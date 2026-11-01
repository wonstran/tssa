"""Homography utilities for image/world coordinate conversion.

This module provides `HomographyTransform` for mapping 2D image coordinates to
2D ground-plane world coordinates (and back) using a 3x3 projective transform.

Example:
    >>> import numpy as np
    >>> from tssa.transform.homography import HomographyTransform
    >>> image_pts = np.array([[100, 200], [200, 200], [200, 300], [100, 300]])
    >>> world_pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    >>> h = HomographyTransform(image_pts, world_pts)
    >>> h.image_to_world(np.array([150, 250])).shape
    (2,)
    >>> h.world_to_image(np.array([[0.5, 0.5]])).shape
    (1, 2)

Practical workflow for roadside-camera use cases:
    1. Collect ground control points (GCPs) in Google Earth Pro.
    2. Label corresponding pixel points on a reference video frame.
    3. Convert WGS84 latitude/longitude to metric coordinates (UTM or State Plane).
    4. Estimate homography with OpenCV from pixel points to ground-plane points.

Notes:
    - This gives a *ground-plane* coordinate system, not full 3D calibration.
    - Use stable pavement/corner features and avoid elevated/moving objects.
    - Prefer 8+ distributed GCPs for robust estimation (minimum is 4).
    - Distortion, slope, and map/video time mismatch can introduce error.
"""

from __future__ import annotations

import json

import cv2
import numpy as np


class HomographyTransform:
    """Convert points between image (x, y) and world (x, y) coordinates."""

    def __init__(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        method: int = cv2.RANSAC,
        ransac_reproj_threshold: float = 3.0,
    ) -> None:
        """Compute homography and inverse-homography matrices.

        Args:
            image_points: Image-space correspondences shaped ``(N, 2)`` with ``N >= 4``.
            world_points: World-space correspondences shaped ``(N, 2)`` with ``N >= 4``.
            method: OpenCV homography estimation method (e.g., ``cv2.RANSAC``).
            ransac_reproj_threshold: Reprojection threshold used when ``method`` is RANSAC-like.

        Raises:
            ValueError: If point arrays are invalid, mismatched in length, or homography fails.
            ValueError: If the estimated homography is singular and cannot be inverted.
        """
        self._ransac_threshold = ransac_reproj_threshold
        src = self._validate_points(image_points, min_points=4, name="image_points")
        dst = self._validate_points(world_points, min_points=4, name="world_points")

        if src.shape[0] != dst.shape[0]:
            raise ValueError("image_points and world_points must contain the same number of points.")

        H, mask = cv2.findHomography(src, dst, method=method, ransacReprojThreshold=ransac_reproj_threshold)
        if H is None:
            raise ValueError("Failed to compute homography matrix from input points.")

        self.H = H.astype(np.float64)
        self.inlier_mask = mask
        self.n_points = int(src.shape[0])

        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Homography matrix is singular and cannot be inverted.") from exc

    @staticmethod
    def _validate_points(points: np.ndarray, min_points: int = 1, name: str = "points") -> np.ndarray:
        """Validate point array format and minimum cardinality.

        Args:
            points: Candidate point array expected in ``(N, 2)`` format.
            min_points: Minimum required number of rows.
            name: Parameter name used in validation error messages.

        Returns:
            A float64 NumPy array with shape ``(N, 2)``.

        Raises:
            ValueError: If shape is not ``(N, 2)`` or point count is less than ``min_points``.
        """
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{name} must have shape (N, 2).")
        if arr.shape[0] < min_points:
            raise ValueError(f"{name} must contain at least {min_points} points.")
        return arr

    @staticmethod
    def _prepare_query_points(points: np.ndarray, name: str = "points") -> tuple[np.ndarray, bool]:
        """Normalize query input points to ``(N, 2)`` format.

        Accepts either a single point shaped ``(2,)`` or batched points shaped ``(N, 2)``.

        Args:
            points: Point(s) to transform.
            name: Parameter name used in validation error messages.

        Returns:
            A tuple of ``(normalized_points, squeeze_back)`` where ``normalized_points`` has
            shape ``(N, 2)`` and ``squeeze_back`` indicates whether the original input was
            a single point that should be returned as shape ``(2,)``.

        Raises:
            ValueError: If input is neither ``(2,)`` nor ``(N, 2)``.
        """
        arr = np.asarray(points, dtype=np.float64)
        squeeze_back = False

        if arr.ndim == 1:
            if arr.shape[0] != 2:
                raise ValueError(f"{name} must have shape (2,) or (N, 2).")
            arr = arr.reshape(1, 2)
            squeeze_back = True
        elif arr.ndim == 2 and arr.shape[1] == 2:
            pass
        else:
            raise ValueError(f"{name} must have shape (2,) or (N, 2).")

        return arr, squeeze_back

    @staticmethod
    def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Project 2D points with a 3x3 homography matrix.

        Args:
            points: Input points shaped ``(N, 2)``.
            matrix: Homography matrix shaped ``(3, 3)``.

        Returns:
            Projected points shaped ``(N, 2)``.

        Raises:
            cv2.error: If OpenCV projection fails for invalid matrix/point inputs.
        """
        if points.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)
        reshaped = points.reshape(-1, 1, 2).astype(np.float64)
        projected = cv2.perspectiveTransform(reshaped, matrix)
        return projected.reshape(-1, 2)

    def image_to_world(self, image_xy: np.ndarray) -> np.ndarray:
        """Transform image-space point(s) into world-space point(s).

        Args:
            image_xy: A single point ``(2,)`` or point array ``(N, 2)`` in image coordinates.

        Returns:
            Transformed point(s), matching input rank:
            - ``(2,)`` for a single input point
            - ``(N, 2)`` for batched input points
        """
        points, squeeze_back = self._prepare_query_points(image_xy, name="image_xy")
        out = self._project(points, self.H)
        return out[0] if squeeze_back else out

    def world_to_image(self, world_xy: np.ndarray) -> np.ndarray:
        """Transform world-space point(s) into image-space point(s).

        Args:
            world_xy: A single point ``(2,)`` or point array ``(N, 2)`` in world coordinates.

        Returns:
            Transformed point(s), matching input rank:
            - ``(2,)`` for a single input point
            - ``(N, 2)`` for batched input points
        """
        points, squeeze_back = self._prepare_query_points(world_xy, name="world_xy")
        out = self._project(points, self.H_inv)
        return out[0] if squeeze_back else out

    def save_matrix(self, file_path: str) -> None:
        """Save homography matrices to a NumPy ``.npz`` file.

        Args:
            file_path: Destination path for the saved matrices.
        """
        arrays = {"H": self.H, "H_inv": self.H_inv}
        if self.inlier_mask is not None:
            arrays["inlier_mask"] = self.inlier_mask
        np.savez(file_path, **arrays)

    @classmethod
    def load_matrix(cls, file_path: str) -> "HomographyTransform":
        """Load homography matrices from a NumPy ``.npz`` file.

        Args:
            file_path: Path to a saved homography ``.npz`` file.

        Returns:
            A `HomographyTransform` instance populated from saved matrices.

        Raises:
            ValueError: If loaded ``H`` or ``H_inv`` does not have shape ``(3, 3)``.
        """
        data = np.load(file_path, allow_pickle=False)
        H = data["H"]
        H_inv = data["H_inv"]
        inlier_mask = data["inlier_mask"] if "inlier_mask" in data.files else None

        if H.shape != (3, 3) or H_inv.shape != (3, 3):
            raise ValueError("Saved matrices must both have shape (3, 3).")

        obj = cls.__new__(cls)
        obj.H = H.astype(np.float64)
        obj.H_inv = H_inv.astype(np.float64)
        obj.inlier_mask = inlier_mask
        obj.n_points = int(inlier_mask.size) if inlier_mask is not None else 0
        obj._ransac_threshold = 3.0
        return obj

    def save_json(self, file_path: str) -> None:
        """Save homography matrices and inlier mask to a JSON file.

        Args:
            file_path: Destination path for the JSON file.
        """
        data = {
            "H": self.H.tolist(),
            "H_inv": self.H_inv.tolist(),
            "inlier_mask": None if self.inlier_mask is None else self.inlier_mask.ravel().astype(int).tolist(),
            "n_points": self.n_points,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, file_path: str) -> "HomographyTransform":
        """Load a homography from a JSON file written by `save_json`.

        The model is restored directly from the saved values; no refitting is done.

        Args:
            file_path: Path to a JSON file written by `save_json`.

        Returns:
            A `HomographyTransform` instance with the saved parameters.

        Raises:
            ValueError: If the saved homography matrices do not have shape ``(3, 3)``.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        H = np.asarray(data["H"], dtype=np.float64)
        H_inv = np.asarray(data["H_inv"], dtype=np.float64)
        if H.shape != (3, 3) or H_inv.shape != (3, 3):
            raise ValueError("Saved matrices must both have shape (3, 3).")

        obj = cls.__new__(cls)
        obj.H = H
        obj.H_inv = H_inv
        mask = data.get("inlier_mask")
        obj.inlier_mask = None if mask is None else np.asarray(mask, dtype=np.uint8).reshape(-1, 1)
        obj.n_points = int(data["n_points"])
        obj._ransac_threshold = 3.0
        return obj

    def __repr__(self) -> str:
        """Return a string representation of the homography transform."""
        n_inliers = int(self.inlier_mask.sum()) if self.inlier_mask is not None else 0
        return f"HomographyTransform(n_points={self.n_points}, inliers={n_inliers})"

    def reprojection_error(self, image_points: np.ndarray, world_points: np.ndarray) -> float:
        """Compute mean reprojection error for validation.

        Projects world_points to image space and compares to the original image_points.
        Lower values indicate better fit.

        Args:
            image_points: Original image points shaped (N, 2).
            world_points: Original world points shaped (N, 2).

        Returns:
            Mean Euclidean distance between projected and original image points.
        """
        src = self._validate_points(image_points, min_points=1, name="image_points")
        dst = self._validate_points(world_points, min_points=1, name="world_points")

        if src.shape[0] != dst.shape[0]:
            raise ValueError("image_points and world_points must contain the same number of points.")

        projected = self._project(dst, self.H_inv)
        errors = np.linalg.norm(src - projected, axis=1)
        return float(errors.mean())


class DistortedHomography:
    """Radial lens distortion followed by a ground-plane homography.

    Image points are corrected with ``p' = c + (p - c)(1 + k1 r^2 + k2 r^4 + ...)``, where
    ``r`` is the distance to the image centre ``c`` normalised by half the larger image
    side, and then mapped to world coordinates with a `HomographyTransform`. The
    distortion coefficients are optimised; the homography is re-fitted by least squares
    for each candidate. World coordinates are re-centred on their mean during fitting to
    keep large values (e.g., UTM) well conditioned.

    Example:
        >>> h = DistortedHomography(image_pts, world_pts, image_size=(640, 480))
        >>> h.image_to_world(np.array([320, 240])).shape
        (2,)
    """

    def __init__(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        image_size: tuple[int, int],
        n_coeffs: int = 2,
    ) -> None:
        """Fit distortion coefficients and homography.

        Args:
            image_points: Image-space correspondences shaped ``(N, 2)``.
            world_points: World-space correspondences shaped ``(N, 2)``.
            image_size: Image ``(width, height)`` in pixels; sets the distortion centre.
            n_coeffs: Number of radial coefficients (``k1``, ``k2``, ...). Keep the total
                parameter count (``8 + n_coeffs``) well below ``2 * N``.

        Raises:
            ValueError: If point arrays are invalid or there are too few points.
        """
        from scipy.optimize import least_squares

        min_points = max(4, (8 + n_coeffs + 1) // 2)
        src = HomographyTransform._validate_points(image_points, min_points=min_points, name="image_points")
        dst = HomographyTransform._validate_points(world_points, min_points=min_points, name="world_points")
        if src.shape[0] != dst.shape[0]:
            raise ValueError("image_points and world_points must contain the same number of points.")

        width, height = image_size
        self.image_size = (float(width), float(height))
        self.center = np.array([width / 2.0, height / 2.0])
        self.scale = max(width, height) / 2.0
        self.origin = dst.mean(axis=0)
        local = dst - self.origin

        def residuals(k: np.ndarray) -> np.ndarray:
            undistorted = self._undistort(src, k)
            h = HomographyTransform(undistorted, local, method=0)
            return (h.image_to_world(undistorted) - local).ravel()

        self.k = least_squares(residuals, np.zeros(n_coeffs)).x
        self.homography = HomographyTransform(self._undistort(src, self.k), local, method=0)
        self.n_points = int(src.shape[0])

    def _factor(self, image_xy: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Return the radial scale factor ``1 + k1 r^2 + k2 r^4 + ...`` shaped ``(N, 1)``."""
        r2 = (((image_xy - self.center) / self.scale) ** 2).sum(axis=1, keepdims=True)
        return 1.0 + sum(kk * r2 ** (j + 1) for j, kk in enumerate(k))

    def _undistort(self, image_xy: np.ndarray, k: np.ndarray) -> np.ndarray:
        return self.center + (image_xy - self.center) * self._factor(image_xy, k)

    def undistort(self, image_xy: np.ndarray) -> np.ndarray:
        """Map raw image point(s) ``(2,)`` or ``(N, 2)`` to distortion-corrected pixels."""
        points, squeeze_back = HomographyTransform._prepare_query_points(image_xy, name="image_xy")
        out = self._undistort(points, self.k)
        return out[0] if squeeze_back else out

    def distort(self, undistorted_xy: np.ndarray, n_iter: int = 20) -> np.ndarray:
        """Invert `undistort` by solving ``r_d (1 + k1 r_d^2 + ...) = r_u`` with Newton's method.

        Args:
            undistorted_xy: Corrected pixel point(s) ``(2,)`` or ``(N, 2)``.
            n_iter: Maximum number of Newton iterations.

        Returns:
            Raw image point(s), matching input rank.
        """
        points, squeeze_back = HomographyTransform._prepare_query_points(undistorted_xy, name="undistorted_xy")
        offset = (points - self.center) / self.scale
        r_u = np.linalg.norm(offset, axis=1)
        r_d = r_u.copy()
        for _ in range(n_iter):
            g = r_d * (1.0 + sum(kk * r_d ** (2 * (j + 1)) for j, kk in enumerate(self.k))) - r_u
            dg = 1.0 + sum((2 * j + 3) * kk * r_d ** (2 * (j + 1)) for j, kk in enumerate(self.k))
            step = g / dg
            r_d = r_d - step
            if np.all(np.abs(step) < 1e-12):
                break
        ratio = np.divide(r_d, r_u, out=np.ones_like(r_u), where=r_u > 0)
        out = self.center + offset * self.scale * ratio[:, None]
        return out[0] if squeeze_back else out

    def image_to_world(self, image_xy: np.ndarray) -> np.ndarray:
        """Transform raw image point(s) ``(2,)`` or ``(N, 2)`` into world point(s)."""
        points, squeeze_back = HomographyTransform._prepare_query_points(image_xy, name="image_xy")
        out = self.homography.image_to_world(self._undistort(points, self.k)) + self.origin
        return out[0] if squeeze_back else out

    def world_to_image(self, world_xy: np.ndarray) -> np.ndarray:
        """Transform world point(s) ``(2,)`` or ``(N, 2)`` into raw image point(s)."""
        points, squeeze_back = HomographyTransform._prepare_query_points(world_xy, name="world_xy")
        out = self.distort(self.homography.world_to_image(points - self.origin))
        return out[0] if squeeze_back else out

    def rescale(self, new_size: tuple[int, int]) -> "DistortedHomography":
        """Return a copy of this model for images resized to ``new_size``.

        Distortion coefficients are unchanged (the radius is normalised); the centre, scale
        and homography are rescaled so the new model takes pixels from the resized image.

        Args:
            new_size: Target image ``(width, height)`` in pixels. Must have the same aspect
                ratio as the calibration image; crops and letterboxing are not supported.

        Returns:
            A new `DistortedHomography`; this instance is not modified.

        Raises:
            ValueError: If ``new_size`` is not positive or its aspect ratio differs.
        """
        new_w, new_h = float(new_size[0]), float(new_size[1])
        old_w, old_h = self.image_size
        if new_w <= 0 or new_h <= 0:
            raise ValueError("new_size must be positive.")
        s = new_w / old_w
        if not np.isclose(new_h / old_h, s, rtol=1e-3):
            raise ValueError(
                f"new_size {tuple(new_size)} must keep the aspect ratio of image_size {self.image_size}."
            )

        S = np.diag([s, s, 1.0])
        inner = HomographyTransform.__new__(HomographyTransform)
        inner.H = self.homography.H @ np.linalg.inv(S)
        inner.H_inv = S @ self.homography.H_inv
        inner.inlier_mask = None
        inner.n_points = self.homography.n_points
        inner._ransac_threshold = self.homography._ransac_threshold

        obj = type(self).__new__(type(self))
        obj.k = self.k.copy()
        obj.image_size = (new_w, new_h)
        obj.center = self.center * s
        obj.scale = self.scale * s
        obj.origin = self.origin.copy()
        obj.n_points = self.n_points
        obj.homography = inner
        return obj

    def __repr__(self) -> str:
        """Return a string representation of the distorted homography."""
        return f"DistortedHomography(n_points={self.n_points}, k={np.array2string(self.k, precision=4)})"

    def save_json(self, file_path: str) -> None:
        """Save distortion coefficients, image size, centre, scale, origin and homography to a JSON file.

        Args:
            file_path: Destination path for the JSON file.
        """
        data = {
            "k": self.k.tolist(),
            "image_size": list(self.image_size),
            "center": self.center.tolist(),
            "scale": float(self.scale),
            "origin": self.origin.tolist(),
            "n_points": self.n_points,
            "H": self.homography.H.tolist(),
            "H_inv": self.homography.H_inv.tolist(),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, file_path: str) -> "DistortedHomography":
        """Load a distorted homography from a JSON file written by `save_json`.

        The model is restored directly from the saved values; no refitting is done.

        Args:
            file_path: Path to a JSON file written by `save_json`.

        Returns:
            A `DistortedHomography` instance with the saved parameters.

        Raises:
            ValueError: If the saved homography matrices do not have shape ``(3, 3)``.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        H = np.asarray(data["H"], dtype=np.float64)
        H_inv = np.asarray(data["H_inv"], dtype=np.float64)
        if H.shape != (3, 3) or H_inv.shape != (3, 3):
            raise ValueError("Saved matrices must both have shape (3, 3).")

        inner = HomographyTransform.__new__(HomographyTransform)
        inner.H = H
        inner.H_inv = H_inv
        inner.inlier_mask = None
        inner.n_points = int(data["n_points"])
        inner._ransac_threshold = 3.0

        obj = cls.__new__(cls)
        obj.k = np.asarray(data["k"], dtype=np.float64)
        obj.center = np.asarray(data["center"], dtype=np.float64)
        obj.image_size = tuple(data.get("image_size", 2.0 * obj.center))
        obj.scale = float(data["scale"])
        obj.origin = np.asarray(data["origin"], dtype=np.float64)
        obj.n_points = int(data["n_points"])
        obj.homography = inner
        return obj

    def save_matrix(self, file_path: str) -> None:
        """Save distortion parameters and homography matrices to a NumPy ``.npz`` file.

        Args:
            file_path: Destination path for the saved arrays.
        """
        np.savez(
            file_path,
            k=self.k,
            image_size=np.array(self.image_size),
            center=self.center,
            scale=np.array(self.scale),
            origin=self.origin,
            n_points=np.array(self.n_points),
            H=self.homography.H,
            H_inv=self.homography.H_inv,
        )

    @classmethod
    def load_matrix(cls, file_path: str) -> "DistortedHomography":
        """Load a distorted homography from a NumPy ``.npz`` file written by `save_matrix`.

        The model is restored directly from the saved arrays; no refitting is done.

        Args:
            file_path: Path to a ``.npz`` file written by `save_matrix`.

        Returns:
            A `DistortedHomography` instance with the saved parameters.

        Raises:
            ValueError: If the saved homography matrices do not have shape ``(3, 3)``.
        """
        data = np.load(file_path, allow_pickle=False)
        H = data["H"].astype(np.float64)
        H_inv = data["H_inv"].astype(np.float64)
        if H.shape != (3, 3) or H_inv.shape != (3, 3):
            raise ValueError("Saved matrices must both have shape (3, 3).")

        inner = HomographyTransform.__new__(HomographyTransform)
        inner.H = H
        inner.H_inv = H_inv
        inner.inlier_mask = None
        inner.n_points = int(data["n_points"])
        inner._ransac_threshold = 3.0

        obj = cls.__new__(cls)
        obj.k = data["k"].astype(np.float64)
        obj.center = data["center"].astype(np.float64)
        obj.image_size = tuple(data["image_size"]) if "image_size" in data else tuple(2.0 * obj.center)
        obj.scale = float(data["scale"])
        obj.origin = data["origin"].astype(np.float64)
        obj.n_points = int(data["n_points"])
        obj.homography = inner
        return obj


if __name__ == "__main__":
    # Example usage
    image_pts = np.array([[100, 200], [200, 200], [200, 300], [100, 300]])
    world_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    homography = HomographyTransform(image_pts, world_pts)
    print("Homography matrix:\n", homography.H)
    print("Inverse homography matrix:\n", homography.H_inv)

    test_image_point = np.array([150, 250])
    world_point = homography.image_to_world(test_image_point)
    print(f"Image point {test_image_point} maps to world point {world_point}")

    test_world_point = np.array([0.5, 0.5])
    image_point = homography.world_to_image(test_world_point)
    print(f"World point {test_world_point} maps to image point {image_point}")
