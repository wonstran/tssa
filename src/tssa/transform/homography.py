"""Homography utilities for image/world coordinate conversion.

This module provides `HomographyTransform` for mapping 2D image coordinates to
2D ground-plane world coordinates (and back) using a 3x3 projective transform.

Example:
    >>> import numpy as np
    >>> from tssa.transformation.homography import HomographyTransform
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
        src = self._validate_points(image_points, min_points=4, name="image_points")
        dst = self._validate_points(world_points, min_points=4, name="world_points")

        if src.shape[0] != dst.shape[0]:
            raise ValueError("image_points and world_points must contain the same number of points.")

        H, mask = cv2.findHomography(src, dst, method=method, ransacReprojThreshold=ransac_reproj_threshold)
        if H is None:
            raise ValueError("Failed to compute homography matrix from input points.")

        self.H = H
        self.inlier_mask = mask

        try:
            self.H_inv = np.linalg.inv(H)
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
        np.savez(file_path, H=self.H, H_inv=self.H_inv, inlier_mask=self.inlier_mask)

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
