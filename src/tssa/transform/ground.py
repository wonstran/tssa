"""GroundModel: one interface over ground-plane homographies, with explicit distortion and image size.

Inputs:  any object with `image_to_world(uv)` and `world_to_image(EN)` in metres (`HomographyTransform`,
         `DistortedHomography`, or a user object), and optionally the image size in pixels.
Outputs: `GroundModel` with `image_to_world`, `world_to_image` (distorted pixels <-> world), `undistort`, `distort`
         (identity when the homography has no distortion model), `world_to_undistorted`, and `image_size`.

Design: docs/2026-11-10-footprint-design.md, section 4.2. A pinhole camera works in undistorted pixels only; this
adapter is where the two pixel spaces are told apart.
"""

from __future__ import annotations

import numpy as np

from .homography import DistortedHomography, HomographyTransform


def _points(xy, name: str) -> tuple[np.ndarray, bool]:
    return HomographyTransform._prepare_query_points(xy, name=name)


class GroundModel:
    """A homography plus its distortion model and image size.

    Build it with `GroundModel.wrap(homography, image_size=None)`. The image size comes from the explicit argument,
    else from the homography's `image_size` attribute; the two must agree, and one of them must exist.
    """

    def __init__(self, homography, image_size) -> None:
        for name in ("image_to_world", "world_to_image"):
            if not callable(getattr(homography, name, None)):
                raise ValueError(f"homography lacks {name}()")
        has_u = callable(getattr(homography, "undistort", None))
        has_d = callable(getattr(homography, "distort", None))
        if has_u != has_d:
            raise ValueError("homography has only one of undistort() / distort(); give both or neither")
        size = tuple(float(v) for v in image_size)
        if len(size) != 2 or not all(np.isfinite(size)) or min(size) <= 0:
            raise ValueError(f"image_size must be a positive (width, height), got {image_size!r}")
        self.homography = homography
        self.image_size = size
        self.has_distortion = has_u

    @classmethod
    def wrap(cls, homography, image_size=None) -> "GroundModel":
        """Wrap `homography` (or re-check an existing GroundModel against `image_size`)."""
        if isinstance(homography, GroundModel):
            if image_size is not None and not np.allclose(image_size, homography.image_size):
                raise ValueError(f"image_size {tuple(image_size)} conflicts with the model's {homography.image_size}")
            return homography
        own = getattr(homography, "image_size", None)
        if image_size is not None and own is not None and not np.allclose(image_size, own):
            raise ValueError(f"image_size {tuple(image_size)} conflicts with the homography's image_size {tuple(own)}")
        size = image_size if image_size is not None else own
        if size is None:
            raise ValueError("image size unknown: pass image_size=(width, height); this homography does not carry one")
        return cls(homography, size)

    # distorted pixels <-> world -----------------------------------------------------------------------------------

    def image_to_world(self, image_xy) -> np.ndarray:
        """Distorted pixel(s) ``(2,)`` / ``(N, 2)`` -> world metres."""
        return np.asarray(self.homography.image_to_world(image_xy), dtype=float)

    def world_to_image(self, world_xy) -> np.ndarray:
        """World metres ``(2,)`` / ``(N, 2)`` -> distorted pixels."""
        return np.asarray(self.homography.world_to_image(world_xy), dtype=float)

    # distortion ---------------------------------------------------------------------------------------------------

    def undistort(self, image_xy) -> np.ndarray:
        """Distorted -> undistorted pixels; the identity without a distortion model."""
        if self.has_distortion:
            return np.asarray(self.homography.undistort(image_xy), dtype=float)
        pts, squeeze = _points(image_xy, "image_xy")
        return pts[0].copy() if squeeze else pts.copy()

    def distort(self, undistorted_xy) -> np.ndarray:
        """Undistorted -> distorted pixels; the identity without a distortion model."""
        if self.has_distortion:
            return np.asarray(self.homography.distort(undistorted_xy), dtype=float)
        pts, squeeze = _points(undistorted_xy, "undistorted_xy")
        return pts[0].copy() if squeeze else pts.copy()

    def world_to_undistorted(self, world_xy) -> np.ndarray:
        """World metres -> undistorted pixels.

        For `DistortedHomography` this goes straight through its inner (undistorted image -> world) homography, so it
        avoids the iteration error of `distort()`; otherwise it is `undistort(world_to_image(.))`.
        """
        h = self.homography
        if isinstance(h, DistortedHomography):
            pts, squeeze = _points(world_xy, "world_xy")
            out = h.homography.world_to_image(pts - h.origin)
            return out[0] if squeeze else out
        return self.undistort(self.world_to_image(world_xy))

    def __repr__(self) -> str:
        kind = type(self.homography).__name__
        return f"GroundModel({kind}, image_size={self.image_size}, distortion={self.has_distortion})"
