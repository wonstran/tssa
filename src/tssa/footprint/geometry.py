"""Angles, box points, Jacobian, rectangles and the cuboid face (ported from the speed study's geometry.py, m3.py,
m6.py and smoothlib/ground.py; snapshot docs/study-snapshot-2026-11-10.json).

World coordinates are the homography's (e.g. UTM metres). Angles are compass degrees (0 = north, clockwise);
directions in [0, 360), axes in [0, 180).
"""

from __future__ import annotations

import numpy as np
import shapely

from .config import FT


def compass_deg(vec) -> np.ndarray:
    """Compass angle in [0, 360) of world vectors (dE, dN): atan2(dE, dN)."""
    v = np.asarray(vec, dtype=float)
    return np.degrees(np.arctan2(v[..., 0], v[..., 1])) % 360.0


def unit_from_compass(deg) -> np.ndarray:
    """Unit (dE, dN) vectors for compass degrees."""
    r = np.radians(np.asarray(deg, dtype=float))
    return np.stack([np.sin(r), np.cos(r)], axis=-1)


def rot_ccw(vec) -> np.ndarray:
    """Rotate 90 degrees counter-clockwise seen from above: (dE, dN) -> (-dN, dE)."""
    v = np.asarray(vec, dtype=float)
    return np.stack([-v[..., 1], v[..., 0]], axis=-1)


def _scalar(d):
    return float(d) if np.ndim(d) == 0 else d


def diff_directed(a, b):
    """Smallest absolute angle between two directions, in [0, 180]."""
    return _scalar(np.abs((np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0))


def to_undirected(deg):
    """Fold a direction onto an axis in [0, 180)."""
    return _scalar(np.asarray(deg, dtype=float) % 180.0)


def diff_undirected(a, b):
    """Smallest angle between two axes, in [0, 90]."""
    return _scalar(np.abs((np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 90.0) % 180.0 - 90.0))


def direct_toward(axis_deg, ref_deg):
    """Pick axis or axis + 180, whichever is closer to ref; result in [0, 360). NaN ref keeps the axis."""
    a = np.asarray(axis_deg, dtype=float) % 360.0
    flip = diff_directed(a + 180.0, ref_deg) < diff_directed(a, ref_deg)
    return _scalar(np.where(flip, (a + 180.0) % 360.0, a))


def bottom_centre(boxes) -> np.ndarray:
    """(n, 2) pixel bottom-centre `[x + w/2, y + h]` of (n, 4) boxes x, y, w, h."""
    b = np.atleast_2d(np.asarray(boxes, dtype=float))
    return np.column_stack([b[:, 0] + b[:, 2] / 2.0, b[:, 1] + b[:, 3]])


def project(ground, uv) -> np.ndarray:
    return np.asarray(ground.image_to_world(np.atleast_2d(np.asarray(uv, dtype=float))), dtype=float).reshape(-1, 2)


def jacobian(ground, uv, eps: float = 0.5) -> np.ndarray:
    """Central-difference Jacobian d(world)/d(u, v), shape (n, 2, 2) (study smoothlib.ground.jacobian)."""
    uv = np.atleast_2d(np.asarray(uv, dtype=float))
    J = np.empty((len(uv), 2, 2))
    for k in range(2):
        step = np.zeros(2)
        step[k] = eps
        J[:, :, k] = (project(ground, uv + step) - project(ground, uv - step)) / (2 * eps)
    return J


def away_dir(ground, uv) -> np.ndarray:
    """Unit world direction of image step (u, v) -> (u, v - 1), i.e. away from the camera (study M2)."""
    uv = np.atleast_2d(np.asarray(uv, dtype=float))
    d = project(ground, uv + np.array([0.0, -1.0])) - project(ground, uv)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def m1_points(ground, boxes) -> np.ndarray:
    """M1: world point under the box bottom-centre."""
    return project(ground, bottom_centre(boxes))


def m2_points(ground, boxes, width_m) -> np.ndarray:
    """M2: M1 moved half the vehicle width away from the camera."""
    uv = bottom_centre(boxes)
    w = np.broadcast_to(np.asarray(width_m, dtype=float), (len(uv),))
    return project(ground, uv) + (w / 2.0)[:, None] * away_dir(ground, uv)


def projection_valid(EN, J, max_ft_per_px: float, zone=None, gate_buffer_ft: float = 100.0) -> np.ndarray:
    """Finite, local scale (largest singular value of J) <= max_ft_per_px, and inside the buffered zone if given."""
    EN = np.atleast_2d(np.asarray(EN, dtype=float))
    J = np.asarray(J, dtype=float).reshape(-1, 2, 2)
    finite = np.isfinite(EN).all(axis=1) & np.isfinite(J).all(axis=(1, 2))
    Jc = np.where(finite[:, None, None], J, 0.0)
    scale = np.linalg.svd(Jc, compute_uv=False)[:, 0] if len(Jc) else np.zeros(0)
    ok = finite & (scale <= max_ft_per_px * FT)
    if zone is not None and len(EN):
        safe = np.where(finite[:, None], EN, 0.0)
        ok &= np.asarray(shapely.contains_xy(zone.buffer(gate_buffer_ft * FT), safe[:, 0], safe[:, 1]), dtype=bool)
    return ok


def edge_flag(boxes, image_size, edge_px: float) -> np.ndarray:
    """Box within edge_px of an image border."""
    b = np.atleast_2d(np.asarray(boxes, dtype=float))
    W, H = image_size
    x, y, w, h = b.T
    return (x < edge_px) | (y < edge_px) | (x + w > W - edge_px) | (y + h > H - edge_px)


def rect_corners(centre, heading_deg, L_m, W_m) -> np.ndarray:
    """Corners front-left, front-right, rear-right, rear-left (front along the heading).

    One rectangle: centre (2,), scalar heading -> (4, 2). Vectorised: centre (n, 2), heading (n,), L and W
    broadcastable to (n,) -> (n, 4, 2).
    """
    c = np.asarray(centre, dtype=float)
    f = unit_from_compass(heading_deg)
    lf = rot_ccw(f)
    L = np.asarray(L_m, dtype=float)
    W = np.asarray(W_m, dtype=float)
    if c.ndim == 1:
        a, b = L / 2.0 * f, W / 2.0 * lf
        return np.array([c + a + b, c + a - b, c - a - b, c - a + b])
    a = (L.reshape(-1, 1) / 2.0) * f
    b = (W.reshape(-1, 1) / 2.0) * lf
    return np.stack([c + a + b, c + a - b, c - a - b, c - a + b], axis=1)


def face_geometry(face_uv, ground):
    """Cuboid bottom face (4 image corners, any start or winding) -> (corners (4, 2), centroid, length_m, width_m,
    axis_deg): opposite-edge means give length and width, the doubled-angle mean of the long edges the axis."""
    corners = project(ground, np.asarray(face_uv, dtype=float).reshape(4, 2))
    edges = np.roll(corners, -1, axis=0) - corners  # edge i: corner i -> i + 1
    lens = np.linalg.norm(edges, axis=1)
    pair_a, pair_b = lens[[0, 2]].mean(), lens[[1, 3]].mean()
    long_i = (0, 2) if pair_a >= pair_b else (1, 3)
    length, width = max(pair_a, pair_b), min(pair_a, pair_b)
    ang = np.radians(compass_deg(edges[list(long_i)]))
    axis = to_undirected(np.degrees(np.arctan2(np.sin(2 * ang).sum(), np.cos(2 * ang).sum())) / 2.0)
    return corners, corners.mean(axis=0), float(length), float(width), float(axis)
