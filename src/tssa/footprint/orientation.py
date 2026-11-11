"""Motion heading rules and native-axis smoothing (design docs/2026-11-10-footprint-design.md section 10; ported from
the speed study's smooth.py and products.py)."""

from __future__ import annotations

import numpy as np

from ..smooth.methods import rts_smooth
from .geometry import compass_deg


def motion_heading(vel, min_speed) -> tuple[np.ndarray, np.ndarray]:
    """Directed heading per row of ONE segment and its reliability, from smoothed velocity (any unit; `min_speed` in
    the same unit). Unreliable rows hold the last reliable heading; rows before the first reliable one are
    back-filled from it; a segment with no reliable row gets NaN everywhere."""
    vel = np.asarray(vel, dtype=float).reshape(-1, 2)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    reliable = np.isfinite(speed) & (speed >= min_speed)
    n = len(vel)
    if not reliable.any():
        return np.full(n, np.nan), reliable
    raw = np.where(reliable, compass_deg(np.where(reliable[:, None], vel, 1.0)), np.nan)
    idx = np.where(reliable, np.arange(n), -1)
    idx = np.maximum.accumulate(idx)  # last reliable row at or before each row
    idx[idx < 0] = np.flatnonzero(reliable)[0]  # back-fill
    return raw[idx], reliable


def smooth_axis(deg, has, q: float, sigma: float, dt: float) -> np.ndarray:
    """Smooth the native axes of ONE segment (rows `has`) in doubled-angle form; result in [0, 180), every row.

    0 native rows: all NaN. 1: that axis on every row. 2 or more: RTS on the doubled-angle unit vector (rows without
    a native axis are predicted). The reach rule is applied by the caller.
    """
    deg = np.asarray(deg, dtype=float)
    has = np.asarray(has, dtype=bool) & np.isfinite(deg)
    n = len(deg)
    k = int(has.sum())
    if k == 0:
        return np.full(n, np.nan)
    if k == 1:
        return np.full(n, deg[has][0] % 180.0)
    th = np.radians(deg * 2.0)
    vec = np.column_stack([np.cos(th), np.sin(th)])
    vec[~has] = np.nan
    R = np.tile(np.eye(2) * sigma**2, (n, 1, 1))
    sm, _ = rts_smooth(vec, R, dt, q)
    out = (np.degrees(np.arctan2(sm[:, 1], sm[:, 0])) / 2.0) % 180.0
    return np.where(out >= 180.0, 0.0, out)


def near(frames, has, reach: int) -> np.ndarray:
    """True on rows within `reach` frames of a row with `has` (one segment, frames sorted)."""
    frames = np.asarray(frames)
    has = np.asarray(has, dtype=bool)
    if not has.any():
        return np.zeros(len(frames), bool)
    return nearest_distance(frames, frames[has]) <= reach


def nearest_distance(frames, targets) -> np.ndarray:
    """Distance (frames) from each frame to the nearest of the sorted `targets`."""
    frames = np.asarray(frames)
    targets = np.asarray(targets)
    k = np.clip(np.searchsorted(targets, frames), 1, max(len(targets) - 1, 1))
    lo = targets[np.minimum(k - 1, len(targets) - 1)]
    hi = targets[np.minimum(k, len(targets) - 1)]
    return np.minimum(np.abs(frames - lo), np.abs(frames - hi))


def nearest_index(frames, targets) -> np.ndarray:
    """Index into the sorted `targets` of the nearest one to each frame."""
    frames = np.asarray(frames)
    targets = np.asarray(targets)
    k = np.clip(np.searchsorted(targets, frames), 1, max(len(targets) - 1, 1))
    lo, hi = np.minimum(k - 1, len(targets) - 1), np.minimum(k, len(targets) - 1)
    return np.where(np.abs(frames - targets[lo]) <= np.abs(frames - targets[hi]), lo, hi)
