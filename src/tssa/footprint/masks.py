"""Mask contours per (track, frame): the `masks` table of M5 and M7, and the outline helpers (ported from the speed
study's m5.lower_outline and m7_candidates.split_contours / join_contours / rasterise).

A `MaskTable` maps (track, frame) -> list of external contour polygons (k, 2) in distorted pixels, largest first.
On disk it is the study's m7 cache layout: an `.npz` with arrays track, frame, count and pts, where pts holds every
row's contours back to back, each polygon followed by a NaN row, and count[i] rows of pts belong to row i.
Rasterising needs OpenCV (imported on first use).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("M5 and M7 rasterise mask contours with OpenCV (pip install opencv-python)") from exc
    return cv2


def split_contours(arr) -> list[np.ndarray]:
    """(n, 2) array with NaN separator rows -> list of polygons."""
    arr = np.asarray(arr, dtype=float).reshape(-1, 2)
    if not len(arr):
        return []
    sep = np.isnan(arr).any(axis=1)
    out, start = [], 0
    for k in np.flatnonzero(sep):
        if k > start:
            out.append(arr[start:k].copy())
        start = k + 1
    if start < len(arr):
        out.append(arr[start:].copy())
    return out


def join_contours(polys) -> np.ndarray:
    """List of polygons -> (n, 2) with a NaN row after each polygon."""
    parts = []
    for p in polys:
        parts.append(np.asarray(p, dtype=float).reshape(-1, 2))
        parts.append(np.full((1, 2), np.nan))
    return np.vstack(parts) if parts else np.empty((0, 2))


def rasterise(polys, img_size) -> np.ndarray:
    """Bool (H, W) mask of the filled polygons."""
    cv2 = _cv2()
    W, H = (int(round(v)) for v in img_size)
    canvas = np.zeros((H, W), np.uint8)
    pts = [np.round(np.asarray(p)).astype(np.int32) for p in polys if len(p) >= 3]
    if pts:
        cv2.fillPoly(canvas, pts, 1)
    return canvas.astype(bool)


def lower_outline(mask, box_xywh) -> np.ndarray:
    """(m, 2) pixels [u, v]: the largest set row per integer column inside the bbox.

    `mask` is a polygon (k, 2) of pixel vertices, a list of polygons, or an (H, W) bool array.
    """
    x, y, w, h = (float(v) for v in box_xywh)
    if isinstance(mask, (list, tuple)):
        polys = [np.asarray(p, dtype=float).reshape(-1, 2) for p in mask]
        if not polys:
            return np.empty((0, 2))
        hi = np.vstack(polys).max(axis=0)
        m = rasterise(polys, (max(hi[0], x + w) + 2, max(hi[1], y + h) + 2))
    else:
        m = np.asarray(mask)
        if m.ndim == 2 and m.shape[1] == 2 and m.dtype != bool:
            if len(m) < 3:
                return np.empty((0, 2))
            m = rasterise([m], (int(np.ceil(max(m[:, 0].max(), x + w))) + 2, int(np.ceil(max(m[:, 1].max(), y + h))) + 2))
    H, Wd = m.shape
    u0, u1 = max(int(np.ceil(x)), 0), min(int(np.floor(x + w)), Wd - 1)
    out = []
    for u in range(u0, u1 + 1):
        rows = np.flatnonzero(m[:, u])
        if len(rows):
            out.append((float(u), float(rows[-1])))
    return np.array(out, dtype=float).reshape(-1, 2)


class MaskTable(Mapping):
    """(track, frame) -> list of contour polygons (distorted px), largest first."""

    def __init__(self, data: dict | None = None) -> None:
        self._d = {(int(t), int(f)): [np.asarray(p, dtype=float).reshape(-1, 2) for p in v]
                   for (t, f), v in (data or {}).items()}  # fmt: skip

    def __getitem__(self, key):
        return self._d[(int(key[0]), int(key[1]))]

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __repr__(self) -> str:
        return f"MaskTable({len(self)} rows)"

    def save_npz(self, path) -> None:
        keys = sorted(self._d)
        joined = [join_contours(self._d[k]) for k in keys]
        np.savez_compressed(
            path, track=np.array([k[0] for k in keys], dtype=np.int64), frame=np.array([k[1] for k in keys], np.int64),
            count=np.array([len(j) for j in joined], dtype=np.int64),
            pts=np.vstack(joined) if joined else np.empty((0, 2)),
        )  # fmt: skip

    @classmethod
    def load_npz(cls, path) -> "MaskTable":
        """Read the layout `save_npz` writes (also the speed study's m7 cache `outlines.npz`)."""
        z = np.load(path, allow_pickle=False)
        track, frame, count, pts = z["track"], z["frame"], z["count"], z["pts"]  # each array read once
        ends = np.cumsum(count)
        starts = ends - count
        return cls({(t, f): split_contours(pts[a:b]) for t, f, a, b in zip(track, frame, starts, ends)})
