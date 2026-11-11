"""One-to-one matching of detections to track boxes (ported from the speed study's infer.py; run report R21).

numpy and scipy only, so users with their own detectors can match without the `detect` extra.
"""

from __future__ import annotations

import numpy as np


def compute_iou(box1, boxes2) -> np.ndarray:
    """IoU between a box [x1, y1, x2, y2] and an (N, 4) array of boxes."""
    boxes2 = np.asarray(boxes2, dtype=float).reshape(-1, 4)
    if len(boxes2) == 0:
        return np.zeros((0,), dtype=float)
    box1 = np.asarray(box1, dtype=float)
    ix1 = np.maximum(box1[0], boxes2[:, 0])
    iy1 = np.maximum(box1[1], boxes2[:, 1])
    ix2 = np.minimum(box1[2], boxes2[:, 2])
    iy2 = np.minimum(box1[3], boxes2[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-6)


def match_frame(track_xyxy, det_xyxy, min_iou: float = 0.3, det_cls=None, prefer=None) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one detection -> track-box assignment for one frame: (detection index or -1, IoU) per track box.

    Pairs below `min_iou` are excluded and the rest are assigned by maximum total IoU (linear_sum_assignment on
    -IoU); a pair is kept only if its IoU >= min_iou (and > 0). With `det_cls` and `prefer`, only detections of a
    preferred class are candidates whenever the frame has any. The IoU of an unmatched box is its best candidate IoU
    (diagnostic). Pass every track box of the frame, in a fixed order, so a row's assignment does not depend on
    which rows were requested.
    """
    from scipy.optimize import linear_sum_assignment

    t = np.asarray(track_xyxy, dtype=float).reshape(-1, 4)
    dets = np.asarray(det_xyxy, dtype=float).reshape(-1, 4)
    idx, best = np.full(len(t), -1, dtype=int), np.zeros(len(t))
    if len(t) == 0 or len(dets) == 0:
        return idx, best
    cand = np.ones(len(dets), dtype=bool)
    if det_cls is not None and prefer is not None:
        ok = np.isin(np.asarray(det_cls).astype(int), list(prefer))
        if ok.any():
            cand = ok
    M = np.vstack([compute_iou(b, dets) for b in t])
    M[:, ~cand] = 0.0
    best = M.max(axis=1)
    A = np.where(M >= min_iou, M, 0.0)
    for i, j in zip(*linear_sum_assignment(-A)):
        if A[i, j] > 0.0:
            idx[i], best[i] = j, A[i, j]
    return idx, best


def xyxy(boxes_xywh) -> np.ndarray:
    """(n, 4) x, y, w, h -> x1, y1, x2, y2."""
    b = np.asarray(boxes_xywh, dtype=float).reshape(-1, 4)
    return np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]])
