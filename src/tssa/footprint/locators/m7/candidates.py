"""Ported from the speed study's m7_candidates.py (sha256 7a664ff6885db247aa0d1b2e942115aece3522ba9e187761d1e7260883ef4261, snapshot docs/study-snapshot-2026-11-10.json).
Changes: settings through the per-solve namespace `S`; image-size and edge defaults resolved at call time;
rasterising and the lower outline from tssa.footprint.masks (OpenCV imported on first use).

M7 wheel-contact candidates from a SAM mask, without a pose (design 2026-11-08-m7, section 5.1).

Per row: lower outline of the mask inside the bbox -> drop untrustworthy columns (image edge, nearer boxes, leaks
below the bbox) -> undistort -> robust one- or two-segment body line (points below it are down-weighted, so tyre
bumps do not pull it) -> runs of columns protruding below the line -> runs split at dips -> tyre-size check
against the expected tyre diameter D_c at the run's own ground point -> candidate = the run's lowest pixel, with
its frozen gate g_c and rejection price kappa_c.

Geometry inputs:
- `dh`: ground model (image_to_world on DISTORTED pixels, undistort);
- `cam`: camera.Camera (project UTM 3-D -> UNDISTORTED pixels).
Contours are lists of (k, 2) distorted pixel polygons (several components allowed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...masks import lower_outline
from ...masks import rasterise as _rasterise
from .settings import S

CAND_COLUMNS = ["cand", "u", "v", "u_und", "v_und", "depth_px", "width_px", "tyre_px", "width_ratio", "gate_px",
                "kappa", "accepted", "reject"]  # fmt: skip
BELOW_BOTTOM_PX = 1.0  # outline points more than this below the bbox bottom are mask leaks (as M5 R23)
N_BREAKS = 17  # candidate breakpoints for the two-segment line (5th..85th percentile of u)


def rho(z):
    """Cauchy loss log(1 + z^2) (design 7.4)."""
    z = np.asarray(z, dtype=float)
    return np.log1p(z * z)


def split_contours(arr) -> list[np.ndarray]:
    """(n, 2) array with NaN separator rows -> list of polygons (the m7 cache format)."""
    arr = np.asarray(arr, dtype=float).reshape(-1, 2)
    out, cur = [], []
    for p in arr:
        if np.isnan(p).any():
            if cur:
                out.append(np.array(cur))
            cur = []
        else:
            cur.append(p)
    if cur:
        out.append(np.array(cur))
    return out


def join_contours(polys) -> np.ndarray:
    """List of polygons -> (n, 2) with a NaN row after each polygon."""
    parts = []
    for p in polys:
        parts.append(np.asarray(p, dtype=float).reshape(-1, 2))
        parts.append(np.full((1, 2), np.nan))
    return np.vstack(parts) if parts else np.empty((0, 2))


def rasterise(polys, img_size=None) -> np.ndarray:
    return _rasterise(polys, S.IMG_SIZE if img_size is None else img_size)


def trusted_mask(outline, box, other_boxes, img_size=None, edge_px=None) -> np.ndarray:
    """Columns of `outline` (distorted px) that are boundary evidence: not at the image edge, not inside a nearer
    box of the same frame (`other_boxes` rows x, y, w, h with a lower bottom edge), not leaking below the bbox."""
    img_size = S.IMG_SIZE if img_size is None else img_size
    edge_px = S.EDGE_PX if edge_px is None else edge_px
    o = np.asarray(outline, dtype=float).reshape(-1, 2)
    W, H = img_size
    x, y, w, h = (float(v) for v in box)
    ok = (o[:, 0] >= edge_px) & (o[:, 0] <= W - 1 - edge_px) & (o[:, 1] <= H - 1 - edge_px)
    ok &= o[:, 1] <= y + h + BELOW_BOTTOM_PX
    for ox, oy, ow, oh in np.asarray(other_boxes, dtype=float).reshape(-1, 4):
        if oy + oh <= y + h:  # not nearer than this vehicle
            continue
        inside = (o[:, 0] >= ox) & (o[:, 0] <= ox + ow) & (o[:, 1] >= oy) & (o[:, 1] <= oy + oh)
        ok &= ~inside
    return ok


def _irls_line(u, v, design, thr, iters=12):
    """Robust fit v ~ design @ beta: points below the line by more than thr (tyres) get weight (thr / r)^2,
    points above by more than 3 thr Huber weights. Returns (beta, residuals, cost)."""
    wts = np.ones(len(u))
    beta = None
    for _ in range(iters):
        sw = np.sqrt(wts)
        beta, *_ = np.linalg.lstsq(design * sw[:, None], v * sw, rcond=None)
        r = v - design @ beta
        wts = np.ones(len(u))
        below = r > thr
        wts[below] = (thr / r[below]) ** 2
        above = r < -3 * thr
        wts[above] = 3 * thr / np.abs(r[above])
    r = v - design @ beta
    cost = float(np.sum(np.minimum(r * r, (3 * thr) ** 2)))
    return beta, r, cost


def body_line(uv, thr):
    """Robust one- or two-segment (continuous hinge) body line through undistorted outline points.

    Returns (v_line at each point, info dict: segments, break_u, cost)."""
    u, v = uv[:, 0], uv[:, 1]
    one = np.column_stack([np.ones_like(u), u])
    _, r1, c1 = _irls_line(u, v, one, thr)
    best = (c1, v - r1, {"segments": 1, "break_u": np.nan, "cost": c1})
    n = len(u)
    for q in np.linspace(5, 85, N_BREAKS):
        b = np.percentile(u, q)
        left = (u <= b).sum()
        if min(left, n - left) < max(0.15 * n, 4):
            continue
        two = np.column_stack([np.ones_like(u), u, np.maximum(0.0, u - b)])
        _, r2, c2 = _irls_line(u, v, two, thr)
        if c2 < best[0] and c2 < S.M7_TWO_SEG_GAIN * c1:
            best = (c2, v - r2, {"segments": 2, "break_u": float(b), "cost": c2})
    return best[1], best[2]


def _runs(flag, cols):
    """Maximal runs of consecutive True with consecutive integer columns -> list of index arrays."""
    out, cur = [], []
    for i in range(len(flag)):
        if flag[i] and (not cur or cols[i] == cols[cur[-1]] + 1):
            cur.append(i)
        elif flag[i]:
            out.append(np.array(cur))
            cur = [i]
        else:
            if cur:
                out.append(np.array(cur))
            cur = []
    if cur:
        out.append(np.array(cur))
    return out


def _split_at_dips(idx, depth, thr):
    """Split a run at an interior dip: two depth peaks separated by a drop of more than thr / 2 below both."""
    d = depth[idx]
    if len(d) < 5:
        return [idx]
    peaks = [i for i in range(1, len(d) - 1) if d[i] >= d[i - 1] and d[i] >= d[i + 1]]
    peaks = [0] * int(d[0] > d[1]) + peaks + [len(d) - 1] * int(d[-1] > d[-2])
    if len(peaks) < 2:
        return [idx]
    p = sorted(peaks, key=lambda i: -d[i])[:2]
    a, b = sorted(p)
    j = a + int(np.argmin(d[a : b + 1]))
    if d[j] < min(d[a], d[b]) - thr / 2 and 0 < j < len(d) - 1:
        return _split_at_dips(idx[: j + 1], depth, thr) + _split_at_dips(idx[j + 1 :], depth, thr)
    return [idx]


def _dedupe(cand: pd.DataFrame) -> pd.DataFrame:
    """One tyre can split into two runs: among accepted candidates closer than M7_DEDUP_TYRES diameters (u, undistorted),
    keep the deepest; the others become reject = 'duplicate'."""
    cand = cand.copy()
    acc = cand.index[cand["accepted"].astype(bool)].tolist()
    for i in sorted(acc, key=lambda j: -cand.at[j, "depth_px"]):
        if not cand.at[i, "accepted"]:
            continue
        for j in acc:
            if j != i and cand.at[j, "accepted"] and cand.at[j, "depth_px"] <= cand.at[i, "depth_px"] and \
                    abs(cand.at[j, "u_und"] - cand.at[i, "u_und"]) < S.M7_DEDUP_TYRES * cand.at[i, "tyre_px"]:
                cand.at[j, "accepted"], cand.at[j, "reject"] = False, "duplicate"
    return cand


def tyre_px(dh, cam, uv_dist, r_ft) -> np.ndarray:
    """Expected tyre diameter in undistorted px at each distorted pixel's own ground point (no pose)."""
    uv = np.atleast_2d(np.asarray(uv_dist, dtype=float))
    G = np.asarray(dh.image_to_world(uv), dtype=float).reshape(-1, 2)
    z = 2.0 * r_ft * S.FT
    p0 = cam.project(np.column_stack([G, np.zeros(len(G))]))
    p1 = cam.project(np.column_stack([G, np.full(len(G), z)]))
    return np.linalg.norm(p1 - p0, axis=1)


def extract(contours, box, other_boxes, dh, cam, cls, img_size=None):
    """Candidates of one row and a summary dict. `contours`: list of distorted pixel polygons."""
    summary = {"n_cols": 0, "n_trusted": 0, "segments": 0, "break_u": np.nan, "n_runs": 0, "n_cand": 0,
               "reason": ""}  # fmt: skip
    empty = pd.DataFrame(columns=CAND_COLUMNS)
    if not contours:
        summary["reason"] = "no_mask"
        return empty, summary, None
    mask = rasterise(contours, img_size)
    o = lower_outline(mask, box)
    summary["n_cols"] = len(o)
    ok = trusted_mask(o, box, other_boxes, img_size)
    o = o[ok]
    summary["n_trusted"] = len(o)
    if len(o) < S.M7_MIN_OUTLINE_COLS:
        summary["reason"] = "few_columns"
        return empty, summary, None
    cols = np.round(o[:, 0]).astype(int)
    uv = np.asarray(dh.undistort(o), dtype=float).reshape(-1, 2)
    h = float(box[3])
    thr = S.M7_PROTRUDE_PX * h / 100.0
    v_line, info = body_line(uv, thr)
    summary.update(segments=info["segments"], break_u=info["break_u"])
    depth = uv[:, 1] - v_line
    r_ft = S.M7_WHEEL_R_REF_FT[int(cls)]
    recs = []
    runs = [s for run in _runs(depth > thr, cols) for s in _split_at_dips(run, depth, thr)]
    summary["n_runs"] = len(runs)
    for run in runs:
        bottom = run[uv[run, 1] >= uv[run, 1].max() - 0.75]  # a raster bottom is flat: take its middle
        k = bottom[len(bottom) // 2]
        pu, pv = float(o[bottom, 0].mean()), float(o[k, 1])
        pu_und = float(uv[bottom, 0].mean())
        D = float(tyre_px(dh, cam, np.array([pu, pv]), r_ft)[0])
        width = float(uv[run[-1], 0] - uv[run[0], 0] + 1.0)
        ratio = width / D if D > 0 else np.nan
        ok_w = S.M7_TYRE_W_MIN <= ratio <= S.M7_TYRE_W_MAX
        reject = "" if ok_w else ("narrow" if ratio < S.M7_TYRE_W_MIN else "wide")
        if not reject and depth[k] > S.M7_MAX_DEPTH_TYRES * D:
            ok_w, reject = False, "deep"
        gate = max(S.M7_GATE_MIN_PX, S.M7_GATE_TYRES * D)
        recs.append([len(recs), pu, pv, pu_und, uv[k, 1], float(depth[k]), width, D, ratio, gate,
                     float(rho(gate / S.M7_SIGMA_WHEEL_PX)), bool(ok_w), reject])  # fmt: skip
    cand = pd.DataFrame(recs, columns=CAND_COLUMNS)
    cand = _dedupe(cand)
    summary["n_cand"] = int(cand["accepted"].sum()) if len(cand) else 0
    debug = {"outline": o, "uv": uv, "v_line": v_line, "depth": depth, "thr": thr, "dropped": None}
    return cand, summary, debug
