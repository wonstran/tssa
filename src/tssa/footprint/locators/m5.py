"""M5: lower mask outline -> nominal-size rectangle fit (ported from the speed study's m5.py; base design section 6).

- R19: the fit starts from fixed axes 0/45/90/135 deg and never sees psi_ref.
- R20: a one-sided outline is mirror-ambiguous; after the fit the centre must lie on the away-from-camera side of the
  outline, else it is reflected across the fitted near side (`mirror_away`; diagnostic `m5_flipped`).
- R23: outline points below the bbox bottom (v > y + h + 1 px) are dropped before fitting.
Change from the study: the outline comes from the mask contours of the `masks` table (one SAM pass serves M5 and M7,
design section 15), rasterised with every stored component.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ..config import FT
from ..geometry import away_dir, project, to_undirected
from ..masks import lower_outline
from . import LocatorResult, register_locator

SIDE_NEAR_FT = 0.5
SIDE_MIN_PTS = 3
START_DEG = (0.0, 45.0, 90.0, 135.0)  # fixed fit starts (R19)
BELOW_BOTTOM_PX = 1.0  # outline points with v > y + h + this are dropped (R23)


def _boundary_dist(pts, p, L, W):
    e, n, th = p
    t = np.array([np.sin(th), np.cos(th)])  # long axis (compass heading)
    s = np.array([-t[1], t[0]])
    d = pts - np.array([e, n])
    a, b = d @ t, d @ s
    qa, qb = np.abs(a) - L / 2.0, np.abs(b) - W / 2.0
    outside = np.hypot(np.maximum(qa, 0), np.maximum(qb, 0))
    inside = np.minimum(np.maximum(qa, qb), 0)
    return outside - inside  # |signed distance| = outside + (-inside)


def _sides_covered(pts, c, th, L, W) -> int:
    t = np.array([np.sin(th), np.cos(th)])
    s = np.array([-t[1], t[0]])
    d = pts - c
    a, b = d @ t, d @ s
    hl, hw = L / 2.0, W / 2.0
    tol = SIDE_NEAR_FT * FT
    cnt = 0
    for fixed, other, hf, ho in ((a, b, hl, hw), (b, a, hw, hl)):
        for sign in (1, -1):
            near = (np.abs(fixed - sign * hf) <= tol) & (np.abs(other) <= ho + tol)
            cnt += int(near.sum() >= SIDE_MIN_PTS)
    return cnt


def fit_rect(points_EN, L_m, W_m, init_centre):
    """(centre, axis_deg in [0, 180), resid_ft RMS of boundary distance, sides_covered); fixed starts (R19)."""
    pts = np.asarray(points_EN, dtype=float)
    c0 = np.asarray(init_centre, dtype=float)
    best = None
    for a0 in START_DEG:
        p0 = np.array([c0[0], c0[1], np.radians(a0)])
        sol = least_squares(lambda p: _boundary_dist(pts, p, L_m, W_m), p0, method="lm", xtol=1e-10, ftol=1e-10,
                            max_nfev=400)  # fmt: skip
        if best is None or sol.cost < best.cost:
            best = sol
    c = best.x[:2]
    th = best.x[2]
    return (c, float(to_undirected(np.degrees(th))), *_resid_sides(pts, c, th, L_m, W_m))


def _resid_sides(pts, c, th, L_m, W_m):
    d = _boundary_dist(pts, np.array([c[0], c[1], th]), L_m, W_m)
    return float(np.sqrt(np.mean(d**2))) / FT, _sides_covered(pts, c, th, L_m, W_m)


def mirror_away(pts, centre, axis_deg, L_m, W_m, away) -> tuple[np.ndarray, bool]:
    """(centre, flipped): unchanged when (centre - outline mean) . away > 0, else reflected across the rectangle side
    closest to the outline points (R20)."""
    pts = np.asarray(pts, dtype=float)
    c = np.asarray(centre, dtype=float)
    if float((c - pts.mean(axis=0)) @ np.asarray(away, dtype=float)) > 0:
        return c, False
    th = np.radians(axis_deg)
    t = np.array([np.sin(th), np.cos(th)])
    s = np.array([-t[1], t[0]])
    best, offset = np.inf, None
    for normal, h, along, ha in ((s, W_m / 2.0, t, L_m / 2.0), (t, L_m / 2.0, s, W_m / 2.0)):
        for sign in (1.0, -1.0):
            off = sign * h * normal  # centre -> side midpoint
            d = pts - (c + off)
            a = np.clip(d @ along, -ha, ha)
            dist = float(np.mean(np.linalg.norm(d - a[:, None] * along, axis=1)))
            if dist < best:
                best, offset = dist, off
    return c + 2.0 * offset, True


def solve_m5(outline_px, box_xywh, ground, L_m, W_m, init, min_cols: int = 8, max_resid_ft: float = 1.0) -> dict:
    """One row: outline pixels -> dict(ok, reason, centre, axis, cols, resid_ft, sides, flipped)."""
    out = dict(ok=False, reason="", centre=np.full(2, np.nan), axis=np.nan, cols=0, resid_ft=np.nan, sides=0,
               flipped=False)  # fmt: skip
    if outline_px is None:
        out["reason"] = "no_match"
        return out
    x, y, w, h = (float(v) for v in box_xywh)
    o = np.asarray(outline_px, dtype=float).reshape(-1, 2)
    o = o[~(o[:, 1] > y + h + BELOW_BOTTOM_PX)]  # R23
    out["cols"] = len(o)
    if len(o) < min_cols:
        out["reason"] = "few_columns"
        return out
    pts = project(ground, o)
    try:
        c, ax, r, s = fit_rect(pts, L_m, W_m, init)
        c, flipped = mirror_away(pts, c, ax, L_m, W_m, away_dir(ground, o.mean(axis=0))[0])
        if flipped:
            r, s = _resid_sides(pts, c, np.radians(ax), L_m, W_m)
    except (ValueError, FloatingPointError):
        out["reason"] = "fit_residual"
        return out
    out.update(resid_ft=r, sides=s, flipped=flipped)
    if not np.isfinite(r) or r > max_resid_ft:
        out["reason"] = "fit_residual"
        return out
    out.update(ok=True, centre=c, axis=ax)
    return out


@register_locator("M5", needs=("masks",), aliases="sam_outline")
def locate_m5(rows, ctx) -> LocatorResult:
    """Fail reasons: no_match (no mask), few_columns, fit_residual."""
    cfg = ctx.config
    n = len(rows)
    res = LocatorResult.empty(n)
    masks = ctx.tables["masks"]
    boxes = rows[list(cfg.box_cols)].to_numpy(dtype=float)
    init = rows[["m2_x", "m2_y"]].to_numpy(dtype=float)
    cls = rows["cls_track"].to_numpy()
    tr, fr = rows["track"].to_numpy(), rows["frame"].to_numpy()
    cols, resid, sides = np.zeros(n, int), np.full(n, np.nan), np.zeros(n, int)
    flipped = np.zeros(n, bool)
    for i in range(n):
        key = (int(tr[i]), int(fr[i]))
        raw = getattr(masks, "outlines", {}).get(key)
        polys = masks.get(key) if hasattr(masks, "get") else None
        outline = raw if raw is not None else (lower_outline(list(polys), boxes[i]) if polys else None)
        L, W = (v * FT for v in cfg.size_of(cls[i]))
        r = solve_m5(outline, boxes[i], ctx.ground, L, W, init[i], cfg.m5_min_cols, cfg.m5_max_resid_ft)
        cols[i], resid[i], sides[i], flipped[i] = r["cols"], r["resid_ft"], r["sides"], r["flipped"]
        if r["ok"]:
            res.ok[i] = True
            res.x[i], res.y[i] = r["centre"]
            res.orient_deg[i], res.orient_kind[i] = r["axis"], "axis"
        else:
            res.fail_reason[i] = r["reason"]
    res.extras.update(m5_cols=cols, m5_resid_ft=resid, m5_sides=sides, m5_flipped=flipped)
    return res
