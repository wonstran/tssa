"""M1 (bbox bottom-centre), M2 (half a nominal width away from the camera), M3 (nominal rectangle at the reference
heading fitted to the bbox). Ported from the speed study's geometry.py and m3.py (base design section 6)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ..config import FT
from ..geometry import rect_corners
from . import LocatorResult, register_locator


@register_locator("M1", point_only=True)
def locate_m1(rows, ctx) -> LocatorResult:
    """The world point under the bbox bottom-centre: the foot of the box, not the body centre."""
    res = LocatorResult.empty(len(rows))
    xy = rows[["m1_x", "m1_y"]].to_numpy(dtype=float)
    ok = np.isfinite(xy).all(axis=1)
    res.ok, res.x, res.y = ok, xy[:, 0], xy[:, 1]
    res.fail_reason[~ok] = "projection"
    return res


@register_locator("M2")
def locate_m2(rows, ctx) -> LocatorResult:
    """M1 moved half the class's nominal width away from the camera (computed for every box row by the printer)."""
    res = LocatorResult.empty(len(rows))
    xy = rows[["m2_x", "m2_y"]].to_numpy(dtype=float)
    ok = np.isfinite(xy).all(axis=1)
    res.ok, res.x, res.y = ok, xy[:, 0], xy[:, 1]
    res.fail_reason[~ok] = "projection"
    return res


def solve_m3(ground, box_xywh, heading_deg, L_m, W_m, init, resid_ok_px: float = 0.5):
    """(centre, corners, ok, resid_px): the nominal rectangle at `heading_deg` whose projection has its lowest point
    on the bbox bottom and its horizontal midpoint at the bbox centre (distorted pixels, like the bbox)."""
    nan2, nan42 = np.full(2, np.nan), np.full((4, 2), np.nan)
    init = np.asarray(init, dtype=float)
    if not np.isfinite(heading_deg) or not np.isfinite(init).all():
        return nan2, nan42, False, np.nan
    x, y, w, h = (float(v) for v in box_xywh)
    tv, tu = y + h, x + w / 2.0

    def resid(d):
        uv = np.asarray(ground.world_to_image(rect_corners(init + d, heading_deg, L_m, W_m)), dtype=float)
        return np.array([uv[:, 1].max() - tv, (uv[:, 0].min() + uv[:, 0].max()) / 2.0 - tu])

    try:
        r0 = resid(np.zeros(2))
        if not np.isfinite(r0).all():
            return nan2, nan42, False, np.nan
        sol = least_squares(resid, np.zeros(2), method="lm", x_scale=1.0, xtol=1e-12, ftol=1e-12, gtol=1e-12,
                            max_nfev=200)  # fmt: skip
    except (ValueError, FloatingPointError):
        return nan2, nan42, False, np.nan
    r = float(np.max(np.abs(sol.fun)))
    if not np.isfinite(r):
        return nan2, nan42, False, np.nan
    centre = init + sol.x
    return centre, rect_corners(centre, heading_deg, L_m, W_m), bool(r < resid_ok_px), r


@register_locator("M3")
def locate_m3(rows, ctx) -> LocatorResult:
    """M3 at the reference heading; the orientation is imposed (directed psi_ref), not measured."""
    cfg = ctx.config
    n = len(rows)
    res = LocatorResult.empty(n)
    boxes = rows[list(cfg.box_cols)].to_numpy(dtype=float)
    psi = rows["ref_heading_deg"].to_numpy(dtype=float)
    init = rows[["m2_x", "m2_y"]].to_numpy(dtype=float)
    cls = rows["cls_track"].to_numpy()
    resid = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(psi[i]):
            res.fail_reason[i] = "no_heading"
            continue
        L, W = (v * FT for v in cfg.size_of(cls[i]))
        c, _, ok, r = solve_m3(ctx.ground, boxes[i], psi[i], L, W, init[i], cfg.m3_resid_px)
        resid[i] = r
        if ok:
            res.ok[i], res.x[i], res.y[i] = True, c[0], c[1]
            res.orient_deg[i], res.orient_kind[i] = psi[i], "directed"
        else:
            res.fail_reason[i] = "no_solution"
    res.extras["m3_resid_px"] = resid
    return res
