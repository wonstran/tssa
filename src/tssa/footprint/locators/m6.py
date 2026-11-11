"""M6: UrbanOmniDetect-2 cuboid bottom face -> measured quadrilateral + undirected axis (study m6.py; base design
section 6, run report "Large-vehicle handling" for the size gate)."""

from __future__ import annotations

import numpy as np

from ..config import FT
from ..geometry import face_geometry
from . import LocatorResult, register_locator

UV_COLS = ["gu0", "gv0", "gu1", "gv1", "gu2", "gv2", "gu3", "gv3"]
RAW_COLS = ["m6_raw_c1x", "m6_raw_c1y", "m6_raw_c2x", "m6_raw_c2y", "m6_raw_c3x", "m6_raw_c3y", "m6_raw_c4x",
            "m6_raw_c4y"]  # fmt: skip


@register_locator("M6", needs=("cuboids",))
def locate_m6(rows, ctx) -> LocatorResult:
    """Fail reasons: no_match (no row, IoU below m6_min_iou, non-finite corners); implausible_size (length or width
    below m6_size_gate of the class's nominal size: a collapsed face)."""
    cfg = ctx.config
    n = len(rows)
    res = LocatorResult.empty(n, "no_match")
    cub = ctx.tables["cuboids"]
    missing = [c for c in ["track", "frame", *UV_COLS] if c not in cub.columns]
    if missing:
        raise ValueError(f"cuboids table lacks column(s) {missing}")
    cub = cub.drop_duplicates(["track", "frame"]).set_index(["track", "frame"])
    hit = cub.reindex(list(zip(rows["track"].to_numpy(), rows["frame"].to_numpy())))
    uv = hit[UV_COLS].to_numpy(dtype=float)
    iou = hit["iou"].to_numpy(dtype=float) if "iou" in hit.columns else np.where(np.isfinite(uv).all(1), 1.0, np.nan)
    cls = rows["cls_track"].to_numpy()
    raw = np.full((n, 8), np.nan)
    for i in range(n):
        if not (iou[i] >= cfg.m6_min_iou) or not np.isfinite(uv[i]).all():
            continue
        corners, cen, L, W, axis = face_geometry(uv[i].reshape(4, 2), ctx.ground)
        nom_l, nom_w = cfg.size_of(cls[i])
        if L < cfg.m6_size_gate * nom_l * FT or W < cfg.m6_size_gate * nom_w * FT:
            res.fail_reason[i] = "implausible_size"
            continue
        res.ok[i], res.fail_reason[i] = True, ""
        res.x[i], res.y[i] = cen
        res.orient_deg[i], res.orient_kind[i] = axis, "axis"
        res.len_ft[i], res.wid_ft[i] = L / FT, W / FT
        raw[i] = corners.ravel()
    res.extras.update({c: raw[:, j] for j, c in enumerate(RAW_COLS)})
    res.extras["m6_len_ft"], res.extras["m6_wid_ft"] = res.len_ft.copy(), res.wid_ft.copy()
    res.extras["m6_iou"] = np.where(res.ok, iou, np.nan)
    return res
