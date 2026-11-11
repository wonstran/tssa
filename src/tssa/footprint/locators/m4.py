"""M4: CarFusion wheel keypoints -> body centre and directed heading (ported from the speed study's m4.py; base design
section 6.1).

Keypoint schema: 0 right-front, 1 left-front, 2 right-back, 3 left-back; only the same-side pairs right (0, 2) and
left (1, 3) are used. Keypoints are distorted pixels (like the bbox); `to_ground` maps them to world contacts
(`ground.image_to_world` for contact landmarks, `hub_to_contact` through the camera for hub landmarks, None when hub
landmarks meet a camera that failed its consistency check: camera_invalid). Image heights ("lower in the image", the
bbox-bottom cue) use the ground contacts re-projected with `world_to_image`, in the same pixels as the bbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import FT, FootprintConfig
from ..geometry import away_dir, compass_deg, diff_directed, diff_undirected, rot_ccw
from . import LocatorResult, register_locator

PAIRS = {"R": (0, 2), "L": (1, 3)}  # label side -> (front index, back index)
KP_COLUMNS = ["u0", "v0", "u1", "v1", "u2", "v2", "u3", "v3"]
_LABEL_CASE = {(False, False): "as_labelled", (True, False): "fb_swapped", (True, True): "perm180",
               (False, True): "lr_swapped"}  # fmt: skip


def _nan2():
    return np.full(2, np.nan)


@dataclass
class M4Result:
    valid: bool = False
    fail_reason: str = ""  # "" | no_match | no_valid_pair | side_ambiguous | camera_invalid | no_heading | no_prior
    centre: np.ndarray = field(default_factory=_nan2)  # world (2,)
    d_front: np.ndarray = field(default_factory=_nan2)  # unit (2,), after the front correction
    pair_side: str = ""  # label side of the chosen pair: "R" (0, 2) or "L" (1, 3)
    label_case: str = ""  # as_labelled | fb_swapped | perm180 | lr_swapped
    interior_source: str = ""  # other_pair | camera_away | ""
    side_reject: str = ""  # "" | nonfinite | degenerate | wheelbase | axis | parallel | lateral | longitudinal
    front_flipped: bool = False
    wheelbase_ft: float = np.nan  # measured length of the chosen pair
    track_ft: float = np.nan  # measured |s|, only when the other pair is credible (both pairs pass)


def _cfg(cfg):
    return FootprintConfig() if cfg is None else cfg


def _pair_accept(F, B, psi, wb_m, cfg):
    """Own acceptance of a pair: finite, length within +-tol of the wheelbase, axis within the axis tolerance."""
    if not (np.isfinite(F).all() and np.isfinite(B).all()):
        return False
    L = np.linalg.norm(F - B)
    if abs(L - wb_m) > cfg.m4_wheelbase_tol * wb_m:
        return False
    return bool(diff_undirected(compass_deg(F - B), psi) <= cfg.m4_axis_tol_deg)


def _credibility(F, B, uvo, axle, d_front, n_left, psi, wb_m, tr_m, cfg):
    """First failed credibility check of the other pair ("" if credible) and its lateral offset s (m)."""
    if not (np.isfinite(F).all() and np.isfinite(B).all() and np.isfinite(uvo).all()):
        return "nonfinite", np.nan
    L = np.linalg.norm(F - B)
    if L <= cfg.m4_min_contact_sep_ft * FT:
        return "degenerate", np.nan
    if abs(L - wb_m) > cfg.m4_wheelbase_tol * wb_m:
        return "wheelbase", np.nan
    axis = compass_deg(F - B)
    if diff_undirected(axis, psi) > cfg.m4_axis_tol_deg:
        return "axis", np.nan
    if diff_undirected(axis, compass_deg(d_front)) > cfg.m4_parallel_tol_deg:
        return "parallel", np.nan
    off = (F + B) / 2.0 - axle
    s = float(off @ n_left)
    if abs(abs(s) - tr_m) > cfg.m4_track_tol * tr_m:
        return "lateral", s
    if abs(float(off @ d_front)) > cfg.m4_long_offset_frac * wb_m:
        return "longitudinal", s
    return "", s


def solve_m4(kp_uv, box_xywh, psi_ref_deg, cls, dh, to_ground, cfg: FootprintConfig | None = None) -> M4Result:
    """Body centre (world) and directed heading from four wheel keypoints [RF, LF, RB, LB] (distorted px)."""
    cfg = _cfg(cfg)
    if to_ground is None:
        return M4Result(fail_reason="camera_invalid")
    if int(cls) not in cfg.priors_ft:
        return M4Result(fail_reason="no_prior")
    kp_uv = np.asarray(kp_uv, dtype=float).reshape(4, 2)
    psi = float(psi_ref_deg)
    pri = cfg.priors_ft[int(cls)]
    wb_m, tr_m = pri["wheelbase"] * FT, pri["track"] * FT
    x, y, w, h = (float(v) for v in box_xywh)

    G = np.full((4, 2), np.nan)
    fin = np.isfinite(kp_uv).all(axis=1)
    if fin.any():
        G[fin] = np.asarray(to_ground(kp_uv[fin]), dtype=float).reshape(-1, 2)
    gfin = np.isfinite(G).all(axis=1)
    v_img = np.full(4, np.nan)  # image v of the ground contacts (same pixels as the bbox)
    if gfin.any():
        v_img[gfin] = np.asarray(dh.world_to_image(G[gfin]), dtype=float).reshape(-1, 2)[:, 1]

    # 1. Acceptance; the lower passing pair (larger mean v) is chosen.
    passing = [(side, np.mean(v_img[list(ij)])) for side, ij in PAIRS.items()
               if np.isfinite(psi) and _pair_accept(G[ij[0]], G[ij[1]], psi, wb_m, cfg)]  # fmt: skip
    if not passing:
        return M4Result(fail_reason="no_valid_pair")
    side = max(passing, key=lambda t: t[1] if np.isfinite(t[1]) else -np.inf)[0]
    other = "L" if side == "R" else "R"
    (fi, bi), (ofi, obi) = PAIRS[side], PAIRS[other]
    F, B = G[fi], G[bi]
    res = M4Result(pair_side=side, wheelbase_ft=float(np.linalg.norm(F - B) / FT))

    # 2. Front.
    d_front = (F - B) / np.linalg.norm(F - B)
    res.front_flipped = bool(diff_directed(compass_deg(d_front), psi) > 90.0)
    if res.front_flipped:
        d_front = -d_front
    res.d_front = d_front
    n_left = rot_ccw(d_front)
    axle = (F + B) / 2.0

    # 3. Interior side: credible other pair, else camera-away, else ambiguous.
    reject, s = _credibility(G[ofi], G[obi], kp_uv[[ofi, obi]], axle, d_front, n_left, psi, wb_m, tr_m, cfg)
    res.side_reject = reject
    sign = 0.0
    if reject == "":
        sign = float(np.sign(s))
        res.interior_source = "other_pair"
        res.track_ft = abs(s) / FT
    else:
        v_ch, v_ot = v_img[[fi, bi]], v_img[[ofi, obi]]
        depth_ok = (not np.isfinite(v_ot).all()) or bool(v_ch.mean() >= v_ot.mean() + cfg.m4_near_dv_px)
        bottom_ok = bool(np.any(np.abs((y + h) - v_ch) <= cfg.m4_near_bottom_frac * h))  # two-sided (R8)
        if depth_ok and bottom_ok:
            a = away_dir(dh, np.asarray(dh.world_to_image(axle[None, :]), dtype=float))[0]
            dot = float(a @ n_left)
            if np.isfinite(dot) and abs(dot) >= cfg.m4_camera_away_dot:
                sign = float(np.sign(dot))
                res.interior_source = "camera_away"
    if sign == 0.0:
        res.fail_reason = "side_ambiguous"
        return res

    # 4. Centre. 5. Label case: interior +n_left means the pair is physically the vehicle's right side.
    res.centre = axle + (tr_m / 2.0) * sign * n_left + ((pri["front_oh"] - pri["rear_oh"]) * FT / 2.0) * d_front
    geo_side = "R" if sign > 0 else "L"
    res.label_case = _LABEL_CASE[(res.front_flipped, geo_side != side)]
    res.valid = True
    return res


def to_ground_for(ctx, cls):
    """The contact function of a class: the homography for contact landmarks, the camera's hub drop for hubs."""
    cfg = ctx.config
    if cfg.wheel_landmark == "contact":
        return lambda uv: ctx.ground.image_to_world(uv)
    cam = ctx.camera()
    if cam is None:
        return None
    from ...transform.camera import hub_to_contact

    r_m = cfg.priors_ft.get(int(cls), {}).get("wheel_r", np.nan) * FT
    return lambda uv: hub_to_contact(cam, ctx.ground.undistort(uv), r_m)


def front_conflict(track, valid, flipped, min_rows: int, share: float) -> np.ndarray:
    """Per row: its track has at least `min_rows` native-valid rows and more than `share` of them front-flipped."""
    df = pd.DataFrame({"track": track, "ok": valid, "f": flipped & valid})
    g = df[df["ok"]].groupby("track")["f"].agg(["size", "mean"])
    flag = (g["size"] >= min_rows) & (g["mean"] > share)
    return df["track"].map(flag).fillna(False).to_numpy(dtype=bool)


@register_locator("M4", needs=("keypoints",), aliases="wheel_keypoints")
def locate_m4(rows, ctx) -> LocatorResult:
    """Fail reasons: no_match, no_heading, camera_invalid, no_valid_pair, side_ambiguous, no_prior."""
    cfg = ctx.config
    n = len(rows)
    res = LocatorResult.empty(n, "no_match")
    kp = ctx.tables["keypoints"]
    missing = [c for c in ["track", "frame", *KP_COLUMNS] if c not in kp.columns]
    if missing:
        raise ValueError(f"keypoints table lacks column(s) {missing}")
    hit = kp.drop_duplicates(["track", "frame"]).set_index(["track", "frame"])
    hit = hit.reindex(list(zip(rows["track"].to_numpy(), rows["frame"].to_numpy())))
    kp_arr = hit[KP_COLUMNS].to_numpy(dtype=float)
    present = hit[KP_COLUMNS].notna().any(axis=1).to_numpy()
    boxes = rows[list(cfg.box_cols)].to_numpy(dtype=float)
    psi = rows["ref_heading_deg"].to_numpy(dtype=float)
    cls = rows["cls_track"].to_numpy()
    diag = {c: np.full(n, "", dtype=object) for c in ("m4_pair_side", "m4_label_case", "m4_interior_source",
                                                       "m4_side_reject")}  # fmt: skip
    flipped = np.zeros(n, bool)
    wb, tr = np.full(n, np.nan), np.full(n, np.nan)
    grounds: dict = {}
    for i in range(n):
        if not present[i]:
            continue
        if not np.isfinite(psi[i]):
            res.fail_reason[i] = "no_heading"
            continue
        c = int(cls[i])
        if c not in grounds:
            grounds[c] = to_ground_for(ctx, c)
        r = solve_m4(kp_arr[i].reshape(4, 2), boxes[i], psi[i], c, ctx.ground, grounds[c], cfg)
        res.fail_reason[i] = r.fail_reason
        diag["m4_pair_side"][i], diag["m4_label_case"][i] = r.pair_side, r.label_case
        diag["m4_interior_source"][i], diag["m4_side_reject"][i] = r.interior_source, r.side_reject
        flipped[i], wb[i], tr[i] = r.front_flipped, r.wheelbase_ft, r.track_ft
        if r.valid:
            res.ok[i] = True
            res.x[i], res.y[i] = r.centre
            res.orient_deg[i], res.orient_kind[i] = float(compass_deg(r.d_front)), "directed"
    res.extras.update(diag)
    res.extras["m4_front_flipped"] = flipped
    res.extras["m4_wheelbase_ft"], res.extras["m4_track_ft"] = wb, tr
    res.extras["m4_front_conflict"] = front_conflict(rows["track"].to_numpy(), res.ok, flipped,
                                                     cfg.m4_front_conflict_min_rows, cfg.m4_front_conflict_share)  # fmt: skip
    return res
