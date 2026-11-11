"""M7: wheel-anchored rigid-body fit per bus or truck segment (M7 design rev 7 plus the M7 notes' decisions to
2026-11-10; footprint design section 7.6). Ported from the speed study's m7.py, m7_candidates.py and the job, cache
and table parts of m7_runner.py (snapshot docs/study-snapshot-2026-11-10.json).

Per segment of a cls 5/7 track: frozen observations from the mask contours (`frame_obs_from_mask`, the other boxes
of each frame from the full track table), an initial pose from the M2 points and psi_ref, `solve_track`, then rows:
wheels2 / wheel_front / wheel_rear / silhouette rows are native; kinematic rows carry the solve's pose and centre
covariance for admission (design 7.7); a failed solve fails every row with its gate reason.
Solves run in `workers` spawned processes (one BLAS thread each) and are cached under `<cache_dir>/m7solve/`.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import FT
from ...masks import keep_components
from .. import LocatorResult, register_locator
from . import solver as m7
from .settings import namespace, using

NATIVE_SUPPORT = ("wheels2", "wheel_front", "wheel_rear", "silhouette")
_CODE_FILES = ("solver.py", "candidates.py", "settings.py", "__init__.py")


def code_version() -> str:
    h = hashlib.sha256()
    for f in _CODE_FILES:
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()


def _interp_angle(deg: np.ndarray) -> np.ndarray:
    """Compass angles with NaN gaps filled by interpolating the unit vector (no 359 -> 0 sweep)."""
    deg = np.asarray(deg, float)
    if not np.isfinite(deg).any():
        return deg
    r = np.radians(deg)
    v = pd.DataFrame({"s": np.sin(r), "c": np.cos(r)}).interpolate(limit_direction="both").to_numpy(float)
    return np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360


def init_pose(m2_local: np.ndarray, psi: np.ndarray):
    """Initial centre (M2 points interpolated across rows without one) and heading (psi_ref interpolated; else the
    path direction), study m7_runner._init_pose. None when no row has an M2 point."""
    ok = np.isfinite(m2_local).all(axis=1)
    if not ok.any():
        return None
    c = m2_local.copy()
    c[~ok] = np.nan
    c0 = pd.DataFrame(c).interpolate(limit_direction="both").to_numpy(float)
    p = _interp_angle(psi)
    if not np.isfinite(p).all():
        d = np.gradient(c0, axis=0) if len(c0) > 1 else np.array([[0.0, 1.0]])
        p = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
    return c0, p


def _upd(h, a) -> None:
    a = np.asarray(a)
    h.update(str((a.dtype.str, a.shape)).encode())
    h.update(np.ascontiguousarray(a).tobytes())


def job_key(tr, ns: dict, camera_ok: bool, ident) -> str:
    """Hash of the solve's inputs: code, settings, identity, frames, initial pose, camera and frozen observations."""
    h = hashlib.sha256()
    h.update(code_version().encode())
    h.update(json.dumps(ns, sort_keys=True, default=str).encode())
    h.update(json.dumps([*ident, bool(camera_ok)], default=str).encode())
    _upd(h, tr.frames)
    for a in (tr.origin, tr.init_centre, tr.init_psi_deg, [tr.cls, tr.dt]):
        _upd(h, np.asarray(a, float))
    for k, v in sorted(vars(tr.cam).items()):
        h.update(k.encode())
        _upd(h, np.asarray(v, float)) if np.asarray(v).dtype.kind in "biuf" else h.update(repr(v).encode())
    for o in tr.obs:
        _upd(h, [o.frame])
        for a in (o.cand, o.kappa, o.env_u, o.env_v, o.ext, o.cand_s):
            _upd(h, np.asarray(a, float))
    return h.hexdigest()


def solve_one(tr, ns: dict, camera_ok: bool = True) -> dict:
    """One solve -> {"frames", "summary", "seconds"} or {"error", "traceback", "seconds"}. Never raises."""
    t0 = time.perf_counter()
    try:
        with using(ns):
            res = m7.solve_track(tr, camera_ok=camera_ok)
        out = {"frames": res["frames"], "summary": res["summary"]}
    except Exception as e:  # noqa: BLE001 - one bad segment must not lose the run; it is reported
        out = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    out["seconds"] = time.perf_counter() - t0
    return out


def _cache_path(cache_dir, key: str) -> Path:
    return Path(cache_dir) / "m7solve" / key[:2] / f"{key}.pkl"


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)


def solve_jobs(jobs: list[dict], ns: dict, cache_dir=None, workers: int = 1, camera_ok: bool = True) -> dict:
    """{key: result}: cached results reused; the rest solved longest first, in `workers` spawned processes."""
    out, todo = {}, []
    for j in jobs:
        p = _cache_path(cache_dir, j["key"]) if cache_dir is not None else None
        if p is not None and p.is_file():
            with open(p, "rb") as f:
                out[j["key"]] = {**pickle.load(f), "cached": True}
        elif j["key"] not in {t["key"] for t in todo}:
            todo.append(j)
    todo.sort(key=lambda j: -len(j["tr"].frames))

    def done(j, r):
        out[j["key"]] = {**r, "cached": False}
        if cache_dir is not None and "error" not in r:
            _save(_cache_path(cache_dir, j["key"]), r)

    if workers <= 1 or len(todo) <= 1:
        for j in todo:
            done(j, solve_one(j["tr"], ns, camera_ok))
        return out
    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    os.environ.update({k: "1" for k in saved})  # spawned children read these when they import numpy
    try:
        with ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=get_context("spawn")) as ex:
            fut = {ex.submit(solve_one, j["tr"], ns, camera_ok): j for j in todo}
            for f in as_completed(fut):
                done(fut[f], f.result())
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def _obs(rows: pd.DataFrame, ctx, cam, by_frame: dict) -> list:
    """Frozen observations of each row of one segment (empty for rows without a box or a mask)."""
    cfg = ctx.config
    masks = ctx.tables["masks"]
    xc, yc, wc, hc = cfg.box_cols
    out = []
    with using(_ns(ctx)):
        for r in rows.itertuples(index=False):
            f, t = int(r.frame), int(r.track)
            polys = masks.get((t, f)) if hasattr(masks, "get") else None
            box = (getattr(r, xc), getattr(r, yc), getattr(r, wc), getattr(r, hc))
            if r.fp_row_kind == "inserted" or not polys or not np.isfinite(box).all():
                out.append(m7.FrameObs(f))
                continue
            fr = by_frame.get(f)
            others = fr.loc[fr["track"] != t, [xc, yc, wc, hc]].to_numpy(float) if fr is not None else np.empty((0, 4))
            polys = keep_components([np.asarray(p, float) for p in polys], ctx.config.m7.M7_MIN_COMPONENT_FRAC)
            out.append(m7.frame_obs_from_mask(f, polys, box, others, ctx.ground, cam, int(r.cls_track)))
    return out


def _ns(ctx) -> dict:
    cfg = ctx.config
    return namespace(cfg.m7, ctx.ground.image_size, cfg.edge_px, cfg.size_ft, cfg.m7_classes)


def _summary_extras(sm: dict) -> dict:
    gf = sm.get("group_frames", {})
    return {
        "m7_subtype": sm.get("subtype", ""), "m7_subtype_ambiguous": sm.get("subtype_ambiguous", np.nan),
        "m7_wheelbase_ft": sm.get("wheelbase_ft", np.nan), "m7_len_ft": sm.get("length_ft", np.nan),
        "m7_wid_ft": sm.get("width_ft", np.nan), "m7_front_oh_ft": sm.get("front_oh_ft", np.nan),
        "m7_rear_oh_ft": sm.get("rear_oh_ft", np.nan), "m7_height_ft": sm.get("height_ft", np.nan),
        "m7_axle_config": sm.get("axle_config", ""), "m7_axle_grouping": sm.get("axle_grouping", ""),
        "m7_group_frames": json.dumps(gf, sort_keys=True) if sm else "",
        "m7_len_source": sm.get("L_source", ""), "m7_wid_source": sm.get("W_source", ""),
        "m7_front_state": sm.get("front_state", ""), "m7_front_source": sm.get("front_source", ""),
        "m7_reversing": "" if sm.get("reversing") is None else str(bool(sm["reversing"])),
        "m7_articulated_suspect": bool(sm.get("articulated_suspect", False)), "m7_w_held": bool(sm.get("w_held", False)),
    }  # fmt: skip


@register_locator("M7", needs=("masks",), scope="segment", classes=(5, 7))
def locate_m7(rows: pd.DataFrame, ctx) -> LocatorResult:
    """See the module docstring. Fail reasons: the solve's gate reason, camera_invalid, too_few_frames,
    no_initial_pose, solve_error; kinematic rows of a passed solve are supplied with support "kinematic"."""
    cfg = ctx.config
    n = len(rows)
    res = LocatorResult.empty(n)
    ex = {k: np.full(n, np.nan) for k in ("m7_x", "m7_y", "m7_psi_deg", "m7_n_inliers", "m7_n_outliers",
                                           "m7_sd_along_ft", "m7_sd_across_ft", "m7_sd_heading_deg",
                                           "m7_cov_centre_ee", "m7_cov_centre_en", "m7_cov_centre_nn")}  # fmt: skip
    obj = {k: np.full(n, "", dtype=object) for k in ("m7_support", "m7_near_side", "m7_fail_reason")}
    solve_ok = np.zeros(n, bool)
    per: dict = {}
    res.seg_len_ft, res.seg_wid_ft = np.full(n, np.nan), np.full(n, np.nan)
    res.seg_dims_source = np.full(n, "", dtype=object)
    res.cov = np.full((n, 2, 2), np.nan)
    cam = ctx.camera()
    origin = np.asarray(ctx.origin, float)
    ns = _ns(ctx)
    by_frame = {int(f): g for f, g in ctx.all_boxes.groupby("frame")}
    groups = [np.asarray(v) for v in rows.groupby(["track", "seg"], sort=True).indices.values()]
    jobs, reason = [], {}
    for g in groups:
        sub = rows.iloc[g]
        frames = sub["frame"].to_numpy(int)
        ident = (int(sub["track"].iloc[0]), int(sub["seg"].iloc[0]), int(sub["cls_track"].iloc[0]))
        if cam is None:
            reason[id(g)] = "camera_invalid"
            continue
        if len(frames) > 1 and not (np.diff(frames) == 1).all():
            raise ValueError(f"M7: segment {ident[:2]} does not have consecutive frames")
        if len(frames) < ns["M7_MIN_SUPPORT_FRAMES"]:
            reason[id(g)] = "too_few_frames"
            continue
        m2 = sub[["m2_x", "m2_y"]].to_numpy(float) - origin
        init = init_pose(m2, sub["ref_heading_deg"].to_numpy(float))
        if init is None:
            reason[id(g)] = "no_initial_pose"
            continue
        tr = m7.Track(frames, _obs(sub, ctx, cam, by_frame), cam, origin, init[0], init[1], ident[2], 1.0 / cfg.fps)
        jobs.append({"g": g, "tr": tr, "ident": ident, "key": job_key(tr, ns, True, ident)})
    results = solve_jobs(jobs, ns, ctx.cache_dir, ctx.workers, True)
    solves = []
    for g in groups:
        if id(g) in reason:
            res.fail_reason[g] = reason[id(g)]
            obj["m7_fail_reason"][g] = reason[id(g)]
    for j in jobs:
        g, r = j["g"], results[j["key"]]
        if "error" in r:
            res.fail_reason[g] = "solve_error"
            obj["m7_fail_reason"][g] = "solve_error"
            solves.append({"track": j["ident"][0], "seg": j["ident"][1], "fail_reason": "solve_error",
                           "error": r["error"], "seconds": r.get("seconds", np.nan)})  # fmt: skip
            continue
        sm, fr = r["summary"], r["frames"].reset_index(drop=True)
        fail = sm["fail_reason"]
        passed = fail == ""
        solves.append({"track": j["ident"][0], "seg": j["ident"][1], "cls": j["ident"][2], "n_frames": len(g),
                       "seconds": r.get("seconds", np.nan), "cached": r.get("cached", False),
                       **{k: (json.dumps(v, default=float) if isinstance(v, dict) else v) for k, v in sm.items()}})  # fmt: skip
        pos = fr.set_index("frame").reindex(rows["frame"].to_numpy()[g])
        cen = pos[["e_nat", "n_nat"]].to_numpy(float) + origin
        sup = pos["m7_support"].fillna("").to_numpy(dtype=object)
        cov = pos[["m7_cov_centre_ee", "m7_cov_centre_en", "m7_cov_centre_en", "m7_cov_centre_nn"]].to_numpy(float)
        ex["m7_x"][g], ex["m7_y"][g], ex["m7_psi_deg"][g] = cen[:, 0], cen[:, 1], pos["psi_deg"].to_numpy(float)
        for k in ("m7_n_inliers", "m7_n_outliers", "m7_sd_along_ft", "m7_sd_across_ft", "m7_sd_heading_deg",
                  "m7_cov_centre_ee", "m7_cov_centre_en", "m7_cov_centre_nn"):
            ex[k][g] = pos[k].to_numpy(float)
        obj["m7_support"][g], obj["m7_near_side"][g] = sup, pos["near_side"].astype(object).to_numpy()
        obj["m7_fail_reason"][g] = fail
        solve_ok[g] = passed
        for k, v in _summary_extras(sm).items():
            per.setdefault(k, np.full(n, np.nan if isinstance(v, float) else "", dtype=float if isinstance(v, float) else object))
            per[k][g] = v
        if not passed:
            res.fail_reason[g] = fail
            continue
        res.seg_len_ft[g], res.seg_wid_ft[g] = sm["length_ft"], sm["width_ft"]
        res.seg_dims_source[g] = sm.get("dims_source", "m7_measured")
        native = np.isin(sup, NATIVE_SUPPORT)
        kin = sup == "kinematic"
        ok = (native | kin) & np.isfinite(cen).all(axis=1)
        res.ok[g] = ok
        res.fail_reason[g] = np.where(ok, "", "no_observation")
        res.x[g], res.y[g] = cen[:, 0], cen[:, 1]
        res.orient_deg[g], res.orient_kind[g] = pos["psi_deg"].to_numpy(float), "directed"
        res.support[g] = np.where(kin, "kinematic", "native").astype(object)
        R = cov.reshape(-1, 2, 2) * cfg.kin_meas_scale**2 + (cfg.kin_sd_floor_ft * FT) ** 2 * np.eye(2)
        res.cov[g] = np.where(kin[:, None, None], R, np.nan)
    res.extras.update(ex)
    res.extras.update(obj)
    res.extras.update(per)
    res.extras["m7_solve_ok"] = solve_ok
    ctx.info.setdefault("m7_solves", []).extend(solves)
    return res
