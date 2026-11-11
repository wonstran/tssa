"""NW-clip validation of tssa.footprint against the speed study (design docs/2026-11-10-footprint-design.md 14.3).

    python nw_clip_validation.py --study <extracted snapshot>/speed --root <footprint output root> --out <dir> \
        --part native|m7|e2e|grid

- study code is imported from the extracted snapshot archive (docs/study-snapshot-2026-11-10.json), never the live
  folder; inputs are the study's pinned copies under <root>/inputs; detections are the study's caches converted to
  the tssa tables (cuboids, keypoints, masks);
- native: M1, M2, M3, M4, M6 per-row native points on the study's rows with the study's display psi_ref injected,
  M5 through solve_m5 on the study's own outlines, the M5 outline from mask contours, the M4 counts on eligible rows
  with the metric psi_ref, and the camera;
- m7: the study's display jobs (build_jobs on the study grid), each converted and solved fresh by the port, against
  the study's run report of 20261110_161113_s0 (m7_tracks.csv);
- m7study: the study's own solver on the same jobs in this environment; m7compare: port against m7study;
- trap: e2e's speed trap against the study's speed_trap.csv, on all and on common vehicles;
- e2e: the port end to end (compare M1-M4, M6 with the M2 fallback, and estimate with the default chains) and the
  speed trap of check_speed_trap.py on the port's trajectories;
- grid: the port's segmentation against the study's grid.
Each part writes <out>/<part>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FT = 0.3048
REF_RUN = "20261110_161113_s0"
NATIVE_RUN = "20261107_222536_s0"


def study_env(study: Path, root: Path) -> dict:
    """The study's grid, natives, reference heading and camera, computed by its own code on its pinned inputs."""
    sys.path.insert(0, str(study))
    import camera
    import geometry
    import grid as G
    import inputs
    import run_footprint as R
    import settings as S
    import smooth
    from smoothlib import ground

    inp = root / "inputs"
    name = {k: Path(v).name for k, v in R.SOURCES.items()}
    tracks = inputs.load_tracks(inp / name["tracks"])
    labels = inputs.load_labels(inp / name["audit"], S.CLIP)
    dh = ground.load_homography(inp / name["homography"], S.IMG_SIZE)
    gcps = inputs.load_gcps(inp / name["gcps"])
    zone = geometry.analysis_zone(gcps, S.ZONE_BUFFER_FT)
    grid = G.build(tracks, labels, dh, zone)
    origin = np.asarray(grid.attrs["origin"], float)
    m1n, m2n = R.native_points(grid, dh, origin)
    ref = smooth.reference_heading(grid, m2n)
    cam = camera.fit_camera(dh, zone)
    cons = camera.consistency(cam, dh, zone)
    return dict(S=S, R=R, tracks=tracks, labels=labels, dh=dh, gcps=gcps, zone=zone, grid=grid, origin=origin,
                m1n=m1n, m2n=m2n, ref=ref, cam=cam, cons=cons, inp=inp, name=name)  # fmt: skip


def caches(root: Path) -> dict:
    """The study's detection caches (the keys its current code reads) as tssa tables."""
    from tssa.footprint.masks import MaskTable

    c = root / "cache"
    m6 = pd.read_csv(c / "m6" / "9a1f4b79288939c9" / "rows.csv")
    m4 = pd.read_csv(c / "m4" / "6a1f0df0d8de1344" / "rows.csv")
    z = np.load(c / "m5" / "62a0ce39913f7eb0" / "outlines.npz")
    t, f, n, p = z["track"], z["frame"], z["count"], z["pts"]
    ends = np.cumsum(n)
    m5 = {(int(a), int(b)): p[e - k:e] for a, b, k, e in zip(t, f, n, ends)}
    return dict(cuboids=m6[m6["matched"] == 1].drop(columns="matched").reset_index(drop=True),
                keypoints=m4[m4["matched"] == 1].drop(columns="matched").reset_index(drop=True),
                masks=MaskTable.load_npz(c / "m7" / "a322c8cecb019168" / "outlines.npz"), m5_outlines=m5)  # fmt: skip


def printer(env, **kw):
    from tssa.footprint import VehFootPrinter

    vfp = VehFootPrinter(env["dh"], zone=env["zone"], **kw)
    vfp.local_origin = env["origin"]
    return vfp


def inject_ref(vfp, ref: pd.DataFrame, col: str) -> None:
    """Replace the port's psi_ref by the study's (`col`), aligned on (track, frame)."""
    r = ref[["track", "frame", col]]

    def f(base):
        return base[["track", "frame"]].merge(r, how="left", on=["track", "frame"])[col].to_numpy(float)

    vfp._reference_heading = f


def _dist_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.hypot(*(a - b).T)
    return {"rows": int(len(d)), "max_m": float(d.max()) if len(d) else None,
            "p99_m": float(np.quantile(d, 0.99)) if len(d) else None}  # fmt: skip


# --- native -------------------------------------------------------------------------------------------------------


def part_native(env, cach) -> dict:
    import m3
    import m6

    from tssa.footprint.locators.m5 import solve_m5
    from tssa.footprint.masks import lower_outline
    from tssa.transform.camera import PinholeCamera

    S, R, grid, origin = env["S"], env["R"], env["grid"], env["origin"]
    out: dict = {}
    # camera
    cam = PinholeCamera.fit(env["dh"], env["zone"])
    sc = env["cam"]
    out["camera"] = {"port": {"f": cam.f, "cx": cam.cx, "cy": cam.cy, "height_m": cam.height},
                     "study": {"f": sc.f, "cx": sc.cx, "cy": sc.cy, "height_m": float(sc.C[2])},
                     "max_abs_diff": max(abs(cam.f - sc.f), abs(cam.cx - sc.cx), abs(cam.cy - sc.cy),
                                         abs(cam.height - float(sc.C[2])))}  # fmt: skip
    rows = grid[R.bbox_rows(grid)]
    psi_d = pd.Series(env["ref"]["ref_display_deg"].to_numpy(float), index=grid.index)
    gate = rows["proj_valid"].to_numpy(bool)
    to_ground = R.m4_to_ground(env["cam"], env["cons"]["ok"], env["dh"])
    study = {
        "M1": env["m1n"], "M2": env["m2n"], "M3": m3.native_m3(rows, psi_d.loc[rows.index], origin, env["dh"]),
        "M4": R.native_m4(rows, cach["keypoints"], psi_d.loc[rows.index], env["dh"], to_ground, origin),
        "M6": m6.native_m6(rows, cach["cuboids"], env["dh"], origin),
    }
    lab = env["tracks"][env["tracks"]["track"].isin(env["labels"]["track"])]
    for m, st in study.items():
        vfp = printer(env, chains={c: (m,) for c in (2, 5, 7)})
        inject_ref(vfp, env["ref"], "ref_display_deg")
        loc = vfp.locate(lab, cuboids=cach["cuboids"], keypoints=cach["keypoints"])
        st = st.assign(_gate=st[["track", "frame"]].merge(rows[["track", "frame"]].assign(g=gate), how="left")["g"].to_numpy())
        st_ok = st["native_valid"].eq(True) & st["_gate"].eq(True)
        s_xy = st[["e_nat", "n_nat"]].to_numpy(float) + origin
        j = st[["track", "frame"]].assign(s_ok=st_ok.to_numpy(), sx=s_xy[:, 0], sy=s_xy[:, 1]).merge(
            loc[["track", "frame", "fp_source", "fp_x", "fp_y", "fp_fail"]], on=["track", "frame"], how="inner")
        p_ok = (j["fp_source"] == m).to_numpy()
        both = j["s_ok"].to_numpy() & p_ok
        out[m] = {"study_valid": int(j["s_ok"].sum()), "port_valid": int(p_ok.sum()), "both": int(both.sum()),
                  "only_study": int((j["s_ok"] & ~p_ok).sum()), "only_port": int((~j["s_ok"] & p_ok).sum()),
                  **_dist_stats(j.loc[both, ["fp_x", "fp_y"]].to_numpy(float), j.loc[both, ["sx", "sy"]].to_numpy(float))}  # fmt: skip
        if m == "M4":
            p4 = loc.merge(st[["track", "frame", "label_case", "interior_source"]], on=["track", "frame"])
            p4 = p4[(p4["fp_source"] == "M4")]
            out[m]["label_case_equal"] = bool((p4["m4_label_case"] == p4["label_case"]).all())
            out[m]["interior_source_equal"] = bool((p4["m4_interior_source"] == p4["interior_source"]).all())
    # M4 counts on eligible rows with the metric psi_ref (run report: 2,185 valid; tracks 230 and 432 flagged)
    el = grid[R.bbox_rows(grid) & grid["eligible"].to_numpy(bool)]
    vfp = printer(env, chains={c: ("M4",) for c in (2, 5, 7)})
    inject_ref(vfp, env["ref"], "ref_metric_deg")
    loc = vfp.locate(el[["frame", "track", "x", "y", "w", "h", "cls", "interp"]], keypoints=cach["keypoints"])
    v = loc[loc["fp_source"] == "M4"]
    fc = loc.groupby("track")["m4_front_conflict"].any()
    out["M4_metric_counts"] = {"valid": int(len(v)), "interior_source": v["m4_interior_source"].value_counts().to_dict(),
                               "label_case": v["m4_label_case"].value_counts().to_dict(),
                               "front_conflict_tracks": sorted(int(t) for t in fc[fc].index)}  # fmt: skip
    # M5: solve_m5 on the study's own outlines against the study's native_m5
    import m5

    m5_st = m5.native_m5(rows, cach["m5_outlines"], env["dh"], origin)
    W = np.array([S.SIZE_FT[int(c)][1] for c in rows["cls_track"]]) * FT
    L = np.array([S.SIZE_FT[int(c)][0] for c in rows["cls_track"]]) * FT
    import geometry

    init = geometry.m2_points(env["dh"], rows, W)
    sx, px = [], []
    for i, (t, f) in enumerate(zip(rows["track"].to_numpy(), rows["frame"].to_numpy())):
        s = m5_st.iloc[i]
        r = solve_m5(cach["m5_outlines"].get((int(t), int(f))), rows.iloc[i][["x", "y", "w", "h"]].to_numpy(float),
                     env["dh"], L[i], W[i], init[i])  # fmt: skip
        if r["ok"] != bool(s.native_valid) or (not r["ok"] and r["reason"] != s.fail_reason):
            px.append((int(t), int(f), r["reason"], s.fail_reason))
        elif r["ok"]:
            sx.append((r["centre"], np.array([s.e_nat, s.n_nat]) + origin))
    a = np.array([p for p, _ in sx]) if sx else np.empty((0, 2))
    b = np.array([q for _, q in sx]) if sx else np.empty((0, 2))
    out["M5_solve"] = {"rows": int(len(rows)), "valid_both": len(sx), "disagreements": len(px), **_dist_stats(a, b)}
    # M5 outlines from mask contours (cls 5/7 rows that both caches hold)
    diffs, n = [], 0
    for key, polys in cach["masks"].items():
        o_st = cach["m5_outlines"].get(key)
        if o_st is None:
            continue
        r = rows[(rows["track"] == key[0]) & (rows["frame"] == key[1])]
        if not len(r):
            continue
        o = lower_outline(list(polys), r.iloc[0][["x", "y", "w", "h"]].to_numpy(float))
        n += 1
        a_ = pd.DataFrame(o, columns=["u", "v"]).set_index("u")["v"]
        b_ = pd.DataFrame(np.asarray(o_st), columns=["u", "v"]).set_index("u")["v"]
        common = a_.index.intersection(b_.index)
        diffs.append((len(a_), len(b_), len(common), float(np.abs(a_[common] - b_[common]).max()) if len(common) else 0.0))
    d = np.array(diffs) if diffs else np.zeros((0, 4))
    out["M5_outline_from_contours"] = {
        "rows": n, "same_columns": int((d[:, 0] == d[:, 1]).sum()) if n else 0,
        "rows_identical": int(((d[:, 0] == d[:, 1]) & (d[:, 2] == d[:, 0]) & (d[:, 3] == 0)).sum()) if n else 0,
        "max_v_diff_px": float(d[:, 3].max()) if n else None}  # fmt: skip
    return out


# --- m7 -----------------------------------------------------------------------------------------------------------


def _convert(tr):
    from tssa.footprint.locators.m7 import solver as pm7
    from tssa.transform.camera import PinholeCamera

    c = tr.cam
    cam = PinholeCamera(c.f, c.R, c.C, c.cx, c.cy, c.origin, c.converged)
    obs = [pm7.FrameObs(o.frame, o.cand, o.kappa, o.env_u, o.env_v, o.ext, o.cand_s) for o in tr.obs]
    return pm7.Track(tr.frames, obs, cam, tr.origin, tr.init_centre, tr.init_psi_deg, tr.cls, tr.dt)


def _m7_jobs(env, cach):
    import m7_runner

    jobs = m7_runner.build_jobs(env["grid"], env["tracks"], dict(cach["masks"].items()), env["dh"], env["cam"],
                                env["m2n"], env["ref"], env["cons"]["ok"])  # fmt: skip
    return [j for j in jobs if j.kind == "display" and j.tr is not None]


def _study_report(root: Path) -> pd.DataFrame:
    """The study's display solves of run REF_RUN (m7_tracks.csv): cache keys hash the observations, so solves built
    on another platform are not found in the study's solve cache; the run report is the reference."""
    st = pd.read_csv(root / "runs" / REF_RUN / "m7_tracks.csv")
    st = st[st["kind"] == "display"]
    return pd.DataFrame({"track": st["track"], "seg": st["seg"], "fail": st["fail_reason"].fillna(""),
                         "L": st["length_ft"], "W": st["width_ft"]})  # fmt: skip


def _agreement(a: pd.DataFrame, b: pd.DataFrame, tol_ft: float = 0.05) -> dict:
    """a, b: track, seg, fail, L, W. Fail reasons equal; L and W within tol_ft where both passed."""
    j = a.merge(b, on=["track", "seg"], suffixes=("_a", "_b"))
    same = j["fail_a"] == j["fail_b"]
    both = j["fail_a"].eq("") & j["fail_b"].eq("")
    dL, dW = (j["L_a"] - j["L_b"]).abs(), (j["W_a"] - j["W_b"]).abs()
    return {"compared": int(len(j)), "same_fail_reason": int(same.sum()),
            "fail_differs": j.loc[~same, ["track", "seg", "fail_a", "fail_b"]].to_dict("records"),
            "both_passed": int(both.sum()), "L_within_tol": int((dL[both] <= tol_ft).sum()),
            "W_within_tol": int((dW[both] <= tol_ft).sum()), "dL_ft_median": float(dL.median()),
            "dL_ft_max": float(dL.max()), "dW_ft_max": float(dW.max()),
            "dL_ft_all": {f"{t}/{g}": round(float(x), 4) for t, g, x in zip(j["track"], j["seg"], dL)}}  # fmt: skip


def part_m7(env, cach, root: Path, workers: int) -> dict:
    """The study's display jobs, each converted and solved fresh by the port."""
    from tssa.footprint.locators.m7 import solve_jobs
    from tssa.footprint.locators.m7.settings import namespace

    S = env["S"]
    disp = _m7_jobs(env, cach)
    ns = namespace(None, S.IMG_SIZE, S.EDGE_PX, S.SIZE_FT, S.M7_CLASSES)
    t0 = time.perf_counter()
    res = solve_jobs([{"key": j.key, "tr": _convert(j.tr)} for j in disp], ns, None, workers, True)
    secs = time.perf_counter() - t0
    rows = []
    for j in disp:
        r = res[j.key]
        sm = r.get("summary", {})
        rows.append({"track": int(j.track), "seg": int(j.seg), "frames": len(j.frames),
                     "fail": sm.get("fail_reason", r.get("error")) or "", "L": sm.get("length_ft"),
                     "W": sm.get("width_ft"), "seconds": r.get("seconds")})  # fmt: skip
    df = pd.DataFrame(rows)
    return {"display_jobs": len(disp), "wall_s": secs, "passed": int(df["fail"].eq("").sum()), "solves": rows,
            "vs_study_report": _agreement(df, _study_report(root))}  # fmt: skip


def part_m7study(env, cach, root: Path, workers: int) -> dict:
    """The study's own solver on the same display jobs, in this environment: separates port error from platform
    numerics (the reference run was solved on Windows)."""
    import m7_runner

    disp = _m7_jobs(env, cach)
    t0 = time.perf_counter()
    res = m7_runner.solve_jobs(disp, cache_dir=None, workers=workers, camera_ok=env["cons"]["ok"], log=lambda m: None)
    secs = time.perf_counter() - t0
    rows = []
    for j in disp:
        sm = res[j.key].get("summary", {})
        rows.append({"track": int(j.track), "seg": int(j.seg), "fail": sm.get("fail_reason", res[j.key].get("error")) or "",
                     "L": sm.get("length_ft"), "W": sm.get("width_ft")})  # fmt: skip
    df = pd.DataFrame(rows)
    return {"display_jobs": len(disp), "wall_s": secs, "passed": int(df["fail"].eq("").sum()), "solves": rows,
            "vs_study_report": _agreement(df, _study_report(root))}  # fmt: skip


def part_m7compare(out_dir: Path) -> dict:
    """Port (m7.json) against the study's solver run here (m7study.json): the port's own error."""
    a = pd.DataFrame(json.loads((out_dir / "m7.json").read_text())["solves"])
    b = pd.DataFrame(json.loads((out_dir / "m7study.json").read_text())["solves"])
    return {"port_vs_study_here": _agreement(a, b)}


# --- e2e ----------------------------------------------------------------------------------------------------------

LINES_PX = [[[108, 160], [47, 320]], [[473, 117], [613, 186]]]  # check_speed_trap.py
TRUE_DISTANCE_FT = 92.71


def speed_trap(table: pd.DataFrame, dh, method_col="method") -> pd.DataFrame:
    """check_speed_trap.py on the port's trajectories (per (track, seg) instead of per metric run)."""
    sys.path.insert(0, str(Path(__file__).parent))
    lines = [dh.image_to_world(np.array(line, float)) for line in LINES_PX]

    def crossings(P, t, AB):
        A, B = AB
        d = B - A
        s = d[0] * (P[:, 1] - A[1]) - d[1] * (P[:, 0] - A[0])
        res = []
        for k in np.where(np.sign(s[:-1]) * np.sign(s[1:]) < 0)[0]:
            f = s[k] / (s[k] - s[k + 1])
            X = P[k] + f * (P[k + 1] - P[k])
            u = ((X - A) @ d) / (d @ d)
            if -0.1 <= u <= 1.1:
                res.append((t[k] + f * (t[k + 1] - t[k]), k, f, X))
        return res

    rows = []
    x = table[np.isfinite(table[["fp_x_smooth", "fp_y_smooth", "speed"]]).all(axis=1) & (table["seg"] >= 0)]
    for (m, tr, sg), g in x.sort_values("frame").groupby([method_col, "track", "seg"]):
        if (np.diff(g["frame"].to_numpy()) != 1).any():
            continue
        P, t, v = g[["fp_x_smooth", "fp_y_smooth"]].to_numpy(), g["frame"].to_numpy() / 10.0, g["speed"].to_numpy()
        c1, c2 = crossings(P, t, lines[0]), crossings(P, t, lines[1])
        if len(c1) != 1 or len(c2) != 1:
            continue
        (ta, ka, _, Xa), (tb, kb, _, Xb) = sorted([c1[0], c2[0]], key=lambda c: c[0])
        dt = tb - ta
        tt = np.concatenate([[ta], t[ka + 1: kb + 1], [tb]])
        vv = np.interp(tt, t, v)
        rows.append(dict(method=m, track=int(tr), dt_s=dt, v_trap_mph=TRUE_DISTANCE_FT / dt / 1.46667,
                         v_reported_mph=np.trapezoid(vv, tt) / dt))  # fmt: skip
    r = pd.DataFrame(rows)
    r["err_mph"] = r["v_reported_mph"] - r["v_trap_mph"]
    r["err_pct"] = (r["v_reported_mph"] / r["v_trap_mph"] - 1) * 100
    return r


def part_e2e(env, cach, root: Path, out_dir: Path, workers: int) -> dict:
    lab = env["tracks"][env["tracks"]["track"].isin(env["labels"]["track"])]
    turns = env["labels"].set_index("track")
    vfp = printer(env, workers=workers, cache_dir=out_dir / "cache")
    t0 = time.perf_counter()
    cmp_ = vfp.compare(lab, methods=("M1", "M2", "M3", "M4", "M6"), fallback="M2", cuboids=cach["cuboids"],
                       keypoints=cach["keypoints"])  # fmt: skip
    t_cmp = time.perf_counter() - t0
    cmp_.to_csv(out_dir / "compare_M1_M6.csv.gz", index=False)
    through = turns.index[turns["turn"].eq("T") & turns["approach"].isin(["NB", "SB"])]
    trap = speed_trap(cmp_[cmp_["track"].isin(through)], env["dh"])
    trap.to_csv(out_dir / "speed_trap_compare.csv", index=False)
    summ = trap.groupby("method").agg(n=("track", "size"), err_pct_med=("err_pct", "median"),
                                      abs_err_med=("err_mph", lambda s: s.abs().median()),
                                      abs_err_p90=("err_mph", lambda s: s.abs().quantile(0.9)))  # fmt: skip
    # smoothed M6 centres vs the study's M6 display product (cars, M6-supplied rows)
    st = pd.read_csv(root / "runs" / NATIVE_RUN / "traj_M6_display.csv", usecols=["track", "frame", "e_m", "n_m"])
    m6 = cmp_[(cmp_["method"] == "M6") & (cmp_["fp_source"] == "M6")].merge(st, on=["track", "frame"])
    m6 = m6[m6["cls_track"] == 2]
    d = np.hypot(m6["fp_x_smooth"] - m6["e_m"], m6["fp_y_smooth"] - m6["n_m"]) / FT
    return {"compare_s": t_cmp, "rows": int(len(lab)), "compare_rows": int(len(cmp_)),
            "rows_per_method": cmp_.groupby("method").size().to_dict(),
            "speed_trap": summ.round(3).reset_index().to_dict("records"),
            "m6_vs_study_display_ft": {"rows": int(len(d)), "median": float(d.median()), "p95": float(d.quantile(0.95))}}  # fmt: skip


def part_trap(root: Path, out_dir: Path) -> dict:
    """The port's speed trap (e2e's speed_trap_compare.csv, per (track, seg)) against the study's (speed_trap.csv of
    NATIVE_RUN, per metric_e2e run): |error| p90 on every vehicle each side measured, and on the vehicles both
    measured exactly once (the like-for-like comparison)."""
    pt = pd.read_csv(out_dir / "speed_trap_compare.csv")
    st = pd.read_csv(root / "runs" / NATIVE_RUN / "speed_trap.csv")
    st = st[st["turn"].eq("T") & st["approach"].isin(["NB", "SB"])]
    p90 = lambda s: round(float(s.abs().quantile(0.9)), 3)  # noqa: E731
    med = lambda s: round(float(s.abs().median()), 3)  # noqa: E731
    out = []
    for m in sorted(set(pt["method"]) & set(st["method"])):
        a, b = pt[pt["method"] == m], st[st["method"] == m]
        a1 = a.groupby("track").filter(lambda g: len(g) == 1).set_index("track")["err_mph"]
        b1 = b.groupby("track").filter(lambda g: len(g) == 1).set_index("track")["err_mph"]
        c = a1.index.intersection(b1.index)
        out.append({"method": m, "port_n": int(len(a)), "port_p90": p90(a["err_mph"]), "study_n": int(len(b)),
                    "study_p90": p90(b["err_mph"]), "common_n": int(len(c)), "port_p90_common": p90(a1[c]),
                    "study_p90_common": p90(b1[c]), "port_med_common": med(a1[c]), "study_med_common": med(b1[c]),
                    "abs_diff_med_common": med(a1[c] - b1[c])})  # fmt: skip
    return {"methods": out}


def part_estimate(env, cach, root: Path, out_dir: Path, workers: int) -> dict:
    """estimate() with the default chains (M7 for buses and trucks) on every labelled track."""
    lab = env["tracks"][env["tracks"]["track"].isin(env["labels"]["track"])]
    turns = env["labels"].set_index("track")
    vfp = printer(env, workers=workers, cache_dir=out_dir / "cache")
    t0 = time.perf_counter()
    est = vfp.estimate(lab, cuboids=cach["cuboids"], keypoints=cach["keypoints"], masks=cach["masks"])
    t_est = time.perf_counter() - t0
    est.to_csv(out_dir / "estimate_default_chains.csv.gz", index=False)
    through = turns.index[turns["turn"].eq("T") & turns["approach"].isin(["NB", "SB"])]
    trap = speed_trap(est[est["track"].isin(through)].assign(method="default"), env["dh"])
    m7s = pd.DataFrame(vfp.info.get("m7_solves", []))
    big = est["cls_track"].isin([5, 7])
    return {"estimate_s": t_est, "rows": int(len(lab)), "estimate_rows": int(len(est)),
            "sources_all": est["fp_source"].value_counts().to_dict(),
            "sources_bus_truck": est.loc[big, "fp_source"].value_counts().to_dict(),
            "support_bus_truck": est.loc[big, "fp_support"].value_counts().to_dict(),
            "dims_source_bus_truck": est.loc[big, "dims_source"].value_counts().to_dict(),
            "speed_trap": {"n": int(len(trap)), "abs_err_med": float(trap["err_mph"].abs().median()) if len(trap) else None,
                           "abs_err_p90": float(trap["err_mph"].abs().quantile(0.9)) if len(trap) else None},
            "m7_solves": {"n": int(len(m7s)), "passed": int((m7s.get("fail_reason", pd.Series()) == "").sum()),
                          "fail_reasons": m7s.get("fail_reason", pd.Series()).value_counts().to_dict()},
            "low_confidence_rows": int(est["fp_low_confidence"].sum()), "info_grid": vfp.info.get("grid")}  # fmt: skip


def part_grid(env) -> dict:
    lab = env["tracks"][env["tracks"]["track"].isin(env["labels"]["track"])]
    vfp = printer(env)
    loc = vfp.locate(lab)
    g = env["grid"]
    st = g[g["seg"] >= 0].groupby(["track", "seg"]).ngroups
    pt = loc[loc["seg"] >= 0].groupby(["track", "seg"]).ngroups
    return {"study_segments": int(st), "port_segments": int(pt), "port_info": vfp.info.get("grid"),
            "study_inserted": int((g["interp"] == 2).sum()), "port_inserted": int((loc["fp_row_kind"] == "inserted").sum()),
            "port_seg_minus_one_rows": int((loc["seg"] < 0).sum())}  # fmt: skip


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--part", required=True, choices=["native", "m7", "m7study", "m7compare", "trap", "e2e", "estimate", "grid"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if a.part in ("m7compare", "trap"):
        res = part_m7compare(a.out) if a.part == "m7compare" else part_trap(a.root, a.out)
        (a.out / f"{a.part}.json").write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(res, indent=1, default=str))
        return
    env = study_env(a.study, a.root)
    cach = caches(a.root) if a.part != "grid" else None
    res = {"native": lambda: part_native(env, cach), "m7": lambda: part_m7(env, cach, a.root, a.workers),
           "m7study": lambda: part_m7study(env, cach, a.root, a.workers),
           "e2e": lambda: part_e2e(env, cach, a.root, a.out, a.workers),
           "estimate": lambda: part_estimate(env, cach, a.root, a.out, a.workers), "grid": lambda: part_grid(env)}[a.part]()
    res["total_s"] = time.perf_counter() - t0
    (a.out / f"{a.part}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
