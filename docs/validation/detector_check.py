"""Detector check (design 14.3): the tssa detectors against the speed study's caches on sampled rows.

Run in the study's environment (torch, ultralytics, CUDA; the same device and package versions as the study):

    python detector_check.py --src <tssa>/src --root <footprint output root> --sam <sam2.1_b.pt> \
        --uod <urbanomnidetect_v2_x_640.pt> --uod-root <urbanomnidetect repo> --out <file.json> [--n 200]

Rows are sampled from the study's current caches (m4 6a1f..., m6 9a1f..., m7 a322...); the whole track file is the
matching context, as in the study. Pass: keypoints and cuboid corners within 0.5 px of the cache, mask IoU >= 0.99
(the port's components filtered by M7's 5% rule, as M7 reads them). Also reported: on --n5 rows of the study's M5
cache (62a0...), whether the port's raw-mask lower outline equals the study's M5 outline.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    for k in ("--src", "--root", "--sam", "--uod", "--uod-root", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n5", type=int, default=200)
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    from tssa.footprint import detect
    from tssa.footprint.m7config import M7Config
    from tssa.footprint.masks import MaskTable, keep_components, rasterise

    root = Path(a.root)
    inp = root / "inputs"
    video = next(inp.glob("*.mp4"))
    raw = pd.read_csv(next(inp.glob("*_veh_tracks_reclass_p3.txt")), header=None,
                      names=["frame", "track", "x", "y", "w", "h", "conf", "cls", "flag", "r4"])  # fmt: skip
    tracks = raw[["track", "frame", "x", "y", "w", "h"]]
    c = root / "cache"
    m4 = pd.read_csv(c / "m4" / "6a1f0df0d8de1344" / "rows.csv")
    m6 = pd.read_csv(c / "m6" / "9a1f4b79288939c9" / "rows.csv")
    m7 = pd.read_csv(c / "m7" / "a322c8cecb019168" / "rows.csv")
    masks = MaskTable.load_npz(c / "m7" / "a322c8cecb019168" / "outlines.npz")
    z = np.load(c / "m5" / "62a0ce39913f7eb0" / "outlines.npz")
    z_t, z_f, z_n, z_p = z["track"], z["frame"], z["count"], z["pts"]  # each array read (decompressed) once
    m5 = {(int(t), int(f)): z_p[e - k:e] for t, f, k, e in zip(z_t, z_f, z_n, np.cumsum(z_n))}
    rng = np.random.default_rng(0)
    n3 = a.n // 3
    pick = lambda df, k: df.iloc[rng.choice(len(df), min(k, len(df)), replace=False)][["track", "frame"]]  # noqa: E731
    s4, s6, s7 = pick(m4[m4["matched"] == 1], n3), pick(m6[m6["matched"] == 1], n3), pick(m7, a.n - 2 * n3)
    k5 = list(m5)
    s5 = pd.DataFrame([k5[i] for i in rng.choice(len(k5), min(a.n5, len(k5)), replace=False)], columns=["track", "frame"])
    sm = pd.concat([s7, s5]).drop_duplicates()
    rows = {t: s.merge(tracks, on=["track", "frame"]) for t, s in (("keypoints", s4), ("cuboids", s6), ("masks", sm))}
    dets = [detect.KeypointDetector(), detect.CuboidDetector(a.uod, a.uod_root), detect.MaskSegmenter(a.sam)]
    t0 = time.perf_counter()
    got = detect.run(str(video), tracks, dets, rows=rows)
    secs = time.perf_counter() - t0
    out = {"seconds": secs, "video": video.name}
    for name, cache, cols in (("keypoints", m4, [f"{c}{i}" for i in range(4) for c in "uv"]),
                              ("cuboids", m6, [f"g{c}{i}" for i in range(4) for c in "uv"])):  # fmt: skip
        want = rows[name][["track", "frame"]].merge(cache, on=["track", "frame"])
        j = want.merge(got[name], on=["track", "frame"], how="left", suffixes=("_st", ""))
        matched_st = j["matched"].eq(1)
        matched_pt = j[cols[0]].notna()
        d = np.abs(j.loc[matched_st & matched_pt, cols].to_numpy(float) - j.loc[matched_st & matched_pt,
                   [f"{c}_st" for c in cols]].to_numpy(float))  # fmt: skip
        out[name] = {"rows": int(len(j)), "matched_study": int(matched_st.sum()), "matched_port": int(matched_pt.sum()),
                     "match_agree": int((matched_st == matched_pt).sum()), "max_px": float(d.max()) if d.size else None,
                     "within_0_5px": int((d.max(axis=1) <= 0.5).sum()) if d.size else 0}  # fmt: skip
    ious = []
    frac = M7Config().M7_MIN_COMPONENT_FRAC
    for t, f in s7[["track", "frame"]].itertuples(index=False):
        p, s = got["masks"].get((t, f)), masks.get((t, f))
        p = keep_components(p, frac) if p is not None else None
        if p is None or s is None:
            ious.append(0.0 if (p is None) != (s is None) else 1.0)
            continue
        A, B = rasterise(p, (640, 480)), rasterise(s, (640, 480))
        ious.append(float((A & B).sum() / max((A | B).sum(), 1)))
    ious = np.array(ious)
    out["masks"] = {"rows": int(len(ious)), "iou_min": float(ious.min()), "iou_median": float(np.median(ious)),
                    "iou_ge_0_99": int((ious >= 0.99).sum())}  # fmt: skip
    same, n5, cols_same, dv = 0, 0, 0, []
    for t, f in s5[["track", "frame"]].itertuples(index=False):
        p, s = got["masks"].outlines.get((int(t), int(f))), m5[(int(t), int(f))]
        n5 += 1
        if p is None:
            continue
        same += int(p.shape == s.shape and np.array_equal(p, s))
        if np.array_equal(p[:, 0], s[:, 0]):
            cols_same += 1
            dv.append(float(np.abs(p[:, 1] - s[:, 1]).max()) if len(p) else 0.0)
    out["m5_outlines"] = {"rows": n5, "identical": same, "same_columns": cols_same,
                          "max_v_diff_px": max(dv) if dv else None}  # fmt: skip
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
