"""Segmentation of ground tracks for smoothing: duplicates, jumps, gap reasons, fill grid.

Inputs:
    track tables with frame, track, the two position columns (metres), optionally interp (0 detected, 1 filled
    upstream) and a per-row sigma; an optional reference table (the tracks before gaps were emptied or filled
    upstream) that tells jump gaps from plain gaps.
Outputs:
    prepare           -> table with canonical columns `_x`, `_y` (metres), `_sigma`, `interp`, deduplicated and sorted
    jump_split        -> boolean split flags between consecutive rows of one track (implied speed rule)
    jump_gaps         -> (track, first, last) of the gaps between detected rows that jump
    classify_gaps     -> table (track, gap_id, first, last, length, implied_speed, reason, fill)
    build_grid        -> `grid`: one row per frame of every segment, filled gaps included, in LOCAL metres
                         (`grid.attrs["origin"] = (X0, Y0)` is the origin of the local frame)
    structure         -> the structural table only (track, frame, seg, seg_len, interp, fill_reason, gap_id), with
                         rows of missing positions kept (`allow_missing`); no positions, so none can be reused
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SmoothConfig

JUMP = "jump"  # the pair of rows jumps (implied speed above split_speed), or the gap lies inside a reference jump
LONG = "long"  # gap longer than fill_max_gap frames
SPEED = "speed"  # implied speed across the gap is too high
OVERLAP = "overlap"  # plain gap, confirmed against the reference table, most likely an occlusion
UNVERIFIED = "unverified"  # plain gap, no reference table: filled on the conservative rule only
NO_POSITION = "no_position"  # an input row without a finite position (segment(allow_missing=True))

FILL_REASONS = (OVERLAP, UNVERIFIED)  # the reasons whose gaps are filled (when fill_gaps is on)
GAP_COLUMNS = ["track", "gap_id", "first", "last", "length", "implied_speed", "reason", "fill"]
COV = ("_cxx", "_cxy", "_cyy")  # canonical per-row covariance columns (m^2)
GRID_COLUMNS = [
    "frame", "track", "seg", "seg_len", "interp", "fill_reason", "gap_id", "sigma", *COV,
    "x_obs", "y_obs", "x_raw", "y_raw",
]  # fmt: skip
STRUCT_COLUMNS = ["track", "frame", "seg", "seg_len", "interp", "fill_reason", "gap_id"]


def prepare(df: pd.DataFrame, cfg: SmoothConfig, allow_missing: bool = False) -> pd.DataFrame:
    """Copy with canonical `_x`, `_y`, `_sigma`, covariance and `interp` columns, sorted by track and frame, repeated
    rows dropped.

    Raises ValueError for a missing required column, or for a non-finite position unless `allow_missing`.
    """
    xc, yc = cfg.xy_cols
    need = ["frame", "track", xc, yc] + ([cfg.sigma_col] if cfg.sigma_col else []) + list(cfg.cov_cols or [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"tracks lack required column(s) {missing}")
    out = df.copy()
    out["_x"], out["_y"] = out[xc].astype(float), out[yc].astype(float)
    out["_sigma"] = out[cfg.sigma_col].astype(float) if cfg.sigma_col else np.nan
    for canon, col in zip(COV, cfg.cov_cols or (None, None, None)):
        out[canon] = out[col].astype(float) if col else np.nan
    if "interp" not in out.columns:
        out["interp"] = 0
    if not allow_missing and not np.isfinite(out[["_x", "_y"]].to_numpy()).all():
        raise ValueError(f"position columns {cfg.xy_cols} contain NaN or infinite values")
    out = out[~out.duplicated(["frame", "track"], keep="first")]
    return out.sort_values(["track", "frame"]).reset_index(drop=True)


def _runs(track: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [lo, hi) of equal consecutive values."""
    if not len(track):
        return []
    cuts = np.flatnonzero(track[1:] != track[:-1]) + 1
    edges = [0, *cuts.tolist(), len(track)]
    return list(zip(edges[:-1], edges[1:]))


def jump_split(f: np.ndarray, xy: np.ndarray, *, speed: float, fps: float) -> np.ndarray:
    """Flag the pairs of consecutive rows (length n - 1) whose implied speed (m/s) is above `speed`."""
    step = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    return step / (np.maximum(np.diff(f), 1) / fps) > speed


def jump_gaps(df: pd.DataFrame, cfg: SmoothConfig) -> list[tuple[int, int, int]]:
    """Return `(track, first frame, last frame)` of the gaps between detected rows (interp 0) that jump."""
    gaps = []
    for tid, obs in df[df["interp"] == 0].sort_values("frame").groupby("track"):
        if len(obs) < 2:
            continue
        f = obs["frame"].to_numpy()
        flag = jump_split(f, obs[["_x", "_y"]].to_numpy(dtype=float), speed=cfg.split_speed, fps=cfg.fps)
        gaps += [(int(tid), int(f[k]), int(f[k + 1])) for k in np.flatnonzero(flag & (np.diff(f) > 1))]
    return gaps


def _pairs(df: pd.DataFrame, cfg: SmoothConfig) -> pd.DataFrame:
    """Consecutive row pairs of a table sorted by track and frame, with `same` track and `jump` flags."""
    track, frame = df["track"].to_numpy(), df["frame"].to_numpy()
    jump = np.zeros(max(len(df) - 1, 0), bool)
    xy = df[["_x", "_y"]].to_numpy(dtype=float)
    for lo, hi in _runs(track):
        if hi - lo > 1:
            jump[lo : hi - 1] = jump_split(frame[lo:hi], xy[lo:hi], speed=cfg.split_speed, fps=cfg.fps)
    return pd.DataFrame({
        "i": np.arange(len(jump)), "track": track[:-1], "first": frame[:-1], "last": frame[1:],
        "same": track[1:] == track[:-1], "jump": jump,
    })  # fmt: skip


def classify_gaps(df: pd.DataFrame, ref_jumps: list[tuple[int, int, int]], cfg: SmoothConfig, *, verified: bool) -> pd.DataFrame:
    """Classify every frame gap of a prepared table; first match wins.

    `ref_jumps` are the jump gaps of the reference table (see `jump_gaps`). A plain gap is `overlap` when the
    reference was given (`verified`) and `unverified` otherwise; both are filled when `cfg.fill_gaps` is on.
    """
    pr = _pairs(df, cfg)
    pr = pr[pr["same"] & (pr["last"] - pr["first"] > 1)].reset_index(drop=True)
    x, y = df["_x"].to_numpy(), df["_y"].to_numpy()
    i = pr["i"].to_numpy()
    span = (pr["last"] - pr["first"]).to_numpy()
    length = span - 1
    speed = np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) / (span / cfg.fps)
    jumped = pr["jump"].to_numpy()
    jump_by_track = {}
    if ref_jumps:
        jt = pd.DataFrame(ref_jumps, columns=["track", "first", "last"])
        jump_by_track = {t: g[["first", "last"]].to_numpy() for t, g in jt.groupby("track")}
    plain = OVERLAP if verified else UNVERIFIED
    out = []
    for k, (t, first, last) in enumerate(zip(pr["track"], pr["first"], pr["last"])):
        rows = jump_by_track.get(t)
        if jumped[k]:
            reason = JUMP
        elif rows is not None and ((rows[:, 0] <= first) & (last <= rows[:, 1])).any():
            reason = JUMP
        elif length[k] > cfg.fill_max_gap:
            reason = LONG
        elif speed[k] > cfg.gap_max_speed:
            reason = SPEED
        else:
            reason = plain
        out.append(reason)
    gaps = pd.DataFrame({
        "track": pr["track"].astype("int64"), "gap_id": np.arange(len(pr)), "first": pr["first"].astype("int64"),
        "last": pr["last"].astype("int64"), "length": length.astype("int64"), "implied_speed": speed,
        "reason": pd.Series(out, dtype=object),
    })  # fmt: skip
    gaps["fill"] = gaps["reason"].isin(FILL_REASONS) & cfg.fill_gaps
    return gaps[GAP_COLUMNS]


def build_grid(df: pd.DataFrame, gaps: pd.DataFrame, cfg: SmoothConfig) -> pd.DataFrame:
    """Per-frame grid of all segments: split at jumps and unfilled gaps, interp = 2 rows inside filled gaps."""
    obs = df[["frame", "track", "interp", "_sigma", *COV, "_x", "_y"]].copy()
    origin = (float(np.floor(obs["_x"].min())), float(np.floor(obs["_y"].min())))
    obs["x_obs"], obs["y_obs"] = obs["_x"] - origin[0], obs["_y"] - origin[1]
    obs["x_raw"], obs["y_raw"] = obs["x_obs"], obs["y_obs"]
    obs["sigma"] = obs["_sigma"]
    obs["fill_reason"], obs["gap_id"] = "", -1
    pr = _pairs(df, cfg)
    pr = pr[pr["same"]].merge(
        gaps[["track", "first", "last", "gap_id", "reason", "fill"]], how="left", on=["track", "first", "last"]
    )
    gapped = (pr["last"] - pr["first"]) > 1
    fills = gapped & pr["fill"].eq(True) & ~pr["jump"]
    cut = pr["jump"] | (gapped & ~fills)
    boundary = np.ones(len(obs), bool)  # track starts stay True
    boundary[pr["i"].to_numpy() + 1] = cut.to_numpy()
    obs["seg"] = np.cumsum(boundary) - 1
    fp = pr[fills]
    if len(fp):
        lo = fp["i"].to_numpy()
        count = (fp["last"] - fp["first"] - 1).to_numpy()
        rep = np.repeat(np.arange(len(fp)), count)
        step = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count) + 1
        t = step / (fp["last"] - fp["first"]).to_numpy()[rep]
        a = lo[rep]
        fill = pd.DataFrame({
            "frame": fp["first"].to_numpy()[rep] + step, "track": fp["track"].to_numpy()[rep],
            "seg": obs["seg"].to_numpy()[a],
        })  # fmt: skip
        for col in ("x_raw", "y_raw"):
            v = obs[col].to_numpy()
            fill[col] = v[a] + t * (v[a + 1] - v[a])
        fill["interp"] = 2
        fill["fill_reason"] = fp["reason"].to_numpy()[rep]
        fill["gap_id"] = fp["gap_id"].astype("int64").to_numpy()[rep]
        grid = pd.concat([obs, fill], ignore_index=True).sort_values(["track", "frame"], kind="stable")
    else:
        grid = obs
    grid["seg_len"] = grid.groupby("seg")["frame"].transform("size")
    grid = grid.reindex(columns=GRID_COLUMNS).reset_index(drop=True)
    grid.attrs["origin"] = origin
    return grid


def structure(df: pd.DataFrame, ref: pd.DataFrame | None, cfg: SmoothConfig) -> pd.DataFrame:
    """Structural table of a prepared table that may hold rows without a finite position.

    Jumps and gaps are classified on the finite rows only; a run of missing rows is part of the gap around it:
    - interior run in a filled gap: kept in that segment with its own interp, fill_reason = "no_position";
    - interior run in a gap that is not filled (split), a prefix, a suffix, or a track with no finite row:
      seg = -1, fill_reason = "no_position", never extrapolated.
    Frames with no input row inside a filled gap are inserted as interp = 2, as `build_grid` does.
    Columns: STRUCT_COLUMNS; seg_len is 0 on seg = -1 rows.
    """
    has = np.isfinite(df[["_x", "_y"]].to_numpy(dtype=float)).all(axis=1)
    fin = df[has].reset_index(drop=True)
    if fin.empty:
        struct = pd.DataFrame({c: pd.Series(dtype="int64") for c in ["track", "frame", "seg", "interp", "gap_id"]})
        struct["fill_reason"] = pd.Series(dtype=object)
    else:
        verified = ref is not None
        ref_jumps = jump_gaps(ref[np.isfinite(ref[["_x", "_y"]].to_numpy(dtype=float)).all(axis=1)], cfg) if verified else []
        gaps = classify_gaps(fin, ref_jumps, cfg, verified=verified)
        struct = build_grid(fin, gaps, cfg)[["track", "frame", "seg", "interp", "fill_reason", "gap_id"]].copy()
    miss = df.loc[~has, ["track", "frame", "interp"]]
    if len(miss):
        key = pd.MultiIndex.from_frame(struct[["track", "frame"]])
        mkey = pd.MultiIndex.from_frame(miss[["track", "frame"]])
        inside = key.isin(mkey)  # inserted (interp = 2) rows of a filled gap at frames that have a missing input row
        m_int = pd.Series(miss["interp"].to_numpy(), index=mkey)
        struct.loc[inside, "interp"] = m_int.reindex(key[inside]).to_numpy()
        struct.loc[inside, "fill_reason"] = NO_POSITION
        outside = miss[~mkey.isin(key)]
        extra = pd.DataFrame({"track": outside["track"].to_numpy(), "frame": outside["frame"].to_numpy(), "seg": -1,
                              "interp": outside["interp"].to_numpy(), "fill_reason": NO_POSITION, "gap_id": -1})
        struct = pd.concat([struct, extra], ignore_index=True)
    struct = struct.sort_values(["track", "frame"], kind="stable").reset_index(drop=True)
    seg = struct["seg"].to_numpy()
    sizes = struct[seg >= 0].groupby("seg")["frame"].transform("size")
    struct["seg_len"] = 0
    struct.loc[seg >= 0, "seg_len"] = sizes.to_numpy()
    for c in ("frame", "seg", "seg_len", "interp", "gap_id"):
        struct[c] = struct[c].astype("int64")
    return struct[STRUCT_COLUMNS]
