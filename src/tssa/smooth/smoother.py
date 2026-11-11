"""TrackSmoother: segment projected ground tracks, fill gaps, smooth the positions with a chosen method.

Inputs:  track tables with frame, track and two position columns in metres (a ground plane the user projected
         beforehand; default names x, y), optionally interp (0 detected, 1 filled upstream), a per-row sigma column
         and three per-row covariance columns. Any other columns are carried through. No homography is involved.
Outputs: the input columns first, unchanged and in their order (NaN on rows filled here), then the added columns.
         `smooth()` adds seg, fill_reason, <pos>_smooth (two columns) and fallback (1 = segment too short for the
         method, raw positions copied); `interp` (0 detected, 1 filled upstream, 2 filled here) is updated in place,
         or added at the front when the input lacks it. `compare()` adds seg, fill_reason, then <pos>_<m> (two
         columns) and fb_<m> per method <m>. Speeds are not computed here: pass the smoothed positions to
         tssa.speed.SpeedCalculator.

Segmenting once, smoothing many times (design docs/2026-11-10-footprint-design.md, section 5.2):
         `segment()` returns the structural table only (track, frame, seg, seg_len, interp, fill_reason, gap_id).
         `smooth(..., segmented=True)` takes those columns from its input and does not re-segment: a row is a
         measurement exactly when seg >= 0 and its position is finite (interp is provenance only), observations come
         only from this call's positions, and seg = -1 rows come back with NaN positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import segments
from .config import SmoothConfig
from .methods import PRESETS, SegmentContext, available_methods, resolve, smooth_segment

INTERNAL = {"_x", "_y", "_sigma", *segments.COV}  # helper columns of segments.prepare
SMOOTH_SUFFIX = "_smooth"  # smoothed positions are <pos>_smooth, so the input positions stay untouched
STRUCTURAL = ("seg", "interp", "fill_reason")  # columns a segmented input must carry


class TrackSmoother:
    """Smooth ground-plane trajectories with a selectable method and parameters.

    method:  a name from `available_methods()` (ma, sg5, sg7, rts, or a kind such as sg).
    params:  overrides of that method's parameters, e.g. `method="sg", params={"win": 9}`.
    config:  a `SmoothConfig`; keyword overrides (fps=15, xy_cols=("e", "n"), ...) are applied on top.

    Gaps are classified against `ref_tracks` when given (the tracks before gaps were emptied or filled upstream,
    in the same position columns, which tells jump gaps from plain occlusion gaps). Without it every gap that is
    short, implies a plausible speed and does not jump is filled ("unverified" fills).
    """

    def __init__(self, method: str = "sg7", params: dict | None = None,
                 config: SmoothConfig | None = None, **config_overrides) -> None:  # fmt: skip
        self.config = (config or SmoothConfig()).with_overrides(**config_overrides)
        self.method = method
        self.params = dict(params or {})
        self._kind, self._merged = resolve(method, self.params)  # fail early on a bad name or parameter

    @staticmethod
    def available_methods() -> dict[str, dict]:
        """Every selectable method name with its default parameters."""
        return available_methods()

    def segment(self, tracks: pd.DataFrame, ref_tracks: pd.DataFrame | None = None,
                allow_missing: bool = False) -> pd.DataFrame:  # fmt: skip
        """Structural table (track, frame, seg, seg_len, interp, fill_reason, gap_id): split at jumps, fill gaps.

        One row per input row (repeated (track, frame) rows dropped) plus the inserted rows (interp = 2). No
        positions are returned. With `allow_missing`, rows without a finite position are kept: an interior run inside
        a filled gap stays in that segment (fill_reason "no_position"); any other such row gets seg = -1.
        """
        df = segments.prepare(tracks, self.config, allow_missing=allow_missing)
        ref = segments.prepare(ref_tracks, self.config, allow_missing=True) if ref_tracks is not None else None
        return segments.structure(df, ref, self.config)

    def smooth(self, tracks: pd.DataFrame, ref_tracks: pd.DataFrame | None = None,
               segmented: bool = False) -> pd.DataFrame:  # fmt: skip
        """Smoothed table for the configured method (see the module docstring for `segmented`)."""
        if segmented:
            return self._smooth_segmented(tracks, [(self.method, self._kind, self._merged)], ref_tracks, single=True)
        df, grid = self._grid(tracks, ref_tracks)
        out = self._apply(grid, [(self.method, self._kind, self._merged)])
        return self._finish(out, grid, df, [self.method], single=True)

    def compare(self, tracks: pd.DataFrame, methods=None, ref_tracks: pd.DataFrame | None = None,
                segmented: bool = False) -> pd.DataFrame:  # fmt: skip
        """Every method side by side on one segmentation and gap fill.

        `methods` is a list of names or a dict name -> parameters; None means every preset (ma, sg5, sg7, rts).
        """
        if methods is None:
            methods = list(PRESETS)
        specs = list(methods.items()) if isinstance(methods, dict) else [(m, None) for m in methods]
        resolved = [(name, *resolve(name, p)) for name, p in specs]
        if segmented:
            return self._smooth_segmented(tracks, resolved, ref_tracks, single=False)
        df, grid = self._grid(tracks, ref_tracks)
        out = self._apply(grid, resolved)
        return self._finish(out, grid, df, [n for n, _, _ in resolved], single=False)

    # steps ------------------------------------------------------------------------------------------------------

    def _grid(self, tracks: pd.DataFrame, ref_tracks: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Steps 1 and 2: prepare the positions, split at jumps and fill gaps."""
        df = segments.prepare(tracks, self.config)
        if df.empty:
            grid = pd.DataFrame({c: pd.Series(dtype=float) for c in segments.GRID_COLUMNS})
            grid.attrs["origin"] = (0.0, 0.0)
            return df, grid
        verified = ref_tracks is not None
        ref = segments.prepare(ref_tracks, self.config) if verified else df
        ref_jumps = segments.jump_gaps(ref, self.config) if verified else []
        gaps = segments.classify_gaps(df, ref_jumps, self.config, verified=verified)
        return df, segments.build_grid(df, gaps, self.config)

    def _apply(self, grid: pd.DataFrame, specs: list[tuple[str, str, dict]]) -> pd.DataFrame:
        """Step 3: smooth every (track, seg) with every method; adds per-method position and fallback columns."""
        dt = 1.0 / self.config.fps
        out = grid.copy()
        for name, _, _ in specs:
            out[f"x_{name}"] = np.nan
            out[f"y_{name}"] = np.nan
            out[f"fb_{name}"] = 0
        keys = grid.groupby(["track", "seg"], sort=False).indices
        col = {c: out.columns.get_loc(c) for c in out.columns}
        cov_all = _cov(grid) if self.config.cov_cols else None
        for rows in keys.values():
            g = grid.iloc[rows]
            t = g["frame"].to_numpy(dtype=float) * dt
            pos = g[["x_raw", "y_raw"]].to_numpy(dtype=float)
            ctx = SegmentContext(
                obs=g[["x_obs", "y_obs"]].to_numpy(dtype=float), sigma=g["sigma"].to_numpy(dtype=float),
                interp=g["interp"].to_numpy(), dt=dt, cov=None if cov_all is None else cov_all[rows],
            )  # fmt: skip
            for name, kind, p in specs:
                sp, fb = smooth_segment(kind, p, t, pos, ctx)
                out.iloc[rows, col[f"x_{name}"]] = sp[:, 0]
                out.iloc[rows, col[f"y_{name}"]] = sp[:, 1]
                out.iloc[rows, col[f"fb_{name}"]] = int(fb)
        return out

    def _finish(self, out: pd.DataFrame, grid: pd.DataFrame, df: pd.DataFrame, names: list[str], single: bool) -> pd.DataFrame:
        """Step 4: input columns first (unchanged), then the added columns; local metres back to input coordinates."""
        xc, yc = self.config.xy_cols
        x0, y0 = grid.attrs.get("origin", (0.0, 0.0))
        for m in names:
            out[f"x_{m}"] = out[f"x_{m}"] + x0
            out[f"y_{m}"] = out[f"y_{m}"] + y0
        given = [c for c in df.columns if c not in INTERNAL]
        ren, added = self._added(names, single)
        added = ["seg", "fill_reason", *added]
        out = out.rename(columns=ren)
        clash = sorted(set(added) & set(given))
        if clash:
            raise ValueError(f"input columns {clash} clash with columns the smoother adds; rename them first")
        out = out.merge(df[["track", "frame", *[c for c in given if c not in ("track", "frame", "interp")]]],
                        how="left", on=["track", "frame"])  # fmt: skip
        if "interp" not in given:
            added.insert(0, "interp")
        cols = [*given, *[c for c in added if c not in given]]
        out = out[cols].copy()
        out = _round(out, added)
        out.attrs["origin"] = (x0, y0)
        return out

    def _added(self, names: list[str], single: bool) -> tuple[dict, list[str]]:
        """Renaming of the per-method columns and the list of position/fallback columns added."""
        xc, yc = self.config.xy_cols
        if single:
            m = names[0]
            ren = {f"x_{m}": f"{xc}{SMOOTH_SUFFIX}", f"y_{m}": f"{yc}{SMOOTH_SUFFIX}", f"fb_{m}": "fallback"}
            return ren, list(ren.values())
        ren = {f"{c}_{m}": f"{xc if c == 'x' else yc}_{m}" for m in names for c in ("x", "y")}
        return ren, [c for m in names for c in (ren[f"x_{m}"], ren[f"y_{m}"], f"fb_{m}")]

    # segmented mode (design section 5.2) -------------------------------------------------------------------------

    def _smooth_segmented(self, tracks: pd.DataFrame, specs: list[tuple[str, str, dict]],
                          ref_tracks: pd.DataFrame | None, single: bool) -> pd.DataFrame:  # fmt: skip
        """Smooth on the structure carried by the input (seg, interp, fill_reason); no re-segmentation."""
        if ref_tracks is not None:
            raise ValueError("ref_tracks is not used with segmented=True: the input already carries its segmentation")
        lack = [c for c in STRUCTURAL if c not in tracks.columns]
        if lack:
            raise ValueError(f"segmented=True needs the structural column(s) {lack} (from segment() or locate())")
        df = segments.prepare(tracks, self.config, allow_missing=True)
        if len(df) != len(tracks):
            raise ValueError("segmented input has repeated (track, frame) rows")
        seg = df["seg"].to_numpy(dtype="int64")
        frame = df["frame"].to_numpy(dtype="int64")
        groups = [r for r in df[seg >= 0].groupby(["track", "seg"], sort=False).indices.values()]
        idx_pos = np.flatnonzero(seg >= 0)
        groups = [idx_pos[r] for r in groups]
        for rows in groups:
            if len(rows) > 1 and not (np.diff(frame[rows]) == 1).all():
                t, s = df["track"].iloc[rows[0]], seg[rows[0]]
                raise ValueError(f"segment {s} of track {t} does not have consecutive frames")
        xy = df[["_x", "_y"]].to_numpy(dtype=float)
        meas = (seg >= 0) & np.isfinite(xy).all(axis=1)
        x0 = float(np.floor(xy[meas, 0].min())) if meas.any() else 0.0
        y0 = float(np.floor(xy[meas, 1].min())) if meas.any() else 0.0
        local = xy - [x0, y0]
        local[~meas] = np.nan
        sigma = df["_sigma"].to_numpy(dtype=float)
        interp = df["interp"].to_numpy()
        cov = _cov(df) if self.config.cov_cols else None
        dt = 1.0 / self.config.fps
        res = {name: (np.full((len(df), 2), np.nan), np.ones(len(df), dtype="int64")) for name, _, _ in specs}
        for rows in groups:
            m = meas[rows]
            n_meas = int(m.sum())
            if n_meas == 0:
                continue  # no measurement: NaN, fallback = 1
            obs = local[rows]
            if n_meas == 1:
                for name, _, _ in specs:
                    res[name][0][rows[m]] = obs[m]  # that position on that row only
                continue
            raw = _interp_hold(obs, m)
            t = frame[rows].astype(float) * dt
            ctx = SegmentContext(obs=obs, sigma=sigma[rows], interp=interp[rows], dt=dt, meas=m,
                                 cov=None if cov is None else cov[rows])  # fmt: skip
            for name, kind, p in specs:
                sp, fb = smooth_segment(kind, p, t, raw, ctx)
                res[name][0][rows] = sp
                res[name][1][rows] = int(fb)
        given = [c for c in df.columns if c not in INTERNAL]
        ren, added = self._added([n for n, _, _ in specs], single)
        clash = sorted(set(added) & set(given))
        if clash:
            raise ValueError(f"input columns {clash} clash with columns the smoother adds; rename them first")
        out = df[given].copy()
        for name, _, _ in specs:
            sp, fb = res[name]
            out[ren[f"x_{name}"]] = sp[:, 0] + x0
            out[ren[f"y_{name}"]] = sp[:, 1] + y0
            out[ren[f"fb_{name}"]] = fb
        if "seg_len" in out.columns:  # structural input from segment(): recompute, never trust a stale count
            sl = np.zeros(len(out), dtype="int64")
            for rows in groups:
                sl[rows] = len(rows)
            out["seg_len"] = sl
        out = _round(out, added)
        out.attrs["origin"] = (x0, y0)
        return out.reset_index(drop=True)


def _cov(table: pd.DataFrame) -> np.ndarray:
    """(n, 2, 2) covariance from the canonical columns; NaN rows where any of the three is not finite."""
    xx, xy, yy = (table[c].to_numpy(dtype=float) for c in segments.COV)
    out = np.stack([np.stack([xx, xy], axis=-1), np.stack([xy, yy], axis=-1)], axis=-2)
    bad = ~np.isfinite(out).all(axis=(1, 2))
    out[bad] = np.nan
    return out


def _interp_hold(obs: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Measured positions linearly interpolated across unmeasured rows; held before the first and after the last."""
    k = np.arange(len(obs))
    return np.column_stack([np.interp(k, k[m], obs[m, j]) for j in range(2)])


def _round(out: pd.DataFrame, added: list[str]) -> pd.DataFrame:
    """Round the added numeric position columns to three decimals (millimetres), as the smoother always has."""
    num = [c for c in added if c not in ("seg", "interp", "fill_reason") and not c.startswith("fb_") and c != "fallback"]
    out[num] = out[num].round(3)
    return out
