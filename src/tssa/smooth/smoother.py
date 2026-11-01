"""TrackSmoother: segment projected ground tracks, fill gaps, smooth the positions with a chosen method.

Inputs:  track tables with frame, track and two position columns in metres (a ground plane the user projected
         beforehand; default names x, y), optionally interp (0 detected, 1 filled upstream) and a per-row sigma
         column. Any other columns are carried through. No homography is involved.
Outputs: the input columns first, unchanged and in their order (NaN on rows filled here), then the added columns.
         `smooth()` adds seg, fill_reason, <pos>_smooth (two columns) and fallback (1 = segment too short for the
         method, raw positions copied); `interp` (0 detected, 1 filled upstream, 2 filled here) is updated in place,
         or added at the front when the input lacks it. `compare()` adds seg, fill_reason, then <pos>_<m> (two
         columns) and fb_<m> per method <m>. Speeds are not computed here: pass the smoothed positions to
         tssa.speed.SpeedCalculator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import segments
from .config import SmoothConfig
from .methods import PRESETS, SegmentContext, available_methods, resolve, smooth_segment

INTERNAL = {"_x", "_y", "_sigma"}  # helper columns of segments.prepare
SMOOTH_SUFFIX = "_smooth"  # smoothed positions are <pos>_smooth, so the input positions stay untouched


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

    def smooth(self, tracks: pd.DataFrame, ref_tracks: pd.DataFrame | None = None) -> pd.DataFrame:
        """Smoothed table for the configured method."""
        df, grid = self._grid(tracks, ref_tracks)
        out = self._apply(grid, [(self.method, self._kind, self._merged)])
        return self._finish(out, grid, df, [self.method], single=True)

    def compare(self, tracks: pd.DataFrame, methods=None, ref_tracks: pd.DataFrame | None = None) -> pd.DataFrame:
        """Every method side by side on one segmentation and gap fill.

        `methods` is a list of names or a dict name -> parameters; None means every preset (ma, sg5, sg7, rts).
        """
        if methods is None:
            methods = list(PRESETS)
        specs = list(methods.items()) if isinstance(methods, dict) else [(m, None) for m in methods]
        resolved = [(name, *resolve(name, p)) for name, p in specs]
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
        for rows in keys.values():
            g = grid.iloc[rows]
            t = g["frame"].to_numpy(dtype=float) * dt
            pos = g[["x_raw", "y_raw"]].to_numpy(dtype=float)
            ctx = SegmentContext(
                obs=g[["x_obs", "y_obs"]].to_numpy(dtype=float), sigma=g["sigma"].to_numpy(dtype=float),
                interp=g["interp"].to_numpy(), dt=dt,
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
        if single:
            m = names[0]
            ren = {f"x_{m}": f"{xc}{SMOOTH_SUFFIX}", f"y_{m}": f"{yc}{SMOOTH_SUFFIX}", f"fb_{m}": "fallback"}
            added = ["seg", "fill_reason", *ren.values()]
        else:
            ren = {f"{c}_{m}": f"{xc if c == 'x' else yc}_{m}" for m in names for c in ("x", "y")}
            added = ["seg", "fill_reason"]
            added += [c for m in names for c in (ren[f"x_{m}"], ren[f"y_{m}"], f"fb_{m}")]
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
        num = [c for c in added if c not in ("seg", "interp", "fill_reason") and not c.startswith("fb_") and c != "fallback"]
        out[num] = out[num].round(3)
        out.attrs["origin"] = (x0, y0)
        return out
