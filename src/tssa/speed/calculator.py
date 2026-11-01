"""SpeedCalculator: velocity and speed of tracks from their (smoothed) ground positions.

Inputs:  a track table with frame, track and two position columns in metres (default x, y; for example the
         `<pos>_smooth` columns written by tssa.smooth.TrackSmoother), optionally the `seg` column of that smoother. A
         velocity never spans two segments: rows are grouped by (track, seg), or by track when there is no seg.
Outputs: the input table (same rows and order) plus vx, vy and speed. Speeds are in mph by default
         (`units="metric"` gives m/s); the unit is in `out.attrs["speed_unit"]`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNITS = ("imperial", "metric")
SPEED_PER_MPS = {"imperial": 2.2369362920544, "metric": 1.0}  # output speed unit per m/s: mph, m/s
SPEED_UNIT = {"imperial": "mph", "metric": "m/s"}
ADDED = ["vx", "vy", "speed"]


def frame_velocity(t: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """(n, 2) velocity at every sample: central differences (one-sided at the ends); NaN for a single sample."""
    if len(t) < 2:
        return np.full((len(t), 2), np.nan)
    return np.gradient(pos, t, axis=0)


def block_velocity(frames: np.ndarray, t: np.ndarray, pos: np.ndarray, interval: int) -> np.ndarray:
    """(n, 2) block velocity: blocks start every `interval` frames from the first frame.

    The velocity of a block is the displacement from the position at its first frame to the position at the start
    of the next block, over the elapsed time; every row of the block carries it. The last block ends at the last
    row (it may be shorter and uses its own duration) and takes all the rows from its start on. A row missing at a
    block boundary is replaced by the next row. NaN where a block holds a single row.
    """
    first, last = int(frames[0]), int(frames[-1])
    starts = np.arange(first, last, interval)
    out = np.full((len(frames), 2), np.nan)
    if not len(starts):
        return out
    lo = np.searchsorted(frames, starts, side="left")
    hi = np.minimum(np.searchsorted(frames, starts + interval, side="left"), len(frames) - 1)
    ok = hi > lo
    vel = np.full((len(starts), 2), np.nan)
    vel[ok] = (pos[hi[ok]] - pos[lo[ok]]) / (t[hi[ok]] - t[lo[ok]])[:, None]
    return vel[np.searchsorted(starts, frames, side="right") - 1]


class SpeedCalculator:
    """Speed from smoothed positions, at every frame or over blocks of N frames.

    fps:       frames per second of the tracks.
    xy_cols:   names of the two position columns (metres).
    interval:  None (default) gives the velocity at every frame by central differences of the positions. An int
               N >= 2 gives block velocities: the position is taken every N frames, the velocity of a block is the
               displacement over the block divided by its duration, and every frame of the block shows it.
    units:     "imperial" (default, mph) or "metric" (m/s) for the output vx, vy and speed.
    """

    def __init__(self, fps: float = 10.0, xy_cols: tuple = ("x", "y"), interval: int | None = None,
                 units: str = "imperial") -> None:  # fmt: skip
        if fps <= 0:
            raise ValueError("fps must be positive")
        if len(xy_cols) != 2 or xy_cols[0] == xy_cols[1]:
            raise ValueError(f"xy_cols must be a pair of two different column names, got {xy_cols!r}")
        if interval is not None and (int(interval) != interval or interval < 2):
            raise ValueError(f"interval must be None or an integer >= 2, got {interval!r}")
        if units not in UNITS:
            raise ValueError(f"units must be one of {UNITS}, got {units!r}")
        self.fps, self.xy_cols, self.interval, self.units = fps, tuple(xy_cols), interval, units

    def calculate(self, tracks: pd.DataFrame) -> pd.DataFrame:
        """The input table plus vx, vy and speed; rows with a missing position get NaN."""
        xc, yc = self.xy_cols
        missing = [c for c in ("frame", "track", xc, yc) if c not in tracks.columns]
        if missing:
            raise ValueError(f"tracks lack required column(s) {missing}")
        clash = sorted(set(ADDED) & set(tracks.columns))
        if clash:
            raise ValueError(f"input columns {clash} clash with columns the calculator adds; rename them first")
        keys = ["track", "seg"] if "seg" in tracks.columns else ["track"]
        frames = tracks["frame"].to_numpy()
        pos = tracks[[xc, yc]].to_numpy(dtype=float)
        vel = np.full((len(tracks), 2), np.nan)
        for rows in tracks.groupby(keys, sort=False).indices.values():
            rows = rows[np.argsort(frames[rows], kind="stable")]
            rows = rows[np.isfinite(pos[rows]).all(axis=1)]
            f = frames[rows]
            if len(rows) < 2:
                continue
            if (np.diff(f) == 0).any():
                raise ValueError(f"track {tracks['track'].to_numpy()[rows[0]]} has repeated frames within a segment")
            t = f / self.fps
            vel[rows] = frame_velocity(t, pos[rows]) if self.interval is None else block_velocity(f, t, pos[rows], self.interval)
        vel = vel * SPEED_PER_MPS[self.units]
        out = tracks.copy()
        out["vx"], out["vy"] = vel[:, 0].round(3), vel[:, 1].round(3)
        out["speed"] = np.hypot(vel[:, 0], vel[:, 1]).round(3)
        out.attrs["speed_unit"] = SPEED_UNIT[self.units]
        return out
