"""Settings of the track smoothing pipeline (segmentation, gap fill, column names).

Inputs:  keyword overrides of the defaults below.
Outputs: `SmoothConfig`, the single source of these settings; a `TrackSmoother` holds one instance, so two smoothers
         with different settings can coexist.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class SmoothConfig:
    """Pipeline settings. Positions are in metres (a projected ground plane); a track is split where it jumps.

    fps:           frames per second of the tracks.
    xy_cols:       names of the input position columns (metres); the smoothed positions keep these names.
    sigma_col:     optional column of the per-row position std-dev in metres; overrides the rts `sigma` row by row.
    split_speed:   a track is cut between two rows whose implied speed (m/s) is above this (a jump / ID switch).
                   This and gap_max_speed are segmentation rules in m/s; the speeds of the output come from
                   tssa.speed.SpeedCalculator.
    fill_gaps:     fill the frames missing inside a track (False cuts the track at every gap instead).
    fill_max_gap:  longest gap in frames that may be filled.
    gap_max_speed: fastest implied speed in m/s across a filled gap.
    """

    fps: float = 10.0
    xy_cols: tuple = ("x", "y")
    sigma_col: str | None = None
    split_speed: float = 6.0
    fill_gaps: bool = True
    fill_max_gap: int = 20
    gap_max_speed: float = 3.0

    def __post_init__(self) -> None:
        if len(self.xy_cols) != 2 or self.xy_cols[0] == self.xy_cols[1] or not all(isinstance(c, str) for c in self.xy_cols):
            raise ValueError(f"xy_cols must be a pair of two different column names, got {self.xy_cols!r}")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.fill_max_gap < 0:
            raise ValueError("fill_max_gap must be >= 0")
        if self.split_speed <= 0 or self.gap_max_speed <= 0:
            raise ValueError("split_speed and gap_max_speed must be positive")

    def with_overrides(self, **overrides) -> "SmoothConfig":
        """Copy with some settings replaced; an unknown name raises ValueError."""
        known = {f.name for f in fields(self)}
        bad = sorted(set(overrides) - known)
        if bad:
            raise ValueError(f"unknown setting(s) {bad}; known: {sorted(known)}")
        return replace(self, **overrides)
