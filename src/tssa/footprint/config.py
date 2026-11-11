"""Settings of VehFootPrinter (design docs/2026-11-10-footprint-design.md, section 12).

Distances and sizes are in FEET (user decision Q2); positions stay in the homography's world metres; speeds in mph.
Defaults are the speed study's values (snapshot docs/study-snapshot-2026-11-10.json, settings.values); the name of
the study setting is given beside each field.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

FT = 0.3048  # m per ft
MPS_TO_MPH = 2.2369362920544


def _size_ft():
    return {2: (15.0, 6.0), 5: (40.0, 8.5), 7: (25.0, 8.0)}  # SIZE_FT (length, width)


def _priors_ft():
    return {  # PRIORS_FT (M4 wheel geometry)
        2: dict(height=5, wheelbase=9, track=5, front_oh=3, rear_oh=3, wheel_r=1.1),
        7: dict(height=10, wheelbase=15, track=6.5, front_oh=4, rear_oh=6, wheel_r=1.6),
        5: dict(height=11, wheelbase=24, track=7, front_oh=7, rear_oh=9, wheel_r=1.6),
    }


def _chains():
    return {2: ("M6", "M2"), 5: ("M7", "M6", "M2"), 7: ("M7", "M6", "M3", "M2")}  # design 7.8


@dataclass(frozen=True)
class FootprintConfig:
    """VehFootPrinter settings; `with_overrides(**kw)` returns a changed copy (unknown names raise)."""

    fps: float = 10.0
    box_cols: tuple = ("x", "y", "w", "h")
    cls_col: str = "cls"
    size_ft: dict = field(default_factory=_size_ft)
    default_size_ft: tuple = (15.0, 6.0)
    priors_ft: dict = field(default_factory=_priors_ft)
    chains: dict = field(default_factory=_chains)
    default_chain: tuple = ("M2",)
    # shared checks (design 7.10)
    max_ft_per_px: float = 2.0  # GATE_MAX_FT_PER_PX
    gate_buffer_ft: float = 100.0  # GATE_BUFFER_FT (only with a zone)
    edge_px: float = 2.0  # EDGE_PX
    # shared noise model (design 7.9)
    sigma_k: float = 0.03  # RTS_SIGMA_K
    fill_meas_scale: float = 4.0  # FILL_MEAS_SCALE
    jacobian_eps_px: float = 0.5  # smoothlib.ground.JACOBIAN_EPS (central difference half step)
    # motion heading and orientation (design 10)
    heading_min_mph: float = 3.0  # HEADING_MIN_MPH
    orient_q: float = 1.0  # ORIENT_Q
    orient_sigma: float = 0.1  # ORIENT_SIGMA
    orient_reach_frames: int = 10  # ORIENT_NATIVE_REACH_FRAMES (R10)
    # M3
    m3_resid_px: float = 0.5  # m3.RESID_OK_PX
    # M4 (base design 6.1)
    m4_wheelbase_tol: float = 0.30  # WHEELBASE_TOL
    m4_axis_tol_deg: float = 30.0  # AXIS_TOL_DEG
    m4_parallel_tol_deg: float = 15.0  # PARALLEL_TOL_DEG
    m4_track_tol: float = 0.30  # TRACK_TOL
    m4_long_offset_frac: float = 0.25  # LONG_OFFSET_FRAC
    m4_min_contact_sep_ft: float = 0.5  # MIN_CONTACT_SEP_FT
    m4_near_dv_px: float = 3.0  # NEAR_DV_PX
    m4_near_bottom_frac: float = 0.10  # NEAR_BOTTOM_FRAC
    m4_camera_away_dot: float = 0.5  # CAMERA_AWAY_DOT
    m4_front_conflict_min_rows: int = 10  # FRONT_CONFLICT_MIN_ROWS
    m4_front_conflict_share: float = 0.5  # FRONT_CONFLICT_SHARE
    wheel_landmark: str = "hub"  # WHEEL_LANDMARK ("hub" or "contact")
    # M5
    m5_min_cols: int = 8  # M5_MIN_COLS
    m5_max_resid_ft: float = 1.0  # M5_MAX_RESID_FT
    # M6
    m6_size_gate: float = 0.5  # M6_MIN_SIZE_FRAC
    m6_min_iou: float = 0.3  # M6_MATCH_IOU
    m6_measured_size_classes: tuple = (2,)  # M6_MEASURED_SIZE_CLASSES
    measured_size_min_rows: int = 5  # my call (design 10.3)
    # M7 rows (design 7.7) and confidence (design 10.4)
    m7_classes: tuple = (5, 7)  # M7_CLASSES
    admit_kinematic: bool = True
    kin_meas_scale: float = 2.0  # M7_KIN_MEAS_SCALE
    kin_sd_floor_ft: float = 0.5  # M7_KIN_SD_FLOOR_FT
    lowconf_s: float = 2.0  # M7_LOWCONF_S
    m7: object = None  # M7Config; None = its defaults (filled in __post_init__)
    # camera (design 8)
    camera_rms_px: float = 1.0  # CAMERA_RMS_PX
    camera_max_px: float = 3.0  # CAMERA_MAX_PX
    camera_pp_prior: float = 1e-3  # CAMERA_PP_PRIOR (R18)
    camera_free_principal: bool = True  # CAMERA_FREE_PRINCIPAL
    # the vehicle smoother estimate() builds when none is passed (design 11.3)
    rts_q: float = 2.0  # RTS_Q, m^2/s^3
    split_speed_mps: float = 30.0  # my call (design 9)
    fill_max_gap: int = 20  # FILL_MAX_GAP, frames
    gap_max_speed_mph: float = 60.0  # GAP_MAX_SPEED_MPH

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if len(self.box_cols) != 4 or len(set(self.box_cols)) != 4:
            raise ValueError(f"box_cols must be four different column names, got {self.box_cols!r}")
        for c, lw in self.size_ft.items():
            if len(lw) != 2 or min(lw) <= 0:
                raise ValueError(f"size_ft[{c}] must be a positive (length, width), got {lw!r}")
        if self.m7 is None:
            from .m7config import M7Config

            object.__setattr__(self, "m7", M7Config())

    def with_overrides(self, **overrides) -> "FootprintConfig":
        """Copy with some settings replaced; an unknown name raises ValueError."""
        known = {f.name for f in fields(self)}
        bad = sorted(set(overrides) - known)
        if bad:
            raise ValueError(f"unknown setting(s) {bad}; known: {sorted(known)}")
        return replace(self, **overrides)

    def size_of(self, cls) -> tuple:
        """Nominal (length, width) in ft of a class."""
        return tuple(self.size_ft.get(int(cls), self.default_size_ft)) if cls == cls else tuple(self.default_size_ft)

    def chain_of(self, cls) -> tuple:
        return tuple(self.chains.get(int(cls), self.default_chain)) if cls == cls else tuple(self.default_chain)
