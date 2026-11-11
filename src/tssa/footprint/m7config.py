"""M7 settings: every M7_* setting of the speed study as of 2026-11-10 (snapshot
docs/study-snapshot-2026-11-10.json; run 20261110_161113_s0 recorded the same values).

Generated from the study's settings.py; field names keep the study's names so the ported solver reads them
unchanged (`S.M7_...`). Lengths are feet, angles degrees, image quantities pixels. The comment above each
field is the study's own (decisions and their dates). Runner settings (M7_KIN_*, M7_LOWCONF_S, M7_CLASSES)
live in FootprintConfig; workers and the solve cache are VehFootPrinter arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _m7_wheel_r_ref_ft():
    return {5: 1.6, 7: 1.45}

def _m7_priors_ft():
    return {'bus': {'Lw': (24.0, 5.0, 12.0, 30.0), 'af': 0.2916666666666667, 'sf': 2.0, 'ar': 0.375, 'sr': 2.0, 'W': (8.5, 0.3, 7.0, 8.8), 'H': (11.0, 1.0), 'c': (1.0, 0.5, 0.3, 3.5), 'i_f': 1.0, 'i_r': 0.3, 'wheel_r': 1.6}, 'truck': {'Lw': (15.0, 5.0, 8.0, 26.0), 'af': 0.26666666666666666, 'sf': 2.0, 'ar': 0.4, 'sr': 3.0, 'W': (8.0, 0.8, 6.0, 8.8), 'H': (10.0, 2.0), 'c': (1.5, 1.0, 0.3, 4.0), 'i_f': 0.8, 'i_r': 0.3, 'wheel_r': 1.6}, 'pickup_van': {'Lw': (12.0, 1.5, 9.0, 15.0), 'af': 0.25, 'sf': 1.5, 'ar': 0.35000000000000003, 'sr': 1.5, 'W': (6.7, 0.4, 5.8, 7.2), 'H': (7.0, 1.5), 'c': (1.0, 0.4, 0.3, 2.0), 'i_f': 0.5, 'i_r': 0.5, 'wheel_r': 1.3}}

def _m7_subtypes():
    return {5: ('bus',), 7: ('truck', 'pickup_van')}

def _m7_loose_w_ft():
    return {'bus': (8.5, 0.5, 7.5, 8.8), 'truck': (8.0, 0.8, 6.5, 8.8), 'pickup_van': (6.7, 0.4, 6.0, 7.2)}

def _m7_inset_sd_lo_hi_ft():
    return {'bus': (0.5, 0.0, 2.0), 'truck': (0.5, 0.0, 2.0), 'pickup_van': (0.3, 0.0, 1.5)}

def _m7_max_rigid_len_ft():
    return {'bus': 45.0, 'truck': 40.0, 'pickup_van': 24.0}

def _m7_measured_sd_ft():
    return {'L': 1.0, 'W': 0.5}


@dataclass(frozen=True)
class M7Config:
    """M7 solver settings (see the module docstring)."""

    # protrusion below the body line, px at a box height of 100 px (scaled by h / 100)
    M7_PROTRUDE_PX: float = 2.0
    # accepted protrusion width, in expected tyre diameters. Phase 0 retune (2026-11-08) of the maximum from 1.6: on
    # the seed-0 crops most missed tyres were rejected as wide (median 2.2 diameters: wheel wells, mud flaps, oblique
    # treads). The minimum stays 0.4: lowering it admits raster stair-steps along an oblique end face, and recovers
    # few tyres
    M7_TYRE_W_MIN: float = 0.4
    # accepted protrusion width, in expected tyre diameters. Phase 0 retune (2026-11-08) of the maximum from 1.6: on
    # the seed-0 crops most missed tyres were rejected as wide (median 2.2 diameters: wheel wells, mud flaps, oblique
    # treads). The minimum stays 0.4: lowering it admits raster stair-steps along an oblique end face, and recovers
    # few tyres
    M7_TYRE_W_MAX: float = 3.0
    # deeper protrusions are mask notches, not tyres (Phase 0 retune)
    M7_MAX_DEPTH_TYRES: float = 1.0
    # accepted candidates closer than this many diameters: keep the deepest (Phase 0 retune)
    M7_DEDUP_TYRES: float = 0.5
    # tyre radius for the size check and gate; independent of the subtype
    M7_WHEEL_R_REF_FT: dict = field(default_factory=_m7_wheel_r_ref_ft)
    # frozen gate g_c = max(min, tyres * D_c); smoke run: 1.5 let
    M7_GATE_MIN_PX: float = 8.0
    # frozen gate g_c = max(min, tyres * D_c); smoke run: 1.5 let
    M7_GATE_TYRES: float = 0.75
    # body-end candidates 18-30 px away match near vehicles (D about 30 px)
    M7_SIGMA_WHEEL_PX: float = 1.5
    # fewer trusted outline columns: no candidates
    M7_MIN_OUTLINE_COLS: int = 8
    # prior sets in feet: (mean, sd, lo, hi) for L_w and W, (mean, sd) for H; overhang mean = ratio * L_w
    M7_PRIORS_FT: dict = field(default_factory=_m7_priors_ft)
    M7_SUBTYPES: dict = field(default_factory=_m7_subtypes)
    # tandem spacing prior (mean, sd)
    M7_TANDEM_FT: tuple = (4.5, 0.7)
    # smoke run 2026-11-08: tandem won on 3 of 5 single-axle vehicles (body-end candidates fill the third rear
    # target); off by default until the user decides. A tandem vehicle then gets its effective rear axle at one of the
    # two tyres.
    M7_ALLOW_TANDEM: bool = False
    # user decision 2026-11-08: width is not fitted. Real widths slid to every lower bound tried (rounded vehicle ends
    # make the visible bottom outline narrower than a box). W is held at the class prior mean by a very tight prior;
    # its source is 'prior'. 2026-11-09 (user: "Loosen width prior"): W fitted again under the loose prior below
    M7_FIX_WIDTH: bool = False
    # when M7_FIX_WIDTH is True
    M7_FIXED_W_SD_FT: float = 0.01
    # loose width prior (mean, sd, lo, hi) ft, used together with the fitted inset; replaces the prior set's W. Lower
    # bounds sit well above where widths drifted when W was fitted with a fixed inset (7.0 ft for buses).
    M7_LOOSE_W_FT: dict = field(default_factory=_m7_loose_w_ft)
    # user decision 2026-11-09: the tyre inset is fitted instead of the width. One static d, the mean inset of the
    # near-side tyres from the body's near bottom edge (ft): i_f = d + (i_f0 - i_r0) / 2, i_r = d - (i_f0 - i_r0) / 2
    # with i_f0, i_r0 the prior-set insets (their difference is kept; symmetric fits use 0). Prior (mean = (i_f0 +
    # i_r0) / 2, sd, lo, hi).
    M7_FIT_INSET: bool = True
    M7_INSET_SD_LO_HI_FT: dict = field(default_factory=_m7_inset_sd_lo_hi_ft)
    # when M7_FIT_INSET is False
    M7_FIXED_INSET_SD_FT: float = 0.01
    M7_TANDEM_SPACING_BOUNDS_FT: tuple = (3.0, 7.0)
    # lower bound of L_f, L_r, H
    M7_MIN_DIM_FT: float = 0.5
    M7_MAX_RIGID_LEN_FT: dict = field(default_factory=_m7_max_rigid_len_ft)
    # far-side wheel targets
    M7_SIGMA_FAR_PX: float = 3.0
    # smoke run: far-side targets mostly captured clutter (track 476: half the matches, 25 px median)
    M7_FAR_TARGETS: bool = False
    M7_SIGMA_ENV_PX: float = 2.0
    M7_SIGMA_EXT_PX: float = 3.0
    M7_ENV_MAX_COLS: int = 40
    M7_ENV_EFF_N: int = 10
    M7_MIN_ENV_COLS: int = 12
    M7_SIGMA_SLIP_FT: float = 0.1
    M7_SIGMA_ACC_FT: float = 0.05
    M7_SIGMA_YAW_DEG: float = 0.5
    M7_SIGMA_CURV_DEG: float = 0.2
    M7_STEER_MAX_DEG: float = 40.0
    M7_CURV_EPS_DEG: float = 0.2
    # 12 -> 20 (2026-11-09): box truck 452 needed 18; 20 -> 30 (2026-11-10, user decision): the W-held refits of
    # trucks 452 and 488 were still descending monotonically at 20 (2 and 4 more)
    M7_OUTER_ITERS: int = 30
    # 12 -> 20 (2026-11-09): box truck 452 needed 18; 20 -> 30 (2026-11-10, user decision): the W-held refits of
    # trucks 452 and 488 were still descending monotonically at 20 (2 and 4 more)
    M7_TOL: float = 0.0001
    # function evaluations per inner robust least-squares solve
    M7_INNER_NFEV: int = 200
    # first outer rounds use IRLS (robust basin), later ones the exact robust loss
    M7_IRLS_ROUNDS: int = 4
    # function evaluations per IRLS round
    M7_IRLS_NFEV: int = 60
    M7_USE_EXTENTS: bool = True
    # envelope columns outside the box's projected u-span are evaluated this far inside the span (px; at most a
    # quarter of the span). 0 = at the extreme vertex, where the envelope steps when two vertices swap order. Bus 417,
    # 2026-11-09.
    M7_ENV_CLAMP_PX: float = 3.0
    # hold each frame's near side fixed within a round of the fit (width debugging 2026-11-09: fits pinned a frame
    # exactly where the camera crosses the vehicle's centreline and the side switches).
    M7_SIDE_FREEZE: bool = True
    # bottom extents use outline columns at most this many protrusion thresholds above the body line
    M7_EXT_ABOVE_THR: float = 3.0
    # evidence-score units: a non-default axle config or subtype must win by more
    M7_MODEL_MARGIN: float = 3.0
    M7_FRONT_MIN_YAW_DPS: float = 3.0
    M7_FRONT_MARGIN: float = 3.0
    # user decision 2026-11-09: "travel" = the front is the end facing the direction of travel (the turning test of
    # design 8.4 is run as a diagnostic, front_turning); "turning" = the design rule. The turning test made 6 of 25
    # real vehicles reverse through the junction.
    M7_FRONT_RULE: str = 'travel'
    # user decision 2026-11-09: a width on its limit does not fail the solve; it is held there and its source is
    # "prior"
    M7_W_BOUND_FAILS: bool = False
    # user decision 2026-11-10 ("Hold W at prior"): when one px of silhouette is worth more than this much width (ft
    # per px at the far end, median over frames), W is held at the class prior (sd M7_FIXED_W_SD_FT) and sourced
    # "prior". The far end is extended one way only (bike racks, rounded corners, mask bleed). None = never held.
    M7_W_HOLD_FT_PER_PX: float = 0.3
    # a static within this of a bound is at the bound (was a relative 1e-4)
    M7_BOUND_TOL_FT: float = 0.01
    M7_GROUP_MIN_FRAMES: int = 5
    M7_MIN_SUPPORT_FRAMES: int = 10
    # wheel 3.0 -> 3.5 px (user, 2026-11-09): real tyre extraction noise
    M7_MAX_WHEEL_PX: float = 3.5
    # wheel 3.0 -> 3.5 px (user, 2026-11-09): real tyre extraction noise
    M7_MAX_ENV_PX: float = 4.0
    # wheel 3.0 -> 3.5 px (user, 2026-11-09): real tyre extraction noise
    M7_MAX_SLIP_FT: float = 0.3
    M7_ARTIC_OUTLIER_SHARE: float = 0.3
    M7_PRIOR_DOM_RATIO: float = 0.7
    M7_PRIOR_ONLY_RATIO: float = 0.9
    M7_MEASURED_SD_FT: dict = field(default_factory=_m7_measured_sd_ft)
    # box truck 452 debugging (2026-11-09): a tyre candidate is the tyre's lowest image point, predicted from a circle
    # of the prior set's wheel_r; near the camera it is far from the ground contact
    M7_TYRE_LOWEST_POINT: bool = True
    # tyre candidates near the left/right image edge (distorted u): sigma x 1 at M7_EDGE_BAND_PX inwards, rising
    # linearly to x M7_EDGE_SIGMA_MULT at the edge (452: 8-17 px residuals at u 20-115 px, 1-3 px beyond 165 px; a
    # lens-model limit)
    M7_EDGE_BAND_PX: float = 160.0
    # tyre candidates near the left/right image edge (distorted u): sigma x 1 at M7_EDGE_BAND_PX inwards, rising
    # linearly to x M7_EDGE_SIGMA_MULT at the edge (452: 8-17 px residuals at u 20-115 px, 1-3 px beyond 165 px; a
    # lens-model limit)
    M7_EDGE_SIGMA_MULT: float = 8.0
    # pair start (bus 16 debugging, 2026-11-09): wheelbase and pose from tyre-candidate pairs with a steady ground
    # distance
    M7_PAIR_START: bool = True
    # pairs within this of the mode support it; needs M7_GROUP_MIN_FRAMES distinct frames
    M7_PAIR_TOL_FT: float = 1.0
    M7_MOVING_MPH: float = 3.0
    # coarse-to-fine starts (smoke run: tight gates alone missed the wheel basin)
    M7_COARSE_GATE_MULT: float = 3.0
    # second L_w start: prior mean minus this (bus and truck)
    M7_START_SHIFT_FT: float = 7.0
    # a two-segment body line is kept when its cost is below this fraction of one line's
    M7_TWO_SEG_GAIN: float = 0.7
    # mask components kept for the observations: at least this share of the summed (area + perimeter) (the study's
    # infer.M7_MIN_COMPONENT_FRAC, applied when its SAM cache was written; tssa stores every component)
    M7_MIN_COMPONENT_FRAC: float = 0.05

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
