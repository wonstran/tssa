"""M4 wheel geometry (ported from the speed study's tests/test_m4.py; base design 6.1). Fixtures are in ft in a local frame (x = E, y = N), placed at a UTM origin.

Every fixture uses PinholeGround(look_at(camera, aimed at the origin, f = 500, principal point (320, 240))), and
each bbox is the projection of the nominal 3D box of the class (bus 8.5 x 40 x 11 ft) centred on the true centre.
"""

import numpy as np
import pandas as pd
import pytest

from tssa.footprint import FootprintConfig, VehFootPrinter
from tssa.footprint.config import FT
from tssa.footprint.geometry import diff_directed, rect_corners, rot_ccw, unit_from_compass
from tssa.footprint.locators.m4 import front_conflict, solve_m4
from tssa.transform.camera import PinholeGround, hub_to_contact, look_at

PRIORS_FT = FootprintConfig().priors_ft
SIZE_FT = FootprintConfig().size_ft

ORIGIN = np.array([580000.0, 2852000.0])  # nonzero UTM origin, so local/UTM mix-ups show
BUS, CAR, TRUCK = 5, 2, 7
TRUE_FT = np.array([0.0, -1.0])
TRUE = ORIGIN + TRUE_FT * FT
PHYS = {"RF": (3.5, 12), "LF": (-3.5, 12), "RB": (3.5, -12), "LB": (-3.5, -12)}
PERMS = {"as_labelled": ["RF", "LF", "RB", "LB"], "fb_swapped": ["RB", "LB", "RF", "LF"],
         "perm180": ["LB", "RB", "LF", "RF"], "lr_swapped": ["LF", "RF", "LB", "RB"]}
REAR_OBLIQUE = (40, -70, 23)  # ft


def make_dh(cam_ft):
    C = np.array([ORIGIN[0], ORIGIN[1], 0.0]) + np.asarray(cam_ft, dtype=float) * FT
    return PinholeGround(look_at(C, target=(ORIGIN[0], ORIGIN[1], 0.0), origin=ORIGIN))


def utm(pts_ft):
    return ORIGIN + np.asarray(pts_ft, dtype=float) * FT


def project(dh, pts_ft):
    return dh.world_to_image(utm(pts_ft))


def nominal_box(dh, cls=BUS, centre_ft=TRUE_FT, heading=0.0):
    """xywh of the projected nominal 3D box (length x width x height) centred on the ground at centre_ft."""
    L, W = SIZE_FT[cls]
    H = PRIORS_FT[cls]["height"]
    base = rect_corners(utm(centre_ft), heading, L * FT, W * FT)
    pts = np.vstack([np.column_stack([base, np.zeros(4)]), np.column_stack([base, np.full(4, H * FT)])])
    uv = dh.cam.project(pts)
    x0, y0 = uv.min(axis=0)
    x1, y1 = uv.max(axis=0)
    return np.array([x0, y0, x1 - x0, y1 - y0])


def wheels_ft(cls, centre_ft, heading):
    """Physical contacts {RF, LF, RB, LB} (ft) of a class-prior vehicle with body centre centre_ft facing heading."""
    p = PRIORS_FT[cls]
    f = unit_from_compass(heading)
    left = rot_ccw(f)
    axle = np.asarray(centre_ft, float) - (p["front_oh"] - p["rear_oh"]) / 2.0 * f
    out = {}
    for side, s in (("R", -1.0), ("L", 1.0)):
        for end, e in (("F", 1.0), ("B", -1.0)):
            out[side + end] = axle + s * p["track"] / 2.0 * left + e * p["wheelbase"] / 2.0 * f
    return out


def kp(dh, phys, order=("RF", "LF", "RB", "LB")):
    return project(dh, [phys[k] for k in order])


def ft_err(centre, true=TRUE):
    return np.linalg.norm(np.asarray(centre) - true) / FT


def test_wheels_helper_matches_design_bus():
    w = wheels_ft(BUS, TRUE_FT, 0.0)
    for k, v in PHYS.items():
        assert np.allclose(w[k], v)


@pytest.mark.parametrize("case", PERMS)
def test_bus_permutations_recover_centre(case):  # psi_ref = 0 (north), rear-oblique camera
    dh = make_dh(REAR_OBLIQUE)
    r = solve_m4(kp(dh, PHYS, PERMS[case]), nominal_box(dh), 0.0, BUS, dh, dh.image_to_world)
    assert r.valid and r.fail_reason == ""
    assert ft_err(r.centre) < 1e-6
    assert r.label_case == case
    assert r.interior_source == "other_pair" and r.side_reject == ""
    assert r.front_flipped == (case in ("fb_swapped", "perm180"))
    assert diff_directed(np.degrees(np.arctan2(*r.d_front)), 0.0) < 1e-6
    assert r.wheelbase_ft == pytest.approx(24.0, abs=1e-6) and r.track_ft == pytest.approx(7.0, abs=1e-6)


def test_malformed_outward_pair_rejected_then_ambiguous():  # camera (60,-30,23) ft; other pair (10.5,+-1)
    dh = make_dh((60, -30, 23))
    phys = dict(PHYS, LF=(10.5, 1), LB=(10.5, -1))
    r = solve_m4(kp(dh, phys), nominal_box(dh), 0.0, BUS, dh, dh.image_to_world)
    assert r.pair_side == "R"
    assert r.side_reject == "wheelbase" and r.fail_reason == "side_ambiguous" and not r.valid
    assert np.isnan(r.centre).all() and r.interior_source == ""


def test_short_far_pair_camera_away_success():  # camera (60,-5,23) ft; other pair (-3.5,+-1)
    dh = make_dh((60, -5, 23))
    phys = dict(PHYS, LF=(-3.5, 1), LB=(-3.5, -1))
    r = solve_m4(kp(dh, phys), nominal_box(dh), 0.0, BUS, dh, dh.image_to_world)
    assert r.side_reject == "wheelbase" and r.interior_source == "camera_away"
    assert r.valid and r.label_case == "as_labelled"
    assert ft_err(r.centre) < 1e-6
    assert np.isnan(r.track_ft) and r.wheelbase_ft == pytest.approx(24.0, abs=1e-6)


@pytest.mark.parametrize("offset,reason", [((0, 8), "longitudinal"), ((-6, 0), "lateral")])
def test_misaligned_credible_pair_rejected(offset, reason):  # far pair passes own acceptance, still rejected
    dh = make_dh((60, -30, 23))
    phys = dict(PHYS, LF=tuple(np.add(PHYS["LF"], offset)), LB=tuple(np.add(PHYS["LB"], offset)))
    r = solve_m4(kp(dh, phys), nominal_box(dh), 0.0, BUS, dh, dh.image_to_world)
    assert r.pair_side == "R"
    assert r.side_reject == reason and r.interior_source == ""
    assert r.fail_reason == "side_ambiguous" and not r.valid  # camera-away then fails its bbox-bottom gate


def test_contact_far_below_bbox_bottom_fails_near_side_gate():  # the bbox-bottom gate is two-sided (R8)
    dh = make_dh((60, -5, 23))  # the camera-away success fixture, with the bbox lifted by 30% of its height
    phys = dict(PHYS, LF=(-3.5, 1), LB=(-3.5, -1))
    b = nominal_box(dh)
    b[1] -= 0.3 * b[3]
    uv = kp(dh, phys)
    assert ((b[1] + b[3]) - uv[[0, 2], 1] < -0.1 * b[3]).all()  # both chosen contacts well below the bottom
    r = solve_m4(uv, b, 0.0, BUS, dh, dh.image_to_world)
    assert r.side_reject == "wheelbase" and r.fail_reason == "side_ambiguous" and r.interior_source == ""


def test_nonfinite_other_pair_waives_depth_cue_camera_away():
    dh = make_dh((60, -5, 23))
    uv = kp(dh, PHYS)
    uv[[1, 3]] = np.nan  # the left pair was not detected
    r = solve_m4(uv, nominal_box(dh), 0.0, BUS, dh, dh.image_to_world)
    assert r.side_reject == "nonfinite" and r.interior_source == "camera_away" and r.valid
    assert ft_err(r.centre) < 1e-6


def _rotated_left_pair(deg):
    """The true left pair rotated by deg (clockwise) about its own midpoint (-3.5, 0); length stays 24 ft."""
    d = 12.0 * unit_from_compass(deg)
    return dict(PHYS, LF=tuple(np.array([-3.5, 0.0]) + d), LB=tuple(np.array([-3.5, 0.0]) - d))


@pytest.mark.parametrize("deg,reason", [(35.0, "axis"), (20.0, "parallel")])
def test_rotated_other_pair_rejected(deg, reason):  # 35 deg fails its own axis check; 20 deg passes it, not parallel
    dh = make_dh(REAR_OBLIQUE)
    r = solve_m4(kp(dh, _rotated_left_pair(deg)), nominal_box(dh), 0.0, BUS, dh, dh.image_to_world)
    assert r.pair_side == "R" and r.side_reject == reason and r.interior_source == ""
    assert r.fail_reason == "side_ambiguous" and not r.valid  # camera-away then fails its gates here


def test_head_on_is_side_ambiguous():  # camera on the axis, far pair collapsed, no depth cue
    dh = make_dh((0, 70, 23))  # straight ahead of the north-facing bus
    phys = dict(PHYS, LB=PHYS["LF"])  # both left labels on the one visible left wheel
    r = solve_m4(kp(dh, phys), nominal_box(dh), 0.0, BUS, dh, dh.image_to_world)
    assert r.pair_side == "R" and r.side_reject == "degenerate"
    assert r.fail_reason == "side_ambiguous" and not r.valid


LAYOUTS = {  # view: (cls, centre ft, heading, camera ft)
    "frontal": (BUS, (0.0, -1.0), 0.0, (6, 75, 23)),
    "rear": (TRUCK, (2.0, 3.0), 180.0, (-5, 70, 23)),
    "oblique": (CAR, (-4.0, 6.0), 35.0, (45, -35, 23)),
}


@pytest.mark.parametrize("view", ["frontal", "rear", "oblique"])
def test_layouts_recover_centre(view):  # genuine far pair -> other_pair, centre exact
    cls, centre_ft, heading, cam = LAYOUTS[view]
    dh = make_dh(cam)
    r = solve_m4(kp(dh, wheels_ft(cls, centre_ft, heading)), nominal_box(dh, cls, centre_ft, heading), heading,
                 cls, dh, dh.image_to_world)
    assert r.valid and r.interior_source == "other_pair" and r.label_case == "as_labelled"
    assert ft_err(r.centre, utm(centre_ft)) < 1e-6


def test_hub_landmarks_recover_centre():
    dh = make_dh(REAR_OBLIQUE)
    rad = PRIORS_FT[BUS]["wheel_r"] * FT
    hubs = np.column_stack([utm([PHYS[k] for k in PERMS["as_labelled"]]), np.full(4, rad)])
    uv = dh.cam.project(hubs)
    r = solve_m4(uv, nominal_box(dh), 0.0, BUS, dh, lambda q: hub_to_contact(dh.cam, dh.undistort(q), rad))
    assert r.valid and ft_err(r.centre) < 1e-6


def test_reversing_bus_off_by_two_feet_and_flagged():
    dh = make_dh(REAR_OBLIQUE)
    r = solve_m4(kp(dh, PHYS), nominal_box(dh), 180.0, BUS, dh, dh.image_to_world)
    # correct labels read against a reversed front: the front flip also flips which side is "right" -> perm180
    assert r.valid and r.front_flipped and r.label_case == "perm180"
    assert ft_err(r.centre) == pytest.approx(2.0, abs=1e-6)
    assert ft_err(r.centre, utm((0.0, 1.0))) < 1e-6  # off along the axis
    t, ok = np.ones(10, int), np.ones(10, bool)
    assert front_conflict(t, ok, ok, 10, 0.5).all()
    assert not front_conflict(t[:9], ok[:9], ok[:9], 10, 0.5).any()  # fewer than 10 rows -> never flagged


def test_no_valid_pair():  # both pairs 2 ft long -> no_valid_pair
    dh = make_dh(REAR_OBLIQUE)
    phys = {"RF": (3.5, 1), "LF": (-3.5, 1), "RB": (3.5, -1), "LB": (-3.5, -1)}
    r = solve_m4(kp(dh, phys), nominal_box(dh), 0.0, BUS, dh, dh.image_to_world)
    assert not r.valid and r.fail_reason == "no_valid_pair" and r.pair_side == ""


def test_axis_off_reference_is_no_valid_pair():
    dh = make_dh(REAR_OBLIQUE)
    r = solve_m4(kp(dh, PHYS), nominal_box(dh), 45.0, BUS, dh, dh.image_to_world)
    assert r.fail_reason == "no_valid_pair"


def test_camera_invalid_for_hub_landmarks():  # to_ground None (camera failed) -> camera_invalid
    dh = make_dh(REAR_OBLIQUE)
    r = solve_m4(kp(dh, PHYS), nominal_box(dh), 0.0, BUS, dh, None)
    assert not r.valid and r.fail_reason == "camera_invalid"


def test_front_conflict_counts_valid_rows_and_needs_majority():
    track = np.array([1] * 12 + [2] * 12 + [3] * 3)
    valid = np.array([True] * 10 + [False] * 2 + [True] * 12 + [False] * 3)
    flipped = np.array([True] * 6 + [False] * 4 + [True] * 2 + [True] * 6 + [False] * 6 + [True] * 3)
    fc = front_conflict(track, valid, flipped, 10, 0.5)
    assert fc[track == 1].all() and not fc[track == 2].any() and not fc[track == 3].any()  # 2: exactly half


def _rows(dh, frames, track=1):
    b = nominal_box(dh)
    return pd.DataFrame([dict(track=track, frame=f, x=b[0], y=b[1], w=b[2], h=b[3], cls=BUS) for f in frames])


def _kp_table(uv, frames, track=1):
    return pd.DataFrame([dict(track=track, frame=f, iou=0.9, **{f"{c}{i}": uv[i, j] for i in range(4)
                                                                 for j, c in enumerate("uv")}) for f in frames])


def test_m4_locator_reasons_and_extras(monkeypatch):
    """Through VehFootPrinter with contact landmarks: a matched row recovers the centre, an unmatched one is no_match."""
    dh = make_dh(REAR_OBLIQUE)
    rows = _rows(dh, range(3))
    kps = _kp_table(kp(dh, PHYS, PERMS["fb_swapped"]), (0, 2))
    vfp = VehFootPrinter(dh, image_size=(640, 480), chains={BUS: ("M4",)}, wheel_landmark="contact")
    monkeypatch.setattr(vfp, "_reference_heading", lambda base: np.zeros(len(base)))  # psi_ref north, as the study
    loc = vfp.locate(rows, keypoints=kps).set_index("frame")
    r0 = loc.loc[0]
    assert r0.fp_source == "M4" and r0.fp_orient_kind == "directed" and diff_directed(r0.fp_orient_deg, 0.0) < 1e-6
    assert np.linalg.norm([r0.fp_x - TRUE[0], r0.fp_y - TRUE[1]]) / FT < 1e-6
    assert r0.m4_label_case == "fb_swapped" and r0.m4_front_flipped and r0.m4_interior_source == "other_pair"
    assert r0.m4_wheelbase_ft == pytest.approx(24.0)
    assert loc.loc[1, "fp_source"] == "none" and loc.loc[1, "fp_fail"] == "M4:no_match"


def test_m4_locator_without_camera_is_camera_invalid(monkeypatch):
    dh = make_dh(REAR_OBLIQUE)
    rows = _rows(dh, range(2))
    kps = _kp_table(kp(dh, PHYS), (0, 1))
    vfp = VehFootPrinter(dh, image_size=(640, 480), chains={BUS: ("M4",)})  # hub landmarks, no zone -> no camera
    monkeypatch.setattr(vfp, "_reference_heading", lambda base: np.zeros(len(base)))
    loc = vfp.locate(rows, keypoints=kps)
    assert (loc["fp_fail"] == "M4:camera_invalid").all()


def test_m4_locator_hub_landmarks_with_a_camera(monkeypatch):
    dh = make_dh(REAR_OBLIQUE)
    rad = PRIORS_FT[BUS]["wheel_r"] * FT
    hubs = np.column_stack([utm([PHYS[k] for k in PERMS["as_labelled"]]), np.full(4, rad)])
    rows = _rows(dh, range(2))
    kps = _kp_table(dh.cam.project(hubs), (0, 1))
    vfp = VehFootPrinter(dh, image_size=(640, 480), camera=dh.cam, chains={BUS: ("M4",)})
    monkeypatch.setattr(vfp, "_reference_heading", lambda base: np.zeros(len(base)))
    loc = vfp.locate(rows, keypoints=kps)
    assert (loc["fp_source"] == "M4").all()
    assert (np.hypot(loc["fp_x"] - TRUE[0], loc["fp_y"] - TRUE[1]) / FT < 1e-6).all()
