"""VehFootPrinter: chains, covariance, size, orientation, units, compare and admission (design section 14.2, tests
9-18; M7-shaped paths use stub segment locators)."""

import numpy as np
import pandas as pd
import pytest

from tssa.footprint import VehFootPrinter
from tssa.footprint.config import FT, MPS_TO_MPH
from tssa.footprint.geometry import jacobian
from tssa.footprint.orientation import smooth_axis
from tssa.speed import SpeedCalculator

from synth import ORIGIN, ground, kinematic_stub, render, straight, stub_locator, tables, turn

G = ground()


def scene(track=1, cls=2, centres=None, headings=None, frames=None, L=15.0, W=6.0, H=5.0, noise=0.0, seed=0):
    if centres is None:
        centres, headings = straight()
    frames = np.arange(len(centres)) if frames is None else np.asarray(frames)
    boxes, faces = render(G, centres, headings, L, W, H, noise, seed)
    return tables(frames, track, cls, boxes, faces)


def stationary(n=30, at=(0.0, 0.0), psi=30.0, **kw):
    c = np.tile(ORIGIN + np.asarray(at), (n, 1))
    return scene(centres=c, headings=np.full(n, psi), **kw)


def path_pose(centres, headings, offset_frames=None, offset_m=0.0):
    """pose_fn for kinematic_stub: the true pose of each row's frame (+ an east offset on `offset_frames`)."""

    def fn(rows):
        f = rows["frame"].to_numpy()
        xy = centres[f].copy()
        if offset_frames is not None:
            xy[np.isin(f, offset_frames), 0] += offset_m
        return xy[:, 0], xy[:, 1], headings[f]

    return fn


# --- 9. chains ---------------------------------------------------------------------------------------------------


def test_truck_chain_records_each_failure_in_order():
    tr, cub = stationary(cls=7, L=25, W=8, H=10)
    faces = cub.copy()
    faces[[f"gu{i}" for i in range(4)]] = faces[["gu0"]].to_numpy() + np.array([0, 2, 2, 0])  # collapsed ~2 px face
    faces[[f"gv{i}" for i in range(4)]] = faces[["gv0"]].to_numpy() + np.array([0, 0, 1, 1])
    with stub_locator("T7", kinematic_stub(None, passed=False), needs=()):
        loc = VehFootPrinter(G, chains={7: ("T7", "M6", "M3", "M2")}).locate(tr, cuboids=faces)
    box = loc[loc["fp_row_kind"] == "box"]
    assert (box["fp_source"] == "M2").all()
    assert (box["fp_fail"] == "T7:param_bound;M6:implausible_size;M3:no_heading").all()


def test_car_without_cuboids_falls_to_m2():
    tr, _ = scene()
    loc = VehFootPrinter(G).locate(tr)
    box = loc[loc["fp_row_kind"] == "box"]
    assert (box["fp_source"] == "M2").all() and (box["fp_fail"] == "M6:not_supplied").all()


# --- 10. covariance per support type -----------------------------------------------------------------------------


def test_box_covariance_and_upstream_fill_inflation():
    tr, _ = scene()
    tr["interp"] = np.where(tr["frame"] == 12, 1, 0)
    vfp = VehFootPrinter(G, chains={2: ("M2",)})
    loc = vfp.locate(tr).set_index("frame")
    for f, scale in ((11, 1.0), (12, 4.0)):
        r = tr.set_index("frame").loc[f]
        J = jacobian(vfp.ground, np.array([[r.x + r.w / 2, r.y + r.h]]))[0]
        R = (0.03 * r.h * scale) ** 2 * J @ J.T
        got = loc.loc[f, ["fp_cov_xx", "fp_cov_xy", "fp_cov_yy"]].to_numpy(dtype=float)
        np.testing.assert_allclose(got, [R[0, 0], R[0, 1], R[1, 1]], rtol=1e-12, atol=0)


def test_admitted_row_keeps_its_own_covariance():
    c, h = straight()
    tr, _ = scene(cls=5, centres=c, headings=h, L=40, W=8.5, H=11)
    with stub_locator("T7", kinematic_stub(path_pose(c, h), cov=0.04, native_frames=range(0, 40, 5)), needs=()):
        loc = VehFootPrinter(G, chains={5: ("T7", "M2")}).locate(tr)
    kin = loc[loc["fp_support"] == "kinematic"]
    assert len(kin) and (kin["fp_source"] == "T7").all()
    np.testing.assert_array_equal(kin[["fp_cov_xx", "fp_cov_xy", "fp_cov_yy"]].to_numpy(), np.tile([0.04, 0, 0.04], (len(kin), 1)))


# --- 11. size ----------------------------------------------------------------------------------------------------


def test_segment_size_applies_to_every_row_including_fallback_rows():
    c, h = straight()
    tr, _ = scene(cls=5, centres=c, headings=h, L=40, W=8.5, H=11)
    with stub_locator("T7", kinematic_stub(path_pose(c, h), native_frames=range(0, 40, 5), seg_size=(30.0, 8.2, "m7_measured")), needs=()):
        out = VehFootPrinter(G, chains={5: ("T7", "M2")}, admit_kinematic=False).estimate(tr)
    poly = out[out["dims_source"] != "none"]
    assert (poly["fp_source"] == "M2").any()  # kinematic rows fell back
    assert (poly["length_ft"] == 30.0).all() and (poly["width_ft"] == 8.2).all()
    assert (poly["dims_source"] == "m7_measured").all()


@pytest.mark.parametrize("n_m6, expect", [(5, "m6_measured"), (4, "nominal")])
def test_m6_measured_size_needs_enough_rows(n_m6, expect):
    tr, cub = scene(L=14.0, W=5.8)
    cub = cub[cub["frame"] < n_m6]
    out = VehFootPrinter(G).estimate(tr, cuboids=cub)
    poly = out[out["dims_source"] != "none"]
    assert (poly["dims_source"] == expect).all()
    if expect == "m6_measured":
        assert poly["length_ft"].iloc[0] == pytest.approx(14.0, abs=0.05)


def test_bus_m6_size_is_nominal():
    c, h = straight()
    tr, cub = scene(cls=5, centres=c, headings=h, L=38, W=8.4, H=11)
    out = VehFootPrinter(G, chains={5: ("M6", "M2")}).estimate(tr, cuboids=cub)
    poly = out[out["dims_source"] != "none"]
    assert (poly["dims_source"] == "nominal").all() and (poly["length_ft"] == 40.0).all()


# --- 12. orientation ---------------------------------------------------------------------------------------------


def test_no_native_axes_uses_motion_heading():
    tr, _ = scene()
    out = VehFootPrinter(G, chains={2: ("M2",)}).estimate(tr)
    poly = out[np.isfinite(out["orient_deg"])]
    assert (poly["orient_source"] == "motion").all()
    assert np.abs(((poly["orient_deg"] + 180) % 360) - 180).max() < 3.0  # north


def test_one_native_axis_and_the_reach_rule():
    c, h = straight()
    tr, cub = scene(centres=c, headings=np.full(len(c), 0.0))
    cub = cub[cub["frame"] == 5]
    out = VehFootPrinter(G).estimate(tr, cuboids=cub).set_index("frame")
    assert out.loc[5, "orient_source"] == "native" and out.loc[15, "orient_source"] == "native"  # within 10 frames
    assert out.loc[16, "orient_source"] == "motion"


def test_axis_never_smooths_through_ninety():
    deg = np.where(np.arange(40) % 2 == 0, 179.0, 1.0)
    out = smooth_axis(deg, np.ones(40, bool), 1.0, 0.1, 0.1)
    assert (np.minimum(out, 180 - out) < 2.0).all()
    assert np.isnan(smooth_axis(deg, np.zeros(40, bool), 1.0, 0.1, 0.1)).all()
    assert (smooth_axis(deg, np.arange(40) == 3, 1.0, 0.1, 0.1) == 1.0).all()


def test_axis_is_directed_toward_motion():
    c, h = straight()
    c = c[::-1]  # driving south
    tr, cub = scene(centres=c, headings=np.full(len(c), 180.0))
    out = VehFootPrinter(G).estimate(tr, cuboids=cub)
    poly = out[out["orient_source"] == "native"]
    assert len(poly) and (np.abs(poly["orient_deg"] - 180.0) < 2.0).all()


# --- 13. never-moving vehicles -------------------------------------------------------------------------------------


def test_stationary_car_keeps_its_axis():
    tr, cub = stationary(psi=30.0)
    out = VehFootPrinter(G).estimate(tr, cuboids=cub)
    poly = out[out["fp_row_kind"] == "box"]
    assert (poly["orient_source"] == "native").all()
    assert (np.abs((poly["orient_deg"] % 180) - 30.0) < 1.0).all()
    assert (poly["front_source"] == "none").all() and (~poly["front_known"]).all()
    assert poly[["c1x", "c4y"]].notna().all().all()


def test_stationary_segment_takes_the_direction_of_the_tracks_moving_segment():
    t1, c1 = stationary(n=20, psi=30.0)
    cs, hs = straight(n=30)
    t2, c2 = scene(centres=cs, headings=np.full(30, 30.0), frames=np.arange(60, 90))  # long gap: a second segment
    tr, cub = pd.concat([t1, t2], ignore_index=True), pd.concat([c1, c2], ignore_index=True)
    out = VehFootPrinter(G).estimate(tr, cuboids=cub)
    still = out[(out["frame"] < 20) & (out["fp_row_kind"] == "box")]
    assert (still["front_source"] == "track").all() and still["front_known"].all()
    assert (np.abs(still["orient_deg"] - 30.0) < 1.0).all()  # 30 rather than 210: toward the moving segment's north


def test_stationary_bus_takes_its_direction_from_a_directed_native():
    n = 30
    c = np.tile(ORIGIN + [0.0, 0.0], (n, 1))
    tr, _ = scene(cls=5, centres=c, headings=np.full(n, 200.0), L=40, W=8.5, H=11)
    with stub_locator("T7", kinematic_stub(path_pose(c, np.full(n, 200.0)), native_frames=range(n)), needs=()):
        out = VehFootPrinter(G, chains={5: ("T7",)}).estimate(tr)
    poly = out[out["fp_source"] == "T7"]
    assert (poly["front_source"] == "native").all()
    assert (np.abs(poly["orient_deg"] - 200.0) < 1.0).all()


def test_car_that_waits_then_drives_is_back_filled():
    wait = np.tile(ORIGIN + [-3.0, -12.0], (15, 1))
    cs, _ = straight(n=25, y0=-12.0)
    c = np.vstack([wait, cs])
    tr, cub = scene(centres=c, headings=np.zeros(len(c)))
    out = VehFootPrinter(G).estimate(tr, cuboids=cub)
    waiting = out[out["frame"] < 10]
    assert (waiting["front_source"] == "motion").all()
    assert (np.abs(((waiting["orient_deg"] + 180) % 360) - 180) < 2.0).all()


# --- 14. speed units ---------------------------------------------------------------------------------------------


def test_metric_and_imperial_give_the_same_orientation():
    c, h = turn()
    tr, cub = scene(centres=c, headings=h)
    imp = VehFootPrinter(G).estimate(tr, cuboids=cub)
    met = VehFootPrinter(G, speed=SpeedCalculator(xy_cols=("fp_x_smooth", "fp_y_smooth"), units="metric")).estimate(tr, cuboids=cub)
    assert imp["heading_reliable"].tolist() == met["heading_reliable"].tolist()
    np.testing.assert_allclose(imp["orient_deg"], met["orient_deg"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(imp["speed"] / MPS_TO_MPH, met["speed"], rtol=0, atol=2e-3)


def test_polygons_need_a_speed_unit():
    tr, cub = scene()
    vfp = VehFootPrinter(G)
    loc = vfp.locate(tr, cuboids=cub)
    sp = vfp.speed.calculate(vfp.smoother.smooth(loc, segmented=True))
    sp.attrs = {}
    with pytest.raises(ValueError, match="speed unit unknown"):
        vfp.polygons(sp)
    assert len(vfp.polygons(sp, speed_unit="mph")) == len(sp)


# --- 15. compare -------------------------------------------------------------------------------------------------


@pytest.fixture
def mixed():
    c, h = straight()
    bus, _ = scene(track=1, cls=5, centres=c, headings=h, L=40, W=8.5, H=11)
    cs, hs = straight(x=4.0)
    car, _ = scene(track=2, cls=2, centres=cs, headings=hs)
    return pd.concat([bus, car], ignore_index=True), c, h


@pytest.mark.parametrize("admit", [None, False])
def test_compare_native_only(mixed, admit):
    tr, c, h = mixed
    with stub_locator("T7", kinematic_stub(path_pose(c, h), native_frames=range(0, 40, 5)), needs=()):
        out = VehFootPrinter(G).compare(tr, methods=("T7", "M2"), fallback=None, admit_kinematic=admit)
    t7 = out[out["method"] == "T7"]
    kin = t7[(t7["track"] == 1) & (t7["m7_support"] == "kinematic")]
    assert len(kin) and (kin["fp_source"] == "none").all() and (kin["fp_support"] == "").all()
    assert (kin["fp_fail"] == "T7:no_observation").all()
    car = t7[t7["track"] == 2]
    assert (car["fp_source"] == "none").all() and car["fp_fail"].str.endswith("T7:class_ineligible").all()


def test_compare_native_only_rejects_admission(mixed):
    with pytest.raises(ValueError, match="native-only"):
        VehFootPrinter(G).compare(mixed[0], methods=("M2",), fallback=None, admit_kinematic=True)


@pytest.mark.parametrize("admit, src", [(None, "T7"), (True, "T7"), (False, "M2")])
def test_compare_end_to_end(mixed, admit, src):
    tr, c, h = mixed
    with stub_locator("T7", kinematic_stub(path_pose(c, h), native_frames=range(0, 40, 5)), needs=()):
        out = VehFootPrinter(G).compare(tr, methods=("T7", "M2"), fallback="M2", admit_kinematic=admit)
    t7 = out[out["method"] == "T7"]
    kin = t7[(t7["track"] == 1) & (t7["m7_support"] == "kinematic") & (t7["fp_row_kind"] == "box")]
    assert len(kin) and (kin["fp_source"] == src).all()
    if src == "T7":
        assert (kin["fp_support"] == "kinematic").all()
    car = t7[t7["track"] == 2]
    assert (car["fp_source"] == "none").all() and car["fp_fail"].str.endswith("T7:class_ineligible").all()  # no fallback


def test_compare_returns_every_row_once_per_method(mixed):
    tr, c, h = mixed
    with stub_locator("T7", kinematic_stub(path_pose(c, h)), needs=()):
        out = VehFootPrinter(G).compare(tr, methods=("T7", "M2", "M6"), only_tracks=[1])
    for m, part in out.groupby("method"):
        assert len(part) == len(tr) and not part.duplicated(["track", "frame"]).any()
    assert (out.loc[out["track"] == 2, "fp_fail"] == "excluded").all()


# --- 16. end to end ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [straight, turn])
def test_end_to_end_recovers_the_vehicle(path):
    c, h = path(n=50)
    keep = ~np.isin(np.arange(50), np.arange(20, 35))  # a 15-frame hole
    tr, cub = scene(centres=c[keep], headings=h[keep], frames=np.arange(50)[keep], noise=0.5, seed=1)
    out = VehFootPrinter(G).estimate(tr, cuboids=cub).set_index("frame")
    f = np.arange(3, 47)
    err_ft = np.linalg.norm(out.loc[f, ["fp_x_smooth", "fp_y_smooth"]].to_numpy() - c[f], axis=1) / FT
    meas = keep[f]
    assert err_ft[meas].max() < 0.3
    dh = np.abs(((out.loc[f, "orient_deg"].to_numpy() - h[f]) + 180) % 360 - 180)
    assert np.median(dh) < 2.0
    speed_true = (4.0 if path is straight else 5.0) * MPS_TO_MPH
    assert np.median(np.abs(out.loc[f, "speed"].to_numpy() - speed_true)) < 0.2
    assert out.loc[25, "interp"] == 2 and np.isfinite(out.loc[25, "c1x"])  # the hole is predicted, with a polygon


# --- 17. every row once, including invalid projections ------------------------------------------------------------


def test_every_input_row_returned_once():
    tr, cub = scene()
    sky = tr["frame"].isin([0, 1, 18, 39])
    tr.loc[sky, "y"], tr.loc[sky, "h"] = 40.0, 30.0  # bottom edge above the horizon: no ground point
    vfp = VehFootPrinter(G)
    for out in (vfp.locate(tr, cuboids=cub), vfp.estimate(tr, cuboids=cub)):
        assert not out.duplicated(["track", "frame"]).any()
        assert set(map(tuple, tr[["track", "frame"]].to_numpy())) <= set(map(tuple, out[["track", "frame"]].to_numpy()))
        rej = out[out["fp_row_kind"] == "box_rejected"]
        assert sorted(rej["frame"]) == [0, 1, 18, 39]
        assert rej["fp_fail"].str.startswith("projection").all() and (rej["fp_source"] == "none").all()
        assert (out.set_index("frame").loc[[0, 1, 39], "seg"] == -1).all()
        assert out.set_index("frame").loc[18, "seg"] >= 0  # bridged, still in its segment


# --- 18. admission through the pipeline --------------------------------------------------------------------------


def test_admission_through_estimate():
    c, h = straight()
    keep = np.arange(40) != 20  # frame 20 has no row: inserted
    tr, _ = scene(cls=5, centres=c[keep], headings=h[keep], frames=np.arange(40)[keep], L=40, W=8.5, H=11)
    sky = tr["frame"].isin([0, 25, 27])
    tr.loc[sky, "y"], tr.loc[sky, "h"] = 40.0, 30.0
    native = [f for f in range(0, 40, 3) if f != 25] + [27]  # 27: native support on a rejected row
    stub = kinematic_stub(path_pose(c, h, offset_frames=[20], offset_m=1.0), cov=0.04, native_frames=native)
    with stub_locator("T7", stub, needs=()):
        on = VehFootPrinter(G, chains={5: ("T7", "M2")}).estimate(tr).set_index("frame")
        off = VehFootPrinter(G, chains={5: ("T7", "M2")}, admit_kinematic=False).estimate(tr).set_index("frame")
    r20 = on.loc[20]
    assert r20["fp_row_kind"] == "inserted" and r20["fp_source"] == "T7" and r20["fp_support"] == "kinematic"
    assert r20["fp_cov_xx"] == 0.04 and r20["fp_cov_xy"] == 0.0 and r20["fp_cov_yy"] == 0.04
    assert abs(on.loc[20, "fp_x_smooth"] - off.loc[20, "fp_x_smooth"]) > 0.01
    r25 = on.loc[25]
    assert r25["fp_row_kind"] == "box_rejected" and r25["fp_source"] == "T7" and r25["fp_fail"].startswith("projection")
    r27 = on.loc[27]
    assert r27["fp_source"] == "none" and r27["fp_fail"] == "projection;T7:projection"
    r0 = on.loc[0]
    assert r0["seg"] == -1 and r0["fp_source"] == "none" and np.isnan(r0["fp_x_smooth"])
