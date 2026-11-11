"""The M7 locator through VehFootPrinter (design 7.6, 7.7, 10.3; tests 4, 10, 13, 15, 18 and 19 of 14.2 against the real
locator). Observations come from m7_synth (frame_obs_from_mask is replaced, as the study's runner tests did), so the
solver, admission, sizes and the solve cache run for real."""

import numpy as np
import pandas as pd
import pytest

import m7_synth as sy
from tssa.footprint import M7Config, VehFootPrinter
from tssa.footprint.config import FT
from tssa.footprint.locators.m7 import solver as m7
from tssa.footprint.masks import MaskTable
from tssa.transform.camera import PinholeGround

KIN = range(30, 40)  # frames with no observation of their own


@pytest.fixture(scope="module")
def scene():
    cam = sy.camera()
    truth = sy.simulate(K=70)
    cfg = m7.Config("bus")
    obs = sy.observations(truth, cam, cfg, seed=3)
    for k in KIN:
        obs[k] = m7.FrameObs(int(truth["frames"][k]))
    track = m7.Track(truth["frames"], obs, cam, np.zeros(2), np.zeros((70, 2)), np.zeros(70), 5, 0.1)
    g = m7.Problem(track, cfg).geometry(sy._pose_x(truth))
    box = g["box_px"]  # (K, 8, 2) undistorted px; the pinhole ground has no distortion
    lo, hi = box.min(axis=1), box.max(axis=1)
    tr = pd.DataFrame({"frame": truth["frames"], "track": 1, "x": lo[:, 0], "y": lo[:, 1], "w": hi[:, 0] - lo[:, 0],
                       "h": hi[:, 1] - lo[:, 1], "cls": 5})  # fmt: skip
    masks = MaskTable({(1, int(f)): [np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]])] for f in truth["frames"]})
    return dict(cam=cam, truth=truth, obs={int(o.frame): o for o in obs}, tracks=tr, masks=masks)


@pytest.fixture
def fake_obs(monkeypatch, scene):
    monkeypatch.setattr(m7, "frame_obs_from_mask", lambda frame, *a, **k: scene["obs"][int(frame)])


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("m7cache")


def printer(scene, cache, **kw):
    g = PinholeGround(scene["cam"], image_size=(640, 480))
    vfp = VehFootPrinter(g, camera=scene["cam"], cache_dir=cache, chains={5: ("M7", "M2")}, **kw)
    vfp.local_origin = np.zeros(2)  # the synthetic truth's frame
    return vfp


def test_m7_rows_support_and_size(scene, cache, fake_obs):
    vfp = printer(scene, cache)
    out = vfp.estimate(scene["tracks"], masks=scene["masks"]).set_index("frame")
    sm = vfp.info["m7_solves"][0]
    assert sm["fail_reason"] == "" and (out["m7_solve_ok"]).all()
    assert abs(sm["length_ft"] - 40.0) < 1.0
    nat = out[out["fp_support"] == "native"]
    kin = out[out["fp_support"] == "kinematic"]
    assert set(kin.index) == set(KIN) and (kin["fp_source"] == "M7").all() and (nat["fp_source"] == "M7").all()
    # admitted covariance (design 7.7): kin_meas_scale^2 * m7_cov_centre + floor^2 I, exactly
    r = kin.iloc[0]
    want = 2.0**2 * r.m7_cov_centre_ee + (0.5 * FT) ** 2
    assert r.fp_cov_xx == pytest.approx(want, rel=1e-12) and r.fp_cov_xy == pytest.approx(4 * r.m7_cov_centre_en, rel=1e-12)
    # size: the fitted L and W on every row of the passed solve
    assert np.allclose(out["length_ft"], sm["length_ft"], rtol=0, atol=1e-9) and out["dims_source"].str.startswith("m7_").all()
    # centre accuracy on native rows, and the directed heading
    c = sy.centre_of(scene["truth"])
    err = np.hypot(nat["fp_x_smooth"] - c[nat.index, 0], nat["fp_y_smooth"] - c[nat.index, 1]) / FT
    assert np.median(err) < 0.5
    assert (out["fp_low_confidence"] == False).all()  # noqa: E712  (kinematic run of 1 s < 2 s)


def test_admission_off_falls_back(scene, cache, fake_obs):
    out = printer(scene, cache, admit_kinematic=False).locate(scene["tracks"], masks=scene["masks"]).set_index("frame")
    kin = out.loc[list(KIN)]
    assert (kin["fp_source"] == "M2").all() and kin["fp_fail"].str.startswith("M7:no_observation").all()
    assert (kin["m7_support"] == "kinematic").all()


def test_native_only_compare_and_cached_solve(scene, cache, fake_obs):
    vfp = printer(scene, cache)
    out = vfp.compare(scene["tracks"], methods=("M7",), fallback=None, masks=scene["masks"])
    kin = out[out["frame"].isin(KIN)]
    assert (kin["fp_source"] == "none").all() and (kin["fp_fail"] == "M7:no_observation").all()
    assert vfp.info["m7_solves"][0]["cached"]  # solved by an earlier test: read from the cache


def test_cache_key_follows_solver_settings_only(scene, cache, fake_obs):
    a = printer(scene, cache, kin_meas_scale=3.0)  # a product setting: same solve
    a.locate(scene["tracks"], masks=scene["masks"])
    assert a.info["m7_solves"][0]["cached"]
    from tssa.footprint.locators.m7 import job_key, _ns
    from tssa.footprint.locators import LocateContext

    ctx = lambda m: LocateContext(PinholeGround(scene["cam"], image_size=(640, 480)),  # noqa: E731
                                  printer(scene, cache, **m).config, {}, None, pd.DataFrame())  # fmt: skip
    tr = m7.Track(np.arange(3), [m7.FrameObs(i) for i in range(3)], scene["cam"], np.zeros(2), np.zeros((3, 2)),
                  np.zeros(3), 5, 0.1)  # fmt: skip
    k0 = job_key(tr, _ns(ctx({})), True, (1, 0, 5))
    assert job_key(tr, _ns(ctx({"kin_meas_scale": 3.0})), True, (1, 0, 5)) == k0
    assert job_key(tr, _ns(ctx({"m7": M7Config(M7_SIGMA_WHEEL_PX=2.0)})), True, (1, 0, 5)) != k0


@pytest.mark.parametrize("w, held", [(0.29, False), (0.3, False), (0.31, True), (np.inf, True)])
def test_width_hold_threshold(w, held):
    # design 14.2 test 19: strictly above the threshold holds; no threshold never holds
    assert m7.width_held(w, 0.3) is held
    assert m7.width_held(w, None) is False


def test_no_camera_and_short_segments(scene, cache, fake_obs):
    g = PinholeGround(scene["cam"], image_size=(640, 480))
    loc = VehFootPrinter(g, chains={5: ("M7", "M2")}).locate(scene["tracks"], masks=scene["masks"])  # no zone
    assert loc["fp_fail"].str.startswith("M7:camera_invalid").all() and (loc["fp_source"] == "M2").all()
    short = scene["tracks"][scene["tracks"]["frame"] < 8]
    loc = printer(scene, cache).locate(short, masks=scene["masks"])
    assert loc["fp_fail"].str.startswith("M7:too_few_frames").all()
