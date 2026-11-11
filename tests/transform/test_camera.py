"""PinholeCamera (ported from the speed study's tests/test_camera.py; design section 8 and test 7 of 14.2)."""

import warnings
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import MultiPoint, box

import tssa.transform.camera as camera_mod
from tssa.transform import HomographyTransform
from tssa.transform.camera import FOCAL_SCALES, PinholeCamera, PinholeGround, consistency, hub_to_contact, look_at, sensitivity

FT = 0.3048
UTM0 = np.array([587240.0, 2852970.0])  # a UTM offset, so local/UTM mix-ups show
IMG = (640, 480)
NW = Path("/mnt/e/videos/miami/output/a1a-20mins/footprint/inputs")


def ground(cam):
    return PinholeGround(cam, image_size=IMG)


class RadialPinhole(PinholeGround):
    """A pinhole ground model with a radial distortion about the image centre, like DistortedHomography."""

    K1 = 4e-7  # per px^2
    center = np.array([320.0, 240.0])

    def distort(self, uv):
        uv = np.asarray(uv, dtype=float)
        d = uv - self.center
        return self.center + d * (1.0 + self.K1 * (d**2).sum(axis=-1, keepdims=True))

    def undistort(self, uv):
        uv = np.asarray(uv, dtype=float)
        dd = uv - self.center
        d = dd.copy()
        for _ in range(100):
            d = dd / (1.0 + self.K1 * (d**2).sum(axis=-1, keepdims=True))
        return self.center + d

    def world_to_image(self, EN):
        return self.distort(super().world_to_image(EN))

    def image_to_world(self, uv):
        return super().image_to_world(self.undistort(uv))


def test_look_at_centres_target_and_has_no_roll():
    cam = look_at(C=(20, -30, 7), f=500)
    assert np.allclose(cam.project([[0, 0, 0]]), [[320, 240]], atol=1e-9)
    assert np.allclose(cam.R @ cam.R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(cam.R) == pytest.approx(1.0)
    assert cam.R[0, 2] == pytest.approx(0.0, abs=1e-12)
    assert cam.project([[0, 0, 1]])[0, 1] < 240


def test_project_ray_round_trip_in_utm():
    cam = look_at(C=UTM0.tolist() + [7.0], target=(UTM0[0] + 20, UTM0[1] + 5, 0), origin=UTM0)
    X = np.array([[UTM0[0] + 15, UTM0[1] + 3, 0.0], [UTM0[0] + 25, UTM0[1] + 9, 1.5]])
    uv = cam.project(X)
    C, d = cam.ray(uv)
    assert np.allclose(C, [*UTM0, 7.0]) and np.allclose(cam.centre, C) and cam.height == pytest.approx(7.0)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    along = X - C
    assert np.allclose(np.cross(along, d), 0.0, atol=1e-9)
    assert (np.einsum("ij,ij->i", along, d) > 0).all()


def test_pinhole_ground_round_trip():
    g = ground(look_at(C=(20, -30, 7), f=500))
    EN = np.array([[0.0, 0.0], [5.0, -3.0], [-10.0, 12.0]])
    assert np.allclose(g.image_to_world(g.world_to_image(EN)), EN, atol=1e-9)


def test_fit_recovers_synthetic_camera():
    g = ground(look_at(C=(20, -30, 7), f=500))
    zone = box(-20, -20, 20, 20)
    cam = PinholeCamera.fit(g, zone)
    assert consistency(cam, g, zone)["rms_px"] < 1e-3
    assert cam.f == pytest.approx(500, rel=1e-3)


def test_free_fit_recovers_camera_centred_on_image():
    g = ground(look_at(C=(15, -35, 9), target=(3, 4, 0), f=420))
    zone = box(-20, -20, 20, 20)
    cam = PinholeCamera.fit(g, zone)
    assert cam.converged
    assert cam.f == pytest.approx(420, rel=5e-3)
    assert cam.cx == pytest.approx(320, abs=0.5) and cam.cy == pytest.approx(240, abs=0.5)
    assert consistency(cam, g, zone)["rms_px"] < 1e-3


def test_fit_is_deterministic():
    g = ground(look_at(C=(20, -30, 7), f=500, cx=334, cy=249))
    zone = box(-20, -20, 20, 20)
    a, b = PinholeCamera.fit(g, zone), PinholeCamera.fit(g, zone)
    assert (a.f, a.cx, a.cy) == (b.f, b.cx, b.cy)
    assert np.array_equal(a.R, b.R) and np.array_equal(a.C, b.C)


def test_free_fit_picks_family_member_nearest_image_centre():
    g = ground(look_at(C=(20, -30, 7), f=500, cx=334, cy=249))
    zone = box(-20, -20, 20, 20)
    cam = PinholeCamera.fit(g, zone)
    assert consistency(cam, g, zone)["rms_px"] < 1e-3
    assert np.hypot(cam.cx - 320, cam.cy - 240) <= np.hypot(14, 9) + 1e-6


def test_fit_in_utm_through_distortion():
    """Nonzero UTM zone and a distorted model: the fit must undistort, and consistency must distort (test 7)."""
    true = look_at(C=(UTM0[0] + 20, UTM0[1] - 30, 7), target=(*UTM0, 0), f=450, origin=UTM0)
    g = RadialPinhole(true, image_size=IMG)
    zone = box(UTM0[0] - 20, UTM0[1] - 20, UTM0[0] + 20, UTM0[1] + 20)
    cam = PinholeCamera.fit(g, zone)
    res = consistency(cam, g, zone)
    assert res["rms_px"] < 1e-3 and res["max_px"] < 1e-2 and res["ok"]
    assert cam.f == pytest.approx(450, rel=1e-3)
    C_utm, _ = cam.ray([[320.0, 240.0]])
    assert np.allclose(C_utm, [UTM0[0] + 20, UTM0[1] - 30, 7], atol=1e-3)
    assert np.allclose(cam.origin, UTM0)  # floor of the zone centroid


def test_fit_through_an_ordinary_homography():
    """An ordinary (undistorted) HomographyTransform fitted to a pinhole's ground plane (test 7)."""
    true = look_at(C=(UTM0[0] + 20, UTM0[1] - 30, 7), target=(*UTM0, 0), f=450, origin=UTM0)
    pg = ground(true)
    uv = np.array([[u, v] for u in (60, 200, 320, 450, 600) for v in (260, 330, 420, 470)], float)
    h = HomographyTransform(uv, pg.image_to_world(uv), method=0)
    zone = box(UTM0[0] - 15, UTM0[1] - 15, UTM0[0] + 15, UTM0[1] + 15)
    with pytest.raises(ValueError, match="image size"):
        PinholeCamera.fit(h, zone)
    from tssa.transform import GroundModel

    g = GroundModel.wrap(h, image_size=IMG)
    cam = PinholeCamera.fit(g, zone)
    res = consistency(cam, g, zone)
    # a single plane fixes f only weakly with a free principal point: the study's fit_camera gives the same 448.96
    assert res["ok"] and res["rms_px"] < 1e-3 and cam.f == pytest.approx(450, rel=5e-3)


def test_consistency_flags_a_wrong_camera():
    g = ground(look_at(C=(20, -30, 7), f=500))
    zone = box(-20, -20, 20, 20)
    res = consistency(look_at(C=(20, -30, 8), f=500), g, zone)
    assert res["rms_px"] > 1.0 and not res["ok"]
    assert set(res) == {"rms_px", "max_px", "ok"}


def test_fit_with_fixed_focal_keeps_f():
    cam = PinholeCamera.fit(ground(look_at(C=(20, -30, 7), f=500)), box(-20, -20, 20, 20), f_fixed=560.0)
    assert cam.f == 560.0


def test_hub_drop_is_vertical_not_along_ray():
    true = look_at(C=(0, 0, 7), target=(20, 0, 0))
    g = ground(true)
    uv = true.project([[20, 0, 1.1 * FT]])
    contact = hub_to_contact(true, uv, 1.1 * FT)
    assert contact.shape == (1, 2)
    assert np.linalg.norm(contact - [20, 0]) / FT < 0.05
    assert np.linalg.norm(g.image_to_world(uv) - [20, 0]) / FT == pytest.approx(3.30, abs=0.01)


def test_hub_to_contact_per_row_radius_in_utm():
    cam = look_at(C=(UTM0[0], UTM0[1], 7), target=(UTM0[0] + 20, UTM0[1], 0), origin=UTM0)
    hubs = np.array([[UTM0[0] + 20, UTM0[1] + 2, 1.1 * FT], [UTM0[0] + 15, UTM0[1] - 3, 1.6 * FT]])
    contact = hub_to_contact(cam, cam.project(hubs), np.array([1.1 * FT, 1.6 * FT]))
    assert np.allclose(contact, hubs[:, :2], atol=1e-6)


def test_sensitivity_returns_three_refits():
    g = ground(look_at(C=(20, -30, 7), f=500))
    zone = box(-20, -20, 20, 20)
    base = PinholeCamera.fit(g, zone)
    out = sensitivity(g, zone, base)
    assert [o["scale"] for o in out] == list(FOCAL_SCALES)
    for o in out:
        assert isinstance(o["camera"], PinholeCamera)
        assert o["camera"].f == pytest.approx(base.f * o["scale"])
        assert o["rms_px"] == pytest.approx(consistency(o["camera"], g, zone)["rms_px"])
    by = {o["scale"]: o for o in out}
    assert by[1.0]["rms_px"] < 1e-3
    assert abs(by[1.13]["camera"].cy - base.cy) > 1.0 and abs(by[0.87]["camera"].cy - base.cy) > 1.0


def test_sensitivity_with_fixed_principal_point():
    g = ground(look_at(C=(20, -30, 7), f=500))
    zone = box(-20, -20, 20, 20)
    out = sensitivity(g, zone, PinholeCamera.fit(g, zone, free_principal=False), free_principal=False)
    by = {o["scale"]: o for o in out}
    assert all((o["camera"].cx, o["camera"].cy) == (320.0, 240.0) for o in out)
    assert by[1.0]["rms_px"] < 1e-3
    assert by[0.87]["rms_px"] > 1e-3 and by[1.13]["rms_px"] > 1e-3


def test_fit_recovers_off_centre_principal_point():
    g = ground(look_at(C=(20, -30, 7), f=500, cx=334, cy=249))
    zone = box(-20, -20, 20, 20)
    cam = PinholeCamera.fit(g, zone)
    assert cam.converged and consistency(cam, g, zone)["rms_px"] < 1e-3
    at_f = PinholeCamera.fit(g, zone, f_fixed=500.0)
    assert consistency(at_f, g, zone)["rms_px"] < 1e-3
    assert at_f.cx == pytest.approx(334, abs=0.5) and at_f.cy == pytest.approx(249, abs=0.5)
    assert np.allclose(at_f.ray([[0.0, 0.0]])[0], [20, -30, 7], atol=1e-3)


def test_fixed_principal_point_uses_the_image_centre():
    g = ground(look_at(C=(20, -30, 7), f=500, cx=334, cy=249))
    zone = box(-20, -20, 20, 20)
    cam = PinholeCamera.fit(g, zone, free_principal=False)
    assert (cam.cx, cam.cy) == (320.0, 240.0)
    assert consistency(cam, g, zone)["rms_px"] > 1e-3


def test_non_convergence_is_flagged(monkeypatch):
    real_ls = camera_mod.least_squares
    monkeypatch.setattr(camera_mod, "least_squares", lambda *a, **k: real_ls(*a, **{**k, "max_nfev": 2}))
    g = ground(look_at(C=(20, -30, 7), f=500))
    zone = box(-20, -20, 20, 20)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        cam = PinholeCamera.fit(g, zone)
    assert cam.converged is False and consistency(cam, g, zone)["ok"] is False


def test_affine_stand_in_fit_is_finite_and_flagged():
    """The orthographic-limit affine map has no pinhole solution: the fit must return finite residuals and, when the
    least squares stops early, flag it. (The study's run happened to converge here, to cx = -22831 px; the path
    depends on round-off at the 1e-16 level, so convergence itself is not a property of this degenerate case.)"""
    from synth_affine import AffineGround

    g = AffineGround()
    zone = MultiPoint(g.image_to_world(np.array([[0.0, 0], [640, 0], [640, 480], [0, 480]]))).convex_hull
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cam = PinholeCamera.fit(g, zone)
    res = consistency(cam, g, zone)
    assert np.isfinite(res["rms_px"]) and (cam.converged or not res["ok"])


def test_too_few_finite_correspondences_raise():
    class Sparse(PinholeGround):
        def world_to_image(self, EN):
            uv = super().world_to_image(EN)
            uv[3:] = np.nan
            return uv

    with pytest.raises(ValueError, match="correspondences"):
        PinholeCamera.fit(Sparse(look_at(C=(20, -30, 7)), image_size=IMG), box(-20, -20, 20, 20))


@pytest.mark.skipif(not (NW / "model_a1a_nw.json").exists(), reason="NW study inputs not available")
def test_real_model_reproduces_the_study_camera():
    """The NW model and GCP zone: the run report's camera (f 319.55 px, (cx, cy) (334.10, 240.13), 6.73 m)."""
    import pandas as pd

    from tssa.footprint import zone_from_points
    from tssa.transform import DistortedHomography

    dh = DistortedHomography.load_json(NW / "model_a1a_nw.json")
    g = pd.read_csv(NW / "a1a_nw_gcps_transfer.csv")
    zone = zone_from_points(g[["utm_e_m", "utm_n_m"]].to_numpy(), 15)
    cam = PinholeCamera.fit(dh, zone)
    assert consistency(cam, dh, zone)["ok"]
    assert cam.f == pytest.approx(319.5459175640886, abs=0.01)
    assert cam.cx == pytest.approx(334.10085400161427, abs=0.01) and cam.cy == pytest.approx(240.1255795054726, abs=0.01)
    assert cam.height == pytest.approx(6.733058272508432, abs=1e-3)
