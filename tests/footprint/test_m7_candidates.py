"""Ported from the speed study's tests/test_m7_candidates.py (imports only changed; OpenCV is optional here)."""
"""M7 Phase 0: pose-free wheel-contact candidates (design 2026-11-08-m7, section 5.1; tests 7-9)."""

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from tssa.footprint.locators.m7 import candidates as mc
from tssa.footprint.locators.m7.settings import S
from tssa.transform.camera import PinholeGround, look_at

R_TYRE = 1.6 * S.FT
Y_NEAR = -1.3  # near body side (camera is at negative y)
Y_TYRE = -1.2  # outer tyre face, 0.1 m inside the body side
WHEELS_X = (3.5, -3.8)


def _cam(C=(0.0, -25.0, 7.0)):
    return look_at(C, target=(0.0, 0.0, 0.0), f=320.0)


def _poly(cam, pts3):
    uv = cam.project(np.asarray(pts3, dtype=float))
    return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2).astype(float)


def _body(cam, clear=0.3, length=12.2, width=2.6, height=3.35):
    xs, ys, zs = (-length / 2, length / 2), (-width / 2, width / 2), (clear, height)
    return _poly(cam, [(x, y, z) for x in xs for y in ys for z in zs])


def _tyre(cam, x0, r=R_TYRE, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return _poly(cam, np.column_stack([x0 + r * np.cos(a), np.full(n, Y_TYRE), r + r * np.sin(a)]))


def _contours(polys):
    m = mc.rasterise(polys).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1, 2).astype(float) for c in cs], m


def _box(mask):
    ys, xs = np.nonzero(mask)
    return (float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))


def _contacts(cam, xs=WHEELS_X):
    return cam.project(np.array([(x, Y_TYRE, 0.0) for x in xs]))


@pytest.mark.parametrize("C", [(0.0, -25.0, 7.0), (-12.0, -22.0, 7.0), (10.0, -18.0, 6.5)])
def test_candidates_at_tyre_contacts(C):  # design test 7
    cam = _cam(C)
    dh = PinholeGround(cam)
    contours, mask = _contours([_body(cam), *[_tyre(cam, x) for x in WHEELS_X]])
    cand, summ, _ = mc.extract(contours, _box(mask), [], dh, cam, 5)
    acc = cand[cand["accepted"]]
    assert len(acc) == 2, cand
    got = acc[["u", "v"]].to_numpy()
    lowest = np.array([(p := _tyre(cam, x))[p[:, 1].argmax()] for x in WHEELS_X])  # what the extractor measures
    d_low = np.linalg.norm(got[:, None] - lowest[None], axis=2).min(axis=0)
    d_contact = np.linalg.norm(got[:, None] - _contacts(cam)[None], axis=2).min(axis=0)
    assert (d_low <= 1.5).all(), (d_low, cand)
    # the lowest image point of a tyre is not exactly its contact (design 5.1 accepts this small bias)
    assert (d_contact <= 2.0).all(), (d_contact, cand)
    assert (acc["kappa"] == mc.rho(acc["gate_px"] / S.M7_SIGMA_WHEEL_PX)).all()


def test_no_protrusion_no_candidates():  # design test 8
    cam = _cam()
    dh = PinholeGround(cam)
    contours, mask = _contours([_body(cam, clear=0.0)])
    cand, summ, _ = mc.extract(contours, _box(mask), [], dh, cam, 5)
    assert summ["n_cand"] == 0, cand


def test_shadow_lobe_rejected_tyres_kept():  # design test 9
    cam = _cam()
    dh = PinholeGround(cam)
    shadow = _poly(cam, [(-2.5, Y_NEAR, 0.0), (2.5, Y_NEAR, 0.0), (2.5, -1.9, 0.0), (-2.5, -1.9, 0.0)])
    contours, mask = _contours([_body(cam), *[_tyre(cam, x) for x in WHEELS_X], shadow])
    cand, summ, _ = mc.extract(contours, _box(mask), [], dh, cam, 5)
    acc = cand[cand["accepted"]]
    assert len(acc) == 2, cand
    assert (cand.loc[~cand["accepted"], "reject"] == "wide").sum() >= 1, cand


def test_mask_notch_rejected_as_deep():
    cam = _cam()
    dh = PinholeGround(cam)
    # long thin spike below the body, like the seed-0 crop 21 notch (about 2.5 tyre diameters deep)
    notch = _poly(cam, [(0.0, Y_NEAR, 0.3), (0.6, Y_NEAR, 0.3), (0.6, -6.0, 0.0), (0.0, -6.0, 0.0)])
    contours, mask = _contours([_body(cam), *[_tyre(cam, x) for x in WHEELS_X], notch])
    cand, _, _ = mc.extract(contours, _box(mask), [], dh, cam, 5)
    assert (cand["accepted"]).sum() == 2, cand
    assert "deep" in set(cand["reject"]), cand


def test_duplicate_runs_on_one_tyre_merged():
    cand = mc._dedupe(pd.DataFrame({"cand": [0, 1, 2], "u_und": [100.0, 104.0, 160.0], "depth_px": [5.0, 3.0, 4.0],
                                    "tyre_px": [12.0, 12.0, 12.0], "accepted": [True, True, True],
                                    "reject": ["", "", ""]}))  # fmt: skip
    assert cand["accepted"].tolist() == [True, False, True] and cand.at[1, "reject"] == "duplicate"


def test_tandem_tyres_split():
    cam = _cam()
    dh = PinholeGround(cam)
    xs = (3.5, -3.1, -4.5)  # tandem rear: 1.4 m apart
    contours, mask = _contours([_body(cam), *[_tyre(cam, x) for x in xs]])
    cand, _, _ = mc.extract(contours, _box(mask), [], dh, cam, 5)
    acc = cand[cand["accepted"]]
    assert len(acc) == 3, cand
    d = np.linalg.norm(acc[["u", "v"]].to_numpy()[:, None] - _contacts(cam, xs)[None], axis=2).min(axis=0)
    assert (d <= 1.5).all()


def test_nearer_box_and_image_edge_columns_dropped():
    o = np.array([[1.0, 100.0], [50.0, 100.0], [60.0, 100.0], [70.0, 100.0], [638.0, 100.0], [80.0, 104.0]])
    box = (0.0, 50.0, 640.0, 52.0)  # bottom 102
    nearer = [(55.0, 60.0, 10.0, 60.0)]  # bottom 120 > 102: nearer, covers u 55-65
    farther = [(65.0, 60.0, 10.0, 30.0)]  # bottom 90: farther, ignored
    ok = mc.trusted_mask(o, box, nearer + farther)
    assert ok.tolist() == [False, True, False, True, False, False]  # edge, ok, occluded, ok, edge, leak


def test_contour_roundtrip_and_frozen_price():
    a, b = np.array([[0, 0], [5, 0], [5, 5.0]]), np.array([[10, 10], [12, 10], [12, 13.0]])
    back = mc.split_contours(mc.join_contours([a, b]))
    assert len(back) == 2 and np.array_equal(back[0], a) and np.array_equal(back[1], b)
    cam = _cam()
    dh = PinholeGround(cam)
    contours, mask = _contours([_body(cam), *[_tyre(cam, x) for x in WHEELS_X]])
    c1, _, _ = mc.extract(contours, _box(mask), [], dh, cam, 5)
    c2, _, _ = mc.extract(contours, _box(mask), [], dh, cam, 5)  # no pose anywhere: identical prices
    assert c1[["gate_px", "kappa"]].equals(c2[["gate_px", "kappa"]])
