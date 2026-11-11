"""M5 SAM outline fit (ported from the speed study's tests/test_m5.py: the row-level checks go through `solve_m5`,
one test through the masks table and VehFootPrinter)."""

import inspect

import numpy as np
import pandas as pd
import pytest

from tssa.footprint import VehFootPrinter
from tssa.footprint.config import FT
from tssa.footprint.geometry import away_dir, diff_undirected, m2_points, rect_corners
from tssa.footprint.locators import m5
from tssa.footprint.locators.m5 import fit_rect, mirror_away, solve_m5
from tssa.footprint.masks import MaskTable, join_contours, lower_outline, split_contours

from synth import AffineGround

L, W = 15 * FT, 6 * FT
DH = AffineGround(A=((0.05, 0.01), (0.0, -0.08)), b=(580000.0, 2852000.0))


def _L_points(c, ax, n=30, seed=0, sides=((2, 1), (1, 0))):
    """Noisy points on two sides of the rectangle (corner index pairs; default right long side + front)."""
    rng = np.random.default_rng(seed)
    q = rect_corners(c, ax, L, W)  # fl, fr, rr, rl
    pts = []
    for a, b in ((q[i], q[j]) for i, j in sides):
        t = rng.uniform(0, 1, n // 2)[:, None]
        pts.append(a + t * (b - a))
    pts = np.vstack(pts)
    return pts + rng.normal(0, 0.05 * FT, pts.shape)


def test_fit_rect_recovers_L_outline():
    c_true, ax_true = np.array([100.0, 200.0]), 35.0
    pts = _L_points(c_true, ax_true)
    c, ax, res, sides = fit_rect(pts, L, W, c_true + 0.5)
    assert np.linalg.norm(c - c_true) / FT < 0.2 and diff_undirected(ax, ax_true) < 2 and sides == 2
    assert 0 <= ax < 180 and res < 0.2


def test_fit_rect_fixed_angle_starts_find_the_axis_without_a_reference():  # R19
    assert list(inspect.signature(fit_rect).parameters) == ["points_EN", "L_m", "W_m", "init_centre"]
    for k, ax_true in enumerate((0.0, 22.0, 67.0, 120.0, 158.0)):
        c_true = np.array([10.0, 20.0])
        pts = _L_points(c_true, ax_true, seed=k + 1)
        c, ax, res, sides = fit_rect(pts, L, W, c_true + 0.3)
        assert np.linalg.norm(c - c_true) / FT < 0.2 and diff_undirected(ax, ax_true) < 2, ax_true


def test_lower_outline_takes_max_v_per_column():
    pytest.importorskip("cv2")
    m = np.zeros((20, 30), bool)
    m[5:10, 4:8] = True
    m[3:15, 8:10] = True
    out = lower_outline(m, (4, 0, 6, 20))
    assert out.shape == (6, 2)
    assert np.allclose(out[:4, 1], 9) and np.allclose(out[4:, 1], 14)
    assert np.allclose(out[:, 0], np.arange(4, 10))
    poly = np.array([[4, 5], [7.5, 5], [7.5, 9.5], [4, 9.5]])
    o2 = lower_outline(poly, (4, 0, 4, 20))
    assert len(o2) >= 3 and (o2[:, 1] <= 10).all() and (o2[:, 1] >= 8).all()
    assert lower_outline(np.zeros((20, 30), bool), (4, 0, 6, 20)).shape == (0, 2)
    assert np.array_equal(lower_outline([poly], (4, 0, 4, 20)), o2)  # a list of contours, as the masks table holds


def _box_around(uv, h=40.0):
    bottom = float(uv[:, 1].max()) + 0.5
    return np.array([float(uv[:, 0].min()), bottom - h, float(np.ptp(uv[:, 0])), h])


def _setup(ncols=40):
    c_true = np.array([580010.0, 2852020.0])
    ax = 60.0
    pts = _L_points(c_true, ax, n=ncols, seed=2, sides=((2, 1), (3, 2)))  # camera-facing L (camera to the south)
    uv = DH.world_to_image(pts)
    box = _box_around(uv)
    return box, uv, c_true, ax, m2_points(DH, box[None, :], W)[0]


def test_solve_m5_ok():
    box, uv, c_true, ax, init = _setup()
    r = solve_m5(uv, box, DH, L, W, init)
    assert r["ok"] and np.linalg.norm(r["centre"] - c_true) / FT < 0.3 and diff_undirected(r["axis"], ax) < 3


def test_missing_outline_no_match():
    box, _, _, _, init = _setup()
    assert solve_m5(None, box, DH, L, W, init)["reason"] == "no_match"


def test_too_few_columns_invalid():
    box, uv, _, _, init = _setup()
    r = solve_m5(uv[:7], box, DH, L, W, init)
    assert not r["ok"] and r["reason"] == "few_columns" and r["cols"] == 7


def test_large_residual_invalid():
    rng = np.random.default_rng(3)
    uv = np.column_stack([rng.uniform(250, 350, 40), rng.uniform(150, 250, 40)])
    box = _box_around(uv)
    r = solve_m5(uv, box, DH, L, W, m2_points(DH, box[None, :], W)[0])
    assert not r["ok"] and r["reason"] == "fit_residual"


def test_mirror_away_reflects_centre_across_the_near_side():
    pts = np.column_stack([np.linspace(-2.0, 2.0, 20), np.zeros(20)])
    away = np.array([0.0, 1.0])
    c, flipped = mirror_away(pts, np.array([0.0, -W / 2]), 90.0, L, W, away)
    assert flipped and np.allclose(c, [0.0, W / 2])
    c, flipped = mirror_away(pts, np.array([0.0, W / 2]), 90.0, L, W, away)
    assert not flipped and np.allclose(c, [0.0, W / 2])


def _one_sided():
    c_true = np.array([580010.0, 2852020.0])
    q = rect_corners(c_true, 90.0, L, W)
    t = np.linspace(0.02, 0.98, 30)[:, None]
    pts = q[2] + t * (q[1] - q[2])
    uv = DH.world_to_image(pts)
    box = _box_around(uv)
    return box, uv, c_true, pts, m2_points(DH, box[None, :], W)[0]


def test_one_sided_outline_centre_on_the_away_side():
    box, uv, c_true, pts, init = _one_sided()
    r = solve_m5(uv, box, DH, L, W, init)
    assert r["ok"] and r["sides"] == 1
    away = away_dir(DH, uv.mean(axis=0))[0]
    assert float((r["centre"] - pts.mean(axis=0)) @ away) > 0
    assert np.linalg.norm(r["centre"] - c_true) / FT < 0.3


def test_solve_m5_flips_a_mirrored_fit(monkeypatch):
    box, uv, c_true, pts, init = _one_sided()
    mirrored = c_true - np.array([0.0, W])
    monkeypatch.setattr(m5, "fit_rect", lambda p, Lm, Wm, init: (mirrored, 90.0, 0.0, 1))
    r = solve_m5(uv, box, DH, L, W, init)
    assert r["flipped"] and r["ok"] and np.allclose(r["centre"], c_true, atol=1e-6)


def test_outline_points_below_bbox_bottom_are_ignored():
    box, uv, _, _, init = _setup()
    clean = solve_m5(uv, box, DH, L, W, init)
    bottom = box[1] + box[3]
    junk = np.column_stack([np.arange(300.0, 340.0), np.full(40, bottom + 1.5)])
    noisy = solve_m5(np.vstack([uv, junk]), box, DH, L, W, init)
    assert noisy["cols"] == clean["cols"] and np.allclose(noisy["centre"], clean["centre"])
    kept = solve_m5(np.vstack([uv, [[250.0, bottom + 1.0]]]), box, DH, L, W, init)
    assert kept["cols"] == clean["cols"] + 1


# --- masks table -------------------------------------------------------------------------------------------------


def test_contours_round_trip_and_npz(tmp_path):
    a = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 4.0]])
    b = np.array([[10.0, 10.0], [12.0, 10.0], [12.0, 13.0], [10.0, 13.0]])
    assert [p.tolist() for p in split_contours(join_contours([a, b]))] == [a.tolist(), b.tolist()]
    mt = MaskTable({(3, 7): [a, b], (3, 8): [b]})
    mt.save_npz(tmp_path / "m.npz")
    back = MaskTable.load_npz(tmp_path / "m.npz")
    assert set(back) == {(3, 7), (3, 8)} and np.array_equal(back[(3, 7)][1], b)


def test_m5_through_the_masks_table():
    pytest.importorskip("cv2")
    c_true = np.array([580010.0, 2851980.0])  # 20 m south: positive image rows under this affine stand-in
    face = DH.world_to_image(rect_corners(c_true, 60.0, L, W))  # a filled footprint-shaped mask
    box = np.array([face[:, 0].min(), face[:, 1].min() - 30, np.ptp(face[:, 0]), np.ptp(face[:, 1]) + 30])
    tr = pd.DataFrame({"frame": [0], "track": [1], "x": [box[0]], "y": [box[1]], "w": [box[2]], "h": [box[3]], "cls": [2]})
    loc = VehFootPrinter(DH, chains={2: ("M5",)}).locate(tr, masks=MaskTable({(1, 0): [face]}))
    r = loc.iloc[0]
    assert r.fp_source == "M5" and r.fp_orient_kind == "axis" and r.m5_cols > 8
    assert np.hypot(r.fp_x - c_true[0], r.fp_y - c_true[1]) / FT < 0.5
    assert VehFootPrinter(DH, chains={2: ("M5",)}).locate(tr, masks=MaskTable({}))["fp_fail"].iloc[0] == "M5:no_match"
