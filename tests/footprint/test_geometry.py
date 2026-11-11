"""Angles, M1/M2 points, the projection gate and rectangles (ported from the speed study's test_geometry.py and
test_m3.py)."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from tssa.footprint.config import FT
from tssa.footprint.geometry import (away_dir, compass_deg, diff_directed, diff_undirected, direct_toward, edge_flag,
                                     jacobian, m1_points, m2_points, projection_valid, rect_corners, rot_ccw,
                                     to_undirected, unit_from_compass)  # fmt: skip

from synth import ORIGIN, AffineGround


def test_compass_convention():
    assert compass_deg(np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])).tolist() == [0, 90, 180, 270]
    assert np.allclose(rot_ccw(np.array([0, 1.0])), [-1, 0])
    assert np.allclose(unit_from_compass(np.array([0.0, 90.0])), [[0, 1], [1, 0]], atol=1e-12)


def test_undirected_diff_and_direct_toward():
    assert diff_undirected(10, 185) == pytest.approx(5)
    assert diff_directed(10, 350) == pytest.approx(20)
    assert to_undirected(190) == pytest.approx(10)
    assert direct_toward(170, 355) == pytest.approx(350)
    assert np.allclose(direct_toward(np.array([170.0]), np.array([355.0])), [350])
    assert direct_toward(170, np.nan) == pytest.approx(170)  # no reference keeps the axis


def test_m2_shift_is_half_width_away_from_camera():
    g = AffineGround()
    box = np.array([[100.0, 100.0, 40.0, 20.0]])
    uv = np.array([[120.0, 120.0]])
    assert np.allclose(m1_points(g, box), g.image_to_world(uv))
    shift = m2_points(g, box, np.array([6 * FT])) - m1_points(g, box)
    assert np.linalg.norm(shift) == pytest.approx(3 * FT)
    assert np.allclose(shift / np.linalg.norm(shift), away_dir(g, uv)[0])


def test_jacobian_matches_the_affine_map():
    g = AffineGround()
    J = jacobian(g, np.array([[100.0, 200.0], [300.0, 50.0]]))
    np.testing.assert_allclose(J, np.stack([g.A, g.A]), rtol=0, atol=1e-9)


def test_projection_gate():
    zone = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    EN = np.array([[50.0, 50.0], [-150 * FT, 50.0], [50.0, 50.0], [np.nan, 50.0]])
    J = np.tile(np.eye(2) * 0.5 * FT, (4, 1, 1))
    J[2] = np.eye(2) * 3 * FT
    assert projection_valid(EN, J, 1.0, zone, 10).tolist() == [True, False, False, False]
    assert projection_valid(EN, J, 1.0).tolist() == [True, True, False, False]  # no zone: scale and finiteness only


def test_edge_flag():
    boxes = np.array([[1.0, 50, 10, 10], [100, 50, 10, 10], [630, 50, 9, 10], [100, 469, 10, 10]])
    assert edge_flag(boxes, (640, 480), 2).tolist() == [True, False, True, True]


def test_rect_corners_order_and_vectorised():
    c = rect_corners(np.array([0.0, 0.0]), 90.0, 4.0, 2.0)  # heading east; front-left (left = north) first
    assert np.allclose(c, [[2, 1], [2, -1], [-2, -1], [-2, 1]])
    v = rect_corners(np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([90.0, 0.0]), np.array([4.0, 4.0]), 2.0)
    assert v.shape == (2, 4, 2)
    assert np.allclose(v[0], c)
    assert np.allclose(v[1], rect_corners(np.array([10.0, 0.0]), 0.0, 4.0, 2.0))
