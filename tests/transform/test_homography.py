"""Tests for tssa.transform.homography."""

import numpy as np
import pytest

from tssa.transform.homography import HomographyTransform

# Square in the image mapped to a 10x scaled, UTM-sized square in the world
IMAGE_PTS = np.array([[100, 200], [200, 200], [200, 300], [100, 300], [150, 250]], dtype=float)
ORIGIN = np.array([500000.0, 3000000.0])
WORLD_PTS = (IMAGE_PTS - [100, 200]) * 0.1 + ORIGIN


@pytest.fixture
def transform():
    return HomographyTransform(IMAGE_PTS, WORLD_PTS)


def test_image_to_world_keeps_sub_unit_precision_at_utm_scale(transform):
    out = transform.image_to_world(np.array([101.0, 200.0]))
    assert out.shape == (2,)
    np.testing.assert_allclose(out, ORIGIN + [0.1, 0.0], atol=1e-6)


def test_round_trip(transform):
    world = transform.image_to_world(IMAGE_PTS)
    np.testing.assert_allclose(transform.world_to_image(world), IMAGE_PTS, atol=1e-6)


def test_empty_input_returns_empty(transform):
    assert transform.image_to_world(np.empty((0, 2))).shape == (0, 2)


def test_reprojection_error_is_zero_for_exact_fit(transform):
    assert transform.reprojection_error(IMAGE_PTS, WORLD_PTS) == pytest.approx(0.0, abs=1e-6)


def test_reprojection_error_measures_pixel_offset(transform):
    shifted = IMAGE_PTS + [3.0, 4.0]
    assert transform.reprojection_error(shifted, WORLD_PTS) == pytest.approx(5.0, abs=1e-6)


def test_repr_reports_point_count(transform):
    assert repr(transform) == "HomographyTransform(n_points=5, inliers=5)"


def test_save_load_round_trip(transform, tmp_path):
    path = tmp_path / "h.npz"
    transform.save_matrix(str(path))
    loaded = HomographyTransform.load_matrix(str(path))
    np.testing.assert_allclose(loaded.H, transform.H)
    assert loaded.n_points == 5


def test_save_load_without_inlier_mask(transform, tmp_path):
    transform.inlier_mask = None
    path = tmp_path / "h.npz"
    transform.save_matrix(str(path))
    loaded = HomographyTransform.load_matrix(str(path))
    assert loaded.inlier_mask is None
    loaded.save_matrix(str(path))
