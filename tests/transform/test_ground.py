"""GroundModel adapter (design docs/2026-11-10-footprint-design.md section 4.2; test 6 of section 14.2)."""

import numpy as np
import pytest

from tssa.transform import DistortedHomography, GroundModel, HomographyTransform

ORIGIN = np.array([587000.0, 2852900.0])


def plane_points(k1=-0.08):
    """Image grid through a known radial distortion, and the world points of a known plane homography."""
    u, v = np.meshgrid(np.linspace(40, 600, 8), np.linspace(200, 460, 6))
    raw = np.column_stack([u.ravel(), v.ravel()])
    c, s = np.array([320.0, 240.0]), 320.0
    r2 = (((raw - c) / s) ** 2).sum(axis=1, keepdims=True)
    und = c + (raw - c) * (1 + k1 * r2)
    H = np.array([[0.05, 0.002, -15.0], [0.001, -0.09, 40.0], [0.0, 0.0012, 1.0]])
    w = np.column_stack([und, np.ones(len(und))]) @ H.T
    return raw, w[:, :2] / w[:, 2:] + ORIGIN


@pytest.fixture(scope="module")
def distorted():
    raw, world = plane_points()
    return DistortedHomography(raw, world, image_size=(640, 480))


@pytest.fixture(scope="module")
def plain():
    raw, world = plane_points(k1=0.0)
    return HomographyTransform(raw, world)


def test_plain_homography_needs_an_explicit_size(plain):
    with pytest.raises(ValueError, match="image size unknown"):
        GroundModel.wrap(plain)
    g = GroundModel.wrap(plain, image_size=(640, 480))
    assert g.image_size == (640.0, 480.0) and not g.has_distortion


def test_plain_homography_has_identity_distortion(plain):
    g = GroundModel.wrap(plain, image_size=(640, 480))
    uv = np.array([[100.0, 300.0], [500.0, 420.0]])
    np.testing.assert_array_equal(g.undistort(uv), uv)
    np.testing.assert_array_equal(g.distort(uv), uv)
    world = g.image_to_world(uv)
    np.testing.assert_array_equal(g.world_to_undistorted(world), g.world_to_image(world))
    assert g.undistort(uv[0]).shape == (2,)


def test_distorted_homography_gives_its_own_size(distorted):
    g = GroundModel.wrap(distorted)
    assert g.image_size == (640.0, 480.0) and g.has_distortion
    assert GroundModel.wrap(distorted, image_size=(640, 480)).image_size == (640.0, 480.0)
    with pytest.raises(ValueError, match="conflicts"):
        GroundModel.wrap(distorted, image_size=(1280, 720))


def test_world_to_undistorted_uses_the_inner_homography(distorted):
    g = GroundModel.wrap(distorted)
    raw, world = plane_points()
    direct = g.world_to_undistorted(world)
    np.testing.assert_allclose(direct, distorted.undistort(raw), rtol=0, atol=1e-3)  # the fit is not exact
    via = g.undistort(g.world_to_image(world))
    np.testing.assert_allclose(direct, via, rtol=0, atol=1e-6)  # round trip through distort() agrees closely
    assert g.world_to_undistorted(world[0]).shape == (2,)


def test_half_a_distortion_model_is_rejected(plain):
    class OnlyUndistort:
        image_size = (640, 480)

        def image_to_world(self, uv):
            return plain.image_to_world(uv)

        def world_to_image(self, en):
            return plain.world_to_image(en)

        def undistort(self, uv):
            return uv

    with pytest.raises(ValueError, match="only one of"):
        GroundModel.wrap(OnlyUndistort())


def test_object_without_projection_is_rejected():
    with pytest.raises(ValueError, match="lacks image_to_world"):
        GroundModel(object(), (640, 480))


def test_wrap_is_idempotent(distorted):
    g = GroundModel.wrap(distorted)
    assert GroundModel.wrap(g) is g
    with pytest.raises(ValueError, match="conflicts"):
        GroundModel.wrap(g, image_size=(320, 240))
