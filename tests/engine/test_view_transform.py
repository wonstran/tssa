"""Tests for tssa.engine.view_transform."""

import numpy as np

from tssa.engine import ViewTransform

SOURCE = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
DESTINATION = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])


def test_transform_returns_float_without_truncation():
    vt = ViewTransform(SOURCE, DESTINATION)
    out = vt.transform(np.array([15, 25]))
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [[1.5, 2.5]], atol=1e-5)


def test_transform_inv_round_trip():
    vt = ViewTransform(SOURCE, DESTINATION)
    pts = np.array([[15.0, 25.0], [33.3, 66.6]])
    np.testing.assert_allclose(vt.transform_inv(vt.transform(pts)), pts, atol=1e-4)
