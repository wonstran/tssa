"""M3 and M6 (ported from the speed study's test_m3.py and test_m6.py) and the face geometry."""

import numpy as np
import pandas as pd
import pytest

from tssa.footprint import FootprintConfig, VehFootPrinter
from tssa.footprint.config import FT
from tssa.footprint.geometry import face_geometry, rect_corners
from tssa.footprint.locators.m1_m2_m3 import solve_m3

from synth import ORIGIN, AffineGround, ground


def _box(g, centre, heading, L, W):
    uv = g.world_to_image(rect_corners(centre, heading, L, W))
    x0, x1 = uv[:, 0].min(), uv[:, 0].max()
    return np.array([x0, uv[:, 1].min(), x1 - x0, uv[:, 1].max() - uv[:, 1].min()])


def _face_uv(g, centre_en, L, W, axis_deg):
    a = np.radians(axis_deg)
    u = np.array([np.sin(a), np.cos(a)])
    v = np.array([np.cos(a), -np.sin(a)])
    en = np.array([centre_en + s * L / 2 * u + t * W / 2 * v for s, t in [(1, 1), (1, -1), (-1, -1), (-1, 1)]])
    return g.world_to_image(en)


# --- M3 ----------------------------------------------------------------------------------------------------------


def test_m3_recovers_synthetic_rectangle():
    g = ground()
    centre_true = ORIGIN + [3.0, 4.0]
    L, W = 15 * FT, 6 * FT
    box = _box(g, centre_true, 40.0, L, W)
    centre, corners, ok, r = solve_m3(g, box, 40.0, L, W, init=centre_true + [1, 1])
    assert ok and r < 0.5
    assert np.linalg.norm(centre - centre_true) / FT < 0.05
    assert corners.shape == (4, 2)


def test_m3_fails_cleanly_without_heading():
    g = ground()
    box = _box(g, ORIGIN + [3.0, 4.0], 40.0, 15 * FT, 6 * FT)
    centre, corners, ok, r = solve_m3(g, box, np.nan, 15 * FT, 6 * FT, init=ORIGIN + [3.0, 4.0])
    assert not ok and np.isnan(centre).all() and np.isnan(corners).all()


# --- M6 ----------------------------------------------------------------------------------------------------------


def test_face_geometry_rectangle():
    g = AffineGround()
    c = ORIGIN + np.array([12.0, 7.0])
    uv = _face_uv(g, c, 15 * FT, 6 * FT, 30.0)
    corners, centroid, L, W, axis = face_geometry(uv, g)
    assert corners.shape == (4, 2)
    assert np.allclose(centroid, c)
    assert np.isclose(L, 15 * FT) and np.isclose(W, 6 * FT)
    assert np.isclose(axis, 30.0)


@pytest.mark.parametrize("flip", [False, True])
def test_face_corner_order_and_winding_irrelevant(flip):
    g = AffineGround()
    uv = _face_uv(g, ORIGIN + [3.0, 4.0], 15 * FT, 6 * FT, 30.0)
    uv = uv[::-1] if flip else uv
    base = face_geometry(_face_uv(g, ORIGIN + [3.0, 4.0], 15 * FT, 6 * FT, 30.0), g)
    for k in (0, 1, 2, 3):
        _, cen, L, W, axis = face_geometry(np.roll(uv, k, axis=0), g)
        assert np.isclose(L, base[2]) and np.isclose(W, base[3])
        assert np.isclose(axis % 180, base[4] % 180)
        assert np.allclose(cen, base[1])


def test_face_axis_survives_the_0_180_wrap():
    g = AffineGround()
    for a in (179.5, 0.5):
        assert face_geometry(_face_uv(g, ORIGIN + [3.0, 4.0], 15 * FT, 6 * FT, a), g)[4] == pytest.approx(a % 180)


def _one_row_table(g, faces, cls, iou=0.8):
    n = len(faces)
    tr = pd.DataFrame({"frame": np.zeros(n, int), "track": np.arange(1, n + 1), "x": 300.0, "y": 300.0, "w": 40.0,
                       "h": 30.0, "cls": cls})  # fmt: skip
    cub = {"track": np.arange(1, n + 1), "frame": np.zeros(n, int), "iou": iou}
    for i in range(4):
        cub[f"gu{i}"] = [f[i, 0] for f in faces]
        cub[f"gv{i}"] = [f[i, 1] for f in faces]
    return tr, pd.DataFrame(cub)


def test_m6_size_gate_and_match():
    """A bus face collapsed to 3 x 2 ft is implausible_size; 38 x 8 ft passes; an IoU of 0.1 is no_match."""
    g = AffineGround()
    faces = [_face_uv(g, ORIGIN + [3.0, 4.0], 3 * FT, 2 * FT, 30.0),
             _face_uv(g, ORIGIN + [3.0, 4.0], 38 * FT, 8 * FT, 30.0),
             _face_uv(g, ORIGIN + [3.0, 4.0], 15 * FT, 6 * FT, 30.0)]  # fmt: skip
    tr, cub = _one_row_table(g, faces, [5, 5, 2])
    cub.loc[2, "iou"] = 0.1
    loc = VehFootPrinter(g, chains={5: ("M6",), 2: ("M6",)}).locate(tr, cuboids=cub)
    assert loc["fp_source"].tolist() == ["none", "M6", "none"]
    assert loc["fp_fail"].tolist() == ["M6:implausible_size", "", "M6:no_match"]
    r = loc.iloc[1]
    assert np.allclose([r.fp_x, r.fp_y], ORIGIN + [3.0, 4.0])
    assert r.fp_orient_kind == "axis" and np.isclose(r.fp_orient_deg, 30.0)
    assert np.isclose(r.fp_len_ft, 38) and np.isclose(r.fp_wid_ft, 8) and r.m6_iou == 0.8
    assert np.isfinite(loc.loc[1, "m6_raw_c1x"]) and np.isnan(loc.loc[0, "m6_raw_c1x"])


def test_m6_gate_fraction_boundary():
    g = AffineGround()
    nom_l = FootprintConfig().size_ft[5][0]
    faces = [_face_uv(g, ORIGIN, 0.49 * nom_l * FT, 8 * FT, 0.0), _face_uv(g, ORIGIN, 0.51 * nom_l * FT, 8 * FT, 0.0)]
    tr, cub = _one_row_table(g, faces, [5, 5])
    loc = VehFootPrinter(g, chains={5: ("M6",)}).locate(tr, cuboids=cub)
    assert loc["fp_fail"].tolist() == ["M6:implausible_size", ""]
