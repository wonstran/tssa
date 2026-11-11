"""OpenCV-free loading and projection (design section 13; test 8 of section 14.2, transform part).

A child interpreter makes any import of cv2, torch or ultralytics raise ImportError (an import hook, not
`sys.modules[name] = None`: scipy probes `sys.modules["torch"]` and breaks on a None entry) before importing tssa.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tssa.transform import DistortedHomography, HomographyTransform

SRC = Path(__file__).resolve().parents[2] / "src"

CHILD = r"""
import json, sys

class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("cv2", "torch", "ultralytics"):
            raise ImportError(f"{{name}} is blocked in this test")
        return None

sys.meta_path.insert(0, Block())
sys.path.insert(0, {src!r})
import numpy as np
import tssa, tssa.transform, tssa.smooth, tssa.speed
from tssa.transform import DistortedHomography, GroundModel, HomographyTransform
d = DistortedHomography.load_json({dj!r})
dm = DistortedHomography.load_matrix({dm!r})
h = HomographyTransform.load_json({hj!r})
hm = HomographyTransform.load_matrix({hm!r})
uv = np.array({uv!r})
out = {{
    "d_w": d.image_to_world(uv).tolist(), "d_uv": d.world_to_image(d.image_to_world(uv)).tolist(),
    "dm_w": dm.image_to_world(uv).tolist(), "h_w": h.image_to_world(uv).tolist(),
    "hm_uv": hm.world_to_image(h.image_to_world(uv)).tolist(),
    "g_und": GroundModel.wrap(d).world_to_undistorted(d.image_to_world(uv)).tolist(),
}}
try:
    sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    HomographyTransform(sq, sq * 10)
    out["fit"] = "no error"
except ImportError as e:
    out["fit"] = str(e)
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    img = np.array([[60, 300], [580, 310], [600, 460], [40, 450], [320, 260], [320, 400], [150, 380], [480, 380]], float)
    world = np.array([[0, 30], [25, 30], [22, 2], [3, 3], [12, 40], [12, 10], [5, 14], [19, 14]], float) + [587000, 2852900]
    dh = DistortedHomography(img, world, image_size=(640, 480))
    h = HomographyTransform(img, world)
    paths = {"dj": d / "d.json", "dm": d / "d.npz", "hj": d / "h.json", "hm": d / "h.npz"}
    dh.save_json(paths["dj"]), dh.save_matrix(paths["dm"]), h.save_json(paths["hj"]), h.save_matrix(paths["hm"])
    return dh, h, {k: str(v) for k, v in paths.items()}


def test_load_and_project_without_opencv(saved):
    dh, h, paths = saved
    uv = [[100.0, 350.0], [500.0, 420.0], [320.0, 300.0]]
    code = CHILD.format(src=str(SRC), uv=uv, **paths)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    a = np.array(uv)
    np.testing.assert_allclose(out["d_w"], dh.image_to_world(a), rtol=0, atol=1e-9)
    np.testing.assert_allclose(out["dm_w"], dh.image_to_world(a), rtol=0, atol=1e-9)
    np.testing.assert_allclose(out["d_uv"], a, rtol=0, atol=1e-6)
    np.testing.assert_allclose(out["h_w"], h.image_to_world(a), rtol=0, atol=1e-9)
    np.testing.assert_allclose(out["hm_uv"], a, rtol=0, atol=1e-6)
    np.testing.assert_allclose(out["g_und"], dh.undistort(a), rtol=0, atol=1e-6)
    assert "OpenCV" in out["fit"]  # fitting a homography still needs it, and says so


def test_numpy_projection_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    for _ in range(20):
        H = rng.normal(size=(3, 3))
        H[2] *= 1e-3
        H[2, 2] = 1.0
        pts = rng.uniform(-2000, 2000, size=(200, 2))
        horizon = np.array([[0.0, -H[2, 2] / H[2, 1]], [1.0, -(H[2, 2] + H[2, 0]) / H[2, 1]]])  # w = 0 exactly
        pts = np.vstack([pts, horizon])
        ref = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        np.testing.assert_allclose(HomographyTransform._project(pts, H), ref, rtol=1e-12, atol=1e-9)
