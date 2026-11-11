"""M2-only and M6-only estimate() without OpenCV, torch or ultralytics (design section 14.2 test 8, footprint part).

A child interpreter makes any import of cv2, torch or ultralytics raise ImportError before importing tssa.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"

CHILD = r"""
import json, sys

class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("cv2", "torch", "ultralytics"):
            raise ImportError(f"{{name}} is blocked in this test")
        return None

sys.meta_path.insert(0, Block())
sys.path[:0] = [{src!r}, {here!r}]
import numpy as np
from tssa.footprint import VehFootPrinter
from synth import ground, render, straight, tables
g = ground()
c, h = straight()
boxes, faces = render(g, c, h)
tr, cub = tables(np.arange(len(c)), 1, 2, boxes, faces)
m2 = VehFootPrinter(g, chains={{2: ("M2",)}}).estimate(tr)
m6 = VehFootPrinter(g).estimate(tr, cuboids=cub)
print(json.dumps({{"m2": sorted(set(m2["fp_source"])), "m6": sorted(set(m6["fp_source"])),
                  "poly": int(m6["c1x"].notna().sum()), "cv2": "cv2" in sys.modules}}))
"""


def test_m2_and_m6_estimate_without_opencv():
    code = CHILD.format(src=str(SRC), here=str(HERE))
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["m2"] == ["M2"] and "M6" in out["m6"] and out["poly"] > 0 and not out["cv2"]
