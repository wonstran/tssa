"""Synthetic scenes for the footprint tests: a pinhole camera over a UTM-sized origin, vehicles rendered to boxes and
cuboid faces, and stub locators for the M7-shaped paths."""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd

from tssa.footprint.config import FT
from tssa.footprint.geometry import rect_corners
from tssa.footprint.locators import REGISTRY, LocatorResult, register_locator
from tssa.transform.camera import PinholeGround, look_at

ORIGIN = np.array([580000.0, 2852000.0])


def ground(f=500.0):
    """A camera 7 m up, 25 m east and 30 m south of the origin, aimed at it; image 640 x 480."""
    cam = look_at(C=(ORIGIN[0] + 25, ORIGIN[1] - 30, 7), target=(ORIGIN[0], ORIGIN[1], 0), f=f, origin=ORIGIN)
    return PinholeGround(cam, image_size=(640, 480))


class AffineGround:
    """Image -> world affine map `uv @ A.T + b` (identity distortion), as the study's test stand-in."""

    image_size = (640, 480)

    def __init__(self, A=((0.05, 0.01), (0.0, -0.08)), b=ORIGIN):
        self.A, self.b = np.asarray(A, float), np.asarray(b, float)

    def image_to_world(self, uv):
        return np.asarray(uv, float) @ self.A.T + self.b

    def world_to_image(self, xy):
        return (np.asarray(xy, float) - self.b) @ np.linalg.inv(self.A).T


def render(g, centres, headings, L_ft=15.0, W_ft=6.0, H_ft=5.0, noise_px=0.0, seed=0):
    """Boxes (x, y, w, h) of the 3-D body and its ground face (4 image corners) per pose."""
    rng = np.random.default_rng(seed)
    boxes, faces = [], []
    for c, psi in zip(np.atleast_2d(centres), np.atleast_1d(headings)):
        q = rect_corners(np.asarray(c, float), float(psi), L_ft * FT, W_ft * FT)
        pts = np.vstack([np.column_stack([q, np.zeros(4)]), np.column_stack([q, np.full(4, H_ft * FT)])])
        uv = g.cam.project(pts)
        face = uv[:4] + rng.normal(0, noise_px, (4, 2))
        x0, y0 = uv[:, 0].min(), uv[:, 1].min()
        box = np.array([x0, y0, uv[:, 0].max() - x0, uv[:, 1].max() - y0]) + rng.normal(0, noise_px, 4)
        boxes.append(box)
        faces.append(face.ravel())
    return np.array(boxes), np.array(faces)


def tables(frames, track, cls, boxes, faces):
    tr = pd.DataFrame({"frame": frames, "track": track, "x": boxes[:, 0], "y": boxes[:, 1], "w": boxes[:, 2],
                       "h": boxes[:, 3], "cls": cls})  # fmt: skip
    cub = pd.DataFrame(faces, columns=["gu0", "gv0", "gu1", "gv1", "gu2", "gv2", "gu3", "gv3"])
    cub.insert(0, "frame", frames)
    cub.insert(0, "track", track)
    cub["iou"] = 0.9
    return tr, cub


def straight(n=40, speed=4.0, fps=10.0, x=-3.0, y0=-14.0):
    """A path due north at `speed` m/s: centres (n, 2) and headings (n,)."""
    t = np.arange(n) / fps
    return ORIGIN + np.column_stack([np.full(n, x), y0 + speed * t]), np.zeros(n)


def turn(n=40, speed=5.0, fps=10.0, R=10.0):
    """A quarter circle, right turn from heading north to east, centre of the circle at origin + (R - 3, -5)."""
    s = speed * np.arange(n) / fps
    a = s / R  # turned angle
    cen = ORIGIN + [R - 3.0, -5.0]
    pts = cen + R * np.column_stack([-np.cos(a), np.sin(a)])
    return pts, np.degrees(a)


@contextlib.contextmanager
def stub_locator(name, fn, needs=(), scope="segment", classes=(5, 7), point_only=False):
    """Register a test locator for the duration of a block."""
    register_locator(name, needs=needs, scope=scope, classes=classes, point_only=point_only)(fn)
    try:
        yield
    finally:
        REGISTRY.pop(name, None)


def kinematic_stub(pose_fn, cov=0.04, native_frames=(), passed=True, seg_size=(30.0, 8.0, "m7_measured"),
                   orient_kind="directed"):  # fmt: skip
    """A segment locator that returns `pose_fn(rows)` (x, y, psi) for every row: native on `native_frames`, kinematic
    elsewhere with covariance cov * I (m^2). `passed=False` fails every row with 'param_bound'."""

    def fn(rows, ctx):
        n = len(rows)
        res = LocatorResult.empty(n)
        if not passed:
            res.fail_reason[:] = "param_bound"
            res.extras["m7_solve_ok"] = np.zeros(n, bool)
            return res
        x, y, psi = pose_fn(rows)
        res.ok[:] = True
        res.fail_reason[:] = ""
        res.x, res.y = np.asarray(x, float), np.asarray(y, float)
        res.orient_deg[:], res.orient_kind[:] = psi, orient_kind
        nat = rows["frame"].isin(native_frames).to_numpy()
        res.support = np.where(nat, "native", "kinematic").astype(object)
        res.extras["m7_support"] = np.where(nat, "wheels2", "kinematic").astype(object)
        res.cov = np.tile(np.eye(2) * cov, (n, 1, 1))
        res.seg_len_ft = np.full(n, seg_size[0])
        res.seg_wid_ft = np.full(n, seg_size[1])
        res.seg_dims_source = np.full(n, seg_size[2], dtype=object)
        res.extras["m7_solve_ok"] = np.ones(n, bool)
        return res

    return fn
