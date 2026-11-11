"""Approximate pinhole camera fitted to a ground homography, and the hub -> contact drop.

Ported from the speed study's `camera.py` (snapshot docs/study-snapshot-2026-11-10.json; design
docs/2026-11-10-footprint-design.md section 8). Changes from the study: no OpenCV (`scipy.spatial.transform.Rotation`
replaces `cv2.Rodrigues`, a normalised DLT replaces the initial `cv2.findHomography(..., 0)`); the settings are
arguments; the ground model is a `GroundModel` (image size and pixel spaces from it).

World frame: right-handed (E, N, Up), metres. Public functions take and return world coordinates (e.g. UTM); the
camera keeps `R` and `C` relative to its own calibration origin (`PinholeCamera.origin`). The camera works on
UNDISTORTED pixels; the ground model's `distort` is applied afterwards.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .ground import GroundModel

GRID_STEP_M = 1.0
FOCAL_SCALES = (0.87, 1.0, 1.13)


@dataclass
class PinholeCamera:
    """Pinhole K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]; x_cam = R (X - origin3 - C). Image u right, v down."""

    f: float
    R: np.ndarray  # (3, 3) world -> camera
    C: np.ndarray  # (3,) camera centre relative to origin
    cx: float
    cy: float
    origin: np.ndarray  # (2,) world (E, N)
    converged: bool = True  # False when the fit's least squares did not converge

    def __post_init__(self):
        self.f = float(self.f)
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.C = np.asarray(self.C, dtype=float).reshape(3)
        self.cx = float(self.cx)
        self.cy = float(self.cy)
        self.origin = np.asarray(self.origin, dtype=float).reshape(2)
        self.converged = bool(self.converged)

    @property
    def _origin3(self) -> np.ndarray:
        return np.array([self.origin[0], self.origin[1], 0.0])

    @property
    def centre(self) -> np.ndarray:
        """Camera centre in world coordinates (3,)."""
        return self.C + self._origin3

    @property
    def height(self) -> float:
        """Camera height above the ground plane (m)."""
        return float(self.C[2])

    def project(self, XYZ) -> np.ndarray:
        """World (n, 3) -> undistorted pixels (n, 2); NaN for points not in front of the camera."""
        X = np.atleast_2d(np.asarray(XYZ, dtype=float)) - self._origin3
        uv, depth = _project_local(self.f, self.R, self.C, self.cx, self.cy, X)
        uv[depth <= 0] = np.nan
        return uv

    def ray(self, uv_undist):
        """Undistorted pixels (n, 2) -> (camera centre world (3,), unit world directions (n, 3))."""
        uv = np.atleast_2d(np.asarray(uv_undist, dtype=float))
        d_cam = np.column_stack([(uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f, np.ones(len(uv))])
        d = d_cam @ self.R  # R.T @ d_cam, row-wise
        return self.C + self._origin3, d / np.linalg.norm(d, axis=1, keepdims=True)

    @classmethod
    def fit(cls, ground, zone, f_fixed: float | None = None, free_principal: bool = True,
            pp_prior: float = 1e-3) -> "PinholeCamera":  # fmt: skip
        """Least-squares pinhole fit to the ground model over a 1 m grid in the zone (world coordinates).

        Correspondences are world (x, y, 0) -> `ground.world_to_undistorted` pixels. Fitted: f (unless f_fixed),
        (cx, cy) when free_principal (starting at the image centre, else fixed there), R and C. With a free principal
        point a single ground plane fixes (f, cx, cy) only up to a one-parameter family of exact fits; two weak prior
        residuals `pp_prior * (cx - cx0, cy - cy0)` pick the member nearest the image centre (study ruling R18).
        If the least squares does not converge, the camera is returned with `converged=False` and a RuntimeWarning.
        """
        g = GroundModel.wrap(ground)
        EN = zone_grid(zone)
        uv_u = np.asarray(g.world_to_undistorted(EN), dtype=float)
        ok = np.isfinite(uv_u).all(axis=1)
        EN, uv_u = EN[ok], uv_u[ok]
        free = bool(free_principal)
        n_par = 6 + (f_fixed is None) + 2 * free
        need = max(4, -(-(n_par - 2 * free) // 2))  # the prior adds 2 residuals when free
        if len(EN) < need:
            raise ValueError(f"PinholeCamera.fit: only {len(EN)} finite correspondences in the zone, need {need}")
        c = zone.centroid
        origin = np.floor([c.x, c.y])
        X = np.column_stack([EN - origin, np.zeros(len(EN))])
        cx0, cy0 = g.image_size[0] / 2.0, g.image_size[1] / 2.0

        Hw2i = _dlt(X[:, :2], uv_u)
        f0, R0, C0 = _decompose(Hw2i, cx0, cy0, f_fixed)
        rvec0 = Rotation.from_matrix(R0).as_rotvec()

        def unpack(p):  # p = [f?] + [cx, cy]? + rvec + C
            i = 0
            if f_fixed is None:
                f, i = p[0], 1
            else:
                f = float(f_fixed)
            if free:
                cx, cy, i = p[i], p[i + 1], i + 2
            else:
                cx, cy = cx0, cy0
            return f, cx, cy, Rotation.from_rotvec(p[i:i + 3]).as_matrix(), p[i + 3:i + 6]

        def resid(p):
            f, cx, cy, R, C = unpack(p)
            uv, _ = _project_local(f, R, C, cx, cy, X)
            r = (uv - uv_u).ravel()
            if free:
                r = np.concatenate([r, [pp_prior * (cx - cx0), pp_prior * (cy - cy0)]])
            return r

        p0 = np.concatenate([[f0] if f_fixed is None else [], [cx0, cy0] if free else [], rvec0, C0])
        sol = least_squares(resid, p0, method="lm", x_scale="jac", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
        f, cx, cy, R, C = unpack(sol.x)
        if not sol.success:
            warnings.warn(f"PinholeCamera.fit: least squares did not converge (status {sol.status}: {sol.message}); "
                          "camera flagged converged=False", RuntimeWarning, stacklevel=2)  # fmt: skip
        return cls(f, R, C, cx, cy, origin, converged=bool(sol.success))


def _project_local(f, R, C, cx, cy, X_local):
    Xc = (X_local - C) @ R.T
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.column_stack([f * Xc[:, 0] / z + cx, f * Xc[:, 1] / z + cy])
    return uv, z


def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares plane homography src -> dst by the normalised DLT (replaces cv2.findHomography(src, dst, 0))."""

    def norm(p):
        m = p.mean(axis=0)
        s = np.sqrt(2.0) / np.mean(np.linalg.norm(p - m, axis=1))
        return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1.0]])

    Ts, Td = norm(src), norm(dst)
    a = np.column_stack([src, np.ones(len(src))]) @ Ts.T
    b = np.column_stack([dst, np.ones(len(dst))]) @ Td.T
    rows = []
    for (x, y, _), (u, v, _) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(rows))
    H = np.linalg.inv(Td) @ Vt[-1].reshape(3, 3) @ Ts
    if abs(H[2, 2]) < 1e-15:
        raise ValueError("PinholeCamera.fit: homography initialisation failed")
    return H / H[2, 2]


def look_at(C, target=(0, 0, 0), f=500.0, cx=320.0, cy=240.0, origin=(0, 0)) -> PinholeCamera:
    """Camera at C (world, 3) aimed at target (world, 3), with no roll (camera x-axis horizontal)."""
    C = np.asarray(C, dtype=float)
    fwd = np.asarray(target, dtype=float) - C
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("look_at: viewing direction is vertical, roll is undefined")
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    origin = np.asarray(origin, dtype=float)
    return PinholeCamera(f, np.vstack([right, down, fwd]), C - [origin[0], origin[1], 0.0], cx, cy, origin)


class PinholeGround:
    """A pinhole camera's z = 0 plane as a ground model (no distortion): for synthetic data and tests."""

    def __init__(self, cam: PinholeCamera, image_size=None):
        self.cam = cam
        self.image_size = tuple(image_size) if image_size is not None else (2.0 * cam.cx, 2.0 * cam.cy)

    def image_to_world(self, uv) -> np.ndarray:
        uv = np.asarray(uv, dtype=float)
        C, d = self.cam.ray(uv.reshape(-1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -C[2] / d[:, 2]
        EN = C[:2] + t[:, None] * d[:, :2]
        EN[~(t > 0)] = np.nan  # the ray does not reach the ground in front of the camera
        return EN.reshape(uv.shape)

    def world_to_image(self, EN) -> np.ndarray:
        EN = np.asarray(EN, dtype=float)
        E2 = np.atleast_2d(EN)
        return self.cam.project(np.column_stack([E2, np.zeros(len(E2))])).reshape(EN.shape)


def zone_grid(zone, step: float = GRID_STEP_M) -> np.ndarray:
    """World (n, 2) points of a `step` m grid inside the zone."""
    x0, y0, x1, y1 = zone.bounds
    xs = np.arange(np.floor(x0), x1 + step, step)
    ys = np.arange(np.floor(y0), y1 + step, step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pts = pts[shapely.contains_xy(zone, pts[:, 0], pts[:, 1])]
    if len(pts) < 8:
        raise ValueError(f"zone too small for a camera fit: {len(pts)} grid points")
    return pts


def _decompose(Hw2i, cx, cy, f_fixed=None):
    """Plane homography world (local x, y) -> undistorted px  =>  (f, R, C) with K^-1 H ∝ [r1 r2 t]."""
    T = np.array([[1.0, 0, -cx], [0, 1.0, -cy], [0, 0, 1.0]])
    H = T @ Hw2i
    h1, h2 = H[:, 0], H[:, 1]
    if f_fixed is None:
        cands = []
        den = h1[2] * h2[2]
        if abs(den) > 1e-15:  # orthogonality r1 . r2 = 0
            cands.append(-(h1[0] * h2[0] + h1[1] * h2[1]) / den)
        den = h2[2] ** 2 - h1[2] ** 2
        if abs(den) > 1e-15:  # equal norms |r1| = |r2|
            cands.append((h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / den)
        fs = [np.sqrt(c) for c in cands if np.isfinite(c) and c > 0]
        f = float(np.mean(fs)) if fs else float(max(cx, cy) * 2.0)
    else:
        f = float(f_fixed)
    M = np.diag([1.0 / f, 1.0 / f, 1.0]) @ H
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    if M[2, 2] < 0:  # the calibration origin (zone centroid) must be in front of the camera
        lam = -lam
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    C = -R.T @ t
    return f, R, C


def consistency(cam: PinholeCamera, ground, zone, rms_px: float = 1.0, max_px: float = 3.0) -> dict:
    """Camera z = 0 projection + distort vs world_to_image over the zone grid (px).

    ok needs rms <= rms_px, max <= max_px and a converged fit.
    """
    g = GroundModel.wrap(ground)
    EN = zone_grid(zone)
    uv_cam = np.asarray(g.distort(cam.project(np.column_stack([EN, np.zeros(len(EN))]))), dtype=float)
    uv_g = np.asarray(g.world_to_image(EN), dtype=float)
    err = np.linalg.norm(uv_cam - uv_g, axis=1)
    rms = float(np.sqrt(np.mean(err**2)))
    mx = float(np.max(err))
    return dict(rms_px=rms, max_px=mx, ok=bool(cam.converged and rms <= rms_px and mx <= max_px))


def sensitivity(ground, zone, base: PinholeCamera, scales=FOCAL_SCALES, free_principal: bool = True,
                pp_prior: float = 1e-3, rms_px: float = 1.0, max_px: float = 3.0) -> list[dict]:  # fmt: skip
    """Refit (cx, cy), R, C with f fixed at base.f x each scale; each refit reports its own ground residual."""
    out = []
    for s in scales:
        cam = PinholeCamera.fit(ground, zone, f_fixed=base.f * s, free_principal=free_principal, pp_prior=pp_prior)
        res = consistency(cam, ground, zone, rms_px, max_px)
        out.append(dict(scale=s, camera=cam, rms_px=res["rms_px"], max_px=res["max_px"], ok=res["ok"]))
    return out


def hub_to_contact(cam: PinholeCamera, uv_undist, r_m) -> np.ndarray:
    """Hub pixel (undistorted) -> ground contact world (n, 2): the ray meets z = r, then drops vertically."""
    C, d = cam.ray(uv_undist)
    r = np.broadcast_to(np.asarray(r_m, dtype=float), (len(d),))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (r - C[2]) / d[:, 2]
    hub = C + t[:, None] * d
    contact = hub[:, :2]
    contact[~(t > 0)] = np.nan
    return contact
