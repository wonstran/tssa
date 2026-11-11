"""Ported from the speed study's m7.py (sha256 d8fc196b1e8375caeaef486eaadcc7314bbf65eaa82c92b26ed67ef8bcbf108f, snapshot docs/study-snapshot-2026-11-10.json).
Changes: settings through the per-solve namespace `S` (an M7Config plus FT, MPS_TO_MPH, IMG_SIZE, EDGE_PX,
SIZE_FT) instead of the study's settings module; image-size default resolved at call time; the width-hold
decision moved into `width_held` (same expression) so it can be tested on its own.

M7: wheel-anchored rigid-body fit of one bus or truck segment (design 2026-11-08-m7, rev 5, sections 4-8).

Frames and units: positions are LOCAL metres (the grid's frame; `origin` converts to UTM for the camera), headings
compass radians internally (0 = north, clockwise), f = (sin psi, cos psi), n_left = (-cos psi, sin psi). Image
residuals are in UNDISTORTED pixels (`cam.project`).

Unknowns: per frame the rear-axle centre R_k (2) and heading psi_k; per solve the statics
theta = (L_w, L_f, L_r, W, H[, s_t]) in metres. Observations per frame (`FrameObs`, frozen before any fit):
wheel candidates with frozen rejection prices kappa_c, lower-envelope columns and the two end extents.

Objective F (section 7.4) = sum over candidates of [matched: rho(|d| / sigma_t) | rejected: kappa_c]
+ envelope and extent rho terms + 1/2 motion z^2 + 1/2 prior z^2, rho(z) = log(1 + z^2). Association (5.2) is
the exact minimiser of F's wheel part for a pose; the pose is fitted by IRLS (a majorise-minimise step for
Cauchy) with backtracking on F. The evidence score E (Laplace over the statics) is computed once per
completed fit and only compares models (axle configuration, subtype).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import lil_matrix
from scipy.stats import multivariate_normal

from .candidates import rho
from .settings import S

FT = 0.3048  # m per ft (S.FT)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
BIG = 1e9


# --- observations ----------------------------------------------------------------------------------------------


@dataclass
class FrameObs:
    """Frozen observations of one frame (undistorted pixels). Empty arrays when the frame has none."""

    frame: int
    cand: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))  # (n, 2)
    kappa: np.ndarray = field(default_factory=lambda: np.empty(0))  # (n,)
    env_u: np.ndarray = field(default_factory=lambda: np.empty(0))
    env_v: np.ndarray = field(default_factory=lambda: np.empty(0))
    ext: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))  # mask u-min, u-max; NaN = not evidence
    # per-candidate sigma scale (box truck 452 debugging, 2026-11-09: tyre positions near the left/right image edge
    # were off by 8-17 px, a lens-model limit); empty = all 1
    cand_s: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def s(self):
        return self.cand_s if len(self.cand_s) == len(self.cand) else np.ones(len(self.cand))


@dataclass
class Track:
    """One segment: consecutive frames, their observations, the camera and an initial pose."""

    frames: np.ndarray
    obs: list
    cam: object
    origin: np.ndarray  # local -> UTM offset (E, N)
    init_centre: np.ndarray  # (K, 2) local m, e.g. the smoothed M2 point
    init_psi_deg: np.ndarray  # (K,) compass deg, e.g. the reference heading
    cls: int = 5
    dt: float = 0.1


@dataclass(frozen=True)
class Config:
    subtype: str
    tandem: bool = False
    symmetric: bool = False  # frontness fits (8.4): equal insets and overhang priors
    hold_w: bool = False  # width held at the class prior (M7_W_HOLD_FT_PER_PX): the silhouette cannot resolve it


# --- priors (4.4) ----------------------------------------------------------------------------------------------


# statics layout: L_w, L_f, L_r, W, H, c, inset d | tandem spacing (tandem only)
IS, IT = 6, 7  # inset and tandem-spacing indices
NS2 = 7  # number of two-axle statics


def prior_model(cfg: Config):
    """(A, b, sig, lo, hi): prior residual z = (A theta - b) / sig per row, box bounds lo/hi (metres)."""
    p = S.M7_PRIORS_FT[cfg.subtype]
    mw, sw, lw, hw = (v * FT for v in p["Lw"])
    af, ar, sf, sr = p["af"], p["ar"], p["sf"] * FT, p["sr"] * FT
    if cfg.symmetric:
        af = ar = (af + ar) / 2
        sf = sr = max(sf, sr)
    mW, sW, lW, hW = (v * FT for v in (p["W"] if S.M7_FIX_WIDTH else S.M7_LOOSE_W_FT[cfg.subtype]))
    if S.M7_FIX_WIDTH:  # user decision 2026-11-08: width held at the class prior (rounded ends bias it low)
        sW = S.M7_FIXED_W_SD_FT * FT
    if cfg.hold_w:  # user decision 2026-11-10 ("Hold W at prior"): this track's silhouette cannot resolve W
        mW, sW = p["W"][0] * FT, S.M7_FIXED_W_SD_FT * FT
    mH, sH = (v * FT for v in p["H"])
    mc, sc, lc, hc = (v * FT for v in p["c"])
    mi = (p["i_f"] + p["i_r"]) / 2 * FT  # mean near-side tyre inset (fitted since 2026-11-09, user decision)
    si, li, hi_ = (v * FT for v in S.M7_INSET_SD_LO_HI_FT[cfg.subtype])
    if not S.M7_FIT_INSET:
        si = S.M7_FIXED_INSET_SD_FT * FT
    n = NS2 + 1 if cfg.tandem else NS2
    A = np.zeros((n, n))
    b = np.zeros(n)
    sig = np.zeros(n)
    A[0, 0], b[0], sig[0] = 1, mw, sw
    A[1, 1], A[1, 0], sig[1] = 1, -af, sf
    A[2, 2], A[2, 0], sig[2] = 1, -ar, sr
    A[3, 3], b[3], sig[3] = 1, mW, sW
    A[4, 4], b[4], sig[4] = 1, mH, sH
    A[5, 5], b[5], sig[5] = 1, mc, sc  # body ground clearance (fitted since the smoke run, design rev 7)
    A[IS, IS], b[IS], sig[IS] = 1, mi, si
    m = S.M7_MIN_DIM_FT * FT
    lo = [lw, m, m, lW, m, lc, li]
    hi = [hw, np.inf, np.inf, hW, np.inf, hc, hi_]
    if cfg.tandem:
        mt, st = (v * FT for v in S.M7_TANDEM_FT)
        A[IT, IT], b[IT], sig[IT] = 1, mt, st
        lo.append(S.M7_TANDEM_SPACING_BOUNDS_FT[0] * FT)
        hi.append(S.M7_TANDEM_SPACING_BOUNDS_FT[1] * FT)
    return A, b, sig, np.array(lo), np.array(hi)


def prior_cov(cfg: Config):
    """(mean, covariance) of the untruncated joint Gaussian of the statics."""
    A, b, sig, _, _ = prior_model(cfg)
    Ip = A.T @ np.diag(1 / sig**2) @ A
    return np.linalg.solve(A, b), np.linalg.inv(Ip)


def log_ZB(cfg: Config) -> float:
    """log P(theta in box B) under the joint prior (4.4); a constant per prior set and axle configuration."""
    mu, C = prior_cov(cfg)
    _, _, _, lo, hi = prior_model(cfg)
    sd = np.sqrt(np.diag(C))
    lo_f = np.where(np.isfinite(lo), lo, mu - 30 * sd)
    hi_f = np.where(np.isfinite(hi), hi, mu + 30 * sd)
    z = multivariate_normal(mu, C, allow_singular=False).cdf(hi_f, lower_limit=lo_f)
    return float(np.log(max(z, 1e-300)))


def edge_scale(u_img, width):
    """Sigma scale of tyre candidates by distance to the left/right image edge (distorted u): 1 from
    M7_EDGE_BAND_PX inwards, rising linearly to M7_EDGE_SIGMA_MULT at the edge."""
    dist = np.minimum(u_img, width - 1 - u_img)
    t = np.clip((S.M7_EDGE_BAND_PX - dist) / S.M7_EDGE_BAND_PX, 0.0, 1.0)
    return 1.0 + (S.M7_EDGE_SIGMA_MULT - 1.0) * t


def frame_obs_from_mask(frame, contours, box, others, dh, cam, cls, img_size=None) -> FrameObs:
    """FrameObs of one real row: accepted candidates (5.1), envelope columns outside the protrusion runs, and the
    end extents (dropped where the mask touches the image edge or a nearer box covers the extreme point)."""
    from . import candidates as mc

    img_size = S.IMG_SIZE if img_size is None else img_size
    cand, _, dbg = mc.extract(contours, box, others, dh, cam, cls, img_size)
    if dbg is None:
        return FrameObs(int(frame))
    acc = cand[cand["accepted"].astype(bool)] if len(cand) else cand
    c_uv = acc[["u_und", "v_und"]].to_numpy(float) if len(acc) else np.empty((0, 2))
    kap = acc["kappa"].to_numpy(float) if len(acc) else np.empty(0)
    c_s = edge_scale(acc["u"].to_numpy(float), img_size[0]) if len(acc) else np.empty(0)
    if len(acc):  # the frozen price at the candidate's own sigma; the gate in pixels is unchanged
        kap = rho(acc["gate_px"].to_numpy(float) / (S.M7_SIGMA_WHEEL_PX * c_s))
    body = dbg["depth"] <= dbg["thr"]
    pts = np.vstack(contours)
    W = img_size[0]
    # bottom extents (design rev 7): the u-range of the lower-outline columns that lie on or near the pose-free body
    # line (side and end bottom edges). Roof-only columns, where the lowest mask pixel is far above the line, are
    # excluded: rounded roofs made full-mask extents pull widths to their bound. Predicted by the near-side bottom corners (2026-11-09).
    near = dbg["depth"] >= -S.M7_EXT_ABOVE_THR * dbg["thr"]
    if not near.any():
        return FrameObs(int(frame), c_uv, kap, dbg["uv"][body, 0], dbg["uv"][body, 1], cand_s=c_s)
    uvn, on = dbg["uv"][near], dbg["outline"][near]
    i_lo, i_hi = int(np.argmin(uvn[:, 0])), int(np.argmax(uvn[:, 0]))
    ext = np.array([uvn[i_lo, 0], uvn[i_hi, 0]])
    bottom = box[1] + box[3]
    sides = (((pts[:, 0] <= S.EDGE_PX).any(), on[i_lo]), ((pts[:, 0] >= W - 1 - S.EDGE_PX).any(), on[i_hi]))
    for j, (side_edge, p) in enumerate(sides):
        covered = any(oy + oh > bottom and ox <= p[0] <= ox + ow and oy <= p[1] <= oy + oh
                      for ox, oy, ow, oh in np.asarray(others, float).reshape(-1, 4))  # fmt: skip
        if side_edge or covered:
            ext[j] = np.nan
    return FrameObs(int(frame), c_uv, kap, dbg["uv"][body, 0], dbg["uv"][body, 1], ext, c_s)


# --- the problem -----------------------------------------------------------------------------------------------


def unit(psi):
    return np.stack([np.sin(psi), np.cos(psi)], axis=-1)


def left(psi):
    return np.stack([-np.cos(psi), np.sin(psi)], axis=-1)


class Problem:
    """Geometry, association, residuals and objective of one Track under one Config."""

    def __init__(self, track: Track, cfg: Config):
        self.t, self.cfg = track, cfg
        self.K = K = len(track.frames)
        p = S.M7_PRIORS_FT[cfg.subtype]
        # front/rear inset difference stays at the prior set's; the mean inset is the fitted static th[IS]
        self.di_half = 0.0 if cfg.symmetric else (p["i_f"] - p["i_r"]) / 2 * FT
        self.r_wheel = p["wheel_r"] * FT  # tyre radius for the lowest-point observation model
        self.A, self.b, self.sig, self.lo_s, self.hi_s = prior_model(cfg)
        self.ns = len(self.sig)
        self.n = 3 * K + self.ns
        self.Cg = np.asarray(track.cam.C[:2], float) + np.asarray(track.cam.origin, float) - np.asarray(track.origin, float)
        # targets: near/far x front/rear(s); group ids 0 = front, 1/2 = rear(s)
        rears = [1, 2] if cfg.tandem else [1]
        self.far = bool(S.M7_FAR_TARGETS)
        # (K,) near side held fixed during one round of `fit` (M7_SIDE_FREEZE); None = from the pose
        self.side_fixed = None
        self.tg_group = np.array([0] + rears + ([0] + rears if self.far else []))
        self.tg_far = np.array([False] * (1 + len(rears)) + ([True] * (1 + len(rears)) if self.far else []))
        self.tg_sigma = np.where(self.tg_far, S.M7_SIGMA_FAR_PX, S.M7_SIGMA_WHEEL_PX)
        self.T = len(self.tg_group)
        # frozen observation arrays
        M = S.M7_ENV_MAX_COLS
        self.env_u = np.full((K, M), np.nan)
        self.env_v = np.full((K, M), np.nan)
        self.env_wg = np.zeros(K)
        self.ext = np.full((K, 2), np.nan)
        for k, o in enumerate(track.obs):
            eu, ev = np.asarray(o.env_u, float), np.asarray(o.env_v, float)
            m = len(eu)
            if m:
                idx = np.linspace(0, m - 1, min(m, M)).round().astype(int)
                self.env_u[k, : len(idx)] = eu[idx]
                self.env_v[k, : len(idx)] = ev[idx]
                self.env_wg[k] = min(1.0, S.M7_ENV_EFF_N / len(idx))
            self.ext[k] = o.ext if S.M7_USE_EXTENTS else np.nan
        self.env_ok = np.isfinite(self.env_u)
        self.ext_ok = np.isfinite(self.ext)
        self.sparsity = self._sparsity()

    # x layout: R (K, 2) | psi (K) | statics
    def unpack(self, x):
        K = self.K
        return x[: 2 * K].reshape(K, 2), x[2 * K : 3 * K], x[3 * K :]

    def bounds(self):
        lo = np.r_[np.full(3 * self.K, -np.inf), self.lo_s]
        hi = np.r_[np.full(3 * self.K, np.inf), self.hi_s]
        return lo, hi

    MIN_DEPTH_M = 0.5  # points closer to (or behind) the image plane are clamped there, keeping residuals finite

    def _proj(self, P3):
        """Local (..., 3) -> undistorted pixels (..., 2). Depth is clamped at MIN_DEPTH_M: a bus passing under the
        pole can put box corners behind the camera mid-optimisation; those frames are image-edge cases anyway."""
        cam = self.t.cam
        sh = P3.shape[:-1]
        Q = P3.reshape(-1, 3) + np.r_[self.t.origin - cam.origin, 0.0]
        Xc = (Q - cam.C) @ cam.R.T
        z = np.maximum(Xc[:, 2], self.MIN_DEPTH_M)
        uv = np.column_stack([cam.f * Xc[:, 0] / z + cam.cx, cam.f * Xc[:, 1] / z + cam.cy])
        return uv.reshape(*sh, 2)

    def _tyre_lowest(self, contact, f):
        """Image of what a tyre candidate measures (box truck 452 debugging, 2026-11-09): the lowest image point of
        the tyre's circle (radius = the prior set's wheel_r, vertical plane along the body heading, touching the
        ground at `contact`), not the contact itself. Near the camera the two differ by tens of pixels: on 452 the
        front tyre's lowest point ran 11-36 px ahead of its contact while the truck turned towards the camera.
        Closed form: on X(phi) = Cw + r cos(phi) f + r sin(phi) z the image row v = (a + b cos + c sin) /
        (d + e cos + g sin) is stationary where (ae - bd) sin + (cd - ag) cos + (ce - bg) = 0; the larger-v root is
        the lowest point. contact (K, T, 2), f (K, 2) -> (K, T, 2) undistorted pixels."""
        cam = self.t.cam
        r = self.r_wheel
        K, T = contact.shape[:2]
        off = np.r_[self.t.origin - cam.origin, 0.0]
        Cw = np.concatenate([contact, np.full((K, T, 1), r)], -1) + off - cam.C  # circle centre, camera-centred
        X0 = Cw @ cam.R.T  # (K, T, 3)
        X1 = np.broadcast_to((r * np.c_[f, np.zeros(K)] @ cam.R.T)[:, None], X0.shape)
        X2 = np.broadcast_to(r * np.array([0.0, 0.0, 1.0]) @ cam.R.T, X0.shape)
        a, b, c = X0[..., 1], X1[..., 1], X2[..., 1]
        d, e, g = X0[..., 2], X1[..., 2], X2[..., 2]
        A, B, C0 = a * e - b * d, c * d - a * g, c * e - b * g
        rho_ = np.hypot(A, B)
        s0 = np.arcsin(np.clip(-C0 / np.maximum(rho_, 1e-12), -1.0, 1.0))
        dl = np.arctan2(B, A)
        best_uv, best_v = None, None
        for phi in (s0 - dl, np.pi - s0 - dl):
            Xc = X0 + np.cos(phi)[..., None] * X1 + np.sin(phi)[..., None] * X2
            z = np.maximum(Xc[..., 2], self.MIN_DEPTH_M)
            uv = np.stack([cam.f * Xc[..., 0] / z + cam.cx, cam.f * Xc[..., 1] / z + cam.cy], -1)
            if best_uv is None:
                best_uv, best_v = uv, uv[..., 1]
            else:
                pick = uv[..., 1] > best_v
                best_uv = np.where(pick[..., None], uv, best_uv)
                best_v = np.maximum(best_v, uv[..., 1])
        return best_uv

    def geometry(self, x):
        R, psi, th = self.unpack(x)
        Lw, Lf, Lr, W, H, c = th[:6]
        f, nl = unit(psi), left(psi)
        if self.side_fixed is not None:
            side = self.side_fixed
        else:
            side = np.sign(np.einsum("kj,kj->k", self.Cg - R, nl))
            side[side == 0] = 1.0
        nn = side[:, None] * nl
        A = R + Lw * f
        offs = [0.0] if not self.cfg.tandem else [-th[IT] / 2, th[IT] / 2]
        i_f, i_r = th[IS] + self.di_half, th[IS] - self.di_half
        pts = [A + (W / 2 - i_f) * nn] + [R + o * f + (W / 2 - i_r) * nn for o in offs]
        if self.far:
            pts += [A - (W / 2 - i_f) * nn] + [R + o * f - (W / 2 - i_r) * nn for o in offs]
        tg = np.stack(pts, axis=1)  # (K, T, 2)
        if S.M7_TYRE_LOWEST_POINT:
            tg_px = self._tyre_lowest(tg, f)
        else:
            tg_px = self._proj(np.concatenate([tg, np.zeros(tg.shape[:2] + (1,))], axis=-1))
        fe, re_ = R + (Lw + Lf) * f, R - Lr * f
        base = np.stack([fe + W / 2 * nl, fe - W / 2 * nl, re_ - W / 2 * nl, re_ + W / 2 * nl], axis=1)  # (K, 4, 2)
        box = np.concatenate([np.concatenate([base, np.full((self.K, 4, 1), c)], -1),
                              np.concatenate([base, np.full((self.K, 4, 1), H)], -1)], axis=1)  # fmt: skip
        bp = self._proj(box)  # (K, 8, 2)
        umin, umax = bp[..., 0].min(1), bp[..., 0].max(1)
        # extents' prediction (2026-11-09, user decision): the u-range of the two NEAR-side bottom corners (z = c).
        # The far bottom corners carried width: rounded vehicle ends hide them, and a box matched the shorter
        # outline only by narrowing (widths sat on every lower bound tried). Corners 0, 3 are on the left (+nl).
        near_c = np.where((side > 0)[:, None], np.array([0, 3]), np.array([1, 2]))  # (K, 2)
        bu = np.take_along_axis(bp[..., 0], near_c, axis=1)
        bmin, bmax = bu.min(1), bu.max(1)
        # columns outside the box's u-span are evaluated a margin inside it (2026-11-09, bus 417 debugging). At the
        # extreme u the silhouette narrows to one vertex, so the envelope there steps when two vertices swap order
        # (17.7 px on 417, pinning W at its bound); a few pixels inside, the lower envelope is continuous.
        m = np.minimum(S.M7_ENV_CLAMP_PX, (umax - umin) / 4)[:, None] if S.M7_ENV_CLAMP_PX > 0 else 1e-6
        u = np.clip(self.env_u, umin[:, None] + m, umax[:, None] - m)
        e = np.array(EDGES)
        p, q = bp[:, e[:, 0]], bp[:, e[:, 1]]  # (K, 12, 2)
        du = q[..., 0] - p[..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (u[:, None, :] - p[..., 0][..., None]) / du[..., None]  # (K, 12, M)
            v = p[..., 1][..., None] + t * (q[..., 1] - p[..., 1])[..., None]
        ok = (t >= 0) & (t <= 1) & (np.abs(du) > 1e-9)[..., None]
        vb = np.where(ok, v, -np.inf).max(axis=1)  # (K, M) lower envelope
        return dict(R=R, psi=psi, th=th, f=f, nl=nl, side=side, tg_px=tg_px, env_pred=vb,
                    ext_pred=np.stack([bmin, bmax], 1), box_px=bp)  # fmt: skip

    # --- association (5.2) -------------------------------------------------------------------------------------
    def associate(self, g, use_wheels=True):
        """Per frame minimum of the wheel part of F: arrays (k, cand, target) of matches."""
        mk, mc, mt = [], [], []
        if not use_wheels:
            return np.array(mk, int), np.array(mc, int), np.array(mt, int)
        for k, o in enumerate(self.t.obs):
            n = len(o.cand)
            if not n:
                continue
            d = np.linalg.norm(o.cand[:, None, :] - g["tg_px"][k][None], axis=2)  # (n, T)
            cost = rho(d / (self.tg_sigma[None] * o.s[:, None]))
            kap = o.kappa[:, None]
            C = np.full((n, self.T + n), BIG)
            C[:, : self.T] = np.where(cost < kap, cost, BIG)
            C[np.arange(n), self.T + np.arange(n)] = o.kappa
            r, c = linear_sum_assignment(C)
            for i, j in zip(r, c):
                if j < self.T:
                    mk.append(k), mc.append(i), mt.append(j)
        return np.array(mk, int), np.array(mc, int), np.array(mt, int)

    # --- objective pieces --------------------------------------------------------------------------------------
    def motion(self, g):
        """Motion residuals z (already divided by sigma): slip, acc (2 each), yaw, curvature."""
        R, psi, th = g["R"], g["psi"], g["th"]
        if self.K < 2:
            return np.empty(0), {}
        d = np.diff(R, axis=0)
        pm = (psi[1:] + psi[:-1]) / 2
        slip = np.einsum("kj,kj->k", d, left(pm)) / (S.M7_SIGMA_SLIP_FT * FT)
        s = np.einsum("kj,kj->k", d, unit(pm))
        lim = np.abs(s) * np.tan(np.radians(S.M7_STEER_MAX_DEG)) / th[0] + np.radians(S.M7_CURV_EPS_DEG)
        curv = np.maximum(0.0, np.abs(np.diff(psi)) - lim) / np.radians(S.M7_SIGMA_CURV_DEG)
        if self.K >= 3:
            acc = (R[2:] - 2 * R[1:-1] + R[:-2]).ravel() / (S.M7_SIGMA_ACC_FT * FT)
            yaw = (psi[2:] - 2 * psi[1:-1] + psi[:-2]) / np.radians(S.M7_SIGMA_YAW_DEG)
        else:
            acc, yaw = np.empty(0), np.empty(0)
        return np.r_[slip, acc, yaw, curv], dict(slip=slip, acc=acc, yaw=yaw, curv=curv, s=s)

    def prior_z(self, th):
        return (self.A @ th - self.b) / self.sig

    def obs_z(self, g, assoc):
        """Observation z: wheels (m, 2), envelope (K, M) NaN where absent, extents (K, 2) NaN where absent."""
        mk, mc, mt = assoc
        if len(mk):
            cand = np.array([self.t.obs[k].cand[c] for k, c in zip(mk, mc)])
            sc = np.array([self.t.obs[k].s[c] for k, c in zip(mk, mc)])
            zw = (g["tg_px"][mk, mt] - cand) / (self.tg_sigma[mt] * sc)[:, None]
        else:
            zw = np.empty((0, 2))
        ze = np.where(self.env_ok, (self.env_v - g["env_pred"]) / S.M7_SIGMA_ENV_PX, np.nan)
        zx = np.where(self.ext_ok, (self.ext - g["ext_pred"]) / S.M7_SIGMA_EXT_PX, np.nan)
        return zw, ze, zx

    def F(self, x, assoc=None, use_wheels=True, parts=False):
        g = self.geometry(x)
        if assoc is None:
            assoc = self.associate(g, use_wheels)
        zw, ze, zx = self.obs_z(g, assoc)
        mk, mc, _ = assoc
        f_w = float(rho(np.linalg.norm(zw, axis=1)).sum()) if len(zw) else 0.0
        if use_wheels:  # every unmatched candidate pays its frozen price
            matched = set(zip(mk.tolist(), mc.tolist()))
            f_w += float(sum(o.kappa[i] for k, o in enumerate(self.t.obs) for i in range(len(o.cand))
                             if (k, i) not in matched))  # fmt: skip
        f_e = float(np.nansum(self.env_wg[:, None] * rho(ze)))
        f_x = float(np.nansum(rho(zx)))
        zm, _ = self.motion(g)
        f_m = 0.5 * float(zm @ zm)
        zp = self.prior_z(g["th"])
        f_p = 0.5 * float(zp @ zp)
        tot = f_w + f_e + f_x + f_m + f_p
        if parts:
            return tot, dict(wheel=f_w, env=f_e, ext=f_x, motion=f_m, prior=f_p, assoc=assoc, geom=g)
        return tot

    def residuals(self, x, assoc, wts):
        """IRLS surrogate residual vector r (F_surrogate = 1/2 r.r) for a fixed association and weights."""
        g = self.geometry(x)
        zw, ze, zx = self.obs_z(g, assoc)
        ww, we, wx = wts
        rw = (np.sqrt(2 * ww)[:, None] * zw).ravel()
        re_ = np.sqrt(2 * we * self.env_wg[:, None]) * np.nan_to_num(ze)
        rx = np.sqrt(2 * wx) * np.nan_to_num(zx)
        zm, _ = self.motion(g)
        return np.r_[rw, re_.ravel(), rx.ravel(), zm, self.prior_z(g["th"])]

    def raw_residuals(self, x, assoc):
        """Unweighted z vector: wheel pairs, envelope, extents (0 where absent), motion, priors. With `loss(assoc)`
        least_squares minimises exactly F minus the rejection constants: 1/2 sum rho_i(z_i^2)."""
        g = self.geometry(x)
        zw, ze, zx = self.obs_z(g, assoc)
        zm, _ = self.motion(g)
        return np.r_[zw.ravel(), np.nan_to_num(ze).ravel(), np.nan_to_num(zx).ravel(), zm, self.prior_z(g["th"])]

    def loss(self, assoc):
        """Callable loss for least_squares (design 7.4): Cauchy on observations (wheel pairs share one
        log(1 + |z|^2)), weighted by the envelope down-weighting; quadratic on motion and prior rows."""
        nw = 2 * len(assoc[0])
        wobs = np.r_[np.ones(nw), (self.env_wg[:, None] * self.env_ok).ravel(), self.ext_ok.ravel().astype(float)]
        nobs = len(wobs)

        def fn(z):
            out = np.empty((3, len(z)))
            out[0], out[1], out[2] = z, 1.0, 0.0  # quadratic rows: 1/2 * z
            s = z[:nobs].copy()
            if nw:
                pair = s[0:nw:2] + s[1:nw:2]
                s[0:nw:2], s[1:nw:2] = pair, pair
            r0 = 2 * wobs * np.log1p(s)
            r1 = 2 * wobs / (1 + s)
            r2 = -2 * wobs / (1 + s) ** 2
            if nw:
                r0[1:nw:2] = 0.0  # the pair's log term is counted once; both rows keep rho' (gradient is exact)
            out[0, :nobs], out[1, :nobs], out[2, :nobs] = r0, r1, r2
            return out

        return fn

    def weights(self, x, assoc):
        g = self.geometry(x)
        zw, ze, zx = self.obs_z(g, assoc)
        ww = 1 / (1 + (zw**2).sum(1)) if len(zw) else np.empty(0)
        we = np.where(self.env_ok, 1 / (1 + np.nan_to_num(ze) ** 2), 0.0)
        wx = np.where(self.ext_ok, 1 / (1 + np.nan_to_num(zx) ** 2), 0.0)
        return ww, we, wx

    def _sparsity(self):
        return None  # built per association (rows depend on the matches)

    def jac_sparsity(self, assoc):
        K, ns = self.K, self.ns
        mk = assoc[0]
        M = self.env_u.shape[1]
        rows = 2 * len(mk) + K * M + 2 * K
        nm = (K - 1) + 2 * max(K - 2, 0) + max(K - 2, 0) + (K - 1)
        sp = lil_matrix((rows + nm + ns, self.n), dtype=int)
        st = list(range(3 * K, 3 * K + ns))

        def frame_cols(k):
            return [2 * k, 2 * k + 1, 2 * K + k]

        r = 0
        for k in mk:
            for _ in range(2):
                sp[r, frame_cols(k) + st] = 1
                r += 1
        for k in range(K):
            for _ in range(M):
                sp[r, frame_cols(k) + st] = 1
                r += 1
        for k in range(K):
            for _ in range(2):
                sp[r, frame_cols(k) + st] = 1
                r += 1
        for k in range(K - 1):  # slip
            sp[r, frame_cols(k) + frame_cols(k + 1)] = 1
            r += 1
        for k in range(K - 2):  # acc e, n
            for j in range(2):
                sp[r, [2 * (k + i) + j for i in range(3)]] = 1
                r += 1
        for k in range(K - 2):  # yaw
            sp[r, [2 * K + k + i for i in range(3)]] = 1
            r += 1
        for k in range(K - 1):  # curvature (needs L_w)
            sp[r, frame_cols(k) + frame_cols(k + 1) + [3 * K]] = 1
            r += 1
        for i in range(ns):
            sp[r, st] = 1
            r += 1
        return sp


# --- fitting (7.2, 7.3) ----------------------------------------------------------------------------------------


def initial_x(prob: Problem, Lw_ft=None, flipped_from=None):
    """x0 from the track's initial centre and heading (7.2), statics at the prior means (L_w optionally set)."""
    t = prob.t
    mu, _ = prior_cov(prob.cfg)
    th = mu.copy()
    if Lw_ft is not None:
        th[0] = Lw_ft * FT
        th[1], th[2] = prob.A[1, 0] * -th[0], prob.A[2, 0] * -th[0]
    th = np.clip(th, prob.lo_s + 1e-6, prob.hi_s - 1e-6)
    psi = np.radians(np.asarray(t.init_psi_deg, float))
    for k in range(1, len(psi)):  # a motion heading flips 180 deg when the vehicle reverses: the body does not
        if abs((psi[k] - psi[k - 1] + np.pi) % (2 * np.pi) - np.pi) > np.pi / 2:
            psi[k:] += np.pi
    psi = np.unwrap(psi)
    R = np.asarray(t.init_centre, float) - ((th[0] + th[1] - th[2]) / 2)[None] * unit(psi)
    return np.r_[R.ravel(), psi, th]


def _ground(track: Track, uv):
    """Undistorted pixels (n, 2) -> local ground points (n, 2) (z = 0)."""
    o, d = track.cam.ray(uv)
    t = -o[2] / d[:, 2]
    return o[:2] + t[:, None] * d[:, :2] - np.asarray(track.origin, float)


def pair_wheelbase(prob: Problem):
    """Wheelbase estimate (m) from tyre-candidate pairs, or None (bus 16 debugging, 2026-11-09). Where two candidates
    keep a steady ground distance in several frames, that distance is the wheelbase. The prior-pose starts build the
    pose from the M2 point, which is biased when the vehicle is cut by the image edge or close to the camera; on bus 16
    they ended in a 20.8 ft minimum while its tyre candidates stayed 25 ft apart."""
    t = prob.t
    lo, hi = prob.lo_s[0], prob.hi_s[0]
    d_all, k_all = [], []
    for k, o in enumerate(t.obs):
        if len(o.cand) < 2:
            continue
        G = _ground(t, o.cand)
        for i in range(len(G)):
            for j in range(i + 1, len(G)):
                d = float(np.linalg.norm(G[i] - G[j]))
                if lo <= d <= hi:
                    d_all.append(d)
                    k_all.append(k)
    if not d_all:
        return None
    d_all, k_all = np.array(d_all), np.array(k_all)
    tol = S.M7_PAIR_TOL_FT * FT
    # the mode: the distance supported by the most distinct frames within +- tol
    support = [len(set(k_all[np.abs(d_all - d) <= tol].tolist())) for d in d_all]
    best = int(np.argmax(support))
    if support[best] < S.M7_GROUP_MIN_FRAMES:
        return None
    return float(np.median(d_all[np.abs(d_all - d_all[best]) <= tol]))


def set_wheelbase(prob: Problem, x, Lw, keep: str):
    """x with L_w replaced, the front ('front') or the rear ('rear') axle kept in place; overhangs unchanged."""
    R, psi, th = prob.unpack(x)
    th = th.copy()
    if keep == "front":
        R = R - (Lw - th[0]) * unit(psi)
    th[0] = np.clip(Lw, prob.lo_s[0] + 1e-6, prob.hi_s[0] - 1e-6)
    return np.r_[R.ravel(), psi, th]


@dataclass
class Fit:
    cfg: Config
    x: np.ndarray
    F: float
    parts: dict
    assoc: tuple
    converged: bool
    iters: int
    start: str = ""
    E: float = np.nan
    cov: np.ndarray | None = None
    info: dict = field(default_factory=dict)


def _solve(prob: Problem, x, assoc, nfev):
    """Minimise F for a fixed association: the robust loss is exact (`Problem.loss`), so no IRLS reweighting is
    needed inside a round (implementation note: frozen IRLS weights converged far too slowly on real masks)."""
    lo, hi = prob.bounds()
    x = np.clip(x, lo + 1e-9, hi - 1e-9)
    sol = least_squares(prob.raw_residuals, x, args=(assoc,), loss=prob.loss(assoc),
                        jac_sparsity=prob.jac_sparsity(assoc), bounds=(lo, hi), method="trf", x_scale="jac",
                        max_nfev=nfev)  # fmt: skip
    return sol.x, sol.status > 0


def _irls_solve(prob: Problem, x, assoc):
    """One IRLS round: weights 1 / (1 + z^2) from x, weighted least squares (a majorise-minimise step for F)."""
    lo, hi = prob.bounds()
    x = np.clip(x, lo + 1e-9, hi - 1e-9)
    sol = least_squares(prob.residuals, x, args=(assoc, prob.weights(x, assoc)), jac_sparsity=prob.jac_sparsity(assoc),
                        bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=S.M7_IRLS_NFEV)  # fmt: skip
    return sol.x, sol.status > 0


def fit(prob: Problem, x0, start="", outer=None) -> Fit:
    """Section 7.3: shape stage (no wheels), then rounds of association + exact robust solve, backtracking on F."""
    outer = S.M7_OUTER_ITERS if outer is None else outer
    x = x0.copy()
    empty = (np.array([], int),) * 3
    # stage 1: silhouette + motion + priors by IRLS (weights from the current pose: a redescending loss started far
    # away can ignore the silhouette and collapse the body; a quadratic one is dragged by outliers)
    for _ in range(3):
        x, _ = _irls_solve(prob, x, empty)
    F0, parts = prob.F(x, parts=True)
    assoc = parts["assoc"]
    converged, it = False, 0
    side = None
    for it in range(1, outer + 1):
        if S.M7_SIDE_FREEZE:
            # (2026-11-09, width debugging) the near side switches where the camera crosses the vehicle's
            # centreline, and F jumps there: 5 of the 12 display solves whose side switches ended with one frame
            # pinned exactly on the switch (|offset| < 0.01 ft), where the profile gradient is not zero and W sat
            # on its bound. Like the association, the side is decided from the pose at the start of each round
            # and held fixed inside it, so every inner solve is smooth.
            prob.side_fixed = None
            side_new = prob.geometry(x)["side"].copy()
            if side is not None and not np.array_equal(side, side_new):
                prob.side_fixed = side_new
                F0, parts = prob.F(x, parts=True)
                assoc = parts["assoc"]
            side = side_new
            prob.side_fixed = side
        # the first rounds are IRLS (robust basin), the later ones minimise F exactly for the association
        if it <= S.M7_IRLS_ROUNDS:
            x1, _ = _irls_solve(prob, x, assoc)
        else:
            x1, _ = _solve(prob, x, assoc, S.M7_INNER_NFEV)
        F1, p1 = prob.F(x1, parts=True)
        step = 1.0
        while F1 > F0 and step > 0.125:  # backtrack on F (7.3)
            step /= 2
            xt = x + step * (x1 - x)
            F1, p1 = prob.F(xt, parts=True)
            x1 = xt
        if F1 > F0:
            converged = True  # no descent left: x is the mode
            break
        same = _same(assoc, p1["assoc"])
        if S.M7_SIDE_FREEZE:
            prob.side_fixed = None
            same = same and np.array_equal(side, prob.geometry(x1)["side"])
            prob.side_fixed = side
        rel = (F0 - F1) / max(1.0, abs(F0))
        x, F0, parts, assoc = x1, F1, p1, p1["assoc"]
        if (same and rel < S.M7_TOL) or rel < S.M7_TOL / 10:  # a benign association flip-flop at the optimum
            converged = True
            break
    if prob.side_fixed is not None:  # leave the problem with the pose's own side (every later use of a Fit)
        prob.side_fixed = None
        F0, parts = prob.F(x, assoc=assoc, parts=True)
    return Fit(prob.cfg, x, F0, parts, assoc, converged, it, start)


def _same(a, b):
    return all(len(u) == len(v) and np.array_equal(u, v) for u, v in zip(a, b))


# --- information, covariance, evidence (7.4, 8.2) --------------------------------------------------------------


def information(prob: Problem, ft: Fit):
    """(I, active, groups): Gauss-Newton information at the mode with IRLS weights, observation groups scaled by
    their robust noise factor s_g >= 1; statics at a bound removed (`active` False)."""
    x, assoc = ft.x, ft.assoc
    wts = prob.weights(x, assoc)
    lo, hi = prob.bounds()
    J = approx_derivative(prob.residuals, x, args=(assoc, wts), sparsity=prob.jac_sparsity(assoc), bounds=(lo, hi))
    J = J.toarray() if hasattr(J, "toarray") else np.asarray(J)
    g = prob.geometry(x)
    zw, ze, zx = prob.obs_z(g, assoc)
    nw = 2 * len(assoc[0])
    ne = ze.size
    nx = zx.size
    scales = {}
    for name, z, a, b in (("wheel", zw.ravel(), 0, nw), ("env", ze[prob.env_ok], nw, nw + ne),
                          ("ext", zx[prob.ext_ok], nw + ne, nw + ne + nx)):  # fmt: skip
        z = z[np.isfinite(z)]
        s = max(1.0, 1.4826 * float(np.median(np.abs(z - np.median(z))))) if len(z) >= 5 else 1.0
        scales[name] = s
        J[a:b] /= s
    I = J.T @ J
    th = prob.unpack(x)[2]
    fin_lo, fin_hi = np.isfinite(prob.lo_s), np.isfinite(prob.hi_s)
    # (2026-11-09) absolute tolerance: the bounded solver stops slightly inside a bound (widths 8.7976-8.7990 ft against
    # 8.8), which the old relative 1e-4 test (about 0.0009 ft) missed
    tol = S.M7_BOUND_TOL_FT * FT
    at_lo = fin_lo & (np.abs(th - np.where(fin_lo, prob.lo_s, 0)) <= tol)
    at_hi = fin_hi & (np.abs(th - np.where(fin_hi, prob.hi_s, 0)) <= tol)
    active = np.r_[np.ones(3 * prob.K, bool), ~(at_lo | at_hi)]
    return I, active, scales


def finish(prob: Problem, ft: Fit) -> Fit:
    """Covariance, evidence E and size information for a converged fit."""
    I, active, scales = information(prob, ft)
    Ia = I[np.ix_(active, active)]
    info = {"noise_scale": scales, "at_bound": (~active[3 * prob.K :]).tolist()}
    try:
        L = np.linalg.cholesky(Ia)
        cov_a = np.linalg.inv(Ia)
    except np.linalg.LinAlgError:
        ft.info = {**info, "ill_conditioned": True}
        ft.E = np.nan
        return ft
    cov = np.full((prob.n, prob.n), np.nan)
    cov[np.ix_(active, active)] = cov_a
    ft.cov = cov
    s_act = active[3 * prob.K :]
    Css = cov_a[-int(s_act.sum()) :, -int(s_act.sum()) :] if s_act.any() else np.zeros((0, 0))
    logdet_static = -np.linalg.slogdet(Css)[1] if len(Css) else 0.0
    ft.E = ft.F + float(np.log(prob.sig).sum()) + log_ZB(prob.cfg) + 0.5 * logdet_static
    ft.info = {**info, "ill_conditioned": False, "logdet_I_static": logdet_static, "chol_ok": L is not None}
    return ft


# --- per-solve results (7.5, 8) --------------------------------------------------------------------------------


def size_sources(prob: Problem, ft: Fit) -> dict:
    """Section 8.3: posterior / marginal-prior sd ratios and length / width sources."""
    _, Cp = prior_cov(prob.cfg)
    th = prob.unpack(ft.x)[2]
    at = np.array(ft.info.get("at_bound", [False] * len(th)))
    Cs = ft.cov[3 * prob.K :, 3 * prob.K :] if ft.cov is not None else np.full((len(th),) * 2, np.nan)
    out = {}
    idx = {"L": [0, 1, 2], "W": [3], "Lw": [0], "Lf": [1], "Lr": [2], "H": [4], "c": [5], "inset": [IS]}
    grads = {k: np.isin(np.arange(len(th)), v).astype(float) for k, v in idx.items()}
    for name, g5 in grads.items():
        g = g5
        sp = float(np.sqrt(g @ Cp @ g))
        sq = float(np.sqrt(g @ Cs @ g)) if np.isfinite(Cs[np.ix_(g != 0, g != 0)]).all() else np.nan
        ratio = sq / sp if np.isfinite(sq) else np.nan
        bound = bool(at[g != 0].any())
        if name in ("L", "W"):
            if bound or not np.isfinite(ratio) or ratio > S.M7_PRIOR_ONLY_RATIO:
                src = "prior"
            elif ratio <= S.M7_PRIOR_DOM_RATIO and sq <= S.M7_MEASURED_SD_FT[name] * FT:
                src = "measured"
            else:
                src = "partly_prior"
            out[f"{name}_source"] = src
        out[f"{name}_sd_ft"] = sq / FT if np.isfinite(sq) else np.nan
        out[f"{name}_prior_sd_ft"] = sp / FT
        out[f"{name}_ratio"] = ratio
    ls, ws = out["L_source"], out["W_source"]
    out["dims_source"] = ("m7_measured" if ls == ws == "measured" else
                          "m7_prior" if ls == ws == "prior" else "m7_partly_prior")  # fmt: skip
    return out


def frame_table(prob: Problem, ft: Fit) -> pd.DataFrame:
    """Per-frame pose, body centre, corners, support and uncertainty (sections 8.1, 8.2)."""
    g = prob.geometry(ft.x)
    R, psi, th = g["R"], g["psi"], g["th"]
    Lw, Lf, Lr, W = th[:4]
    K = prob.K
    f, nl = g["f"], g["nl"]
    h = (Lw + Lf - Lr) / 2
    centre = R + h * f
    mk, mc, mt = ft.assoc
    n_in = np.bincount(mk, minlength=K)
    front = np.zeros(K, bool)
    rear = np.zeros(K, bool)
    for k, t in zip(mk, mt):
        (front if prob.tg_group[t] == 0 else rear)[k] = True
    n_c = np.array([len(o.cand) for o in prob.t.obs])
    env_n = prob.env_ok.sum(1)
    sil = (env_n >= S.M7_MIN_ENV_COLS) & (prob.ext_ok.any(1) | (not S.M7_USE_EXTENTS))
    support = np.where(n_in >= 2, "wheels2", np.where(n_in == 1, np.where(front, "wheel_front", "wheel_rear"),
                                                      np.where(sil, "silhouette", "kinematic")))  # fmt: skip
    cov = ft.cov
    sd_al, sd_ac, sd_h = np.full(K, np.nan), np.full(K, np.nan), np.full(K, np.nan)
    cc = np.full((K, 3), np.nan)
    if cov is not None:
        s0 = 3 * K
        for k in range(K):
            idx = [2 * k, 2 * k + 1, 2 * K + k, s0, s0 + 1, s0 + 2, s0 + 3]
            C = cov[np.ix_(idx, idx)]
            if not np.isfinite(C).all():
                continue
            df = np.array([np.cos(psi[k]), -np.sin(psi[k])])
            dn = np.array([np.sin(psi[k]), np.cos(psi[k])]) * g["side"][k]
            Gc = np.column_stack([np.eye(2), h * df, f[k] / 2, f[k] / 2, -f[k] / 2, np.zeros(2)])
            cm = Gc @ C @ Gc.T
            cc[k] = cm[0, 0], cm[0, 1], cm[1, 1]
            nn = g["side"][k] * nl[k]
            Gq = np.column_stack([np.eye(2), (Lw + Lf) * df + W / 2 * dn, f[k], f[k], np.zeros(2), nn / 2])
            cq = Gq @ C @ Gq.T
            sd_al[k] = np.sqrt(f[k] @ cq @ f[k]) / FT
            sd_ac[k] = np.sqrt(nn @ cq @ nn) / FT
            sd_h[k] = np.degrees(np.sqrt(C[2, 2]))
    corners = np.stack([centre + (Lw + Lf + Lr) / 2 * f + W / 2 * nl, centre + (Lw + Lf + Lr) / 2 * f - W / 2 * nl,
                        centre - (Lw + Lf + Lr) / 2 * f - W / 2 * nl, centre - (Lw + Lf + Lr) / 2 * f + W / 2 * nl],
                       axis=1)  # fmt: skip
    out = pd.DataFrame({"frame": prob.t.frames, "R_e": R[:, 0], "R_n": R[:, 1],
                        "psi_deg": np.degrees(psi) % 360, "e_nat": centre[:, 0], "n_nat": centre[:, 1],
                        "m7_support": support, "m7_n_inliers": n_in, "m7_n_outliers": n_c - n_in,
                        "near_side": np.where(g["side"] > 0, "left", "right"),
                        "m7_sd_along_ft": sd_al, "m7_sd_across_ft": sd_ac, "m7_sd_heading_deg": sd_h,
                        "m7_cov_centre_ee": cc[:, 0], "m7_cov_centre_en": cc[:, 1], "m7_cov_centre_nn": cc[:, 2]})  # fmt: skip
    for j in range(4):
        out[f"c{j + 1}e"], out[f"c{j + 1}n"] = corners[:, j, 0], corners[:, j, 1]
    return out


def gates(prob: Problem, ft: Fit, frames: pd.DataFrame, camera_ok=True) -> tuple[str, dict]:
    """Section 7.5 gates and the 8.4 articulated flag: (fail_reason or '', diagnostics)."""
    g = prob.geometry(ft.x)
    th = g["th"]
    zw, ze, zx = prob.obs_z(g, ft.assoc)
    mo = prob.motion(g)[1]
    speed = np.abs(mo.get("s", np.zeros(0))) / prob.t.dt * S.MPS_TO_MPH
    moving = speed >= S.M7_MOVING_MPH
    slip_ft = mo["slip"] * S.M7_SIGMA_SLIP_FT if len(mo) else np.zeros(0)
    diag = {
        "wheel_resid_px": float(np.median(np.linalg.norm(zw * prob.tg_sigma[ft.assoc[2]][:, None], axis=1)))
        if len(zw) else np.nan,
        "env_resid_px": float(np.nanmedian(np.abs(ze))) * S.M7_SIGMA_ENV_PX if np.isfinite(ze).any() else np.nan,
        "slip_rms_ft": float(np.sqrt(np.mean(slip_ft[moving] ** 2))) if moving.any() else np.nan,
        "support_frames": int((frames["m7_support"] != "kinematic").sum()),
        "length_ft": float((th[0] + th[1] + th[2]) / FT),
    }  # fmt: skip
    at = ft.info.get("at_bound", [False] * len(th))
    artic = diag["length_ft"] > S.M7_MAX_RIGID_LEN_FT[prob.cfg.subtype] or _outlier_cluster(prob, ft, g)
    diag["articulated_suspect"] = bool(artic)
    if not camera_ok:
        return "camera_invalid", diag
    if diag["support_frames"] < S.M7_MIN_SUPPORT_FRAMES:
        return "few_support", diag
    if not np.isfinite(ft.F) or not ft.converged:
        return "solve_failed", diag
    if ft.info.get("ill_conditioned"):
        return "ill_conditioned", diag
    if at[0] or (at[3] and S.M7_W_BOUND_FAILS):  # user decision 2026-11-09: a width on its limit is 'prior', not a failure
        return "param_bound", diag
    if np.isfinite(diag["wheel_resid_px"]) and diag["wheel_resid_px"] > S.M7_MAX_WHEEL_PX:
        return "wheel_resid", diag
    if np.isfinite(diag["env_resid_px"]) and diag["env_resid_px"] > S.M7_MAX_ENV_PX:
        return "sil_resid", diag
    if np.isfinite(diag["slip_rms_ft"]) and diag["slip_rms_ft"] > S.M7_MAX_SLIP_FT:
        return "slip", diag
    if artic:
        return "articulated_suspect", diag
    return "", diag


def _outlier_cluster(prob: Problem, ft: Fit, g) -> bool:
    """Rejected candidates clustered at one body position behind the rear axle (8.4)."""
    mk, mc, _ = ft.assoc
    matched = set(zip(mk.tolist(), mc.tolist()))
    total = sum(len(o.cand) for o in prob.t.obs)
    if total == 0:
        return False
    along = []
    for k, o in enumerate(prob.t.obs):
        for i in range(len(o.cand)):
            if (k, i) in matched:
                continue
            # body-frame position along the axis from the candidate's ground point
            C, d = prob.t.cam.ray(o.cand[i : i + 1])
            if d[0, 2] >= 0:
                continue
            tt = -C[2] / d[0, 2]
            G = C[:2] + tt * d[0, :2] - prob.t.origin
            along.append(float((G - g["R"][k]) @ g["f"][k]))
    along = np.array(along)
    behind = along[along < -0.5 * g["th"][2]]
    return len(behind) > S.M7_ARTIC_OUTLIER_SHARE * total and np.std(behind) < 1.0


def axle_grouping(prob: Problem, ft: Fit) -> tuple[str, dict]:
    """Section 4.3: per axle group, frames with a matched inlier; observed when every group has enough."""
    mk, _, mt = ft.assoc
    groups = sorted(set(prob.tg_group.tolist()))
    counts = {}
    for gid in groups:
        ks = set(mk[prob.tg_group[mt] == gid].tolist())
        counts[{0: "F", 1: "R1" if prob.cfg.tandem else "R", 2: "R2"}[gid]] = len(ks)
    ok = all(c >= S.M7_GROUP_MIN_FRAMES for c in counts.values())
    return ("observed" if ok else "unresolved"), counts


# --- model selection and frontness (4.3, 4.4, 8.4) -------------------------------------------------------------


def _widen_gates(track: Track, mult: float) -> Track:
    """The same track with every candidate's gate times `mult` (kappa = rho(g / sigma) recomputed). Only a start
    strategy: F, acceptance and convergence of the kept fit always use the frozen prices."""
    sw = S.M7_SIGMA_WHEEL_PX
    obs = []
    for o in track.obs:
        g = np.sqrt(np.expm1(o.kappa)) * sw * o.s
        obs.append(FrameObs(o.frame, o.cand, rho(mult * g / (sw * o.s)), o.env_u, o.env_v, o.ext, o.cand_s))
    return Track(track.frames, obs, track.cam, track.origin, track.init_centre, track.init_psi_deg, track.cls, track.dt)


def fit_config(track: Track, cfg: Config, x0=None, finish_fit=True) -> tuple[Problem, Fit]:
    """Best of the L_w starts (7.2) for one configuration, by F; then covariance and E."""
    prob = Problem(track, cfg)
    if x0 is not None:
        ft = fit(prob, x0, "given")
    else:
        mean = S.M7_PRIORS_FT[cfg.subtype]["Lw"][0]
        starts = [("prior", mean)]
        if cfg.subtype != "pickup_van":
            starts.append(("short", mean - S.M7_START_SHIFT_FT))
        fits = [fit(prob, initial_x(prob, lw), name) for name, lw in starts]
        if S.M7_COARSE_GATE_MULT > 1:  # coarse-to-fine starts: wide gates find the wheels, the frozen ones refine
            coarse = Problem(_widen_gates(track, S.M7_COARSE_GATE_MULT), cfg)
            for name, lw in starts:
                xc = fit(coarse, initial_x(coarse, lw), "coarse").x
                fits.append(fit(prob, xc, name + "+coarse"))
        ft = min(fits, key=lambda q: q.F)
        lw = pair_wheelbase(prob) if S.M7_PAIR_START else None
        if lw is not None and abs(lw - prob.unpack(ft.x)[2][0]) > S.M7_PAIR_TOL_FT * FT:
            # pair restarts: the best fit's pose with the pair wheelbase, either axle kept in place
            for keep in ("front", "rear"):
                fits.append(fit(prob, set_wheelbase(prob, ft.x, lw, keep), f"pairs({lw / FT:.1f},{keep})"))
            ft = min(fits, key=lambda q: q.F)
    if finish_fit:
        ft = finish(prob, ft)
    return prob, ft


def flip_x(prob: Problem, x, prob_to: Problem):
    """The 180 deg flip that keeps the body (8.4): R on the other axle, psi + pi, L_f <-> L_r."""
    R, psi, th = prob.unpack(x)
    A = R + th[0] * unit(psi)
    th2 = th[: prob_to.ns].copy() if len(th) >= prob_to.ns else np.r_[th, np.zeros(prob_to.ns - len(th))]
    th2[1], th2[2] = th[2], th[1]
    return np.r_[A.ravel(), psi + np.pi, th2]


def convert_x(prob: Problem, x, prob_to: Problem):
    """x of one configuration as a start for another: shared statics kept (clipped to the target box), tandem
    spacing added at its prior mean or dropped."""
    R, psi, th = prob.unpack(x)
    mu, _ = prior_cov(prob_to.cfg)
    th2 = mu.copy()
    n = min(len(th), NS2)
    th2[:n] = th[:n]
    th2 = np.clip(th2, prob_to.lo_s + 1e-6, prob_to.hi_s - 1e-6)
    return np.r_[R.ravel(), psi, th2]


def to_two_axle(prob: Problem, x, prob2: Problem):
    R, psi, th = prob.unpack(x)
    return np.r_[R.ravel(), psi, th[:NS2]]


def frontness(track: Track, prob: Problem, ft: Fit, grouping: str) -> dict:
    """Section 8.4: tandem with observed grouping decides; otherwise compare the motion part of F of the main and
    flipped two-axle fits with symmetric assumptions on turning frames."""
    if prob.cfg.tandem and grouping == "observed":
        return {"front_state": "established", "front_source": "tandem", "flip": False, "margin": np.nan}
    cfg_s = Config(prob.cfg.subtype, False, True)
    ps = Problem(track, cfg_s)
    x_main = to_two_axle(prob, ft.x, ps) if prob.cfg.tandem else ft.x.copy()
    f_main = fit(ps, x_main, "main")
    f_flip = fit(ps, flip_x(ps, f_main.x, ps), "flip")
    yaw = np.abs(np.diff(ps.unpack(f_main.x)[1])) / track.dt
    turning = np.degrees(yaw) >= S.M7_FRONT_MIN_YAW_DPS
    if turning.sum() == 0:
        return {"front_state": "assumed", "front_source": "assumed", "flip": False, "margin": 0.0}

    def data_motion_cost(x):
        # design rev 6: F without its prior part. The motion part alone is not enough: a wrong-end hypothesis
        # escapes its slip cost by rejecting wheel candidates, so the comparison must charge for that too.
        tot, p = ps.F(x, parts=True)
        return tot - p["prior"]

    m_main, m_flip = data_motion_cost(f_main.x), data_motion_cost(f_flip.x)
    margin = m_main - m_flip
    if abs(margin) <= S.M7_FRONT_MARGIN:
        return {"front_state": "assumed", "front_source": "assumed", "flip": False, "margin": margin}
    return {"front_state": "established", "front_source": "turning", "flip": margin > 0, "margin": margin,
            "x_flip_sym": f_flip.x}  # fmt: skip


def _runs_backwards(prob: Problem, ft: Fit) -> bool:
    """More than half of the moving steps go against the fitted heading (rear axle displacement along -f)."""
    s = prob.motion(prob.geometry(ft.x))[1].get("s", np.zeros(0))
    moving = np.abs(s) / prob.t.dt * S.MPS_TO_MPH >= S.M7_MOVING_MPH
    return bool(moving.any() and (s[moving] < 0).mean() > 0.5)


def width_resolution(prob: Problem, ft: Fit) -> float:
    """ft of width per px of silhouette: the median over frames of how far the box's u-extreme moves when W grows
    with the near side held (only the far side moves; width debugging 2026-11-10). inf when no frame's extreme moves
    (the far side never forms the silhouette's end)."""
    x = ft.x
    g = prob.geometry(x)
    v = np.zeros_like(x)
    v[3 * prob.K + 3] = 1.0
    v[: 2 * prob.K] = (-0.5 * g["side"][:, None] * g["nl"]).ravel()
    d = 0.1 * FT
    bm, bp = prob.geometry(x - d * v)["box_px"][..., 0], prob.geometry(x + d * v)["box_px"][..., 0]
    du = np.maximum(np.abs(bp.min(1) - bm.min(1)), np.abs(bp.max(1) - bm.max(1))) / (2 * d / FT)  # px per ft
    med = float(np.median(du)) if len(du) else 0.0
    return 1.0 / med if med > 1e-9 else np.inf


def width_held(w_ft_per_px: float, threshold) -> bool:
    """Hold W at the class prior when a pixel of silhouette is worth more than `threshold` ft of width (strictly
    greater; `None` never holds). The study's inline expression, unchanged (design 14.2 test 19)."""
    return bool(threshold is not None and w_ft_per_px > threshold)


def solve_track(track: Track, camera_ok=True, configs=None) -> dict:
    """Full M7 solve of one segment: model selection on E, frontness, gates, per-frame table and summary."""
    subtypes = S.M7_SUBTYPES[int(track.cls)]
    if configs is None:
        configs = [Config(s, t) for s in subtypes for t in ((False, True) if S.M7_ALLOW_TANDEM else (False,))]
    results = []
    for cfg in configs:
        results.append(fit_config(track, cfg, finish_fit=False))
    # cross-start (smoke run, trolley 120: its two-axle fit stuck in a bad minimum that the tandem fit avoided):
    # refit every configuration from the pose and body of the lowest-F fit, keep whichever F is lower
    ref_prob, ref_ft = min(results, key=lambda r: r[1].F)
    for i, (prob, ft) in enumerate(results):
        if prob is ref_prob:
            continue
        x0 = convert_x(ref_prob, ref_ft.x, prob)
        alt = fit(prob, x0, "cross")
        if alt.F < ft.F:
            results[i] = (prob, alt)
    results = [(p, finish(p, f)) for p, f in results]
    default = next(r for r in results if r[0].cfg == Config(subtypes[0], False)) \
        if any(r[0].cfg == Config(subtypes[0], False) for r in results) else results[0]  # fmt: skip
    best = default
    for r in results:
        if r is default or not np.isfinite(r[1].E):
            continue
        if r[1].E < best[1].E - S.M7_MODEL_MARGIN or (best is not default and r[1].E < best[1].E):
            best = r
    prob, ft = best
    e_sorted = sorted(q[1].E for q in results if q[0].cfg.tandem == prob.cfg.tandem and np.isfinite(q[1].E))
    subtype_ambiguous = len(subtypes) > 1 and len(e_sorted) > 1 and e_sorted[1] - e_sorted[0] < S.M7_MODEL_MARGIN
    grouping, group_frames = axle_grouping(prob, ft)
    fr = frontness(track, prob, ft, grouping)
    if S.M7_FRONT_RULE == "travel":
        # user decision 2026-11-09: the turning test put 6 of 25 real vehicles in reverse through the junction. The
        # front is the end facing the direction of travel; the turning test is kept as a diagnostic (front_turning).
        turning_flip = fr.get("flip") if fr["front_state"] == "established" else None
        flip = _runs_backwards(prob, ft)
        front_turning = "undecided" if turning_flip is None else ("agrees" if turning_flip == flip else "disagrees")
        fr = {"front_state": "assumed", "front_source": "travel", "flip": flip, "margin": fr.get("margin", np.nan),
              "front_turning": front_turning}
    if fr.get("flip"):  # refit in the winning orientation with the normal (asymmetric) assumptions
        prob2 = Problem(track, Config(prob.cfg.subtype, False))
        x0 = to_two_axle(prob, ft.x, prob2) if prob.cfg.tandem else ft.x
        prob, ft = fit_config(track, prob2.cfg, x0=flip_x(prob2, x0, prob2))
        grouping, group_frames = axle_grouping(prob, ft)
    w_ft_per_px = width_resolution(prob, ft)
    w_held = width_held(w_ft_per_px, S.M7_W_HOLD_FT_PER_PX)
    if w_held:
        # user decision 2026-11-10 ("Hold W at prior"): when one pixel of silhouette is worth more than this much
        # width, the far end (bike racks, rounded corners, mask bleed: all one-sided) pins W on its upper bound;
        # refit with W at the class prior instead
        prob2 = Problem(track, replace(prob.cfg, hold_w=True))
        x0 = ft.x.copy()
        x0[3 * prob.K + 3] = S.M7_PRIORS_FT[prob.cfg.subtype]["W"][0] * FT
        prob, ft = fit_config(track, prob2.cfg, x0=x0)
        grouping, group_frames = axle_grouping(prob, ft)
    frames = frame_table(prob, ft)
    reason, diag = gates(prob, ft, frames, camera_ok)
    sizes = size_sources(prob, ft)
    if w_held:
        sizes["W_source"] = "prior"
        ls = sizes["L_source"]
        sizes["dims_source"] = "m7_prior" if ls == "prior" else "m7_partly_prior"
    g = prob.geometry(ft.x)
    s = prob.motion(g)[1].get("s", np.zeros(0))
    moving_sup = (np.abs(s) / track.dt * S.MPS_TO_MPH >= S.M7_MOVING_MPH) & \
        (frames["m7_support"].to_numpy()[:-1] != "kinematic")  # fmt: skip
    reversing = None
    if fr["front_state"] == "established" and moving_sup.any():
        reversing = bool((s[moving_sup] < 0).mean() > 0.5)
    th = g["th"] / FT
    summary = {
        "subtype": prob.cfg.subtype, "subtype_ambiguous": bool(subtype_ambiguous),
        "axle_config": "tandem" if prob.cfg.tandem else "two_axle", "axle_grouping": grouping,
        "group_frames": group_frames, "start_used": ft.start, "F": ft.F, "E": ft.E, "converged": ft.converged,
        "iters": ft.iters, "wheelbase_ft": th[0], "front_oh_ft": th[1], "rear_oh_ft": th[2], "width_ft": th[3],
        "height_ft": th[4], "clearance_ft": th[5], "inset_ft": th[IS],
        "tandem_ft": th[IT] if len(th) > IT else np.nan, "length_ft": th[0] + th[1] + th[2],
        "fail_reason": reason, **diag, **sizes, **{k: v for k, v in fr.items() if k != "x_flip_sym"},
        "reversing": reversing, "w_ft_per_px": w_ft_per_px, "w_held": w_held, "models": {f"{q[0].cfg.subtype}/{'tandem' if q[0].cfg.tandem else 'two'}": q[1].E
                                           for q in results},
    }  # fmt: skip
    frames["native_valid"] = (reason == "") & (frames["m7_support"] != "kinematic")
    frames["fail_reason"] = np.where(reason != "", reason, np.where(frames["m7_support"] == "kinematic",
                                                                     "no_observation", ""))  # fmt: skip
    return {"frames": frames, "summary": summary, "problem": prob, "fit": ft}
