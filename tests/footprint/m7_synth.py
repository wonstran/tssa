"""Ported from the speed study's tests/m7_synth.py (imports only changed)."""
"""Synthetic turning vehicles for the M7 solver tests (design section 13 fixture).

The truth is an exact kinematic bicycle: the rear-axle centre moves along the heading every step (no slip), and
the heading turns at v tan(delta) / L_w with a smooth steering schedule. Observations are generated from the
same geometry the solver uses (section 4.2: wheel contacts on the near side, body box at z in [c, H]).
"""

import numpy as np

from tssa.footprint.locators.m7 import solver as m7
from tssa.footprint.locators.m7.candidates import rho, tyre_px
from tssa.footprint.locators.m7.settings import S
from tssa.transform.camera import look_at

FT = S.FT
BUS = dict(Lw=24.0, Lf=7.0, Lr=9.0, W=8.5, H=11.0)  # ft


def camera(C=(-5.0, -35.0, 8.0), target=(0.0, -5.0, 0.0)):
    return look_at(C, target=target, f=320.0)


def simulate(K=70, v=4.0, dims=BUS, start=(-22.0, 0.0), psi0_deg=90.0, turn_deg=90.0, R_turn=12.2, k_turn=20,
             ramp=8, dt=0.1, reverse=False, tandem_ft=None, reverse_from=None):
    """Truth: dict with R (K, 2), psi (K,) rad, th (statics m), plus frames."""
    th = np.array([dims["Lw"], dims["Lf"], dims["Lr"], dims["W"], dims["H"], dims.get("c", 1.0), dims.get("i", 0.65)]) * FT
    if tandem_ft is not None:
        th = np.r_[th, tandem_ft * FT]
    R = np.zeros((K, 2))
    psi = np.zeros(K)
    R[0], psi[0] = start, np.radians(psi0_deg)
    delta_max = np.arctan(th[0] / R_turn) if turn_deg else 0.0
    turned = 0.0
    sgn = -1.0 if reverse else 1.0
    for k in range(K - 1):
        if k < k_turn or turned >= np.radians(abs(turn_deg)):
            target = 0.0
        else:
            target = delta_max * np.sign(turn_deg)
        prev = getattr(simulate, "_d", 0.0) if k else 0.0
        d = prev + np.clip(target - prev, -delta_max / ramp if ramp else np.inf, delta_max / ramp if ramp else np.inf)
        simulate._d = d
        if reverse_from is not None and k >= reverse_from:
            d, sgn = 0.0, -1.0
            simulate._d = 0.0
        step = sgn * v * dt
        R[k + 1] = R[k] + step * m7.unit(psi[k])
        dpsi = abs(step) * np.tan(d) / th[0]
        psi[k + 1] = psi[k] + dpsi
        turned += abs(dpsi)
    simulate._d = 0.0
    return dict(R=R, psi=psi, th=th, frames=np.arange(K), dt=dt)


def _pose_x(truth):
    return np.r_[truth["R"].ravel(), truth["psi"], truth["th"]]


def observations(truth, cam, cfg, visible=None, noise=0.5, outlier_frac=0.0, seed=0, env_cols=30, ext=(True, True),
                 truncate_u=None, far=False, cut_left_frac=0.0):
    """FrameObs list from the truth. `visible(k) -> set of target names` among 'front', 'rear' (and 'rear1',
    'rear2' for tandem); default both. `truncate_u`: image u limit; columns/extents beyond it are dropped."""
    rng = np.random.default_rng(seed)
    K = len(truth["frames"])
    track = m7.Track(truth["frames"], [m7.FrameObs(int(f)) for f in truth["frames"]], cam, np.zeros(2),
                     np.zeros((K, 2)), np.zeros(K), 5, truth["dt"])  # fmt: skip
    prob = m7.Problem(track, cfg)
    g = prob.geometry(_pose_x(truth))
    names = ["front"] + (["rear1", "rear2"] if cfg.tandem else ["rear"])
    r_ft = S.M7_PRIORS_FT[cfg.subtype]["wheel_r"]
    obs = []
    for k in range(K):
        vis = set(names) if visible is None else set(visible(k))
        idx = [i for i, nme in enumerate(names) if nme in vis]
        if far:
            idx += [i + len(names) for i, nme in enumerate(names) if nme in vis]
        cand = g["tg_px"][k, idx] + rng.normal(0, noise, (len(idx), 2))
        n_out = rng.binomial(len(idx) + 1, outlier_frac) if outlier_frac else 0
        if n_out:
            lo, hi = g["box_px"][k].min(0), g["box_px"][k].max(0)
            out = np.column_stack([rng.uniform(lo[0], hi[0], n_out), rng.uniform(hi[1] - 5, hi[1] + 15, n_out)])
            cand = np.vstack([cand, out])
        if truncate_u is not None and len(cand):
            cand = cand[cand[:, 0] < truncate_u]
        D = np.full(len(cand), 15.0)
        if len(cand):
            G = np.array([cam.ray(c[None])[0][:2] + (-cam.ray(c[None])[0][2] / cam.ray(c[None])[1][0, 2]) *
                          cam.ray(c[None])[1][0, :2] for c in cand])  # fmt: skip
            p0 = cam.project(np.column_stack([G, np.zeros(len(G))]))
            p1 = cam.project(np.column_stack([G, np.full(len(G), 2 * r_ft * FT)]))
            D = np.linalg.norm(p1 - p0, axis=1)
        gate = np.maximum(S.M7_GATE_MIN_PX, S.M7_GATE_TYRES * D)
        kappa = rho(gate / S.M7_SIGMA_WHEEL_PX)
        bp = g["box_px"][k]
        umin, umax = bp[:, 0].min(), bp[:, 0].max()
        u = np.linspace(umin + 2 + cut_left_frac * (umax - umin), umax - 2, env_cols)
        prob.env_u = np.full((K, S.M7_ENV_MAX_COLS), np.nan)
        prob.env_u[k, :env_cols] = u
        vb = prob.geometry(_pose_x(truth))["env_pred"][k, :env_cols]
        vv = vb + rng.normal(0, noise, env_cols)
        e = g["ext_pred"][k] + rng.normal(0, noise, 2)  # near-side bottom corners: rounded ends hide the far ones
        e = np.where(np.array(ext), e, np.nan)
        if truncate_u is not None:
            keep = u < truncate_u
            u, vv = u[keep], vv[keep]
            if umax >= truncate_u:
                e[1] = np.nan
            if umin >= truncate_u:
                e[0] = np.nan
        obs.append(m7.FrameObs(int(truth["frames"][k]), cand, kappa, u, vv, e))
    return obs


def track(truth, cam, obs, init_noise=0.3, seed=1, cls=5, centre_offset=(0.4, -0.3)):
    """Track with an M2-like initial centre (truth + offset + noise) and the centre's motion heading."""
    rng = np.random.default_rng(seed)
    th = truth["th"]
    centre = truth["R"] + ((th[0] + th[1] - th[2]) / 2) * m7.unit(truth["psi"])
    c0 = centre + np.array(centre_offset) + rng.normal(0, init_noise, centre.shape)
    d = np.gradient(centre, axis=0)
    psi0 = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
    return m7.Track(truth["frames"], obs, cam, np.zeros(2), c0, psi0, cls, truth["dt"])


def centre_of(truth):
    th = truth["th"]
    return truth["R"] + ((th[0] + th[1] - th[2]) / 2) * m7.unit(truth["psi"])
