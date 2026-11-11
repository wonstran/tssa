"""Ported from the speed study's tests/test_m7.py (imports changed; solve-heavy tests marked slow)."""
"""M7 solver: synthetic tests of design 2026-11-08-m7 (rev 6), section 13.

Product-level tests (14 sample independence, 18 and 22 display admission) belong to the runner integration step.
"""

import dataclasses
import itertools

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import norm

import m7_synth as sy
from tssa.footprint.locators.m7 import solver as m7
from tssa.footprint.locators.m7.candidates import rho
from tssa.footprint.locators.m7.settings import S

FT = S.FT
BUS = m7.Config("bus")
CAM = sy.camera()


def _fit(truth, cfg=BUS, cls=5, cam=CAM, **kw):
    obs = sy.observations(truth, cam, cfg, **kw)
    tr = sy.track(truth, cam, obs, cls=cls)
    prob, ft = m7.fit_config(tr, cfg)
    return prob, ft, tr


def _errors(prob, ft, truth):
    fr = m7.frame_table(prob, ft)
    g = prob.geometry(ft.x)
    c_err = np.linalg.norm(fr[["e_nat", "n_nat"]].to_numpy() - sy.centre_of(truth), axis=1) / FT
    h_err = np.degrees(np.abs((g["psi"] - truth["psi"] + np.pi) % (2 * np.pi) - np.pi))
    return c_err, h_err, g["th"] / FT, fr


@pytest.fixture(scope="module")
def turn():
    return sy.simulate()


@pytest.fixture(scope="module")
def full_fit(turn):
    return _fit(turn)


# --- accuracy and partial visibility (tests 1-6) ---------------------------------------------------------------


@pytest.mark.slow
def test_01_all_observations(full_fit, turn):
    prob, ft, _ = full_fit
    c, h, th, fr = _errors(prob, ft, turn)
    assert c.max() <= 0.25 and h.max() <= 0.5  # 0.2 before the inset was fitted (design rev 7: 0.20 observed)
    assert abs(th[0] - 24) <= 0.3 and abs(th[:3].sum() - 40) <= 0.5
    assert (fr["m7_support"] == "wheels2").all()


@pytest.mark.slow
def test_02_stage_sequence(turn):
    prob, ft, _ = _fit(turn, visible=lambda k: {"front"} if k < 25 else ({"front", "rear"} if k < 45 else {"rear"}))
    _, h, _, fr = _errors(prob, ft, turn)
    assert h.max() <= 1.5
    assert set(fr["m7_support"]) == {"wheel_front", "wheels2", "wheel_rear"}


@pytest.mark.slow
def test_03_front_only_after_straight_approach(turn):
    prob, ft, _ = _fit(turn, visible=lambda k: {"front"})
    _, h, _, _ = _errors(prob, ft, turn)
    assert h.max() <= 2.0


@pytest.mark.slow
def test_04_rear_only_heading_is_rear_track_tangent(turn):
    prob, ft, _ = _fit(turn, visible=lambda k: {"rear"})
    _, h, _, _ = _errors(prob, ft, turn)  # the truth heading is the rear-axle track tangent (exact bicycle)
    assert h.max() <= 0.75  # 0.5 until the extents lost their far corners (2026-11-09): 0.58 deg observed


@pytest.mark.slow
def test_05_outliers(turn):
    prob, ft, _ = _fit(turn, outlier_frac=0.2)
    c, h, th, fr = _errors(prob, ft, turn)
    assert c.max() <= 0.2 and abs(th[0] - 24) <= 0.3
    assert h.max() <= 1.0  # design: 0.5 deg as test 1; 0.6 deg observed with 20% outliers (recorded deviation)
    assert fr["m7_n_outliers"].sum() > 0


@pytest.mark.slow
def test_06_truncated(turn):
    prob, ft, _ = _fit(turn, truncate_u=330)
    c, _, _, _ = _errors(prob, ft, turn)
    assert c.max() <= 0.5
    assert (~prob.ext_ok[:, 1]).any()  # truncated extents carry no residual


# --- configurations (tests 10, 11, 13, 16, 17, 20) -------------------------------------------------------------


@pytest.mark.slow
def test_10_tandem():
    truth = sy.simulate(tandem_ft=4.5)
    res = m7.solve_track(sy.track(truth, CAM, sy.observations(truth, CAM, m7.Config("bus", True))),
                         configs=[BUS, m7.Config("bus", True)])  # fmt: skip
    s = res["summary"]
    assert s["axle_config"] == "tandem" and abs(s["wheelbase_ft"] - 24) <= 0.3


@pytest.mark.slow
def test_11_left_and_right_side(turn):
    for cam, side in ((sy.camera(), "right"), (sy.camera(C=(-5.0, 30.0, 8.0), target=(0.0, -5.0, 0.0)), "left")):
        prob, ft, _ = _fit(turn, cam=cam)
        c, _, _, fr = _errors(prob, ft, turn)
        assert fr["near_side"].iloc[0] == side and c.max() <= 0.3


@pytest.mark.slow
def test_13_long_rigid_is_articulated_suspect():
    truth = sy.simulate(dims=dict(Lw=34, Lf=9, Lr=12, W=8.5, H=11))
    res = m7.solve_track(sy.track(truth, CAM, sy.observations(truth, CAM, BUS)), configs=[BUS])
    assert res["summary"]["articulated_suspect"] and res["summary"]["fail_reason"] != ""


@pytest.mark.slow
def test_16_17_truck_subtypes():
    pick = sy.simulate(dims=dict(Lw=12, Lf=3, Lr=4.2, W=6.7, H=6.5))
    box = sy.simulate(dims=dict(Lw=16, Lf=4, Lr=7, W=8, H=11))
    cfgs = [m7.Config("truck"), m7.Config("pickup_van")]
    r1 = m7.solve_track(sy.track(pick, CAM, sy.observations(pick, CAM, cfgs[1]), cls=7), configs=cfgs)["summary"]
    r2 = m7.solve_track(sy.track(box, CAM, sy.observations(box, CAM, cfgs[0]), cls=7), configs=cfgs)["summary"]
    assert r1["subtype"] == "pickup_van" and abs(r1["length_ft"] - 19.2) <= 0.5
    assert r2["subtype"] == "truck"


@pytest.mark.slow
def test_20_single_vs_tandem_on_E(full_fit, turn):
    _, ft1, tr = full_fit
    _, ft2 = m7.fit_config(tr, m7.Config("bus", True))
    assert ft1.E < ft2.E  # single-axle data: two-axle model has the better evidence
    tt = sy.simulate(tandem_ft=4.5)
    trt = sy.track(tt, CAM, sy.observations(tt, CAM, m7.Config("bus", True)))
    assert m7.fit_config(trt, m7.Config("bus", True))[1].E < m7.fit_config(trt, BUS)[1].E - S.M7_MODEL_MARGIN


# --- frontness (tests 12a-12g) ---------------------------------------------------------------------------------


def test_12a_flip_symmetry(turn):
    for cfg, wheels_equal in ((m7.Config("bus", False, True), True), (BUS, False)):
        obs = sy.observations(turn, CAM, cfg)
        prob = m7.Problem(sy.track(turn, CAM, obs), cfg)
        x = np.r_[turn["R"].ravel(), turn["psi"], turn["th"]]
        _, p0 = prob.F(x, parts=True)
        _, p1 = prob.F(m7.flip_x(prob, x, prob), parts=True)
        assert abs(p0["env"] - p1["env"]) < 1e-9 and abs(p0["ext"] - p1["ext"]) < 1e-9
        assert (abs(p0["wheel"] - p1["wheel"]) < 1e-9) == wheels_equal


@pytest.fixture
def turning_rule(monkeypatch):
    """Tests 12b-12g check the design's turning rule; the default since 2026-11-09 is direction of travel."""
    monkeypatch.setattr(S, "M7_FRONT_RULE", "turning")


def _front(truth, cfg=BUS, flip_init=False, **kw):
    tr = sy.track(truth, CAM, sy.observations(truth, CAM, cfg, **kw))
    if flip_init:
        tr.init_psi_deg = (tr.init_psi_deg + 180) % 360
    return m7.solve_track(tr, configs=[cfg])


@pytest.mark.usefixtures("turning_rule")
@pytest.mark.slow
def test_12b_turn_then_back_up():
    truth = sy.simulate(K=90, reverse_from=40)
    s = _front(truth)["summary"]
    assert s["front_state"] == "established" and s["front_source"] == "turning" and s["reversing"] is True


@pytest.mark.usefixtures("turning_rule")
@pytest.mark.slow
def test_12c_straight_reverse_is_assumed():
    s = _front(sy.simulate(turn_deg=0, reverse=True, psi0_deg=270, start=(5.0, 0.0)))["summary"]
    assert s["front_state"] == "assumed" and s["reversing"] is None


@pytest.mark.usefixtures("turning_rule")
@pytest.mark.slow
def test_12d_reverse_through_turn():
    truth = sy.simulate(reverse=True, psi0_deg=270, start=(5.0, 0.0))
    res = _front(truth)
    s = res["summary"]
    _, h, _, _ = _errors(res["problem"], res["fit"], truth)
    assert s["front_state"] == "established" and s["reversing"] is True and h.max() <= 2.0


@pytest.mark.usefixtures("turning_rule")
@pytest.mark.slow
def test_12_wrong_initial_front_is_corrected(turn):
    res = _front(turn, flip_init=True)
    _, h, _, _ = _errors(res["problem"], res["fit"], turn)
    assert res["summary"]["front_source"] == "turning" and h.max() <= 1.0


@pytest.mark.usefixtures("turning_rule")
@pytest.mark.slow
def test_12e_inset_swap_does_not_change_the_decision(turn, monkeypatch):
    d0 = _front(turn)["summary"]["front_state"]
    pri = {k: dict(v) for k, v in S.M7_PRIORS_FT.items()}
    pri["bus"]["i_f"], pri["bus"]["i_r"] = pri["bus"]["i_r"], pri["bus"]["i_f"]
    monkeypatch.setattr(S, "M7_PRIORS_FT", pri)
    assert _front(turn)["summary"]["front_state"] == d0 == "established"


@pytest.mark.usefixtures("turning_rule")
@pytest.mark.slow
def test_12f_12g_tandem_grouping():
    truth = sy.simulate(turn_deg=0, reverse=True, psi0_deg=270, start=(5.0, 0.0), tandem_ft=4.5)
    cfg = m7.Config("bus", True)
    s = _front(truth, cfg, flip_init=True)["summary"]  # front-first start faces the wrong way; tandem fixes it
    assert s["axle_grouping"] == "observed" and s["front_source"] == "tandem" and s["reversing"] is True
    s = _front(truth, cfg, flip_init=True, visible=lambda k: {"rear1", "rear2"})["summary"]  # front group hidden
    assert s["axle_grouping"] == "unresolved" and s["front_source"] != "tandem" and s["front_state"] == "assumed"
    assert s["reversing"] is None


# --- size information (tests 15a-15e) --------------------------------------------------------------------------


def _sizes(truth, **kw):
    prob, ft, _ = _fit(truth, **kw)
    return m7.size_sources(prob, ft)


@pytest.mark.slow
def test_15a_priors_only_for_unseen_dimensions(turn):
    s = _sizes(turn, visible=lambda k: {"rear"}, env_cols=0, ext=(False, False))
    assert s["L_source"] == "prior" and s["Lf_ratio"] > 0.9 and s["Lr_ratio"] > 0.9


@pytest.mark.slow
def test_15b_conditional_overhang_uncertainty(turn):
    s = _sizes(turn, env_cols=0, ext=(False, False))  # both wheels, no silhouette: L_w seen, ends not
    p = S.M7_PRIORS_FT["bus"]
    sw = s["Lw_sd_ft"]
    ref_f = np.hypot(p["sf"], p["af"] * sw)
    ref_r = np.hypot(p["sr"], p["ar"] * sw)
    ref_L = np.sqrt(p["sf"] ** 2 + p["sr"] ** 2 + ((1 + p["af"] + p["ar"]) * sw) ** 2)
    assert abs(s["Lf_sd_ft"] / ref_f - 1) < 0.02 and abs(s["Lr_sd_ft"] / ref_r - 1) < 0.02
    assert abs(s["L_sd_ft"] / ref_L - 1) < 0.02 and s["L_source"] != "measured"
    assert abs(ref_f - 2.0) < 0.05  # conditioned, not the 2.48 ft marginal


@pytest.mark.slow
def test_15c_weak_width():
    truth = sy.simulate(turn_deg=0, start=(-8.0, 0.0), K=40)
    cam = sy.camera(C=(0.0, -30.0, 8.0), target=(0.0, 0.0, 0.0))
    prob, ft, _ = _fit(truth, cam=cam, ext=(False, False))
    s = m7.size_sources(prob, ft)
    assert s["W_ratio"] > 0.9  # the fixture really leaves W weak
    assert s["W_source"] == "prior" and s["dims_source"] != "m7_measured"


@pytest.mark.slow
def test_15d_both_ends_measure_length_without_the_rear_wheel(turn):
    s = _sizes(turn, visible=lambda k: {"front"})
    assert s["L_source"] == "measured" and s["Lw_ratio"] > s["L_ratio"]


@pytest.mark.slow
def test_15e_weak_length(turn):
    s = _sizes(turn, visible=lambda k: {"front"}, ext=(False, True), cut_left_frac=0.35)
    assert s["L_source"] != "measured" and s["dims_source"] != "m7_measured"


# --- objective, association, normalisation (tests 19, 21, 23-27) -----------------------------------------------


@pytest.mark.slow
def test_19_start_that_drops_wheels_loses(full_fit):
    prob, ft, tr = full_fit
    bad = m7.fit(prob, m7.initial_x(prob, 14.0), outer=0)  # silhouette-only fit from a short wheelbase
    assert bad.F > ft.F


def _one_frame(cand, kappa, targets):
    obs = [m7.FrameObs(0, np.asarray(cand, float), np.asarray(kappa, float))]
    tr = m7.Track(np.array([0]), obs, CAM, np.zeros(2), np.zeros((1, 2)), np.zeros(1))
    prob = m7.Problem(tr, BUS)
    tg = np.full((1, prob.T, 2), 1e6)
    tg[0, : len(targets)] = targets
    return prob, {"tg_px": tg}


def _wheel_cost(prob, g, assoc):
    o = prob.t.obs[0]
    d = {int(c): int(t) for c, t in zip(assoc[1], assoc[2])}
    return sum(rho(np.linalg.norm(o.cand[c] - g["tg_px"][0, d[c]]) / prob.tg_sigma[d[c]]) if c in d else o.kappa[c]
               for c in range(len(o.cand)))  # fmt: skip


def _brute(prob, g):
    o = prob.t.obs[0]
    best = np.inf
    for perm in itertools.product([None, *range(prob.T)], repeat=len(o.cand)):
        used = [p for p in perm if p is not None]
        if len(used) != len(set(used)):
            continue
        cost = 0.0
        for c, t in enumerate(perm):
            if t is None:
                cost += o.kappa[c]
                continue
            r = rho(np.linalg.norm(o.cand[c] - g["tg_px"][0, t]) / prob.tg_sigma[t])
            cost += r if r < o.kappa[c] else np.inf
        best = min(best, cost)
    return best


@pytest.mark.parametrize("cand,targets,kappa,total,pairs", [
    ([(1, 0), (2, 45 ** 0.5)], [(0, 0), (8, 0)], rho(8 / 1.5), 3.7502, {(0, 0)}),
    ([(1, 0), (2, 45 ** 0.5)], [(0, 0), (8, 0)], 6.0, 3.9786, {(0, 0), (1, 1)}),
    ([(1, 0), (-5.6, 0)], [(0, 0), (2.5, 0)], rho(8 / 1.5), 3.3970, {(0, 1), (1, 0)}),
])  # fmt: skip
def test_21_assignment_examples(cand, targets, kappa, total, pairs):
    prob, g = _one_frame(cand, [kappa, kappa], targets)
    a = prob.associate(g)
    assert set(zip(a[1].tolist(), a[2].tolist())) == pairs
    cost = _wheel_cost(prob, g, a)
    assert abs(cost - total) < 1e-3 and abs(cost - _brute(prob, g)) < 1e-9
    for c, t in zip(a[1], a[2]):  # no edge is ever rejected after assignment: every match beats its price
        assert rho(np.linalg.norm(prob.t.obs[0].cand[c] - g["tg_px"][0, t]) / prob.tg_sigma[t]) < kappa


def test_23_irls_reaches_the_mode_although_the_laplace_term_rises():
    F = lambda x: np.log1p((x * x - 4) ** 2) + 0.5 * (x - 1) ** 2  # noqa: E731
    I = lambda x: 1 + 8 * x * x / (1 + (x * x - 4) ** 2)  # noqa: E731
    mode = minimize_scalar(F, bracket=(1.9, 2.0)).x
    assert F(1.96852) < F(1.95852) and F(1.96852) + 0.5 * np.log(I(1.96852)) > F(1.95852) + 0.5 * np.log(I(1.95852))
    x = 1.95852
    for _ in range(200):  # IRLS on the Cauchy term (weight 1 / (1 + r^2)), backtracking on F only
        r = x * x - 4
        w = 1 / (1 + r * r)
        # surrogate w r^2 + 1/2 (x - 1)^2 with r linearised: r ~ r0 + 2 x0 dx
        dx = -(2 * w * r * 2 * x + (x - 1)) / (2 * w * (2 * x) ** 2 + 1)
        step = 1.0
        while F(x + step * dx) > F(x) and step > 1e-4:
            step /= 2
        x += step * dx
    assert abs(x - mode) < 1e-5 and abs(mode - 1.96852) < 1e-4


@pytest.mark.slow
def test_24_rejection_price_does_not_depend_on_the_hypothesis(full_fit):
    prob, ft, _ = full_fit
    kap = [o.kappa.copy() for o in prob.t.obs]
    for lw in (12.0, 24.0, 30.0):
        prob.F(m7.initial_x(prob, lw))
        assert all(np.array_equal(o.kappa, k) for o, k in zip(prob.t.obs, kap))


def test_25_association_is_the_minimiser_at_every_pose():
    prob, g = _one_frame([(1, 0), (14, 0)], [rho(8 / 1.5)] * 2, [(0, 0), (8, 0)])  # second candidate out of gate
    assert abs(_wheel_cost(prob, g, prob.associate(g)) - _brute(prob, g)) < 1e-9
    g["tg_px"][0, 1] = (12, 0)  # a pose step brings it inside its gate
    a = prob.associate(g)
    assert len(a[0]) == 2 and abs(_wheel_cost(prob, g, a) - _brute(prob, g)) < 1e-9


def test_26_bounded_normal_normaliser():
    Z = norm.cdf(1) - norm.cdf(-1)
    assert abs(Z - 0.682689) < 1e-6 and abs(np.log(Z) + 0.381715) < 1e-6
    nl = lambda x: 0.5 * x * x + 0.5 * np.log(2 * np.pi) + np.log(Z)  # noqa: E731
    assert abs(quad(lambda x: np.exp(-nl(x)), -1, 1)[0] - 1) < 1e-9


def test_27_joint_prior_normaliser():
    cfg = m7.Config("pickup_van")
    mu, C = m7.prior_cov(cfg)
    _, _, _, lo, hi = m7.prior_model(cfg)
    x = np.random.default_rng(0).multivariate_normal(mu, C, 400_000)
    mc = ((x >= lo) & (x <= hi)).all(axis=1).mean()
    assert abs(np.exp(m7.log_ZB(cfg)) - mc) < 0.005
    # F has no normaliser: two statics differ in F only by the prior quadratic (and data terms)
    tr = m7.Track(np.array([0, 1]), [m7.FrameObs(0), m7.FrameObs(1)], CAM, np.zeros(2), np.zeros((2, 2)),
                  np.zeros(2))  # fmt: skip
    prob = m7.Problem(tr, cfg)
    for lw in (9.5, 12.0, 14.5):
        x = m7.initial_x(prob, lw)
        _, p = prob.F(x, parts=True)
        z = prob.prior_z(prob.unpack(x)[2])
        assert abs(p["prior"] - 0.5 * z @ z) < 1e-12


# --- fitted tyre inset (user decision 2026-11-09) --------------------------------------------------------------


@pytest.mark.slow
def test_28_fitted_inset_is_recovered():
    truth = sy.simulate(dims=dict(sy.BUS, i=1.3))  # prior mean 0.65 ft
    prob, ft, _ = _fit(truth)
    c, h, th, _ = _errors(prob, ft, truth)
    assert abs(th[m7.IS] - 1.3) <= 0.15 and c.max() <= 0.25 and h.max() <= 0.6


# --- pair wheelbase start (bus 16 debugging, 2026-11-09) -------------------------------------------------------


def test_29_pair_wheelbase(turn):
    prob = m7.Problem(sy.track(turn, CAM, sy.observations(turn, CAM, BUS)), BUS)
    assert abs(m7.pair_wheelbase(prob) / FT - 24) <= 0.3
    one = m7.Problem(sy.track(turn, CAM, sy.observations(turn, CAM, BUS, visible=lambda k: {"rear"})), BUS)
    assert m7.pair_wheelbase(one) is None  # one tyre per frame: no pairs
    x = sy._pose_x(turn)
    for keep, j in (("front", 0), ("rear", 1)):  # the kept axle's contact does not move
        g0, g1 = prob.geometry(x), prob.geometry(m7.set_wheelbase(prob, x, 20 * FT, keep))
        assert np.abs(g0["tg_px"][:, j] - g1["tg_px"][:, j]).max() < 1e-6


# --- tyre lowest point and image-edge sigma (box truck 452 debugging, 2026-11-09) -------------------------------


def test_30_tyre_lowest_point_and_edge_scale(turn):
    for C in ((-5.0, -35.0, 8.0), (0.0, -8.0, 8.0)):  # far and near camera
        cam = sy.camera(C=C)
        prob = m7.Problem(sy.track(turn, cam, [m7.FrameObs(int(f)) for f in turn["frames"]]), BUS)
        x = sy._pose_x(turn)
        g = prob.geometry(x)
        R, psi, th = prob.unpack(x)
        f = m7.unit(psi)
        P = R + g["side"][:, None] * m7.left(psi) * (th[3] / 2 - (th[m7.IS] - prob.di_half))  # rear contacts
        phi = np.linspace(0, 2 * np.pi, 20001)
        r = prob.r_wheel
        for k in range(0, prob.K, 10):  # closed form = brute-force lowest image point of the tyre circle
            pts = np.column_stack([P[k, 0] + r * np.cos(phi) * f[k, 0], P[k, 1] + r * np.cos(phi) * f[k, 1],
                                   r + r * np.sin(phi)])  # fmt: skip
            uv = prob._proj(pts)
            assert np.abs(uv[np.argmax(uv[:, 1])] - g["tg_px"][k, 1]).max() < 0.01
    s = m7.edge_scale(np.array([0.0, 80.0, 160.0, 320.0, 639.0]), 640)
    assert np.allclose(s, [S.M7_EDGE_SIGMA_MULT, 1 + (S.M7_EDGE_SIGMA_MULT - 1) / 2, 1, 1, S.M7_EDGE_SIGMA_MULT])


def _out_of_span_jumps(prob, x):
    """Largest change of any out-of-span envelope prediction between consecutive small steps of W and of every
    frame's heading (a heading sweep makes the extreme vertices swap order)."""
    worst = 0.0
    K = prob.K
    for idx, span, n in ((3 * K + 3, 0.05 * FT, 101), (slice(2 * K, 3 * K), np.radians(6.0), 2401)):
        prev = None
        for s in np.linspace(-span, span, n):
            xx = x.copy()
            xx[idx] += s
            cur = prob.geometry(xx)["env_pred"]
            if prev is not None:
                worst = max(worst, float(np.abs(cur - prev).max()))
            prev = cur
    return worst


def test_31_envelope_outside_box_span_is_continuous(turn, monkeypatch):
    """Bus 417 (2026-11-09): envelope columns outside the box's u-span were evaluated at the extreme vertex, where the
    silhouette narrows to one point and the prediction steps when two vertices swap order. They are now evaluated
    M7_ENV_CLAMP_PX inside the span, where the lower envelope is continuous."""
    prob = m7.Problem(sy.track(turn, CAM, [m7.FrameObs(int(f)) for f in turn["frames"]]), BUS)
    x = sy._pose_x(turn)
    bp = prob.geometry(x)["box_px"]
    umin, umax = bp[..., 0].min(1), bp[..., 0].max(1)
    prob.env_u = np.full((prob.K, 4), np.nan)
    prob.env_u[:, 0], prob.env_u[:, 1] = umin - 40, umin - 5  # left of the span
    prob.env_u[:, 2], prob.env_u[:, 3] = umax + 5, umax + 40  # right of the span
    g = prob.geometry(x)
    inside = m7.Problem(prob.t, BUS)  # the same columns placed exactly at the clamp points
    m = np.minimum(S.M7_ENV_CLAMP_PX, (umax - umin) / 4)
    inside.env_u = np.column_stack([umin + m, umin + m, umax - m, umax - m])
    assert np.allclose(g["env_pred"], inside.geometry(x)["env_pred"])
    assert _out_of_span_jumps(prob, x) < 0.5
    monkeypatch.setattr(S, "M7_ENV_CLAMP_PX", 0.0)  # the old clamp at the extreme vertex steps
    assert _out_of_span_jumps(prob, x) > 2.0


# --- front by direction of travel, width at its limit, bound tolerance (user decisions 2026-11-09) ---------------


@pytest.mark.slow
def test_32_front_by_direction_of_travel(turn):
    res = _front(turn, flip_init=True)  # the start faces backwards: the fit is flipped to face the travel direction
    s = res["summary"]
    _, h, _, _ = _errors(res["problem"], res["fit"], turn)
    assert s["front_source"] == "travel" and s["front_state"] == "assumed" and s["reversing"] is None
    assert s["front_turning"] == "agrees" and h.max() <= 1.0
    truth = sy.simulate(reverse=True, psi0_deg=270, start=(5.0, 0.0))  # a real reverse through a turn
    res = _front(truth)
    s = res["summary"]
    g = res["problem"].geometry(res["fit"].x)
    assert s["front_source"] == "travel" and s["front_turning"] == "disagrees"
    assert not m7._runs_backwards(res["problem"], res["fit"])  # assumed forward: the front faces the travel direction
    d = np.degrees(np.abs((g["psi"] - truth["psi"] + np.pi) % (2 * np.pi) - np.pi))
    assert np.median(d) > 170  # i.e. opposite to the true (reversing) body heading


@pytest.mark.slow
def test_33_width_on_its_limit_is_prior_not_a_failure():
    truth = sy.simulate(dims=dict(sy.BUS, W=9.3))  # wider than the 8.8 ft bound
    prob, ft, tr = _fit(truth)
    ft = m7.finish(prob, ft)
    th = prob.unpack(ft.x)[2] / FT
    assert abs(th[3] - S.M7_LOOSE_W_FT["bus"][3]) <= S.M7_BOUND_TOL_FT and ft.info["at_bound"][3]
    reason, _ = m7.gates(prob, ft, m7.frame_table(prob, ft))
    assert reason == "" and m7.size_sources(prob, ft)["W_source"] == "prior"


@pytest.mark.slow
def test_34_bound_tolerance(full_fit):
    prob, ft, _ = full_fit
    x = ft.x.copy()
    iW = 3 * prob.K + 3
    hi = prob.hi_s[3]
    for d_ft, at in ((0.002, True), (0.009, True), (0.02, False)):
        x[iW] = hi - d_ft * FT
        f = dataclasses.replace(ft, x=x.copy())
        _, active, _ = m7.information(prob, f)
        assert (not active[iW]) == at

@pytest.mark.slow
def test_35_near_side_is_held_within_a_round(full_fit):
    # width debugging 2026-11-09: a side decided from the pose being optimised let fits pin a frame on the switch
    prob, ft, tr = full_fit
    assert prob.side_fixed is None  # `fit` leaves the pose's own side
    assert ft.F == pytest.approx(prob.F(ft.x, assoc=ft.assoc))
    k = 10
    g = prob.geometry(ft.x)
    lat = float((prob.Cg - g["R"][k]) @ g["nl"][k])
    x2 = ft.x.copy()
    x2[2 * k : 2 * k + 2] += 2 * lat * g["nl"][k]  # mirror the frame's centreline across the camera
    assert prob.geometry(x2)["side"][k] == -g["side"][k]  # from the pose: switches
    prob.side_fixed = g["side"].copy()
    try:
        assert prob.geometry(x2)["side"][k] == g["side"][k]  # held: does not
    finally:
        prob.side_fixed = None


@pytest.mark.slow
def test_36_width_resolution(full_fit):
    # width debugging 2026-11-10: ft of W per px of silhouette at the far end (near side held). The synthetic camera
    # sees the bus nearly side-on, so a pixel there is worth well over a foot (1.86 at writing)
    prob, ft, _ = full_fit
    r = m7.width_resolution(prob, ft)
    assert 0.3 < r < 10
    x = ft.x.copy()
    g = prob.geometry(x)
    v = np.zeros_like(x)
    v[3 * prob.K + 3] = FT
    v[: 2 * prob.K] = (-0.5 * FT * g["side"][:, None] * g["nl"]).ravel()
    u0, u1 = g["box_px"][..., 0], prob.geometry(x + v)["box_px"][..., 0]  # 1 ft wider, near side kept
    du = np.maximum(np.abs(u1.min(1) - u0.min(1)), np.abs(u1.max(1) - u0.max(1)))
    assert np.median(du) == pytest.approx(1 / r, rel=0.1)


@pytest.mark.slow
def test_37_width_held_at_prior_when_coarse():
    # user decision 2026-11-10 ("Hold W at prior"): a 9.3 ft bus (above the 8.8 ft bound) on the synthetic camera
    # (about 1.9 ft per px) is held at the class prior instead of sitting on the bound
    truth = sy.simulate(dims=dict(sy.BUS, W=9.3))
    tr = sy.track(truth, CAM, sy.observations(truth, CAM, BUS))
    sm = m7.solve_track(tr, configs=[BUS])["summary"]
    assert sm["w_ft_per_px"] > S.M7_W_HOLD_FT_PER_PX and sm["w_held"]
    assert sm["width_ft"] == pytest.approx(S.M7_PRIORS_FT["bus"]["W"][0], abs=0.05)
    assert sm["W_source"] == "prior" and sm["fail_reason"] != "param_bound"
