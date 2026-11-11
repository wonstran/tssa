"""M7 speed-ups that must not change results (2026-11-11): the numpy sparsity pattern and `_FDJac` against scipy's
2-point sparse finite differences, bit for bit."""

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

import m7_synth as sy
from tssa.footprint.locators.m7 import solver as m7

EMPTY = (np.array([], int),) * 3


@pytest.fixture(scope="module", params=["bus", "truck", "pickup_van"])
def problem(request):
    truth = sy.simulate(K=24)
    cam = sy.camera()
    cfg = m7.Config(request.param)
    prob = m7.Problem(sy.track(truth, cam, sy.observations(truth, cam, cfg)), cfg)
    lo, hi = prob.bounds()
    x = np.clip(m7.initial_x(prob), lo + 1e-9, hi - 1e-9)
    x[: 3 * prob.K] += np.random.default_rng(0).normal(0, 0.05, 3 * prob.K)
    return prob, x, lo, hi


def _lil_pattern(prob, assoc):
    """The study's construction of the pattern, row by row."""
    from scipy.sparse import lil_matrix

    K, ns, M = prob.K, prob.ns, prob.env_u.shape[1]
    nm = (K - 1) + 2 * max(K - 2, 0) + max(K - 2, 0) + (K - 1)
    sp = lil_matrix((2 * len(assoc[0]) + K * M + 2 * K + nm + ns, prob.n), dtype=int)
    st = list(range(3 * K, 3 * K + ns))
    fc = lambda k: [2 * k, 2 * k + 1, 2 * K + k]  # noqa: E731
    r = 0
    for rows in ([fc(k) + st for k in assoc[0] for _ in range(2)] + [fc(k) + st for k in range(K) for _ in range(M)]
                 + [fc(k) + st for k in range(K) for _ in range(2)] + [fc(k) + fc(k + 1) for k in range(K - 1)]
                 + [[2 * (k + i) + j for i in range(3)] for k in range(K - 2) for j in range(2)]
                 + [[2 * K + k + i for i in range(3)] for k in range(K - 2)]
                 + [fc(k) + fc(k + 1) + [3 * K] for k in range(K - 1)] + [st] * ns):  # fmt: skip
        sp[r, rows] = 1
        r += 1
    return sp.tocsr()


@pytest.mark.parametrize("matched", [False, True])
def test_pattern_is_the_studys(problem, matched):
    prob, x, _, _ = problem
    assoc = prob.associate(prob.geometry(x)) if matched else EMPTY
    a, b = _lil_pattern(prob, assoc), prob.jac_sparsity(assoc)
    a.sort_indices(), b.sort_indices()
    assert a.shape == b.shape and np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)


@pytest.mark.parametrize("matched, at_bound", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("which", ["raw_residuals", "residuals"])
def test_fd_jacobian_equals_scipys_bit_for_bit(problem, matched, at_bound, which):
    prob, x, lo, hi = problem
    x = x.copy()
    if at_bound:  # forces scipy's bounds adjustment of the step (a backward step on the last static)
        j = np.flatnonzero(np.isfinite(hi))[-1]
        x[j] = hi[j] - 1e-12
    assoc = prob.associate(prob.geometry(x)) if matched else EMPTY
    fun = getattr(prob, which)
    args = (assoc,) if which == "raw_residuals" else (assoc, prob.weights(x, assoc))
    a = approx_derivative(fun, x, method="2-point", args=args, sparsity=prob.jac_sparsity(assoc), bounds=(lo, hi))
    b = m7._FDJac(prob, fun, args, assoc, lo, hi)(x)
    a = a.tocsr()
    a.sort_indices()
    assert np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
    assert np.array_equal(a.data, b.data)


def test_memoised_residuals_are_copies(problem):
    # least_squares scales f in place under a robust loss; the Jacobian must still see the unscaled f
    prob, x, lo, hi = problem
    jac = m7._FDJac(prob, prob.raw_residuals, (EMPTY,), EMPTY, lo, hi)
    f = jac.f(x)
    f *= 2.0
    assert np.array_equal(jac.f(x), prob.raw_residuals(x, EMPTY))
