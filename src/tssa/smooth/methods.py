"""Smoothers for ground trajectories, selectable by name and tunable by parameters.

Inputs:  for one segment, times `t` (n,), raw positions `pos` (n, 2 metres; filled rows hold interpolated
         positions) and a `SegmentContext` (observed positions with NaN on filled rows, per-row sigma, interp flags).
Outputs: `(position, velocity)`, each (n, 2). `REGISTRY` maps a kind (ma, sg, rts) to its function, default
         parameters and minimum length; `PRESETS` maps a name (ma, sg5, sg7, rts) to a kind and parameters.
         `resolve(name, params)` returns the kind and the merged parameters; `register_method` adds a kind.
         `smooth_segment` applies a method to one segment, with the raw fallback for short segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import savgol_filter

FILL_MEAS_SCALE = 4.0  # std-dev multiplier on interp=1 rows (covariance becomes 16x)
RTS_P0 = (1.0, 1.0, 4.0, 4.0)  # initial state variances: e, n, ve, vn


@dataclass
class SegmentContext:
    """Per-segment data some smoothers need beyond the raw positions."""

    obs: np.ndarray  # (n, 2) observed positions, NaN on interp = 2 rows
    sigma: np.ndarray  # (n,) per-row position std-dev in metres, NaN where the user gave none
    interp: np.ndarray  # (n,) 0 detected, 1 filled upstream, 2 filled here
    dt: float


@dataclass
class Method:
    """A registered smoothing kind."""

    fn: Callable
    defaults: dict
    min_len: Callable[[dict], int]
    usable: Callable[[SegmentContext], int]  # rows that count towards min_len


REGISTRY: dict[str, Method] = {}
PRESETS: dict[str, tuple[str, dict]] = {}  # name -> (kind, parameters)


def register_method(name: str, defaults: dict, min_len: Callable[[dict], int], usable=None):
    """Decorator: register `fn(t, pos, ctx, **params) -> (pos, vel)` as smoothing kind `name`."""

    def wrap(fn):
        REGISTRY[name] = Method(fn, dict(defaults), min_len, usable or (lambda ctx: len(ctx.interp)))
        return fn

    return wrap


def register_preset(name: str, kind: str, **params) -> None:
    """Name a kind with fixed parameters, e.g. `sg7` = `sg` with win 7."""
    if kind not in REGISTRY:
        raise ValueError(f"unknown method kind {kind!r}")
    PRESETS[name] = (kind, params)


def resolve(name: str, params: dict | None = None) -> tuple[str, dict]:
    """Kind and merged parameters (kind defaults, preset, then `params`) of a method name; ValueError if unknown."""
    if name in PRESETS:
        kind, preset = PRESETS[name]
    elif name in REGISTRY:
        kind, preset = name, {}
    else:
        raise ValueError(f"unknown method {name!r}; available: {sorted(set(PRESETS) | set(REGISTRY))}")
    merged = {**REGISTRY[kind].defaults, **preset}
    bad = sorted(set(params or {}) - set(merged))
    if bad:
        raise ValueError(f"unknown parameter(s) {bad} for method {name!r}; known: {sorted(merged)}")
    merged.update(params or {})
    return kind, merged


def available_methods() -> dict[str, dict]:
    """Every selectable name with its default parameters."""
    out = {k: dict(m.defaults) for k, m in REGISTRY.items()}
    out.update({name: resolve(name)[1] for name in PRESETS})
    return out


def raw_velocity(t: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """(n, 2) velocity: NaN for n = 1, dpos/dt for n = 2, central differences otherwise."""
    n = len(t)
    if n == 1:
        return np.full((1, 2), np.nan)
    if n == 2:
        return np.repeat((pos[1:] - pos[:1]) / (t[1] - t[0]), 2, axis=0)
    return np.gradient(pos, t, axis=0)


def block_velocity(t: np.ndarray, pos: np.ndarray, n: int) -> np.ndarray:
    """(len(t), 2) block velocity: every `n` samples the displacement to the next block start over the elapsed time.

    All rows of a block carry its velocity; the last block ends at the last sample (shorter, its own duration).
    NaN for a single sample.
    """
    m = len(t)
    out = np.full((m, 2), np.nan)
    for lo in range(0, m - 1, n):
        hi = min(lo + n, m - 1)
        end = lo + n if lo + n < m - 1 else m  # rows of the block; the last block takes the tail
        out[lo:end] = (pos[hi] - pos[lo]) / (t[hi] - t[lo])
    return out


def _check_odd(win: int, order: int | None = None) -> None:
    if win < 3 or win % 2 == 0:
        raise ValueError(f"win must be an odd integer >= 3, got {win}")
    if order is not None and order >= win:
        raise ValueError(f"order ({order}) must be smaller than win ({win})")


@register_method("ma", {"win": 5}, min_len=lambda p: int(p["win"]))
def moving_average(t: np.ndarray, pos: np.ndarray, ctx: SegmentContext, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Centered moving average; edges average over the samples available."""
    _check_odd(win)
    ma = pd.DataFrame(pos).rolling(win, center=True, min_periods=1).mean().to_numpy()
    return ma, raw_velocity(t, ma)


@register_method("sg", {"win": 7, "order": 2, "med": 3}, min_len=lambda p: int(p["win"]))
def median_sg(t: np.ndarray, pos: np.ndarray, ctx: SegmentContext, win: int, order: int, med: int) -> tuple[np.ndarray, np.ndarray]:
    """Median filter (`med` = 1 skips it) then Savitzky-Golay (mode='interp'); smoothed position and velocity."""
    _check_odd(win, order)
    if med < 1 or med % 2 == 0:
        raise ValueError(f"med must be an odd integer >= 1, got {med}")
    m = ndimage.median_filter(pos, size=(med, 1), mode="nearest") if med > 1 else pos
    p = savgol_filter(m, win, order, axis=0, mode="interp")
    v = savgol_filter(m, win, order, deriv=1, delta=ctx.dt, axis=0, mode="interp")
    return p, v


def rts_smooth(pos: np.ndarray, R: np.ndarray, dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant-velocity Kalman filter + RTS smoother; rows with NaN pos get no measurement update."""
    n = len(pos)
    I2 = np.eye(2)
    F = np.block([[I2, dt * I2], [np.zeros((2, 2)), I2]])
    Q = q * np.block([[dt**3 / 3 * I2, dt**2 / 2 * I2], [dt**2 / 2 * I2, dt * I2]])
    H = np.hstack([I2, np.zeros((2, 2))])
    meas = np.flatnonzero(~np.isnan(pos[:, 0]))
    i0, i1 = meas[0], meas[1]
    x0 = np.concatenate([pos[i0], (pos[i1] - pos[i0]) / ((i1 - i0) * dt)])
    xp, Pp = np.zeros((n, 4)), np.zeros((n, 4, 4))
    xf, Pf = np.zeros((n, 4)), np.zeros((n, 4, 4))
    x, P = x0, np.diag(RTS_P0)
    for k in range(n):
        if k > 0:
            x, P = F @ x, F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        if not np.isnan(pos[k, 0]):
            S = H @ P @ H.T + R[k]
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (pos[k] - H @ x)
            P = (np.eye(4) - K @ H) @ P
        xf[k], Pf[k] = x, P
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
    return xs[:, :2], xs[:, 2:]


@register_method(
    "rts", {"q": 0.5, "sigma": 0.05, "fill_scale": FILL_MEAS_SCALE}, min_len=lambda p: 2,
    usable=lambda ctx: int((ctx.interp < 2).sum()),
)  # fmt: skip
def rts(t: np.ndarray, pos: np.ndarray, ctx: SegmentContext, q: float, sigma: float, fill_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """RTS smoother on the observed positions.

    The measurement std-dev is `sigma` metres (the per-row sigma column replaces it where given), times
    `fill_scale` on rows filled upstream (interp = 1); rows filled here get no measurement.
    """
    if q <= 0 or sigma <= 0 or fill_scale <= 0:
        raise ValueError("q, sigma and fill_scale must be positive")
    ok = ctx.interp < 2
    s = np.where(np.isnan(ctx.sigma), sigma, ctx.sigma) * np.where(ctx.interp == 1, fill_scale, 1.0)
    R = np.zeros((len(t), 2, 2))
    R[ok] = (s[ok] ** 2)[:, None, None] * np.eye(2)
    obs = ctx.obs.copy()
    obs[~ok] = np.nan
    return rts_smooth(obs, R, ctx.dt, q)


register_preset("ma", "ma", win=5)
register_preset("sg5", "sg", win=5)
register_preset("sg7", "sg", win=7)
register_preset("rts", "rts")


def smooth_segment(kind: str, params: dict, t: np.ndarray, pos: np.ndarray, ctx: SegmentContext) -> tuple[np.ndarray, np.ndarray, bool]:
    """Apply a method to one segment; a segment shorter than the method's minimum gets the raw values (fallback)."""
    spec = REGISTRY[kind]
    if spec.usable(ctx) < spec.min_len(params):
        return pos, raw_velocity(t, pos), True
    p, v = spec.fn(t, pos, ctx, **params)
    return p, v, False
