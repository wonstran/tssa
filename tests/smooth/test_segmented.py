"""Per-row covariance, segment(allow_missing), smooth(segmented=True) and the rts_smooth fixes.

Design docs/2026-11-10-footprint-design.md, section 5; tests 1-5 of section 14.2.
"""

import numpy as np
import pandas as pd
import pytest

from tssa.smooth import TrackSmoother, register_method
from tssa.smooth import methods as M
from tssa.smooth import smoother as S
from tssa.smooth.segments import STRUCT_COLUMNS

from conftest import walker

POS = ["x_smooth", "y_smooth"]


def with_cov(df: pd.DataFrame, sd: float) -> pd.DataFrame:
    out = df.copy()
    out["s"] = sd
    out["cxx"], out["cxy"], out["cyy"] = sd**2, 0.0, sd**2
    return out


# --- 1. covariance -----------------------------------------------------------------------------------------------


def test_cov_equals_sigma_on_detected_rows():
    df = with_cov(walker(n=80, noise=0.1), 0.07)
    a = TrackSmoother("rts", sigma_col="s").smooth(df)
    b = TrackSmoother("rts", cov_cols=("cxx", "cxy", "cyy")).smooth(df)
    pd.testing.assert_frame_equal(a[POS], b[POS])


def test_cov_is_final_on_upstream_filled_rows():
    df = with_cov(walker(n=80, noise=0.1), 0.07)
    df["interp"] = np.where(np.arange(len(df)) % 4 == 0, 1, 0)
    cov = TrackSmoother("rts", cov_cols=("cxx", "cxy", "cyy")).smooth(df)
    sig_unscaled = TrackSmoother("rts", params={"fill_scale": 1.0}, sigma_col="s").smooth(df)
    sig_scaled = TrackSmoother("rts", sigma_col="s").smooth(df)
    pd.testing.assert_frame_equal(cov[POS], sig_unscaled[POS])  # covariance not inflated on interp = 1 rows
    diff = np.abs(cov[POS].to_numpy() - sig_scaled[POS].to_numpy()).max()
    assert diff > 0.01  # sigma is inflated; absolute check (a relative tolerance at UTM scale hides metres)


def test_cov_takes_precedence_and_partial_cov_falls_back_to_sigma():
    df = with_cov(walker(n=60, noise=0.1), 0.07)
    df["s"] = 0.07
    df.loc[5, "cxy"] = np.nan  # this row: no full covariance -> sigma_col
    both = TrackSmoother("rts", sigma_col="s", cov_cols=("cxx", "cxy", "cyy")).smooth(df)
    sig = TrackSmoother("rts", sigma_col="s").smooth(df)
    pd.testing.assert_frame_equal(both[POS], sig[POS])  # equal values here, and the NaN row did not raise


def test_non_positive_definite_cov_raises():
    df = with_cov(walker(n=40), 0.05)
    df.loc[10, "cxy"] = 0.01  # |xy| > sqrt(xx * yy) = 0.0025
    with pytest.raises(ValueError, match="positive definite"):
        TrackSmoother("rts", cov_cols=("cxx", "cxy", "cyy")).smooth(df)


def test_cov_cols_validation():
    with pytest.raises(ValueError, match="three different"):
        TrackSmoother("rts", cov_cols=("a", "b"))
    with pytest.raises(ValueError, match="differ from xy_cols"):
        TrackSmoother("rts", cov_cols=("x", "b", "c"))


def test_cov_ignored_by_kinds_without_noise_model():
    df = with_cov(walker(n=40), 0.05)
    df.loc[10, "cxy"] = 1.0  # not PD, but sg never reads it
    a = TrackSmoother("sg7", cov_cols=("cxx", "cxy", "cyy")).smooth(df)
    b = TrackSmoother("sg7").smooth(df)
    pd.testing.assert_frame_equal(a[POS], b[POS])


# --- 2. segment(allow_missing) -----------------------------------------------------------------------------------


def missing_table() -> pd.DataFrame:
    a = walker(n=60, track=1, drop=(15,))  # frame 15 has no row at all -> inserted (interp = 2)
    nan = a["frame"].isin([0, 1, 2, 10, 58, 59]) | a["frame"].between(25, 49)  # prefix, 1 interior, long run, suffix
    a.loc[nan, ["x", "y"]] = np.nan
    b = walker(n=5, track=2)
    b[["x", "y"]] = np.nan  # a track with no valid position
    return pd.concat([a, b], ignore_index=True)


def test_segment_keeps_every_row_and_returns_structure_only():
    df = missing_table()
    st = TrackSmoother("rts").segment(df, allow_missing=True)
    assert list(st.columns) == STRUCT_COLUMNS
    assert len(st) == len(df) + 1  # every input row once, plus the inserted frame 15
    assert not st.duplicated(["track", "frame"]).any()
    with pytest.raises(ValueError, match="NaN"):
        TrackSmoother("rts").segment(df)


def test_segment_missing_row_rules():
    st = TrackSmoother("rts").segment(missing_table(), allow_missing=True).set_index(["track", "frame"])
    one = st.loc[(1, 10)]
    assert one["seg"] >= 0 and one["fill_reason"] == "no_position" and one["interp"] == 0  # interior, bridged
    assert st.loc[(1, 15), "interp"] == 2 and st.loc[(1, 15), "seg"] == one["seg"]  # inserted frame, same segment
    for f in (0, 1, 2, 58, 59, *range(25, 50)):  # prefix, suffix, run longer than fill_max_gap (20)
        assert st.loc[(1, f), "seg"] == -1 and st.loc[(1, f), "fill_reason"] == "no_position"
        assert st.loc[(1, f), "seg_len"] == 0
    assert (st.loc[2, "seg"] == -1).all()  # no valid position at all
    assert st.loc[(1, 24), "seg"] != st.loc[(1, 50), "seg"]  # the long run split the track
    assert st.loc[(1, 3), "seg_len"] == 22  # frames 3..24, frame 15 inserted


# --- 3. segmented smoothing ---------------------------------------------------------------------------------------


def segmented_input(positions: pd.DataFrame, struct: pd.DataFrame) -> pd.DataFrame:
    """The structure plus this method's positions (NaN where the method has none), as locate() would give it."""
    return struct.merge(positions[["track", "frame", "x", "y"]], how="left", on=["track", "frame"])


@pytest.mark.parametrize("method", ["rts", "ma", "sg7"])
def test_segmented_uses_only_this_calls_positions(method):
    m1 = walker(n=80, drop=(20, 21, 22), noise=0.1)
    m6 = m1.copy()
    m6["x"] += 3.0  # a different method, offset 3 m: the segmentation must not leak M1 into it
    sm = TrackSmoother(method)
    struct = sm.segment(m1)
    seg = sm.smooth(segmented_input(m6, struct), segmented=True)
    legacy = sm.smooth(m6)  # same segmentation (a constant offset changes no jump or gap)
    np.testing.assert_allclose(seg[POS].to_numpy(), legacy[POS].to_numpy(), rtol=0, atol=1e-9)
    assert (seg["fallback"] == 0).all()


def test_segmented_one_measured_row():
    m1 = walker(n=30)
    sm = TrackSmoother("rts")
    struct = sm.segment(m1)
    m6 = m1.copy()
    m6.loc[m6["frame"] != 12, ["x", "y"]] = np.nan
    out = sm.smooth(segmented_input(m6, struct), segmented=True)
    row = out["frame"] == 12
    np.testing.assert_allclose(out.loc[row, POS].to_numpy()[0], m6.loc[m6["frame"] == 12, ["x", "y"]].to_numpy()[0],
                               rtol=0, atol=5e-4)  # fmt: skip
    assert out.loc[~row, POS].isna().all().all()
    assert (out["fallback"] == 1).all()


def test_segmented_failed_method_returns_nan_without_raising():
    m1 = walker(n=30)
    sm = TrackSmoother("rts")
    nothing = m1.assign(x=np.nan, y=np.nan)
    out = sm.smooth(segmented_input(nothing, sm.segment(m1)), segmented=True)
    assert out[POS].isna().all().all() and (out["fallback"] == 1).all()


def test_segmented_minus_one_rows_are_nan():
    df = missing_table()
    sm = TrackSmoother("rts")
    out = sm.smooth(segmented_input(df, sm.segment(df, allow_missing=True)), segmented=True)
    minus = out["seg"] == -1
    assert minus.any() and out.loc[minus, POS].isna().all().all()
    assert out.loc[~minus, POS].notna().all().all()  # inside segments every row gets a position (predicted or not)


def test_segmented_utm_origin_tolerances(monkeypatch):
    register_method("_ident_test", {}, min_len=lambda p: 1)(lambda t, pos, ctx: pos)
    n = 20
    df = pd.DataFrame({"frame": np.arange(n), "track": 1,
                       "x": 587000.123456 + 0.1 * np.arange(n), "y": 5200000.123456 + 0.05 * np.arange(n)})  # fmt: skip
    sm = TrackSmoother("_ident_test")
    inp = segmented_input(df, sm.segment(df))
    truth = df[["x", "y"]].to_numpy()
    monkeypatch.setattr(S, "_round", lambda out, added: out)  # internal: before output rounding
    raw = sm.smooth(inp, segmented=True)
    np.testing.assert_allclose(raw[POS].to_numpy(), truth, rtol=0, atol=1e-6)
    monkeypatch.undo()
    pub = sm.smooth(inp, segmented=True)  # public: three-decimal output, as the legacy path
    np.testing.assert_allclose(pub[POS].to_numpy(), truth, rtol=0, atol=5e-4)
    assert (pub[POS].to_numpy() == np.round(pub[POS].to_numpy(), 3)).all()


def test_segmented_structural_columns_pass_through():
    m1 = walker(n=40, drop=(10,))
    sm = TrackSmoother("rts")
    inp = segmented_input(m1, sm.segment(m1))
    out = sm.smooth(inp, segmented=True)
    assert list(out.columns[: len(inp.columns)]) == list(inp.columns)  # no clash on seg / interp / fill_reason
    assert list(out.columns[len(inp.columns):]) == [*POS, "fallback"]
    for c in ("seg", "interp", "fill_reason"):
        assert out[c].tolist() == inp.sort_values(["track", "frame"])[c].tolist()


def test_segmented_input_validation():
    m1 = walker(n=40)
    sm = TrackSmoother("rts")
    inp = segmented_input(m1, sm.segment(m1))
    with pytest.raises(ValueError, match="structural"):
        sm.smooth(inp.drop(columns="fill_reason"), segmented=True)
    with pytest.raises(ValueError, match="consecutive"):
        sm.smooth(inp[inp["frame"] != 20], segmented=True)
    with pytest.raises(ValueError, match="repeated"):
        sm.smooth(pd.concat([inp, inp.iloc[:1]]), segmented=True)
    with pytest.raises(ValueError, match="ref_tracks"):
        sm.smooth(inp, ref_tracks=m1, segmented=True)


# --- 4. admission follows positions, not interp -------------------------------------------------------------------


def test_finite_position_on_inserted_frame_is_measured():
    m1 = walker(n=40, drop=(20,), noise=0.0)
    sm = TrackSmoother("rts")
    inp = segmented_input(m1, sm.segment(m1))
    ins = inp["frame"] == 20
    assert inp.loc[ins, "interp"].item() == 2
    without = sm.smooth(inp, segmented=True)
    with_pt = inp.copy()
    with_pt.loc[ins, "x"] = m1["x"].mean() + 2.0  # an admitted pose 2 m off the straight path
    with_pt.loc[ins, "y"] = m1["y"].iloc[0]
    used = sm.smooth(with_pt, segmented=True)
    assert abs(used.loc[ins, "x_smooth"].item() - without.loc[ins, "x_smooth"].item()) > 0.01


def test_legacy_mode_still_clears_inserted_rows():
    m1 = walker(n=40, drop=(20,))
    out = TrackSmoother("rts").smooth(m1)
    assert out.loc[out["frame"] == 20, "interp"].item() == 2  # unchanged legacy behaviour


# --- 5. rts_smooth -------------------------------------------------------------------------------------------------


def _old_rts(pos, R, dt, q):
    """rts_smooth as it was in 0.0.2.4 (every caller then started with a measured row)."""
    n = len(pos)
    I2 = np.eye(2)
    F = np.block([[I2, dt * I2], [np.zeros((2, 2)), I2]])
    Q = q * np.block([[dt**3 / 3 * I2, dt**2 / 2 * I2], [dt**2 / 2 * I2, dt * I2]])
    H = np.hstack([I2, np.zeros((2, 2))])
    meas = np.flatnonzero(~np.isnan(pos[:, 0]))
    i0, i1 = meas[0], meas[1]
    x = np.concatenate([pos[i0], (pos[i1] - pos[i0]) / ((i1 - i0) * dt)])
    P = np.diag(M.RTS_P0)
    xp, Pp, xf, Pf = np.zeros((n, 4)), np.zeros((n, 4, 4)), np.zeros((n, 4)), np.zeros((n, 4, 4))
    for k in range(n):
        if k > 0:
            x, P = F @ x, F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        if not np.isnan(pos[k, 0]):
            K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R[k])
            x, P = x + K @ (pos[k] - H @ x), (np.eye(4) - K @ H) @ P
        xf[k], Pf[k] = x, P
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        xs[k] = xf[k] + Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1]) @ (xs[k + 1] - xp[k + 1])
    return xs[:, :2], xs[:, 2:]


@pytest.mark.parametrize("n_meas", [0, 1])
def test_rts_smooth_needs_two_measurements(n_meas):
    pos = np.full((6, 2), np.nan)
    pos[:n_meas] = 0.0
    with pytest.raises(ValueError, match="at least 2"):
        M.rts_smooth(pos, np.tile(np.eye(2), (6, 1, 1)), 0.1, 1.0)


def test_rts_smooth_missing_prefix_is_extrapolated_from_the_first_measurement():
    t = np.arange(40) * 0.1
    pos = np.column_stack([10.0 * t, -2.0 * t])
    pos[:10] = np.nan
    xs, v = M.rts_smooth(pos, np.tile(np.eye(2) * 1e-10, (40, 1, 1)), 0.1, 0.5)
    np.testing.assert_allclose(xs[0], [0.0, 0.0], rtol=0, atol=1e-6)  # 0.0.2.4: 1.54 m off
    np.testing.assert_allclose(v[:10], np.tile([10.0, -2.0], (10, 1)), rtol=0, atol=1e-6)


def test_rts_smooth_unchanged_when_first_row_is_measured():
    rng = np.random.default_rng(1)
    pos = np.cumsum(rng.normal(0, 0.1, (50, 2)), axis=0)
    pos[[7, 8, 30]] = np.nan  # interior gaps only
    R = np.tile(np.eye(2) * 0.01, (50, 1, 1))
    new, old = M.rts_smooth(pos, R, 0.1, 2.0), _old_rts(pos, R, 0.1, 2.0)
    np.testing.assert_array_equal(new[0], old[0])
    np.testing.assert_array_equal(new[1], old[1])
