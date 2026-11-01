"""TrackSmoother: method selection, parameters, gap handling, fallback, compare, column names."""

import numpy as np
import pandas as pd
import pytest

from tssa.smooth import SmoothConfig, TrackSmoother, available_methods, register_method
from tssa.smooth import methods as M

from conftest import E0, N0, walker

OUT_COLS = ["x_smooth", "y_smooth", "vx", "vy", "speed", "seg", "interp", "fill_reason", "fallback"]


def test_smooth_columns_and_coordinates():
    out = TrackSmoother().smooth(walker())
    assert set(OUT_COLS) <= set(out.columns) and {"conf", "cls"} <= set(out.columns)  # extras carried through
    assert len(out) == 60 and out["x_smooth"].between(E0 - 1, E0 + 7).all()  # input coordinates, not local
    assert (out["fallback"] == 0).all() and (out["interp"] == 0).all()


def test_speed_matches_ground_truth():
    out = TrackSmoother("sg7", units="metric").smooth(walker(noise=0.0))
    assert np.allclose(out["speed"].iloc[5:-5], 1.0, atol=0.01)


@pytest.mark.parametrize("method", ["ma", "sg5", "sg7", "rts"])
def test_every_method_reduces_jitter(method):
    tr = walker(noise=0.1)
    raw = TrackSmoother().compare(tr, ["ma"])["speed_raw"].std()
    assert TrackSmoother(method).smooth(tr)["speed"].std() < raw


def test_params_override_and_unknown_rejected():
    a = TrackSmoother("sg", {"win": 9}).smooth(walker())
    b = TrackSmoother("sg", {"win": 5}).smooth(walker())
    assert (a["x_smooth"] - b["x_smooth"]).abs().max() > 1e-3
    with pytest.raises(ValueError, match="unknown method"):
        TrackSmoother("nope")
    with pytest.raises(ValueError, match="unknown parameter"):
        TrackSmoother("sg7", {"window": 9})
    with pytest.raises(ValueError, match="unknown setting"):
        TrackSmoother(fpss=10)
    with pytest.raises(ValueError, match="odd"):
        TrackSmoother("sg", {"win": 6}).smooth(walker())


def test_available_methods_lists_presets_and_kinds():
    av = available_methods()
    assert {"ma", "sg", "sg5", "sg7", "rts"} <= set(av)
    assert av["sg7"]["win"] == 7 and av["rts"]["q"] == 0.5 and av["rts"]["sigma"] == 0.05


def test_short_segment_falls_back_to_raw():
    out = TrackSmoother("sg7").smooth(walker(n=4))
    assert (out["fallback"] == 1).all()
    assert np.allclose(out["x_smooth"], walker(n=4)["x"], rtol=0, atol=1e-3)


def test_gap_filled_conservatively_without_reference():
    out = TrackSmoother("sg7").smooth(walker(drop=range(20, 25)))
    assert len(out) == 60 and (out["interp"] == 2).sum() == 5
    assert out["seg"].nunique() == 1 and set(out.loc[out["interp"] == 2, "fill_reason"]) == {"unverified"}
    assert out.loc[out["interp"] == 2, ["conf", "cls"]].isna().all().all()


def test_fill_gaps_off_cuts_the_track():
    out = TrackSmoother(fill_gaps=False).smooth(walker(drop=range(20, 25)))
    assert len(out) == 55 and out["seg"].nunique() == 2


def test_long_gap_and_fast_gap_not_filled():
    assert TrackSmoother(fill_max_gap=3).smooth(walker(drop=range(20, 25)))["seg"].nunique() == 2
    assert TrackSmoother(gap_max_speed=0.5).smooth(walker(drop=range(20, 25)))["seg"].nunique() == 2


def test_jump_splits_even_without_gap():
    tr = pd.concat([walker(n=30), walker(n=30, start=30).assign(x=lambda d: d["x"] + 50)], ignore_index=True)
    assert TrackSmoother().smooth(tr)["seg"].nunique() == 2
    assert TrackSmoother(split_speed=1000).smooth(tr)["seg"].nunique() == 1


def test_reference_marks_jump_gap_unfilled_and_plain_gap_overlap():
    tr = walker(drop=range(20, 25))
    out = TrackSmoother().smooth(tr, ref_tracks=walker())
    assert set(out.loc[out["interp"] == 2, "fill_reason"]) == {"overlap"}
    ref_jump = walker()
    ref_jump.loc[ref_jump["frame"] >= 25, "x"] += 4.0  # 4.6 m over the 6-frame gap = 7.7 m/s in the reference
    ref_jump = ref_jump[~ref_jump["frame"].between(20, 24)]
    out = TrackSmoother().smooth(tr, ref_tracks=ref_jump)
    assert out["seg"].nunique() == 2 and (out["interp"] == 2).sum() == 0


def test_compare_matches_smooth_per_method():
    tr = walker(drop=range(20, 25))
    cmp = TrackSmoother().compare(tr)
    for m in ("ma", "sg5", "sg7", "rts"):
        one = TrackSmoother(m).smooth(tr)
        assert np.allclose(cmp[f"x_{m}"], one["x_smooth"], rtol=0, atol=1e-3, equal_nan=True)
        assert np.allclose(cmp[f"speed_{m}"], one["speed"], equal_nan=True)
        assert (cmp[f"fb_{m}"] == one["fallback"]).all()
    assert {"vx_raw", "vy_raw", "speed_raw"} <= set(cmp.columns)


def test_compare_accepts_parameter_dict():
    cmp = TrackSmoother().compare(walker(), {"sg": {"win": 9}, "rts": {"q": 0.1}})
    assert {"x_sg", "x_rts"} <= set(cmp.columns)


def test_rts_runs_through_gap():
    out = TrackSmoother("rts", units="metric").smooth(walker(drop=range(20, 25), noise=0.0))
    assert out["speed"].notna().all() and np.allclose(out["speed"], 1.0, atol=0.05)


def test_rts_sigma_param_and_sigma_column_change_the_fit():
    tr = walker(noise=0.1)
    lo = TrackSmoother("rts", {"sigma": 0.01}).smooth(tr)
    hi = TrackSmoother("rts", {"sigma": 1.0}).smooth(tr)
    assert hi["speed"].std() < lo["speed"].std()  # larger measurement noise, smoother result
    tr["sd"] = 1.0
    col = TrackSmoother("rts", {"sigma": 0.01}, sigma_col="sd").smooth(tr)
    assert np.allclose(col["speed"], hi["speed"], atol=1e-3)


def test_two_tracks_are_smoothed_independently():
    tr = pd.concat([walker(track=1), walker(track=2, seed=3).assign(y=lambda d: d["y"] + 100)], ignore_index=True)
    out = TrackSmoother().smooth(tr)
    assert out.groupby("track").size().tolist() == [60, 60]


def test_duplicates_dropped_and_minimal_columns():
    tr = walker()[["frame", "track", "x", "y"]]
    out = TrackSmoother().smooth(pd.concat([tr, tr.iloc[:5]], ignore_index=True))
    assert len(out) == 60


def test_empty_input_and_bad_input():
    empty = TrackSmoother().smooth(walker().iloc[:0])
    assert empty.empty and set(OUT_COLS) <= set(empty.columns)
    with pytest.raises(ValueError, match="required"):
        TrackSmoother().smooth(walker().drop(columns=["y"]))
    with pytest.raises(ValueError, match="NaN"):
        TrackSmoother().smooth(walker().assign(x=np.nan))


def test_custom_position_columns_keep_their_names():
    tr = walker().rename(columns={"x": "e", "y": "n"})
    out = TrackSmoother(xy_cols=("e", "n")).smooth(tr)
    assert {"e", "n", "e_smooth", "n_smooth"} <= set(out.columns) and "x" not in out.columns
    cmp = TrackSmoother(xy_cols=("e", "n")).compare(tr, ["sg7"])
    assert {"e_sg7", "n_sg7"} <= set(cmp.columns)
    with pytest.raises(ValueError, match="required"):
        TrackSmoother().smooth(tr)  # default names x, y are absent
    with pytest.raises(ValueError, match="xy_cols"):
        TrackSmoother(xy_cols=("e", "e"))


def test_unrelated_input_columns_named_x_y_are_carried():
    tr = walker().rename(columns={"x": "e", "y": "n"}).assign(x=5.0, y=6.0)
    out = TrackSmoother(xy_cols=("e", "n")).smooth(tr)
    assert (out["x"] == 5.0).all() and (out["y"] == 6.0).all()


def test_upstream_interp_flag_is_kept():
    tr = walker().assign(interp=0)
    tr.loc[10:14, "interp"] = 1
    out = TrackSmoother("rts").smooth(tr)
    assert (out["interp"] == 1).sum() == 5


def test_register_custom_method():
    @register_method("hold", {"k": 1}, min_len=lambda p: 2)
    def hold(t, pos, ctx, k):
        return pos, M.raw_velocity(t, pos)

    try:
        out = TrackSmoother("hold").smooth(walker())
        assert (out["fallback"] == 0).all()
    finally:
        M.REGISTRY.pop("hold")


def test_config_is_per_instance():
    a, b = TrackSmoother(fps=10), TrackSmoother(fps=20)
    assert a.config.fps == 10 and b.config.fps == 20 and SmoothConfig().fps == 10.0


def test_output_is_input_columns_first_then_added_columns():
    tr = walker(drop=range(20, 25))[["frame", "track", "cls", "x", "y", "conf"]]
    out = TrackSmoother().smooth(tr)
    assert list(out.columns[:6]) == list(tr.columns)  # input fields, same order
    assert list(out.columns[6:]) == ["interp", "seg", "fill_reason", "x_smooth", "y_smooth", "vx", "vy", "speed", "fallback"]
    det = out["interp"] == 0
    assert np.array_equal(out.loc[det, "x"].to_numpy(), tr["x"].to_numpy())  # input positions untouched
    assert out.loc[~det, ["x", "y", "conf", "cls"]].isna().all().all()
    assert out["x_smooth"].notna().all()


def test_input_interp_stays_in_place_and_clashing_names_raise():
    tr = walker(drop=range(20, 25)).assign(interp=0)
    out = TrackSmoother().smooth(tr)
    assert list(out.columns).index("interp") == list(tr.columns).index("interp")
    with pytest.raises(ValueError, match="clash"):
        TrackSmoother().smooth(walker().assign(speed=1.0))


def test_units_default_imperial_and_metric_option():
    tr = walker(noise=0.0)
    mph = TrackSmoother("sg7").smooth(tr)
    ms = TrackSmoother("sg7", units="metric").smooth(tr)
    assert mph.attrs["speed_unit"] == "mph" and ms.attrs["speed_unit"] == "m/s"
    assert np.allclose(ms["speed"].iloc[5:-5], 1.0, atol=0.01)
    assert np.allclose(mph["speed"].iloc[5:-5], 2.237, atol=0.02)
    assert np.allclose(mph[["vx", "vy", "speed"]], ms[["vx", "vy", "speed"]] * 2.23694, atol=0.002)
    assert np.allclose(mph["x_smooth"], ms["x_smooth"])  # positions are not converted
    cmp = TrackSmoother().compare(tr, ["sg7"])
    assert np.allclose(cmp["speed_sg7"].iloc[5:-5], 2.237, atol=0.02) and cmp.attrs["speed_unit"] == "mph"


def test_units_do_not_change_segmentation_thresholds():
    tr = walker(drop=range(20, 25))
    assert TrackSmoother(gap_max_speed=0.5).smooth(tr)["seg"].nunique() == 2  # still m/s
    assert (TrackSmoother().smooth(tr)["interp"] == 2).sum() == 5


def test_bad_units_raise():
    with pytest.raises(ValueError, match="units"):
        TrackSmoother(units="si")


def test_speed_interval_blocks():
    tr = walker(n=23, noise=0.0)
    out = TrackSmoother("sg7", units="metric", speed_interval=5).smooth(tr)
    assert np.allclose(out["speed"], 1.0, atol=0.01)
    # the displacement of the smoothed positions over each block, divided by the block's duration
    pos = out[["x_smooth", "y_smooth"]].to_numpy()
    starts = [0, 5, 10, 15, 20]  # the last block (rows 20-22) is 2 frames long and uses its own duration
    for lo in starts:
        hi = min(lo + 5, 22)
        v = np.hypot(*(pos[hi] - pos[lo])) / ((hi - lo) / 10)
        end = lo + 5 if lo + 5 < 22 else 23
        assert np.allclose(out["speed"].iloc[lo:end], v, atol=0.002)  # constant within the block


def test_speed_interval_changes_with_motion_and_values_are_constant_per_block():
    tr = walker(n=40, noise=0.0)
    tr.loc[tr["frame"] >= 20, "x"] = tr.loc[tr["frame"] >= 20, "x"].iloc[0]  # stands still from frame 20 on
    out = TrackSmoother("sg5", units="metric", speed_interval=10).smooth(tr)
    assert out["speed"].iloc[0:10].nunique() == 1 and out["speed"].iloc[10:20].nunique() == 1
    assert out["speed"].iloc[25] < 0.1 and out["speed"].iloc[2] > 0.8


def test_speed_interval_off_by_default_and_validated():
    tr = walker()
    assert TrackSmoother().smooth(tr)["speed"].nunique() > 10  # per-frame values
    for bad in (1, 0, 2.5):
        with pytest.raises(ValueError, match="speed_interval"):
            TrackSmoother(speed_interval=bad)


def test_speed_interval_per_segment_and_compare_and_short_segment():
    tr = pd.concat([walker(n=12, noise=0.0), walker(n=12, start=40, noise=0.0, track=2)], ignore_index=True)
    out = TrackSmoother(units="metric", speed_interval=5).compare(tr, ["sg5", "sg7"])
    assert np.allclose(out["speed_sg5"], 1.0, atol=0.01) and np.allclose(out["speed_raw"], 1.0, atol=0.01)
    one = TrackSmoother(units="metric", speed_interval=5).smooth(walker(n=1))
    assert one["speed"].isna().all()
