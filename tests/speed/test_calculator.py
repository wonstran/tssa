"""SpeedCalculator: per-frame and block speeds from smoothed positions, units, segments, columns."""

import numpy as np
import pandas as pd
import pytest

from tssa.smooth import TrackSmoother
from tssa.speed import SpeedCalculator

from conftest import walker

MPH = 2.2369362920544


def test_per_frame_speed_of_a_straight_walk():
    out = SpeedCalculator(units="metric").calculate(walker(noise=0.0))
    assert np.allclose(out["speed"], 1.0, atol=0.001) and np.allclose(out["vx"], 1.0, atol=0.001)
    assert out["vy"].abs().max() < 1e-6
    assert out.attrs["speed_unit"] == "m/s"


def test_units_default_imperial_and_metric_option():
    tr = walker(noise=0.0)
    mph = SpeedCalculator().calculate(tr)
    ms = SpeedCalculator(units="metric").calculate(tr)
    assert mph.attrs["speed_unit"] == "mph"
    assert np.allclose(mph["speed"], MPH, atol=0.002)
    assert np.allclose(mph[["vx", "vy", "speed"]], ms[["vx", "vy", "speed"]] * MPH, atol=0.002)
    with pytest.raises(ValueError, match="units"):
        SpeedCalculator(units="si")


def test_fps_scales_the_speed():
    out = SpeedCalculator(fps=20, units="metric").calculate(walker(noise=0.0))  # same 0.1 m / frame at 20 fps
    assert np.allclose(out["speed"], 2.0, atol=0.001)


def test_input_is_kept_and_columns_are_added():
    tr = walker(noise=0.0)
    out = SpeedCalculator().calculate(tr)
    assert list(out.columns) == [*tr.columns, "vx", "vy", "speed"] and len(out) == len(tr)
    assert out.index.equals(tr.index)
    with pytest.raises(ValueError, match="clash"):
        SpeedCalculator().calculate(tr.assign(speed=1.0))
    with pytest.raises(ValueError, match="required"):
        SpeedCalculator(xy_cols=("e", "n")).calculate(tr)


def test_interval_blocks_hold_one_value_and_match_displacement():
    tr = walker(n=23, noise=0.0)
    tr.loc[tr["frame"] >= 10, "x"] = tr.loc[tr["frame"] == 10, "x"].iloc[0]  # stands still from frame 10 on
    out = SpeedCalculator(units="metric", interval=5).calculate(tr)
    s = out["speed"].to_numpy()
    assert np.allclose(s[0:5], 1.0, atol=0.001)  # block [0, 5): 0.5 m in 0.5 s
    assert np.allclose(s[5:10], 1.0, atol=0.001)  # block [5, 10): still walking
    assert np.allclose(s[10:], 0.0, atol=0.001)  # the rest, including the short last block
    assert pd.Series(s[0:5]).nunique() == 1


def test_interval_last_block_is_shorter_and_uses_its_duration():
    out = SpeedCalculator(units="metric", interval=5).calculate(walker(n=23, noise=0.0))
    assert np.allclose(out["speed"], 1.0, atol=0.001)  # last block has 2 frames: 0.2 m / 0.2 s, not / 0.5 s
    lone = SpeedCalculator(units="metric", interval=5).calculate(walker(n=11, noise=0.0))
    assert np.allclose(lone["speed"], 1.0, atol=0.001)  # a lone final frame joins the block before it


def test_interval_validation():
    for bad in (1, 0, 2.5):
        with pytest.raises(ValueError, match="interval"):
            SpeedCalculator(interval=bad)
    with pytest.raises(ValueError, match="fps"):
        SpeedCalculator(fps=0)
    with pytest.raises(ValueError, match="xy_cols"):
        SpeedCalculator(xy_cols=("x", "x"))


def test_velocity_does_not_cross_segments_or_tracks():
    a, b = walker(n=20, noise=0.0), walker(n=20, start=20, noise=0.0).assign(x=lambda d: d["x"] + 30.0)
    tr = pd.concat([a, b], ignore_index=True).assign(seg=lambda d: (d["frame"] >= 20).astype(int))
    out = SpeedCalculator(units="metric").calculate(tr)
    assert np.allclose(out["speed"], 1.0, atol=0.001)  # no 30 m jump between the two segments
    no_seg = SpeedCalculator(units="metric").calculate(tr.drop(columns="seg"))
    assert no_seg["speed"].max() > 5  # without seg the jump is inside one track
    two = pd.concat([a, a.assign(track=2)], ignore_index=True)
    assert np.allclose(SpeedCalculator(units="metric").calculate(two)["speed"], 1.0, atol=0.001)


def test_missing_frames_use_real_time_and_nan_positions_stay_nan():
    tr = walker(n=30, noise=0.0, drop=range(10, 14))  # 4 frames missing, positions on the same line
    out = SpeedCalculator(units="metric").calculate(tr)
    assert np.allclose(out["speed"], 1.0, atol=0.001)
    holes = tr.copy()
    holes.loc[5, ["x", "y"]] = np.nan
    out = SpeedCalculator(units="metric", xy_cols=("x", "y")).calculate(holes)
    assert np.isnan(out.loc[5, "speed"]) and np.allclose(out.drop(index=5)["speed"], 1.0, atol=0.001)


def test_short_and_unsorted_input():
    one = SpeedCalculator().calculate(walker(n=1))
    assert one["speed"].isna().all()
    two = SpeedCalculator(units="metric").calculate(walker(n=2, noise=0.0))
    assert np.allclose(two["speed"], 1.0, atol=0.001)
    shuffled = walker(noise=0.0).sample(frac=1.0, random_state=1)
    out = SpeedCalculator(units="metric").calculate(shuffled)
    assert out.index.equals(shuffled.index) and np.allclose(out["speed"], 1.0, atol=0.001)
    with pytest.raises(ValueError, match="repeated"):
        SpeedCalculator().calculate(pd.concat([walker(n=5), walker(n=5)], ignore_index=True))


def test_speed_from_the_smoother_output_end_to_end():
    tr = walker(noise=0.05, drop=range(20, 25))
    sm = TrackSmoother("sg7", xy_cols=("x", "y")).smooth(tr)
    out = SpeedCalculator(xy_cols=("x_smooth", "y_smooth"), units="metric").calculate(sm)
    assert len(out) == 60 and out["speed"].notna().all()  # the filled rows have a speed too
    assert abs(out["speed"].mean() - 1.0) < 0.05
    cmp = TrackSmoother().compare(tr, ["rts"])
    rts = SpeedCalculator(xy_cols=("x_rts", "y_rts"), units="metric", interval=10).calculate(cmp)
    assert abs(rts["speed"].mean() - 1.0) < 0.05
