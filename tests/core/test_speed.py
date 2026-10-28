"""Tests for tssa.core.speed.SpeedMeterByTransformation."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

# speed.py imports dnt.track.Tracker, which is only needed for frame infill
if "dnt.track" not in sys.modules:
    dnt = types.ModuleType("dnt")
    dnt_track = types.ModuleType("dnt.track")
    dnt_track.Tracker = object
    dnt.track = dnt_track
    sys.modules.setdefault("dnt", dnt)
    sys.modules["dnt.track"] = dnt_track

from tssa.core import speed as speed_module  # noqa: E402

SOURCE = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
# 1 px = 1 world unit
DESTINATION = SOURCE.copy()


def _fake_match(tracks, zone, **kwargs):
    out = tracks.copy()
    out["in_zone"] = True
    return out


@pytest.fixture
def tracks():
    # Moves 1 world unit per frame at 30 fps -> 30 units/s
    frames = np.arange(10)
    return pd.DataFrame({
        "track_id": 1,
        "frame": frames,
        "ref_x": 10.0 + frames,
        "ref_y": 50.0,
    })


@pytest.mark.parametrize("distance_unit, expected_mph", [("ft", 20), ("m", 67)])
def test_speed_respects_distance_unit(monkeypatch, tracks, distance_unit, expected_mph):
    monkeypatch.setattr(speed_module, "match_zone_by_point", _fake_match)
    meter = speed_module.SpeedMeterByTransformation(
        SOURCE, DESTINATION, fps=30, distance_unit=distance_unit)
    result = meter.detect(tracks, verbose=False)
    # 30 ft/s = 20.45 mph; 30 m/s = 67.1 mph
    assert (result["speed"] == expected_mph).all()


def test_invalid_distance_unit():
    with pytest.raises(ValueError):
        speed_module.SpeedMeterByTransformation(SOURCE, DESTINATION, distance_unit="yd")
