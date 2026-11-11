"""Matching, row selection and the cached detection runner (design section 6; tests 20-21 of 14.2). A fake
detector stands in for the models, so no torch or ultralytics is needed."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from tssa.footprint import VehFootPrinter, detect
from tssa.footprint.detect.detectors import CUBOID_COLUMNS, Detector
from tssa.footprint.match import match_frame, xyxy

from synth import AffineGround


# --- 20. matching ------------------------------------------------------------------------------------------------


def test_dense_frame_crossing_boxes_are_assigned_one_to_one():
    tracks = np.array([[0, 0, 10, 10], [6, 0, 16, 10], [12, 0, 22, 10]], float)
    # detection 0 overlaps tracks 0 and 1; greedy best-per-track would give both track 0 and 1 detection 0
    dets = np.array([[3, 0, 13, 10], [0, 0, 9, 10], [13, 0, 23, 10]], float)
    idx, iou = match_frame(tracks, dets, 0.3)
    assert sorted(idx.tolist()) == [0, 1, 2] and len(set(idx.tolist())) == 3
    assert idx.tolist() == [1, 0, 2]


def test_low_iou_is_unmatched_and_preferred_classes_win():
    idx, iou = match_frame(np.array([[0, 0, 10, 10]], float), np.array([[8, 0, 18, 10]], float), 0.3)
    assert idx.tolist() == [-1] and 0 < iou[0] < 0.3
    t = np.array([[0, 0, 10, 10]], float)
    d = np.array([[0, 0, 10, 10], [0, 0, 10, 9]], float)
    assert match_frame(t, d, 0.3, det_cls=[0, 2], prefer=(2, 5, 7))[0].tolist() == [1]  # the vehicle detection


# --- fake detector ------------------------------------------------------------------------------------------------


@dataclass
class FakeCuboids(Detector):
    """Fixed detections per frame, matched over every box of the frame like the real pose detectors."""

    dets: dict = field(default_factory=dict)  # frame -> (n, 4) xyxy
    version: str = "v1"
    table: str = field(default="cuboids", init=False)
    columns: list = field(default_factory=lambda: list(CUBOID_COLUMNS), init=False)
    code_version: str = field(default="fake", init=False)
    loads: int = 0
    calls: int = 0

    def weights_id(self):
        return "fake"

    def params(self):
        return {"version": self.version}

    def load(self):
        self.loads += 1
        return object()

    def infer(self, model, img, req, ctx):
        self.calls += 1
        d = np.asarray(self.dets.get(int(req["frame"].iloc[0]), np.empty((0, 4))), float)
        idx, iou = match_frame(xyxy(ctx[["x", "y", "w", "h"]].to_numpy(float)), d, 0.3)
        where = {int(t): k for k, t in enumerate(ctx["track"])}
        recs = []
        for r in req.itertuples(index=False):
            k = where[int(r.track)]
            i = int(idx[k])
            face = np.tile(d[i, :2], 4) if i >= 0 else np.full(8, np.nan)
            recs.append([int(r.track), int(r.frame), float(iou[k]), *face, int(i >= 0)])
        return recs, {}


def crowded():
    tr = pd.DataFrame({"track": [1, 2, 3], "frame": 0, "x": [0.0, 6, 12], "y": 0.0, "w": 10.0, "h": 10.0,
                       "cls": [2, 2, 5]})  # fmt: skip
    dets = {0: np.array([[3, 0, 13, 10], [0, 0, 9, 10], [13, 0, 23, 10]], float)}
    return tr, dets


# --- 21. row selection --------------------------------------------------------------------------------------------


def test_one_tracks_rows_match_as_in_the_whole_frame():
    tr, dets = crowded()
    images = {0: np.zeros((20, 30, 3), np.uint8)}
    all_rows = detect.run(images, tr, [FakeCuboids(dets)])["cuboids"].set_index("track")
    one = detect.run(images, tr, [FakeCuboids(dets)], rows=tr[tr["track"] == 2])["cuboids"].set_index("track")
    assert list(one.index) == [2]
    assert one.loc[2, "gu0"] == all_rows.loc[2, "gu0"] == 3.0  # matched with the whole frame as context


def test_required_rows_follow_the_chains():
    tr, _ = crowded()
    g = AffineGround()
    req = VehFootPrinter(g).required_rows(tr)
    assert set(req["masks"]["track"]) == {3} and set(req["cuboids"]["track"]) == {1, 2, 3}  # M7 masks: bus only
    req5 = VehFootPrinter(g, chains={2: ("M5", "M2"), 5: ("M2",)}).required_rows(tr)
    assert set(req5["masks"]["track"]) == {1, 2} and "cuboids" not in req5
    assert set(VehFootPrinter(g).required_rows(tr, only_tracks=[1])["cuboids"]["track"]) == {1}


def test_cache_infers_only_missing_rows_and_keys_on_inputs(tmp_path):
    tr, dets = crowded()
    images = {0: np.zeros((20, 30, 3), np.uint8)}
    a = FakeCuboids(dets)
    detect.run(images, tr, [a], rows=tr[tr["track"] == 1], cache_dir=tmp_path)
    b = FakeCuboids(dets)
    out = detect.run(images, tr, [b], cache_dir=tmp_path)  # tracks 2 and 3 are new
    assert b.loads == 1 and b.calls == 1 and len(out["cuboids"]) == 3
    c = FakeCuboids(dets)
    detect.run(images, tr, [c], cache_dir=tmp_path)
    assert c.loads == 0 and c.calls == 0  # everything cached: no model load
    d = FakeCuboids(dets, version="v2")
    detect.run(images, tr, [d], cache_dir=tmp_path)
    assert d.calls == 1  # changed parameters: a new key, not a silent reuse
    assert len(list((tmp_path / "cuboids").iterdir())) == 2


def test_runner_needs_every_requested_frame():
    tr, dets = crowded()
    with pytest.raises(RuntimeError, match="frame 0"):
        detect.run({}, tr, [FakeCuboids(dets)])


def test_keypoint_detector_needs_a_pinned_revision():
    with pytest.raises(ValueError, match="pinned"):
        detect.KeypointDetector(revision="")
