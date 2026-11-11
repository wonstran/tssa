"""Cached detection runner (design docs/2026-11-10-footprint-design.md section 6.4; study infer.py `_run`).

`run(video, tracks, detectors, rows=None, cache_dir=None)`:
- `tracks` is the MATCHING CONTEXT: every box of every frame, always used whole;
- `rows` selects the OUTPUT (and cached) rows: None = every row of `tracks` for every detector; a DataFrame for all
  detectors; or {table name: DataFrame} (e.g. `VehFootPrinter.required_rows(tracks)`);
- frames are read once, in order, and every detector with rows in a frame runs on it;
- cache (optional): `<cache_dir>/<table>/<key>/` with rows.csv, meta.json and (masks) outlines.npz; the key hashes
  the video file, the full track table, the weights, the detector parameters and its code version. A changed input
  gives a new key, never silent reuse; a run infers only rows the cache lacks and saves every SAVE_EVERY frames.
`video` is a path (read with OpenCV) or a mapping {frame: BGR image} (in-memory use and tests).
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..masks import MaskTable, join_contours, split_contours

log = logging.getLogger("tssa.footprint.detect")
SAVE_EVERY = 200
ROW_COLUMNS = ["track", "frame", "x", "y", "w", "h"]


@functools.lru_cache(maxsize=None)
def _sha_cached(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha(path) -> str:
    """sha256 of a file, memoised per (path, size, mtime)."""
    st = os.stat(path)
    return _sha_cached(str(Path(path).resolve()), st.st_size, st.st_mtime_ns)


def table_sha(df: pd.DataFrame) -> str:
    """sha256 of the canonical CSV of track, frame, x, y, w, h (sorted)."""
    t = _norm(df).sort_values(["track", "frame"]).reset_index(drop=True)
    return hashlib.sha256(t.to_csv(index=False, float_format="%.6f").encode()).hexdigest()


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _norm(rows: pd.DataFrame) -> pd.DataFrame:
    r = rows[ROW_COLUMNS].copy()
    r[["track", "frame"]] = r[["track", "frame"]].astype("int64")
    r[["x", "y", "w", "h"]] = r[["x", "y", "w", "h"]].astype(float)
    return r.drop_duplicates(["track", "frame"]).reset_index(drop=True)


def iter_frames(video, frames):
    """Yield (frame, image) for the requested zero-based frames, in order. A mapping is read directly; a path is
    read sequentially with OpenCV, and a frame that cannot be read raises (frames never shift)."""
    wanted = sorted({int(f) for f in frames})
    if wanted and wanted[0] < 0:
        raise ValueError(f"negative frame number {wanted[0]}")
    if isinstance(video, Mapping):
        for f in wanted:
            if f not in video:
                raise RuntimeError(f"frame {f} is not in the supplied images")
            yield f, video[f]
        return
    from .detectors import _need

    cv2 = _need("cv2")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"video could not be opened: {video}")
    pos = 0
    try:
        for f in wanted:
            while pos < f:
                if not cap.grab():
                    raise RuntimeError(f"frame {f} could not be read (the video stops at frame {pos})")
                pos += 1
            ok, img = cap.read()
            if not ok or img is None:
                raise RuntimeError(f"frame {f} could not be read")
            pos += 1
            yield f, img
    finally:
        cap.release()


class _Entry:
    """One detector's cache entry (or an in-memory one without cache_dir)."""

    def __init__(self, det, video_id: str, tracks_id: str, cache_dir):
        self.det = det
        self.inputs = {"table": det.table, "video": video_id, "tracks_sha": tracks_id, "weights_id": det.weights_id(),
                       "params": det.params(), "code_version": det.code_version}  # fmt: skip
        self.key = hashlib.sha256(_canonical(self.inputs)).hexdigest()[:16]
        self.dir = None if cache_dir is None else Path(cache_dir) / det.table / self.key
        self.cols = [*det.columns, "matched"]
        self.df = pd.DataFrame(columns=self.cols)
        self.outlines: dict = {}
        self.lower: dict = {}  # masks: raw-mask lower outline per row
        self.timing: list = []
        if self.dir is not None and (self.dir / "meta.json").exists():
            meta = json.loads((self.dir / "meta.json").read_text())
            if meta.get("inputs") != json.loads(_canonical(self.inputs)):
                raise RuntimeError(f"cache entry {self.dir} does not match its key inputs; refusing to reuse it")
            self.timing = meta.get("timing", [])
            self.df = pd.read_csv(self.dir / "rows.csv").reindex(columns=self.cols)
            self.df[["track", "frame"]] = self.df[["track", "frame"]].astype("int64")
            if (self.dir / "outlines.npz").exists():
                mt = MaskTable.load_npz(self.dir / "outlines.npz")
                self.outlines, self.lower = dict(mt), dict(mt.outlines)
        self.pending: list = []
        self.pending_out: dict = {}

    def have(self) -> pd.MultiIndex:
        if not len(self.df):
            return pd.MultiIndex.from_tuples([], names=["track", "frame"])
        return pd.MultiIndex.from_frame(self.df[["track", "frame"]])

    def flush(self, seconds: float) -> None:
        if not self.pending:
            return
        add = pd.DataFrame(self.pending, columns=self.cols)
        self.df = add if not len(self.df) else pd.concat([self.df, add], ignore_index=True)
        for k, v in self.pending_out.items():
            if isinstance(v, tuple):  # masks: (contours, raw-mask lower outline)
                self.outlines[k], self.lower[k] = v
            else:
                self.outlines[k] = v
        n = len(add)
        self.pending, self.pending_out = [], {}
        if self.dir is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timing.append({"added_rows": n, "seconds": round(seconds, 3),
                            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})  # fmt: skip
        df = self.df.sort_values(["track", "frame"]).reset_index(drop=True)
        _atomic(self.dir / "rows.csv", lambda p: df.to_csv(p, index=False))
        if self.det.table == "masks":
            _atomic(self.dir / "outlines.npz", lambda p: MaskTable(self.outlines, self.lower).save_npz(p))
        meta = {"key": self.key, "inputs": json.loads(_canonical(self.inputs)), "rows": int(len(df)),
                "matched": int(df["matched"].sum()) if len(df) else 0, "timing": self.timing}  # fmt: skip
        _atomic(self.dir / "meta.json", lambda p: p.write_text(json.dumps(meta, indent=2)))


def _atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp" + path.suffix)
    write(tmp)
    os.replace(tmp, path)


def run(video, tracks: pd.DataFrame, detectors, rows=None, cache_dir=None, models: dict | None = None) -> dict:
    """{table: result} for each detector: keypoints and cuboids as DataFrames of matched rows, masks as a MaskTable.

    `models`: optional {table: loaded model}, to skip loading. See the module docstring for the rest.
    """
    ctx_all = _norm(tracks)
    tracks_id = table_sha(ctx_all)
    video_id = file_sha(video) if not isinstance(video, Mapping) else "in-memory"
    entries = []
    for det in detectors:
        want = rows.get(det.table) if isinstance(rows, Mapping) else rows
        want = ctx_all if want is None else _norm(want)
        e = _Entry(det, video_id, tracks_id, cache_dir)
        e.want = want
        e.todo = want[~pd.MultiIndex.from_frame(want[["track", "frame"]]).isin(e.have())]
        entries.append(e)
    frames = sorted(set().union(*[set(e.todo["frame"].unique()) for e in entries])) if entries else []
    if frames:
        loaded = dict(models or {})
        by = {id(e): {f: g for f, g in e.todo.groupby("frame")} for e in entries}
        for e in entries:
            if len(e.todo) and e.det.table not in loaded:
                loaded[e.det.table] = e.det.load()
        ctx_by = {f: g.sort_values("track", kind="stable").reset_index(drop=True)
                  for f, g in ctx_all[ctx_all["frame"].isin(frames)].groupby("frame")}  # fmt: skip
        t0, n = time.perf_counter(), 0
        try:
            for f, img in iter_frames(video, frames):
                for e in entries:
                    req = by[id(e)].get(f)
                    if req is None:
                        continue
                    ctx = pd.concat([ctx_by.get(f, req.iloc[:0]), req]).drop_duplicates(["track", "frame"])
                    ctx = ctx.sort_values("track", kind="stable").reset_index(drop=True)
                    recs, outs = e.det.infer(loaded[e.det.table], img, req, ctx)
                    e.pending.extend(recs)
                    e.pending_out.update(outs)
                n += 1
                if n % SAVE_EVERY == 0:
                    for e in entries:
                        e.flush(time.perf_counter() - t0)
                    log.info("detect: %d of %d frames", n, len(frames))
        finally:
            for e in entries:
                e.flush(time.perf_counter() - t0)
    out = {}
    for e in entries:
        key = pd.MultiIndex.from_frame(e.want[["track", "frame"]])
        got = e.df[pd.MultiIndex.from_frame(e.df[["track", "frame"]]).isin(key)] if len(e.df) else e.df
        got = got.sort_values(["track", "frame", "iou"], ascending=[True, True, False])
        got = got.drop_duplicates(["track", "frame"]).reset_index(drop=True)
        if e.det.table == "masks":
            keys = set(map(tuple, got.loc[got["matched"].astype(bool), ["track", "frame"]].to_numpy()))
            out["masks"] = MaskTable({k: v for k, v in e.outlines.items() if k in keys},
                                     {k: v for k, v in e.lower.items() if k in keys})
        else:
            m = got[got["matched"].astype(bool)][e.det.columns].reset_index(drop=True)
            m[["track", "frame"]] = m[["track", "frame"]].astype("int64")
            out[e.det.table] = m
        if e.dir is not None:
            log.info("detect: %s cache %s", e.det.table, e.dir)
    return out


__all__ = ["run", "iter_frames", "file_sha", "table_sha", "join_contours", "split_contours"]
