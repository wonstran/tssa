"""Model wrappers (optional install `tssa[detect]`): keypoints, 3-D cuboids, SAM masks.

Ported from the speed study's infer.py (snapshot docs/study-snapshot-2026-11-10.json). Every model import is lazy.
Each detector turns one frame into records for the requested rows; matching is one-to-one over every track box of
the frame (`ctx`), so a row's result does not depend on which rows were requested (run report R21).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..masks import lower_outline
from ..match import match_frame, xyxy

KP_REVISION = "48b5795f3b79cdfe7ff6588812aa388a6a7fdd9d"  # the study's pinned HF revision of kiselyovd/vehicle-keypoints
KP_COLUMNS = ["track", "frame", "iou", "u0", "v0", "u1", "v1", "u2", "v2", "u3", "v3"]
CUBOID_COLUMNS = ["track", "frame", "iou", "gu0", "gv0", "gu1", "gv1", "gu2", "gv2", "gu3", "gv3"]
MASK_COLUMNS = ["track", "frame", "iou", "n_comp", "area_px"]


def _need(module: str):
    try:
        return __import__(module)
    except ImportError as exc:
        raise ImportError(f"the detectors need {module}: pip install tssa[detect] (torch, ultralytics, "
                          f"opencv-python, huggingface_hub)") from exc  # fmt: skip


def _np(t) -> np.ndarray:
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)


class Detector:
    """Base: `table` is the output table name; `columns` the record layout; `infer` does one frame."""

    table: str = ""
    columns: list = []
    code_version: str = ""

    def weights_id(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def params(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def load(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def infer(self, model, img, req, ctx) -> tuple[list, dict]:  # pragma: no cover - abstract
        """(records with `columns` plus `matched`, {(track, frame): contours}) for the requested rows `req`;
        `ctx` holds every track box of the frame (track, frame, x, y, w, h), sorted by track."""
        raise NotImplementedError


def _pose_infer(model, img, req, ctx, n_keep, conf, imgsz, min_iou, prefer=None):
    res = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    has = res.boxes is not None and len(res.boxes) > 0 and res.keypoints is not None
    boxes = _np(res.boxes.xyxy).reshape(-1, 4) if has else np.empty((0, 4))
    cls = _np(res.boxes.cls) if has else None
    kpts = _np(res.keypoints.data) if has else None
    idx, ious = match_frame(xyxy(ctx[["x", "y", "w", "h"]].to_numpy(float)), boxes, min_iou, det_cls=cls, prefer=prefer)
    where = {int(t): k for k, t in enumerate(ctx["track"].to_numpy())}
    recs = []
    for r in req.itertuples(index=False):
        k = where[int(r.track)]
        i, iou = int(idx[k]), float(ious[k])
        pts = kpts[i, :n_keep, :2].reshape(-1) if i >= 0 else np.full(2 * n_keep, np.nan)
        recs.append([int(r.track), int(r.frame), iou, *[float(v) for v in pts], int(i >= 0)])
    return recs, {}


def _ckpt_id(path) -> str:
    from .runner import file_sha

    return f"{Path(path).name}:{file_sha(path)}"


@dataclass
class KeypointDetector(Detector):
    """CarFusion 14-point YOLO pose from Hugging Face at a pinned revision; wheels 0-3 kept (RF, LF, RB, LB)."""

    repo: str = "kiselyovd/vehicle-keypoints"
    filename: str = "weights.pt"
    revision: str = KP_REVISION
    conf: float = 0.08
    imgsz: int = 640
    min_iou: float = 0.3
    table: str = field(default="keypoints", init=False)
    columns: list = field(default_factory=lambda: list(KP_COLUMNS), init=False)
    code_version: str = field(default="m4-v2", init=False)

    def __post_init__(self):
        if not self.revision or len(self.revision) < 7:
            raise ValueError("KeypointDetector needs a pinned Hugging Face revision (a commit hash)")

    def weights_id(self) -> str:
        return f"{self.repo}/{self.filename}@{self.revision}"

    def params(self) -> dict:
        return {"input": "full_frame", "imgsz": self.imgsz, "conf": self.conf, "match_iou": self.min_iou,
                "kept": "wheels 0-3 (RF, LF, RB, LB)"}  # fmt: skip

    def load(self):
        _need("ultralytics")
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        return YOLO(hf_hub_download(repo_id=self.repo, filename=self.filename, revision=self.revision))

    def infer(self, model, img, req, ctx):
        return _pose_infer(model, img, req, ctx, 4, self.conf, self.imgsz, self.min_iou)


@dataclass
class CuboidDetector(Detector):
    """UrbanOmniDetect-2: keypoints 0-3 are the cuboid's bottom face. `repo_root` is the urbanomnidetect checkout
    (not pip-installable): it goes first on sys.path and `v2.hybrid_v2` is imported before the checkpoint loads."""

    weights: str = ""
    repo_root: str = ""
    conf: float = 0.08
    imgsz: int = 640
    min_iou: float = 0.3
    prefer_cls: tuple = (2, 5, 7)
    table: str = field(default="cuboids", init=False)
    columns: list = field(default_factory=lambda: list(CUBOID_COLUMNS), init=False)
    code_version: str = field(default="m6-v2", init=False)

    def weights_id(self) -> str:
        return _ckpt_id(self.weights)

    def params(self) -> dict:
        return {"input": "full_frame", "imgsz": self.imgsz, "conf": self.conf, "match_iou": self.min_iou,
                "prefer_cls": list(self.prefer_cls), "kept": "keypoints 0-3 (bottom face)"}  # fmt: skip

    def load(self):
        _need("ultralytics")
        root = str(self.repo_root)
        if not sys.path or sys.path[0] != root:
            sys.path.insert(0, root)
        import v2.hybrid_v2  # noqa: F401  (registers HybridPoseModel26 for unpickling)
        from ultralytics import YOLO

        return YOLO(str(self.weights))

    def infer(self, model, img, req, ctx):
        return _pose_infer(model, img, req, ctx, 4, self.conf, self.imgsz, self.min_iou, self.prefer_cls)


def mask_contours(m: np.ndarray, min_frac: float) -> list:
    """External contours (k, 2) of a bool mask, components with >= min_frac of its area, largest first."""
    cv2 = _need("cv2")
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    areas = np.array([cv2.contourArea(c) + len(c) for c in cs], dtype=float)  # + perimeter: thin parts count
    keep = [i for i in np.argsort(-areas) if areas[i] >= min_frac * areas.sum()]
    return [cs[i].reshape(-1, 2).astype(float) for i in keep]


@dataclass
class MaskSegmenter(Detector):
    """SAM2.1 prompted with every requested track box of a frame in one call; masks > 0.5 kept as external contours
    of components with at least `min_component_frac` of the mask area. The default keeps every component, so M5's
    outline sees the whole mask as in the study; M7 applies the study's 5% rule itself (M7_MIN_COMPONENT_FRAC)."""

    weights: str = ""
    min_component_frac: float = 0.0
    table: str = field(default="masks", init=False)
    columns: list = field(default_factory=lambda: list(MASK_COLUMNS), init=False)
    code_version: str = field(default="masks-v2", init=False)  # v2: every component kept, raw-mask lower outline

    def weights_id(self) -> str:
        return _ckpt_id(self.weights)

    def params(self) -> dict:
        return {"input": "full_frame", "prompt": "track bbox xyxy, requested boxes of a frame in one call",
                "mask": "masks.data > 0.5", "stored": "external contours, CHAIN_APPROX_NONE, NaN-separated",
                "min_component_frac": self.min_component_frac}  # fmt: skip

    def load(self):
        _need("ultralytics")
        from ultralytics import SAM

        return SAM(str(self.weights))

    def infer(self, model, img, req, ctx):
        cv2 = _need("cv2")
        boxes = xyxy(req[["x", "y", "w", "h"]].to_numpy(float)).tolist()
        res = model(img, bboxes=boxes, verbose=False)[0]
        data = _np(res.masks.data) if res.masks is not None else np.zeros((0,) + img.shape[:2], bool)
        if len(data) != len(boxes):
            raise RuntimeError(f"SAM returned {len(data)} masks for {len(boxes)} box prompts")
        recs, outs = [], {}
        for r, m in zip(req.itertuples(index=False), data):
            m = m > 0.5
            if m.shape != img.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            cs = mask_contours(m, self.min_component_frac)
            ok = len(cs) > 0
            if ok:
                outs[(int(r.track), int(r.frame))] = (cs, lower_outline(m, (r.x, r.y, r.w, r.h)))
            recs.append([int(r.track), int(r.frame), 1.0 if ok else 0.0, len(cs), int(m.sum()), int(ok)])
        return recs, outs
