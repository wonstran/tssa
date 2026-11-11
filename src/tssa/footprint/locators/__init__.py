"""Locator registry: a locator turns rows (bbox, class, extra tables) into native reference points.

Design docs/2026-11-10-footprint-design.md, section 7.1. A locator is registered with

    @register_locator("M5", needs=("masks",), scope="row", point_only=False)
    def locate_m5(rows: pd.DataFrame, ctx: LocateContext) -> LocatorResult: ...

- scope "row": `rows` are original rows with a valid projection (fp_row_kind == "box"), any subset.
- scope "segment": `rows` are whole segments (consecutive frames, inserted rows included) of one class.
- `classes`: the cls_track values the locator serves (None = all); other rows fail with "class_ineligible".
- A locator whose `needs` tables were not supplied fails every row with "not_supplied".

`rows` carries: track, frame, the box columns (x, y, w, h under their configured names, NaN on inserted rows),
cls_track, seg, interp, fp_row_kind, u_bc, v_bc (bbox bottom-centre px), m1_x, m1_y, m2_x, m2_y (world m) and
ref_heading_deg.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class LocatorResult:
    """Per-row result aligned to the rows a locator was given."""

    ok: np.ndarray  # bool: supplies a position
    fail_reason: np.ndarray  # object: "" where ok
    x: np.ndarray  # world m
    y: np.ndarray
    orient_deg: np.ndarray  # native orientation (compass deg), NaN = none
    orient_kind: np.ndarray  # object: "axis" | "directed" | "none"
    len_ft: np.ndarray  # measured size of this row, NaN = none
    wid_ft: np.ndarray
    support: np.ndarray  # object: "native" | "kinematic" (only meaningful where ok)
    cov: np.ndarray | None = None  # (n, 2, 2) own covariance (kinematic rows), m^2
    extras: dict = field(default_factory=dict)  # name -> (n,) array, method-prefixed (m4_*, m6_*, m7_*)
    seg_len_ft: np.ndarray | None = None  # size of the vehicle for every row of a passed segment solve (M7)
    seg_wid_ft: np.ndarray | None = None
    seg_dims_source: np.ndarray | None = None

    @classmethod
    def empty(cls, n: int, reason: str = "") -> "LocatorResult":
        return cls(
            ok=np.zeros(n, bool), fail_reason=np.full(n, reason, dtype=object), x=np.full(n, np.nan),
            y=np.full(n, np.nan), orient_deg=np.full(n, np.nan), orient_kind=np.full(n, "none", dtype=object),
            len_ft=np.full(n, np.nan), wid_ft=np.full(n, np.nan), support=np.full(n, "native", dtype=object),
        )  # fmt: skip


@dataclass
class LocateContext:
    """What a locator may use besides its rows."""

    ground: object  # tssa.transform.GroundModel
    config: object  # FootprintConfig
    tables: dict  # detection tables by name
    camera: Callable[[], object]  # lazily fitted PinholeCamera, or None when unavailable / inconsistent
    all_boxes: pd.DataFrame  # every original row (track, frame, x, y, w, h): the matching / occlusion context
    workers: int = 1
    cache_dir: object = None


@dataclass
class Locator:
    name: str
    fn: Callable
    needs: tuple = ()
    scope: str = "row"
    point_only: bool = False
    classes: tuple | None = None


REGISTRY: dict[str, Locator] = {}


def register_locator(name: str, needs=(), scope: str = "row", point_only: bool = False, classes=None):
    """Decorator: register `fn(rows, ctx) -> LocatorResult` as locator `name`."""
    if scope not in ("row", "segment"):
        raise ValueError(f"scope must be 'row' or 'segment', got {scope!r}")

    def wrap(fn):
        REGISTRY[name] = Locator(name, fn, tuple(needs), scope, bool(point_only),
                                 None if classes is None else tuple(classes))  # fmt: skip
        return fn

    return wrap


def available_locators() -> dict[str, dict]:
    """Every registered locator with its needs, scope and flags."""
    return {n: {"needs": L.needs, "scope": L.scope, "point_only": L.point_only, "classes": L.classes}
            for n, L in REGISTRY.items()}  # fmt: skip


def _load_builtins() -> None:
    from . import m1_m2_m3, m6  # noqa: F401  (registers M1, M2, M3, M6)

    for mod in ("m4", "m5", "m7"):
        try:
            __import__(f"{__name__}.{mod}")
        except ModuleNotFoundError as exc:  # pragma: no cover - only while a phase is not built yet
            if exc.name != f"{__name__}.{mod}":
                raise


_load_builtins()
