"""Locator registry: a locator turns rows (bbox, class, extra tables) into native reference points.

Design docs/2026-11-10-footprint-design.md, section 7.1. A locator is registered with

    @register_locator("M5", needs=("masks",), scope="row", point_only=False)
    def locate_m5(rows: pd.DataFrame, ctx: LocateContext) -> LocatorResult: ...

- scope "row": `rows` are original rows with a valid projection (fp_row_kind == "box"), any subset.
- scope "segment": `rows` are whole segments (consecutive frames, inserted rows included) of one class.
- `classes`: the cls_track values the locator serves (None = all); other rows fail with "class_ineligible".
- A locator whose `needs` tables were not supplied fails every row with "not_supplied".
- `aliases`: further names that resolve to the locator (design 16, Q7), such as "cuboid" for M6. Chains and
  `compare()` accept either; results always carry the canonical name (`fp_source`, `fp_fail`, `method`).

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
    origin: object = None  # (2,) world offset of the local frame segment solvers work in
    info: dict = field(default_factory=dict)  # per-locator diagnostics, copied into VehFootPrinter.info


@dataclass
class Locator:
    name: str
    fn: Callable
    needs: tuple = ()
    scope: str = "row"
    point_only: bool = False
    classes: tuple | None = None


REGISTRY: dict[str, Locator] = {}
ALIASES: dict[str, str] = {}  # alias -> canonical locator name


def register_locator(name: str, needs=(), scope: str = "row", point_only: bool = False, classes=None, aliases=()):
    """Decorator: register `fn(rows, ctx) -> LocatorResult` as locator `name`, also reachable as each of `aliases`."""
    if scope not in ("row", "segment"):
        raise ValueError(f"scope must be 'row' or 'segment', got {scope!r}")
    if name in ALIASES:
        raise ValueError(f"{name!r} is already an alias of {ALIASES[name]!r}")
    aliases = (aliases,) if isinstance(aliases, str) else tuple(aliases)

    def wrap(fn):
        REGISTRY[name] = Locator(name, fn, tuple(needs), scope, bool(point_only),
                                 None if classes is None else tuple(classes))  # fmt: skip
        for a in aliases:
            register_alias(a, name)
        return fn

    return wrap


def register_alias(alias: str, name: str) -> None:
    """Make `alias` another name of the registered locator `name`."""
    if name not in REGISTRY:
        raise ValueError(f"unknown locator {name!r}; known: {sorted(REGISTRY)}")
    if alias in REGISTRY:
        raise ValueError(f"{alias!r} is a locator name, not available as an alias")
    if ALIASES.get(alias, name) != name:
        raise ValueError(f"{alias!r} is already an alias of {ALIASES[alias]!r}")
    ALIASES[alias] = name


def canonical(name: str) -> str:
    """The registered name behind `name` (itself, or the locator an alias points to); ValueError if unknown."""
    if name in REGISTRY:
        return name
    if name in ALIASES and ALIASES[name] in REGISTRY:
        return ALIASES[name]
    raise ValueError(f"unknown locator {name!r}; known: {sorted(REGISTRY)}, aliases: {sorted(ALIASES)}")


def available_locators() -> dict[str, dict]:
    """Every registered locator with its aliases, needs, scope and flags."""
    return {n: {"aliases": tuple(a for a, c in ALIASES.items() if c == n), "needs": L.needs, "scope": L.scope,
                "point_only": L.point_only, "classes": L.classes}
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
