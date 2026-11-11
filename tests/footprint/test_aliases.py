"""Descriptive locator aliases (design 16, Q7): accepted wherever a locator is named, results keep the M name."""

import numpy as np
import pandas as pd
import pytest

from tssa.footprint import VehFootPrinter, available_locators, register_alias
from tssa.footprint.locators import ALIASES, canonical

from synth import ORIGIN, ground, render, stub_locator, tables

G = ground()
Q7 = {"bottom_centre": "M1", "nominal_shift": "M2", "nominal_box": "M3", "wheel_keypoints": "M4",
      "sam_outline": "M5", "cuboid": "M6", "wheel_fit": "M7"}  # fmt: skip


def car(n=30):
    c = np.tile(ORIGIN, (n, 1))
    boxes, faces = render(G, c, np.full(n, 30.0), 15.0, 6.0, 5.0, 0.3, 0)
    return tables(np.arange(n), 1, 2, boxes, faces)


def test_builtin_aliases():
    assert {a: canonical(a) for a in Q7} == Q7
    listed = available_locators()
    assert {a: m for m, d in listed.items() for a in d["aliases"]} == Q7
    assert canonical("M6") == "M6"
    with pytest.raises(ValueError, match="unknown locator"):
        canonical("wheel")


def test_chain_by_alias_equals_chain_by_name():
    tr, cub = car()
    a = VehFootPrinter(G, chains={2: ("cuboid", "nominal_shift")}, default_chain=("nominal_shift",))
    b = VehFootPrinter(G, chains={2: ("M6", "M2")})
    assert a.config.chains[2] == ("M6", "M2") and a.config.default_chain == ("M2",)
    out_a, out_b = a.estimate(tr, cuboids=cub), b.estimate(tr, cuboids=cub)
    pd.testing.assert_frame_equal(out_a, out_b)
    assert set(out_a["fp_source"]) <= {"M6", "M2", "none"}
    assert a.required_rows(tr).keys() == b.required_rows(tr).keys() == {"cuboids"}


def test_compare_by_alias_reports_the_m_name():
    tr, cub = car()
    vfp = VehFootPrinter(G)
    a = vfp.compare(tr, methods=("cuboid", "bottom_centre", "M1"), fallback="nominal_shift", cuboids=cub)
    b = vfp.compare(tr, methods=("M6", "M1"), fallback="M2", cuboids=cub)
    assert list(a["method"].unique()) == ["M6", "M1"]  # an alias next to its own name is one method
    pd.testing.assert_frame_equal(a, b)


def test_unknown_names_raise():
    with pytest.raises(ValueError, match="unregistered locator"):
        VehFootPrinter(G, chains={2: ("cuboids", "M2")})
    tr, cub = car(5)
    with pytest.raises(ValueError, match="unknown locator"):
        VehFootPrinter(G).compare(tr, methods=("M6",), fallback="nominal", cuboids=cub)


def test_alias_conflicts():
    with pytest.raises(ValueError, match="is a locator name"):
        register_alias("M2", "M6")
    with pytest.raises(ValueError, match="already an alias"):
        register_alias("cuboid", "M2")
    with pytest.raises(ValueError, match="unknown locator"):
        register_alias("x_alias", "M99")
    register_alias("cuboid", "M6")  # re-registering the same pair is harmless
    try:
        with stub_locator("T9", lambda rows, ctx: None, scope="row", classes=None):
            register_alias("my_method", "T9")
            assert canonical("my_method") == "T9" and available_locators()["T9"]["aliases"] == ("my_method",)
        with pytest.raises(ValueError, match="unknown locator"):  # the alias does not outlive its locator
            canonical("my_method")
    finally:
        ALIASES.pop("my_method", None)
    with pytest.raises(ValueError, match="already an alias"):
        with stub_locator("cuboid", lambda rows, ctx: None):
            pass
