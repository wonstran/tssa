"""tssa.footprint: vehicle ground footprints from tracked boxes (design docs/2026-11-10-footprint-design.md)."""

from .config import FootprintConfig as FootprintConfig
from . import detect as detect
from .locators import LocateContext as LocateContext
from .locators import LocatorResult as LocatorResult
from .locators import available_locators as available_locators
from .locators import register_locator as register_locator
from .m7config import M7Config as M7Config
from .printer import VehFootPrinter as VehFootPrinter


def zone_from_points(world_points, buffer_ft: float = 15.0):
    """Convex hull of world points (e.g. GCPs, metres) buffered by `buffer_ft` (base design section 7)."""
    import numpy as np
    from shapely.geometry import MultiPoint

    from .config import FT

    pts = np.asarray(world_points, dtype=float).reshape(-1, 2)
    return MultiPoint(pts).convex_hull.buffer(buffer_ft * FT)
