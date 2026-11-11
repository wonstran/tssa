"""M7 placeholder until phase 5 of the design's build order: registered so the default chains validate."""

from . import register_locator


@register_locator("M7", needs=("masks",), scope="segment", classes=(5, 7))
def locate_m7(rows, ctx):
    raise NotImplementedError("M7 is ported in phase 5 (docs/2026-11-10-footprint-design.md section 17)")
