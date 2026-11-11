"""The settings namespace the ported M7 code reads as `S.<NAME>` (the study's `settings` module names).

A solve runs inside `using(namespace(...))`, so two printers with different M7Config values can coexist and worker
processes get their settings with the job, not from module globals. Outside a `using` block the defaults apply.
"""

from __future__ import annotations

import contextlib
import contextvars

from ...config import FT, MPS_TO_MPH

_current: contextvars.ContextVar = contextvars.ContextVar("tssa_m7_settings", default=None)


def namespace(m7cfg=None, img_size=(640, 480), edge_px=2.0, size_ft=None, m7_classes=(5, 7)) -> dict:
    """The study's names: every M7Config field plus FT, MPS_TO_MPH, IMG_SIZE, EDGE_PX, SIZE_FT and M7_CLASSES."""
    from ...config import _size_ft
    from ...m7config import M7Config

    cfg = M7Config() if m7cfg is None else m7cfg
    return {**cfg.as_dict(), "FT": FT, "MPS_TO_MPH": MPS_TO_MPH, "IMG_SIZE": tuple(img_size), "EDGE_PX": edge_px,
            "SIZE_FT": dict(size_ft or _size_ft()), "M7_CLASSES": tuple(m7_classes)}  # fmt: skip


class _Proxy:
    def __getattr__(self, name):
        ns = _current.get()
        if ns is None:
            ns = namespace()
            _current.set(ns)
        try:
            return ns[name]
        except KeyError:
            raise AttributeError(f"M7 setting {name!r} is not defined") from None


S = _Proxy()


@contextlib.contextmanager
def using(ns: dict):
    token = _current.set(ns)
    try:
        yield
    finally:
        _current.reset(token)
