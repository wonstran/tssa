"""Shared builder: a synthetic walker already projected to metres (the smoother takes ground coordinates)."""

import numpy as np
import pandas as pd

E0, N0 = 587000.0, 2852900.0  # UTM-sized offsets, to exercise the local-origin handling


def walker(n=60, track=1, start=0, drop=(), noise=0.05, seed=0, speed=1.0, fps=10.0, cls=0) -> pd.DataFrame:
    """One walker moving east at `speed` m/s, position noise sd `noise` m; `drop` frames removed."""
    rng = np.random.default_rng(seed)
    f = np.arange(start, start + n)
    x = E0 + speed * np.arange(n) / fps + rng.normal(0, noise, n)
    y = N0 + rng.normal(0, noise, n)
    df = pd.DataFrame({"frame": f, "track": track, "x": x, "y": y, "conf": 0.9, "cls": cls})
    return df[~df["frame"].isin(drop)].reset_index(drop=True)
