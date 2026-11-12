# Smoothing (`tssa.smooth`)

`TrackSmoother` smooths ground trajectories that are already in metres. It splits tracks where they jump (an
implied speed above `split_speed`), fills short gaps, and smooths each segment with a method chosen by name:

| Name | Kind | Notes |
|---|---|---|
| `ma` | moving average | centred |
| `sg5`, `sg7` | median filter + Savitzky-Golay | window 5 or 7; `method="sg", params={"win": 9}` for others |
| `rts` | constant-velocity Kalman filter + RTS smoother | uses per-row sigma or covariance when given |

```python
from tssa.smooth import TrackSmoother

sm = TrackSmoother(method="rts", fps=10, xy_cols=("x", "y"))
out = sm.smooth(tracks)                        # adds seg, fill_reason, x_smooth, y_smooth, fallback
side_by_side = sm.compare(tracks)              # every preset on one segmentation
```

To segment once and smooth many times, call `segment()` and then `smooth(..., segmented=True)`.
`VehFootPrinter.estimate()` works this way. Speeds are not computed here: pass the smoothed positions to
[`SpeedCalculator`](../speed/calculator.md).

New kinds are added with `register_method`, and named presets with `register_preset`.

::: tssa.smooth
    options:
      show_root_heading: false
      members: false
