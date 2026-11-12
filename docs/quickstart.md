# Quickstart

## Vehicle footprints and speed

```python
import pandas as pd
from tssa.footprint import VehFootPrinter, detect, zone_from_points
from tssa.transform import DistortedHomography

dh = DistortedHomography.load_json("model_a1a_nw.json")       # no OpenCV needed to load and project
tracks = pd.read_csv("veh_tracks.csv")                         # frame, track, x, y, w, h, cls (pixels)
gcps = pd.read_csv("gcps.csv")[["utm_e_m", "utm_n_m"]].to_numpy()
vfp = VehFootPrinter(dh, zone=zone_from_points(gcps, buffer_ft=15), workers=8, cache_dir="cache")

# optional: run the detectors (pip install tssa[detect]) for the rows the default chains need
tables = detect.run("clip.mp4", tracks,
                    [detect.KeypointDetector(), detect.CuboidDetector("uod.pt", "path/to/urbanomnidetect"),
                     detect.MaskSegmenter("sam2.1_b.pt")],
                    rows=vfp.required_rows(tracks), cache_dir="cache")

out = vfp.estimate(tracks, **tables)                           # reference point, smoothed position, speed, polygon
gdf = vfp.to_geodataframe(out, crs="EPSG:32617")               # one polygon per row, for PET
```

- `tracks` has one row per box: `frame`, `track`, `x`, `y`, `w`, `h` in pixels of the original (distorted) image,
  and `cls`. Rename the columns with `box_cols=` and `cls_col=`.
- Without detections, the default chains fall back to the methods that need none (M2, and M3 for trucks).
- Without a `zone` no camera is fitted, so M4 and M7 are skipped and their rows go to the next method in the chain.

### Choosing methods

Each class has a chain of methods, tried in order until one places the row:

```python
vfp = VehFootPrinter(dh, zone=zone, chains={2: ("cuboid", "nominal_shift"), 7: ("M7", "M3", "M2")})
```

Methods can be named `M1`-`M7` or by their aliases (`bottom_centre`, `nominal_shift`, `nominal_box`,
`wheel_keypoints`, `sam_outline`, `cuboid`, `wheel_fit`). Outputs always carry the M names.

To compare methods on the same rows:

```python
long = vfp.compare(tracks, methods=("M2", "M6", "M7"), fallback=None, **tables)   # each method's own rows only
long.groupby("method")["fp_source"].apply(lambda s: s.ne("none").mean())   # share of rows each method placed
```

### Reading the output

Every input row appears once, plus the rows inserted to fill gaps (`fp_row_kind == "inserted"`).

| Columns | Meaning |
|---|---|
| `fp_x`, `fp_y`, `fp_cov_xx`, `fp_cov_xy`, `fp_cov_yy`, `fp_sigma` | Reference point (world m) and its covariance, before smoothing |
| `fp_source`, `fp_support`, `fp_fail` | Which method placed the row, `native` or `kinematic`, and why the earlier methods failed |
| `fp_x_smooth`, `fp_y_smooth`, `vx`, `vy`, `speed` | Smoothed position and speed (mph by default) |
| `orient_deg`, `motion_heading_deg`, `length_ft`, `width_ft`, `dims_source` | Orientation, heading and size of the polygon |
| `c1x` ... `c4y` | The four corners (world m) |
| `fp_low_confidence` | The polygon should not be trusted for PET |

## Smoothing and speed on their own

If you already have ground positions in metres:

```python
from tssa.smooth import TrackSmoother
from tssa.speed import SpeedCalculator

sm = TrackSmoother(method="rts", fps=10)                       # or "sg7", "ma", or method="sg", params={"win": 9}
smoothed = sm.smooth(ground_tracks)                            # adds seg, fill_reason, x_smooth, y_smooth, fallback
speeds = SpeedCalculator(fps=10, xy_cols=("x_smooth", "y_smooth")).calculate(smoothed)
```

## Fitting a homography

```python
import pandas as pd
from tssa.transform import DistortedHomography

g = pd.read_csv("gcps.csv")                                    # pixel u, v and world utm_e_m, utm_n_m
dh = DistortedHomography(g[["u", "v"]].to_numpy(), g[["utm_e_m", "utm_n_m"]].to_numpy(), image_size=(1920, 1080))
dh.save_json("model.json")                                     # fitting needs OpenCV; loading does not
```

`kmz_to_csv` converts Google Earth placemarks into a CSV of latitude and longitude. Project them to UTM (or State
Plane) before fitting.
