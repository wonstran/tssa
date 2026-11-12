# tssa API Manual

**Version {{ tssa_version }}** ([changelog](changelog.md))

`tssa` (Transportation Surrogate Safety Analysis) turns tracked road users from roadside video into ground
trajectories, vehicle footprints and speeds for safety measures such as post-encroachment time (PET). It is the
analysis side of [dnt](https://wonstran.github.io/dnt/): dnt detects and tracks, and tssa works on the track files.

This site is generated with MkDocs and `mkdocstrings` from the Python source in `src/tssa`.

## Install

```bash
pip install -e .                 # the package
pip install -e ".[detect]"       # plus the detectors used by tssa.footprint.detect (torch, ultralytics, OpenCV)
pip install -e ".[docs]"         # plus the tools that build this manual
```

Python 3.9 or newer. OpenCV is needed only to fit a homography, run the detectors or rasterise masks. Loading a
saved homography and projecting with it use NumPy only.

## Packages

| Package | What it does | Main entry point |
|---|---|---|
| [`tssa.footprint`](api/footprint/index.md) | Vehicle ground footprints (reference point, oriented polygon, speed) from tracked boxes, with seven methods M1-M7 | `VehFootPrinter` |
| [`tssa.smooth`](api/smooth/index.md) | Segment ground tracks at jumps, fill gaps and smooth positions | `TrackSmoother` |
| [`tssa.speed`](api/speed/calculator.md) | Velocity and speed from smoothed positions | `SpeedCalculator` |
| [`tssa.transform`](api/transform/index.md) | Image to ground homographies, lens distortion, pinhole camera fit | `DistortedHomography`, `GroundModel` |
| [`tssa.engine`](api/engine/index.md) | Point, line and zone geometry for track matching | functions |
| [`tssa.core`](api/core/index.md) | Earlier analyses: counts, lanes, speed meters, interactions, near misses, yielding | `Count`, `LaneDetector`, ... |
| [`tssa.annotation`](api/annotation.md), [`tssa.io`](api/io.md) | Drawing results on video, reading track files | `Annotation`, `read_track` |

## Pipeline

```
track file (pixels)
  -> tssa.transform     image <-> world (m) homography, with lens distortion
  -> tssa.footprint     reference point per row (M1-M7, per-class fallback chain) + 2 x 2 covariance
  -> tssa.smooth        segment, fill gaps, RTS smoothing
  -> tssa.speed         velocity and speed (mph)
  -> tssa.footprint     oriented polygon per row -> GeoDataFrame for PET
```

`VehFootPrinter.estimate()` runs the last four steps in one call. See the [Quickstart](quickstart.md).

## Units

- Positions are in the homography's world metres (for example UTM).
- Vehicle sizes and distance settings of `FootprintConfig` are in feet.
- Speeds are in mph unless you pass a metric `SpeedCalculator`.
