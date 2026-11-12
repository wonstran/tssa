# Changelog

## Unreleased

- API manual at <https://wonstran.github.io/tssa/>, built with MkDocs Material and mkdocstrings from the package
  docstrings (`pip install -e ".[docs]"`, `mkdocs serve`). It shows the package version, read from
  `tssa.__version__`.
- `docs` extra in `pyproject.toml`.
- GitHub releases for every tag from v0.0.2.3.

## 0.0.2.12 (2026-11-11)

- Descriptive aliases for the footprint methods: `bottom_centre` (M1), `nominal_shift` (M2), `nominal_box` (M3),
  `wheel_keypoints` (M4), `sam_outline` (M5), `cuboid` (M6), `wheel_fit` (M7). They are accepted in `chains`,
  `default_chain` and `compare(methods=..., fallback=...)`; outputs keep the M names.
- `register_alias()`, and `register_locator(aliases=...)` for your own methods.

## 0.0.2.11 (2026-11-11)

- Validation of `VehFootPrinter` on the NW clip against the speed study, recorded in §19 of the footprint design.
- README section on vehicle footprints.

## 0.0.2.10 (2026-11-11)

- M7 is about 2.1x faster with the same results: a grouped finite-difference Jacobian (identical to SciPy's) and
  racing of the fit candidates (`M7_RACE_ROUND`, `M7_RACE_MARGIN`).

## 0.0.2.9 (2026-11-11)

- M7, the wheel-anchored rigid fit for buses and trucks, with parallel solves (`workers`) and a solve cache
  (`cache_dir`).

## 0.0.2.8 (2026-11-10)

- `tssa.footprint.detect`: keypoint, cuboid and SAM mask detectors, one-to-one matching to track boxes, and a cached
  runner that reads the video once.

## 0.0.2.7 (2026-11-10)

- M4 (wheel keypoints) and M5 (mask outline), the `MaskTable`, and the pinhole camera fit.

## 0.0.2.6 (2026-11-10)

- `tssa.footprint.VehFootPrinter` with M1-M3 and M6, per-class fallback chains and per-row covariance.

## 0.0.2.5 (2026-11-10)

- Segmented smoothing (`segment()` once, `smooth(segmented=True)` many times), per-row covariance columns, and
  `GroundModel`.

## 0.0.2.4 (2026-11-01)

- `tssa.speed.SpeedCalculator`; `TrackSmoother` now returns positions only.

## 0.0.2.3 (2026-10-31)

- `tssa.smooth.TrackSmoother`.
- `DistortedHomography` in `tssa.transform`.

## 0.0.1.0 (2026-10-28)

- Version 0.0.1.0.

## 0.0.1 (2024-10-02)

- `Count` and spatial calculation; track-zone matching and lane detection followed in October 2024.
