# Vehicle Footprints (`tssa.footprint`)

`VehFootPrinter` turns tracked vehicle boxes into ground footprints. For each row it finds a reference point at the
body centre before smoothing. After smoothing it builds an oriented polygon of the right size. It carries the seven
methods of the speed study, per-class fallback chains, and a 2 x 2 measurement covariance per row. The full design
is in the [footprint design](../../2026-11-10-footprint-design.md).

```python
vfp = VehFootPrinter(dh, zone=zone, workers=8, cache_dir="cache")
tables = detect.run("clip.mp4", tracks, [detect.CuboidDetector(...), detect.MaskSegmenter(...)],
                    rows=vfp.required_rows(tracks), cache_dir="cache")
out = vfp.estimate(tracks, **tables)
gdf = vfp.to_geodataframe(out, crs="EPSG:32617")
```

## Methods

| Method | Alias | Reference point from | Needs |
|---|---|---|---|
| M1 | `bottom_centre` | bottom centre of the box (a point, no size) | - |
| M2 | `nominal_shift` | half a nominal width away from the camera | - |
| M3 | `nominal_box` | nominal rectangle fitted to the box | - |
| M4 | `wheel_keypoints` | CarFusion wheel keypoints | `keypoints`, camera |
| M5 | `sam_outline` | lower outline of the SAM mask | `masks` |
| M6 | `cuboid` | bottom face of the UrbanOmniDetect cuboid | `cuboids` |
| M7 | `wheel_fit` | rigid two-axle fit over a whole segment (buses, trucks) | `masks`, camera |

The default chains are cars M6, M2; buses M7, M6, M2; trucks M7, M6, M3, M2; any other class M2. A method whose table
is not passed fails with `not_supplied`, and the row goes to the next method.

## Calls

| Call | Returns |
|---|---|
| `estimate(tracks, only_tracks=None, **tables)` | Reference point, smoothed position, speed and polygon for every row |
| `locate(tracks, only_tracks=None, **tables)` | Reference points only, before smoothing |
| `polygons(table, speed_unit=None)` | The polygon step, for a table you smoothed yourself |
| `compare(tracks, methods=..., fallback="M2", **tables)` | Each method alone on one grid, in one long table with a `method` column |
| `required_rows(tracks)` | `{table: rows}` the chains need, for `detect.run(rows=...)` |
| `to_geodataframe(table, crs=None)` | One shapely polygon per row |
| `available_locators()` | Each method with its aliases, tables, scope and classes |
| `camera()` | The fitted pinhole camera, or None |
| `info` | Diagnostics: `camera`, `grid`, `sources`, `m7_solves` |

## Pages

- [VehFootPrinter](printer.md): the class and its output columns.
- [Configuration](config.md): `FootprintConfig` and `M7Config`.
- [Locators](locators.md): the registry, aliases and methods M1-M6; how to add your own.
- [M7 Rigid Fit](m7.md): the bus and truck solver.
- [Detection](detect.md): detectors, matching and the cached runner.
- [Masks](masks.md): `MaskTable` and outline helpers.
- [Geometry and Orientation](geometry.md): angles, projections and heading rules.

::: tssa.footprint
    options:
      show_root_heading: false
      members:
        - zone_from_points
