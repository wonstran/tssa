# tssa.footprint: vehicle ground footprint from tracked boxes (VehFootPrinter)

Date: 2026-11-10
Status: **approved by the user, 2026-11-10** (rev 4.1); rev 4.2 is an implementation note (§4.3). §19 adds the validation results (2026-11-11). Q6 and Q7 answered the same day (§16); nothing is open.
- rev 4.1: addresses [review_2026-11-10-213328.md](review_2026-11-10-213328.md) ([response](review_2026-11-10-213328-response.md)); §15 summary of invalid-projection rows only (editorial).
- rev 4: addresses [review_2026-11-10-211138.md](review_2026-11-10-211138.md) ([response](review_2026-11-10-211138-response.md)):
  - M7 admission is separate from the box-projection gate, with three row kinds (§7.7, §7.10, §9);
  - the width-hold inequality now matches the study code (§7.6);
  - one precedence rule for admission in `compare()` (§7.8, §11.2);
  - the UTM round-trip test is split into an internal and a public tolerance (§14.2).
- rev 3: addresses [review_2026-11-10-205948.md](review_2026-11-10-205948.md) ([response](review_2026-11-10-205948-response.md)):
  - invalid projections are kept on the grid, never measured (§5.2, §9);
  - a complete segmented-input contract for the smoother (§5.2);
  - measurement admission follows finite positions, not `interp`, and supplied covariances are final (§5.1, §5.2, §7.9);
  - `rts_smooth` gets a guard and the correct start time (§5.3), and orientation handles zero or one native rows (§10.2);
  - native axes are kept on vehicles that never move, with a `front_known` flag (§10.2);
  - OpenCV-free loading and projection, including the existing homography module (§2, §8, §13);
  - an explicit ground-model adapter with image size (§4.2);
  - unit-aware heading threshold (§10.1);
  - native-only `compare()` disables admission, and keeps every row (§11.2);
  - detector row selection is independent of the config (§6.4);
  - a study snapshot before porting (§17, phase 0).
- rev 2: the user's answers to rev 1's open questions (2026-11-10):
  - **Q1:** the class is `VehFootPrinter`.
  - **Q2:** sizes and distance settings are in **feet**.
  - **Q3:** port **M1–M7 together**, not only the recommended subset.
  - **Q4:** "your suggestion": take the 2 × 2 covariance path, which needs two backward-compatible additions to `tssa.smooth` (§5).
  - **Q5:** `tssa` provides the detectors (keypoints, cuboids, SAM masks) and the detection-to-track matching (§6), as an optional install.
- rev 1: first draft (M1, M2, M3, M6 only; detectors and covariance left open).

Source study: `D:\apps\miami-win\test\speed` (NW afternoon clip, A1A Collins Ave & 17th St), referred to below as **the study**. Its documents are the algorithm specification for every method; this design says how each one becomes part of `tssa`, and what changes on the way.
- `docs/2026-11-05-vehicle-footprint-design.md` (rev 5.1), **the base design**: M1–M6, camera, smoothing, orientation.
- `docs/2026-11-06-vehicle-footprint-run-report.md`, **the run report**: results, recommendation, large-vehicle changes, rulings R0–R23.
- `docs/2026-11-08-m7-wheel-anchored-design.md` (rev 7), **the M7 design**, and `docs/2026-11-08-m7-build-notes.md`, **the M7 notes**: decisions made while building M7 (up to "Round limit 20 -> 30", 2026-11-10).

Target package: `tssa` 0.0.2.4. New subpackage `src/tssa/footprint/`, new `src/tssa/transform/camera.py` and `ground.py`, additions to `src/tssa/smooth/`, and lazy OpenCV in `src/tssa/transform/homography.py`.

## 1. Intent

The study compared seven ways to get a vehicle's ground position and footprint from a DNT bbox, through one shared segmentation and smoother. This design moves all seven into `tssa`, so a later project (PET, conflicts, speed) can get vehicle footprints from tracked boxes and video without the study's scripts, and can still compare methods.

`tssa` already has the steps around it:
- `tssa.transform.DistortedHomography`: image ↔ ground;
- `tssa.smooth.TrackSmoother`: segments and smooths ground positions;
- `tssa.speed.SpeedCalculator`: speed from smoothed positions.

None of them knows where on the vehicle its input point is. Today a user projects the bbox bottom-centre (the study's M1) and smooths that. The study showed this point is the worst choice for PET: it sits **4.5 ft** from the consensus of the independent cues, its bias depends on direction (left turns +3.8 ft ahead), it shifts PET by up to **0.43 s** (p90; 0.75 s on right turns), and it reads trap speeds **0.9 mph** lower than M6.

`VehFootPrinter` supplies the missing step: **video + pixel boxes → a ground reference point at the body centre, with its covariance (before smoothing) → an oriented polygon of the right size (after smoothing).**

### 1.1 The seven methods

| Name | Study method | Needs | Native output | Study result |
|---|---|---|---|---|
| `M1` | bbox bottom-centre | homography | point (foot of the box, not the body centre) | 4.59 ft from M6; PET shift p90 0.43 s |
| `M2` | M1 moved half a nominal width away from the camera | homography | point | 1.65 ft from M6; \|Δ speed\| 0.40 mph; no inference |
| `M3` | nominal rectangle at the reference heading, fitted to the bbox | homography, ψ_ref | rectangle, orientation imposed | trucks 1.4 ft from M6 (M2: 6.7 ft) |
| `M4` | CarFusion wheel keypoints | keypoints, camera, ψ_ref | point + directed heading | 6.1% coverage; 1.18 ft from M6 |
| `M5` | SAM lower outline, nominal rectangle fit | masks | rectangle + axis | 58.7% coverage; e2e jitter doubles |
| `M6` | UrbanOmniDetect-2 cuboid bottom face | cuboids | measured quadrilateral + axis | 99.5% coverage; **recommended for PET and speed** |
| `M7` | wheel-anchored rigid-body fit per segment (buses, trucks) | masks, camera, M2, ψ_ref | centre + directed heading + fitted L, W, covariance | 19 of 25 display solves pass (run `20261110_161113_s0`); the user reviewed the M7 clips on 2026-11-10 |

The default chains (§7.8) follow the study's recommendation (M6 for cars) and its combined PET product (M7 for buses and trucks, M6 next, then the class fallback).

## 2. Goals, non-goals, anti-goals

**Goals**
- One public class, `VehFootPrinter`, with a frozen `FootprintConfig`, in the style of `TrackSmoother` / `SmoothConfig`.
- All seven methods as registered **locators** (§7), usable alone, in per-class fallback chains, or side by side (`compare()`).
- Per row: reference point, 2 × 2 position covariance, the locator that supplied it, why the others failed, native orientation and size.
- After smoothing, per row: footprint orientation, motion heading, length and width in feet with their source, four corners; optionally shapely polygons.
- Detector wrappers and matching (§6), so `tssa` can go from video to footprints.
- A pinhole camera fitted to a ground homography (§8), needed by M4 and M7.
- The study's synthetic tests ported under `tests/`, and one validation run that reproduces the study's NW-clip results (§14).

**Non-goals (this round)**
- The study's **evaluation machinery**: the three products (metric-native, metric end-to-end, display), eligibility and analysis-zone rules for metrics, plausibility table, stop mask, review clips, pinned-input hashing, sample-independence checks, camera-sensitivity reports. `compare()` gives a project what it needs to build its own.
- M7's metric solves per eligible run. `tssa` has one product, so M7 solves once per segment, the study's display solve.
- Manual footprint keyframes (the user decided not to annotate, 2026-11-08).
- Training models, shipping weights, real-time processing, PET itself.

**Anti-goals (failure even if it "works")**
- **A method looks better because it was smoothed differently.** Every locator goes through the same segmentation and smoother, with the same noise model (§7.9). The only exception is M7's admitted pseudo-measurements, which carry M7's own covariance and are marked (§7.7).
- **Imposed sizes reported as measured.** `dims_source` says whether a size was measured, partly prior, or nominal (M7: by information, M7 design §8.3).
- **Silent drops or invented support.** Every input row comes back once. A row without a reference point says which locators were tried and why each failed. An M7 row with no observation of its own is never `native`.
- **Heavy imports for light use.** Without torch, ultralytics or OpenCV installed, these must work: `import tssa`; **loading** a saved `HomographyTransform` or `DistortedHomography` (`load_json`, `load_matrix`) and projecting with it; fitting the pinhole camera; and M1, M2, M3, M4 and M6. OpenCV is still needed to **fit** a new homography, and for M5 and M7, which rasterise mask contours. torch and ultralytics load only when a detector runs (§6, §8, §13).
- **Study constants or machine paths in the code.** NW-clip values are defaults with a cited source; checkpoints and repository paths are arguments.
- **Silent drift from the study.** Every deliberate change is listed in §15.

## 3. Where it sits in the pipeline

```text
video ─► detect.run(...)            keypoints, cuboids, masks    (optional; §6; or bring your own tables)
                 │
tracks ──────────┼─► VehFootPrinter.locate()
                 │      1. project M1, build ONE segment grid  (TrackSmoother.segment, §5.2, §9)
                 │      2. M2 everywhere, reference pass: smooth M2 → ψ_ref        (§9)
                 │      3. per class, run its chain of locators on the grid        (§7)
                 │         → seg, fp_x, fp_y, fp_cov_*, fp_source, native orientation and size
                 ▼
            TrackSmoother.smooth(segmented=True)  same grid, covariance columns  (§5)
                 ▼
            SpeedCalculator.calculate()
                 ▼
            VehFootPrinter.polygons()  orientation, size, corners               (§10)
```

`estimate()` runs `locate → smooth → speed → polygons` with vehicle defaults (§11.3). `locate()` and `polygons()` stay public for users who want their own smoother or speed settings. `compare()` runs several methods on the same grid (§11.2).

The order follows base design §8.1: segmentation is built **once**, before any method; reference points are fixed **before** smoothing; orientation and polygon come **after**, from the smoothed motion.

## 4. Inputs

### 4.1 Tracks

A DataFrame, one row per `(track, frame)`; column names configurable (§12):

| Column | Meaning |
|---|---|
| `frame` | integer frame number, zero-based, the video's frame index |
| `track` | track id |
| `x, y, w, h` | bbox top-left and size in **distorted image pixels** |
| `cls` | class id per row (DNT: 2 car, 5 bus, 7 truck). The class used is the track's majority class, `cls_track`. |
| `interp` (optional) | 0 detected, 1 filled upstream |

**All tracks of a frame should be passed**, including tracks the user does not want footprints for. M7 drops outline columns that a nearer box covers, and detector matching is one-to-one over every box of a frame (run report R21). `locate(..., only_tracks=[...])` limits the output without hiding the other boxes.

### 4.2 Homography and the ground-model adapter

`VehFootPrinter` wraps whatever it is given in `tssa.transform.ground.GroundModel` (new):

```python
GroundModel.wrap(homography, image_size=None) -> GroundModel
    image_to_world(uv) / world_to_image(EN)   # distorted pixels <-> world metres
    undistort(uv) / distort(uv_u)             # identity when the homography has no distortion model
    world_to_undistorted(EN)                  # direct through the inner undistorted-image homography when there is one
    image_size                                # (width, height) px
```

- **Distortion.**
  - For `DistortedHomography`, `undistort` and `distort` are its own, and `world_to_undistorted` uses its inner homography directly. The round trip through `distort()` would add the iteration error of its 20-step inverse.
  - For `HomographyTransform`, or any object with only `image_to_world` / `world_to_image`, both are the identity and `world_to_undistorted = world_to_image`.
  - An object that has one of `undistort` / `distort` but not the other is rejected.
- **Image size.** Taken from the explicit `image_size` argument (`VehFootPrinter(..., image_size=(640, 480))`), else from the homography's `image_size` attribute (`DistortedHomography` has one). If the two disagree, `ValueError`. If neither exists, construction fails with `ValueError`: the edge flag (§7.10) and the camera's principal-point prior (§8) both need it.
- **Pixel spaces.** The camera works only in undistorted pixels. Its fit targets are `world_to_undistorted` of the zone grid, and its consistency check compares `distort(camera.project(X, z = 0))` with `world_to_image(X)`, as base design §5 specifies. M4 hub points and M7 tyre candidates are undistorted before they meet the camera (M7 design §5).

### 4.3 Detection tables

Written by `tssa.footprint.detect` (§6), or by the user's own pipeline in the same format. All pixels are distorted image pixels, keyed by `(track, frame)` after matching.

| Table | Columns | Used by |
|---|---|---|
| `keypoints` | `track, frame, iou, u0, v0 … u3, v3` (CarFusion wheels 0 RF, 1 LF, 2 RB, 3 LB) | M4, M7 (optional cue) |
| `cuboids` | `track, frame, iou, gu0, gv0 … gu3, gv3` (bottom-face corners, any start, any winding) | M6 |
| `masks` | `MaskTable`: `(track, frame) →` list of contour polygons `(k, 2)` of every mask component, largest first, plus `.outlines`: the lower outline taken from the raw mask at segmentation time | M5 (the stored outline, else one rebuilt from the contours), M7 (components with at least 5% of the area) |

One SAM pass serves both M5 and M7 (the study ran SAM twice, §15):
- M5 reads the lower outline that `MaskSegmenter` takes from the raw mask, exactly the study's M5 input. Without one (masks from elsewhere), it rebuilds the outline from the contours.
- M7 keeps the components holding at least `M7_MIN_COMPONENT_FRAC` (5%) of the summed area + perimeter, the rule the study applied when it wrote its SAM cache. Applied to an already filtered table it keeps every component, so the study's cache works as is.

rev 4.2 (implementation, 2026-11-10): rev 4.1 stored only the 5% components and rebuilt M5's outline from them. The validation run measured the cost: outlines identical to the study's on 3,769 of 4,129 bus and truck rows, and M5 validity changed on 62 of them (§19).

### 4.4 Camera zone

M4 and M7 need the pinhole camera of §8, which is fitted over a ground region. `zone`: a shapely polygon in world metres. Helper: `zone_from_points(world_points, buffer_ft=15)` (base design §7: GCP convex hull + 15 ft). Without a zone, camera-dependent locators fail with `camera_invalid`.

## 5. Changes to `tssa.smooth` (the answer to Q4)

**Recommendation.** Use the study's full measurement covariance, not a scalar σ. Two reasons:
- The ground scale of a pixel is strongly directional in this view: the largest-to-smallest ratio has median 3.3 and p95 5.0 (base design §9). A scalar σ under-smooths along the camera ray and over-smooths across it.
- M7's display pseudo-measurements carry their own 2 × 2 covariance (M7 design §9.3). A per-row covariance column carries them with no special path, replacing the study's `R_override`.

All additions below are backward compatible: calls that use none of the new arguments give identical results, and the existing smooth and speed tests must pass unchanged.

### 5.1 Per-row covariance

`SmoothConfig.cov_cols: tuple[str, str, str] | None = None`: names of per-row covariance columns `(xx, xy, yy)` in m². Three columns make `R` symmetric by construction.
- The `rts` kind uses them as the measurement covariance `R` on measured rows where all three are finite; otherwise `sigma_col`; otherwise the method's `sigma`.
- **A supplied covariance is final.** `fill_scale` is applied only to noise that comes from `sigma_col` or the method's `sigma`, never to `cov_cols`. Whoever writes the covariance applies any inflation. For `VehFootPrinter` that rule is §7.9.
- A measured row whose covariance is not positive definite (`xx > 0`, `yy > 0`, `xx·yy − xy² > 0`) raises `ValueError`.
- Kinds without a noise model (`ma`, `sg`) ignore it, as they ignore `sigma_col` today.

### 5.2 Segmenting once, smoothing many times

**`TrackSmoother.segment(tracks, ref_tracks=None, allow_missing=False) -> DataFrame`**

The public form of today's `_grid`. It returns the **structural** table only: `track, frame, seg, seg_len, interp, fill_reason, gap_id`, one row per input row plus the inserted rows (`interp = 2`). No positions are returned, so none can be reused by mistake.

With `allow_missing=True`, rows with a non-finite position are accepted and **kept**:
- Jump detection and gap classification use only the finite rows. A run of missing rows between two finite rows counts as part of the gap between them.
- **Interior run, gap filled** (short enough, plausible implied speed, no jump): the missing rows stay in the segment with their own `interp` (0 or 1) and `fill_reason = "no_position"`. Frames with no row at all are inserted as today (`interp = 2`).
- **Interior run, gap not filled:** the track is split there, and the missing rows get `seg = −1`, `fill_reason = "no_position"`.
- **Prefix, suffix, or whole track** (missing rows before a track's first finite row, after its last, or on a track with no finite row): `seg = −1`, `fill_reason = "no_position"`. They are never extrapolated.

Without `allow_missing`, a non-finite position raises, as today.

**`TrackSmoother.smooth(tracks, ref_tracks=None, segmented=False)`**

With `segmented=True`, the input already carries the structural columns `seg, interp, fill_reason` (from `segment()` or from `VehFootPrinter.locate()`). Then:
- **No re-segmentation.** The input is validated: no repeated `(track, frame)`, and every `(track, seg ≥ 0)` has consecutive frames; otherwise `ValueError`. `seg_len` is recomputed. `seg`, `interp` and `fill_reason` are input columns in this mode, so they pass through unchanged and are not clashes. The smoother adds only `<pos>_smooth` and `fallback` (or the `compare()` columns).
- **Measurement admission follows the positions, not `interp`.** A row is a measurement exactly when it has `seg ≥ 0` and both position values are finite. `interp` is provenance only. It selects `fill_scale` for σ-based noise on `interp = 1` rows, and it no longer clears observations. A finite position on an `interp = 2` row (an M7 pseudo-measurement on an inserted frame, §7.7) is measured.
- **Nothing from the segmentation pass is reused.** Observations come only from this call's position columns. Kinds that need a value on every row (`ma`, `sg`, and the short-segment fallback) get this call's measurements linearly interpolated across unmeasured rows of the segment, and held constant before the first and after the last measurement.
- **Coordinates.** The local origin is computed from this call's finite positions (floor of the minimum, as today), and the output is returned in input coordinates, rounded to three decimals like every added numeric column today.
- **Segments without enough measurements:**
  - 0 measured rows: NaN positions, `fallback = 1`;
  - 1 measured row: that position on that row, NaN elsewhere, `fallback = 1`;
  - fewer than the method's minimum otherwise: the interpolated raw positions, `fallback = 1`, as today.
- **`seg = −1` rows** are returned with NaN positions and `fallback = 1`.

Without `segmented`, behaviour is unchanged.

This is how one segmentation serves every method, as the study required (base design §7, anti-goal 1), without any method seeing another's positions.

### 5.3 `rts_smooth` fixes

`tssa.smooth.methods.rts_smooth` needs two fixes, both found by the review:
- **Guard.** Fewer than 2 measured rows raises `ValueError` with a clear message, not `IndexError`. Every caller checks first (§5.2, §10.2).
- **Start time.** The forward pass starts at the first measured row `i0`, from the state of rows `i0`, `i1` as today. Rows before `i0` take the smoothed state at `i0`, extrapolated backward at its constant velocity: `x_k = x̂_{i0} − (i0 − k)·dt·v̂_{i0}`. Today the filter starts at row 0 with the state of row `i0`. On exact constant-velocity data with a 10-frame missing prefix, that gives 1.54 m at frame 0 where the truth is 0.
- Existing callers are unaffected: today every segment starts with a measured row (inserted rows lie only between observations), so `i0 = 0`.

## 6. Detectors (`tssa.footprint.detect`, optional install)

### 6.1 Packaging

- Extra: `pip install tssa[detect]` adds `torch`, `ultralytics`, `opencv-python`, `huggingface_hub`. `requirements.txt` already lists OpenCV.
- Every import of these is lazy, inside the function that needs it. A missing package raises `ImportError` naming the extra.
- **UrbanOmniDetect-2 is not pip-installable.** `CuboidDetector(weights, repo_root)` puts `repo_root` first on `sys.path` and imports `v2.hybrid_v2` before loading, as the study's `infer.load_uod` does.
- No weights ship with `tssa`. The keypoint model downloads from Hugging Face at a **pinned revision** (default: the study's `48b5795f3b79cdfe7ff6588812aa388a6a7fdd9d` of `kiselyovd/vehicle-keypoints`, `weights.pt`). An unpinned revision raises.

### 6.2 Classes

| Class | Model | Output table | Study defaults |
|---|---|---|---|
| `KeypointDetector(repo, filename, revision, conf=0.08, imgsz=640)` | CarFusion YOLO pose | `keypoints` | `infer.run_keypoints` |
| `CuboidDetector(weights, repo_root, conf=0.08, imgsz=640, prefer_cls=(2, 5, 7))` | UrbanOmniDetect-2 | `cuboids` | `infer.run_uod` |
| `MaskSegmenter(weights, min_component_frac=0.0)` | SAM2.1, box-prompted with each track bbox | `masks` | `infer.run_sam_full` |

Full-frame inference for keypoints and cuboids; SAM is prompted with the requested rows' boxes (all of a frame in one call). Masks are `> 0.5`, resized to the frame if needed, and stored as external contours (`CHAIN_APPROX_NONE`) of every component (by default), largest first, together with the row's lower outline from the raw mask.

### 6.3 Matching

`match_frame(track_xyxy, det_xyxy, min_iou=0.3, det_cls=None, prefer=None)`: one-to-one assignment of detections to **every** track box of the frame, maximising total IoU (`scipy.optimize.linear_sum_assignment`), edges below `min_iou` removed, preferred classes first (run report R21). It needs only numpy and scipy, so it lives in the core package (`footprint/match.py`) for users with their own detectors.

### 6.4 Runner and cache

```python
tables = detect.run(video, tracks, detectors=[KeypointDetector(), CuboidDetector(w, root), MaskSegmenter(w)],
                    rows=None, cache_dir=None)          # -> {"keypoints": df, "cuboids": df, "masks": MaskTable}
```
- Frames are read once, sequentially (`cv2.VideoCapture`), and every detector runs on each frame read.
- **Two separate inputs.**
  - `tracks` is the **matching context**: every box of every frame. It is always used whole, for the one-to-one matching of §6.3, however few rows are requested.
  - `rows` selects the **output** (and cached) rows.
- **`rows` default, independent of any configuration:** every row of `tracks`, for every detector. The runner knows nothing about chains.
- **To infer only what a configuration needs**, ask the printer: `vfp.required_rows(tracks) -> {"keypoints": df, "cuboids": df, "masks": df}`. It returns, per table, the rows of tracks whose class chain contains a locator that needs that table (a custom car chain with M5 makes car rows appear under `masks`). Pass the result as `rows=`, either as one dict or per detector.
- **Cache** (base design §10), optional: `cache_dir/<detector>/<key>/`, where `key` hashes the video file, the **full** track table (the matching context), the weights (file SHA-256 or HF revision), the detector parameters and a per-detector `CODE_VERSION`. A changed setting gives a new key, never silent reuse. A run infers only rows the cache lacks, and saves every 200 frames. `MaskTable` is stored as `npz`. The study's lesson applies: arrays are read once per load, not per row (run report, "Performance fix").

## 7. Locators (before smoothing)

### 7.1 Contract

```python
@register_locator("M5", needs=("masks",), scope="row", point_only=False)
def locate_m5(rows: pd.DataFrame, ctx: LocateContext) -> LocatorResult: ...
```
- `scope="row"`: the locator sees any subset of rows. `scope="segment"` (M7): it sees whole segments of the grid, consecutive frames including filled ones.
- `ctx`: the homography, the camera (or None), the config, the grid with `cls_track`, `ref_heading_deg` and the M2 points, and the detection tables by name.
- `LocatorResult`, arrays aligned to `rows`:
  - `ok`, `fail_reason`;
  - `x, y` (world m);
  - optional `cov` (n, 2, 2): only M7 supplies one; the others get the shared model of §7.9;
  - `orient_deg`, `orient_kind` (`axis` / `directed` / `none`);
  - `len_ft`, `wid_ft` (measured size, or NaN);
  - `support` (`native` / `kinematic`);
  - an extras dict of diagnostic columns, namespaced by the method (`m4_*`, `m5_*`, `m7_*`).
- A locator whose `needs` are not supplied fails every row with `not_supplied`.

Algorithms are **ported from the study's code**, not rewritten. §7.2–7.7 name the source and say only what changes.

### 7.2 M1, M2, M3

Ported from `geometry.m1_points`, `geometry.m2_points` and `m3.solve_m3` (base design §6):
- **M1** is `point_only`: `polygons()` gives it no corners.
- **M2:** `away = unit(world(u, v − 1) − world(u, v))` at the bbox bottom-centre; moved `W_nom / 2` along it.
- **M3:** solve the centre so the nominal rectangle at `ψ_ref` has its lowest projected corner on the bbox bottom and its projected horizontal midpoint at the bbox centre. LM from the M2 point; valid below 0.5 px. Fail reasons `no_heading`, `no_solution`. The orientation is `ψ_ref`, imposed.

### 7.3 M4 (wheel keypoints)

Ported from `m4.solve_m4` and `m4.front_conflict` (base design §6.1). Unchanged, including:
- the four label permutations;
- the interior side from the other pair or from camera-away, never from the labels;
- the hub-to-contact drop, because the landmark is the hub (run report: confirmed on crops; `wheel_landmark = "hub"` in the config);
- the per-track `front_conflict` flag.

Fail reasons: `no_match`, `no_valid_pair`, `side_ambiguous`, `camera_invalid`, `no_heading`. Extras: `m4_pair_side, m4_label_case, m4_interior_source, m4_side_reject, m4_front_flipped, m4_wheelbase_ft, m4_track_ft, m4_front_conflict`.

### 7.4 M5 (SAM outline)

Ported from `m5.native_m5`:
- lower outline from the mask contours;
- R23: drop points more than 1 px below the bbox bottom;
- fit a nominal-size rectangle from fixed starts at 0/45/90/135° (R19);
- R20: mirror the fit away from the camera.

Fail reasons: `no_match`, `few_columns` (< 8), `fit_residual` (> `m5_max_resid_ft`, 1.0 ft, untuned in the study). Extras: `m5_cols, m5_resid_ft, m5_sides, m5_flipped`.

### 7.5 M6 (cuboid)

Ported from `m6.native_m6` and `face_geometry`:
- project the four corners;
- opposite-edge means give length and width;
- doubled-angle mean of the long edges gives the axis;
- the corner mean is the reference point.

Fail reasons: `no_match` (no row, IoU < 0.3, non-finite), `implausible_size` (length or width under `m6_size_gate` = 0.5 of nominal; run report "Large-vehicle handling"). The raw projected corners are kept as `m6_raw_c1x … m6_raw_c4y`, with `m6_len_ft, m6_wid_ft, m6_iou`.

### 7.6 M7 (wheel-anchored rigid fit), `scope="segment"`

Ported from `m7.py`, `m7_candidates.py`, and the job and table parts of `m7_runner.py`, **in the state of 2026-11-10** (run `20261110_161113_s0`). This is the M7 design rev 7 plus the M7 notes' later decisions:
- front rule `travel` (2026-11-09);
- width held at the class prior when the silhouette resolves width **coarser** than the threshold (2026-11-10). The study code is authoritative (`m7.py`: `w_held = M7_W_HOLD_FT_PER_PX is not None and w_ft_per_px > M7_W_HOLD_FT_PER_PX`):
  - `w_ft_per_px` is the width change per pixel of far-end silhouette, median over frames (`width_resolution`). Larger means coarser.
  - Held when `w_ft_per_px > 0.3`, a strict inequality: exactly 0.3 is **not** held.
  - `w_ft_per_px = inf` (no frame's extreme column moves with W) is held.
  - `w_hold_ft_per_px = None` never holds.
- a width on its bound does not fail the solve (`M7_W_BOUND_FAILS = False`);
- fitted tyre inset;
- tyre lowest-point model;
- image-edge σ;
- pair-wheelbase restarts;
- 30 outer rounds;
- tandem and far-side targets off.

All `M7_*` settings move into `FootprintConfig.m7` (§12.2) with these values.

**Per segment** of a cls 5/7 track (`m7_classes`):
1. **Frozen observations:** `frame_obs_from_mask` on every row with a mask. Other boxes of the same frame come from the full track table (§4.1).
2. **Initial pose** (study `m7_runner._init_pose`, then `m7.initial_x`):
   - centre: the M2 points (§9), linearly interpolated across rows without one;
   - heading: `ψ_ref` interpolated as a unit vector, or the path direction where `ψ_ref` is missing.
3. **Fit:** `solve_track`. Segments shorter than `min_support_frames` (10) fail with `too_few_frames`.
4. **Rows:**
   - `wheels2` / `wheel_front` / `wheel_rear` / `silhouette` rows are `native`;
   - `kinematic` rows are `ok = False`, `fail_reason = no_observation`, and their pose is kept in `m7_x, m7_y, m7_psi_deg`;
   - a failed solve fails every row with its gate reason (`param_bound`, `wheel_resid`, `articulated_suspect`, …).

**Runtime.**
- The study's 49 solves took 46 min wall on 12 workers; one 592-frame truck segment alone took 45 min.
- `VehFootPrinter(..., workers=N)` solves segments in a process pool (one BLAS thread each).
- `cache_dir` stores each solve under a key that hashes its observations, initial pose, camera, the `m7` settings and the solver's code version (study `m7_runner.job_key`).

Extras: the M7 design §9.4 columns with the `m7_` prefix (support, own pose, per-solve parameters, size sources, uncertainty, flags), plus `m7_w_held`.

### 7.7 M7 pseudo-measurements

M7 design §9.3, user decision Q2, adapted to the one-product world.
- **Admission** (the study's predicate, `products._product`): a row becomes a measurement when
  - its segment's M7 solve passed,
  - `m7_support = kinematic`, and
  - its M7 pose and `m7_cov_centre` are finite.
  
  **Nothing about the box is checked.** Admission does not look at the projection gate (§7.10), the box fields or the Jacobian, so it applies to all three row kinds of §7.10 that lie in a segment (`seg ≥ 0`):
  - **inserted frames** (`interp = 2`): no box at all;
  - **original rows whose projection was rejected:** inside a segment, `fill_reason = "no_position"`. Admission overrides the rejection, as in the study, where the display gate was `proj_valid` and admission ignored it. The M7 pose comes from the camera and motion model, not from the bbox bottom.
  - **original rows with a valid projection.**
  
  **Never admitted:** rows with `seg = −1`, which belong to no segment and no solve. The smoother measures any segment row with a finite position (§5.2), so an admitted point is used, not cleared.
- **No box-derived values on admitted rows.** Their `fp_x, fp_y` is the M7 pose and their covariance is the one below. `fp_sigma` is computed from it; no box, Jacobian or `fill_meas_scale` enters.
- **Native M7 rows are gated like any box row,** also as in the study (`nat_ok = native_valid & gate`): a `wheels2` / `wheel_*` / `silhouette` row whose projection was rejected is **not** used as a native measurement. It is not admitted either, because admission requires `kinematic` support, so the next locator in the chain tries it.
- **Covariance:** `R = kin_meas_scale² · m7_cov_centre + (kin_sd_floor_ft · 0.3048)² · I` (defaults 2.0 and 0.5 ft). It is final (§5.1), so no `interp`-based inflation is added, whatever the row's `interp`.
- **Marking:** `fp_source = "M7"`, `fp_support = "kinematic"`.
- **Precedence:** M7 native rows, then admitted rows, then the next locator in the chain.
- `admit_kinematic=False` turns it off, giving the study's metric behaviour (no pseudo-measurements).

### 7.8 Chains

Each class has an ordered chain. The **first locator that succeeds** on a row supplies its reference point. Defaults:

| `cls_track` | Default chain | Source |
|---|---|---|
| 2 car | `M6 → M2` | run report recommendation |
| 5 bus | `M7 → M6 → M2` | M7 design §9.3 and §9.6 |
| 7 truck | `M7 → M6 → M3 → M2` | same; `FALLBACK_BY_CLASS = {7: "M3"}`; the final M2 catches rows with no heading |
| other | `M2`, with `default_size_ft` | my call |

- Locators whose tables were not supplied fail with `not_supplied`, so without detections the chains reduce to `M2` / `M3 → M2`.
- M4 and M5 are in no default chain (the study rejected them for production). Users add them through `chains`.
- **Aliases** (added in 0.0.2.12, Q7 in §16): `bottom_centre` M1, `nominal_shift` M2, `nominal_box` M3, `wheel_keypoints` M4, `sam_outline` M5, `cuboid` M6, `wheel_fit` M7. Chains and `compare()` accept either name; the printer resolves aliases when it is built, so results (`fp_source`, `fp_fail`, `method`) always carry the `M` name. `register_locator(..., aliases=...)` and `register_alias()` add names for user locators.
- **A chain of one locator** has no fallback: failed rows have no measurement, and the smoother predicts them. For M7 this is the study's **native-only** product only with `admit_kinematic=False`; with admission on (the default), kinematic rows are still measured. `compare(fallback=None)` turns admission off for you (§11.2).
- **Mixing sources** inside a track adds jumps, because the locators disagree by 1–2 ft (cars: M2 vs M6 1.6 ft). `fp_source` records it per row.

### 7.9 Shared noise model

For every locator except M7's admitted rows, the measurement covariance is the study's (`smoothlib.methods.measurement_cov`):
```text
R = (sigma_k · h)² · J Jᵀ      J = 1-px Jacobian of image_to_world at the bbox bottom-centre, h = box height
```
`sigma_k` = 0.03 (study `RTS_SIGMA_K`). It comes from the box, not from the locator, so no locator looks steadier because it was given a smaller σ. It is written as `fp_cov_xx, fp_cov_xy, fp_cov_yy`, plus `fp_sigma = sqrt(trace(R) / 2)` for smoothers that take a scalar.

**Scaling by support type.** Supplied covariances are final (§5.1), so `locate()` applies every inflation itself:

| Row | Covariance |
|---|---|
| box-based locator (M1–M7 native) on an original row with a valid projection, `interp = 0` | `R` as above |
| box-based locator, `interp = 1` (box filled upstream) | `fill_meas_scale² · R`, `fill_meas_scale` = 4.0 (study `FILL_MEAS_SCALE`, applied as in the study's `smooth_series`) |
| M7 admitted pseudo-measurement, any `interp` | §7.7, no further scaling |
| no reference point | NaN; the row is not measured |

`fp_sigma` is computed from the scaled `R`. A smoother given `sigma_col="fp_sigma"` instead of the covariance columns would apply its own `fill_scale` on top, so `estimate()` always passes the covariance columns (§11.3).

### 7.10 Shared per-row checks

- **Three row kinds.** `fp_row_kind` tells them apart; `fp_proj_valid` is the gate result (False on inserted rows, as the study's `proj_valid`):

  | `fp_row_kind` | Row | Box observation | M7 admission (§7.7) |
  |---|---|---|---|
  | `box` | original row, projection valid | yes: every locator may run | if `kinematic` |
  | `box_rejected` | original row, projection rejected | no: every **native** locator fails it with `projection` | if `seg ≥ 0` and `kinematic` |
  | `inserted` | frame inserted into a gap (`interp = 2`), no box | no: row-scope locators are not run, `fp_fail = "no_box"`. M7's segment solve spans the frame but has no observation there, so the frame is `kinematic`. | if `kinematic` |

- **Projection gate** (base design §7), on original rows only: the bottom-centre projection is invalid when non-finite, when the local scale (largest singular value of `J`) exceeds `max_ft_per_px` (2.0), or, if a `zone` is given, when it is outside the zone buffered by `gate_buffer_ft` (100). The gate governs **box observations** (every locator's native result). It does not govern M7 admission.
- **Image edge:** `fp_edge = True` when the bbox is within `edge_px` (2) of an image border. A flag only.

### 7.11 `locate()` output

Input columns first, unchanged; the grid's inserted rows are added (`interp = 2`). **Every input row is present exactly once**, including rows with an invalid projection (`seg = −1` or `fill_reason = "no_position"`, §9) and rows of tracks outside `only_tracks` (which keep their structural columns but get no `fp_*` values). Then:

| Column | Meaning |
|---|---|
| `cls_track`, `seg`, `fill_reason`, `gap_id` | class and the shared grid (§5.2). The input's `interp` is updated in place, as `TrackSmoother` does. |
| `fp_x, fp_y` | reference point (world m); NaN when no locator succeeded |
| `fp_cov_xx, fp_cov_xy, fp_cov_yy`, `fp_sigma` | §7.9 / §7.7 |
| `fp_source` | supplying locator, or `none` |
| `fp_support` | `native`, `kinematic` (M7 admitted) or empty |
| `fp_fail` | failures before it, `;`-joined, e.g. `M7:param_bound;M6:implausible_size` |
| `fp_orient_deg, fp_orient_kind` | native orientation of the supplying locator |
| `fp_len_ft, fp_wid_ft` | measured size of the supplying locator (M6, M7), else NaN |
| `fp_row_kind`, `fp_proj_valid` | `box` / `box_rejected` / `inserted`, and the gate result (§7.10) |
| `fp_edge`, `ref_heading_deg` | §7.10, §9 |
| `m4_*`, `m5_*`, `m6_*`, `m7_*` | extras of every locator that ran on the row |

## 8. Pinhole camera (`tssa.transform.camera`)

Ported from the study's `camera.py` (base design §5). It is a transform, so it goes into `tssa.transform` next to the homography.
- `PinholeCamera`: `project(XYZ)` (world 3-D → undistorted px) and `ray(uv)`.
- `PinholeCamera.fit(homography, zone, f_fixed=None, free_principal=True, pp_prior=1e-3)`: least-squares fit of `f`, `(cx, cy)`, `R` and `C` to a 1 m grid over the zone, mapped with the homography. A weak prior pulls `(cx, cy)` toward the image centre, which picks one exact camera out of the family (R18).
- `consistency(cam, homography, zone)`: RMS and max ground-projection residual; ok at RMS ≤ 1 px and max ≤ 3 px.
- `sensitivity(...)`: refits at f × 0.87 / 1.0 / 1.13.
- `hub_to_contact(cam, uv, r_m)`: the ray meets `z = r`, then drops vertically.

- **Pixel space and image size** come from the `GroundModel` adapter (§4.2):
  - fit targets: `world_to_undistorted` of the zone grid;
  - consistency check: `distort(project(X, z = 0))` against `world_to_image(X)`;
  - principal-point prior: toward the centre of `image_size`.
  
  `fit` takes a `GroundModel`; a plain homography is wrapped, and fails without an image size.
- **No OpenCV.**
  - `cv2.Rodrigues` becomes `scipy.spatial.transform.Rotation.from_rotvec` / `as_rotvec`.
  - The initial plane homography (`cv2.findHomography(…, 0)`, plain least squares) becomes a normalised DLT in numpy.
  - The ported `test_camera` checks parity, and so does the NW-clip camera of §14.3 (f = 319.55 px, (cx, cy) = (334.10, 240.13), height 6.73 m, within 0.01 px / 0.001 m).

`VehFootPrinter` fits the camera lazily, the first time a locator needs it. A failed consistency check makes M4 and M7 fail with `camera_invalid` (base design §6.1; M7 design §7.5). The fit and its residuals go into `VehFootPrinter.info` (§11.4).

## 9. Grid, M2 pass and reference heading

Inside `locate()`, before any chain runs:
1. **M1 points** for every row. Rows that fail the projection gate (§7.10) get NaN, not a point.
2. **Grid:** `smoother.segment(m1, allow_missing=True)` (§5.2). Every method uses this grid. Rows with an invalid projection keep their identity:
   - **one invalid row (or a short run) between valid observations:** it stays in the segment, unmeasured, `fill_reason = "no_position"`;
   - **a run too long to bridge:** the track splits there, and the run gets `seg = −1`;
   - **an invalid prefix or suffix, or a track with no valid projection at all:** `seg = −1`.
   
   In every case these are `box_rejected` rows (§7.10): `fp_fail` starts with `projection`, and no **box-derived** ground point is invented for them. A `box_rejected` row inside a segment can still receive an admitted M7 pose (§7.7), and then `fp_source = "M7"` and `fp_support = "kinematic"`. Otherwise `fp_source = "none"`. `seg = −1` rows are never admitted, and reach every later step with NaN outputs.
3. **M2 points** for every row with a valid projection.
4. **Reference pass:** smooth M2 on the grid (`segmented=True`), compute velocities, and apply the motion-heading rules (§10.1) to get `ref_heading_deg`. It is needed by M3, M4 (pair acceptance, front) and M7 (initial heading), and stored for M5/M6 axis direction.

**Dropped from the study:** the approach-direction fallback for tracks that never move (it needs movement labels, which `tssa` inputs do not carry). Those rows get NaN `ψ_ref`; M3 and M4 fail with `no_heading`; M7 starts from the path direction, as `m7_runner._init_pose` already does.

**Segmentation basis.** The study split on **pixel** jumps (0.5 box heights per frame, `SPLIT_SPEED`, `SPLIT_LOGH`). `TrackSmoother` splits on ground speed. The vehicle defaults in §11.3 use ground speed. The validation run (§14) counts segments against the study's grid and reports the difference. A pixel-jump rule in `tssa.smooth` is a follow-up, not part of this round.

## 10. Orientation and polygons (after smoothing)

`polygons(table)` takes the `SpeedCalculator` output (the `locate()` columns, smoothed positions, `vx, vy`).

### 10.1 Motion heading

**Units.** The heading is unit-free, but the reliability threshold is not.
- `polygons(table, speed_unit=None)` computes the speed itself from `vx, vy`.
- Their unit is the `speed_unit` argument, else `table.attrs["speed_unit"]` (`"mph"` or `"m/s"`, as `SpeedCalculator` writes it), and the threshold is converted to it.
- If both are given and differ, or the unit is unknown, or neither is given: `ValueError`. Some pandas operations drop `attrs`, so the message says to pass `speed_unit`.

Per `(track, seg)` (base design §8.2):
- `heading_reliable` at speed ≥ `heading_min_mph` (3);
- otherwise hold the last reliable heading, and back-fill before the first;
- a segment that is never reliable gets NaN motion heading (handled in §10.2).

### 10.2 Footprint orientation

Per `(track, seg)`:
1. **Native rows:** those whose supplying locator gave an orientation (M3–M7). Every native orientation is folded to an axis for smoothing; the directed value is kept for step 4.
2. **Axis smoothing, by the number `n` of native rows in the segment** (as the study's `smooth_orientation`):
   - `n = 0`: no native axis;
   - `n = 1`: that axis, unchanged;
   - `n ≥ 2`: RTS on the doubled-angle unit vector (`rts_smooth`, guarded and correctly started, §5.3; `orient_q` 1.0, `orient_sigma` 0.1; study `ORIENT_Q`, `ORIENT_SIGMA`, both untuned).
3. **Reach:** the axis is used only within `orient_reach_frames` (10) of a native row (R10).
4. **Direction of an axis**, first available:
   1. the row's motion heading (reliable, held or back-filled; §10.1). `front_source = motion`;
   2. the segment never moves: the nearest directed native orientation in the segment (M7's `ψ`, M4's `d_front`, M3's `ψ_ref`). `front_source = native`;
   3. the motion heading of the same track's nearest other segment in time. `front_source = track`;
   4. none: the axis is kept with an arbitrary direction. `front_source = none`, `front_known = False`.
5. **Rows without a native axis in reach** use the motion heading, or else step 4.3. If neither exists, `orient_deg` is NaN and the row has no corners.

`orient_source` = `native` / `motion` / `none`. `front_known` is True unless step 4.4 was used.

**Why fold, and when it matters.**
- **While the vehicle moves**, folding directed headings (M3, M4, M7) to axes loses nothing. M7's front rule is `travel`, so its front already faces the motion. For M4, the front correction already points `d_front` within 90° of `ψ_ref`.
- **A vehicle that never moves** keeps its measured axis (step 4.2–4.4), so a parked bus with a passed M7 solve, or a stopped car with good M6 faces, still has an oriented footprint. Only its front may be unknown.

The directed native value stays in `fp_orient_deg`, and M4's `m4_front_conflict` and M7's `m7_reversing` remain as flags.

### 10.3 Size

Per track and segment, in this order:
1. **M7 passed solve** (the segment's M7 solve has no fail reason): the solve's fitted `L` and `W` for **every** row of that segment (M7 design §9.3). `dims_source` = `m7_measured` / `m7_partly_prior` / `m7_prior` (M7 design §8.3, with the width-hold rule).
2. **Measured classes** (`m6_measured_size_classes`, default `(2,)`): the median M6 length and width over the track's M6 rows, when there are at least `measured_size_min_rows` (5; my call). `dims_source = m6_measured`.
3. Otherwise the class's nominal size (`size_ft`). `dims_source = nominal`.

### 10.4 Corners and output

`corners = rect(centre = smoothed position, orient_deg, length, width)`, front-left first. Rows supplied by a `point_only` locator (M1), or without a smoothed position, get NaN corners and `dims_source = none`.

Added columns: `motion_heading_deg, heading_reliable, orient_deg, orient_source, front_source, front_known, length_ft, width_ft, dims_source, c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y` (corners in world m). With `front_known = False`, the corner set is correct but which end is "front-left" is arbitrary.

`VehFootPrinter.to_geodataframe(table, crs=None)` returns one shapely polygon per row (empty where corners are NaN).

**Low confidence** (M7 design §9.6, `combine_pet.low_confidence`): `fp_low_confidence = True` on rows of a failed M7 solve that fell back, or on admitted `kinematic` rows more than `lowconf_s` (2.0 s) from the nearest supported row. This flag is what a PET step needs.

## 11. Public API

### 11.1 Construction and the main calls

```python
from tssa.footprint import VehFootPrinter, FootprintConfig, detect
from tssa.transform import DistortedHomography

dh = DistortedHomography.load_json("model_a1a_nw.json")
tables = detect.run(video, tracks, detectors=[detect.KeypointDetector(),
                                              detect.CuboidDetector(uod_ckpt, uod_root),
                                              detect.MaskSegmenter(sam_ckpt)], cache_dir=cache)
vfp = VehFootPrinter(dh, zone=zone_from_points(gcps_utm, buffer_ft=15), workers=12, cache_dir=cache)

out = vfp.estimate(tracks, **tables)                 # locate -> smooth -> speed -> polygons
gdf = vfp.to_geodataframe(out, crs="EPSG:32617")

# infer only what this configuration needs (masks for buses and trucks only, with the default chains)
tables = detect.run(video, tracks, detectors=[...], rows=vfp.required_rows(tracks), cache_dir=cache)
```

```python
class VehFootPrinter:
    def __init__(self, homography, config: FootprintConfig | None = None, image_size=None, zone=None,
                 camera=None, smoother: TrackSmoother | None = None, speed: SpeedCalculator | None = None,
                 workers: int = 1, cache_dir=None, **config_overrides) -> None: ...
    def locate(self, tracks, only_tracks=None, **tables) -> pd.DataFrame: ...
    def polygons(self, table, speed_unit=None) -> pd.DataFrame: ...
    def required_rows(self, tracks, only_tracks=None) -> dict[str, pd.DataFrame]: ...
    def estimate(self, tracks, only_tracks=None, **tables) -> pd.DataFrame: ...
    def compare(self, tracks, methods=("M1", "M2", "M3", "M4", "M5", "M6", "M7"), fallback="M2",
                admit_kinematic=None, only_tracks=None, **tables) -> pd.DataFrame: ...
    @staticmethod
    def to_geodataframe(table, crs=None): ...
    @staticmethod
    def available_locators() -> dict[str, dict]: ...
    info: dict      # camera fit and consistency, grid counts, M7 solve summaries (one row per solve)
```

- `homography`, `image_size`: wrapped in `GroundModel` (§4.2); construction fails without an image size.
- `camera`: pass a fitted `PinholeCamera` to skip the fit.
- `smoother`: a user smoother must use `xy_cols=("fp_x", "fp_y")` and `cov_cols=("fp_cov_xx", "fp_cov_xy", "fp_cov_yy")`; anything else raises `ValueError` at construction, so the fill-scaling rule of §7.9 cannot be bypassed by accident.
- `**tables`: the detection tables by name, so a later locator needs no new parameter.
- Errors raise `ValueError` early, as the existing classes do: bad config, an unknown locator name, a chain naming an unregistered locator, a column clash, a table missing its columns.

### 11.2 `compare()`

Runs each method as a single-locator chain on the **same grid and reference heading**, smooths and polygons each, and returns one long table with a `method` column.

**Modes.**
**One precedence rule for `admit_kinematic` in `compare()`:**

| `fallback` | `admit_kinematic` | Result |
|---|---|---|
| `None` (native-only) | `None` or `False` | admission off |
| `None` (native-only) | `True` | `ValueError`: admission contradicts native-only |
| set (end-to-end) | `None` | `config.admit_kinematic` (default True) |
| set (end-to-end) | `True` / `False` | as given |

- **Native-only** means measurements of the method's own observations only. A `kinematic` M7 row has no measurement: `fp_source = none`, `fp_support` empty, `fp_fail = "M7:no_observation"`, while `m7_support = kinematic` stays in the extras.
- **End-to-end** (base design §7, without eligibility): admitted rows have `fp_source = M7` and `fp_support = kinematic`; rows the method fails on take the fallback.

**Row contract: every input row, once per method.** No row is ever dropped, so the per-method tables pair row by row.
- Rows a method does not cover keep their identity, with `fp_source = none` and `fp_fail = "<method>:class_ineligible"`. Today that is M7 on classes outside `m7_classes`, such as cars.
- **The fallback is not applied to `class_ineligible` rows.** An "M7 end-to-end" car would otherwise be pure M2 under the M7 label, and paired M7-vs-M6 comparisons on cars would compare M2 with M6.
- Rows outside `only_tracks` are present with `fp_source = none` and `fp_fail = "excluded"`.

It is what a project needs to rebuild the study's comparison: offsets between methods, speed traps, its own metrics.

### 11.3 Vehicle defaults inside `estimate()`

`TrackSmoother`'s defaults are tuned for pedestrians: they would cut vehicle tracks at every frame faster than 6 m/s (13 mph). When no smoother is passed, `VehFootPrinter` builds:

```python
TrackSmoother("rts", params={"q": 2.0},                 # study RTS_Q, m²/s³ (gap regression: turn error 0.116 mph)
              xy_cols=("fp_x", "fp_y"), cov_cols=("fp_cov_xx", "fp_cov_xy", "fp_cov_yy"),
              fps=config.fps, split_speed=30.0,         # m/s (~67 mph): a jump or id switch; my call, §9
              fill_max_gap=20, gap_max_speed=26.8)      # frames; m/s = 60 mph (study FILL_MAX_GAP, GAP_MAX_SPEED_MPH)
SpeedCalculator(fps=config.fps, xy_cols=("fp_x_smooth", "fp_y_smooth"))   # mph, the tssa default
```

`estimate()` calls `smoother.smooth(loc, segmented=True)` on `locate()`'s output (§5.2): no re-segmentation, every row returned, and rows without a reference point unmeasured. `polygons()` reads the speed unit from the calculator's `attrs` (§10.1). `estimate()` passes it explicitly, so the result does not depend on `attrs` surviving.

### 11.4 `info`

The camera (`f`, `cx`, `cy`, height, consistency RMS/max), segment and fill counts, per-locator success and failure counts, and the M7 per-solve table (study `m7_tracks.csv` columns). These are what the study's manifest recorded and what a user needs to judge a run.

## 12. Configuration

### 12.1 `FootprintConfig` (frozen; `with_overrides()` like `SmoothConfig`)

Distances and sizes in **feet** (user Q2); positions in world metres; speeds in mph.

| Field | Default | Source |
|---|---|---|
| `fps` | 10.0 | study clip |
| `box_cols`, `cls_col` | `("x", "y", "w", "h")`, `"cls"` | DNT track files |
| `size_ft` | `{2: (15, 6), 5: (40, 8.5), 7: (25, 8)}` (L, W) | base design §4, user decision |
| `priors_ft` | height, wheelbase, track width, front/rear overhang per class (base design §4 table) | M4 |
| `wheel_r_ft` | car 1.1, truck/bus 1.6 | base design §6.1 |
| `default_size_ft` | `(15, 6)` | my call |
| `chains`, `default_chain` | §7.8; `("M2",)` | |
| `max_ft_per_px`, `gate_buffer_ft`, `edge_px` | 2.0, 100, 2 | base design §7 |
| `sigma_k` | 0.03 | `RTS_SIGMA_K` |
| `fill_meas_scale` | 4.0 (std-dev multiplier on `interp = 1` rows, §7.9) | `FILL_MEAS_SCALE` |
| `heading_min_mph` | 3 | `HEADING_MIN_MPH` |
| `orient_q`, `orient_sigma`, `orient_reach_frames` | 1.0, 0.1, 10 | `ORIENT_*`, R10 |
| `m3_resid_px` | 0.5 | `m3.RESID_OK_PX` |
| `m4` | the base-design §6.1 tolerances (`WHEELBASE_TOL` 0.30, `AXIS_TOL_DEG` 30, `PARALLEL_TOL_DEG` 15, `TRACK_TOL` 0.30, `LONG_OFFSET_FRAC` 0.25, `MIN_CONTACT_SEP_FT` 0.5, `NEAR_DV_PX` 3, `NEAR_BOTTOM_FRAC` 0.10, `CAMERA_AWAY_DOT` 0.5, front-conflict 10 rows / 0.5), `wheel_landmark = "hub"` | study `settings.py` |
| `m5_min_cols`, `m5_max_resid_ft` | 8, 1.0 | `M5_*` |
| `m6_size_gate`, `m6_min_iou`, `m6_measured_size_classes` | 0.5, 0.3, `(2,)` | `M6_*` |
| `measured_size_min_rows` | 5 | my call |
| `m7` | `M7Config` (§12.2) | |
| `admit_kinematic`, `kin_meas_scale`, `kin_sd_floor_ft`, `lowconf_s` | True, 2.0, 0.5, 2.0 | `M7_KIN_*`, `M7_LOWCONF_S` |
| `camera_rms_px`, `camera_max_px`, `camera_pp_prior` | 1.0, 3.0, 1e-3 | base design §5, R18 |

### 12.2 `M7Config`

A frozen dataclass holding every `M7_*` setting of the study's `settings.py` as of 2026-11-10, with the same values and their comments as docstrings. These include the prior sets (bus, truck, pickup/van), subtypes, loose width priors, inset priors, gates, σ values, solver iteration limits, front rule, width hold and pair start. The study code reads module globals (`S.M7_*`); the port passes an `M7Config` instead. Its hash enters the solve cache key.

## 13. Module layout

```text
src/tssa/smooth/config.py, methods.py, smoother.py, segments.py
                                                      + cov_cols, segment(allow_missing), smooth(segmented), rts_smooth fixes   §5
src/tssa/transform/homography.py                      lazy cv2 (below)
src/tssa/transform/ground.py                          GroundModel adapter                         §4.2
src/tssa/transform/camera.py                          PinholeCamera, fit, consistency, hub_to_contact (numpy/scipy only)   §8
src/tssa/footprint/
    __init__.py          VehFootPrinter, FootprintConfig, M7Config, register_locator, zone_from_points
    config.py            FootprintConfig, M7Config
    geometry.py          compass helpers, rect_corners, away_dir, jacobian, to_local/to_utm   (study geometry.py)
    match.py             match_frame, compute_iou                                           (study infer.py)
    masks.py             MaskTable, lower_outline, contour helpers                          (study m5.py, m7_candidates.py)
    locators/
        __init__.py      registry, LocateContext, LocatorResult
        m1_m2_m3.py      (study geometry.py, m3.py)
        m4.py            (study m4.py)
        m5.py            (study m5.py)
        m6.py            (study m6.py)
        m7/
            candidates.py  (study m7_candidates.py)
            solver.py      (study m7.py)
            runner.py      segment jobs, process pool, solve cache, native rows   (study m7_runner.py)
    orientation.py       motion-heading rules, axis smoothing                               (study smooth.py, products.py)
    printer.py           VehFootPrinter
    detect/              optional: KeypointDetector, CuboidDetector, MaskSegmenter, run, cache   (study infer.py)
tests/
    smooth/test_cov_and_grid.py
    transform/test_camera.py
    footprint/  test_geometry.py  test_match.py  test_m1_m2_m3.py  test_m4.py  test_m5.py  test_m6.py
                test_m7.py  test_m7_candidates.py  test_m7_runner.py  test_orientation.py  test_printer.py
                m7_synth.py  conftest.py
```

Helpers stay inside `footprint/` (AGENTS.md: keep packages narrowly scoped); `engine/` is not touched.

**Lazy OpenCV in `homography.py`** (review comment 6: today it imports `cv2` at module load, so `import tssa.transform` fails without OpenCV):
- `import cv2` moves into the functions that fit (`cv2.findHomography`).
- **Projection** (`_project`) becomes the homogeneous product in numpy, replacing `cv2.perspectiveTransform`. It must agree with the current results to 1e-9 on the existing `test_homography` cases, and those tests must pass unchanged.
- The constructor default `method: int = cv2.RANSAC` becomes `method: int | None = None`, meaning RANSAC, resolved inside the fit. Passing OpenCV constants still works.
- **Guarantee:** `load_json` / `load_matrix` and both projection directions work without OpenCV; constructing a homography from point pairs needs it.

**Provenance.** Every ported file starts with a docstring naming its study source file and that file's SHA-256, as listed in the snapshot manifest (§17, phase 0). The manifest, not the per-file docstrings, establishes that all ported files come from one state.

## 14. Tests and validation

### 14.1 Ported synthetic tests

The study's suite had 210 tests for M1–M6 and 53 more for M7 at the last count. The method tests are ported with their fixtures:
- `test_geometry`, `test_m3`;
- `test_m4`: frontal/rear/oblique; the unequal-overhang bus under all four label permutations to 1e-6 ft; malformed-pair rejection; camera-away success; misaligned far pairs; head-on `side_ambiguous`; reversing bus; elevated hub to 0.05 ft;
- `test_m5`, `test_m6`, `test_camera`;
- `test_m7` and `test_m7_candidates`, using `m7_synth.py`: the 40 ft bus on a 90° right turn with 0.5 px noise, the association examples of test 21, and the rest;
- the parts of `test_smooth` covering motion heading and orientation smoothing.

Tests of the study's evaluation machinery (products, metrics, stops, sampling, clips, export) are not ported (§2).

### 14.2 New tests

**`tssa.smooth` (§5)**
1. **Covariance:**
   - `cov_cols` with `R = σ²I` gives the same result as `sigma_col = σ` on `interp = 0` rows;
   - on `interp = 1` rows the covariance is **not** inflated, while `sigma_col` is (§5.1);
   - a non-PD `R` raises;
   - the existing smoother tests pass unchanged.
2. **Segmenting with missing positions** (`allow_missing=True`, review comment 1):
   - one invalid row between valid observations stays in its segment, unmeasured, `fill_reason = "no_position"`;
   - an invalid run longer than `fill_max_gap` splits the track and gets `seg = −1`;
   - an invalid prefix and an invalid suffix get `seg = −1`;
   - a track with no valid position is returned whole, with `seg = −1`;
   - every input row is returned once.
3. **Segmented smoothing** (review comment 2):
   - **stale positions:** segment on M1 positions, then smooth M6 positions that are deliberately offset by 3 m. The output follows M6, not M1, for `rts`, `ma` and `sg`, and for a 1-row segment;
   - **large origin, two tolerances**, both absolute (`rtol = 0`; a default relative tolerance at UTM scale would hide millimetres), on coordinates with six decimals (e.g. 5200000.123456 m):
     - **internal:** the local ↔ world conversion, before output rounding, round-trips to 1e-6 m;
     - **public:** the returned `<pos>_smooth` columns are within 0.0005 m. The smoother rounds its added columns to three decimals in `_finish()`, and **segmented mode keeps that rounding**, the same as the legacy path. No output precision changes.
   - **failed method:** with every position NaN, a segment returns NaN and `fallback = 1`, without raising;
   - **structural columns:** input with `seg`/`fill_reason` does not clash, and a non-consecutive segment raises.
4. **Admission** (review comment 3): an M7 point on an inserted `interp = 2` frame changes the smoothed result; the same table with that point removed does not use it.
5. **`rts_smooth`:**
   - 0 and 1 measured rows raise `ValueError`;
   - exact constant-velocity data with a 10-frame missing prefix recovers frame 0 to 1e-6 m (today: 1.54 m off);
   - segments that start with a measurement give the same result as before.

**Ground model and camera (§4.2, §8)**

6. **Adapter:**
   - a `HomographyTransform` has identity distortion and needs an explicit `image_size`;
   - a `DistortedHomography` gives its own size, and a conflicting explicit size raises;
   - an object with only `undistort` raises.
7. **Camera fit:** fits through both an ordinary and a distorted homography (synthetic radial distortion), with consistency residuals under 1e-6 px on exact data.
8. **Lazy imports** (review comment 6), with `cv2`, `torch` and `ultralytics` blocked in `sys.modules`:
   - `import tssa`, `tssa.transform` and `tssa.footprint`;
   - `DistortedHomography.load_json` of a saved model, and projection in both directions;
   - `PinholeCamera.fit`;
   - an M2-only and an M6-only `estimate()`.
   
   Fitting a homography and running M5 raise `ImportError` naming OpenCV.

**Footprint**

9. **Chains:**
   - a truck row: M7 failed solve → collapsed M6 face → M3 → with no heading, M2; `fp_fail` lists each failure in order;
   - a car without cuboids: `M6:not_supplied`, then M2.
10. **Covariance per support type** (§7.9): an `interp = 1` box row has `16 · R`; an admitted M7 row has exactly the §7.7 covariance.
11. **Size:** an M7 passed segment uses the fitted L and W on every row, fallback rows included; a car with 5 M6 rows uses their median, with 4 nominal; a bus is always nominal unless M7 passed.
12. **Orientation** (review comment 4):
    - no native axes: the motion heading is used, without error;
    - one native axis: that axis within reach;
    - an axis around 179°/1° never smooths to 90°;
    - the reach rule;
    - direction toward motion.
13. **Never-moving vehicles** (review comment 5):
    - a stationary car with M6 faces keeps the M6 axis. `front_source` is `track` when another segment of the track moves, otherwise `none` with `front_known = False`. Corners are present either way;
    - a stationary M7 bus takes its direction from `ψ` (`front_source = native`);
    - a car that waits, then drives: the waiting rows are back-filled from the first moving heading (`front_source = motion`).
14. **Speed units** (review comment 8): one track through `SpeedCalculator(units="metric")` and `units="imperial"` gives identical `heading_reliable` and `orient_deg`. A table without `attrs` and without `speed_unit` raises.
15. **`compare()`** (first review 9 and 10; second review 3):
    - **`fallback=None`:**
      - with `admit_kinematic=None` and with `False`: the `kinematic` M7 row has no measurement, `fp_source = none`, `fp_support` empty, `fp_fail = "M7:no_observation"`, and `m7_support = kinematic` stays;
      - with `True`: `ValueError`;
    - **`fallback="M2"`:**
      - with `admit_kinematic=None` (config default True) and with `True`: the same row is admitted, `fp_source = M7`, `fp_support = kinematic`;
      - with `False`: it takes M2;
    - a mixed car/bus input returns every row once per method, with M7 car rows `class_ineligible` and no fallback, in both modes;
    - `only_tracks` rows are `excluded`.
16. **End to end:** a straight car and a turning car rendered to boxes and cuboid faces through a synthetic camera with 0.5 px noise and a 15-frame hole. The centre is within 0.3 ft, the heading within 2°, and the speed within 0.2 mph of truth (base design §8.2 bound).
17. **Every input row returned once** by `locate()` and `estimate()`, including invalid-projection rows.
18. **Admission through the real pipeline** (second review 1). `estimate()` runs on a synthetic bus with a passed M7 solve (a stub M7 locator in phase 2, the real one in phase 5), whose segment contains:
    - an **inserted** gap frame (`interp = 2`, NaN box fields, no Jacobian): it is admitted, `fp_row_kind = inserted`, and its `fp_cov_*` equal the §7.7 covariance exactly. The smoothed position moves toward the M7 pose compared with the same run with `admit_kinematic=False`;
    - an **original `box_rejected` row** inside the segment with `kinematic` support: it is admitted, and `fp_fail` still starts with `projection`;
    - an original `box_rejected` row with **native** M7 support: not used by M7, so the next locator in the chain is tried;
    - a **`seg = −1`** prefix row: never admitted, NaN outputs.
19. **Width hold** (second review 2): `w_ft_per_px` = 0.29, exactly 0.3, 0.31 and `inf`, with threshold 0.3, gives held = False, False, True, True, and `None` as the threshold never holds. This is the same decision as the study's expression, evaluated on those inputs.

**Detectors (§6)**

20. **Matching:** a dense frame with crossing boxes is assigned one-to-one; an IoU under 0.3 is unmatched.
21. **Row selection** (review comment 11):
    - `required_rows` with a custom car chain containing M5 lists car rows under `masks`;
    - the default chains list only cls 5/7 rows there;
    - in a crowded frame, requesting one track's rows gives the same match as requesting all rows (matching context is the whole frame). This is tested with a fake detector, so no model is needed.

### 14.3 Validation run on the NW clip (not a unit test)

On the study's pinned inputs (`footprint/inputs/`), fed with the study's caches converted to the §4.3 tables:

| Check | Pass condition |
|---|---|
| Camera | f, (cx, cy) and height equal to the run report's (319.55 px, (334.10, 240.13), 6.73 m) within 0.01 px / 0.001 m |
| M1, M2, M3, M4, M5, M6 native points on the study's grid | equal to the study's native tables (run `20261107_222536_s0`) to 1e-6 m |
| M4 `label_case`, `interior_source`, `front_conflict` | counts equal to the run report (2,185 valid; tracks 230 and 432 flagged) |
| M7 solves on the study's display segments, solved fresh by the port (not read from the study's solve cache), with the study's frames, observations and initial poses | pass/fail and fail reasons equal to run `20261110_161113_s0`; fitted L and W within 0.05 ft. About 46 min with 12 workers. |
| Detectors | on 200 sampled rows, same device and package versions: `detect.run` reproduces the study's cache rows (keypoints and cuboid corners within 0.5 px; mask IoU ≥ 0.99) |
| Grid | segment count vs the study's pixel-jump grid; the difference is reported, not gated (§9) |
| Smoothed M6 centres vs the study's M6 display | median and p95 difference reported, not gated (study products used per-product segments) |
| Speed trap (two screen lines 92.71 ft apart, NB/SB through vehicles) | \|error\| p90 within the study's 0.38 mph |

The results are written into this document as a new §19.

## 15. Deliberate changes from the study

| Change | Why | Effect |
|---|---|---|
| One product per call; `compare()` instead of metric-native / metric-e2e / display | a library has no eligibility rules or clip sample | no sample-independence guarantees; users build their own metric sets |
| M7 solves per segment only (display solve) | one product | metric-run solves are not available |
| One SAM pass for M5 and M7: every component's contour plus the raw-mask lower outline; M7 applies the 5% component rule itself | the study ran SAM twice (M5 outlines, M7 contours) | M5 lower outlines identical to the study's on 200 of 200 sampled rows; M7 masks IoU ≥ 0.99 after the 5% rule (§19) |
| Approach-direction fallback dropped | needs movement labels | never-moving tracks: M3/M4 fail `no_heading`; M7 starts from the path direction |
| Segmentation on ground speed, not pixel jumps | `TrackSmoother` has no pixel rule | segment boundaries may differ; counted in §14.3 |
| Native orientations folded to axes, directed toward the final motion heading | one rule for mixed-source tracks | equal to the study's while moving; differs only while stopped (hold rule) |
| Never-moving segments keep their native axis; direction from a directed native or another segment, else `front_known = False` | the study's approach fallback is gone (§9) | stationary vehicles keep an oriented footprint |
| Invalid-projection input rows are retained without box-derived measurements. Prefixes, suffixes and unbridged missing runs get `seg = −1` and no position. Interior rows in valid segments may receive smoothed predictions, or admitted M7 kinematic measurements under §7.7. | every row returned; no box-derived ground point from a rejected projection | the study built segments from the boxes, so invalid-projection rows at a display segment's ends were also predicted there; here only interior rows are |
| `rts_smooth` starts at the first measurement | segments may now start unmeasured | identical wherever the first row is measured (every study case) |
| Camera without OpenCV (scipy rotations, numpy DLT) | OpenCV-free core | parity checked in §14.3 |
| `R_override` replaced by covariance columns | general mechanism | identical numbers |
| Settings passed as config objects, not module globals | library use, two printers in one process | identical numbers |

## 16. Decisions on the open points (user, 2026-11-10)

- **Q6. Freeze point for M7: snapshot now.** The user has reviewed the M7 clips, so the snapshot (§17, phase 0) is taken now. The study's code has not changed since run `20261110_161113_s0`: the newest code file is `settings.py` (2026-11-10 16:10, just before that run), and the later edit to the M7 notes (18:50) only reports that run. The snapshot is therefore exactly the state §7.6 describes. Later study changes are ported deliberately, each with a new snapshot and a validation rerun.
- **Q7. Method names: `M1`–`M7` only, for now.** Descriptive aliases (`bottom_centre`, `nominal_shift`, `nominal_box`, `wheel_keypoints`, `sam_outline`, `cuboid`, `wheel_fit`) are added later, as a follow-up outside phases 0–6. (Done in 0.0.2.12, §7.8.) The registry (§7.1) must therefore allow more than one name per locator without changing its result columns: `fp_source` keeps the canonical `M` name.

## 17. Build order

Each phase ends with its tests passing and a commit. The version bumps by one patch per phase.

0. **Study snapshot** (review: source handoff). A read-only archive of the study folder's code, documents and settings, plus `docs/study-snapshot-<date>.json` committed in `tssa`. The manifest records:
   - every `.py` and `.md` file with its SHA-256;
   - the full `settings.py` values;
   - the column schemas and one example row of each cache table (`m4`, `m5`, `m6`, `m7`, `m7solve`);
   - `pip freeze` of the study's Python 3.14 environment, and the model checkpoint hashes;
   - the run directories used in §14.3.
   
   Code is ported only from the archive. Taken now (Q6, §16).
1. **`tssa.transform` and `tssa.smooth` foundations:**
   - lazy OpenCV in `homography.py` and the `GroundModel` adapter (§4.2, §13);
   - `cov_cols`, `segment(allow_missing)`, `smooth(segmented)` and the `rts_smooth` fixes (§5);
   - tests 1–6, plus test 8 for the transform part;
   - the existing smooth, speed and homography tests unchanged.
2. **Core footprint:** config, geometry, registry, M1–M3, M6, chains, grid and reference pass, orientation, polygons, `estimate()`, `compare()`; ported tests for geometry, M3 and M6; tests 9–18 (M7 parts with a stub locator).
3. **Camera and M4, M5:** `transform/camera.py` without OpenCV, M4, M5, `masks.py`; ported tests; test 7; the rest of test 8.
4. **Detectors:** `match.py`, `detect/` with cache, `required_rows()`; tests 20–21; detector check of §14.3.
5. **M7:** candidates, solver, runner with workers and cache, admission, low-confidence flag; ported M7 tests; tests 4, 10, 13, 15 and 18 against the real locator; test 19.
6. **Validation run** (§14.3); results into §19; `README.md` section; `pyproject.toml` extra `detect`.

## 18. Left to the builder

- `LocatorResult` and `LocateContext` field details.
- How M7's process pool is started on Windows (`spawn`) and on Linux.
- Whether `split_speed = 30 m/s` needs changing after the §14.3 segment count; reported either way.
- Whether to vectorise M3 and M5 (the study solves them row by row).

## 19. Validation results (NW clip, 2026-11-11)

Run with `docs/validation/nw_clip_validation.py` (parts `native`, `grid`, `e2e`, `trap`, `m7`, `m7study`, `m7compare`,
`estimate`) and `docs/validation/detector_check.py`, on the pinned inputs of §14.3 and the study code from the
snapshot (§17 phase 0). The port ran on Linux (WSL2, Python 3.12, scipy 1.17); the study's reports were made on
Windows (Python 3.14).

| Check | Result | Verdict |
|---|---|---|
| Camera | f 319.5459, (cx, cy) (334.1009, 240.1256), height 6.7331 m; largest difference from the study 5e-8 | pass |
| M1, M2, M3, M6 native points | 46,213 / 46,213 / 46,213 / 34,628 rows valid in both, none in one only; max difference 0 m | pass |
| M4 native points | 2,364 rows valid in both; max difference 1.4e-9 m; `label_case` and `interior_source` equal | pass |
| M4 counts (metric `psi_ref`) | 2,185 valid (2,175 `camera_away`, 10 `other_pair`); `front_conflict` on tracks 230 and 432 | pass |
| M5 (`solve_m5` on the study's outlines) | 23,526 rows valid in both, no disagreement, max difference 0 m | pass |
| M7 port vs the study's solver, same jobs, same machine | 25 / 25 same pass/fail and reason; 17 passed; L and W differences 0 ft on all | pass |
| M7 port vs the study's run report (`20261110_161113_s0`) | 23 / 25 same pass/fail (315 `param_bound` and 452 `wheel_resid` here, passed there); on the 17 passed in both, \|dL\| median 0.11 ft, max 2.97 ft (track 176), 6 within 0.05 ft; \|dW\| max 0.21 ft | platform noise, see below |
| Detectors, 200 sampled rows | keypoints 66 / 66 rows, max 0.15 px; cuboid corners 66 / 66, max 0.05 px; masks 68 / 68 with IoU ≥ 0.99 (min 0.99996) after the 5% component rule; M5 lower outlines from the port's single SAM pass identical to the study's M5 cache on 200 / 200 rows | pass |
| Grid | 445 segments here, 732 in the study (426 labelled tracks) | reported, see below |
| Smoothed M6 centres vs the study's M6 display | 32,852 rows; median 0.0014 ft, p95 0.18 ft | reported |
| Speed trap, NB/SB through vehicles common to both | \|error\| p90 (mph) here / study: M1 0.332 / 0.332, M2 0.365 / 0.365, M3 0.384 / 0.385, M4 0.365 / 0.361, M6 0.384 / 0.384; medians within 0.006 mph | pass (study's 0.38 mph) |

**M7 and the platform.** The study's own solver, run on the same jobs on the Linux machine, gives exactly the port's
results, and differs from its Windows report in the same way the port does (the two pass/fail flips, the same \|dL\|
per track). The Windows-vs-Linux difference is most likely floating-point differences in the least-squares path (BLAS,
scipy version; not traced further): the fit is multi-start with pruning, so a small change in one start's final cost can change which
start wins. The 0.05 ft gate of §14.3 therefore holds on one platform, not across platforms; across platforms expect
a median \|dL\| near 0.1 ft, rare multi-foot outliers, and some pass/fail changes (2 of 25 here).

**Grid.** Ground-speed segmentation (§9, §15) cuts far less than the study's pixel-jump rule. Of the study's 306 cuts
inside tracks, 292 are between consecutive frames, at a median box jump of 20 px (1.3 m), and 257 of its 732 segments
have 5 rows or fewer. Lowering `split_speed` does not reproduce the study's grid: 456 segments at 20 m/s; 558 at
15 m/s, with 41 tracks cut less and 40 cut more than the study; 2,589 at 10 m/s. §18's question is answered: keep
30 m/s. The coarser grid does not move the smoothed centres materially (row above) and leaves the speed trap unchanged
on common vehicles.

**Speed trap, all vehicles.** On every NB/SB through vehicle (251 to 254 per method, against the study's 139 to 167
eligible ones), the p90 is 0.375 to 0.461 mph; the study's eligibility rules (§15, first row) are not applied here.

**`estimate()` with the default chains.** On the 426 labelled tracks (54,241 rows), with every detection
table, 21 minutes with 10 workers (0.0.2.10, nothing cached). Sources: M6 33,650 rows, M2 10,555, M7 1,980, M3 28,
none 8,028 (rejected projections). M7 solved the 25 bus and truck segments of the port's grid: 16 passed, 4
`param_bound`, 3 `articulated_suspect`, 2 `wheel_resid` (these segments are cut by §9, not the study's display
segments, so the 17 / 25 above is not directly comparable). On bus and truck rows: M7 1,980 (18 of them admitted
`kinematic`), M6 798, M2 735, M3 28; sizes `m7_measured` 394, `m7_partly_prior` 1,106, `m7_prior` 480, `nominal`
1,579. `fp_low_confidence` is set on 1,579 rows, the fallback rows of the failed solves. Speed trap on all 253 NB/SB through vehicles:
\|error\| median 0.20 mph, p90 0.45 mph.

**M7 speed (0.0.2.10).** The 0.0.2.9 solver (the study's algorithm as ported) took about 2 minutes per 70 to 90-frame segment on one uncontended core,
and up to 20 minutes per segment with 10 to 16 solves sharing the machine. Two changes leave the results unchanged (the 25
display solves are identical in pass/fail, F, L and W): a finite-difference Jacobian computed in grouped evaluations
that equals scipy's sparse 2-point Jacobian bit for bit (tested), and racing (from outer round 4, a start, pair
restart or cross refit is dropped while its cost is more than 5% above the best finished fit). One segment
uncontended: 113 s to 53 s (88 frames) and 132 s to 62 s (69 frames). Cheaper search settings (`M7_COARSE_OUTER`,
`M7_SEARCH_TOL`) saved only 11% more and changed 3 pass/fail outcomes, so they are off by default. The runner gives each worker one BLAS
thread; more workers than physical cores lengthens every solve.
