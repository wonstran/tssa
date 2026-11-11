"""VehFootPrinter: vehicle ground footprints from tracked boxes (design docs/2026-11-10-footprint-design.md).

Inputs:  tracks (frame, track, x, y, w, h in distorted image pixels, cls; optional interp), a ground homography,
         optional detection tables by name (cuboids, keypoints, masks) and a camera zone.
Outputs: `locate()`  every input row once (plus the rows the segmentation inserts), with a reference point at the
                     body centre, its 2 x 2 covariance, the supplying locator and why the others failed;
         `polygons()` after smoothing and speed: orientation, motion heading, length and width (ft) with their source
                     and four corners (world m);
         `estimate()` locate -> TrackSmoother(segmented) -> SpeedCalculator -> polygons;
         `compare()`  each method alone (native-only or with a fallback) on one grid, in one long table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..smooth import TrackSmoother
from ..speed import SpeedCalculator
from ..transform.ground import GroundModel
from . import orientation as O
from .config import FT, MPS_TO_MPH, FootprintConfig
from .geometry import away_dir, bottom_centre, direct_toward, edge_flag, jacobian, project, projection_valid, rect_corners
from .locators import REGISTRY, LocateContext, LocatorResult

XY = ("fp_x", "fp_y")
COV = ("fp_cov_xx", "fp_cov_xy", "fp_cov_yy")
SMOOTH_XY = ("fp_x_smooth", "fp_y_smooth")
CORNERS = ["c1x", "c1y", "c2x", "c2y", "c3x", "c3y", "c4x", "c4y"]
LOCATE_COLUMNS = [
    "cls_track", "seg", "seg_len", "fill_reason", "gap_id",
    "fp_x", "fp_y", *COV, "fp_sigma", "fp_source", "fp_support", "fp_fail",
    "fp_orient_deg", "fp_orient_kind", "fp_len_ft", "fp_wid_ft",
    "fp_row_kind", "fp_proj_valid", "fp_edge", "ref_heading_deg",
    "fp_seg_len_ft", "fp_seg_wid_ft", "fp_seg_dims_source",
]  # fmt: skip
POLYGON_COLUMNS = ["motion_heading_deg", "heading_reliable", "orient_deg", "orient_source", "front_source",
                   "front_known", "length_ft", "width_ft", "dims_source", *CORNERS, "fp_low_confidence"]  # fmt: skip
NEEDS_HEADING = ("M3", "M4", "M7")  # locators that read ref_heading_deg
SPEED_UNITS = {"mph": 1.0, "m/s": 1.0 / MPS_TO_MPH}  # heading_min_mph -> unit


class VehFootPrinter:
    """Vehicle footprints from tracked boxes with the speed study's methods M1-M7 (see the module docstring).

    homography:  any object with image_to_world / world_to_image (metres); wrapped in `GroundModel` with
                 `image_size` (needed unless the homography carries one).
    config:      a `FootprintConfig`; keyword overrides are applied on top.
    zone:        shapely polygon (world m) for the camera fit (M4, M7) and the projection gate's zone test.
    camera:      a fitted `PinholeCamera` (skips the fit).
    smoother:    a TrackSmoother using xy_cols ("fp_x", "fp_y") and cov_cols ("fp_cov_xx", "fp_cov_xy", "fp_cov_yy");
                 default: rts with the study's tuning (design 11.3).
    speed:       a SpeedCalculator on ("fp_x_smooth", "fp_y_smooth"); default: mph at config.fps.
    workers, cache_dir: M7 solve processes and solve cache.
    """

    def __init__(self, homography, config: FootprintConfig | None = None, image_size=None, zone=None, camera=None,
                 smoother: TrackSmoother | None = None, speed: SpeedCalculator | None = None, workers: int = 1,
                 cache_dir=None, **config_overrides) -> None:  # fmt: skip
        self.config = (config or FootprintConfig()).with_overrides(**config_overrides)
        cfg = self.config
        self.ground = GroundModel.wrap(homography, image_size)
        self.zone = zone
        self._camera = camera
        self._camera_done = camera is not None
        self.smoother = smoother or TrackSmoother(
            "rts", params={"q": cfg.rts_q}, xy_cols=XY, cov_cols=COV, fps=cfg.fps, split_speed=cfg.split_speed_mps,
            fill_max_gap=cfg.fill_max_gap, gap_max_speed=cfg.gap_max_speed_mph / MPS_TO_MPH,
        )  # fmt: skip
        sc = self.smoother.config
        if tuple(sc.xy_cols) != XY or tuple(sc.cov_cols or ()) != COV:
            raise ValueError(f"the smoother must use xy_cols={XY} and cov_cols={COV} (design 7.9: the printer writes the "
                             "final covariance; a sigma column would be inflated twice)")  # fmt: skip
        self.speed = speed or SpeedCalculator(fps=cfg.fps, xy_cols=SMOOTH_XY)
        if tuple(self.speed.xy_cols) != SMOOTH_XY:
            raise ValueError(f"the speed calculator must use xy_cols={SMOOTH_XY}")
        for c, chain in [*cfg.chains.items(), ("default", cfg.default_chain)]:
            bad = [m for m in chain if m not in REGISTRY]
            if bad:
                raise ValueError(f"chain of class {c} names unregistered locator(s) {bad}; known: {sorted(REGISTRY)}")
        self.workers = int(workers)
        self.cache_dir = cache_dir
        self.local_origin = None  # world offset of the local frame M7 solves in; None = floor of the M1 minimum
        self.info: dict = {}

    @staticmethod
    def available_locators() -> dict[str, dict]:
        from .locators import available_locators

        return available_locators()

    # camera ------------------------------------------------------------------------------------------------------

    def camera(self):
        """The fitted camera, or None when there is no zone or it fails its consistency check (design 8)."""
        if self._camera_done:
            return self._camera
        from ..transform.camera import PinholeCamera, consistency

        self._camera_done = True
        cfg = self.config
        if self.zone is None:
            self.info["camera"] = {"ok": False, "reason": "no zone"}
            return None
        cam = PinholeCamera.fit(self.ground, self.zone, free_principal=cfg.camera_free_principal,
                                pp_prior=cfg.camera_pp_prior)  # fmt: skip
        res = consistency(cam, self.ground, self.zone, cfg.camera_rms_px, cfg.camera_max_px)
        self.info["camera"] = {"f": cam.f, "cx": cam.cx, "cy": cam.cy, "height_m": cam.height, **res}
        self._camera = cam if res["ok"] else None
        return self._camera

    # public ------------------------------------------------------------------------------------------------------

    def locate(self, tracks: pd.DataFrame, only_tracks=None, **tables) -> pd.DataFrame:
        """Reference points before smoothing (design 7.11)."""
        base = self._prepare(tracks, only_tracks, self._needs_heading(self.config.chains, self.config.default_chain))
        chains = {c: self.config.chain_of(c) for c in np.unique(base["cls_track"].dropna())}
        return self._run_chains(base, chains, self.config.admit_kinematic, tables, stop_on_ineligible=False)

    def polygons(self, table: pd.DataFrame, speed_unit: str | None = None) -> pd.DataFrame:
        """Orientation, size and corners after smoothing and speed (design 10)."""
        return _polygons(table, self.config, speed_unit)

    def estimate(self, tracks: pd.DataFrame, only_tracks=None, **tables) -> pd.DataFrame:
        """locate -> smooth (segmented) -> speed -> polygons."""
        loc = self.locate(tracks, only_tracks=only_tracks, **tables)
        return self._finish(loc)

    def compare(self, tracks: pd.DataFrame, methods=("M1", "M2", "M3", "M4", "M5", "M6", "M7"), fallback="M2",
                admit_kinematic=None, only_tracks=None, **tables) -> pd.DataFrame:  # fmt: skip
        """Each method alone on the same grid and reference heading; one long table with a `method` column.

        fallback None: native-only (M7 admission off; admit_kinematic=True raises). fallback set: end-to-end,
        admission from admit_kinematic or the config. Rows a method does not serve are `<method>:class_ineligible`
        and get no fallback. Every input row appears once per method (design 11.2).
        """
        if fallback is None and admit_kinematic is True:
            raise ValueError("admit_kinematic=True contradicts a native-only comparison (fallback=None)")
        admit = False if fallback is None else (self.config.admit_kinematic if admit_kinematic is None
                                                else bool(admit_kinematic))  # fmt: skip
        bad = [m for m in [*methods, *([fallback] if fallback else [])] if m not in REGISTRY]
        if bad:
            raise ValueError(f"unknown locator(s) {bad}; known: {sorted(REGISTRY)}")
        base = self._prepare(tracks, only_tracks, self._needs_heading({0: tuple(methods)}, ()))
        classes = np.unique(base["cls_track"].dropna())
        parts = []
        for m in methods:
            chain = (m,) if not fallback or fallback == m else (m, fallback)
            loc = self._run_chains(base, {c: chain for c in classes}, admit, tables, stop_on_ineligible=True)
            parts.append(self._finish(loc).assign(method=m))
        return pd.concat(parts, ignore_index=True)

    def required_rows(self, tracks: pd.DataFrame, only_tracks=None) -> dict[str, pd.DataFrame]:
        """Per detection table, the rows (track, frame, box) of tracks whose class chain needs it (design 6.4)."""
        df = self._input(tracks)
        cls = _cls_track(df, self.config.cls_col)
        keep = df["track"].isin(only_tracks) if only_tracks is not None else np.ones(len(df), bool)
        out: dict[str, list] = {}
        for c in np.unique(cls.dropna()):
            for name in self.config.chain_of(c):
                for t in REGISTRY[name].needs:
                    out.setdefault(t, []).append(df[(cls == c).to_numpy() & np.asarray(keep)])
        cols = ["track", "frame", *self.config.box_cols]
        return {t: pd.concat(v).drop_duplicates(["track", "frame"])[cols].reset_index(drop=True) for t, v in out.items()}

    @staticmethod
    def to_geodataframe(table: pd.DataFrame, crs=None):
        """One shapely polygon per row (empty where the corners are NaN)."""
        import geopandas as gpd
        from shapely.geometry import Polygon

        c = table[CORNERS].to_numpy(dtype=float).reshape(-1, 4, 2)
        geoms = [Polygon(q) if np.isfinite(q).all() else Polygon() for q in c]
        return gpd.GeoDataFrame(table.copy(), geometry=geoms, crs=crs)

    # steps -------------------------------------------------------------------------------------------------------

    @staticmethod
    def _needs_heading(chains: dict, default_chain) -> bool:
        return any(m in NEEDS_HEADING for ch in [*chains.values(), default_chain] for m in ch)

    def _input(self, tracks: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        need = ["frame", "track", *cfg.box_cols, cfg.cls_col]
        missing = [c for c in need if c not in tracks.columns]
        if missing:
            raise ValueError(f"tracks lack required column(s) {missing}")
        df = tracks[~tracks.duplicated(["track", "frame"], keep="first")]
        return df.sort_values(["track", "frame"], kind="stable").reset_index(drop=True)

    def _prepare(self, tracks: pd.DataFrame, only_tracks, need_heading: bool) -> pd.DataFrame:
        """Every row with its box quantities, the shared grid, M1/M2 points, box covariance and psi_ref."""
        cfg = self.config
        df = self._input(tracks)
        given = list(df.columns)
        clash = sorted((set(LOCATE_COLUMNS) | set(POLYGON_COLUMNS) | {"u_bc", "v_bc"}) & set(given) - {"seg"})
        if clash:
            raise ValueError(f"input columns {clash} clash with columns the footprinter adds; rename them first")
        if "interp" not in df.columns:
            df["interp"] = 0
        df["cls_track"] = _cls_track(df, cfg.cls_col).to_numpy()
        boxes = df[list(cfg.box_cols)].to_numpy(dtype=float)
        uv = bottom_centre(boxes)
        finite_box = np.isfinite(boxes).all(axis=1)
        J = np.full((len(df), 2, 2), np.nan)
        m1 = np.full((len(df), 2), np.nan)
        if finite_box.any():
            J[finite_box] = jacobian(self.ground, uv[finite_box], cfg.jacobian_eps_px)
            m1[finite_box] = project(self.ground, uv[finite_box])
        gate = finite_box & projection_valid(m1, J, cfg.max_ft_per_px, self.zone, cfg.gate_buffer_ft)
        m1[~gate] = np.nan

        # one grid for every method (design 5.2, 9 step 2)
        seg_in = pd.DataFrame({"track": df["track"].to_numpy(), "frame": df["frame"].to_numpy(),
                               "interp": df["interp"].to_numpy(), XY[0]: m1[:, 0], XY[1]: m1[:, 1]})  # fmt: skip
        for c in COV:
            seg_in[c] = np.nan
        struct = self.smoother.segment(seg_in, allow_missing=True)
        df = df.drop(columns=[c for c in ("seg", "interp") if c in df.columns]).assign(_row=np.arange(len(df)))
        base = struct.merge(df, how="left", on=["track", "frame"], validate="one_to_one")
        row = base["_row"].to_numpy()
        has = np.isfinite(row)
        r = row[has].astype(int)

        def spread(a, fill=np.nan):
            out = np.full((len(base),) + a.shape[1:], fill, dtype=float)
            out[has] = a[r]
            return out

        base["cls_track"] = base.groupby("track")["cls_track"].transform("first")
        gate_b = spread(gate.astype(float), 0.0).astype(bool)
        base["fp_row_kind"] = np.where(~has, "inserted", np.where(gate_b, "box", "box_rejected")).astype(object)
        base["fp_proj_valid"] = gate_b
        uvb, Jb, m1b = spread(uv), spread(J), spread(m1)
        base["u_bc"], base["v_bc"] = uvb[:, 0], uvb[:, 1]
        base["m1_x"], base["m1_y"] = m1b[:, 0], m1b[:, 1]
        W = np.array([cfg.size_of(c)[1] for c in base["cls_track"].to_numpy()]) * FT
        m2 = np.full((len(base), 2), np.nan)
        if gate_b.any():
            m2[gate_b] = m1b[gate_b] + (W[gate_b] / 2.0)[:, None] * away_dir(self.ground, uvb[gate_b])
        base["m2_x"], base["m2_y"] = m2[:, 0], m2[:, 1]
        edge = np.zeros(len(base), bool)
        b_box = base[list(cfg.box_cols)].to_numpy(dtype=float)
        edge[has] = edge_flag(b_box[has], self.ground.image_size, cfg.edge_px)
        base["fp_edge"] = edge
        h = b_box[:, 3]
        scale = np.where(base["interp"].to_numpy() == 1, cfg.fill_meas_scale, 1.0)
        s2 = (cfg.sigma_k * h * scale) ** 2
        R = s2[:, None, None] * (Jb @ np.swapaxes(Jb, 1, 2))  # design 7.9 (NaN where no box)
        base["_rxx"], base["_rxy"], base["_ryy"] = R[:, 0, 0], R[:, 0, 1], R[:, 1, 1]
        base["_given"] = has
        base.attrs["given"] = given
        base["_included"] = base["track"].isin(only_tracks).to_numpy() if only_tracks is not None else True
        base["ref_heading_deg"] = self._reference_heading(base) if need_heading else np.nan
        segs = base["seg"].to_numpy()
        self.info["grid"] = {"rows": int(has.sum()), "inserted": int((~has).sum()),
                             "segments": int(len(np.unique(segs[segs >= 0]))), "seg_minus_one": int((segs < 0).sum()),
                             "box_rejected": int((base["fp_row_kind"] == "box_rejected").sum())}  # fmt: skip
        return base

    def _reference_heading(self, base: pd.DataFrame) -> np.ndarray:
        """psi_ref: M2 smoothed on the grid, then the motion-heading rules (design 9 step 4)."""
        tab = base[["track", "frame", "seg", "seg_len", "interp", "fill_reason"]].copy()
        tab[XY[0]], tab[XY[1]] = base["m2_x"].to_numpy(), base["m2_y"].to_numpy()
        tab[COV[0]], tab[COV[1]], tab[COV[2]] = base["_rxx"].to_numpy(), base["_rxy"].to_numpy(), base["_ryy"].to_numpy()
        sm = self.smoother.smooth(tab, segmented=True)
        sp = SpeedCalculator(fps=self.config.fps, xy_cols=SMOOTH_XY, units="metric").calculate(sm)
        sp = base[["track", "frame"]].merge(sp[["track", "frame", "vx", "vy"]], how="left", on=["track", "frame"])
        vel = sp[["vx", "vy"]].to_numpy(dtype=float)
        out = np.full(len(base), np.nan)
        for rows in _segments(base):
            out[rows] = O.motion_heading(vel[rows], self.config.heading_min_mph / MPS_TO_MPH)[0]
        return out

    def _run_chains(self, base: pd.DataFrame, chains: dict, admit: bool, tables: dict,
                    stop_on_ineligible: bool) -> pd.DataFrame:  # fmt: skip
        """Run each class's chain; the first locator that succeeds supplies a row (design 7.7-7.11)."""
        cfg = self.config
        n = len(base)
        kind = base["fp_row_kind"].to_numpy(dtype=object)
        seg = base["seg"].to_numpy()
        cls = base["cls_track"].to_numpy()
        incl = base["_included"].to_numpy(dtype=bool)
        x, y = np.full(n, np.nan), np.full(n, np.nan)
        cov = np.full((n, 2, 2), np.nan)
        source = np.full(n, "none", dtype=object)
        support = np.full(n, "", dtype=object)
        orient, okind = np.full(n, np.nan), np.full(n, "none", dtype=object)
        lft, wft = np.full(n, np.nan), np.full(n, np.nan)
        seg_l, seg_w = np.full(n, np.nan), np.full(n, np.nan)
        seg_src = np.full(n, "", dtype=object)
        fails: list[list[str]] = [[] for _ in range(n)]
        for i in np.flatnonzero(kind == "box_rejected"):
            fails[i].append("projection")
        for i in np.flatnonzero(kind == "inserted"):
            fails[i].append("no_box")
        for i in np.flatnonzero(~incl):
            fails[i] = ["excluded"]
        done = ~incl
        stopped = np.zeros(n, bool)
        extras: dict[str, np.ndarray] = {}
        Rbox = np.stack([np.stack([base["_rxx"], base["_rxy"]], -1), np.stack([base["_rxy"], base["_ryy"]], -1)], -2)
        all_boxes = base.loc[base["_given"].to_numpy(bool), ["track", "frame", *cfg.box_cols]].reset_index(drop=True)
        b_xy = base[["m1_x", "m1_y"]].to_numpy(dtype=float)
        fin = np.isfinite(b_xy).all(axis=1)
        origin = self.local_origin if self.local_origin is not None else (
            np.floor(b_xy[fin].min(axis=0)) if fin.any() else np.zeros(2))  # fmt: skip
        ctx = LocateContext(self.ground, cfg, tables, self.camera, all_boxes, self.workers, self.cache_dir,
                            np.asarray(origin, dtype=float))  # fmt: skip
        counts: dict = {}
        for c, chain in chains.items():
            of_c = (cls == c) & incl
            for name in chain:
                L = REGISTRY[name]
                todo = of_c & ~done & ~stopped
                if not todo.any():
                    break
                if L.classes is not None and c not in L.classes:
                    for i in np.flatnonzero(todo):
                        fails[i].append(f"{name}:class_ineligible")
                    if stop_on_ineligible:
                        stopped |= todo
                    continue
                if any(t not in tables or tables[t] is None for t in L.needs):
                    for i in np.flatnonzero(todo):
                        fails[i].append(f"{name}:not_supplied")
                    continue
                if L.scope == "row":
                    run = np.flatnonzero(todo & (kind == "box"))
                else:
                    want = np.unique(seg[todo & (seg >= 0)])
                    run = np.flatnonzero(of_c & np.isin(seg, want) & (seg >= 0))
                if not len(run):
                    continue
                rows = base.iloc[run]
                res: LocatorResult = L.fn(rows, ctx)
                _check_result(res, len(run), name)
                for k, v in res.extras.items():
                    if k not in extras:
                        extras[k] = np.full(n, np.nan) if np.asarray(v).dtype.kind in "fiub" else np.full(n, None, object)
                    extras[k][run] = v
                if res.seg_len_ft is not None:
                    seg_l[run], seg_w[run] = res.seg_len_ft, res.seg_wid_ft
                    seg_src[run] = res.seg_dims_source
                for j, i in enumerate(run):
                    if not todo[i]:
                        continue
                    ok = bool(res.ok[j]) and np.isfinite([res.x[j], res.y[j]]).all()
                    if ok and res.support[j] == "native":
                        if kind[i] != "box":
                            fails[i].append(f"{name}:projection")
                            continue
                        cov[i] = Rbox[i]
                    elif ok and res.support[j] == "kinematic":
                        own = None if res.cov is None else res.cov[j]
                        if not admit or seg[i] < 0 or own is None or not np.isfinite(own).all():
                            fails[i].append(f"{name}:no_observation")
                            continue
                        cov[i] = own
                    else:
                        fails[i].append(f"{name}:{res.fail_reason[j] or 'failed'}")
                        continue
                    x[i], y[i] = res.x[j], res.y[j]
                    source[i], support[i] = name, res.support[j]
                    orient[i], okind[i] = res.orient_deg[j], res.orient_kind[j]
                    lft[i], wft[i] = res.len_ft[j], res.wid_ft[j]
                    done[i] = True
                    counts[name] = counts.get(name, 0) + 1
        out = base.copy()
        out["fp_x"], out["fp_y"] = x, y
        out["fp_cov_xx"], out["fp_cov_xy"], out["fp_cov_yy"] = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
        out["fp_sigma"] = np.sqrt((cov[:, 0, 0] + cov[:, 1, 1]) / 2.0)
        out["fp_source"], out["fp_support"] = source, support
        out["fp_fail"] = [";".join(f) for f in fails]
        out["fp_orient_deg"], out["fp_orient_kind"] = orient, okind
        out["fp_len_ft"], out["fp_wid_ft"] = lft, wft
        out["fp_seg_len_ft"], out["fp_seg_wid_ft"], out["fp_seg_dims_source"] = seg_l, seg_w, seg_src
        self.info["sources"] = {**counts, "none": int((source == "none").sum())}
        self.info.update(ctx.info)
        given = base.attrs["given"]
        cols = [*given, *[c for c in ("interp",) if c not in given], *LOCATE_COLUMNS, *sorted(extras)]
        for k, v in extras.items():
            out[k] = v
        out = out[[c for c in dict.fromkeys(cols)]].copy()
        out.attrs["point_only"] = sorted(n for n, L in REGISTRY.items() if L.point_only)
        return out.reset_index(drop=True)

    def _finish(self, loc: pd.DataFrame) -> pd.DataFrame:
        """smooth (segmented) -> speed -> polygons, every row kept."""
        sm = self.smoother.smooth(loc, segmented=True)
        sp = self.speed.calculate(sm)
        sp.attrs["point_only"] = loc.attrs.get("point_only", [])
        return _polygons(sp, self.config, sp.attrs.get("speed_unit"))


# helpers ------------------------------------------------------------------------------------------------------------


def _cls_track(df: pd.DataFrame, cls_col: str) -> pd.Series:
    """Majority class of each track's rows (ties: the smallest class), mapped back to the rows."""
    modal = df.groupby("track")[cls_col].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    return df["track"].map(modal)


def _segments(table: pd.DataFrame) -> list[np.ndarray]:
    """Positional rows of each (track, seg >= 0), in frame order (the table is sorted by track, frame)."""
    seg = table["seg"].to_numpy()
    pos = np.flatnonzero(seg >= 0)
    if not len(pos):
        return []
    g = table.iloc[pos].groupby(["track", "seg"], sort=False).indices
    return [pos[v] for v in g.values()]


def _check_result(res: LocatorResult, n: int, name: str) -> None:
    for f in ("ok", "fail_reason", "x", "y", "orient_deg", "orient_kind", "len_ft", "wid_ft", "support"):
        if len(getattr(res, f)) != n:
            raise ValueError(f"locator {name} returned {f} of length {len(getattr(res, f))} for {n} rows")


def _polygons(table: pd.DataFrame, cfg: FootprintConfig, speed_unit: str | None) -> pd.DataFrame:
    """Design section 10."""
    need = ["track", "frame", "seg", "cls_track", *SMOOTH_XY, "vx", "vy", "fp_source", "fp_support", "fp_orient_deg",
            "fp_orient_kind", "fp_len_ft", "fp_wid_ft", "fp_seg_len_ft", "fp_seg_wid_ft", "fp_seg_dims_source"]  # fmt: skip
    missing = [c for c in need if c not in table.columns]
    if missing:
        raise ValueError(f"polygons() needs the locate() columns, smoothed positions and vx, vy; missing {missing}")
    unit_attr = table.attrs.get("speed_unit")
    if speed_unit is not None and unit_attr is not None and speed_unit != unit_attr:
        raise ValueError(f"speed_unit {speed_unit!r} conflicts with the table's {unit_attr!r}")
    unit = speed_unit or unit_attr
    if unit not in SPEED_UNITS:
        raise ValueError(f"speed unit unknown ({unit!r}): pass speed_unit='mph' or 'm/s' (pandas may drop attrs)")
    t = table.sort_values(["track", "frame"], kind="stable").reset_index(drop=True)
    n = len(t)
    thr = cfg.heading_min_mph * SPEED_UNITS[unit]
    vel = t[["vx", "vy"]].to_numpy(dtype=float)
    pos = t[list(SMOOTH_XY)].to_numpy(dtype=float)
    frames = t["frame"].to_numpy()
    track = t["track"].to_numpy()
    src = t["fp_source"].to_numpy(dtype=object)
    groups = _segments(t)

    heading, reliable = np.full(n, np.nan), np.zeros(n, bool)
    for rows in groups:
        heading[rows], reliable[rows] = O.motion_heading(vel[rows], thr)
    # step 4.3: the motion heading of the same track's nearest other segment in time
    track_heading = np.full(n, np.nan)
    fin = np.isfinite(heading)
    for tid, idx in t.groupby("track", sort=False).indices.items():
        idx = np.asarray(idx)
        good = idx[fin[idx]]
        if len(good):
            track_heading[idx] = heading[good][O.nearest_index(frames[idx], frames[good])]

    o_deg = t["fp_orient_deg"].to_numpy(dtype=float)
    o_kind = t["fp_orient_kind"].to_numpy(dtype=object)
    has_nat = (src != "none") & np.isin(o_kind, ["axis", "directed"]) & np.isfinite(o_deg)
    directed = has_nat & (o_kind == "directed")
    orient = np.full(n, np.nan)
    o_src = np.full(n, "none", dtype=object)
    f_src = np.full(n, "none", dtype=object)
    dt = 1.0 / cfg.fps
    for rows in groups:
        f = frames[rows]
        ax = O.smooth_axis(o_deg[rows], has_nat[rows], cfg.orient_q, cfg.orient_sigma, dt)
        ax = np.where(O.near(f, has_nat[rows], cfg.orient_reach_frames), ax, np.nan)
        h = heading[rows]
        d_rows = rows[directed[rows]]
        nat_dir = o_deg[d_rows][O.nearest_index(f, frames[d_rows])] if len(d_rows) else np.full(len(rows), np.nan)
        th = track_heading[rows]
        ref = np.where(np.isfinite(h), h, np.where(np.isfinite(nat_dir), nat_dir, th))
        rs = np.where(np.isfinite(h), "motion", np.where(np.isfinite(nat_dir), "native",
                                                          np.where(np.isfinite(th), "track", "none")))  # fmt: skip
        has_ax = np.isfinite(ax)
        directed_ax = np.where(np.isfinite(ref), direct_toward(np.where(has_ax, ax, 0.0), np.where(np.isfinite(ref), ref, 0.0)), ax)
        orient[rows] = np.where(has_ax, directed_ax, np.where(np.isfinite(h), h, th))
        o_src[rows] = np.where(has_ax, "native", np.where(np.isfinite(h) | np.isfinite(th), "motion", "none"))
        f_src[rows] = np.where(has_ax, rs, np.where(np.isfinite(h), "motion", np.where(np.isfinite(th), "track", "none")))
    orient = np.where(np.isfinite(orient), orient % 360.0, np.nan)

    # size (design 10.3)
    cls = t["cls_track"].to_numpy()
    nom = np.array([cfg.size_of(c) for c in cls], dtype=float).reshape(-1, 2)
    L, W = nom[:, 0].copy(), nom[:, 1].copy()
    dims = np.full(n, "nominal", dtype=object)
    lf, wf = t["fp_len_ft"].to_numpy(dtype=float), t["fp_wid_ft"].to_numpy(dtype=float)
    m6 = (src == "M6") & np.isfinite(lf) & np.isfinite(wf)
    if m6.any():
        med = pd.DataFrame({"track": track[m6], "L": lf[m6], "W": wf[m6]}).groupby("track").agg(["median", "size"])
        mL = pd.Series(track).map(med[("L", "median")]).to_numpy(dtype=float)
        mW = pd.Series(track).map(med[("W", "median")]).to_numpy(dtype=float)
        cnt = pd.Series(track).map(med[("L", "size")]).fillna(0).to_numpy()
        use = np.isin(cls, cfg.m6_measured_size_classes) & (cnt >= cfg.measured_size_min_rows) & np.isfinite(mL)
        L, W = np.where(use, mL, L), np.where(use, mW, W)
        dims = np.where(use, "m6_measured", dims).astype(object)
    sl, sw = t["fp_seg_len_ft"].to_numpy(dtype=float), t["fp_seg_wid_ft"].to_numpy(dtype=float)
    seg_ok = np.isfinite(sl) & np.isfinite(sw)
    L, W = np.where(seg_ok, sl, L), np.where(seg_ok, sw, W)
    dims = np.where(seg_ok, t["fp_seg_dims_source"].to_numpy(dtype=object), dims).astype(object)

    # point-only segments (every supplied row from a point-only locator) get no polygon
    point_only = set(table.attrs.get("point_only", ["M1"]))
    po = np.zeros(n, bool)
    for rows in groups:
        s = src[rows][src[rows] != "none"]
        if len(s) and all(v in point_only for v in s):
            po[rows] = True
    po |= np.isin(src, list(point_only))
    ok = np.isfinite(pos).all(axis=1) & np.isfinite(orient) & ~po
    corners = np.full((n, 8), np.nan)
    if ok.any():
        corners[ok] = rect_corners(pos[ok], orient[ok], L[ok] * FT, W[ok] * FT).reshape(-1, 8)
    dims = np.where(ok, dims, "none").astype(object)
    seg = t["seg"].to_numpy()
    front_known = (f_src != "none") & np.isfinite(orient)

    out = t.copy()
    out["motion_heading_deg"], out["heading_reliable"] = heading, reliable
    out["orient_deg"], out["orient_source"] = orient, o_src
    out["front_source"], out["front_known"] = f_src, front_known
    out["length_ft"], out["width_ft"] = np.where(ok, L, np.nan), np.where(ok, W, np.nan)
    out["dims_source"] = dims
    for j, c in enumerate(CORNERS):
        out[c] = corners[:, j]
    out["fp_low_confidence"] = _low_confidence(t, cfg, groups)
    out.loc[seg < 0, ["orient_source", "front_source"]] = "none"
    return out


def _low_confidence(t: pd.DataFrame, cfg: FootprintConfig, groups) -> np.ndarray:
    """Rows of a failed M7 solve that fell back, and admitted kinematic rows far from a native row (design 10.4)."""
    n = len(t)
    out = np.zeros(n, bool)
    if "m7_solve_ok" in t.columns:
        attempted = t["m7_solve_ok"].notna().to_numpy()
        failed = attempted & ~t["m7_solve_ok"].fillna(False).astype(bool).to_numpy()
        out |= failed & (t["fp_source"].to_numpy(dtype=object) != "M7")
    sup = t["fp_support"].to_numpy(dtype=object)
    src = t["fp_source"].to_numpy(dtype=object)
    frames = t["frame"].to_numpy()
    reach = cfg.lowconf_s * cfg.fps
    for rows in groups:
        kin = rows[(sup[rows] == "kinematic")]
        if not len(kin):
            continue
        nat = rows[(sup[rows] == "native") & (src[rows] == src[kin[0]])]
        if not len(nat):
            out[kin] = True
            continue
        out[kin] = O.nearest_distance(frames[kin], frames[nat]) > reach
    return out
