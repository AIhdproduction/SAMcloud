"""The classified outdoor point cloud of a drone project for the AI assistant (read tools and evidence).

Source: the working cloud of the environment chain, in the local frame of the Scene: the refined classes of
``cache/semantics.pkl`` (``{"pointset": PointSet}``) when present, else ``cache/load.pkl`` (``(PointSet, LocalFrame)``),
the same cloud the pipeline steps use.  It is loaded once per project and file version (the real cloud has millions of
points; loading takes a few seconds) and never written.

All methods take and return LOCAL metres (the domain converts LV95 for the model); results are small JSON-safe dicts.
Class codes are LAS codes (:mod:`samcloud.drone.contracts`).
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from samcloud.assistant.pointcloud import CloudUnavailable

LOGGER = logging.getLogger(__name__)

CLASS_NAMES: dict[int, str] = {
    2: "ground", 3: "low vegetation", 4: "medium vegetation", 5: "tree canopy", 6: "building", 11: "road",
    71: "sealed terrain", 72: "garden wall", 73: "sidewalk", 74: "curb", 75: "manhole cover", 77: "fence",
    82: "road marking", 83: "path", 94: "shrub", 95: "tree trunk", 103: "door", 104: "window",
    144: "inclined roof", 145: "flat roof",
}
TREE_CLASSES = (5, 95)
CANOPY_CLASS = 5
TRUNK_CLASS = 95
MANHOLE_CLASS = 75
HISTOGRAM_BIN_M = 0.5
MAX_CLASSES = 8
MAX_HISTOGRAM_BINS = 40
MIN_TREE_POINTS = 80
MIN_MANHOLE_POINTS = 12
DEFAULT_TREE_RADIUS_M = 3.0
DEFAULT_MANHOLE_RADIUS_M = 1.0
_CACHE_SIZE = 2


def _r(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


def class_name(code: int) -> str:
    return CLASS_NAMES.get(int(code), f"class {int(code)}")


class Region:
    """A local box ``(x0, x1, y0, y1)`` with optional ``z`` range or a circle ``(cx, cy, r)``."""

    def __init__(self, *, box: tuple[float, float, float, float] | None = None, circle: tuple[float, float, float] | None = None,
                 z: tuple[float, float] | None = None) -> None:
        if box is None and circle is None:
            raise ValueError("a region needs a box or a circle")
        self.box = box
        self.circle = circle
        self.z = z

    @property
    def center(self) -> tuple[float, float]:
        if self.circle is not None:
            return float(self.circle[0]), float(self.circle[1])
        x0, x1, y0, y1 = self.box  # type: ignore[misc]
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        if self.box is not None:
            return self.box
        cx, cy, r = self.circle  # type: ignore[misc]
        return cx - r, cx + r, cy - r, cy + r

    def mask(self, xyz: np.ndarray) -> np.ndarray:
        x0, x1, y0, y1 = self.bounds
        inside = (xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1)
        if self.circle is not None:
            cx, cy, r = self.circle
            inside &= (xyz[:, 0] - cx) ** 2 + (xyz[:, 1] - cy) ** 2 <= r * r
        if self.z is not None:
            inside &= (xyz[:, 2] >= self.z[0]) & (xyz[:, 2] <= self.z[1])
        return inside


class OutdoorCloud:
    """Classified working cloud (local metres, LAS codes) with the measuring helpers of the outdoor tools."""

    _cache: "OrderedDict[tuple[Any, ...], OutdoorCloud]" = OrderedDict()
    _lock = threading.Lock()

    def __init__(self, xyz: np.ndarray, classes: np.ndarray) -> None:
        self.xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        self.classes = np.asarray(classes).astype(np.int64).reshape(-1)
        if len(self.xyz) != len(self.classes):
            raise ValueError("xyz and classes differ in length")
        order = np.argsort(self.xyz[:, 0], kind="stable")  # sorted by x: box queries slice instead of scanning
        self.xyz = self.xyz[order]
        self.classes = self.classes[order]

    # --------------------------------------------------------------------------------------------------- loading
    @classmethod
    def from_pointset(cls, pointset: Any) -> "OutdoorCloud":
        return cls(np.asarray(pointset.xyz), np.asarray(pointset.classes))

    @classmethod
    def load(cls, project: Path | str) -> "OutdoorCloud":
        """The working cloud of a drone project (cached per file version); raises :class:`CloudUnavailable`."""
        from samcloud.assistant.domains.outdoor_scene import cache_dir

        cache = cache_dir(project)
        source = None
        for name in ("semantics.pkl", "load.pkl"):
            if (cache / name).is_file():
                source = cache / name
                break
        if source is None:
            raise CloudUnavailable("no environment working cloud (products/environment/cache/load.pkl)")
        stat = source.stat()
        key = (str(source.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        with cls._lock:
            hit = cls._cache.get(key)
            if hit is not None:
                cls._cache.move_to_end(key)
                return hit
        import pickle

        with source.open("rb") as handle:
            data = pickle.load(handle)
        pointset = data.get("pointset") if isinstance(data, dict) else (data[0] if isinstance(data, tuple) else data)
        if pointset is None and isinstance(data, dict) and (cache / "load.pkl").is_file() and source.name != "load.pkl":
            with (cache / "load.pkl").open("rb") as handle:
                pointset = pickle.load(handle)[0]
        if pointset is None or not hasattr(pointset, "xyz"):
            raise CloudUnavailable("the environment working cloud holds no points")
        cloud = cls.from_pointset(pointset)
        with cls._lock:
            cls._cache[key] = cloud
            while len(cls._cache) > _CACHE_SIZE:
                cls._cache.popitem(last=False)
        return cloud

    # ---------------------------------------------------------------------------------------------------- access
    def select(self, region: Region, codes: tuple[int, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(xyz, classes) of the points in ``region`` (optionally only the LAS ``codes``)."""
        x0, x1, _y0, _y1 = region.bounds
        lo = int(np.searchsorted(self.xyz[:, 0], x0, side="left"))
        hi = int(np.searchsorted(self.xyz[:, 0], x1, side="right"))
        xyz, classes = self.xyz[lo:hi], self.classes[lo:hi]
        mask = region.mask(xyz)
        if codes is not None:
            mask &= np.isin(classes, np.asarray(codes, dtype=np.int64))
        return xyz[mask], classes[mask]

    def count(self, region: Region, codes: tuple[int, ...]) -> int:
        return int(len(self.select(region, codes)[0]))

    # ------------------------------------------------------------------------------------------------ read tools
    def points_in_box(self, region: Region, surface: Callable[[np.ndarray], np.ndarray] | None = None) -> dict[str, Any]:
        """Point count, classes (count, z range, bounding box), z histogram (0.5 m bins, absolute local z)."""
        xyz, classes = self.select(region)
        result: dict[str, Any] = {"points": int(len(xyz))}
        if not len(xyz):
            result["classes"] = []
            return result
        rows = []
        for code in np.unique(classes):
            sel = xyz[classes == code]
            rows.append({"class": class_name(int(code)), "code": int(code), "points": int(len(sel)),
                         "z_min": _r(sel[:, 2].min()), "z_max": _r(sel[:, 2].max()),
                         "bbox": [[_r(sel[:, 0].min()), _r(sel[:, 1].min())], [_r(sel[:, 0].max()), _r(sel[:, 1].max())]]})
        rows.sort(key=lambda row: -row["points"])
        result["classes"] = rows[:MAX_CLASSES]
        if len(rows) > MAX_CLASSES:
            result["more_classes"] = len(rows) - MAX_CLASSES
        z0 = math.floor(float(xyz[:, 2].min()) / HISTOGRAM_BIN_M) * HISTOGRAM_BIN_M
        bins = min(MAX_HISTOGRAM_BINS, max(1, int(math.ceil((float(xyz[:, 2].max()) - z0) / HISTOGRAM_BIN_M + 1e-9))))
        width = max(HISTOGRAM_BIN_M, (float(xyz[:, 2].max()) - z0) / bins + 1e-9)
        counts = np.bincount(np.clip(((xyz[:, 2] - z0) / width).astype(int), 0, bins - 1), minlength=bins)
        result["z_histogram"] = {"z0": _r(z0), "bin_m": _r(width), "counts": [int(v) for v in counts]}
        if surface is not None:
            ground = np.asarray(surface(np.asarray([region.center])), float)
            if np.isfinite(ground).all():
                result["terrain_z"] = _r(ground[0])
        return result

    def profile(self, xy: np.ndarray, step_m: float, half_width_m: float, surface: Callable[[np.ndarray], np.ndarray] | None) -> list[dict[str, Any]]:
        """Terrain height and the highest / lowest cloud point (with its class) per station along a polyline."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        seg = np.hypot(*np.diff(xy, axis=0).T)
        total = float(seg.sum())
        stations = np.unique(np.r_[np.arange(0.0, total, step_m), total])
        cum = np.r_[0.0, np.cumsum(seg)]
        px, py = np.interp(stations, cum, xy[:, 0]), np.interp(stations, cum, xy[:, 1])
        ground = surface(np.column_stack([px, py])) if surface is not None else np.full(len(px), np.nan)
        rows = []
        for s, x, y, g in zip(stations, px, py, ground):
            pts, classes = self.select(Region(circle=(float(x), float(y), half_width_m)))
            row: dict[str, Any] = {"s": _r(s), "x": _r(x), "y": _r(y), "terrain_z": _r(g) if math.isfinite(float(g)) else None,
                                   "points": int(len(pts))}
            if len(pts):
                top = int(np.argmax(pts[:, 2]))
                row["top_z"], row["top_class"] = _r(pts[top, 2]), class_name(int(classes[top]))
                row["low_z"] = _r(pts[:, 2].min())
            rows.append(row)
        return rows


# ------------------------------------------------------------------------------------------------------- measuring

def measure_tree(cloud: OutdoorCloud, region: Region, surface: Callable[[np.ndarray], np.ndarray] | None,
                 config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Measure a tree from its canopy (5) and trunk (95) points in ``region`` (local metres).

    Crown shape / diameter / crown base from :func:`samcloud.environment.vegetation.fit_crown_shape`, trunk diameter
    from the circle fit of :func:`samcloud.environment.vegetation._trunk_diameter` (None when its gates fail), base from
    the terrain surface at the trunk.  ``points`` is the evidence count (canopy + trunk).
    """
    from samcloud.environment.vegetation import _trunk_diameter, fit_crown_shape

    veg_cfg = (config or {}).get("vegetation", {}) if isinstance(config, dict) else {}
    canopy, _ = cloud.select(region, (CANOPY_CLASS,))
    trunk, _ = cloud.select(region, (TRUNK_CLASS,))
    result: dict[str, Any] = {"points": int(len(canopy) + len(trunk)), "canopy_points": int(len(canopy)), "trunk_points": int(len(trunk))}
    if len(canopy) + len(trunk) == 0:
        return result
    crown = fit_crown_shape(canopy, veg_cfg) if len(canopy) else None
    if len(trunk) >= 3:
        trunk_xy = (float(np.median(trunk[:, 0])), float(np.median(trunk[:, 1])))
    elif crown is not None and crown.centre_xy is not None:
        trunk_xy = (float(crown.centre_xy[0]), float(crown.centre_xy[1]))
    else:
        pts = canopy if len(canopy) else trunk
        trunk_xy = (float(np.median(pts[:, 0])), float(np.median(pts[:, 1])))
    ground = float(surface(np.asarray([trunk_xy]))[0]) if surface is not None else float("nan")
    lowest = float(np.percentile(np.vstack([p for p in (canopy, trunk) if len(p)])[:, 2], 1))
    base_z = ground if math.isfinite(ground) else lowest
    every = np.vstack([p for p in (canopy, trunk) if len(p)])
    top_z = float(np.percentile(every[:, 2], 99.5)) if len(every) >= 20 else float(every[:, 2].max())
    result.update({"trunk_xy": [_r(trunk_xy[0]), _r(trunk_xy[1])], "base_z": _r(base_z), "top_z": _r(top_z),
                   "height": _r(top_z - base_z), "terrain_found": bool(math.isfinite(ground))})
    diameter = None
    if crown is not None:
        result["crown_shape"] = crown.shape
        result["crown_shape_status"] = crown.status
        diameter = crown.diameter_m
        if crown.base_z is not None and base_z < float(crown.base_z) < top_z:
            result["crown_base_z"] = _r(crown.base_z)
        centre = crown.centre_xy
        if centre is not None:
            result["crown_offset"] = [_r(centre[0] - trunk_xy[0]), _r(centre[1] - trunk_xy[1])]
    if diameter is None and len(canopy) >= 10:
        centre = np.median(canopy[:, :2], axis=0)
        diameter = 2.0 * float(np.percentile(np.hypot(*(canopy[:, :2] - centre).T), 95))
    if diameter is not None:
        result["crown_diameter"] = _r(diameter)
    trunk_d, rms = (None, None)
    if len(trunk):
        try:
            trunk_d, rms = _trunk_diameter(trunk, trunk_xy, base_z, veg_cfg.get("trunk"))
        except Exception as error:  # noqa: BLE001 - an unfittable trunk stays unmeasured
            LOGGER.info("assistant: trunk fit failed (%s)", type(error).__name__)
    if trunk_d is not None:
        result["trunk_diameter"] = _r(trunk_d)
        result["trunk_rms"] = None if rms is None else _r(rms, 4)
    return result


def measure_manhole(cloud: OutdoorCloud, region: Region, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Measure a manhole cover from its points (75) in ``region``: the detection of
    :func:`samcloud.outdoor_model.civil.manhole_detections` (PCA extents, roundness, standard size), else a plain circle
    from the extents.  ``points`` is the evidence count."""
    xyz, classes = cloud.select(region, (MANHOLE_CLASS,))
    result: dict[str, Any] = {"points": int(len(xyz))}
    if len(xyz) < 3:
        return result
    detection = None
    try:
        from samcloud.drone.contracts import PointSet
        from samcloud.outdoor_model.civil import manhole_detections

        found = manhole_detections(PointSet(xyz=xyz, classes=classes.astype(np.uint16)), min_points=MIN_MANHOLE_POINTS,
                                   config=config)
        ok = [d for d in found if d.status == "ok"]
        if ok:
            cx, cy = region.center
            detection = min(ok, key=lambda d: math.hypot(float(d.center[0]) - cx, float(d.center[1]) - cy))
    except Exception as error:  # noqa: BLE001 - the plain circle below still measures it
        LOGGER.info("assistant: manhole detection failed (%s)", type(error).__name__)
    if detection is not None:
        result.update({"center": [_r(detection.center[0]), _r(detection.center[1]), _r(detection.center[2])],
                       "diameter": _r(detection.diameter), "diameter_measured": _r(detection.diameter_measured),
                       "snapped": bool(detection.snapped), "standard": str(detection.standard or ""),
                       "roundness": detection.evidence.notes.get("roundness"), "method": "detection"})
        return result
    xy = xyz[:, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    centre = (lo + hi) / 2.0
    diameter = float(np.mean(hi - lo))
    result.update({"center": [_r(centre[0]), _r(centre[1]), _r(np.median(xyz[:, 2]))], "diameter": _r(diameter),
                   "diameter_measured": _r(diameter), "snapped": False, "standard": "", "method": "extent"})
    return result


__all__ = ["CANOPY_CLASS", "CLASS_NAMES", "DEFAULT_MANHOLE_RADIUS_M", "DEFAULT_TREE_RADIUS_M", "MANHOLE_CLASS",
           "MIN_MANHOLE_POINTS", "MIN_TREE_POINTS", "OutdoorCloud", "Region", "TRUNK_CLASS", "class_name", "measure_manhole",
           "measure_tree"]
