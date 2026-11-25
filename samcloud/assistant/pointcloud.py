"""Measuring in the classified project point cloud for the AI assistant (read tools and evidence).

Coordinates of the API ("plan metres"): x / y are the local plan coordinates of ``plan_model.json`` divided by
``units_per_metre`` (for metric plans, ``units_per_metre == 1``, they are identical); z is the height above the
storey floor in metres.  Boxes are ``{"x": [x0, x1], "y": [y0, y1], "z": [z0, z1]}`` (``z`` optional).

Every public method returns a small JSON-safe dict with metres: never raw point arrays.
"""

from __future__ import annotations

import json
import logging
import math
import struct
import threading
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

# floor-plan slice: points within this distance of the requested height belong to the section
SECTION_HALF_BAND_M = 0.05
# z histogram bin of points_in_box
HISTOGRAM_BIN_M = 0.10
# fit_surface: plane inlier distance, minimum points of the requested class before all points are used
SURFACE_TOLERANCE_M = 0.02
SURFACE_MIN_CLASS_POINTS = 30
# a second wall face must be this parallel (degrees) and this far (metres) from the first one
WALL_FACE_MAX_ANGLE_DEG = 10.0
WALL_THICKNESS_RANGE_M = (0.04, 0.80)
# result size limits (JSON for a language model)
MAX_INSTANCES = 10
MAX_CLASSES = 6
MAX_OUTLINES = 40
_CACHE_SIZE = 2

# SAM3 indoor class names (config/classes.json order); project specific names do not change the indices.
DEFAULT_CLASS_NAMES: dict[int, str] = {
    0: "unknown", 1: "loose items", 2: "wall", 3: "ceiling", 4: "floor", 5: "door", 6: "window", 7: "stairs",
    8: "railing", 9: "column", 10: "beam", 11: "steel structure", 12: "pipe", 13: "cable", 14: "electrical",
    15: "hvac", 16: "ceiling equipment", 17: "furniture", 18: "appliances", 19: "sanitary fixtures",
}
FURNITURE_CLASSES = (17, 18, 19)
_SURFACE_CLASSES = {"wall": (2,), "table": (17, 18), "floor": (4,), "ceiling": (3,)}


class CloudUnavailable(FileNotFoundError):
    """The project has no usable classified point cloud (``products/classified/classified.ply``)."""


def _r(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


def parse_box(box: dict[str, Any] | None) -> tuple[float, float, float, float, float | None, float | None]:
    """(x0, x1, y0, y1, z0, z1) in plan metres / floor-relative metres; raises ValueError for a malformed box."""
    if not isinstance(box, dict):
        raise ValueError("box must be an object with 'x' and 'y' ranges")
    try:
        x0, x1 = sorted(float(v) for v in box["x"])
        y0, y1 = sorted(float(v) for v in box["y"])
        z = box.get("z")
        z0, z1 = (None, None) if z is None else sorted(float(v) for v in z)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"malformed box {box!r}: {error}") from error
    values = [x0, x1, y0, y1] + ([] if z0 is None else [z0, z1])
    if not all(math.isfinite(v) for v in values):
        raise ValueError("box values must be finite")
    if x1 - x0 <= 0.0 or y1 - y0 <= 0.0:
        raise ValueError("box must have a positive size")
    return x0, x1, y0, y1, z0, z1


class ProjectCloud:
    """The classified cloud of one project level in plan-local coordinates (class + instance arrays kept)."""

    _cache: "OrderedDict[tuple[Any, ...], ProjectCloud]" = OrderedDict()
    _lock = threading.Lock()

    def __init__(
        self, xyz: np.ndarray, cls: np.ndarray, *, units_per_metre: float, floor_z: float, ceiling_z: float | None,
        instance: np.ndarray | None = None, fine: np.ndarray | None = None, fine_names: list[str] | None = None,
        instance_labels: dict[int, str] | None = None, class_names: dict[int, str] | None = None,
    ) -> None:
        self.xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        count = len(self.xyz)
        self.cls = np.asarray(cls, dtype=np.uint8).reshape(-1)
        self.instance = np.zeros(count, np.int64) if instance is None else np.asarray(instance, dtype=np.int64).reshape(-1)
        self.fine = np.zeros(count, np.int32) if fine is None else np.asarray(fine, dtype=np.int32).reshape(-1)
        if not (len(self.cls) == len(self.instance) == len(self.fine) == count):
            raise ValueError("point arrays differ in length")
        self.units_per_metre = max(float(units_per_metre), 1e-9)
        self.floor_z = float(floor_z)
        self.ceiling_z = None if ceiling_z is None else float(ceiling_z)
        self.fine_names = list(fine_names or [])
        self.instance_labels = dict(instance_labels or {})
        self.class_names = dict(class_names or DEFAULT_CLASS_NAMES)
        upm = self.units_per_metre
        self._xm = self.xyz[:, 0] / upm
        self._ym = self.xyz[:, 1] / upm
        self._hm = (self.xyz[:, 2] - self.floor_z) / upm

    # ----------------------------------------------------------------------------------------------------- loading
    @classmethod
    def load(cls, project: Path | str, level_id: str, plan: Any | None = None) -> "ProjectCloud":
        """Cached per (project, level, PLY mtime / size, plan frame); raises :class:`CloudUnavailable` without a cloud."""
        from samcloud.assistant.paths import classified_ply_path, plan_model_path

        ply = classified_ply_path(project)
        if not ply.is_file():
            raise CloudUnavailable(f"no classified point cloud: {ply.name} missing in products/classified")
        if plan is None:
            from samcloud.bim.plan_model import load_plan_model

            source = plan_model_path(project, level_id)
            if not source.is_file():
                raise CloudUnavailable("no plan_model.json for this level: the plan frame is unknown")
            plan = load_plan_model(source)
        stat = ply.stat()
        key = (str(Path(project).resolve()), str(level_id), stat.st_mtime_ns, stat.st_size,
               json.dumps(plan.frame, sort_keys=True, default=str), float(plan.floor_z), plan.ceiling_z, float(plan.units_per_metre))
        with cls._lock:
            cached = cls._cache.get(key)
            if cached is not None:
                cls._cache.move_to_end(key)
                return cached
        loaded = cls._read(ply, plan)
        with cls._lock:
            cls._cache[key] = loaded
            while len(cls._cache) > _CACHE_SIZE:
                cls._cache.popitem(last=False)
        return loaded

    @classmethod
    def clear_cache(cls) -> None:
        with cls._lock:
            cls._cache.clear()

    @classmethod
    def _read(cls, ply: Path, plan: Any) -> "ProjectCloud":
        from samcloud.dxf.las_source import Frame
        from samcloud.engine.ply_io import read_vertex_array

        try:
            vertices = read_vertex_array(str(ply))
        except Exception as error:  # noqa: BLE001 - any parse problem means: no usable cloud
            raise CloudUnavailable(f"classified point cloud could not be read: {error}") from error
        names = vertices.dtype.names or ()
        if "scalar_class" not in names or not {"x", "y", "z"} <= set(names):
            raise CloudUnavailable("classified point cloud has no scalar_class field")
        frame_data = plan.frame or {}
        frame = Frame(
            origin=np.asarray(frame_data.get("origin", [0.0, 0.0, 0.0]), dtype=float),
            scale_factor=float(frame_data.get("scale_factor", 1.0)),
            rotation=np.asarray(frame_data.get("rotation", np.eye(3).tolist()), dtype=float),
        )
        xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float64)
        local = np.asarray(frame.to_local(xyz), dtype=np.float64) if len(xyz) else xyz
        instance = np.asarray(vertices["instance_id"], dtype=np.int64) if "instance_id" in names else None
        fine = np.asarray(vertices["fine_term"], dtype=np.int32) if "fine_term" in names else None
        fine_names: list[str] = []
        labels_file = ply.with_name("fine_labels.json")
        if labels_file.is_file():
            try:
                fine_names = [str(n).strip().lower() for n in json.loads(labels_file.read_text(encoding="utf-8")).get("labels", [])]
            except (OSError, ValueError):
                fine_names = []
        instance_labels: dict[int, str] = {}
        instances_file = ply.with_name("instances.json")
        if instances_file.is_file():
            try:
                for item in json.loads(instances_file.read_text(encoding="utf-8")).get("instances", []):
                    if "id" in item:
                        instance_labels[int(item["id"])] = str(item.get("label") or item.get("class_name") or "").strip()
            except (OSError, ValueError, TypeError):
                instance_labels = {}
        if fine is not None and fine_names:
            fine[(fine < 0) | (fine > len(fine_names))] = 0
        elif fine is not None:
            fine[:] = 0
        return cls(local, np.asarray(vertices["scalar_class"]), units_per_metre=float(plan.units_per_metre),
                   floor_z=float(plan.floor_z), ceiling_z=plan.ceiling_z, instance=instance, fine=fine,
                   fine_names=fine_names, instance_labels=instance_labels)

    # ------------------------------------------------------------------------------------------------- primitives
    def __len__(self) -> int:
        return int(len(self.xyz))

    def class_name(self, code: int) -> str:
        return self.class_names.get(int(code), f"class {int(code)}")

    def storey_height_m(self) -> float | None:
        if self.ceiling_z is None or self.ceiling_z <= self.floor_z:
            return None
        return (self.ceiling_z - self.floor_z) / self.units_per_metre

    def select(self, box: dict[str, Any] | None, *, min_height_m: float | None = None, classes: tuple[int, ...] | None = None) -> np.ndarray:
        """Indices of the points inside ``box`` (None = all), optionally above ``min_height_m`` and of ``classes``."""
        mask = np.ones(len(self.xyz), dtype=bool)
        if box is not None:
            x0, x1, y0, y1, z0, z1 = parse_box(box)
            mask &= (self._xm >= x0) & (self._xm <= x1) & (self._ym >= y0) & (self._ym <= y1)
            if z0 is not None:
                mask &= (self._hm >= z0) & (self._hm <= z1)
        if min_height_m is not None:
            mask &= self._hm >= float(min_height_m)
        if classes is not None:
            mask &= np.isin(self.cls, np.asarray(classes, dtype=self.cls.dtype))
        return np.flatnonzero(mask)

    def height_m(self, indices: np.ndarray) -> np.ndarray:
        return self._hm[indices]

    def xy_m(self, indices: np.ndarray) -> np.ndarray:
        return np.column_stack((self._xm[indices], self._ym[indices]))

    def xyz_m(self, indices: np.ndarray) -> np.ndarray:
        """Plan metres with floor-relative z."""
        return np.column_stack((self._xm[indices], self._ym[indices], self._hm[indices]))

    def class_counts(self, indices: np.ndarray) -> dict[str, int]:
        if not len(indices):
            return {}
        codes, counts = np.unique(self.cls[indices], return_counts=True)
        order = np.argsort(-counts, kind="stable")
        return {self.class_name(int(codes[i])): int(counts[i]) for i in order}

    def instance_label(self, identifier: int) -> str:
        return self.instance_labels.get(int(identifier), "")

    def scene_cloud(self, box: dict[str, Any] | None = None, classes: tuple[int, ...] = FURNITURE_CLASSES, margin_m: float = 0.0) -> Any:
        """``samcloud.bim.furniture_recon.cloud.SceneCloud`` (plan-local units) of the furniture-class points in ``box``."""
        from samcloud.bim.furniture_recon.cloud import SceneCloud

        grown = None
        if box is not None:
            x0, x1, y0, y1, z0, z1 = parse_box(box)
            grown = {"x": [x0 - margin_m, x1 + margin_m], "y": [y0 - margin_m, y1 + margin_m]}
            if z0 is not None:
                grown["z"] = [z0, z1]
        indices = self.select(grown, classes=classes)
        return SceneCloud(self.xyz[indices].copy(), self.fine[indices].copy(), self.instance[indices].copy(), list(self.fine_names))

    # ----------------------------------------------------------------------------------------------------- read tools
    def points_in_box(self, box: dict[str, Any]) -> dict[str, Any]:
        """Count, floor-relative z histogram (0.1 m bins), top SAM3 classes and instances of the points in ``box``."""
        indices = self.select(box)
        result: dict[str, Any] = {"count": int(len(indices)), "box": _box_json(box)}
        if not len(indices):
            result.update({"z_histogram": [], "classes": [], "instances": []})
            return result
        heights = self._hm[indices]
        bins = np.floor(heights / HISTOGRAM_BIN_M).astype(np.int64)
        values, counts = np.unique(bins, return_counts=True)
        result["z_histogram"] = [
            {"z0": _r(v * HISTOGRAM_BIN_M, 2), "z1": _r((v + 1) * HISTOGRAM_BIN_M, 2), "count": int(c)} for v, c in zip(values, counts)
        ]
        result["z_range_m"] = [_r(np.percentile(heights, 2)), _r(np.percentile(heights, 98))]
        codes, counts = np.unique(self.cls[indices], return_counts=True)
        order = np.argsort(-counts, kind="stable")[:MAX_CLASSES]
        result["classes"] = [{"id": int(codes[i]), "name": self.class_name(int(codes[i])), "count": int(counts[i])} for i in order]
        instances = self.instance[indices]
        present = instances > 0
        rows: list[dict[str, Any]] = []
        if present.any():
            ids, counts = np.unique(instances[present], return_counts=True)
            for i in np.argsort(-counts, kind="stable")[:MAX_INSTANCES]:
                member = indices[instances == ids[i]]
                xy, h = self.xy_m(member), self._hm[member]
                rows.append({
                    "id": int(ids[i]), "label": self.instance_label(int(ids[i])), "count": int(counts[i]),
                    "bbox": {"x": [_r(xy[:, 0].min()), _r(xy[:, 0].max())], "y": [_r(xy[:, 1].min()), _r(xy[:, 1].max())],
                             "z": [_r(h.min()), _r(h.max())]},
                })
        result["instances"] = rows
        return result

    def _section_cells(self, z_m: float, box: dict[str, Any] | None, cell_m: float) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        if not (math.isfinite(z_m) and math.isfinite(cell_m) and cell_m > 0.0):
            raise ValueError("section height and cell size must be finite, the cell positive")
        selection = {"z": [z_m - SECTION_HALF_BAND_M, z_m + SECTION_HALF_BAND_M]}
        if box is not None:
            x0, x1, y0, y1, _z0, _z1 = parse_box(box)
            selection.update({"x": [x0, x1], "y": [y0, y1]})
            indices = self.select(selection)
        else:
            indices = np.flatnonzero(np.abs(self._hm - z_m) <= SECTION_HALF_BAND_M)
        if not len(indices):
            return indices, np.zeros((0, 2), np.int64), (0, 0)
        cells = np.floor(self.xy_m(indices) / cell_m).astype(np.int64)
        return indices, cells, (int(cells[:, 0].min()), int(cells[:, 1].min()))

    def section(self, z_m: float, box: dict[str, Any] | None = None, cell_m: float = 0.05) -> dict[str, Any]:
        """Floor-plan slice at height ``z_m`` (floor-relative, +-0.05 m): occupied-cell outlines (plan metres) + counts."""
        from shapely.geometry import box as shapely_box
        from shapely.ops import unary_union

        indices, cells, _origin = self._section_cells(float(z_m), box, float(cell_m))
        result: dict[str, Any] = {"z_m": _r(z_m), "band_m": SECTION_HALF_BAND_M, "cell_m": _r(cell_m), "points": int(len(indices))}
        if not len(indices):
            result.update({"cells": 0, "outlines": [], "classes": []})
            return result
        unique = np.unique(cells, axis=0)
        result["cells"] = int(len(unique))
        result["classes"] = [{"name": name, "count": count} for name, count in list(self.class_counts(indices).items())[:MAX_CLASSES]]
        # row runs -> rectangles -> union (far fewer shapes than single cells)
        rectangles = []
        order = np.lexsort((unique[:, 0], unique[:, 1]))
        unique = unique[order]
        start = 0
        for index in range(1, len(unique) + 1):
            if index == len(unique) or unique[index, 1] != unique[start, 1] or unique[index, 0] != unique[index - 1, 0] + 1:
                cx0, cx1, cy = unique[start, 0], unique[index - 1, 0] + 1, unique[start, 1]
                rectangles.append(shapely_box(cx0 * cell_m, cy * cell_m, cx1 * cell_m, (cy + 1) * cell_m))
                start = index
        merged = unary_union(rectangles).simplify(cell_m * 0.5)
        polygons = [merged] if merged.geom_type == "Polygon" else list(getattr(merged, "geoms", []))
        polygons = sorted((p for p in polygons if not p.is_empty), key=lambda p: -p.area)
        result["outlines"] = [
            {"points": [[_r(x), _r(y)] for x, y in list(p.exterior.coords)[:-1]], "area_m2": _r(p.area)} for p in polygons[:MAX_OUTLINES]
        ]
        result["outlines_total"] = len(polygons)
        return result

    def section_png(self, z_m: float, box: dict[str, Any] | None = None, cell_m: float = 0.05, max_pixels: int = 1024) -> bytes:
        """The section as a grayscale PNG (occupied cells dark, north up); only used when images are allowed."""
        indices, cells, (cx0, cy0) = self._section_cells(float(z_m), box, float(cell_m))
        if box is not None:
            x0, x1, y0, y1, _z0, _z1 = parse_box(box)
            cx0, cy0 = int(math.floor(x0 / cell_m)), int(math.floor(y0 / cell_m))
            width, height = int(math.ceil(x1 / cell_m)) - cx0 + 1, int(math.ceil(y1 / cell_m)) - cy0 + 1
        elif len(cells):
            width, height = int(cells[:, 0].max()) - cx0 + 1, int(cells[:, 1].max()) - cy0 + 1
        else:
            width, height = 1, 1
        width, height = max(1, min(width, max_pixels)), max(1, min(height, max_pixels))
        image = np.full((height, width), 255, dtype=np.uint8)
        if len(cells):
            cols, rows = cells[:, 0] - cx0, cells[:, 1] - cy0
            keep = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
            image[height - 1 - rows[keep], cols[keep]] = 0
        return _png_gray(image)

    def fit_surface(self, box: dict[str, Any], kind: str) -> dict[str, Any]:
        """Plane fit in ``box`` for ``kind`` (wall | table | floor | ceiling): normal, height / offset, rms, inliers, extent.

        Walls also get a centreline and thickness estimate when both faces are visible.
        """
        from samcloud.bim.room_objects.geometry import fit_plane, upright_plane

        kind = str(kind or "").strip().lower()
        if kind not in _SURFACE_CLASSES:
            raise ValueError(f"unknown surface kind {kind!r} (wall, table, floor, ceiling)")
        indices = self.select(box, classes=_SURFACE_CLASSES[kind])
        class_filter = True
        if len(indices) < SURFACE_MIN_CLASS_POINTS:
            indices, class_filter = self.select(box), False
        result: dict[str, Any] = {"kind": kind, "class_filter": class_filter, "points": int(len(indices)), "ok": False}
        if len(indices) < 10:
            result["reason"] = "too few points"
            return result
        pts = self.xyz_m(indices)
        if kind == "wall":
            inlier = upright_plane(pts, SURFACE_TOLERANCE_M, 10.0)
            if inlier is None or int(np.count_nonzero(inlier)) < 10:
                result["reason"] = "no upright plane"
                return result
            face = _upright_face(pts[inlier], fit_plane)
            result.update({"ok": True, "normal": face["normal"], "offset_m": face["offset_m"], "rms_m": face["rms_m"],
                           "inliers": int(np.count_nonzero(inlier)), "extent": face["extent"]})
            rest = pts[~inlier]
            second = upright_plane(rest, SURFACE_TOLERANCE_M, 10.0) if len(rest) >= 20 else None
            if second is not None and int(np.count_nonzero(second)) >= 20:
                other = _upright_face(rest[second], fit_plane)
                n1, n2 = np.asarray(face["normal"][:2]), np.asarray(other["normal"][:2])
                cosine = abs(float(n1 @ n2))
                if cosine >= math.cos(math.radians(WALL_FACE_MAX_ANGLE_DEG)):
                    sign = 1.0 if float(n1 @ n2) >= 0 else -1.0
                    distance = abs(face["offset_m"] - sign * other["offset_m"])
                    if WALL_THICKNESS_RANGE_M[0] <= distance <= WALL_THICKNESS_RANGE_M[1]:
                        mid_offset = (face["offset_m"] + sign * other["offset_m"]) / 2.0
                        normal = n1 / max(float(np.linalg.norm(n1)), 1e-12)
                        direction = np.asarray((-normal[1], normal[0]))
                        s0 = min(face["extent"]["s"][0], other["extent"]["s"][0])
                        s1 = max(face["extent"]["s"][1], other["extent"]["s"][1])
                        base = normal * mid_offset
                        result["thickness_m"] = _r(distance)
                        result["centreline"] = [[_r(v) for v in base + direction * s0], [_r(v) for v in base + direction * s1]]
                        result["second_face"] = {"offset_m": other["offset_m"], "rms_m": other["rms_m"], "inliers": int(np.count_nonzero(second))}
            return result
        heights = pts[:, 2]
        storey = self.storey_height_m() or 3.0
        band = {"table": (0.25, 1.40), "floor": (-0.30, 0.30), "ceiling": (0.5 * storey, storey + 0.5)}[kind]
        in_band = (heights >= band[0]) & (heights <= band[1])
        if int(np.count_nonzero(in_band)) < 10:
            result["reason"] = "no points in the height band of the kind"
            return result
        values = heights[in_band]
        edges = np.arange(band[0], band[1] + 0.02, 0.02)
        counts, edges = np.histogram(values, bins=edges)
        if kind == "table":
            # the top surface: the highest bin that holds at least half of the densest bin
            strong = np.flatnonzero(counts >= 0.5 * counts.max())
            level_bin = int(strong.max())
        elif kind == "ceiling":
            level_bin = int(np.argmax(counts))
        else:
            level_bin = int(np.argmax(counts))
        level = 0.5 * (edges[level_bin] + edges[level_bin + 1])
        inlier = np.abs(heights - level) <= 1.5 * SURFACE_TOLERANCE_M
        surface = pts[inlier]
        centre, normal, rms = fit_plane(surface)
        if normal[2] < 0:
            normal = -normal
        xy = surface[:, :2]
        result.update({
            "ok": True, "normal": [_r(v, 4) for v in normal], "height_m": _r(float(np.median(surface[:, 2]))), "rms_m": _r(rms, 4),
            "inliers": int(len(surface)),
            "extent": {"x": [_r(xy[:, 0].min()), _r(xy[:, 0].max())], "y": [_r(xy[:, 1].min()), _r(xy[:, 1].max())]},
        })
        if kind == "table" and len(xy) >= 3:
            from shapely.geometry import MultiPoint

            rectangle = MultiPoint([tuple(p) for p in np.unique(np.round(xy, 3), axis=0)]).minimum_rotated_rectangle
            if rectangle.geom_type == "Polygon":
                corners = list(rectangle.exterior.coords)
                sides = sorted(math.dist(corners[i], corners[i + 1]) for i in range(2))
                result["top_size_m"] = [_r(sides[1]), _r(sides[0])]
        return result


def _upright_face(points: np.ndarray, fit_plane: Any) -> dict[str, Any]:
    centre, normal, rms = fit_plane(points)
    normal = np.asarray((normal[0], normal[1], 0.0))
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    if normal[0] < -1e-9 or (abs(normal[0]) <= 1e-9 and normal[1] < 0):  # canonical orientation
        normal = -normal
    direction = np.asarray((-normal[1], normal[0]))
    s = points[:, :2] @ direction
    offset = float(centre[:2] @ normal[:2])
    return {
        "normal": [_r(normal[0], 4), _r(normal[1], 4), 0.0], "offset_m": _r(offset, 4), "rms_m": _r(rms, 4),
        "extent": {"s": [_r(s.min()), _r(s.max())], "z": [_r(points[:, 2].min()), _r(points[:, 2].max())],
                   "length_m": _r(s.max() - s.min())},
    }


def _box_json(box: dict[str, Any]) -> dict[str, Any]:
    x0, x1, y0, y1, z0, z1 = parse_box(box)
    data: dict[str, Any] = {"x": [_r(x0), _r(x1)], "y": [_r(y0), _r(y1)]}
    if z0 is not None:
        data["z"] = [_r(z0), _r(z1)]
    return data


def _png_gray(image: np.ndarray) -> bytes:
    """Minimal 8-bit grayscale PNG encoder (standard library only)."""
    height, width = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
