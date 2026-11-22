"""Measured evidence for room objects: classified cloud, COLMAP-supported depth voxels and the mirrors of the scan.

Everything in :class:`RoomEvidence` is metric: plan x / y in metres, z in metres above the level floor.  Consumers
convert back with ``units_per_metre`` and ``floor_z`` (see :meth:`RoomEvidence.xy_local`, :meth:`RoomEvidence.z_local`).

The depth product adds measured points only where COLMAP supports the voxel (``source & 1``): Depth-Anything-only
voxels are occupancy and never give a dimension, an edge or a face.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)

# ``scalar_class`` indices of the indoor class set (own classes of a project are appended, these never move)
CLUTTER, WALL, CEILING, FLOOR, ELECTRICAL, HVAC, CEILING_EQUIPMENT, FURNITURE, APPLIANCE, SANITARY = 1, 2, 3, 4, 14, 15, 16, 17, 18, 19

SCREEN_LABELS = ("monitor", "laptop", "television", "screen")
LIGHT_LABELS = ("pendant light", "ceiling lamp", "spotlight")
CEILING_LABELS = ("smoke detector", "sprinkler", "projector", *LIGHT_LABELS, "ceiling fixture")
VENT_LABEL = "ventilation grille"

# every spelling a fine-term vocabulary or an instances.json of any age may contain
_CANONICAL = {
    "tv": "television", "tv set": "television", "flat screen tv": "television", "tv screen": "television",
    "computer monitor": "monitor", "computer screen": "monitor", "desktop monitor": "monitor", "flat screen": "screen",
    "laptop computer": "laptop", "ceiling light": "ceiling lamp", "led ceiling light": "ceiling lamp",
    "chandelier": "pendant light", "ceiling spotlight": "spotlight", "track light": "spotlight", "downlight": "spotlight",
    "smoke alarm": "smoke detector", "ceiling projector": "projector", "ceiling mounted fixture": "ceiling fixture",
    "air vent": "ventilation grille", "vent": "ventilation grille",
}


def canonical(name: object) -> str:
    text = str(name or "").strip().lower()
    return _CANONICAL.get(text, text)


@dataclass(frozen=True, slots=True)
class Mirror:
    """A planar mirror in plan metres; ``normal`` points to the side the reflections are seen from."""

    centre: np.ndarray
    normal: np.ndarray
    area_m2: float = 0.0


@dataclass(slots=True)
class RoomEvidence:
    xyz: np.ndarray  # (N, 3) metres, z above the level floor
    cls: np.ndarray  # scalar_class
    instance: np.ndarray  # 0 = none (always 0 for depth voxels)
    fine: np.ndarray  # fine term id, 0 = none
    brightness: np.ndarray  # mean RGB 0..255, NaN for depth voxels
    from_depth: np.ndarray
    fine_names: list[str] = field(default_factory=list)  # canonical, position + 1 = id
    instance_labels: dict[int, str] = field(default_factory=dict)  # canonical
    mirrors: list[Mirror] = field(default_factory=list)
    units_per_metre: float = 1.0
    floor_z: float = 0.0  # local units
    ceiling_m: float | None = None  # level ceiling above the floor
    screen_views: list[Any] = field(default_factory=list, repr=False)  # screen_frames.MaskView
    by_x: np.ndarray | None = field(default=None, repr=False)
    sorted_x: np.ndarray | None = field(default=None, repr=False)

    def xy_local(self, ring: Any) -> list[list[float]]:
        return [[float(point[0]) * self.units_per_metre, float(point[1]) * self.units_per_metre] for point in ring]

    def z_local(self, z_m: float) -> float:
        return self.floor_z + float(z_m) * self.units_per_metre

    def vertices_local(self, vertices: np.ndarray) -> list[list[float]]:
        scale = self.units_per_metre
        return [[float(x) * scale, float(y) * scale, self.floor_z + float(z) * scale] for x, y, z in vertices]

    def z_metres(self, z_local: float) -> float:
        return (float(z_local) - self.floor_z) / self.units_per_metre

    def term_ids(self, labels: tuple[str, ...]) -> np.ndarray:
        return np.asarray([index for index, name in enumerate(self.fine_names, start=1) if name in labels], dtype=np.int32)

    def instance_ids(self, labels: tuple[str, ...]) -> np.ndarray:
        return np.asarray(sorted(key for key, name in self.instance_labels.items() if name in labels), dtype=np.int64)

    def in_box(
        self, low: Any, high: Any, mask: np.ndarray | None = None, *, classes: tuple[int, ...] | None = None,
        cloud_only: bool = False,
    ) -> np.ndarray:
        """Ascending indices of the points inside an axis-aligned box (indexed along x: a scan has many million points)."""
        low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        if self.by_x is None:
            self.by_x = np.argsort(self.xyz[:, 0])
            self.sorted_x = self.xyz[self.by_x, 0]
        first, last = np.searchsorted(self.sorted_x, low[0], "left"), np.searchsorted(self.sorted_x, high[0], "right")
        near = self.by_x[first:last]
        y, z = self.xyz[near, 1], self.xyz[near, 2]
        keep = (y >= low[1]) & (y <= high[1]) & (z >= low[2]) & (z <= high[2])
        if mask is not None:
            keep &= mask[near]
        if classes is not None:
            keep &= np.isin(self.cls[near], classes)
        if cloud_only:
            keep &= ~self.from_depth[near]
        return np.sort(near[keep])


def build_evidence(
    xyz_local: np.ndarray, cls: np.ndarray, *, units_per_metre: float, floor_z: float, ceiling_z: float | None = None,
    instance: np.ndarray | None = None, fine: np.ndarray | None = None, rgb: np.ndarray | None = None,
    fine_names: list[str] | None = None, instance_labels: dict[int, str] | None = None,
    depth_xyz_local: np.ndarray | None = None, depth_cls: np.ndarray | None = None, mirrors: list[Mirror] | None = None,
) -> RoomEvidence:
    """Evidence from arrays in local plan units (the classified cloud and, optionally, measured depth voxels)."""
    upm = max(float(units_per_metre), 1e-9)
    cloud = np.asarray(xyz_local, dtype=np.float64).reshape(-1, 3)
    count = len(cloud)
    depth = np.zeros((0, 3)) if depth_xyz_local is None else np.asarray(depth_xyz_local, dtype=np.float64).reshape(-1, 3)
    metres = np.vstack((cloud, depth)) / upm
    metres[:, 2] -= float(floor_z) / upm
    classes = np.asarray(cls, dtype=np.uint8)
    if len(depth):
        classes = np.concatenate((classes, np.asarray(depth_cls, dtype=np.uint8)))
    names = [canonical(name) for name in (fine_names or [])]
    terms = np.zeros(len(metres), dtype=np.int32)
    if fine is not None and names:
        values = np.asarray(fine, dtype=np.int32)
        terms[:count] = np.where((values >= 0) & (values <= len(names)), values, 0)
    ids = np.zeros(len(metres), dtype=np.int64)
    if instance is not None:
        ids[:count] = np.asarray(instance, dtype=np.int64)
    brightness = np.full(len(metres), np.nan, dtype=np.float32)
    if rgb is not None:
        brightness[:count] = np.asarray(rgb, dtype=np.float32).mean(axis=1)
    from_depth = np.zeros(len(metres), dtype=bool)
    from_depth[count:] = True
    return RoomEvidence(
        xyz=metres, cls=classes, instance=ids, fine=terms, brightness=brightness, from_depth=from_depth, fine_names=names,
        instance_labels={int(key): canonical(value) for key, value in (instance_labels or {}).items()},
        mirrors=list(mirrors or []), units_per_metre=upm, floor_z=float(floor_z),
        ceiling_m=None if ceiling_z is None else (float(ceiling_z) - float(floor_z)) / upm,
    )


def load_room_evidence(
    instances_path: Path | str | None, frame: Any, *, units_per_metre: float, floor_z: float, ceiling_z: float | None,
    depth_min_views: int = 2,
) -> RoomEvidence | None:
    """Evidence of one level from the project products beside ``instances.json``; None without a classified PLY."""
    if instances_path is None:
        return None
    ply = Path(instances_path).with_name("classified.ply")
    if not ply.is_file():
        return None
    from samcloud.engine.ply_io import read_vertex_array

    vertex = read_vertex_array(ply)
    names = vertex.dtype.names or ()
    if "scalar_class" not in names or not len(vertex):
        return None
    world = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    local = np.asarray(frame.to_local(world), dtype=np.float64)
    low = float(floor_z) - 0.20 * units_per_metre
    high = math.inf if ceiling_z is None else float(ceiling_z) + 0.50 * units_per_metre
    keep = (local[:, 2] >= low) & (local[:, 2] <= high)
    depth = _depth_voxels(Path(instances_path).parents[1], frame, depth_min_views, low, high)
    has_rgb = all(name in names for name in ("red", "green", "blue"))
    evidence = build_evidence(
        local[keep], np.asarray(vertex["scalar_class"])[keep], units_per_metre=units_per_metre, floor_z=floor_z, ceiling_z=ceiling_z,
        instance=np.asarray(vertex["instance_id"])[keep] if "instance_id" in names else None,
        fine=np.asarray(vertex["fine_term"])[keep] if "fine_term" in names else None,
        rgb=np.column_stack((vertex["red"], vertex["green"], vertex["blue"]))[keep] if has_rgb else None,
        fine_names=_fine_names(ply.with_name("fine_labels.json")),
        instance_labels=_instance_labels(Path(instances_path)),
        depth_xyz_local=depth[0], depth_cls=depth[1],
        mirrors=_mirrors(Path(instances_path).parents[1], frame, units_per_metre, floor_z),
    )
    from samcloud.engine.screen_masks import FILE_NAME as SCREEN_MASKS_NAME

    masks = ply.with_name(SCREEN_MASKS_NAME)
    if masks.is_file():
        from samcloud.bim.room_objects.screen_frames import load_mask_views

        evidence.screen_views = load_mask_views(
            masks, world, frame.to_local, units_per_metre=units_per_metre, floor_z=float(floor_z),
        )
    return evidence


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError):
        return None


def _fine_names(path: Path) -> list[str]:
    payload = _read_json(path)
    return [str(name) for name in payload.get("labels", [])] if isinstance(payload, dict) else []


def _instance_labels(path: Path) -> dict[int, str]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {}
    return {int(item["id"]): str(item.get("label") or "") for item in payload.get("instances", []) if isinstance(item, dict) and "id" in item}


def _depth_voxels(products: Path, frame: Any, min_views: int, low: float, high: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = products / "depth" / "depth_points.npz"
    if not path.is_file():
        return None, None
    try:
        with np.load(path, allow_pickle=False) as data:
            classes = np.asarray(data["class_id"], dtype=np.uint8)
            measured = ((np.asarray(data["source"]) & 1) > 0) & (np.asarray(data["n_views"]) >= int(min_views))
            measured &= np.isin(classes, (WALL, CEILING, HVAC, CEILING_EQUIPMENT, FURNITURE))
            world = np.asarray(data["xyz"][measured], dtype=np.float64)
    except (OSError, ValueError, KeyError):
        _LOGGER.warning("depth product unreadable; room objects use the classified cloud only", exc_info=True)
        return None, None
    if not len(world):
        return None, None
    local = np.asarray(frame.to_local(world), dtype=np.float64)
    keep = (local[:, 2] >= low) & (local[:, 2] <= high)
    return local[keep], classes[measured][keep]


def _mirrors(products: Path, frame: Any, units_per_metre: float, floor_z: float) -> list[Mirror]:
    """Mirrors the surface fusion detected (``products/dense/surface_fusion.json``)."""
    payload = _read_json(products / "dense" / "surface_fusion.json")
    if not isinstance(payload, dict) or not isinstance(payload.get("mirrors"), list):
        return []
    mirrors: list[Mirror] = []
    for item in payload["mirrors"]:
        try:
            centre = np.asarray(item["centre"], dtype=np.float64)
            local = np.asarray(frame.to_local(np.vstack((centre, centre + np.asarray(item["normal"], dtype=np.float64)))), dtype=np.float64)
            area = float(item.get("area") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        direction = local[1] - local[0]
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            continue
        position = (local[0] - np.asarray((0.0, 0.0, floor_z))) / units_per_metre
        mirrors.append(Mirror(position, direction / length, area * (length / units_per_metre) ** 2))
    return mirrors
