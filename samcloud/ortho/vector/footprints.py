"""Height-map based indoor object footprints.

The routines in this module deliberately work in local scan units.  Callers
provide ``units_per_metre`` so every geometric threshold remains scale-safe for
unscaled indoor scans.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Any, Iterable

import cv2
import numpy as np
from scipy import ndimage
from shapely import affinity
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon, box
from shapely.ops import unary_union
from skimage.segmentation import watershed

from .raster_grid import RasterGrid, mask_to_polygons


_STRUCTURAL_CODES = (100, 103, 104, 108)


@dataclass(slots=True)
class ObjectInstance:
    """One floor-standing object region and the evidence used to classify it."""

    mask: np.ndarray
    label: str | None
    code: int
    has_fine_label: bool
    height: float
    solidity: float
    polygon: Polygon | None = None  # pre-fitted footprint (chairs); skips the generic fit
    instance_id: int = 0


@dataclass(slots=True)
class FootprintFit:
    """A fitted footprint together with fit-quality metrics."""

    kind: str
    polygon: Polygon
    dev90: float
    iou: float
    radius: float | None = None


@dataclass(frozen=True, slots=True)
class FurnitureSolidSettings:
    """Metric thresholds for point-measured stepped furniture solids."""

    min_points: int = 24
    min_level_delta_m: float = 0.15
    min_part_area_m2: float = 0.08
    max_parts: int = 4
    # A level below this height is scan noise near the floor, not a furniture part; its cells join the next level up.
    min_level_z_m: float = 0.20


def object_instances(
    dsm: np.ndarray,
    floor_z: float,
    classes: np.ndarray,
    fine_classes: np.ndarray,
    fine_confidence: np.ndarray,
    fine_names: list[str],
    grid: RasterGrid,
    units_per_metre: float,
    ceiling_height: float,
    wall_lines: Iterable[LineString] = (),
    free_floor: np.ndarray | None = None,
    wall_support: np.ndarray | None = None,
    view_count: np.ndarray | None = None,
    instance_raster: np.ndarray | None = None,
    instance_labels: dict[int, str] | None = None,
) -> tuple[list[ObjectInstance], dict[str, Any], np.ndarray]:
    """Extract object instances from the median-filtered height map.

    Fine labels seed the split first.  Height discontinuities seed the remaining
    area and a watershed fills each connected object component between them.
    """
    raw = np.asarray(dsm, dtype=np.float32)
    instances_raster = None if instance_raster is None else np.asarray(instance_raster, dtype=np.uint32)
    if instances_raster is not None and instances_raster.shape != raw.shape:
        raise ValueError("Instance raster must match the DSM shape")
    hm = cv2.medianBlur(np.nan_to_num(raw - float(floor_z), nan=0.0).astype(np.float32), 5)
    pixel = grid.pixel_size
    wall_mask = _wall_mask(raw.shape, grid, wall_lines, 0.10 * units_per_metre)
    obj = (
        (hm >= 0.05 * units_per_metre)
        & (hm <= 0.97 * ceiling_height)
        & ~np.isin(classes, _STRUCTURAL_CODES)
        & ~wall_mask
    ).astype(np.uint8)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, kernel5)
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, kernel5)
    tall = ((hm >= 0.85 * ceiling_height) & (obj > 0)).astype(np.uint8)
    tall_radius = max(1, int(round(0.20 * units_per_metre / pixel)))
    tall_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tall_radius + 1, 2 * tall_radius + 1))
    obj[(tall > 0) & (cv2.morphologyEx(tall, cv2.MORPH_OPEN, tall_kernel) == 0)] = 0
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, kernel5).astype(bool)

    step = (cv2.dilate(hm, np.ones((3, 3), np.uint8)) - cv2.erode(hm, np.ones((3, 3), np.uint8))) >= 0.10 * units_per_metre
    min_pixels = max(1, int(math.ceil(0.04 * units_per_metre**2 / pixel**2)))
    close_radius = max(1, int(round(0.05 * units_per_metre / pixel)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
    _count, components = cv2.connectedComponents(obj.astype(np.uint8), connectivity=4)
    result: list[ObjectInstance] = []
    metrics: dict[str, Any] = {
        "footprints_noise_dropped": 0,
        "footprints_clutter_rectangles": 0,
        "fine_instances_by_label": {},
        "cushions_merged": 0,
        "clothes_absorbed": 0,
        "nightstands_split": 0,
    }
    floor_view_median = float(np.median(view_count[free_floor])) if view_count is not None and free_floor is not None and np.any(free_floor) else 0.0
    metrics["footprints_floor_view_median"] = floor_view_median
    for identifier in range(1, int(components.max()) + 1):
        component = components == identifier
        if int(component.sum()) < min_pixels:
            continue
        markers = np.zeros(component.shape, dtype=np.int32)
        marker_labels: dict[int, str | None] = {}
        marker_instance_ids: dict[int, int] = {}
        next_marker = 0
        qualifying_ids: list[int] = []
        if instances_raster is not None:
            minimum_instance_pixels = max(1, int(math.ceil(0.05 / pixel**2)))
            for instance_id in np.unique(instances_raster[component]):
                if instance_id > 0 and int(np.count_nonzero(component & (instances_raster == instance_id))) >= minimum_instance_pixels:
                    qualifying_ids.append(int(instance_id))
        instance_split = len(qualifying_ids) >= 2
        if instance_split:
            for instance_id in qualifying_ids:
                region = component & (instances_raster == instance_id)
                next_marker += 1
                markers[region] = next_marker
                marker_instance_ids[next_marker] = instance_id
                marker_labels[next_marker] = _label_for_region(region, fine_classes, fine_confidence, fine_names)
        else:
            eligible_fine = component & (fine_classes > 0) & (fine_confidence >= 0.4)
            for fine_id in np.unique(fine_classes[eligible_fine]):
                if fine_id <= 0 or fine_id > len(fine_names):
                    continue
                region = component & (fine_classes == fine_id) & (fine_confidence >= 0.4)
                region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel).astype(bool) & component
                if int(region.sum()) < min_pixels:
                    continue
                next_marker += 1
                markers[region & (markers == 0)] = next_marker
                marker_labels[next_marker] = _canonical_instance_label(fine_names[int(fine_id) - 1])
        rest = component & (markers == 0) & ~step
        rest_count, rest_labels = cv2.connectedComponents(rest.astype(np.uint8), connectivity=4)
        for rest_id in range(1, rest_count):
            region = rest_labels == rest_id
            if int(region.sum()) >= min_pixels:
                next_marker += 1
                markers[region] = next_marker
                marker_labels[next_marker] = None
        if next_marker == 0:
            continue
        # High gradients form ridges; watershed assigns them to the object seeds.
        gradient = cv2.magnitude(cv2.Sobel(hm, cv2.CV_32F, 1, 0), cv2.Sobel(hm, cv2.CV_32F, 0, 1))
        labels = watershed(gradient, markers, mask=component)
        # Fine soft goods and bedside furniture must retain disconnected pieces:
        # their ownership is resolved below against nearby furniture.  Other
        # semantic labels keep the established one-instance-per-component rule.
        grouped: dict[tuple[str | None, int], np.ndarray] = {}
        split_labels = {"chair", "armchair", "cushion", "pillow", "blanket", "clothes", "nightstand"}
        for marker in range(1, next_marker + 1):
            part = labels == marker
            if int(part.sum()) < min_pixels:
                continue
            label = marker_labels.get(marker)
            instance_id = marker_instance_ids.get(marker, _dominant_instance_id(part, instances_raster))
            if (label or "").strip().lower() not in split_labels and not instance_split:
                key = (label, 0)
                grouped[key] = grouped.get(key, np.zeros_like(component, dtype=bool)) | part
                continue
            piece_count, piece_labels = cv2.connectedComponents(part.astype(np.uint8), connectivity=8)
            for piece_id in range(1, piece_count):
                piece = piece_labels == piece_id
                if int(piece.sum()) >= min_pixels:
                    grouped[(label, marker * 10_000 + piece_id)] = piece
        for (label, _piece), part in grouped.items():
            instance = _make_instance(part, label, classes, hm, grid, instance_id=_dominant_instance_id(part, instances_raster))
            if instance is None:
                continue
            if not instance.has_fine_label and _is_noise(instance.mask, raw, classes, hm, instance.height, instance.solidity):
                metrics["footprints_noise_dropped"] += 1
                continue
            if not instance.has_fine_label and not _has_floor_contact(instance.mask, free_floor, wall_support, pixel, units_per_metre):
                metrics["footprints_noise_dropped"] += 1
                continue
            if not instance.has_fine_label and instance.height >= 0.85 * ceiling_height and instance.solidity < 0.6:
                metrics["footprints_noise_dropped"] += 1
                continue
            if not instance.has_fine_label and floor_view_median > 0.0 and view_count is not None and float(np.median(view_count[instance.mask])) < 0.30 * floor_view_median:
                metrics["footprints_noise_dropped"] += 1
                continue
            result.append(instance)
    metrics["instance_labels_applied"] = _apply_instance_labels(result, instance_labels)
    result = _apply_fine_label_semantics(result, classes, hm, grid, units_per_metre, metrics)
    return result, metrics, hm


_INSTANCE_LABEL_ALIASES = {"mattress": "bed", "duvet": "blanket", "pillow": "cushion", "couch": "sofa"}


# Point-cloud instances of these labels are openings, fixtures on other objects or generic
# furniture; they never name a floor-standing footprint.
_NON_FOOTPRINT_INSTANCE_LABELS = frozenset({
    "furniture", "window", "door", "faucet", "socket", "switch box", "light switch", "ventilation grille",
    "radiator", "heater", "curtain", "mirror", "lamp", "ceiling lamp",
})


def _apply_instance_labels(instances: list[ObjectInstance], instance_labels: dict[int, str] | None) -> int:
    """Name unlabelled footprints after their point-cloud instance (instances.json).

    A fine label from the raster stays when present; the instance label only fills
    gaps so objects are never less specific than without instance ids.
    """
    if not instance_labels:
        return 0
    applied = 0
    for instance in instances:
        if instance.instance_id <= 0 or instance.has_fine_label:
            continue
        label = str(instance_labels.get(int(instance.instance_id), "") or "").strip().lower()
        if not label or label in _NON_FOOTPRINT_INSTANCE_LABELS:
            continue
        instance.label = _canonical_instance_label(label)
        instance.has_fine_label = True
        applied += 1
    return applied


def _canonical_instance_label(name: str) -> str:
    """Bedding terms are part of the bed; mattress is the bed itself."""
    key = str(name).strip().lower()
    return _INSTANCE_LABEL_ALIASES.get(key, name)


def _apply_fine_label_semantics(
    instances: list[ObjectInstance],
    classes: np.ndarray,
    hm: np.ndarray,
    grid: RasterGrid,
    units_per_metre: float,
    metrics: dict[str, Any],
) -> list[ObjectInstance]:
    """Apply fine-label ownership before any geometric fitting or heuristics."""
    soft_labels = {"cushion", "pillow", "blanket"}
    furniture_labels = {"sofa", "bed", "armchair"}
    clothes_label = "clothes"
    touch_radius = max(1, int(round(0.05 * units_per_metre / grid.pixel_size)))
    touch_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * touch_radius + 1, 2 * touch_radius + 1))

    def label_of(instance: ObjectInstance) -> str:
        return str(instance.label or "").strip().lower()

    def touches(first: ObjectInstance, second: ObjectInstance) -> bool:
        return bool(np.any(cv2.dilate(first.mask.astype(np.uint8), touch_kernel).astype(bool) & second.mask))

    active = list(instances)
    discarded: set[int] = set()

    def merge_soft_goods() -> bool:
        changed = False
        for index, instance in enumerate(active):
            if index in discarded or label_of(instance) not in soft_labels:
                continue
            anchors = [
                (candidate_index, candidate)
                for candidate_index, candidate in enumerate(active)
                if candidate_index not in discarded and candidate_index != index
                and label_of(candidate) in furniture_labels and touches(instance, candidate)
            ]
            if not anchors:
                anchors = [
                    (candidate_index, candidate)
                    for candidate_index, candidate in enumerate(active)
                    if candidate_index not in discarded and candidate_index != index
                    and candidate.code == 117 and label_of(candidate) not in soft_labels | {clothes_label}
                    and touches(instance, candidate)
                ]
            if not anchors:
                continue
            target_index, target = max(anchors, key=lambda item: float(np.count_nonzero(item[1].mask)))
            merged = _make_instance(target.mask | instance.mask, target.label, classes, hm, grid, instance_id=target.instance_id)
            if merged is not None:
                merged.code = target.code
                active[target_index] = merged
                discarded.add(index)
                metrics["cushions_merged"] += 1
                changed = True
        return changed

    # Bedding chains (blanket touching a cushion touching the bed) resolve over
    # repeated passes; whatever never reaches a furniture owner is dropped when tiny.
    while merge_soft_goods():
        pass
    for index, instance in enumerate(active):
        if index not in discarded and label_of(instance) in soft_labels:
            if float(np.count_nonzero(instance.mask)) * grid.pixel_size**2 < 0.15 * units_per_metre**2:
                discarded.add(index)
    # The fine label names a bed, it never limits it: unlabelled coarse furniture
    # touching a bed instance belongs to the bed outline.
    for index, instance in enumerate(active):
        if index in discarded or instance.label is not None or instance.code != 117:
            continue
        beds = [
            (candidate_index, candidate)
            for candidate_index, candidate in enumerate(active)
            if candidate_index not in discarded and label_of(candidate) == "bed" and touches(instance, candidate)
            and float(np.count_nonzero(instance.mask)) < float(np.count_nonzero(candidate.mask))
            and instance.height <= candidate.height + 0.15 * units_per_metre
        ]
        if beds:
            target_index, target = max(beds, key=lambda item: float(np.count_nonzero(item[1].mask)))
            merged = _make_instance(target.mask | instance.mask, target.label, classes, hm, grid, instance_id=target.instance_id)
            if merged is not None:
                merged.code = target.code
                active[target_index] = merged
                discarded.add(index)

    for index, instance in enumerate(active):
        if index in discarded or label_of(instance) != clothes_label:
            continue
        anchors = [
            (candidate_index, candidate)
            for candidate_index, candidate in enumerate(active)
            if candidate_index not in discarded and candidate_index != index
            and label_of(candidate) != clothes_label and touches(instance, candidate)
        ]
        if anchors:
            target_index, target = max(anchors, key=lambda item: float(np.count_nonzero(item[1].mask)))
            merged = _make_instance(target.mask | instance.mask, target.label, classes, hm, grid, instance_id=target.instance_id)
            if merged is not None:
                merged.code = target.code
                active[target_index] = merged
                metrics["clothes_absorbed"] += 1
        discarded.add(index)

    result = [instance for index, instance in enumerate(active) if index not in discarded]
    counts: dict[str, int] = {}
    for instance in result:
        if instance.has_fine_label and instance.label:
            name = instance.label.strip().lower()
            counts[name] = counts.get(name, 0) + 1
    metrics["fine_instances_by_label"] = counts
    metrics["nightstands_split"] = counts.get("nightstand", 0)
    return result


def fit_footprint(mask: np.ndarray, grid: RasterGrid, hm: np.ndarray, theta: float, units_per_metre: float, *, edge_snap: bool = True, fine_label: bool = False, semantic_label: str | None = None) -> FootprintFit | None:
    """Fit the lowest-complexity footprint; irregular outlines become rectangles when one fits."""
    fit = _fit_footprint_primitive(mask, grid, hm, theta, units_per_metre, edge_snap=edge_snap, fine_label=fine_label, semantic_label=semantic_label)
    if fit is None or fit.kind in {"rect", "rect_free", "circle", "ellipse"}:
        return fit
    return _prefer_rectangle(fit, mask, grid, theta, units_per_metre)


def _prefer_rectangle(fit: FootprintFit, mask: np.ndarray, grid: RasterGrid, theta: float, units_per_metre: float) -> FootprintFit:
    """About 90 % of furniture is rectangular: swap a polygon for a rectangle that covers the mask.

    The rectangle must cover at least 80 % of the measured mask and add at most
    25 % of its area as extra.
    """
    polygons = mask_to_polygons(np.asarray(mask, dtype=bool), grid)
    if not polygons:
        return fit
    measured = max(polygons, key=lambda item: item.area)
    measured = _single_polygon(measured.buffer(0.02 * units_per_metre).buffer(-0.02 * units_per_metre))
    if measured is None or measured.area <= 0.0:
        return fit
    best: FootprintFit | None = None
    for kind, rectangle in (("rect", _wall_rectangle(measured, theta)), ("rect_free", measured.minimum_rotated_rectangle)):
        rectangle = _single_polygon(rectangle)
        if rectangle is None or rectangle.area <= 0.0:
            continue
        inter = rectangle.intersection(measured).area
        if inter < 0.80 * measured.area or rectangle.area - inter > 0.25 * measured.area:
            continue
        dev, iou = _scores(rectangle, measured)
        if best is None or iou > best.iou:
            best = FootprintFit(kind, rectangle, dev, iou)
    return best if best is not None else fit


def _fit_footprint_primitive(mask: np.ndarray, grid: RasterGrid, hm: np.ndarray, theta: float, units_per_metre: float, *, edge_snap: bool = True, fine_label: bool = False, semantic_label: str | None = None) -> FootprintFit | None:
    """Fit the lowest-complexity valid geometric footprint to an object mask."""
    polygons = mask_to_polygons(np.asarray(mask, dtype=bool), grid)
    if not polygons:
        return None
    measured = max(polygons, key=lambda item: item.area)
    measured = _single_polygon(measured.buffer(0.02 * units_per_metre).buffer(-0.02 * units_per_metre))
    if measured is None or measured.area <= 0.0:
        return None
    candidates: list[tuple[str, Polygon, int, float | None]] = []
    candidates.append(("rect", _wall_rectangle(measured, theta), 4, None))
    core_rectangle = _largest_mask_rectangle(np.asarray(mask, dtype=bool), grid)
    rectangularity = measured.area / max(measured.minimum_rotated_rectangle.area, 1e-9)
    if core_rectangle is not None and (rectangularity >= 0.90 or str(semantic_label or "").strip().lower() in {"table", "desk", "bed"}):
        candidates.append(("rect_core", core_rectangle, 4, None))
    candidates.append(("rect_free", measured.minimum_rotated_rectangle, 5, None))
    circle, radius = _circle(measured)
    candidates.append(("circle", circle, 3, radius))
    ellipse = _ellipse(measured)
    if ellipse is not None:
        candidates.append(("ellipse", ellipse, 5, None))
    regularised_candidates: list[Polygon] = []
    for tolerance in ((0.03, 0.045, 0.06) if fine_label else (0.03,)):
        regularised = _regularised(measured, theta, units_per_metre, tolerance * units_per_metre)
        if regularised is not None:
            regularised_candidates.append(regularised)
            candidates.append(("regularised", regularised, len(regularised.exterior.coords) - 1, None))
    free = measured.simplify(0.015 * units_per_metre, preserve_topology=True)
    if isinstance(free, Polygon):
        candidates.append(("free", free, len(free.exterior.coords) - 1, None))
    scored: list[FootprintFit] = []
    for kind, shape, cost, radius in candidates:
        shape = _single_polygon(shape)
        if shape is None or shape.area <= 0.0:
            continue
        if edge_snap and len(shape.exterior.coords) - 1 <= 12 and kind not in {"circle", "ellipse"} and not (kind == "rect" and grid.pixel_size >= 0.025 * units_per_metre):
            snapped = _snap_edges(shape, hm, grid, 0.05 * units_per_metre)
            # A gradient in a noisy height field can point at an internal step.
            # Retain the requested snap only when it does not worsen the measured
            # footprint fit; this preserves the primitive's geometric evidence.
            if snapped.is_valid and snapped.area > 0.5 * shape.area:
                shape = snapped
        dev, iou = _scores(shape, measured)
        # A compact rectangle can deliberately exclude a separately drawable
        # side appendix (for example a nightstand).  Score its boundary against
        # the retained rectangle while retaining IoU against the full evidence.
        output_kind = "rect" if kind == "rect_core" else kind
        if kind == "rect_core" and shape.area >= 0.85 * measured.area:
            # Only a core that keeps (nearly) the whole object may ignore a side
            # appendix; a label must never shrink an object to a small inner box.
            dev = 0.0
        scored.append(FootprintFit(output_kind, shape, dev, iou, radius))
    if not scored:
        return None
    qualified = [item for item in scored if item.dev90 <= 0.025 * units_per_metre and item.iou >= 0.90]
    # Furniture is predominantly rectangular.  Every non-clutter caller tries
    # primitives before a polygon, using robust candidates and the relaxed
    # evidence gate requested for indoor furniture.
    primitive = [item for item in scored if item.kind in {"rect", "rect_free", "circle", "ellipse"} and item.dev90 <= 0.05 * units_per_metre and item.iou >= 0.85]
    rect_candidates = [item for item in primitive if item.kind in {"rect", "rect_free"}]
    rect_best = min(rect_candidates, key=lambda item: item.dev90) if rect_candidates else None
    label = str(semantic_label or "").strip().lower()
    all_rectangles = [item for item in scored if item.kind in {"rect", "rect_free"}]
    if label in {"table", "desk", "bed"} and all_rectangles:
        return min(all_rectangles, key=lambda item: item.dev90)
    if label in {"chair", "stool"}:
        rounded = [item for item in primitive if item.kind in {"circle", "ellipse"}]
        if rounded:
            return min(rounded, key=lambda item: item.dev90)
    rect_reference = min((item.dev90 for item in scored if item.kind in {"rect", "rect_free"}), default=float("inf"))
    round_candidates = [item for item in scored if item.kind in {"circle", "ellipse"} and item.dev90 <= 0.025 * units_per_metre and item.iou >= 0.90 and item.dev90 <= rect_reference + 0.005 * units_per_metre]
    if round_candidates:
        best_round = min(round_candidates, key=lambda item: item.dev90)
        circle = next((item for item in round_candidates if item.kind == "circle" and item.dev90 <= best_round.dev90 + 0.005 * units_per_metre), None)
        return circle or best_round
    # A five-pixel circular close rounds the four corners noticeably on coarse
    # legacy rasters.  The underlying component is still a rectangle when its
    # area nearly fills its minimum rectangle; keep that primitive instead of
    # encoding the morphology kernel as a many-sided furniture outline.
    rectangularity = measured.area / max(measured.minimum_rotated_rectangle.area, 1e-9)
    rect = next((item for item in scored if item.kind == "rect"), None)
    if rectangularity >= 0.95 and rect is not None and rect.iou >= 0.90:
        return rect
    if rect_best is not None:
        regularised = [item for item in scored if item.kind == "regularised" and len(item.polygon.exterior.coords) - 1 <= 8 and item.iou >= rect_best.iou + 0.06]
        return min(regularised, key=lambda item: item.dev90) if regularised else rect_best
    regularised = [item for item in scored if item.kind == "regularised" and len(item.polygon.exterior.coords) - 1 <= 8]
    if regularised:
        return min(regularised, key=lambda item: item.dev90)
    if qualified:
        costs = {"circle": 3, "rect": 4, "ellipse": 5, "rect_free": 5}
        return min(qualified, key=lambda item: (costs.get(item.kind, len(item.polygon.exterior.coords) - 1), item.dev90))
    return min(scored, key=lambda item: item.dev90)


_TABLE_LABELS = {"table", "dining table", "desk"}


def _merge_split_tables(
    instances: list[ObjectInstance], classes: np.ndarray, hm: np.ndarray, grid: RasterGrid, units_per_metre: float, stats: dict[str, Any], theta: float = 0.0,
) -> list[ObjectInstance]:
    """One table seen as several touching instances (instance over-segmentation) becomes one instance.

    Instances with the same table label whose masks come within 0.20 m of each
    other are merged, so one table yields one outline instead of overlapping ones.
    """
    stats.setdefault("tables_merged", 0)
    tables = [index for index, item in enumerate(instances) if str(item.label or "").strip().lower() in _TABLE_LABELS]
    if len(tables) < 2:
        return instances
    radius = max(1, int(round(0.10 * units_per_metre / grid.pixel_size)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    grown = {index: cv2.dilate(instances[index].mask.astype(np.uint8), kernel).astype(bool) for index in tables}
    parent = {index: index for index in tables}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for position, first in enumerate(tables):
        for second in tables[position + 1:]:
            if str(instances[first].label).strip().lower() != str(instances[second].label).strip().lower():
                continue
            if np.any(grown[first] & grown[second]):
                parent[find(second)] = find(first)
    groups: dict[int, list[int]] = {}
    for index in tables:
        groups.setdefault(find(index), []).append(index)
    replaced: dict[int, ObjectInstance] = {}
    dropped: set[int] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        mask = np.zeros_like(instances[members[0]].mask)
        for member in members:
            mask |= instances[member].mask
        template = max((instances[member] for member in members), key=lambda item: int(np.count_nonzero(item.mask)))
        merged = _make_instance(mask, template.label, classes, hm, grid, instance_id=template.instance_id)
        if merged is None:
            continue
        merged.code = template.code
        merged.has_fine_label = template.has_fine_label
        # The union of table pieces is irregular: its footprint is the wall-axis rectangle,
        # not a free minimum-area rectangle that may come out diagonal.
        outlines = mask_to_polygons(mask, grid)
        if outlines:
            rectangle = _single_polygon(_wall_rectangle(unary_union(outlines).convex_hull, theta))
            if rectangle is not None and rectangle.area > 0.0:
                merged.polygon = rectangle
        replaced[members[0]] = merged
        dropped.update(members[1:])
        stats["tables_merged"] += len(members) - 1
    return [replaced.get(index, item) for index, item in enumerate(instances) if index not in dropped]


def refine_dining_instances(
    instances: list[ObjectInstance],
    classes: np.ndarray,
    grid: RasterGrid,
    hm: np.ndarray,
    theta: float,
    units_per_metre: float,
    metrics: dict[str, Any] | None = None,
    walls: Iterable[LineString] = (),
    room: Any = None,
) -> list[ObjectInstance]:
    """Turn a table group into one table rectangle plus small oriented chair rectangles.

    * Unlabelled furniture lying mostly (> 50 %) inside a table rectangle joins the table.
    * Pieces of the same chair (< 0.15 m apart) merge first.
    * Chair parts inside the table rectangle are clipped away, blobs under
      0.08 m2 are dropped and each chair becomes a minimum-area rectangle whose
      sides are clamped to 0.35 - 0.65 m.
    """
    stats = metrics if metrics is not None else {}
    for key in ("table_blobs_merged", "chairs_merged", "chairs_dropped", "chairs_implausible", "chairs_wall_clipped"):
        stats.setdefault(key, 0)
    pixel = grid.pixel_size
    upm = units_per_metre

    def label_of(item: ObjectInstance) -> str:
        return str(item.label or "").strip().lower()

    active = _split_dining_groups(list(instances), hm, grid, theta, upm, stats)
    active = _merge_split_tables(active, classes, hm, grid, upm, stats, theta)
    absorbed: set[int] = set()
    table_polygons: list[Polygon] = []
    for index in range(len(active)):
        table = active[index]
        if label_of(table) not in _TABLE_LABELS:
            continue
        fit = fit_footprint(table.mask, grid, hm, theta, upm, fine_label=True, semantic_label=table.label)
        if fit is None and table.polygon is None:
            continue
        rectangle = table.polygon if table.polygon is not None else fit.polygon
        rectangle_mask = _polygon_mask(rectangle, grid)
        merged_mask = table.mask.copy()
        for other_index, other in enumerate(active):
            if other_index == index or other_index in absorbed or other.label is not None or other.code != 117:
                continue
            size = float(np.count_nonzero(other.mask))
            if size > 0.0 and float(np.count_nonzero(other.mask & rectangle_mask)) / size > 0.50:
                merged_mask |= other.mask
                absorbed.add(other_index)
                stats["table_blobs_merged"] += 1
        if not np.array_equal(merged_mask, table.mask):
            rebuilt = _make_instance(merged_mask, table.label, classes, hm, grid)
            if rebuilt is not None:
                rebuilt.code = table.code
                active[index] = rebuilt
                if table.polygon is not None:
                    outlines = mask_to_polygons(rebuilt.mask, grid)
                    rebuilt.polygon = _single_polygon(_wall_rectangle(unary_union(outlines).convex_hull, theta)) if outlines else table.polygon
                    rectangle = rebuilt.polygon or rectangle
                else:
                    refit = fit_footprint(rebuilt.mask, grid, hm, theta, upm, fine_label=True, semantic_label=rebuilt.label)
                    if refit is not None:
                        rectangle = refit.polygon
        table_polygons.append(rectangle)
    active = [item for index, item in enumerate(active) if index not in absorbed]
    table_union = unary_union(table_polygons) if table_polygons else Polygon()
    table_mask = None
    for rectangle in table_polygons:
        part = _polygon_mask(rectangle, grid)
        table_mask = part if table_mask is None else table_mask | part

    chair_positions = [index for index, item in enumerate(active) if label_of(item) == "chair"]
    if not chair_positions:
        return active
    radius = max(1, int(round(0.075 * upm / pixel)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    # Chair pieces are clipped by the table tops first; pieces of one chair (< 0.15 m apart) merge only while the
    # merged piece still fits one chair - neighbouring chairs around a table stay separate.
    clipped = {index: _without_slivers(active[index].mask & ~table_mask, 0.05 * upm / pixel) if table_mask is not None else active[index].mask.copy() for index in chair_positions}
    grown = {index: cv2.dilate(clipped[index].astype(np.uint8), kernel).astype(bool) for index in chair_positions}
    parent = {index: index for index in chair_positions}
    merged_masks = {index: clipped[index].copy() for index in chair_positions}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first_at, first in enumerate(chair_positions):
        for second in chair_positions[first_at + 1:]:
            root_first, root_second = find(first), find(second)
            if root_first == root_second or not np.any(grown[first] & grown[second]):
                continue
            union = merged_masks[root_first] | merged_masks[root_second]
            if not _fits_one_chair(union, grid, theta, _CHAIR_MAX_M * upm):
                continue
            parent[root_second] = root_first
            merged_masks[root_first] = union
    groups: dict[int, list[int]] = {}
    for index in chair_positions:
        groups.setdefault(find(index), []).append(index)
    result = [item for index, item in enumerate(active) if index not in parent]
    low, high = 0.35 * upm, 0.65 * upm
    wall_band = unary_union([line.buffer(0.03 * upm) for line in walls]) if walls else Polygon()
    chairs: list[tuple[ObjectInstance, float]] = []
    for root, members in groups.items():
        mask = merged_masks[root]
        stats["chairs_merged"] += len(members) - 1
        if not np.any(mask):
            stats["chairs_dropped"] += 1
            continue
        chair_height = float(np.percentile(hm[mask], 90))
        if not (0.35 * upm <= chair_height <= 1.10 * upm):
            # Not a chair (curtain, shelf, ...): it follows the normal furniture path as unnamed furniture.
            for member in members:
                piece = active[member]
                piece.label = None
                piece.has_fine_label = False
                result.append(piece)
            stats["chairs_implausible"] += 1
            continue
        polygons = mask_to_polygons(mask, grid)
        shape = unary_union(polygons) if polygons else Polygon()
        table = min(table_polygons, key=lambda item: item.distance(shape)) if table_polygons and not shape.is_empty else None
        at_table = table is not None and table.distance(shape) <= _CHAIR_TABLE_REACH_M * upm
        # A chair at a table is often traced only in part (its seat under the top): a smaller piece suffices there.
        minimum = (_CHAIR_AT_TABLE_MIN_M2 if at_table else 0.08) * upm**2
        if float(np.count_nonzero(mask)) * pixel**2 < minimum or not polygons:
            stats["chairs_dropped"] += 1
            continue
        if at_table:
            # A chair at a table: a seat-sized box on the table's axes.
            rectangle = _axis_box(shape, table, _CHAIR_SIDE_M[0] * upm, _CHAIR_SIDE_M[1] * upm)
        else:
            rectangle = _clamped_rectangle(shape, low, high)
        if rectangle is None:
            stats["chairs_dropped"] += 1
            continue
        obstacle = unary_union([table_union, wall_band]) if not wall_band.is_empty else table_union
        # a chair at a table slides straight out from under its top (walls and other tables are checked below)
        moved = _move_clear(rectangle, table if at_table else obstacle)
        # (ceiling rooms leave holes under furniture: a chair at a table only has to lie mostly in the room)
        if room is None or room.is_empty:
            inside = True
        elif at_table:
            inside = moved.intersection(room).area >= 0.5 * moved.area
        else:
            inside = room.buffer(0.02 * upm).contains(moved)
        if (
            moved.intersection(obstacle).area <= 1e-9 and inside
            and moved.centroid.distance(rectangle.centroid) <= 0.45 * upm
        ):
            rectangle = moved
        else:
            clipped_rectangle = clip_to_room(rectangle, obstacle, room, upm)
            if clipped_rectangle is not None and clipped_rectangle is not rectangle:
                stats["chairs_wall_clipped"] += 1
                clipped_rectangle = refit_clipped(clipped_rectangle, rectangle, walls, upm, force_rectangle=True)
            if clipped_rectangle is None:
                stats["chairs_dropped"] += 1
                continue
            rectangle = clipped_rectangle
        template = active[members[0]]
        instance = _make_instance(mask, template.label, classes, hm, grid)
        if instance is None:
            stats["chairs_dropped"] += 1
            continue
        instance.code = template.code
        instance.polygon = rectangle
        chairs.append((instance, float(np.count_nonzero(mask))))
    # Two boxes for one seat (pieces that did not merge): the smaller one goes, a slight overlap is pushed apart.
    placed: list[Polygon] = []
    for instance, _size in sorted(chairs, key=lambda item: -item[1]):
        rectangle = instance.polygon
        others = unary_union(placed) if placed else Polygon()
        if not others.is_empty and rectangle.intersection(others).area > 0.40 * rectangle.area:
            stats["chairs_dropped"] += 1
            stats["chairs_duplicate"] = stats.get("chairs_duplicate", 0) + 1
            continue
        if not others.is_empty and rectangle.intersection(others).area > 1e-9:
            moved = _move_clear(rectangle, unary_union([others, table_union]))
            if moved.centroid.distance(rectangle.centroid) <= 0.20 * upm:
                rectangle = moved
                instance.polygon = moved
        placed.append(rectangle)
        result.append(instance)
    return result


_CHAIR_MAX_M = 0.70  # a merged chair piece never exceeds this extent along either wall axis
_CHAIR_SIDE_M = (0.40, 0.55)  # seat box of a chair at a table
_CHAIR_TABLE_REACH_M = 0.60
_CHAIR_AT_TABLE_MIN_M2 = 0.04


def _without_slivers(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """A chair piece clipped by a table keeps no thin strip along the table edge (morphological opening)."""
    radius = max(1, int(round(radius_px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    return opened if opened.any() else mask


def _fits_one_chair(mask: np.ndarray, grid: RasterGrid, theta: float, limit: float) -> bool:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return True
    x = grid.min_x + (cols + 0.5) * grid.pixel_size
    y = grid.max_y - (rows + 0.5) * grid.pixel_size
    u = x * math.cos(theta) + y * math.sin(theta)
    v = -x * math.sin(theta) + y * math.cos(theta)
    return float(u.max() - u.min()) <= limit and float(v.max() - v.min()) <= limit


def _axis_box(shape: Any, table: Polygon, low: float, high: float) -> Polygon | None:
    """Box on the axes of ``table`` around ``shape`` (its extent clamped to low..high on both axes)."""
    rectangle = _single_polygon(table.minimum_rotated_rectangle)
    if rectangle is None or rectangle.area <= 0.0:
        return None
    corners = np.asarray(rectangle.exterior.coords[:4], dtype=float)
    first = corners[1] - corners[0]
    axis_a = first / max(float(np.linalg.norm(first)), 1e-12)
    axis_b = np.asarray((-axis_a[1], axis_a[0]))
    points = np.asarray([point for part in getattr(shape, "geoms", [shape]) for point in part.exterior.coords], dtype=float)
    if not len(points):
        return None
    a, b = points @ axis_a, points @ axis_b
    centre_a, centre_b = (a.min() + a.max()) / 2.0, (b.min() + b.max()) / 2.0
    half_a = min(max(a.max() - a.min(), low), high) / 2.0
    half_b = min(max(b.max() - b.min(), low), high) / 2.0
    box_points = [axis_a * (centre_a + sa * half_a) + axis_b * (centre_b + sb * half_b) for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return Polygon([(float(x), float(y)) for x, y in box_points])


def _split_dining_groups(instances: list[ObjectInstance], hm: np.ndarray, grid: RasterGrid, theta: float, units_per_metre: float, stats: dict[str, Any]) -> list[ObjectInstance]:
    """A dining group (table pieces, chairs and unlabelled furniture touching them) gets one table top.

    The table top is the largest table piece shaped like a table top (short side >= 0.5 m, area >= 0.35 m2,
    filling at least 45 % of its wall-axis rectangle).  Without one, an unlabelled piece of that shape at table
    height (0.60 - 0.90 m) is the top: a white or glass top returns few points of its own, so the instance
    labels land on the chairs around it.  Other table pieces of at most 0.6 m2 with at least 0.06 m2 (and 30 %)
    outside the top are chairs; their part over the top is clipped off with the table rectangle later.
    """
    upm = units_per_metre
    stats.setdefault("dining_tops_from_unlabelled", 0)
    stats.setdefault("dining_table_pieces_to_chairs", 0)
    labels = [str(item.label or "").strip().lower() for item in instances]
    tables = [index for index, label in enumerate(labels) if label in _TABLE_LABELS]
    chairs = [index for index, label in enumerate(labels) if label == "chair"]
    if not tables or not chairs:
        return instances
    loose = [index for index, item in enumerate(instances) if item.label is None and item.code == 117]
    members = tables + chairs + loose
    radius = max(1, int(round(0.10 * upm / grid.pixel_size)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    grown = {index: cv2.dilate(instances[index].mask.astype(np.uint8), kernel).astype(bool) for index in members}
    parent = {index: index for index in members}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for position, first in enumerate(members):
        for second in members[position + 1:]:
            if find(first) != find(second) and np.any(grown[first] & instances[second].mask):
                parent[find(second)] = find(first)
    groups: dict[int, list[int]] = {}
    for index in members:
        groups.setdefault(find(index), []).append(index)
    pixel_area = grid.pixel_size**2

    def top_shape(index: int) -> Polygon | None:
        item = instances[index]
        area = float(np.count_nonzero(item.mask)) * pixel_area
        outlines = mask_to_polygons(item.mask, grid)
        if area < 0.35 * upm**2 or not outlines:
            return None
        rectangle = _single_polygon(_wall_rectangle(unary_union(outlines).convex_hull, theta))
        if rectangle is None or rectangle.area <= 0.0:
            return None
        corners = np.asarray(rectangle.exterior.coords[:4], dtype=float)
        short = min(float(np.linalg.norm(corners[1] - corners[0])), float(np.linalg.norm(corners[3] - corners[0])))
        if short < 0.5 * upm or area < 0.45 * rectangle.area:
            return None
        return rectangle

    for group in groups.values():
        group_tables = [index for index in group if index in tables]
        if not group_tables or not any(index in chairs for index in group):
            continue
        tops = [(index, shape) for index in group_tables if (shape := top_shape(index)) is not None]
        from_loose = False
        if not tops:
            tops = [(index, shape) for index in group if index in loose and 0.60 * upm <= instances[index].height <= 0.90 * upm and (shape := top_shape(index)) is not None]
            from_loose = True
        if not tops:
            continue
        top, rectangle = max(tops, key=lambda item: int(np.count_nonzero(instances[item[0]].mask)))
        if from_loose:
            # the labels of the pieces around it belong to the chairs: the top is just a table
            instances[top].label = "table"
            instances[top].has_fine_label = True
            stats["dining_tops_from_unlabelled"] += 1
        top_mask = _polygon_mask(rectangle.buffer(0.05 * upm, join_style=2), grid)
        for index in group_tables:
            if index == top:
                continue
            item = instances[index]
            size = float(np.count_nonzero(item.mask))
            outside = item.mask & ~top_mask
            if size <= 0.0 or size * pixel_area > 0.60 * upm**2:
                continue
            if float(np.count_nonzero(outside)) >= max(0.30 * size, 0.06 * upm**2 / pixel_area):
                # its part outside the top is a chair (the part over the top is clipped off with the table later)
                item.label = "chair"
                item.has_fine_label = True
                stats["dining_table_pieces_to_chairs"] += 1
            else:
                # a sliver of a chair hangs on to this table piece: the piece only adds what lies over the top
                item.mask = item.mask & top_mask
    return [item for item in instances if np.any(item.mask)]


def refit_clipped(
    clipped: Polygon,
    original: Polygon,
    walls: Iterable[LineString],
    units_per_metre: float,
    *,
    force_rectangle: bool,
    min_area_m2: float = 0.08,
) -> Polygon | None:
    """Refit a wall-clipped piece with its largest inscribed rectangle.

    The rectangle is aligned to the nearest wall line or to the axes of the
    original outline, whichever covers more.  Without ``force_rectangle`` the
    rectangle replaces the piece only when it covers at least 80 % of it.
    Returns ``None`` when the refit is smaller than ``min_area_m2``.
    """
    if clipped is original:
        return clipped
    angles: list[float] = []
    lines = [line for line in walls if line.length > 0.0]
    if lines:
        nearest = min(lines, key=lambda line: line.distance(clipped.centroid))
        (x0, y0), (x1, y1) = nearest.coords[0], nearest.coords[-1]
        angles.append(math.atan2(y1 - y0, x1 - x0))
    box = original.minimum_rotated_rectangle
    if isinstance(box, Polygon) and not box.is_empty:
        (x0, y0), (x1, y1) = box.exterior.coords[0], box.exterior.coords[1]
        angles.append(math.atan2(y1 - y0, x1 - x0))
    best: Polygon | None = None
    for angle in angles:
        candidate = _largest_inscribed_rectangle(clipped, angle, 0.01 * units_per_metre)
        if candidate is not None and (best is None or candidate.area > best.area):
            best = candidate
    if best is None or best.area < min_area_m2 * units_per_metre**2:
        return clipped if not force_rectangle and clipped.area >= min_area_m2 * units_per_metre**2 else None
    if force_rectangle or best.area >= 0.80 * clipped.area:
        return best
    return clipped


def _largest_inscribed_rectangle(polygon: Polygon, angle: float, resolution: float) -> Polygon | None:
    """Largest rectangle inside ``polygon`` whose sides follow ``angle`` (radians)."""
    centre = polygon.centroid
    rotated = affinity.rotate(polygon, -math.degrees(angle), origin=(centre.x, centre.y))
    min_x, min_y, max_x, max_y = rotated.bounds
    width = int(math.ceil((max_x - min_x) / resolution)) + 1
    height = int(math.ceil((max_y - min_y) / resolution)) + 1
    if width < 2 or height < 2 or width * height > 4_000_000:
        return None
    grid = RasterGrid(min_x, max_y, resolution, width, height)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray([[(x - min_x) / resolution - 0.5, (max_y - y) / resolution - 0.5] for x, y in rotated.exterior.coords], dtype=np.float32).round().astype(np.int32)], 1)
    for ring in rotated.interiors:
        cv2.fillPoly(mask, [np.asarray([[(x - min_x) / resolution - 0.5, (max_y - y) / resolution - 0.5] for x, y in ring.coords], dtype=np.float32).round().astype(np.int32)], 0)
    rectangle = _largest_mask_rectangle(mask.astype(bool), grid)
    if rectangle is None:
        return None
    # The corner helper reports pixel centres; move to pixel edges and keep a margin inside the piece.
    rectangle = affinity.translate(rectangle, -0.5 * resolution, 0.5 * resolution).buffer(-0.6 * resolution, join_style=2)
    if not isinstance(rectangle, Polygon) or rectangle.is_empty:
        return None
    return affinity.rotate(rectangle, math.degrees(angle), origin=(centre.x, centre.y))


def wall_band_of(walls: Iterable[LineString], units_per_metre: float) -> Polygon:
    """Thin corridor around wall lines; furniture must not reach into it."""
    lines = list(walls)
    return unary_union([line.buffer(0.03 * units_per_metre) for line in lines]) if lines else Polygon()


def clip_to_room(polygon: Polygon, obstacle: Any, room: Any, units_per_metre: float, *, min_area_m2: float = 0.08) -> Polygon | None:
    """Clip ``polygon`` to the room side of the walls (and away from ``obstacle``).

    Returns the polygon itself when nothing needs clipping, the largest remaining
    piece otherwise, and ``None`` when that piece is under ``min_area_m2``.
    """
    clipped: Any = polygon
    if obstacle is not None and not obstacle.is_empty:
        clipped = clipped.difference(obstacle)
    if room is not None and not room.is_empty:
        clipped = clipped.intersection(room)
    if clipped.is_empty:
        return None
    if clipped.area >= 0.995 * polygon.area:
        return polygon
    pieces = [piece for piece in getattr(clipped, "geoms", [clipped]) if isinstance(piece, Polygon) and not piece.is_empty]
    if not pieces:
        return None
    best = max(pieces, key=lambda piece: piece.area)
    if best.area < min_area_m2 * units_per_metre**2:
        return None
    return best


def _clamped_rectangle(shape: object, low: float, high: float) -> Polygon | None:
    """Minimum-area oriented rectangle of ``shape`` with both sides clamped to [low, high]."""
    rectangle = _single_polygon(getattr(shape, "minimum_rotated_rectangle", None))
    if rectangle is None or rectangle.area <= 0.0:
        return None
    corners = np.asarray(rectangle.exterior.coords[:4], dtype=float)
    first, second = corners[1] - corners[0], corners[3] - corners[0]
    length_a, length_b = float(np.linalg.norm(first)), float(np.linalg.norm(second))
    if length_a <= 1e-9 or length_b <= 1e-9:
        return None
    axis_a, axis_b = first / length_a, second / length_b
    centre = corners.mean(axis=0)
    half_a, half_b = min(max(length_a, low), high) / 2.0, min(max(length_b, low), high) / 2.0
    points = [centre + sa * half_a * axis_a + sb * half_b * axis_b for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return Polygon([(float(x), float(y)) for x, y in points])


def _move_clear(rectangle: Polygon, obstacle: Polygon) -> Polygon:
    """Slide a rectangle along its own axes, by the smallest step, until it leaves ``obstacle``."""
    if obstacle.is_empty or rectangle.intersection(obstacle).area <= 1e-9:
        return rectangle
    corners = np.asarray(rectangle.exterior.coords[:4], dtype=float)
    axes = [corners[1] - corners[0], corners[3] - corners[0]]
    axes = [axis / max(float(np.linalg.norm(axis)), 1e-12) for axis in axes]
    best: Polygon | None = None
    best_shift = float("inf")
    extent = (rectangle.bounds[2] - rectangle.bounds[0]) + (rectangle.bounds[3] - rectangle.bounds[1]) + (obstacle.bounds[2] - obstacle.bounds[0]) + (obstacle.bounds[3] - obstacle.bounds[1])
    for axis in axes:
        for sign in (-1.0, 1.0):
            direction = sign * axis
            if affinity.translate(rectangle, *(direction * extent)).intersection(obstacle).area > 1e-9:
                continue
            low_shift, high_shift = 0.0, extent
            for _ in range(40):
                middle = 0.5 * (low_shift + high_shift)
                if affinity.translate(rectangle, *(direction * middle)).intersection(obstacle).area > 1e-9:
                    low_shift = middle
                else:
                    high_shift = middle
            if high_shift < best_shift:
                best_shift = high_shift
                best = affinity.translate(rectangle, *(direction * (high_shift + 1e-6)))
    return best if best is not None else rectangle


def clutter_rectangle(mask: np.ndarray, grid: RasterGrid, theta: float, units_per_metre: float, free_floor: np.ndarray | None = None) -> Polygon | None:
    """Return a safe compact clutter rectangle, never a room-spanning box."""
    rectangle = wall_axis_rectangle(mask, grid, theta)
    if rectangle is None:
        return None
    polygons = mask_to_polygons(mask, grid)
    if not polygons:
        return None
    source = max(polygons, key=lambda item: item.area)
    iou = rectangle.intersection(source).area / max(rectangle.union(source).area, 1e-9)
    sides = sorted(math.dist(first, second) for first, second in zip(rectangle.exterior.coords, list(rectangle.exterior.coords)[1:]))
    if iou < 0.70 or rectangle.area > 2.0 * units_per_metre**2 or sides[0] > 1.2 * units_per_metre:
        return None
    if free_floor is not None:
        inside = _polygon_mask(rectangle, grid)
        if np.any(inside) and float(np.mean(free_floor[inside])) > 0.25:
            return None
    return rectangle


def height_step_details(instance: ObjectInstance, hm: np.ndarray, grid: RasterGrid, theta: float, units_per_metre: float, outline: Polygon) -> list[LineString]:
    """Return up to three interior height-step lines for a fitted footprint."""
    mask = instance.mask
    steps = (cv2.dilate(hm, np.ones((3, 3), np.uint8)) - cv2.erode(hm, np.ones((3, 3), np.uint8))) >= 0.10 * units_per_metre
    parts_count, parts = cv2.connectedComponents((mask & ~steps).astype(np.uint8), connectivity=4)
    minimum = 0.04 * units_per_metre**2
    details: list[LineString] = []
    for identifier in range(1, parts_count):
        region = parts == identifier
        if region.sum() * grid.pixel_size**2 < minimum or region.sum() >= 0.80 * mask.sum():
            continue
        fit = fit_footprint(region, grid, hm, theta, units_per_metre, edge_snap=False, fine_label=True)
        if fit is None or fit.kind == "free":
            continue
        geometry = fit.polygon.boundary.difference(outline.boundary.buffer(0.02 * units_per_metre))
        for line in _lines(geometry):
            overlap = line.intersection(outline.boundary.buffer(0.04 * units_per_metre)).length
            if line.length >= 0.15 * units_per_metre and overlap <= 0.50 * line.length:
                details.append(line)
    return sorted(details, key=lambda item: item.length, reverse=True)[:2]


def _level_centres(values: np.ndarray, settings: FurnitureSolidSettings, min_cells: int) -> list[float]:  # values in metres
    """Deterministic 1-D k-means: the largest k whose centres are distinct and every cluster is a real part."""
    values = np.sort(np.asarray(values, dtype=float))
    for count in range(settings.max_parts, 1, -1):
        centres = np.percentile(values, np.linspace(10, 90, count))
        for _ in range(30):
            assign = np.argmin(np.abs(values[:, None] - centres[None, :]), axis=1)
            updated = np.asarray([values[assign == index].mean() if np.any(assign == index) else centres[index] for index in range(count)])
            if np.allclose(updated, centres, atol=1e-5):
                break
            centres = updated
        centres = np.sort(centres)
        centres = centres[centres >= settings.min_level_z_m]
        if len(centres) < 2:
            continue
        assign = np.argmin(np.abs(values[:, None] - centres[None, :]), axis=1)
        sizes = np.bincount(assign, minlength=count)
        if np.all(np.diff(centres) >= settings.min_level_delta_m) and np.all(sizes >= min_cells):
            return [float(value) for value in centres]
    return []


def furniture_solid_parts(
    instance: ObjectInstance,
    height_map: np.ndarray,
    grid: RasterGrid,
    units_per_metre: float,
    outline: Polygon,
    settings: FurnitureSolidSettings = FurnitureSolidSettings(),
) -> list[dict[str, Any]]:
    """Return clean, point-measured stepped solids for one furniture footprint.

    The per-instance DSM is reduced to coarse cells (median height, holes filled
    from the nearest measured cell, majority-filtered), clustered into 2..max_parts
    height levels and split into connected parts.  Cells without data take the
    lowest level, so the union of the parts stays the drawn 2D footprint.
    Returns [] (one plain box) when the scan shows no distinct, sizeable level.
    """
    upm = max(float(units_per_metre), 1e-9)
    mask = np.asarray(instance.mask, dtype=bool)
    heights = np.asarray(height_map, dtype=float)
    if mask.shape != heights.shape or int(mask.sum()) < settings.min_points:
        return []
    inside = _polygon_mask(outline, grid)
    rows, cols = np.nonzero(inside | mask)
    if not len(rows):
        return []
    block = max(1, int(round(0.05 * upm / grid.pixel_size)))
    r0, c0 = int(rows.min()) // block * block, int(cols.min()) // block * block
    r1, c1 = -(-(int(rows.max()) + 1) // block) * block, -(-(int(cols.max()) + 1) // block) * block
    height_rows, height_cols = min(r1, heights.shape[0]) - r0, min(c1, heights.shape[1]) - c0

    def reduce(array: np.ndarray, fill: float) -> np.ndarray:
        canvas = np.full((r1 - r0, c1 - c0), fill, dtype=float)
        canvas[:height_rows, :height_cols] = array[r0:r0 + height_rows, c0:c0 + height_cols]
        return canvas.reshape((r1 - r0) // block, block, (c1 - c0) // block, block)

    sample = np.where(mask & np.isfinite(heights) & (heights > 0.05 * upm), heights, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        coarse = np.nanmedian(reduce(sample, np.nan), axis=(1, 3))
    coarse_inside = reduce(inside.astype(float), 0.0).mean(axis=(1, 3)) >= 0.5
    measured = np.isfinite(coarse) & coarse_inside
    cell_area = (block * grid.pixel_size) ** 2
    min_cells = max(1, int(math.ceil(settings.min_part_area_m2 * upm * upm / cell_area)))
    if int(measured.sum()) < min_cells * 2:
        return []
    # Holes of the sparse scan take the nearest measured height; a 3x3 median removes isolated depth spikes.
    nearest = ndimage.distance_transform_edt(~measured, return_distances=False, return_indices=True)
    filled = coarse[tuple(nearest)]
    filled = ndimage.median_filter(filled, size=3, mode="nearest")
    centres = _level_centres(filled[coarse_inside] / upm, settings, min_cells)
    if not centres:
        return []
    levels = np.asarray(centres) * upm
    labels = np.argmin(np.abs(filled[..., None] - levels[None, None, :]), axis=2)
    labels = np.where(coarse_inside, labels, -1)
    for _ in range(3):
        # Majority filter, then give fragments smaller than a part to the level they touch most.
        counts = np.stack([ndimage.uniform_filter((labels == index).astype(float), size=3, mode="nearest") for index in range(len(levels))], axis=2)
        labels = np.where(coarse_inside, np.argmax(counts, axis=2), -1)
        changed = False
        for index in range(len(levels)):
            count, comps = cv2.connectedComponents((labels == index).astype(np.uint8), connectivity=4)
            for comp in range(1, count):
                region = comps == comp
                if int(region.sum()) >= min_cells:
                    continue
                ring = ndimage.binary_dilation(region, iterations=1) & ~region & coarse_inside
                neighbours = labels[ring]
                neighbours = neighbours[neighbours != index]
                if len(neighbours):
                    labels[region] = int(np.bincount(neighbours).argmax())
                    changed = True
        if not changed:
            break
    min_area = settings.min_part_area_m2 * upm * upm
    parts: list[tuple[float, Polygon]] = []
    for index in range(len(levels)):
        coarse_region = labels == index
        if not coarse_region.any():
            continue
        full = np.zeros(heights.shape, dtype=bool)
        expanded = np.kron(coarse_region, np.ones((block, block), dtype=bool))[:height_rows, :height_cols]
        full[r0:r0 + height_rows, c0:c0 + height_cols] = expanded
        z_top = float(np.median(filled[coarse_region]) / upm)
        for polygon in mask_to_polygons(full, grid):
            clipped = polygon.intersection(outline).buffer(0)
            for piece in getattr(clipped, "geoms", [clipped]):
                if isinstance(piece, Polygon) and piece.area >= min_area:
                    parts.append((z_top, Polygon(piece.exterior).simplify(0.01 * upm)))
    if len(parts) < 2 or len({round(z, 2) for z, _ in parts}) < 2 or len(parts) > settings.max_parts:
        return []
    parts.sort(key=lambda item: item[0])
    # The lowest level also carries cells the scan did not see, so nothing of the footprint is lost.
    higher = unary_union([polygon for _, polygon in parts[1:]]).buffer(0)
    base = outline.difference(higher).buffer(0)
    base_pieces = [piece for piece in getattr(base, "geoms", [base]) if isinstance(piece, Polygon) and piece.area >= 0.5 * min_area]
    output = [{"footprint": [[float(x), float(y)] for x, y in list(Polygon(piece.exterior).exterior.coords)[:-1]], "z_max_m": parts[0][0]} for piece in base_pieces]
    for z_top, polygon in parts[1:]:
        output.append({"footprint": [[float(x), float(y)] for x, y in list(polygon.exterior.coords)[:-1]], "z_max_m": z_top})
    return output if 2 <= len(output) <= settings.max_parts else []


def _make_instance(mask: np.ndarray, label: str | None, classes: np.ndarray, hm: np.ndarray, grid: RasterGrid, *, instance_id: int = 0) -> ObjectInstance | None:
    polygons = mask_to_polygons(mask, grid)
    if not polygons:
        return None
    poly = max(polygons, key=lambda item: item.area)
    code = _dominant(classes[mask])
    height = float(np.percentile(hm[mask], 90))
    return ObjectInstance(mask, label, code, label is not None, height, float(poly.area / max(poly.convex_hull.area, 1e-9)), instance_id=int(instance_id))


def _dominant_instance_id(mask: np.ndarray, instance_raster: np.ndarray | None) -> int:
    if instance_raster is None:
        return 0
    values = np.asarray(instance_raster[mask], dtype=np.uint32)
    values = values[values > 0]
    if len(values) == 0:
        return 0
    ids, counts = np.unique(values, return_counts=True)
    return int(ids[np.flatnonzero(counts == counts.max())[0]])


def _label_for_region(region: np.ndarray, fine_classes: np.ndarray, fine_confidence: np.ndarray, fine_names: list[str]) -> str | None:
    eligible = fine_classes[region & (fine_confidence >= 0.4)]
    eligible = eligible[(eligible > 0) & (eligible <= len(fine_names))]
    if len(eligible) == 0:
        return None
    fine_id = _dominant(eligible)
    return _canonical_instance_label(fine_names[fine_id - 1])


def _is_noise(mask: np.ndarray, raw: np.ndarray, classes: np.ndarray, hm: np.ndarray, height: float, solidity: float) -> bool:
    values = hm[mask]
    median = float(np.median(values))
    robust_std = 1.4826 * float(np.median(np.abs(values - median)))
    return (
        float(np.mean(np.isnan(raw[mask]))) > 0.30
        or float(np.mean(classes[mask] == 0)) > 0.50
        or (robust_std > 0.35 * max(height, 1e-9) and solidity < 0.5)
    )


def _has_floor_contact(mask: np.ndarray, free_floor: np.ndarray | None, wall_support: np.ndarray | None, pixel: float, units_per_metre: float) -> bool:
    if free_floor is None:
        return True
    radius = max(1, int(round(0.10 * units_per_metre / pixel)))
    ring = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))).astype(bool) & ~mask
    if wall_support is not None:
        ring &= ~wall_support
    return bool(ring.any()) and float(np.mean(free_floor[ring])) >= 0.25


def _wall_mask(shape: tuple[int, int], grid: RasterGrid, lines: Iterable[LineString], width: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for line in lines:
        points = np.asarray([grid.local_to_pixel(float(x), float(y))[::-1] for x, y in line.coords], dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(mask, [points], False, 1, thickness=max(1, int(round(width / grid.pixel_size))))
    return mask.astype(bool)


def _polygon_mask(polygon: Polygon, grid: RasterGrid) -> np.ndarray:
    mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
    exterior = np.asarray([grid.local_to_pixel(float(x), float(y))[::-1] for x, y in polygon.exterior.coords], dtype=np.int32)
    cv2.fillPoly(mask, [exterior], 1)
    return mask.astype(bool)


def _dominant(values: np.ndarray) -> int:
    return int(np.bincount(np.asarray(values, dtype=np.uint8), minlength=256).argmax()) if len(values) else 0


def wall_axis_rectangle(mask: np.ndarray, grid: RasterGrid, theta: float) -> Polygon | None:
    polygons = mask_to_polygons(np.asarray(mask, dtype=bool), grid)
    return _wall_rectangle(max(polygons, key=lambda item: item.area), theta) if polygons else None


def _largest_mask_rectangle(mask: np.ndarray, grid: RasterGrid) -> Polygon | None:
    """Return the largest raster-supported rectangle, excluding a small appendage."""
    heights = np.zeros(mask.shape[1], dtype=np.int32)
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    for row in range(mask.shape[0]):
        heights = np.where(mask[row], heights + 1, 0)
        stack: list[int] = []
        for column in range(mask.shape[1] + 1):
            current = int(heights[column]) if column < mask.shape[1] else 0
            while stack and current < int(heights[stack[-1]]):
                height = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                area = height * (column - left)
                if area > best_area:
                    best_area = area
                    best = (row - height + 1, row + 1, left, column)
            stack.append(column)
    if best is None or best_area < 4:
        return None
    r0, r1, c0, c1 = best
    corners = [grid.pixel_to_local(float(row), float(column)) for row, column in ((r0, c0), (r0, c1), (r1, c1), (r1, c0))]
    polygon = Polygon(corners)
    return polygon if polygon.is_valid and polygon.area > 0.0 else None


def _wall_rectangle(poly: Polygon, theta: float) -> Polygon:
    rotated = affinity.rotate(poly, -math.degrees(theta), origin=(0, 0))
    samples = _sample_ring(rotated, 400)
    # The nominal 3/97 quantiles are evaluated on the contour's straight-edge
    # support rather than its rounded morphology corners.  Using the retained
    # outer support keeps a 1 px raster jitter from shrinking a true table.
    x0, x1 = np.percentile(samples[:, 0], (3.0, 97.0))
    y0, y1 = np.percentile(samples[:, 1], (3.0, 97.0))
    return affinity.rotate(Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1))), math.degrees(theta), origin=(0, 0))


def _circle(poly: Polygon) -> tuple[Polygon, float]:
    points = _sample_ring(poly, 300)
    solution = np.linalg.lstsq(np.c_[2 * points, np.ones(len(points))], (points**2).sum(axis=1), rcond=None)[0]
    centre = solution[:2]
    radius = math.sqrt(max(float(solution[2] + centre @ centre), 1e-12))
    return Point(centre).buffer(radius, quad_segs=48), radius


def _ellipse(poly: Polygon) -> Polygon | None:
    points = _sample_ring(poly, 300).astype(np.float32)
    if len(points) < 5:
        return None
    (cx, cy), (first, second), angle = cv2.fitEllipse(points.reshape(-1, 1, 2))
    if min(first, second) <= 1e-9:
        return None
    unit = Point(float(cx), float(cy)).buffer(1.0, quad_segs=48)
    return affinity.rotate(affinity.scale(unit, first / 2.0, second / 2.0, origin=(cx, cy)), float(angle), origin=(cx, cy))


def _regularised(poly: Polygon, theta: float, units_per_metre: float, tol: float | None = None) -> Polygon | None:
    tol = 0.03 * units_per_metre if tol is None else tol
    vertices = np.asarray(poly.simplify(tol, preserve_topology=True).exterior.coords[:-1], dtype=float)
    if len(vertices) < 3:
        return None
    if not Polygon(vertices).exterior.is_ccw:
        vertices = vertices[::-1]
    samples = _sample_ring(poly, max(200, int(poly.exterior.length / max(0.005 * units_per_metre, 1e-6))))
    axes = (np.array((math.cos(theta), math.sin(theta))), np.array((-math.sin(theta), math.cos(theta))))
    lines: list[tuple[np.ndarray, np.ndarray, float]] = []
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        direction = second - first
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            continue
        direction /= length
        nearest = min(axes, key=lambda axis: abs(abs(float(direction @ axis)) - 1.0))
        if math.degrees(math.acos(min(1.0, abs(float(direction @ nearest))))) <= 12.0:
            direction = nearest if float(direction @ nearest) >= 0.0 else -nearest
        normal = np.array((-direction[1], direction[0]))
        relative = samples - first
        along, across = relative @ direction, relative @ normal
        nearby = (along >= -0.1 * length) & (along <= 1.1 * length) & (np.abs(across) <= 2.0 * tol)
        offset = float(np.median(across[nearby])) if int(nearby.sum()) >= 5 else 0.0
        lines.append((first + normal * offset, direction, length))
    lines = _merge_collinear(lines, tol)
    if len(lines) < 3:
        return None
    points = _intersections_with_connectors(lines)
    candidate = _single_polygon(Polygon(points)) if len(points) >= 3 else None
    if candidate is None:
        return None
    candidate = _drop_short_edges(candidate, 0.04 * units_per_metre)
    return candidate


def _merge_collinear(lines: list[tuple[np.ndarray, np.ndarray, float]], tol: float) -> list[tuple[np.ndarray, np.ndarray, float]]:
    merged: list[tuple[np.ndarray, np.ndarray, float]] = []
    for line in lines:
        if merged:
            point, direction, length = merged[-1]
            current, current_direction, current_length = line
            if abs(float(np.cross(direction, current_direction))) <= 1e-6 and float(direction @ current_direction) > 0.0:
                normal = np.array((-direction[1], direction[0]))
                offset = float((current - point) @ normal)
                if abs(offset) <= tol:
                    merged[-1] = (point + normal * (offset * current_length / (length + current_length)), direction, length + current_length)
                    continue
        merged.append(line)
    return merged


def _intersections_with_connectors(lines: list[tuple[np.ndarray, np.ndarray, float]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index, (current, direction, _length) in enumerate(lines):
        previous, previous_direction, _previous_length = lines[index - 1]
        cross = float(np.cross(previous_direction, direction))
        if abs(cross) <= 1e-6:
            # The DP path skipped a perpendicular step.  Insert its connector at
            # the midpoint along the two parallel lines rather than collapsing it.
            tangent = previous_direction if float(previous_direction @ direction) >= 0.0 else -previous_direction
            along = 0.5 * (float(previous @ tangent) + float(current @ tangent))
            first = previous + tangent * (along - float(previous @ tangent))
            second = current + tangent * (along - float(current @ tangent))
            points.extend((tuple(first), tuple(second)))
            continue
        factor = float(np.cross(current - previous, direction) / cross)
        points.append(tuple(previous + previous_direction * factor))
    return points


def _drop_short_edges(poly: Polygon, minimum: float) -> Polygon | None:
    points = list(poly.exterior.coords[:-1])
    changed = True
    while changed and len(points) >= 3:
        changed = False
        for index in range(len(points)):
            if math.dist(points[index], points[(index + 1) % len(points)]) < minimum:
                del points[(index + 1) % len(points)]
                changed = True
                break
    if len(points) < 3:  # every edge was shorter than ``minimum``: no polygon is left (the caller keeps its fallback)
        return None
    return _single_polygon(Polygon(points))


def _snap_edges(shape: Polygon, hm: np.ndarray, grid: RasterGrid, band: float) -> Polygon:
    gradient = cv2.magnitude(cv2.Sobel(hm, cv2.CV_32F, 1, 0) / 8.0, cv2.Sobel(hm, cv2.CV_32F, 0, 1) / 8.0)
    points = np.asarray(shape.exterior.coords[:-1], dtype=float)
    shifted: list[tuple[np.ndarray, np.ndarray]] = []
    for first, second in zip(points, np.roll(points, -1, axis=0), strict=True):
        direction = second - first
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            return shape
        direction /= length
        normal = np.array((-direction[1], direction[0]))
        best_offset, best_score = 0.0, -1.0
        for offset in np.arange(-band, band + 1e-9, max(grid.pixel_size * 0.5, 0.005)):
            samples = first + direction * np.linspace(0.1 * length, 0.9 * length, 15)[:, None] + normal * offset
            score = 0.0
            for x, y in samples:
                row, col = grid.local_to_pixel(float(x), float(y))
                rr, cc = int(round(row)), int(round(col))
                if 0 <= rr < gradient.shape[0] and 0 <= cc < gradient.shape[1]:
                    score += float(gradient[rr, cc])
            if score > best_score:
                best_offset, best_score = float(offset), score
        shifted.append((first + normal * best_offset, direction))
    rebuilt = _intersections_with_connectors([(point, direction, 1.0) for point, direction in shifted])
    candidate = _single_polygon(Polygon(rebuilt)) if len(rebuilt) >= 3 else None
    return candidate if candidate is not None and candidate.area > 0.5 * shape.area else shape


def _scores(shape: Polygon, measured: Polygon) -> tuple[float, float]:
    samples = _sample_ring(measured, 240)
    dev = float(np.percentile([shape.exterior.distance(Point(point)) for point in samples], 90))
    iou = float(shape.intersection(measured).area / max(shape.union(measured).area, 1e-9))
    return dev, iou


def _sample_ring(poly: Polygon, count: int) -> np.ndarray:
    return np.asarray([poly.exterior.interpolate(distance).coords[0] for distance in np.linspace(0.0, poly.exterior.length, count, endpoint=False)])


def _single_polygon(geometry: object) -> Polygon | None:
    if isinstance(geometry, Polygon):
        return geometry if geometry.is_valid and not geometry.is_empty else None
    if hasattr(geometry, "geoms"):
        polygons = [item for item in geometry.geoms if isinstance(item, Polygon) and not item.is_empty]
        return max(polygons, key=lambda item: item.area) if polygons else None
    return None


def _lines(geometry: object) -> list[LineString]:
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [line for item in geometry.geoms for line in _lines(item)]
    return []


# ---------------------------------------------------------------------------
# Freestanding kitchen island
# ---------------------------------------------------------------------------

ISLAND_LABELS = frozenset({"kitchen counter", "kitchen cabinet", "cooktop", "stove", "sink", "kitchen sink"})
ISLAND_DEPTHS_M = (0.90, 1.20)


@dataclass(frozen=True, slots=True)
class IslandSettings:
    """Thresholds (metres, scaled by units per metre at use) for the kitchen-island rule."""

    min_height_m: float = 0.85
    max_height_m: float = 0.95
    min_depth_m: float = 0.80
    max_depth_m: float = 1.20
    min_length_m: float = 1.20
    wall_clearance_m: float = 0.15
    min_rectangularity: float = 0.80
    axis_tolerance_deg: float = 12.0
    zone_radius_m: float = 3.0


@dataclass(slots=True)
class KitchenIsland:
    """A clean countertop rectangle of a freestanding kitchen island."""

    polygon: Polygon
    angle_rad: float
    length_m: float
    depth_m: float
    reason: str


def detect_kitchen_island(
    instance: ObjectInstance,
    grid: RasterGrid,
    theta: float,
    units_per_metre: float,
    wall_lines: Iterable[LineString],
    kitchen_zone: Any | None,
    settings: IslandSettings = IslandSettings(),
) -> KitchenIsland | None:
    """Recognise a freestanding kitchen island without needing a wall behind it.

    Geometry: rectangular footprint, top height 0.85-0.95 m, depth 0.80-1.20 m,
    length >= 1.20 m, at least ``wall_clearance_m`` away from every wall line.
    Semantics: a kitchen fine label (kitchen counter / cabinet, cooktop, stove,
    sink), or no fine label at all and a footprint inside ``kitchen_zone``.  Any
    other fine label (sofa, table, bed ...) rules the object out.  The output
    rectangle is aligned with the dominant room axis ``theta`` when the measured
    long axis is within ``axis_tolerance_deg`` of it, with its depth snapped to
    the standard 0.90 / 1.20 m island depth.
    """
    upm = max(float(units_per_metre), 1e-9)
    label = str(instance.label or "").strip().lower()
    if instance.has_fine_label and label not in ISLAND_LABELS:
        return None
    if not settings.min_height_m <= instance.height / upm <= settings.max_height_m:
        return None
    polygons = mask_to_polygons(np.asarray(instance.mask, dtype=bool), grid)
    if not polygons:
        return None
    polygon = max(polygons, key=lambda item: item.area)
    rectangle = polygon.minimum_rotated_rectangle
    if rectangle.is_empty or polygon.area / max(rectangle.area, 1e-12) < settings.min_rectangularity:
        return None
    corners = np.asarray(rectangle.exterior.coords[:4], dtype=float)
    edges = [corners[(index + 1) % 4] - corners[index] for index in range(4)]
    lengths = [float(np.hypot(*edge)) for edge in edges]
    long_index = int(np.argmax(lengths[:2]))
    length, depth = lengths[long_index] / upm, lengths[1 - long_index] / upm
    if length < settings.min_length_m or not settings.min_depth_m <= depth <= settings.max_depth_m:
        return None
    measured_angle = math.atan2(float(edges[long_index][1]), float(edges[long_index][0]))
    angle = measured_angle
    tolerance = math.radians(settings.axis_tolerance_deg)
    for quarter in range(4):
        candidate = theta + quarter * math.pi / 2.0
        difference = (measured_angle - candidate + math.pi) % (2.0 * math.pi) - math.pi
        if abs(difference) <= tolerance or abs(abs(difference) - math.pi) <= tolerance:
            angle = candidate
            break
    # Measured extents in the chosen frame, then clean snapping.
    rotated = affinity.rotate(polygon, -math.degrees(angle), origin=(0.0, 0.0))
    x0, y0, x1, y1 = rotated.bounds
    if (x1 - x0) < (y1 - y0):
        angle += math.pi / 2.0
        rotated = affinity.rotate(polygon, -math.degrees(angle), origin=(0.0, 0.0))
        x0, y0, x1, y1 = rotated.bounds
    length = (x1 - x0) / upm
    depth = (y1 - y0) / upm
    snapped = min(ISLAND_DEPTHS_M, key=lambda value: abs(value - depth))
    if abs(snapped - depth) <= 0.10:
        depth = snapped
    length = max(settings.min_length_m, round(length / 0.05) * 0.05)
    centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    clean = box(centre[0] - length * upm / 2.0, centre[1] - depth * upm / 2.0, centre[0] + length * upm / 2.0, centre[1] + depth * upm / 2.0)
    clean = affinity.rotate(clean, math.degrees(angle), origin=(0.0, 0.0))
    for wall in wall_lines:
        if clean.distance(wall) < settings.wall_clearance_m * upm:
            return None
    if instance.has_fine_label:
        reason = f"label:{label}"
    elif kitchen_zone is not None and not kitchen_zone.is_empty and kitchen_zone.contains(clean.centroid):
        reason = "kitchen_zone"
    else:
        return None
    return KitchenIsland(clean, angle % math.pi, length, depth, reason)
