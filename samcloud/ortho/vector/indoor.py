"""Indoor wall, opening, stair, fixture, and furniture vector extraction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math

import cv2
import numpy as np
from scipy import ndimage
from shapely import affinity
from shapely.geometry import LineString, Polygon, box
from skimage.measure import find_contours
from skimage.morphology import skeletonize

from .raster_grid import RasterGrid, mask_to_polygons
from .regularize import regularize_polygon
from .symbols import CATALOGUE, CatalogueEntry


@dataclass(frozen=True, slots=True)
class Opening:
    """A measured interruption in an indoor wall."""

    kind: str
    center: tuple[float, float]
    width: float
    axis_angle: float
    wall_thickness: float
    evidence: str = "gap"
    class_evidence: float = 0.0
    lintel_evidence: float = 0.0
    leaf_angle_deg: float | None = None
    leaf_direction_deg: float | None = None
    leaf_mode: str | None = None
    hinge_side: str | None = None
    swing_side: int | None = None
    leaf_detected: bool | None = None
    # Measured by the 3D refinement (samcloud.bim.openings3d); None / empty for 2D-only openings.
    sill_m: float | None = None
    head_m: float | None = None
    height_source: str | None = None
    mullion_u: tuple[float, ...] = ()  # metres from the left edge along the axis
    glazed: bool = False
    confidence: float | None = None


def wall_faces_from_section(
    section_mask: np.ndarray,
    grid: RasterGrid,
    *,
    min_length: float,
    angle_tolerance_deg: float,
    max_face_thickness: float,
) -> list[LineString]:
    """Extract regularised wall-face centre lines from a thin section raster.

    Interior scans normally observe only one face of a wall.  Hough segments on
    a skeleton are deliberately used here instead of area polygons, because a
    one-pixel face is meaningful geometry at a 1 cm orthophoto resolution.
    """
    mask = np.asarray(section_mask, dtype=bool)
    if mask.shape != (grid.height, grid.width):
        raise ValueError("section_mask shape must match RasterGrid")
    skeleton = skeletonize(mask).astype(np.uint8) * 255
    threshold = max(2, int(round(min_length / grid.pixel_size)))
    raw = cv2.HoughLinesP(
        skeleton, 1, np.pi / 360.0, threshold=threshold, minLineLength=threshold,
        maxLineGap=max(1, int(round(0.25 / grid.pixel_size))),
    )
    if raw is None:
        return []
    candidates: list[LineString] = []
    for x0, y0, x1, y1 in raw[:, 0, :]:
        line = LineString((grid.pixel_to_local(float(y0), float(x0)), grid.pixel_to_local(float(y1), float(x1))))
        if line.length >= min_length:
            candidates.append(line)
    snapped = _snap_faces_to_axes(candidates, angle_tolerance_deg)
    return _merge_face_lines(snapped, angle_tolerance_deg, 0.08, 1.3)


def walls_from_faces(
    faces: list[LineString],
    free_space_mask: np.ndarray,
    grid: RasterGrid,
    *,
    max_pair_distance: float,
    min_overlap: float,
    default_thickness: float,
) -> list[Polygon]:
    """Turn parallel wall faces into measured or room-side-aware wall polygons."""
    free = np.asarray(free_space_mask, dtype=bool)
    if free.shape != (grid.height, grid.width):
        raise ValueError("free_space_mask shape must match RasterGrid")
    used: set[int] = set()
    walls: list[Polygon] = []
    for index, face in enumerate(faces):
        if index in used or face.length <= 0.0:
            continue
        axis = _line_axis(face)
        normal = np.asarray((-axis[1], axis[0]))
        origin = np.asarray(face.coords[0], dtype=float)
        best: tuple[float, int] | None = None
        for other_index in range(index + 1, len(faces)):
            if other_index in used:
                continue
            other = faces[other_index]
            if _axis_angle_distance(axis, _line_axis(other)) > math.radians(8.0):
                continue
            distance = abs(float(np.dot(np.asarray(other.centroid.coords[0]) - origin, normal)))
            if not 0.05 <= distance <= max_pair_distance:
                continue
            overlap = _line_overlap(face, other, axis)
            if overlap < min_overlap * min(face.length, other.length):
                continue
            if best is None or distance < best[0]:
                best = (distance, other_index)
        if best is not None:
            other = faces[best[1]]
            wall = _polygon_between_faces(face, other, axis)
            if wall is not None and wall.area > 0.0:
                walls.append(wall)
                used.update((index, best[1]))
                continue
        side = _outside_normal(face, normal, free, grid)
        coords = np.asarray(face.coords, dtype=float)
        shifted = coords + side * default_thickness
        wall = Polygon([tuple(coords[0]), tuple(coords[-1]), tuple(shifted[-1]), tuple(shifted[0])])
        if wall.is_valid and wall.area > 0.0:
            walls.append(wall)
            used.add(index)
    return walls


def wall_polygons_from_section(
    section_mask: np.ndarray,
    grid: RasterGrid,
    *,
    min_thickness: float,
    max_thickness: float,
    min_length: float,
    angle_tolerance_deg: float,
) -> list[Polygon]:
    """Clean section walls, bridge practical opening gaps, and regularise outlines."""
    mask = np.asarray(section_mask, dtype=bool)
    if mask.shape != (grid.height, grid.width):
        raise ValueError("section_mask shape must match RasterGrid")
    radius = max(1, int(round(min_thickness / (2.0 * grid.pixel_size))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    cleaned = _bridge_axis_gaps(cleaned, grid, max_gap=2.5)
    output: list[Polygon] = []
    for polygon in mask_to_polygons(cleaned, grid):
        if polygon.area <= 0.0 or polygon.length < min_length * 2.0:
            continue
        regular = polygon if polygon.interiors else regularize_polygon(polygon, angle_tolerance_deg=angle_tolerance_deg, min_edge_length=min_thickness * 0.5)
        thickness = _wall_thickness(mask, polygon, grid)
        if min_thickness <= thickness <= max_thickness and regular.length >= min_length * 2.0:
            output.append(regular)
    return output


def detect_openings(
    section_mask: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
    walls: list[Polygon],
    grid: RasterGrid,
    *,
    door_mask: np.ndarray | None = None,
    door_min_width: float = 0.6,
    door_max_width: float = 2.5,
    window_min_width: float = 0.4,
    window_max_width: float = 4.0,
) -> list[Opening]:
    """Detect section gaps inside bridged wall polygons and classify their masks."""
    section = np.asarray(section_mask, dtype=bool)
    low = np.asarray(low_mask, dtype=bool)
    high = np.asarray(high_mask, dtype=bool)
    if any(array.shape != (grid.height, grid.width) for array in (section, low, high)):
        raise ValueError("all masks must match RasterGrid")
    doors = None if door_mask is None else np.asarray(door_mask, dtype=bool)
    if doors is not None and doors.shape != section.shape:
        raise ValueError("door_mask must match RasterGrid")
    openings: list[Opening] = []
    for wall in walls:
        candidate = _polygon_mask(wall, grid) & _bridge_axis_gaps(section, grid, max_gap=2.5)
        gaps = candidate & ~section
        _count, components, stats, _centroids = cv2.connectedComponentsWithStats(gaps.astype(np.uint8), connectivity=8)
        thickness = _wall_thickness(section, wall, grid)
        for identifier, component_slice in enumerate(ndimage.find_objects(components), start=1):
            if component_slice is None:
                continue
            if stats[identifier, cv2.CC_STAT_AREA] == 0:
                continue
            component = components[component_slice] == identifier
            rows, cols = np.nonzero(component)
            rows = rows + component_slice[0].start
            cols = cols + component_slice[1].start
            points = np.asarray([grid.pixel_to_local(row, col) for row, col in zip(rows, cols, strict=True)])
            angle = _component_axis_angle(points)
            axis = np.asarray((math.cos(angle), math.sin(angle)))
            width = float(np.ptp(points @ axis)) + grid.pixel_size
            has_high = bool((high[component_slice] & component).any())
            has_low = bool((low[component_slice] & component).any())
            if has_high and has_low and window_min_width <= width <= window_max_width:
                kind = "WINDOW"
            elif has_high and not has_low and door_min_width <= width <= door_max_width:
                kind = "DOOR"
            elif not has_high and not has_low and door_min_width <= width <= door_max_width:
                kind = "passage"
            else:
                continue
            center = tuple(float(value) for value in points.mean(axis=0))
            openings.append(Opening(kind, center, width, angle, thickness))
    if doors is not None and doors.any():
        _count, components, _stats, _centres = cv2.connectedComponentsWithStats(doors.astype(np.uint8), connectivity=8)
        for identifier, component_slice in enumerate(ndimage.find_objects(components), start=1):
            if component_slice is None:
                continue
            component = components[component_slice] == identifier
            rows, cols = np.nonzero(component)
            rows = rows + component_slice[0].start
            cols = cols + component_slice[1].start
            points = np.asarray([grid.pixel_to_local(row, col) for row, col in zip(rows, cols, strict=True)])
            if len(points) < 2:
                continue
            angle = _component_axis_angle(points)
            axis = np.asarray((math.cos(angle), math.sin(angle)))
            width = float(np.ptp(points @ axis)) + grid.pixel_size
            if not door_min_width <= width <= door_max_width:
                continue
            center = tuple(float(value) for value in points.mean(axis=0))
            nearby = min(walls, key=lambda wall: wall.distance(LineString(points))) if walls else None
            if nearby is not None and nearby.distance(LineString(points)) <= max(0.15, grid.pixel_size * 3.0):
                openings.append(Opening("DOOR", center, width, angle, max(grid.pixel_size * 2.0, _wall_thickness(section, nearby, grid))))
    return _deduplicate_openings(openings)


def cut_openings(walls: list[Polygon], openings: list[Opening]) -> list[Polygon]:
    """Subtract rectangular wall cuts at every detected opening."""
    output: list[Polygon] = []
    for wall in walls:
        cut = None
        for opening in openings:
            shape = affinity.rotate(
                box(-opening.width / 2.0 - 0.02, -opening.wall_thickness, opening.width / 2.0 + 0.02, opening.wall_thickness),
                math.degrees(opening.axis_angle),
                origin=(0.0, 0.0),
            )
            shape = affinity.translate(shape, opening.center[0], opening.center[1])
            cut = shape if cut is None else cut.union(shape)
        difference = wall.difference(cut) if cut is not None else wall
        if isinstance(difference, Polygon):
            output.append(difference)
        else:
            output.extend(item for item in getattr(difference, "geoms", ()) if isinstance(item, Polygon))
    return output


@lru_cache(maxsize=1)
def _default_heights() -> tuple[float, float, float]:
    """Metric defaults (window sill, window head, door head) shared with the BIM export."""
    from samcloud.bim.plan_model import load_bim_export_config

    defaults = load_bim_export_config()["defaults"]
    return float(defaults["window_sill_m"]), float(defaults["window_head_m"]), float(defaults["door_head_m"])


def _head_suffix(opening: Opening) -> str:
    return f" H {opening.head_m:.2f}" if opening.head_m is not None else ""


def _window_attributes(opening: Opening) -> dict[str, str]:
    """LABEL plus BRH (sill) / STH (head) in metres and MULLIONS; unmeasured windows carry the BIM defaults."""
    default_sill, default_head, _ = _default_heights()
    sill = default_sill if opening.sill_m is None else float(opening.sill_m)
    head = default_head if opening.head_m is None else float(opening.head_m)
    return {"LABEL": f"Window {opening.width:.2f} m", "BRH": f"{sill:.2f}", "STH": f"{head:.2f}", "MULLIONS": str(len(opening.mullion_u))}


def _door_attributes(opening: Opening) -> dict[str, str]:
    """LABEL plus H (head height, m) and GLAZED=1 for doors with a glass leaf."""
    _, _, default_head = _default_heights()
    head = default_head if opening.head_m is None else float(opening.head_m)
    attributes = {"LABEL": f"Door {opening.width:.2f} m", "H": f"{head:.2f}"}
    if opening.glazed:
        attributes["GLAZED"] = "1"
    return attributes


def opening_entities(
    openings: list[Opening], free_space_mask: np.ndarray | None = None, *, grid: RasterGrid | None = None
) -> list[dict[str, object]]:
    """Create opening geometry, drawing a door symbol only for observed leaves.

    The DOOR block has its closed leaf along positive x and open leaf along
    positive y.  Its x scale therefore always points from the hinge to the
    other jamb; a right hinge turns the block by 180 degrees, so the y scale
    carries the swing sign times -1 there: the open leaf always stands on the
    ``swing_side`` side of the opening normal.
    """
    if free_space_mask is not None:
        if grid is None:
            raise ValueError("grid is required when free_space_mask is supplied")
        if np.asarray(free_space_mask, dtype=bool).shape != (grid.height, grid.width):
            raise ValueError("free_space_mask must match RasterGrid")
    entities: list[dict[str, object]] = []
    for opening in openings:
        axis = np.asarray((math.cos(opening.axis_angle), math.sin(opening.axis_angle)))
        if opening.kind in {"DOOR", "passage"}:
            if opening.kind == "passage" or opening.leaf_detected is False:
                normal = np.asarray((-axis[1], axis[0]))
                half = axis * (opening.width / 2.0)
                thickness = max(opening.wall_thickness, 0.01)
                centre_marker = [float(opening.center[0]), float(opening.center[1])]
                entities.extend([
                    {"type": "line", "start": (np.asarray(opening.center) - half - normal * thickness / 2.0).tolist(), "end": (np.asarray(opening.center) - half + normal * thickness / 2.0).tolist(), "_jamb": True, "_opening_center": centre_marker},
                    {"type": "line", "start": (np.asarray(opening.center) + half - normal * thickness / 2.0).tolist(), "end": (np.asarray(opening.center) + half + normal * thickness / 2.0).tolist(), "_jamb": True, "_opening_center": centre_marker},
                    {"type": "text", "text": f"{'Tür' if opening.kind == 'DOOR' else 'Durchgang'} {opening.width:.2f} m" + _head_suffix(opening), "point": (np.asarray(opening.center) + normal * 0.15 * max(opening.wall_thickness / 0.25, 1.0)).tolist(), "height": 0.10, "rotation": math.degrees(opening.axis_angle), "_opening_label": True, "_opening_center": centre_marker},
                ])
                continue
            hinge_side = opening.hinge_side or "left"
            swing_side = opening.swing_side
            if swing_side is None and free_space_mask is not None and grid is not None:
                positive, negative = _free_space_by_normal(opening, np.asarray(free_space_mask, dtype=bool), grid)
                swing_side = 1 if positive >= negative else -1
            swing_side = 1 if swing_side is None else swing_side
            direction = axis if hinge_side == "left" else -axis
            hinge = np.asarray(opening.center) - axis * (opening.width / 2.0) if hinge_side == "left" else np.asarray(opening.center) + axis * (opening.width / 2.0)
            normal = np.asarray((-axis[1], axis[0]))
            half = axis * (opening.width / 2.0)
            thickness = max(opening.wall_thickness, 0.01)
            centre_marker = [float(opening.center[0]), float(opening.center[1])]
            entities.append({"type": "insert", "name": "DOOR", "point": hinge.tolist(), "rotation": math.degrees(math.atan2(direction[1], direction[0])), "xscale": opening.width, "yscale": float(swing_side) * opening.width * (1.0 if hinge_side == "left" else -1.0), "attributes": _door_attributes(opening), "_leaf_angle_deg": opening.leaf_angle_deg, "_leaf_mode": opening.leaf_mode, "_hinge_side": hinge_side, "_swing_side": swing_side, "_opening_center": list(opening.center), "_opening_kind": opening.kind, "_opening_evidence": opening.evidence})
            # The opening's jambs stay drawn beside the swing symbol (CAD convention: the wall ends at the opening are marked).
            entities.extend([
                {"type": "line", "start": (np.asarray(opening.center) - half - normal * thickness / 2.0).tolist(), "end": (np.asarray(opening.center) - half + normal * thickness / 2.0).tolist(), "_jamb": True, "_opening_center": centre_marker},
                {"type": "line", "start": (np.asarray(opening.center) + half - normal * thickness / 2.0).tolist(), "end": (np.asarray(opening.center) + half + normal * thickness / 2.0).tolist(), "_jamb": True, "_opening_center": centre_marker},
            ])
        elif opening.kind == "WINDOW":
            normal = np.asarray((-axis[1], axis[0]))
            exterior_sign = 1.0
            if free_space_mask is not None and grid is not None:
                positive, negative = _free_space_by_normal(opening, np.asarray(free_space_mask, dtype=bool), grid)
                exterior_sign = -1.0 if positive >= negative else 1.0
            origin = np.asarray((opening.center[0] - opening.width / 2.0 * axis[0], opening.center[1] - opening.width / 2.0 * axis[1]))
            entities.append({"type": "insert", "name": "WINDOW", "point": origin.tolist(), "rotation": math.degrees(opening.axis_angle), "xscale": opening.width, "yscale": exterior_sign * opening.wall_thickness, "attributes": _window_attributes(opening), "_opening_center": list(opening.center), "_opening_kind": opening.kind, "_opening_evidence": opening.evidence})
            # Mullion bars (3D refinement): a line across the wall thickness at every measured position, drawn in the window layer.
            for position in opening.mullion_u:
                start = origin + axis * float(position)
                entities.append({"type": "line", "start": start.tolist(), "end": (start + normal * exterior_sign * opening.wall_thickness).tolist(), "_mullion": True, "_opening_center": list(opening.center), "_opening_kind": opening.kind})
    return entities


def stair_entities(stair_mask: np.ndarray, grid: RasterGrid, *, tread_depth: float = 0.28, min_area: float) -> list[dict[str, object]]:
    """Return tread lines, a travel arrow, and a Stairs text label per stair component."""
    mask = np.asarray(stair_mask, dtype=bool)
    output: list[dict[str, object]] = []
    for polygon in mask_to_polygons(mask, grid):
        if polygon.area < min_area:
            continue
        rectangle = polygon.minimum_rotated_rectangle
        coordinates = list(rectangle.exterior.coords)[:-1]
        edges = [(np.asarray(coordinates[(index + 1) % 4]) - np.asarray(point)) for index, point in enumerate(coordinates)]
        lengths = [float(np.hypot(*edge)) for edge in edges]
        long_index = int(np.argmax(lengths))
        origin = np.asarray(coordinates[long_index])
        long_axis = edges[long_index] / lengths[long_index]
        short_axis = edges[(long_index + 1) % 4] / lengths[(long_index + 1) % 4]
        long_length, short_length = lengths[long_index], lengths[(long_index + 1) % 4]
        for distance in np.arange(0.0, long_length + 1e-9, tread_depth):
            centre = origin + long_axis * min(distance, long_length)
            start, end = centre - short_axis * short_length / 2.0, centre + short_axis * short_length / 2.0
            output.append({"type": "line", "start": start.tolist(), "end": end.tolist(), "layer_role": "detail"})
        middle = origin + long_axis * (long_length * 0.5)
        arrow_end = middle + long_axis * (long_length * 0.3)
        arrow_left = arrow_end - long_axis * 0.18 + short_axis * 0.12
        arrow_right = arrow_end - long_axis * 0.18 - short_axis * 0.12
        output.append({"type": "polyline", "points": [middle.tolist(), arrow_end.tolist(), arrow_left.tolist(), arrow_end.tolist(), arrow_right.tolist()], "closed": False, "layer_role": "detail"})
        output.append({"type": "text", "text": "Stairs", "point": list(polygon.centroid.coords[0]), "height": 0.12, "rotation": math.degrees(math.atan2(long_axis[1], long_axis[0])), "layer_role": "label"})
    return output


def fit_block_insert(
    component_mask: np.ndarray,
    grid: RasterGrid,
    block: str,
    label: str,
    *,
    row_offset: int = 0,
    col_offset: int = 0,
    units_per_metre: float = 1.0,
) -> dict[str, object]:
    """Fit a block insert to OpenCV's minimum-area rectangle.

    Catalogue blocks (``CATALOGUE`` kinds such as BATHTUB) are never stretched:
    the nearest standard size is inserted at one uniform scale of
    ``units_per_metre`` (origin convention of the block).  Unit blocks without a
    catalogue (COLUMN, FURNITURE, ...) keep the measured long/short scales.

    ``minAreaRect`` measures point-centre extents and reports its angle in image
    coordinates (positive rows point down).  One pixel is added to each extent
    to convert centres to pixel edges; the longest transformed box edge supplies
    the local-metre rotation and the xscale direction.
    """
    mask = np.asarray(component_mask, dtype=bool)
    rows, cols = np.nonzero(mask)
    if not len(rows):
        raise ValueError("component_mask contains no foreground pixels")
    points = np.column_stack(((cols + col_offset).astype(np.float32), (rows + row_offset).astype(np.float32)))
    (col, row), (width_px, height_px), _angle = cv2.minAreaRect(points)
    corners = cv2.boxPoints(((col, row), (width_px, height_px), _angle))
    local_corners = np.asarray([grid.pixel_to_local(float(point[1]), float(point[0])) for point in corners])
    edges = [local_corners[(index + 1) % 4] - point for index, point in enumerate(local_corners)]
    lengths = [float(np.hypot(*edge)) for edge in edges]
    long_index = int(np.argmax(lengths))
    long_axis = edges[long_index]
    long_size = max(width_px, height_px) + 1.0
    short_size = min(width_px, height_px) + 1.0
    center = grid.pixel_to_local(float(row), float(col))
    measured = {"type": "insert", "name": str(block).upper(), "point": list(center), "rotation": math.degrees(math.atan2(long_axis[1], long_axis[0])) % 180.0, "xscale": float(long_size * grid.pixel_size), "yscale": float(short_size * grid.pixel_size), "attributes": {"LABEL": str(label)}}
    if str(block).upper() in CATALOGUE:
        uniform = centred_catalogue_insert(measured, str(block).upper(), units_per_metre)
        if uniform is not None:
            return uniform
    return measured


@dataclass(frozen=True, slots=True)
class CatalogueMatch:
    """The nearest standard size for a measured footprint (relative error, 0 = exact)."""

    entry: CatalogueEntry
    error: float


def match_catalogue(kinds: tuple[str, ...] | str, along_m: float, depth_m: float, *, tolerance: float | None = None) -> CatalogueMatch | None:
    """Pick the standard size nearest to the measured ``along`` (X) and ``depth`` (Y) extents in metres.

    The relative error is the larger of the two per-axis deviations.  With a
    ``tolerance`` a worse match returns None (the object is not that catalogue
    item); without one the nearest size always wins.
    """
    names = (kinds,) if isinstance(kinds, str) else tuple(kinds)
    best: CatalogueMatch | None = None
    for name in names:
        for entry in CATALOGUE[name]:
            error = max(abs(along_m - entry.width_m) / entry.width_m, abs(depth_m - entry.depth_m) / entry.depth_m)
            if best is None or error < best.error:
                best = CatalogueMatch(entry, error)
    if best is None or (tolerance is not None and best.error > tolerance):
        return None
    return best


def _uniform_insert(entry: CatalogueEntry, point: np.ndarray, x_axis: np.ndarray, units_per_metre: float, source: dict[str, object], error: float, snapped: bool) -> dict[str, object]:
    """Build an insert at uniform scale; ``point`` is the block origin and ``x_axis`` its +X direction."""
    scale = float(units_per_metre)
    return {
        "type": "insert", "name": entry.block, "point": [float(point[0]), float(point[1])],
        "rotation": math.degrees(math.atan2(float(x_axis[1]), float(x_axis[0]))) % 360.0,
        "xscale": scale, "yscale": scale, "attributes": dict(source.get("attributes", {})),  # type: ignore[call-overload]
        "_catalogue": {"kind": entry.kind, "width_m": entry.width_m, "depth_m": entry.depth_m, "layer": entry.layer, "wall_snapped": bool(snapped), "match_error": float(error)},
    }


def snap_insert_to_wall(
    measured: dict[str, object],
    kinds: tuple[str, ...] | str,
    wall_lines: list[LineString],
    units_per_metre: float,
    *,
    max_gap_m: float = 0.30,
    angle_tolerance_deg: float = 12.0,
    tolerance: float | None = None,
) -> dict[str, object] | None:
    """Snap a measured rectangular object flush onto the nearest wall.

    ``measured`` is a ``fit_block_insert`` style dict (centre ``point``, long axis
    ``rotation``, ``xscale`` = long and ``yscale`` = short extent in local units).
    The rear edge of the object must be within ``max_gap_m`` (metres) of the wall.

    Normal mounting: the block's mounting edge (origin) lies on the wall line, +X
    along the wall, +Y into free space.  When the object's long axis runs
    perpendicular to the wall (for example a bathtub with its short end against
    the wall) the block is rotated so that its long X axis follows the measured
    long axis and its end edge touches the wall ("end-on").  The better of the
    two orientations against the catalogue size wins, so the long side is only
    snapped to the wall when it is really there.  A fixture that is not parallel
    to the wall is rotated onto it using its projected extents.  Returns None
    when no wall is near enough or no standard size matches within ``tolerance``.
    """
    upm = max(float(units_per_metre), 1e-9)
    centre = np.asarray(measured["point"], dtype=float)
    long_size, short_size = float(measured["xscale"]), float(measured["yscale"])
    rotation = math.radians(float(measured["rotation"]))
    long_axis = np.asarray((math.cos(rotation), math.sin(rotation)))
    short_axis = np.asarray((-long_axis[1], long_axis[0]))
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for line in wall_lines:
        coords = np.asarray(line.coords, dtype=float)
        for first, second in zip(coords[:-1], coords[1:], strict=True):
            vector = second - first
            length = float(np.hypot(*vector))
            if length < 1e-9:
                continue
            tangent = vector / length
            along = float(np.clip(np.dot(centre - first, tangent), 0.0, length))
            distance = float(np.hypot(*(centre - (first + tangent * along))))
            if best is None or distance < best[0]:
                best = (distance, tangent, first)
    if best is None:
        return None
    tangent = best[1]
    perpendicular = np.asarray((-tangent[1], tangent[0]))
    cos_long, cos_short = abs(float(np.dot(tangent, long_axis))), abs(float(np.dot(tangent, short_axis)))
    limit = math.cos(math.radians(angle_tolerance_deg))
    if cos_long >= limit:
        along_size, depth_size = long_size, short_size
    elif cos_short >= limit:
        along_size, depth_size = short_size, long_size
    else:
        # Not parallel to the wall: use the extents of the rectangle projected onto the wall frame.
        along_size = long_size * cos_long + short_size * cos_short
        depth_size = long_size * abs(float(np.dot(perpendicular, long_axis))) + short_size * abs(float(np.dot(perpendicular, short_axis)))
    offset = float(np.dot(centre - best[2], perpendicular))
    height = abs(offset)
    if height < 1e-9 or height - depth_size / 2.0 > max_gap_m * upm:
        return None
    normal = perpendicular if offset > 0.0 else -perpendicular
    mounted = match_catalogue(kinds, along_size / upm, depth_size / upm)
    end_on = match_catalogue(kinds, depth_size / upm, along_size / upm)
    if mounted is None or end_on is None:
        return None
    use_end_on = end_on.error < mounted.error - 1e-9
    match = end_on if use_end_on else mounted
    if tolerance is not None and match.error > tolerance:
        return None
    if not use_end_on:
        x_axis = tangent if float(np.dot(np.asarray((-tangent[1], tangent[0])), normal)) > 0.0 else -tangent
        return _uniform_insert(match.entry, centre - normal * height, x_axis, upm, measured, match.error, True)
    # End-on: block +X points into the room (along the measured long axis), +Y runs along the wall.
    x_axis = normal
    y_axis = np.asarray((-x_axis[1], x_axis[0]))
    shifted = centre - normal * (height - depth_size / 2.0)
    insert = _uniform_insert(match.entry, shifted - y_axis * (match.entry.depth_m * upm / 2.0), x_axis, upm, measured, match.error, True)
    insert["_catalogue"]["end_on"] = True  # type: ignore[index]
    return insert


def centred_catalogue_insert(
    measured: dict[str, object],
    kinds: tuple[str, ...] | str,
    units_per_metre: float,
    *,
    tolerance: float | None = None,
    centred_origin: bool = False,
) -> dict[str, object] | None:
    """Uniform-scale catalogue insert for an object that is not wall-mounted.

    The block is centred on the measured footprint and oriented along the
    measured long axis.  Wall-mounted blocks (origin on the mounting edge) are
    shifted by half their depth so their footprint still covers the object.
    """
    upm = max(float(units_per_metre), 1e-9)
    centre = np.asarray(measured["point"], dtype=float)
    long_size, short_size = float(measured["xscale"]) / upm, float(measured["yscale"]) / upm
    rotation = math.radians(float(measured["rotation"]))
    long_axis = np.asarray((math.cos(rotation), math.sin(rotation)))
    short_axis = np.asarray((-long_axis[1], long_axis[0]))
    direct = match_catalogue(kinds, long_size, short_size, tolerance=tolerance)
    swapped = match_catalogue(kinds, short_size, long_size, tolerance=tolerance)
    if direct is None and swapped is None:
        return None
    use_swapped = direct is None or (swapped is not None and swapped.error < direct.error)
    match = swapped if use_swapped else direct
    assert match is not None
    x_axis, y_axis = (short_axis, long_axis) if use_swapped else (long_axis, short_axis)
    # y_axis must be x_axis rotated by +90 degrees (block convention).
    if float(x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0]) < 0.0:
        y_axis = -y_axis
    origin = centre if centred_origin else centre - y_axis * (match.entry.depth_m * upm / 2.0)
    return _uniform_insert(match.entry, origin, x_axis, upm, measured, match.error, False)


def guess_fixture(class_code: int, length: float, width: float, height: float | None, fine_label: str | None) -> str:
    """Map fine labels first, then dimensions: WC 0.35-0.45 x 0.5-0.8 m and 0.35-0.80 m high; washbasin 0.4-0.7 x 0.3-0.6 m at 0.80-0.95 m. Overlapping footprints prefer WC when height is unknown. Bathtub 0.65-0.9 x 1.4-1.9 m; shower 0.7-1.2 m square and low. Unknown objects use FURNITURE."""
    aliases = {
        "toilet": "WC", "wc": "WC", "bathtub": "BATHTUB", "washbasin": "WASHBASIN",
        "shower": "SHOWER", "shower tray": "SHOWER", "stove": "STOVE", "sink": "SINK",
        "kitchen sink": "SINK", "radiator": "RADIATOR", "column": "COLUMN",
    }
    if fine_label and fine_label.strip():
        return aliases.get(fine_label.strip().lower(), "FURNITURE")
    short, long = sorted((float(length), float(width)))
    if 0.35 <= short <= 0.45 and 0.5 <= long <= 0.8 and (height is None or 0.35 <= height <= 0.80):
        return "WC"
    if 0.3 <= short <= 0.6 and 0.4 <= long <= 0.7 and height is not None and 0.80 <= height <= 0.95:
        return "WASHBASIN"
    if 0.65 <= short <= 0.9 and 1.4 <= long <= 1.9 and height is not None and 0.35 <= height <= 0.70:
        return "BATHTUB"
    if 0.7 <= short <= 1.2 and 0.7 <= long <= 1.2 and (height is None or height < 0.5):
        return "SHOWER"
    return "FURNITURE"


def furniture_entities(component_mask: np.ndarray, dsm: np.ndarray, grid: RasterGrid, *, label: str, floor_z: float, step_height: float = 0.15, min_contour_length: float = 0.15, units_per_metre: float = 1.0, regularize: bool = True) -> list[dict[str, object]]:
    """Return a verified true footprint, height steps, and an internal label."""
    mask = np.asarray(component_mask, dtype=bool)
    heights = np.asarray(dsm, dtype=float)
    if mask.shape != (grid.height, grid.width) or heights.shape != mask.shape:
        raise ValueError("component_mask and dsm must match RasterGrid")
    raw_polygons = mask_to_polygons(mask, grid)
    if not raw_polygons:
        return []
    raw_outline = max(raw_polygons, key=lambda polygon: polygon.area)
    radius = max(1, int(round(0.03 * units_per_metre / grid.pixel_size)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(bool)
    polygons = mask_to_polygons(mask, grid)
    if not polygons:
        return []
    smoothed_outline = max(polygons, key=lambda polygon: polygon.area)
    outline, hausdorff = _simplify_outline(smoothed_outline, 0.02 * units_per_metre)
    outline = _remove_short_spikes(outline, smoothed_outline, 0.02 * units_per_metre)
    smooth_distance = float(outline.boundary.hausdorff_distance(smoothed_outline.boundary))
    raw_distance = float(outline.boundary.hausdorff_distance(raw_outline.boundary))
    entities: list[dict[str, object]] = [{"type": "polyline", "points": _coords(outline.exterior.coords), "closed": True, "holes": [_coords(ring.coords) for ring in outline.interiors], "layer_role": "outline", "_hausdorff": smooth_distance, "_raw_hausdorff": raw_distance, "_label": label}]
    maximum = float(np.nanmax(np.where(mask, heights, np.nan)))
    details: list[tuple[float, dict[str, object]]] = []
    for level in np.arange(floor_z + step_height, maximum + 1e-9, step_height):
        for contour in find_contours(np.where(mask, heights, floor_z), level=float(level)):
            points = [grid.pixel_to_local(row, col) for row, col in contour]
            raw_line = LineString(points)
            line = _simplify_line(raw_line, 0.02 * units_per_metre)
            if line.length >= min_contour_length and (line.length < outline.length * 0.9 or line.distance(outline.boundary) > grid.pixel_size * 0.75):
                details.append((line.length, {"type": "polyline", "points": [[float(x), float(y)] for x, y in line.coords], "closed": False, "layer_role": "detail"}))
    entities.extend(entity for _length, entity in sorted(details, key=lambda item: item[0], reverse=True)[:3])
    metric_height = (maximum - floor_z) / max(units_per_metre, 1e-9)
    text_height = (0.08 if outline.area < units_per_metre**2 else 0.12) * units_per_metre
    entities.append({"type": "text", "text": f"{label} h={metric_height:.2f}", "point": list(outline.representative_point().coords[0]), "height": text_height, "rotation": 0.0, "layer_role": "label"})
    return entities


def _simplify_outline(raw: Polygon, limit: float) -> tuple[Polygon, float]:
    """Simplify only while the complete boundary stays within the 2 cm bound."""
    tolerance = limit
    while tolerance >= 1e-5:
        candidate = raw.simplify(tolerance, preserve_topology=True)
        if isinstance(candidate, Polygon) and not candidate.is_empty:
            distance = float(raw.boundary.hausdorff_distance(candidate.boundary))
            if distance <= limit + 1e-9:
                return candidate, distance
        tolerance *= 0.5
    return raw, 0.0


def _simplify_line(raw: LineString, limit: float) -> LineString:
    tolerance = limit
    while tolerance >= 1e-5:
        candidate = raw.simplify(tolerance, preserve_topology=False)
        if candidate.geom_type == "LineString" and raw.hausdorff_distance(candidate) <= limit + 1e-9:
            return candidate
        tolerance *= 0.5
    return raw


def _remove_short_spikes(outline: Polygon, reference: Polygon, limit: float) -> Polygon:
    """Remove only short acute vertices when the smoothed-boundary limit remains valid."""
    coordinates = list(outline.exterior.coords)[:-1]
    if len(coordinates) < 4:
        return outline
    retained: list[tuple[float, float]] = []
    for index, current in enumerate(coordinates):
        previous, following = np.asarray(coordinates[index - 1]), np.asarray(coordinates[(index + 1) % len(coordinates)])
        first, second = previous - current, following - current
        angle = math.degrees(math.acos(float(np.clip(np.dot(first, second) / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12), -1.0, 1.0))))
        if angle < 35.0 and np.linalg.norm(first) < 0.06 and np.linalg.norm(second) < 0.06:
            candidate_points = retained + coordinates[index + 1:]
            if len(candidate_points) >= 3:
                candidate = Polygon(candidate_points)
                if candidate.is_valid and candidate.boundary.hausdorff_distance(reference.boundary) <= limit + 1e-9:
                    continue
        retained.append(current)
    candidate = Polygon(retained) if len(retained) >= 3 else outline
    return candidate if candidate.is_valid and candidate.boundary.hausdorff_distance(reference.boundary) <= limit + 1e-9 else outline


def _bridge_axis_gaps(mask: np.ndarray, grid: RasterGrid, max_gap: float) -> np.ndarray:
    pixels = max(1, int(round(max_gap / grid.pixel_size)))
    horizontal = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, pixels + 1), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    vertical = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((pixels + 1, 1), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (horizontal | vertical).astype(bool)


def _polygon_mask(polygon: Polygon, grid: RasterGrid) -> np.ndarray:
    mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
    def points(coordinates: object) -> np.ndarray:
        return np.asarray([[round(grid.local_to_pixel(x, y)[1]), round(grid.local_to_pixel(x, y)[0])] for x, y in list(coordinates)], dtype=np.int32)
    cv2.fillPoly(mask, [points(polygon.exterior.coords)], 1)
    for ring in polygon.interiors:
        cv2.fillPoly(mask, [points(ring.coords)], 0)
    return mask.astype(bool)


def _wall_thickness(mask: np.ndarray, polygon: Polygon, grid: RasterGrid) -> float:
    raster = _polygon_mask(polygon, grid) & mask
    distances = ndimage.distance_transform_edt(raster)
    skeleton = skeletonize(raster)
    values = distances[skeleton]
    return float(np.median(values) * 2.0 * grid.pixel_size) if len(values) else 0.0


def _free_space_by_normal(opening: Opening, free_space_mask: np.ndarray, grid: RasterGrid) -> tuple[int, int]:
    """Count free pixels on either side of an opening, just beyond wall material."""
    axis = np.asarray((math.cos(opening.axis_angle), math.sin(opening.axis_angle)))
    normal = np.asarray((-axis[1], axis[0]))
    offset = opening.wall_thickness / 2.0 + 2.0 * grid.pixel_size
    samples = np.linspace(-opening.width * 0.35, opening.width * 0.35, 5)
    counts: list[int] = []
    for sign in (1.0, -1.0):
        total = 0
        for along in samples:
            point = np.asarray(opening.center) + axis * along + normal * offset * sign
            row, col = grid.local_to_pixel(float(point[0]), float(point[1]))
            row_index, col_index = int(round(row)), int(round(col))
            if 0 <= row_index < grid.height and 0 <= col_index < grid.width and free_space_mask[row_index, col_index]:
                total += 1
        counts.append(total)
    return counts[0], counts[1]


def _component_axis_angle(points: np.ndarray) -> float:
    """Return the local-metre direction of a rectangular wall-gap component."""
    rectangle = cv2.minAreaRect(np.asarray([(x, -y) for x, y in points], dtype=np.float32))
    corners = cv2.boxPoints(rectangle)
    local = np.asarray([(float(x), -float(y)) for x, y in corners])
    edges = [local[(index + 1) % 4] - point for index, point in enumerate(local)]
    edge = max(edges, key=lambda item: float(np.hypot(*item)))
    return math.atan2(float(edge[1]), float(edge[0])) % math.pi


def _deduplicate_openings(openings: list[Opening]) -> list[Opening]:
    result: list[Opening] = []
    priority = {"DOOR": 3, "WINDOW": 2, "passage": 1}
    for opening in sorted(openings, key=lambda item: (-priority.get(item.kind, 0), -item.class_evidence, -item.lintel_evidence, item.center)):
        duplicate = next((existing for existing in result if opening.kind in {"DOOR", "passage"} and existing.kind in {"DOOR", "passage"} and _opening_spans_overlap(opening, existing)), None)
        if duplicate is None:
            duplicate = next((existing for existing in result if opening.kind == existing.kind and math.dist(opening.center, existing.center) < max(opening.width, existing.width) * 0.15), None)
        if duplicate is None:
            result.append(opening)
        elif opening.kind == "DOOR" and duplicate.kind == "passage":
            result[result.index(duplicate)] = opening
    return result


def _opening_spans_overlap(first: Opening, second: Opening) -> bool:
    first_axis = np.asarray((math.cos(first.axis_angle), math.sin(first.axis_angle)))
    second_axis = np.asarray((math.cos(second.axis_angle), math.sin(second.axis_angle)))
    if _axis_angle_distance(first_axis, second_axis) > math.radians(8.0):
        return False
    normal = np.asarray((-first_axis[1], first_axis[0]))
    if abs(float(np.dot(np.asarray(second.center) - np.asarray(first.center), normal))) > 0.08:
        return False
    distance = abs(float(np.dot(np.asarray(second.center) - np.asarray(first.center), first_axis)))
    return distance < 0.5 * max(first.width, second.width) or distance < (first.width + second.width) / 2.0


def _line_axis(line: LineString) -> np.ndarray:
    start, end = np.asarray(line.coords[0], dtype=float), np.asarray(line.coords[-1], dtype=float)
    vector = end - start
    return vector / max(float(np.hypot(*vector)), 1e-9)


def _axis_angle_distance(first: np.ndarray, second: np.ndarray) -> float:
    return abs(math.asin(float(np.clip(abs(first[0] * second[1] - first[1] * second[0]), 0.0, 1.0))))


def _line_overlap(first: LineString, second: LineString, axis: np.ndarray) -> float:
    first_values = [float(np.dot(point, axis)) for point in first.coords]
    second_values = [float(np.dot(point, axis)) for point in second.coords]
    return max(0.0, min(max(first_values), max(second_values)) - max(min(first_values), min(second_values)))


def _polygon_between_faces(first: LineString, second: LineString, axis: np.ndarray) -> Polygon | None:
    origin = np.asarray(first.coords[0], dtype=float)
    first_values = [float(np.dot(np.asarray(point) - origin, axis)) for point in first.coords]
    second_values = [float(np.dot(np.asarray(point) - origin, axis)) for point in second.coords]
    low, high = max(min(first_values), min(second_values)), min(max(first_values), max(second_values))
    if high <= low:
        return None
    normal = np.asarray((-axis[1], axis[0]))
    first_offset = float(np.dot(np.asarray(first.centroid.coords[0]) - origin, normal))
    second_offset = float(np.dot(np.asarray(second.centroid.coords[0]) - origin, normal))
    points = [origin + axis * low + normal * first_offset, origin + axis * high + normal * first_offset, origin + axis * high + normal * second_offset, origin + axis * low + normal * second_offset]
    return Polygon([tuple(point) for point in points])


def _outside_normal(face: LineString, normal: np.ndarray, free_space_mask: np.ndarray, grid: RasterGrid) -> np.ndarray:
    midpoint = np.asarray(face.interpolate(0.5, normalized=True).coords[0])
    distance = max(2.0 * grid.pixel_size, 0.06)
    scores: list[int] = []
    for sign in (1.0, -1.0):
        point = midpoint + normal * distance * sign
        row, col = grid.local_to_pixel(float(point[0]), float(point[1]))
        row, col = int(round(row)), int(round(col))
        scores.append(int(0 <= row < grid.height and 0 <= col < grid.width and free_space_mask[row, col]))
    # The wall offset is away from the room side, which is the free side.
    return -normal if scores[0] > scores[1] else normal


def _snap_faces_to_axes(lines: list[LineString], angle_tolerance_deg: float) -> list[LineString]:
    """Snap segments close to the two length-dominant orthogonal axes."""
    if not lines:
        return []
    angles = np.asarray([math.atan2(_line_axis(line)[1], _line_axis(line)[0]) % (math.pi / 2.0) for line in lines])
    weights = np.asarray([line.length for line in lines])
    primary_angle = 0.25 * math.atan2(float(np.sum(weights * np.sin(4.0 * angles))), float(np.sum(weights * np.cos(4.0 * angles))))
    primary = np.asarray((math.cos(primary_angle), math.sin(primary_angle)))
    secondary = np.asarray((-primary[1], primary[0]))
    tolerance = math.radians(angle_tolerance_deg)
    output: list[LineString] = []
    for line in lines:
        axis = _line_axis(line)
        chosen = primary if _axis_angle_distance(axis, primary) <= _axis_angle_distance(axis, secondary) else secondary
        if _axis_angle_distance(axis, chosen) > tolerance:
            continue
        midpoint = np.asarray(line.interpolate(0.5, normalized=True).coords[0])
        length = line.length
        output.append(LineString((midpoint - chosen * length / 2.0, midpoint + chosen * length / 2.0)))
    return output


def _merge_face_lines(lines: list[LineString], angle_tolerance_deg: float, offset_tol: float, gap_tol: float) -> list[LineString]:
    """Merge overlapping or almost-touching collinear Hough segments."""
    output: list[LineString] = []
    tolerance = math.radians(angle_tolerance_deg)
    for line in sorted(lines, key=lambda item: item.length, reverse=True):
        axis = _line_axis(line)
        merged = False
        for index, current in enumerate(output):
            current_axis = _line_axis(current)
            if _axis_angle_distance(axis, current_axis) > tolerance:
                continue
            origin = np.asarray(current.coords[0])
            normal = np.asarray((-current_axis[1], current_axis[0]))
            offset = abs(float(np.dot(np.asarray(line.centroid.coords[0]) - origin, normal)))
            if offset > offset_tol:
                continue
            axis_used = current_axis
            current_values = [float(np.dot(np.asarray(point) - origin, axis_used)) for point in current.coords]
            line_values = [float(np.dot(np.asarray(point) - origin, axis_used)) for point in line.coords]
            if max(min(current_values), min(line_values)) - min(max(current_values), max(line_values)) > gap_tol:
                continue
            values = [*current_values, *line_values]
            normal = np.asarray((-axis_used[1], axis_used[0]))
            offset = float(np.mean([np.dot(np.asarray(point) - origin, normal) for point in (*current.coords, *line.coords)]))
            output[index] = LineString((origin + axis_used * min(values) + normal * offset, origin + axis_used * max(values) + normal * offset))
            merged = True
            break
        if not merged:
            output.append(line)
    return output


def _coords(coordinates: object) -> list[list[float]]:
    return [[float(x), float(y)] for x, y, *_rest in list(coordinates)[:-1]]
