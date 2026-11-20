"""Double-line indoor walls: both faces of every wall, with measured or learned thickness.

An indoor scan normally sees one face of a wall from each room.  This module
turns the tidied single wall faces of the clean plan into real wall bodies:

1. **Solid side.**  For every face the side that faces a room (ceiling room
   polygon or free floor) is the room side; the other side is the wall body.
2. **Measured pairs.**  Two parallel faces whose solid sides face each other at
   a plausible distance (``min_thickness_m`` .. ``max_thickness_m``) are the two
   faces of one wall: the distance is the measured thickness.
3. **Far-side probe.**  Behind the solid side the next room (another ceiling
   room or free floor) is searched.  A room within the thickness range gives
   the thickness from the room edge; a room further away marks an interior
   wall of unknown thickness; nothing (void, outside) marks an exterior wall.
4. **Learned thickness.**  Measured pairs (and room-edge distances) are
   clustered into the typical interior thicknesses; exterior walls use the
   measured exterior pairs or the configured default.  Collinear faces of the
   same wall share their measured thickness, so a face seen only in part
   still gets one consistent second face.
5. **Wall bodies.**  Each face run becomes a band (face + thickness on the
   solid side).  Ends are extended into L corners, short gaps to another wall
   are closed, bands are unioned (clean L/T/X joints), and every door/window
   span is cut through the full thickness, so the jambs are closed.

The result is written as closed polylines (outline of each wall body, with
holes) and an optional hatch.  All distances scale with ``units_per_metre``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import cv2
import numpy as np
from shapely import make_valid
from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.ops import unary_union

from .raster_grid import RasterGrid


@dataclass(frozen=True, slots=True)
class DoubleWallSettings:
    """Double-line wall settings (metres, scaled by the project units per metre)."""

    double_line: bool = True
    min_thickness_m: float = 0.06
    max_thickness_m: float = 0.45
    exterior_thickness_m: float = 0.30
    interior_thickness_m: float = 0.12
    probe_m: float = 0.80
    side_probe_m: float = 0.25
    sample_step_m: float = 0.05
    min_run_m: float = 1.00
    pair_min_overlap_m: float = 0.10
    pair_gap_m: float = 1.0
    pair_angle_deg: float = 3.0
    collinear_offset_m: float = 0.04
    # Collinear fragments can be separated by a doorway or a room-edge artefact.  They
    # share a centreline and thickness unless two independently measured faces prove a step.
    flush_line_offset_m: float = 0.20
    flush_chain_gap_m: float = 1.50
    step_proof_face_length_m: float = 0.30
    step_proof_offset_m: float = 0.06
    cluster_gap_m: float = 0.04
    cluster_snap_m: float = 0.04
    joint_tolerance_m: float = 0.04
    joint_reach_m: float = 0.35
    jamb_reach_m: float = 0.15
    min_face_m: float = 0.10
    min_stub_m: float = 0.15
    jog_m: float = 0.04
    min_piece_area_m2: float = 0.003
    isolated_max_length_m: float = 1.2
    stub_min_length_m: float = 0.6
    stub_end_reach_m: float = 0.10
    isolated_free_max_length_m: float = 2.5
    isolated_free_room_distance_m: float = 0.6
    keep_isolated_faces: bool = False
    room_edge_walls: bool = True
    room_edge_min_area_m2: float = 2.0
    room_edge_min_length_m: float = 0.30
    room_edge_opening_reach_m: float = 0.8  # room edges this close to a parallel door/window span invent no wall there (0 = off)
    hatch: bool = True
    bound_rooms: bool = True
    room_gap_m: float = 0.20
    room_merge_max_area_m2: float = 2.0
    room_fill_min_share: float = 0.25
    face_test: bool = True
    face_min_height_m: float = 1.0
    face_reach_m: float = 1.7
    face_min_share: float = 0.25
    visibility_test: bool = True
    visibility_max_m: float = 12.0
    visibility_min_share: float = 0.15
    door_leaf_reorient: bool = True
    door_detect: bool = True
    # The openings come from the 3D refinement (measured frames, leaves, passages): the 2D door heuristics (leaf reorientation,
    # solid-door rejection, doorway detection, doors made of leaf faces) must not move, drop or add openings; a leaf face is still
    # removed from the wall faces.
    refined_openings: bool = False
    door_min_width_m: float = 0.55
    door_max_width_m: float = 1.40
    door_floor_share: float = 0.15
    # Tall built-in kitchen units (fridge / oven tower / tall cabinets) whose front was taken as a wall face:
    # the unit becomes a built-in and the wall stays (or is continued) behind it.
    tall_units: bool = True
    tall_unit_band_m: float = 0.12
    tall_unit_min_m: float = 0.45
    tall_unit_gap_m: float = 0.30
    tall_unit_reach_m: float = 1.0
    tall_unit_min_depth_m: float = 0.40
    tall_unit_max_depth_m: float = 0.90
    tall_unit_side_gap_m: float = 0.60
    tall_unit_neighbour_gap_m: float = 0.40
    tall_unit_room_m: float = 0.40
    tall_unit_room_free: float = 0.50
    tall_unit_max_free: float = 0.15
    tall_unit_max_low: float = 0.25
    tall_unit_fridge_m: float = 0.60
    # A wall run whose body lies on kitchen / cabinet labels and whose points show no wall (wall-class points of wall
    # height) is a cabinet front taken as a wall: the run is dropped (the counter run / unit outlines draw the cabinets).
    # A real wall behind cabinets keeps its wall-class points above and beside them and stays.
    cabinet_walls: bool = True
    cabinet_min_share: float = 0.50
    cabinet_max_wall_share: float = 0.50
    cabinet_dilate_m: float = 0.05
    # A stretch of a measured wall run without any wall-class column (no wall points of wall height) that lies on or is flanked
    # (within cabinet_flank_m, either side) by kitchen cabinet labels, at least cabinet_stretch_min_m long, is a cabinet column
    # (its side panels were taken as a wall): the stretch is dropped and, where a built-in unit stands there, becomes a tall unit.
    cabinet_flank_m: float = 0.50
    cabinet_stretch_min_m: float = 0.60
    cabinet_stretch_gap_m: float = 0.25
    # ... and only when the run is mostly such a column: flagged samples >= cabinet_run_min_share of the run and wall-supported
    # samples <= cabinet_run_max_support (a real wall that is merely weakly measured beside a kitchen unit keeps its run).
    cabinet_remnant_min_m: float = 0.45  # what is left of the run beside a removed column is kept only when it is at least this long (no wall stubs)
    cabinet_run_min_share: float = 0.40
    cabinet_run_max_support: float = 0.15
    # An open door leaf is never a wall: a wall run whose centreline lies along the measured open leaf of a door (both ends within
    # leaf_run_distance_m of the leaf segment, not longer than the leaf + leaf_run_extra_m) is that leaf and is dropped.
    leaf_walls: bool = True
    leaf_run_distance_m: float = 0.20
    leaf_run_extra_m: float = 0.25
    # Steps in a wall line that the room outline invented (parallel room-edge run within jog_merge_m of a neighbouring parallel run that
    # touches or overlaps it) are straightened onto the neighbour's line; remaining free ends that face each other across a step of at
    # most close_step_max_m (ends aligned within close_step_align_m) are joined by a connecting wall, so the walls form closed outlines.
    close_walls: bool = True
    jog_merge_m: float = 0.25
    close_step_max_m: float = 0.80
    close_step_align_m: float = 0.35
    free_end_reach_m: float = 0.12
    # A door / passage is an opening in ONE host wall: parallel wall runs other than the host (offset more than door_host_collinear_m from the
    # host's centreline) are cut back to the jambs, perpendicular runs entering the span by more than door_span_enter_m are cut back, and a
    # door without any wall run gets a door-host piece between its jambs (thickness of the nearest wall within door_fill_reach_m).
    # Depth evidence (COLMAP-supported class-2 voxels the cloud lacks, ``WallEvidence.depth_points``): a measured face continues along
    # its own line (within depth_face_tolerance_m) over unseen stretches of at most depth_extend_max_m; a wall whose thickness is not
    # measured from two cloud faces takes the offset of a depth face on its solid side (>= depth_cover_min_m of the run, voxels on
    # >= depth_min_height_bins height bins).  Never used without depth points.
    depth_faces: bool = True
    depth_extend: bool = True
    depth_thickness: bool = True
    depth_face_tolerance_m: float = 0.03
    depth_extend_max_m: float = 0.50
    depth_bin_m: float = 0.05
    depth_min_voxels: int = 3
    depth_min_height_bins: int = 3
    depth_cover_min_m: float = 0.30
    depth_cover_min_share: float = 0.30
    door_spans: bool = True
    door_host_collinear_m: float = 0.03
    door_span_enter_m: float = 0.20
    door_min_piece_m: float = 0.17
    door_fill_reach_m: float = 1.5


@dataclass(slots=True)
class TallUnitEvidence:
    """Rasters (plan grid) for the tall built-in unit test.

    ``front``: pixels of tall built-in kitchen units (fine label kitchen cabinet at full height, fridge, oven).
    ``fridge``: fridge pixels (the unit is labelled "KS" there, else "HS").  ``low``: observed surfaces below
    unit height (counter tops, floor furniture, floor) - a tall unit's footprint holds none of them.
    """

    front: np.ndarray
    fridge: np.ndarray
    low: np.ndarray
    cabinet: np.ndarray | None = None  # kitchen / cabinet-labelled pixels (counter, cabinet, stove, fridge, sink ...)


@dataclass(slots=True)
class WallEvidence:
    """Optional measured evidence for the wall-body builder (all rasters on the plan grid).

    ``wall_bits``: per pixel, bit ``i`` set when wall-class points lie in height bin ``i``
    of ``bins`` equal bins from floor to ``storey_height``; ``all_bits`` the same for
    points of any class.  ``section`` / ``low``: any
    points at mid height / in the sill band.  ``floor``: measured floor points (no
    morphological closing).  ``door``: door-class pixels.  ``cameras``: N x 2 camera
    centres of the level.
    """

    wall_bits: np.ndarray | None = None
    all_bits: np.ndarray | None = None
    bins: int = 16
    storey_height: float = 0.0
    section: np.ndarray | None = None
    low: np.ndarray | None = None
    floor: np.ndarray | None = None
    door: np.ndarray | None = None
    cameras: np.ndarray | None = None
    depth_points: np.ndarray | None = None  # (N, 3) plan x, y and height above floor of COLMAP class-2 gap-fill voxels (local units)


@dataclass(slots=True)
class _Face:
    index: int
    start: np.ndarray
    end: np.ndarray
    entity: dict[str, Any]
    side: int = 0
    side_margin: float = 0.0

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        vector = self.end - self.start
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    @property
    def normal(self) -> np.ndarray:
        direction = self.direction
        return np.asarray((-direction[1], direction[0]))

    @property
    def solid_normal(self) -> np.ndarray:
        return self.normal * (self.side if self.side else 1)

    def point(self, along: float, across: float = 0.0) -> np.ndarray:
        return self.start + self.direction * along + self.solid_normal * across


@dataclass(slots=True)
class WallRun:
    """One piece of a wall face with its wall body on the solid side."""

    face: int
    start: float
    end: float
    thickness: float
    kind: str
    evidence: str
    group: int = -1
    extend_start: float = 0.0
    extend_end: float = 0.0


@dataclass(slots=True)
class DoubleWallResult:
    """Wall bodies, their DXF entities, updated openings and statistics."""

    entities: list[dict[str, Any]]
    hatches: list[dict[str, Any]]
    polygons: list[Polygon]
    closed_polygons: list[Polygon]
    faces: list[LineString]
    caps: list[LineString]
    opening_spans: list[tuple[str, LineString]]
    opening_slabs: list[tuple[float, float]]
    stats: dict[str, Any] = field(default_factory=dict)
    pieces: list[dict[str, Any]] = field(default_factory=list)
    opening_cuts: list[Polygon] = field(default_factory=list)
    openings: list[Any] = field(default_factory=list)
    openings_changed: list[tuple[Any, Any]] = field(default_factory=list)
    tall_units: list[tuple[Polygon, str]] = field(default_factory=list)


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------

def build_double_walls(
    entities: list[dict[str, Any]],
    openings: list[Any],
    rooms: Iterable[Polygon],
    free_floor: np.ndarray,
    valid: np.ndarray,
    grid: RasterGrid,
    units_per_metre: float,
    settings: DoubleWallSettings = DoubleWallSettings(),
    furniture: np.ndarray | None = None,
    evidence: WallEvidence | None = None,
    builtins: Iterable[Polygon] = (),
    tall_units: TallUnitEvidence | None = None,
    close_spans: Iterable[dict[str, Any]] = (),
) -> DoubleWallResult:
    """Turn single measured wall faces into closed double-line wall bodies.

    ``furniture`` (optional) marks furniture and fixtures; two faces with furniture
    between them (a wardrobe in front of a wall) are not the two faces of one wall.
    ``evidence`` (optional) enables the wall-height face test, the camera visibility
    test, door-leaf correction and door detection in the wall bodies.  ``builtins``
    (closed built-in units whose wall was moved behind them) are not outside: no
    outline wall is completed through them.  ``tall_units`` (optional) enables the tall built-in unit test:
    a wall face standing at the front of a tall kitchen unit (fridge, oven tower) is removed, the unit is
    returned in ``tall_units`` and the wall behind it is kept or continued (see ``_tall_unit_runs``).

    ``close_spans`` ({"center", "axis_angle", "width"} each) are spans the 3D refinement classified as no opening: where the
    measured walls on both sides leave a gap there, the host wall is continued across it (``door-fill`` run).

    The returned ``openings`` are the input openings after door-leaf correction,
    overlap resolution and door detection; ``openings_changed`` lists (old, new) pairs,
    with old None for a new opening and new None for a removed one.
    """
    upm = max(float(units_per_metre), 1e-9)
    room_list = [polygon for polygon in rooms if isinstance(polygon, Polygon) and not polygon.is_empty]
    faces, passthrough = _collect_faces(entities, upm, settings.min_face_m)
    original_openings = list(openings)
    face_stats: dict[str, Any] = {}
    if evidence is not None:
        faces, face_stats = _filter_faces_by_evidence(faces, evidence, np.asarray(valid, dtype=bool), None if furniture is None else np.asarray(furniture, dtype=bool), grid, upm, settings)
        openings = _resolve_overlapping_openings(openings, upm)
        if settings.door_leaf_reorient:
            if not settings.refined_openings:
                openings = _reorient_door_leaves(openings, faces, evidence, grid, upm, settings)
            faces, leaf_doors = _leaf_faces_to_doors(faces, openings, evidence, grid, upm, settings)
            face_stats["double_wall_leaf_faces"] = len(leaf_doors)
            if not settings.refined_openings:
                openings = list(openings) + leaf_doors
    faces, reveals = _split_reveals(faces, openings, upm, settings)
    faces, isolated = _split_isolated(faces, upm, settings, room_list)
    if settings.keep_isolated_faces:
        passthrough.extend((-1, {**face.entity, "points": [face.start.tolist(), face.end.tolist()], "_isolated_face": True}) for face in isolated)
    depth_points = None if evidence is None or not settings.depth_faces else evidence.depth_points
    depth_stats: dict[str, Any] = {}
    if depth_points is not None and len(depth_points) and faces and settings.depth_extend:
        depth_stats["double_wall_depth_faces_extended"] = _depth_extend_faces(faces, depth_points, openings, upm, settings, evidence, np.asarray(free_floor, dtype=bool), grid)
    stats: dict[str, Any] = {"double_wall_faces": len(faces), "double_wall_passthrough": len(passthrough), "double_wall_isolated_faces": len(isolated), "double_wall_reveals_dropped": len(reveals), **face_stats, **depth_stats}
    if not faces:
        return DoubleWallResult(list(entities), [], [], [], [], [], [], [(0.0, 0.0)] * len(openings), stats, openings=list(openings), openings_changed=_opening_changes(original_openings, openings))

    room_labels = _room_label_raster(room_list, grid)
    free = np.asarray(free_floor, dtype=bool)
    valid_mask = np.asarray(valid, dtype=bool)
    roomish = free | (room_labels > 0)
    fallback: dict[int, int] = {}
    for face in faces:
        face.side, face.side_margin, fallback[face.index] = _solid_side(face, roomish, valid_mask, grid, upm, settings)
    pairs = _measured_pairs(faces, free, grid, upm, settings, None if furniture is None else np.asarray(furniture, dtype=bool))
    _adopt_collinear_sides(faces, upm, settings)
    for face in faces:
        if not face.side:
            face.side = fallback[face.index]
    stats["double_wall_ambiguous_faces"] = sum(face.side == 0 for face in faces)
    stats["double_wall_faces_aligned"] = _align_collinear(faces, upm, settings)
    outside = {face.index: _is_outer_face(face, roomish, grid, upm) for face in faces}
    stats["double_wall_outer_faces"] = sum(outside.values())
    samples = [_far_samples(face, pairs.get(face.index, []), room_labels, free, valid_mask, grid, upm, settings, outside[face.index], openings) for face in faces]
    groups = _collinear_groups(faces, upm, settings)
    runs: list[WallRun] = []
    for face, face_samples in zip(faces, samples, strict=True):
        runs.extend(_face_runs(face, face_samples, upm, settings))
    stats["double_wall_run_boundaries_snapped"] = _snap_run_boundaries(runs, faces, upm, settings)
    interior_clusters, exterior_value, learn_stats = _learn_thickness(runs, samples, pairs, upm, settings)
    stats.update(learn_stats)
    _assign_thickness(runs, faces, samples, groups, interior_clusters, exterior_value, upm, settings, pairs)
    if depth_points is not None and len(depth_points) and settings.depth_thickness:
        stats["double_wall_depth_thickness_runs"] = _depth_thickness(runs, faces, depth_points, upm, settings, evidence)
    stats["double_wall_flush_runs"] = _flush_collinear_runs(runs, faces, upm, settings)
    if settings.room_edge_walls:
        added = _room_edge_runs(faces, runs, room_list, roomish, valid_mask, openings, grid, upm, settings, exterior_value, [item for item in builtins if isinstance(item, Polygon) and not item.is_empty])
        stats["double_wall_room_edge_pieces"] = len(added)
        stats["double_wall_room_edge_m"] = float(sum(run.end - run.start for run in added) / upm)
    unit_shapes: list[tuple[Polygon, str]] = []
    if tall_units is not None and settings.tall_units:
        unit_shapes, unit_stats = _tall_unit_runs(runs, faces, free, tall_units, grid, upm, settings)
        stats.update(unit_stats)
    if tall_units is not None and tall_units.cabinet is not None and evidence is not None and settings.cabinet_walls:
        cabinet_stats, cabinet_units = _cabinet_wall_runs(runs, faces, tall_units.cabinet, evidence, grid, upm, settings, [item for item in builtins if isinstance(item, Polygon) and not item.is_empty], free)
        stats.update(cabinet_stats)
        unit_shapes.extend(cabinet_units)
    if settings.leaf_walls:
        stats["double_wall_leaf_runs_dropped"] = _drop_leaf_runs(runs, faces, openings, upm, settings)
    close_stats = _close_spans(list(close_spans), runs, faces, upm)
    if close_stats:
        stats["double_wall_spans_closed"] = close_stats
    if evidence is not None and settings.door_detect and not settings.refined_openings:
        openings, rejected = _reject_solid_doors(openings, runs, faces, evidence, grid, upm)
        stats["double_wall_doors_rejected"] = rejected
        detected = _detect_wall_doors(runs, faces, openings, evidence, grid, upm, settings)
        stats["double_wall_doors_detected"] = len(detected)
        openings = list(openings) + detected

    # after the door detection: every opening, also the ones found in the wall bodies, is judged
    if settings.door_spans:
        stats["double_wall_door_spans"] = _enforce_door_spans(runs, faces, openings, upm, settings)
    if settings.close_walls:
        stats["double_wall_jogs_merged"] = _merge_jogs(runs, faces, upm, settings)
        stats["double_wall_closing_pieces"] = _close_free_ends(runs, faces, openings, upm, settings)

    bands_closed, opening_slabs, cut_shapes, jamb_lines = _wall_bodies(runs, faces, openings, upm, settings, roomish, grid)
    closed_union = _clean_union(bands_closed, upm, settings)
    cut_union = unary_union(cut_shapes) if cut_shapes else GeometryCollection()
    body = closed_union.difference(cut_union) if not cut_union.is_empty else closed_union
    polygons = _polygons_of(body, upm, settings)
    closed_polygons = _polygons_of(closed_union, upm, settings)

    output: list[dict[str, Any]] = [entity for _, entity in passthrough]
    hatches: list[dict[str, Any]] = []
    for polygon in polygons:
        exterior = _ring(polygon.exterior.coords)
        holes = [_ring(ring.coords) for ring in polygon.interiors]
        output.append({"type": "polyline", "points": exterior, "closed": True, "holes": holes, "_double_wall": True, "_axis_snapped": True, "_rms": 0.0, "_max_residual": 0.0})
        if settings.hatch:
            hatches.append({"type": "hatch", "points": exterior, "holes": holes, "_double_wall": True})

    face_lines, cap_lines = _classify_outline(polygons, runs, faces, jamb_lines, upm)
    spans: list[tuple[str, LineString]] = []
    for opening, slab in zip(openings, opening_slabs, strict=True):
        axis = _axis_of(opening)
        centre = np.asarray(getattr(opening, "center"), dtype=float)[:2]
        normal = np.asarray((-axis[1], axis[0]))
        middle = centre + normal * (slab[0] + slab[1]) / 2.0
        half = axis * float(getattr(opening, "width")) / 2.0
        spans.append((str(getattr(opening, "kind", "")), LineString([tuple(middle - half), tuple(middle + half)])))

    thicknesses = {"interior": [], "exterior": [], "partition": []}
    for run in runs:
        thicknesses.setdefault(run.kind, []).append((run.thickness / upm, (run.end - run.start) / upm, run.evidence))
    stats.update({
        "double_wall_runs": len(runs),
        "double_wall_bodies": len(polygons),
        "double_wall_length_m": float(sum((run.end - run.start) for run in runs) / upm),
        "double_wall_interior_m": float(sum(length for _, length, _ in thicknesses.get("interior", []))),
        "double_wall_exterior_m": float(sum(length for _, length, _ in thicknesses.get("exterior", []))),
        "double_wall_partition_m": float(sum(length for _, length, _ in thicknesses.get("partition", []))),
        "double_wall_evidence_m": {key: float(sum((run.end - run.start) / upm for run in runs if run.evidence == key)) for key in sorted({run.evidence for run in runs})},
        "double_wall_thickness_by_kind": {kind: sorted({round(value, 3) for value, _, _ in items}) for kind, items in thicknesses.items() if items},
        "double_wall_measured_pairs": int(sum(len(items) for items in pairs.values()) // 2),
    })
    face_of = {face.index: face for face in faces}
    pieces = [
        {
            "face": [face_of[run.face].point(run.start - run.extend_start).tolist(), face_of[run.face].point(run.end + run.extend_end).tolist()],
            "face_b": [face_of[run.face].point(run.start - run.extend_start, run.thickness).tolist(), face_of[run.face].point(run.end + run.extend_end, run.thickness).tolist()],
            "solid_normal": face_of[run.face].solid_normal.tolist(), "thickness": run.thickness / upm, "kind": run.kind, "evidence": run.evidence,
            "source": face_of[run.face].entity.get("_source"),
        }
        for run in runs
    ]
    return DoubleWallResult(output, hatches, polygons, closed_polygons, face_lines, cap_lines, spans, opening_slabs, stats, pieces, list(cut_shapes), list(openings), _opening_changes(original_openings, openings), unit_shapes)


def bound_rooms_by_walls(
    rooms: list[Polygon],
    wall_bodies: list[Polygon],
    units_per_metre: float,
    settings: DoubleWallSettings = DoubleWallSettings(),
    barriers: Iterable[Polygon] = (),
) -> tuple[list[Polygon], dict[str, int]]:
    """Grow each ceiling room to the inner wall faces and merge ceiling-hole fragments.

    A room grows by at most ``room_gap_m`` into free space but never into a wall
    body or another room.  Two rooms that touch without any wall between them are
    one room when either is smaller than ``room_merge_max_area_m2`` (a ceiling
    hole split one room into fragments).
    """
    upm = max(float(units_per_metre), 1e-9)
    stats = {"rooms_bounded": 0, "rooms_merged": 0}
    if not rooms:
        return list(rooms), stats
    walls = unary_union(wall_bodies) if wall_bodies else GeometryCollection()
    # Door and window spans: a room may not grow into an opening (the next room starts there).
    barrier_list = [item for item in barriers if isinstance(item, Polygon) and not item.is_empty]
    if barrier_list:
        walls = unary_union([walls, *barrier_list]) if not walls.is_empty else unary_union(barrier_list)
    gap = settings.room_gap_m * upm
    grown: list[Polygon] = []
    for index, room in enumerate(rooms):
        others = unary_union([other for other_index, other in enumerate(rooms) if other_index != index]) if len(rooms) > 1 else GeometryCollection()
        candidate = room.buffer(gap, join_style=2)
        if not walls.is_empty:
            candidate = candidate.difference(walls)
        if not others.is_empty:
            candidate = candidate.difference(others)
        candidate = candidate.union(room.difference(walls) if not walls.is_empty else room)
        part = _largest_overlap(candidate, room)
        if part is None:
            grown.append(room)
            continue
        part = part.buffer(0.005 * upm, join_style=2).buffer(-0.005 * upm, join_style=2)
        part = _largest_overlap(part, room) or part
        grown.append(Polygon(part.exterior) if isinstance(part, Polygon) else room)
        stats["rooms_bounded"] += 1
    # Resolve overlaps between grown rooms: the original owner keeps a contested cell.
    for index in range(len(grown)):
        for other in range(index + 1, len(grown)):
            if grown[index].intersects(grown[other]):
                shared = grown[index].intersection(grown[other])
                if shared.area <= 1e-12:
                    continue
                if rooms[index].intersection(shared).area >= rooms[other].intersection(shared).area:
                    trimmed = _largest_overlap(grown[other].difference(shared), rooms[other])
                    if trimmed is not None:
                        grown[other] = trimmed
                else:
                    trimmed = _largest_overlap(grown[index].difference(shared), rooms[index])
                    if trimmed is not None:
                        grown[index] = trimmed
    merged = list(grown)
    # The fragment size is judged on the measured ceiling area, not on the grown outline.
    measured_area = [room.area for room in rooms]
    limit = settings.room_merge_max_area_m2 * upm * upm
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for index in range(len(merged)):
            for other in range(index + 1, len(merged)):
                first, second = merged[index], merged[other]
                if min(measured_area[index], measured_area[other]) >= limit:
                    continue
                contact = first.buffer(0.03 * upm, join_style=2).intersection(second.buffer(0.03 * upm, join_style=2))
                if contact.is_empty or contact.area <= 0.0:
                    continue
                free_contact = contact.difference(walls) if not walls.is_empty else contact
                if free_contact.area < 0.5 * contact.area:
                    continue
                if _contact_length(contact) < 0.30 * upm:
                    continue
                joined = unary_union([first, second, contact.difference(walls) if not walls.is_empty else contact])
                joined = joined.buffer(0.01 * upm, join_style=2).buffer(-0.01 * upm, join_style=2)
                part = _largest_overlap(joined, first.union(second))
                if part is None:
                    continue
                merged[index] = Polygon(part.exterior)
                measured_area[index] += measured_area[other]
                del merged[other]
                del measured_area[other]
                stats["rooms_merged"] += 1
                changed = True
                break
            if changed:
                break
    # A wall-enclosed cell (doors and windows closed) that holds exactly one room is that room:
    # its outline follows the inner wall faces all round, also where the ceiling was not seen.
    stats["rooms_filled_to_walls"] = 0
    for cell in _enclosed_cells(walls, upm):
        inside = [index for index, room in enumerate(merged) if room.intersection(cell).area >= 0.5 * room.area]
        if len(inside) != 1:
            continue
        index = inside[0]
        if merged[index].area < settings.room_fill_min_share * cell.area:
            continue
        others = [room for other, room in enumerate(merged) if other != index and room.intersects(cell)]
        filled = cell.difference(unary_union(others)) if others else cell
        part = _largest_overlap(filled, merged[index])
        if part is not None and part.area >= merged[index].area:
            merged[index] = Polygon(part.exterior)
            stats["rooms_filled_to_walls"] += 1
    return merged, stats


def _enclosed_cells(walls: Any, upm: float) -> list[Polygon]:
    """Free cells completely enclosed by wall bodies (the holes of the wall union)."""
    cells: list[Polygon] = []
    if walls is None or walls.is_empty:
        return cells
    parts = [part for part in _parts(walls) if isinstance(part, Polygon)]
    for part in parts:
        for ring in part.interiors:
            cell = Polygon(ring)
            # Islands (free-standing wall bodies) inside the cell are not floor.
            cell = cell.difference(walls)
            for piece in _parts(cell):
                if isinstance(piece, Polygon) and piece.area >= 0.5 * upm * upm:
                    cells.append(piece)
    return cells


def update_opening_symbols(layers: dict[str, list[dict[str, Any]]], openings: list[Any], slabs: list[tuple[float, float]], units_per_metre: float) -> int:
    """Move door hinges, window frames and jamb lines onto the measured wall thickness."""
    updated = 0
    upm = max(float(units_per_metre), 1e-9)
    tolerance = 0.05 * upm
    for opening, (low, high) in zip(openings, slabs, strict=True):
        if high - low <= 1e-9:
            continue
        axis = _axis_of(opening)
        normal = np.asarray((-axis[1], axis[0]))
        centre = np.asarray(getattr(opening, "center"), dtype=float)[:2]
        width = float(getattr(opening, "width"))
        for layer in ("A-DOOR", "A-WIND"):
            for entity in layers.get(layer, []):
                own_centre = entity.get("_opening_center")
                if own_centre is None or float(np.linalg.norm(np.asarray(own_centre[:2], dtype=float) - centre)) > tolerance:
                    continue
                if entity.get("type") == "insert" and entity.get("name") == "WINDOW":
                    entity["point"] = (centre - axis * width / 2.0 + normal * low).tolist()
                    entity["rotation"] = math.degrees(math.atan2(axis[1], axis[0]))
                    entity["yscale"] = float(high - low)
                    updated += 1
                elif entity.get("type") == "insert" and entity.get("name") == "DOOR":
                    rotation = math.radians(float(entity.get("rotation", 0.0)))
                    leaf = np.asarray((-math.sin(rotation), math.cos(rotation))) * (1.0 if float(entity.get("yscale", 1.0)) >= 0.0 else -1.0)
                    hinge = np.asarray(entity["point"][:2], dtype=float)
                    along = float(np.dot(hinge - centre, axis))
                    face = high if float(np.dot(leaf, normal)) > 0.0 else low
                    entity["point"] = (centre + axis * along + normal * face).tolist()
                    updated += 1
                elif entity.get("type") == "line" and (entity.get("_jamb") or entity.get("_mullion")):
                    start = np.asarray(entity["start"][:2], dtype=float)
                    along = float(np.dot(start - centre, axis))
                    entity["start"] = (centre + axis * along + normal * low).tolist()
                    entity["end"] = (centre + axis * along + normal * high).tolist()
                    updated += 1
                elif entity.get("type") == "text" and entity.get("_opening_label"):
                    point = np.asarray(entity["point"][:2], dtype=float)
                    across = float(np.dot(point - centre, normal))
                    if low - 0.02 * upm <= across <= high + 0.02 * upm:
                        target = high + 0.12 * upm if across >= (low + high) / 2.0 else low - 0.12 * upm
                        entity["point"] = (point + normal * (target - across)).tolist()
                        updated += 1
    return updated


# --------------------------------------------------------------------------
# faces and sides
# --------------------------------------------------------------------------

def _collect_faces(entities: list[dict[str, Any]], upm: float, minimum_m: float = 0.02) -> tuple[list[_Face], list[tuple[int, dict[str, Any]]]]:
    faces: list[_Face] = []
    passthrough: list[tuple[int, dict[str, Any]]] = []
    for index, entity in enumerate(entities):
        if entity.get("type") != "polyline" or entity.get("closed"):
            passthrough.append((index, entity))
            continue
        points = [np.asarray(point[:2], dtype=float) for point in entity.get("points", [])]
        if len(points) < 2:
            passthrough.append((index, entity))
            continue
        for first, second in zip(points[:-1], points[1:]):
            # Fragments shorter than ``minimum_m`` are covered by the neighbouring wall body.
            if float(np.linalg.norm(second - first)) >= minimum_m * upm:
                faces.append(_Face(len(faces), first, second, entity))
    return faces, passthrough


def _align_collinear(faces: list[_Face], upm: float, settings: DoubleWallSettings) -> int:
    """Move nearly collinear faces of one wall line (same solid side) onto their length-weighted line."""
    groups = _collinear_groups(faces, upm, settings)
    members: dict[int, list[_Face]] = {}
    for face in faces:
        members.setdefault(groups[face.index], []).append(face)
    moved = 0
    for group in members.values():
        if len(group) < 2:
            continue
        reference = max(group, key=lambda item: item.length)
        normal = reference.normal
        offsets = [float(np.dot((face.start + face.end) / 2.0 - reference.start, normal)) for face in group]
        weights = [face.length for face in group]
        target = float(np.average(offsets, weights=weights))
        for face, offset in zip(group, offsets, strict=True):
            shift = target - offset
            if abs(shift) > 1e-9:
                face.start = face.start + normal * shift
                face.end = face.end + normal * shift
                moved += 1
    return moved


def _opening_changes(before: list[Any], after: list[Any]) -> list[tuple[Any, Any]]:
    """(old, new) pairs: replaced openings, removed (new None) and added (old None) ones."""
    changes: list[tuple[Any, Any]] = []
    after_ids = {id(item) for item in after}
    before_ids = {id(item) for item in before}
    replaced = {id(getattr(item, "_replaces", None)) for item in after}
    for item in before:
        if id(item) not in after_ids and id(item) not in replaced:
            changes.append((item, None))
    for item in after:
        if id(item) in before_ids:
            continue
        changes.append((getattr(item, "_replaces", None), item))
    return changes


def _neighbour_slab(centre: np.ndarray, axis: np.ndarray, half: float, runs: list[WallRun], face_of: dict[int, _Face], upm: float, settings: DoubleWallSettings) -> tuple[float, float] | None:
    """Slab (offsets across the wall) of an opening that no wall run hosts: the wall continues the nearest parallel run.

    Used for an opening that fills a whole short wall segment (no wall data inside the gap, so no run covers it): the
    collinear run that ends beside the gap decides which slab the opening cuts.  The run must lie on the opening's line
    (within the wall thickness) and end within ``jamb_reach_m`` + 0.15 m of the span.  None when no run qualifies.
    """
    normal = np.asarray((-axis[1], axis[0]))
    limit = settings.max_thickness_m * upm
    best: tuple[float, float, float] | None = None
    for run in runs:
        face = face_of[run.face]
        if not _parallel(face.direction, axis, 10.0):
            continue
        offset = float(np.dot(centre - face.start, face.normal))
        if abs(offset) > limit + run.thickness:
            continue
        along = float(np.dot(centre - face.start, face.direction))
        gap = max(run.start - run.extend_start - (along + half), (along - half) - (run.end + run.extend_end), 0.0)
        if gap > (settings.jamb_reach_m + 0.15) * upm:
            continue
        face_across = float(np.dot(face.start - centre, normal))
        far_across = float(np.dot(face.point(0.0, run.thickness) - centre, normal))
        low, high = min(face_across, far_across), max(face_across, far_across)
        distance = abs(offset) + gap  # the run nearest to the opening (across and along) decides
        if best is None or distance < best[0]:
            best = (distance, low, high)
    if best is None or best[2] - best[1] < 0.05 * upm:
        return None
    return best[1], best[2]


def _bit_heights(bits: np.ndarray, bins: int, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Covered height and top height of height-bin bitmasks."""
    values = np.asarray(bits, dtype=np.uint32)
    occupied = np.stack([((values >> np.uint32(index)) & np.uint32(1)).astype(bool) for index in range(bins)], axis=-1)
    count = occupied.sum(axis=-1)
    top = np.where(occupied.any(axis=-1), bins - np.argmax(occupied[..., ::-1], axis=-1), 0)
    step = height / max(bins, 1)
    return count * step, top * step


def _filter_faces_by_evidence(faces: list[_Face], evidence: WallEvidence, valid: np.ndarray, furniture: np.ndarray | None, grid: RasterGrid, upm: float, settings: DoubleWallSettings) -> tuple[list[_Face], dict[str, Any]]:
    """Wall-height test and camera visibility test.

    Height test: a face is rejected when the measured data along it does not form a
    wall (wall points over ``face_min_height_m`` reaching ``face_reach_m``, or a
    floor-connected surface reaching ``face_reach_m``) AND there is measured data
    right behind it on both sides AND furniture or fixtures directly at it - the
    front of a counter, cabinet or stub, not a wall whose body is empty of points.
    """
    stats: dict[str, Any] = {"double_wall_faces_low": 0, "double_wall_faces_low_m": 0.0, "double_wall_faces_unseen": 0, "double_wall_faces_unseen_m": 0.0}
    exempt = {"niche_panel", "builtin_front", "room_edge"}
    kept = list(faces)
    if settings.face_test and evidence.wall_bits is not None and evidence.storey_height > 0.0:
        survivors = []
        for face in kept:
            if face.entity.get("_source") in exempt:
                survivors.append(face)
                continue
            share = _face_height_share(face, evidence, grid, upm, settings)
            if share >= settings.face_min_share or not _data_on_both_sides(face, valid, grid, upm) or not _furniture_at(face, furniture, grid, upm):
                survivors.append(face)
            else:
                stats["double_wall_faces_low"] += 1
                stats["double_wall_faces_low_m"] += face.length / upm
                stats.setdefault("double_wall_faces_low_list", []).append([np.round(face.start, 2).tolist(), np.round(face.end, 2).tolist(), round(share, 2), face.entity.get("_source")])
        kept = survivors
    cameras = evidence.cameras
    stats["double_wall_cameras"] = 0 if cameras is None else int(len(cameras))
    if settings.visibility_test and cameras is not None and len(cameras):
        lines = np.asarray([[face.start, face.end] for face in kept], dtype=float).reshape(-1, 2, 2)
        survivors = []
        for index, face in enumerate(kept):
            if face.entity.get("_source") in exempt:
                survivors.append(face)
                continue
            share = _visible_share(face, index, lines, np.asarray(cameras, dtype=float), upm, settings)
            if share >= settings.visibility_min_share:
                survivors.append(face)
            else:
                stats.setdefault("double_wall_faces_unseen_list", []).append([np.round(face.start, 2).tolist(), np.round(face.end, 2).tolist(), round(share, 2)])
                stats["double_wall_faces_unseen"] += 1
                stats["double_wall_faces_unseen_m"] += face.length / upm
        kept = survivors
    for number, face in enumerate(kept):
        face.index = number
    return kept, stats


def _furniture_at(face: _Face, furniture: np.ndarray | None, grid: RasterGrid, upm: float) -> bool:
    if furniture is None:
        return True
    along = _sample_positions(face, 0.05 * upm, min(0.03 * upm, face.length / 4.0))
    hits = np.zeros(len(along), dtype=bool)
    for offset in (-0.20, -0.12, -0.06, 0.06, 0.12, 0.20):
        points = face.start[None, :] + along[:, None] * face.direction[None, :] + offset * upm * face.normal[None, :]
        hits |= _lookup(furniture, grid, points)
    return float(np.mean(hits)) >= 0.4


def _data_on_both_sides(face: _Face, valid: np.ndarray, grid: RasterGrid, upm: float) -> bool:
    along = _sample_positions(face, 0.05 * upm, min(0.03 * upm, face.length / 4.0))
    shares = []
    for sign in (1.0, -1.0):
        hits = np.zeros(len(along), dtype=bool)
        # Beyond the few centimetres that fused surface points spread behind a face.
        for offset in (0.07, 0.09, 0.11):
            points = face.start[None, :] + along[:, None] * face.direction[None, :] + sign * offset * upm * face.normal[None, :]
            hits |= _lookup(valid, grid, points)
        shares.append(float(np.mean(hits)))
    return min(shares) >= 0.6


def _face_height_share(face: _Face, evidence: WallEvidence, grid: RasterGrid, upm: float, settings: DoubleWallSettings) -> float:
    """Share of the face where measured data supports a wall; 1.0 when there is too little data to judge.

    A sample passes when wall-class points cover ``face_min_height_m`` and reach
    ``face_reach_m``, or when a floor-connected vertical surface of any class (a wall
    behind curtains or glazing) reaches ``face_reach_m``; the surface may stand up to
    10 cm in front of or behind the fitted face.  A sample with data that does neither
    (a counter, an upper cabinet, a sideboard front) counts against the face.
    """
    along = _sample_positions(face, 0.05 * upm, min(0.03 * upm, face.length / 4.0))
    offsets = np.linspace(-0.10, 0.10, 11)
    wall = np.zeros(len(along), dtype=np.uint32)
    any_class = np.zeros(len(along), dtype=np.uint32)
    passed = np.zeros(len(along), dtype=bool)
    for offset in offsets:
        points = face.start[None, :] + along[:, None] * face.direction[None, :] + offset * upm * face.normal[None, :]
        wall_here = _lookup(evidence.wall_bits, grid, points, 0).astype(np.uint32)
        wall |= wall_here
        if evidence.all_bits is not None:
            here = _lookup(evidence.all_bits, grid, points, 0).astype(np.uint32)
            any_class |= here
            # Each lateral position on its own: a floor-to-ceiling surface (curtain, glazing frame).
            covered, top = _bit_heights(here, evidence.bins, evidence.storey_height)
            passed |= ((here & np.uint32(0b11)) > 0) & (top >= settings.face_reach_m * upm) & (covered >= 0.7 * top)
    covered, top = _bit_heights(wall, evidence.bins, evidence.storey_height)
    passed |= (covered >= settings.face_min_height_m * upm) & (top >= settings.face_reach_m * upm)
    has_data = (wall | any_class) > 0
    judged = int(np.count_nonzero(has_data | passed))
    if judged < 0.30 * len(along):
        return 1.0
    return float(np.count_nonzero(passed)) / max(judged, 1)


def _visible_share(face: _Face, index: int, lines: np.ndarray, cameras: np.ndarray, upm: float, settings: DoubleWallSettings) -> float:
    """Share of face samples that some camera sees (from either side) without crossing another face."""
    along = _sample_positions(face, 0.25 * upm, min(0.10 * upm, face.length / 4.0))
    others = np.delete(lines, index, axis=0)
    seen = np.zeros(len(along), dtype=bool)
    for side in (1.0, -1.0):
        targets = face.start[None, :] + along[:, None] * face.direction[None, :] + side * 0.05 * upm * face.normal[None, :]
        for k, target in enumerate(targets):
            if seen[k]:
                continue
            vectors = cameras - target[None, :]
            distance = np.linalg.norm(vectors, axis=1)
            facing = (vectors @ (side * face.normal)) > 0.0
            candidates = cameras[facing & (distance <= settings.visibility_max_m * upm) & (distance > 0.05 * upm)]
            if not len(candidates):
                continue
            if not len(others):
                seen[k] = True
                continue
            blocked = _segments_blocked(candidates, target, others)
            seen[k] = bool(np.any(~blocked))
    return float(np.mean(seen)) if len(seen) else 0.0


def _segments_blocked(starts: np.ndarray, end: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """For each segment start->end, whether it properly crosses any of ``lines`` (M x 2 x 2)."""
    p = starts[:, None, :]
    r = (end[None, :] - starts)[:, None, :]
    q = lines[None, :, 0, :]
    s = (lines[:, 1, :] - lines[:, 0, :])[None, :, :]
    denominator = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / denominator
        u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / denominator
    hit = (np.abs(denominator) > 1e-12) & (t > 1e-6) & (t < 1.0 - 1e-6) & (u >= 0.0) & (u <= 1.0)
    return hit.any(axis=1)


def _replace_opening(opening: Any, **changes: Any) -> Any:
    from dataclasses import replace as dataclass_replace

    try:
        new = dataclass_replace(opening, **changes)
    except TypeError:
        return opening
    try:
        object.__setattr__(new, "_replaces", getattr(opening, "_replaces", opening))
    except (AttributeError, TypeError):
        pass
    return new


def _resolve_overlapping_openings(openings: list[Any], upm: float) -> list[Any]:
    """A window overlapping a door in the same wall keeps only its part beside the door."""
    output = list(openings)
    for door_index, door in enumerate(output):
        if door is None or str(getattr(door, "kind", "")) not in {"DOOR", "passage"}:
            continue
        axis = _axis_of(door)
        normal = np.asarray((-axis[1], axis[0]))
        centre = np.asarray(getattr(door, "center"), dtype=float)[:2]
        half = float(getattr(door, "width")) / 2.0
        for index, window in enumerate(output):
            if window is None or index == door_index or str(getattr(window, "kind", "")) != "WINDOW":
                continue
            if not _parallel(axis, _axis_of(window), 10.0):
                continue
            other = np.asarray(getattr(window, "center"), dtype=float)[:2]
            if abs(float(np.dot(other - centre, normal))) > 0.45 * upm:
                continue
            middle = float(np.dot(other - centre, axis))
            low, high = middle - float(getattr(window, "width")) / 2.0, middle + float(getattr(window, "width")) / 2.0
            if high <= -half + 0.05 * upm or low >= half - 0.05 * upm:
                continue
            pieces = [(low, min(high, -half)), (max(low, half), high)]
            pieces = [(a, b) for a, b in pieces if b - a >= 0.40 * upm]
            if not pieces:
                output[index] = None
                continue
            a, b = max(pieces, key=lambda item: item[1] - item[0])
            new_centre = centre + axis * (a + b) / 2.0 + normal * float(np.dot(other - centre, normal))
            output[index] = _replace_opening(window, center=(float(new_centre[0]), float(new_centre[1])), width=float(b - a))
    return [item for item in output if item is not None]


def _leaf_faces_to_doors(faces: list[_Face], openings: list[Any], evidence: WallEvidence, grid: RasterGrid, upm: float, settings: DoubleWallSettings) -> tuple[list[_Face], list[Any]]:
    """A measured face made of door-class points standing at a wall is an open door leaf: it becomes that door."""
    from .indoor import Opening

    if evidence.door is None:
        return faces, []
    kept: list[_Face] = []
    doors: list[Any] = []
    for face in faces:
        if face.entity.get("_source") not in (None, "measured") or not settings.door_min_width_m * upm <= face.length <= settings.door_max_width_m * upm:
            kept.append(face)
            continue
        along = _sample_positions(face, 0.05 * upm, min(0.03 * upm, face.length / 4.0))
        door_hits = np.zeros(len(along), dtype=bool)
        for offset in (-0.05, -0.025, 0.0, 0.025, 0.05):
            door_hits |= _lookup(evidence.door, grid, face.start[None, :] + along[:, None] * face.direction[None, :] + offset * upm * face.normal[None, :])
        if float(np.mean(door_hits)) < 0.30:
            kept.append(face)
            continue
        others = [other for other in faces if other is not face]
        angle = math.atan2(float(face.direction[1]), float(face.direction[0])) % math.pi
        middle = (face.start + face.end) / 2.0
        leaf = Opening("DOOR", (float(middle[0]), float(middle[1])), float(face.length), angle, 0.04 * upm, evidence="leaf-face", leaf_detected=False)
        converted = _reorient_door_leaves([leaf], others, evidence, grid, upm, settings, host_tolerance=0.15, existing=list(openings) + doors)
        if converted and converted[0] is not leaf:
            doors.append(converted[0])
            # A leaf standing free in the room is no wall; a leaf opened flat against a wall
            # (nothing but wall or outside behind it) still marks that wall.
            sides = []
            for sign in (1.0, -1.0):
                hits = np.zeros(len(along), dtype=bool)
                for offset in (0.08, 0.15, 0.25):
                    hits |= _lookup(evidence.floor, grid, face.start[None, :] + along[:, None] * face.direction[None, :] + sign * offset * upm * face.normal[None, :])
                sides.append(float(np.mean(hits)))
            if min(sides) < 0.2:
                kept.append(face)
        else:
            kept.append(face)
    for number, face in enumerate(kept):
        face.index = number
    return kept, doors


def _overlaps_existing(centre: np.ndarray, axis: np.ndarray, width: float, existing: list[Any], upm: float) -> bool:
    for opening in existing:
        other_axis = _axis_of(opening)
        other = np.asarray(getattr(opening, "center"), dtype=float)[:2]
        normal = np.asarray((-axis[1], axis[0]))
        if _parallel(axis, other_axis, 10.0) and abs(float(np.dot(other - centre, normal))) <= 0.45 * upm and abs(float(np.dot(other - centre, axis))) < (width + float(getattr(opening, "width"))) / 2.0 - 0.05 * upm:
            return True
    return False


def _refine_open_span(centre: np.ndarray, direction: np.ndarray, width: float, evidence: WallEvidence, grid: RasterGrid, upm: float, settings: DoubleWallSettings) -> tuple[np.ndarray, float]:
    """Fit a door derived from its leaf to the stretch of the wall line that is open at sill and mid height."""
    if evidence.section is None:
        return centre, width
    normal = np.asarray((-direction[1], direction[0]))
    step = 0.02 * upm
    reach = 0.25 * upm
    positions = np.arange(-width / 2.0 - reach, width / 2.0 + reach + step / 2.0, step)
    blocked = np.zeros(len(positions), dtype=bool)
    for offset in np.linspace(-0.15, 0.15, 7) * upm:
        points = centre[None, :] + positions[:, None] * direction[None, :] + offset * normal[None, :]
        hit = _lookup(evidence.section, grid, points)
        if evidence.low is not None:
            hit &= _lookup(evidence.low, grid, points)
        blocked |= hit
    middle = int(np.argmin(np.abs(positions)))
    if blocked[middle]:
        return centre, width
    low = middle
    while low > 0 and not blocked[low - 1]:
        low -= 1
    high = middle
    while high < len(positions) - 1 and not blocked[high + 1]:
        high += 1
    # Only a stretch closed by wall on both sides is a measured door width.
    if low == 0 or high == len(positions) - 1:
        return centre, width
    a, b = positions[low] - step / 2.0, positions[high] + step / 2.0
    if not settings.door_min_width_m * upm <= b - a <= settings.door_max_width_m * upm:
        return centre, width
    return centre + direction * (a + b) / 2.0, float(b - a)


def _reorient_door_leaves(openings: list[Any], faces: list[_Face], evidence: WallEvidence, grid: RasterGrid, upm: float, settings: DoubleWallSettings, host_tolerance: float = 0.25, existing: list[Any] | None = None) -> list[Any]:
    """A door detected as its open leaf (perpendicular to a wall, one end at the wall) becomes the opening in that wall.

    The leaf of an open door stands at 90 degrees to the wall at the hinge jamb.  The
    opening is the wall span next to the hinge, as long as the leaf, on the side where
    the wall is open at mid height.
    """
    output: list[Any] = []
    for opening in openings:
        if str(getattr(opening, "kind", "")) not in {"DOOR", "passage"}:
            output.append(opening)
            continue
        axis = _axis_of(opening)
        centre = np.asarray(getattr(opening, "center"), dtype=float)[:2]
        half = float(getattr(opening, "width")) / 2.0
        hosted = False
        for face in faces:
            if _parallel(face.direction, axis, 10.0):
                offset = abs(float(np.dot(centre - face.start, face.normal)))
                along = float(np.dot(centre - face.start, face.direction))
                # In a wall line (also beside a wall piece of it): a real opening, not a leaf.
                if offset <= host_tolerance * upm and -half - 1.0 * upm <= along <= face.length + half + 1.0 * upm:
                    hosted = True
                    break
        if hosted:
            output.append(opening)
            continue
        others_open = [item for item in (existing if existing is not None else openings) if item is not opening]
        candidates: list[tuple[float, float, np.ndarray, float, np.ndarray]] = []
        for face in faces:
            if _parallel(face.direction, axis, 80.0):
                continue
            for end_sign in (-1.0, 1.0):
                end = centre + end_sign * axis * half
                far = centre - end_sign * axis * half
                across = float(np.dot(end - face.start, face.normal))
                if abs(across) > 0.25 * upm:
                    continue
                hinge = end - face.normal * across
                along_hinge = float(np.dot(hinge - face.start, face.direction))
                if along_hinge < -0.30 * upm or along_hinge > face.length + 0.30 * upm:
                    continue
                width = float(min(max(float(np.linalg.norm(far - hinge)), settings.door_min_width_m * upm), settings.door_max_width_m * upm))
                for sign in (-1.0, 1.0):
                    span = [along_hinge + sign * value for value in (0.0, width)]
                    if min(span) < -0.10 * upm or max(span) > face.length + 0.10 * upm:
                        continue  # the opening must lie in the wall the leaf stands at
                    new_centre = hinge + sign * face.direction * width / 2.0
                    if _overlaps_existing(new_centre, face.direction, width, others_open, upm):
                        continue
                    points = hinge[None, :] + sign * (np.linspace(0.05, 0.95, 10) * width)[:, None] * face.direction[None, :]
                    occupied = np.zeros(len(points), dtype=bool)
                    if evidence.section is not None:
                        for offset in (-0.04, 0.0, 0.04):
                            occupied |= _lookup(evidence.section, grid, points + offset * upm * face.normal[None, :])
                    candidates.append((float(np.mean(occupied)), abs(across), new_centre, width, face.direction))
        if not candidates:
            output.append(opening)
            continue
        occupancy, _distance, new_centre, width, direction = min(candidates, key=lambda item: (item[0], item[1]))
        if occupancy > 0.5:
            output.append(opening)
            continue
        new_centre, width = _refine_open_span(new_centre, direction, width, evidence, grid, upm, settings)
        angle = math.atan2(float(direction[1]), float(direction[0])) % math.pi
        output.append(_replace_opening(opening, center=(float(new_centre[0]), float(new_centre[1])), width=width, axis_angle=angle, leaf_detected=False, hinge_side=None, swing_side=None, evidence=f"{getattr(opening, 'evidence', '')}+leaf"))
    return output


def _close_spans(spans: list[dict[str, Any]], runs: list[WallRun], faces: list[_Face], upm: float, reach_m: float = 1.5) -> int:
    """Continue the host wall across spans that are no opening (a gap between the measured faces); returns how many were closed.

    The host is the nearest wall run parallel to the span and standing in its line.  A span the runs already cover (at least
    80 %, e.g. by an invented room-edge wall) is left alone.
    """
    face_of = {face.index: face for face in faces}
    reach = reach_m * upm
    closed = 0
    for span in spans:
        centre = np.asarray(span["center"], dtype=float)[:2]
        axis = np.asarray((math.cos(float(span["axis_angle"])), math.sin(float(span["axis_angle"]))))
        half = float(span["width"]) / 2.0
        candidates: list[tuple[float, WallRun, _Face, float, float]] = []
        covered: list[LineString] = []
        for run in runs:
            face = face_of[run.face]
            if run.thickness <= 0.0 or not _parallel(face.direction, axis, 10.0):
                continue
            across = float(np.dot(centre - face.start, face.solid_normal))
            if not -0.10 * upm <= across <= run.thickness + 0.10 * upm:
                continue
            ends = sorted(float(np.dot(face.point(position) - centre, axis)) for position in (run.start, run.end))
            overlap = (max(ends[0], -half), min(ends[1], half))
            if overlap[1] > overlap[0]:
                covered.append(LineString([(overlap[0], 0.0), (overlap[1], 0.0)]))
            a = float(np.dot(centre - axis * half - face.start, face.direction))
            b = float(np.dot(centre + axis * half - face.start, face.direction))
            low, high = min(a, b), max(a, b)
            candidates.append((max(run.start - high, low - run.end, 0.0), run, face, low, high))
        if not candidates or (covered and unary_union(covered).length >= 0.8 * 2.0 * half):
            continue
        distance, run, face, low, high = min(candidates, key=lambda item: item[0])
        if distance > reach:
            continue
        low, high = low - 0.02 * upm, high + 0.02 * upm
        filler = _Face(len(faces), face.point(low), face.point(high), {"type": "polyline", "points": [], "closed": False, "_source": "door_fill"}, side=face.side)
        filler.start, filler.end = face.start + face.direction * low, face.start + face.direction * high
        faces.append(filler)
        face_of[filler.index] = filler
        runs.append(WallRun(filler.index, 0.0, filler.length, run.thickness, run.kind, "door-fill", group=run.group))
        closed += 1
    return closed


def _run_centre(run: WallRun, face: _Face, along: float) -> np.ndarray:
    return face.point(along, run.thickness / 2.0)


def _segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    vector = b - a
    norm2 = float(vector @ vector)
    t = 0.0 if norm2 <= 0.0 else float(np.clip((point - a) @ vector / norm2, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + vector * t)))


def _drop_leaf_runs(runs: list[WallRun], faces: list[_Face], openings: list[Any], upm: float, settings: DoubleWallSettings) -> int:
    """Remove wall runs that are the measured open leaf of a door (a leaf is evidence for hinge/swing, never a wall); returns how many."""
    leaves = []
    for opening in openings:
        hinge, free = getattr(opening, "leaf_hinge", None), getattr(opening, "leaf_free", None)
        if str(getattr(opening, "kind", "")) == "DOOR" and hinge is not None and free is not None:
            leaves.append((np.asarray(hinge, dtype=float), np.asarray(free, dtype=float)))
    if not leaves:
        return 0
    face_of = {face.index: face for face in faces}
    reach = settings.leaf_run_distance_m * upm
    dropped = 0
    for run in list(runs):
        face = face_of[run.face]
        a, b = _run_centre(run, face, run.start), _run_centre(run, face, run.end)
        length = float(np.linalg.norm(b - a))
        for hinge, free in leaves:
            if length > float(np.linalg.norm(free - hinge)) + settings.leaf_run_extra_m * upm:
                continue
            if _segment_distance(a, hinge, free) <= reach and _segment_distance(b, hinge, free) <= reach:
                runs.remove(run)
                dropped += 1
                break
    return dropped


def _enforce_door_spans(runs: list[WallRun], faces: list[_Face], openings: list[Any], upm: float, settings: DoubleWallSettings) -> dict[str, int]:
    """A door is never a wall: every door / passage has exactly one host wall run and nothing else stands in its span.

    Per opening: the host is the parallel run whose centreline passes the opening centre (smallest offset, then longest cover of the span).
    Other parallel runs (a duplicate / learned run beside the host) are cut back to the jambs, perpendicular runs that reach into the span
    (an open leaf, a stub) are cut back to the span edge.  Without a host a door-host piece between the jambs is built.
    """
    counts = {"hosts_built": 0, "runs_cut": 0, "runs_removed": 0}
    face_of = {face.index: face for face in faces}
    min_piece = settings.door_min_piece_m * upm
    for opening in openings:
        if str(getattr(opening, "kind", "")) not in ("DOOR", "passage"):
            continue
        axis = _axis_of(opening)
        normal = np.asarray((-axis[1], axis[0]))
        centre = np.asarray(getattr(opening, "center"), dtype=float)[:2]
        half = float(getattr(opening, "width")) / 2.0
        depth = max(float(getattr(opening, "wall_thickness", 0.0)), 0.1 * upm) / 2.0
        parallel: list[tuple[float, WallRun, _Face, float, float, float]] = []  # (distance to the centre, run, face, span low, span high, offset)
        for run in runs:
            face = face_of[run.face]
            if run.thickness <= 0.0 or not _parallel(face.direction, axis, 10.0):
                continue
            left = np.asarray((-face.direction[1], face.direction[0]))
            first, last = _run_centre(run, face, run.start), _run_centre(run, face, run.end)
            offset = float(np.dot(centre - first, left))
            distance = float(LineString([tuple(first), tuple(last)]).distance(Point(float(centre[0]), float(centre[1]))))
            a = float(np.dot(centre - axis * half - face.start, face.direction))
            b = float(np.dot(centre + axis * half - face.start, face.direction))
            # same host criterion as the plan model: the centreline passes the opening centre within half its width plus the wall body
            if abs(offset) <= run.thickness / 2.0 + 0.05 * upm and distance <= half + run.thickness / 2.0 + 0.05 * upm:
                parallel.append((distance, run, face, min(a, b), max(a, b), offset))
        if not parallel:
            nearest = None
            for run in runs:
                face = face_of[run.face]
                if run.thickness <= 0.0 or not _parallel(face.direction, axis, 10.0):
                    continue
                gap = float(LineString([tuple(_run_centre(run, face, run.start)), tuple(_run_centre(run, face, run.end))]).distance(Point(float(centre[0]), float(centre[1]))))
                if gap <= settings.door_fill_reach_m * upm and (nearest is None or gap < nearest[0]):
                    nearest = (gap, run)
            thickness = nearest[1].thickness if nearest is not None else max(float(getattr(opening, "wall_thickness", 0.0)), 0.15 * upm)
            kind = nearest[1].kind if nearest is not None else "interior"
            start = centre - axis * (half + 0.02 * upm) - normal * thickness / 2.0
            end = centre + axis * (half + 0.02 * upm) - normal * thickness / 2.0
            filler = _Face(len(faces), start, end, {"type": "polyline", "points": [], "closed": False, "_source": "door_host"}, side=1)
            faces.append(filler)
            face_of[filler.index] = filler
            runs.append(WallRun(filler.index, 0.0, filler.length, thickness, kind, "door-host"))
            counts["hosts_built"] += 1
            continue
        host = min(parallel, key=lambda item: (round(item[0], 3), abs(item[5])))
        host_normal = np.asarray((-host[2].direction[1], host[2].direction[0]))
        host_line = float(np.dot(_run_centre(host[1], host[2], host[1].start), host_normal))
        for _, run, face, low, high, _ in parallel:
            if run is host[1] or max(0.0, min(run.end, high) - max(run.start, low)) <= 0.0:
                continue  # not in the span: a piece beside the opening
            if abs(float(np.dot(_run_centre(run, face, run.start), host_normal)) - host_line) <= settings.door_host_collinear_m * upm:
                continue  # a collinear piece of the host wall: the opening cut ends it at the jamb
            runs.remove(run)
            kept = [(lo, hi) for lo, hi in ((run.start, min(run.end, low)), (max(run.start, high), run.end)) if hi - lo >= min_piece]
            counts["runs_cut" if kept else "runs_removed"] += 1
            for lo, hi in kept:
                runs.append(WallRun(run.face, lo, hi, run.thickness, run.kind, run.evidence, group=run.group, extend_start=run.extend_start if lo == run.start else 0.0, extend_end=run.extend_end if hi == run.end else 0.0))
        zone = Polygon([tuple(centre - axis * (half - 0.05 * upm) - normal * depth), tuple(centre + axis * (half - 0.05 * upm) - normal * depth), tuple(centre + axis * (half - 0.05 * upm) + normal * depth), tuple(centre - axis * (half - 0.05 * upm) + normal * depth)])
        for run in list(runs):
            face = face_of[run.face]
            if run.thickness <= 0.0 or _parallel(face.direction, axis, 30.0):
                continue
            segment = LineString([tuple(_run_centre(run, face, run.start)), tuple(_run_centre(run, face, run.end))])
            inside = segment.intersection(zone)
            if inside.is_empty or inside.length <= settings.door_span_enter_m * upm:
                continue
            # only a run that BEGINS inside the span (hangs from the jamb into the opening: an open leaf, a stub) is cut; a wall that merely
            # passes the lower or upper end of the span (a door at a wall junction) is a real crossing wall
            reach_zone = zone.buffer(0.05 * upm)
            if not any(reach_zone.contains(Point(float(end[0]), float(end[1]))) for end in segment.coords):
                continue
            runs.remove(run)
            counts["runs_removed"] += 1
            length = run.end - run.start
            first = float(segment.project(Point(*inside.coords[0]))) if inside.geom_type == "LineString" else 0.0
            last = float(segment.project(Point(*inside.coords[-1]))) if inside.geom_type == "LineString" else length
            lo_in, hi_in = min(first, last), max(first, last)
            for lo, hi in ((0.0, lo_in), (hi_in, length)):
                if hi - lo >= min_piece:
                    runs.append(WallRun(run.face, run.start + lo, run.start + hi, run.thickness, run.kind, run.evidence, group=run.group))
                    counts["runs_removed"] -= 1
                    counts["runs_cut"] += 1
    return counts


def _free_ends(runs: list[WallRun], faces: list[_Face], openings: list[Any], upm: float, settings: DoubleWallSettings) -> list[tuple[WallRun, int, np.ndarray, np.ndarray]]:
    """(run, end 0/1, centreline end point, outward direction) of every run end that touches no other wall and no opening jamb."""
    face_of = {face.index: face for face in faces}
    centre_lines = []
    for run in runs:
        face = face_of[run.face]
        centre_lines.append((run, LineString([tuple(_run_centre(run, face, run.start)), tuple(_run_centre(run, face, run.end))])))
    jambs = []
    for opening in openings:
        axis = _axis_of(opening)
        centre = np.asarray(getattr(opening, "center"), dtype=float)[:2]
        half = float(getattr(opening, "width")) / 2.0
        jambs.extend([centre - axis * half, centre + axis * half])
    free = []
    for run, line in centre_lines:
        face = face_of[run.face]
        reach = run.thickness / 2.0 + settings.free_end_reach_m * upm
        for end, along in enumerate((run.start, run.end)):
            point = _run_centre(run, face, along)
            probe = Point(float(point[0]), float(point[1]))
            if any(other is not run and probe.distance(other_line) <= reach for other, other_line in centre_lines):
                continue
            if any(float(np.linalg.norm(point - jamb)) <= reach + 0.10 * upm for jamb in jambs):
                continue
            free.append((run, end, point, face.direction * (-1.0 if end == 0 else 1.0)))
    return free


def _merge_jogs(runs: list[WallRun], faces: list[_Face], upm: float, settings: DoubleWallSettings) -> int:
    """Straighten a room-edge run that is offset by a small step from a longer parallel neighbour it touches or overlaps."""
    face_of = {face.index: face for face in faces}
    limit = settings.jog_merge_m * upm
    merged = 0
    uses: dict[int, int] = {}
    for run in runs:
        uses[run.face] = uses.get(run.face, 0) + 1
    for run in sorted(runs, key=lambda item: item.end - item.start):
        if run.evidence != "room-edge" or uses.get(run.face, 0) != 1:
            continue
        face = face_of[run.face]
        a, b = _run_centre(run, face, run.start), _run_centre(run, face, run.end)
        length = float(np.linalg.norm(b - a))
        best = None
        for other in runs:
            if other is run:
                continue
            other_face = face_of[other.face]
            if not _parallel(face.direction, other_face.direction, 5.0):
                continue
            oa, ob = _run_centre(other, other_face, other.start), _run_centre(other, other_face, other.end)
            if float(np.linalg.norm(ob - oa)) < length:
                continue
            offset = float(np.dot(oa - a, face.normal))
            if not 0.04 * upm < abs(offset) <= limit:
                continue
            lo, hi = sorted((float(np.dot(oa - a, face.direction)), float(np.dot(ob - a, face.direction))))
            if max(lo - length, -hi, 0.0) > 0.30 * upm:
                continue
            if best is None or abs(offset) < abs(best):
                best = offset
        if best is None:
            continue
        shift = face.normal * best
        face.start, face.end = face.start + shift, face.end + shift
        merged += 1
    return merged


def _close_free_ends(runs: list[WallRun], faces: list[_Face], openings: list[Any], upm: float, settings: DoubleWallSettings) -> int:
    """Join two free ends that face each other across a small step with a connecting wall piece; returns the pieces added."""
    free = _free_ends(runs, faces, openings, upm, settings)
    used: set[tuple[int, int]] = set()
    added = 0
    limit, align = settings.close_step_max_m * upm, settings.close_step_align_m * upm
    for i, (run_a, end_a, point_a, direction_a) in enumerate(free):
        if (id(run_a), end_a) in used:
            continue
        step = np.asarray((-direction_a[1], direction_a[0]))
        best = None
        for run_b, end_b, point_b, direction_b in free[i + 1:]:
            if run_b is run_a or (id(run_b), end_b) in used or not _parallel(direction_a, direction_b, 10.0):
                continue
            delta = point_b - point_a
            along, across = abs(float(np.dot(delta, direction_a))), abs(float(np.dot(delta, step)))
            if along > align or not 0.04 * upm < across <= limit:
                continue
            if best is None or across < best[0]:
                best = (across, run_b, end_b, point_b)
        if best is None:
            continue
        across, run_b, end_b, point_b = best
        thickness = (run_a.thickness + run_b.thickness) / 2.0
        sign = 1.0 if float(np.dot(point_b - point_a, step)) >= 0.0 else -1.0
        e = step * sign  # from the centreline of A to the centreline of B
        n = np.asarray((-e[1], e[0]))
        midpoint_a = point_a + direction_a * (float(np.dot(point_b - point_a, direction_a)) / 2.0)  # between the two ends along the walls
        base = midpoint_a - e * (thickness / 2.0)  # the piece overlaps both walls by half a thickness: no notch at the corners
        face = _Face(len(faces), base - n * thickness / 2.0, base + e * (across + thickness) - n * thickness / 2.0, {"type": "polyline", "points": [], "closed": False, "_source": "wall_closure"}, side=1)
        faces.append(face)
        runs.append(WallRun(face.index, 0.0, face.length, thickness, run_a.kind, "closure"))
        used.update({(id(run_a), end_a), (id(run_b), end_b)})
        added += 1
    return added


def _reject_solid_doors(openings: list[Any], runs: list[WallRun], faces: list[_Face], evidence: WallEvidence, grid: RasterGrid, upm: float) -> tuple[list[Any], int]:
    """Drop a door whose span is solid wall at sill and mid height (a gap between face pieces, not a doorway).

    Door-class points (a closed leaf) do not count as solid.  Only doors lying in a wall body
    are judged; the wall slab is sampled across its whole thickness.
    """
    if evidence.section is None:
        return list(openings), 0
    face_of = {face.index: face for face in faces}
    kept: list[Any] = []
    rejected = 0
    for opening in openings:
        if str(getattr(opening, "kind", "")) != "DOOR" or str(getattr(opening, "evidence", "")).endswith("leaf"):
            kept.append(opening)
            continue
        axis = _axis_of(opening)
        centre = np.asarray(getattr(opening, "center"), dtype=float)[:2]
        half = float(getattr(opening, "width")) / 2.0
        host = None
        for run in runs:
            face = face_of[run.face]
            if run.thickness <= 0.0 or not _parallel(face.direction, axis, 10.0):
                continue
            across = float(np.dot(centre - face.start, face.solid_normal))
            if -0.10 * upm <= across <= run.thickness + 0.10 * upm:
                host = (face, run)
                break
        if host is None:
            kept.append(opening)
            continue
        face, run = host
        along = np.linspace(-0.8, 0.8, 9) * half
        base = centre[None, :] + along[:, None] * axis[None, :] - float(np.dot(centre - face.start, face.solid_normal)) * face.solid_normal[None, :]
        solid = np.zeros(len(along), dtype=bool)
        for offset in np.arange(-0.03 * upm, run.thickness + 0.035 * upm, 0.01 * upm):
            points = base + offset * face.solid_normal[None, :]
            hit = _lookup(evidence.section, grid, points)
            if evidence.low is not None:
                hit &= _lookup(evidence.low, grid, points)
            if evidence.door is not None:
                hit &= ~_lookup(evidence.door, grid, points)
            solid |= hit
        if float(np.mean(solid)) >= 0.7:
            rejected += 1
            # The span is wall: close it with the host wall's body (the faces had a gap there).
            a = float(np.dot(centre - axis * half - face.start, face.direction))
            b = float(np.dot(centre + axis * half - face.start, face.direction))
            low, high = min(a, b) - 0.02 * upm, max(a, b) + 0.02 * upm
            filler = _Face(len(faces), face.point(low), face.point(high), {"type": "polyline", "points": [], "closed": False, "_source": "door_fill"}, side=face.side)
            filler.start, filler.end = face.start + face.direction * low, face.start + face.direction * high
            faces.append(filler)
            face_of[filler.index] = filler
            runs.append(WallRun(filler.index, 0.0, filler.length, run.thickness, run.kind, "door-fill", group=run.group))
            continue
        kept.append(opening)
    return kept, rejected


def _detect_wall_doors(runs: list[WallRun], faces: list[_Face], openings: list[Any], evidence: WallEvidence, grid: RasterGrid, upm: float, settings: DoubleWallSettings) -> list[Any]:
    """Doors in a wall body: nothing at sill and mid height across the wall, floor seen through it.

    A wall (also one only inferred, e.g. behind a coat niche) has points at mid height
    unless something stands in front of it; then its floor line is hidden too.  A
    door-wide stretch with neither sill nor mid-height points across the wall line and
    with measured floor inside the wall body is a doorway.
    """
    from .indoor import Opening

    if evidence.section is None or evidence.floor is None:
        return []
    face_of = {face.index: face for face in faces}
    found: list[Any] = []
    step = 0.05 * upm
    for run in runs:
        face = face_of[run.face]
        if run.thickness <= 0.0 or face.entity.get("_source") == "room_edge":
            continue
        along = np.arange(run.start + step / 2.0, run.end, step)
        if len(along) < 3:
            continue
        base = face.start[None, :] + along[:, None] * face.direction[None, :]
        blocked = np.zeros(len(along), dtype=bool)
        floor = np.zeros(len(along), dtype=bool)
        for fraction in np.linspace(-0.2, 1.2, 8):
            shifted = base + fraction * run.thickness * face.solid_normal[None, :]
            blocked |= _lookup(evidence.section, grid, shifted)
            if evidence.low is not None:
                blocked |= _lookup(evidence.low, grid, shifted)
            if 0.0 < fraction < 1.0:
                floor |= _lookup(evidence.floor, grid, shifted)
        open_column = ~blocked
        # bridge single blocked samples (noise)
        for index in range(1, len(open_column) - 1):
            if not open_column[index] and open_column[index - 1] and open_column[index + 1]:
                open_column[index] = True
        start = None
        for index, flag in enumerate(list(open_column) + [False]):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                first, last = start, index - 1
                start = None
                a, b = along[first] - step / 2.0, along[last] + step / 2.0
                width = b - a
                if not settings.door_min_width_m * upm <= width <= settings.door_max_width_m * upm:
                    continue
                if float(np.mean(floor[first:last + 1])) < settings.door_floor_share:
                    continue
                centre = face.point((a + b) / 2.0, run.thickness / 2.0)
                axis = face.direction
                if _overlaps_existing(centre, axis, width, list(openings) + found, upm):
                    continue
                angle = math.atan2(float(axis[1]), float(axis[0])) % math.pi
                found.append(Opening("DOOR", (float(centre[0]), float(centre[1])), float(width), angle, float(run.thickness), evidence="wall-floor", leaf_detected=False))
    return found


def _adopt_collinear_sides(faces: list[_Face], upm: float, settings: DoubleWallSettings) -> None:
    """An undecided face continues the nearest collinear decided face (same wall line)."""
    sine = math.sin(math.radians(settings.pair_angle_deg))
    for face in faces:
        if face.side:
            continue
        best: tuple[float, _Face] | None = None
        for other in faces:
            if other is face or not other.side:
                continue
            d1, d2 = face.direction, other.direction
            if abs(float(d1[0] * d2[1] - d1[1] * d2[0])) > sine:
                continue
            offset = abs(float(np.dot((other.start + other.end) / 2.0 - face.start, face.normal)))
            if offset > settings.collinear_offset_m * upm:
                continue
            gap = LineString([tuple(face.start), tuple(face.end)]).distance(LineString([tuple(other.start), tuple(other.end)]))
            if best is None or gap < best[0]:
                best = (gap, other)
        if best is not None and best[0] <= 2.0 * upm:
            face.side = 1 if float(np.dot(best[1].solid_normal, face.normal)) > 0 else -1


def _is_outer_face(face: _Face, roomish: np.ndarray, grid: RasterGrid, upm: float) -> bool:
    """True when the room side of a face is outside: neither a ceiling room nor free floor in front of it."""
    if not face.side:
        return False
    along = _sample_positions(face, 0.05 * upm, min(0.05 * upm, face.length / 4.0))
    offsets = np.linspace(0.02, 0.30, 8) * upm
    points = (face.start[None, None, :] + along[:, None, None] * face.direction[None, None, :] - offsets[None, :, None] * face.solid_normal[None, None, :]).reshape(-1, 2)
    return float(np.mean(_lookup(roomish, grid, points))) < 0.05


def _split_reveals(faces: list[_Face], openings: list[Any], upm: float, settings: DoubleWallSettings) -> tuple[list[_Face], list[_Face]]:
    """A short face across an opening edge is the reveal of a door or window, drawn by the wall's jamb."""
    kept: list[_Face] = []
    reveals: list[_Face] = []
    limit = (settings.max_thickness_m + 0.15) * upm
    for face in faces:
        reveal = False
        if face.length <= limit:
            middle = (face.start + face.end) / 2.0
            for opening in openings:
                axis = _axis_of(opening)
                if not _parallel(face.direction, np.asarray((-axis[1], axis[0])), 10.0):
                    continue
                centre = np.asarray(getattr(opening, "center"), dtype=float)[:2]
                half = float(getattr(opening, "width")) / 2.0
                along = float(np.dot(middle - centre, axis))
                across = [float(np.dot(point - centre, np.asarray((-axis[1], axis[0])))) for point in (face.start, face.end)]
                if abs(abs(along) - half) <= 0.08 * upm and min(across) <= 0.05 * upm and max(across) >= -0.05 * upm:
                    reveal = True
                    break
        (reveals if reveal else kept).append(face)
    return kept, reveals


def _split_isolated(faces: list[_Face], upm: float, settings: DoubleWallSettings, rooms: Iterable[Polygon] = ()) -> tuple[list[_Face], list[_Face]]:
    """Faces that are furniture sides or stubs, not walls.

    * A face shorter than ``isolated_max_length_m`` with no other wall near either end.
    * (stub rule) A face shorter than ``stub_min_length_m`` with a free end: a real jamb or return touches walls at both ends.
    * (free-face rule) A face shorter than ``isolated_free_max_length_m`` that is free at both ends and farther than
      ``isolated_free_room_distance_m`` from every ceiling room: a free-standing object side, not a wall.
    Built-in fronts, niche panels and room-edge faces are exempt from the two extra rules.
    """
    lines = [LineString([tuple(face.start), tuple(face.end)]) for face in faces]
    reach = settings.joint_reach_m * upm
    room_union = unary_union([room for room in rooms if room.area > 0.5 * upm * upm]) if settings.isolated_free_max_length_m > 0 else None
    kept: list[_Face] = []
    isolated: list[_Face] = []
    for face, line in zip(faces, lines, strict=True):
        exempt = face.entity.get("_source") in ("niche_panel", "builtin_front", "room_edge")
        short = face.length < settings.isolated_max_length_m * upm
        stub = not exempt and face.length < settings.stub_min_length_m * upm
        free_candidate = not exempt and face.length < settings.isolated_free_max_length_m * upm
        if not (short or stub or free_candidate):
            kept.append(face)
            continue
        others = [other for other_line, other in zip(lines, faces, strict=True) if other is not face]
        ends_near = [
            bool(others) and min(LineString([tuple(other.start), tuple(other.end)]).distance(Point(tuple(point))) for other in others) <= reach
            for point in (face.start, face.end)
        ]
        drop = False
        if stub:
            # A real jamb / return touches a wall at both ends (stub_end_reach_m); a face 0.15-0.35 m beside one is a free object edge.
            touching = [
                bool(others) and min(LineString([tuple(other.start), tuple(other.end)]).distance(Point(tuple(point))) for other in others) <= settings.stub_end_reach_m * upm
                for point in (face.start, face.end)
            ] if settings.stub_end_reach_m > 0.0 else ends_near
            drop = not all(touching)
        elif short:
            drop = not any(ends_near)
        if not drop and free_candidate and not any(ends_near) and (room_union is None or room_union.is_empty or room_union.distance(line) > settings.isolated_free_room_distance_m * upm):
            drop = True
        (isolated if drop else kept).append(face)
    for number, face in enumerate(kept):
        face.index = number
    return kept, isolated


def _room_label_raster(rooms: list[Polygon], grid: RasterGrid) -> np.ndarray:
    labels = np.zeros((grid.height, grid.width), dtype=np.int32)
    for index, room in enumerate(rooms, start=1):
        exterior = np.asarray([[grid.local_to_pixel(x, y)[1], grid.local_to_pixel(x, y)[0]] for x, y in room.exterior.coords], dtype=np.float64)
        cv2.fillPoly(labels, [np.round(exterior).astype(np.int32)], int(index))
        for ring in room.interiors:
            hole = np.asarray([[grid.local_to_pixel(x, y)[1], grid.local_to_pixel(x, y)[0]] for x, y in ring.coords], dtype=np.float64)
            cv2.fillPoly(labels, [np.round(hole).astype(np.int32)], 0)
    return labels


def _lookup(mask: np.ndarray, grid: RasterGrid, points: np.ndarray, outside: Any = False) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    rows = np.round((grid.max_y - points[:, 1]) / grid.pixel_size - 0.5).astype(np.int64)
    cols = np.round((points[:, 0] - grid.min_x) / grid.pixel_size - 0.5).astype(np.int64)
    inside = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])
    values = np.full(len(points), outside, dtype=mask.dtype)
    values[inside] = mask[rows[inside], cols[inside]]
    return values


def _sample_positions(face: _Face, step: float, margin: float = 0.0) -> np.ndarray:
    length = face.length
    count = max(2, int(math.ceil(max(length - 2.0 * margin, 0.0) / max(step, 1e-9))) + 1)
    if length <= 2.0 * margin:
        return np.asarray([length / 2.0])
    return np.linspace(margin, length - margin, count)


def _solid_side(face: _Face, roomish: np.ndarray, valid: np.ndarray, grid: RasterGrid, upm: float, settings: DoubleWallSettings) -> tuple[int, float, int]:
    """Return (solid side, margin, fallback side) of a face; 0 means undecided.

    The room side is the side with room (ceiling room or free floor) right in
    front of the face.  A face with no room in front of it on either side but a
    room a little further away on one side is an outer face: the wall body lies
    between it and that room.  The void side is only a fallback, because an
    outer face seen from outside has the void on its room-less side as well.
    """
    along = _sample_positions(face, settings.sample_step_m * upm, min(0.05 * upm, face.length / 4.0))
    direction, normal = face.direction, face.normal

    def strip(sign: int, low: float, high: float, mask: np.ndarray) -> float:
        offsets = np.linspace(low, high, 5) * upm
        points = (face.start[None, None, :] + along[:, None, None] * direction[None, None, :] + sign * offsets[None, :, None] * normal[None, None, :]).reshape(-1, 2)
        return float(np.mean(_lookup(mask, grid, points)))

    for low, high in ((0.02, 0.12), (0.03, settings.side_probe_m)):
        margin = strip(1, low, high, roomish) - strip(-1, low, high, roomish)
        if abs(margin) >= 0.15:
            return (-1 if margin > 0 else 1), abs(margin), 0
    # Undecided: a measured partner face (pairing) decides first.  Otherwise the void side is
    # the outside (a face behind kitchen units or a wardrobe with nothing measured behind it);
    # only without void, a room further away on one side marks an outer face.
    void_margin = strip(1, 0.03, settings.side_probe_m, ~valid) - strip(-1, 0.03, settings.side_probe_m, ~valid)
    if abs(void_margin) >= 0.25:
        return 0, abs(void_margin), (1 if void_margin > 0 else -1)
    beyond = settings.max_thickness_m + 0.10
    margin = strip(1, 0.12, beyond, roomish) - strip(-1, 0.12, beyond, roomish)
    if abs(margin) >= 0.15:
        return 0, abs(margin), (1 if margin > 0 else -1)
    return 0, 0.0, 0


# --------------------------------------------------------------------------
# measured face pairs
# --------------------------------------------------------------------------

def _measured_pairs(faces: list[_Face], free: np.ndarray, grid: RasterGrid, upm: float, settings: DoubleWallSettings, furniture: np.ndarray | None = None) -> dict[int, list[tuple[float, float, float, int, bool]]]:
    """Return per face the (start, end, thickness, partner, strong) intervals of measured pairs.

    A strong pair overlaps along the wall.  A weak pair is the opposite face of the
    same wall line seen only beside this face (within ``pair_gap_m``); it lends its
    offset to the whole face where nothing better is measured.
    """
    pairs: dict[int, list[tuple[float, float, float, int, bool]]] = {}
    gap_limit = settings.pair_gap_m * upm
    low, high = settings.min_thickness_m * upm, settings.max_thickness_m * upm
    minimum_overlap = settings.pair_min_overlap_m * upm
    sine = math.sin(math.radians(settings.pair_angle_deg))
    for first in faces:
        for second in faces:
            if second.index <= first.index:
                continue
            d1, d2 = first.direction, second.direction
            if abs(float(d1[0] * d2[1] - d1[1] * d2[0])) > sine:
                continue
            normal = first.normal
            offsets = [float(np.dot(point - first.start, normal)) for point in (second.start, second.end)]
            offset = float(np.mean(offsets))
            if not low <= abs(offset) <= high:
                continue
            towards = 1 if offset > 0 else -1
            # Solid sides must face each other (ambiguous faces may pair either way).
            if first.side and first.side != towards:
                continue
            second_towards = -towards if float(np.dot(second.normal, normal)) > 0 else towards
            if second.side and second.side != second_towards:
                continue
            positions = sorted(float(np.dot(point - first.start, d1)) for point in (second.start, second.end))
            start, end = max(0.0, positions[0]), min(first.length, positions[1])
            if end - start < minimum_overlap:
                gap = max(positions[0] - first.length, 0.0 - positions[1], 0.0)
                if gap > gap_limit or not (first.side or second.side):
                    continue
                # The decided face orients an undecided partner (an outer face beside an opening).
                if not first.side:
                    first.side = towards
                if not second.side:
                    second.side = second_towards
                thickness = abs(offset)
                pairs.setdefault(first.index, []).append((0.0, first.length, thickness, second.index, False))
                pairs.setdefault(second.index, []).append((0.0, second.length, thickness, first.index, False))
                continue
            # The strip between the faces must be wall, not floor.
            along = np.linspace(start, end, max(3, int((end - start) / (0.05 * upm)) + 1))
            across = np.linspace(0.2, 0.8, 4) * abs(offset)
            points = (first.start[None, None, :] + along[:, None, None] * d1[None, None, :] + towards * across[None, :, None] * normal[None, None, :]).reshape(-1, 2)
            if float(np.mean(_lookup(free, grid, points))) > 0.25:
                continue
            if furniture is not None and float(np.mean(_lookup(furniture, grid, points))) > 0.30:
                continue
            thickness = abs(offset)
            pairs.setdefault(first.index, []).append((start, end, thickness, second.index, True))
            second_positions = sorted(float(np.dot(first.point(t) - second.start, d2)) for t in (start, end))
            pairs.setdefault(second.index, []).append((max(0.0, second_positions[0]), min(second.length, second_positions[1]), thickness, first.index, True))
            if not first.side:
                first.side = towards
            if not second.side:
                second.side = second_towards
    return pairs


# --------------------------------------------------------------------------
# far-side probe
# --------------------------------------------------------------------------

@dataclass(slots=True)
class _Samples:
    along: np.ndarray
    pair: np.ndarray
    far: np.ndarray
    void: np.ndarray
    skip: np.ndarray | None = None


def _far_samples(face: _Face, pairs: list[tuple[float, float, float, int, bool]], room_labels: np.ndarray, free: np.ndarray, valid: np.ndarray, grid: RasterGrid, upm: float, settings: DoubleWallSettings, outer: bool = False, openings: list[Any] = ()) -> _Samples:
    along = _sample_positions(face, settings.sample_step_m * upm, min(0.02 * upm, face.length / 4.0))
    pair_values = np.full(len(along), np.nan)
    for start, end, thickness, _partner, strong in pairs:
        if not strong:
            continue
        inside = (along >= start - 1e-9) & (along <= end + 1e-9)
        pair_values[inside] = np.where(np.isnan(pair_values[inside]), thickness, np.minimum(pair_values[inside], thickness))
    if not face.side or outer:
        # An outer face looks at the outside: what lies behind it is its own building.
        return _Samples(along, pair_values, np.full(len(along), np.nan), np.ones(len(along), dtype=bool))
    step = 0.01 * upm
    distances = np.arange(0.02 * upm, settings.probe_m * upm + step / 2.0, step)
    base = face.start[None, None, :] + along[:, None, None] * face.direction[None, None, :]
    points = (base + distances[None, :, None] * face.solid_normal[None, None, :]).reshape(-1, 2)
    labels = _lookup(room_labels, grid, points, 0).reshape(len(along), len(distances))
    # The face's own room (the one on its room side) does not count as the neighbour.
    own_points = (face.start[None, :] + along[:, None] * face.direction[None, :] - 0.08 * upm * face.solid_normal[None, :])
    own = _lookup(room_labels, grid, own_points, 0)
    own_set = set(int(value) for value in own if value > 0)
    other_room = (labels > 0) & ~np.isin(labels, list(own_set)) if own_set else labels > 0
    floor = _lookup(free, grid, points).reshape(len(along), len(distances))
    sustained = floor.copy()
    for shift in (1, 2):
        rolled = np.zeros_like(floor)
        rolled[:, :-shift] = floor[:, shift:]
        sustained &= rolled
    # Free floor is a neighbour only right behind the wall; further away it may be a
    # balcony or terrace seen through a window, which does not make the wall interior.
    sustained &= distances[None, :] <= (settings.max_thickness_m + 0.05) * upm
    hit = other_room | sustained
    first = np.where(hit.any(axis=1), np.argmax(hit, axis=1), -1)
    far = np.where(first >= 0, distances[np.maximum(first, 0)], np.nan)
    far[first < 0] = np.nan
    void_fraction = np.mean(~_lookup(valid, grid, points).reshape(len(along), len(distances)), axis=1)
    # Behind a door or window span the probe sees the reveal floor or the outside;
    # such samples take the kind of their neighbours.
    skip = np.zeros(len(along), dtype=bool)
    positions = face.start[None, :] + along[:, None] * face.direction[None, :]
    for opening in openings:
        axis = _axis_of(opening)
        if not _parallel(face.direction, axis, 10.0):
            continue
        centre = np.asarray(getattr(opening, "center"), dtype=float)[:2]
        half = float(getattr(opening, "width")) / 2.0 + 0.05 * upm
        across = (positions - centre) @ face.solid_normal
        if not np.any(np.abs(across) <= (settings.max_thickness_m + 0.15) * upm):
            continue
        skip |= (np.abs((positions - centre) @ axis) <= half) & (np.abs(across) <= (settings.max_thickness_m + 0.15) * upm)
    return _Samples(along, pair_values, far, void_fraction > 0.5, skip)


# --------------------------------------------------------------------------
# thickness learning and runs
# --------------------------------------------------------------------------

def _clusters(values: list[float], gap: float) -> list[tuple[float, int]]:
    if not values:
        return []
    ordered = sorted(values)
    groups: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] > gap:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [(float(np.median(group)), len(group)) for group in groups]


def _learn_thickness(runs: list[WallRun], samples: list[_Samples], pairs: dict[int, list[tuple[float, float, float, int, bool]]], upm: float, settings: DoubleWallSettings) -> tuple[list[tuple[float, int]], float, dict[str, Any]]:
    """Learn the interior thickness classes and the exterior thickness from the measured evidence.

    Interior: measured face pairs on interior runs, plus room-edge distances where a
    cluster is backed by at least ~0.5 m of wall.  Exterior: measured pairs on exterior
    runs (overlapping or beside), else the configured default.
    """
    low, high = settings.min_thickness_m * upm, settings.max_thickness_m * upm
    interior_pairs: list[float] = []
    exterior_pairs: list[float] = []
    room_values: list[float] = []
    step = settings.sample_step_m * upm
    for run in runs:
        sample = samples[run.face]
        inside = (sample.along >= run.start) & (sample.along <= run.end)
        values = [float(value) for value in sample.pair[inside] if not np.isnan(value)]
        weak = [item[2] for item in pairs.get(run.face, []) if not item[4]]
        if run.kind == "interior":
            interior_pairs.extend(values)
            if not values:
                room_values.extend(float(value) for value in sample.far[inside] if not np.isnan(value) and low <= value <= high)
        elif run.kind == "exterior":
            exterior_pairs.extend(values)
            if not values and weak:
                exterior_pairs.extend([float(np.median(weak))] * max(1, int((run.end - run.start) / max(step, 1e-9))))
    gap = settings.cluster_gap_m * upm
    interior = _clusters(interior_pairs, gap) if interior_pairs else []
    minimum = max(3, int(round(0.5 / max(settings.sample_step_m, 1e-9))))
    for value, count in _clusters(room_values, gap):
        if count >= minimum and all(abs(value - existing) > gap for existing, _ in interior):
            interior.append((value, count))
    interior.sort()
    measured = len(exterior_pairs) >= minimum
    exterior_value = float(np.median(exterior_pairs)) if measured else settings.exterior_thickness_m * upm
    stats = {
        "double_wall_interior_classes_m": [[round(value / upm, 3), count] for value, count in interior],
        "double_wall_exterior_thickness_m": round(exterior_value / upm, 3),
        "double_wall_exterior_measured": measured,
        "double_wall_pair_samples": len(interior_pairs) + len(exterior_pairs),
    }
    return interior, exterior_value, stats


def _sample_kind(pair: float, far: float, void: bool, upm: float, settings: DoubleWallSettings) -> str:
    if not np.isnan(far):
        reach = (pair if not np.isnan(pair) else 0.0) + settings.probe_m * upm / 2.0
        if far <= max(reach, settings.max_thickness_m * upm):
            return "interior"
        if far <= settings.probe_m * upm:
            return "interior"
    return "exterior"


def _face_runs(face: _Face, samples: _Samples, upm: float, settings: DoubleWallSettings) -> list[WallRun]:
    if not face.side:
        return [WallRun(face.index, 0.0, face.length, 0.0, "partition", "ambiguous")]
    kinds: list[str | None] = [_sample_kind(pair, far, void, upm, settings) for pair, far, void in zip(samples.pair, samples.far, samples.void)]
    if samples.skip is not None and samples.skip.any():
        decided = [index for index, flag in enumerate(samples.skip) if not flag]
        if not decided:
            kinds = ["exterior"] * len(kinds)
        else:
            kinds = [kinds[min(decided, key=lambda item: abs(item - index))] if flag else kind for index, (kind, flag) in enumerate(zip(kinds, samples.skip))]
    # Contiguous runs of the same kind; short runs join the longer neighbour.
    runs: list[list[Any]] = []
    for index, kind in enumerate(kinds):
        if runs and runs[-1][0] == kind:
            runs[-1][2] = index
        else:
            runs.append([kind, index, index])
    minimum = settings.min_run_m * upm
    step = settings.sample_step_m * upm

    def span(run: list[Any]) -> float:
        return float(samples.along[run[2]] - samples.along[run[1]]) + step

    changed = True
    while changed and len(runs) > 1:
        changed = False
        shortest = min(range(len(runs)), key=lambda item: span(runs[item]))
        if span(runs[shortest]) >= minimum:
            break
        neighbours = [item for item in (shortest - 1, shortest + 1) if 0 <= item < len(runs)]
        target = max(neighbours, key=lambda item: span(runs[item]))
        low, high = min(shortest, target), max(shortest, target)
        runs[low] = [runs[target][0], runs[low][1], runs[high][2]]
        del runs[high]
        merged: list[list[Any]] = []
        for run in runs:
            if merged and merged[-1][0] == run[0]:
                merged[-1][2] = run[2]
            else:
                merged.append(run)
        runs = merged
        changed = True
    output: list[WallRun] = []
    for number, (kind, first, last) in enumerate(runs):
        start = 0.0 if number == 0 else float((samples.along[first] + samples.along[runs[number - 1][2]]) / 2.0)
        end = face.length if number == len(runs) - 1 else float((samples.along[last] + samples.along[runs[number + 1][1]]) / 2.0)
        output.append(WallRun(face.index, start, end, 0.0, kind, "pending"))
    return output


def _room_edge_runs(faces: list[_Face], runs: list[WallRun], rooms: list[Polygon], roomish: np.ndarray, valid: np.ndarray, openings: list[Any], grid: RasterGrid, upm: float, settings: DoubleWallSettings, exterior: float, builtins: list[Polygon] | None = None) -> list[WallRun]:
    """Close the building outline where a room edge borders the outside but no wall face was measured.

    Kitchen units, wardrobes or curtains can hide an exterior wall face completely.  A room
    edge (ceiling room, at least ``room_edge_min_area_m2``) that is not covered by any wall
    body, has unmeasured outside behind it (no floor or other room, data ending within
    the probe, no closed built-in unit) and meets an existing wall at one end becomes an
    exterior wall of the learned exterior thickness at the end of the data.  Doors and
    windows in it are cut afterwards like in any other wall.
    """
    face_of = {face.index: face for face in faces}
    bands = [_band(face_of[run.face], run.start, run.end, run.thickness) for run in runs if run.thickness > 0.0]
    if not bands:
        return []
    walls = unary_union(bands)
    near = 0.10 * upm
    reach = settings.joint_reach_m * upm
    step = 0.05 * upm
    units = unary_union([item.buffer(-0.01 * upm) for item in builtins]) if builtins else GeometryCollection()
    added: list[WallRun] = []
    # An opening marks the wall line itself: a room edge beside it (parallel, within ``room_edge_opening_reach_m``) is that wall,
    # not a hidden one, so no wall is invented in front of the opening (it would stay as an outside block beside the cut).
    spans = [(_axis_of(opening), np.asarray(getattr(opening, "center"), dtype=float)[:2], float(getattr(opening, "width")) / 2.0 + 0.02 * upm) for opening in openings]
    opening_reach = settings.room_edge_opening_reach_m * upm
    for room in rooms:
        if room.area < settings.room_edge_min_area_m2 * upm * upm:
            continue
        coordinates = list(room.exterior.coords)
        orientation = 1.0 if room.exterior.is_ccw else -1.0
        for first, second in zip(coordinates[:-1], coordinates[1:]):
            a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
            length = float(np.linalg.norm(b - a))
            if length < settings.room_edge_min_length_m * upm:
                continue
            direction = (b - a) / length
            # Outward normal: right of a counter-clockwise ring.
            outward = np.asarray((direction[1], -direction[0])) * orientation
            along = np.arange(step / 2.0, length, step)
            flags = []
            depths: list[float] = []
            distances = np.arange(0.03, settings.max_thickness_m + 0.35, 0.01) * upm
            for t in along:
                point = a + direction * t
                if opening_reach > 0.0 and any(
                    _parallel(direction, axis, 10.0) and abs(float(np.dot(point - centre, axis))) <= half and abs(float(np.dot(point - centre, np.asarray((-axis[1], axis[0]))))) <= opening_reach
                    for axis, centre, half in spans
                ):
                    flags.append(False)
                    continue
                probe = point[None, :] + outward[None, :] * distances[:, None]
                if bool(np.any(_lookup(roomish, grid, probe))):
                    flags.append(False)
                    continue
                # Units in front of a hidden wall (counters, cabinets) are measured up to the wall:
                # the wall face is where the data ends for good.
                void = ~_lookup(valid, grid, probe)
                # First stretch of at least 10 cm without any data.
                window = 10
                sustained = np.array([bool(void[index:index + window].all()) for index in range(max(len(void) - window + 1, 0))])
                if not sustained.any():
                    flags.append(False)
                    continue
                depth = float(distances[int(np.argmax(sustained))]) if not sustained[0] else 0.0
                if walls.distance(Point(tuple(point + outward * depth))) <= near or walls.distance(Point(tuple(point))) <= near:
                    flags.append(False)
                    continue
                # A closed built-in unit (its wall already moved behind it) is not the outside.
                if not units.is_empty and LineString([tuple(point), tuple(point + outward * (depth + 0.05 * upm))]).intersects(units):
                    flags.append(False)
                    continue
                # A wall body just beyond means the gap is a built-in or cavity, not a missing wall.
                beyond = LineString([tuple(point + outward * depth), tuple(point + outward * (depth + 0.20 * upm))])
                if beyond.intersects(walls):
                    flags.append(False)
                    continue
                flags.append(True)
                depths.append(depth)
            # contiguous uncovered outside runs
            depth_of: list[float] = []
            cursor = 0
            for flag in flags:
                if flag:
                    depth_of.append(depths[cursor])
                    cursor += 1
                else:
                    depth_of.append(0.0)
            start = None
            for index, flag in enumerate(flags + [False]):
                if flag and start is None:
                    start = index
                elif not flag and start is not None:
                    t0, t1 = max(0.0, along[start] - step / 2.0), min(length, along[index - 1] + step / 2.0)
                    start_index = start
                    start = None
                    if t1 - t0 < settings.room_edge_min_length_m * upm:
                        continue
                    shift = float(np.median(depth_of[start_index:index])) if depth_of[start_index:index] else 0.0
                    # Continue the inner face of a neighbouring wall on the same line, if there is one.
                    candidates = []
                    for run in runs:
                        if run.thickness <= 0.0:
                            continue
                        other = face_of[run.face]
                        if not _parallel(other.direction, direction, 3.0) or float(np.dot(other.solid_normal, outward)) <= 0.5:
                            continue
                        offset = float(np.dot(other.start - a, outward))
                        line = LineString([tuple(other.point(run.start)), tuple(other.point(run.end))])
                        ends = [a + direction * t0 + outward * offset, a + direction * t1 + outward * offset]
                        if abs(offset - shift) <= 0.35 * upm and min(line.distance(Point(tuple(end))) for end in ends) <= reach:
                            candidates.append(offset)
                    if candidates:
                        shift = min(candidates, key=lambda value: abs(value - shift))
                    p0, p1 = a + direction * t0 + outward * shift, a + direction * t1 + outward * shift
                    if min(walls.distance(Point(tuple(p0))), walls.distance(Point(tuple(p1)))) > reach + shift:
                        continue
                    # An end that stopped next to a wall body reaches into it (closed joint, no sliver gap).
                    if walls.distance(Point(tuple(p0))) <= near + 0.05 * upm:
                        p0 = p0 - direction * (walls.distance(Point(tuple(p0))) + 0.05 * upm)
                    if walls.distance(Point(tuple(p1))) <= near + 0.05 * upm:
                        p1 = p1 + direction * (walls.distance(Point(tuple(p1))) + 0.05 * upm)
                    # Run ends that stop just short of a wall body reach it.
                    face = _Face(len(faces), p0, p1, {"type": "polyline", "points": [p0.tolist(), p1.tolist()], "closed": False, "_source": "room_edge"})
                    face.side = 1 if float(np.dot(face.normal, outward)) > 0 else -1
                    # An unmeasured mass between this face and another wall body close behind it
                    # (a closet, shaft or chimney) is solid: the body reaches that wall.
                    thickness = float(exterior)
                    behind = []
                    for t in np.linspace(0.1, 0.9, 5):
                        ray = LineString([tuple(face.point(t * face.length)), tuple(face.point(t * face.length, (settings.max_thickness_m + 0.35) * upm))])
                        hit = ray.intersection(walls)
                        if not hit.is_empty:
                            behind.append(min(Point(tuple(face.point(t * face.length))).distance(part) for part in _parts(hit)))
                    if behind and len(behind) >= 3:
                        thickness = float(min(max(thickness, max(behind) + 0.01 * upm), settings.max_thickness_m * upm))
                    faces.append(face)
                    face_of[face.index] = face
                    run = WallRun(face.index, 0.0, face.length, thickness, "exterior", "room-edge")
                    runs.append(run)
                    added.append(run)
    return added


# --------------------------------------------------------------------------
# tall built-in units
# --------------------------------------------------------------------------

@dataclass(slots=True)
class _Unit:
    face: _Face
    run: WallRun
    start: float
    end: float
    offset: float
    depth: float = 0.0
    neighbour: WallRun | None = None

    def polygon(self) -> Polygon:
        face = self.face
        return Polygon([tuple(face.point(self.start, self.offset)), tuple(face.point(self.end, self.offset)),
                        tuple(face.point(self.end, self.offset + self.depth)), tuple(face.point(self.start, self.offset + self.depth))])


def _flag_runs(flags: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Index ranges of True stretches that may be interrupted by up to ``gap`` False samples."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    last = -10**9
    for index, flag in enumerate(flags):
        if flag:
            if start is None:
                start = index
            last = index
        elif start is not None and index - last > gap:
            runs.append((start, last))
            start = None
    if start is not None:
        runs.append((start, last))
    return runs


def _box_samples(face: _Face, a: float, b: float, c0: float, c1: float, step: float) -> np.ndarray:
    """Grid points of the face-frame box along a..b, across c0..c1 (solid side positive)."""
    along = np.arange(min(a, b) + step / 2.0, max(a, b), step)
    across = np.arange(min(c0, c1) + step / 2.0, max(c0, c1), step)
    if not len(along) or not len(across):
        return np.zeros((0, 2))
    grid_along, grid_across = np.meshgrid(along, across, indexing="ij")
    return face.start[None, :] + face.direction[None, :] * grid_along.reshape(-1, 1) + face.solid_normal[None, :] * grid_across.reshape(-1, 1)


def _share(mask: np.ndarray, grid: RasterGrid, points: np.ndarray) -> float:
    return float(np.mean(_lookup(mask, grid, points))) if len(points) else 0.0


def _first_entry(origin: np.ndarray, target: np.ndarray, bands: list[Polygon], minimum: float) -> float | None:
    """Distance from ``origin`` to the nearest band the segment origin->target enters at ``minimum`` or later."""
    ray = LineString([tuple(origin), tuple(target)])
    start = Point(tuple(origin))
    best: float | None = None
    for band in bands:
        if not ray.intersects(band):
            continue
        distance = start.distance(ray.intersection(band))
        if distance >= minimum and (best is None or distance < best):
            best = distance
    return best


def _tall_unit_runs(
    runs: list[WallRun], faces: list[_Face], free: np.ndarray, evidence: TallUnitEvidence, grid: RasterGrid, upm: float, settings: DoubleWallSettings,
) -> tuple[list[tuple[Polygon, str]], dict[str, Any]]:
    """Tall built-in units (fridge, oven tower, tall kitchen cabinets) whose front was taken as a wall face.

    A wall run (measured, or completed from a room edge) with tall built-in evidence along at least
    ``tall_unit_min_m`` within ``tall_unit_band_m`` of its face is a unit front when

    * the room side in front of it is open floor (``tall_unit_room_free`` of a ``tall_unit_room_m`` strip);
    * a wall stands behind it at a unit depth (``tall_unit_min_depth_m`` .. ``tall_unit_max_depth_m``): a wall
      body that rays from the front enter, or else a parallel wall on the same side whose end lies next to the
      unit (``tall_unit_neighbour_gap_m``) - the wall the unit hides, which is then continued behind the unit;
    * its footprint holds no open floor (``tall_unit_max_free``) and few observed low surfaces
      (``tall_unit_max_low``: counter tops, furniture, floor are never inside a floor-to-ceiling unit).

    The unit reaches a perpendicular wall body within ``tall_unit_side_gap_m`` of its evidence.  Wall runs
    whose body lies inside a unit are removed (split where only part of them does).  Returns the unit
    rectangles labelled "KS" (fridge evidence; a long run is split into a fridge module and the rest) or "HS".
    """
    stats: dict[str, Any] = {"tall_units": 0, "tall_unit_runs_removed": 0, "tall_unit_walls_continued": 0}
    face_of = {face.index: face for face in faces}
    solid = [run for run in runs if run.thickness > 0.0]
    front_mask = np.asarray(evidence.front, dtype=bool)
    if not solid or not front_mask.any():
        return [], stats
    step = 0.02 * upm
    band = settings.tall_unit_band_m * upm
    across = np.arange(-band, band + 1e-9, max(grid.pixel_size, 0.01 * upm))
    min_depth, max_depth = settings.tall_unit_min_depth_m * upm, settings.tall_unit_max_depth_m * upm
    candidates: list[_Unit] = []
    for run in solid:
        face = face_of[run.face]
        reach = settings.tall_unit_reach_m * upm
        along = np.arange(run.start - reach, run.end + reach + 1e-9, step)
        points = face.start[None, None, :] + face.direction[None, None, :] * along[:, None, None] + face.solid_normal[None, None, :] * across[None, :, None]
        hits = _lookup(front_mask, grid, points.reshape(-1, 2)).reshape(len(along), len(across))
        for first, last in _flag_runs(hits.any(axis=1), max(1, int(round(settings.tall_unit_gap_m * upm / step)))):
            a, b = float(along[first]), float(along[last])
            if b - a < settings.tall_unit_min_m * upm or min(b, run.end) - max(a, run.start) < 0.20 * upm:
                continue
            offsets = np.broadcast_to(across[None, :], hits.shape)[first:last + 1][hits[first:last + 1]]
            candidates.append(_Unit(face, run, a, b, float(np.median(offsets))))
    if not candidates:
        return [], stats
    front_runs = {id(unit.run) for unit in candidates}
    backs = [(run, _band(face_of[run.face], run.start, run.end, run.thickness)) for run in solid if id(run) not in front_runs]
    back_bands = [item for _, item in backs]
    units: list[_Unit] = []
    for unit in candidates:
        face = unit.face
        samples = np.arange(unit.start, unit.end + 1e-9, 0.05 * upm)
        hits = [hit for s in samples if (hit := _first_entry(face.point(s, unit.offset), face.point(s, unit.offset + max_depth), back_bands, min_depth)) is not None]
        if len(hits) >= 0.5 * len(samples):
            unit.depth = float(np.median(hits))
        else:
            best: tuple[float, float, WallRun] | None = None
            for other, _ in backs:
                other_face = face_of[other.face]
                if not _parallel(other_face.direction, face.direction, settings.pair_angle_deg) or float(np.dot(other_face.solid_normal, face.solid_normal)) < 0.9:
                    continue
                depth = float(np.dot(other_face.start - face.start, face.solid_normal)) - unit.offset
                if not min_depth <= depth <= max_depth:
                    continue
                u0, u1 = sorted(float(np.dot(other_face.point(value) - face.start, face.direction)) for value in (other.start, other.end))
                lateral = max(u0 - unit.end, unit.start - u1)
                if lateral > settings.tall_unit_neighbour_gap_m * upm:
                    continue
                if best is None or lateral < best[0]:
                    best = (lateral, depth, other)
            if best is None:
                continue
            unit.depth, unit.neighbour = best[1], best[2]
        # The room lies in front of the unit's evidence (beyond it may stand a neighbouring unit).
        room = _box_samples(face, unit.start, unit.end, unit.offset - settings.tall_unit_room_m * upm, unit.offset, 0.04 * upm)
        # The unit reaches a perpendicular wall body close beyond its evidence (a blind corner, an end panel gap).
        middle = unit.offset + unit.depth / 2.0
        for sign in (-1.0, 1.0):
            end = unit.end if sign > 0 else unit.start
            origin = face.point(end, middle)
            hit = _first_entry(origin, origin + face.direction * sign * settings.tall_unit_side_gap_m * upm, back_bands, 0.02 * upm)
            if hit is None:
                continue
            filler = _box_samples(face, end, end + sign * hit, unit.offset, unit.offset + unit.depth, 0.04 * upm)
            if _share(free, grid, filler) <= settings.tall_unit_max_free:
                if sign > 0:
                    unit.end = end + hit
                else:
                    unit.start = end - hit
        footprint = _box_samples(face, unit.start, unit.end, unit.offset, unit.offset + unit.depth, 0.04 * upm)
        low = np.asarray(evidence.low, dtype=bool) & ~front_mask
        if (_share(free, grid, footprint) > settings.tall_unit_max_free or _share(low, grid, footprint) > settings.tall_unit_max_low
                or _share(free, grid, room) < settings.tall_unit_room_free):
            continue
        units.append(unit)
    # the same front found from two runs: keep the longer one
    units.sort(key=lambda item: -(item.end - item.start) * item.depth)
    accepted: list[_Unit] = []
    for unit in units:
        if any(unit.polygon().intersection(other.polygon()).area > 0.5 * unit.polygon().area for other in accepted):
            continue
        keep = True
        for other in accepted:
            overlap = unit.polygon().intersection(other.polygon())
            if overlap.is_empty or overlap.area <= 1e-6 * upm * upm:
                continue
            # trim the unit that the overlap crosses over the shorter stretch of its own axis
            trimmed = min((unit, other), key=lambda item: _axis_extent(overlap, item.face.direction))
            extent = _axis_extent(overlap, trimmed.face.direction)
            centre = float(np.dot(np.asarray(overlap.centroid.coords[0]) - trimmed.face.start, trimmed.face.direction))
            if abs(centre - trimmed.end) <= abs(centre - trimmed.start):
                trimmed.end -= extent
            else:
                trimmed.start += extent
            if trimmed.end - trimmed.start < settings.tall_unit_min_m * upm:
                keep = keep and trimmed is not unit
        if keep and unit.end - unit.start >= settings.tall_unit_min_m * upm:
            accepted.append(unit)
    accepted = [unit for unit in accepted if unit.end - unit.start >= settings.tall_unit_min_m * upm]
    if not accepted:
        return [], stats
    # wall runs whose body lies in a unit go; a run reaching beyond a unit keeps its outside parts
    shapes = unary_union([unit.polygon().buffer(0.03 * upm, join_style=2) for unit in accepted])
    kept: list[WallRun] = []
    split: set[int] = set()
    for run in runs:
        if run.thickness <= 0.0:
            kept.append(run)
            continue
        face = face_of[run.face]
        samples = np.arange(run.start, run.end + 1e-9, step)
        middles = np.asarray([face.point(s, run.thickness / 2.0) for s in samples])
        inside = np.asarray([shapes.contains(Point(tuple(point))) for point in middles])
        if not inside.any():
            kept.append(run)
            continue
        stats["tall_unit_runs_removed"] += 1
        for first, last in _flag_runs(~inside, 0):
            a = run.start if first == 0 else float(samples[first])
            b = run.end if last == len(samples) - 1 else float(samples[last])
            if b - a >= settings.min_stub_m * upm:
                piece_run = WallRun(run.face, a, b, run.thickness, run.kind, run.evidence, run.group,
                                    run.extend_start if first == 0 else 0.0, run.extend_end if last == len(samples) - 1 else 0.0)
                kept.append(piece_run)
                split.add(id(piece_run))
    # a leftover piece that lies inside another wall body adds nothing
    final: list[WallRun] = []
    for run in kept:
        if id(run) in split:
            piece = _band(face_of[run.face], run.start, run.end, run.thickness)
            others = [_band(face_of[item.face], item.start, item.end, item.thickness) for item in kept if item is not run and item.thickness > 0.0 and item.face != run.face]
            if others and piece.area > 0.0 and piece.difference(unary_union(others)).area <= 0.10 * piece.area:
                continue
        final.append(run)
    runs[:] = final
    # the hidden wall behind a unit is continued along its face to the unit's far end
    for unit in accepted:
        if unit.neighbour is None or not any(unit.neighbour is run for run in runs):
            continue
        run = unit.neighbour
        face = face_of[run.face]
        corners = [np.asarray(point) for point in unit.polygon().exterior.coords[:-1]]
        along = [float(np.dot(point - face.start, face.direction)) for point in corners]
        low_end, high_end = min(along), max(along)
        if high_end > run.end and abs(high_end - run.end) <= abs(run.start - low_end):
            run.end = high_end  # the joint step extends an end that meets a wall body into it
            if run.end > face.length:
                face.end = face.point(run.end)
        elif low_end < run.start:
            shift = max(0.0, -low_end)
            if shift > 0.0:
                face.start = face.point(-shift)
                for item in runs:
                    if item.face == run.face:
                        item.start += shift
                        item.end += shift
            run.start = low_end + shift
        else:
            continue
        stats["tall_unit_walls_continued"] += 1
    output: list[tuple[Polygon, str]] = []
    fridge = np.asarray(evidence.fridge, dtype=bool)
    for unit in accepted:
        stats["tall_units"] += 1
        output.extend(_label_unit(unit, fridge, grid, upm, settings))
    stats["tall_unit_rects"] = [[round(float(value), 3) for value in shape.bounds] + [label] for shape, label in output]
    return output, stats


def _cabinet_wall_runs(
    runs: list[WallRun], faces: list[_Face], cabinet: np.ndarray, evidence: WallEvidence, grid: RasterGrid, upm: float, settings: DoubleWallSettings, builtins: list[Polygon] = (), free: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[tuple[Polygon, str]]]:
    """Drop wall runs that are cabinet fronts: the body lies on cabinet labels and no wall stands there.

    ``cabinet_share`` = share of the run body (samples across its thickness) on cabinet-labelled pixels (dilated by
    ``cabinet_dilate_m``); ``wall_share`` = share of the samples along the run where wall-class points of wall height
    (``face_min_height_m`` covered, ``face_reach_m`` reached, any lateral position inside the body) exist.  A run is a
    cabinet front when ``cabinet_share >= cabinet_min_share`` and ``wall_share <= cabinet_max_wall_share``.  A real wall
    behind a cabinet carries wall-class points up its height and stays.
    """
    stats: dict[str, Any] = {"cabinet_wall_runs": 0, "cabinet_wall_m": 0.0, "cabinet_column_stretches": 0, "cabinet_column_m": 0.0}
    units: list[tuple[Polygon, str]] = []
    if evidence.wall_bits is None or evidence.storey_height <= 0.0 or not np.asarray(cabinet).any():
        return stats, units
    pixels = max(1, int(round(settings.cabinet_dilate_m * upm / grid.pixel_size)))
    mask = cv2.dilate(np.asarray(cabinet, dtype=np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))).astype(bool)
    face_of = {face.index: face for face in faces}
    kept: list[WallRun] = []
    for run in runs:
        face = face_of.get(run.face)
        if face is None or run.thickness <= 0.0 or run.end - run.start < 0.3 * upm:
            kept.append(run)
            continue
        along = np.arange(run.start, run.end + 1e-9, 0.04 * upm)
        offsets = np.linspace(0.1 * run.thickness, 0.9 * run.thickness, 5)
        wall = np.zeros(len(along), dtype=np.uint32)
        hits = []
        for offset in offsets:
            points = face.start[None, :] + along[:, None] * face.direction[None, :] + face.solid_normal[None, :] * offset
            wall |= _lookup(evidence.wall_bits, grid, points, 0).astype(np.uint32)
            hits.append(_lookup(mask, grid, points))
        cabinet_share = float(np.mean(np.asarray(hits)))
        covered, top = _bit_heights(wall, evidence.bins, evidence.storey_height)
        wall_share = float(np.mean((covered >= settings.face_min_height_m * upm) & (top >= settings.face_reach_m * upm)))
        if cabinet_share >= settings.cabinet_min_share and wall_share <= settings.cabinet_max_wall_share:
            stats["cabinet_wall_runs"] += 1
            stats["cabinet_wall_m"] += (run.end - run.start) / upm
            stats.setdefault("cabinet_wall_list", []).append([np.round(face.point(run.start), 2).tolist(), np.round(face.point(run.end), 2).tolist(), round(cabinet_share, 2), round(wall_share, 2)])
            continue
        kept.append(run)
    # Second rule: stretches without a wall column that lie on / beside kitchen cabinets (a cabinet column whose panels read as wall).
    step = 0.04 * upm
    final: list[WallRun] = []
    for run in kept:
        face = face_of.get(run.face)
        if face is None or run.thickness <= 0.0 or run.evidence == "room-edge" or run.end - run.start < settings.cabinet_stretch_min_m * upm:
            final.append(run)
            continue
        along = np.arange(run.start, run.end + 1e-9, step)
        wall = np.zeros(len(along), dtype=np.uint32)
        on_cabinet = np.zeros(len(along), dtype=bool)
        for offset in np.linspace(0.1 * run.thickness, 0.9 * run.thickness, 5):
            points = face.start[None, :] + along[:, None] * face.direction[None, :] + face.solid_normal[None, :] * offset
            wall |= _lookup(evidence.wall_bits, grid, points, 0).astype(np.uint32)
            on_cabinet |= _lookup(mask, grid, points)
        # cabinets beside the run on one side and measured room floor on the other: a column standing in the room, not the outer
        # wall behind a counter (that one has no floor on its far side)
        beside = [np.zeros(len(along), dtype=bool), np.zeros(len(along), dtype=bool)]
        floor_beside = [np.zeros(len(along), dtype=bool), np.zeros(len(along), dtype=bool)]
        for side, sign in enumerate((-1.0, 1.0)):
            for distance in (0.1, 0.25, 0.4, settings.cabinet_flank_m):
                lateral = -distance * upm if sign < 0 else run.thickness + distance * upm
                points = face.start[None, :] + along[:, None] * face.direction[None, :] + face.solid_normal[None, :] * lateral
                beside[side] |= _lookup(mask, grid, points)
            if free is not None:
                for distance in (0.3, 0.45, 0.6):
                    lateral = -distance * upm if sign < 0 else run.thickness + distance * upm
                    points = face.start[None, :] + along[:, None] * face.direction[None, :] + face.solid_normal[None, :] * lateral
                    floor_beside[side] |= _lookup(free, grid, points)
        covered, top = _bit_heights(wall, evidence.bins, evidence.storey_height)
        supported = (covered >= settings.face_min_height_m * upm) & (top >= settings.face_reach_m * upm)
        column = (beside[0] & floor_beside[1]) | (beside[1] & floor_beside[0])
        flagged = (on_cabinet | column) & ~supported
        if float(np.mean(flagged)) < settings.cabinet_run_min_share or float(np.mean(supported)) > settings.cabinet_run_max_support:
            final.append(run)
            continue
        stretches = [(float(along[first]), float(along[last])) for first, last in _flag_runs(flagged, max(1, int(round(settings.cabinet_stretch_gap_m * upm / step))))]
        stretches = [(a, b) for a, b in stretches if b - a >= settings.cabinet_stretch_min_m * upm]
        if not stretches:
            final.append(run)
            continue
        cursor = run.start
        for a, b in stretches:
            stats["cabinet_column_stretches"] += 1
            stats["cabinet_column_m"] += (b - a) / upm
            stats.setdefault("cabinet_column_list", []).append([np.round(face.point(a), 2).tolist(), np.round(face.point(b), 2).tolist()])
            band = _band(face, a, b, run.thickness)
            touching = [item for item in builtins if item.intersects(band) and item.intersection(band).area > 0.2 * band.area]
            if touching:
                hull = unary_union([band, *touching]).convex_hull
                if isinstance(hull, Polygon) and not hull.is_empty:
                    rect = hull.minimum_rotated_rectangle
                    if isinstance(rect, Polygon) and not rect.is_empty:
                        units.append((rect, "HS"))
            if a - cursor >= settings.cabinet_remnant_min_m * upm:
                final.append(WallRun(run.face, cursor, a, run.thickness, run.kind, run.evidence, run.group, run.extend_start if cursor == run.start else 0.0, 0.0))
            cursor = b
        if run.end - cursor >= settings.cabinet_remnant_min_m * upm:
            final.append(WallRun(run.face, cursor, run.end, run.thickness, run.kind, run.evidence, run.group, 0.0, run.extend_end))
    runs[:] = final
    return stats, units


def _axis_extent(geometry: Any, direction: np.ndarray) -> float:
    coordinates = [np.asarray(point) for part in _parts(geometry) if isinstance(part, Polygon) for point in part.exterior.coords]
    if not coordinates:
        return 0.0
    values = [float(np.dot(point, direction)) for point in coordinates]
    return max(values) - min(values)


def _label_unit(unit: _Unit, fridge: np.ndarray, grid: RasterGrid, upm: float, settings: DoubleWallSettings) -> list[tuple[Polygon, str]]:
    """"KS" where fridge evidence lies in the front zone (a long unit is split into the fridge module at the
    end holding it and a tall cabinet "HS" for the rest), "HS" without fridge evidence."""
    face = unit.face
    step = 0.02 * upm
    zone = _box_samples(face, unit.start, unit.end, unit.offset - 0.05 * upm, unit.offset + 0.20 * upm, step)
    if not len(zone):
        return [(unit.polygon(), "HS")]
    flags = _lookup(fridge, grid, zone)
    along = np.sort((zone[flags] - face.start) @ face.direction)
    if len(np.unique(np.floor(along / (0.05 * upm)))) < 3:  # a stray fridge pixel at a corner is no fridge
        return [(unit.polygon(), "HS")]
    module = settings.tall_unit_fridge_m * upm
    if unit.end - unit.start <= 1.5 * module:
        return [(unit.polygon(), "KS")]
    # stretches of fridge evidence (gaps up to 0.25 m); a stretch at one end of the unit is its fridge module
    breaks = np.nonzero(np.diff(along) > 0.25 * upm)[0]
    stretches = [(float(chunk[0]), float(chunk[-1])) for chunk in np.split(along, breaks + 1)]

    def part(a: float, b: float) -> Polygon:
        return Polygon([tuple(face.point(a, unit.offset)), tuple(face.point(b, unit.offset)), tuple(face.point(b, unit.offset + unit.depth)), tuple(face.point(a, unit.offset + unit.depth))])

    first, last = stretches[0], stretches[-1]
    if first[0] - unit.start <= 0.30 * upm:
        split = min(max(unit.start + module, first[1] + step), unit.start + 1.5 * module)
        if unit.end - split >= 0.30 * upm:
            return [(part(unit.start, split), "KS"), (part(split, unit.end), "HS")]
    if unit.end - last[1] <= 0.30 * upm:
        split = max(min(unit.end - module, last[0] - step), unit.end - 1.5 * module)
        if split - unit.start >= 0.30 * upm:
            return [(part(unit.start, split), "HS"), (part(split, unit.end), "KS")]
    return [(unit.polygon(), "KS")]


def _snap_run_boundaries(runs: list[WallRun], faces: list[_Face], upm: float, settings: DoubleWallSettings) -> int:
    """Move a change of wall kind along a face onto the crossing wall that causes it.

    The kind (interior / exterior) of a face changes where another wall meets it
    from the far side; the sampled boundary lies up to a sample step beside it,
    which would leave a small step in the outer face.
    """
    reach = max(settings.sample_step_m * 2.0, 0.10) * upm
    face_of = {face.index: face for face in faces}
    snapped = 0
    by_face: dict[int, list[WallRun]] = {}
    for run in runs:
        by_face.setdefault(run.face, []).append(run)
    for index, face_runs in by_face.items():
        face_runs.sort(key=lambda item: item.start)
        face = face_of[index]
        crossings: list[float] = []
        for other in faces:
            if other is face or _parallel(face.direction, other.direction, 60.0):
                continue
            segment = LineString([tuple(other.start), tuple(other.end)])
            if segment.distance(LineString([tuple(face.start), tuple(face.end)])) > settings.joint_reach_m * upm:
                continue
            # Intersection of the two infinite lines, as a position along this face.
            d1, d2 = face.direction, other.direction
            denominator = float(d1[0] * d2[1] - d1[1] * d2[0])
            if abs(denominator) <= 1e-9:
                continue
            delta = other.start - face.start
            crossings.append(float((delta[0] * d2[1] - delta[1] * d2[0]) / denominator))
        for first, second in zip(face_runs[:-1], face_runs[1:]):
            boundary = first.end
            candidates = [value for value in crossings if abs(value - boundary) <= reach and first.start < value < second.end]
            if not candidates:
                continue
            target = min(candidates, key=lambda value: abs(value - boundary))
            if abs(target - boundary) > 1e-9:
                first.end = second.start = target
                snapped += 1
    return snapped


def _collinear_groups(faces: list[_Face], upm: float, settings: DoubleWallSettings) -> dict[int, int]:
    parent = list(range(len(faces)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    sine = math.sin(math.radians(settings.pair_angle_deg))
    for first in faces:
        for second in faces:
            if second.index <= first.index or not first.side or not second.side:
                continue
            d1, d2 = first.direction, second.direction
            if abs(float(d1[0] * d2[1] - d1[1] * d2[0])) > sine:
                continue
            if float(np.dot(first.solid_normal, second.solid_normal)) <= 0.0:
                continue
            offset = abs(float(np.dot((second.start + second.end) / 2.0 - first.start, first.normal)))
            if offset <= settings.collinear_offset_m * upm:
                parent[find(second.index)] = find(first.index)
    return {face.index: find(face.index) for face in faces}


def _snap(value: float, clusters: list[tuple[float, int]], tolerance: float) -> float:
    if not clusters:
        return value
    nearest = min(clusters, key=lambda item: abs(item[0] - value))[0]
    return nearest if abs(nearest - value) <= tolerance else value


def _depth_frame(face: _Face, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Along / across coordinates (relative to the face line, ``across`` along ``face.normal``) and height of depth voxels near a face."""
    relative = points[:, :2] - face.start
    return relative @ face.direction, relative @ face.normal, points[:, 2]


def _covered_bins(along: np.ndarray, height: np.ndarray, low: float, count: int, bin_size: float, height_bin: float, settings: DoubleWallSettings) -> np.ndarray:
    """Per along-bin flag: enough voxels spread over enough height bins to be a wall surface (not a lamp, not a floor spill)."""
    covered = np.zeros(count, dtype=bool)
    if not len(along) or count <= 0:
        return covered
    index = np.floor((along - low) / bin_size).astype(np.int64)
    inside = (index >= 0) & (index < count)
    index, bins = index[inside], np.clip(np.floor(height[inside] / max(height_bin, 1e-9)).astype(np.int64), 0, 40)
    if not len(index):
        return covered
    voxels = np.bincount(index, minlength=count)
    pairs = np.unique(index * 64 + bins)
    height_bins = np.bincount(pairs // 64, minlength=count)
    return (voxels >= settings.depth_min_voxels) & (height_bins >= settings.depth_min_height_bins)


def _depth_extend_faces(faces: list[_Face], points: np.ndarray, openings: list[Any], upm: float, settings: DoubleWallSettings, evidence: WallEvidence | None, free: np.ndarray | None = None, grid: RasterGrid | None = None) -> int:
    """Continue a measured face over an unseen stretch where COLMAP depth voxels lie on its own line (within the face tolerance).

    The extension stops before an opening span and never exceeds ``depth_extend_max_m`` per end; a gap bin of one is tolerated.
    With the measured free floor (``free``) the extension also stops at the first cell of walkable floor: a door or passage
    gap has floor in it, an unseen stretch of wall does not.  Returns the number of face ends extended.
    """
    tolerance = settings.depth_face_tolerance_m * upm
    bin_size = settings.depth_bin_m * upm
    reach = settings.depth_extend_max_m * upm
    height_bin = 0.10 * max(float(evidence.storey_height if evidence is not None else 0.0), 1e-9) if evidence is not None and evidence.storey_height else 0.10 * upm
    spans = []
    for opening in openings:
        axis = _axis_of(opening)
        centre = np.asarray(getattr(opening, "center"), dtype=float)[:2]
        half = float(getattr(opening, "width")) / 2.0
        spans.append((centre - axis * half, centre + axis * half))
    extended = 0
    for face in faces:
        if face.length < settings.min_face_m * upm:
            continue
        box = np.maximum(np.abs(points[:, 0] - (face.start[0] + face.end[0]) / 2.0), np.abs(points[:, 1] - (face.start[1] + face.end[1]) / 2.0)) <= face.length / 2.0 + reach + tolerance
        near = points[box]
        if not len(near):
            continue
        along, across, height = _depth_frame(face, near)
        on_line = np.abs(across) <= tolerance
        along, height = along[on_line], height[on_line]
        if not len(along):
            continue
        new_ends: list[np.ndarray | None] = [None, None]
        for end in (0, 1):
            sign = -1.0 if end == 0 else 1.0
            origin = 0.0 if end == 0 else face.length
            low = origin if end == 1 else origin - reach
            count = int(math.ceil(reach / bin_size))
            if end == 1:
                covered = _covered_bins(along, height, low, count, bin_size, height_bin, settings)
            else:
                covered = _covered_bins(along, height, low, count, bin_size, height_bin, settings)[::-1]
            # bin k (k = 0 next to the face end) is covered[k]
            length = 0
            gap = 0
            for flag in covered:
                if flag:
                    length += 1 + gap
                    gap = 0
                else:
                    gap += 1
                    if gap > 1:
                        break
            if length * bin_size < 0.10 * upm:
                continue
            grow = length * bin_size
            if free is not None and grid is not None:
                origin_point = face.start if end == 0 else face.end
                steps = np.arange(0.0, grow + 0.5 * bin_size, 0.02 * upm)
                probes = origin_point[None, :] + sign * face.direction[None, :] * steps[:, None]
                walkable = np.flatnonzero(_lookup(free, grid, probes))
                if len(walkable):
                    grow = max(0.0, float(steps[walkable[0]]) - 0.02 * upm)
                    if grow < 0.10 * upm:
                        continue
            target = (face.start if end == 0 else face.end) + sign * face.direction * grow
            if any(float(np.linalg.norm(target - jamb)) <= 0.05 * upm or LineString([tuple(a), tuple(b)]).distance(Point(float(target[0]), float(target[1]))) <= 0.05 * upm for a, b in spans for jamb in (a, b)):
                continue
            new_ends[end] = target
        if new_ends[0] is not None:
            face.start = new_ends[0]
            extended += 1
        if new_ends[1] is not None:
            face.end = new_ends[1]
            extended += 1
    return extended


def _depth_thickness(runs: list[WallRun], faces: list[_Face], points: np.ndarray, upm: float, settings: DoubleWallSettings, evidence: WallEvidence | None) -> int:
    """Measure the thickness of a wall run whose second face the cloud did not see from COLMAP depth voxels on its solid side.

    Only runs whose thickness is a room-edge / learned estimate are touched.  The far face is the dominant offset (1 cm bins)
    of wall voxels over at least ``depth_cover_min_m`` (and ``depth_cover_min_share`` of the run) of the run, between the minimum and
    maximum wall thickness; its offset becomes the run's thickness (evidence ``depth``).
    """
    low_thickness, high_thickness = settings.min_thickness_m * upm, settings.max_thickness_m * upm
    tolerance = settings.depth_face_tolerance_m * upm
    bin_size = settings.depth_bin_m * upm
    height_bin = 0.10 * evidence.storey_height if evidence is not None and evidence.storey_height else 0.10 * upm
    face_of = {face.index: face for face in faces}
    changed = 0
    for run in runs:
        # Only interior estimates (room edge / learned): an exterior default stays, because the outer face is mostly seen from a window
        # only and a measured interior-looking offset there would step against its (default-thickness) neighbours.
        if run.kind == "partition" or run.evidence not in {"room", "learned"}:
            continue
        face = face_of[run.face]
        normal = face.solid_normal
        direction = face.direction
        length = run.end - run.start
        if length < settings.depth_cover_min_m * upm:
            continue
        middle = face.point((run.start + run.end) / 2.0)
        reach = length / 2.0 + tolerance
        box = np.maximum(np.abs(points[:, 0] - middle[0]), np.abs(points[:, 1] - middle[1])) <= reach + high_thickness + tolerance
        near = points[box]
        if not len(near):
            continue
        relative = near[:, :2] - face.start
        along, across, height = relative @ direction, relative @ normal, near[:, 2]
        keep = (along >= run.start) & (along <= run.end) & (across >= low_thickness - tolerance) & (across <= high_thickness + tolerance)
        along, across, height = along[keep], across[keep], height[keep]
        if len(along) < settings.depth_min_voxels:
            continue
        histogram, edges = np.histogram(across, bins=np.arange(low_thickness - tolerance, high_thickness + 2 * tolerance, 0.01 * upm))
        smoothed = np.convolve(histogram, np.ones(3), mode="same")
        peak = int(np.argmax(smoothed))
        offset = float((edges[peak] + edges[peak + 1]) / 2.0)
        on_face = np.abs(across - offset) <= tolerance
        count = max(1, int(math.ceil(length / bin_size)))
        covered = _covered_bins(along[on_face], height[on_face], run.start, count, bin_size, height_bin, settings)
        covered_length = float(covered.sum()) * bin_size
        if covered_length < min(settings.depth_cover_min_m * upm, length) or covered_length < settings.depth_cover_min_share * length:
            continue
        measured = float(np.median(across[on_face]))
        if not low_thickness <= measured <= high_thickness:
            continue
        run.thickness, run.evidence = measured, "depth"
        changed += 1
    return changed


def _flush_collinear_runs(runs: list[WallRun], faces: list[_Face], upm: float, settings: DoubleWallSettings) -> int:
    """Give chained pieces of one wall a common centreline and thickness.

    A room outline can split a single measured wall around a door and displace one
    fragment by the estimated wall thickness.  Keeping both fragments makes a false
    jog (and sometimes a diagonal bridge) in the wall body.  A real step is retained
    only when both overlapping pieces are independently supported by measured face
    pairs for the configured minimum length.
    """
    if len(runs) < 2:
        return 0
    face_of = {face.index: face for face in faces}
    parent = list(range(len(runs)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(first: int, second: int) -> None:
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    limit = settings.flush_line_offset_m * upm
    chain_gap = settings.flush_chain_gap_m * upm
    proof_length = settings.step_proof_face_length_m * upm
    proof_offset = settings.step_proof_offset_m * upm
    for first_index, first in enumerate(runs):
        first_face = face_of[first.face]
        first_start = _run_centre(first, first_face, first.start)
        first_end = _run_centre(first, first_face, first.end)
        for second_index in range(first_index + 1, len(runs)):
            second = runs[second_index]
            second_face = face_of[second.face]
            if not _parallel(first_face.direction, second_face.direction, settings.pair_angle_deg):
                continue
            if float(np.dot(first_face.solid_normal, second_face.solid_normal)) <= 0.0:
                continue
            offset = abs(float(np.dot(_run_centre(second, second_face, (second.start + second.end) / 2.0) - first_start, first_face.solid_normal)))
            if offset > limit:
                continue
            second_positions = sorted(float(np.dot(point - first_start, first_face.direction)) for point in (_run_centre(second, second_face, second.start), _run_centre(second, second_face, second.end)))
            first_positions = sorted((0.0, float(np.dot(first_end - first_start, first_face.direction))))
            overlap = min(first_positions[1], second_positions[1]) - max(first_positions[0], second_positions[0])
            gap = max(first_positions[0] - second_positions[1], second_positions[0] - first_positions[1], 0.0)
            if gap > chain_gap:
                continue
            # ``face`` is emitted only when the paired face measurements support
            # this run.  Two such long, offset measurements are the available
            # generic proof of an intentional step.
            if offset >= proof_offset and overlap >= proof_length and first.evidence == second.evidence == "face":
                continue
            join(first_index, second_index)

    components: dict[int, list[WallRun]] = {}
    for index, run in enumerate(runs):
        components.setdefault(find(index), []).append(run)
    strength = {"face": 4, "depth": 3, "face-collinear": 3, "face-beside": 3, "room": 2, "niche": 1, "learned": 1, "exterior": 1, "door-host": 1, "door-fill": 1, "closure": 1}
    changed = 0
    for component in components.values():
        if len(component) < 2:
            continue
        best_strength = max(strength.get(run.evidence, 0) for run in component)
        anchors = [run for run in component if strength.get(run.evidence, 0) == best_strength]
        reference = anchors[0]
        reference_face = face_of[reference.face]
        reference_point = _run_centre(reference, reference_face, (reference.start + reference.end) / 2.0)
        weights = np.asarray([max(run.end - run.start, 1e-9) for run in anchors], dtype=float)
        offsets = np.asarray([
            float(np.dot(_run_centre(run, face_of[run.face], (run.start + run.end) / 2.0) - reference_point, reference_face.solid_normal))
            for run in anchors
        ])
        target_offset = float(np.average(offsets, weights=weights))
        target_thickness = float(np.average(np.asarray([run.thickness for run in anchors]), weights=weights))
        target_thickness = min(max(target_thickness, settings.min_thickness_m * upm), settings.max_thickness_m * upm)
        shifted_faces: set[int] = set()
        for run in component:
            face = face_of[run.face]
            if run.face not in shifted_faces:
                # Compare using the shared target thickness: changing a run's
                # thickness must not displace its already selected centreline.
                centre = face.point((run.start + run.end) / 2.0, target_thickness / 2.0)
                delta = target_offset - float(np.dot(centre - reference_point, reference_face.solid_normal))
                if abs(delta) > 1e-9:
                    face.start = face.start + face.solid_normal * delta
                    face.end = face.end + face.solid_normal * delta
                    changed += 1
                shifted_faces.add(run.face)
            if abs(run.thickness - target_thickness) > 1e-9:
                run.thickness = target_thickness
                changed += 1
            run.group = reference.group
    return changed


def _assign_thickness(runs: list[WallRun], faces: list[_Face], samples: list[_Samples], groups: dict[int, int], interior: list[tuple[float, int]], exterior: float, upm: float, settings: DoubleWallSettings, measured_pairs: dict[int, list[tuple[float, float, float, int, bool]]] | None = None) -> None:
    low, high = settings.min_thickness_m * upm, settings.max_thickness_m * upm
    snap = settings.cluster_snap_m * upm
    default_interior = max(interior, key=lambda item: item[1])[0] if interior else settings.interior_thickness_m * upm
    group_pairs: dict[tuple[int, str], list[float]] = {}
    run_values: list[tuple[list[float], list[float]]] = []
    for run in runs:
        face_samples = samples[run.face]
        inside = (face_samples.along >= run.start) & (face_samples.along <= run.end)
        pairs = [float(value) for value in face_samples.pair[inside] if not np.isnan(value)]
        rooms = [float(value) for value in face_samples.far[inside] if not np.isnan(value) and low <= value <= high]
        run_values.append((pairs, rooms))
        run.group = groups.get(run.face, run.face)
        if pairs:
            group_pairs.setdefault((run.group, run.kind), []).extend(pairs)
    lightest = min(interior)[0] if interior else settings.interior_thickness_m * upm
    face_of = {face.index: face for face in faces}
    weak = {index: [item[2] for item in items if not item[4]] for index, items in (measured_pairs or {}).items()}
    for run, (pairs, rooms) in zip(runs, run_values, strict=True):
        if run.kind == "partition":
            # Undecided face: a light partition centred on the measured line.
            face = face_of[run.face]
            run.thickness, run.evidence = float(min(max(lightest, low), high)), "partition"
            face.side = 1
            face.start = face.start - face.normal * run.thickness / 2.0
            face.end = face.end - face.normal * run.thickness / 2.0
            continue
        if face_of[run.face].entity.get("_source") == "niche_panel" and not pairs:
            # The side panel of a recessed niche is a light partition, whatever lies behind it.
            run.kind, run.thickness, run.evidence = "interior", float(min(max(lightest, low), high)), "niche"
            continue
        count = max(1, int(np.sum((samples[run.face].along >= run.start) & (samples[run.face].along <= run.end))))
        if pairs and len(pairs) >= 0.2 * count:
            run.thickness, run.evidence = float(np.median(pairs)), "face"
        elif (run.group, run.kind) in group_pairs:
            run.thickness, run.evidence = float(np.median(group_pairs[(run.group, run.kind)])), "face-collinear"
        elif weak.get(run.face):
            run.thickness, run.evidence = float(np.median(weak[run.face])), "face-beside"
        elif run.kind == "interior" and rooms and len(rooms) >= 0.3 * count:
            run.thickness, run.evidence = _snap(float(np.median(rooms)), interior, snap), "room"
        elif run.kind == "interior":
            run.thickness, run.evidence = default_interior, "learned"
        else:
            run.thickness, run.evidence = exterior, "exterior"
        run.thickness = float(min(max(run.thickness, low), high))
    # Adjacent runs of one wall line with nearly the same thickness share it (no outer-face jog).
    by_group: dict[int, list[WallRun]] = {}
    for run in runs:
        if run.kind != "partition":
            by_group.setdefault(run.group, []).append(run)
    for group_runs in by_group.values():
        if len(group_runs) < 2:
            continue
        strength = {"face": 4, "depth": 3, "face-collinear": 3, "face-beside": 3, "room": 2, "niche": 1, "learned": 1, "exterior": 1}
        ordered = sorted(group_runs, key=lambda item: (-strength.get(item.evidence, 0), -(item.end - item.start)))
        for run in group_runs:
            for anchor in ordered:
                if anchor is run or strength.get(anchor.evidence, 0) < strength.get(run.evidence, 0):
                    continue
                if 0.0 < abs(anchor.thickness - run.thickness) <= settings.cluster_gap_m * upm:
                    run.thickness = anchor.thickness
                break


# --------------------------------------------------------------------------
# wall bodies
# --------------------------------------------------------------------------

def _axis_of(opening: Any) -> np.ndarray:
    angle = float(getattr(opening, "axis_angle", 0.0))
    return np.asarray((math.cos(angle), math.sin(angle)))


def _band(face: _Face, start: float, end: float, thickness: float) -> Polygon:
    a, b = face.point(start), face.point(end)
    c, d = face.point(end, thickness), face.point(start, thickness)
    return Polygon([tuple(a), tuple(b), tuple(c), tuple(d)])


def _slab(face: _Face, thickness: float, centre: np.ndarray, reach: float) -> Polygon:
    """The wall slab of a face (0..thickness on its solid side) around a point."""
    along = float(np.dot(centre - face.start, face.direction))
    return _band(face, along - reach, along + reach, thickness)


def _wall_bodies(runs: list[WallRun], faces: list[_Face], openings: list[Any], upm: float, settings: DoubleWallSettings, roomish: np.ndarray | None = None, grid: RasterGrid | None = None) -> tuple[list[Polygon], list[tuple[float, float]], list[Polygon], list[LineString]]:
    tolerance = settings.joint_tolerance_m * upm
    reach = settings.joint_reach_m * upm
    solid_runs = [run for run in runs if run.thickness > 0.0]
    face_of = {face.index: face for face in faces}
    # Opening intervals per run (to avoid extending into door and window spans).
    opening_geometry = []
    for opening in openings:
        axis = _axis_of(opening)
        centre = np.asarray(getattr(opening, "center"), dtype=float)[:2]
        half = float(getattr(opening, "width")) / 2.0
        opening_geometry.append((opening, axis, centre, half))

    def run_line(run: WallRun) -> LineString:
        face = face_of[run.face]
        return LineString([tuple(face.point(run.start)), tuple(face.point(run.end))])

    def blocked(segment: LineString) -> bool:
        for _opening, axis, centre, half in opening_geometry:
            span = LineString([tuple(centre - axis * (half + 0.02 * upm)), tuple(centre + axis * (half + 0.02 * upm))])
            if segment.distance(span) <= 0.03 * upm and _parallel(np.asarray(segment.coords[1]) - np.asarray(segment.coords[0]), axis, 10.0):
                return True
        return False

    # 1. corner extension: a run end that meets another wall's slab is extended
    #    by that wall's thickness, clipped to its slab (fills the outer corner).
    extra: list[Polygon] = []
    for run in solid_runs:
        face = face_of[run.face]
        for at_start in (True, False):
            position = run.start if at_start else run.end
            end_point = face.point(position)
            outward = -face.direction if at_start else face.direction
            for other in solid_runs:
                if other is run:
                    continue
                other_face = face_of[other.face]
                if _parallel(face.direction, other_face.direction, 30.0):
                    continue
                line = run_line(other)
                other_band = _band(other_face, other.start, other.end, other.thickness)
                if line.distance(Point(tuple(end_point))) > tolerance and other_band.distance(Point(tuple(end_point))) > tolerance:
                    continue
                along = float(np.dot(end_point - other_face.start, other_face.direction))
                if along < other.start - tolerance - other.thickness or along > other.end + tolerance + other.thickness:
                    continue
                length = other.thickness + tolerance
                tip = end_point + outward * length
                extension = Polygon([tuple(end_point), tuple(tip), tuple(tip + face.solid_normal * run.thickness), tuple(end_point + face.solid_normal * run.thickness)])
                slab = _slab(other_face, other.thickness, end_point, max(run.thickness, other.thickness) + 2 * tolerance)
                piece = extension.intersection(slab)
                if not piece.is_empty and piece.area > 0.0:
                    extra.append(piece if isinstance(piece, Polygon) else piece.convex_hull)
    bands = [_band(face_of[run.face], run.start, run.end, run.thickness) for run in solid_runs]
    # 2. free ends within reach of another body are closed along the wall axis.
    union = unary_union(bands + extra)
    for run, band in zip(solid_runs, bands, strict=True):
        face = face_of[run.face]
        others = unary_union([item for item in bands + extra if item is not band]) if len(bands) > 1 else GeometryCollection()
        if others.is_empty:
            continue
        for at_start in (True, False):
            position = run.start if at_start else run.end
            outward = -face.direction if at_start else face.direction
            middle = face.point(position, run.thickness / 2.0)
            probe_edge = LineString([tuple(face.point(position, 0.0)), tuple(face.point(position, run.thickness))])
            if probe_edge.distance(others) <= 0.01 * upm:
                continue
            distance = None
            for fraction in (0.05, 0.5, 0.95):
                origin = face.point(position, run.thickness * fraction)
                ray = LineString([tuple(origin), tuple(origin + outward * reach)])
                hit = ray.intersection(others)
                if hit.is_empty:
                    continue
                value = min(Point(tuple(origin)).distance(geometry) for geometry in _parts(hit))
                if value > 0.0 and (distance is None or value < distance):
                    distance = value
            if distance is None or distance > reach:
                continue
            if blocked(LineString([tuple(middle), tuple(middle + outward * distance)])):
                continue
            gap = distance + 0.01 * upm
            if at_start:
                run.extend_start = max(run.extend_start, gap)
            else:
                run.extend_end = max(run.extend_end, gap)
    bands = [_band(face_of[run.face], run.start - run.extend_start, run.end + run.extend_end, run.thickness) for run in solid_runs]
    extra.extend(_corner_pieces(solid_runs, face_of, bands, tolerance))
    if roomish is not None and grid is not None:
        extra.extend(_bridge_staggered(solid_runs, face_of, roomish, grid, upm, settings))
    # 3. opening slabs: every opening cuts the full thickness of its host wall.
    slabs: list[tuple[float, float]] = []
    cuts: list[Polygon] = []
    cut_hosts: list[tuple[np.ndarray, list[WallRun]]] = []
    jambs: list[LineString] = []
    for opening, axis, centre, half in opening_geometry:
        normal = np.asarray((-axis[1], axis[0]))
        lows: list[float] = []
        highs: list[float] = []
        hosts: list[WallRun] = []
        for run in solid_runs:
            face = face_of[run.face]
            if not _parallel(face.direction, axis, 10.0):
                continue
            offset = float(np.dot(centre - face.start, face.normal))
            near = abs(offset)
            if near > max(0.15 * upm, run.thickness + 0.05 * upm):
                continue
            along = float(np.dot(centre - face.start, face.direction))
            if along + half < run.start - run.extend_start - settings.jamb_reach_m * upm or along - half > run.end + run.extend_end + settings.jamb_reach_m * upm:
                continue
            # The opening centre must lie on or inside the slab of this run (with tolerance).
            across = float(np.dot(centre - face.start, face.solid_normal))
            if across < -0.15 * upm or across > run.thickness + 0.15 * upm:
                continue
            face_across = float(np.dot(face.start - centre, normal))
            far_across = float(np.dot(face.point(0.0, run.thickness) - centre, normal))
            lows.append(min(face_across, far_across))
            highs.append(max(face_across, far_across))
            hosts.append(run)
        if not hosts:
            # No wall run carries the opening, e.g. a door that fills a whole short wall segment (no wall data inside the gap).
            # The collinear wall run beside the gap gives the slab of a door, so the gap is closed like any other cut; without one
            # (and for a window) the opening cuts nothing (a symbol in a wall gap).
            fallback = _neighbour_slab(centre, axis, half, solid_runs, face_of, upm, settings) if str(getattr(opening, "kind", "")) in {"DOOR", "passage"} else None
            if fallback is None:
                slabs.append((0.0, 0.0))
                continue
            low, high = fallback
        else:
            # The thickest consistent host decides the slab (a thin face run beside a thick wall is part of it).
            low, high = float(min(lows)), float(max(highs))
        if hosts and high - low > settings.max_thickness_m * upm + 0.05 * upm:
            best = max(range(len(hosts)), key=lambda item: hosts[item].end - hosts[item].start)
            low, high = lows[best], highs[best]
        slabs.append((low, high))
        margin = 0.01 * upm
        cut = Polygon([
            tuple(centre - axis * half + normal * (low - margin)), tuple(centre + axis * half + normal * (low - margin)),
            tuple(centre + axis * half + normal * (high + margin)), tuple(centre - axis * half + normal * (high + margin)),
        ])
        cuts.append(cut)
        cut_hosts.append((axis, hosts))
        for sign in (-1.0, 1.0):
            jambs.append(LineString([tuple(centre + sign * axis * half + normal * low), tuple(centre + sign * axis * half + normal * high)]))
        # Host runs that stop just short of a jamb are extended to it, so both faces end at the same jamb.
        for run in hosts:
            face = face_of[run.face]
            along = float(np.dot(centre - face.start, face.direction))
            first, second = along - half, along + half
            jamb_reach = settings.jamb_reach_m * upm
            start, end = run.start - run.extend_start, run.end + run.extend_end
            if first - jamb_reach <= end < first:
                run.extend_end += first - end
            if second < start <= second + jamb_reach:
                run.extend_start += start - second
    _trim_overshoots(solid_runs, face_of, upm, settings)
    bands = [_band(face_of[run.face], run.start - run.extend_start, run.end + run.extend_end, run.thickness) for run in solid_runs]
    # A door or window cuts its own wall only: where its span reaches into a crossing
    # wall (detected opening a few centimetres too wide), the crossing wall stays whole.
    trimmed: list[Polygon] = []
    stub = settings.min_stub_m * upm
    for cut, (axis, hosts) in zip(cuts, cut_hosts, strict=True):
        host_ids = {id(run) for run in hosts}
        crossing = [band for run, band in zip(solid_runs, bands, strict=True) if id(run) not in host_ids and not _parallel(face_of[run.face].direction, axis, 30.0)]
        if crossing and stub > 0.0:
            # A wall stub shorter than ``min_stub_m`` between an opening and a crossing wall is
            # not built: the opening reaches the crossing wall (no sliver jamb).
            near = unary_union([band for band in crossing if band.distance(cut) <= stub])
            if not near.is_empty:
                corners = np.asarray(cut.exterior.coords[:4], dtype=float)
                along = (corners - corners.mean(axis=0)) @ axis
                lo, hi = float(along.min()), float(along.max())
                normal = np.asarray((-axis[1], axis[0]))
                across = (corners - corners.mean(axis=0)) @ normal
                centre = corners.mean(axis=0)
                for sign in (-1.0, 1.0):
                    edge = centre + axis * (hi if sign > 0 else lo)
                    probes = [LineString([tuple(edge + normal * offset), tuple(edge + normal * offset + sign * axis * stub)]) for offset in np.linspace(float(across.min()), float(across.max()), 5)]
                    gaps = [Point(probe.coords[0]).distance(probe.intersection(near)) for probe in probes if probe.intersects(near)]
                    if len(gaps) >= 3:
                        gap = max(gaps)
                        if sign > 0:
                            hi += gap
                        else:
                            lo -= gap
                cut = Polygon([
                    tuple(centre + axis * lo + normal * float(across.min())), tuple(centre + axis * hi + normal * float(across.min())),
                    tuple(centre + axis * hi + normal * float(across.max())), tuple(centre + axis * lo + normal * float(across.max())),
                ])
        blockers = [band for band in crossing if band.intersects(cut)]
        if blockers:
            reduced = cut.difference(unary_union(blockers))
            trimmed.extend(part for part in _parts(reduced) if isinstance(part, Polygon) and part.area > 1e-12)
        else:
            trimmed.append(cut)
    return bands + extra, slabs, trimmed, jambs


def _bridge_staggered(runs: list[WallRun], face_of: dict[int, _Face], roomish: np.ndarray, grid: RasterGrid, upm: float, settings: DoubleWallSettings) -> list[Polygon]:
    """Close the step between two parallel wall bodies that end side by side with a small gap between them.

    Where one wall line steps to another (e.g. a facade recess) and the face of the
    step was not measured, the two bodies overlap along the wall only near their ends
    and a strip of at most ``max_thickness_m`` without any room lies between them.
    """
    pieces: list[Polygon] = []
    limit = settings.max_thickness_m * upm
    for index, first in enumerate(runs):
        face = face_of[first.face]
        for second in runs[index + 1:]:
            other = face_of[second.face]
            if not _parallel(face.direction, other.direction, 3.0):
                continue
            normal = face.normal
            a_low, a_high = sorted((0.0, first.thickness * (1.0 if face.side >= 0 else -1.0)))
            offset = float(np.dot(other.start - face.start, normal))
            sign = 1.0 if float(np.dot(other.solid_normal, normal)) > 0 else -1.0
            b_low, b_high = sorted((offset, offset + sign * second.thickness))
            if a_high < b_low:
                gap_low, gap_high = a_high, b_low
            elif b_high < a_low:
                gap_low, gap_high = b_high, a_low
            else:
                continue
            if not 0.005 * upm < gap_high - gap_low <= limit:
                continue
            span_a = (first.start - first.extend_start, first.end + first.extend_end)
            span_b = sorted(float(np.dot(other.point(value) - face.start, face.direction)) for value in (second.start - second.extend_start, second.end + second.extend_end))
            low, high = max(span_a[0], span_b[0]), min(span_a[1], span_b[1])
            # Only an end-to-end overlap (a step), never two long parallel walls with a gap.
            if high - low < -0.20 * upm or high - low > 0.6 * upm:
                continue
            if high < low:
                low, high = high, low  # the two ends are apart: the step wall fills the gap between them
            elif high - low < 0.02 * upm:
                middle = (low + high) / 2.0
                low, high = middle - 0.01 * upm, middle + 0.01 * upm
            corners = [face.start + face.direction * along + normal * across for along, across in ((low, gap_low), (high, gap_low), (high, gap_high), (low, gap_high))]
            strip = Polygon([tuple(point) for point in corners])
            centre = np.asarray(strip.centroid.coords[0])
            samples = centre[None, :] + np.asarray([[dx, dy] for dx in (-0.4, 0.0, 0.4) for dy in (-0.4, 0.0, 0.4)]) * np.asarray([high - low, gap_high - gap_low])[None, :] @ np.vstack((face.direction, normal))
            if bool(np.any(_lookup(roomish, grid, samples))):
                continue
            full_low, full_high = min(a_low, b_low), max(a_high, b_high)
            corners = [face.start + face.direction * along + normal * across for along, across in ((low, full_low), (high, full_low), (high, full_high), (low, full_high))]
            pieces.append(Polygon([tuple(point) for point in corners]).buffer(0.005 * upm, join_style=2))
    return pieces


def _corner_pieces(runs: list[WallRun], face_of: dict[int, _Face], bands: list[Polygon], tolerance: float) -> list[Polygon]:
    """Corner squares where an (extended) run end meets a perpendicular wall body."""
    pieces: list[Polygon] = []
    for run in runs:
        face = face_of[run.face]
        for at_start in (True, False):
            position = run.start - run.extend_start if at_start else run.end + run.extend_end
            end_point = face.point(position)
            outward = -face.direction if at_start else face.direction
            for other, band in zip(runs, bands, strict=True):
                if other is run:
                    continue
                other_face = face_of[other.face]
                if _parallel(face.direction, other_face.direction, 30.0):
                    continue
                if band.distance(Point(tuple(end_point))) > tolerance and band.distance(Point(tuple(face.point(position, run.thickness)))) > tolerance:
                    continue
                length = other.thickness + tolerance
                tip = end_point + outward * length
                extension = Polygon([tuple(end_point), tuple(tip), tuple(tip + face.solid_normal * run.thickness), tuple(end_point + face.solid_normal * run.thickness)])
                slab = _slab(other_face, other.thickness, end_point, max(run.thickness, other.thickness) + 2 * tolerance)
                piece = extension.intersection(slab)
                if not piece.is_empty and piece.area > 0.0:
                    pieces.append(piece if isinstance(piece, Polygon) else piece.convex_hull)
    return pieces


def _trim_overshoots(runs: list[WallRun], face_of: dict[int, _Face], upm: float, settings: DoubleWallSettings) -> None:
    """A wall end that passes a crossing wall body by less than ``min_stub_m`` stops at that body's far face."""
    stub = settings.min_stub_m * upm
    if stub <= 0.0:
        return
    bands = [(run, _band(face_of[run.face], run.start - run.extend_start, run.end + run.extend_end, run.thickness)) for run in runs]
    for run in runs:
        face = face_of[run.face]
        start, end = run.start - run.extend_start, run.end + run.extend_end
        middle_offset = run.thickness / 2.0
        centre_line = LineString([tuple(face.point(start, middle_offset)), tuple(face.point(end, middle_offset))])
        for other, band in bands:
            if other is run or _parallel(face.direction, face_of[other.face].direction, 30.0):
                continue
            inside = centre_line.intersection(band)
            if inside.is_empty:
                continue
            positions = [float(np.dot(np.asarray(point) - face.start, face.direction)) for part in _parts(inside) for point in getattr(part, "coords", [])]
            if not positions:
                continue
            low, high = min(positions), max(positions)
            if 1e-6 < end - high < stub and high > start + stub:
                run.extend_end -= end - high
                end = high
            if 1e-6 < low - start < stub and low < end - stub:
                run.extend_start -= low - start
                start = low


def _parts(geometry: Any) -> list[Any]:
    if hasattr(geometry, "geoms"):
        return [part for item in geometry.geoms for part in _parts(item)]
    return [geometry]


def _parallel(first: np.ndarray, second: np.ndarray, tolerance_deg: float) -> bool:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norm <= 1e-12:
        return False
    cross = abs(float(first[0] * second[1] - first[1] * second[0])) / norm
    return math.degrees(math.asin(min(1.0, cross))) <= tolerance_deg


def _clean_union(polygons: list[Polygon], upm: float, settings: DoubleWallSettings) -> Any:
    valid = [make_valid(polygon) for polygon in polygons if polygon.area > 1e-12]
    if not valid:
        return GeometryCollection()
    union = unary_union(valid)
    # Close hairline gaps between touching bands without moving faces.
    epsilon = 0.002 * upm
    return union.buffer(epsilon, join_style=2).buffer(-epsilon, join_style=2)


def _polygons_of(geometry: Any, upm: float, settings: DoubleWallSettings) -> list[Polygon]:
    minimum = settings.min_piece_area_m2 * upm * upm
    jog = settings.jog_m * upm
    output: list[Polygon] = []
    if jog > 0.0 and not geometry.is_empty:
        # Close notches and drop spikes narrower than ``jog_m`` (numerical remnants of joints).
        half = jog / 2.0
        geometry = geometry.buffer(half, join_style=2).buffer(-half, join_style=2)
        geometry = geometry.buffer(-half, join_style=2).buffer(half, join_style=2)
    for part in _parts(geometry):
        if not isinstance(part, Polygon) or part.is_empty or part.area < minimum:
            continue
        part = _remove_jogs(part.simplify(0.002 * upm, preserve_topology=True), jog)
        simplified = part.simplify(0.003 * upm, preserve_topology=True)
        if isinstance(simplified, Polygon) and simplified.is_valid and simplified.area >= minimum:
            # A sliver thinner than 2 cm is a numerical remnant, not a wall.
            if simplified.buffer(-0.01 * upm).is_empty:
                continue
            output.append(simplified)
    output.sort(key=lambda polygon: (-polygon.area, polygon.bounds))
    return output


def _remove_jogs(polygon: Polygon, tolerance: float) -> Polygon:
    """Remove orthogonal steps shorter than ``tolerance`` from every ring.

    A step is a short edge whose two neighbours run in the same direction (two
    nearly collinear faces offset by a few centimetres).  The shorter neighbour
    is moved onto the longer one's line, so the faces stay straight and parallel.
    """
    if tolerance <= 0.0 or not isinstance(polygon, Polygon) or polygon.is_empty:
        return polygon

    def clean(coordinates: list[tuple[float, float]]) -> list[tuple[float, float]]:
        points = [np.asarray(point, dtype=float) for point in coordinates[:-1]]
        for _ in range(4 * len(points)):
            count = len(points)
            if count < 5:
                break
            changed = False
            for index in range(count):
                a, b, c, d = (points[(index + offset) % count] for offset in (-1, 0, 1, 2))
                z = points[(index - 2) % count]
                e = points[(index + 3) % count]
                step = c - b
                if float(np.linalg.norm(step)) >= tolerance or float(np.linalg.norm(step)) <= 1e-12:
                    continue
                before, after = b - a, d - c
                if float(np.linalg.norm(before)) <= 1e-12 or float(np.linalg.norm(after)) <= 1e-12:
                    continue
                if not _parallel(before, after, 2.0) or float(np.dot(before, after)) <= 0.0:
                    continue
                if float(np.linalg.norm(before)) <= float(np.linalg.norm(after)):
                    # Shift edge a-b by the step; a slides along its own incoming edge z-a.
                    if not _parallel(a - z, step, 2.0):
                        continue
                    points[(index - 1) % count] = a + step
                    del points[index % count]
                else:
                    if not _parallel(e - d, step, 2.0):
                        continue
                    points[(index + 2) % count] = d - step
                    del points[(index + 1) % count]
                changed = True
                break
            if not changed:
                break
        return [tuple(point) for point in points] + [tuple(points[0])]

    try:
        shell = clean(list(polygon.exterior.coords))
        holes = [clean(list(ring.coords)) for ring in polygon.interiors]
        candidate = Polygon(shell, [hole for hole in holes if len(hole) >= 4])
    except (ValueError, IndexError):
        return polygon
    if not candidate.is_valid or abs(candidate.area - polygon.area) > max(0.02 * polygon.area, tolerance * tolerance * 50):
        return polygon
    return candidate


def _ring(coordinates: Any) -> list[list[float]]:
    points = [[float(x), float(y)] for x, y, *_ in list(coordinates)]
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    return points


def _classify_outline(polygons: list[Polygon], runs: list[WallRun], faces: list[_Face], jambs: list[LineString], upm: float) -> tuple[list[LineString], list[LineString]]:
    """Split the wall outlines into face lines and end caps / jambs."""
    face_of = {face.index: face for face in faces}
    caps_reference: list[LineString] = list(jambs)
    solid = [run for run in runs if run.thickness > 0.0]
    bands = [_band(face_of[run.face], run.start - run.extend_start, run.end + run.extend_end, run.thickness) for run in solid]
    for index, run in enumerate(solid):
        face = face_of[run.face]
        others = [band for other, band in enumerate(bands) if other != index]
        for position in (run.start - run.extend_start, run.end + run.extend_end):
            cap = LineString([tuple(face.point(position)), tuple(face.point(position, run.thickness))])
            # Only a free end is an end cap; where another wall body meets it the edge is a face (step, corner).
            if any(band.distance(cap) <= 0.05 * upm and band.intersection(cap.buffer(0.05 * upm)).area > 0.0 for band in others):
                continue
            caps_reference.append(cap)
    reference = unary_union(caps_reference) if caps_reference else GeometryCollection()
    face_lines: list[LineString] = []
    cap_lines: list[LineString] = []
    tolerance = 0.012 * upm
    for polygon in polygons:
        for ring in [polygon.exterior, *polygon.interiors]:
            coordinates = list(ring.coords)
            for first, second in zip(coordinates[:-1], coordinates[1:]):
                edge = LineString([first, second])
                if edge.length <= 1e-9:
                    continue
                is_cap = False
                if not reference.is_empty and edge.length <= 0.5 * upm:
                    covered = edge.intersection(reference.buffer(tolerance, cap_style=2)).length
                    is_cap = covered >= 0.8 * edge.length
                (cap_lines if is_cap else face_lines).append(edge)
    return face_lines, cap_lines


# --------------------------------------------------------------------------
# room helpers
# --------------------------------------------------------------------------

def _largest_overlap(geometry: Any, reference: Polygon) -> Polygon | None:
    parts = [part for part in _parts(geometry) if isinstance(part, Polygon) and not part.is_empty]
    if not parts:
        return None
    return max(parts, key=lambda part: (part.intersection(reference).area, part.area))


def _contact_length(contact: Any) -> float:
    parts = [part for part in _parts(contact) if isinstance(part, Polygon)]
    if not parts:
        return 0.0
    best = max(parts, key=lambda part: part.area)
    rectangle = best.minimum_rotated_rectangle
    coordinates = list(rectangle.exterior.coords)
    if len(coordinates) < 4:
        return 0.0
    return max(math.dist(coordinates[0], coordinates[1]), math.dist(coordinates[1], coordinates[2]))
