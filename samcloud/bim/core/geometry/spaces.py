"""Space geometry derived from the walls, slabs and ceilings of a storey.

Interior polygons: the union of all wall footprints of a storey is closed against hairline gaps
(1e-6 m), subtracted from the bounding box of the walls (enlarged by 1 m), and every resulting
polygon that does not touch the box border is a room polygon. Islands (for example a wall piece
inside a room) become holes of the room.

Spaces with an explicit ``SpaceParams.outline`` (rooms from a plan model) use that outline directly
and skip the matching below.

Matching spaces to polygons. Order of attempts per space:

1. a seed point (the ``seeds`` argument; ``compute_scene_geometry`` fills it from
   ``SpaceParams.seed`` unless an explicit entry exists): the polygon that covers it;
2. the walls that list the space in ``space_ids``: the polygon that contains the centroid of their
   footprints, else the polygon sharing the most boundary length with them;
3. otherwise by deterministic order. A single remaining space and a single remaining polygon are
   assigned silently. More than one: spaces are sorted by ``(number, name, id)``, polygons by
   ``(min x, min y, area)`` and assigned pairwise, and every such space gets an issue "ambiguous".
   Spaces without a polygon get an issue and no geometry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from samcloud.bim.core.geometry._math import Vec2, oriented, shifted, unshifted
from samcloud.bim.core.geometry.slabs import CeilingGeometry, SlabGeometry, local_polygon
from samcloud.bim.core.geometry.walls import WallGeometry
from samcloud.bim.core.model import SpaceElement, Storey

CLOSING = 1e-6
"""Gaps narrower than twice this value between wall footprints are closed (m)."""
BORDER_MARGIN = 1.0
MIN_ROOM_AREA = 1e-6
"""Smallest polygon area that counts as a room (m^2)."""
TOUCH_TOLERANCE = 1e-3
"""Walls closer than this to the room boundary bound the room (m)."""
MIN_OVERLAP_AREA = 1e-6


@dataclass(frozen=True)
class RoomPolygon:
    """Interior polygon of a storey with its bounding elements (project coordinates)."""

    storey_id: str | None
    exterior: tuple[Vec2, ...]
    holes: tuple[tuple[Vec2, ...], ...]
    gross_area: float
    net_area: float
    perimeter: float
    representative_point: Vec2
    bounding_wall_ids: tuple[str, ...]
    floor_ids: tuple[str, ...]
    ceiling_ids: tuple[str, ...]

    def polygon(self, origin: Vec2 = (0.0, 0.0)) -> Polygon:
        """Net polygon shifted by ``origin``."""
        return local_polygon(self.exterior, self.holes, origin)


@dataclass(frozen=True)
class SpaceGeometry:
    """Geometry and quantities of a space."""

    element_id: str
    storey_id: str | None
    exterior: tuple[Vec2, ...]
    holes: tuple[tuple[Vec2, ...], ...]
    floor_z: float
    ceiling_z: float
    height: float
    height_source: str
    gross_area: float
    net_area: float
    perimeter: float
    bounding_wall_ids: tuple[str, ...]
    floor_ids: tuple[str, ...]
    ceiling_ids: tuple[str, ...]
    matched_by: str
    issues: tuple[str, ...] = ()

    @property
    def gross_volume(self) -> float:
        return self.gross_area * self.height

    @property
    def net_volume(self) -> float:
        return self.net_area * self.height

    @property
    def quantities(self) -> dict[str, float]:
        """IFC4 Qto_SpaceBaseQuantities names."""
        return {
            "Height": self.height,
            "GrossFloorArea": self.gross_area,
            "NetFloorArea": self.net_area,
            "GrossPerimeter": self.perimeter,
            "GrossVolume": self.gross_volume,
            "NetVolume": self.net_volume,
        }

    def polygon(self, origin: Vec2 = (0.0, 0.0)) -> Polygon:
        """Net polygon shifted by ``origin``."""
        return local_polygon(self.exterior, self.holes, origin)


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if not part.is_empty]
    return [
        part
        for part in getattr(geometry, "geoms", ())
        if isinstance(part, Polygon) and not part.is_empty
    ]


def _ring(coords: Sequence[tuple[float, float]], origin: Vec2, ccw: bool) -> tuple[Vec2, ...]:
    points = [(float(x), float(y)) for x, y in coords[:-1]]
    return unshifted(oriented(points, ccw), origin)


def derive_storey_rooms(
    storey_id: str | None,
    walls: Sequence[WallGeometry],
    slabs: Sequence[SlabGeometry] = (),
    ceilings: Sequence[CeilingGeometry] = (),
    origin: Vec2 = (0.0, 0.0),
) -> list[RoomPolygon]:
    """Return all interior polygons of a storey with their bounding element ids.

    Only elements whose ``storey_id`` equals ``storey_id`` are used. The result is sorted by
    ``(min x, min y, area)`` and therefore deterministic. A later work package can match these
    polygons to spaces by a seed point.
    """
    storey_walls = [wall for wall in walls if wall.storey_id == storey_id]
    if not storey_walls:
        return []
    footprints = {
        wall.element_id: Polygon(wall.footprint_relative_to(origin)) for wall in storey_walls
    }
    union = unary_union(list(footprints.values()))
    union = union.buffer(CLOSING, join_style=2, mitre_limit=10.0).buffer(
        -CLOSING, join_style=2, mitre_limit=10.0
    )
    min_x, min_y, max_x, max_y = union.bounds
    frame = box(
        min_x - BORDER_MARGIN, min_y - BORDER_MARGIN, max_x + BORDER_MARGIN, max_y + BORDER_MARGIN
    )
    rooms: list[Polygon] = []
    for candidate in _polygons(frame.difference(union)):
        if candidate.area < MIN_ROOM_AREA or candidate.intersects(frame.exterior):
            continue
        rooms.append(candidate)
    rooms.sort(key=lambda p: (p.bounds[0], p.bounds[1], p.area))

    floors = {s.element_id: s.polygon(origin) for s in slabs if s.storey_id == storey_id}
    plates = {c.element_id: c.polygon(origin) for c in ceilings if c.storey_id == storey_id}
    result: list[RoomPolygon] = []
    for room in rooms:
        simplified = room.simplify(1e-9, preserve_topology=True)
        if not isinstance(simplified, Polygon) or simplified.is_empty:
            simplified = room
        boundary = simplified.boundary
        point = simplified.representative_point()
        result.append(
            RoomPolygon(
                storey_id=storey_id,
                exterior=_ring(simplified.exterior.coords, origin, True),
                holes=tuple(_ring(ring.coords, origin, False) for ring in simplified.interiors),
                gross_area=float(Polygon(simplified.exterior).area),
                net_area=float(simplified.area),
                perimeter=float(simplified.exterior.length),
                representative_point=(float(point.x) + origin[0], float(point.y) + origin[1]),
                bounding_wall_ids=tuple(
                    wall_id
                    for wall_id, footprint in footprints.items()
                    if footprint.distance(boundary) <= TOUCH_TOLERANCE
                ),
                floor_ids=tuple(
                    key
                    for key, polygon in floors.items()
                    if polygon.intersection(simplified).area > MIN_OVERLAP_AREA
                ),
                ceiling_ids=tuple(
                    key
                    for key, polygon in plates.items()
                    if polygon.intersection(simplified).area > MIN_OVERLAP_AREA
                ),
            )
        )
    return result


def _levels(
    room: RoomPolygon,
    slabs: Mapping[str, SlabGeometry],
    ceilings: Mapping[str, CeilingGeometry],
    walls: Mapping[str, WallGeometry],
    storey: Storey | None,
    origin: Vec2,
) -> tuple[float, float, str]:
    """Return ``(floor top z, ceiling bottom z, height source)`` for a room."""
    point = Point(
        room.representative_point[0] - origin[0], room.representative_point[1] - origin[1]
    )
    floor_tops = [
        slabs[key].top_z
        for key in room.floor_ids
        if key in slabs and slabs[key].polygon(origin).covers(point)
    ]
    ceiling_bottoms = [
        ceilings[key].bottom_z_at(*room.representative_point)
        for key in room.ceiling_ids
        if key in ceilings and ceilings[key].polygon(origin).covers(point)
    ]
    floor_z = max(floor_tops) if floor_tops else (storey.elevation if storey is not None else 0.0)
    if ceiling_bottoms:
        ceiling_z = min(ceiling_bottoms)
        if ceiling_z > floor_z:
            return floor_z, ceiling_z, "ceiling"
    if storey is not None and storey.height is not None:
        return floor_z, floor_z + storey.height, "storey_height"
    heights = [walls[key].height for key in room.bounding_wall_ids if key in walls]
    if heights:
        return floor_z, floor_z + max(heights), "wall_height"
    return floor_z, floor_z, "none"


def _shared_length(room: Polygon, footprints: Sequence[Polygon]) -> float:
    return math.fsum(
        float(room.boundary.intersection(f.buffer(TOUCH_TOLERANCE)).length) for f in footprints
    )


def _assign(
    claimed: dict[int, str],
    chosen: dict[str, tuple[int, str, list[str]]],
    space_id: str,
    index: int,
    method: str,
    notes: list[str],
) -> None:
    if index in claimed:
        notes.append(f"polygon is also claimed by space {claimed[index]!r}")
    else:
        claimed[index] = space_id
    chosen[space_id] = (index, method, notes)


def _explicit_space_geometry(
    space: SpaceElement,
    storey_id: str | None,
    storey: Storey | None,
    walls: Mapping[str, WallGeometry],
    slabs: Mapping[str, SlabGeometry],
    ceilings: Mapping[str, CeilingGeometry],
    origin: Vec2,
) -> SpaceGeometry:
    """Space geometry from ``SpaceParams.outline`` (no room derivation from the wall network)."""
    params = space.params
    assert params.outline is not None
    exterior = oriented([(float(x), float(y)) for x, y in params.outline], True)
    holes = tuple(oriented([(float(x), float(y)) for x, y in hole], False) for hole in params.holes)
    polygon = local_polygon(exterior, holes, origin)
    issues: list[str] = []
    if not polygon.is_valid:
        issues.append("space outline is not a valid polygon (self-intersection)")
    boundary = polygon.boundary
    footprints = {
        key: Polygon(wall.footprint_relative_to(origin))
        for key, wall in walls.items()
        if wall.storey_id == storey_id
    }
    point = polygon.representative_point()
    shell = Polygon(shifted(exterior, origin))
    room = RoomPolygon(
        storey_id=storey_id,
        exterior=exterior,
        holes=holes,
        gross_area=float(shell.area),
        net_area=float(polygon.area),
        perimeter=float(shell.exterior.length),
        representative_point=(float(point.x) + origin[0], float(point.y) + origin[1]),
        bounding_wall_ids=tuple(
            key
            for key, footprint in footprints.items()
            if footprint.distance(boundary) <= TOUCH_TOLERANCE
        ),
        floor_ids=tuple(
            key
            for key, slab in slabs.items()
            if slab.storey_id == storey_id
            and slab.polygon(origin).intersection(polygon).area > MIN_OVERLAP_AREA
        ),
        ceiling_ids=tuple(
            key
            for key, ceiling in ceilings.items()
            if ceiling.storey_id == storey_id
            and ceiling.polygon(origin).intersection(polygon).area > MIN_OVERLAP_AREA
        ),
    )
    floor_z, ceiling_z, source = _levels(room, slabs, ceilings, walls, storey, origin)
    if source == "none":
        issues.append("space height unknown: no ceiling, storey height or wall")
    elif source != "ceiling":
        issues.append(f"space height from {source}: no ceiling above the floor found")
    return SpaceGeometry(
        element_id=space.id,
        storey_id=storey_id,
        exterior=room.exterior,
        holes=room.holes,
        floor_z=floor_z,
        ceiling_z=ceiling_z,
        height=ceiling_z - floor_z,
        height_source=source,
        gross_area=room.gross_area,
        net_area=room.net_area,
        perimeter=room.perimeter,
        bounding_wall_ids=room.bounding_wall_ids,
        floor_ids=room.floor_ids,
        ceiling_ids=room.ceiling_ids,
        matched_by="outline",
        issues=tuple(issues),
    )


def compute_space_geometries(
    spaces: Sequence[SpaceElement],
    walls: Mapping[str, WallGeometry],
    slabs: Mapping[str, SlabGeometry],
    ceilings: Mapping[str, CeilingGeometry],
    storeys: Mapping[str, Storey],
    origin: Vec2 = (0.0, 0.0),
    seeds: Mapping[str, Vec2] | None = None,
) -> tuple[dict[str, SpaceGeometry], dict[str, list[str]]]:
    """Derive the geometry of all spaces.

    Returns ``(geometries by space id, issues by space id for spaces without geometry)``.
    Matching rules are described in the module docstring.
    """
    seeds = seeds or {}
    geometries: dict[str, SpaceGeometry] = {}
    failures: dict[str, list[str]] = {}
    storey_ids: list[str | None] = []
    for space in spaces:
        if space.storey_id not in storey_ids:
            storey_ids.append(space.storey_id)

    for storey_id in storey_ids:
        group = [space for space in spaces if space.storey_id == storey_id]
        explicit = [space for space in group if space.params.outline is not None]
        if explicit:
            storey = storeys.get(storey_id) if storey_id is not None else None
            for space in explicit:
                geometries[space.id] = _explicit_space_geometry(
                    space, storey_id, storey, walls, slabs, ceilings, origin
                )
            group = [space for space in group if space.params.outline is None]
            if not group:
                continue
        rooms = derive_storey_rooms(
            storey_id, list(walls.values()), list(slabs.values()), list(ceilings.values()), origin
        )
        polygons = [room.polygon(origin) for room in rooms]
        storey = storeys.get(storey_id) if storey_id is not None else None
        claimed: dict[int, str] = {}
        chosen: dict[str, tuple[int, str, list[str]]] = {}

        pending: list[SpaceElement] = []
        for space in group:
            if space.id in seeds:
                seed = Point(seeds[space.id][0] - origin[0], seeds[space.id][1] - origin[1])
                hits = [i for i, polygon in enumerate(polygons) if polygon.covers(seed)]
                if hits:
                    _assign(claimed, chosen, space.id, hits[0], "seed", [])
                else:
                    failures.setdefault(space.id, []).append("seed lies in no room polygon")
                continue
            bounding = [
                w for w in walls.values() if w.storey_id == storey_id and space.id in w.space_ids
            ]
            if bounding:
                footprints = [Polygon(w.footprint_relative_to(origin)) for w in bounding]
                centroid = unary_union(footprints).centroid
                hits = [i for i, polygon in enumerate(polygons) if polygon.covers(centroid)]
                if hits:
                    _assign(claimed, chosen, space.id, hits[0], "bounding_walls", [])
                    continue
                shares = [_shared_length(polygon, footprints) for polygon in polygons]
                if shares and max(shares) > 0.0:
                    _assign(
                        claimed,
                        chosen,
                        space.id,
                        shares.index(max(shares)),
                        "bounding_walls_shared",
                        [],
                    )
                    continue
                failures.setdefault(space.id, []).append("bounding walls touch no room polygon")
                continue
            pending.append(space)

        free = [i for i in range(len(polygons)) if i not in claimed]
        pending.sort(key=lambda s: (s.params.number, s.params.name, s.id))
        if len(pending) == 1 and len(free) == 1:
            _assign(claimed, chosen, pending[0].id, free[0], "single_room", [])
        else:
            for position, space in enumerate(pending):
                if position < len(free):
                    _assign(
                        claimed,
                        chosen,
                        space.id,
                        free[position],
                        "order",
                        [
                            "ambiguous room assignment: no seed and no bounding walls; matched "
                            f"by order ({len(pending)} spaces, {len(free)} free polygons)"
                        ],
                    )
                else:
                    failures.setdefault(space.id, []).append(
                        "no free room polygon for this space (no seed, no bounding walls)"
                    )

        for space_id, (index, method, notes) in chosen.items():
            room = rooms[index]
            issues = list(notes)
            floor_z, ceiling_z, source = _levels(room, slabs, ceilings, walls, storey, origin)
            if source == "none":
                issues.append("space height unknown: no ceiling, storey height or wall")
            elif source != "ceiling":
                issues.append(f"space height from {source}: no ceiling above the floor found")
            geometries[space_id] = SpaceGeometry(
                element_id=space_id,
                storey_id=storey_id,
                exterior=room.exterior,
                holes=room.holes,
                floor_z=floor_z,
                ceiling_z=ceiling_z,
                height=ceiling_z - floor_z,
                height_source=source,
                gross_area=room.gross_area,
                net_area=room.net_area,
                perimeter=room.perimeter,
                bounding_wall_ids=room.bounding_wall_ids,
                floor_ids=room.floor_ids,
                ceiling_ids=room.ceiling_ids,
                matched_by=method,
                issues=tuple(issues),
            )
    ordered = {space.id: geometries[space.id] for space in spaces if space.id in geometries}
    return ordered, failures
