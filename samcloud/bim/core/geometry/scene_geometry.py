"""Scene-level geometry: everything in dependency order with quantities and issues."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from shapely.geometry import Polygon

from samcloud.bim.core.geometry._math import Vec2
from samcloud.bim.core.geometry.openings import (
    FillingGeometry,
    OpeningGeometry,
    apply_openings,
    compute_filling_geometry,
    compute_opening_geometry,
)
from samcloud.bim.core.geometry.slabs import (
    CeilingGeometry,
    SlabGeometry,
    compute_ceiling_geometry,
    compute_slab_geometry,
)
from samcloud.bim.core.geometry.spaces import SpaceGeometry, compute_space_geometries, derive_storey_rooms
from samcloud.bim.core.geometry.stairs import StairGeometry, compute_stair_geometry
from samcloud.bim.core.geometry.walls import WallGeometry, compute_wall_geometries
from samcloud.bim.core.model import (
    CeilingElement,
    DoorElement,
    OpeningElement,
    Scene,
    SlabElement,
    SpaceElement,
    StairElement,
    WallElement,
    WallJoinElement,
    WindowElement,
)


@dataclass(frozen=True)
class SceneGeometry:
    """All derived geometry of a scene, keyed by element id (insertion order of the scene)."""

    origin: Vec2
    walls: dict[str, WallGeometry] = field(default_factory=dict)
    openings: dict[str, OpeningGeometry] = field(default_factory=dict)
    windows: dict[str, FillingGeometry] = field(default_factory=dict)
    doors: dict[str, FillingGeometry] = field(default_factory=dict)
    slabs: dict[str, SlabGeometry] = field(default_factory=dict)
    ceilings: dict[str, CeilingGeometry] = field(default_factory=dict)
    spaces: dict[str, SpaceGeometry] = field(default_factory=dict)
    stairs: dict[str, StairGeometry] = field(default_factory=dict)
    issues: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Issues per element id (also for elements without geometry and for wall joins)."""

    def geometry_of(
        self, element_id: str
    ) -> (
        WallGeometry
        | OpeningGeometry
        | FillingGeometry
        | SlabGeometry
        | CeilingGeometry
        | SpaceGeometry
        | StairGeometry
        | None
    ):
        """Return the geometry object of an element, or None."""
        for pool in (
            self.walls,
            self.openings,
            self.windows,
            self.doors,
            self.slabs,
            self.ceilings,
            self.spaces,
            self.stairs,
        ):
            if element_id in pool:
                return pool[element_id]
        return None

    @property
    def quantities(self) -> dict[str, dict[str, float]]:
        """Quantities per element id (IFC4 Qto names)."""
        result: dict[str, dict[str, float]] = {}
        for pool in (
            self.walls,
            self.openings,
            self.windows,
            self.doors,
            self.slabs,
            self.ceilings,
            self.spaces,
            self.stairs,
        ):
            for key, geometry in pool.items():
                result[key] = geometry.quantities
        return result

    def quantities_of(self, element_id: str) -> dict[str, float]:
        """Quantities of one element (empty if it has no geometry)."""
        geometry = self.geometry_of(element_id)
        return {} if geometry is None else geometry.quantities


def _origin_of(scene: Scene) -> Vec2:
    xs: list[float] = []
    ys: list[float] = []
    for element in scene.elements.values():
        if isinstance(element, WallElement):
            points = [element.params.start, element.params.end]
        elif isinstance(element, SlabElement | CeilingElement):
            points = list(element.params.outline)
        else:
            continue
        xs.extend(p[0] for p in points)
        ys.extend(p[1] for p in points)
    if not xs:
        return (0.0, 0.0)
    return (min(xs), min(ys))


def compute_scene_geometry(scene: Scene, seeds: Mapping[str, Vec2] | None = None) -> SceneGeometry:
    """Compute all geometry of a scene in dependency order.

    Order: walls with joins, openings (net wall quantities), windows and doors, slabs and
    ceilings, spaces. ``seeds`` optionally maps a space id to a plan point inside its room; an
    entry wins over ``SpaceParams.seed``, which is used for every space without an entry. The
    function is pure and deterministic; math runs in float64 relative to an internal local origin
    (the minimum corner of all outlines).
    """
    origin = _origin_of(scene)
    elevations = {key: storey.elevation for key, storey in scene.storeys.items()}
    issues: dict[str, list[str]] = {}

    def report(element_id: str, messages: list[str] | tuple[str, ...]) -> None:
        if messages:
            issues.setdefault(element_id, []).extend(messages)

    wall_elements = [e for e in scene.elements.values() if isinstance(e, WallElement)]
    join_elements = [e for e in scene.elements.values() if isinstance(e, WallJoinElement)]
    walls, join_issues = compute_wall_geometries(wall_elements, join_elements, elevations, origin)
    for join_id, messages in join_issues.items():
        report(join_id, messages)

    openings: dict[str, OpeningGeometry] = {}
    for element in scene.elements.values():
        if isinstance(element, OpeningElement):
            host = walls.get(element.params.host_wall_id)
            if host is None:
                report(element.id, [f"host wall {element.params.host_wall_id!r} not found"])
                continue
            openings[element.id] = compute_opening_geometry(element, host)
    for wall_id, wall in list(walls.items()):
        on_wall = [o for o in openings.values() if o.host_wall_id == wall_id]
        walls[wall_id] = apply_openings(wall, on_wall)

    windows: dict[str, FillingGeometry] = {}
    doors: dict[str, FillingGeometry] = {}
    for element in scene.elements.values():
        if isinstance(element, WindowElement | DoorElement):
            opening = openings.get(element.params.opening_id)
            if opening is None:
                report(element.id, [f"opening {element.params.opening_id!r} has no geometry"])
                continue
            filling = compute_filling_geometry(element, opening)
            (doors if isinstance(element, DoorElement) else windows)[element.id] = filling

    slabs = {
        e.id: compute_slab_geometry(e, origin)
        for e in scene.elements.values()
        if isinstance(e, SlabElement)
    }
    wall_list = list(walls.values())
    ceiling_elements = [e for e in scene.elements.values() if isinstance(e, CeilingElement)]
    rooms_by_storey: dict[str | None, list[Polygon]] = {}
    for element in ceiling_elements:
        if element.params.trim_to_walls and element.storey_id not in rooms_by_storey:
            rooms_by_storey[element.storey_id] = [
                room.polygon(origin)
                for room in derive_storey_rooms(element.storey_id, wall_list, origin=origin)
            ]
    ceilings = {
        e.id: compute_ceiling_geometry(e, origin, wall_list, rooms_by_storey.get(e.storey_id, ()))
        for e in ceiling_elements
    }

    space_elements = [e for e in scene.elements.values() if isinstance(e, SpaceElement)]
    effective_seeds: dict[str, Vec2] = {
        e.id: (float(e.params.seed[0]), float(e.params.seed[1]))
        for e in space_elements
        if e.params.seed is not None
    }
    effective_seeds.update(seeds or {})
    spaces, space_failures = compute_space_geometries(
        space_elements, walls, slabs, ceilings, scene.storeys, origin, effective_seeds
    )
    for space_id, messages in space_failures.items():
        report(space_id, messages)

    stairs: dict[str, StairGeometry] = {}
    for element in scene.elements.values():
        if isinstance(element, StairElement):
            storey = scene.storeys.get(element.storey_id or "")
            stairs[element.id] = compute_stair_geometry(
                element,
                storey.elevation if storey is not None else 0.0,
                storey.height if storey is not None else None,
            )

    for pool in (walls, openings, windows, doors, slabs, ceilings, spaces, stairs):
        for key, geometry in pool.items():
            report(key, geometry.issues)

    return SceneGeometry(
        origin=origin,
        walls=walls,
        openings=openings,
        windows=windows,
        doors=doors,
        slabs=slabs,
        ceilings=ceilings,
        spaces=spaces,
        stairs=stairs,
        issues={key: tuple(value) for key, value in issues.items()},
    )
