"""Dependency graph and deterministic recomputation of derived values (MASTERPROMPT 5.1).

The graph has one node per element id and per storey id. An edge ``(dependency, dependent)``
means that the dependent's derived values must be recomputed when the dependency changes:

* storey -> every element with that ``storey_id`` (an elevation change moves them);
* wall -> opening placed in it (``host_wall_id``);
* opening -> window or door filling it (``opening_id``);
* wall -> wall join that lists it (a join depends on both walls);
* wall -> trimmed ceiling of the same storey (``CeilingGeometry.wall_ids``): the ceiling outline
  follows the inner wall faces;
* wall, floor slab, ceiling -> space they bound (``space_ids`` of the element, united with the
  bounding ids of the space geometry, which also covers elements without ``space_ids``).

"Quantities depend on the geometry" needs no node of its own: the quantities of an element are
derived together with its geometry, so they are recomputed whenever the element is.

Wall joins couple walls mutually: the trimmed geometry of a wall depends on the parameters of the
walls it is joined to, and a changed join moves both walls. Mutual edges would be cycles, and a
transitive closure over them would wrongly mark a whole room as affected when one wall changes
(wall C of a ring A-B-C-D does not depend on wall A). They are therefore kept apart as *couplings*
that are followed exactly one hop from a changed id: a changed wall affects the walls it is joined
to and the joins it takes part in, and a changed join affects its two walls. From there the
directed edges are followed transitively. The directed edges are acyclic for a valid scene; a cycle
(for example an opening whose host "wall" is a window that fills the opening) raises
:class:`DependencyCycleError`.

Derived values live in ``Scene.derived`` and are written only by :func:`recompute`; they are never
edited directly and never patched. See docs/DATA_MODEL.md, section "Derived values".
"""

from __future__ import annotations

import dataclasses
import heapq
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from samcloud.bim.core.model.elements import (
    CeilingElement,
    DoorElement,
    OpeningElement,
    SlabElement,
    WallElement,
    WallJoinElement,
    WindowElement,
)
from samcloud.bim.core.model.scene import Scene

if TYPE_CHECKING:
    from samcloud.bim.core.geometry import SceneGeometry

STOREY_KIND = "storey"
FLOAT_TOLERANCE = 1e-12
"""Derived floats closer than this (absolute and relative) count as unchanged."""

ELEMENT_KEYS = ("quantities", "geometry", "issues")
"""Keys of ``derived["elements"][id]`` owned by :func:`recompute` (others are preserved)."""
SPACE_KEYS = (
    "exterior",
    "holes",
    "area",
    "gross_area",
    "perimeter",
    "height",
    "volume",
    "gross_volume",
    "floor_z",
    "ceiling_z",
    "height_source",
    "matched_by",
    "bounding_wall_ids",
    "floor_ids",
    "ceiling_ids",
)
"""Keys of ``derived["spaces"][id]`` owned by :func:`recompute` (others are preserved)."""

_GEOMETRY_SKIP = frozenset({"element_id", "storey_id", "space_ids", "issues", "local_footprint"})


class DependencyCycleError(ValueError):
    """The dependency graph contains a cycle; ``cycle`` lists the ids involved."""

    def __init__(self, cycle: Iterable[str]) -> None:
        self.cycle = tuple(sorted(cycle))
        super().__init__("dependency cycle between elements: " + ", ".join(self.cycle))


class DependencyGraph:
    """Directed dependency graph over element and storey ids (see module docstring)."""

    def __init__(self) -> None:
        self._kinds: dict[str, str] = {}
        self._dependents: dict[str, set[str]] = {}
        self._dependencies: dict[str, set[str]] = {}
        self._coupled: dict[str, set[str]] = {}

    # ----------------------------------------------------------------- building

    def add_node(self, node_id: str, kind: str = "node") -> None:
        """Add a node (idempotent)."""
        if node_id not in self._kinds:
            self._kinds[node_id] = kind
            self._dependents[node_id] = set()
            self._dependencies[node_id] = set()
            self._coupled[node_id] = set()

    def add_edge(self, dependency: str, dependent: str) -> None:
        """Add the directed edge ``dependency -> dependent`` (both nodes must exist)."""
        for node_id in (dependency, dependent):
            if node_id not in self._kinds:
                raise KeyError(f"unknown node {node_id!r}")
        self._dependents[dependency].add(dependent)
        self._dependencies[dependent].add(dependency)

    def add_coupling(self, first: str, second: str) -> None:
        """Couple two nodes mutually; a change of one affects the other (one hop only)."""
        for node_id in (first, second):
            if node_id not in self._kinds:
                raise KeyError(f"unknown node {node_id!r}")
        self._coupled[first].add(second)
        self._coupled[second].add(first)

    @classmethod
    def from_scene(cls, scene: Scene, geometry: SceneGeometry | None = None) -> DependencyGraph:
        """Build the graph of a scene.

        ``geometry`` (the result of ``compute_scene_geometry``) supplies the bounding ids of the
        spaces; it is computed here when not given.
        """
        if geometry is None:
            from samcloud.bim.core.geometry import compute_scene_geometry

            geometry = compute_scene_geometry(scene)
        graph = cls()
        for storey_id in scene.storeys:
            graph.add_node(storey_id, STOREY_KIND)
        for element in scene.elements.values():
            graph.add_node(element.id, element.kind.value)

        def link(dependency: str, dependent: str) -> None:
            if dependency in graph._kinds:
                graph.add_edge(dependency, dependent)

        def couple(first: str, second: str) -> None:
            if first in graph._kinds and second in graph._kinds:
                graph.add_coupling(first, second)

        for element in scene.elements.values():
            if element.storey_id is not None:
                link(element.storey_id, element.id)
            if isinstance(element, WallJoinElement):
                for wall_id in (element.params.wall_a, element.params.wall_b):
                    link(wall_id, element.id)
                    couple(wall_id, element.id)
                couple(element.params.wall_a, element.params.wall_b)
            elif isinstance(element, OpeningElement):
                link(element.params.host_wall_id, element.id)
            elif isinstance(element, WindowElement | DoorElement):
                link(element.params.opening_id, element.id)
        # wall -> ceiling: a trimmed ceiling ends at the inner faces of the walls of its storey
        for ceiling_id, ceiling_geometry in geometry.ceilings.items():
            for wall_id in ceiling_geometry.wall_ids:
                link(wall_id, ceiling_id)
        # space edges: element -> space, from space_ids and from the geometry's bounding ids
        for element in scene.elements.values():
            if isinstance(element, WallElement | SlabElement | CeilingElement):
                for space_id in element.space_ids:
                    if space_id in graph._kinds and space_id != element.id:
                        graph.add_edge(element.id, space_id)
        for space_id, space in geometry.spaces.items():
            for bound in (*space.bounding_wall_ids, *space.floor_ids, *space.ceiling_ids):
                if bound in graph._kinds and bound != space_id:
                    graph.add_edge(bound, space_id)
        return graph

    # ------------------------------------------------------------------ queries

    @property
    def nodes(self) -> list[str]:
        """All node ids, sorted."""
        return sorted(self._kinds)

    def kind_of(self, node_id: str) -> str:
        """Kind of a node: an element kind value or ``"storey"``."""
        return self._kinds[node_id]

    @property
    def edges(self) -> list[tuple[str, str]]:
        """All directed edges ``(dependency, dependent)``, sorted."""
        return sorted(
            (dependency, dependent)
            for dependency, targets in self._dependents.items()
            for dependent in targets
        )

    @property
    def couplings(self) -> list[tuple[str, str]]:
        """All couplings as sorted pairs ``(a, b)`` with ``a < b``."""
        return sorted(
            (first, second)
            for first, others in self._coupled.items()
            for second in others
            if first < second
        )

    def has_edge(self, dependency: str, dependent: str) -> bool:
        """True if the directed edge exists."""
        return dependent in self._dependents.get(dependency, ())

    def dependencies_of(self, node_id: str) -> list[str]:
        """Direct dependencies of a node, sorted."""
        return sorted(self._dependencies[node_id])

    def dependents_of(self, node_id: str) -> list[str]:
        """Direct dependents of a node, sorted."""
        return sorted(self._dependents[node_id])

    def _ordered(self, subset: set[str]) -> list[str]:
        """Topological order of ``subset`` (dependencies first, ties by id)."""
        indegree = {
            node: sum(1 for dep in self._dependencies[node] if dep in subset) for node in subset
        }
        heap = [node for node, degree in indegree.items() if degree == 0]
        heapq.heapify(heap)
        order: list[str] = []
        while heap:
            node = heapq.heappop(heap)
            order.append(node)
            for dependent in self._dependents[node]:
                if dependent in subset:
                    indegree[dependent] -= 1
                    if indegree[dependent] == 0:
                        heapq.heappush(heap, dependent)
        if len(order) != len(subset):
            raise DependencyCycleError(node for node, degree in indegree.items() if degree > 0)
        return order

    def topological_order(self) -> list[str]:
        """All nodes, dependencies before dependents, ties broken by id."""
        return self._ordered(set(self._kinds))

    def affected_by(self, changed_ids: Iterable[str]) -> list[str]:
        """Return the changed ids and everything that depends on them, in dependency order.

        Couplings (wall joins) are followed one hop from a changed id, directed edges
        transitively. Raises ``KeyError`` for an unknown id and :class:`DependencyCycleError` if
        the directed edges contain a cycle.
        """
        seeds = set(changed_ids)
        for node_id in seeds:
            if node_id not in self._kinds:
                raise KeyError(f"unknown id {node_id!r}")
        # Detect cycles of the whole graph first so that they are reported even when the cycle is
        # not reachable from the changed ids.
        self.topological_order()
        reached = set(seeds)
        for node_id in seeds:
            reached.update(self._coupled[node_id])
        stack = sorted(reached)
        while stack:
            node = stack.pop()
            for dependent in self._dependents[node]:
                if dependent not in reached:
                    reached.add(dependent)
                    stack.append(dependent)
        return self._ordered(reached)


# ------------------------------------------------------------------ recomputation


@dataclass(frozen=True)
class RecomputeResult:
    """Outcome of :func:`recompute`."""

    affected: tuple[str, ...]
    """Ids that depend on ``changed_ids`` (all nodes when none were given), dependency order."""
    changed: tuple[str, ...]
    """Element ids whose derived values differ from before (dependency order)."""
    geometry: SceneGeometry
    graph: DependencyGraph


def _plain(value: Any) -> Any:
    """Convert geometry values into JSON-serializable plain data."""
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.name not in _GEOMETRY_SKIP
        }
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, bool) or value is None or isinstance(value, str | int):
        return value
    if isinstance(value, float):
        return float(value)
    raise TypeError(f"cannot store {type(value).__name__} in scene.derived")


def _geometry_snapshot(geometry: Any) -> dict[str, Any]:
    data = _plain(geometry)
    origin = data.get("origin")
    if isinstance(origin, list) and len(origin) == 2:
        del data["origin"]  # the internal local frame origin, not a property of the element
    return dict(data)


def _space_values(space: Any) -> dict[str, Any]:
    return {
        "exterior": _plain(space.exterior),
        "holes": _plain(space.holes),
        "area": float(space.net_area),
        "gross_area": float(space.gross_area),
        "perimeter": float(space.perimeter),
        "height": float(space.height),
        "volume": float(space.net_volume),
        "gross_volume": float(space.gross_volume),
        "floor_z": float(space.floor_z),
        "ceiling_z": float(space.ceiling_z),
        "height_source": space.height_source,
        "matched_by": space.matched_by,
        "bounding_wall_ids": list(space.bounding_wall_ids),
        "floor_ids": list(space.floor_ids),
        "ceiling_ids": list(space.ceiling_ids),
    }


def _same(old: Any, new: Any) -> bool:
    """Structural equality with a float tolerance of 1e-12."""
    if isinstance(old, bool) or isinstance(new, bool):
        return bool(old == new) and type(old) is type(new)
    if isinstance(old, int | float) and isinstance(new, int | float):
        return math.isclose(old, new, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        return old.keys() == new.keys() and all(_same(old[key], new[key]) for key in old)
    if isinstance(old, list | tuple) and isinstance(new, list | tuple):
        return len(old) == len(new) and all(_same(a, b) for a, b in zip(old, new, strict=True))
    return bool(old == new)


def _owned(entry: Mapping[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
    return {} if entry is None else {key: entry[key] for key in keys if key in entry}


def _merge(
    pool: dict[str, Any], key: str, owned_keys: tuple[str, ...], new: dict[str, Any]
) -> None:
    """Write ``new`` into ``pool[key]`` keeping foreign keys; drop the entry when empty."""
    kept = {k: v for k, v in pool.get(key, {}).items() if k not in owned_keys}
    merged = {**kept, **new}
    if merged:
        pool[key] = {k: merged[k] for k in sorted(merged)}
    else:
        pool.pop(key, None)


def canonical_view(scene: Scene) -> Scene:
    """Shallow copy of ``scene`` whose storeys, types and elements are ordered by id.

    The geometry code walks these dicts, so the order of lists in the derived values (bounding
    walls of a space, opening cuts of a wall) and the summation order of floats would otherwise
    depend on the history of the scene (an undo re-inserts elements at the end). Computing on
    the canonical view makes ``scene.derived`` a pure function of the current scene.
    """
    return scene.model_copy(
        update={
            "storeys": {key: scene.storeys[key] for key in sorted(scene.storeys)},
            "types": {key: scene.types[key] for key in sorted(scene.types)},
            "elements": {key: scene.elements[key] for key in sorted(scene.elements)},
        }
    )


def recompute(scene: Scene, changed_ids: Iterable[str] | None = None) -> RecomputeResult:
    """Recompute all derived values of a scene and write them into ``scene.derived``.

    Geometry is cheap, so everything is recomputed; ``changed_ids`` only determines the
    ``affected`` list of the result (all nodes when ``None``). ``changed`` lists exactly the
    element ids whose derived values differ from the previous ones (float tolerance 1e-12).
    Only ``scene.derived`` is written; element params, properties and origins are never touched.
    The result is a pure function of the current scene, independent of its history and of the
    insertion order of storeys, types and elements (everything is recomputed from scratch on
    :func:`canonical_view`). The function is not a patch. Layout: docs/DATA_MODEL.md.
    """
    from samcloud.bim.core.geometry import compute_scene_geometry

    canonical = canonical_view(scene)
    geometry = compute_scene_geometry(canonical)
    graph = DependencyGraph.from_scene(canonical, geometry)
    affected = graph.topological_order() if changed_ids is None else graph.affected_by(changed_ids)

    derived_elements: dict[str, Any] = scene.derived.setdefault("elements", {})
    derived_spaces: dict[str, Any] = scene.derived.setdefault("spaces", {})
    quantities = geometry.quantities
    changed: set[str] = set()

    for element_id in sorted(scene.elements):
        new: dict[str, Any] = {}
        geo = geometry.geometry_of(element_id)
        if geo is not None:
            new["quantities"] = {
                name: float(v) for name, v in sorted(quantities[element_id].items())
            }
            if element_id not in geometry.spaces:
                new["geometry"] = _geometry_snapshot(geo)
        issues = list(geometry.issues.get(element_id, ()))
        if issues:
            new["issues"] = issues
        if not _same(_owned(derived_elements.get(element_id), ELEMENT_KEYS), new):
            changed.add(element_id)
        _merge(derived_elements, element_id, ELEMENT_KEYS, new)

        new_space = (
            _space_values(geometry.spaces[element_id]) if element_id in geometry.spaces else {}
        )
        if not _same(_owned(derived_spaces.get(element_id), SPACE_KEYS), new_space):
            changed.add(element_id)
        _merge(derived_spaces, element_id, SPACE_KEYS, new_space)

    removed = sorted((set(derived_elements) | set(derived_spaces)) - set(scene.elements))
    for element_id in removed:
        for pool, keys in ((derived_elements, ELEMENT_KEYS), (derived_spaces, SPACE_KEYS)):
            if _owned(pool.get(element_id), keys):
                changed.add(element_id)
            if element_id in pool:
                _merge(pool, element_id, keys, {})

    for pool in (derived_elements, derived_spaces):
        ordered = {key: pool[key] for key in sorted(pool)}
        pool.clear()
        pool.update(ordered)

    position = {node: index for index, node in enumerate(graph.topological_order())}
    changed_sorted = sorted(changed, key=lambda node: (position.get(node, len(position)), node))
    return RecomputeResult(
        affected=tuple(affected),
        changed=tuple(changed_sorted),
        geometry=geometry,
        graph=graph,
    )


__all__ = [
    "ELEMENT_KEYS",
    "FLOAT_TOLERANCE",
    "SPACE_KEYS",
    "DependencyCycleError",
    "DependencyGraph",
    "RecomputeResult",
    "recompute",
]
