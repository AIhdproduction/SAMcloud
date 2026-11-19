"""Dependency graph and recomputation of derived values."""

from __future__ import annotations

import copy
import json

import pytest

from samcloud.bim.core.model import (
    CeilingElement,
    DoorElement,
    OpeningElement,
    Scene,
    SlabElement,
    SpaceElement,
    WallElement,
    WallJoinElement,
    WallParams,
    WindowElement,
)
from samcloud.bim.core.model.graph import (
    DependencyCycleError,
    DependencyGraph,
    RecomputeResult,
    recompute,
)
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene


def rectangle() -> Scene:
    return build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))


def two_rooms() -> Scene:
    return build_room_scene(RoomSpec(layout=RoomLayout.TWO_ROOMS))


def by_tag(scene: Scene, tag: str, cls: type) -> str:
    matches = [e.id for e in scene.elements.values() if isinstance(e, cls) and e.tag == tag]
    assert len(matches) == 1, (tag, matches)
    return matches[0]


def of_class(scene: Scene, cls: type) -> list[str]:
    return [e.id for e in scene.elements.values() if isinstance(e, cls)]


def rebuild_wall_thickness(scene: Scene, wall_id: str, thickness: float) -> None:
    """Replace a wall through the model API with a new thickness (layers are reset)."""
    wall = scene.elements[wall_id]
    assert isinstance(wall, WallElement)
    data = wall.params.model_dump()
    data["thickness"] = thickness
    data["layers"] = []
    scene.elements[wall_id] = WallElement(
        **{**wall.model_dump(exclude={"params"}), "params": WallParams(**data)}
    )


def without_derived(scene: Scene) -> str:
    data = scene.model_dump(mode="json")
    data.pop("derived")
    return json.dumps(data, sort_keys=True)


# ----------------------------------------------------------------------------- edges


def test_rectangle_edges() -> None:
    scene = rectangle()
    graph = DependencyGraph.from_scene(scene)
    storey = next(iter(scene.storeys))
    assert len(graph.nodes) == len(scene.elements) + 1
    assert graph.kind_of(storey) == "storey"
    for element in scene.elements.values():
        assert graph.has_edge(storey, element.id)
    space = of_class(scene, SpaceElement)[0]
    walls = of_class(scene, WallElement)
    assert len(walls) == 4
    for wall_id in walls:
        assert graph.has_edge(wall_id, space)
    assert graph.has_edge(of_class(scene, SlabElement)[0], space)
    assert graph.has_edge(of_class(scene, CeilingElement)[0], space)
    # openings and fillings
    window = by_tag(scene, "window_south", WindowElement)
    opening = by_tag(scene, "window_south", OpeningElement)
    south = by_tag(scene, "exterior_south", WallElement)
    assert graph.has_edge(south, opening)
    assert graph.has_edge(opening, window)
    door = by_tag(scene, "door_west", DoorElement)
    assert graph.has_edge(by_tag(scene, "door_west", OpeningElement), door)
    assert graph.has_edge(
        by_tag(scene, "exterior_west", WallElement), by_tag(scene, "door_west", OpeningElement)
    )
    # joins depend on both walls and couple them
    joins = of_class(scene, WallJoinElement)
    assert len(joins) == 4
    for join_id in joins:
        join = scene.elements[join_id]
        assert isinstance(join, WallJoinElement)
        assert graph.has_edge(join.params.wall_a, join_id)
        assert graph.has_edge(join.params.wall_b, join_id)
        pair = tuple(sorted((join.params.wall_a, join.params.wall_b)))
        assert pair in graph.couplings
        assert (
            min(join.params.wall_a, join_id),
            max(join.params.wall_a, join_id),
        ) in graph.couplings
    # the space has no dependents; nothing depends on a window
    assert graph.dependents_of(space) == []
    assert graph.dependents_of(window) == []
    assert not graph.has_edge(space, storey)


def test_two_rooms_edges() -> None:
    scene = two_rooms()
    graph = DependencyGraph.from_scene(scene)
    partition = by_tag(scene, "partition", WallElement)
    spaces = of_class(scene, SpaceElement)
    assert len(spaces) == 2
    for space in spaces:
        assert graph.has_edge(partition, space)
    door_opening = by_tag(scene, "door_partition", OpeningElement)
    assert graph.has_edge(partition, door_opening)
    assert graph.has_edge(door_opening, by_tag(scene, "door_partition", DoorElement))
    east = by_tag(scene, "exterior_east", WallElement)
    west = by_tag(scene, "exterior_west", WallElement)
    assert graph.has_edge(east, spaces[1])
    assert not graph.has_edge(east, spaces[0])
    assert graph.has_edge(west, spaces[0])
    assert not graph.has_edge(west, spaces[1])
    t_join = by_tag(scene, "T_partition_south", WallJoinElement)
    south = by_tag(scene, "exterior_south", WallElement)
    assert graph.has_edge(partition, t_join)
    assert graph.has_edge(south, t_join)
    assert tuple(sorted((partition, south))) in graph.couplings
    assert graph.edges == sorted(graph.edges)


def test_space_edges_fall_back_to_geometry_when_space_ids_are_missing() -> None:
    scene = rectangle()
    for element in scene.elements.values():
        element.space_ids = []
    graph = DependencyGraph.from_scene(scene)
    space = of_class(scene, SpaceElement)[0]
    for wall_id in of_class(scene, WallElement):
        assert graph.has_edge(wall_id, space)
    assert graph.has_edge(of_class(scene, SlabElement)[0], space)
    assert graph.has_edge(of_class(scene, CeilingElement)[0], space)


# ---------------------------------------------------------------- change propagation


def test_wall_thickness_change_propagates_to_joined_walls_openings_and_space() -> None:
    scene = rectangle()
    first = recompute(scene)
    assert set(first.changed) == {
        e.id for e in scene.elements.values() if not isinstance(e, WallJoinElement)
    }
    before = copy.deepcopy(scene.derived)
    space = of_class(scene, SpaceElement)[0]
    assert before["spaces"][space]["area"] == pytest.approx((5.0 - 0.3) * (4.0 - 0.3), abs=1e-9)
    assert before["spaces"][space]["area"] == pytest.approx(17.39, abs=1e-9)

    west = by_tag(scene, "exterior_west", WallElement)
    rebuild_wall_thickness(scene, west, 0.5)
    result = recompute(scene, [west])

    south = by_tag(scene, "exterior_south", WallElement)
    north = by_tag(scene, "exterior_north", WallElement)
    east = by_tag(scene, "exterior_east", WallElement)
    door_opening = by_tag(scene, "door_west", OpeningElement)
    door = by_tag(scene, "door_west", DoorElement)
    # the ceiling follows the inner wall faces (trim_to_walls): its outline changes as well
    ceiling = of_class(scene, CeilingElement)[0]
    expected = {west, south, north, door_opening, door, space, ceiling}
    assert set(result.changed) == expected
    assert east not in result.changed
    assert set(result.changed) <= set(result.affected)

    after = scene.derived
    # the wall itself
    old_wall = before["elements"][west]["quantities"]
    new_wall = after["elements"][west]["quantities"]
    assert old_wall["Width"] == pytest.approx(0.3)
    assert new_wall["Width"] == pytest.approx(0.5)
    assert new_wall["GrossFootprintArea"] == pytest.approx(0.5 * 4.0)
    assert new_wall["GrossVolume"] > old_wall["GrossVolume"]
    # the joined walls: their trimmed footprint changes
    for joined in (south, north):
        assert (
            after["elements"][joined]["geometry"]["footprint"]
            != (before["elements"][joined]["geometry"]["footprint"])
        )
    # opening and door in the wall: depth follows the wall thickness
    assert before["elements"][door_opening]["geometry"]["depth"] == pytest.approx(0.3)
    assert after["elements"][door_opening]["geometry"]["depth"] == pytest.approx(0.5)
    assert (
        after["elements"][door]["geometry"]["corners"]
        != (before["elements"][door]["geometry"]["corners"])
    )
    # space: the west wall's inner face moves from x = 0.15 to x = 0.25;
    # net area = (5 - 0.25 - 0.15) * (4 - 0.3) = 4.6 * 3.7 = 17.02
    assert after["spaces"][space]["area"] == pytest.approx(4.6 * 3.7, abs=1e-9)
    assert after["spaces"][space]["area"] == pytest.approx(17.02, abs=1e-9)
    assert after["elements"][space]["quantities"]["NetFloorArea"] == pytest.approx(17.02, abs=1e-9)
    assert after["spaces"][space]["volume"] == pytest.approx(17.02 * 2.8, abs=1e-9)
    # untouched elements keep identical values
    assert after["elements"][east] == before["elements"][east]
    window = by_tag(scene, "window_south", WindowElement)
    assert after["elements"][window] == before["elements"][window]
    slab = of_class(scene, SlabElement)[0]
    assert after["elements"][slab] == before["elements"][slab]


def test_recompute_twice_reports_no_change() -> None:
    scene = two_rooms()
    recompute(scene)
    snapshot = copy.deepcopy(scene.derived)
    again = recompute(scene)
    assert again.changed == ()
    assert scene.derived == snapshot


def test_changed_uses_float_tolerance() -> None:
    scene = rectangle()
    recompute(scene)
    wall = by_tag(scene, "exterior_south", WallElement)
    scene.derived["elements"][wall]["quantities"]["Length"] += 1e-13  # below 1e-12: unchanged
    assert recompute(scene).changed == ()
    scene.derived["elements"][wall]["quantities"]["Length"] += 1e-9  # above: reported
    assert recompute(scene).changed == (wall,)


def test_storey_elevation_change_moves_elements() -> None:
    scene = rectangle()
    recompute(scene)
    storey_id = next(iter(scene.storeys))
    storey = scene.storeys[storey_id]
    scene.storeys[storey_id] = storey.model_copy(update={"elevation": 3.0})
    result = recompute(scene, [storey_id])
    assert storey_id in result.affected
    assert set(result.affected) == {storey_id, *scene.elements}
    wall = by_tag(scene, "exterior_south", WallElement)
    assert result.changed and wall in result.changed
    assert scene.derived["elements"][wall]["geometry"]["base_z"] == pytest.approx(3.0)


def test_removed_element_is_reported_and_dropped() -> None:
    scene = rectangle()
    recompute(scene)
    window = by_tag(scene, "window_north", WindowElement)
    del scene.elements[window]
    result = recompute(scene)
    assert result.changed == (window,)
    assert window not in scene.derived["elements"]


def test_foreign_derived_keys_are_preserved() -> None:
    scene = rectangle()
    wall = by_tag(scene, "exterior_south", WallElement)
    recompute(scene)
    scene.derived["elements"][wall]["metrics"] = {"rms": 0.003}
    result = recompute(scene)
    assert result.changed == ()
    assert scene.derived["elements"][wall]["metrics"] == {"rms": 0.003}


# --------------------------------------------------------------------- affected_by


def test_affected_by_order_respects_dependencies() -> None:
    scene = two_rooms()
    graph = DependencyGraph.from_scene(scene)
    south = by_tag(scene, "exterior_south", WallElement)
    storey = next(iter(scene.storeys))
    for changed in ([south], [storey], [by_tag(scene, "door_partition", OpeningElement)]):
        order = graph.affected_by(changed)
        position = {node: index for index, node in enumerate(order)}
        assert len(position) == len(order)
        assert set(changed) <= set(order)
        for dependency, dependent in graph.edges:
            if dependency in position and dependent in position:
                assert position[dependency] < position[dependent]
    full = graph.topological_order()
    assert sorted(full) == graph.nodes
    assert full[0] == storey


def test_affected_by_scope() -> None:
    scene = rectangle()
    graph = DependencyGraph.from_scene(scene)
    window = by_tag(scene, "window_south", WindowElement)
    assert graph.affected_by([window]) == [window]
    opening = by_tag(scene, "window_south", OpeningElement)
    assert graph.affected_by([opening]) == [opening, window]
    slab = of_class(scene, SlabElement)[0]
    space = of_class(scene, SpaceElement)[0]
    assert graph.affected_by([slab]) == [slab, space]
    east = by_tag(scene, "exterior_east", WallElement)
    affected = graph.affected_by([east])
    south = by_tag(scene, "exterior_south", WallElement)
    north = by_tag(scene, "exterior_north", WallElement)
    west = by_tag(scene, "exterior_west", WallElement)
    assert {east, south, north, space} <= set(affected)
    # couplings are followed one hop only: the opposite wall is not affected
    assert west not in affected
    assert by_tag(scene, "door_west", DoorElement) not in affected
    assert affected[-1] == space


def test_affected_by_is_deterministic_and_independent_of_insertion_order() -> None:
    scene = two_rooms()
    reversed_scene = scene.model_copy(deep=True)
    reversed_scene.elements = dict(reversed(list(scene.elements.items())))
    south = by_tag(scene, "exterior_south", WallElement)
    first = DependencyGraph.from_scene(scene).affected_by([south])
    second = DependencyGraph.from_scene(scene).affected_by([south])
    third = DependencyGraph.from_scene(reversed_scene).affected_by([south])
    assert first == second == third


def test_unknown_id_raises() -> None:
    graph = DependencyGraph.from_scene(rectangle())
    with pytest.raises(KeyError, match="unknown id"):
        graph.affected_by(["nope"])


# ------------------------------------------------------------------------- cycles


def test_cycle_in_manual_graph_is_detected() -> None:
    graph = DependencyGraph()
    for node in ("a", "b", "c", "d"):
        graph.add_node(node)
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    assert graph.affected_by(["a"]) == ["a", "b", "c"]
    graph.add_edge("c", "b")
    with pytest.raises(DependencyCycleError, match="b, c") as info:
        graph.affected_by(["d"])
    assert info.value.cycle == ("b", "c")
    with pytest.raises(DependencyCycleError):
        graph.topological_order()


def test_cycle_in_scene_references_is_detected() -> None:
    scene = rectangle()
    window = by_tag(scene, "window_south", WindowElement)
    opening = by_tag(scene, "window_south", OpeningElement)
    # an opening whose "host wall" is the window that fills it
    scene.elements[opening].params.host_wall_id = window  # type: ignore[attr-defined]
    graph = DependencyGraph.from_scene(scene)
    assert graph.has_edge(window, opening)
    with pytest.raises(DependencyCycleError):
        graph.affected_by([window])


# ----------------------------------------------------------------------- recompute


def test_recompute_does_not_touch_the_model() -> None:
    scene = two_rooms()
    before = without_derived(scene)
    result = recompute(scene)
    assert isinstance(result, RecomputeResult)
    assert without_derived(scene) == before
    assert scene.derived["elements"] and scene.derived["spaces"]


def test_derived_structure_and_json() -> None:
    scene = two_rooms()
    recompute(scene)
    derived = json.loads(json.dumps(scene.derived))
    assert derived == scene.derived
    assert set(derived) == {"elements", "spaces"}
    assert list(derived["elements"]) == sorted(derived["elements"])
    wall = by_tag(scene, "partition", WallElement)
    entry = derived["elements"][wall]
    assert set(entry) == {"quantities", "geometry"}
    assert entry["quantities"]["Width"] == pytest.approx(0.12)
    assert entry["quantities"]["Length"] == pytest.approx(4.7)
    assert entry["geometry"]["thickness"] == pytest.approx(0.12)
    join = by_tag(scene, "T_partition_south", WallJoinElement)
    assert join not in derived["elements"]
    for space_id in of_class(scene, SpaceElement):
        space = derived["spaces"][space_id]
        assert space["area"] == pytest.approx(17.813, abs=1e-9)
        assert space["height"] == pytest.approx(2.8)
        assert space["volume"] == pytest.approx(space["area"] * space["height"])
        assert space["matched_by"] == "seed"
        assert wall in space["bounding_wall_ids"]
        assert "issues" not in derived["elements"][space_id]
        assert derived["elements"][space_id]["quantities"]["NetFloorArea"] == space["area"]


def test_recompute_is_deterministic() -> None:
    first = two_rooms()
    second = two_rooms()
    recompute(first)
    recompute(second, [by_tag(second, "partition", WallElement)])
    assert json.dumps(first.derived, sort_keys=False) == json.dumps(second.derived, sort_keys=False)
    assert first.to_json() == second.to_json()


def test_recompute_is_independent_of_element_insertion_order() -> None:
    """Derived values are a pure function of the scene, not of the order of its dicts."""
    reference = two_rooms()
    recompute(reference)
    shuffled = two_rooms()
    shuffled.elements = dict(reversed(list(shuffled.elements.items())))
    shuffled.storeys = dict(reversed(list(shuffled.storeys.items())))
    recompute(shuffled)
    assert json.dumps(shuffled.derived) == json.dumps(reference.derived)


def test_recompute_reports_issues() -> None:
    scene = rectangle()
    opening_id = by_tag(scene, "window_south", OpeningElement)
    opening = scene.elements[opening_id]
    assert isinstance(opening, OpeningElement)
    opening.params.position_along_axis = 0.1  # before the wall start: geometry issue
    result = recompute(scene)
    assert opening_id in result.changed
    assert scene.derived["elements"][opening_id]["issues"]
