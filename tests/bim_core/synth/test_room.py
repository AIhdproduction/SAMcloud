from __future__ import annotations

import math
from collections import Counter

import pytest

from samcloud.bim.core.geometry import compute_scene_geometry
from samcloud.bim.core.model import (
    CeilingElement,
    DoorElement,
    ElementKind,
    JoinEnd,
    JoinType,
    OpeningElement,
    ParamOrigin,
    Scene,
    SlabElement,
    SpaceElement,
    WallElement,
    WallFunction,
    WallJoinElement,
    WindowElement,
)
from samcloud.bim.core.synth.room import (
    RoomLayout,
    RoomSpec,
    build_room_scene,
)

LAYOUTS = [RoomLayout.RECTANGLE, RoomLayout.TWO_ROOMS]
EXPECTED = {
    RoomLayout.RECTANGLE: {
        ElementKind.WALL: 4,
        ElementKind.WALL_JOIN: 4,
        ElementKind.SLAB: 1,
        ElementKind.CEILING: 1,
        ElementKind.OPENING: 3,
        ElementKind.WINDOW: 2,
        ElementKind.DOOR: 1,
        ElementKind.SPACE: 1,
    },
    RoomLayout.TWO_ROOMS: {
        ElementKind.WALL: 5,
        ElementKind.WALL_JOIN: 6,
        ElementKind.SLAB: 1,
        ElementKind.CEILING: 2,
        ElementKind.OPENING: 4,
        ElementKind.WINDOW: 2,
        ElementKind.DOOR: 2,
        ElementKind.SPACE: 2,
    },
}


def _scene(layout: RoomLayout) -> Scene:
    return build_room_scene(RoomSpec(layout=layout))


def _walls(scene: Scene) -> list[WallElement]:
    return [e for e in scene.elements.values() if isinstance(e, WallElement)]


def _joins(scene: Scene) -> list[WallJoinElement]:
    return [e for e in scene.elements.values() if isinstance(e, WallJoinElement)]


def _length(wall: WallElement) -> float:
    return math.dist(wall.params.start, wall.params.end)


def _end(wall: WallElement, end: JoinEnd) -> tuple[float, float]:
    return wall.params.start if end == JoinEnd.START else wall.params.end


@pytest.mark.parametrize("layout", LAYOUTS)
def test_references_consistent(layout: RoomLayout) -> None:
    assert _scene(layout).check_references() == []


def test_default_spec_is_rectangle() -> None:
    assert build_room_scene().to_json() == _scene(RoomLayout.RECTANGLE).to_json()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_counts(layout: RoomLayout) -> None:
    scene = _scene(layout)
    counts = Counter(element.kind for element in scene.elements.values())
    assert dict(counts) == EXPECTED[layout]
    joins = Counter(j.params.join_type for j in _joins(scene))
    assert joins[JoinType.L] == 4
    assert joins[JoinType.T] == (0 if layout == RoomLayout.RECTANGLE else 2)
    assert len(scene.types) == 5


@pytest.mark.parametrize("layout", LAYOUTS)
def test_structure_and_defaults(layout: RoomLayout) -> None:
    scene = _scene(layout)
    assert scene.settings.origin_offset == (2_600_000.0, 1_200_000.0, 400.0)
    assert scene.settings.crs is None
    (storey,) = scene.storeys.values()
    assert storey.elevation == 0.0
    assert storey.height == 2.8
    typed = (ElementKind.WALL, ElementKind.SLAB, ElementKind.WINDOW, ElementKind.DOOR)
    for element in scene.elements.values():
        assert element.storey_id == storey.id
        assert element.status.value == "unsure"
        assert element.provenance.value == "script"
        assert element.name
        assert element.tag
        assert set(element.param_origin) == set(type(element.params).model_fields)
        assert set(element.param_origin.values()) == {ParamOrigin.TEMPLATE}
        if element.kind in typed:
            assert element.type_id in scene.types
    for wall in _walls(scene):
        assert wall.params.height == 2.8
        if wall.params.function == WallFunction.EXTERIOR:
            assert wall.params.thickness == 0.30
        else:
            assert wall.params.thickness == 0.12
    (slab,) = scene.elements_of(ElementKind.SLAB)
    assert isinstance(slab, SlabElement)
    assert slab.params.thickness == 0.25
    assert slab.params.top_elevation == 0.0
    ceiling = scene.elements_of(ElementKind.CEILING)[0]
    assert isinstance(ceiling, CeilingElement)
    assert ceiling.params.elevation == 2.8
    assert ceiling.params.thickness == 0.20
    assert ceiling.params.sag == 0.0
    spaces = [e for e in scene.elements.values() if isinstance(e, SpaceElement)]
    assert [s.params.name for s in spaces] == ["Room 1", "Room 2"][: len(spaces)]
    assert [s.params.number for s in spaces] == ["001", "002"][: len(spaces)]


def test_spec_values_flow_into_scene() -> None:
    spec = RoomSpec(
        storey_elevation=3.0,
        ceiling_sag=0.05,
        origin_offset=(1.0, 2.0, 3.0),
        rectangle_size=(6.0, 5.0),
    )
    scene = build_room_scene(spec)
    assert scene.settings.origin_offset == (1.0, 2.0, 3.0)
    (ceiling,) = scene.elements_of(ElementKind.CEILING)
    assert isinstance(ceiling, CeilingElement)
    assert ceiling.params.sag == 0.05
    assert ceiling.params.elevation == pytest.approx(5.8)
    assert 6.0 in {wall.params.end[0] for wall in _walls(scene)}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_openings_inside_walls(layout: RoomLayout) -> None:
    scene = _scene(layout)
    openings = [e for e in scene.elements.values() if isinstance(e, OpeningElement)]
    assert openings
    for opening in openings:
        wall = scene.get(opening.params.host_wall_id)
        assert isinstance(wall, WallElement)
        pos, w = opening.params.position_along_axis, opening.params.width
        assert pos - w / 2 >= 0.0
        assert pos + w / 2 <= _length(wall)
        assert opening.params.sill_height + opening.params.height <= wall.params.height


@pytest.mark.parametrize("layout", LAYOUTS)
def test_join_geometry(layout: RoomLayout) -> None:
    scene = _scene(layout)
    for join in _joins(scene):
        wall_a, wall_b = scene.get(join.params.wall_a), scene.get(join.params.wall_b)
        assert isinstance(wall_a, WallElement)
        assert isinstance(wall_b, WallElement)
        if join.params.join_type == JoinType.L:
            assert join.params.end_a != JoinEnd.PATH
            assert join.params.end_b != JoinEnd.PATH
            assert _end(wall_a, join.params.end_a) == _end(wall_b, join.params.end_b)
        else:
            assert join.params.end_a in (JoinEnd.START, JoinEnd.END)
            assert join.params.end_b == JoinEnd.PATH
            px, py = _end(wall_a, join.params.end_a)
            (ax, ay), (bx, by) = wall_b.params.start, wall_b.params.end
            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            assert cross == pytest.approx(0.0, abs=1e-12)
            dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
            assert 0.0 <= dot <= _length(wall_b) ** 2


def test_two_rooms_partition_joins_both_ends() -> None:
    t_joins = [j for j in _joins(_scene(RoomLayout.TWO_ROOMS)) if j.params.join_type == JoinType.T]
    assert {j.params.end_a for j in t_joins} == {JoinEnd.START, JoinEnd.END}
    assert len({j.params.wall_a for j in t_joins}) == 1
    assert len({j.params.wall_b for j in t_joins}) == 2


@pytest.mark.parametrize("layout", LAYOUTS)
def test_deterministic(layout: RoomLayout) -> None:
    first = build_room_scene(RoomSpec(layout=layout))
    second = build_room_scene(RoomSpec(layout=layout))
    assert first.to_json() == second.to_json()
    assert list(first.elements) == list(second.elements)
    assert Scene.model_validate_json(first.to_json()) == first


def test_ids_independent_of_seed_and_unique() -> None:
    a = build_room_scene(RoomSpec(seed=1))
    b = build_room_scene(RoomSpec(seed=2))
    assert list(a.elements) == list(b.elements)
    assert len(set(a.elements) | set(a.types)) == len(a.elements) + len(a.types)


def test_invalid_specs() -> None:
    with pytest.raises(ValueError, match="window"):
        RoomSpec(window_sill=2.0, window_height=1.4)
    with pytest.raises(ValueError, match="fit"):
        build_room_scene(RoomSpec(rectangle_size=(1.0, 1.0)))


@pytest.mark.parametrize("layout", list(RoomLayout))
def test_space_membership_storey_and_seeds(layout: RoomLayout) -> None:
    scene = build_room_scene(RoomSpec(layout=layout))
    storey = next(iter(scene.storeys))
    spaces = [e for e in scene.elements.values() if isinstance(e, SpaceElement)]
    space_ids = [s.id for s in spaces]
    assert all(e.storey_id == storey for e in scene.elements.values())
    width, depth = RoomSpec(layout=layout).size
    assert all(s.params.seed is not None for s in spaces)
    if layout == RoomLayout.RECTANGLE:
        assert spaces[0].params.seed == (width / 2.0, depth / 2.0)
    else:
        assert spaces[0].params.seed == (width / 4.0, depth / 2.0)
        assert spaces[1].params.seed == (3.0 * width / 4.0, depth / 2.0)
    ceilings = scene.elements_of(ElementKind.CEILING)
    for element in scene.elements.values():
        if element.kind == ElementKind.CEILING and layout == RoomLayout.TWO_ROOMS:
            # one ceiling per room (the partition cuts the envelope); each belongs to its room
            assert element.space_ids == [space_ids[ceilings.index(element)]]
        elif element.kind in (ElementKind.SLAB, ElementKind.CEILING):
            assert element.space_ids == space_ids
        elif element.kind == ElementKind.WALL_JOIN:
            assert element.space_ids == []
    walls = {e.tag: e for e in scene.elements.values() if e.kind == ElementKind.WALL}
    assert all(w.space_ids for w in walls.values())
    if layout == RoomLayout.TWO_ROOMS:
        assert walls["partition"].space_ids == space_ids
        assert walls["exterior_west"].space_ids == [space_ids[0]]
        assert walls["exterior_east"].space_ids == [space_ids[1]]
    for element in scene.elements.values():
        if isinstance(element, OpeningElement):
            host = scene.elements[element.params.host_wall_id]
            assert element.space_ids == host.space_ids
        if isinstance(element, WindowElement | DoorElement):
            opening = scene.elements[element.params.opening_id]
            assert element.space_ids == opening.space_ids
    assert scene.check_references() == []
    geometry = compute_scene_geometry(scene)
    assert set(geometry.spaces) == set(space_ids)
    assert all(not any("ambiguous" in i for i in geometry.issues.get(s, ())) for s in space_ids)
    assert all(g.matched_by == "seed" for g in geometry.spaces.values())
