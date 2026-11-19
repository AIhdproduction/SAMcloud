"""Re-run policy: matching, manual and locked protection, removal, remapping and group undo."""

from __future__ import annotations

import json
from typing import Any

import pytest

from samcloud.bim.core.model import (
    Element,
    OpeningElement,
    OpeningParams,
    ParamOrigin,
    Provenance,
    Scene,
    WallElement,
    WallParams,
    parse_element,
)
from samcloud.bim.core.model.graph import recompute
from samcloud.bim.core.model.patches import PatchEngine
from samcloud.bim.core.model.rerun import REMOVAL_REASON, RerunResult, RerunSettings, merge_rerun
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene

TWO = RoomSpec(layout=RoomLayout.TWO_ROOMS)


def make(spec: RoomSpec = TWO) -> tuple[Scene, PatchEngine]:
    scene = build_room_scene(spec)
    recompute(scene)
    return scene, PatchEngine(scene)


def fresh(spec: RoomSpec = TWO) -> tuple[list[Element], list[Any]]:
    other = build_room_scene(spec)
    return list(other.elements.values()), list(other.types.values())


def document(scene: Scene) -> str:
    """Scene JSON without the derived values (compared separately, see ``derived_document``)."""
    data = scene.model_dump(mode="json")
    data.pop("derived")
    return json.dumps(data, sort_keys=True, indent=2)


def derived_document(scene: Scene) -> str:
    """The derived values byte for byte: no key sorting, floats exactly as stored."""
    return json.dumps(scene.derived, indent=2)


def by_tag(elements: Any, tag: str) -> Element:
    return next(e for e in elements if e.tag == tag)


def window_of(elements: Any, tag: str) -> Element:
    return next(e for e in elements if e.kind.value == "window" and e.tag == tag)


def run(
    scene: Scene, engine: PatchEngine, elements: list[Element], types: list[Any], **kw: Any
) -> RerunResult:
    return merge_rerun(scene, elements, types, engine, run_id="r1", **kw)


def rename(elements: list[Element], prefix: str = "new-") -> list[Element]:
    """Copies with all element ids replaced (references follow)."""
    text = json.dumps([e.model_dump(mode="json") for e in elements])
    for element in elements:
        text = text.replace(element.id, prefix + element.id)
    return [parse_element(d) for d in json.loads(text)]


def shift_wall(wall: Element, dx: float, thickness: float | None = None) -> None:
    assert isinstance(wall, WallElement)
    data = wall.params.model_dump(mode="json")
    data["start"] = (data["start"][0] + dx, data["start"][1])
    data["end"] = (data["end"][0] + dx, data["end"][1])
    if thickness is not None:
        data["thickness"] = thickness
        data["layers"] = [{"material": "Unspecified", "thickness": thickness}]
    wall.params = WallParams.model_validate(data)


def hosted_openings(elements: Any, wall_id: str) -> list[str]:
    return [
        e.id for e in elements if isinstance(e, OpeningElement) and e.params.host_wall_id == wall_id
    ]


def drop_south(elements: list[Element]) -> list[Element]:
    """The re-run does not find the south wall, its openings, windows and joins."""
    south = by_tag(elements, "exterior_south").id
    openings = set(hosted_openings(elements, south))
    kept: list[Element] = []
    for e in elements:
        if e.id == south or e.id in openings:
            continue
        if e.kind.value == "window" and e.params.opening_id in openings:  # type: ignore[union-attr]
            continue
        if e.kind.value == "wall_join" and south in (e.params.wall_a, e.params.wall_b):  # type: ignore[union-attr]
            continue
        kept.append(e)
    return kept


def extra_wall(template: Element, new_id: str = "extra-wall") -> WallElement:
    assert isinstance(template, WallElement)
    extra = template.model_copy(deep=True)
    extra.id = new_id
    extra.tag = "extra"
    extra.space_ids = []
    data = extra.params.model_dump(mode="json")
    data["start"], data["end"] = (1.0, 1.0), (3.0, 1.0)
    extra.params = WallParams.model_validate(data)
    return extra


def test_identical_rerun_changes_nothing() -> None:
    scene, engine = make()
    before = scene.to_json()
    elements, types = fresh()
    result = run(scene, engine, elements, types)
    assert result.patch_ids == []
    assert not result.changed
    assert result.added_ids == [] and result.removed_ids == []
    assert len(result.matches) == len(scene.elements)
    assert all(m.updated_params == () and m.score > 0.99 for m in result.matches)
    assert all(result.id_map[e.id] == e.id for e in elements)
    assert engine.history == []
    assert scene.to_json() == before


def test_identical_rerun_with_new_ids_keeps_old_ids() -> None:
    scene, engine = make()
    before = scene.to_json()
    elements, types = fresh()
    result = run(scene, engine, rename(elements), types)
    assert result.patch_ids == []
    assert scene.to_json() == before
    assert all(result.id_map["new-" + e.id] == e.id for e in elements)


def test_shifted_rerun_updates_non_manual_and_keeps_manual_and_locked() -> None:
    scene, engine = make()
    pool = list(scene.elements.values())
    east = by_tag(pool, "exterior_east")
    north = by_tag(pool, "exterior_north")
    west = by_tag(pool, "exterior_west")
    south = by_tag(pool, "exterior_south")
    assert isinstance(east, WallElement) and isinstance(south, WallElement)
    # user-edited thickness (origin manual) on the east wall, provenance still script
    east.param_origin["thickness"] = ParamOrigin.MANUAL
    east.param_origin["layers"] = ParamOrigin.MANUAL
    manual_thickness = east.params.thickness
    west.locked = True
    west_before = west.model_dump(mode="json")
    south_start = south.params.start
    elements, types = fresh()
    for tag in ("exterior_east", "exterior_north", "exterior_west", "exterior_south"):
        shift_wall(by_tag(elements, tag), 0.03, thickness=0.35)
    result = run(scene, engine, elements, types)

    assert result.removed_ids == [] and result.added_ids == []
    assert set(result.id_map.values()) == set(scene.elements)
    new_east = scene.elements[east.id]
    assert isinstance(new_east, WallElement)
    assert new_east.params.thickness == manual_thickness
    assert new_east.param_origin["thickness"] == ParamOrigin.MANUAL
    assert new_east.params.start[0] == pytest.approx(east.params.start[0] + 0.03)
    assert "thickness" in result.kept_manual_params[east.id]
    new_north = scene.elements[north.id]
    assert isinstance(new_north, WallElement)
    assert new_north.params.thickness == 0.35
    assert new_north.params.layers[0].thickness == 0.35
    new_south = scene.elements[south.id]
    assert isinstance(new_south, WallElement)
    assert new_south.params.start[0] == pytest.approx(south_start[0] + 0.03)
    assert scene.elements[west.id].model_dump(mode="json") == west_before
    protected = [m for m in result.matches if m.existing_id == west.id]
    assert protected and protected[0].protected
    assert {r.group_id for r in engine.history} == {"rerun-r1"}
    assert all(r.author == "samcloud rerun" and r.author_kind == "script" for r in engine.history)


def test_missing_wall_is_removed_with_its_openings_and_windows() -> None:
    scene, engine = make()
    south = by_tag(scene.elements.values(), "exterior_south")
    hosted = hosted_openings(scene.elements.values(), south.id)
    windows = [
        e.id
        for e in scene.elements.values()
        if e.kind.value == "window" and e.params.opening_id in hosted  # type: ignore[union-attr]
    ]
    assert hosted and windows
    elements, types = fresh()
    result = run(scene, engine, drop_south(elements), types)
    for gone in [south.id, *hosted, *windows]:
        assert gone in result.removed_ids
        assert gone not in scene.elements
    assert not scene.check_references()
    removal = engine.history[-1]
    assert removal.reason.endswith(REMOVAL_REASON)
    removed_paths = {op["path"] for op in removal.forward if op["op"] == "remove"}
    assert f"/elements/{south.id}" in removed_paths
    assert result.added_ids == []


def test_new_wall_is_added() -> None:
    scene, engine = make()
    elements, types = fresh()
    extra = extra_wall(by_tag(elements, "partition"))
    result = run(scene, engine, [*elements, extra], types)
    assert result.added_ids == ["extra-wall"]
    added = scene.elements["extra-wall"]
    assert added.provenance == Provenance.SCRIPT and not added.locked
    assert result.removed_ids == []


def test_unmatched_manual_element_is_kept() -> None:
    scene, engine = make()
    partition = by_tag(scene.elements.values(), "partition")
    partition.provenance = Provenance.MANUAL
    elements, types = fresh()
    kept = [e for e in elements if "partition" not in e.tag]
    result = run(scene, engine, kept, types)
    assert partition.id in scene.elements
    assert partition.id in result.kept_unmatched_ids
    assert partition.id not in result.removed_ids
    assert not scene.check_references()


def test_locked_window_keeps_its_opening_and_host_wall() -> None:
    scene, engine = make()
    south = by_tag(scene.elements.values(), "exterior_south")
    window = window_of(scene.elements.values(), "window_south_1")
    window.locked = True
    elements, types = fresh()
    result = run(scene, engine, drop_south(elements), types)
    assert window.id in scene.elements
    assert south.id in scene.elements
    assert south.id not in result.removed_ids
    assert any("kept" in note for note in result.notes)
    assert not scene.check_references()


def test_undo_group_restores_scene_exactly() -> None:
    scene, engine = make()
    by_tag(scene.elements.values(), "exterior_east").locked = True
    before = document(scene)
    derived_before = derived_document(scene)
    elements, types = fresh()
    elements = drop_south(elements)
    for tag in ("exterior_north", "exterior_west"):
        shift_wall(by_tag(elements, tag), 0.03, thickness=0.35)
    extra = extra_wall(by_tag(elements, "partition"))
    result = run(scene, engine, [*elements, extra], types)
    assert len(result.patch_ids) == 3
    assert document(scene) != before
    assert derived_document(scene) != derived_before
    engine.undo_group(result.group_id)
    assert document(scene) == before
    assert derived_document(scene) == derived_before


def test_references_of_new_elements_are_remapped_to_existing_ids() -> None:
    scene, engine = make()
    elements, types = fresh()
    elements = rename(elements)
    north_new = by_tag(elements, "exterior_north")
    north_old = by_tag(scene.elements.values(), "exterior_north")
    extra = OpeningElement(
        id="new-extra-opening",
        tag="extra",
        params=OpeningParams(
            host_wall_id=north_new.id, position_along_axis=0.5, width=0.6, height=1.0
        ),
        storey_id=north_old.storey_id,
    )
    result = run(scene, engine, [*elements, extra], types)
    assert result.added_ids == [result.id_map["new-extra-opening"]]
    added = scene.elements[result.added_ids[0]]
    assert isinstance(added, OpeningElement)
    assert added.params.host_wall_id == north_old.id
    assert result.removed_ids == []
    assert not scene.check_references()


def test_rectangle_rerun_of_two_room_project_removes_partition_stuff() -> None:
    scene, engine = make()
    elements, types = fresh(RoomSpec())
    result = run(scene, engine, elements, types)
    assert all(e.tag != "partition" for e in scene.elements.values())
    assert result.removed_ids
    assert not scene.check_references()
    assert len(set(result.id_map.values())) == len(result.id_map)


def test_new_id_colliding_with_removed_element_gets_fresh_id() -> None:
    scene, engine = make()
    elements, types = fresh()
    # a new window with the id of an old one, hosted on the north wall where no window matches
    old_window = window_of(scene.elements.values(), "window_south_1")
    north = by_tag(elements, "exterior_north")
    opening = OpeningElement(
        id="o-new",
        tag="o-new",
        params=OpeningParams(host_wall_id=north.id, position_along_axis=0.5, width=0.6, height=1.0),
    )
    clash = old_window.model_copy(deep=True)
    clash.id = old_window.id
    clash.params.opening_id = opening.id  # type: ignore[union-attr]
    kept = [e for e in elements if e.id != old_window.id]
    result = run(scene, engine, [*kept, opening, clash], types)
    new_id = result.id_map[old_window.id]
    assert new_id != old_window.id
    assert old_window.id in result.removed_ids and old_window.id not in scene.elements
    assert scene.elements[new_id].params.opening_id == result.id_map["o-new"]  # type: ignore[union-attr]


def test_thresholds_are_configurable() -> None:
    scene, engine = make()
    east = by_tag(scene.elements.values(), "exterior_east")
    elements, types = fresh()
    shift_wall(by_tag(elements, "exterior_east"), 0.2)
    strict = RerunSettings(wall_iou_min=0.9)
    result = run(scene, engine, elements, types, settings=strict)
    # shifted by 0.2 m on a 0.3 m wall: IoU 0.2 -> no match at 0.9; the old wall is replaced
    matched = {m.existing_id for m in result.matches}
    assert east.id not in matched
    assert result.added_ids and east.id in result.removed_ids
    assert RerunSettings().wall_iou_min == 0.5
    assert RerunSettings().wall_angle_max_deg == 10.0
