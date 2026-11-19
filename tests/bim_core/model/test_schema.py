"""Tests of the parametric scene model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from samcloud.bim.core.model import (
    PARAM_SCHEMAS,
    Building,
    CrsInfo,
    ElementKind,
    ElementType,
    Metrics,
    ParamOrigin,
    PropertyValue,
    Scene,
    Site,
    SlabElement,
    SpaceElement,
    Storey,
    TerrainElement,
    WallElement,
    check_param_ranges,
    get_param_schema,
    parse_element,
)
from samcloud.bim.core.model.elements import PARAMS_MODELS

SQUARE = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]


def wall_data(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "start": [0.0, 0.0],
        "end": [4.0, 0.0],
        "thickness": 0.2,
        "height": 2.6,
    }
    params.update(overrides)
    return {"kind": "wall", "params": params}


def build_scene() -> Scene:
    scene = Scene()
    site = Site(name="Site")
    b = Building(name="House", site_id=site.id)
    storey = Storey(name="EG", elevation=0.0, height=2.8, building_id=b.id)
    scene.sites[site.id] = site
    scene.buildings[b.id] = b
    scene.storeys[storey.id] = storey
    wall = WallElement(
        params={"start": (0, 0), "end": (4, 0), "thickness": 0.2, "height": 2.6},
        param_origin={"thickness": ParamOrigin.MEASURED, "height": ParamOrigin.ASSUMED},
        storey_id=storey.id,
        building_id=b.id,
        site_id=site.id,
        properties={
            "IsExternal": PropertyValue(value=True, value_type="boolean", pset="Pset_WallCommon"),
            "Note": PropertyValue(value="x", value_type="text"),
            "Area": PropertyValue(value=3, value_type="number"),
        },
        metrics=Metrics(deviation_rms=0.004, coverage=0.8, point_count=1200),
    )
    scene.add(wall)
    return scene


def test_round_trip_save_load(tmp_path: Path) -> None:
    scene = build_scene()
    target = tmp_path / "scene" / "scene.json"
    scene.save(target)
    loaded = Scene.load(target)
    assert loaded == scene
    assert loaded.model_dump(mode="json") == scene.model_dump(mode="json")
    wall = loaded.elements_of(ElementKind.WALL)[0]
    assert isinstance(wall, WallElement)
    assert wall.properties["Area"].value == 3.0
    assert wall.ifc_class == "IfcWall"
    assert wall.predefined_type == "STANDARD"


def test_save_is_deterministic(tmp_path: Path) -> None:
    scene = build_scene()
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    scene.save(first)
    scene.save(second)
    assert first.read_bytes() == second.read_bytes()
    reloaded = tmp_path / "c.json"
    Scene.load(first).save(reloaded)
    assert reloaded.read_bytes() == first.read_bytes()
    text = first.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\r" not in text
    data = json.loads(text)
    assert list(data) == sorted(data)
    assert not (tmp_path / "a.json.tmp").exists()


def test_float64_precision_for_large_coordinates(tmp_path: Path) -> None:
    x, y = 2_600_000.123456789, 1_200_000.987654321
    wall = WallElement(
        params={"start": (x, y), "end": (x + 4.0, y), "thickness": 0.2, "height": 2.6}
    )
    scene = Scene()
    scene.settings.origin_offset = (2_600_000.0, 1_200_000.0, 400.123456789)
    scene.add(wall)
    target = tmp_path / "s.json"
    scene.save(target)
    loaded = Scene.load(target)
    loaded_wall = loaded.get(wall.id)
    assert isinstance(loaded_wall, WallElement)
    assert loaded_wall.params.start == (x, y)
    assert loaded_wall.params.start[0] == 2_600_000.123456789
    assert loaded.settings.origin_offset == (2_600_000.0, 1_200_000.0, 400.123456789)


@pytest.mark.parametrize(
    "kind,params",
    [
        ("wall", {"start": [0, 0], "end": [1, 0], "thickness": 0.2, "height": 2.5}),
        (
            "wall_join",
            {"wall_a": "a", "wall_b": "b", "join_type": "L", "end_a": "end", "end_b": "start"},
        ),
        ("slab", {"outline": [[0, 0], [4, 0], [4, 3]], "thickness": 0.25, "top_elevation": 0.0}),
        ("ceiling", {"outline": [[0, 0], [4, 0], [4, 3]], "elevation": 2.6}),
        (
            "opening",
            {"host_wall_id": "w", "position_along_axis": 1.0, "width": 0.9, "height": 2.1},
        ),
        ("window", {"opening_id": "o", "operation": "tilt_and_turn"}),
        ("door", {"opening_id": "o", "operation": "sliding", "swing_side": "left"}),
        ("space", {"name": "Office", "number": "1.01"}),
        (
            "stair",
            {
                "start": [0, 0],
                "width": 1.0,
                "step_count": 14,
                "riser_height": 0.18,
                "tread_length": 0.28,
            },
        ),
        ("furniture", {"template_id": "chair"}),
        ("terrain", {"resolution": 0.5}),
        ("building", {"roof_type": "flat"}),
    ],
)
def test_discriminated_union_parses_each_kind(kind: str, params: dict[str, Any]) -> None:
    element = parse_element({"kind": kind, "params": params})
    assert element.kind == ElementKind(kind)
    assert element.ifc_class.startswith("Ifc")
    assert type(element).__name__.lower().startswith(kind.replace("_", ""))
    again = parse_element(json.loads(element.model_dump_json()))
    assert again == element
    scene = Scene()
    scene.add(element)
    assert Scene.model_validate_json(scene.to_json()) == scene


def test_unknown_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_element({"kind": "banana", "params": {}})


def test_terrain_is_non_parametric_by_default() -> None:
    terrain = parse_element({"kind": "terrain", "params": {}})
    assert isinstance(terrain, TerrainElement)
    assert terrain.non_parametric is True
    assert terrain.predefined_type == "TERRAIN"


def test_wall_default_layer_equals_thickness() -> None:
    wall = parse_element(wall_data(thickness=0.175))
    assert isinstance(wall, WallElement)
    assert len(wall.params.layers) == 1
    assert wall.params.layers[0].thickness == 0.175


def test_wall_layers_must_sum_to_thickness() -> None:
    good = parse_element(
        wall_data(
            thickness=0.3,
            layers=[
                {"material": "Brick", "thickness": 0.2},
                {"material": "Plaster", "thickness": 0.1},
            ],
        )
    )
    assert isinstance(good, WallElement)
    assert [layer.material for layer in good.params.layers] == ["Brick", "Plaster"]
    with pytest.raises(ValidationError, match="layer thickness sum"):
        parse_element(
            wall_data(
                thickness=0.3,
                layers=[
                    {"material": "Brick", "thickness": 0.2},
                    {"material": "P", "thickness": 0.11},
                ],
            )
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"end": [0.0, 0.0]},
        {"thickness": 0.0},
        {"thickness": -0.1},
        {"height": 0.0},
        {"height": float("nan")},
        {"start": [float("inf"), 0.0]},
        {"layers": [{"material": "", "thickness": 0.2}]},
        {"layers": [{"material": "x", "thickness": 0.0}]},
        {"justification": "middle"},
        {"unknown_field": 1},
    ],
)
def test_wall_rejects_invalid(overrides: dict[str, Any]) -> None:
    data = wall_data(**overrides)
    # json cannot carry nan/inf, so use python validation here
    with pytest.raises(ValidationError):
        parse_element(data)


def test_wall_origin_keys_must_be_parameters() -> None:
    data = wall_data()
    data["param_origin"] = {"colour": "measured"}
    with pytest.raises(ValidationError, match="unknown parameters"):
        parse_element(data)
    data["param_origin"] = {"thickness": "guessed"}
    with pytest.raises(ValidationError):
        parse_element(data)


@pytest.mark.parametrize(
    "outline",
    [
        [[0, 0], [1, 0]],
        [],
        [[0, 0], [4, 0], [4, 3], [0, 0]],
        [[0, 0], [1, 1], [2, 2]],
    ],
)
def test_slab_and_ceiling_reject_bad_outlines(outline: list[list[float]]) -> None:
    with pytest.raises(ValidationError):
        parse_element(
            {"kind": "slab", "params": {"outline": outline, "thickness": 0.2, "top_elevation": 0}}
        )
    with pytest.raises(ValidationError):
        parse_element({"kind": "ceiling", "params": {"outline": outline, "elevation": 2.6}})


def test_slab_opening_validation_and_layers() -> None:
    base = {"outline": [[0, 0], [4, 0], [4, 3], [0, 3]], "thickness": 0.2, "top_elevation": 0.0}
    slab = parse_element(
        {"kind": "slab", "params": {**base, "openings": [[[1, 1], [2, 1], [2, 2]]]}}
    )
    assert isinstance(slab, SlabElement)
    assert len(slab.params.openings) == 1
    assert slab.params.layers[0].thickness == 0.2
    with pytest.raises(ValidationError):
        parse_element({"kind": "slab", "params": {**base, "openings": [[[1, 1], [2, 1]]]}})
    with pytest.raises(ValidationError, match="layer thickness sum"):
        parse_element(
            {"kind": "slab", "params": {**base, "layers": [{"material": "C", "thickness": 0.1}]}}
        )
    with pytest.raises(ValidationError):
        parse_element({"kind": "slab", "params": {**base, "thickness": 0}})


def test_ceiling_plane_rules() -> None:
    base = {"outline": [[0, 0], [4, 0], [4, 3]], "elevation": 2.6}
    ok = parse_element(
        {"kind": "ceiling", "params": {**base, "plane_normal": [0, 0.1, 1], "plane_offset": 2.6}}
    )
    assert ok.kind == ElementKind.CEILING
    with pytest.raises(ValidationError, match="together"):
        parse_element({"kind": "ceiling", "params": {**base, "plane_normal": [0, 0, 1]}})
    with pytest.raises(ValidationError, match="zero vector"):
        parse_element(
            {"kind": "ceiling", "params": {**base, "plane_normal": [0, 0, 0], "plane_offset": 0}}
        )
    with pytest.raises(ValidationError):
        parse_element({"kind": "ceiling", "params": {**base, "sag": -0.01}})


@pytest.mark.parametrize(
    "kind,params",
    [
        ("opening", {"host_wall_id": "w", "position_along_axis": -1, "width": 1, "height": 1}),
        ("opening", {"host_wall_id": "w", "position_along_axis": 1, "width": 0, "height": 1}),
        ("opening", {"host_wall_id": "w", "position_along_axis": 1, "width": 1, "height": -1}),
        (
            "opening",
            {
                "host_wall_id": "w",
                "position_along_axis": 1,
                "width": 1,
                "height": 1,
                "sill_height": -0.1,
            },
        ),
        (
            "opening",
            {
                "host_wall_id": "w",
                "position_along_axis": 1,
                "width": 1,
                "height": 1,
                "inside_side": "up",
            },
        ),
        ("opening", {"host_wall_id": "", "position_along_axis": 1, "width": 1, "height": 1}),
        ("window", {"opening_id": "o", "frame_depth": 0}),
        ("window", {"opening_id": "o", "sill_thickness": -1}),
        ("window", {"opening_id": "o", "operation": "swing"}),
        ("door", {"opening_id": "o", "swing_side": "middle"}),
        ("door", {"opening_id": "o", "frame_depth": -0.1}),
        (
            "wall_join",
            {"wall_a": "a", "wall_b": "a", "join_type": "L", "end_a": "end", "end_b": "start"},
        ),
        (
            "wall_join",
            {"wall_a": "a", "wall_b": "b", "join_type": "X", "end_a": "end", "end_b": "start"},
        ),
    ],
)
def test_other_kinds_reject_invalid(kind: str, params: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        parse_element({"kind": kind, "params": params})


def test_property_value_type_checks() -> None:
    assert PropertyValue(value=2, value_type="number").value == 2.0
    for value, value_type in [("a", "number"), (True, "number"), (1.0, "boolean"), (1.0, "text")]:
        with pytest.raises(ValidationError):
            PropertyValue(value=value, value_type=value_type)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        PropertyValue(value=1.0, value_type="other")  # type: ignore[arg-type]


def test_metrics_ranges() -> None:
    with pytest.raises(ValidationError):
        Metrics(coverage=1.5)
    with pytest.raises(ValidationError):
        Metrics(confidence=-0.1)
    with pytest.raises(ValidationError):
        Metrics(deviation_rms=-0.001)
    with pytest.raises(ValidationError):
        Metrics(point_count=-1)


def test_scene_key_must_match_id() -> None:
    wall = WallElement(params={"start": (0, 0), "end": (1, 0), "thickness": 0.2, "height": 2.5})
    with pytest.raises(ValidationError, match="differs from id"):
        Scene(elements={"other": wall})
    scene = Scene()
    scene.add(wall)
    with pytest.raises(ValueError, match="duplicate"):
        scene.add(wall)


def test_scene_get_and_elements_of() -> None:
    scene = build_scene()
    scene.add(SpaceElement(params={"name": "Office", "number": "1"}))
    assert len(scene.elements_of(ElementKind.WALL)) == 1
    assert len(scene.elements_of(ElementKind.SPACE)) == 1
    assert scene.elements_of(ElementKind.DOOR) == []
    space = scene.elements_of(ElementKind.SPACE)[0]
    assert scene.get(space.id) is space
    with pytest.raises(KeyError):
        scene.get("missing")


def test_check_references() -> None:
    scene = build_scene()
    assert scene.check_references() == []
    wall = scene.elements_of(ElementKind.WALL)[0]
    opening = parse_element(
        {
            "kind": "opening",
            "params": {"host_wall_id": wall.id, "position_along_axis": 1, "width": 1, "height": 2},
        }
    )
    scene.add(opening)
    door = parse_element({"kind": "door", "params": {"opening_id": opening.id}})
    scene.add(door)
    assert scene.check_references() == []
    scene.add(parse_element({"kind": "window", "params": {"opening_id": wall.id}}))
    scene.add(
        parse_element(
            {
                "kind": "opening",
                "storey_id": "nope",
                "params": {
                    "host_wall_id": "ghost",
                    "position_along_axis": 0,
                    "width": 1,
                    "height": 1,
                },
            }
        )
    )
    problems = scene.check_references()
    assert any("expected opening" in p for p in problems)
    assert any("ghost" in p for p in problems)
    assert any("storey_id" in p for p in problems)


def test_settings_defaults_and_crs() -> None:
    scene = Scene()
    assert scene.settings.ceiling_ifc_class.value == "slab"
    assert scene.settings.auto_lock_on_manual_edit is False
    assert scene.settings.crs is None
    assert scene.settings.origin_offset == (0.0, 0.0, 0.0)
    assert scene.derived == {}
    assert scene.schema_version == 1
    scene.settings.crs = CrsInfo(epsg=2056, name="CH1903+ / LV95", source="input")
    assert Scene.model_validate_json(scene.to_json()).settings.crs == scene.settings.crs
    with pytest.raises(ValidationError):
        CrsInfo(source="guess")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Scene.model_validate({"settings": {"scope": "everywhere"}})
    with pytest.raises(ValidationError):
        Scene.model_validate({"settings": {"unknown": 1}})


def test_element_type_layers_only_on_wall_and_slab() -> None:
    layer = {"material": "Brick", "thickness": 0.2}
    assert ElementType(kind=ElementKind.WALL, layers=[layer]).layers[0].material == "Brick"  # type: ignore[list-item]
    with pytest.raises(ValidationError):
        ElementType(kind=ElementKind.DOOR, layers=[layer])  # type: ignore[list-item]


def test_storey_height_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Storey(name="x", height=0.0)


def test_get_param_schema_for_all_m1_kinds() -> None:
    m1 = [
        ElementKind.WALL,
        ElementKind.WALL_JOIN,
        ElementKind.SLAB,
        ElementKind.CEILING,
        ElementKind.OPENING,
        ElementKind.WINDOW,
        ElementKind.DOOR,
        ElementKind.SPACE,
    ]
    for kind in m1:
        schema = get_param_schema(kind)
        json.dumps(schema)
        assert schema["kind"] == kind.value
        names = [p["name"] for p in schema["params"]]
        assert set(names) == set(PARAMS_MODELS[kind].model_fields), kind
        for param in schema["params"]:
            assert set(param) >= {"name", "unit", "min", "max", "step", "type", "description"}
            if param["type"] == "number":
                assert param["unit"] == "m"
                assert param["min"] < param["max"]
                assert param["step"] > 0
            if param["type"] == "enum":
                assert param["choices"]
    wall = {p["name"]: p for p in get_param_schema("wall")["params"]}
    assert (wall["thickness"]["min"], wall["thickness"]["max"], wall["thickness"]["step"]) == (
        0.02,
        2.0,
        0.005,
    )
    assert (wall["height"]["min"], wall["height"]["max"]) == (0.5, 20.0)
    assert "left" in wall["justification"]["choices"]
    stair = {p["name"]: p for p in get_param_schema(ElementKind.STAIR)["params"]}
    assert stair["riser_height"]["unit"] == "m"
    assert (stair["step_count"]["min"], stair["step_count"]["max"]) == (2, 200)
    assert ElementKind.STAIR in PARAM_SCHEMAS
    with pytest.raises(KeyError):
        get_param_schema(ElementKind.FURNITURE)


def test_schema_ranges_accepted_by_models() -> None:
    wall = {p["name"]: p for p in get_param_schema("wall")["params"]}
    for key in ("min", "max"):
        element = parse_element(
            wall_data(thickness=wall["thickness"][key], height=wall["height"][key])
        )
        assert isinstance(element, WallElement)
    opening = {p["name"]: p for p in get_param_schema("opening")["params"]}
    for key in ("min", "max"):
        parse_element(
            {
                "kind": "opening",
                "params": {
                    "host_wall_id": "w",
                    "position_along_axis": opening["position_along_axis"][key],
                    "width": opening["width"][key],
                    "height": opening["height"][key],
                    "sill_height": opening["sill_height"][key],
                    "reveal_depth": opening["reveal_depth"][key],
                },
            }
        )


def test_check_param_ranges() -> None:
    assert check_param_ranges("wall", {"thickness": 0.2, "height": 2.6}) == []
    messages = check_param_ranges("wall", {"thickness": 5.0, "height": 0.1})
    assert len(messages) == 2
    assert "thickness" in messages[0] and "maximum" in messages[0]
    assert "height" in messages[1] and "minimum" in messages[1]


def test_check_references_wall_join() -> None:
    scene = build_scene()
    wall = scene.elements_of(ElementKind.WALL)[0]
    join = parse_element(
        {
            "kind": "wall_join",
            "params": {
                "wall_a": wall.id,
                "wall_b": "ghost",
                "join_type": "T",
                "end_a": "end",
                "end_b": "path",
            },
        }
    )
    scene.add(join)
    problems = scene.check_references()
    assert len(problems) == 1
    assert "wall_b" in problems[0] and "ghost" in problems[0]


def test_space_seed_param_and_schema() -> None:
    from samcloud.bim.core.model import SpaceParams, get_param_schema

    assert SpaceParams().seed is None
    assert SpaceParams(seed=(1.0, 2.0)).seed == (1.0, 2.0)
    seed = next(p for p in get_param_schema("space")["params"] if p["name"] == "seed")
    assert seed["unit"] == "m"
    assert seed["type"] == "point2"
