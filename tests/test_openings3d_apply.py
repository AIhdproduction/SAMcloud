"""The clean-plan hook: refined openings replace the 2D ones; DXF symbols carry sill/head/mullions; no re-plan loop."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import ezdxf
import numpy as np
import pytest
from shapely.geometry import LineString

import test_ortho_plan as scene_tests
from openings3d_synth import make_plan, seed_door
from samcloud.bim.openings3d import apply as apply_module
from samcloud.bim.openings3d import pipeline
from samcloud.bim.openings3d.apply import apply_refined_openings, load_matching_openings
from samcloud.bim.openings3d.config import config_hash, load_openings_config
from samcloud.bim.openings3d.model import OpeningsResult, RefinedOpening, openings_3d_path, openings_3d_sha256, save_openings_3d
from samcloud.bim.openings3d.walls import walls_fingerprint, wall_lines
from samcloud.bim.plan_model import load_plan_model, save_plan_model
from samcloud.dxf.writer import build_dxf_document
from samcloud.ortho.plan import clean_plan_cache_dir, generate_clean_plan
from samcloud.ortho.vector.indoor import Opening, opening_entities


def _refined(kind: str = "window", centre_x: float = 2.0, width: float = 1.2, **kwargs) -> RefinedOpening:
    values = dict(
        id="C001", kind=kind, center=[centre_x, 0.0], width_m=width, axis_angle=0.0, wall_thickness_m=0.25, host_wall_id="W001", sill_m=1.05, head_m=2.0,
        height_source="photo+points", mullion_count=1, mullion_u=[0.5], glazed=False, evidence="synthetic", confidence=0.8,
    )
    values.update(kwargs)
    return RefinedOpening(**values)


def _write(directory: Path, openings: list[RefinedOpening], walls_from=None) -> None:
    config = load_openings_config()
    plan = walls_from or make_plan(4.0)
    save_plan_model(plan, directory)
    inputs = {"walls": walls_fingerprint(wall_lines(plan, config)), "config_hash": config_hash(config)}
    save_openings_3d(OpeningsResult(openings=openings, inputs=inputs, mode="full"), directory)


FACE = [LineString([(0.0, -0.125), (4.0, -0.125)])]  # measured wall face 12.5 cm in front of the wall centreline


def test_apply_replaces_openings_and_snaps_them_to_the_face(tmp_path: Path) -> None:
    _write(tmp_path, [_refined(), _refined("door", 3.2, 0.9, id="C002", sill_m=0.0, head_m=2.13, mullion_count=0, mullion_u=[], glazed=True)])
    two_d = [Opening("WINDOW", (2.0, -0.125), 1.0, 0.0, 0.25)]
    replaced, info = apply_refined_openings(two_d, tmp_path, FACE, 1.0)
    assert info["applied"] is True and info["count"] == 2 and info["dropped"] == 0
    assert info["sha256"] == openings_3d_sha256(tmp_path)
    window, door = replaced
    assert (window.kind, door.kind) == ("WINDOW", "DOOR")
    assert window.width == pytest.approx(1.2) and (window.sill_m, window.head_m) == (1.05, 2.0)
    assert window.mullion_u == (0.5,) and window.height_source == "photo+points"
    assert window.center == pytest.approx((2.0, -0.125))  # projected onto the measured face
    assert door.glazed is True and door.head_m == pytest.approx(2.13)
    assert door.leaf_detected is None  # leaf detection still runs after the hook


def test_apply_without_a_file_changes_nothing(tmp_path: Path) -> None:
    save_plan_model(make_plan(4.0), tmp_path)
    two_d = [Opening("WINDOW", (2.0, -0.125), 1.0, 0.0, 0.25)]
    replaced, info = apply_refined_openings(two_d, tmp_path, FACE, 1.0)
    assert replaced is two_d and info == {}


def test_openings_of_another_wall_set_are_not_applied(tmp_path: Path) -> None:
    _write(tmp_path, [_refined()])
    save_plan_model(make_plan(7.0), tmp_path)  # the level has since been re-planned with different walls
    assert load_matching_openings(tmp_path) is None
    two_d = [Opening("WINDOW", (2.0, -0.125), 1.0, 0.0, 0.25)]
    replaced, info = apply_refined_openings(two_d, tmp_path, FACE, 1.0)
    assert replaced is two_d
    assert info == {"sha256": openings_3d_sha256(tmp_path), "applied": False}


def test_refined_opening_without_a_wall_face_nearby_is_dropped(tmp_path: Path) -> None:
    stray = _refined(id="C002", width=0.5, center=[3.5, 1.0])  # 1.1 m away from every measured face line
    _write(tmp_path, [_refined(), stray])
    replaced, info = apply_refined_openings([], tmp_path, FACE, 1.0)
    assert [o.width for o in replaced] == [pytest.approx(1.2)]
    assert info["dropped"] == 1 and info["dropped_ids"] == ["C002"] and info["count"] == 1


def test_openings_file_with_wrong_schema_is_ignored(tmp_path: Path) -> None:
    _write(tmp_path, [_refined()])
    path = openings_3d_path(tmp_path)
    path.write_text(path.read_text().replace('"schema_version": 1', '"schema_version": 99', 1), encoding="utf-8")
    assert load_matching_openings(tmp_path) is None


# ---------------------------------------------------------------- DXF symbols


def _window_opening(**kwargs) -> Opening:
    return Opening("WINDOW", (2.0, 0.0), 1.2, 0.0, 0.25, sill_m=1.05, head_m=2.0, height_source="photo+points", mullion_u=(0.5,), confidence=0.8, **kwargs)


def test_window_symbol_has_sill_head_mullions_and_mullion_line() -> None:
    entities = opening_entities([_window_opening()])
    window = next(e for e in entities if e.get("type") == "insert")
    assert window["attributes"]["BRH"] == "1.05" and window["attributes"]["STH"] == "2.00" and window["attributes"]["MULLIONS"] == "1"
    bars = [e for e in entities if e.get("_mullion")]
    assert len(bars) == 1 and bars[0]["_opening_kind"] == "WINDOW"
    start, end = np.asarray(bars[0]["start"]), np.asarray(bars[0]["end"])
    assert start[0] == pytest.approx(1.4 + 0.5) and end[0] == pytest.approx(1.9)  # window starts at x = 1.4
    assert abs(end[1] - start[1]) == pytest.approx(0.25)  # across the wall thickness


def test_door_symbol_has_head_and_glazed_flag() -> None:
    door = Opening("DOOR", (1.0, 0.0), 0.9, 0.0, 0.25, leaf_detected=True, hinge_side="left", swing_side=1, head_m=2.13, glazed=True)
    insert = next(e for e in opening_entities([door]) if e.get("type") == "insert")
    assert insert["attributes"]["H"] == "2.13" and insert["attributes"]["GLAZED"] == "1"
    plain = Opening("DOOR", (1.0, 0.0), 0.9, 0.0, 0.25, leaf_detected=True, hinge_side="left", swing_side=1, head_m=2.05)
    attributes = next(e for e in opening_entities([plain]) if e.get("type") == "insert")["attributes"]
    assert attributes["H"] == "2.05" and "GLAZED" not in attributes
    jamb_door = Opening("DOOR", (1.0, 0.0), 0.9, 0.0, 0.25, leaf_detected=False, head_m=2.13)
    text = next(e for e in opening_entities([jamb_door]) if e.get("type") == "text")
    assert "H 2.13" in text["text"]


def test_unmeasured_windows_carry_the_bim_default_heights() -> None:
    attributes = next(e for e in opening_entities([Opening("WINDOW", (2.0, 0.0), 1.2, 0.0, 0.25)]) if e.get("type") == "insert")["attributes"]
    assert (attributes["BRH"], attributes["STH"], attributes["MULLIONS"]) == ("0.90", "2.10", "0")


# ---------------------------------------------------------------- full clean plan, twice


@pytest.fixture(scope="module")
def two_passes(request: pytest.FixtureRequest):
    root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    base = Path(root) / f"openings_apply_{uuid.uuid4().hex[:8]}" if root else request.getfixturevalue("tmp_path_factory").mktemp("openings_apply")
    base.mkdir(parents=True, exist_ok=True)
    project = base / "indoor"
    level, _room = scene_tests._build_indoor_wall_face_scene(project, units_per_metre=0.8)
    first = generate_clean_plan(project, level=level, mode="indoor")
    directory = clean_plan_cache_dir(project, level)
    plan1 = load_plan_model(directory)
    config = load_openings_config()
    # a "refinement" of every pass-1 opening: new heights; the first window gets two panes
    refined = []
    for index, opening in enumerate(plan1.openings, start=1):
        window = opening.kind == "window"
        refined.append(RefinedOpening(
            id=f"C{index:03d}", kind=opening.kind, center=list(opening.center), width_m=opening.width_m, axis_angle=opening.axis_angle, wall_thickness_m=opening.wall_thickness_m,
            host_wall_id=opening.host_wall_id, sill_m=1.05 if window else 0.0, head_m=2.0 if window else 2.15, height_source="photo+points",
            mullion_count=1 if window else 0, mullion_u=[opening.width_m / 2.0 - 0.1] if window else [], glazed=opening.kind == "door" and index == 1, evidence="test", confidence=0.8,
        ))
    inputs = pipeline.current_inputs(plan1, wall_lines(plan1, config), project, level.id, config, pipeline.resolve_point_source(project, level.id))
    save_openings_3d(OpeningsResult(openings=refined, inputs=inputs, mode="full"), directory)
    second = generate_clean_plan(project, level=level, mode="indoor")
    return project, level, first, second, plan1, load_plan_model(directory), refined


def test_second_pass_uses_the_refined_openings(two_passes) -> None:
    _project, _level, first, second, plan1, plan2, refined = two_passes
    assert len(plan1.openings) >= 2 and {o.kind for o in plan1.openings} >= {"door", "window"}
    assert all(o.height_source == "default" for o in plan1.openings)
    assert len(plan2.openings) == len(refined)
    windows = [o for o in plan2.openings if o.kind == "window"]
    assert windows and all((o.sill_m, o.head_m, o.mullion_count) == (1.05, 2.0, 1) for o in windows)
    assert all(o.height_source == "photo+points" and o.confidence == pytest.approx(0.8) for o in plan2.openings)
    doors = [o for o in plan2.openings if o.kind == "door"]
    assert doors and all(o.head_m == pytest.approx(2.15) for o in doors)
    assert any(o.glazed for o in doors)
    assert [len(o.mullion_u) for o in windows] == [1] * len(windows)


def test_dxf_window_and_door_inserts_carry_the_measured_data(two_passes, tmp_path: Path) -> None:
    _project, _level, _first, second, _plan1, plan2, _refined_list = two_passes
    wind = second.layers["A-WIND"]
    inserts = [e for e in wind if e.get("type") == "insert"]
    assert inserts and all(e["attributes"]["BRH"] == "1.05" and e["attributes"]["STH"] == "2.00" and e["attributes"]["MULLIONS"] == "1" for e in inserts)
    assert len([e for e in wind if e.get("_mullion")]) == len(inserts)
    document = build_dxf_document(second)
    references = [r for r in document.modelspace().query("INSERT") if r.dxf.name == "WINDOW"]
    assert len(references) == len(inserts)
    for reference in references:
        values = {a.dxf.tag: a.dxf.text for a in reference.attribs}
        assert values["BRH"] == "1.05" and values["STH"] == "2.00" and values["MULLIONS"] == "1"
    door_refs = [r for r in document.modelspace().query("INSERT") if r.dxf.name == "DOOR"]
    for reference in door_refs:
        values = {a.dxf.tag: a.dxf.text for a in reference.attribs}
        assert values["H"] == "2.15"
    door_texts = [e for e in second.layers["A-DOOR"] if e.get("type") == "text"]
    assert all("H 2.15" in e["text"] for e in door_texts)
    mullion_lines = [e for e in document.modelspace().query("LINE") if e.dxf.layer == "A-WIND"]
    assert len(mullion_lines) >= len(inserts)


def test_plan_records_which_openings_file_it_used_and_does_not_loop(two_passes) -> None:
    project, level, _first, _second, plan1, plan2, _refined_list = two_passes
    directory = clean_plan_cache_dir(project, level)
    used = plan2.source["openings_3d"]
    assert used["applied"] is True and used["sha256"] == openings_3d_sha256(directory) and used["count"] == len(plan2.openings)
    assert "openings_3d" not in plan1.source
    assert pipeline.plan_has_current_openings(project, level.id)
    # the refinement fingerprint only contains the walls: the plan with the refined openings is still "fresh"
    assert pipeline.refine_status(project, level.id) == "fresh"


def test_plan_with_stale_openings_file_keeps_2d_openings_and_still_records_the_file(tmp_path: Path) -> None:
    project = tmp_path / "indoor"
    level, _ = scene_tests._build_indoor_wall_face_scene(project, units_per_metre=0.8)
    generate_clean_plan(project, level=level, mode="indoor")
    directory = clean_plan_cache_dir(project, level)
    plan1 = load_plan_model(directory)
    other = make_plan(9.0)  # fingerprint of a different wall set
    config = load_openings_config()
    save_openings_3d(OpeningsResult(openings=[_refined(centre_x=1.0)], inputs={"walls": walls_fingerprint(wall_lines(other, config))}, mode="full"), directory)
    assert not pipeline.plan_has_current_openings(project, level.id)
    generate_clean_plan(project, level=level, mode="indoor")
    plan2 = load_plan_model(directory)
    assert len(plan2.openings) == len(plan1.openings)
    assert plan2.source["openings_3d"]["applied"] is False
    assert pipeline.plan_has_current_openings(project, level.id)  # recorded: run_all will not re-plan forever
