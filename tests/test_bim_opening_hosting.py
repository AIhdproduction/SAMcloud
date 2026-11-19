"""Every opening of the plan model must reach the IFC: lintel hosts, axis override, join-cut shift, QA check."""

from __future__ import annotations

import json
import math
from pathlib import Path

import ifcopenshell
import pytest

from samcloud.bim.export import check_openings_consistency, export_level, ifc_opening_ids
from samcloud.bim.plan_model import BuildingPlan, PlanOpening, PlanRoom, PlanWall, save_plan_model
from samcloud.bim.scene_builder import build_scene


def _wall(wall_id: str, a: tuple[float, float], b: tuple[float, float], thickness: float = 0.2, kind: str = "interior") -> PlanWall:
    return PlanWall(id=wall_id, centreline=[list(a), list(b)], thickness_m=thickness, kind=kind)


def _plan(walls: list[PlanWall], openings: list[PlanOpening], rooms: list[PlanRoom] | None = None) -> BuildingPlan:
    return BuildingPlan(
        project_name="hosting", level_id="level_1", level_name="1", units_per_metre=1.0,
        frame={"origin": [0, 0, 0], "scale_factor": 1.0, "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        floor_z=0.0, ceiling_z=2.6, walls=walls, rooms=rooms or [], openings=openings,
    )


def _door(opening_id: str, centre: tuple[float, float], width: float, angle: float, thickness: float = 0.2) -> PlanOpening:
    return PlanOpening(opening_id, "door", [centre[0], centre[1]], width, angle, thickness, sill_m=0.0, head_m=2.05)


def _hosted_ids(scene) -> set[str]:
    return {e.tag for e in scene.elements.values() if type(e).__name__ == "OpeningElement"}


def _export(tmp_path: Path, plan: BuildingPlan):
    project = tmp_path / "proj"
    save_plan_model(plan, project / "products" / "dxf" / "level_1" / "ortho")
    return export_level(project, "level_1")


def test_bridge_that_would_cross_a_wall_gets_a_lintel_host(tmp_path: Path) -> None:
    # two collinear pieces with a 0.9 m gap; a stub wall ends inside the gap, so bridging the pieces would cross it
    walls = [
        _wall("W001", (0, 0), (4.0, 0)),
        _wall("W002", (4.9, 0), (9, 0)),
        _wall("W003", (4.45, -3), (4.45, 0.02)),
    ]
    plan = _plan(walls, [_door("O001", (4.45, 0.0), 0.9, 0.0)])
    scene = build_scene(plan)
    report = scene.derived["samcloud_build"]

    assert report["openings_hosted"] == 1 and report["openings_skipped"] == 0
    assert report["openings_lintel"] == 1
    assert any("lintel" in w for w in report["warnings"])
    opening = next(e for e in scene.elements.values() if type(e).__name__ == "OpeningElement")
    assert opening.params.width == pytest.approx(0.9)

    result = _export(tmp_path, plan)
    assert ifc_opening_ids(result.ifc_path) == {"O001": "door"}
    assert result.openings_consistency["ok"] is True and result.openings_consistency_errors == []
    assert result.counts["IfcDoor"] == 1


def test_perpendicular_axis_trusts_the_host_wall_direction(tmp_path: Path) -> None:
    walls = [_wall("W001", (0, 0), (6, 0)), _wall("W002", (0, 0), (0, 4))]
    plan = _plan(walls, [_door("O001", (3.0, 0.0), 0.9, math.pi / 2)])  # axis vertical, wall horizontal
    scene = build_scene(plan)
    report = scene.derived["samcloud_build"]

    assert report["openings_hosted"] == 1 and report["openings_axis_overridden"] == 1
    assert any("axis disagrees" in w for w in report["warnings"])
    assert ifc_opening_ids(_export(tmp_path, plan).ifc_path) == {"O001": "door"}


def test_opening_in_a_join_cut_region_is_shifted_not_dropped(tmp_path: Path) -> None:
    # T join: the horizontal wall ends at the face of the vertical wall (cut 0.15 m), the door starts inside the cut
    walls = [_wall("W001", (0, 0), (6, 0), 0.3), _wall("W002", (0, -3), (0, 4), 0.3)]
    plan = _plan(walls, [_door("O001", (0.45, 0.0), 0.7, 0.0)])
    scene = build_scene(plan)
    report = scene.derived["samcloud_build"]

    assert report["openings_hosted"] == 1
    assert report["openings_shifted"] == 1
    assert any("shifted" in w and "O001" in w for w in report["warnings"])
    opening = next(e for e in scene.elements.values() if type(e).__name__ == "OpeningElement")
    assert 0.0 < opening.params.position_along_axis - 0.45 <= 0.10 + 1e-9
    result = _export(tmp_path, plan)
    assert not any("trimmed part of the wall" in w for w in result.warnings)
    assert result.openings_consistency["ok"] is True


def test_far_join_cut_is_kept_with_warning_not_dropped() -> None:
    walls = [_wall("W001", (0, 0), (6, 0), 0.2), _wall("W002", (0, -3), (0, 4), 0.8)]
    scene = build_scene(_plan(walls, [_door("O001", (0.5, 0.0), 0.7, 0.0)]))
    report = scene.derived["samcloud_build"]

    assert report["openings_hosted"] == 1 and report["openings_skipped"] == 0
    assert any("join-cut region" in w for w in report["warnings"])


def test_consistency_check_catches_a_removed_opening(tmp_path: Path) -> None:
    walls = [_wall("W001", (0, 0), (8, 0)), _wall("W002", (0, 3), (8, 3))]
    window = PlanOpening("O002", "window", [6.0, 0.0], 1.2, 0.0, 0.2, sill_m=0.9, head_m=2.1)
    plan = _plan(walls, [_door("O001", (2.0, 0.0), 0.9, 0.0), window])
    result = _export(tmp_path, plan)
    assert result.openings_consistency["plan"] == {"door": 1, "window": 1, "passage": 0}
    assert result.openings_consistency["ifc"] == {"door": 1, "window": 1, "passage": 0}

    model = ifcopenshell.open(str(result.ifc_path))
    door = model.by_type("IfcDoor")[0]
    model.remove(door)
    damaged = tmp_path / "damaged.ifc"
    model.write(str(damaged))
    check = check_openings_consistency(plan, damaged)

    assert check["ok"] is False
    assert [(e["id"], e["kind"]) for e in check["errors"]] == [("O001", "door")]
    assert check["ifc"]["door"] == 0 and check["plan"]["door"] == 1


def test_skipped_opening_is_an_error_with_its_reason_in_qa_and_result(tmp_path: Path) -> None:
    walls = [_wall("W001", (0, 0), (8, 0))]
    tiny = _door("O009", (4.0, 0.0), 0.1, 0.0)
    result = _export(tmp_path, _plan(walls, [_door("O001", (2.0, 0.0), 0.9, 0.0), tiny]))

    assert result.openings_consistency["ok"] is False
    assert [e["id"] for e in result.openings_consistency_errors] == ["O009"]
    assert "width" in result.openings_consistency_errors[0]["reason"]
    assert any(w.startswith("openings_consistency:") and "O009" in w for w in result.warnings)
    qa = json.loads(result.qa_json_path.read_text(encoding="utf-8"))
    assert qa["openings_consistency"]["errors"][0]["id"] == "O009"
    assert "O009" in result.qa_html_path.read_text(encoding="utf-8")
    assert "ERRORS" in qa["summary"]


def _lintel_corner_plan() -> BuildingPlan:
    # real-project corner (v11, plan units 0.9887 per metre): a door whose lintel host ends 0.2 m short of a
    # thick wall, a thin wall that ends in the corner of that wall and a perpendicular one; the lintel and
    # the thin wall used to be joined as an L whose corner lies inside the thick wall (bodies stacked on it)
    walls = [
        _wall("W006", (5.8856, 10.9698), (6.8174, 10.9722), 0.2583, "exterior"),
        _wall("W012", (6.1027, 12.2078), (7.3872, 12.2112), 0.2583, "exterior"),
        _wall("W026", (5.9786, 12.0798), (5.9732, 14.1231), 0.2583, "exterior"),
        _wall("W027", (5.4983, 12.2818), (6.1057, 12.2834), 0.09),
    ]
    plan = _plan(walls, [_door("O007", (6.3588, 11.5783), 0.85, 1.5734, thickness=0.3)])
    plan.units_per_metre = 0.9887072043809466
    return plan


def test_lintel_host_does_not_overlap_regular_walls() -> None:
    scene = build_scene(_lintel_corner_plan())
    report = scene.derived["samcloud_build"]

    assert report["openings_lintel"] == 1 and report["openings_hosted"] == 1 and report["openings_skipped"] == 0
    assert report["wall_overlaps"] == []
    assert _hosted_ids(scene) == {"O007"}
