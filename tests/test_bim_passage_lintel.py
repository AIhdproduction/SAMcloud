"""A passage is open up to the ceiling (no lintel above it); a door keeps its head; the QA reports the door invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from samcloud.bim.export import export_level
from samcloud.bim.plan_model import BuildingPlan, PlanOpening, PlanWall, save_plan_model
from samcloud.bim.scene_builder import build_scene
import json


def _plan(openings, upm=1.0):
    walls = [PlanWall(id="W001", centreline=[[0.0, 0.0], [8.0 * upm, 0.0]], thickness_m=0.2, kind="interior"), PlanWall(id="W002", centreline=[[0.0, 3.0 * upm], [8.0 * upm, 3.0 * upm]], thickness_m=0.2, kind="interior")]
    frame = {"origin": [0, 0, 0], "scale_factor": 1.0, "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    return BuildingPlan("lintel", "level_1", "1", upm, frame, 0.0, 2.6 * upm, walls=walls, openings=openings)


def _heights(plan):
    scene = build_scene(plan)
    return {e.tag: float(e.params.height) for e in scene.elements.values() if type(e).__name__ == "OpeningElement"}


def test_passage_has_no_lintel_but_door_keeps_its_head():
    passage = PlanOpening("O001", "passage", [2.0, 0.0], 0.9, 0.0, 0.2, sill_m=0.0, head_m=2.0, height_source="default")
    door = PlanOpening("O002", "door", [5.0, 0.0], 0.9, 0.0, 0.2, sill_m=0.0, head_m=2.05, height_source="points", evidence="backing=lintel", confidence=0.9)
    heights = _heights(_plan([passage, door]))
    assert heights["O001"] == pytest.approx(2.6, abs=0.01)  # up to the ceiling
    assert heights["O002"] == pytest.approx(2.05, abs=0.01)


def test_qa_contains_door_invariants(tmp_path: Path):
    door = PlanOpening("O002", "door", [5.0, 0.0], 0.9, 0.0, 0.2, sill_m=0.0, head_m=2.05, height_source="points", evidence="backing=frame", confidence=0.9, host_wall_id="W001")
    project = tmp_path / "proj"
    save_plan_model(_plan([door]), project / "products" / "dxf" / "level_1" / "ortho")
    result = export_level(project, "level_1")
    qa = json.loads(result.qa_json_path.read_text(encoding="utf-8"))
    assert qa["door_invariants"] == {"ok": True, "violations": []}
