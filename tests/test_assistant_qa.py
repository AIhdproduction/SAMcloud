"""QA of assistant proposals: valid edits pass against the baseline, the IFC agrees with the plan, new findings fail."""

from __future__ import annotations

import copy

import ifcopenshell
import ifcopenshell.util.element as element_util
import pytest

from assistant_synthetic import EMPTY_BOX, make_plan
from samcloud.assistant.edits import EditOp, Proposal
from samcloud.assistant.qa import finding_key, run_qa
from samcloud.assistant.selection import global_id_for
from samcloud.bim.scene_builder import build_scene


@pytest.fixture(scope="module")
def work(tmp_path_factory: pytest.TempPathFactory):
    return tmp_path_factory.mktemp("assistant_qa")


def _opening_width(model, name: str) -> float:
    return float(next(e for e in model.by_type("IfcDoor") + model.by_type("IfcWindow") if e.Name == name).OverallWidth)


def test_valid_edits_pass_and_ifc_matches_the_plan(work) -> None:
    base = make_plan()
    proposal = Proposal(base)
    proposal.apply(EditOp("move_wall", "W005", {"offset_m": 0.4}))
    proposal.apply(EditOp("set_wall_thickness", "W005", {"thickness_m": 0.25}))
    proposal.apply(EditOp("set_opening", "O001", {"width_m": 1.0}))
    proposal.apply(EditOp("set_opening", "O002", {"width_m": 1.5, "sill_m": 0.8}))
    proposal.apply(EditOp("move_furniture", "F001", {"dx_m": 0.3, "dy_m": 0.5}))
    proposal.apply(EditOp("add_furniture", None, {"box": EMPTY_BOX, "kind": "chair", "manual": True}))
    proposal.apply(EditOp("rename_room", "R002", {"name": "Gast"}))
    result = run_qa(proposal.plan, baseline=base, work_dir=work / "valid")
    assert result.ok, result.new_findings
    assert result.ifc_path is not None and result.ifc_path.is_file()
    model = ifcopenshell.open(str(result.ifc_path))
    assert _opening_width(model, "Door O001") == pytest.approx(1.0)
    assert _opening_width(model, "Window O002") == pytest.approx(1.5)
    partition = model.by_guid(global_id_for(proposal.plan, "wall", "W005"))
    quantities = element_util.get_pset(partition, "Qto_WallBaseQuantities") or {}
    assert quantities.get("Width") == pytest.approx(0.25)
    spaces = {s.Name for s in model.by_type("IfcSpace")}
    assert spaces == {"Wohnen", "Gast"}
    manual = [f for f in model.by_type("IfcFurniture") if element_util.get_pset(f, "SAMcloud_Scan").get("SAMcloud_Evidence") == "manual"]
    assert [f.Name for f in manual] == ["Stuhl 01"]
    table = next(f for f in model.by_type("IfcFurniture") if f.Name == "Tisch 01")
    assert "SAMcloud_Evidence" not in element_util.get_pset(table, "SAMcloud_Scan")


def test_single_op_qa_and_new_findings(work) -> None:
    base = make_plan()
    for op in (EditOp("delete_wall", "W005", {}), EditOp("rotate_furniture", "F001", {"angle_deg": 30.0})):
        proposal = Proposal(base)
        proposal.apply(op)
        assert run_qa(proposal.plan, baseline=base, work_dir=work / op.op).ok
    broken = copy.deepcopy(base)
    broken.openings[0].center = [5.5, 2.5]  # a door 1.5 m away from its host wall
    result = run_qa(broken, baseline=base, work_dir=work / "broken")
    assert not result.ok
    assert any(item["check"] == "opening_off_wall" and item["subject"] == "O001" for item in result.new_findings)
    # the same finding already in the baseline does not count
    again = run_qa(copy.deepcopy(broken), baseline=broken, work_dir=work / "baseline_finding")
    assert again.ok and any(finding_key(item) == "opening_off_wall|O001" for item in again.findings)


def test_plans_without_manual_items_carry_no_evidence_flag() -> None:
    scene = build_scene(make_plan())
    for element in scene.elements.values():
        extra = getattr(getattr(element, "params", None), "model_extra", None) or {}
        assert "evidence" not in extra


@pytest.mark.parametrize(
    "op",
    [
        EditOp("move_wall", "W005", {"offset_m": -0.35}),
        EditOp("move_wall", "W002", {"offset_m": -0.4}),
        EditOp("set_wall_thickness", "W005", {"thickness_m": 0.12, "anchor": "face_b"}),
        EditOp("set_opening", "O001", {"shift_m": -0.8, "head_m": 2.2}),
        EditOp("delete_opening", "O002", {}),
        EditOp("move_furniture", "F001", {"dx_m": 4.5, "dy_m": 2.0}),
        EditOp("delete_furniture", "F001", {}),
        EditOp("set_furniture_class", "F001", {"kind": "desk"}),
    ],
    ids=lambda op: op.op,
)
def test_every_op_passes_qa(work, op: EditOp) -> None:
    base = make_plan()
    proposal = Proposal(base)
    proposal.apply(op)
    result = run_qa(proposal.plan, baseline=base, work_dir=work / f"op_{op.op}_{op.target}")
    assert result.ok, result.new_findings
