"""The "no evidence = nothing" rule: add_furniture needs point-cloud evidence unless the user forces a manual item."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from assistant_synthetic import CHAIR_BOX, EMPTY_BOX, FLOOR_Z, LEVEL_ID, TABLE_BOX, make_plan, make_project
from samcloud.assistant.edits import EditError, EditOp, Proposal
from samcloud.assistant.evidence import FURNITURE_KINDS, MANUAL_LAYER, check_evidence, resolve_kind
from samcloud.assistant.pointcloud import ProjectCloud

CHAIR = {"x": [CHAIR_BOX[0] - 0.05, CHAIR_BOX[1] + 0.05], "y": [CHAIR_BOX[2] - 0.05, CHAIR_BOX[3] + 0.05]}


@pytest.fixture(scope="module")
def cloud(tmp_path_factory: pytest.TempPathFactory):
    return ProjectCloud.load(make_project(tmp_path_factory.mktemp("assistant_evidence")), LEVEL_ID)


def test_check_evidence(cloud) -> None:
    found = check_evidence(cloud, CHAIR, "chair")
    assert found.ok and found.points > 500 and found.classes["furniture"] > 500
    assert found.height_spread_m > 0.3
    missing = check_evidence(cloud, EMPTY_BOX, "chair")
    assert not missing.ok and missing.points == 0
    assert not check_evidence(cloud, CHAIR, "toilet").ok  # wrong class: furniture points are no sanitary evidence
    assert not check_evidence(None, CHAIR, "chair").ok
    assert not check_evidence(cloud, CHAIR, "warp drive").ok


def test_add_without_evidence_is_refused(cloud) -> None:
    proposal = Proposal(make_plan())
    with pytest.raises(EditError) as raised:
        proposal.apply(EditOp("add_furniture", None, {"box": EMPTY_BOX, "kind": "chair"}), cloud=cloud)
    assert raised.value.key == "assistant.edit.no_evidence"
    assert raised.value.params["points"] == 0
    with pytest.raises(EditError) as raised:
        proposal.apply(EditOp("add_furniture", None, {"box": CHAIR, "kind": "chair"}), cloud=None)
    assert raised.value.key == "assistant.edit.cloud_required"
    assert proposal.ops == [] and len(proposal.plan.furniture) == 1


def test_manual_item_is_flagged(cloud) -> None:
    proposal = Proposal(make_plan())
    changes = proposal.apply(EditOp("add_furniture", None, {"box": EMPTY_BOX, "kind": "Stuhl", "manual": True}), cloud=cloud)
    item = proposal.plan.furniture[-1]
    assert changes[0].change == "added" and changes[0].element_id == item.id == "F002"
    assert item.fit["evidence"] == "manual" and item.layer == MANUAL_LAYER
    assert (item.label, item.semantic_label) == ("Stuhl", "chair")
    polygon = Polygon(item.footprint)
    assert polygon.centroid.coords[0] == pytest.approx((7.0, 4.0))
    assert sorted(_sides(polygon)) == pytest.approx(sorted(FURNITURE_KINDS["chair"].size_m[:2]))
    assert item.default_height_m == pytest.approx(FURNITURE_KINDS["chair"].size_m[2])
    assert any("manually" in note for note in proposal.notes)
    # the stored item re-applies deterministically (history), without cloud
    op = proposal.ops[-1]
    assert op.params["_item"]["fit"]["evidence"] == "manual" and op.params["_created_ids"] == ["F002"]
    again = Proposal(make_plan())
    again.apply(op, replay=True)
    assert again.plan.furniture[-1].footprint == item.footprint


def test_forged_internal_item_cannot_bypass_the_evidence(cloud) -> None:
    proposal = Proposal(make_plan())
    proposal.apply(EditOp("add_furniture", None, {"box": EMPTY_BOX, "kind": "table", "manual": True}))
    forged = dict(proposal.ops[-1].params["_item"], fit={"added": "assistant", "evidence": "measured"}, layer="A-FURN")
    for params in ({"box": EMPTY_BOX, "kind": "table", "_item": forged}, {"box": EMPTY_BOX, "kind": "table", "_item": forged, "_created_ids": ["F009"]}):
        attempt = Proposal(make_plan())
        with pytest.raises(EditError) as raised:
            attempt.apply(EditOp("add_furniture", None, params), cloud=cloud)
        assert raised.value.key == "assistant.edit.no_evidence"
        with pytest.raises(EditError) as raised:
            attempt.apply({"op": "add_furniture", "target": None, "params": params}, cloud=None)
        assert raised.value.key == "assistant.edit.cloud_required"
        assert len(attempt.plan.furniture) == 1


def test_measured_chair_dimensions(cloud) -> None:
    proposal = Proposal(make_plan())
    proposal.apply(EditOp("add_furniture", None, {"box": CHAIR, "kind": "chair", "manual": True}), cloud=cloud)
    item = proposal.plan.furniture[-1]
    assert item.fit["evidence"] == "measured" and item.fit["added"] == "assistant"
    assert item.layer == "A-FURN" and item.label == "Stuhl"
    assert item.instance_id == 9
    sides = sorted(_sides(Polygon(item.footprint)))
    assert sides == pytest.approx([0.45, 0.50], abs=0.06)
    assert (item.z_max - FLOOR_Z) == pytest.approx(0.92, abs=0.05)
    assert any("measured instead of manual" in note for note in proposal.notes)


def test_measured_table_top(cloud) -> None:
    x0, x1, y0, y1 = TABLE_BOX
    plan = make_plan()
    plan.furniture = []
    proposal = Proposal(plan)
    proposal.apply(EditOp("add_furniture", None, {"box": {"x": [x0 - 0.05, x1 + 0.05], "y": [y0 - 0.05, y1 + 0.05]}, "kind": "table"}), cloud=cloud)
    item = proposal.plan.furniture[-1]
    assert item.label == "Tisch"
    assert sorted(_sides(Polygon(item.footprint))) == pytest.approx([0.8, 1.2], abs=0.06)
    assert (item.z_max - FLOOR_Z) == pytest.approx(0.75, abs=0.04)


def test_kind_table_matches_the_ifc_mapping() -> None:
    from samcloud.bim.furniture import load_furniture_mapping

    mapping = load_furniture_mapping()
    expected = {"chair": ("IfcFurniture", "CHAIR"), "table": ("IfcFurniture", "TABLE"), "bed": ("IfcFurniture", "BED"),
                "toilet": ("IfcSanitaryTerminal", "TOILETPAN"), "refrigerator": ("IfcElectricAppliance", "REFRIGERATOR")}
    for key, (ifc_class, predefined) in expected.items():
        kind = FURNITURE_KINDS[key]
        klass, _object_type = mapping.resolve(label=kind.label, semantic_label=kind.semantic_label, catalogue_kind=kind.catalogue_kind, layer=kind.layer)
        assert (klass.ifc_class, klass.predefined_type) == (ifc_class, predefined)
    assert resolve_kind("Schreibtisch").key == "desk" and resolve_kind("couch").key == "sofa" and resolve_kind("nothing") is None
    for kind in FURNITURE_KINDS.values():
        assert "_" not in kind.label  # display labels, never internal keys like tall_unit


def _sides(polygon: Polygon) -> list[float]:
    corners = list(polygon.minimum_rotated_rectangle.exterior.coords)
    return [((corners[i + 1][0] - corners[i][0]) ** 2 + (corners[i + 1][1] - corners[i][1]) ** 2) ** 0.5 for i in range(2)]
