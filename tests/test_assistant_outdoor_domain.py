"""Outdoor assistant domain: registry, selection (hit test, IFC ids, chips), read tools, evidence rule and the manual marking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from assistant_outdoor_synthetic import (
    EMPTY_POINT, EXTRA_MANHOLE, EXTRA_TREE, FAKE_ORIGIN, FENCE_AXIS, MANHOLE, TREE_1, TREE_2, lv95, shared_project,
    terrain_z,
)
from samcloud.assistant.domains.outdoor import OutdoorDomain
from samcloud.assistant.domains.outdoor_selection import global_id_of
from samcloud.assistant.domains.registry import domain_for
from samcloud.assistant.edits import EditError, EditOp
from samcloud.assistant.providers.base import ToolCall
from samcloud.assistant.tools import dispatch, tool_names


@pytest.fixture(scope="module")
def domain() -> OutdoorDomain:
    return OutdoorDomain(shared_project(), "site", mode="outdoor")


@pytest.fixture(scope="module")
def base(domain: OutdoorDomain) -> Any:
    return domain.load_base()


def _ctx(domain: OutdoorDomain, base: Any, *, allow_manual: bool = False, tier: str = "elements") -> Any:
    return domain.tool_context(base, allow_manual=allow_manual, tier=tier)


def _call(ctx: Any, name: str, **arguments: Any) -> dict[str, Any]:
    result = dispatch(ctx, ToolCall("c1", name, arguments))
    payload = json.loads(result.content)
    payload["_error"] = result.error
    payload["_key"] = result.error_key
    return payload


# ---------------------------------------------------------------------------------------------------------- registry

def test_registry_returns_the_outdoor_domain(domain: OutdoorDomain) -> None:
    found = domain_for(shared_project(), "site", mode="outdoor")
    assert isinstance(found, OutdoorDomain) and found.availability() == (True, None)
    assert found.supported_tiers == ("elements", "full")
    frame = found.view_frame()
    assert frame["origin_offset"] == list(FAKE_ORIGIN) and frame["crs"]["epsg"] == 2056 and frame["units_per_metre"] == 1.0
    assert found.export_actions() == ["env:scene,export,qa:cad_only"]


def test_base_model_matches_the_history_model_file(domain: OutdoorDomain, base: Any) -> None:
    from samcloud.assistant.history import History

    backend = domain.history_backend()
    history = History(shared_project(), "site", backend)
    assert history.plan_path.name == "scene.json" and history.plan_path.is_file()
    assert backend.semantic_hash(history.current_plan()) == backend.semantic_hash(base)
    assert backend.base_filename == "scene.base.json"


# --------------------------------------------------------------------------------------------------------- selection

def test_hit_test_prefers_manhole_tree_line_building(domain: OutdoorDomain, base: Any) -> None:
    tree = domain.hit_test(TREE_1["xy"][0] + 1.0, TREE_1["xy"][1], model=base)
    assert (tree["type"], tree["kind"], tree["id"]) == ("element", "tree", TREE_1["id"])
    manhole = domain.hit_test(*MANHOLE["xy"], model=base)
    assert (manhole["kind"], manhole["id"]) == ("manhole", "manhole-0")  # a manhole in the road wins over the pavement
    fence = domain.hit_test(FENCE_AXIS[1][0] + 0.2, FENCE_AXIS[1][1], model=base)
    assert (fence["kind"], fence["id"]) == ("fence", "fence-0")
    assert domain.hit_test(26.0, 13.0, model=base)["kind"] == "building"
    assert domain.hit_test(30.0, 3.0, model=base)["kind"] == "pavement"
    assert domain.hit_test(*EMPTY_POINT, model=base) == {"type": "point", "x": EMPTY_POINT[0], "y": EMPTY_POINT[1]}


def test_global_ids_chips_outlines(domain: OutdoorDomain, base: Any) -> None:
    sel = domain.selection_from_global_id(global_id_of(TREE_2["id"]), model=base)
    assert sel["id"] == TREE_2["id"] and sel["kind"] == "tree"
    assert domain.selection_from_global_id(global_id_of("curb-0-gutter"), model=base)["id"] == "curb-0"
    assert domain.selection_from_global_id("unknown", [1.0, 2.0, 3.0], model=base) == {"type": "point", "x": 1.0, "y": 2.0}
    assert domain.selection_from_global_id("unknown", model=base)["type"] == "unknown"
    assert domain.global_ids_for([sel, {"type": "point", "x": 1, "y": 2}, {"type": "element", "id": "nope"}], model=base) == [global_id_of(TREE_2["id"])]
    assert sel["ref"] == "B1"  # short display label: trees numbered north to south (TREE_2 lies north of TREE_1)
    assert domain.chip_label(sel, "de").startswith("Baum B1 (Ellipsoid, ")
    assert domain.chip_label(sel, "en").startswith("Tree ")
    fence = domain.hit_test(FENCE_AXIS[0][0], FENCE_AXIS[0][1] + 1.0, model=base)
    assert domain.chip_label(fence, "de") == "Zaun Z1 (1.20 m)"
    point = domain.chip_label({"type": "point", "x": 1.0, "y": 2.0}, "en")
    assert point == f"Point E {FAKE_ORIGIN[0] + 1.0:.2f} / N {FAKE_ORIGIN[1] + 2.0:.2f}"
    rings = domain.outline(sel, model=base)
    assert len(rings) == 1 and len(rings[0]) >= 12
    z0, z1 = domain.z_range(sel, model=base)
    assert z1 - z0 == pytest.approx(base.elements[TREE_2["id"]].params.height, abs=1e-3) and z0 > FAKE_ORIGIN[2] - 5
    assert domain.outline({"type": "point", "x": 0, "y": 0}, model=base) == []
    assert domain.refresh_selection(sel, model=base)["id"] == TREE_2["id"]
    gone = base.model_copy(update={"elements": {k: v for k, v in base.elements.items() if k != TREE_2["id"]}})
    assert domain.refresh_selection(sel, model=gone) is None


def test_preview_and_context(domain: OutdoorDomain, base: Any) -> None:
    proposal = domain.new_proposal(base)
    proposal.apply(EditOp("set_tree", TREE_1["id"], {"height_delta": 1.0}))
    proposal.apply(EditOp("delete_fence", "fence-0", {}))
    preview = domain.preview(proposal)
    assert [item["id"] for item in preview["moved"]] == [TREE_1["id"]] and [item["id"] for item in preview["removed"]] == ["fence-0"]
    moved = preview["moved"][0]
    assert moved["global_id"] == global_id_of(TREE_1["id"]) and moved["before"] and moved["after"] and moved["z_max"] > moved["z_min"]
    assert preview["removed"][0]["after"] is None and preview["removed"][0]["before"]
    message = json.loads(domain.context_message(base, [{"type": "point", "x": 1.0, "y": 2.0}], "de"))
    assert message["project_type"] == "Aussenprojekt (Drohne)" and message["selection"][0]["e"] == pytest.approx(FAKE_ORIGIN[0] + 1.0)
    assert any(row["id"] == "fence-0" for row in message["elements"])
    vocabulary = domain.vocabulary("de")
    assert vocabulary["element_types"]["tree"] == "Baum" and "{tier}" in vocabulary["rules"]
    assert "LV95" in " ".join(domain.vocabulary("en")["rules"])


# ------------------------------------------------------------------------------------------------------- read tools

def test_read_tools(domain: OutdoorDomain, base: Any) -> None:
    ctx = _ctx(domain, base)
    assert set(tool_names(domain, "elements")) >= {"list_elements", "element_details", "points_in_box", "terrain_height", "profile", "measure_tree"}
    trees = _call(ctx, "list_elements", category="tree")
    assert trees["total"] == 2 and {row["id"] for row in trees["elements"]} == {TREE_1["id"], TREE_2["id"]}
    details = _call(ctx, "element_details", id="fence-0")
    assert details["found"] and details["axis"][0] == pytest.approx(list(lv95(*FENCE_AXIS[0])), abs=1e-3)
    assert _call(ctx, "element_details", id="nope")["_key"] == "assistant.edit.not_found"
    e, n = lv95(*TREE_1["xy"])
    box = _call(ctx, "points_in_box", box={"e": [e - 3, e + 3], "n": [n - 3, n + 3]})
    names = {row["class"] for row in box["classes"]}
    assert {"tree canopy", "tree trunk", "ground"} <= names and box["terrain_ln02"] > FAKE_ORIGIN[2] - 2
    empty_e, empty_n = lv95(*EMPTY_POINT)
    empty = _call(ctx, "points_in_box", box={"e": [empty_e - 1, empty_e + 1], "n": [empty_n - 1, empty_n + 1]})
    assert {row["class"] for row in empty["classes"]} == {"ground"}
    height = _call(ctx, "terrain_height", e=empty_e, n=empty_n)
    assert height["terrain_ln02"] == pytest.approx(FAKE_ORIGIN[2] + float(terrain_z(*EMPTY_POINT)), abs=0.08)
    profile = _call(ctx, "profile", polyline=[list(lv95(2.0, 15.0)), list(lv95(20.0, 15.0))], step_m=2.0)
    assert profile["length_m"] == pytest.approx(18.0) and len(profile["stations"]) == 10
    assert all(row["terrain_ln02"] is not None for row in profile["stations"])
    tree = _call(ctx, "measure_tree", e=e, n=n, radius_m=3.0)
    assert tree["evidence"] == "enough" and tree["height"] == pytest.approx(TREE_1["height"], abs=0.6)
    assert tree["crown_diameter"] == pytest.approx(TREE_1["crown"], abs=0.8)
    bad = _call(ctx, "points_in_box", box={"e": [e, e], "n": [n, n + 1]})
    assert bad["_key"] == "assistant.agent.bad_arguments"


# --------------------------------------------------------------------------------------------------------- evidence

def test_add_tree_needs_evidence_and_is_measured(domain: OutdoorDomain, base: Any) -> None:
    ctx = _ctx(domain, base)
    e, n = lv95(*EMPTY_POINT)
    failed = _call(ctx, "add_tree", point={"e": e, "n": n})
    assert failed["_key"] == "assistant.edit.no_evidence" and "hint" in failed
    assert _call(ctx, "add_tree", point={"e": e, "n": n}, manual=True)["_key"] == "assistant.edit.manual_not_allowed"
    assert not ctx.proposal.summary()
    e, n = lv95(*EXTRA_TREE["xy"])
    added = _call(ctx, "add_tree", point={"e": e, "n": n}, radius_m=3.0)
    assert added["ok"] and added["changes"][0]["change"] == "added"
    element = ctx.proposal.plan.elements[added["changes"][0]["element_id"]]
    p = element.params
    assert p.height == pytest.approx(EXTRA_TREE["height"], abs=0.6) and p.crown_diameter == pytest.approx(EXTRA_TREE["crown"], abs=0.8)
    assert p.trunk_xy == pytest.approx(EXTRA_TREE["xy"], abs=0.1) and element.provenance.value == "ai"
    assert element.param_origin["height"].value == "measured" and element.evidence["added"] == "assistant"
    assert "evidence" not in element.evidence  # measured, not manual
    stored = ctx.proposal.ops[-1].params
    assert stored["_element"]["id"] == element.id and stored["_created_ids"] == [element.id]
    e1, n1 = lv95(*TREE_1["xy"])
    assert _call(ctx, "add_tree", point={"e": e1, "n": n1})["_key"] == "assistant.outdoor.duplicate"


def test_manual_tree_and_manhole_are_marked(domain: OutdoorDomain, base: Any) -> None:
    ctx = _ctx(domain, base, allow_manual=True)
    e, n = lv95(*EMPTY_POINT)
    added = _call(ctx, "add_tree", point={"e": e, "n": n}, manual=True, height=6.0, crown_diameter=3.0)
    tree = ctx.proposal.plan.elements[added["changes"][0]["element_id"]]
    assert tree.evidence["evidence"] == "manual" and tree.provenance.value == "manual"
    assert tree.param_origin["height"].value == "assumed" and tree.param_origin["trunk_xy"].value == "assumed"
    assert tree.param_origin["crown_diameter"].value == "manual" and tree.params.height == 6.0
    assert tree.params.base_z == pytest.approx(float(terrain_z(*EMPTY_POINT)), abs=0.08)
    e, n = lv95(EMPTY_POINT[0], EMPTY_POINT[1] - 6.0)
    manhole = _call(ctx, "add_manhole", point={"e": e, "n": n}, manual=True, diameter=0.8)
    element = ctx.proposal.plan.elements[manhole["changes"][0]["element_id"]]
    assert element.evidence["evidence"] == "manual" and element.params.diameter == 0.8


def test_add_manhole_measured_or_no_evidence(domain: OutdoorDomain, base: Any) -> None:
    from samcloud.assistant.domains.outdoor_cloud import OutdoorCloud

    proposal = domain.new_proposal(base)
    cloud = OutdoorCloud.load(shared_project())
    e, n = lv95(*EXTRA_MANHOLE["xy"])
    changes = proposal.apply(EditOp("add_manhole", None, {"point": {"e": e, "n": n}}), cloud=cloud)
    manhole = proposal.plan.elements[changes[0].element_id]
    assert manhole.params.diameter == pytest.approx(EXTRA_MANHOLE["diameter"], abs=0.12)
    assert manhole.params.center[:2] == pytest.approx(EXTRA_MANHOLE["xy"], abs=0.08)
    assert manhole.params.center[2] == pytest.approx(float(terrain_z(*EXTRA_MANHOLE["xy"])), abs=0.03)
    e, n = lv95(*EMPTY_POINT)
    with pytest.raises(EditError) as error:
        proposal.apply(EditOp("add_manhole", None, {"point": {"e": e, "n": n}}), cloud=cloud)
    assert error.value.key == "assistant.edit.no_evidence"
    with pytest.raises(EditError) as error:
        proposal.apply(EditOp("add_manhole", None, {"point": {"e": e, "n": n}}), cloud=None)
    assert error.value.key == "assistant.edit.cloud_required"


def test_manual_marking_reaches_ifc_dxf_and_passes_the_qa(tmp_path: Path, domain: OutdoorDomain, base: Any) -> None:
    import ezdxf
    import ifcopenshell
    import ifcopenshell.util.element

    from samcloud.assistant.domains.outdoor_qa import cad_qa, compare, run_outdoor_qa

    proposal = domain.new_proposal(base)
    e, n = lv95(*EMPTY_POINT)
    tree_id = proposal.apply(EditOp("add_tree", None, {"point": {"e": e, "n": n}, "manual": True}))[0].element_id
    results, ifc_path, dxf_path = cad_qa(shared_project(), proposal.plan, tmp_path / "qa")
    doc = ezdxf.readfile(dxf_path)
    manual = [entity for entity in doc.modelspace() if entity.dxf.layer == "S-TREE-MANUAL"]
    assert manual and all(entity.dxftype() == "INSERT" for entity in manual)
    assert "S-TREE-MANUAL" in doc.layers and doc.layers.get("S-TREE-MANUAL").dxf.linetype == "DASHED"
    assert len([entity for entity in doc.modelspace().query('INSERT[name=="TREE"]') if entity.dxf.layer == "S-TREE"]) == 2
    model = ifcopenshell.open(str(ifc_path))  # keep the file alive while its entities are used
    product = model.by_guid(global_id_of(tree_id))
    psets = ifcopenshell.util.element.get_psets(product)
    assert psets["SAMcloud_Herkunft"]["SAMcloud_Evidence"] == "manual"
    other = model.by_guid(global_id_of(TREE_1["id"]))
    assert "SAMcloud_Evidence" not in ifcopenshell.util.element.get_psets(other)["SAMcloud_Herkunft"]
    rules = {result["rule"]: result for result in results}
    assert rules["1_block_inserts"]["ok"] and rules["15_count_tree"]["ok"] and rules["15_count_tree"]["value"]["dxf"] == 3
    qa = run_outdoor_qa(shared_project(), proposal.plan, base, work_dir=tmp_path / "qa_run")
    assert qa.ok and not qa.new_findings and qa.counts["rules"] == len(results)
    failing, new = compare([{"rule": "a", "ok": False}, {"rule": "b", "ok": False}, {"rule": "c", "ok": True}],
                           [{"rule": "a", "ok": False}, {"rule": "b", "ok": True}])
    assert [f["check"] for f in failing] == ["a", "b"] and [f["check"] for f in new] == ["b"]


def test_qa_without_checkpoints_blocks(tmp_path: Path, domain: OutdoorDomain, base: Any) -> None:
    from samcloud.assistant.domains.outdoor_qa import run_outdoor_qa

    qa = run_outdoor_qa(tmp_path / "missing", base, base, work_dir=tmp_path / "qa")
    assert not qa.ok and qa.new_findings[0]["check"] == "qa_failed"


def test_availability_without_scene(tmp_path: Path) -> None:
    project = tmp_path / "drone_without_scene"
    (project / "products" / "environment" / "cache").mkdir(parents=True)
    (project / "products" / "environment" / "state.json").write_text('{"steps": {}}', encoding="utf-8")
    domain = domain_for(project, "site")
    assert isinstance(domain, OutdoorDomain) and domain.availability() == (False, "assistant.agent.outdoor_no_scene")


def test_qa_gate_blocks_new_offenders_of_an_already_failing_rule() -> None:
    from samcloud.assistant.domains.outdoor_qa import compare, offender_key

    baseline = [
        {"rule": "1_block_inserts", "ok": False, "value": {}, "offenders": ["INSERT@S-TREE#1A2 missing attributes TRUNK_DIA"]},
        {"rule": "21_ifc_attributes_vs_dxf", "ok": False, "value": {},
         "offenders": ["IfcGeographicElement#537:Baum 1: Kronendurchmesser_m 4.2 vs DXF 4.19 (deviation 0.012 m)"]},
        {"rule": "15_count_tree", "ok": False, "value": {"scene": 6, "dxf": 5, "ifc": 6}, "offenders": ["dxf=5 (scene=6)"]},
        {"rule": "3_tin_only_on_tin_layer_frozen", "ok": True, "value": {}, "offenders": []},
    ]
    # same offenders (other handles / entity numbers / magnitudes), a delete that keeps the count gap: nothing new
    same = [
        {"rule": "1_block_inserts", "ok": False, "value": {}, "offenders": ["INSERT@S-TREE#2F0 missing attributes TRUNK_DIA"]},
        {"rule": "21_ifc_attributes_vs_dxf", "ok": False, "value": {},
         "offenders": ["IfcGeographicElement#612:Baum 1: Kronendurchmesser_m 4.3 vs DXF 4.28 (deviation 0.02 m)"]},
        {"rule": "15_count_tree", "ok": False, "value": {"scene": 5, "dxf": 4, "ifc": 5}, "offenders": ["dxf=4 (scene=5)"]},
        {"rule": "3_tin_only_on_tin_layer_frozen", "ok": True, "value": {}, "offenders": []},
    ]
    failing, new = compare(same, baseline)
    assert len(failing) == 3 and new == []
    assert offender_key("IfcGeographicElement#537:Baum 1: x 4.2 m") == offender_key("IfcGeographicElement#99:Baum 1: x 7.25 m")
    # an edit that adds an offender to a failing rule, widens a count gap or breaks a passing rule is blocked
    worse = [
        {"rule": "1_block_inserts", "ok": False, "value": {},
         "offenders": ["INSERT@S-TREE#2F0 missing attributes TRUNK_DIA", "INSERT@S-MANH#3A1 missing attributes TYP"]},
        {"rule": "21_ifc_attributes_vs_dxf", "ok": False, "value": {},
         "offenders": ["IfcGeographicElement#612:Baum 1: Kronendurchmesser_m 4.3 vs DXF 4.28 (deviation 0.02 m)",
                       "IfcGeographicElement#640:Baum 7: no matching DXF TREE insert within 1.0 m"]},
        {"rule": "15_count_tree", "ok": False, "value": {"scene": 7, "dxf": 5, "ifc": 7}, "offenders": ["dxf=5 (scene=7)"]},
        {"rule": "3_tin_only_on_tin_layer_frozen", "ok": False, "value": {}, "offenders": []},
    ]
    failing, new = compare(worse, baseline)
    by_rule = {finding["check"]: finding["detail"] for finding in new}
    assert set(by_rule) == {"1_block_inserts", "21_ifc_attributes_vs_dxf", "15_count_tree", "3_tin_only_on_tin_layer_frozen"}
    assert by_rule["1_block_inserts"]["new_offenders"] == ["INSERT@S-MANH#3A1 missing attributes TYP"]
    assert by_rule["21_ifc_attributes_vs_dxf"]["new_offenders"] == ["IfcGeographicElement#640:Baum 7: no matching DXF TREE insert within 1.0 m"]
    assert by_rule["15_count_tree"]["reason"] == "count gap grew" and by_rule["15_count_tree"]["new_offenders"] == ["dxf: 1 -> 2"]
    assert by_rule["3_tin_only_on_tin_layer_frozen"]["reason"] == "new failure"
