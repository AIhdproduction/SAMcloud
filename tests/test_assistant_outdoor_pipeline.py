"""Outdoor assistant in the environment chain: the ``_step_scene`` hook, re-apply after a recompute, ``cad_only`` and QA per op."""

from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path
from typing import Any

import ezdxf
import numpy as np
import pytest

from assistant_outdoor_synthetic import (
    EMPTY_POINT, TREE_1, TREE_2, copy_project, lv95, renumbered_recompute, run_scene_step, shared_project,
)
from samcloud.assistant.domains.outdoor import OutdoorDomain
from samcloud.assistant.domains.outdoor_selection import global_id_of
from samcloud.assistant.edits import EditOp
from samcloud.assistant.history import History
from samcloud.drone import run_environment as env
from samcloud.drone.contracts import load_drone_config


def _cache(project: Path) -> Path:
    return project / "products" / "environment" / "cache"


def _scene(project: Path) -> Any:
    with (_cache(project) / "scene.pkl").open("rb") as handle:
        return pickle.load(handle)["scene"]


def _write_cad(project: Path, scene: Any, dxf_result: Any, folder: Path) -> tuple[Path, Path]:
    from samcloud.assistant.domains.outdoor_qa import frame_of

    folder.mkdir(parents=True, exist_ok=True)
    ifc_path, dxf_path = folder / "outdoor.ifc", folder / "outdoor.dxf"
    env.write_cad_exports(scene, dxf_result, frame_of(scene), load_drone_config(), ifc_path, dxf_path)
    return ifc_path, dxf_path


def _stable_ifc(path: Path) -> str:
    return re.sub(r"FILE_NAME\(.*?\);", "FILE_NAME();", path.read_text(encoding="utf-8"), flags=re.S)


def _stable_dxf(path: Path) -> list[tuple[str, str, str]]:
    doc = ezdxf.readfile(path)
    rows = []
    for entity in doc.modelspace():
        if entity.dxftype() == "TEXT" and entity.dxf.layer == "X-NOTE":
            continue  # carries the export date
        rows.append((entity.dxftype(), entity.dxf.layer, str(entity.dxf.all_existing_dxf_attribs().get("name", "")),
                     json.dumps([round(v, 4) for v in _points(entity)])))
    return sorted(rows)


def _points(entity: Any) -> list[float]:
    kind = entity.dxftype()
    if kind == "LWPOLYLINE":
        return [c for p in entity.get_points("xy") for c in p] + [float(entity.dxf.elevation)]
    if kind == "POLYLINE":
        return [float(c) for v in entity.vertices for c in v.dxf.location]
    if kind == "INSERT":
        return [float(c) for c in entity.dxf.insert] + [float(entity.dxf.xscale)]
    if kind == "3DFACE":
        return [float(c) for i in range(4) for c in entity[i]]
    return []


def _accept(domain: OutdoorDomain, *ops: EditOp, allow_cloud: bool = False) -> str:
    base = domain.load_base()
    proposal = domain.new_proposal(base)
    cloud = domain.cloud_loader(base)() if allow_cloud else None
    for op in ops:
        proposal.apply(op, cloud=cloud)
    return History.load(domain.project, "site", domain.history_backend()).accept(proposal, prompt="synthetic", provider="fake", model="fake")


def _psets(ifc: Any, element_id: str) -> dict[str, Any]:
    import ifcopenshell.util.element

    product = ifc.by_guid(global_id_of(element_id))
    return ifcopenshell.util.element.get_psets(product)


def _tree_height_dxf(doc: Any, near_xy: tuple[float, float]) -> float:
    e, n = lv95(*near_xy)
    best = min(doc.modelspace().query('INSERT[name=="TREE"]'), key=lambda i: (i.dxf.insert.x - e) ** 2 + (i.dxf.insert.y - n) ** 2)
    text = {a.dxf.tag: a.dxf.text for a in best.attribs}["HEIGHT"]
    return float(re.search(r"-?\d+(\.\d+)?", text).group(0))


# ------------------------------------------------------------------------------------------------- without edits

def test_without_edits_the_scene_step_output_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = copy_project(tmp_path)
    cfg = load_drone_config()
    scene, dxf_result, qa = env.build_scene_outputs(project, cfg, _cache(project))
    assert not (project / "products" / "assistant" / "site" / "edits.json").exists()
    monkeypatch.setattr(env, "_assistant_scene_plan", lambda *args, **kwargs: None)  # hook-free reference
    ref_scene, ref_dxf, ref_qa = env.build_scene_outputs(project, cfg, _cache(project))
    assert scene.to_json() == ref_scene.to_json() and len(qa) == len(ref_qa)
    assert dxf_result.layers == ref_dxf.layers and dxf_result.blocks == ref_dxf.blocks
    ifc_a, dxf_a = _write_cad(project, scene, dxf_result, tmp_path / "a")
    ifc_b, dxf_b = _write_cad(project, ref_scene, ref_dxf, tmp_path / "b")
    assert _stable_ifc(ifc_a) == _stable_ifc(ifc_b)
    assert _stable_dxf(dxf_a) == _stable_dxf(dxf_b)
    assert "S-TREE-MANUAL" not in ezdxf.readfile(dxf_a).layers
    assert not (project / "products" / "assistant" / "site" / "scene.base.json").exists()


# --------------------------------------------------------------------------------------------------- with edits

def test_accepted_edits_reach_scene_dxf_and_ifc_and_survive_a_recompute(tmp_path: Path) -> None:
    import ifcopenshell

    project = copy_project(tmp_path)
    domain = OutdoorDomain(project, "site", mode="outdoor")
    base = domain.load_base()
    old_height = base.elements[TREE_1["id"]].params.height
    e, n = lv95(*EMPTY_POINT)
    tree_edit = _accept(domain, EditOp("set_tree", TREE_1["id"], {"height": old_height + 2.0}))
    fence_edit = _accept(domain, EditOp("delete_fence", "fence-0", {}))
    manual_edit = _accept(domain, EditOp("add_manhole", None, {"point": {"e": e, "n": n}, "manual": True, "diameter": 0.8}))
    curb_edit = _accept(domain, EditOp("set_curb", "curb-0", {"width": 0.16}))
    scene = run_scene_step(project)
    assistant = project / "products" / "assistant" / "site"
    assert (assistant / "scene.base.json").is_file() and (assistant / "scene.json").is_file()
    assert scene.elements[TREE_1["id"]].params.height == pytest.approx(old_height + 2.0)
    assert "fence-0" not in scene.elements and scene.elements["curb-0"].params.width == 0.16
    manual = [el for el in scene.elements.values() if el.kind.value == "manhole" and el.evidence.get("evidence") == "manual"]
    assert len(manual) == 1
    report = json.loads((assistant / "pipeline_report.json").read_text(encoding="utf-8"))["environment_scene"]
    assert set(report["applied"]) == {tree_edit, fence_edit, manual_edit, curb_edit} and report["stale"] == [] and report["error"] is None
    with (_cache(project) / "scene.pkl").open("rb") as handle:
        data = pickle.load(handle)
    ifc_path, dxf_path = _write_cad(project, data["scene"], data["dxf"], tmp_path / "export")
    doc = ezdxf.readfile(dxf_path)
    assert _tree_height_dxf(doc, TREE_1["xy"]) == pytest.approx(old_height + 2.0, abs=0.01)
    assert not list(doc.modelspace().query('*[layer=="S-FENC"]'))
    assert len(list(doc.modelspace().query('INSERT[layer=="S-MANH-MANUAL"]'))) == 1
    ifc = ifcopenshell.open(str(ifc_path))
    assert _psets(ifc, TREE_1["id"])["SAMcloud_Tree"]["Hoehe_m"] == pytest.approx(old_height + 2.0, abs=1e-3)
    assert ifc.by_guid(global_id_of(manual[0].id)) is not None and not ifc.by_type("IfcRailing")
    assert _psets(ifc, "curb-0")["SAMcloud_Curb"]["Breite_m"] == pytest.approx(0.16)

    # a recompute renumbers the order-dependent ids and gives the trees new uuids: re-applied by signature
    scene = renumbered_recompute(project)
    trees = sorted((el for el in scene.elements.values() if el.kind.value == "tree" and el.evidence.get("added") is None),
                   key=lambda el: el.params.trunk_xy[0])
    assert TREE_1["id"] not in scene.elements and trees[0].params.height == pytest.approx(old_height + 2.0)
    assert not [el for el in scene.elements.values() if el.kind.value == "fence"]  # fence-3 deleted again
    assert scene.elements["curb-3"].params.width == 0.16
    history = History.load(project, "site", domain.history_backend())
    assert {edit["status"] for edit in history.edits()} == {"active"}

    # the fence vanished in the next recompute: its edit is stale and reported, the others stay applied
    scene = renumbered_recompute(project, drop_fence=True)
    history = History.load(project, "site", domain.history_backend())
    statuses = {edit["id"]: edit["status"] for edit in history.edits()}
    assert statuses[fence_edit] == "stale" and statuses[tree_edit] == statuses[curb_edit] == statuses[manual_edit] == "active"
    report = json.loads((assistant / "pipeline_report.json").read_text(encoding="utf-8"))["environment_scene"]
    assert [item["id"] for item in report["stale"]] == [fence_edit]
    assert history.edit(fence_edit)["stale_code"] == "assistant.history.element_gone"


def test_undo_restores_the_automatic_scene(tmp_path: Path) -> None:
    project = copy_project(tmp_path)
    domain = OutdoorDomain(project, "site", mode="outdoor")
    base = domain.load_base()
    edit = _accept(domain, EditOp("set_tree", TREE_2["id"], {"height_delta": 1.0}))
    assert domain.current_model().elements[TREE_2["id"]].params.height == pytest.approx(base.elements[TREE_2["id"]].params.height + 1.0)
    History.load(project, "site", domain.history_backend()).undo(edit)
    assert domain.history_backend().semantic_hash(domain.current_model()) == domain.history_backend().semantic_hash(base)
    scene = run_scene_step(project)  # all edits undone: the chain writes its own scene again
    assert scene.elements[TREE_2["id"]].params.height == pytest.approx(base.elements[TREE_2["id"]].params.height)


# ------------------------------------------------------------------------------------------------------- cad_only

def _stub_heavy(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    import samcloud.drone.export as export

    monkeypatch.setattr(export, "export_point_cloud", lambda ps, path, frame, extra=None: (calls.append("laz"), Path(path).write_bytes(b"LAZ")))
    monkeypatch.setattr(export, "export_mesh", lambda *args, **kwargs: calls.append("obj"))
    monkeypatch.setattr(env, "_write_environment_glb", lambda nodes, path, frame: (calls.append("glb"), Path(path).write_bytes(b"glTF")))
    monkeypatch.setattr(env, "_dsm_mesh", lambda path, frame: None)


def test_cad_only_export_writes_ifc_dxf_and_publishes_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = copy_project(tmp_path)
    out = project / "products" / "environment"
    calls: list[str] = []
    _stub_heavy(monkeypatch, calls)
    domain = OutdoorDomain(project, "site", mode="outdoor")
    _accept(domain, EditOp("set_tree", TREE_1["id"], {"height_delta": 1.5}))
    started = time.monotonic()
    state = env.run_environment(project, steps="scene,export,qa", cad_only=True, config=load_drone_config())
    elapsed = time.monotonic() - started
    assert calls == [] and elapsed < 120
    assert state["steps"]["export"]["status"] == "done" and state["steps"]["export"]["outputs"]["cad_only"] is True
    assert state["steps"]["scene"]["status"] == "done" and state["steps"]["qa"]["status"] == "done"
    assert (out / "outdoor.ifc").is_file() and (out / "outdoor.dxf").is_file()
    assert not (out / "classified.laz").exists() and not (out / "environment.glb").exists()
    published = state["steps"]["export"]["outputs"]["published"]
    assert Path(published["dxf"]).is_file() and Path(published["ifc"]).is_file()
    doc = ezdxf.readfile(published["dxf"])
    height = domain.current_model().elements[TREE_1["id"]].params.height
    assert _tree_height_dxf(doc, TREE_1["xy"]) == pytest.approx(height, abs=0.01)


def test_default_export_still_writes_the_heavy_products(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = copy_project(tmp_path)
    out = project / "products" / "environment"
    calls: list[str] = []
    _stub_heavy(monkeypatch, calls)
    result = env._step_export(project, out, load_drone_config(), out / "cache")
    assert "laz" in calls and "glb" in calls and "obj" in calls and "cad_only" not in result
    assert (out / "outdoor.ifc").is_file() and (out / "outdoor.dxf").is_file()


def test_env_job_command_with_steps_and_cad_only() -> None:
    from samcloud.drone.env_job import build_command

    plain = build_command("P", False)
    assert plain[-2:] == ["--project", "P"]
    command = build_command("P", False, steps="scene,export,qa", cad_only=True)
    assert command[-3:] == ["--steps", "scene,export,qa", "--cad-only"]
    assert command[command.index("--steps") + 1] == "scene,export,qa" and command[-1] == "--cad-only"
    assert build_command("P", True, steps=["export"])[-3:] == ["--no-resume", "--steps", "export"]
    with pytest.raises(ValueError):
        build_command("P", False, steps="scene,paint")
    calls: list[dict[str, Any]] = []
    original = env.run_environment
    try:
        env.run_environment = lambda project, **kwargs: calls.append(kwargs) or {}  # type: ignore[assignment]
        env.main(["--project", "P", "--steps", "scene,export,qa", "--cad-only"])
        env.main(["--project", "P"])
    finally:
        env.run_environment = original  # type: ignore[assignment]
    assert calls[0]["cad_only"] is True and calls[0]["steps"] == "scene,export,qa" and calls[1]["cad_only"] is False


# ------------------------------------------------------------------------------------------------- QA after each op

OP_CASES = [
    ("set_tree", lambda base: EditOp("set_tree", TREE_1["id"], {"height_delta": 2.0})),
    ("move_tree", lambda base: EditOp("move_tree", TREE_2["id"], {"dx_m": 1.0, "dy_m": 0.5})),
    ("set_manhole", lambda base: EditOp("set_manhole", "manhole-0", {"diameter": 0.8})),
    ("set_fence", lambda base: EditOp("set_fence", "fence-0", {"height": 1.5})),
    ("offset_fence", lambda base: EditOp("offset_fence", "fence-0", {"distance_m": 0.4})),
    ("set_site_wall", lambda base: EditOp("set_site_wall", "site-wall-0", {"height": 1.3})),
    ("offset_curb", lambda base: EditOp("offset_curb", "curb-0", {"distance_m": 0.1})),
    ("set_curb", lambda base: EditOp("set_curb", "curb-0", {"height": 0.15})),
    ("delete_tree", lambda base: EditOp("delete_tree", TREE_2["id"], {})),
]


@pytest.mark.parametrize("name", [case[0] for case in OP_CASES])
def test_qa_and_exports_after_each_op(tmp_path: Path, name: str) -> None:
    import ifcopenshell

    from samcloud.assistant.domains.outdoor_qa import run_outdoor_qa

    domain = OutdoorDomain(shared_project(), "site", mode="outdoor")
    base = domain.load_base()
    proposal = domain.new_proposal(base)
    op = dict(OP_CASES)[name](base)
    proposal.apply(op)
    qa = run_outdoor_qa(shared_project(), proposal.plan, base, work_dir=tmp_path / "qa")
    assert qa.ok, qa.new_findings
    doc = ezdxf.readfile(tmp_path / "qa" / "proposal" / "outdoor.dxf")
    ifc = ifcopenshell.open(str(tmp_path / "qa" / "proposal" / "outdoor.ifc"))
    edited = proposal.plan.elements.get(op.target)
    if name == "set_tree":
        assert _tree_height_dxf(doc, TREE_1["xy"]) == pytest.approx(edited.params.height, abs=0.01)
        assert _psets(ifc, op.target)["SAMcloud_Tree"]["Hoehe_m"] == pytest.approx(edited.params.height, abs=1e-3)
    elif name == "move_tree":
        e, n = lv95(*edited.params.trunk_xy)
        assert any(abs(i.dxf.insert.x - e) < 0.01 and abs(i.dxf.insert.y - n) < 0.01 for i in doc.modelspace().query('INSERT[name=="TREE"]'))
        assert _psets(ifc, op.target)["SAMcloud_Tree"]["Stamm_LV95_E"] == pytest.approx(e, abs=1e-3)
    elif name == "set_manhole":
        insert = next(iter(doc.modelspace().query('INSERT[name=="MANHOLE"]')))
        assert _psets(ifc, op.target)["SAMcloud_Manhole"]["Durchmesser_m"] == pytest.approx(0.8)
        assert insert.dxf.xscale != pytest.approx(next(iter(_baseline_manhole_scale(tmp_path, base))))
    elif name in ("set_fence", "offset_fence"):
        lines = list(doc.modelspace().query('*[layer=="S-FENC"]'))
        dxf_xyz = np.asarray([tuple(v.dxf.location) for v in lines[0].vertices])
        expected = np.asarray(edited.params.axis) + np.asarray(proposal.plan.settings.origin_offset)
        assert dxf_xyz == pytest.approx(expected, abs=1e-3)
        assert _psets(ifc, op.target)["SAMcloud_Fence"]["Laenge_m"] > 0
    elif name == "set_site_wall":
        assert _psets(ifc, op.target)["SAMcloud_SiteWall"]
        assert edited.params.top_z - edited.params.base_z >= 1.3 - 1e-6
    elif name in ("offset_curb", "set_curb"):
        curb = list(doc.modelspace().query('*[layer=="S-CURB"]'))
        dxf_xyz = np.asarray([tuple(v.dxf.location) for v in curb[0].vertices])
        expected = np.asarray(edited.params.axis) + np.asarray(proposal.plan.settings.origin_offset)
        assert dxf_xyz[:, :2] == pytest.approx(expected[[0, -1]][:, :2] if len(dxf_xyz) == 2 else expected[:, :2], abs=0.02)
        assert dxf_xyz[:, 2].max() == pytest.approx(expected[:, 2].max(), abs=0.02)
    elif name == "delete_tree":
        assert len(list(doc.modelspace().query('INSERT[name=="TREE"]'))) == 1
        with pytest.raises(RuntimeError):
            ifc.by_guid(global_id_of(op.target))


def _baseline_manhole_scale(tmp_path: Path, base: Any) -> list[float]:
    folder = tmp_path / "qa"
    baseline = next(folder.glob("baseline/outdoor.dxf"), None)
    if baseline is None:  # the baseline came from the memory cache: write it once
        from samcloud.assistant.domains.outdoor_qa import cad_qa

        baseline = cad_qa(shared_project(), base, tmp_path / "baseline_check")[2]
    return [i.dxf.xscale for i in ezdxf.readfile(baseline).modelspace().query('INSERT[name=="MANHOLE"]')]


# ------------------------------------------------------------------------------------------- golden (HEAD code)

GOLDEN = Path(__file__).parent / "fixtures" / "outdoor_scene_golden.json"


def test_without_edits_the_output_matches_the_golden_snapshot_of_the_original_code(tmp_path: Path) -> None:
    """The snapshot was generated once with the original (pre-assistant) ``_step_scene`` on the same synthetic checkpoints:
    Scene JSON, neutral DXF entities, the DXF document and the IFC must be unchanged by the refactor and the hook."""
    from assistant_outdoor_synthetic import golden_digest

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    project = copy_project(tmp_path)
    scene = run_scene_step(project)  # the new _step_scene (build_scene_outputs + hook), no edits.json
    with (_cache(project) / "scene.pkl").open("rb") as handle:
        data = pickle.load(handle)
    digest = golden_digest(scene, data["dxf"], tmp_path / "digest")
    assert digest["counts"] == golden["counts"]
    for key in ("scene_sha256", "dxf_entities_sha256", "dxf_styling_sha256", "dxf_document_sha256", "ifc_sha256"):
        assert digest[key] == golden[key], key
    assert not (project / "products" / "assistant" / "site" / "scene.base.json").exists()  # the hook did nothing


def test_plain_chain_exports_a_tree_with_a_measured_trunk(tmp_path: Path) -> None:
    """Regression: TREE_STEM (measured trunk) used to reach the ortho symbol table and raise KeyError in the DXF writer."""
    from samcloud.bim.core.model import ParamOrigin
    from samcloud.outdoor_model.vegetation import VegetationResult

    project = copy_project(tmp_path)
    cache = _cache(project)
    with (cache / "vegetation.pkl").open("rb") as handle:
        veg = pickle.load(handle)
    measured = []
    for tree in veg.elements:
        origin = {**tree.param_origin, "trunk_diameter": ParamOrigin.MEASURED}
        measured.append(tree.model_copy(update={"params": tree.params.model_copy(update={"trunk_measured": True}), "param_origin": origin}))
    with (cache / "vegetation.pkl").open("wb") as handle:
        pickle.dump(VegetationResult(measured, [], []), handle)
    out = project / "products" / "environment"
    run_scene_step(project)
    assert not (project / "products" / "assistant" / "site" / "edits.json").exists()
    result = env._step_export(project, out, load_drone_config(), cache, cad_only=True)
    doc = ezdxf.readfile(result["dxf"])
    assert "TREE_STEM" in doc.blocks
    stems = [i for i in doc.modelspace().query('INSERT[name=="TREE_STEM"]')]
    assert len(stems) == 2 and all(i.dxf.layer == "S-TREE" for i in stems)
    assert stems[0].dxf.xscale == pytest.approx(TREE_1["trunk"] / 2.0) or stems[1].dxf.xscale == pytest.approx(TREE_1["trunk"] / 2.0)
