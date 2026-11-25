"""Assistant pipeline hooks (DXF export, clean plan, refinement, IFC export, share) and the drone DXF language record."""

from __future__ import annotations

from dataclasses import replace
import json
import logging
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ezdxf
import numpy as np
import pytest

from assistant_synthetic import LEVEL_ID, make_project, renumbered_recompute
from samcloud.bim import jobs as bim_jobs
from samcloud.dxf import jobs
from samcloud.dxf.cancel import CancelToken
from samcloud.dxf.pipeline import load_cached_result
from samcloud.dxf.writer import export_plan

LOGGER = logging.getLogger("samcloud.tests.dxf_language_assistant")
CACHE = ("products", "dxf", LEVEL_ID, "ortho")


def _cache(project: Path) -> Path:
    return project.joinpath(*CACHE)


def _patch_job_inputs(monkeypatch: pytest.MonkeyPatch, project: Path) -> None:
    """Cheap stand-ins for the LAS header, the DXF state and the clean-plan staleness check of the synthetic project."""
    result = load_cached_result(_cache(project))
    level = {"id": LEVEL_ID, "name": result.level_name, "floor_z": 0.2, "ceiling_z": 2.8, "checked": True}
    state = SimpleNamespace(mode_override="indoor", levels=[level], view={"coordinates": "local"}, segment_source="combined")
    monkeypatch.setattr(jobs, "load_state", lambda _project: state)
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode="indoor", frame=result.frame))
    monkeypatch.setattr(jobs, "plan_settings_for", lambda *_a, **_k: None)
    monkeypatch.setattr("samcloud.ortho.plan.clean_plan_fingerprint", lambda *_a, **_k: result.settings_fingerprint)
    monkeypatch.setattr("samcloud.ortho.segment.load_segment_result", lambda *_a, **_k: None)


def _export(project: Path, language: str | None = None) -> tuple[Path, dict[str, Any]]:
    kwargs = {} if language is None else {"language": language}
    jobs.run_export(project, project / "cloud.ply", LEVEL_ID, lambda *_a: None, CancelToken(), LOGGER, mode="indoor", variant="ortho", **kwargs)
    export_dir = project / "products" / "dxf" / "export"
    return next(export_dir.glob("*_clean.dxf")), json.loads((export_dir / "last_export.json").read_text(encoding="utf-8"))


def _dxf_texts(path: Path) -> list[str]:
    texts: list[str] = []
    for entity in ezdxf.readfile(str(path)).modelspace():
        if entity.dxftype() == "TEXT":
            texts.append(str(entity.dxf.text))
        elif entity.dxftype() == "INSERT":
            texts.extend(f"{attrib.dxf.tag}={attrib.dxf.text}" for attrib in entity.attribs)
    return texts


def _stable(text: str) -> str:
    text = re.sub(r"\{[0-9A-Fa-f-]{36}\}", "{GUID}", text)
    text = re.sub(r"\d+\.\d+\.\d+ @ \S+", "EZDXF-STAMP", text)
    return re.sub(r"(\$TD[A-Z]+\n\s*\d+\n)[^\n]*", r"\1TIME", text)


def _ifc_text(project: Path, manifest: dict[str, Any]) -> str:
    text = (project / "products" / "bim" / LEVEL_ID / str(manifest["ifc"])).read_text(encoding="utf-8")
    return re.sub(r"FILE_NAME\([^\n]*", "FILE_NAME(...);", text)


def _accept(project: Path, *ops: Any, prompt: str = "Bitte das Fenster breiter machen") -> str:
    from samcloud.assistant.edits import Proposal
    from samcloud.assistant.history import History
    from samcloud.assistant.paths import plan_model_path
    from samcloud.bim.plan_model import load_plan_model

    proposal = Proposal(load_plan_model(plan_model_path(project, LEVEL_ID)))
    for op in ops:
        proposal.apply(op)
    return History.load(project, LEVEL_ID).accept(proposal, prompt=prompt, provider="ollama", model="dummy-model")


def _ifc_values(project: Path, manifest: dict[str, Any]) -> tuple[list[float], list[str]]:
    import ifcopenshell

    model = ifcopenshell.open(str(project / "products" / "bim" / LEVEL_ID / str(manifest["ifc"])))
    widths = sorted(round(float(window.OverallWidth), 3) for window in model.by_type("IfcWindow"))
    names = sorted(f"{space.Name} {space.LongName}" for space in model.by_type("IfcSpace"))
    return widths, names


# --------------------------------------------------------------------------- no edits.json: outputs unchanged


def test_without_edits_dxf_and_ifc_are_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.bim.furniture_recon.dxf_outline import apply_reconstructed_outlines

    hooked = make_project(tmp_path / "hooked", with_cloud=False)
    reference = make_project(tmp_path / "reference", with_cloud=False)
    plan_before = _cache(hooked).joinpath("plan_model.json").read_bytes()
    _patch_job_inputs(monkeypatch, hooked)
    # Importing the assistant would fail now: the hooks must not even try without edits.json.
    monkeypatch.setitem(sys.modules, "samcloud.assistant.history", None)
    exported, manifest = _export(hooked)
    assert "assistant" not in manifest
    cached = replace(load_cached_result(_cache(hooked)), project_name=hooked.name)  # run_export names the plan after the project
    expected = export_plan(apply_reconstructed_outlines(cached, _cache(hooked)), tmp_path / "expected_dxf")
    assert _stable(exported.read_text(encoding="utf-8", errors="replace")) == _stable(expected.read_text(encoding="utf-8", errors="replace"))
    ifc_manifest = bim_jobs.run_ifc_export(hooked, LEVEL_ID)
    assert "assistant" not in ifc_manifest
    with monkeypatch.context() as patch:
        patch.setattr("samcloud.dxf.assistant_hooks.ensure_applied_safely", lambda *_a, **_k: None)  # the code path before the hook
        reference_manifest = bim_jobs.run_ifc_export(reference, LEVEL_ID)
    assert _ifc_text(hooked, ifc_manifest) == _ifc_text(reference, reference_manifest)
    assert _cache(hooked).joinpath("plan_model.json").read_bytes() == plan_before
    assert not (hooked / "products" / "assistant").exists()


# --------------------------------------------------------------------------- accepted edit reaches DXF and IFC


def test_accepted_edit_reaches_dxf_and_ifc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.assistant.edits import EditOp

    project = make_project(tmp_path, with_cloud=False)
    _patch_job_inputs(monkeypatch, project)
    edit_id = _accept(project, EditOp("set_opening", "O002", {"width_m": 1.5}), EditOp("rename_room", "R002", {"name": "Gast"}))
    exported, manifest = _export(project, "en")
    texts = _dxf_texts(exported)
    assert "LABEL=Window 1.50 m" in texts and any(text.startswith("Gast  ") for text in texts)
    assert manifest["assistant"][0]["applied"] == [edit_id] and manifest["assistant"][0]["stale"] == []
    assert manifest["assistant"][0]["error"] is None
    ifc_manifest = bim_jobs.run_ifc_export(project, LEVEL_ID)
    widths, names = _ifc_values(project, ifc_manifest)
    assert 1.5 in widths and any("Gast" in name for name in names)
    assert ifc_manifest["assistant"]["applied"] == [edit_id]


def test_recompute_reapplies_and_reports_stale_edits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.assistant.edits import EditOp
    from samcloud.assistant.history import History
    from samcloud.bim.plan_model import load_plan_model, save_plan_model

    project = make_project(tmp_path, with_cloud=False)
    _patch_job_inputs(monkeypatch, project)
    window = _accept(project, EditOp("set_opening", "O002", {"width_m": 1.5}))
    table = _accept(project, EditOp("move_furniture", "F001", {"dx_m": 0.2, "dy_m": 0.0}))
    base = load_plan_model(project / "products" / "assistant" / LEVEL_ID / "plan_model.base.json")

    def recompute(*_args: Any, **_kwargs: Any) -> Any:
        recomputed = renumbered_recompute(base)
        recomputed.furniture = []  # the table vanished in the recompute
        save_plan_model(recomputed, _cache(project))
        return SimpleNamespace(project_name=project.name)

    monkeypatch.setattr(jobs, "generate_clean_plan", recompute)
    jobs.run_clean_plan(project, project / "cloud.ply", LEVEL_ID, lambda *_a: None, CancelToken(), LOGGER, mode="indoor")
    report = json.loads((project / "products" / "assistant" / LEVEL_ID / "pipeline_report.json").read_text(encoding="utf-8"))
    assert report["clean_plan"]["changed"] and report["clean_plan"]["applied"] == [window]
    assert [item["id"] for item in report["clean_plan"]["stale"]] == [table]
    plan = load_plan_model(_cache(project) / "plan_model.json")
    assert next(o for o in plan.openings if o.kind == "window").width_m == pytest.approx(1.5)

    exported, manifest = _export(project)
    assert "LABEL=Window 1.50 m" in _dxf_texts(exported)
    assert [item["id"] for item in manifest["assistant"][0]["stale"]] == [table] and manifest["assistant"][0]["stale"][0]["reason"]

    # a second recompute right before the IFC export: the IFC hook re-applies before hashing and reading the plan
    recompute()
    ifc_manifest = bim_jobs.run_ifc_export(project, LEVEL_ID)
    assert 1.5 in _ifc_values(project, ifc_manifest)[0]
    assert ifc_manifest["assistant"]["changed"] and [item["id"] for item in ifc_manifest["assistant"]["stale"]] == [table]
    assert ifc_manifest["plan_model_sha256"] == bim_jobs.plan_model_sha256(project, LEVEL_ID)
    assert [edit["status"] for edit in History.load(project, LEVEL_ID).edits()] == ["active", "stale"]


def test_refinement_reapplies_edits_through_the_clean_plan_rewrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.assistant.edits import EditOp
    from samcloud.bim.plan_model import load_plan_model, save_plan_model

    project = make_project(tmp_path, with_cloud=False)
    _patch_job_inputs(monkeypatch, project)
    window = _accept(project, EditOp("set_opening", "O002", {"width_m": 1.5}))
    base = load_plan_model(project / "products" / "assistant" / LEVEL_ID / "plan_model.base.json")
    calls: list[str] = []

    def regenerate(*_args: Any, **_kwargs: Any) -> Any:  # pass 2 of the clean plan with the refined openings: new plan_model.json
        calls.append("clean_plan")
        save_plan_model(renumbered_recompute(base), _cache(project))
        return SimpleNamespace(project_name=project.name)

    module = types.ModuleType("samcloud.bim.openings3d")
    module.refine_level_openings = lambda *_a, **_k: (calls.append("refine"), SimpleNamespace(status="ok", mode="3d", openings=2, changed=True, seconds=0.1, message=""))[1]
    module.plan_has_current_openings = lambda *_a, **_k: False
    pipeline = types.ModuleType("samcloud.bim.openings3d.pipeline")
    pipeline.refined_openings_applied = lambda *_a, **_k: (True, "")
    monkeypatch.setitem(sys.modules, "samcloud.bim.openings3d", module)
    monkeypatch.setitem(sys.modules, "samcloud.bim.openings3d.pipeline", pipeline)
    monkeypatch.setattr(jobs, "generate_clean_plan", regenerate)
    jobs.run_refine_openings(project, project / "cloud.ply", LEVEL_ID, lambda *_a: None, CancelToken(), LOGGER, mode="indoor")
    assert calls == ["refine", "clean_plan"]
    report = json.loads((project / "products" / "assistant" / LEVEL_ID / "pipeline_report.json").read_text(encoding="utf-8"))
    assert set(report) == {"clean_plan"}  # one hook, after the final plan_model.json write
    assert report["clean_plan"]["changed"] and report["clean_plan"]["applied"] == [window]
    plan = load_plan_model(_cache(project) / "plan_model.json")
    assert next(o for o in plan.openings if o.kind == "window").width_m == pytest.approx(1.5)


def test_run_all_reexports_after_an_accepted_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from samcloud.app.dxf_controller import DxfPageController
    from samcloud.assistant.edits import EditOp

    project = make_project(tmp_path, with_cloud=False)
    _patch_job_inputs(monkeypatch, project)
    exported, _manifest = _export(project)
    monkeypatch.setattr(jobs, "workflow_step_status", lambda *_a: {"ortho": "ok", "segment": "ok", "clean_plan": "ok"})
    monkeypatch.setattr(jobs, "refine_openings_status", lambda *_a: "fresh")
    monkeypatch.setattr(jobs, "ifc_export_status", lambda *_a: "fresh")
    monkeypatch.setattr("samcloud.dxf.writer3d.dxf3d_is_current", lambda *_a, **_k: True)
    steps: list[str] = []
    real_export = jobs.run_export
    monkeypatch.setattr(jobs, "run_export", lambda *a, **k: (steps.append("export"), real_export(*a, **k))[1])
    for name in ("run_ortho", "run_segment", "run_clean_plan", "run_refine_openings", "run_export_ifc"):
        monkeypatch.setattr(jobs, name, lambda *_a, _name=name, **_k: steps.append(_name))
    jobs.run_all(project, project / "cloud.ply", LEVEL_ID, lambda *_a: None, CancelToken(), LOGGER, mode="indoor")
    assert steps == []  # no edits: the export is fresh

    monkeypatch.setattr(DxfPageController, "plan_status", lambda self, level_id, variant="pointcloud": "ok")
    monkeypatch.setattr(DxfPageController, "level_label", lambda self, level_id: load_cached_result(_cache(project)).level_name)
    controller = DxfPageController(project)
    assert controller.export_misses_assistant_edits(LEVEL_ID) is False
    now = exported.stat().st_mtime_ns  # coarse file-system clocks: date the cache and the export back, the edit is then clearly newer
    os.utime(_cache(project) / "result.json", ns=(now - 10_000_000_000, now - 10_000_000_000))
    os.utime(exported, ns=(now - 5_000_000_000, now - 5_000_000_000))
    _accept(project, EditOp("set_opening", "O002", {"width_m": 1.5}))
    assert controller.export_misses_assistant_edits(LEVEL_ID) is True  # the export row shows "stale"
    jobs.run_all(project, project / "cloud.ply", LEVEL_ID, lambda *_a: None, CancelToken(), LOGGER, mode="indoor")
    assert steps == ["export"]
    assert "LABEL=Window 1.50 m" in _dxf_texts(exported)
    assert controller.export_misses_assistant_edits(LEVEL_ID) is False


def test_building_ifc_reapplies_the_edits_of_every_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.dxf import assistant_hooks

    project = tmp_path / "multi"
    for key in ("level_1", "level_2", "level_3"):
        folder = project / "products" / "assistant" / key
        folder.mkdir(parents=True)
        if key != "level_3":
            (folder / "edits.json").write_text(json.dumps({"version": 1, "edits": []}), encoding="utf-8")
    assert assistant_hooks.levels_with_edits(project) == ["level_1", "level_2"]
    state = SimpleNamespace(mode_override="indoor", levels=[{"id": "level_1", "name": "EG", "floor_z": 0.0, "ceiling_z": 2.6}])
    monkeypatch.setattr(jobs, "load_state", lambda _project: state)
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode="indoor"))
    order: list[str] = []
    monkeypatch.setattr(jobs, "run_ifc_export", lambda *_a, **_k: (order.append("ifc level_1"), {"ifc": "x.ifc", "ids_failures": 0})[1])
    monkeypatch.setattr("samcloud.dxf.writer3d.export_level_3d_dxf", lambda *_a, **_k: None)
    monkeypatch.setattr(jobs, "ensure_applied_safely", lambda _project, key, step, _logger=None: order.append(f"{step} {key}"))
    monkeypatch.setattr("samcloud.bim.building_export.export_building_ifc", lambda *_a, **_k: order.append("building") or None)
    jobs.run_export_ifc(project, project / "cloud.ply", "level_1", lambda *_a: None, CancelToken(), LOGGER, mode="indoor")
    assert order == ["ifc level_1", "building_ifc level_2", "building"]


def test_pipeline_report_survives_unserialisable_values(tmp_path: Path) -> None:
    from samcloud.dxf import assistant_hooks

    folder = tmp_path / "products" / "assistant" / "level_1"
    folder.mkdir(parents=True)
    assistant_hooks._record(tmp_path, "level_1", {"step": "x", "bad": object()})  # TypeError inside json.dumps: logged, not raised
    assistant_hooks._record(tmp_path, "level_1", {"step": "y", "value": float("nan")})
    assert json.loads((folder / "pipeline_report.json").read_text(encoding="utf-8").replace("NaN", "null"))["y"]["value"] is None


def test_assistant_failures_never_break_the_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.assistant import history
    from samcloud.assistant.edits import EditOp

    project = make_project(tmp_path, with_cloud=False)
    _patch_job_inputs(monkeypatch, project)
    _accept(project, EditOp("set_opening", "O002", {"width_m": 1.5}))

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("edits.json is broken")

    monkeypatch.setattr(history, "apply_to_result", broken)
    monkeypatch.setattr(history, "ensure_applied", broken)
    exported, manifest = _export(project)
    assert exported.is_file() and "RuntimeError: edits.json is broken" in manifest["assistant"][0]["error"]
    ifc_manifest = bim_jobs.run_ifc_export(project, LEVEL_ID)
    assert "edits.json is broken" in ifc_manifest["assistant"]["error"]


# --------------------------------------------------------------------------- share: edits without prompt, never the chat


def test_share_contains_active_edits_without_prompt_or_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.assistant.edits import EditOp
    from samcloud.assistant.history import History
    from samcloud.share import export, project as project_share

    project = make_project(tmp_path, with_cloud=False)
    kept = _accept(project, EditOp("set_opening", "O002", {"width_m": 1.5}), prompt="Mach das Fenster 1.5 m breit")
    undone = _accept(project, EditOp("rename_room", "R002", {"name": "Gast"}), prompt="Nenne den Raum Gast")
    History.load(project, LEVEL_ID).undo(undone)
    (project / "products" / "assistant" / LEVEL_ID / "chat.json").write_text(json.dumps([{"role": "user", "text": "geheim"}]), encoding="utf-8")
    (project / "products" / "classified").mkdir(parents=True)
    (project / "products" / "classified" / "classified.ply").write_bytes(b"ply")
    bim = project / "products" / "bim" / LEVEL_ID
    bim.mkdir(parents=True)
    (bim / "model.glb").write_bytes(b"glTF")
    (bim / "props.json").write_text(json.dumps({"elements": {}}), encoding="utf-8")
    export_dir = project / "products" / "dxf" / "export"
    export_dir.mkdir(parents=True)
    ezdxf.new().saveas(export_dir / f"{project.name}_{LEVEL_ID}_clean.dxf")
    monkeypatch.setattr(project_share, "_dxf_point_source", lambda _directory: None)
    monkeypatch.setattr(export, "prepare_points", lambda *_a, **_k: (np.zeros((5, 3)), None))
    monkeypatch.setattr(export, "build_octree", lambda *_a, **_k: {})
    monkeypatch.setattr(export, "assemble_package", lambda *_a, **_k: None)
    scanned: list[list[str]] = []
    real_scan = export.scan_package
    monkeypatch.setattr(export, "scan_package", lambda package: scanned.append(sorted(p.name for p in Path(package).rglob("*.json"))) or [])
    package = tmp_path / "out" / "package"
    project_share.build_project_package(project, package, config={}, check=lambda: None, max_points=100,
                                        progress_for_phase=lambda _index: (lambda _fraction, message="": None))
    shared = json.loads((package / "data" / "edits" / f"{LEVEL_ID}.json").read_text(encoding="utf-8"))
    assert [edit["id"] for edit in shared["edits"]] == [kept]
    assert set(shared["edits"][0]) == {"id", "timestamp", "ops", "changes", "provider", "model"}
    payload = (package / "data" / "edits" / f"{LEVEL_ID}.json").read_text(encoding="utf-8")
    assert "Fenster 1.5 m breit" not in payload and "prompt" not in payload
    assert not list(package.rglob("chat.json")) and "geheim" not in "".join(p.read_text(encoding="utf-8", errors="ignore") for p in package.rglob("*.json"))
    assert f"{LEVEL_ID}.json" in scanned[0]  # written before the privacy scan
    assert real_scan(package) == []
    data = json.loads((package / "data" / "project.json").read_text(encoding="utf-8"))
    assert next(row for row in data["levels"] if row["key"] == LEVEL_ID)["edits"] == f"data/edits/{LEVEL_ID}.json"


# --------------------------------------------------------------------------- drone DXF: recorded language


def test_drone_export_records_the_language_and_publish_copies_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.drone.contracts import LocalFrame, PointSet
    from samcloud.drone.run_environment import _dump, _step_export
    from samcloud.dxf.text_language import dxf_file_language, language_sidecar_path
    from samcloud.outdoor_model.scene import assemble_dxf, assemble_scene

    frame = LocalFrame((2_600_000.0, 1_200_000.0, 500.0))
    project = tmp_path / "drone"
    output = project / "products" / "environment"
    cache = output / "cache"
    cache.mkdir(parents=True)
    _dump(cache / "load.pkl", (PointSet(np.array([[0.013, 0.021, 0.007]]), np.array([2], np.uint16)), frame))
    _dump(cache / "scene.pkl", {"scene": assemble_scene(frame=frame), "dxf": assemble_dxf({}, frame)})
    monkeypatch.setattr("samcloud.drone.run_environment._dsm_mesh", lambda *_args: None)
    result = _step_export(project, output, {"dxf": {}}, cache)
    assert dxf_file_language(output / "outdoor.dxf") == "en"  # the autouse fixture runs the UI in English
    published = Path(result["published"]["dxf"])
    assert published.name.endswith("_Site_clean.dxf") and dxf_file_language(published) == "en"
    assert jobs.exported_file_language(project, published) == "en"
    assert language_sidecar_path(published).is_file()


def test_publish_without_recorded_language_writes_no_sidecar(tmp_path: Path) -> None:
    from samcloud.drone.publish import publish_environment_exports
    from samcloud.dxf.text_language import language_sidecar_path, record_dxf_language

    env = tmp_path / "drone2" / "products" / "environment"
    env.mkdir(parents=True)
    source = env / "outdoor.dxf"
    ezdxf.new().saveas(source)
    published = publish_environment_exports(tmp_path / "drone2", {"dxf": source})["dxf"]
    assert not language_sidecar_path(published).exists()
    record_dxf_language(source, "de")
    published = publish_environment_exports(tmp_path / "drone2", {"dxf": source})["dxf"]
    assert jobs.exported_file_language(tmp_path / "drone2", published) == "de"
