"""AI assistant on the /dxf page: after accept / undo the DXF and the IFC update by themselves and visibly.

* the 2D plan shows the accepted / undone edit right away (payload patched with the edits, new version token),
* the export runs on the cached plan even when it counts as stale (``--assistant`` / ``allow_stale_plan``),
* without a cached clean plan the full computation ("Compute everything") starts by itself,
* the IFC viewer loads the new model (versioned URL) once the IFC export is done, without a page reload,
* a failed update keeps the edit and offers "Export again"; one chat card between the DXF and the IFC card.

Synthetic data only (``assistant_synthetic``); no job subprocess runs (``DxfJobRunner.start`` is recorded).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import yaml
from nicegui import ElementFilter

from assistant_synthetic import FLOOR_Z, LEVEL_ID, make_plan, make_project, write_cloud
from samcloud.app.dxf_controller import DxfPageController
from samcloud.assistant.service import AssistantService
from samcloud.dxf import job_runner, jobs
from samcloud.dxf.cancel import CancelToken
from samcloud.dxf.job_runner import DxfJobRunner
from samcloud.dxf.pipeline import load_cached_result
from samcloud.dxf.result import vector_payload
from samcloud.dxf.state import load_state, save_state
from test_assistant_agent import FakeProvider, call, say
from test_assistant_panel import _ask, _dxf_user, _element, _enable_assistant, _eventually, _has, _project, _use

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

LOGGER = logging.getLogger("samcloud.tests.assistant_live_update")
CACHE = ("products", "dxf", LEVEL_ID, "ortho")


@pytest.fixture(autouse=True)
def _private_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the real user's AI settings / credentials."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


class Jobs:
    """Recorded ``DxfJobRunner.start`` calls; each job ends right away with ``outcomes[action]`` (default "done") after
    running the optional side effect ``effects[action](project)`` (writes the files the real job would write)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, project: Path) -> None:
        self.project = project
        self.calls: list[tuple[str, str | None, dict[str, Any]]] = []
        self.outcomes: dict[str, list[tuple[str, str]]] = {}
        self.effects: dict[str, Callable[[Path], None]] = {}
        self.last: tuple[str, str, str] = ("", "idle", "")
        recorder = self

        def start(runner: DxfJobRunner, action: str, las: Any, level: str | None = None, **kwargs: Any) -> None:
            recorder.calls.append((str(action), level, dict(kwargs)))
            queue = recorder.outcomes.get(str(action)) or []
            state, message = queue.pop(0) if queue else ("done", "")
            if state == "done" and str(action) in recorder.effects:
                recorder.effects[str(action)](recorder.project)
            recorder.last = (str(action), state, message)

        def status(runner: DxfJobRunner) -> dict[str, Any]:
            action, state, message = self.last
            return {"action": action, "state": state, "fraction": 1.0 if state == "done" else 0.0, "stage": "", "message": message,
                    "started": "", "finished": None, "pid": None}

        monkeypatch.setattr(DxfJobRunner, "start", start)
        monkeypatch.setattr(DxfJobRunner, "status", status)
        monkeypatch.setattr(DxfJobRunner, "is_running", lambda runner: False)

    def actions(self) -> list[tuple[str, str | None]]:
        return [(action, level) for action, level, _kwargs in self.calls]


def _glb(project: Path) -> Path:
    from samcloud.bim import ui_data

    path = ui_data.glb_path(project, LEVEL_ID)
    assert path is not None
    return path


def _new_ifc(project: Path) -> None:
    """Side effect of the IFC export: the viewer GLB is rewritten (newer mtime)."""
    path = _glb(project)
    stamp = path.stat().st_mtime + 30.0
    os.utime(path, (stamp, stamp))


def _plan_stale_manifest(project: Path) -> None:
    """Side effect of the assistant DXF export on a stale plan: last_export.json records plan_stale."""
    export_dir = project / "products" / "dxf" / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "last_export.json").write_text(json.dumps({"files": [], "plan_stale": True}), encoding="utf-8")


def _payload_texts(project: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Texts of the ortho payload exactly as the browser fetches it (HTTP route of the 2D canvas)."""
    from samcloud.app import application
    from samcloud.app.state import state as app_state

    monkeypatch.setattr(app_state, "active_project_directory", project)
    response = application.dxf_review_payload(LEVEL_ID, variant="ortho")
    assert response.status_code == 200
    payload = json.loads(response.body)
    return [str(entity.get("text") or "") for entity in payload["layers"]["A-ANNO"]["entities"]]


def _rename_provider(monkeypatch: pytest.MonkeyPatch) -> FakeProvider:
    return _use(monkeypatch, FakeProvider([call("rename_room", target="R002", name="Gast"), say("Raum R002 heisst jetzt Gast.")]))


def _text(user: Any, marker: str) -> str:
    """Text of the element marked ``marker`` (visible or not); '' while it does not exist (e.g. during a refresh)."""
    with user:
        return " ".join(str(getattr(element, "text", "")) for element in ElementFilter(marker=marker))


def _order(user: Any, markers: tuple[str, ...]) -> list[str]:
    """Markers in document order (first occurrence)."""
    seen: list[str] = []
    with user:
        for element in ElementFilter():
            for marker in element._markers:
                if marker in markers and marker not in seen:
                    seen.append(marker)
    return seen


# ------------------------------------------------------------------------------------------------- 1. accept -> live update

@pytest.mark.anyio
async def test_accept_shows_the_edit_at_once_and_reloads_the_ifc_after_its_export(tmp_path: Path,
                                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_assistant()
    project = _project(tmp_path)
    recorded = Jobs(monkeypatch, project)
    recorded.effects["export_ifc"] = _new_ifc
    _rename_provider(monkeypatch)
    assert not any(text.startswith("Gast") for text in _payload_texts(project, monkeypatch))
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-panel")
        canvas = _element(user, "result-canvas")
        viewer = _element(user, "ifc-viewer")
        url_before, src_before = canvas.props["payload-url"], viewer.props["src"]
        await _ask(user, "Raum umbenennen", "Raum R002 heisst jetzt Gast.")
        user.find(marker="assistant-accept").click()

        # right after accept: same canvas element, new payload URL, and that payload already holds the edited name
        assert canvas.props["payload-url"] != url_before
        assert any(text.startswith("Gast") for text in _payload_texts(project, monkeypatch))
        assert recorded.calls[0][:2] == ("export", LEVEL_ID) and recorded.calls[0][2].get("assistant") is True
        assert recorded.calls[0][2].get("variant") == "ortho"

        # the IFC export follows automatically; afterwards the viewer loads the new model (new version token)
        await _eventually(lambda: ("export_ifc", LEVEL_ID) in recorded.actions(), timeout=5.0)
        new_src = f"/api/bim/{LEVEL_ID}/model.glb?v={_glb(project).stat().st_mtime_ns}"
        await _eventually(lambda: _has(user, "ifc-viewer") and _element(user, "ifc-viewer").props["src"] == new_src)
        assert new_src != src_before
        await _eventually(lambda: "DXF and IFC show the change" in _text(user, "assistant-export-status"))
        assert recorded.actions() == [("export", LEVEL_ID), ("export_ifc", LEVEL_ID)]
        assert not _has(user, "assistant-export-error")
        # after the rebuild the canvas still shows the edited plan
        assert "e" in _element(user, "result-canvas").props["payload-url"].split("&v=")[1]
    assert [edit["status"] for edit in AssistantService(project, LEVEL_ID).edits()] == ["active"]


@pytest.mark.anyio
async def test_undo_shows_the_original_plan_at_once_and_exports_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_assistant()
    project = _project(tmp_path)
    recorded = Jobs(monkeypatch, project)
    _rename_provider(monkeypatch)
    service = AssistantService(project, LEVEL_ID)
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Raum umbenennen", "Raum R002 heisst jetzt Gast.")
        user.find(marker="assistant-accept").click()
        await _eventually(lambda: len(recorded.calls) == 2)
        edit_id = service.edits()[0]["id"]
        await user.should_see(marker=f"assistant-undo-{edit_id}")
        await asyncio.sleep(1.2)  # the page rebuilt after the last export step
        url_accepted = _element(user, "result-canvas").props["payload-url"]
        user.find(marker=f"assistant-undo-{edit_id}").click()
        assert _element(user, "result-canvas").props["payload-url"] != url_accepted
        assert not any(text.startswith("Gast") for text in _payload_texts(project, monkeypatch))
        await _eventually(lambda: recorded.actions()[2:] == [("export", LEVEL_ID), ("export_ifc", LEVEL_ID)], timeout=5.0)
        assert recorded.calls[2][2].get("assistant") is True
    assert [edit["status"] for edit in service.edits()] == ["undone"]


# ------------------------------------------------------------------------------------------------- 2. stale plan / missing cache

@pytest.mark.anyio
async def test_stale_plan_note_after_the_assistant_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    recorded = Jobs(monkeypatch, project)
    recorded.effects["export"] = _plan_stale_manifest
    _rename_provider(monkeypatch)
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Raum umbenennen", "Raum R002 heisst jetzt Gast.")
        user.find(marker="assistant-accept").click()
        await user.should_see(marker="assistant-export-plan-stale", retries=60)
        await user.should_see("Compute everything")
    assert recorded.actions() == [("export", LEVEL_ID), ("export_ifc", LEVEL_ID)]


def _project_without_clean_plan(tmp_path: Path) -> Path:
    """Like the panel test project, but only plan_model.json is cached (no clean-plan result.json): the export cannot run."""
    from samcloud.bim.plan_model import save_plan_model

    project = tmp_path / "project"
    plan = make_plan()
    save_plan_model(plan, project.joinpath(*CACHE))
    write_cloud(project, plan)
    document = {"project": {"name": "assistant_synthetic"}, "workflow": {},
                "settings": {"classification": {"class_set": "indoor", "model_profile": "sam3_indoor"}}}
    (project / "project.samcloud.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    state = load_state(project)
    state.levels = [{"id": LEVEL_ID, "name": "EG", "floor_z": FLOOR_Z, "ceiling_z": 2.8, "point_count": 1000, "confidence": 1.0,
                     "low_confidence": False, "checked": True}]
    state.plan_variant = "ortho"
    save_state(project, state)
    AssistantService(project, LEVEL_ID).discard()
    return project


@pytest.mark.anyio
async def test_missing_clean_plan_starts_compute_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_assistant()
    project = _project_without_clean_plan(tmp_path)
    controller = DxfPageController(project)
    assert controller.assistant_export_ready(LEVEL_ID) is False
    recorded = Jobs(monkeypatch, project)
    recorded.outcomes["run_all"] = [("running", "")]  # still computing: the progress line stays visible
    _rename_provider(monkeypatch)
    monkeypatch.setattr(DxfJobRunner, "is_running", lambda runner: bool(recorded.calls))
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-panel")
        await _ask(user, "Raum umbenennen", "Raum R002 heisst jetzt Gast.")
        user.find(marker="assistant-accept").click()
        assert recorded.actions() == [("run_all", LEVEL_ID)]
        assert "assistant" not in recorded.calls[0][2]
        await _eventually(lambda: "computing everything" in _text(user, "assistant-export-status"))
    assert [edit["status"] for edit in AssistantService(project, LEVEL_ID).edits()] == ["active"]


def test_export_ready_needs_the_cached_clean_plan(tmp_path: Path) -> None:
    project = _project(tmp_path, ifc=False)
    assert DxfPageController(project).assistant_export_ready(LEVEL_ID) is True
    other = _project_without_clean_plan(tmp_path / "other")
    assert DxfPageController(other).assistant_export_ready(LEVEL_ID) is False


# ------------------------------------------------------------------------------------------------- 3. failure + retry

@pytest.mark.anyio
async def test_failed_update_keeps_the_edit_and_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    recorded = Jobs(monkeypatch, project)
    recorded.outcomes["export"] = [("failed", "Disk full while writing the DXF")]
    _rename_provider(monkeypatch)
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Raum umbenennen", "Raum R002 heisst jetzt Gast.")
        user.find(marker="assistant-accept").click()
        await user.should_see(marker="assistant-export-error", retries=60)
        await user.should_see("Disk full while writing the DXF")
        await user.should_see("The accepted change stays stored.")
        assert recorded.actions() == [("export", LEVEL_ID)]  # no IFC export after a failed DXF export
        assert any(text.startswith("Gast") for text in _payload_texts(project, monkeypatch))  # the plan view keeps the edit
        user.find(marker="assistant-export-retry").click()
        await _eventually(lambda: recorded.actions() == [("export", LEVEL_ID), ("export", LEVEL_ID), ("export_ifc", LEVEL_ID)],
                          timeout=5.0)
        await _eventually(lambda: not _has(user, "assistant-export-error"))
        await _eventually(lambda: "DXF and IFC show the change" in _text(user, "assistant-export-status"))
    assert [edit["status"] for edit in AssistantService(project, LEVEL_ID).edits()] == ["active"]


# ------------------------------------------------------------------------------------------------- 4. layout

@pytest.mark.anyio
async def test_one_chat_card_between_dxf_and_ifc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_assistant()
    project = _project(tmp_path)
    Jobs(monkeypatch, project)
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-panel")
        await user.should_see(marker="ifc-viewer")
        await user.should_not_see(marker="assistant-panel-ifc")
        assert _order(user, ("result-canvas", "assistant-panel", "ifc-card")) == ["result-canvas", "assistant-panel", "ifc-card"]
        with user:
            assert len(list(ElementFilter(marker="assistant-input"))) == 1  # one question field on the page
        # the selection stays shared: a 3D pick highlights in 2D and 3D
        from samcloud.assistant.selection import global_id_for

        door_gid = global_id_for(make_plan(), "opening", "O001")
        viewer = _element(user, "ifc-viewer")
        with user:
            listeners = [item for item in viewer._event_listeners.values() if item.type == "pick"]
            for listener in listeners:
                viewer._handle_event({"listener_id": listener.id, "args": {"global_id": door_gid, "ifc_class": "IfcDoor",
                                                                          "ctrl": False, "point": [4.0, 1.0, -2.5]}})
        await _eventually(lambda: viewer.props.get("selected-many") == [door_gid])
        assert len(_element(user, "result-canvas").props["selection-overlay"]) == 1


@pytest.mark.anyio
async def test_variant_hint_sits_between_dxf_and_ifc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_assistant()
    project = _project(tmp_path, variant="pointcloud")
    Jobs(monkeypatch, project)
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-variant-hint")
        assert _order(user, ("result-canvas", "assistant-variant-hint", "ifc-card")) == [
            "result-canvas", "assistant-variant-hint", "ifc-card"]


# ------------------------------------------------------------------------------------------------- 5. payload / versions

def test_plan_payload_and_version_unchanged_without_edits(tmp_path: Path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    state = load_state(project)
    state.levels = [{"id": LEVEL_ID, "name": "EG", "floor_z": FLOOR_Z, "ceiling_z": 2.8, "checked": True}]
    state.mode_override = "indoor"
    save_state(project, state)
    controller = DxfPageController(project)
    result_file = project.joinpath(*CACHE, "result.json")
    stat = result_file.stat()
    assert controller.plan_version(LEVEL_ID, "ortho") == f"{stat.st_mtime_ns:x}-{stat.st_size:x}"
    assert controller.assistant_edits_version(LEVEL_ID) == ""
    expected = vector_payload(load_cached_result(project.joinpath(*CACHE)), grid_size=controller_grid())
    assert controller.plan_payload(LEVEL_ID, "ortho") == expected
    assert not (project / "products" / "assistant").exists()


def controller_grid() -> int:
    from samcloud.dxf.config import load_dxf_config

    return int(load_dxf_config().geometry.vector_grid_size)


def test_plan_payload_with_edits_is_read_only_and_versioned(tmp_path: Path) -> None:
    from samcloud.assistant.edits import EditOp, Proposal
    from samcloud.assistant.history import History
    from samcloud.assistant.paths import plan_model_path
    from samcloud.bim.plan_model import load_plan_model

    project = make_project(tmp_path, with_cloud=False)
    state = load_state(project)
    state.levels = [{"id": LEVEL_ID, "name": "EG", "floor_z": FLOOR_Z, "ceiling_z": 2.8, "checked": True}]
    state.mode_override = "indoor"
    save_state(project, state)
    controller = DxfPageController(project)
    version_before = controller.plan_version(LEVEL_ID, "ortho")
    proposal = Proposal(load_plan_model(plan_model_path(project, LEVEL_ID)))
    proposal.apply(EditOp("rename_room", "R002", {"name": "Gast"}))
    History.load(project, LEVEL_ID).accept(proposal, prompt="umbenennen", provider="ollama", model="dummy-model")
    version_after = controller.plan_version(LEVEL_ID, "ortho")
    assert version_after != version_before and version_after.startswith(version_before + "-e")
    watched = [project.joinpath(*CACHE, "plan_model.json"), project.joinpath(*CACHE, "result.json")]
    watched += sorted((project / "products" / "assistant").rglob("*.json"))
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in watched}
    texts = [entity["text"] for entity in controller.plan_payload(LEVEL_ID, "ortho")["layers"]["A-ANNO"]["entities"]]
    assert any(text.startswith("Gast") for text in texts)
    assert {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in watched} == before  # nothing written
    # the point-cloud variant is never patched
    assert controller.plan_version(LEVEL_ID, "pointcloud") == "none"


def test_broken_edits_fall_back_to_the_cached_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   caplog: pytest.LogCaptureFixture) -> None:
    project = make_project(tmp_path, with_cloud=False)
    state = load_state(project)
    state.levels = [{"id": LEVEL_ID, "name": "EG", "floor_z": FLOOR_Z, "ceiling_z": 2.8, "checked": True}]
    state.mode_override = "indoor"
    save_state(project, state)
    edits = project / "products" / "assistant" / LEVEL_ID / "edits.json"
    edits.parent.mkdir(parents=True)
    edits.write_text("{not json", encoding="utf-8")

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("broken history")

    monkeypatch.setattr("samcloud.assistant.history.edit_changes", boom)
    controller = DxfPageController(project)
    expected = vector_payload(load_cached_result(project.joinpath(*CACHE)), grid_size=controller_grid())
    with caplog.at_level(logging.WARNING):
        assert controller.plan_payload(LEVEL_ID, "ortho") == expected
    assert any("could not be drawn into the plan view" in record.getMessage() for record in caplog.records)


# ------------------------------------------------------------------------------------------------- 6. assistant export mode

def _patch_job_inputs(monkeypatch: pytest.MonkeyPatch, project: Path, *, fingerprint: str | None = None) -> None:
    """Cheap stand-ins for the LAS header and the DXF state; ``fingerprint`` = what the clean-plan check computes now."""
    result = load_cached_result(project.joinpath(*CACHE))
    level = {"id": LEVEL_ID, "name": result.level_name, "floor_z": 0.2, "ceiling_z": 2.8, "checked": True}
    state = SimpleNamespace(mode_override="indoor", levels=[level], view={"coordinates": "local"}, segment_source="combined")
    monkeypatch.setattr(jobs, "load_state", lambda _project: state)
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode="indoor", frame=result.frame))
    monkeypatch.setattr(jobs, "plan_settings_for", lambda *_a, **_k: None)
    current = result.settings_fingerprint if fingerprint is None else fingerprint
    monkeypatch.setattr("samcloud.ortho.plan.clean_plan_fingerprint", lambda *_a, **_k: current)
    monkeypatch.setattr("samcloud.ortho.segment.load_segment_result", lambda *_a, **_k: None)


def _run_export(project: Path, **kwargs: Any) -> dict[str, Any]:
    jobs.run_export(project, project / "cloud.ply", LEVEL_ID, lambda *_a: None, CancelToken(), LOGGER, mode="indoor", variant="ortho",
                    **kwargs)
    return json.loads((project / "products" / "dxf" / "export" / "last_export.json").read_text(encoding="utf-8"))


def _accept_rename(project: Path) -> None:
    from samcloud.assistant.edits import EditOp, Proposal
    from samcloud.assistant.history import History
    from samcloud.assistant.paths import plan_model_path
    from samcloud.bim.plan_model import load_plan_model

    proposal = Proposal(load_plan_model(plan_model_path(project, LEVEL_ID)))
    proposal.apply(EditOp("rename_room", "R002", {"name": "Gast"}))
    History.load(project, LEVEL_ID).accept(proposal, prompt="umbenennen", provider="ollama", model="dummy-model")


def _dxf_texts(path: Path) -> list[str]:
    import ezdxf

    return [str(entity.dxf.text) for entity in ezdxf.readfile(str(path)).modelspace() if entity.dxftype() == "TEXT"]


def test_stale_plan_blocks_the_normal_export_but_not_the_assistant_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                         caplog: pytest.LogCaptureFixture) -> None:
    project = make_project(tmp_path, with_cloud=False)
    _patch_job_inputs(monkeypatch, project, fingerprint="code-changed-since")
    _accept_rename(project)
    with pytest.raises(RuntimeError, match="is stale. Run clean_plan first"):
        jobs.run_export(project, project / "cloud.ply", LEVEL_ID, lambda *_a: None, CancelToken(), LOGGER, mode="indoor", variant="ortho")
    assert not (project / "products" / "dxf" / "export").exists()  # the normal export writes nothing
    with caplog.at_level(logging.WARNING):
        manifest = _run_export(project, allow_stale_plan=True)
    assert manifest["plan_stale"] is True
    assert manifest["assistant"][0]["applied"] and not manifest["assistant"][0]["error"]
    exported = next((project / "products" / "dxf" / "export").glob("*_clean.dxf"))
    assert any(text.startswith("Gast") for text in _dxf_texts(exported))  # outlines -> assistant edits -> DXF as usual
    assert any("uses it as it is" in record.getMessage() for record in caplog.records)


def test_assistant_export_of_a_fresh_plan_records_plan_stale_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = make_project(tmp_path, with_cloud=False)
    _patch_job_inputs(monkeypatch, project)
    manifest = _run_export(project, allow_stale_plan=True, language="en")
    assert manifest["plan_stale"] is False and manifest["language"] == "en"
    normal = _run_export(project)
    assert "plan_stale" not in normal  # the normal export manifest is unchanged


def test_assistant_export_without_segmentation_fingerprint_counts_as_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = make_project(tmp_path, with_cloud=False)
    _patch_job_inputs(monkeypatch, project)

    def no_segmentation(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("Run segmentation first")

    monkeypatch.setattr("samcloud.ortho.plan.clean_plan_fingerprint", no_segmentation)
    assert _run_export(project, allow_stale_plan=True)["plan_stale"] is True


def test_assistant_export_still_needs_the_cached_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project_without_clean_plan(tmp_path)
    level = {"id": LEVEL_ID, "name": "EG", "floor_z": 0.2, "ceiling_z": 2.8, "checked": True}
    state = SimpleNamespace(mode_override="indoor", levels=[level], view={"coordinates": "local"}, segment_source="combined")
    monkeypatch.setattr(jobs, "load_state", lambda _project: state)
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode="indoor", frame=None))
    monkeypatch.setattr(jobs, "plan_settings_for", lambda *_a, **_k: None)
    with pytest.raises(RuntimeError, match="No cached result"):
        _run_export(project, allow_stale_plan=True)


def test_cli_assistant_flag_reaches_run_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(jobs, "run_export", lambda *_a, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(jobs, "setup_job_logging", lambda _path: LOGGER)
    project = tmp_path / "cli_project"
    project.mkdir()
    base = ["jobs", "export", "--project", str(project), "--las", str(project / "c.las"), "--variant", "ortho", "--lang", "en"]
    monkeypatch.setattr(sys, "argv", [*base, "--assistant"])
    jobs.main()
    monkeypatch.setattr(sys, "argv", base)
    jobs.main()
    assert calls == [{"mode": None, "variant": "ortho", "language": "en", "allow_stale_plan": True},
                     {"mode": None, "variant": "ortho", "language": "en"}]


class _FakePopen:
    commands: list[list[str]] = []

    def __init__(self, command: list[str], **_kwargs: Any) -> None:
        _FakePopen.commands.append(list(command))

    def poll(self) -> int:
        return 0


@pytest.mark.parametrize(("action", "assistant", "flag"), [("export", True, True), ("export", False, False),
                                                           ("export_ifc", True, False), ("run_all", True, False)])
def test_runner_passes_the_assistant_flag_for_export_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str,
                                                         assistant: bool, flag: bool) -> None:
    _FakePopen.commands = []
    monkeypatch.setattr(job_runner.subprocess, "Popen", _FakePopen)
    runner = job_runner.DxfJobRunner(tmp_path / "runner_project")
    runner.start(action, tmp_path / "cloud.las", level=LEVEL_ID, assistant=assistant)
    assert ("--assistant" in _FakePopen.commands[-1]) is flag


def test_controller_assistant_export_is_always_the_ortho_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, ifc=False, variant="pointcloud")
    seen: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(DxfJobRunner, "start", lambda runner, action, las, level=None, **kwargs: seen.append((action, kwargs)))
    controller = DxfPageController(project)
    controller.start_job("export", LEVEL_ID, assistant=True)
    controller.start_job("export", LEVEL_ID)
    assert seen[0][1]["variant"] == "ortho" and seen[0][1]["assistant"] is True
    assert seen[1][1]["variant"] == "pointcloud" and "assistant" not in seen[1][1]


def _plain(text: str | None) -> str:
    """Thousands separators are non-breaking spaces in the UI; compare with plain spaces."""
    return str(text or "").replace(" ", " ")


def _subtree_text(user: Any, marker: str) -> str:
    """Texts of the element marked ``marker`` and all its descendants."""
    with user:
        return " ".join(str(getattr(item, "text", "") or "") for element in ElementFilter(marker=marker)
                        for item in [element, *element.descendants()])


# ------------------------------------------------------------------------------------------------- 7. queue while a job runs

def _capture_pages(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    from samcloud.app.pages import dxf as dxf_page

    pages: list[Any] = []
    original = dxf_page.DxfPage.__init__

    def init(self: Any, controller: Any) -> None:
        original(self, controller)
        pages.append(self)

    monkeypatch.setattr(dxf_page.DxfPage, "__init__", init)
    return pages


@pytest.mark.anyio
async def test_request_while_a_job_runs_replaces_the_queue_and_poll_starts_it(tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_assistant()
    project = _project(tmp_path)
    recorded = Jobs(monkeypatch, project)
    busy = {"running": True}
    recorded.last = ("clean_plan", "running", "")  # a job of the user is running when the page opens
    monkeypatch.setattr(DxfJobRunner, "is_running", lambda runner: busy["running"])
    pages = _capture_pages(monkeypatch)
    _rename_provider(monkeypatch)
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-panel")
        await _ask(user, "Raum umbenennen", "Raum R002 heisst jetzt Gast.")
        user.find(marker="assistant-accept").click()  # locked while the job runs: nothing accepted, nothing queued
        assert AssistantService(project, LEVEL_ID).edits() == []
        page = pages[-1]
        assert page.request_assistant_exports() is True  # e.g. accept / undo / retry arriving while the job runs
        assert page.request_assistant_exports() is True
        assert page.assistant_queue == [("export", LEVEL_ID), ("export_ifc", LEVEL_ID)]  # replaced, not appended
        await asyncio.sleep(1.3)
        assert recorded.calls == []  # nothing starts while the other job runs
        recorded.last = ("clean_plan", "done", "")
        busy["running"] = False
        await _eventually(lambda: recorded.actions() == [("export", LEVEL_ID), ("export_ifc", LEVEL_ID)], timeout=5.0)
        await asyncio.sleep(1.3)
        assert recorded.actions() == [("export", LEVEL_ID), ("export_ifc", LEVEL_ID)]  # each step started exactly once
        assert page.assistant_queue == [] and page.assistant_job is None
        assert page.assistant_outcome["state"] == "done"


# ------------------------------------------------------------------------------------------------- 8. failures of later steps

def _dxf_written(project: Path) -> None:
    """Side effect of the DXF export: an exported DXF plus last_export.json."""
    export_dir = project / "products" / "dxf" / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "assistant_synthetic_EG_clean.dxf").write_text("0\nEOF\n", encoding="utf-8")
    (export_dir / "last_export.json").write_text(json.dumps({"files": [], "plan_stale": False}), encoding="utf-8")


@pytest.mark.anyio
async def test_failed_ifc_export_keeps_the_dxf_and_offers_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_assistant()
    project = _project(tmp_path)
    recorded = Jobs(monkeypatch, project)
    recorded.effects["export"] = _dxf_written
    recorded.outcomes["export_ifc"] = [("failed", "IFC writer crashed")]
    pages = _capture_pages(monkeypatch)
    _rename_provider(monkeypatch)
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Raum umbenennen", "Raum R002 heisst jetzt Gast.")
        user.find(marker="assistant-accept").click()
        await user.should_see(marker="assistant-export-error", retries=60)
        await user.should_see("IFC writer crashed")
        await user.should_see(marker="assistant-export-retry")
        page = pages[-1]
        assert page.assistant_queue == [] and page.assistant_job is None
        assert recorded.actions() == [("export", LEVEL_ID), ("export_ifc", LEVEL_ID)]
        assert (project / "products" / "dxf" / "export" / "assistant_synthetic_EG_clean.dxf").is_file()  # the DXF stays
        assert any(text.startswith("Gast") for text in _payload_texts(project, monkeypatch))
        assert not _text(user, "assistant-export-status")
    assert [edit["status"] for edit in AssistantService(project, LEVEL_ID).edits()] == ["active"]


@pytest.mark.anyio
async def test_failed_compute_everything_shows_message_and_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_assistant()
    project = _project_without_clean_plan(tmp_path)
    recorded = Jobs(monkeypatch, project)
    recorded.outcomes["run_all"] = [("failed", "Build orthophoto first")]
    _rename_provider(monkeypatch)
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Raum umbenennen", "Raum R002 heisst jetzt Gast.")
        user.find(marker="assistant-accept").click()
        await user.should_see(marker="assistant-export-error", retries=60)
        await user.should_see("Build orthophoto first")
        user.find(marker="assistant-export-retry").click()
        await _eventually(lambda: recorded.actions() == [("run_all", LEVEL_ID), ("run_all", LEVEL_ID)], timeout=5.0)
    assert [edit["status"] for edit in AssistantService(project, LEVEL_ID).edits()] == ["active"]


# ------------------------------------------------------------------------------------------------- 9. context / fresh retry display

def test_usage_text_shows_the_context_use() -> None:
    from samcloud.app.components.assistant_panel import usage_text
    from samcloud.assistant.agent import TurnResult
    from samcloud.assistant.providers.base import Usage

    local = TurnResult(usage=Usage(1200, 80), provider="ollama", prompt_tokens=16414, context_window=32768)
    assert _plain(usage_text(local)) == "1 200 in / 80 out · 16 414 / 32 768 tokens context · local"
    cloud = TurnResult(usage=Usage(1200, 80), provider="openai", cost_usd=0.004, prompt_tokens=900, context_window=None)
    assert _plain(usage_text(cloud)) == "1 200 in / 80 out · ≈ $0.004"  # without a known window: no context part
    assert usage_text(TurnResult(provider="ollama")) is None
    only_context = usage_text(TurnResult(provider="ollama", prompt_tokens=2048, context_window=8192))
    assert _plain(only_context) == "0 in / 0 out · 2 048 / 8 192 tokens context · local"


def test_context_texts_in_german(german_ui: None) -> None:
    from samcloud.app.components.assistant_panel import error_title, usage_text
    from samcloud.assistant.agent import TurnResult

    result = TurnResult(provider="ollama", prompt_tokens=16414, context_window=32768)
    assert _plain(usage_text(result)) == "0 ein / 0 aus · 16 414 / 32 768 Token Kontext · lokal"
    assert error_title("context") == "Kontext voll"


def _answer(monkeypatch: pytest.MonkeyPatch, result: Any) -> None:
    monkeypatch.setattr(AssistantService, "ask", lambda self, prompt, selection, allow_manual=False: result)


@pytest.mark.anyio
async def test_fresh_retry_note_and_context_line_are_shown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.app.i18n import t
    from samcloud.assistant.agent import TurnResult
    from samcloud.assistant.providers.base import Usage

    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    Jobs(monkeypatch, project)
    _answer(monkeypatch, TurnResult(reply="Der Raum heisst Schlafen.", usage=Usage(16000, 120), provider="ollama", model="qwen",
                                    retried_fresh=True, notes=[t("assistant.agent.retried_fresh"), "Second note"],
                                    prompt_tokens=16414, context_window=32768))
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Wie heisst der Raum?", "Der Raum heisst Schlafen.")
        await user.should_see("The question was asked again in a new chat context.", marker="assistant-retried-fresh")
        assert "16 414 / 32 768 tokens context · local" in _plain(_text(user, "assistant-usage"))
        await user.should_see("Second note")
        with user:  # the fresh-context note is shown once, not again in the plain note list
            assert sum("new chat context" in str(getattr(element, "text", "")) for element in ElementFilter()) == 1


@pytest.mark.anyio
async def test_no_fresh_retry_note_without_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.assistant.agent import TurnResult

    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    Jobs(monkeypatch, project)
    _answer(monkeypatch, TurnResult(reply="Der Raum heisst Schlafen.", provider="ollama"))
    async with _dxf_user(project, monkeypatch) as user:
        await _ask(user, "Wie heisst der Raum?", "Der Raum heisst Schlafen.")
        assert not _has(user, "assistant-retried-fresh")
        assert not _has(user, "assistant-usage")


@pytest.mark.anyio
async def test_context_error_is_shown_like_other_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.app.components.assistant_panel import error_title
    from samcloud.app.i18n import t
    from samcloud.assistant.agent import TurnResult

    _enable_assistant()
    project = _project(tmp_path, ifc=False)
    Jobs(monkeypatch, project)
    message = t("assistant.agent.context_failed", detail=t("assistant.agent.context_full", tokens="40 000", window="8 192"))
    _answer(monkeypatch, TurnResult(error=message, error_kind="context", provider="ollama", retried_fresh=True,
                                    notes=[t("assistant.agent.retried_fresh")], prompt_tokens=40000, context_window=8192))
    async with _dxf_user(project, monkeypatch) as user:
        with user:
            next(iter(user.find(marker="assistant-input").elements)).value = "Alles auf einmal ändern"
        user.find(marker="assistant-send").click()
        await user.should_see(marker="assistant-error", retries=60)
        error_text = _subtree_text(user, "assistant-error")
        assert "Context full" in error_text and "Tip: choose a larger model" in error_text
        await user.should_see(marker="assistant-retried-fresh")
    assert error_title("context") == "Context full"
