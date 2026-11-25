"""Outdoor (drone) AI assistant on the /dxf page: one chat card between the DXF preview and the IFC viewer, picks in the
SVG preview (LV95) and in the IFC viewer, coloured proposal, accept / undo / full control with the fast environment run
(``scene,export,qa`` with ``cad_only``) and the in-place reload of both viewers.

Synthetic drone project only (``assistant_outdoor_synthetic``, fake LV95 origin); the environment job is replaced by an
in-process run of the real chain in a thread (no subprocess), the provider by a scripted fake.  No network.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml
from nicegui import ElementFilter, ui

from assistant_outdoor_synthetic import EMPTY_POINT, FENCE_AXIS, TREE_1, lv95, make_outdoor_project
from samcloud.app.components import assistant_panel as panel_module
from samcloud.app.pages import dxf as dxf_page
from samcloud.assistant.domains.outdoor_selection import global_id_of
from samcloud.drone import dxf_preview, env_job
from test_assistant_agent import FakeProvider, call, say
from test_assistant_panel import _chips, _dxf_user, _element, _emit, _enable_assistant, _has, _set, _use

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

TREE_GID = global_id_of(TREE_1["id"])
TREE_REF = "B2"  # short display label of TREE_1 (trees numbered north to south; TREE_2 lies further north)
AMBER = "#ffb74d"


@pytest.fixture(autouse=True)
def _private_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the real user's AI settings / credentials."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


# ------------------------------------------------------------------------------------------------------ project

_BASE: dict[str, Path] = {}


def _run_cad_only(project: Path) -> dict[str, Any]:
    from samcloud.drone import run_environment as env
    from samcloud.drone.contracts import load_drone_config

    return env.run_environment(project, steps="scene,export,qa", cad_only=True, config=load_drone_config())


def _published_base() -> Path:
    """The synthetic drone project with a classified cloud (the page needs a point source), published DXF / IFC / viewer
    assets from one ``cad_only`` run and the SVG preview (one per test process; copied per test, never used directly).

    Built from scratch (not from the shared synthetic project other test modules read) so no other test's files leak in."""
    if "base" in _BASE and (_BASE["base"] / "products" / "bim" / "site" / "model.glb").is_file():
        assert not (_BASE["base"] / "products" / "assistant").exists()
        return _BASE["base"]
    from dxf_synthetic import write_outdoor_scene

    root = Path(os.path.realpath(os.environ["SAMCLOUD_TEST_WORKDIR"])) / f"outdoor_ui_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    project = make_outdoor_project(root, "outdoor_ui")
    document = {"project": {"name": "outdoor_ui"}, "mode": "outdoor", "acquisition": {"type": "drone_gps"},
                "settings": {"classification": {"class_set": "outdoor"}}, "workflow": {}}
    (project / "project.samcloud.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    write_outdoor_scene(project / "products" / "classified" / "classified.las")
    state = _run_cad_only(project)
    assert state["steps"]["export"]["status"] == "done"
    dxf = next((project / "products" / "dxf" / "export").glob("*Site*clean.dxf"))
    dxf_preview.render_dxf_svg(dxf, dxf.parent / dxf_preview.SVG_NAME)
    assert not (project / "products" / "assistant").exists()  # building and exporting writes no assistant files
    _BASE["base"] = project
    return project


@pytest.fixture
def project(tmp_path: Path) -> Path:
    destination = tmp_path / "outdoor_ui"
    shutil.copytree(_published_base(), destination)
    return destination


class EnvJobs:
    """Replaces ``env_job.start_environment_job``: records the call and runs the real chain in a thread (``running``
    with this process' pid until it is done).  ``fail_next`` makes the next run end ``failed`` right away."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_next: list[str] = []
        self.threads: list[threading.Thread] = []
        monkeypatch.setattr(env_job, "start_environment_job", self.start)

    def start(self, project_dir: Any, rerun_all: bool = False, *, steps: Any = None, cad_only: bool = False) -> dict[str, Any]:
        project = Path(project_dir)
        self.calls.append({"project": project, "rerun_all": rerun_all, "steps": steps, "cad_only": cad_only})
        status = env_job.job_status_path(project)
        if self.fail_next:
            record = {"state": "failed", "pid": None, "started_at": env_job._now(), "finished_at": env_job._now(),
                      "error": self.fail_next.pop(0), "rerun_all": False}
            env_job._write_json(status, record)
            return record
        record = {"state": "running", "pid": os.getpid(), "started_at": env_job._now(), "finished_at": None, "error": None,
                  "rerun_all": False, "steps": steps, "cad_only": cad_only}
        env_job._write_json(status, record)

        def work() -> None:
            try:
                _run_cad_only(project)
                env_job._write_json(status, {**record, "state": "finished", "pid": None, "finished_at": env_job._now()})
            except Exception as error:  # noqa: BLE001 - reported to the page like a failed job
                env_job._write_json(status, {**record, "state": "failed", "pid": None, "error": type(error).__name__})

        thread = threading.Thread(target=work, daemon=True)
        self.threads.append(thread)
        thread.start()
        return record

    def join(self) -> None:
        for thread in self.threads:
            thread.join(timeout=180)


@pytest.fixture
def envjobs(monkeypatch: pytest.MonkeyPatch) -> EnvJobs:
    jobs = EnvJobs(monkeypatch)
    yield jobs
    jobs.join()


@pytest.fixture
def pages(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """The DxfPage objects built during the test (to read the shown preview URL)."""
    built: list[Any] = []
    original = dxf_page.DxfPage.build

    def build(self: Any) -> None:
        built.append(self)
        original(self)

    monkeypatch.setattr(dxf_page.DxfPage, "build", build)
    return built


@pytest.fixture
def js(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """JavaScript sent by the preview's assistant layer."""
    sent: list[str] = []
    original = dxf_page._preview_js

    def record(code: str, element: Any = None) -> None:
        sent.append(code)
        original(code, element)

    monkeypatch.setattr(dxf_page, "_preview_js", record)
    return sent


# ------------------------------------------------------------------------------------------------------ helpers

def _viewport(user: Any) -> Any:
    return _element(user, "dxf-preview-view")


def _click_svg(user: Any, e: float, n: float, ctrl: bool = False) -> None:
    """A click in the SVG preview as the browser reports it (LV95 E / N plus the SVG coordinates)."""
    origin = tuple(json.loads(_viewport(user).props["data-sam-origin"]))
    sx, sy = dxf_preview.dxf_to_svg(origin, e, n)
    with user:
        _emit(_viewport(user), "sampick", {"detail": {"e": e, "n": n, "sx": sx, "sy": sy, "ctrl": ctrl}})


def _ifc_pick(user: Any, global_id: str, local_xyz: tuple[float, float, float], ctrl: bool = False) -> None:
    x, y, z = local_xyz
    with user:
        _emit(_element(user, "ifc-viewer"), "pick", {"global_id": global_id, "ifc_class": "IfcGeographicElement", "ctrl": ctrl,
                                                     "point": [x, z, -y]})


def _last_overlay(js: list[str]) -> dict[str, Any]:
    calls = [code for code in js if code.startswith("samDxfpOverlay(")]
    assert calls, "no overlay sent"
    return json.loads(calls[-1].split(", ", 1)[1].rstrip(")"))


def _order(user: Any, markers: tuple[str, ...]) -> list[str]:
    seen: list[str] = []
    with user:
        for element in ElementFilter():
            for marker in element._markers:
                if marker in markers and marker not in seen:
                    seen.append(marker)
    return seen


async def _eventually(check: Any, timeout: float = 90.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not check():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.2)


async def _ask(user: Any, text: str, expect_marker: str) -> None:
    _set(user, "assistant-input", text)
    user.find(marker="assistant-send").click()
    await _eventually(lambda: _has(user, expect_marker) or _has(user, "assistant-error"), timeout=180)
    assert not _has(user, "assistant-error"), _element(user, "assistant-error").text if _has(user, "assistant-error") else ""


def _centroid(ring: list[list[float]]) -> tuple[float, float]:
    return sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring)


def _tree_height_in_dxf(project: Path) -> float:
    import re

    import ezdxf

    dxf = next((project / "products" / "dxf" / "export").glob("*Site*clean.dxf"))
    e, n = lv95(*TREE_1["xy"])
    best = min(ezdxf.readfile(dxf).modelspace().query('INSERT[name=="TREE"]'),
               key=lambda item: (item.dxf.insert.x - e) ** 2 + (item.dxf.insert.y - n) ** 2)
    text = {attrib.dxf.tag: attrib.dxf.text for attrib in best.attribs}["HEIGHT"]
    return float(re.search(r"-?\d+(\.\d+)?", text).group(0))


# ------------------------------------------------------------------------------------------------------ pure helpers

def test_svg_coordinate_helpers_and_env_action() -> None:
    origin = dxf_preview.svg_origin({"origin": [2600000.0, 1200031.0]})
    assert origin == (2600000.0, 1200031.0)
    assert dxf_preview.svg_to_dxf(origin, 12.5, 4.25) == (2600012.5, 1200026.75)
    assert dxf_preview.dxf_to_svg(origin, 2600012.5, 1200026.75) == (12.5, 4.25)
    assert dxf_preview.svg_origin({}) is None and dxf_preview.svg_origin({"origin": ["x"]}) is None
    assert dxf_page.parse_env_action("env:scene,export,qa:cad_only") == ("scene,export,qa", True)
    assert dxf_page.parse_env_action("env:scene,export,qa") == ("scene,export,qa", False)
    assert dxf_page.parse_env_action("export") is None
    assert panel_module.rings_of([[0, 0], [1, 0], [1, 1]]) == [[[0, 0], [1, 0], [1, 1]]]
    assert panel_module.rings_of([[[0, 0], [1, 0]], [[2, 2], [3, 3]]]) == [[[0, 0], [1, 0]], [[2, 2], [3, 3]]]
    assert panel_module.rings_of(None) == []
    assert panel_module.shift_ring([[1.0, 2.0]], [10.0, 20.0, 5.0]) == [[11.0, 22.0]]
    box = panel_module.model_box([[[0.0, 0.0], [1.0, 1.0]], [[4.0, 2.0], [5.0, 3.0]]], 0.5, 2.0, [0.0, 0.0, 0.0], AMBER)
    assert box["min"] == [0.0, 0.5, -3.0] and box["max"] == [5.0, 2.0, 0.0]


# ------------------------------------------------------------------------------------------------------ gating

@pytest.mark.anyio
async def test_assistant_off_leaves_the_drone_page_unchanged(project: Path, monkeypatch: pytest.MonkeyPatch, envjobs: EnvJobs,
                                                             js: list[str]) -> None:
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="dxf-preview-svg", retries=100)
        await user.should_see(marker="ifc-viewer")
        assert not _has(user, "assistant-panel") and not _has(user, "assistant-unsupported-hint")
        props = dict(_viewport(user).props)
        listeners = [listener.type for listener in _viewport(user)._event_listeners.values()]
        viewer = dict(_element(user, "ifc-viewer").props)
    assert not any(name.startswith("data-sam") for name in props) and "tabindex" not in props
    assert "sampick" not in listeners
    assert not any(name in viewer for name in ("pick-enabled", "selected-many", "highlight", "boxes"))
    assert js == [] and envjobs.calls == []
    assert not (project / "products" / "assistant").exists()


# ------------------------------------------------------------------------------------------------------ selection

@pytest.mark.anyio
async def test_picks_in_svg_preview_and_ifc_share_one_selection(project: Path, monkeypatch: pytest.MonkeyPatch, envjobs: EnvJobs,
                                                               js: list[str], german_ui: None) -> None:
    _enable_assistant()
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-panel")
        await user.should_see(marker="dxf-preview-svg", retries=100)
        assert _order(user, ("dxf-preview-card", "assistant-panel", "ifc-card")) == ["dxf-preview-card", "assistant-panel", "ifc-card"]
        await user.should_see(marker="assistant-outdoor-scope")
        meta = dxf_preview.load_svg_meta(next((project / "products" / "dxf" / "export").glob("preview.svg")))
        viewport = _viewport(user)
        assert viewport.props["data-sam-pick"] == "1"
        assert tuple(json.loads(viewport.props["data-sam-origin"])) == dxf_preview.svg_origin(meta)
        assert _element(user, "ifc-viewer").props["pick-enabled"] is True

        _click_svg(user, *lv95(TREE_1["xy"][0] + 0.4, TREE_1["xy"][1] - 0.3))  # inside the crown
        chips = await _chips(user)
        assert len(chips) == 1 and chips[0].startswith(f"Baum {TREE_REF} (")  # the short label, not the uuid
        ring = _last_overlay(js)["selection"][0]["ring"]
        cx, cy = _centroid(ring)
        assert (cx, cy) == pytest.approx(lv95(*TREE_1["xy"]), abs=0.05)  # overlay in LV95
        assert _element(user, "ifc-viewer").props["selected-many"] == [TREE_GID]

        fence = FENCE_AXIS[1]
        _click_svg(user, *lv95(fence[0] + 0.05, fence[1]), ctrl=True)  # Ctrl+click adds the fence
        chips = await _chips(user)
        assert len(chips) == 2 and chips[1] == "Zaun Z1 (1.20 m)"
        assert len(_last_overlay(js)["selection"]) == 2

        e, n = lv95(*EMPTY_POINT)
        _click_svg(user, e, n)  # a plain click into empty space: one point chip in LV95
        chips = await _chips(user)
        assert chips == [f"Punkt E {e:.2f} / N {n:.2f}"]
        assert _last_overlay(js)["selection"][0]["point"] == pytest.approx([e, n], abs=0.01)
        assert _element(user, "ifc-viewer").props["selected-many"] == []

        _ifc_pick(user, TREE_GID, (TREE_1["xy"][0], TREE_1["xy"][1], 5.0))  # the same tree in the IFC viewer
        chips = await _chips(user)
        assert len(chips) == 1 and chips[0].startswith(f"Baum {TREE_REF} (")
        assert _element(user, "ifc-viewer").props["selected-many"] == [TREE_GID]
    assert envjobs.calls == []


# ------------------------------------------------------------------------------------------------------ proposal -> accept

@pytest.mark.anyio
async def test_proposal_colours_accept_and_undo_refresh_both_viewers(project: Path, monkeypatch: pytest.MonkeyPatch,
                                                                    envjobs: EnvJobs, js: list[str], pages: list[Any]) -> None:
    _enable_assistant()
    # the model names the tree by its short label: the tool resolves it, the history stores the real id
    _use(monkeypatch, FakeProvider([call("set_tree", target=TREE_REF, height_delta=2.0), say("Tree B2 is 2 m taller now.")]))
    before = _tree_height_in_dxf(project)
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-panel")
        await user.should_see(marker="dxf-preview-svg", retries=100)
        page = pages[-1]
        _click_svg(user, *lv95(*TREE_1["xy"]))
        await _ask(user, "make this tree 2 m taller", "assistant-summary")
        await user.should_see(f"Tree {TREE_REF}")
        await user.should_see(f"Height {before:.2f} → {before + 2.0:.2f} m")
        preview = _last_overlay(js)["preview"]
        assert preview and all(item["color"] == AMBER for item in preview)
        assert any(item.get("dashed") for item in preview) and any(item.get("fill") for item in preview)
        viewer = _element(user, "ifc-viewer")
        assert viewer.props["highlight"] == {TREE_GID: AMBER}
        box = viewer.props["boxes"][0]
        assert box["min"][1] < 2.0 and box["max"][1] == pytest.approx(box["min"][1] + before + 2.0, abs=0.3)  # local heights

        url, src = page.dxfp_url, viewer.props["src"]
        assert url and "?v=" in url
        user.find(marker="assistant-accept").click()
        await _eventually(lambda: len(envjobs.calls) == 1, timeout=10)
        assert envjobs.calls[0]["steps"] == "scene,export,qa" and envjobs.calls[0]["cad_only"] is True
        assert envjobs.calls[0]["rerun_all"] is False
        await _eventually(lambda: page.dxfp_url != url and _element(user, "ifc-viewer").props["src"] != src)
        await user.should_see("Done – DXF and IFC show the change.", retries=50)
        assert _last_overlay(js)["preview"] == []
        assert _tree_height_in_dxf(project) == pytest.approx(before + 2.0, abs=0.05)

        edit = page.assistant.service.edits()[-1]
        edit_id = edit["id"]
        assert edit["ops"][0]["target"] == TREE_1["id"]  # stored with the real id
        history = _element(user, f"assistant-edit-{edit_id}")
        with user:
            texts = [element.text for element in ElementFilter(kind=ui.label).within(instance=history)]
        assert any(f"Tree {TREE_REF} (" in text and f"Height {before:.2f} → {before + 2.0:.2f} m" in text for text in texts), texts
        url, src = page.dxfp_url, _element(user, "ifc-viewer").props["src"]
        user.find(marker=f"assistant-undo-{edit_id}").click()
        await _eventually(lambda: len(envjobs.calls) == 2, timeout=10)
        await _eventually(lambda: page.dxfp_url != url and _element(user, "ifc-viewer").props["src"] != src)
        await user.should_see("Done – DXF and IFC show the change.", retries=50)
    assert _tree_height_in_dxf(project) == pytest.approx(before, abs=0.05)


@pytest.mark.anyio
async def test_full_control_applies_directly_and_refreshes(project: Path, monkeypatch: pytest.MonkeyPatch, envjobs: EnvJobs,
                                                           pages: list[Any]) -> None:
    _enable_assistant(full_control=True)
    _use(monkeypatch, FakeProvider([call("set_tree", target=TREE_1["id"], height=11.0), say("Set to 11 m.")]))
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-full-mode")
        await user.should_see(marker="dxf-preview-svg", retries=100)
        page = pages[-1]
        url, src = page.dxfp_url, _element(user, "ifc-viewer").props["src"]
        await _ask(user, "set the tree near the house to 11 m", "assistant-applied")
        assert not _has(user, "assistant-accept")
        await _eventually(lambda: len(envjobs.calls) == 1, timeout=10)
        assert envjobs.calls[0]["cad_only"] is True and envjobs.calls[0]["steps"] == "scene,export,qa"
        await _eventually(lambda: page.dxfp_url != url and _element(user, "ifc-viewer").props["src"] != src)
        await user.should_see("Done – DXF and IFC show the change.", retries=50)
    assert _tree_height_in_dxf(project) == pytest.approx(11.0, abs=0.05)


@pytest.mark.anyio
async def test_failed_update_keeps_the_edit_and_retries(project: Path, monkeypatch: pytest.MonkeyPatch, envjobs: EnvJobs,
                                                        pages: list[Any]) -> None:
    _enable_assistant()
    _use(monkeypatch, FakeProvider([call("set_tree", target=TREE_1["id"], height_delta=1.0), say("One metre taller.")]))
    envjobs.fail_next.append("worker crashed")
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-panel")
        await user.should_see(marker="dxf-preview-svg", retries=100)
        page = pages[-1]
        await _ask(user, "tree one metre taller", "assistant-summary")
        user.find(marker="assistant-accept").click()
        await _eventually(lambda: _has(user, "assistant-export-error"), timeout=15)
        await user.should_see("worker crashed")
        assert len(page.assistant.service.edits()) == 1 and page.assistant.service.edits()[0]["status"] == "active"
        url = page.dxfp_url
        user.find(marker="assistant-export-retry").click()
        await _eventually(lambda: len(envjobs.calls) == 2, timeout=10)
        assert envjobs.calls[1]["cad_only"] is True
        await _eventually(lambda: page.dxfp_url != url)
        await user.should_see("Done – DXF and IFC show the change.", retries=50)
        assert not _has(user, "assistant-export-error")


# ------------------------------------------------------------------------------------------------------ rebuild during a turn

@pytest.mark.anyio
async def test_turn_result_reaches_the_panel_after_a_rebuild(project: Path, monkeypatch: pytest.MonkeyPatch, envjobs: EnvJobs,
                                                            pages: list[Any]) -> None:
    """The page body (and then a whole new page) is rebuilt while the model works: the reply and the proposal appear in
    the live panel afterwards, without a reload."""
    _enable_assistant()
    release = threading.Event()
    started = threading.Event()

    def blocked(_messages: Any) -> Any:
        started.set()
        assert release.wait(60)
        return call("set_tree", target=TREE_REF, height_delta=1.0)

    _use(monkeypatch, FakeProvider([blocked, say("Tree B2 is one metre taller.")]))
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-panel")
        await user.should_see(marker="dxf-preview-svg", retries=100)
        page = pages[-1]
        _set(user, "assistant-input", "tree one metre taller")
        user.find(marker="assistant-send").click()
        await _eventually(started.is_set, timeout=30)
        await user.should_see(marker="assistant-busy")
        with user:
            page.body.refresh()  # e.g. an environment poll or a language refresh rebuilds the page body
        await asyncio.sleep(0.2)
        await user.should_see(marker="assistant-busy")  # the rebuilt card still waits for the turn
        release.set()
        await _eventually(lambda: _has(user, "assistant-summary"), timeout=120)
        await user.should_see("Tree B2 is one metre taller.")
        assert _has(user, "assistant-accept") and not _has(user, "assistant-busy")

    # a second question, and this time the whole page is opened again (new page, new panel) while the model works
    release.clear()
    started.clear()
    _use(monkeypatch, FakeProvider([blocked, say("Done, another metre.")]))
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see(marker="assistant-panel")
        user.find(marker="assistant-discard").click()
        _set(user, "assistant-input", "one more metre")
        user.find(marker="assistant-send").click()
        await _eventually(started.is_set, timeout=30)
        await user.open("/dxf")  # a new page while the turn runs (reconnect / reload)
        await user.should_see(marker="assistant-busy")
        release.set()
        await _eventually(lambda: _has(user, "assistant-summary"), timeout=120)
        await user.should_see("Done, another metre.")
        assert len(pages) >= 3 and pages[-1].assistant is not pages[-2].assistant
