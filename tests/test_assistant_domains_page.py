"""/dxf page: one hint line between DXF and IFC when the assistant is on but the project type is not supported (yet)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from nicegui import ElementFilter

from samcloud.app.i18n import t
from samcloud.dxf.state import load_state, save_state
from test_assistant_panel import _dxf_user, _enable_assistant, _has, jobs  # noqa: F401,F811 - fixture re-export
from test_dxf_gui import _result_project

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

HINT = "assistant-unsupported-hint"


@pytest.fixture(autouse=True)
def _private_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the real user's AI settings / credentials."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _order(user: Any, markers: tuple[str, ...]) -> list[str]:
    seen: list[str] = []
    with user:
        for element in ElementFilter():
            for marker in element._markers:
                if marker in markers and marker not in seen:
                    seen.append(marker)
    return seen


def _ortho_outdoor(tmp_path: Path, name: str) -> Path:
    project = _result_project(tmp_path, name)
    state = load_state(project)
    state.plan_variant = "ortho"
    save_state(project, state)
    return project


def _make_drone(project: Path) -> None:
    project_file = project / "project.samcloud.yml"
    document = yaml.safe_load(project_file.read_text(encoding="utf-8")) or {}
    document["acquisition"] = {"type": "drone_gps"}
    project_file.write_text(yaml.safe_dump(document), encoding="utf-8")


@pytest.mark.anyio
async def test_drone_project_without_site_model_shows_the_no_scene_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                       caplog: pytest.LogCaptureFixture, jobs: list) -> None:
    project = _ortho_outdoor(tmp_path, "DroneHint")
    _make_drone(project)
    async with _dxf_user(project, monkeypatch) as user:  # assistant off: no hint
        await user.should_see(marker="ifc-card")
        assert not _has(user, HINT) and not _has(user, "assistant-panel")
    _enable_assistant()
    with caplog.at_level(logging.WARNING):
        async with _dxf_user(project, monkeypatch) as user:
            # the outdoor domain is registered, but without scene.pkl it has no site model to edit yet
            await user.should_see(t("assistant.agent.outdoor_no_scene"), marker=HINT)
            assert not _has(user, "assistant-panel") and not _has(user, "assistant-variant-hint")
            order = _order(user, ("dxf-preview-card", HINT, "ifc-card"))
            assert order[-2:] == [HINT, "ifc-card"]
            with user:
                assert len(list(ElementFilter(marker=HINT))) == 1
    assert jobs == []


@pytest.mark.anyio
async def test_ortho_outdoor_without_environment_chain_shows_its_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                     jobs: list) -> None:
    project = _ortho_outdoor(tmp_path, "OrthoHint")
    _enable_assistant()
    async with _dxf_user(project, monkeypatch) as user:
        await user.should_see("environment", marker=HINT)
        assert not _has(user, "assistant-panel") and not _has(user, "assistant-variant-hint")
        assert "pick-enabled" not in next(iter(user.find(marker="result-canvas").elements)).props
        assert _order(user, ("result-canvas", HINT, "ifc-card")) == ["result-canvas", HINT, "ifc-card"]
