"""Project share: the packaged DXF texts follow the app language (export step redone in-process when needed)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from samcloud.dxf import jobs
from samcloud.dxf.cancel import CancelToken
from samcloud.dxf.las_source import Frame
from samcloud.dxf.pipeline import save_cached_result
from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus
from samcloud.share import export, project
from samcloud.share import job as share_job

LOGGER = logging.getLogger("samcloud.tests.share_language")


def _result() -> DxfPlanResult:
    texts = [
        {"type": "text", "text": "Küche  12.4 m²", "point": [1.013, 2.027], "height": 0.12, "rotation": 0.0},
        {"type": "text", "text": "Gebüsch h=1.2 m", "point": [3.011, 1.019], "height": 0.4, "rotation": 0.0},
        {"type": "text", "text": "Stairs", "point": [5.017, 4.023], "height": 0.12, "rotation": 0.0},
    ]
    window = {"type": "insert", "name": "WINDOW", "point": [6.013, 0.011], "rotation": 0.0, "xscale": 1.2, "yscale": 0.25,
              "attributes": {"LABEL": "Window 1.20 m"}}
    layers = {"S-ANNO": texts, "S-SYMB": [window], "S-GRND": [{"type": "polyline", "points": [[0.0, 0.0], [8.0, 0.0], [8.0, 6.0]], "closed": True, "holes": []}]}
    return DxfPlanResult(
        project_name="ShareLang", level_name="site", mode="outdoor", profile_name=None, scale_factor=1.0,
        frame=Frame(np.zeros(3, dtype=np.float64), 1.0, np.eye(3, dtype=np.float64)), settings_fingerprint="fp-share-lang",
        layers=layers, statuses={name: LayerStatus(name, "ok", None, len(items)) for name, items in layers.items()},
        layer_colors={name: 7 for name in layers}, background=BackgroundImage(None, None, (-0.013, -0.017, 8.21, 6.13)), variant="ortho",
        blocks=["WINDOW"],
    )


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[str]]:
    """Outdoor project with a cached clean plan and the cheap job inputs patched; returns (dir, re-export calls)."""
    directory = tmp_path / "ShareLang"
    (directory / "products" / "classified").mkdir(parents=True)
    (directory / "products" / "classified" / "classified.ply").write_bytes(b"ply")
    result = _result()
    save_cached_result(result, directory / "products" / "dxf" / "site" / "ortho")
    bim = directory / "products" / "bim" / "site"
    bim.mkdir(parents=True)
    (bim / "model.glb").write_bytes(b"glTF")
    (bim / "props.json").write_text(json.dumps({"elements": {"id": {"ifc_class": "IfcRoof"}}}), encoding="utf-8")
    state = SimpleNamespace(mode_override="outdoor", levels=[], view={"coordinates": "local"}, segment_source="combined")
    monkeypatch.setattr(jobs, "load_state", lambda _project: state)
    monkeypatch.setattr(jobs, "read_las_info", lambda *_a, **_k: SimpleNamespace(mode="outdoor", frame=result.frame))
    monkeypatch.setattr(jobs, "plan_settings_for", lambda *_a, **_k: None)
    monkeypatch.setattr("samcloud.ortho.plan.clean_plan_fingerprint", lambda *_a, **_k: result.settings_fingerprint)
    monkeypatch.setattr("samcloud.ortho.segment.load_segment_result", lambda *_a, **_k: None)
    monkeypatch.setattr(project, "_dxf_point_source", lambda _directory: directory / "products" / "classified" / "classified.ply")
    calls: list[str] = []
    original = jobs.run_export

    def recording(*args: Any, **kwargs: Any) -> None:
        calls.append(str(kwargs.get("language")))
        original(*args, **kwargs)

    monkeypatch.setattr(jobs, "run_export", recording)
    return directory, calls


def _build(directory: Path, package: Path, monkeypatch: pytest.MonkeyPatch, messages: list[str]) -> dict[str, Any]:
    monkeypatch.setattr(export, "prepare_points", lambda *_a, **_k: (np.zeros((7, 3)), None))
    monkeypatch.setattr(export, "build_octree", lambda *_a, **_k: {})
    monkeypatch.setattr(export, "assemble_package", lambda *_a, **_k: None)
    monkeypatch.setattr(export, "scan_package", lambda *_a, **_k: [])
    return project.build_project_package(directory, package, config={}, check=lambda: None, max_points=100,
                                         progress_for_phase=lambda _index: (lambda _fraction, message="": messages.append(message)))


def _package_texts(package: Path) -> list[str]:
    data = json.loads((package / "data" / "dxf" / "site.json").read_text(encoding="utf-8"))
    return [item["s"] for layer in data["layers"] for item in layer["texts"]]


def test_share_reexports_a_german_dxf_in_english(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, calls = _project(tmp_path, monkeypatch)
    jobs.run_export(directory, directory / "cloud.las", None, lambda *_a: None, CancelToken(), LOGGER, variant="ortho", language="de")
    exported = directory / "products" / "dxf" / "export" / "ShareLang_site_clean.dxf"
    assert jobs.exported_file_language(directory, exported) == "de"
    calls.clear()

    messages: list[str] = []
    package = tmp_path / "out" / "package"
    _build(directory, package, monkeypatch, messages)  # the autouse fixture runs the UI in English

    assert calls == ["en"]
    assert jobs.exported_file_language(directory, exported) == "en"
    texts = _package_texts(package)
    assert "Kitchen  12.4 m²" in texts and "Shrub h=1.2 m" in texts and "Stairs" in texts and "Window 1.20 m" in texts
    assert not [text for text in texts if any(char in text for char in "äöüÄÖÜß")]
    assert any("unknown profile | scale 1" in text for text in texts)
    assert json.loads((package / "data" / "project.json").read_text(encoding="utf-8"))["lang"] == "en"
    assert "Preparing DXF" in messages and "Preparing IFC" in messages and "Re-exporting the DXF in the app language" in messages
    assert not [message for message in messages if "aufbereiten" in message]


def test_share_keeps_an_export_in_the_current_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, german_ui: None) -> None:
    directory, calls = _project(tmp_path, monkeypatch)
    jobs.run_export(directory, directory / "cloud.las", None, lambda *_a: None, CancelToken(), LOGGER, variant="ortho", language="de")
    calls.clear()
    messages: list[str] = []
    package = tmp_path / "out" / "package"
    _build(directory, package, monkeypatch, messages)
    assert calls == []
    texts = _package_texts(package)
    assert "Küche  12.4 m²" in texts and "Treppe" in texts and "Fenster 1.20 m" in texts
    assert json.loads((package / "data" / "project.json").read_text(encoding="utf-8"))["lang"] == "de"
    assert "DXF aufbereiten" in messages and "IFC aufbereiten" in messages


def test_share_without_clean_plan_cache_relabels_the_export_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, calls = _project(tmp_path, monkeypatch)
    jobs.run_export(directory, directory / "cloud.las", None, lambda *_a: None, CancelToken(), LOGGER, variant="ortho", language="de")
    (directory / "products" / "dxf" / "site" / "ortho" / "result.json").unlink()
    calls.clear()
    package = tmp_path / "out" / "package"
    _build(directory, package, monkeypatch, [])
    assert calls == []  # no cache to export from: the existing file is relabelled instead
    assert "Kitchen  12.4 m²" in _package_texts(package)
    assert jobs.exported_file_language(directory, directory / "products" / "dxf" / "export" / "ShareLang_site_clean.dxf") == "en"


def test_project_share_privacy_error_follows_the_ui_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, _calls = _project(tmp_path, monkeypatch)
    jobs.run_export(directory, directory / "cloud.las", None, lambda *_a: None, CancelToken(), LOGGER, variant="ortho", language="en")
    monkeypatch.setattr(export, "prepare_points", lambda *_a, **_k: (np.zeros((7, 3)), None))
    monkeypatch.setattr(export, "build_octree", lambda *_a, **_k: {})
    monkeypatch.setattr(export, "assemble_package", lambda *_a, **_k: None)
    monkeypatch.setattr(export, "scan_package", lambda *_a, **_k: ["image file: x.jpg"])
    with pytest.raises(RuntimeError, match=r"^Privacy check failed: image file: x\.jpg$"):
        project.build_project_package(directory, tmp_path / "out" / "package", config={}, check=lambda: None, max_points=100,
                                      progress_for_phase=lambda _index: (lambda _fraction, message="": None))
    assert share_job.PROJECT.phases[2][0] == "dxf"  # phase keys stay stable for the UI translation


def test_share_relabels_a_drone_dxf_without_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.dxf.text_language import dxf_file_language, translate_document_notes, translate_result
    from samcloud.dxf.writer import build_dxf_document

    directory, calls = _project(tmp_path, monkeypatch)
    (directory / "products" / "dxf" / "site" / "ortho" / "result.json").unlink()  # drone layout: no clean plan cache
    export_dir = directory / "products" / "dxf" / "export"
    export_dir.mkdir(parents=True)
    document = build_dxf_document(translate_result(_result(), "de")[0])
    translate_document_notes(document, "de")
    drone = export_dir / "ShareLang_Site_clean.dxf"
    document.saveas(drone)
    package = tmp_path / "out" / "package"
    _build(directory, package, monkeypatch, [])
    assert calls == []  # relabelled in place, no export
    assert dxf_file_language(drone) == "en"
    texts = _package_texts(package)
    assert "Kitchen  12.4 m²" in texts and "Shrub h=1.2 m" in texts and "Stairs" in texts and "Window 1.20 m" in texts
    assert any("unknown profile | scale 1" in text for text in texts)
