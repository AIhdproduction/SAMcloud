"""Camera-video projects must find their extracted frames."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from samcloud.ortho.cameras import _project_image_roots, resolve_image_path
from samcloud.viewer.cameras import _source_roots


def _project(tmp_path: Path, with_key: bool) -> Path:
    project = tmp_path / "video-project"
    frames = project / "inputs" / "video_frames"
    frames.mkdir(parents=True)
    (frames / "frame_000001.jpg").write_bytes(b"x")
    acquisition: dict = {"type": "camera", "sources": {"image_directory": ""}}
    if with_key:
        acquisition["video"] = {"frame_directory": "inputs/video_frames"}
    (project / "project.samcloud.yml").write_text(yaml.safe_dump({"acquisition": acquisition}), encoding="utf-8")
    _project_image_roots.cache_clear()
    return project


@pytest.mark.parametrize("with_key", [True, False])
def test_ortho_resolves_video_frames(tmp_path: Path, with_key: bool) -> None:
    project = _project(tmp_path, with_key)
    found = resolve_image_path(project, "frame_000001.jpg")
    assert found is not None and found.name == "frame_000001.jpg"
    assert resolve_image_path(project, "missing.jpg") is None


@pytest.mark.parametrize("with_key", [True, False])
def test_viewer_roots_include_video_frames(tmp_path: Path, with_key: bool) -> None:
    project = _project(tmp_path, with_key)
    assert (project / "inputs" / "video_frames").resolve() in _source_roots(project)
