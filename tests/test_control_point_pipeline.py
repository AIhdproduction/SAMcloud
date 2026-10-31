"""Regression tests for the control-point pipeline boundary and resume path."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from samcloud.engine import pipeline
from samcloud.engine.run_summary import RunSummary


def _write_text_model(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cameras.txt").write_text("1 PINHOLE 10 10 5 5 5 5\n", encoding="utf-8")
    (directory / "images.txt").write_text("1 1 0 0 0 0 0 0 1 image.jpg\n\n", encoding="utf-8")
    (directory / "points3D.txt").write_text("1 0 0 1 255 255 255 0 1 0\n", encoding="utf-8")
    (directory / "rigs.txt").write_text("1 1 CAMERA 1\n", encoding="utf-8")
    (directory / "frames.txt").write_text("1 1 1 0 0 0 0 0 0 1 CAMERA 1 1\n", encoding="utf-8")


def _fake_run(commands: list[list[str]], cache_states: list[bool] | None = None):
    def run(command, **_kwargs):
        commands.append(command)
        action = command[1]
        if action == "mapper":
            _write_text_model(Path(command[command.index("--output_path") + 1]) / "0")
        elif action == "image_undistorter":
            output = Path(command[command.index("--output_path") + 1])
            (output / "images").mkdir(parents=True, exist_ok=True)
            _write_text_model(output / "sparse")
        elif action == "model_converter":
            output = Path(command[command.index("--output_path") + 1])
            assert output.is_dir(), "COLMAP requires a pre-existing model_converter output directory"
            _write_text_model(output)
        elif action == "patch_match_stereo":
            workspace = Path(command[command.index("--workspace_path") + 1])
            if cache_states is not None:
                cache_states.append(
                    not (workspace / "stereo" / "depth_maps" / "stale.bin").exists()
                    and not (workspace / "stereo" / "normal_maps" / "stale.bin").exists()
                )
            (workspace / "stereo" / "depth_maps").mkdir(parents=True, exist_ok=True)
        elif action == "stereo_fusion":
            Path(command[command.index("--output_path") + 1]).write_bytes(b"ply")
    return run


def test_after_alignment_stops_after_undistortion_and_text_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images = tmp_path / "images"; images.mkdir(); (images / "image.jpg").write_bytes(b"image")
    commands: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run", _fake_run(commands))
    monkeypatch.setattr(pipeline, "find_colmap_exe", lambda _: "colmap")
    monkeypatch.setattr(sys, "argv", ["pipeline", "--capture-type", "camera", "--images", str(images), "--outdir", str(tmp_path / "products"), "--control-points-mode", "after_alignment", "--skip-export"])
    pipeline.main()
    actions = [command[1] for command in commands]
    assert "image_undistorter" in actions and "model_converter" in actions
    assert "patch_match_stereo" not in actions


def _resume_args(control_points: Path) -> SimpleNamespace:
    return SimpleNamespace(control_points_dir=str(control_points), control_points_crs="EPSG:4979", colmap_bin="colmap", skip_export=True)


def test_resume_uses_pristine_backup_clears_cache_and_never_repeats_alignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outdir, dense, control_points = tmp_path / "products", tmp_path / "products" / "dense", tmp_path / "control_points"
    _write_text_model(dense / "sparse"); _write_text_model(dense / "sparse_txt")
    (dense / "images").mkdir(); (dense / "stereo" / "depth_maps").mkdir(parents=True); (dense / "stereo" / "normal_maps").mkdir()
    control_points.mkdir()
    (dense / "stereo" / "depth_maps" / "stale.bin").write_bytes(b"stale")
    (dense / "stereo" / "normal_maps" / "stale.bin").write_bytes(b"stale")
    commands: list[list[str]] = []; received: list[Path] = []; transformed: list[Path] = []; cache_states: list[bool] = []
    monkeypatch.setattr(pipeline, "run", _fake_run(commands, cache_states)); monkeypatch.setattr(pipeline, "find_colmap_exe", lambda _: "colmap")
    def georeference(model, *_args, **_kwargs):
        received.append(Path(model)); return ({"scale": 1., "rotation": np.eye(3).tolist(), "translation": [0., 0., 0.], "pairs": 3, "residual_m": {"mean": 0., "p95": 0., "max": 0.}}, {"points": []}, (1., np.eye(3), np.zeros(3)))
    def transform(source, destination, *_args): transformed.append(Path(source)); _write_text_model(Path(destination))
    monkeypatch.setattr(pipeline, "georeference_from_control_points", georeference)
    monkeypatch.setattr(pipeline, "transform_text_model", transform)
    monkeypatch.setattr(pipeline, "_classify_and_finish", lambda *_args: None)
    summary = RunSummary(outdir, {})
    for index in range(2):
        pipeline._resume_after_control_points(_resume_args(control_points), outdir, tmp_path / "reports", summary, None, {}, 10, 1, 0, outdir / "classified.ply")
        if index == 0:
            (dense / "stereo" / "depth_maps" / "stale.bin").write_bytes(b"stale")
            (dense / "stereo" / "normal_maps").mkdir(parents=True, exist_ok=True)
            (dense / "stereo" / "normal_maps" / "stale.bin").write_bytes(b"stale")
    pristine = dense / "sparse_txt_pre_gcp"
    assert received == [pristine, pristine]
    assert transformed == [pristine, pristine]
    assert cache_states == [True, True]
    actions = [command[1] for command in commands]
    assert "patch_match_stereo" in actions and "stereo_fusion" in actions
    assert not {"feature_extractor", "exhaustive_matcher", "mapper", "image_undistorter"}.intersection(actions)


def test_resume_rejects_insufficient_control_points_before_dense_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outdir, dense, control_points = tmp_path / "products", tmp_path / "products" / "dense", tmp_path / "control_points"
    _write_text_model(dense / "sparse"); _write_text_model(dense / "sparse_txt"); (dense / "images").mkdir(); control_points.mkdir()
    commands: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run", _fake_run(commands))
    monkeypatch.setattr(pipeline, "georeference_from_control_points", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("At least 3 usable control points are required; found 2")))
    with pytest.raises(ValueError, match="At least 3 usable"):
        pipeline._resume_after_control_points(_resume_args(control_points), outdir, tmp_path / "reports", RunSummary(outdir, {}), None, {}, 10, 1, 0, outdir / "classified.ply")
    assert commands == []
