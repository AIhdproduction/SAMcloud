"""Storage compaction, safe deletion, and adaptive dense-resolution tests."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from samcloud.app.pages import projects, settings
from samcloud.colmap.alignment import adaptive_dense_size
from samcloud.core.project_manager import ProjectManager
from samcloud.core.storage import (
    DEFAULT_STORAGE,
    compact_project,
    compress_file,
    ensure_available,
    ensure_text_model,
    release_after_fusion,
    remove_generated,
    storage_settings,
)
from samcloud.engine import pipeline
from samcloud.viewer import potree
from synthetic_scene import make_scene


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    products = project / "products"
    products.mkdir(parents=True)
    (project / "project.samcloud.yml").write_text(yaml.safe_dump({"settings": {}}), encoding="utf-8")
    return products


def _file(path: Path, content: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _model(path: Path) -> None:
    _file(path / "cameras.bin")
    _file(path / "images.bin")
    _file(path / "points3D.bin")


def _complete_products(products: Path) -> None:
    dense, sparse, classified = products / "dense", products / "sparse", products / "classified"
    for path in (
        dense / "stereo" / "depth_maps" / "depth.bin",
        dense / "stereo" / "normal_maps" / "normal.bin",
        dense / "images" / "image.jpg",
        dense / "sparse_txt" / "cameras.txt",
        dense / "sparse_txt_extra" / "cameras.txt",
        dense / "sparse_pre_gcp" / "cameras.bin",
        dense / "sparse_gcp" / "cameras.bin",
        dense / "run-colmap-a.sh",
        sparse / "0" / "points3D.bin",
        sparse / "0_geo" / "points3D.bin",
        sparse / "pre_gcp" / "points3D.bin",
        sparse / "pre_gcp_txt" / "cameras.txt",
        classified / "classified_alt.ply",
        products / "viewer" / "potree" / "dense.las",
        products / "database.db",
        dense / "fused.ply",
        dense / "fused.ply.vis",
        dense / "sparse" / "cameras.bin",
        classified / "classified_raw.ply",
        classified / "classified_scores.npz",
        classified / "classified.ply",
    ):
        _file(path, b"generated-data" * 5)


def test_remove_generated_refuses_unsafe_paths(tmp_path: Path) -> None:
    products = _project(tmp_path)
    outside = _file(tmp_path / "outside.bin")
    with pytest.raises(ValueError):
        remove_generated(outside, products)
    invalid_products = tmp_path / "invalid" / "products"
    invalid_products.mkdir(parents=True)
    with pytest.raises(ValueError):
        remove_generated(_file(invalid_products / "file.bin"), invalid_products)
    link = products / "linked.bin"
    try:
        link.symlink_to(outside)
    except OSError:
        return
    with pytest.raises(ValueError):
        remove_generated(link, products)


def test_compress_and_ensure_available_round_trip(tmp_path: Path) -> None:
    source = _file(tmp_path / "input.bin", b"same bytes\x00" * 100)
    original = source.read_bytes()
    compressed = compress_file(source)
    assert compressed.is_file() and not source.exists()
    assert ensure_available(source).read_bytes() == original
    with pytest.raises(FileNotFoundError):
        ensure_available(tmp_path / "missing.bin")


def test_release_after_fusion_removes_dense_caches(tmp_path: Path) -> None:
    products = _project(tmp_path)
    depth = _file(products / "dense" / "stereo" / "depth_maps" / "depth.bin", b"1234")
    normal = _file(products / "dense" / "stereo" / "normal_maps" / "normal.bin", b"56")
    expected = depth.stat().st_size + normal.stat().st_size
    assert release_after_fusion(products / "dense") == expected
    assert not depth.parent.exists() and not normal.parent.exists()


def test_compact_project_keeps_re_run_inputs_when_enabled(tmp_path: Path) -> None:
    products = _project(tmp_path)
    _complete_products(products)
    control = _file(products.parent / "control_points" / "source.csv")
    report = _file(products.parent / "reports" / "kept.txt")
    result = compact_project(products, DEFAULT_STORAGE)
    assert not (products / "dense" / "stereo").exists()
    assert not (products / "dense" / "images").exists()
    assert (products / "sparse" / "0_geo").is_dir() and (products / "sparse" / "pre_gcp").is_dir()
    for path in (products / "database.db", products / "dense" / "fused.ply", products / "dense" / "fused.ply.vis", products / "classified" / "classified_raw.ply", products / "classified" / "classified_scores.npz"):
        assert not path.exists() and path.with_name(path.name + ".gz").is_file()
    assert (products / "classified" / "classified.ply").is_file() and control.is_file() and report.is_file()
    assert (products.parent / "reports" / "storage_summary.json").is_file() and result["compressed"]


def test_compact_project_removes_re_run_inputs_when_disabled(tmp_path: Path) -> None:
    products = _project(tmp_path)
    _complete_products(products)
    result = compact_project(products, {key: False for key in DEFAULT_STORAGE})
    for path in (products / "sparse" / "pre_gcp", products / "database.db", products / "dense" / "fused.ply", products / "dense" / "fused.ply.vis", products / "dense" / "sparse", products / "classified" / "classified_raw.ply", products / "classified" / "classified_scores.npz"):
        assert not path.exists() and not path.with_name(path.name + ".gz").exists()
    assert result["deleted"]


def test_compact_project_keeps_expected_final_sparse_model(tmp_path: Path) -> None:
    for final in ("gcp_adjusted", "0_geo", "0"):
        products = _project(tmp_path / final)
        for name in ("0", "0_geo", "gcp_adjusted"):
            _model(products / "sparse" / name)
        if final == "0_geo":
            remove_generated(products / "sparse" / "gcp_adjusted", products)
        if final == "0":
            remove_generated(products / "sparse" / "gcp_adjusted", products)
            remove_generated(products / "sparse" / "0_geo", products)
        compact_project(products, DEFAULT_STORAGE)
        assert (products / "sparse" / final).is_dir()
        assert sum(path.is_dir() for path in (products / "sparse").iterdir()) == 1


def test_compact_project_keeps_final_model_named_in_marker_file(tmp_path: Path) -> None:
    products = _project(tmp_path)
    for name in ("0", "0_geo", "0_geo_filter_1", "0_geo_ba_1"):
        _model(products / "sparse" / name)
    (products / "sparse" / "final_model.txt").write_text("0_geo_ba_1\n", encoding="utf-8")
    compact_project(products, DEFAULT_STORAGE)
    assert (products / "sparse" / "0_geo_ba_1").is_dir()
    assert not (products / "sparse" / "0").exists()
    assert not (products / "sparse" / "0_geo").exists()
    assert not (products / "sparse" / "0_geo_filter_1").exists()


def test_ensure_text_model_regenerates_colmap_text_model(tmp_path: Path) -> None:
    scene = make_scene(tmp_path / "scene")
    text = ensure_text_model(scene.model_dir, tmp_path / "text")
    assert all((text / name).is_file() for name in ("cameras.txt", "images.txt", "points3D.txt"))


def test_ensure_text_model_leaves_empty_placeholder_unchanged(tmp_path: Path) -> None:
    text = tmp_path / "text"
    text.mkdir()
    assert ensure_text_model(tmp_path / "missing-binary", text) == text
    assert list(text.iterdir()) == []


def test_adaptive_dense_size_uses_input_and_free_space() -> None:
    assert adaptive_dense_size(5280, 1, 10**18)[0] == 2900
    assert adaptive_dense_size(4000, 1, 10**18)[0] == 2200
    assert adaptive_dense_size(8000, 1, 10**18)[0] == 3200
    assert adaptive_dense_size(1500, 1, 10**18)[0] == 1500
    size, reason = adaptive_dense_size(8000, 291, 1)
    assert size < 3200 and "disk space" in reason


def test_pipeline_compacts_only_after_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    products = _project(tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    _file(images / "image.jpg")
    calls: list[Path] = []
    monkeypatch.setattr(pipeline, "compact_project", lambda path, _settings: calls.append(path) or {"freed_bytes": 0, "deleted": [], "compressed": []})
    monkeypatch.setattr(pipeline, "find_colmap_exe", lambda _value: "colmap")

    def fake_run(command, **_kwargs):
        action = command[1]
        if action == "mapper":
            _model(Path(command[command.index("--output_path") + 1]) / "0")
        elif action == "image_undistorter":
            output = Path(command[command.index("--output_path") + 1])
            (output / "images").mkdir(parents=True, exist_ok=True)
            _model(output / "sparse")
        elif action == "model_converter":
            _file(Path(command[command.index("--output_path") + 1]) / "cameras.txt")
        elif action == "stereo_fusion":
            _file(Path(command[command.index("--output_path") + 1]), b"ply")
        elif Path(command[1]).name == "classify.py":
            _file(Path(command[command.index("--out") + 1]), b"classified")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline, "run", fake_run)
    monkeypatch.setattr(pipeline.sys, "argv", ["pipeline", "--capture-type", "camera", "--images", str(images), "--outdir", str(products), "--control-points-mode", "after_alignment", "--skip-export"])
    pipeline.main()
    assert calls == []
    monkeypatch.setattr(pipeline.sys, "argv", ["pipeline", "--capture-type", "camera", "--images", str(images), "--outdir", str(products), "--skip-export"])
    pipeline.main()
    assert calls == [products.resolve()]


def test_release_sparse_intermediates_preserves_continuing_model(tmp_path: Path) -> None:
    products = _project(tmp_path)
    sparse = products / "sparse"
    for name in ("0", "0_geo", "0_geo_filter_1", "0_geo_ba_1", "pre_gcp", "gcp_adjusted"):
        _model(sparse / name)
    pipeline._release_sparse_intermediates(sparse, sparse / "0_geo_ba_1")
    assert (sparse / "0_geo_ba_1").is_dir()
    assert (sparse / "pre_gcp").is_dir() and (sparse / "gcp_adjusted").is_dir()
    assert not (sparse / "0").exists()
    assert not (sparse / "0_geo").exists()
    assert not (sparse / "0_geo_filter_1").exists()


def test_potree_uses_brotli_encoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    source = _file(project / "products" / "classified" / "classified.ply", b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n")
    command: list[str] = []
    monkeypatch.setattr(potree, "potree_available", lambda: True)
    monkeypatch.setattr(potree, "_ply_points", lambda _source: (__import__("numpy").empty((0, 3)), __import__("numpy").empty((0, 3), dtype="uint8"), None))
    monkeypatch.setattr(potree, "write_viewer_las", lambda path, *_args: _file(path))

    def fake_run(value, **_kwargs):
        command.extend(value)
        output = Path(value[value.index("-o") + 1])
        _file(output / "metadata.json")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(potree.subprocess, "run", fake_run)
    potree.ensure_potree(project, "dense")
    assert command[command.index("--encoding") + 1] == "BROTLI"


def test_project_storage_defaults_and_ui_option_mapping(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path)
    created = manager.create_project("storage", str(tmp_path), capture_type="camera")
    assert manager.load(created)["settings"]["storage"] == DEFAULT_STORAGE
    assert settings.STORAGE_OPTIONS == projects.STORAGE_OPTIONS


def test_storage_settings_accepts_flat_project_and_settings_forms() -> None:
    expected = {
        "keep_control_point_inputs": True,
        "keep_classification_inputs": False,
        "keep_refinement_inputs": True,
    }
    flat = {
        "keep_control_point_inputs": 1,
        "keep_classification_inputs": 0,
        "keep_refinement_inputs": ["enabled"],
    }
    assert storage_settings(flat) == expected
    assert storage_settings({"storage": expected}) == expected
    assert storage_settings({"settings": {"storage": expected}}) == expected
