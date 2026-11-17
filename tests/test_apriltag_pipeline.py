"""AprilTag scale step of the pipeline, its command-line wiring and the sparse-model clean-up."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from samcloud.app.processing import ProcessingManager
from samcloud.control_points.apriltags import load_tag_detections, save_tag_detections
from samcloud.engine import pipeline
from colmap_test_model import write_usable_sparse_model
from test_apriltag_scale import TRUE_SCALE, _build_reconstruction, _observations

SETTINGS_INDOOR = {
    "workflow": {}, "classification": {"class_set": "indoor"}, "coordinate_systems": {},
    "control_points": {"crs": "EPSG:4979"}, "apriltags": {"enabled": True},
}


def _project(settings: dict) -> dict:
    return {
        "settings": settings,
        "acquisition": {"type": "camera", "sources": {"image_directory": "D:/images"}, "georeferencing": {}},
    }


def _command(settings: dict, **kwargs) -> list[str]:
    return ProcessingManager._pipeline_command(object(), Path("D:/demo"), _project(settings), "none", **kwargs)


def _images_for(reconstruction, images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for image in reconstruction.images.values():
        (images_dir / image.name).write_bytes(b"not a real image")


def _full_cache(reconstruction, detections: dict) -> dict:
    """Like detect_apriltags: every readable image has an entry, empty when no tag was seen."""
    return {image.name: detections.get(image.name, []) for image in reconstruction.images.values()}


def test_pipeline_argparse_accepts_apriltags_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "image.jpg").write_bytes(b"image")
    calls: list[dict] = []

    def fake_step(model0, sparse_dir, images_dir, control_points_dir, report_dir, *, outlier_pct, **_kwargs):
        calls.append({"model0": model0, "outlier_pct": outlier_pct})
        raise RuntimeError("stop here")

    def fake_run(command, **_kwargs):
        if command[1] == "mapper":
            write_usable_sparse_model(Path(command[command.index("--output_path") + 1]) / "0")

    monkeypatch.setattr(pipeline, "run", fake_run)
    monkeypatch.setattr(pipeline, "find_colmap_exe", lambda _: "colmap")
    monkeypatch.setattr(pipeline, "_apriltag_step", fake_step)
    monkeypatch.setattr(sys, "argv", [
        "pipeline", "--capture-type", "camera", "--images", str(images), "--outdir", str(tmp_path / "products"),
        "--apriltags", "--geometry-refinement", "0", "--skip-export",
    ])
    pipeline.main()  # a failing tag step must not abort the pipeline
    assert len(calls) == 1
    assert calls[0]["model0"].name.startswith("0")
    assert calls[0]["outlier_pct"] == pytest.approx(3.0)


def test_apriltag_step_scales_model_and_writes_report(tmp_path: Path) -> None:
    reconstruction, model_dir = _build_reconstruction(tmp_path)
    detections = _observations(reconstruction, {11: 0.150, 12: 0.150, 13: 0.150})
    images_dir, control_dir, report_dir = tmp_path / "images", tmp_path / "control_points", tmp_path / "reports"
    sparse_dir = tmp_path / "sparse"
    _images_for(reconstruction, images_dir)
    save_tag_detections(control_dir / "apriltag_detections.json", _full_cache(reconstruction, detections))

    new_model, report = pipeline._apriltag_step(model_dir, sparse_dir, images_dir, control_dir, report_dir, outlier_pct=3.0)[:2]

    assert new_model == sparse_dir / "0_tagscale"
    assert (new_model / "images.bin").is_file()
    assert report["mode"] == "scale"
    assert report["scale_factor"] == pytest.approx(TRUE_SCALE, rel=0.005)
    assert sorted(tag["id"] for tag in report["tags"]) == [11, 12, 13]
    stored = json.loads((report_dir / "apriltag_report.json").read_text(encoding="utf-8"))
    assert stored["mode"] == "scale"
    assert all(tag["size_mm_after"] == pytest.approx(150.0, rel=0.01) for tag in stored["tags"])


def test_apriltag_step_without_tags_keeps_original_model(tmp_path: Path) -> None:
    reconstruction, model_dir = _build_reconstruction(tmp_path)
    images_dir, control_dir, report_dir = tmp_path / "images", tmp_path / "control_points", tmp_path / "reports"
    sparse_dir = tmp_path / "sparse"
    _images_for(reconstruction, images_dir)
    save_tag_detections(control_dir / "apriltag_detections.json", {image.name: [] for image in reconstruction.images.values()})

    new_model, report = pipeline._apriltag_step(model_dir, sparse_dir, images_dir, control_dir, report_dir, outlier_pct=3.0)[:2]

    assert new_model == model_dir
    assert not (sparse_dir / "0_tagscale").exists()
    assert report["mode"] == "none"
    assert report["scale_factor"] is None
    assert report["warnings"]
    assert json.loads((report_dir / "apriltag_report.json").read_text(encoding="utf-8"))["mode"] == "none"


def test_apriltag_step_reuses_cache_covering_all_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reconstruction, model_dir = _build_reconstruction(tmp_path)
    detections = _observations(reconstruction, {11: 0.150, 13: 0.150})
    images_dir, control_dir = tmp_path / "images", tmp_path / "control_points"
    _images_for(reconstruction, images_dir)
    cache = control_dir / "apriltag_detections.json"
    searched: list[int] = []
    monkeypatch.setattr(pipeline, "detect_apriltags", lambda paths: searched.append(len(paths)) or {})

    full = _full_cache(reconstruction, detections)
    save_tag_detections(cache, full)
    pipeline._apriltag_step(model_dir, tmp_path / "sparse", images_dir, control_dir, tmp_path / "reports", outlier_pct=3.0)
    assert searched == []

    # A cache that misses images of the model is rebuilt from the images.
    first = next(iter(full))
    save_tag_detections(cache, {first: full[first]})
    pipeline._apriltag_step(model_dir, tmp_path / "sparse", images_dir, control_dir, tmp_path / "reports", outlier_pct=3.0)
    assert searched == [reconstruction.num_images()]
    assert set(load_tag_detections(cache)) == set()


def test_release_sparse_intermediates_keeps_tagscale_model(tmp_path: Path) -> None:
    products = tmp_path / "products"
    sparse_dir = products / "sparse"
    for name in ("0", "0_filter_2", "0_tagscale", "pre_gcp"):
        (sparse_dir / name).mkdir(parents=True)
        (sparse_dir / name / "cameras.bin").write_bytes(b"x")
    (tmp_path / "project.samcloud.yml").write_text("project: {}\n", encoding="utf-8")

    pipeline._release_sparse_intermediates(sparse_dir, sparse_dir / "0_tagscale")

    assert (sparse_dir / "0_tagscale" / "cameras.bin").is_file()
    assert (sparse_dir / "pre_gcp").is_dir()
    assert not (sparse_dir / "0").exists()
    assert not (sparse_dir / "0_filter_2").exists()


def test_indoor_command_has_apriltags_and_one_crs_flag() -> None:
    command = _command(SETTINGS_INDOOR)

    assert "--apriltags" in command
    assert command.count("--control-points-crs") == 1
    assert command[command.index("--control-points-crs") + 1] == "EPSG:4979"


def test_outdoor_command_has_no_apriltags_but_default_crs() -> None:
    command = _command({"workflow": {}, "classification": {}, "coordinate_systems": {}})

    assert "--apriltags" not in command
    assert command.count("--control-points-crs") == 1
    assert command[command.index("--control-points-crs") + 1] == "EPSG:2056+5728"


def test_resume_command_has_exactly_one_crs_flag() -> None:
    command = _command(SETTINGS_INDOOR, resume_after_control_points=True)

    assert "--resume-after-control-points" in command
    assert command.count("--control-points-crs") == 1
    assert command[command.index("--control-points-dir") + 1] == "D:\\demo\\control_points"


def _stub_colmap(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    from test_control_point_pipeline import _fake_run

    commands: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run", _fake_run(commands))
    monkeypatch.setattr(pipeline, "find_colmap_exe", lambda _: "colmap")
    return commands


def test_automatic_control_point_mode_never_runs_the_tag_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "image.jpg").write_bytes(b"image")
    control_points = tmp_path / "control_points"
    control_points.mkdir()
    (control_points / "control_points.csv").write_text("name,x,y,z,role\nP1,0,0,0,pass\n", encoding="utf-8")
    tag_calls: list[Path] = []
    automatic_calls: list[Path] = []
    _stub_colmap(monkeypatch)
    monkeypatch.setattr(pipeline, "_apriltag_step", lambda model0, *_args, **_kwargs: tag_calls.append(model0))
    monkeypatch.setattr(pipeline, "_automatic_control_points", lambda model, *_args, **_kwargs: automatic_calls.append(model))
    monkeypatch.setattr(sys, "argv", [
        "pipeline", "--capture-type", "camera", "--images", str(images), "--outdir", str(tmp_path / "products"),
        "--control-points-mode", "automatic", "--control-points-dir", str(control_points), "--apriltags",
        "--geometry-refinement", "0", "--skip-export",
    ])

    pipeline.main()

    assert len(automatic_calls) == 1  # the gate "failed" (stub returns None) and the run waits for review
    assert tag_calls == []


_GRAVITY_ROTATION = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]  # +90 degrees about x
_GRAVITY_TRANSLATION = [1.0, 2.0, 3.0]


def _run_panorama_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tag_result_factory, gravity_extra: dict | None = None) -> list[Path]:
    cubemap = tmp_path / "inputs" / "cubemap"
    for face in ("front",):
        (cubemap / face).mkdir(parents=True, exist_ok=True)
        (cubemap / face / "pano_0001.jpg").write_bytes(b"image")
    gravity_calls: list[Path] = []
    _stub_colmap(monkeypatch)
    monkeypatch.setattr(pipeline, "resolve_input_images", lambda *_args: (cubemap, True, 256))
    monkeypatch.setattr(pipeline, "_apriltag_step", lambda model0, *_a, **_k: tag_result_factory(model0))
    monkeypatch.setattr(
        pipeline, "_align_panorama_gravity",
        lambda sparse_dir, model0, report_dir: gravity_calls.append(model0) or (
            model0, {"frame_count": 1, "up_spread_deg_rms": 0.0, **(gravity_extra or {})},
        ),
    )
    monkeypatch.setattr(sys, "argv", [
        "pipeline", "--capture-type", "panorama_360", "--panorama-workdir", str(tmp_path / "inputs"),
        "--outdir", str(tmp_path / "products"), "--apriltags", "--control-points-mode", "after_alignment",
        "--geometry-refinement", "0", "--skip-export",
    ])
    pipeline.main()
    return gravity_calls


def test_successful_tag_georeference_skips_panorama_gravity_alignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    georeferenced = lambda model0: pipeline.TagStepResult(model0, {"mode": "georef"}, True, False)  # noqa: E731

    assert _run_panorama_main(tmp_path, monkeypatch, georeferenced) == []


def test_scale_only_tag_result_keeps_panorama_gravity_alignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scaled = lambda model0: pipeline.TagStepResult(model0, {"mode": "scale"}, False, False)  # noqa: E731

    assert len(_run_panorama_main(tmp_path, monkeypatch, scaled)) == 1


def _scale_report(mode: str) -> dict:
    return {"mode": mode, "tags": [{
        "id": 7, "center_xyz": [1.0, 2.0, 3.0],
        "corners_xyz": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    }]}


def _tag_step_writing_report(tmp_path: Path, mode: str, georeferenced: bool):
    def factory(model0: Path):
        pipeline.save_tag_report(tmp_path / "reports" / "apriltag_report.json", _scale_report(mode))
        return pipeline.TagStepResult(model0, {"mode": mode}, georeferenced, False)
    return factory


def test_scale_only_panorama_run_saves_tag_report_in_gravity_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gravity = {"rotation": _GRAVITY_ROTATION, "translation": _GRAVITY_TRANSLATION}

    calls = _run_panorama_main(tmp_path, monkeypatch, _tag_step_writing_report(tmp_path, "scale", False), gravity)

    assert len(calls) == 1
    report = json.loads((tmp_path / "reports" / "apriltag_report.json").read_text(encoding="utf-8"))
    assert report["frame"] == "gravity_aligned"
    # p' = R p + t with R = 90 degrees about x: (1, 2, 3) -> (1, -3, 2) + (1, 2, 3)
    assert report["tags"][0]["center_xyz"] == pytest.approx([2.0, -1.0, 5.0])
    assert report["tags"][0]["corners_xyz"][2] == pytest.approx([2.0, 2.0, 4.0])


def test_georeferenced_tag_report_is_not_transformed_by_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gravity = {"rotation": _GRAVITY_ROTATION, "translation": _GRAVITY_TRANSLATION}

    calls = _run_panorama_main(tmp_path, monkeypatch, _tag_step_writing_report(tmp_path, "georef", True), gravity)

    assert calls == []
    report = json.loads((tmp_path / "reports" / "apriltag_report.json").read_text(encoding="utf-8"))
    assert "frame" not in report
    assert report["tags"][0]["center_xyz"] == [1.0, 2.0, 3.0]
    # even if a georef-mode report were handed to the helper it stays untouched
    path = tmp_path / "reports" / "apriltag_report.json"
    pipeline._transform_tag_report_to_gravity(path, gravity)
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_optimize_control_points_on_tag_project_uses_tag_seeding_and_cubemap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.app import processing
    from test_apriltag_georef import CRS, SETTINGS, _run_control_points, _surveyed

    project = tmp_path / "project"
    rng = np.random.default_rng(21)
    surveyed = {tag_id: _surveyed(tag_id, rng) for tag_id in (11, 12, 13, 14)}
    first, _, _ = _run_control_points(project, {tag_id: "pass" for tag_id in surveyed}, surveyed)
    assert first is not None
    (project / "inputs" / "cubemap").mkdir(parents=True)
    (project / "reports").mkdir(exist_ok=True)
    assert not (project / "products" / "gps_ref.txt").exists()
    seen: list[Path] = []
    original = pipeline._apriltag_control_points

    def spy(model, model_txt, images_dir, *args, **kwargs):
        seen.append(images_dir)
        return original(model, model_txt, images_dir, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_apriltag_control_points", spy)
    monkeypatch.setattr(processing, "load_global_config", lambda _name: {"control_points": dict(SETTINGS)})
    project_data = {
        "acquisition": {"type": "panorama_360", "sources": {}, "georeferencing": {}},
        "settings": {"control_points": {"crs": CRS}},
    }

    passed = processing.optimize_control_points(project, project_data)

    assert passed is True
    assert seen == [(project / "inputs" / "cubemap").resolve()]
    report = json.loads((project / "reports" / "control_points_report.json").read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is True
    assert sorted(point["name"] for point in report["points"]) == ["TAG11", "TAG12", "TAG13", "TAG14"]


def _local_tag_project(tmp_path: Path):
    from test_apriltag_georef import SETTINGS as GEOREF_SETTINGS, _surveyed_local, _world, _write_csv

    reconstruction, model_dir, detections = _world(tmp_path)
    rng = np.random.default_rng(33)
    _write_csv(tmp_path / "control_points", {tag_id: (_surveyed_local(tag_id, rng), "pass") for tag_id in (11, 12, 13, 14)})
    _images_for(reconstruction, tmp_path / "images")
    save_tag_detections(tmp_path / "control_points" / "apriltag_detections.json", _full_cache(reconstruction, detections))
    return model_dir, GEOREF_SETTINGS


def test_apriltag_step_with_local_csv_georeferences_without_georeference_file(tmp_path: Path) -> None:
    model_dir, settings = _local_tag_project(tmp_path)
    sparse = tmp_path / "products" / "sparse"

    outcome = pipeline._apriltag_step(
        model_dir, sparse, tmp_path / "images", tmp_path / "control_points", tmp_path / "reports",
        outlier_pct=3.0, crs="LOCAL", settings=settings,
    )

    assert outcome.georeferenced is True and outcome.gate_failed is False
    assert outcome.report["mode"] == "georef"
    assert outcome.model == sparse / "gcp_adjusted"
    assert not (tmp_path / "products" / "georeference.json").exists()
    assert pipeline._is_georeferenced(tmp_path / "products") is True


def test_local_tag_georeference_skips_panorama_gravity_alignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local = lambda model0: pipeline.TagStepResult(model0, {"mode": "georef"}, True, False)  # noqa: E731

    assert _run_panorama_main(tmp_path, monkeypatch, local) == []
    assert not (tmp_path / "products" / "georeference.json").exists()


def test_local_crs_with_gps_rerun_path_returns_none_with_clear_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from synthetic_scene import make_scene
    from test_control_point_pipeline import _write_points

    products = tmp_path / "products"
    scene = make_scene(products / "sparse")
    control_points, model_txt = tmp_path / "control_points", products / "sparse_txt"
    control_points.mkdir()
    model_txt.mkdir()
    _write_points(control_points, scene.references)
    gps_ref = products / "gps_ref.txt"
    gps_ref.write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(pipeline, "fit_enu_to_ecef", lambda *_args: {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [6371000.0, 0.0, 0.0]})
    report_path = tmp_path / "reports" / "control_points_report.json"

    result = pipeline._rerun_control_points(
        scene.model_dir, model_txt, tmp_path / "images", control_points, "LOCAL",
        gps_ref, {}, report_path, rtk=False,
    )

    assert result is None
    reasons = json.loads(report_path.read_text(encoding="utf-8"))["gate"]["reasons"]
    assert reasons == ["Lokale Koordinaten sind nur für AprilTag-Projekte ohne GPS möglich"]
