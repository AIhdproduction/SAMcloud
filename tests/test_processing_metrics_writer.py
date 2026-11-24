"""reports/processing_metrics.json writer, per-run log copy and the video manifest ``discarded`` list."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from samcloud.report import metrics, run_metrics_writer as writer
from test_report_metrics import IMAGES, build_model, two_cluster_points


def make_project(root: Path, with_model: bool = True) -> Path:
    project = root / "writer_project"
    (project / "reports").mkdir(parents=True)
    (project / "products").mkdir()
    (project / "project.samcloud.yml").write_text(
        "project:\n  name: writer_project\nacquisition:\n  type: drone_gps\n  sources:\n    image_directory: ''\n",
        encoding="utf-8")
    (project / "products" / "image_list.txt").write_text("a.jpg\nb.jpg\nc.jpg\nd.jpg\nmissing.jpg\n", encoding="utf-8")
    (project / "reports" / "pipeline.log").write_text("x\nWARNING: from the log\n", encoding="utf-8")
    if with_model:
        folder = project / "products" / "sparse" / "0"
        folder.mkdir(parents=True)
        build_model(IMAGES, two_cluster_points(40)).write(str(folder))
    return project


RUN_METRICS = {"run_id": "20300101_000000Z", "images": 5, "settings": {
    "capture_type": "drone_gps", "class_set": "outdoor", "gps_mode": "RTK", "image_count": 5,
    "surface_fusion": {"enabled": True, "original_points": 1200, "final_points": 900}}}


def test_write_processing_metrics_creates_file_and_per_run_copy(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    target = writer.write_processing_metrics(project, "20300101_000000Z", RUN_METRICS, dense_points_fused=4242)
    per_run = project / "reports" / "processing_metrics_20300101_000000Z.json"
    assert target == project / "reports" / "processing_metrics.json" and target.is_file() and per_run.is_file()
    data = json.loads(target.read_text(encoding="utf-8"))
    for key in ("report_version", "run_id", "created_at", "software", "hardware", "import", "alignment", "overlap",
                "dense", "refinement_stages", "classification", "warnings"):
        assert key in data
    assert data["report_version"] == 1 and data["run_id"] == "20300101_000000Z"
    assert data["alignment"]["registered_images"] == 4 and data["alignment"]["total_images"] == 5
    assert data["alignment"]["unregistered_images"] == ["missing.jpg"]
    assert data["dense"]["points"] == 4242
    assert data["import"]["gps"]["mode"] == "RTK" and data["import"]["source_type"] == "photos"
    assert "WARNING: from the log" in data["warnings"]
    assert data["overlap"]["max"] >= 2
    assert [row["stage"] for row in data["refinement_stages"]][0] == "depth_prior"


def test_per_run_copy_is_never_overwritten(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    writer.write_processing_metrics(project, "20300101_000000Z", RUN_METRICS, dense_points_fused=1)
    per_run = project / "reports" / "processing_metrics_20300101_000000Z.json"
    first = per_run.read_text(encoding="utf-8")
    writer.write_processing_metrics(project, "20300101_000000Z", RUN_METRICS, dense_points_fused=2)
    assert per_run.read_text(encoding="utf-8") == first
    latest = json.loads((project / "reports" / "processing_metrics.json").read_text(encoding="utf-8"))
    assert latest["dense"]["points"] == 2


def test_failing_metric_only_adds_a_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = make_project(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(metrics, "overlap_grid", boom)
    monkeypatch.setattr(metrics, "hardware_info", boom)
    target = writer.write_processing_metrics(project, "20300101_000001Z", RUN_METRICS)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["overlap"] is None and data["hardware"] is None
    assert any("overlap" in w and "synthetic failure" in w for w in data["warnings"])
    assert any("hardware" in w for w in data["warnings"])
    assert data["alignment"]["registered_images"] == 4  # the other metrics are unaffected


def test_project_without_model_records_warning(tmp_path: Path) -> None:
    project = make_project(tmp_path, with_model=False)
    data = json.loads(writer.write_processing_metrics(project, "20300101_000002Z", RUN_METRICS)
                      .read_text(encoding="utf-8"))
    assert data["alignment"] is None and any("no sparse model" in w for w in data["warnings"])
    assert data["dense"]["points"] == 1200  # surface_fusion.original_points fallback


def test_run_summary_write_hook_creates_processing_metrics(tmp_path: Path) -> None:
    from samcloud.engine.run_summary import RunSummary

    project = make_project(tmp_path)
    summary = RunSummary(project / "products", {
        "capture_type": "drone_gps", "classified_output": str(project / "products" / "classified.ply"),
        "classification_summary": str(project / "reports" / "classification_summary.json")})
    summary.dense_points_fused = 777
    summary.write("Waiting for control points")
    assert not (project / "reports" / "processing_metrics.json").exists()
    summary.write("Completed")
    data = json.loads((project / "reports" / "processing_metrics.json").read_text(encoding="utf-8"))
    assert data["dense"]["points"] == 777
    run_id = json.loads((project / "reports" / "run_metrics.json").read_text(encoding="utf-8"))["run_id"]
    assert data["run_id"] == run_id and (project / "reports" / f"processing_metrics_{run_id}.json").is_file()


def test_archive_pipeline_log_copies_once(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    reports = project / "reports"
    (reports / "run_metrics.json").write_text(json.dumps({"run_id": "20300101_000003Z"}), encoding="utf-8")
    copy = writer.archive_pipeline_log(project)
    assert copy == reports / "pipeline_20300101_000003Z.log" and "from the log" in copy.read_text(encoding="utf-8")
    (reports / "pipeline.log").write_text("newer content\n", encoding="utf-8")
    again = writer.archive_pipeline_log(project)
    assert again == copy and "from the log" in copy.read_text(encoding="utf-8")  # not overwritten
    stale = writer.archive_pipeline_log(project, not_before=(reports / "run_metrics.json").stat().st_mtime + 100)
    assert stale is None  # metrics older than the process start: a failed run must not claim the old run id
    assert writer.archive_pipeline_log(tmp_path / "nowhere") is None


def test_video_frame_extraction_records_discarded_frames(tmp_path: Path) -> None:
    from samcloud.acquisition.video import extract_frames, prepare_camera_video_frames

    rng = np.random.default_rng(5)
    base = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    video = tmp_path / "clip.avi"
    writer_cv = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64))
    assert writer_cv.isOpened()
    for index in range(20):  # two 1 s windows of 10 frames; frames 3 and 15 are the sharp ones
        sigma = 0.0 if index in (3, 15) else 3.0
        frame = base if sigma == 0 else cv2.GaussianBlur(base, (0, 0), sigma)
        writer_cv.write(frame)
    writer_cv.release()
    discarded: list[dict] = []
    frames = extract_frames([video], tmp_path / "frames", 1.0, pick="sharpest", discarded=discarded)
    assert len(frames) == 2 and len(discarded) == 18
    assert {item["kept_frame"] for item in discarded} == {"frame_000000.jpg", "frame_000001.jpg"}
    assert all(item["sharpness"] < item["kept_sharpness"] for item in discarded)
    assert not any(abs(item["time_seconds"] - 0.3) < 1e-6 for item in discarded)  # the kept frame is not listed
    assert frames[0].time_seconds == pytest.approx(0.3)
    plain = extract_frames([video], tmp_path / "plain", 1.0, pick="sharpest")  # default behaviour unchanged
    assert [f.time_seconds for f in plain] == [f.time_seconds for f in frames]
    prepared = prepare_camera_video_frames(str(video), tmp_path / "project" / "inputs" / "video_frames", 1.0)
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert manifest["pick"] == "sharpest" and len(manifest["frames"]) == 2
    assert manifest["discarded_count"] == 18 and len(manifest["discarded"]) == 18
