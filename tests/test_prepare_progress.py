"""Progress bar and ETA for workflow step 1 (image preparation)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from samcloud.acquisition import panorama
from samcloud.acquisition.panorama import expected_video_frame_count, prepare_panorama_inputs
from samcloud.app.pages import home
from samcloud.core.configuration import load_global_config
from samcloud.engine import pipeline
from samcloud.engine.eta import (
    CAPTURE_DEFAULT_RATES,
    DEFAULT_RATES,
    PANORAMA_RATES,
    STAGE_LABELS,
    ProgressEstimator,
    RateStore,
)


def _write_panorama(path: Path, width: int = 80, height: int = 40) -> None:
    assert cv2.imwrite(str(path), np.full((height, width, 3), 128, dtype=np.uint8))


class _CountingCapture:
    """Fake cv2 capture with a configurable frame count and frame rate."""

    def __init__(self, frame_count: float, fps: float, frames_to_read: int | None = None) -> None:
        self.frame_count = frame_count
        self.fps = fps
        self.remaining = int(frame_count if frames_to_read is None else frames_to_read)

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        return True, np.full((40, 80, 3), 100, dtype=np.uint8)

    def release(self) -> None:
        return None


def _prepare_with_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame_count: float, callback):
    photo_directory = tmp_path / "photos"
    photo_directory.mkdir()
    _write_panorama(photo_directory / "photo.jpg")
    video_path = tmp_path / "tour.mp4"
    video_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        panorama.cv2, "VideoCapture", lambda _: _CountingCapture(frame_count, 2.0, frames_to_read=6)
    )
    return prepare_panorama_inputs(
        str(photo_directory), str(video_path), tmp_path / "project-inputs", face_size=16,
        progress_callback=callback,
    )


# 1. expected_video_frame_count

def test_expected_video_frame_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(panorama.cv2, "VideoCapture", lambda _: _CountingCapture(300, 30.0))
    assert expected_video_frame_count(Path("tour.mp4"), 1.0) == 10
    assert expected_video_frame_count(Path("tour.mp4"), 0.5) == 20
    monkeypatch.setattr(panorama.cv2, "VideoCapture", lambda _: _CountingCapture(301, 30.0))
    assert expected_video_frame_count(Path("tour.mp4"), 1.0) == 11
    for frame_count, fps in ((0, 30.0), (-1, 30.0), (300, 0.0)):
        monkeypatch.setattr(panorama.cv2, "VideoCapture", lambda _, c=frame_count, f=fps: _CountingCapture(c, f))
        assert expected_video_frame_count(Path("tour.mp4"), 1.0) is None


# 2. prepare_panorama_inputs progress callback

def test_prepare_panorama_reports_monotonic_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int, str]] = []
    # 6 frames at 2 fps, 1 s interval -> 3 kept frames, plus 1 photo = 4 sources, total 8
    prepared = _prepare_with_video(tmp_path, monkeypatch, 6, lambda d, t, s: calls.append((d, t, s)))

    assert prepared.panorama_count == 4
    dones = [done for done, _, _ in calls]
    assert dones and dones == sorted(dones)
    assert calls[-1][0] == calls[-1][1] == 8
    assert any("Frames" in detail for _, _, detail in calls)
    assert any("Würfelflächen" in detail for _, _, detail in calls)
    assert all(total == 8 for _, total, _ in calls)


def test_prepare_panorama_progress_with_unknown_frame_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int, str]] = []
    _prepare_with_video(tmp_path, monkeypatch, 0, lambda d, t, s: calls.append((d, t, s)))

    extraction = [call for call in calls if "Frames" in call[2]]
    assert extraction and all(total == 0 for _, total, _ in extraction)
    assert calls[-1][0] == calls[-1][1] == 8
    dones = [done for done, _, _ in calls]
    assert dones == sorted(dones)


def test_prepare_panorama_without_callback_and_with_failing_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    photo_directory = tmp_path / "photos"
    photo_directory.mkdir()
    _write_panorama(photo_directory / "photo.jpg")
    assert prepare_panorama_inputs(
        str(photo_directory), "", tmp_path / "a", face_size=16, progress_callback=None,
    ).cubemap_image_count == 6

    def broken(*_args) -> None:
        raise RuntimeError("callback failure")

    assert prepare_panorama_inputs(
        str(photo_directory), "", tmp_path / "b", face_size=16, progress_callback=broken,
    ).cubemap_image_count == 6
    nested = tmp_path / "video"
    nested.mkdir()
    assert _prepare_with_video(nested, monkeypatch, 6, broken).panorama_count == 4


# 3. eta tables and rate learning

def test_prepare_images_rates_and_label() -> None:
    assert "prepare_images" in DEFAULT_RATES
    assert "prepare_images" in PANORAMA_RATES
    assert STAGE_LABELS["prepare_images"] == "Bilder vorbereiten"
    assert CAPTURE_DEFAULT_RATES["panorama_360"]["prepare_images"] == PANORAMA_RATES["prepare_images"]
    assert PANORAMA_RATES["prepare_images"] > DEFAULT_RATES["prepare_images"]


def test_rate_store_learns_prepare_images_per_profile(tmp_path: Path) -> None:
    path = tmp_path / "rates.json"
    store = RateStore(path, profile="panorama_360")
    store.update("prepare_images", 100.0, 10.0)
    store.save()

    assert RateStore(path, profile="panorama_360").rate("prepare_images") == pytest.approx(0.6 * 1.0 + 0.4 * 10.0)
    assert RateStore(path, profile="camera").rate("prepare_images") == DEFAULT_RATES["prepare_images"]


# 4. estimator replan / set_units

def test_estimator_replan_keeps_done_prepare_stage_and_elapsed() -> None:
    now = [0.0]
    estimator = ProgressEstimator(
        [("prepare_images", 10), ("feature_extraction", 60), ("mapping", 60)],
        RateStore(None, profile="panorama_360"), clock=lambda: now[0],
    )
    estimator.start_stage("prepare_images")
    estimator.update(10, 10)
    now[0] = 20.0
    estimator.finish_stage()
    estimator.start_stage("feature_extraction")
    before = estimator.snapshot()

    estimator.replan([("feature_extraction", 120), ("mapping", 120), ("undistortion", 120)])
    after = estimator.snapshot()

    assert [stage["key"] for stage in after["stages"]] == [
        "prepare_images", "feature_extraction", "mapping", "undistortion",
    ]
    assert after["stages"][0]["status"] == "done"
    assert after["stages"][1]["status"] == "running"
    assert estimator.stages[2].units == 120
    assert after["eta_seconds"] > before["eta_seconds"]
    assert now[0] - estimator._run_started == 20.0
    estimator.set_units("mapping", 6)
    assert estimator.stages[2].units == 6
    estimator.set_units("prepare_images", 999)
    assert estimator.stages[0].units == 10  # finished stages stay untouched


# 5. pipeline integration

def test_panorama_main_writes_prepare_progress_with_eta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_control_point_pipeline import _fake_run

    cubemap = tmp_path / "inputs" / "cubemap"
    for face in ("front",):
        (cubemap / face).mkdir(parents=True, exist_ok=True)
        (cubemap / face / "pano_0001.jpg").write_bytes(b"image")
    progress_file = tmp_path / "reports" / "progress.json"
    seen: list[dict] = []

    def fake_resolve(args, outdir, parser, callback=None):
        seen.append(json.loads(progress_file.read_text(encoding="utf-8")))  # written by progress.begin
        assert callback is not None
        callback(3, 12, "Frames extrahieren 3/6")
        callback(12, 12, "Würfelflächen 6/6")  # the final report is written immediately
        seen.append(json.loads(progress_file.read_text(encoding="utf-8")))
        return cubemap, True, 256

    monkeypatch.setattr(pipeline, "run", _fake_run([]))
    monkeypatch.setattr(pipeline, "find_colmap_exe", lambda _: "colmap")
    monkeypatch.setattr(pipeline, "resolve_input_images", fake_resolve)
    monkeypatch.setattr(pipeline, "_align_panorama_gravity", lambda sparse_dir, model0, report_dir: (model0, {"frame_count": 1, "up_spread_deg_rms": 0.0}))
    monkeypatch.setattr(sys, "argv", [
        "pipeline", "--capture-type", "panorama_360", "--panorama-workdir", str(tmp_path / "inputs"),
        "--outdir", str(tmp_path / "products"), "--control-points-mode", "after_alignment",
        "--geometry-refinement", "0", "--skip-export", "--progress-file", str(progress_file),
    ])
    pipeline.main()

    first, second = seen
    assert first["stage_key"] == "prepare_images"
    assert first["stages"][0]["key"] == "prepare_images"
    assert first["eta_seconds"] is not None and first["eta_seconds"] > 0
    assert first["stage"] == "Bilder vorbereiten"
    assert second["stage_percent"] == 100.0
    assert second["detail"] == "Würfelflächen 6/6"
    final = json.loads(progress_file.read_text(encoding="utf-8"))
    assert final["stages"][0]["key"] == "prepare_images"
    assert final["stages"][0]["status"] == "done"
    assert [stage["key"] for stage in final["stages"]].count("feature_extraction") == 1


def test_resume_estimator_has_no_prepare_stage() -> None:
    args = argparse.Namespace(control_points_mode="none", stop_after_alignment=False, geometry_refinement="0")
    colmap_config = load_global_config("colmap.yml")
    resumed = pipeline._build_estimator(args, 10, colmap_config, resume=True, prepare_units=5)
    assert "prepare_images" not in [name for name, _ in resumed.plan]
    full = pipeline._build_estimator(args, 10, colmap_config, resume=False, prepare_units=5)
    assert [name for name, _ in full.plan][0] == "prepare_images"
    plain = pipeline._build_estimator(args, 10, colmap_config, resume=False)
    assert "prepare_images" not in [name for name, _ in plain.plan]


# 6. home.py

def test_home_step_one_shows_prepare_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    progress = {
        "stage_key": "prepare_images",
        "detail": "Würfelflächen 80/316",
        "stage_eta_seconds": 125.0,
        "stages": [
            {"key": "prepare_images", "status": "running", "percent": 40, "expected_seconds": 300},
            {"key": "feature_extraction", "status": "pending", "percent": 0, "expected_seconds": 600},
        ],
    }
    project = {"workflow": {"alignment": "running"}}
    monkeypatch.setattr(home, "_read_progress", lambda *_args: progress)

    statuses = home._live_workflow(project)
    assert home._phase_of_stage("prepare_images") == "acquisition"
    assert statuses["acquisition"] == "running"
    assert home._phase_percent("acquisition", statuses, progress) == pytest.approx(40.0)
    text = home._phase_text("acquisition", statuses, progress)
    assert text.startswith("40% · Würfelflächen 80/316 · ")
    assert "2m 05s" in text

    later = {**progress, "stage_key": "feature_extraction", "stages": [
        {"key": "prepare_images", "status": "done", "percent": 100, "expected_seconds": 300},
        {"key": "feature_extraction", "status": "running", "percent": 10, "expected_seconds": 600},
    ]}
    monkeypatch.setattr(home, "_read_progress", lambda *_args: later)
    done_statuses = home._live_workflow(project)
    assert done_statuses["acquisition"] == "completed"
    assert done_statuses["alignment"] == "running"
    assert home._phase_text("acquisition", done_statuses, later) == "100%"


def test_home_old_progress_without_prepare_stage_renders_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    old = {
        "stage_key": "feature_extraction",
        "stage_eta_seconds": 60.0,
        "percent": 10,
        "stages": [{"key": "feature_extraction", "status": "running", "percent": 10, "expected_seconds": 100}],
    }
    project = {"workflow": {"alignment": "running"}}
    monkeypatch.setattr(home, "_read_progress", lambda *_args: old)

    statuses = home._live_workflow(project)
    assert "acquisition" not in statuses  # step 1 stays "ready"
    assert statuses["alignment"] == "running"
    assert home._phase_text("alignment", statuses, old).startswith("10% · ")


def test_home_resume_run_shows_no_acquisition_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    resume = {
        "stage_key": "dense_stereo",
        "stages": [{"key": "dense_stereo", "status": "running", "percent": 10, "expected_seconds": 100}],
    }
    project = {"workflow": {"dense_cloud": "running"}}
    monkeypatch.setattr(home, "_read_progress", lambda *_args: resume)

    assert not home._has_prepare_stage(resume)
    assert "acquisition" not in home._live_workflow(project)
    assert home._has_prepare_stage({"stages": [{"key": "prepare_images", "status": "done"}]})
    assert not home._has_prepare_stage({})


def test_estimate_prepare_work_survives_photo_listing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_directory):
        raise OSError("unreadable")

    monkeypatch.setattr(panorama, "find_panorama_images", broken)
    monkeypatch.setattr(panorama, "expected_video_frame_count", lambda *_args: 7)
    args = argparse.Namespace(
        capture_type="panorama_360", panorama_images="D:/missing", panorama_video="tour.mp4", frame_interval=1.0,
    )
    assert pipeline._estimate_prepare_work(args) == (7, 42)
