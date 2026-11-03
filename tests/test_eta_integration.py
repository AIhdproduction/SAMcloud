"""Integration tests for pipeline ETA estimation and progress tracking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from samcloud.app.pages import home
from samcloud.core.configuration import load_global_config
from samcloud.engine import pipeline
from samcloud.engine.eta import ProgressEstimator, RateStore
from samcloud.engine.pipeline import ProgressTracker


def test_progress_tracker_writes_eta_snapshot(tmp_path: Path) -> None:
    progress_file = tmp_path / "p.json"
    estimator = ProgressEstimator(
        [("mapping", 10), ("classification", 10)],
        RateStore(None),
    )
    tracker = ProgressTracker(progress_file, estimator)
    tracker.begin("mapping")
    tracker.feed_line("Registering image #3 (num_reg_frames=5)")
    tracker.write(force=True)

    assert progress_file.is_file()
    data = json.loads(progress_file.read_text(encoding="utf-8"))
    assert data["stage_key"] == "mapping"
    assert data["stage_percent"] == 50.0
    assert data["eta_seconds"] > 0
    assert len(data["stages"]) == 2
    for stage in data["stages"]:
        assert "expected_seconds" in stage
        assert stage["expected_seconds"] is not None


def test_run_streams_lines_to_tracker(tmp_path: Path) -> None:
    tracker = ProgressTracker(
        tmp_path / "p.json",
        ProgressEstimator([("dense_fusion", 4)], RateStore(None)),
    )
    tracker.begin("dense_fusion")
    try:
        pipeline._ACTIVE_PROGRESS = tracker
        pipeline.run([sys.executable, "-c", "print('Fusing image [2/4] with index 1')"])
        assert tracker.estimator is not None
        assert tracker.estimator.current is not None
        assert tracker.estimator.current.done_fraction == 0.5
    finally:
        pipeline._ACTIVE_PROGRESS = None


def test_build_estimator_plan_for_automatic_full_run() -> None:
    args = argparse.Namespace(
        control_points_mode="automatic",
        stop_after_alignment=False,
        geometry_refinement="1",
    )
    colmap_config = load_global_config("colmap.yml")
    estimator = pipeline._build_estimator(args, 291, colmap_config, resume=False)
    expected_full = [
        "feature_extraction",
        "matching",
        "mapping",
        "tiepoint_filtering",
        "control_points",
        "undistortion",
        "dense_stereo",
        "dense_fusion",
        "classification",
        "geometry_refinement",
    ]
    assert [name for name, _ in estimator.plan] == expected_full

    estimator_resumed = pipeline._build_estimator(args, 291, colmap_config, resume=True)
    expected_resumed = [
        "control_points",
        "undistortion",
        "dense_stereo",
        "dense_fusion",
        "classification",
        "geometry_refinement",
    ]
    assert [name for name, _ in estimator_resumed.plan] == expected_resumed


def test_phase_percent_uses_weighted_stages() -> None:
    progress = {
        "stages": [
            {"key": "undistortion", "status": "done", "percent": 100, "expected_seconds": 10},
            {"key": "dense_stereo", "status": "running", "percent": 50, "expected_seconds": 90},
            {"key": "dense_fusion", "status": "pending", "percent": 0, "expected_seconds": 0},
        ],
    }
    assert home._phase_percent("dense_cloud", {}, progress) == pytest.approx((100 * 10 + 50 * 90 + 0 * 1) / 101)


def test_tracker_finish_persists_learned_rates(tmp_path: Path) -> None:
    rates_file = tmp_path / "rates.json"
    rates = RateStore(rates_file)
    clock = [0.0]
    estimator = ProgressEstimator(
        [("mapping", 10)],
        rates,
        clock=lambda: clock[0],
    )
    tracker = ProgressTracker(tmp_path / "progress.json", estimator)
    tracker.begin("mapping")
    clock[0] = 40.0
    tracker.finish()

    assert rates_file.is_file()
    data = json.loads(rates_file.read_text(encoding="utf-8"))
    expected_rate = (1 - 0.4) * 1.9 + 0.4 * 4.0
    assert data["rates"]["mapping"] == pytest.approx(expected_rate)
