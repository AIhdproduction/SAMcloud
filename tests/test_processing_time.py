"""Tests for processing time calculation and stage grouping."""

import json
from pathlib import Path

from samcloud.app.pages.home import _processing_time_summary


def test_chain_alignment_and_resume_run(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()

    aborted_run = {
        "status": "Failed",
        "started_at": "2026-11-03T12:00:00Z",
        "total_seconds": 50,
        "stage_seconds": {
            "Sparse reconstruction": 40,
        },
    }
    alignment_run = {
        "status": "Waiting for control points",
        "started_at": "2026-11-03T13:53:00Z",
        "total_seconds": 1094,
        "stage_seconds": {
            "Feature extraction": 100,
            "Sparse reconstruction": 489,
            "Bundle adjustment pass 1": 117,
            "Preparing automatic control points": 7,
        },
    }
    resume_run = {
        "status": "Completed",
        "started_at": "2026-11-03T15:56:00Z",
        "total_seconds": 10980,
        "stage_seconds": {
            "Dense stereo": 8947,
            "Dense fusion": 423,
            "SAM3 classification": 1332,
            "Geometric refinement": 136,
        },
    }

    (reports / "run_metrics_20261103_120000Z.json").write_text(
        json.dumps(aborted_run), encoding="utf-8"
    )
    (reports / "run_metrics_20261103_135300Z.json").write_text(
        json.dumps(alignment_run), encoding="utf-8"
    )
    (reports / "run_metrics_20261103_155600Z.json").write_text(
        json.dumps(resume_run), encoding="utf-8"
    )

    summary = _processing_time_summary(tmp_path)
    assert summary is not None
    assert summary["total_seconds"] == 12074.0
    assert summary["runs"] == 2
    groups = summary["groups"]
    assert groups["Ausrichtung"] == 706.0
    assert groups["Passpunkte"] == 7.0
    assert groups["Dichte Punktwolke"] == 9370.0
    assert groups["Klassifizierung"] == 1332.0
    assert groups["Verbessern & filtern"] == 136.0  # geometric refinement is workflow step 6
    assert "Übrige Schritte" in groups
    assert groups["Übrige Schritte"] == 523.0


def test_single_completed_run(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()

    single_run = {
        "status": "Completed",
        "started_at": "2026-11-03T10:00:00Z",
        "total_seconds": 3600.0,
        "stage_seconds": {
            "Feature extraction": 600.0,
            "Sparse reconstruction": 2500.0,
            "Dense stereo": 500.0,
        },
    }
    (reports / "run_metrics_20261103_100000Z.json").write_text(
        json.dumps(single_run), encoding="utf-8"
    )

    summary = _processing_time_summary(tmp_path)
    assert summary is not None
    assert summary["runs"] == 1
    assert summary["total_seconds"] == 3600.0


def test_no_completed_run_returns_none(tmp_path: Path) -> None:
    assert _processing_time_summary(None) is None
    assert _processing_time_summary(tmp_path) is None

    reports = tmp_path / "reports"
    reports.mkdir()

    aborted_run = {
        "status": "Failed",
        "started_at": "2026-11-03T09:00:00Z",
        "total_seconds": 120.0,
        "stage_seconds": {
            "Feature extraction": 120.0,
        },
    }
    (reports / "run_metrics_20261103_090000Z.json").write_text(
        json.dumps(aborted_run), encoding="utf-8"
    )

    assert _processing_time_summary(tmp_path) is None
