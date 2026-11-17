"""Consistent ETAs: workflow phase ETAs vs total ETA, stage labels, skipped refinement, initial plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from samcloud.app.pages import home
from samcloud.core.configuration import load_global_config
from samcloud.engine import pipeline
from samcloud.engine.eta import CAPTURE_DEFAULT_RATES, DEFAULT_RATES, PANORAMA_RATES, STAGE_LABELS, ProgressEstimator, RateStore

LOCALES = Path(__file__).resolve().parents[1] / "samcloud" / "app" / "locales"


@pytest.fixture
def default_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan with the built-in rates and never read or write the real learned-rates file."""
    monkeypatch.setattr(pipeline, "RateStore", lambda _path, profile=None: RateStore(None, profile=profile))


def _v11_progress(*, geometry_skipped: bool = False) -> dict:
    """progress.json of Test_Home_V11_Tag while surface fusion runs (step 6)."""
    done = [
        {"key": key, "status": "done", "percent": 100.0, "expected_seconds": 100.0}
        for key in ("prepare_images", "feature_extraction", "matching", "mapping", "tiepoint_filtering",
                    "undistortion", "dense_stereo", "dense_fusion", "classification", "depth_prior")
    ]
    stages = done + [
        {"key": "surface_fusion", "status": "running", "percent": 0.0, "expected_seconds": 1048.0},
        {"key": "superpoint_vote", "status": "pending", "percent": 0.0, "expected_seconds": 1451.0},
        {"key": "geometry_refinement", "status": "skipped" if geometry_skipped else "pending", "percent": 0.0,
         "expected_seconds": 682.0},
    ]
    eta = 1048.0 + 1451.0 + (0.0 if geometry_skipped else 682.0)
    return {
        "stage_key": "surface_fusion", "stage": "Oberflächen zusammenführen", "stage_percent": 0.0,
        "stage_eta_seconds": 1048.0, "eta_seconds": eta, "percent": 70.0, "stages": stages,
    }


def _planned_keys() -> set[str]:
    """Every stage key a fresh run, a resume run or a step 6 re-run can plan."""
    config = load_global_config("colmap.yml")
    keys: set[str] = set()
    variants = [
        dict(capture_type="panorama_360", class_set="indoor", depth_model="large"),
        dict(capture_type="panorama_360", class_set="indoor", depth_model="large", surface_fusion="0"),
        dict(capture_type="drone_gps", class_set="outdoor", depth_model="large"),
        dict(capture_type="camera", class_set="outdoor", depth_model="off"),
    ]
    for variant in variants:
        for mode in ("none", "automatic"):
            args = argparse.Namespace(control_points_mode=mode, stop_after_alignment=False, geometry_refinement="1",
                                      **variant)
            for resume in (False, True):
                estimator = pipeline._build_estimator(args, 120, config, resume=resume, prepare_units=20)
                keys.update(name for name, _ in estimator.plan)
    return keys


# 1. step 6 remaining time == total ETA

@pytest.mark.parametrize("geometry_skipped", [False, True])
def test_step_six_remaining_time_equals_total_eta(monkeypatch: pytest.MonkeyPatch, geometry_skipped: bool) -> None:
    progress = _v11_progress(geometry_skipped=geometry_skipped)
    project = {"workflow": {"alignment": "completed", "dense_cloud": "completed", "classification": "running"}}
    statuses = home._live_workflow(project, progress)

    assert statuses["refinement"] == "running"
    remaining = home._phase_eta("refinement", statuses, progress)
    assert remaining == pytest.approx(progress["eta_seconds"])
    assert remaining >= progress["stage_eta_seconds"]
    assert home._phase_text("refinement", statuses, progress).endswith(home._format_span(progress["eta_seconds"]))
    if geometry_skipped:
        assert remaining == pytest.approx(1048.0 + 1451.0)


def test_superpoint_vote_running_keeps_step_six_running() -> None:
    progress = _v11_progress()
    progress["stage_key"] = "superpoint_vote"
    progress["stages"][-3]["status"] = "done"
    progress["stages"][-2]["status"] = "running"
    progress["stage_eta_seconds"] = 1200.0
    progress["eta_seconds"] = 1200.0 + 682.0
    statuses = home._live_workflow({"workflow": {"classification": "running"}}, progress)

    assert statuses["classification"] == "completed" and statuses["refinement"] == "running"
    assert home._phase_eta("refinement", statuses, progress) == pytest.approx(progress["eta_seconds"])


def test_phase_etas_add_up_to_estimator_total() -> None:
    """With a real estimator snapshot the running and pending phases add up to eta_seconds."""
    clock = [0.0]
    plan = [(key, 50.0) for key in DEFAULT_RATES]
    estimator = ProgressEstimator(plan, RateStore(None, profile="panorama_360"), clock=lambda: clock[0])
    for key in ("prepare_images", "feature_extraction", "matching"):
        estimator.start_stage(key)
        clock[0] += 5.0
    estimator.update(1, 4)
    estimator.skip_stage("geometry_refinement")
    snapshot = json.loads(json.dumps(estimator.snapshot()))
    statuses = home._live_workflow({"workflow": {"alignment": "running"}}, snapshot)

    total = sum(home._phase_eta(phase, statuses, snapshot) or 0.0 for phase in home._PHASE_ORDER)
    assert total == pytest.approx(snapshot["eta_seconds"], abs=1.0)
    assert home._phase_eta("alignment", statuses, snapshot) >= snapshot["stage_eta_seconds"]


def test_unknown_stage_joins_the_phase_before_it() -> None:
    progress = {
        "stage_key": "future_step", "stage_eta_seconds": 30.0, "eta_seconds": 30.0 + 40.0,
        "stages": [
            {"key": "plane_fill", "status": "done", "percent": 100.0, "expected_seconds": 10.0},
            {"key": "future_step", "status": "running", "percent": 0.0, "expected_seconds": 30.0},
            {"key": "geometry_refinement", "status": "pending", "percent": 0.0, "expected_seconds": 40.0},
        ],
    }
    statuses = home._live_workflow({"workflow": {"classification": "running"}}, progress)
    assert statuses["refinement"] == "running"
    assert home._phase_eta("refinement", statuses, progress) == pytest.approx(70.0)


def test_old_progress_without_new_stages_still_renders() -> None:
    old = {"stage_key": "surface_fusion", "stage_eta_seconds": 100.0, "eta_seconds": 300.0, "stages": [
        {"key": "surface_fusion", "status": "running", "percent": 10, "expected_seconds": 120},
        {"key": "geometry_refinement", "status": "pending", "percent": 0, "expected_seconds": 200},
    ]}
    statuses = home._live_workflow({"workflow": {"classification": "running"}}, old)
    assert home._phase_text("refinement", statuses, old).startswith("4% · ")
    assert home._phase_eta("refinement", statuses, old) == pytest.approx(300.0)
    assert home._phase_text("dense_cloud", statuses, {}) == "100%"


def test_progress_file_snapshot_is_shared_until_the_file_changes(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    path = reports / "progress.json"
    path.write_text(json.dumps(_v11_progress()), encoding="utf-8")
    first = home._read_progress({}, tmp_path)
    second = home._read_progress({}, tmp_path)
    assert first == second and first["eta_seconds"] == 3181.0
    path.write_text("{ half written", encoding="utf-8")
    assert home._read_progress({}, tmp_path)["eta_seconds"] == 3181.0  # keeps the last good snapshot
    path.write_text(json.dumps(_v11_progress(geometry_skipped=True)), encoding="utf-8")
    assert home._read_progress({}, tmp_path)["eta_seconds"] == 2499.0


# 2. every planned stage key belongs to exactly one phase

def test_every_planned_stage_maps_to_exactly_one_phase() -> None:
    keys = set(DEFAULT_RATES) | set(PANORAMA_RATES) | _planned_keys()
    for rates in CAPTURE_DEFAULT_RATES.values():
        keys |= set(rates)
    assert "superpoint_vote" in keys
    for key in sorted(keys):
        phases = [phase for phase, members in home._PHASE_STAGES.items() if key in members]
        assert len(phases) == 1, f"{key} belongs to {phases}"
    assert _planned_keys() <= set(DEFAULT_RATES)  # a planned stage without a rate would plan 1 s per unit


def test_step_six_run_summary_stages_are_grouped() -> None:
    refinement = dict(home._TIME_GROUPS)["Verbessern & filtern"]
    for name in ("Surface fusion", "Superpoint voting", "Mirror detection", "Depth fill", "Geometric refinement"):
        assert name in refinement
    all_members = [member for _, members in home._TIME_GROUPS for member in members]
    assert len(all_members) == len(set(all_members))
    assert set(dict(home._TIME_GROUPS)) | {"Übrige Schritte"} == set(home._TIME_GROUP_KEYS)


def test_processing_time_summary_puts_surface_fusion_into_step_six(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "run_metrics_1.json").write_text(json.dumps({
        "status": "Completed", "started_at": "2026-11-15T10:00:00", "total_seconds": 5000.0,
        "stage_seconds": {"Sparse reconstruction": 1000.0, "Surface fusion": 3000.0, "Superpoint voting": 70.0,
                          "Mirror detection": 30.0},
    }), encoding="utf-8")
    timing = home._processing_time_summary(tmp_path)
    assert timing["groups"]["Verbessern & filtern"] == pytest.approx(3100.0)
    assert timing["groups"]["Übrige Schritte"] == pytest.approx(900.0)


# 3. labels for every rate key

def test_every_rate_key_has_a_label_and_ui_text() -> None:
    catalogues = {
        language: json.loads((LOCALES / language / "workflow.json").read_text(encoding="utf-8"))["stage"]
        for language in ("de", "en")
    }
    keys = set(DEFAULT_RATES) | set(PANORAMA_RATES) | _planned_keys()
    for key in sorted(keys):
        assert STAGE_LABELS.get(key), key
        for language, stages in catalogues.items():
            assert stages.get(key), f"workflow.stage.{key} missing in {language}"
    assert STAGE_LABELS["superpoint_vote"] == "Superpoint-Abstimmung"
    assert catalogues["de"]["superpoint_vote"] == "Superpoint-Abstimmung"
    assert catalogues["en"]["superpoint_vote"] == "Superpoint voting"
    snapshot = ProgressEstimator([("superpoint_vote", 10)], RateStore(None))
    snapshot.start_stage("superpoint_vote")
    assert snapshot.snapshot()["stage"] == "Superpoint-Abstimmung"


def test_superpoint_vote_stage_text_is_translated(german_ui: None) -> None:
    assert home._stage_text({"stage_key": "superpoint_vote", "stage": "superpoint_vote"}) == "Superpoint-Abstimmung"


# 4. geometric refinement leaves the ETA as soon as it is known to be skipped

def _run_to_georef_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, tag_georeferenced: bool) -> dict:
    from test_control_point_pipeline import _fake_run

    images = tmp_path / "images"
    images.mkdir()
    (images / "image.jpg").write_bytes(b"image")
    progress_file = tmp_path / "reports" / "progress.json"
    seen: list[dict] = []
    original_release = pipeline._release_sparse_intermediates

    def fake_tag_step(model0, *_args, **_kwargs):
        return pipeline.TagStepResult(model0, {"mode": "local"}, georeferenced=tag_georeferenced)

    def release(*args, **kwargs):  # first call after the georeference decision
        seen.append(json.loads(progress_file.read_text(encoding="utf-8")))
        return original_release(*args, **kwargs)

    class _StopBeforeDenseStereo(Exception):
        pass

    def stop(*_args, **_kwargs):
        seen.append(json.loads(progress_file.read_text(encoding="utf-8")))
        raise _StopBeforeDenseStereo

    monkeypatch.setattr(pipeline, "run", _fake_run([]))
    monkeypatch.setattr(pipeline, "find_colmap_exe", lambda _: "colmap")
    monkeypatch.setattr(pipeline, "_apriltag_step", fake_tag_step)
    monkeypatch.setattr(pipeline, "_release_sparse_intermediates", release)
    monkeypatch.setattr(pipeline, "_run_dense_stereo", stop)
    monkeypatch.setattr(sys, "argv", [
        "pipeline", "--capture-type", "camera", "--images", str(images), "--outdir", str(tmp_path / "products"),
        "--control-points-mode", "none", "--apriltags", "--geometry-refinement", "1", "--skip-export",
        "--progress-file", str(progress_file),
    ])
    with pytest.raises(_StopBeforeDenseStereo):
        pipeline.main()
    after_decision, at_dense_stereo = seen
    return {"after_decision": after_decision, "at_dense_stereo": at_dense_stereo}


def _geometry(snapshot: dict) -> dict:
    return next(stage for stage in snapshot["stages"] if stage["key"] == "geometry_refinement")


def _pending_sum(snapshot: dict) -> float:
    return sum(stage["expected_seconds"] for stage in snapshot["stages"] if stage["status"] == "pending")


def test_local_project_skips_geometry_refinement_right_after_georef_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, default_rates: None,
) -> None:
    seen = _run_to_georef_decision(tmp_path, monkeypatch, tag_georeferenced=False)
    for snapshot in seen.values():
        assert _geometry(snapshot)["status"] == "skipped"
        assert snapshot["eta_seconds"] == pytest.approx(
            (snapshot["stage_eta_seconds"] or 0.0) + _pending_sum(snapshot), abs=1.0)


def test_georeferenced_project_keeps_geometry_refinement_planned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, default_rates: None,
) -> None:
    seen = _run_to_georef_decision(tmp_path, monkeypatch, tag_georeferenced=True)
    for snapshot in seen.values():
        geometry = _geometry(snapshot)
        assert geometry["status"] == "pending" and geometry["expected_seconds"] > 0
        assert snapshot["eta_seconds"] >= geometry["expected_seconds"]


def test_resume_runs_skip_geometry_refinement_from_the_start(tmp_path: Path) -> None:
    outdir = tmp_path / "products"
    outdir.mkdir()
    assert not pipeline._may_be_georeferenced(outdir)
    estimator = ProgressEstimator([("classification", 10), ("geometry_refinement", 10)], RateStore(None))
    tracker = pipeline.ProgressTracker(tmp_path / "progress.json", estimator)
    pipeline._plan_geometry_refinement(tracker, pipeline._may_be_georeferenced(outdir))
    snapshot = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert _geometry(snapshot)["status"] == "skipped"
    assert snapshot["eta_seconds"] == pytest.approx(10 * DEFAULT_RATES["classification"], abs=0.2)

    (outdir / "gps_ref.txt").write_text("image.jpg 47 8 400\n", encoding="utf-8")
    assert pipeline._may_be_georeferenced(outdir)
    planned = ProgressEstimator([("geometry_refinement", 10)], RateStore(None))
    pipeline._plan_geometry_refinement(pipeline.ProgressTracker(tmp_path / "p2.json", planned), True)
    assert planned.stages[0].status == "pending"
    pipeline._plan_geometry_refinement(None, False)  # no progress file: nothing to do


# 5. the initial plan uses the same dense-stereo basis as the replanned one

def _dense_seconds(estimator: ProgressEstimator) -> float:
    return next(stage["expected_seconds"] for stage in estimator.snapshot()["stages"] if stage["key"] == "dense_stereo")


def test_initial_panorama_plan_matches_replan_dense_stereo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, default_rates: None,
) -> None:
    sources = tmp_path / "panoramas"
    sources.mkdir()
    Image.new("RGB", (5760, 2880), (90, 90, 90)).save(sources / "pano_0001.jpg", quality=50)
    outdir = tmp_path / "products"
    outdir.mkdir()
    face_size = int(load_global_config("defaults.yml")["acquisition"]["panorama"]["cubemap_face_size"])
    args = argparse.Namespace(
        capture_type="panorama_360", panorama_images=str(sources), panorama_video=None, frame_interval=1.0,
        cubemap_face_size=face_size, images=None, class_set="indoor", depth_model="large",
        control_points_mode="none", stop_after_alignment=False, geometry_refinement="1",
    )
    monkeypatch.setattr(pipeline, "_estimate_prepare_work", lambda _args: (356, 2136))
    colmap_config = load_global_config("colmap.yml")
    dense_config = colmap_config["dense_reconstruction"]
    camera_size = int(colmap_config.get("camera", {}).get("max_image_size", 4000))
    fallback = camera_size if dense_config.get("max_image_size") == "auto" else int(dense_config["max_image_size"])

    initial = pipeline._initial_estimator(args, colmap_config, outdir, fallback)

    # what the replan after write_image_list sees: the real cubemap faces
    faces = tmp_path / "cubemap"
    (faces / "front").mkdir(parents=True)
    Image.new("RGB", (face_size, face_size), (90, 90, 90)).save(faces / "front" / "pano_0001.jpg")
    (outdir / "image_list.txt").write_text("front/pano_0001.jpg\n", encoding="utf-8")
    replan_size, _reason = pipeline._dense_size_from_inputs(dense_config, outdir, faces)
    replanned = pipeline._build_estimator(args, 2136, colmap_config, resume=False, dense_max_image_size=replan_size)

    before, after = _dense_seconds(initial), _dense_seconds(replanned)
    assert after > 0
    assert abs(before - after) / after <= 0.30, (before, after)
    assert next(stage.units for stage in initial.stages if stage.name == "feature_extraction") == 2136


def test_initial_photo_plan_reads_the_first_image_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (4000, 3000), (90, 90, 90)).save(images / "a.jpg", quality=50)
    args = argparse.Namespace(capture_type="camera", images=str(images))
    assert pipeline._expected_input_long_side(args) == 4000
    auto = {"max_image_size": "auto", "auto": {"ratio": 0.55, "minimum": 2000, "maximum": 3200}}
    (tmp_path / "image_list.txt").write_text("a.jpg\n", encoding="utf-8")
    real, _reason = pipeline._dense_size_from_inputs(auto, tmp_path, images)
    assert pipeline._planned_dense_size(auto, 4000, 4000, 1, tmp_path) == real
    assert pipeline._planned_dense_size(auto, 4000, None, 1, tmp_path) == 3200
    assert pipeline._planned_dense_size({"max_image_size": 1600}, 4000, 4000, 1, tmp_path) == 1600
    assert pipeline._expected_input_long_side(argparse.Namespace(capture_type="camera", images=None)) is None
    no_source = argparse.Namespace(capture_type="panorama_360", panorama_images=None, panorama_video=None,
                                   cubemap_face_size=1232)
    assert pipeline._expected_input_long_side(no_source) == 1232
