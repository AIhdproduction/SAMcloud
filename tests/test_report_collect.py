"""Collector of the processing protocol on synthetic project folders."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from samcloud.report.collect import SECTION_KEYS, collect_protocol_data, write_protocol_data

ALLOWED_STATUS = {"ok", "not_run", "no_data", "error"}


def put(project: Path, relative: str, content) -> None:
    path = project / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")


def make_project(root: Path) -> Path:
    project = root / "synthetic_project"
    put(project, "project.samcloud.yml",
        "project:\n  name: synthetic_project\n  created_at: '2030-01-01T00:00:00+00:00'\n"
        "acquisition:\n  type: drone_gps\n  sources:\n    image_directory: ''\n")
    put(project, "products/image_list.txt", "img_a.jpg\nimg_b.jpg\nimg_c.jpg\nimg_d.jpg\n")
    put(project, "reports/run_metrics.json", {
        "run_id": "20300101_000000Z", "finished_at": "2030-01-01T05:00:00+00:00", "total_seconds": 1234.5,
        "stage_seconds": {"Dense stereo": 100.0, "Dense fusion": 20.0, "SAM3 classification": 300.0},
        "settings": {
            "capture_type": "drone_gps", "class_set": "outdoor", "export_crs": "EPSG:99999", "image_count": 4,
            "gps_mode": "RTK", "gps_max_error_m": 0.1, "dense_max_image_size": 2000, "dense_num_iterations": 3,
            "dense_size_reason": "synthetic", "dense_stereo_excluded_images": ["img_d.jpg"],
            "sparse_models": [{"registered_images": 3, "points": 500}],
            "alignment_quality": {"matching_strategy": "exhaustive", "camera_model": "OPENCV", "tiepoint_filter": {
                "passes": [{"pass": 1, "after_filter": {"points": 480.0, "mean_track_length": 3.5,
                                                         "mean_reprojection_error": 0.61}}]}},
            "point_cleaning": {"enabled": True, "original_points": 10000, "removed_points": 100},
        },
        "dense_points": 10000, "classified_points": 9900,
    })
    put(project, "reports/classification_summary.json", {
        "point_count": 9900, "unknown_points": 900, "mode": "segearth_ov3", "engine": "sam3",
        "background_threshold": 0.4, "query_threshold": 0.45, "class_counts": {"ground": 6000, "building": 3000},
        "instances": {"count": 5, "per_class": {"building": 5}}})
    put(project, "reports/control_points_report.json", {
        "gate": {"passed": True, "reasons": []},
        "points": [
            {"name": "PP1", "role": "pass", "enabled": True, "status": "accepted", "projections": 10,
             "error_x_m": 0.01, "error_y_m": -0.02, "error_z_m": 0.003, "error_3d_m": 0.0224},
            {"name": "TAG7", "role": "control", "enabled": False, "status": "rejected", "projections": 2,
             "error_x_m": None, "error_y_m": None, "error_z_m": None, "error_3d_m": None}],
        "totals": {"pass": {"rms_x_m": 0.01, "rms_y_m": 0.02, "rms_z_m": 0.003, "rms_xy_m": 0.0224,
                            "rms_3d_m": 0.0226, "mean_reprojection_px": 0.5, "count": 1},
                   "control": {"rms_x_m": None, "count": 0}},
        "quick_correction": {"height_offset_m": -0.2, "applied": False}})
    put(project, "reports/apriltag_report.json", {
        "mode": "scale", "scale_factor": 1.25, "scale_applied": 1.25, "warnings": [],
        "tags": [{"id": 7, "images": 4, "size_mm_before": 100.0, "size_mm_after": 125.0, "deviation_pct": -0.8,
                  "status": "ok"}, {"id": 8, "images": 3, "deviation_pct": 0.3, "status": "ok"}]})
    put(project, "reports/checkpoints_pre_gcp.json", {
        "points": [{"name": "PP1", "dE": 0.01, "dN": 0.0, "dH": 0.2, "d3D": 0.2, "status": "ok",
                    "measured": [10.0, 20.0, 1.0]}],
        "totals": {"n_points": 1, "rms_E": 0.01, "rms_N": 0.0, "rms_H": 0.2, "rms_2D": 0.01, "rms_3D": 0.2,
                   "bias_H": 0.2}})
    put(project, "reports/geometry_summary.json", {"unknown_before": 20, "unknown_after": 10, "set_to_unknown": 3,
                                                    "smoothed_points": 77, "runtime_seconds": 5.5})
    put(project, "reports/run_summary.txt", "Summary\nWarnings\nWARNING: Point cleaning was skipped: synthetic\n")
    put(project, "reports/pipeline.log", "line\nWARNING: log warning one\n")
    put(project, "reports/dxf_run_metrics.json", {"step_seconds": {"Export DXF": 1.5}})
    put(project, "products/environment/qa_cad_report.json",
        [{"rule": "r1", "ok": True, "value": 1, "threshold": "t"}, {"rule": "r2", "ok": False, "value": 2, "threshold": "t"}])
    put(project, "products/environment/outdoor.dxf", "0\nEOF\n")
    return project


def test_sections_are_complete_ordered_and_match_inputs(tmp_path: Path) -> None:
    data = collect_protocol_data(make_project(tmp_path), language="de")
    assert tuple(data.sections) == SECTION_KEYS
    assert data.language == "de" and data.project["name"] == "synthetic_project"
    for key, section in data.sections.items():
        assert section.key == key and section.status in ALLOWED_STATUS and "not_recorded" in section.data
    s = {key: section.data for key, section in data.sections.items()}
    assert s["title"]["project_name"] == "synthetic_project" and s["title"]["date"] == "2030-01-01"
    assert s["title"]["crs"] == "EPSG:99999" and s["title"]["control_points_mode"] == "with"
    assert s["summary"]["images_registered"] == 3 and s["summary"]["images_total"] == 4
    assert s["summary"]["rms_pass_3d_m"] == pytest.approx(0.0226)
    assert s["summary"]["apriltag_max_deviation_pct"] == pytest.approx(0.8)
    assert s["summary"]["filtered_points"] == 9900 and s["summary"]["total_seconds"] == 1234.5
    assert s["import_"]["source"] == "photos" and s["import_"]["image_count"] == 4
    assert s["alignment"]["registered"] == 3 and s["alignment"]["registered_pct"] == pytest.approx(75.0)
    assert s["alignment"]["tie_points"] == 480 and s["alignment"]["matching_strategy"] == "exhaustive"
    assert [p["id"] for p in s["control_points"]["points"]] == ["PP1", "TAG7"]
    assert s["control_points"]["points"][1]["type"] == "apriltag"
    assert s["control_points"]["points"][1]["excluded_reason"] == "disabled"
    assert s["control_points"]["height_shift_m"] == -0.2
    assert s["control_points"]["gcp_vs_shift"]["pre_gcp"]["bias_h_m"] == pytest.approx(0.2)
    assert s["control_points"]["apriltag"]["scale_factor"] == 1.25
    assert all("measured" not in p for p in s["control_points"]["checkpoints"]["pre_gcp"]["points"])
    assert s["densify"]["points"] == 10000 and s["densify"]["seconds"] == pytest.approx(120.0)
    assert s["densify"]["settings"]["excluded_images"] == ["img_d.jpg"]
    assert s["refine"]["executed"] is True and s["refine"]["metrics"]["unknown_after"] == 10
    assert s["classification"]["classes"][0] == {"name": "ground", "points": 6000,
                                                 "share_pct": pytest.approx(6000 / 9900 * 100), "instances": None}
    assert s["classification"]["thresholds"]["query"] == 0.45
    assert s["refinement"]["stages"][3]["stage"] == "point_cleaning" and s["refinement"]["stages"][3]["skipped"]
    assert s["products"]["dxf_qa"]["passed"] == 1 and s["products"]["dxf_qa"]["failed"] == 1
    assert [f["kind"] for f in s["products"]["files"]] == ["dxf"]
    assert {"name": "Export DXF", "seconds": 1.5} in s["runtimes"]["steps"]
    assert "WARNING: log warning one" in s["appendix"]["warnings"]
    assert any("Point cleaning was skipped" in w for w in s["appendix"]["warnings"])
    assert data.sections["alignment"].source == "derived"
    assert "hardware" in s["runtimes"]["not_recorded"]


def test_empty_project_gives_no_data_without_exception(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    data = collect_protocol_data(empty)
    assert tuple(data.sections) == SECTION_KEYS
    assert all(section.status in ALLOWED_STATUS for section in data.sections.values())
    assert data.sections["control_points"].status == "not_run"
    assert data.sections["control_points"].data["reason"] == "without_control_points"
    assert data.sections["alignment"].status == "no_data"
    assert data.sections["classification"].status == "no_data"
    assert data.sections["title"].data["project_name"] == "empty"
    path = write_protocol_data(empty, data)
    assert path == empty / "reports" / "protocol_data.json"
    assert json.loads(path.read_text(encoding="utf-8"))["sections"]["title"]["key"] == "title"


def test_corrupt_file_marks_only_its_section(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    project = make_project(tmp_path)
    put(project, "reports/classification_summary.json", "{broken")
    put(project, "reports/geometry_summary.json", [1, 2, 3])  # wrong type
    with caplog.at_level(logging.WARNING):
        data = collect_protocol_data(project)
    classification = data.sections["classification"]
    assert classification.status == "error" and classification.warnings
    assert data.sections["refine"].status == "error" and data.sections["refine"].warnings
    assert data.sections["alignment"].status == "ok"
    assert data.sections["control_points"].status == "ok"
    assert data.sections["densify"].status == "ok"
    assert any("classification_summary.json" in record.getMessage() for record in caplog.records)


def test_corrupt_run_metrics_does_not_raise(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    put(project, "reports/run_metrics.json", "{broken")
    data = collect_protocol_data(project)
    assert all(section.status in ALLOWED_STATUS for section in data.sections.values())
    assert data.sections["control_points"].status == "ok"
    assert any("run_metrics.json" in w for w in data.sections["densify"].warnings)


def test_processing_metrics_gives_source_run(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    put(project, "reports/processing_metrics.json", {
        "report_version": 1, "run_id": "20300101_000000Z", "software": {"version": "9.9.9", "git_hash": "abc1234"},
        "hardware": {"gpu_name": "Synthetic GPU", "ram_gb": 64},
        "import": {"source_type": "photos", "image_count": 4, "frame_interval": None, "discarded_frames": [],
                   "camera": {"make": "Synthetic", "model": "Cam"}, "gps": {"mode": "RTK", "rtk": True}},
        "alignment": {"registered_images": 4, "total_images": 4, "unregistered_images": [], "submodels": 1,
                      "tie_points": 777, "mean_track_length": 4.5, "mean_reprojection_error_px": 0.4,
                      "cameras": [], "images": [{"name": "img_a.jpg", "camera_id": 1, "reprojection_rms_px": 0.3}]},
        "overlap": {"cell_size": 1.0, "origin_xy": [0.0, 0.0], "shape": [1, 1], "counts": [[3]], "max": 3,
                    "mean_over_occupied": 3.0},
        "dense": {"points": 20000, "area_m2": 100.0, "density_pts_m2": 200.0},
        "refinement_stages": [{"stage": "depth_fill", "skipped": False, "added": 5}],
        "classification": {"confidence_histogram": {"bins": 2, "counts": [1, 2], "per_class": {
            "ground": {"p10": 0.1, "p50": 0.5, "p90": 0.9, "points": 3}}}},
        "warnings": ["WARNING: recorded warning"]})
    data = collect_protocol_data(project)
    assert data.sections["alignment"].source in ("run", "mixed")
    assert data.sections["alignment"].data["tie_points"] == 777
    assert data.sections["alignment"].data["overlap"]["max"] == 3
    assert data.sections["densify"].data["points"] == 20000 and data.sections["densify"].data["density_pts_m2"] == 200.0
    assert data.sections["title"].data["samcloud_version"] == "9.9.9"
    assert data.sections["import_"].source == "run" and data.sections["import_"].data["camera"]["make"] == "Synthetic"
    assert data.sections["refinement"].source == "run" and data.sections["refinement"].data["stages"][0]["added"] == 5
    assert data.sections["classification"].data["per_class_percentiles"]["ground"]["p50"] == 0.5
    assert data.sections["runtimes"].data["hardware"]["gpu_name"] == "Synthetic GPU"
    assert data.sections["appendix"].data["warnings"] == ["WARNING: recorded warning"]
    assert data.sections["appendix"].data["cameras_per_image"][0]["name"] == "img_a.jpg"


def test_collector_writes_only_protocol_data(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    before = {p for p in project.rglob("*") if p.is_file()}
    write_protocol_data(project, collect_protocol_data(project))
    after = {p for p in project.rglob("*") if p.is_file()}
    assert after - before == {project / "reports" / "protocol_data.json"}
