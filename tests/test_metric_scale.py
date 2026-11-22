"""Metric-scale decision (samcloud.core.metric_scale) and its use by the indoor plan and the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from samcloud.core import metric_scale as metric_scale_module
from samcloud.core.metric_scale import MetricScaleDecision, auto_metric_enabled, metric_scale_decision, plan_scale


@pytest.fixture()
def evidence_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-detected metric evidence enabled (config ``clean_plan.metric_scale: true``)."""
    monkeypatch.setattr(metric_scale_module, "auto_metric_enabled", lambda: True)


def _project(tmp_path: Path, name: str = "p") -> Path:
    project = tmp_path / name
    (project / "reports").mkdir(parents=True)
    (project / "products" / "dxf").mkdir(parents=True)
    return project


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _setting(project: Path, value: str) -> None:
    (project / "project.samcloud.yml").write_text(yaml.safe_dump({"project": {"name": project.name}, "settings": {"scale": {"cloud_units": value}}}), encoding="utf-8")


def test_unscaled_project_keeps_storey_height_heuristic(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _json(project / "products" / "dxf" / "state.json", {"scale": {"factor": 1.0, "source": "none"}})
    _json(project / "reports" / "apriltag_report.json", {"mode": "none", "scale_factor": None, "scale_applied": None})
    decision = metric_scale_decision(project)
    assert not decision.metric and decision.source == "none" and decision.evidence == ()
    scale = plan_scale(decision, 2.7, 2.5)
    assert scale.scale_source == "storey_height" and scale.units_per_metre == pytest.approx(2.7 / 2.5)
    assert plan_scale(decision, None).units_per_metre == 1.0 and plan_scale(decision, None).scale_source == "default"


def test_missing_project_folder_is_unscaled(tmp_path: Path) -> None:
    assert not metric_scale_decision(tmp_path / "does_not_exist").metric
    assert not metric_scale_decision(None).metric


@pytest.mark.parametrize(
    ("files", "source"),
    [
        ({"reports/apriltag_report.json": {"mode": "scale", "scale_factor": 1.24, "scale_applied": 1.24}}, "apriltag_scale"),
        ({"reports/apriltag_report.json": {"mode": "georef", "tags": []}}, "apriltag_georef"),
        ({"reports/control_points_report.json": {"source_frame": "local_tags"}}, "apriltag_survey"),
        ({"products/georeference.json": {"source_frame": "colmap_enu", "method": "gcp_bundle_adjustment", "scale": 1.0}}, "control_points"),
        ({"products/georeference.json": {"source_frame": "colmap_enu", "method": "camera_centres", "scale": 1.0}}, "georeferenced"),
        ({"products/dxf/state.json": {"scale": {"factor": 0.81, "source": "two_point"}}}, "dxf_scale"),
        ({"products/dxf/state.json": {"scale": {"factor": 1.1, "source": "numeric"}}}, "dxf_scale"),
    ],
)
def test_each_metric_source_is_recognised(tmp_path: Path, files: dict[str, object], source: str) -> None:
    project = _project(tmp_path)
    for relative, payload in files.items():
        _json(project / relative, payload)
    off = metric_scale_decision(project, use_evidence=False)
    assert not off.metric and off.evidence == (source,) and plan_scale(off, 2.46, 2.5).scale_source == "storey_height"
    decision = metric_scale_decision(project, use_evidence=True)
    assert decision.metric and decision.source == source and source in decision.evidence
    scale = plan_scale(decision, 2.46, 2.5)
    assert scale.units_per_metre == 1.0 and scale.scale_source == source


def test_gps_reference_is_metric(tmp_path: Path) -> None:
    project = _project(tmp_path)
    (project / "products" / "gps_ref.txt").write_text("img.jpg 47.0 8.0 450.0\n", encoding="utf-8")
    assert metric_scale_decision(project, use_evidence=True).source == "gps_reference"


def test_scale_report_without_applied_factor_is_not_metric(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _json(project / "reports" / "apriltag_report.json", {"mode": "scale", "scale_applied": None})
    assert not metric_scale_decision(project).metric


def test_explicit_project_setting_overrides_evidence(tmp_path: Path) -> None:
    metric = _project(tmp_path, "metric")
    _setting(metric, "metric")
    decision = metric_scale_decision(metric)
    assert decision.metric and decision.source == "project_setting"
    unscaled = _project(tmp_path, "unscaled")
    _json(unscaled / "reports" / "apriltag_report.json", {"mode": "scale", "scale_applied": 1.2})
    _setting(unscaled, "unscaled")
    decision = metric_scale_decision(unscaled)
    assert not decision.metric and decision.source == "project_setting" and decision.evidence == ("apriltag_scale",)
    auto = _project(tmp_path, "auto")
    _json(auto / "reports" / "apriltag_report.json", {"mode": "scale", "scale_applied": 1.2})
    _setting(auto, "auto")
    assert metric_scale_decision(auto, use_evidence=True).source == "apriltag_scale"


def test_explicit_units_per_metre_setting(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _json(project / "reports" / "apriltag_report.json", {"mode": "scale", "scale_applied": 1.2})
    (project / "project.samcloud.yml").write_text(yaml.safe_dump({"settings": {"scale": {"units_per_metre": 100, "cloud_units": "metric"}}}), encoding="utf-8")
    decision = metric_scale_decision(project)
    assert decision.units_per_metre == 100.0 and decision.known_scale and not decision.metric and decision.source == "project_setting"
    scale = plan_scale(decision, 250.0)
    assert scale.units_per_metre == 100.0 and scale.scale_source == "project_setting"
    assert "100" in decision.fingerprint_key()
    for bad in ("abc", -2, 0, True):
        (project / "project.samcloud.yml").write_text(yaml.safe_dump({"settings": {"scale": {"units_per_metre": bad}}}), encoding="utf-8")
        assert metric_scale_decision(project, use_evidence=True).source == "apriltag_scale"  # invalid numbers are ignored (auto)


def test_primary_source_priority_and_serialisation(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _json(project / "reports" / "apriltag_report.json", {"mode": "scale", "scale_applied": 1.2})
    _json(project / "products" / "dxf" / "state.json", {"scale": {"factor": 1.01, "source": "numeric"}})
    decision = metric_scale_decision(project, use_evidence=True)
    assert decision.source == "dxf_scale" and decision.evidence == ("dxf_scale", "apriltag_scale")
    payload = plan_scale(decision, 2.5).as_dict()
    assert payload["units_per_metre"] == 1.0 and payload["scale_source"] == "dxf_scale" and payload["metric_metric"] is True
    json.dumps(payload)


def test_indoor_plan_scale_and_bands(tmp_path: Path, evidence_on: None) -> None:
    from samcloud.dxf.levels import Level
    from samcloud.ortho.plan import CleanPlanSettings, _indoor_plan_scale, _indoor_scale_and_bands

    settings = CleanPlanSettings()
    level = Level(id="level_1", name="1", floor_z=0.1, ceiling_z=2.8, point_count=1000, confidence=1.0, low_confidence=False)
    height = 2.7
    upm, bands = _indoor_scale_and_bands(level, settings)
    assert upm == pytest.approx(height / settings.nominal_storey_height)
    metric_upm, metric_bands = _indoor_scale_and_bands(level, settings, MetricScaleDecision(True, "apriltag_scale"))
    assert metric_upm == 1.0
    assert metric_bands == bands  # bands follow the measured storey height in both cases
    assert metric_bands["wall"][1] == pytest.approx(height * settings.indoor_wall_band[1])

    project = _project(tmp_path)
    assert _indoor_plan_scale(project, level, settings).scale_source == "storey_height"
    _json(project / "reports" / "apriltag_report.json", {"mode": "scale", "scale_applied": 1.238})
    scale = _indoor_plan_scale(project, level, settings)
    assert scale.units_per_metre == 1.0 and scale.scale_source == "apriltag_scale"


def test_pipeline_surface_fusion_scale_follows_the_same_decision(tmp_path: Path, evidence_on: None) -> None:
    from samcloud.engine.pipeline import _resolve_units_per_metre

    project = _project(tmp_path)
    dense = project / "products" / "dense"
    dense.mkdir(parents=True)
    _json(project / "reports" / "apriltag_report.json", {"mode": "scale", "scale_applied": 1.238})
    summary = SimpleNamespace(settings={})
    args = SimpleNamespace(classes_json=str(tmp_path / "missing.json"), class_set="indoor")
    fill, fusion = _resolve_units_per_metre(args, dense, dense / "missing.ply", {"units_per_metre": "auto"}, {"units_per_metre": "auto"}, summary)
    assert fill["units_per_metre"] == 1.0 and fusion["units_per_metre"] == 1.0
    assert summary.settings["units_per_metre"] == {"units_per_metre": 1.0, "source": "apriltag_scale"}


def test_config_switch_gates_auto_evidence_but_not_explicit_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import samcloud.core.paths as paths

    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setattr(paths, "CONFIG_DIRECTORY", config)
    assert auto_metric_enabled() is False  # no config file -> off
    (config / "ortho_plan.yaml").write_text(yaml.safe_dump({"clean_plan": {"metric_scale": True}}), encoding="utf-8")
    assert auto_metric_enabled() is True
    project = _project(tmp_path)
    _json(project / "reports" / "apriltag_report.json", {"mode": "scale", "scale_applied": 1.2})
    assert metric_scale_decision(project).source == "apriltag_scale"
    (config / "ortho_plan.yaml").write_text(yaml.safe_dump({"clean_plan": {"metric_scale": False}}), encoding="utf-8")
    decision = metric_scale_decision(project)
    assert not decision.metric and decision.evidence == ("apriltag_scale",) and "metric_scale is off" in decision.detail
    _setting(project, "metric")
    assert metric_scale_decision(project).source == "project_setting"  # explicit setting applies with the switch off


def test_repository_config_declares_the_switch() -> None:
    from samcloud.core.configuration import load_yaml
    from samcloud.core.paths import CONFIG_DIRECTORY

    assert isinstance(load_yaml(CONFIG_DIRECTORY / "ortho_plan.yaml")["clean_plan"].get("metric_scale"), bool)


def test_detector_units_follow_the_storey_not_the_metric_scale() -> None:
    from samcloud.dxf.levels import Level
    from samcloud.ortho.plan import CleanPlanSettings, _indoor_detector_units

    settings = CleanPlanSettings()
    level = Level(id="level_1", name="1", floor_z=0.0, ceiling_z=2.47, point_count=1000, confidence=1.0, low_confidence=False)
    units = [_indoor_detector_units(level, settings, metric) for metric in (0.984, 0.9891, 0.994, 1.0, 1.01)]
    assert units == [pytest.approx(2.47 / settings.nominal_storey_height)] * 5  # one detector unit whatever the metric scale
    no_ceiling = Level(id="level_1", name="1", floor_z=0.0, ceiling_z=None, point_count=1000, confidence=1.0, low_confidence=False)
    assert _indoor_detector_units(no_ceiling, settings, 1.0) == 1.0


def test_detector_metres_are_converted_to_true_metres() -> None:
    from samcloud.ortho.plan import Opening, _layers_in_metres, _openings_in_metres, _pieces_in_metres

    detector, metric = 0.988, 1.0
    pieces = _pieces_in_metres([{"thickness": 0.30, "kind": "exterior"}, {"kind": "x"}], detector, metric)
    assert pieces[0]["thickness"] == pytest.approx(0.30 * detector / metric) and pieces[1] == {"kind": "x"}
    layers = {"A-OBEN": [{"_z_range_m": [1.5, 2.1], "points": []}], "A-EINB": [{"_height_m": 2.0}, {"type": "text"}]}
    converted = _layers_in_metres(layers, detector, metric)
    assert converted["A-OBEN"][0]["_z_range_m"] == pytest.approx([1.5 * detector, 2.1 * detector])
    assert converted["A-EINB"][0]["_height_m"] == pytest.approx(2.0 * detector) and converted["A-EINB"][1] is layers["A-EINB"][1]
    assert layers["A-OBEN"][0]["_z_range_m"] == [1.5, 2.1]  # the DXF entities stay untouched
    assert _layers_in_metres(layers, 1.0, 1.0) is layers and _pieces_in_metres(None, detector, metric) is None
    opening = Opening(kind="DOOR", center=(0.0, 0.0), width=0.9, axis_angle=0.0, wall_thickness=0.2, sill_m=0.0, head_m=2.05)
    assert _openings_in_metres([opening], detector, metric)[0].head_m == pytest.approx(2.05 * detector)


def test_refine_plan_is_returned_in_detector_units(tmp_path: Path) -> None:
    from samcloud.bim.plan_model import BuildingPlan, PlanOpening, PlanRoom, PlanWall, plan_in_detector_units

    plan = BuildingPlan(
        project_name="p", level_id="level_1", level_name="1", units_per_metre=1.0, frame={"origin": [0, 0, 0], "scale_factor": 1.0, "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        floor_z=0.0, ceiling_z=2.47, walls=[PlanWall(id="W1", centreline=[[0, 0], [4, 0]], thickness_m=0.3)],
        rooms=[PlanRoom(id="R1", name="Bad", polygon={"exterior": [[0, 0], [2, 0], [2, 2], [0, 2]], "holes": []}, area_m2=4.0, label_point=[1, 1])],
        openings=[PlanOpening(id="O1", kind="door", center=[1.0, 0.0], width_m=0.9, axis_angle=0.0, wall_thickness_m=0.3, host_wall_id="W1", sill_m=0.0, head_m=2.0)],
        furniture=[], wall_bodies=[], source={"scale": {"detector_units_per_metre": 0.988}},
    )
    detector = plan_in_detector_units(plan)
    assert detector.units_per_metre == 0.988
    # every *_m value times units_per_metre is the same local length as before
    assert detector.walls[0].thickness_m * 0.988 == pytest.approx(0.3)
    assert detector.openings[0].width_m * 0.988 == pytest.approx(0.9) and detector.openings[0].head_m * 0.988 == pytest.approx(2.0)
    assert detector.rooms[0].area_m2 * 0.988 ** 2 == pytest.approx(4.0)
    plain = BuildingPlan(**{**{f: getattr(plan, f) for f in plan.__slots__}, "source": {}})
    assert plan_in_detector_units(plain) is plain
