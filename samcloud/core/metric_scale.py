"""Decide whether a project's point cloud is already metric, and why.

Indoor plans and the BIM export convert local cloud units to metres with ``units_per_metre``.  An unscaled 360 scan
has an arbitrary unit, so the plan falls back to the storey-height heuristic (floor-to-ceiling distance against a
nominal storey).  A cloud that is ALREADY metric must not be rescaled by that heuristic: a flat with 2.70 m ceilings
would otherwise come out 8 % too small.

Recognised metric evidence (the pipeline records each one as a product or report of the project folder):

* ``project_setting``   ``project.samcloud.yml`` -> ``settings.scale.cloud_units``: ``metric`` | ``unscaled`` | ``auto``
                         or ``settings.scale.units_per_metre: <number>`` (known unit, e.g. 100 for centimetres);
                         an explicit setting overrides all evidence, in both directions
* ``dxf_scale``         ``products/dxf/state.json`` scale source ``numeric`` / ``two_point`` (the DXF frame applies the
                         user's factor, so the plan's local units are metres)
* ``control_points``    ``products/georeference.json`` from a control-point (GCP) bundle adjustment
* ``georeferenced``     ``products/georeference.json`` (GPS / RTK fit of the camera centres)
* ``apriltag_survey``   ``reports/control_points_report.json`` with the local AprilTag survey frame
* ``apriltag_georef``   ``reports/apriltag_report.json`` mode ``georef``
* ``apriltag_scale``    ``reports/apriltag_report.json`` mode ``scale`` with an applied scale factor
* ``gps_reference``     ``products/gps_ref.txt`` (camera GPS / RTK reference the sparse model is aligned to)

Without evidence the decision is ``metric=False`` (``source="none"``) and callers keep their storey-height heuristic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_FILENAME = "project.samcloud.yml"
LOCAL_TAGS_FRAME = "local_tags"
METRIC_SETTING_VALUES = {"metric", "metre", "metres", "meter", "meters", "m", "true", "yes", "1"}
UNSCALED_SETTING_VALUES = {"unscaled", "arbitrary", "storey_height", "false", "no", "0"}

# Order = priority of the reported primary source (all evidence is listed in ``evidence``).
_SOURCE_ORDER = ("dxf_scale", "control_points", "georeferenced", "apriltag_survey", "apriltag_georef", "apriltag_scale", "gps_reference")


@dataclass(frozen=True)
class MetricScaleDecision:
    """Whether the project's cloud coordinates are metres, the primary reason, and every piece of evidence found."""

    metric: bool
    source: str = "none"
    detail: str = "no metric reference found (unscaled cloud)"
    evidence: tuple[str, ...] = field(default_factory=tuple)
    units_per_metre: float | None = None  # explicit known scale (project setting ``settings.scale.units_per_metre``)

    def as_dict(self) -> dict[str, Any]:
        return {"metric": bool(self.metric), "source": self.source, "detail": self.detail, "evidence": list(self.evidence),
                **({"units_per_metre": float(self.units_per_metre)} if self.units_per_metre is not None else {})}

    def fingerprint_key(self) -> str:
        return f"{int(self.metric)}:{self.source}" + (f":{self.units_per_metre:.9g}" if self.units_per_metre is not None else "")

    @property
    def known_scale(self) -> bool:
        """True when the cloud unit is known (metric, or an explicit units-per-metre setting)."""
        return bool(self.metric) or self.units_per_metre is not None


@dataclass(frozen=True)
class PlanScale:
    """Units per metre chosen for an indoor plan and the reason (``scale_source``)."""

    units_per_metre: float
    scale_source: str
    metric: MetricScaleDecision

    def as_dict(self) -> dict[str, Any]:
        return {"units_per_metre": float(self.units_per_metre), "scale_source": self.scale_source, **{f"metric_{k}": v for k, v in self.metric.as_dict().items()}}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _project_setting(project: Path) -> str | float | None:
    """``settings.scale`` of project.samcloud.yml: ``units_per_metre`` (a positive number, wins) or ``cloud_units``
    normalised to ``metric`` / ``unscaled``; None = auto."""
    path = project / PROJECT_FILENAME
    if not path.is_file():
        return None
    try:
        import yaml

        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - an unreadable project file means "no explicit setting"
        return None
    settings = document.get("settings") if isinstance(document, dict) else None
    scale = settings.get("scale") if isinstance(settings, dict) else None
    explicit = scale.get("units_per_metre") if isinstance(scale, dict) else None
    if explicit is not None and not isinstance(explicit, bool):
        try:
            number = float(explicit)
        except (TypeError, ValueError):
            number = float("nan")
        if math.isfinite(number) and number > 0.0:
            return number
    value = scale.get("cloud_units") if isinstance(scale, dict) else None
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in METRIC_SETTING_VALUES:
        return "metric"
    if text in UNSCALED_SETTING_VALUES:
        return "unscaled"
    return None


def _dxf_scale(project: Path) -> str | None:
    state = _read_json(project / "products" / "dxf" / "state.json")
    scale = state.get("scale") if isinstance(state, dict) else None
    if not isinstance(scale, dict) or str(scale.get("source", "none")) not in ("numeric", "two_point"):
        return None
    try:
        factor = float(scale.get("factor", 1.0))
    except (TypeError, ValueError):
        return None
    return f"DXF scale {scale.get('source')} factor {factor:.6g}" if math.isfinite(factor) and factor > 0 else None


def _georeference(project: Path) -> tuple[str, str] | None:
    path = project / "products" / "georeference.json"
    if not path.is_file():
        return None
    data = _read_json(path)
    method = str(data.get("method", "")) if isinstance(data, dict) else ""
    if "gcp" in method.lower() or "control" in method.lower():
        return "control_points", f"georeference.json ({method})"
    return "georeferenced", f"georeference.json ({method or 'gps fit'})"


def _apriltag(project: Path) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    survey = _read_json(project / "reports" / "control_points_report.json")
    if isinstance(survey, dict) and survey.get("source_frame") == LOCAL_TAGS_FRAME:
        found.append(("apriltag_survey", "control_points_report.json (local AprilTag survey frame)"))
    report = _read_json(project / "reports" / "apriltag_report.json")
    if isinstance(report, dict):
        mode = str(report.get("mode", "none"))
        if mode == "georef":
            found.append(("apriltag_georef", "apriltag_report.json mode georef"))
        elif mode == "scale":
            try:
                applied = float(report.get("scale_applied"))
            except (TypeError, ValueError):
                applied = float("nan")
            if math.isfinite(applied) and applied > 0.0:
                found.append(("apriltag_scale", f"apriltag_report.json mode scale, factor {applied:.6g}"))
    return found


def auto_metric_enabled() -> bool:
    """Switch ``clean_plan.metric_scale`` of config/ortho_plan.yaml: use AUTO-DETECTED metric evidence.

    Off (or missing), the evidence is still detected and recorded, but plans keep the storey-height scale; explicit project
    settings (``settings.scale``) are honoured either way.  The plan detectors run in storey-normalised units whatever the
    switch says (ortho/plan.py ``_indoor_detector_units``), so the switch only changes the metric conversion of the results.
    """
    try:
        from samcloud.core.configuration import load_yaml
        from samcloud.core.paths import CONFIG_DIRECTORY

        data = load_yaml(CONFIG_DIRECTORY / "ortho_plan.yaml") or {}
        section = data.get("clean_plan") if isinstance(data, dict) else None
        value = section.get("metric_scale", False) if isinstance(section, dict) else False
    except Exception:  # noqa: BLE001 - an unreadable config keeps today's behaviour
        return False
    if isinstance(value, str):
        return value.strip().lower() in METRIC_SETTING_VALUES | {"on", "auto", "enabled"}
    return bool(value)


def metric_scale_decision(project_dir: Path | str | None, *, use_evidence: bool | None = None) -> MetricScaleDecision:
    """Return whether ``project_dir``'s cloud is metric (never raises; missing files simply give no evidence).

    ``use_evidence`` (None = config switch :func:`auto_metric_enabled`) decides whether auto-detected evidence makes
    the cloud metric; explicit project settings always apply.
    """
    if project_dir is None:
        return MetricScaleDecision(False)
    project = Path(project_dir)
    details: dict[str, str] = {}
    try:
        if (text := _dxf_scale(project)) is not None:
            details["dxf_scale"] = text
        if (georef := _georeference(project)) is not None:
            details[georef[0]] = georef[1]
        for source, text in _apriltag(project):
            details[source] = text
        if (project / "products" / "gps_ref.txt").is_file():
            details["gps_reference"] = "products/gps_ref.txt"
        setting = _project_setting(project)
    except OSError:
        return MetricScaleDecision(False, "none", "project folder unreadable")
    evidence = tuple(source for source in _SOURCE_ORDER if source in details)
    if isinstance(setting, float):
        return MetricScaleDecision(abs(setting - 1.0) <= 1e-12, "project_setting", f"project.samcloud.yml settings.scale.units_per_metre = {setting:.9g}", evidence, setting)
    if setting == "metric":
        return MetricScaleDecision(True, "project_setting", "project.samcloud.yml settings.scale.cloud_units = metric", evidence)
    if setting == "unscaled":
        return MetricScaleDecision(False, "project_setting", "project.samcloud.yml settings.scale.cloud_units = unscaled", evidence)
    if evidence:
        if use_evidence is None:
            use_evidence = auto_metric_enabled()
        if use_evidence:
            return MetricScaleDecision(True, evidence[0], details[evidence[0]], evidence)
        return MetricScaleDecision(False, "none", f"metric evidence found ({details[evidence[0]]}) but clean_plan.metric_scale is off", evidence)
    return MetricScaleDecision(False)


def plan_scale(decision: MetricScaleDecision, storey_height_local: float | None, nominal_storey_height_m: float = 2.5) -> PlanScale:
    """Units per metre of an indoor plan: an explicit setting, 1.0 for a metric cloud, else the storey-height heuristic
    (1.0 without a storey)."""
    if decision.units_per_metre is not None:
        return PlanScale(float(decision.units_per_metre), decision.source, decision)
    if decision.metric:
        return PlanScale(1.0, decision.source, decision)
    height = None if storey_height_local is None else float(storey_height_local)
    if height is None or not math.isfinite(height) or height <= 0.0 or not nominal_storey_height_m > 0.0:
        return PlanScale(1.0, "default", decision)
    return PlanScale(height / float(nominal_storey_height_m), "storey_height", decision)
