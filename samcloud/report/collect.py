"""Collector of the processing protocol data (schema contract for the PDF layer).

``collect_protocol_data(project_dir, language=None) -> ProtocolData`` reads a project (read-only) and
``write_protocol_data(project_dir, data) -> Path`` writes ``reports/protocol_data.json`` (the ONLY file the
collector writes). Future runs record their metrics in ``reports/processing_metrics.json`` (source "run");
for old projects the collector derives what it can from existing reports, logs, the COLMAP model and the score
files (source "derived"; "mixed" when both were used). Values that cannot be obtained are ``None`` and listed in
the section's ``data["not_recorded"]`` (dotted names for one nested level). Nothing is guessed.

Structure
---------
``ProtocolData`` = ``project{name, path, created_at}``, ``language``, ``generated_at``, ``version`` (1) and
``sections`` (an ordered dict, in exactly this order): ``title, summary, import_, alignment, control_points,
densify, refine, classification, refinement, products, runtimes, appendix``.
Every section is ``Section(key, status, source, data, warnings)``; ``status`` in {"ok", "not_run", "no_data",
"error"}, ``source`` in {"run", "derived", "mixed", None}. A section is collected in its own try/except; a missing
file gives "no_data"/"not_run", a corrupt one (invalid JSON, wrong types) "error" or a warning, never an exception.
Units stay in key suffixes (``_m``, ``_px``, ``_s``, ``_pct``, ``_mm``, ``_m2``); the PDF layer converts to cm.

Section data keys
-----------------
title: project_name, date (ISO date of the run), samcloud_version, git_hash, capture_type (drone_gps,
    panorama_360, camera, ...), class_set, crs, overview_image (project-relative path or None),
    control_points_mode ("with" | "without").
summary: images_registered, images_total, reprojection_error_px, rms_pass_x_m, rms_pass_y_m, rms_pass_z_m,
    rms_pass_3d_m, rms_control_x_m, rms_control_y_m, rms_control_z_m, rms_control_3d_m, apriltag_scale_factor,
    apriltag_max_deviation_pct, dense_points, filtered_points (final classified cloud), total_seconds.
import_: source ("photos" | "video" | "panorama"), image_count, frame_count, frame_interval_s,
    panorama_to_cubemap{panoramas, faces, face_size} (panorama only, else None), camera{make, model, focal_mm,
    focal_35mm, sensor{pixel_width, pixel_height}} or None, discarded_frames[{kept_frame, source_path,
    time_seconds, sharpness, kept_sharpness}] (at most 5000), discarded_count, gps{mode, rtk, max_error_m}.
alignment: registered, total, registered_pct, unregistered_images[], submodels, tie_points, mean_track_length,
    reprojection_error_px, matching_strategy, camera_model, cameras[{camera_id, model, width, height,
    params{...}}], images[{name, camera_id, center_xyz, reprojection_rms_px, observations}],
    overlap{cell_size, origin_xy, shape, counts, max, mean_over_occupied}, tiepoint_filter_passes[].
control_points: points[{id, type ("target" | "apriltag"), role ("pass" | "control"), enabled, status,
    error_x_m, error_y_m, error_z_m, error_3d_m, images, excluded_reason}], rms_pass{x_m, y_m, z_m, xy_m, 3d_m,
    mean_reprojection_px, count}, rms_control{same}, height_shift_m, height_shift_applied, gcp_vs_shift{pre_gcp,
    post_gcp (checkpoint totals: rms_e_m, rms_n_m, rms_h_m, rms_2d_m, rms_3d_m, bias_h_m, n_points, name),
    height_shift_m}, checkpoints{name: {totals, points[{name, d_e_m, d_n_m, d_h_m, d_3d_m, status}]}},
    apriltag{mode, scale_factor, scale_applied, warnings[], tags[{id, images, size_mm_before, size_mm_after,
    deviation_pct, status}]} or None, gate{passed, reasons[]}, local_points_xy[{name, role, x_m, y_m, err_x_m,
    err_y_m}] (positions relative to the centroid of the points, for the error-vector plot only), reason
    ("without_control_points" when status is "not_run").
densify: settings{max_image_size, num_iterations, patch_match, size_reason, excluded_images[]}, points,
    density_pts_m2, area_m2, density_reason, density_cloud, seconds.
refine: executed, skipped_reason, metrics{unknown_before, unknown_after, set_to_unknown, smoothed_points,
    runtime_seconds}.
classification: model (mode), engine, thresholds{background, query}, classes[{name, points, share_pct,
    instances}], unknown_points, total_points, confidence_histogram{bins, edges, counts, points, unknown_points,
    mean_conf, per_class}, per_class_percentiles{name: {p10, p50, p90, points}}.
refinement: stages[{stage, before, after, removed, added, skipped, reason, seconds}], final_points.
products: files[{kind (las, ortho, dsm, dtm, dxf, ifc, glb, potree), path (project-relative), size_bytes, crs}],
    dxf_qa{passed, failed, rules[{rule, ok, value, threshold}]} or None.
runtimes: steps[{name, seconds}] (pipeline stages, DXF/IFC steps, environment steps), total_seconds, hardware{
    gpu_name, vram_gb, ram_gb, cpu, cpu_count, os}.
appendix: cameras_per_image[{name, camera_id, reprojection_rms_px}], warnings[].
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from samcloud.report import metrics
from samcloud.report import run_metrics_writer as writer

LOGGER = logging.getLogger(__name__)
PROTOCOL_DATA_VERSION = 1
SECTION_KEYS = ("title", "summary", "import_", "alignment", "control_points", "densify", "refine",
                "classification", "refinement", "products", "runtimes", "appendix")
MAX_DISCARDED = 5000
_TAG_NAME = re.compile(r"(?i)^tag[\s_-]?\d+$")


@dataclass
class Section:
    key: str
    status: str
    source: str | None
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ProtocolData:
    project: dict[str, Any]
    sections: dict[str, Section]
    language: str | None = None
    generated_at: str = ""
    version: int = PROTOCOL_DATA_VERSION

    def section(self, key: str) -> Section:
        return self.sections[key]

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "generated_at": self.generated_at, "language": self.language,
                "project": self.project, "sections": {key: asdict(value) for key, value in self.sections.items()}}


# ----------------------------------------------------------------------------------- plumbing


class _Builder:
    """Collects one section: JSON access with corrupt-file handling, run/derived tracking, final status."""

    def __init__(self, project_dir: Path, key: str, cache: dict[Path, Any]) -> None:
        self.project_dir, self.key, self._cache = project_dir, key, cache
        self.data: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.run = self.derived = self.has = self.corrupt = False
        self.na: dict[str, str] = {}  # field -> reason code: "not applicable" (differs from "not recorded")

    def json(self, relative: str, kind: type | tuple = dict) -> Any:
        """Parsed JSON of ``relative`` or None (missing). Corrupt/wrong-type files add a warning, return None."""
        path = self.project_dir / relative
        if path not in self._cache:
            if not path.is_file():
                self._cache[path] = None
            else:
                try:
                    self._cache[path] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as error:
                    self._cache[path] = _Corrupt(f"{relative}: unreadable ({error})")
        value = self._cache[path]
        if isinstance(value, _Corrupt):
            return self._bad(value.message)
        if value is not None and not isinstance(value, kind):
            return self._bad(f"{relative}: unexpected content type {type(value).__name__}")
        return value

    def _bad(self, message: str) -> None:
        self.corrupt = True
        if message not in self.warnings:
            self.warnings.append(message)
            LOGGER.warning("protocol %s: %s", self.key, message)
        return None

    def processing(self) -> dict[str, Any] | None:
        return self.json("reports/processing_metrics.json")

    def run_metrics(self) -> dict[str, Any] | None:
        return self.json("reports/run_metrics.json")

    def finish(self, status: str | None = None) -> Section:
        data = _with_not_recorded(self.data)
        if self.na:
            data["not_recorded"] = [name for name in data["not_recorded"]
                                    if not any(name == key or name.startswith(key + ".") for key in self.na)]
        data["not_applicable"] = dict(self.na)
        if status is None:
            status = "ok" if self.has else ("error" if self.corrupt else "no_data")
        source = "mixed" if self.run and self.derived else "run" if self.run else "derived" if self.derived else None
        return Section(self.key, status, source, data, self.warnings)


class _Corrupt:
    def __init__(self, message: str) -> None:
        self.message = message


def _with_not_recorded(data: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    for key, value in data.items():
        if key == "not_recorded" or key.endswith("reason"):
            continue
        if value is None:
            missing.append(key)
        elif isinstance(value, dict) and key not in ("params", "per_class", "per_class_percentiles"):
            missing.extend(f"{key}.{sub}" for sub, inner in value.items() if inner is None and not sub.endswith("reason"))
    data["not_recorded"] = missing
    return data


def _num(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


_clean = writer.clean_json


def _rel(project_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_dir.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


# ----------------------------------------------------------------------------------- sections


def _title(b: _Builder, config: dict[str, Any], config_error: str | None) -> Section:
    run = b.run_metrics()
    processing = b.processing()
    settings = _dict(_dict(run).get("settings"))
    project = _dict(config.get("project"))
    acquisition = _dict(config.get("acquisition"))
    software = _dict(_dict(processing).get("software"))
    if processing and software:
        b.run = True
    crs = settings.get("export_crs")
    if not crs:
        terrain = _dict(b.json("products/environment/terrain_report.json"))
        crs = _dict(terrain.get("crs")).get("epsg")
        crs = f"EPSG:{crs}" if isinstance(crs, int) else crs
    finished = _dict(run).get("finished_at") or _dict(config.get("processing")).get("completed_at")
    overview = _overview_image(b.project_dir)
    mode = _control_points_mode(b.project_dir, config, b)
    b.data = {
        "project_name": project.get("name") or b.project_dir.name,
        "date": str(finished)[:10] if finished else None,
        "samcloud_version": software.get("version"), "git_hash": software.get("git_hash"),
        "capture_type": settings.get("capture_type") or acquisition.get("type"),
        "class_set": settings.get("class_set"), "crs": crs or None,
        "overview_image": overview, "control_points_mode": mode,
    }
    b.derived = True
    b.has = bool(config) or bool(run)
    if config_error:
        b.warnings.append(config_error)
    return b.finish()


def _control_points_mode(project_dir: Path, config: dict[str, Any], b: _Builder) -> str:
    report = _dict(b.json("reports/control_points_report.json"))
    if _list(report.get("points")):
        return "with"
    workflow = _dict(config.get("settings", {}).get("workflow") if isinstance(config.get("settings"), dict) else {})
    mode = workflow.get("control_points_mode")
    if mode in ("automatic", "manual", "csv", "file") and (project_dir / "control_points" / "control_points.csv").is_file():
        return "with"
    return "without"


def _overview_image(project_dir: Path) -> str | None:
    candidates = [project_dir / "products" / "environment" / "ortho_rgb.tif"]
    ortho = project_dir / "products" / "ortho"
    if ortho.is_dir():
        # Real photos first; colour_source.png is a label mask (values 0..2, almost black), only a last resort.
        for name in ("ortho.png", "preview.jpg", "ortho_4096.jpg"):
            candidates += sorted(ortho.glob(f"*/{name}"))
        candidates += sorted(ortho.glob("**/*.tif"))
    candidates.append(project_dir / "products" / "environment" / "hillshade.png")
    if ortho.is_dir():
        candidates += sorted(ortho.glob("*/colour_source.png"))
    found = next((path for path in candidates if path.is_file()), None)
    return _rel(project_dir, found) if found else None


def _import(b: _Builder) -> Section:
    processing = _dict(b.processing())
    recorded = _dict(processing.get("import"))
    run_metrics = b.run_metrics()
    base = recorded if recorded else writer.import_block(b.project_dir, run_metrics)
    b.run, b.derived = bool(recorded), not recorded
    discarded = _list(base.get("discarded_frames"))
    frames = frame_count = None
    cube: dict[str, Any] | None = None
    manifest = b.json("inputs/video_frames_manifest.json")
    if manifest:
        frames = len(_list(manifest.get("frames")))
        b.derived = True
        if not discarded:
            discarded = _list(manifest.get("discarded"))
    pano = b.json("inputs/panorama_manifest.json")
    if pano:
        panoramas = _list(pano.get("panoramas"))
        cube = {"panoramas": len(panoramas),
                "faces": sum(len(_list(_dict(item).get("faces"))) for item in panoramas),
                "face_size": pano.get("face_size")}
        frames = sum(1 for item in panoramas if _dict(item).get("source_kind") == "video_frame") or None
        b.derived = True
    frame_count = frames
    if base.get("source_type") in ("video", "panorama") and base.get("camera") is None:
        b.na["camera"] = "not_applicable"
    b.has = bool(base) and base.get("source_type") is not None
    b.data = {
        "source": base.get("source_type"), "image_count": base.get("image_count"), "frame_count": frame_count,
        "frame_interval_s": base.get("frame_interval") if base.get("source_type") != "photos" else None,
        "panorama_to_cubemap": cube, "camera": base.get("camera"),
        "discarded_frames": discarded[:MAX_DISCARDED], "discarded_count": len(discarded),
        "gps": _dict(base.get("gps")) or {"mode": None, "rtk": None, "max_error_m": None},
    }
    return b.finish()


def _tiepoint_passes(run_metrics: dict[str, Any] | None) -> list[Any]:
    quality = _dict(_dict(_dict(run_metrics).get("settings")).get("alignment_quality"))
    return _list(_dict(quality.get("tiepoint_filter")).get("passes"))


def _alignment(b: _Builder) -> Section:
    processing = _dict(b.processing())
    run_metrics = b.run_metrics()
    settings = _dict(_dict(run_metrics).get("settings"))
    align = _dict(processing.get("alignment")) or None
    overlap = _dict(processing.get("overlap")) or None
    if align:
        b.run = True
    else:
        model = writer.final_sparse_model(b.project_dir)
        if model is not None:
            try:
                align = metrics.alignment_metrics(model, writer.input_image_names(b.project_dir))
                b.derived = True
            except Exception as error:  # noqa: BLE001 - unreadable model
                b.warnings.append(f"alignment: sparse model unreadable ({error})")
    if overlap is None and align is not None and writer.final_sparse_model(b.project_dir) is not None:
        try:
            from samcloud.report.config import load_protocol_config

            overlap = metrics.overlap_grid(writer.final_sparse_model(b.project_dir),
                                           max_cells=int(load_protocol_config()["overlap"]["max_cells"]))
            b.derived = True
        except Exception as error:  # noqa: BLE001
            b.warnings.append(f"overlap: not computed ({error})")
    elif overlap:
        b.run = True
    passes = _tiepoint_passes(run_metrics)
    quality = _dict(settings.get("alignment_quality"))
    if align is None:
        # Old project whose sparse model was removed: only the run metrics remain.
        last = _dict(_dict(passes[-1]).get("after_filter")) if passes else {}
        models = _list(settings.get("sparse_models"))
        registered = _dict(models[0]).get("registered_images") if models else None
        if last or registered is not None:
            b.derived = True
            align = {"registered_images": registered, "total_images": settings.get("image_count"),
                     "unregistered_images": [], "submodels": len(models) or None, "tie_points": last.get("points"),
                     "mean_track_length": last.get("mean_track_length"),
                     "mean_reprojection_error_px": last.get("mean_reprojection_error"), "cameras": [], "images": []}
    align = align or {}
    registered, total = align.get("registered_images"), align.get("total_images")
    if total is None:
        total = settings.get("image_count") or _dict(run_metrics).get("images")
    b.has = bool(align)
    b.data = {
        "registered": registered, "total": total,
        "registered_pct": (registered / total * 100.0) if _num(registered) is not None and _num(total) else None,
        "unregistered_images": _list(align.get("unregistered_images")),
        "submodels": align.get("submodels"), "tie_points": align.get("tie_points"),
        "mean_track_length": align.get("mean_track_length"),
        "reprojection_error_px": align.get("mean_reprojection_error_px"),
        "matching_strategy": quality.get("matching_strategy"), "camera_model": quality.get("camera_model"),
        "cameras": _list(align.get("cameras")), "images": _list(align.get("images")),
        "overlap": overlap, "tiepoint_filter_passes": passes,
    }
    if align and total is not None and not _list(align.get("unregistered_images")) and align.get("total_images") is None:
        b.warnings.append("alignment: unregistered image names unknown (input list missing)")
    return b.finish()


def _error_points(report: dict[str, Any]) -> list[dict[str, Any]]:
    points = []
    for item in _list(report.get("points")):
        item = _dict(item)
        name = str(item.get("name"))
        status = item.get("status")
        enabled = item.get("enabled")
        excluded = None
        if enabled is False:
            excluded = "disabled"
        elif status not in (None, "accepted", "ok"):
            excluded = str(item.get("reason") or status)
        points.append({
            "id": name, "type": "apriltag" if _TAG_NAME.match(name) else "target", "role": item.get("role"),
            "enabled": enabled, "status": status, "error_x_m": _num(item.get("error_x_m")),
            "error_y_m": _num(item.get("error_y_m")), "error_z_m": _num(item.get("error_z_m")),
            "error_3d_m": _num(item.get("error_3d_m")), "images": item.get("projections"),
            "excluded_reason": excluded,
        })
    return points


def _checkpoint_sets(b: _Builder) -> dict[str, Any]:
    sets: dict[str, Any] = {}
    reports = b.project_dir / "reports"
    for path in sorted(reports.glob("checkpoints_*.json")) if reports.is_dir() else []:
        data = b.json(f"reports/{path.name}")
        if not data:
            continue
        totals = _dict(data.get("totals"))
        sets[path.stem[len("checkpoints_"):]] = {
            "totals": {"rms_e_m": _num(totals.get("rms_E")), "rms_n_m": _num(totals.get("rms_N")),
                       "rms_h_m": _num(totals.get("rms_H")), "rms_2d_m": _num(totals.get("rms_2D")),
                       "rms_3d_m": _num(totals.get("rms_3D")), "bias_h_m": _num(totals.get("bias_H")),
                       "n_points": totals.get("n_points", len(_list(data.get("points"))))},
            "points": [{"name": p.get("name"), "d_e_m": _num(p.get("dE")), "d_n_m": _num(p.get("dN")),
                        "d_h_m": _num(p.get("dH")), "d_3d_m": _num(p.get("d3D")), "status": p.get("status"),
                        "measured": p.get("measured")} for p in map(_dict, _list(data.get("points")))],
        }
    return sets


def _strip_measured(sets: dict[str, Any]) -> dict[str, Any]:
    """Checkpoint sets without the absolute survey coordinates (only used for the local plot)."""
    return {name: {"totals": entry["totals"],
                   "points": [{k: v for k, v in p.items() if k != "measured"} for p in entry["points"]]}
            for name, entry in sets.items()}


def _rms_block(raw: Any) -> dict[str, Any] | None:
    raw = _dict(raw)
    if not raw:
        return None
    return {"x_m": _num(raw.get("rms_x_m")), "y_m": _num(raw.get("rms_y_m")), "z_m": _num(raw.get("rms_z_m")),
            "xy_m": _num(raw.get("rms_xy_m")), "3d_m": _num(raw.get("rms_3d_m")),
            "mean_reprojection_px": _num(raw.get("mean_reprojection_px")), "count": raw.get("count")}


def _control_points(b: _Builder) -> Section:
    report = b.json("reports/control_points_report.json")
    tags = b.json("reports/apriltag_report.json")
    if tags is None:
        embedded = _dict(_dict(b.run_metrics()).get("settings")).get("apriltags")
        tags = embedded if isinstance(embedded, dict) and embedded.get("tags") else None
    sets = _checkpoint_sets(b)
    apriltag = None
    if tags:
        apriltag = {"mode": tags.get("mode"), "scale_factor": _num(tags.get("scale_factor")),
                    "scale_applied": _num(tags.get("scale_applied")), "warnings": _list(tags.get("warnings")),
                    "tags": [{key: _dict(tag).get(key) for key in
                              ("id", "images", "size_mm_before", "size_mm_after", "deviation_pct", "status")}
                             for tag in _list(tags.get("tags"))]}
    b.derived = bool(report or tags or sets)
    if not report or not _list(report.get("points")):
        b.has = bool(apriltag or sets)
        b.data = {"points": [], "rms_pass": None, "rms_control": None, "height_shift_m": None,
                  "gcp_vs_shift": None, "checkpoints": _strip_measured(sets), "apriltag": apriltag, "gate": None,
                  "local_points_xy": [], "reason": "without_control_points"}
        b.data["checkpoints"] = _strip_measured(sets)
        b.na.update({"rms_pass": "without_control_points", "rms_control": "without_control_points",
                     "height_shift_m": "without_control_points", "gcp_vs_shift": "without_control_points",
                     "gate": "without_control_points"})
        if not apriltag:
            b.na["apriltag"] = "no_apriltags"
        return b.finish("error" if b.corrupt else "not_run")
    totals = _dict(report.get("totals"))
    quick = _dict(report.get("quick_correction"))
    points = _error_points(report)
    pre = sets.get("pre_gcp")
    post_name = next((name for name in sets if name != "pre_gcp" and ("gcp" in name or "after" in name)), None)
    gcp_vs_shift = None
    if pre or post_name:
        gcp_vs_shift = {"pre_gcp": dict(pre["totals"], name="pre_gcp") if pre else None,
                        "post_gcp": dict(sets[post_name]["totals"], name=post_name) if post_name else None,
                        "height_shift_m": _num(quick.get("mean_dH_m", quick.get("height_offset_m")))}
    b.has = True
    b.data = {
        "points": points, "rms_pass": _rms_block(_dict(totals.get("pass"))),
        "rms_control": _rms_block(_dict(totals.get("control"))),
        "height_shift_m": _num(quick.get("height_offset_m", quick.get("mean_dH_m"))),
        "height_shift_applied": quick.get("applied"), "gcp_vs_shift": gcp_vs_shift,
        "checkpoints": _strip_measured(sets), "apriltag": apriltag,
        "gate": {"passed": _dict(report.get("gate")).get("passed"), "reasons": _list(_dict(report.get("gate")).get("reasons"))},
        "local_points_xy": _local_points(points, sets, tags), "reason": None,
    }
    if not any(p["role"] == "control" for p in points):
        b.na["rms_control"] = "no_control_points"
    if not any(p["role"] == "pass" for p in points):
        b.na["rms_pass"] = "no_pass_points"
    if gcp_vs_shift is None:
        b.na["gcp_vs_shift"] = "no_gcp_run"
    if not apriltag:
        b.na["apriltag"] = "no_apriltags"
    return b.finish()


def _local_points(points: list[dict[str, Any]], sets: dict[str, Any], tags: Any) -> list[dict[str, Any]]:
    """Point positions relative to their centroid: survey positions of the check points, else AprilTag centres."""
    position: dict[str, tuple[float, float]] = {}
    for entry in sets.values():
        for item in entry["points"]:
            measured = item.get("measured")
            if isinstance(measured, list) and len(measured) >= 2 and item.get("name") not in position:
                position[str(item["name"])] = (float(measured[0]), float(measured[1]))
    for tag in _list(_dict(tags).get("tags")):
        centre = _dict(tag).get("center_xyz")
        name = f"TAG{_dict(tag).get('id')}"
        if isinstance(centre, list) and len(centre) >= 2 and name not in position:
            position[name] = (float(centre[0]), float(centre[1]))
    known = [(p, position[p["id"]]) for p in points if p["id"] in position]
    if not known:
        return []
    cx = sum(xy[0] for _, xy in known) / len(known)
    cy = sum(xy[1] for _, xy in known) / len(known)
    return [{"name": p["id"], "role": p["role"], "x_m": xy[0] - cx, "y_m": xy[1] - cy,
             "err_x_m": p["error_x_m"], "err_y_m": p["error_y_m"]} for p, xy in known]


def _stage_seconds(run_metrics: dict[str, Any] | None, *names: str) -> float | None:
    seconds = _dict(_dict(run_metrics).get("stage_seconds"))
    values = [_num(seconds.get(name)) for name in names if _num(seconds.get(name)) is not None]
    return float(sum(values)) if values else None


def _densify(b: _Builder) -> Section:
    run_metrics = b.run_metrics()
    settings = _dict(_dict(run_metrics).get("settings"))
    processing = _dict(b.processing())
    dense = _dict(processing.get("dense"))
    points = dense.get("points")
    area = density = reason = cloud = None
    if dense:
        b.run = True
        area, density = dense.get("area_m2"), dense.get("density_pts_m2")
        reason, cloud = dense.get("density_reason"), dense.get("density_cloud")
    if points is None:
        points = _dict(run_metrics).get("dense_points")
        if points is None:
            points = metrics.ply_vertex_count(b.project_dir / "products" / "dense" / "fused.ply") or \
                metrics.ply_vertex_count(b.project_dir / "products" / "dense" / "fused.ply.gz")
        if points is None:
            fusion = _dict(settings.get("surface_fusion"))
            points = fusion.get("original_points") or _dict(settings.get("point_cleaning")).get("original_points")
        b.derived = points is not None or b.derived
    if area is None and reason is None:
        source = writer.cloud_for_density(b.project_dir, run_metrics)
        if source is not None:
            try:
                from samcloud.report.config import load_protocol_config

                result = metrics.dense_density(source, writer.is_metric(b.project_dir),
                                               float(load_protocol_config()["density"]["cell_m"]))
                area, reason, cloud = result["area_m2"], result["reason"], source.name
                density = (points / area) if points and area else result["density_pts_m2"]
                b.derived = True
            except Exception as error:  # noqa: BLE001
                b.warnings.append(f"densify: density not computed ({error})")
    patch = settings.get("dense_patch_match")
    b.has = points is not None or bool(settings.get("dense_max_image_size"))
    b.data = {
        "settings": {"max_image_size": settings.get("dense_max_image_size"),
                     "num_iterations": settings.get("dense_num_iterations"), "patch_match": patch,
                     "size_reason": settings.get("dense_size_reason"),
                     "excluded_images": _list(settings.get("dense_stereo_excluded_images"))},
        "points": points, "density_pts_m2": density, "area_m2": area, "density_reason": reason,
        "density_cloud": cloud, "seconds": _stage_seconds(run_metrics, "Dense stereo", "Dense fusion"),
    }
    if settings:
        b.derived = b.derived or not b.run
    return b.finish()


def _refine(b: _Builder) -> Section:
    summary = b.json("reports/geometry_summary.json")
    if summary is None:
        return b.finish("error") if b.corrupt else _no_geometry(b)
    b.derived = True
    skipped = summary.get("skipped")
    b.has = True
    b.data = {"executed": not skipped, "skipped_reason": str(skipped) if skipped else None,
              "metrics": None if skipped else {
                  key: _num(summary.get(key)) for key in
                  ("unknown_before", "unknown_after", "set_to_unknown", "smoothed_points", "runtime_seconds")}}
    return b.finish()


def _no_geometry(b: _Builder) -> Section:
    settings = _dict(_dict(b.run_metrics()).get("settings"))
    b.data = {"executed": False, "skipped_reason": "no geometry_summary.json" if settings.get("geometry_refinement")
              else "geometry refinement not enabled", "metrics": None}
    return b.finish("not_run")


def _class_names_from_summary(counts: dict[str, Any]) -> list[str]:
    return list(counts)


def _classification(b: _Builder) -> Section:
    summary = b.json("reports/classification_summary.json")
    run_metrics = b.run_metrics()
    settings = _dict(_dict(run_metrics).get("settings"))
    processing = _dict(b.processing())
    if not summary:
        return b.finish()
    b.derived = True
    counts = _dict(summary.get("class_counts"))
    unknown = int(_num(summary.get("unknown_points")) or 0)
    total = _num(summary.get("point_count")) or (sum(int(_num(v) or 0) for v in counts.values()) + unknown)
    instances = _dict(_dict(summary.get("instances")).get("per_class"))
    classes = [{"name": name, "points": int(_num(value) or 0),
                "share_pct": (int(_num(value) or 0) / total * 100.0) if total else None,
                "instances": instances.get(name)}
               for name, value in sorted(counts.items(), key=lambda item: -(_num(item[1]) or 0))]
    histogram = _dict(_dict(processing.get("classification")).get("confidence_histogram")) or None
    if histogram:
        b.run = True
    else:
        path = writer.scores_path(b.project_dir)
        if path is not None:
            try:
                histogram = metrics.confidence_histogram(
                    path, class_names=writer.classification_class_names(settings.get("class_set")),
                    placeholder_values=writer.placeholder_confidences(b.project_dir))
            except Exception as error:  # noqa: BLE001
                b.warnings.append(f"classification: confidence histogram not computed ({error})")
    percentiles = None
    if histogram:
        percentiles = {name: {k: v for k, v in item.items() if k in ("p10", "p50", "p90", "points")}
                       for name, item in _dict(histogram.get("per_class")).items()}
    b.has = True
    b.data = {
        "model": summary.get("mode"), "engine": summary.get("engine"),
        "thresholds": {"background": _num(summary.get("background_threshold")),
                       "query": _num(summary.get("query_threshold"))},
        "classes": classes, "unknown_points": unknown, "total_points": total,
        "confidence_histogram": histogram, "per_class_percentiles": percentiles,
    }
    return b.finish()


def _refinement(b: _Builder) -> Section:
    processing = _dict(b.processing())
    run_metrics = b.run_metrics()
    stages = _list(processing.get("refinement_stages"))
    if stages:
        b.run = True
    elif run_metrics:
        warnings = metrics.log_warnings(b.project_dir / "reports" / "run_summary.txt")
        stages = metrics.refinement_stages(run_metrics, warnings)
        b.derived = True
    if not stages:
        return b.finish("not_run" if not b.corrupt else "error")
    fusion = _dict(_dict(_dict(run_metrics).get("settings")).get("surface_fusion"))
    final = fusion.get("final_points") or _dict(run_metrics).get("classified_points")
    b.has = True
    b.data = {"stages": stages, "final_points": final}
    status = "ok" if any(not _dict(s).get("skipped") for s in stages) else "not_run"
    return b.finish(status)


_PRODUCT_PATTERNS = (
    ("las", ("products/classified/*.las", "products/classified/*.laz", "products/environment/*.laz",
             "products/environment/*.las")),
    ("ortho", ("products/environment/ortho_rgb.tif", "products/ortho/**/*ortho*.tif")),
    ("dsm", ("products/environment/dsm.tif",)), ("dtm", ("products/environment/dtm.tif",)),
    ("dxf", ("products/environment/*.dxf", "products/dxf/export/*.dxf")),
    ("ifc", ("products/environment/*.ifc", "products/bim/**/*.ifc")),
    ("glb", ("products/environment/*.glb", "products/bim/**/*.glb")),
)


def _products(b: _Builder) -> Section:
    settings = _dict(_dict(b.run_metrics()).get("settings"))
    crs = settings.get("export_crs")
    files = []
    seen: set[Path] = set()
    for kind, patterns in _PRODUCT_PATTERNS:
        for pattern in patterns:
            for path in sorted(b.project_dir.glob(pattern)):
                if path.is_file() and path not in seen:
                    seen.add(path)
                    use_crs = None if kind in ("dxf", "ifc", "glb") else crs
                    files.append({"kind": kind, "path": _rel(b.project_dir, path),
                                  "size_bytes": path.stat().st_size,
                                  "crs": ("local" if str(use_crs).upper() == "LOCAL" else use_crs) if use_crs else None})
    potree = b.project_dir / "products" / "viewer" / "potree"
    if potree.is_dir():
        size = sum(item.stat().st_size for item in potree.rglob("*") if item.is_file())
        files.append({"kind": "potree", "path": _rel(b.project_dir, potree), "size_bytes": size, "crs": None})
    qa = b.json("products/environment/qa_cad_report.json", list)
    dxf_qa = None
    if qa:
        rules = [{"rule": r.get("rule"), "ok": r.get("ok"), "value": r.get("value"), "threshold": r.get("threshold")}
                 for r in map(_dict, qa)]
        dxf_qa = {"passed": sum(1 for r in rules if r["ok"] is True), "failed": sum(1 for r in rules if r["ok"] is False),
                  "rules": rules}
    else:
        last = b.json("products/dxf/export/last_export.json")
        if last and last.get("audit") is not None:
            ok = last.get("audit") == "passed"
            dxf_qa = {"passed": 1 if ok else 0, "failed": 0 if ok else 1,
                      "rules": [{"rule": "ezdxf_audit", "ok": ok, "value": last.get("audit"), "threshold": "no errors"}]}
    b.derived = bool(files or dxf_qa)
    b.has = bool(files or dxf_qa)
    b.data = {"files": files, "dxf_qa": dxf_qa}
    return b.finish()


def _runtimes(b: _Builder) -> Section:
    processing = _dict(b.processing())
    run_metrics = b.run_metrics()
    steps = [{"name": name, "seconds": _num(value)} for name, value in
             _dict(_dict(run_metrics).get("stage_seconds")).items()]
    dxf = b.json("reports/dxf_run_metrics.json")
    for name, value in _dict(_dict(dxf).get("step_seconds")).items():
        steps.append({"name": name, "seconds": _num(value)})
    environment = b.json("products/environment/environment_report.json")
    for name, value in _dict(_dict(environment).get("steps")).items():
        steps.append({"name": f"environment: {name}", "seconds": _num(_dict(value).get("seconds"))})
    hardware = _dict(processing.get("hardware")) or None
    if hardware:
        b.run = True
    else:
        dataset = _dict(b.json("reports/dataset_report.json"))
        hardware = _dict(dataset.get("hardware")) or None
    b.derived = bool(steps)
    b.has = bool(steps) or bool(hardware)
    b.data = {"steps": steps, "total_seconds": _num(_dict(run_metrics).get("total_seconds")), "hardware": hardware}
    return b.finish()


def _appendix(b: _Builder, alignment: Section) -> Section:
    processing = _dict(b.processing())
    images = [{"name": i.get("name"), "camera_id": i.get("camera_id"), "reprojection_rms_px": i.get("reprojection_rms_px")}
              for i in map(_dict, _list(alignment.data.get("images")))]
    warnings = _list(processing.get("warnings"))
    if warnings:
        b.run = True
    else:
        run_id = _dict(b.run_metrics()).get("run_id")
        logs = [b.project_dir / "reports" / f"pipeline_{run_id}.log"] if run_id else []
        logs.append(b.project_dir / "reports" / "pipeline.log")
        for log in logs:
            if log.is_file():
                warnings = metrics.log_warnings(log)
                b.derived = True
                break
        summary_warnings = metrics.log_warnings(b.project_dir / "reports" / "run_summary.txt")
        merged = list(dict.fromkeys([*warnings, *summary_warnings]))
        b.derived = b.derived or bool(summary_warnings)
        warnings = merged
    b.data = {"cameras_per_image": images, "warnings": [str(w) for w in warnings]}
    b.has = bool(images) or bool(warnings) or b.run or b.derived
    if alignment.source:
        b.derived = b.derived or alignment.source in ("derived", "mixed")
        b.run = b.run or alignment.source in ("run", "mixed")
    return b.finish("ok" if b.has else "no_data")


def _summary(b: _Builder, sections: dict[str, Section]) -> Section:
    run_metrics = _dict(b.run_metrics())
    alignment = sections["alignment"].data
    control = sections["control_points"].data
    densify = sections["densify"].data
    april = _dict(control.get("apriltag"))
    deviations = [abs(_num(t.get("deviation_pct"))) for t in map(_dict, _list(april.get("tags")))
                  if _num(t.get("deviation_pct")) is not None]
    rms_pass = _dict(control.get("rms_pass"))
    rms_control = _dict(control.get("rms_control"))
    data: dict[str, Any] = {
        "images_registered": alignment.get("registered"), "images_total": alignment.get("total"),
        "reprojection_error_px": alignment.get("reprojection_error_px"),
    }
    for prefix, block in (("rms_pass", rms_pass), ("rms_control", rms_control)):
        for axis in ("x", "y", "z", "3d"):
            data[f"{prefix}_{axis}_m"] = block.get(f"{axis}_m")
    data.update({
        "apriltag_scale_factor": april.get("scale_factor"),
        "apriltag_max_deviation_pct": max(deviations) if deviations else None,
        "dense_points": densify.get("points"),
        "filtered_points": run_metrics.get("classified_points") or sections["classification"].data.get("total_points"),
        "total_seconds": _num(run_metrics.get("total_seconds")),
    })
    sources = {sections[k].source for k in ("alignment", "control_points", "densify", "classification") if sections[k].source}
    b.run, b.derived = "run" in sources or "mixed" in sources, "derived" in sources or "mixed" in sources
    control_na = _dict(sections["control_points"].data.get("not_applicable"))
    for prefix, block in (("rms_pass", "rms_pass"), ("rms_control", "rms_control")):
        if block in control_na:
            for axis in ("x", "y", "z", "3d"):
                b.na[f"{prefix}_{axis}_m"] = control_na[block]
    if "apriltag" in control_na:
        b.na["apriltag_scale_factor"] = b.na["apriltag_max_deviation_pct"] = control_na["apriltag"]
    b.data = data
    b.has = any(value is not None for value in data.values())
    return b.finish()


# --------------------------------------------------------------------------------------- entry


def _guard(key: str, project_dir: Path, cache: dict[Path, Any], build: Callable[[_Builder], Section]) -> Section:
    builder = _Builder(project_dir, key, cache)
    started = time.perf_counter()
    try:
        return build(builder)
    except Exception as error:  # noqa: BLE001 - one broken section never stops the protocol
        LOGGER.warning("protocol section %s failed: %s", key, error, exc_info=True)
        return Section(key, "error", None, {"not_recorded": []},
                       [*builder.warnings, f"{key}: {type(error).__name__}: {error}"])
    finally:
        LOGGER.debug("section %s took %.2fs", key, time.perf_counter() - started)


def collect_protocol_data(project_dir: Any, language: str | None = None) -> ProtocolData:
    """Collect all protocol sections of a project (read-only). Never raises for bad project content."""
    project_dir = Path(project_dir)
    cache: dict[Path, Any] = {}
    config = writer.read_project_config(project_dir)
    config_error = None
    if (project_dir / "project.samcloud.yml").is_file() and not config:
        config_error = "project.samcloud.yml is unreadable or empty"
    project_block = _dict(config.get("project"))
    sections: dict[str, Section] = {}
    sections["title"] = _guard("title", project_dir, cache, lambda b: _title(b, config, config_error))
    sections["import_"] = _guard("import_", project_dir, cache, _import)
    sections["alignment"] = _guard("alignment", project_dir, cache, _alignment)
    sections["control_points"] = _guard("control_points", project_dir, cache, _control_points)
    sections["densify"] = _guard("densify", project_dir, cache, _densify)
    sections["refine"] = _guard("refine", project_dir, cache, _refine)
    sections["classification"] = _guard("classification", project_dir, cache, _classification)
    sections["refinement"] = _guard("refinement", project_dir, cache, _refinement)
    sections["products"] = _guard("products", project_dir, cache, _products)
    sections["runtimes"] = _guard("runtimes", project_dir, cache, _runtimes)
    sections["appendix"] = _guard("appendix", project_dir, cache, lambda b: _appendix(b, sections["alignment"]))
    sections["summary"] = _guard("summary", project_dir, cache, lambda b: _summary(b, sections))
    ordered = {key: sections[key] for key in SECTION_KEYS}
    return ProtocolData(
        project={"name": project_block.get("name") or project_dir.name, "path": str(project_dir),
                 "created_at": project_block.get("created_at")},
        sections=ordered, language=language, generated_at=datetime.now(timezone.utc).isoformat())


def write_protocol_data(project_dir: Any, data: ProtocolData) -> Path:
    """Write ``reports/protocol_data.json`` (the only file the collector writes)."""
    target = Path(project_dir) / "reports" / "protocol_data.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_clean(data.to_dict()), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return target
