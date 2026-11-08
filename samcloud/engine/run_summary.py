"""Compact, comparable reports for SAMcloud processing runs."""

from __future__ import annotations

import argparse
import json
import re
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from samcloud.core.sparse_model import sparse_points_path


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "Not measured"
    total = max(0, round(seconds))
    hours, remaining = divmod(total, 3600)
    minutes, seconds = divmod(remaining, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _point_count_from_colmap(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as file:
            raw_count = file.read(8)
        return struct.unpack("<Q", raw_count)[0] if len(raw_count) == 8 else None
    except OSError:
        return None


def _point_count_from_ply(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as file:
            for _ in range(200):
                line = file.readline().decode("ascii", errors="replace").strip()
                if line.startswith("element vertex "):
                    return int(line.rsplit(" ", 1)[1])
                if line == "end_header":
                    return None
    except (OSError, ValueError):
        return None
    return None


def _file_size(path: Path) -> str:
    if not path.exists():
        compressed = path.with_name(path.name + ".gz")
        if compressed.exists():
            return f"Compressed ({_file_size(compressed)})"
        return "Not available"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return "Not available"


def _first_label(label: str) -> str:
    return label.split(",", 1)[0].strip()


class RunSummary:
    """Collect precise stage timings and write a human-readable run report."""

    def __init__(self, outdir: Path, settings: dict[str, Any]) -> None:
        self.outdir = outdir
        self.report_dir = outdir.parent / "reports"
        self.settings = settings
        self.started_at = datetime.now(timezone.utc)
        self.finished_at: datetime | None = None
        self.started_clock = time.perf_counter()
        self.stages: dict[str, float] = {}

    def record_stage(self, name: str, elapsed_seconds: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + elapsed_seconds

    def write(self, status: str = "Completed", note: str = "") -> Path:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        metrics = self._collect_metrics(status, note)
        metrics_text = json.dumps(metrics, indent=2, ensure_ascii=False)
        report_text = self._render(metrics)
        run_suffix = self.started_at.strftime("%Y%m%d_%H%M%SZ")
        (self.report_dir / "run_metrics.json").write_text(metrics_text, encoding="utf-8")
        (self.report_dir / f"run_metrics_{run_suffix}.json").write_text(metrics_text, encoding="utf-8")
        report_path = self.report_dir / "run_summary.txt"
        report_path.write_text(report_text, encoding="utf-8")
        (self.report_dir / f"run_summary_{run_suffix}.txt").write_text(report_text, encoding="utf-8")
        return report_path

    def _collect_metrics(self, status: str, note: str) -> dict[str, Any]:
        classified_path = Path(self.settings["classified_output"])
        classification_data = _read_json(Path(self.settings["classification_summary"]))
        dense_path = self.outdir / "dense" / "fused.ply"
        sparse_path = sparse_points_path(self.outdir / "sparse") or self.outdir / "sparse" / "0_geo" / "points3D.bin"
        dense_points = _point_count_from_ply(dense_path)
        classified_points = _point_count_from_ply(classified_path)
        class_counts = classification_data.get("class_counts", {})
        unknown_points = int(classification_data.get("unknown_points", 0))
        classified_points_from_summary = int(classification_data.get("point_count", 0))
        if classified_points is None and classified_points_from_summary:
            classified_points = classified_points_from_summary
        coverage = None
        if classified_points:
            coverage = (classified_points - unknown_points) / classified_points * 100
        return {
            "report_version": 1,
            "run_id": self.started_at.strftime("%Y%m%d_%H%M%SZ"),
            "status": status,
            "note": note,
            "project": self.outdir.parent.name,
            "started_at": self.started_at.isoformat(),
            "finished_at": (self.finished_at or datetime.now(timezone.utc)).isoformat(),
            "total_seconds": time.perf_counter() - self.started_clock,
            "stage_seconds": self.stages,
            "settings": self.settings,
            "images": self.settings.get("image_count"),
            "sparse_points": _point_count_from_colmap(sparse_path),
            "dense_points": dense_points,
            "classified_points": classified_points,
            "unknown_points": unknown_points,
            "classified_coverage_percent": coverage,
            "classification_images": classification_data.get("images_processed"),
            "class_counts": class_counts,
            "files": {
                "sparse_model": str(sparse_path),
                "dense_cloud": str(dense_path),
                "classified_cloud": str(classified_path),
                "dense_cloud_size": _file_size(dense_path),
                "classified_cloud_size": _file_size(classified_path),
            },
        }

    @staticmethod
    def _render(metrics: dict[str, Any]) -> str:
        settings = metrics["settings"]
        lines = [
            "SAMcloud processing summary",
            "=" * 28,
            f"Project: {metrics['project']}",
            f"Run ID: {metrics['run_id']}",
            f"Status: {metrics['status']}",
            f"Started: {metrics['started_at']}",
            f"Finished: {metrics['finished_at']}",
            f"Total duration: {_format_duration(metrics['total_seconds'])}",
        ]
        if metrics["note"]:
            lines.append(f"Note: {metrics['note']}")
        lines.extend([
            "",
            "Input and settings",
            "-" * 18,
            f"Images: {_number(metrics['images'])}",
            f"Capture type: {settings.get('capture_type', 'Not recorded')}",
            f"GPS mode: {settings.get('gps_mode', 'Not recorded')}",
            f"GPS maximum error: {settings.get('gps_max_error_m', 'Not recorded')} m",
            f"GPU enabled: {settings.get('gpu', 'Not recorded')}",
            f"Dense image size: {settings.get('dense_max_image_size', 'Not recorded')} px",
            f"Dense iterations: {settings.get('dense_num_iterations', 'Not recorded')}",
            f"SAM class set: {settings.get('class_set', 'Not recorded')}",
            f"SAM minimum votes: {settings.get('min_votes', 'Not recorded')}",
            f"SAM classification mode: {settings.get('classification_mode', 'Not recorded')}",
            f"SAM background threshold: {settings.get('background_threshold', 'Not recorded')}",
            f"SAM query threshold: {settings.get('query_threshold', 'Not recorded')}",
            f"SAM tile size: {settings.get('sam_tile_size', 'Not recorded')} px",
            f"SAM tile overlap: {settings.get('sam_tile_min_overlap', 'Not recorded')}",
            f"Geometric refinement: {settings.get('geometry_refinement', 'Not recorded')}",
            f"GPS georeference residual: {_residual_summary(settings.get('georeference_residual_m'))}",
            "",
            "Stage durations",
            "-" * 15,
        ])
        if metrics["stage_seconds"]:
            lines.extend(
                f"{name}: {_format_duration(seconds)}" for name, seconds in metrics["stage_seconds"].items()
            )
        else:
            lines.append("No individual stage timings are available for this older run.")
        lines.extend([
            "",
            "Reconstruction results",
            "-" * 22,
            f"Sparse points: {_number(metrics['sparse_points'])}",
            f"Dense points: {_number(metrics['dense_points'])}",
            f"Classified points: {_number(metrics['classified_points'])}",
            f"Classified coverage: {_percent(metrics['classified_coverage_percent'])}",
            f"Unknown/noise points: {_number(metrics['unknown_points'])}",
            f"Photos processed by SAM: {_number(metrics['classification_images'])}",
            "",
            "Output files",
            "-" * 12,
            f"Dense cloud: {metrics['files']['dense_cloud']} ({metrics['files']['dense_cloud_size']})",
            f"Classified cloud: {metrics['files']['classified_cloud']} ({metrics['files']['classified_cloud_size']})",
        ])
        class_counts = metrics["class_counts"]
        if class_counts:
            total = metrics["classified_points"] or sum(int(count) for count in class_counts.values())
            lines.extend(["", "Class distribution", "-" * 18])
            for name, count in sorted(class_counts.items(), key=lambda item: int(item[1]), reverse=True):
                ratio = int(count) / total * 100 if total else 0
                lines.append(f"{name}: {_number(count)} ({ratio:.2f}%)")
        lines.extend([
            "",
            "Comparison guidance",
            "-" * 20,
            "Compare runs with the same input images by total duration, dense points, classified coverage, and class distribution.",
            "A higher point count is not automatically better; check the cloud visually for noise and missing surfaces.",
        ])
        return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _number(value: Any) -> str:
    return f"{int(value):,}" if value is not None else "Not available"


def _percent(value: float | None) -> str:
    return f"{value:.2f}%" if value is not None else "Not available"


def _residual_summary(value: Any) -> str:
    if not isinstance(value, dict):
        return "Not available"
    try:
        return (
            f"mean {float(value['mean']):.3f} m, p95 {float(value['p95']):.3f} m, "
            f"max {float(value['max']):.3f} m"
        )
    except (KeyError, TypeError, ValueError):
        return "Not available"


def write_legacy_summary(outdir: Path, project_file: Path) -> Path:
    """Create a baseline report for a run completed before timing capture existed."""
    project = yaml.safe_load(project_file.read_text(encoding="utf-8")) or {}
    settings = project.get("settings", {})
    acquisition = project.get("acquisition", {})
    classification = settings.get("classification", {})
    coordinates = settings.get("coordinate_systems", {})
    image_directory = Path(project.get("paths", {}).get("image_directory", ""))
    image_count = sum(
        1 for path in image_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
    ) if image_directory.is_dir() else None
    report_settings = {
        "image_count": image_count,
        "capture_type": acquisition.get("type", "Not recorded"),
        "gps_mode": "RTK" if acquisition.get("georeferencing", {}).get("rtk_enabled") else "Standard GPS",
        "gps_max_error_m": "0.1" if acquisition.get("georeferencing", {}).get("rtk_enabled") else "3.0",
        "gpu": settings.get("workflow", {}).get("use_gpu", "Not recorded"),
        "dense_max_image_size": "Not recorded",
        "dense_num_iterations": "Not recorded",
        "class_set": classification.get("class_set", "Not recorded"),
        "min_votes": classification.get("min_votes", "Not recorded"),
        "classification_mode": classification.get("mode", "Not recorded"),
        "background_threshold": classification.get("background_threshold", "Not recorded"),
        "query_threshold": classification.get("query_threshold", "Not recorded"),
        "sam_tile_size": classification.get("tile_size", "Not recorded"),
        "sam_tile_min_overlap": classification.get("tile_min_overlap", "Not recorded"),
        "geometry_refinement": classification.get("geometry_refinement", "Not recorded"),
        "source_crs": coordinates.get("working_crs", "Not recorded"),
        "export_crs": coordinates.get("export_crs", "Not recorded"),
        "classified_output": str(outdir / "classified" / "classified.ply"),
        "classification_summary": str(outdir.parent / "reports" / "classification_summary.json"),
    }
    summary = RunSummary(outdir, report_settings)
    processing = project.get("processing", {})
    started_at = _parse_iso(processing.get("started_at"))
    completed_at = _parse_iso(processing.get("completed_at"))
    if started_at and completed_at:
        summary.started_at = started_at
        summary.finished_at = completed_at
        summary.started_clock = time.perf_counter() - (completed_at - started_at).total_seconds()
    _add_legacy_classification_data(summary, outdir.parent / "reports" / "pipeline.log")
    return summary.write(
        status="Completed (legacy baseline)",
        note="Individual stage timings were added after this run. Future runs record every stage exactly.",
    )


def _parse_iso(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _add_legacy_classification_data(summary: RunSummary, log_path: Path) -> None:
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    match = re.findall(r"images: 100%.*?\[(\d+):(\d+)<", content)
    if match:
        minutes, seconds = match[-1]
        summary.record_stage("SAM3 classification (legacy log)", int(minutes) * 60 + int(seconds))
    processed_match = re.findall(r"images: 100%.*?(\d+)/(\d+)", content)
    counts = {
        _first_label(name): int(count.replace(",", ""))
        for _, name, count in re.findall(r"^\s*(\d+)\s+(.+?)\s+([\d,]+)\s+points$", content, re.MULTILINE)
    }
    if counts:
        point_count = sum(counts.values())
        unknown = next((count for name, count in counts.items() if name.lower().startswith("unknown")), 0)
        (summary.report_dir / "classification_summary.json").parent.mkdir(parents=True, exist_ok=True)
        (summary.report_dir / "classification_summary.json").write_text(
            json.dumps({
                "point_count": point_count,
                "unknown_points": unknown,
                "images_processed": int(processed_match[-1][0]) if processed_match else None,
                "class_counts": counts,
            }, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a SAMcloud legacy run summary")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--project-file", required=True, type=Path)
    args = parser.parse_args()
    path = write_legacy_summary(args.outdir.resolve(), args.project_file.resolve())
    print(f"Written run summary: {path}")


if __name__ == "__main__":
    main()
