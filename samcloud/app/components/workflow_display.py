"""Display helpers of the project workflow panel: the AprilTag status of the control-point step (display only)."""

from __future__ import annotations

import json
from pathlib import Path

from samcloud.app.i18n import t

# apriltag_report.json snapshots by path: (file signature, parsed content). Only valid reports are cached.
_REPORT_CACHE: dict[str, tuple[tuple[int, int], dict]] = {}

# Display kinds of the control-point step: "completed" highlights the step, "neutral" does not.
KIND_NEUTRAL = "neutral"
KIND_COMPLETED = "completed"


def load_apriltag_report(path: Path) -> tuple[dict | None, float | None]:
    """Parsed apriltag_report.json and its mtime; (None, None) when missing, half written or malformed.

    An unchanged file is not read again; a broken one is retried on every call until it parses.
    """
    key = str(path)
    try:
        stat = path.stat()
    except OSError:
        return None, None
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _REPORT_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1], stat.st_mtime
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(data, dict) or not isinstance(data.get("tags"), list):
        return None, None
    _REPORT_CACHE[key] = (signature, data)
    return data, stat.st_mtime


def apriltag_found_text(report: dict) -> tuple[str, str] | None:
    """(status text, kind) of a finished search that found tags; None when no tag was found."""
    tags = [tag for tag in report.get("tags", []) if isinstance(tag, dict)]
    if not tags:
        return None
    found = t("home.apriltag.found_one") if len(tags) == 1 else t("home.apriltag.found_many", count=len(tags))
    mode = report.get("mode")
    if mode == "georef":
        suffix, kind = t("home.apriltag.georeferenced"), KIND_COMPLETED
    elif mode == "scale" and report.get("scale_applied"):
        suffix, kind = t("home.apriltag.scale_set"), KIND_COMPLETED
    else:
        suffix, kind = t("home.apriltag.no_scale"), KIND_NEUTRAL
    text = f"{found} · {suffix}"
    rejected = sum(1 for tag in tags if tag.get("status") != "ok")
    if rejected:
        text += " " + t("home.apriltag.rejected", count=rejected)
    return text, kind


def apriltag_step_display(
    project_directory: Path | None,
    *,
    enabled: bool,
    alignment_running: bool,
    past_search: bool,
    run_started_at: float | None,
) -> tuple[str, str] | None:
    """Status text and kind of the control-point step in AprilTag projects without control points.

    Returns None when the step keeps its normal text ("skipped (no control points)"): AprilTags off,
    nothing found, or the search ended without a usable report. While the alignment of the current run
    is still going, a report older than the run belongs to an earlier run and is ignored.
    """
    if not enabled or project_directory is None:
        return None
    report, modified = load_apriltag_report(Path(project_directory) / "reports" / "apriltag_report.json")
    if report is not None and alignment_running and run_started_at is not None and modified is not None \
            and modified < run_started_at - 2.0:
        report = None
    if report is not None:
        return apriltag_found_text(report)
    if alignment_running and not past_search:
        return t("home.apriltag.pending"), KIND_NEUTRAL
    return None
