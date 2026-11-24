"""Public entry points of the processing protocol: ``generate_protocol``, ``next_protocol_path``, ``list_protocols``."""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from samcloud.report.charts import build_all_charts
from samcloud.report.collect import collect_protocol_data, write_protocol_data
from samcloud.report.config import load_protocol_config
from samcloud.report.i18n_text import normalize_language
from samcloud.report.pdf import build_protocol_pdf

LOGGER = logging.getLogger(__name__)
REPORTS_DIRECTORY = "reports"


def next_protocol_path(project_dir: Path, now: datetime | None = None) -> Path:
    """``<project>/reports/protocol_<YYYYMMDD_HHMM>.pdf``; an existing name gets ``_2``, ``_3``, ... (never overwrites)."""
    moment = now or datetime.now()
    folder = Path(project_dir) / REPORTS_DIRECTORY
    base = f"protocol_{moment:%Y%m%d_%H%M}"
    candidate = folder / f"{base}.pdf"
    counter = 2
    while candidate.exists():
        candidate = folder / f"{base}_{counter}.pdf"
        counter += 1
    return candidate


def list_protocols(project_dir: Path) -> list[Path]:
    """Existing ``reports/protocol_*.pdf`` of a project, newest first (by modification time)."""
    folder = Path(project_dir) / REPORTS_DIRECTORY
    if not folder.is_dir():
        return []
    files = [path for path in folder.glob("protocol_*.pdf") if path.is_file()]
    return sorted(files, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def generate_protocol(project_dir: Path, *, language: str | None = None, options: dict[str, Any] | None = None,
                      progress: Callable[[float, str], None] | None = None,
                      cancel_check: Callable[[], None] | None = None, now: datetime | None = None) -> Path:
    """Collect the data, render the charts and write the PDF; returns the PDF path.

    Progress (fraction 0..1, phase): collect 0.0-0.3, charts 0.3-0.7, pdf 0.7-1.0. ``cancel_check`` is called between
    the phases and raises to cancel. ``options``: ``camera_list`` and ``class_details`` (defaults from the config).
    The chart folder is a temp directory that is left in place.
    """
    project_dir = Path(project_dir)
    code = normalize_language(language)
    sections = load_protocol_config()["sections"]
    merged = {"camera_list": bool(sections.get("camera_list", False)), "class_details": bool(sections.get("class_details", True))}
    merged.update({key: value for key, value in (options or {}).items() if value is not None})

    def report(fraction: float, phase: str) -> None:
        if progress is not None:
            try:
                progress(max(0.0, min(1.0, fraction)), phase)
            except Exception:  # noqa: BLE001 - a broken progress callback must not stop the report
                LOGGER.debug("progress callback failed", exc_info=True)

    def check() -> None:
        if cancel_check is not None:
            cancel_check()

    report(0.0, "collect")
    data = collect_protocol_data(project_dir, language=code)
    report(0.2, "collect")
    write_protocol_data(project_dir, data)
    report(0.3, "collect")
    check()
    chart_dir = Path(tempfile.mkdtemp(prefix="samcloud_protocol_charts_"))
    report(0.3, "charts")
    build_all_charts(data, chart_dir, code, progress=lambda fraction, _name: report(0.3 + 0.4 * fraction, "charts"))
    report(0.7, "charts")
    check()
    target = next_protocol_path(project_dir, now)
    report(0.7, "pdf")
    build_protocol_pdf(data, target, language=code, options=merged, chart_dir=chart_dir,
                       progress=lambda fraction, _name: report(0.7 + 0.3 * fraction, "pdf"))
    report(1.0, "pdf")
    LOGGER.info("processing protocol written: %s (charts in %s)", target, chart_dir)
    return target
