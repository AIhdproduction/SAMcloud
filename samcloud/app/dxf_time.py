"""Time spent creating the DXF/IFC of a project, for the overview runtime card (pure, no UI)."""

from __future__ import annotations

import json
from pathlib import Path


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def dxf_ifc_exists(project_dir: Path | str | None) -> bool:
    """True when a DXF export of the project exists (published ``products/dxf/export/*.dxf``)."""
    if project_dir is None:
        return False
    export_dir = Path(project_dir) / "products" / "dxf" / "export"
    return export_dir.is_dir() and any(export_dir.glob("*.dxf"))


def dxf_ifc_seconds(project_dir: Path | str | None) -> float | None:
    """Seconds spent creating the DXF/IFC, or None when it was not created (or no time is recorded).

    Indoor/auto export: wall time of the completed ``reports/dxf_run_metrics.json`` job (kept apart from the
    photogrammetry run metrics, so nothing is counted twice). Drone/outdoor: sum of the ``seconds`` of all
    finished steps in ``products/environment/state.json``, counted only once the DXF is published.
    """
    if project_dir is None:
        return None
    root = Path(project_dir)
    metrics = _read_json(root / "reports" / "dxf_run_metrics.json")
    if metrics is not None and metrics.get("status") == "completed":
        seconds = _number(metrics.get("total_seconds"))
        if seconds is not None:
            return seconds
    if not dxf_ifc_exists(root):
        return None
    state = _read_json(root / "products" / "environment" / "state.json")
    steps = state.get("steps") if state else None
    if not isinstance(steps, dict):
        return None
    total = sum(
        _number(step.get("seconds")) or 0.0
        for step in steps.values()
        if isinstance(step, dict) and step.get("status") == "done"
    )
    return total if total > 0 else None


def dxf_ifc_created(project_dir: Path | str | None) -> bool:
    """True when the DXF/IFC exists or a completed export job is on record."""
    if project_dir is None:
        return False
    metrics = _read_json(Path(project_dir) / "reports" / "dxf_run_metrics.json")
    return (metrics is not None and metrics.get("status") == "completed") or dxf_ifc_exists(project_dir)
