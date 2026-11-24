"""Whether the processing protocol can be created for a project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONTROL_REPORT = Path("reports") / "control_points_report.json"
CONTROL_CSV = Path("control_points") / "control_points.csv"


@dataclass
class ProtocolAvailability:
    enabled: bool
    without_control_points: bool
    missing: list[str] = field(default_factory=list)


def _has_control_points(csv_path: Path) -> bool:
    """True when the CSV holds at least one data row below the header."""
    try:
        lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return False
    return len([line for line in lines if line.strip(" ,;\t\r")]) > 1


def protocol_availability(project_dir: Path | str, project: dict[str, Any] | None = None) -> ProtocolAvailability:
    """Step 6 must be done; control points need their check report unless the project has none."""
    from samcloud.share.job import share_available

    directory = Path(project_dir)
    missing: list[str] = []
    without = False
    if not (directory / CONTROL_REPORT).is_file():
        if _has_control_points(directory / CONTROL_CSV):
            missing.append("control_points")
        else:
            without = True
    try:
        step6 = share_available(directory, project)
    except Exception:  # unreadable project file: treat as not finished
        step6 = False
    if not step6:
        missing.append("step6")
    return ProtocolAvailability(enabled=not missing, without_control_points=without, missing=missing)
