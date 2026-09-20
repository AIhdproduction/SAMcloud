"""CSV and observation persistence for ground control points."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("name", "x", "y", "z")


def read_control_points(csv_path: Path) -> pd.DataFrame:
    """Load and validate a ground-control-point CSV file."""
    points = pd.read_csv(csv_path)
    missing_columns = set(REQUIRED_COLUMNS) - set(points.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Control point CSV is missing columns: {missing}")
    return points.loc[:, list(REQUIRED_COLUMNS)].copy()


def write_control_points(points: pd.DataFrame, csv_path: Path) -> None:
    """Store validated control points with a stable CSV column order."""
    missing_columns = set(REQUIRED_COLUMNS) - set(points.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Control point table is missing columns: {missing}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    points.loc[:, list(REQUIRED_COLUMNS)].to_csv(csv_path, index=False)


def load_observations(path: Path) -> dict[str, list[dict[str, float | str]]]:
    """Load image clicks keyed by control-point name."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_observations(path: Path, observations: dict[str, list[dict[str, float | str]]]) -> None:
    """Store image clicks for later project reopening."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(observations, indent=2), encoding="utf-8")
