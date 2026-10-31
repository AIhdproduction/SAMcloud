"""CSV and observation persistence for ground control points."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("name", "x", "y", "z")

# observations.json maps a CSV control-point name to clicks in undistorted COLMAP
# images. Coordinates use image pixels with a top-left origin; pixel centres are
# at (0.5, 0.5), matching COLMAP's camera projection convention.
OBSERVATION_REQUIRED_KEYS = ("image", "u", "v")


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
    """Load and validate image clicks keyed by control-point name."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return validate_observations(data)


def save_observations(path: Path, observations: dict[str, list[dict[str, float | str]]]) -> None:
    """Store image clicks for later project reopening."""
    validated = validate_observations(observations)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated, indent=2) + "\n", encoding="utf-8")


def validate_observations(data: object) -> dict[str, list[dict[str, float | str]]]:
    """Validate the documented observations.json format and normalise values."""
    if not isinstance(data, dict):
        raise ValueError("Control-point observations must be a JSON object")
    validated: dict[str, list[dict[str, float | str]]] = {}
    for name, clicks in data.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(clicks, list):
            raise ValueError("Each control-point observation entry needs a name and a list of clicks")
        normalized: list[dict[str, float | str]] = []
        image_names: set[str] = set()
        for click in clicks:
            if not isinstance(click, dict) or set(OBSERVATION_REQUIRED_KEYS) - set(click):
                raise ValueError("Each observation requires image, u, and v")
            image = click["image"]
            if not isinstance(image, str) or not image or image in image_names:
                raise ValueError("Each control point may have one observation per image")
            try:
                u, v = float(click["u"]), float(click["v"])
            except (TypeError, ValueError) as error:
                raise ValueError("Observation pixel coordinates must be numbers") from error
            if not all(value == value and abs(value) != float("inf") for value in (u, v)):
                raise ValueError("Observation pixel coordinates must be finite")
            normalized.append({"image": image, "u": u, "v": v})
            image_names.add(image)
        validated[name] = normalized
    return validated
