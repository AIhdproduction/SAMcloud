"""CSV and observation persistence for ground control points."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("name", "x", "y", "z")
OUTPUT_COLUMNS = (*REQUIRED_COLUMNS, "role")

# observations.json maps a CSV control-point name to clicks in undistorted COLMAP
# images. Coordinates use image pixels with a top-left origin; pixel centres are
# at (0.5, 0.5), matching COLMAP's camera projection convention.
OBSERVATION_REQUIRED_KEYS = ("image", "u", "v")
DETECTION_STATUSES = {"accepted", "needs_review", "disabled", "not_found"}


def read_control_points(csv_path: Path) -> pd.DataFrame:
    """Load control points, accepting CSV and common Agisoft text exports.

    The returned table always includes a normalised ``role`` column containing
    either ``pass`` (used for adjustment) or ``control`` (check point only).
    """
    text = csv_path.read_text(encoding="utf-8-sig")
    points = _read_delimited_control_points(text)
    missing_columns = set(REQUIRED_COLUMNS) - set(points.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Control point CSV is missing columns: {missing}")
    result = points.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in ("x", "y", "z"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    roles = points["role"] if "role" in points.columns else pd.Series("", index=points.index)
    result["role"] = [_normalise_role(role, name) for role, name in zip(roles, result["name"], strict=True)]
    return result.loc[:, list(OUTPUT_COLUMNS)]


def write_control_points(points: pd.DataFrame, csv_path: Path) -> None:
    """Store validated control points with a stable CSV column order."""
    missing_columns = set(REQUIRED_COLUMNS) - set(points.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Control point table is missing columns: {missing}")
    result = points.loc[:, list(REQUIRED_COLUMNS)].copy()
    roles = points["role"] if "role" in points.columns else pd.Series("", index=points.index)
    result["role"] = [_normalise_role(role, name) for role, name in zip(roles, result["name"], strict=True)]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.loc[:, list(OUTPUT_COLUMNS)].to_csv(csv_path, index=False)


def apply_roles(points: pd.DataFrame, roles: dict[str, object]) -> pd.DataFrame:
    """Return a copy with UI-selected pass/control roles applied by point name."""
    if "name" not in points.columns:
        raise ValueError("Control point table is missing columns: name")
    result = points.copy()
    result["role"] = [
        _normalise_role(roles.get(str(name), current), name)
        for name, current in zip(result["name"], result.get("role", pd.Series("", index=result.index)), strict=True)
    ]
    return result


def _read_delimited_control_points(text: str) -> pd.DataFrame:
    """Read semicolon, tab, or whitespace point lists without locale surprises."""
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if not rows:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    delimiter = _detect_delimiter(rows)
    split = (lambda line: [part.strip() for part in line.split(delimiter)]) if delimiter else (lambda line: re.split(r"\s+", line.strip()))
    values = [split(row) for row in rows]
    header = [value.strip().lower() for value in values[0]]
    has_header = set(REQUIRED_COLUMNS).issubset(header)
    if not has_header and len(values[0]) >= 4:
        has_header = not _coordinates_are_numeric(values[0][1:4], decimal_comma=delimiter != ",")
    if has_header:
        columns = header
        data = values[1:]
    else:
        # Agisoft exports may add accuracy columns. Only a recognised fifth
        # value is a role; later columns are intentionally ignored.
        include_role = len(values[0]) > 4 and _is_role(values[0][4])
        columns = list(REQUIRED_COLUMNS) + (["role"] if include_role else [])
        data = [row[:5] if include_role else row[:4] for row in values]
    points = pd.DataFrame(data, columns=columns)
    for column in ("x", "y", "z"):
        if column in points:
            points[column] = points[column].astype(str).str.replace("'", "", regex=False).str.replace(",", ".", regex=False)
    return points


def _detect_delimiter(rows: list[str]) -> str | None:
    """Choose a delimiter before interpreting decimal commas.

    A comma within a semicolon-, tab-, or whitespace-delimited number is not
    CSV punctuation. The count-based rule mirrors Agisoft's forgiving import.
    """
    sample = rows[: min(10, len(rows))]
    counts = {
        "\t": sum(line.count("\t") for line in sample),
        ";": sum(line.count(";") for line in sample),
        ",": sum(line.count(",") for line in sample),
    }
    for delimiter in ("\t", ";"):
        if counts[delimiter]:
            return delimiter
    # Four whitespace tokens are already a complete headerless point row.
    # Prefer that evidence over commas, which are then decimal separators.
    whitespace_columns = sum(max(0, len(re.findall(r"\S+", line)) - 1) for line in sample)
    if counts[","] and whitespace_columns < len(sample) * 3:
        return ","
    return None


def _coordinates_are_numeric(values: list[str], *, decimal_comma: bool) -> bool:
    try:
        for value in values:
            normalized = value.replace("'", "")
            if decimal_comma:
                normalized = normalized.replace(",", ".")
            float(normalized)
    except ValueError:
        return False
    return True


def _is_role(value: object) -> bool:
    return str(value).strip().casefold() in {"pass", "pp", "passpunkt", "control", "cp", "kontrollpunkt"}


def _normalise_role(value: object, name: object) -> str:
    """Map English and German pass/check point labels to stable API values."""
    role = "" if pd.isna(value) else str(value).strip().casefold()
    if role in {"pass", "pp", "passpunkt"}:
        return "pass"
    if role in {"control", "cp", "kontrollpunkt"}:
        return "control"
    if role:
        raise ValueError(f"Unknown control point role: {value}")
    return "control" if str(name).strip().casefold().startswith("cp") else "pass"


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


def load_detections(path: Path) -> dict[str, dict[str, object]]:
    """Load the automatic-detection store without loosening its schema."""
    if not path.exists():
        return {}
    return validate_detections(json.loads(path.read_text(encoding="utf-8")))


def save_detections(path: Path, detections: dict[str, dict[str, object]]) -> None:
    """Persist validated original-image detections for review and re-runs."""
    validated = validate_detections(detections)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated, indent=2) + "\n", encoding="utf-8")


def validate_detections(data: object) -> dict[str, dict[str, object]]:
    """Validate detections.json and normalise all numeric measurements."""
    if not isinstance(data, dict):
        raise ValueError("Control-point detections must be a JSON object")
    result: dict[str, dict[str, object]] = {}
    for name, entry in data.items():
        if not isinstance(name, str) or not name or not isinstance(entry, dict):
            raise ValueError("Each detection needs a point name and an object")
        role = _normalise_role(entry.get("role", ""), name)
        status = str(entry.get("status", "accepted"))
        if status not in DETECTION_STATUSES:
            raise ValueError(f"Unknown detection status: {status}")
        xyz = entry.get("triangulated_model_xyz")
        if xyz is not None and (not isinstance(xyz, list) or len(xyz) != 3):
            raise ValueError("triangulated_model_xyz must contain three coordinates")
        normalised: dict[str, object] = {
            "role": role,
            "enabled": bool(entry.get("enabled", status not in {"disabled", "not_found"})),
            "status": status,
            "triangulated_model_xyz": None if xyz is None else [_finite_number(value, "triangulated_model_xyz") for value in xyz],
            "observations": [],
        }
        normalised["disabled_by_reviewer"] = bool(entry.get("disabled_by_reviewer", False))
        if entry.get("marker_rms_px") is not None:
            marker_rms = float(entry["marker_rms_px"])
            normalised["marker_rms_px"] = marker_rms if np.isfinite(marker_rms) else None
        predictions = entry.get("predictions")
        if predictions is not None:
            if not isinstance(predictions, list):
                raise ValueError("Detection predictions must be a list")
            normalised["predictions"] = []
            predicted_images: set[str] = set()
            for prediction in predictions:
                if not isinstance(prediction, dict) or set(OBSERVATION_REQUIRED_KEYS) - set(prediction):
                    raise ValueError("Each detection prediction requires image, u, and v")
                image = prediction["image"]
                if not isinstance(image, str) or not image or image in predicted_images:
                    raise ValueError("Each control point may have one prediction per image")
                normalised["predictions"].append({
                    "image": image,
                    "u": _finite_number(prediction["u"], "u"),
                    "v": _finite_number(prediction["v"], "v"),
                })
                predicted_images.add(image)
        if "reason" in entry:
            normalised["reason"] = str(entry["reason"])
        images: set[str] = set()
        observations = entry.get("observations", [])
        if not isinstance(observations, list):
            raise ValueError("Detection observations must be a list")
        for observation in observations:
            if not isinstance(observation, dict) or set(OBSERVATION_REQUIRED_KEYS) - set(observation):
                raise ValueError("Each detection observation requires image, u, and v")
            image = observation["image"]
            if not isinstance(image, str) or not image or image in images:
                raise ValueError("Each control point may have one detection per image")
            source = observation.get("source", "auto")
            if source not in {"auto", "manual"}:
                raise ValueError("Detection source must be auto or manual")
            item: dict[str, object] = {"image": image, "u": _finite_number(observation["u"], "u"), "v": _finite_number(observation["v"], "v"), "u_undist": _finite_number(observation.get("u_undist", observation["u"]), "u_undist"), "v_undist": _finite_number(observation.get("v_undist", observation["v"]), "v_undist"), "score": _finite_number(observation.get("score", 1.0), "score"), "source": source, "enabled": bool(observation.get("enabled", True)), "reprojection_px": _finite_number(observation.get("reprojection_px", 0.0), "reprojection_px")}
            if "reprojected_u" in observation:
                item["reprojected_u"] = _finite_number(observation["reprojected_u"], "reprojected_u")
            if "reprojected_v" in observation:
                item["reprojected_v"] = _finite_number(observation["reprojected_v"], "reprojected_v")
            normalised["observations"].append(item)
            images.add(image)
        result[name] = normalised
    return result


def move_detection_observation(path: Path, name: str, image: str, u: object, v: object) -> dict[str, object]:
    """Move or create one original-image observation as a manual placement."""
    detections = load_detections(path)
    entry = _detection_entry(detections, name)
    image_name = _image_name(image)
    u_value, v_value = _finite_number(u, "u"), _finite_number(v, "v")
    observations = entry["observations"]
    existing = next((item for item in observations if item["image"] == image_name), None)
    if existing is None:
        existing = {
            "image": image_name,
            "u": u_value,
            "v": v_value,
            "u_undist": u_value,
            "v_undist": v_value,
            "score": 1.0,
            "source": "manual",
            "enabled": True,
            "reprojection_px": 0.0,
        }
        observations.append(existing)
    else:
        existing.update({
            "u": u_value,
            "v": v_value,
            "u_undist": u_value,
            "v_undist": v_value,
            "source": "manual",
            "enabled": True,
        })
    save_detections(path, detections)
    return dict(existing)


def remove_detection_observation(path: Path, name: str, image: str) -> None:
    """Remove one observation while preserving every other marker projection."""
    detections = load_detections(path)
    entry = _detection_entry(detections, name)
    image_name = _image_name(image)
    before = len(entry["observations"])
    entry["observations"] = [item for item in entry["observations"] if item["image"] != image_name]
    if len(entry["observations"]) == before:
        raise ValueError("Observation does not exist")
    save_detections(path, detections)


def set_detection_observation_enabled(path: Path, name: str, image: str, enabled: object) -> bool:
    """Persist whether one image observation participates in adjustment."""
    detections = load_detections(path)
    entry = _detection_entry(detections, name)
    image_name = _image_name(image)
    observation = next((item for item in entry["observations"] if item["image"] == image_name), None)
    if observation is None:
        raise ValueError("Observation does not exist")
    observation["enabled"] = _boolean(enabled, "enabled")
    save_detections(path, detections)
    return bool(observation["enabled"])


def set_detection_enabled(path: Path, name: str, enabled: object) -> bool:
    """Persist whether a marker is used in the next adjustment."""
    detections = load_detections(path)
    entry = _detection_entry(detections, name)
    entry["enabled"] = _boolean(enabled, "enabled")
    entry["disabled_by_reviewer"] = not entry["enabled"]
    entry["status"] = "accepted" if entry["enabled"] else "disabled"
    save_detections(path, detections)
    return bool(entry["enabled"])


def set_detection_role(path: Path, name: str, role: object) -> str:
    """Persist a marker's pass/control role in the detection store."""
    detections = load_detections(path)
    entry = _detection_entry(detections, name)
    entry["role"] = _normalise_role(role, name)
    save_detections(path, detections)
    return str(entry["role"])


def set_control_point_role(csv_path: Path, name: str, role: object) -> str:
    """Keep the imported reference table aligned with a reviewed marker role."""
    points = read_control_points(csv_path)
    matches = points["name"].astype(str) == name
    if not bool(matches.any()):
        raise ValueError("Control point does not exist")
    normalized = _normalise_role(role, name)
    points.loc[matches, "role"] = normalized
    write_control_points(points, csv_path)
    return normalized


def _detection_entry(detections: dict[str, dict[str, object]], name: str) -> dict[str, object]:
    if not isinstance(name, str) or not name.strip() or name not in detections:
        raise ValueError("Control point does not exist")
    return detections[name]


def _image_name(image: object) -> str:
    if not isinstance(image, str) or not image.strip():
        raise ValueError("Image name is required")
    return image


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _finite_number(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not pd.notna(number) or number in {float("inf"), float("-inf")}:
        raise ValueError(f"{field} must be a finite number")
    return number
