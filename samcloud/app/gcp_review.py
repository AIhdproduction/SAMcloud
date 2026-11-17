"""Validated data helpers for the sparse-cloud control-point review."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from samcloud.control_points.apriltags import load_tag_detections
from samcloud.control_points.repository import load_detections


MIN_CROP_SIZE = 16
MAX_CROP_SIZE = 512
DEFAULT_CROP_SIZE = 96


def displayed_pixel_to_original(
    display_x: float, display_y: float, *, origin_x: float, origin_y: float, zoom: float,
) -> tuple[float, float]:
    """Map a displayed pixel-centre click to original top-left pixel coordinates."""
    if not math.isfinite(zoom) or zoom <= 0:
        raise ValueError("Zoom must be a positive finite number")
    return ((float(display_x) - float(origin_x)) / zoom, (float(display_y) - float(origin_y)) / zoom)


def _json_safe(value: Any) -> Any:
    """Replace NaN/inf (not valid JSON) with None, recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def marker_payload(project_directory: Path) -> dict[str, object]:
    """Return every persisted marker in the final sparse-model coordinate frame."""
    control_directory = project_directory / "control_points"
    detections = load_detections(control_directory / "detections.json")
    report = _load_report(project_directory / "reports" / "control_points_report.json")
    report_points = {str(point.get("name")): point for point in report.get("points", []) if isinstance(point, dict)}
    tag_report = _load_report(project_directory / "reports" / "apriltag_report.json")
    tag_entries = {}
    for tag in tag_report.get("tags", []):
        if isinstance(tag, dict):
            try:
                tag_entries[int(tag.get("id"))] = tag
            except (TypeError, ValueError):
                continue
    markers = []
    for name, entry in detections.items():
        report_point = report_points.get(name, {})
        enabled = bool(entry.get("enabled", report_point.get("enabled", True)))
        status = str(entry.get("status", report_point.get("status", "needs_review")))
        xyz = entry.get("triangulated_model_xyz")
        photos: list[dict[str, Any]] = []
        observed_images: set[str] = set()
        for obs in entry.get("observations", []):
            if not isinstance(obs, dict):
                continue
            image_name = obs.get("image")
            if image_name is not None:
                observed_images.add(str(image_name))
            photos.append({
                "image": image_name,
                "u": obs.get("u"),
                "v": obs.get("v"),
                "kind": "manual" if obs.get("source") == "manual" else "auto",
                "enabled": bool(obs.get("enabled", True)),
                "reprojected_u": obs.get("reprojected_u"),
                "reprojected_v": obs.get("reprojected_v"),
                "reprojection_px": obs.get("reprojection_px"),
                "score": obs.get("score"),
            })
        seen_predicted_images: set[str] = set()
        for pred in entry.get("predictions") or []:
            if not isinstance(pred, dict):
                continue
            image_name = pred.get("image")
            if image_name is None:
                continue
            image_str = str(image_name)
            if image_str in observed_images or image_str in seen_predicted_images:
                continue
            seen_predicted_images.add(image_str)
            pred_u = pred.get("u")
            pred_v = pred.get("v")
            photos.append({
                "image": image_name,
                "u": pred_u,
                "v": pred_v,
                "kind": "predicted",
                "enabled": False,
                "reprojected_u": pred_u,
                "reprojected_v": pred_v,
                "reprojection_px": None,
                "score": None,
            })
        marker_entry = {
            "name": name,
            "role": str(entry.get("role", report_point.get("role", "pass"))),
            "enabled": enabled,
            "state": status,
            "state_color": _state_color(enabled, status, str(entry.get("role", "pass"))),
            "coordinates": xyz if isinstance(xyz, list) and len(xyz) == 3 else None,
            "errors": {
                "x_m": _finite_or_none(report_point.get("error_x_m")),
                "y_m": _finite_or_none(report_point.get("error_y_m")),
                "z_m": _finite_or_none(report_point.get("error_z_m")),
                "total_m": _finite_or_none(report_point.get("error_3d_m", report_point.get("residual_m"))),
                "reprojection_px": _finite_or_none(report_point.get("reprojection_rms_px")),
            },
            "projections": int(report_point.get("projections", len(entry.get("observations", [])))),
            "observations": entry.get("observations", []),
            "photos": photos,
        }
        tag_match = re.fullmatch(r"TAG(\d+)", str(name))
        if tag_match:
            tag_id = int(tag_match.group(1))
            tag_info = tag_entries.get(tag_id, {})
            marker_entry.update(kind="apriltag", tag_id=tag_id)
            if tag_info.get("size_mm_after") is not None:
                marker_entry["tag_size_mm"] = tag_info.get("size_mm_after")
            if tag_info.get("deviation_pct") is not None:
                marker_entry["tag_deviation_pct"] = tag_info.get("deviation_pct")
        markers.append(marker_entry)
    present = {str(marker["name"]) for marker in markers}
    markers.extend(marker for marker in apriltag_marker_payload(project_directory) if marker["name"] not in present)
    return _json_safe({"gate": report.get("gate", {"passed": False, "reasons": []}), "totals": report.get("totals", {}), "markers": markers})


def apriltag_marker_payload(project_directory: Path) -> list[dict[str, Any]]:
    """One marker per usable AprilTag of reports/apriltag_report.json (empty without a report).

    Coordinate frame: centre and corners are stored in the frame of the final model. For scale-only
    panorama runs the pipeline moves the report into the gravity-aligned frame right after the gravity
    step (report["frame"] == "gravity_aligned", see transform_tag_report); in georeference mode the
    report is already in the final frame. The viewers place their points raw in the final frame, so the
    coordinates are served unchanged.

    Photos: every image of control_points/apriltag_detections.json that shows the tag becomes a
    read-only photo entry (source "apriltag"); a missing or corrupt detections file yields no photos.
    """
    tag_report = _load_report(Path(project_directory) / "reports" / "apriltag_report.json")
    tag_photos = _apriltag_photos(Path(project_directory))
    markers: list[dict[str, Any]] = []
    for tag in tag_report.get("tags", []):
        if not isinstance(tag, dict) or tag.get("status") == "insufficient":
            continue
        center = tag.get("center_xyz")
        try:
            tag_id = int(tag.get("id"))
        except (TypeError, ValueError):
            continue
        if not (isinstance(center, (list, tuple)) and len(center) == 3):
            continue
        state = "needs_review" if tag.get("status") == "outlier" else "apriltag"
        error_total = _finite_or_none(tag.get("error_3d_m"))
        photos = tag_photos.get(tag_id, [])
        marker: dict[str, Any] = {
            "name": f"TAG{tag_id}", "kind": "apriltag", "tag_id": tag_id, "role": "apriltag", "enabled": True,
            "state": state, "state_color": state,
            "coordinates": [float(value) for value in center], "corners": tag.get("corners_xyz"),
            "tag_size_mm": tag.get("size_mm_after"), "tag_deviation_pct": tag.get("deviation_pct"),
            "images": tag.get("images"), "status": tag.get("status"),
            "photos": [dict(photo) for photo in photos], "observations": [dict(photo) for photo in photos],
        }
        if error_total is not None:
            marker["errors"] = {"total_m": error_total}
        markers.append(marker)
    return _json_safe(markers)


def _apriltag_photos(project_directory: Path) -> dict[int, list[dict[str, Any]]]:
    """Photo entries per tag id from control_points/apriltag_detections.json (empty on any problem)."""
    path = project_directory / "control_points" / "apriltag_detections.json"
    if not path.is_file():
        return {}
    try:
        detections = load_tag_detections(path)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return {}
    by_tag: dict[int, list[dict[str, Any]]] = {}
    for image_name in sorted(detections):
        for observation in detections[image_name]:
            by_tag.setdefault(observation.tag_id, []).append({
                "image": image_name, "u": observation.center[0], "v": observation.center[1],
                "kind": "auto", "enabled": True,
                "corners": [[float(u), float(v)] for u, v in observation.corners], "source": "apriltag",
            })
    return by_tag


def has_apriltag_review_data(project_directory: Path) -> bool:
    """True when an AprilTag report or detection file exists, so the tag photos can be reviewed."""
    root = Path(project_directory)
    return (root / "reports" / "apriltag_report.json").is_file() or (root / "control_points" / "apriltag_detections.json").is_file()


def apriltag_warnings(project_directory: Path) -> list[str]:
    """Warnings of the AprilTag report (for the viewer info card)."""
    report = _load_report(Path(project_directory) / "reports" / "apriltag_report.json")
    return [str(item) for item in report.get("warnings", []) if isinstance(item, str)]


def image_crop(
    project_directory: Path, image: str, u: object, v: object, size: object, source: str,
) -> tuple[bytes, int, int, int]:
    """Encode an exact, unclamped integer-sliced crop without interpolation."""
    image_path = resolve_project_image(project_directory, image, source)
    try:
        center_u, center_v = float(u), float(v)
    except (TypeError, ValueError) as error:
        raise ValueError("Crop coordinates must be numbers") from error
    if not all(math.isfinite(value) for value in (center_u, center_v)):
        raise ValueError("Crop coordinates must be finite")
    crop_size = clamp_crop_size(size)
    pixels = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if pixels is None:
        raise FileNotFoundError("Image cannot be read")
    height, width = pixels.shape[:2]
    origin_u = math.floor(center_u - crop_size / 2.0)
    origin_v = math.floor(center_v - crop_size / 2.0)
    crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    img_v_min = max(0, origin_v)
    img_v_max = min(height, origin_v + crop_size)
    img_u_min = max(0, origin_u)
    img_u_max = min(width, origin_u + crop_size)
    if img_v_min < img_v_max and img_u_min < img_u_max:
        crop_v_min = img_v_min - origin_v
        crop_v_max = img_v_max - origin_v
        crop_u_min = img_u_min - origin_u
        crop_u_max = img_u_max - origin_u
        crop[crop_v_min:crop_v_max, crop_u_min:crop_u_max] = pixels[img_v_min:img_v_max, img_u_min:img_u_max]
    rendered = cv2.resize(crop, (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_NEAREST)
    success, encoded = cv2.imencode(".png", rendered)
    if not success:
        raise OSError("Could not encode image crop")
    return bytes(encoded), crop_size, origin_u, origin_v


def clamp_crop_size(value: object) -> int:
    """Keep thumbnail and zoom crop reads bounded for large source images."""
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        numeric = DEFAULT_CROP_SIZE
    return max(MIN_CROP_SIZE, min(MAX_CROP_SIZE, numeric))


def resolve_project_image(project_directory: Path, image: str, source: str) -> Path:
    """Resolve a project image only below the selected known image roots."""
    if not isinstance(image, str) or not image or Path(image).is_absolute():
        raise ValueError("Invalid image path")
    if source not in {"original", "undistorted"}:
        raise ValueError("Invalid image source")
    roots = _image_roots(project_directory, source)
    for root in roots:
        candidate = (root / image).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    if any(part == ".." for part in Path(image).parts):
        raise ValueError("Invalid image path")
    raise FileNotFoundError("Image not found")


def _image_roots(project_directory: Path, source: str) -> list[Path]:
    dense = project_directory / "products" / "dense" / "images"
    if source == "undistorted" and dense.is_dir():
        return [dense.resolve()]
    roots = [project_directory / "inputs" / "images", project_directory / "inputs" / "cubemap", project_directory / "inputs"]
    project_path = project_directory / "project.samcloud.yml"
    try:
        project = yaml.safe_load(project_path.read_text(encoding="utf-8")) or {}
        configured = project.get("acquisition", {}).get("sources", {}).get("image_directory") or project.get("paths", {}).get("image_directory")
        if isinstance(configured, str) and configured:
            roots.insert(0, Path(configured))
    except (OSError, ValueError, TypeError):
        pass
    return [root.resolve() for root in roots if root.exists() and root.is_dir()]


def _load_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _state_color(enabled: bool, status: str, role: str) -> str:
    if not enabled:
        return "disabled"
    if status in {"needs_review", "not_found", "disabled"}:
        return "needs_review"
    return "pass" if role == "pass" else "control"


def _finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None
