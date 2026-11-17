"""AprilTag detection, triangulation and scale estimation for indoor models.

Pixel coordinates follow COLMAP's convention: the centre of the top-left pixel
is (0.5, 0.5).  With CORNER_REFINE_APRILTAG OpenCV already reports corners on
pixel edges (verified on an axis-aligned marker: a square spanning pixels
150..549 is returned as 150..550), which is exactly that convention, so no
+0.5 shift is added here.  (CORNER_REFINE_SUBPIX would return centre-based
values and need the shift.)
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from samcloud.control_points.auto_detect import project_point, triangulate_observations
from samcloud.control_points.repository import apriltag_point_name

APRILTAG_FAMILY = "tag36h11"
APRILTAG_SIZE_M = 0.150
APRILTAG_ID_MAX = 586
DEFAULT_OUTLIER_PCT = 3.0
DETECTIONS_SCHEMA_VERSION = 1

WARNING_NO_TAGS = "Keine AprilTags gefunden – Modell bleibt unskaliert"
WARNING_NO_CONTROL = "Nur {n} AprilTag(s) gefunden – keine Kontrolle möglich"
WARNING_GEOREF_UNAVAILABLE = "Weniger als 3 eingemessene AprilTags – nur Maßstab"
MIN_COLLINEARITY_RATIO = 0.1
MIN_GEOREF_TAGS = 3


class TagGeorefUnavailable(Exception):
    """Raised when surveyed AprilTags cannot georeference the model (too few or collinear)."""


@dataclass(frozen=True)
class TagObservation:
    image: str
    tag_id: int
    corners: tuple[tuple[float, float], ...]
    center: tuple[float, float]


@dataclass(frozen=True)
class TagGeometry:
    tag_id: int
    corners_xyz: np.ndarray
    center_xyz: np.ndarray
    images: tuple[str, ...]
    reprojection_rms_px: float


@dataclass(frozen=True)
class ScaleResult:
    scale_factor: float | None
    per_tag: dict[int, dict]
    warnings: list[str]
    inliers: tuple[int, ...]
    outliers: tuple[int, ...]


def make_detector() -> Any:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def _diagonal_intersection(corners: np.ndarray) -> tuple[float, float]:
    p0, p1, p2, p3 = (np.asarray(c, dtype=float) for c in corners)
    d1, d2 = p2 - p0, p3 - p1
    matrix = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    if abs(np.linalg.det(matrix)) < 1e-12:
        centre = np.asarray(corners, dtype=float).mean(axis=0)
    else:
        t, _ = np.linalg.solve(matrix, p1 - p0)
        centre = p0 + t * d1
    return float(centre[0]), float(centre[1])


def detect_apriltags_in_image(image: np.ndarray, image_name: str) -> list[TagObservation]:
    gray = image
    if gray.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if gray.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(gray, code)
    gray = np.ascontiguousarray(gray)
    corners_list, ids, _ = make_detector().detectMarkers(gray)
    results: list[TagObservation] = []
    if ids is None:
        return results
    for corners, tag_id in zip(corners_list, ids.ravel(), strict=True):
        tag_id = int(tag_id)
        if tag_id > APRILTAG_ID_MAX:
            continue
        pts = np.asarray(corners, dtype=float).reshape(4, 2)
        results.append(TagObservation(
            image=image_name,
            tag_id=tag_id,
            corners=tuple((float(x), float(y)) for x, y in pts),
            center=_diagonal_intersection(pts),
        ))
    return results


def detect_apriltags(image_paths: dict[str, Path], *, workers: int = 4) -> dict[str, list[TagObservation]]:
    def work(item: tuple[str, Path]) -> tuple[str, list[TagObservation] | None]:
        name, path = item
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return name, None
        return name, detect_apriltags_in_image(image, name)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        outcomes = list(pool.map(work, image_paths.items()))
    return {name: found for name, found in outcomes if found is not None}


def save_tag_detections(path: Path, detections: dict[str, list[TagObservation]]) -> None:
    payload = {
        "schema_version": DETECTIONS_SCHEMA_VERSION,
        "family": APRILTAG_FAMILY,
        "images": {
            name: [
                {"tag_id": o.tag_id, "corners": [list(c) for c in o.corners], "center": list(o.center)}
                for o in observations
            ]
            for name, observations in detections.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_tag_detections(path: Path) -> dict[str, list[TagObservation]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != DETECTIONS_SCHEMA_VERSION:
        raise ValueError("Unsupported AprilTag detection file")
    result: dict[str, list[TagObservation]] = {}
    for name, items in data.get("images", {}).items():
        result[name] = [
            TagObservation(
                image=name,
                tag_id=int(item["tag_id"]),
                corners=tuple((float(c[0]), float(c[1])) for c in item["corners"]),
                center=(float(item["center"][0]), float(item["center"][1])),
            )
            for item in items
        ]
    return result


def triangulate_tags(
    reconstruction: Any, detections: dict[str, list[TagObservation]], *,
    inlier_px: float = 3.0, min_images: int = 2,
) -> dict[int, TagGeometry]:
    by_tag: dict[int, list[TagObservation]] = {}
    for observations in detections.values():
        for observation in observations:
            by_tag.setdefault(observation.tag_id, []).append(observation)
    geometries: dict[int, TagGeometry] = {}
    for tag_id, observations in sorted(by_tag.items()):
        if len({o.image for o in observations}) < min_images:
            continue
        try:
            corners = []
            for index in range(4):
                rows = [{"image": o.image, "u": o.corners[index][0], "v": o.corners[index][1], "enabled": True} for o in observations]
                xyz, _ = triangulate_observations(reconstruction, rows, inlier_px=inlier_px)
                corners.append(xyz)
            rows = [{"image": o.image, "u": o.center[0], "v": o.center[1], "enabled": True} for o in observations]
            center, _ = triangulate_observations(reconstruction, rows, inlier_px=inlier_px)
        except ValueError:
            continue
        corners_xyz = np.asarray(corners, dtype=float)
        squared: list[float] = []
        used: set[str] = set()
        for o in observations:
            image = reconstruction.find_image_with_name(o.image)
            if image is None:
                continue
            camera = reconstruction.camera(image.camera_id)
            for index in range(4):
                projected = project_point(image, camera, corners_xyz[index], margin_px=float("inf"))
                if projected is None:
                    continue
                squared.append((projected[0] - o.corners[index][0]) ** 2 + (projected[1] - o.corners[index][1]) ** 2)
                used.add(o.image)
        rms = float(np.sqrt(np.mean(squared))) if squared else float("nan")
        images = tuple(sorted(used or {o.image for o in observations}))
        geometries[tag_id] = TagGeometry(tag_id, corners_xyz, np.asarray(center, dtype=float), images, rms)
    return geometries


def tag_edge_lengths(corners_xyz: np.ndarray) -> np.ndarray:
    """Four edges followed by the two diagonals."""
    c = np.asarray(corners_xyz, dtype=float)
    pairs = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3))
    return np.array([np.linalg.norm(c[a] - c[b]) for a, b in pairs])


def expected_edge_lengths() -> np.ndarray:
    return np.array([APRILTAG_SIZE_M] * 4 + [APRILTAG_SIZE_M * np.sqrt(2.0)] * 2)


def tag_scale(geometries: dict[int, TagGeometry], *, outlier_pct: float = DEFAULT_OUTLIER_PCT) -> ScaleResult:
    if not geometries:
        return ScaleResult(None, {}, [WARNING_NO_TAGS], (), ())
    factors: dict[int, float] = {}
    before: dict[int, float] = {}
    for tag_id, geometry in geometries.items():
        lengths = tag_edge_lengths(geometry.corners_xyz)
        factors[tag_id] = float(np.median(expected_edge_lengths() / lengths))
        before[tag_id] = float(lengths[:4].mean() * 1000.0)
    warnings: list[str] = []
    if len(factors) < 3:
        warnings.append(WARNING_NO_CONTROL.format(n=len(factors)))
        inliers, outliers = tuple(sorted(factors)), ()
        scale = float(np.mean(list(factors.values())))
    else:
        median = float(np.median(list(factors.values())))
        inliers = tuple(t for t in sorted(factors) if abs(factors[t] / median - 1.0) * 100.0 <= outlier_pct)
        outliers = tuple(t for t in sorted(factors) if t not in inliers)
        scale = float(np.mean([factors[t] for t in inliers])) if inliers else median
        if outliers:
            warnings.append("AprilTag-Ausreißer ignoriert: " + ", ".join(str(t) for t in outliers))
    per_tag = {
        t: {
            "factor": factors[t],
            "size_mm_before": before[t],
            "deviation_pct": (factors[t] / scale - 1.0) * 100.0,
            "status": "outlier" if t in outliers else "ok",
        }
        for t in sorted(factors)
    }
    return ScaleResult(scale, per_tag, warnings, inliers, outliers)


def apply_scale(model_dir: Path, out_dir: Path, scale: float) -> Path:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(Path(model_dir))
    reconstruction.transform(pycolmap.Sim3d(float(scale), pycolmap.Rotation3d(np.eye(3)), np.zeros(3)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(out_dir)
    return out_dir


def build_tag_report(
    result: ScaleResult, geometries: dict[int, TagGeometry], *, scale_applied: float | None, mode: str,
) -> dict:
    factor = 1.0 if scale_applied is None else float(scale_applied)
    tags = []
    for tag_id in sorted(geometries):
        geometry = geometries[tag_id]
        info = result.per_tag.get(tag_id, {})
        before = float(info.get("size_mm_before", tag_edge_lengths(geometry.corners_xyz)[:4].mean() * 1000.0))
        tags.append({
            "id": int(tag_id),
            "images": len(geometry.images),
            "size_mm_before": before,
            "size_mm_after": before * factor,
            "deviation_pct": info.get("deviation_pct"),
            "status": info.get("status", "ok"),
            "center_xyz": [float(v) for v in geometry.center_xyz * factor],
            "corners_xyz": [[float(v) for v in row] for row in geometry.corners_xyz * factor],
            "reprojection_rms_px": float(geometry.reprojection_rms_px),
        })
    return {
        "mode": mode,
        "scale_factor": None if result.scale_factor is None else float(result.scale_factor),
        "scale_applied": None if scale_applied is None else float(scale_applied),
        "warnings": list(result.warnings),
        "tags": tags,
    }


def save_tag_report(path: Path, report: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


GRAVITY_ALIGNED_FRAME = "gravity_aligned"


def transform_tag_report(report: dict, rotation: Any, translation: Any) -> dict:
    """Move every tag centre/corner of a report with p' = R p + t and mark it as gravity aligned.

    Idempotent: a report that already carries frame "gravity_aligned" is returned unchanged.
    """
    if report.get("frame") == GRAVITY_ALIGNED_FRAME:
        return report
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    shift = np.asarray(translation, dtype=float).reshape(3)
    for tag in report.get("tags", []):
        if not isinstance(tag, dict):
            continue
        center = tag.get("center_xyz")
        if center is not None:
            tag["center_xyz"] = [float(v) for v in matrix @ np.asarray(center, dtype=float).reshape(3) + shift]
        corners = tag.get("corners_xyz")
        if corners is not None:
            moved = np.asarray(corners, dtype=float).reshape(-1, 3) @ matrix.T + shift
            tag["corners_xyz"] = [[float(v) for v in row] for row in moved]
    report["frame"] = GRAVITY_ALIGNED_FRAME
    report["gravity_transform"] = {"rotation": matrix.tolist(), "translation": shift.tolist()}
    return report


def tags_are_non_collinear(xyz: np.ndarray, *, min_ratio: float = MIN_COLLINEARITY_RATIO) -> bool:
    """True when the second/first singular value of the centred coordinates reaches min_ratio."""
    points = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if len(points) < MIN_GEOREF_TAGS:
        return False
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return bool(singular[0] > 0.0 and singular[1] / singular[0] >= min_ratio)


def tag_detection_entries(
    table: Any, geometries: dict[int, TagGeometry], detections: dict[str, list[TagObservation]],
) -> dict[str, dict]:
    """Control-point detections (detections.json schema) for every surveyed, triangulated tag."""
    entries: dict[str, dict] = {}
    for row in table.itertuples(index=False):
        tag_id = int(row.id)
        geometry = geometries.get(tag_id)
        if geometry is None:
            continue
        used = set(geometry.images)
        observations = []
        seen: set[str] = set()
        for observations_in_image in detections.values():
            for observation in observations_in_image:
                if observation.tag_id != tag_id or observation.image not in used or observation.image in seen:
                    continue
                seen.add(observation.image)
                observations.append({
                    "image": observation.image, "u": float(observation.center[0]), "v": float(observation.center[1]),
                    "score": 1.0, "source": "apriltag", "enabled": True,
                })
        entries[apriltag_point_name(tag_id)] = {
            "role": str(row.role) if str(row.role) in {"pass", "control"} else "pass",
            "enabled": True, "status": "accepted",
            "triangulated_model_xyz": [float(v) for v in geometry.center_xyz],
            "observations": observations, "predictions": [],
        }
    return entries
