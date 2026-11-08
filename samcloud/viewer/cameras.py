"""Camera metadata and safe thumbnails for sparse-model inspection."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
import yaml

from samcloud.core.sparse_model import resolve_sparse_model


_PAYLOAD_CACHE: dict[Path, tuple[tuple[tuple[str, int, int], ...], dict[str, object], dict[str, Path]]] = {}
_THUMBNAIL_CACHE: dict[tuple[Path, int, int, int], bytes] = {}
_CACHE_LOCK = Lock()


@dataclass(frozen=True)
class _CameraSource:
    """A displayable source image selected for one registered camera."""

    path: Path | None
    display_name: str
    display_path: str
    order: int | None = None
    is_panorama: bool = False


def camera_payload(project_directory: Path) -> dict[str, object]:
    """Return sparse camera centres in exactly the final sparse-model frame.

    Points in the preview and Potree octree are read directly from the selected
    final model, so no additional coordinate transform is appropriate here.
    """
    project_directory = Path(project_directory).resolve()
    model_directory = resolve_sparse_model(project_directory / "products" / "sparse")
    if model_directory is None:
        raise FileNotFoundError("No final sparse model is available")
    signature = _model_signature(model_directory, project_directory)
    with _CACHE_LOCK:
        cached = _PAYLOAD_CACHE.get(project_directory)
        if cached is not None and cached[0] == signature:
            return cached[1]

    payload, sources = _read_cameras(project_directory, model_directory)
    with _CACHE_LOCK:
        _PAYLOAD_CACHE[project_directory] = (signature, payload, sources)
    return payload


def camera_thumbnail(project_directory: Path, camera_id: str, size: object = 320) -> bytes:
    """Return a cached JPEG thumbnail for an ID returned by :func:`camera_payload`."""
    project_directory = Path(project_directory).resolve()
    payload = camera_payload(project_directory)
    del payload  # Ensure the source-ID map is current before looking it up.
    with _CACHE_LOCK:
        cached = _PAYLOAD_CACHE.get(project_directory)
        source = cached[2].get(str(camera_id)) if cached else None
    if source is None or not _is_allowed_source(project_directory, source):
        raise FileNotFoundError("Camera image not found")
    thumbnail_size = _thumbnail_size(size)
    stat = source.stat()
    cache_key = (source, stat.st_mtime_ns, stat.st_size, thumbnail_size)
    with _CACHE_LOCK:
        encoded = _THUMBNAIL_CACHE.get(cache_key)
    if encoded is not None:
        return encoded
    pixels = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if pixels is None:
        raise FileNotFoundError("Camera image cannot be read")
    height, width = pixels.shape[:2]
    scale = min(1.0, thumbnail_size / max(width, height))
    if scale < 1.0:
        pixels = cv2.resize(pixels, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded_array = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise OSError("Camera thumbnail could not be encoded")
    encoded = bytes(encoded_array)
    with _CACHE_LOCK:
        _THUMBNAIL_CACHE[cache_key] = encoded
    return encoded


def _read_cameras(project_directory: Path, model_directory: Path) -> tuple[dict[str, object], dict[str, Path]]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(model_directory)
    source_index = _panorama_sources(project_directory)
    registered = [reconstruction.image(image_id) for image_id in reconstruction.reg_image_ids()]
    by_frame: dict[int, list[Any]] = {}
    for image in registered:
        if image.has_frame_id():
            by_frame.setdefault(int(image.frame_id), []).append(image)
    rig_model = (model_directory / "rigs.bin").is_file() or (model_directory / "frames.bin").is_file()
    records: list[dict[str, object]] = []
    sources: dict[str, Path] = {}

    if rig_model and by_frame:
        for frame_id, images in by_frame.items():
            frame = reconstruction.frame(frame_id)
            if not frame.has_pose():
                continue
            image_sources = [_camera_source(project_directory, image.name, source_index) for image in images]
            selected_index = _representative_image(images, image_sources)
            selected = images[selected_index]
            source = image_sources[selected_index]
            centre = np.asarray(frame.rig_from_world.inverse().translation, dtype=float)
            direction = np.asarray(selected.viewing_direction(), dtype=float)
            seen, mean_error = _image_metrics(reconstruction, images)
            record = _record(
                f"frame-{frame_id}", selected.name, source, centre, direction, frame_id, seen, mean_error,
            )
            records.append(record)
            if source.path is not None:
                sources[str(record["id"])] = source.path
    else:
        for image in registered:
            source = _camera_source(project_directory, image.name, source_index)
            seen, mean_error = _image_metrics(reconstruction, [image])
            records.append(_record(
                f"image-{int(image.image_id)}", image.name, source,
                np.asarray(image.projection_center(), dtype=float), np.asarray(image.viewing_direction(), dtype=float),
                None, seen, mean_error,
            ))
            if source.path is not None:
                sources[str(records[-1]["id"])] = source.path

    has_capture_order = _has_capture_order(project_directory, records)
    records.sort(key=lambda item: _record_sort_key(item, has_capture_order))
    for index, record in enumerate(records):
        record["index"] = index
    return {
        "model": str(model_directory),
        "path_available": has_capture_order and len(records) > 1,
        "is_panorama": bool(rig_model and by_frame),
        "cameras": records,
    }, sources


def _record(
    camera_id: str, image_name: str, source: _CameraSource, centre: np.ndarray, direction: np.ndarray,
    frame_id: int | None, seen: int, mean_error: float | None,
) -> dict[str, object]:
    return {
        "id": camera_id,
        "image_name": image_name,
        "image_path": source.display_path,
        "source_name": source.display_name,
        "position": [float(value) for value in centre.reshape(3)],
        "direction": [float(value) for value in direction.reshape(3)],
        "frame_id": frame_id,
        "index": 0,
        "points3d": seen,
        "mean_reprojection_error": mean_error,
    }


def _image_metrics(reconstruction: Any, images: list[Any]) -> tuple[int, float | None]:
    errors: list[float] = []
    seen = 0
    for image in images:
        point_count = image.num_points3D
        seen += int(point_count() if callable(point_count) else point_count)
        for point2d in image.points2D:
            if not point2d.has_point3D():
                continue
            point = reconstruction.point3D(point2d.point3D_id)
            if point.has_error() and math.isfinite(float(point.error)):
                errors.append(float(point.error))
    return seen, (float(np.mean(errors)) if errors else None)


def _representative_image(images: list[Any], sources: list[_CameraSource]) -> int:
    """Prefer the cubemap front face, then use a stable image-name order."""
    ranked = sorted(
        range(len(images)),
        key=lambda index: (0 if "/front/" in f"/{images[index].name.replace('\\', '/')}/" else 1, str(images[index].name)),
    )
    return ranked[0]


def _record_sort_key(record: dict[str, object], ordered: bool) -> tuple[int, object, str]:
    if ordered:
        image_path = str(record.get("image_path") or "")
        order = _panorama_order_from_path(image_path)
        if order is not None:
            return (0, order, str(record["image_name"]))
        frame_id = record.get("frame_id")
        if isinstance(frame_id, int):
            return (1, frame_id, str(record["image_name"]))
    return (2, str(record["image_name"]).casefold(), str(record["id"]))


def _panorama_order_from_path(value: str) -> int | None:
    stem = Path(value).stem
    if stem.startswith("panorama_"):
        try:
            return int(stem.rsplit("_", 1)[1])
        except ValueError:
            return None
    if stem.startswith("frame_"):
        try:
            return int(stem.rsplit("_", 1)[1])
        except ValueError:
            return None
    return None


def _has_capture_order(project_directory: Path, records: list[dict[str, object]]) -> bool:
    if any(_panorama_order_from_path(str(record.get("image_path") or "")) is not None for record in records):
        return True
    try:
        project = yaml.safe_load((project_directory / "project.samcloud.yml").read_text(encoding="utf-8")) or {}
        return project.get("acquisition", {}).get("type") == "panorama_360"
    except (OSError, TypeError, ValueError):
        return False


def _panorama_sources(project_directory: Path) -> dict[str, _CameraSource]:
    manifest = project_directory / "inputs" / "panorama_manifest.json"
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        panoramas = value.get("panoramas", []) if isinstance(value, dict) else []
    except (OSError, ValueError, TypeError):
        return {}
    result: dict[str, _CameraSource] = {}
    for order, entry in enumerate(panoramas):
        if not isinstance(entry, dict) or not isinstance(entry.get("source_id"), str):
            continue
        source = Path(str(entry.get("source_path", "")))
        if not source.is_file() or not _is_allowed_source(project_directory, source):
            fallback = project_directory / "inputs" / "panorama_frames" / source.name
            source = fallback if fallback.is_file() else Path()
        path = source.resolve() if source and source.is_file() else None
        result[entry["source_id"]] = _CameraSource(
            path, path.name if path else entry["source_id"], str(path) if path else entry["source_id"], order, True,
        )
    return result


def _camera_source(project_directory: Path, image_name: str, panorama_sources: dict[str, _CameraSource]) -> _CameraSource:
    image_path = Path(image_name.replace("\\", "/"))
    source_id = image_path.stem
    if source_id in panorama_sources:
        return panorama_sources[source_id]
    for root in _source_roots(project_directory):
        candidate = (root / image_path).resolve()
        if candidate.is_file() and _is_allowed_source(project_directory, candidate):
            return _CameraSource(candidate, candidate.name, str(candidate))
    return _CameraSource(None, image_name, image_name)


def _source_roots(project_directory: Path) -> list[Path]:
    roots = [project_directory / "inputs" / "images", project_directory / "inputs", project_directory / "inputs" / "panorama_frames"]
    try:
        project = yaml.safe_load((project_directory / "project.samcloud.yml").read_text(encoding="utf-8")) or {}
        if not isinstance(project, dict):
            return [path.resolve() for path in roots if path.is_dir()]
        acquisition = project.get("acquisition", {})
        sources = acquisition.get("sources", {}) if isinstance(acquisition, dict) else {}
        paths = project.get("paths", {})
        if not isinstance(sources, dict):
            sources = {}
        if not isinstance(paths, dict):
            paths = {}
        for value in (
            sources.get("image_directory") or paths.get("image_directory"),
            sources.get("panorama_image_directory"),
        ):
            if isinstance(value, str) and value:
                roots.append(Path(value))
    except (OSError, TypeError, ValueError):
        pass
    return [path.resolve() for path in roots if path.is_dir()]


def _is_allowed_source(project_directory: Path, candidate: Path) -> bool:
    try:
        resolved = candidate.resolve()
    except OSError:
        return False
    for root in _source_roots(project_directory):
        try:
            resolved.relative_to(root)
            return resolved.is_file()
        except ValueError:
            continue
    return False


def _model_signature(model_directory: Path, project_directory: Path) -> tuple[tuple[str, int, int], ...]:
    paths = [model_directory / name for name in ("images.bin", "points3D.bin", "rigs.bin", "frames.bin")]
    paths.extend([project_directory / "inputs" / "panorama_manifest.json", project_directory / "project.samcloud.yml"])
    result = []
    for path in paths:
        try:
            stat = path.stat()
            result.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            result.append((str(path), -1, -1))
    return tuple(result)


def _thumbnail_size(value: object) -> int:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        numeric = 320
    return max(64, min(1024, numeric))
