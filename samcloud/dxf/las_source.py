"""Chunked access to classified point sources (LAS, or the project's classified PLY) and their coordinate frame.

A classified LAS is read as is. A project's products/classified/classified.ply is read directly with the PLY
layout of the LAS export: its scalar class ids are mapped to Cyclone codes with the project's model profile
and its coordinates get the same transform the LAS export would apply, so DXF coordinates match the LAS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import laspy
import numpy as np

from samcloud.classification.profiles import (
    NEVER_CLASSIFIED_ID,
    ModelProfile,
    ProfileInfo,
    profile_for_engine,
    resolve_las_profile,
    sidecar_path,
)
from samcloud.dxf.cancel import CancelToken, ProgressCallback, check_cancel, report_progress
from samcloud.dxf.config import DxfExportConfig, load_dxf_config


@dataclass(frozen=True, slots=True)
class Frame:
    """Reversible source-to-local transform: origin subtraction, scale, then rotation."""

    origin: np.ndarray
    scale_factor: float = 1.0
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=np.float64)
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if origin.shape != (3,):
            raise ValueError("Frame origin must have exactly three coordinates")
        if rotation.shape != (3, 3):
            raise ValueError("Frame rotation must be a 3 by 3 matrix")
        if not np.isfinite(origin).all() or not np.isfinite(rotation).all() or self.scale_factor <= 0:
            raise ValueError("Frame values must be finite and scale must be positive")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError("Frame rotation must be orthonormal")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "rotation", rotation)

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """Transform file coordinates to local scaled and aligned coordinates."""
        values = _points_array(points)
        return ((values - self.origin) * self.scale_factor) @ self.rotation.T

    def to_original(self, points: np.ndarray) -> np.ndarray:
        """Transform local scaled and aligned coordinates back to file coordinates."""
        values = _points_array(points)
        return (values @ self.rotation) / self.scale_factor + self.origin

    def with_scale(self, factor: float) -> "Frame":
        """Return a frame with an explicitly selected scale factor."""
        return Frame(self.origin, factor, self.rotation)

    def with_rotation(self, rotation: np.ndarray) -> "Frame":
        """Return a frame with a replacement local alignment rotation."""
        return Frame(self.origin, self.scale_factor, rotation)


@dataclass(frozen=True, slots=True)
class LasInfo:
    """Metadata collected from a LAS file without loading its point cloud at once."""

    path: Path
    point_count: int
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    has_rgb: bool
    class_histogram: dict[int, int]
    profile: ProfileInfo | None
    mode: Literal["indoor", "outdoor"]
    frame: Frame

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return minimum and maximum XYZ coordinates in the file coordinate system."""
        return self.bounds_min, self.bounds_max

    @property
    def profile_info(self) -> ProfileInfo | None:
        """Return LAS provenance when the source provided it."""
        return self.profile

    @property
    def extent(self) -> np.ndarray:
        """Return XYZ extent in file coordinate units."""
        return self.bounds_max - self.bounds_min


def iter_las_chunks(
    path: Path,
    chunk_size: int,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
    stage: str = "Reading LAS points",
) -> Iterator[laspy.ScaleAwarePointRecord]:
    """Yield LAS chunks while checking cancellation between disk reads."""
    if chunk_size < 1:
        raise ValueError("Chunk size must be at least 1")
    source_path = Path(path)
    with laspy.open(source_path) as reader:
        total = int(reader.header.point_count)
        processed = 0
        for chunk in reader.chunk_iterator(chunk_size):
            check_cancel(cancel)
            processed += len(chunk)
            report_progress(progress, processed / total if total else 1.0, stage)
            yield chunk
    check_cancel(cancel)


@dataclass(frozen=True, slots=True)
class _PlyPointFormat:
    dimension_names: tuple[str, ...]


class PlyPointChunk:
    """One chunk of a classified PLY in the attribute layout of a laspy point record.

    x, y, z are float64 in the LAS export frame, classification holds Cyclone codes and red/green/blue are
    16-bit like LAS colours (8-bit PLY value * 257), so LAS consumers handle both sources identically.
    """

    DIMENSIONS: tuple[str, ...] = ("X", "Y", "Z", "classification", "red", "green", "blue")

    def __init__(self, xyz: np.ndarray, classification: np.ndarray, rgb: np.ndarray) -> None:
        self.x = np.ascontiguousarray(xyz[:, 0], dtype=np.float64)
        self.y = np.ascontiguousarray(xyz[:, 1], dtype=np.float64)
        self.z = np.ascontiguousarray(xyz[:, 2], dtype=np.float64)
        self.classification = np.asarray(classification, dtype=np.uint8)
        self.red = rgb[:, 0].astype(np.uint16) * 257
        self.green = rgb[:, 1].astype(np.uint16) * 257
        self.blue = rgb[:, 2].astype(np.uint16) * 257
        self.point_format = _PlyPointFormat(self.DIMENSIONS)

    def __len__(self) -> int:
        return len(self.classification)


@dataclass(frozen=True, slots=True)
class ClassifiedPlySource:
    """A project's classified PLY plus what is needed to read it like the LAS export would write it."""

    path: Path
    project_directory: Path
    class_set: Literal["indoor", "outdoor"]
    profile: ModelProfile
    source_crs: str
    target_crs: str

    @property
    def profile_info(self) -> ProfileInfo:
        """Provenance equivalent to the profile sidecar the LAS export writes."""
        return ProfileInfo(self.profile.name, self.class_set, self.class_set, self.profile.code_to_name())

    @property
    def needs_georeference(self) -> bool:
        """LOCAL GPS-aligned projects exported to a CRS are placed with products/georeference.json."""
        return self.source_crs.upper() == "LOCAL" and self.target_crs.upper() != "LOCAL"

    def transform_key(self) -> str:
        """Identify the coordinate transform so cached bounds are dropped when it changes."""
        stamp: list[Any] = []
        georeference = self.project_directory / "products" / "georeference.json"
        if self.needs_georeference and georeference.is_file():
            stat = georeference.stat()
            stamp = [stat.st_mtime_ns, stat.st_size]
        payload = json.dumps([self.source_crs, self.target_crs, stamp])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def transformer(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the LAS export's coordinate transform (CRS transform or ENU georeference) for PLY coordinates."""
        from samcloud.engine.export_las import _create_transformer, _load_georeference_for_export, _transform_xyz
        from samcloud.pointcloud.georeference import ensure_project_georeference, export_target_crs, require_export_grids

        output_crs = export_target_crs(self.target_crs)
        require_export_grids(self.target_crs)
        georeference_path = ensure_project_georeference(self.project_directory) if self.needs_georeference else None
        georeference = _load_georeference_for_export(self.source_crs, self.target_crs, georeference_path)
        transformer = _create_transformer(self.source_crs, output_crs, georeference is not None)
        target_crs = self.target_crs

        def transform(xyz: np.ndarray) -> np.ndarray:
            return np.asarray(_transform_xyz(np.asarray(xyz, dtype=np.float64), transformer, georeference, target_crs), dtype=np.float64)

        return transform

    def classify(self, scalar_classes: np.ndarray) -> np.ndarray:
        """Map PLY scalar class ids to Cyclone codes exactly like the LAS export (255 = never classified, 0 = unknown)."""
        codes = self.profile.codes_for_scalar_classes()
        values = np.asarray(scalar_classes, dtype=np.uint8)
        classified = values != NEVER_CLASSIFIED_ID
        if values[classified].max(initial=0) >= len(codes):
            raise ValueError("PLY contains a class ID not defined by this export")
        result = np.full(len(values), self.profile.point_states["never_classified"], dtype=np.uint8)
        result[classified] = codes[values[classified]]
        return result


def is_ply_source(path: Path | str) -> bool:
    """True when the point source is a classified PLY instead of a LAS file."""
    return Path(path).suffix.lower() == ".ply"


def classified_ply_source(path: Path | str) -> ClassifiedPlySource:
    """Describe a project's classified PLY; the profile follows the project's engine and class set.

    The project folder is the one holding products/classified/classified.ply. No points are read.
    """
    from samcloud.classification.custom import migrate_classes_config, project_classes_path
    from samcloud.core.configuration import load_yaml
    from samcloud.engine.export_las import read_ply_header

    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise ValueError(f"Classified PLY was not found: {source_path}")
    read_ply_header(source_path)
    project_directory = source_path.parents[2]
    project_file = project_directory / "project.samcloud.yml"
    settings: dict[str, Any] = {}
    if project_file.is_file():
        settings = dict(load_yaml(project_file).get("settings", {}) or {})
    classification = dict(settings.get("classification", {}) or {})
    coordinates = dict(settings.get("coordinate_systems", {}) or {})
    class_set_value = str(classification.get("class_set", "outdoor"))
    if class_set_value not in ("indoor", "outdoor"):
        raise ValueError(f"Unsupported class set: {class_set_value}")
    class_set: Literal["indoor", "outdoor"] = "indoor" if class_set_value == "indoor" else "outdoor"
    engine = str(classification.get("engine", "sam3"))
    config = migrate_classes_config(json.loads(project_classes_path(project_directory).read_text(encoding="utf-8")))
    profile = profile_for_engine(engine, class_set, config)
    return ClassifiedPlySource(
        source_path,
        project_directory,
        class_set,
        profile,
        str(coordinates.get("working_crs", "LOCAL")),
        str(coordinates.get("export_crs", "LOCAL")),
    )


def iter_classified_ply_chunks(
    path: Path,
    chunk_size: int,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
    stage: str = "Reading classified PLY points",
    source: ClassifiedPlySource | None = None,
) -> Iterator[PlyPointChunk]:
    """Yield classified PLY chunks with Cyclone codes and LAS-export coordinates, checking cancellation per chunk."""
    from samcloud.engine.export_las import _ply_chunks, read_ply_header

    if chunk_size < 1:
        raise ValueError("Chunk size must be at least 1")
    described = source or classified_ply_source(path)
    transform = described.transformer()
    payload_offset, vertex_count = read_ply_header(described.path)
    processed = 0
    with described.path.open("rb") as handle:
        for points in _ply_chunks(handle, payload_offset, vertex_count, chunk_size):
            check_cancel(cancel)
            xyz = transform(np.column_stack((points["x"], points["y"], points["z"])))
            if not np.all(np.isfinite(xyz)):
                raise ValueError("Coordinate transformation produced non-finite values")
            rgb = np.column_stack((points["red"], points["green"], points["blue"]))
            processed += len(points)
            report_progress(progress, processed / vertex_count if vertex_count else 1.0, stage)
            yield PlyPointChunk(xyz, described.classify(points["scalar_class"]), rgb)
    check_cancel(cancel)


def iter_source_chunks(
    path: Path,
    chunk_size: int,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
    stage: str | None = None,
) -> Iterator[laspy.ScaleAwarePointRecord | PlyPointChunk]:
    """Yield point chunks from a classified LAS or a project's classified PLY, chosen by the file suffix."""
    if is_ply_source(path):
        return iter_classified_ply_chunks(path, chunk_size, progress=progress, cancel=cancel, stage=stage or "Reading classified PLY points")
    return iter_las_chunks(path, chunk_size, progress=progress, cancel=cancel, stage=stage or "Reading LAS points")


def _ply_cache_path(source: ClassifiedPlySource) -> Path:
    return source.project_directory / "products" / "dxf" / "ply_source.json"


def _ply_cache_key(source: ClassifiedPlySource) -> list[Any]:
    stat = source.path.stat()
    return [str(source.path), stat.st_mtime_ns, stat.st_size, source.transform_key()]


def _read_ply_cache(source: ClassifiedPlySource) -> dict[str, Any] | None:
    path = _ply_cache_path(source)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if payload.get("key") == _ply_cache_key(source) else None


def _write_ply_cache(source: ClassifiedPlySource, payload: dict[str, Any]) -> None:
    target = _ply_cache_path(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".ply_source.{os.getpid()}.tmp.json")
    temporary.write_text(json.dumps({"key": _ply_cache_key(source), **payload}, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def _scan_classified_ply(
    source: ClassifiedPlySource,
    chunk_size: int,
    progress: ProgressCallback | None,
    cancel: CancelToken | None,
) -> dict[str, Any]:
    """Collect transformed bounds and the Cyclone-code histogram of a classified PLY in one pass, then cache them."""
    histogram = np.zeros(256, dtype=np.int64)
    bounds_min = np.full(3, np.inf)
    bounds_max = np.full(3, -np.inf)
    count = 0
    for chunk in iter_classified_ply_chunks(source.path, chunk_size, progress=progress, cancel=cancel,
                                            stage="Reading classified PLY classifications", source=source):
        if len(chunk) == 0:
            continue
        xyz = np.column_stack((chunk.x, chunk.y, chunk.z))
        bounds_min = np.minimum(bounds_min, xyz.min(axis=0))
        bounds_max = np.maximum(bounds_max, xyz.max(axis=0))
        histogram += np.bincount(chunk.classification, minlength=256)
        count += len(chunk)
    if count == 0:
        raise ValueError(f"Classified PLY contains no points: {source.path}")
    payload: dict[str, Any] = {
        "point_count": count,
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "class_histogram": {str(index): int(value) for index, value in enumerate(histogram) if value},
    }
    _write_ply_cache(source, payload)
    return payload


def _read_classified_ply_info(
    path: Path,
    chunk_size: int,
    progress: ProgressCallback | None,
    cancel: CancelToken | None,
) -> LasInfo:
    source = classified_ply_source(path)
    scan = _scan_classified_ply(source, chunk_size, progress, cancel)
    bounds_min = np.asarray(scan["bounds_min"], dtype=np.float64)
    bounds_max = np.asarray(scan["bounds_max"], dtype=np.float64)
    profile = source.profile_info
    return LasInfo(
        source.path,
        int(scan["point_count"]),
        bounds_min,
        bounds_max,
        True,
        {int(code): int(value) for code, value in scan["class_histogram"].items()},
        profile,
        profile.mode,
        Frame(np.round(bounds_min).astype(np.float64)),
    )


def read_las_info(
    path: Path,
    *,
    chunk_size: int | None = None,
    config: DxfExportConfig | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> LasInfo:
    """Read point-source metadata and a class histogram using bounded chunks only (LAS or classified PLY)."""
    settings = config or load_dxf_config()
    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"LAS file was not found: {source_path}")
    chosen_chunk_size = settings.loading.chunk_size if chunk_size is None else chunk_size
    if is_ply_source(source_path):
        return _read_classified_ply_info(source_path, chosen_chunk_size, progress, cancel)
    profile: ProfileInfo | None = None
    with laspy.open(source_path) as reader:
        header = reader.header
        bounds_min = np.asarray(header.mins, dtype=np.float64)
        bounds_max = np.asarray(header.maxs, dtype=np.float64)
        point_count = int(header.point_count)
        has_rgb = all(name in header.point_format.dimension_names for name in ("red", "green", "blue"))
        profile = _profile_from_header(source_path, header)
    histogram = np.zeros(256, dtype=np.int64)
    for chunk in iter_las_chunks(source_path, chosen_chunk_size, progress=progress, cancel=cancel, stage="Reading LAS classifications"):
        classes = np.asarray(chunk.classification, dtype=np.uint8)
        histogram += np.bincount(classes, minlength=256)
    if profile is not None:
        mode: Literal["indoor", "outdoor"] = profile.mode
    else:
        mode = _detect_histogram_mode(histogram, settings)
    origin = np.round(bounds_min).astype(np.float64)
    return LasInfo(
        source_path,
        point_count,
        bounds_min,
        bounds_max,
        has_rgb,
        {index: int(count) for index, count in enumerate(histogram) if count},
        profile,
        mode,
        Frame(origin),
    )


def read_las_header_profile(path: Path) -> ProfileInfo | None:
    """Resolve provenance without reading points: LAS sidecar or VLRs, or the project profile of a classified PLY."""
    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"LAS file was not found: {source_path}")
    if is_ply_source(source_path):
        return classified_ply_source(source_path).profile_info
    with laspy.open(source_path) as reader:
        return _profile_from_header(source_path, reader.header)


def read_las_header_origin(path: Path) -> np.ndarray:
    """Return the local-frame origin used by read_las_info, reading only the LAS header.

    A classified PLY has no header bounds: its origin comes from products/dxf/ply_source.json written by the last
    read_las_info, and only without a matching cache are the points scanned once.
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"LAS file was not found: {source_path}")
    if is_ply_source(source_path):
        source = classified_ply_source(source_path)
        cached = _read_ply_cache(source) or _scan_classified_ply(source, load_dxf_config().loading.chunk_size, None, None)
        return np.round(np.asarray(cached["bounds_min"], dtype=np.float64)).astype(np.float64)
    with laspy.open(source_path) as reader:
        return np.round(np.asarray(reader.header.mins, dtype=np.float64)).astype(np.float64)


def plausibility_warnings(info: LasInfo, config: DxfExportConfig | None = None) -> list[str]:
    """Return non-blocking extent warnings for a loaded point cloud."""
    settings = config or load_dxf_config()
    if info.mode == "indoor":
        minimum, maximum = settings.plausibility.indoor_min_extent, settings.plausibility.indoor_max_extent
    else:
        minimum, maximum = settings.plausibility.outdoor_min_extent, settings.plausibility.outdoor_max_extent
    extent = info.extent * info.frame.scale_factor
    warnings: list[str] = []
    largest = float(np.max(extent))
    if largest < minimum:
        warnings.append(f"Point-cloud extent {largest:.3f} m is smaller than the expected {info.mode} minimum of {minimum:.3f} m.")
    if largest > maximum:
        warnings.append(f"Point-cloud extent {largest:.3f} m exceeds the expected {info.mode} maximum of {maximum:.3f} m.")
    return warnings


def _detect_histogram_mode(histogram: np.ndarray, config: DxfExportConfig) -> Literal["indoor", "outdoor"]:
    total = int(histogram.sum())
    if total == 0:
        return "outdoor"
    indoor = int(histogram[100] + histogram[101] + histogram[102])
    if indoor / total > config.mode_detection.indoor_share:
        return "indoor"
    return "outdoor"


def _profile_from_header(path: Path, header: laspy.LasHeader) -> ProfileInfo | None:
    """Resolve sidecar provenance first, then inspect VLRs without reading points."""
    if sidecar_path(path).is_file():
        return resolve_las_profile(path)
    for vlr in header.vlrs:
        if vlr.user_id == "colmap_sam" and vlr.record_id == 1:
            payload = json.loads(bytes(vlr.record_data).decode("utf-8"))
            if "model_profile" in payload:
                return ProfileInfo(
                    str(payload["model_profile"]),
                    payload["mode"],
                    str(payload.get("semantic_class_set", payload["mode"])),
                    payload.get("class_mapping", {}),
                )
    return None


def _points_array(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.shape[-1:] != (3,):
        raise ValueError("Points must have a final dimension of length three")
    return values
