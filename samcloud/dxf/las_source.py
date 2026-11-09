"""Chunked access to classified LAS sources and their coordinate frame."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterator, Literal

import laspy
import numpy as np

from samcloud.classification.profiles import ProfileInfo, resolve_las_profile, sidecar_path
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


def read_las_info(
    path: Path,
    *,
    chunk_size: int | None = None,
    config: DxfExportConfig | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> LasInfo:
    """Read LAS metadata and a class histogram using bounded chunks only."""
    settings = config or load_dxf_config()
    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"LAS file was not found: {source_path}")
    chosen_chunk_size = settings.loading.chunk_size if chunk_size is None else chunk_size
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
    """Resolve LAS provenance from the sidecar or header VLRs without reading any points."""
    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"LAS file was not found: {source_path}")
    with laspy.open(source_path) as reader:
        return _profile_from_header(source_path, reader.header)


def read_las_header_origin(path: Path) -> np.ndarray:
    """Return the local-frame origin used by read_las_info, reading only the LAS header."""
    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"LAS file was not found: {source_path}")
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
