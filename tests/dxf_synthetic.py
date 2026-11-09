"""Deterministic classified LAS fixtures shared by DXF export tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import laspy
import numpy as np

from samcloud.classification.profiles import flurseg_profile, sam3_profile, write_profile_sidecar


def write_indoor_building(
    path: Path,
    *,
    tilt_degrees: float = 0.0,
    z_rotation_degrees: float = 0.0,
    scale: float = 1.0,
    origin_offset: tuple[float, float, float] | None = None,
    two_windows: bool = False,
    seed: int = 20261109,
) -> Path:
    """Write a noisy two-storey, Cyclone-classified LAS 1.4 building fixture."""
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    codes: list[np.ndarray] = []

    def add(points: np.ndarray, code: int) -> None:
        parts.append(points + rng.normal(0.0, 0.005, points.shape))
        codes.append(np.full(len(points), code, dtype=np.uint8))

    for height in (0.0, 3.0):
        add(_plane_xy(10.0, 6.0, height, 34, 22), 102)
    for height in (2.6, 5.6):
        add(_plane_xy(10.0, 6.0, height, 34, 22), 101)
    for height in (0.0, 3.0):
        add(_wall_x(0.0, 6.0, height, height + 2.6, 121, 27), 100)
        add(_wall_x(10.0, 6.0, height, height + 2.6, 121, 27), 100)
        south = _wall_y(0.0, 10.0, height, height + 2.6, 201, 27)
        north = _wall_y(6.0, 10.0, height, height + 2.6, 201, 27)
        if height == 0.0:
            south = south[~((np.abs(south[:, 0] - 4.2) < 0.6) & (south[:, 2] < 2.1))]
            north = north[~((np.abs(north[:, 0] - 6.2) < 0.6) & (north[:, 2] > 0.9) & (north[:, 2] < 1.8))]
            if two_windows:
                north = north[~((np.abs(north[:, 0] - 2.8) < 0.6) & (north[:, 2] > 0.9) & (north[:, 2] < 1.8))]
        add(south, 100)
        add(north, 100)
    door = np.column_stack((rng.uniform(3.62, 4.78, 140), rng.uniform(0.01, 0.08, 140), rng.uniform(0.0, 2.1, 140)))
    window = np.column_stack((rng.uniform(5.62, 6.78, 120), rng.uniform(5.92, 5.99, 120), rng.uniform(0.9, 1.8, 120)))
    table = np.column_stack((rng.uniform(3.4, 5.0, 140), rng.uniform(2.3, 3.1, 140), np.full(140, 0.75)))
    stairs = np.column_stack((rng.uniform(0.8, 2.0, 320), rng.uniform(1.0, 2.1, 320), np.linspace(0.0, 3.0, 320)))
    add(door, 103)
    add(window, 104)
    if two_windows:
        second_window = np.column_stack((rng.uniform(2.22, 3.38, 120), rng.uniform(5.92, 5.99, 120), rng.uniform(0.9, 1.8, 120)))
        add(second_window, 104)
    add(table, 119)
    add(stairs, 105)
    xyz = np.vstack(parts)
    classification = np.concatenate(codes)
    rotation = _rotation_x(np.radians(tilt_degrees))
    xyz = (xyz @ rotation.T) * scale
    if z_rotation_degrees:
        xyz = xyz @ _rotation_z(np.radians(z_rotation_degrees)).T
    if origin_offset is not None:
        xyz += np.asarray(origin_offset, dtype=np.float64)
    return _write_las(path, xyz, classification, "indoor")


def write_outdoor_scene(
    path: Path,
    *,
    profile: Literal["sam3", "flurseg"] = "sam3",
    include_tree: bool = True,
    seed: int = 20261109,
) -> Path:
    """Write a compact outdoor scene using the selected exported model profile."""
    rng = np.random.default_rng(seed)
    objects = [
        (_plane_xy(30.0, 20.0, 0.0, 40, 28), 2),
        (_plane_xy(18.0, 4.0, 0.03, 32, 8) + np.array([6.0, 8.0, 0.0]), 11 if profile == "sam3" else 71),
        (_box_surface((5.0, 5.0, 1.5), (10.0, 5.0, 4.0), rng), 6),
        (_box_surface((18.0, 5.0, 0.3), (2.2, 1.0, 1.2), rng), 91),
        (_line_points((4.0, 13.0, 0.0), (12.0, 13.0, 0.15), 80), 74),
        (_disc_points((15.0, 13.0, 0.02), 0.35, 50, rng), 75),
        (_line_points((23.0, 12.0, 0.0), (23.0, 12.0, 4.0), 50), 84 if profile == "sam3" else 64),
        (_box_surface((25.0, 12.0, 2.0), (0.7, 0.08, 0.5), rng), 81 if profile == "sam3" else 64),
        (_line_points((2.0, 18.0, 0.0), (20.0, 18.0, 1.3), 100), 77 if profile == "sam3" else 64),
    ]
    if profile == "flurseg":
        marking = np.column_stack((rng.uniform(8.0, 22.0, 400), rng.uniform(9.93, 10.07, 400), np.full(400, 0.035)))
        objects.append((marking, 82))
    if include_tree:
        objects.insert(3, (_tree_points(rng), 5))
    xyz = np.vstack([item[0] for item in objects])
    xyz += rng.normal(0.0, 0.007, xyz.shape)
    classification = np.concatenate([np.full(len(item[0]), item[1], dtype=np.uint8) for item in objects])
    return _write_las(path, xyz, classification, "outdoor", profile)


def write_classified_las(
    path: Path,
    xyz: np.ndarray,
    classification: np.ndarray,
    *,
    mode: Literal["indoor", "outdoor"] = "outdoor",
    profile_name: str = "sam3",
) -> Path:
    """Write caller-supplied noisy classified points using the standard DXF provenance."""
    return _write_las(path, np.asarray(xyz, dtype=np.float64), np.asarray(classification, dtype=np.uint8), mode, profile_name)


def _write_las(path: Path, xyz: np.ndarray, classification: np.ndarray, mode: str, profile_name: str = "sam3") -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor(np.min(xyz, axis=0))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    cloud.classification = classification
    cloud.red = np.full(len(xyz), 50000, dtype=np.uint16)
    cloud.green = np.full(len(xyz), 35000, dtype=np.uint16)
    cloud.blue = np.full(len(xyz), 18000, dtype=np.uint16)
    cloud.write(destination)
    classes = json.loads((Path(__file__).parents[1] / "config" / "classes.json").read_text(encoding="utf-8"))
    profile = flurseg_profile("outdoor") if profile_name == "flurseg" else sam3_profile(mode, classes)
    write_profile_sidecar(destination, profile, mode, mode)
    return destination


def _plane_xy(width: float, depth: float, z: float, nx: int, ny: int) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(0.0, width, nx), np.linspace(0.0, depth, ny))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))


def _wall_x(x: float, depth: float, lower: float, upper: float, ny: int, nz: int) -> np.ndarray:
    y, z = np.meshgrid(np.linspace(0.0, depth, ny), np.linspace(lower, upper, nz))
    return np.column_stack((np.full(y.size, x), y.ravel(), z.ravel()))


def _wall_y(y: float, width: float, lower: float, upper: float, nx: int, nz: int) -> np.ndarray:
    x, z = np.meshgrid(np.linspace(0.0, width, nx), np.linspace(lower, upper, nz))
    return np.column_stack((x.ravel(), np.full(x.size, y), z.ravel()))


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)))


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def _box_surface(origin: tuple[float, float, float], size: tuple[float, float, float], rng: np.random.Generator) -> np.ndarray:
    return np.asarray(origin) + rng.uniform(0.0, 1.0, (220, 3)) * np.asarray(size)


def _tree_points(rng: np.random.Generator) -> np.ndarray:
    trunk = np.column_stack((rng.normal(20.0, 0.12, 70), rng.normal(16.0, 0.12, 70), rng.uniform(0.0, 4.0, 70)))
    crown = rng.normal((20.0, 16.0, 5.0), (1.3, 1.3, 1.0), (280, 3))
    return np.vstack((trunk, crown))


def _line_points(start: tuple[float, float, float], end: tuple[float, float, float], count: int) -> np.ndarray:
    return np.linspace(np.asarray(start), np.asarray(end), count)


def _disc_points(center: tuple[float, float, float], radius: float, count: int, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    distance = radius * np.sqrt(rng.uniform(0.0, 1.0, count))
    return np.column_stack((center[0] + np.cos(angle) * distance, center[1] + np.sin(angle) * distance, np.full(count, center[2])))
