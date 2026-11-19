"""Synthetic classified point cloud of one wall with doors, passages, frames and open leaves (LAS, metres * units_per_metre).

The wall runs along +x at y = 0 (plan frame = file frame, identity), 4.4 m long, 2.5 m high, 0.25 m thick.  Every
surface is a random sample (seeded, noisy, never on a grid boundary), so the tests exercise the real streaming pass
(:func:`collect_point_evidence`) and the real classification at any ``units_per_metre``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import laspy
import numpy as np

from samcloud.bim.openings3d.config import Settings, load_openings_config
from samcloud.bim.openings3d.points import PointEvidence, collect_point_evidence
from samcloud.bim.openings3d.walls import WallLine, plan_frame, wall_lines
from samcloud.bim.plan_model import BuildingPlan, PlanWall

HEIGHT = 2.5
LENGTH = 4.4
WALL, CEILING, FLOOR, DOOR, FURNITURE = 100, 101, 102, 103, 117


def make_wall_plan(upm: float, length: float = LENGTH) -> BuildingPlan:
    walls = [PlanWall(id="W001", centreline=[[0.0, 0.0], [length * upm, 0.0]], thickness_m=0.25, kind="interior")]
    frame = {"origin": [0.0, 0.0, 0.0], "scale_factor": 1.0, "rotation": np.eye(3).tolist()}
    return BuildingPlan("synthetic", "level_1", "Level 1", upm, frame, 0.0, HEIGHT * upm, walls=walls)


@dataclass
class Cloud:
    """Surface samples in metres; ``write`` scales them to plan units."""

    seed: int = 5
    parts: list[np.ndarray] = field(default_factory=list)
    classes: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def _add(self, xyz: np.ndarray, code: int) -> None:
        self.parts.append(xyz)
        self.classes.append(np.full(len(xyz), code, dtype=np.uint8))

    def box(self, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, code: int, density: float = 2500.0, jitter: float = 0.004) -> None:
        """Random points inside an axis-aligned box (a thin box is a plane)."""
        volume_faces = max((x1 - x0) * (z1 - z0), (y1 - y0) * (z1 - z0), (x1 - x0) * (y1 - y0))
        count = max(int(density * volume_faces), 30)
        xyz = np.column_stack([self.rng.uniform(x0, x1, count), self.rng.uniform(y0, y1, count), self.rng.uniform(z0, z1, count)])
        self._add(xyz + self.rng.normal(0.0, jitter, xyz.shape), code)

    def wall(self, u0: float, u1: float, z0: float, z1: float) -> None:
        self.box(u0, u1, -0.012, 0.012, z0, z1, WALL, density=2200.0)

    def room_shell(self, length: float = LENGTH) -> None:
        self.box(0.0, length, -0.4, 0.4, 0.0, 0.01, FLOOR, density=300.0)
        self.box(0.0, length, -0.4, 0.4, HEIGHT - 0.03, HEIGHT - 0.01, CEILING, density=300.0)

    def wall_with_openings(self, openings: list[tuple[float, float, float]], length: float = LENGTH) -> None:
        """Wall pieces between the openings ``(u0, u1, head)``; lintel above each opening up to the ceiling."""
        cursor = 0.0
        for u0, u1, head in sorted(openings):
            self.wall(cursor, u0, 0.0, HEIGHT - 0.04)
            if head < HEIGHT - 0.12:
                self.wall(u0, u1, head, HEIGHT - 0.04)
            cursor = u1
        self.wall(cursor, length, 0.0, HEIGHT - 0.04)

    def frame(self, u0: float, u1: float, head: float = 2.05, jambs: tuple[str, ...] = ("left", "right"), head_strip: bool = True) -> None:
        """Door-class casing: jamb strips (0.2 m wide, across the wall) and a head strip, plus a wall-thickness reveal."""
        if "left" in jambs:
            self.box(u0 - 0.10, u0 + 0.10, -0.13, 0.13, 0.05, head, DOOR, density=1800.0)
        if "right" in jambs:
            self.box(u1 - 0.10, u1 + 0.10, -0.13, 0.13, 0.05, head, DOOR, density=1800.0)
        if head_strip:
            self.box(u0 - 0.10, u1 + 0.10, -0.13, 0.13, head - 0.08, head + 0.04, DOOR, density=1800.0)

    def closed_leaf(self, u0: float, u1: float, top: float = 2.0) -> None:
        self.box(u0 + 0.08, u1 - 0.08, -0.02, 0.02, 0.05, top, DOOR, density=2500.0)

    def open_leaf(self, hinge_u: float, angle_deg: float, length: float = 0.88, swing: int = 1, toward: int = -1) -> None:
        """Leaf of ``length`` hinged at x = hinge_u, y = 0; ``toward`` is -1 (opens over the -x jamb side) or +1; ``angle_deg`` from the closed leaf."""
        theta = math.radians(angle_deg)
        direction = np.array([toward * math.cos(theta), swing * math.sin(theta)])
        along = self.rng.uniform(0.03, length, int(3500 * length * 2.0))
        z = self.rng.uniform(0.05, 2.0, len(along))
        across = self.rng.normal(0.0, 0.006, len(along))
        normal = np.array([-direction[1], direction[0]])
        xy = np.array([hinge_u, 0.0]) + along[:, None] * direction[None, :] + across[:, None] * normal[None, :]
        self._add(np.column_stack((xy, z)), DOOR)

    def furniture(self, u0: float, u1: float, depth: float = 0.35, top: float = 1.9) -> None:
        self.box(u0, u1, -depth, depth, 0.0, top, FURNITURE, density=1500.0, jitter=0.01)

    def write(self, path: Path, upm: float) -> Path:
        xyz = np.concatenate(self.parts) * upm
        classes = np.concatenate(self.classes)
        header = laspy.LasHeader(point_format=7, version="1.4")
        header.scales = np.array([0.0005, 0.0005, 0.0005])
        header.offsets = np.floor(np.min(xyz, axis=0))
        data = laspy.LasData(header)
        data.x, data.y, data.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        data.classification = classes
        data.write(path)
        return path


def collect(cloud: Cloud, directory: Path, upm: float, length: float = LENGTH, config: Settings | None = None) -> tuple[PointEvidence, list[WallLine], BuildingPlan, Settings]:
    """Write the cloud and run the real point-evidence pass on it."""
    config = config or load_openings_config()
    plan = make_wall_plan(upm, length)
    lines = wall_lines(plan, config)
    source = cloud.write(directory / f"synthetic_{upm}.las", upm)
    evidence = collect_point_evidence(source, lines, plan, plan_frame(plan), config, want_cloud=False)
    return evidence, lines, plan, config
