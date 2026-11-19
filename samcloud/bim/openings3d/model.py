"""Result types of the 3D opening refinement and ``openings_3d.json`` persistence."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
OPENINGS_3D_FILENAME = "openings_3d.json"


@dataclass(slots=True)
class RefinedOpening:
    """One opening in PlanOpening-compatible form (centre in plan units, every ``*_m`` in metres)."""

    id: str
    kind: str  # door | window | passage
    center: list[float]
    width_m: float
    axis_angle: float
    wall_thickness_m: float
    host_wall_id: str | None
    sill_m: float
    head_m: float
    height_source: str  # photo+points | photo | points | default
    mullion_count: int = 0
    mullion_u: list[float] = field(default_factory=list)  # metres from the left edge (along the axis)
    glazed: bool = False
    evidence: str = ""
    confidence: float = 0.0
    line: str | None = None  # wall line the opening was measured on
    u0: float = 0.0  # extent along the wall line (m), informative
    u1: float = 0.0
    plan_seed: str | None = None
    # Door classification (classify.py): frame / head / headroom measured on the wall plane, open leaf of a door.
    classification: str = ""  # door | passage (| window) as classified from the evidence
    headroom_m: float | None = None  # ceiling minus the measured head (None: head not measured)
    frame_jambs: list[str] = field(default_factory=list)  # jambs with a frame strip, "left" / "right" along the opening axis
    head_frame: bool = False  # door-class head strip above the opening
    hinge_side: str | None = None  # left | right (hinge at the -axis / +axis end of the opening), only with a detected leaf
    swing_side: int | None = None  # +1 / -1: side of the +normal the leaf swings to
    leaf_angle_deg: float | None = None  # 0 closed ... 90 square to the wall ... 180 flat against the wall
    leaf_detected: bool = False
    leaf: dict[str, Any] | None = None  # measured leaf (hinge / free end in plan metres, length, points)

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RefinedOpening":
        names = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})


@dataclass(slots=True)
class OpeningsResult:
    openings: list[RefinedOpening]
    curtain_regions: list[dict[str, Any]] = field(default_factory=list)
    unconfirmed: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    closed: list[dict[str, Any]] = field(default_factory=list)  # door candidates classified as no opening: the wall closes there (center, width_m, axis_angle, ...)
    inputs: dict[str, Any] = field(default_factory=dict)  # fingerprint: walls, config hash, image set
    mode: str = "full"  # full (photos + points) | points
    stats: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "inputs": self.inputs,
            "stats": self.stats,
            "openings": [item.to_dict() for item in self.openings],
            "curtain_regions": self.curtain_regions,
            "unconfirmed": self.unconfirmed,
            "rejected": self.rejected,
            "closed": self.closed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpeningsResult":
        version = int(data.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported openings_3d schema_version {version}")
        return cls(
            openings=[RefinedOpening.from_dict(item) for item in data.get("openings", [])],
            curtain_regions=list(data.get("curtain_regions", [])), unconfirmed=list(data.get("unconfirmed", [])),
            rejected=list(data.get("rejected", [])), closed=list(data.get("closed", [])), inputs=dict(data.get("inputs", {})), mode=str(data.get("mode", "full")),
            stats=dict(data.get("stats", {})), schema_version=version,
        )


def openings_3d_path(directory: Path | str) -> Path:
    return Path(directory) / OPENINGS_3D_FILENAME


def save_openings_3d(result: OpeningsResult, directory: Path | str) -> Path:
    """Write ``openings_3d.json`` atomically into ``directory`` (the level's clean-plan cache folder)."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    destination = openings_3d_path(target)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(result.to_dict(), indent=1, default=float), encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def load_openings_3d(directory_or_file: Path | str) -> OpeningsResult:
    path = Path(directory_or_file)
    if path.is_dir():
        path = openings_3d_path(path)
    return OpeningsResult.from_dict(json.loads(path.read_text(encoding="utf-8")))


def openings_3d_sha256(directory_or_file: Path | str) -> str | None:
    path = Path(directory_or_file)
    if path.is_dir():
        path = openings_3d_path(path)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None
