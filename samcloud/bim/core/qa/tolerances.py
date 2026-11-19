"""Tolerance profiles loaded from configs/tolerances.yaml (MASTERPROMPT section 7)."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from samcloud.bim.core.model import ElementKind

from samcloud.core.paths import CONFIG_DIRECTORY

TOLERANCES_PATH = CONFIG_DIRECTORY / "bim" / "tolerances.yaml"
"""Tolerance profiles in ``config/bim``."""
TOLERANCES_NAME = "tolerances.yaml"
CONFIGS_ENV = "SAMCLOUD_BIM_CONFIGS_DIR"

Pos = Annotated[float, Field(gt=0)]
Rgb = tuple[
    Annotated[int, Field(ge=0, le=255)],
    Annotated[int, Field(ge=0, le=255)],
    Annotated[int, Field(ge=0, le=255)],
]

_KIND_TO_KEY = {
    ElementKind.WALL: "wall",
    ElementKind.SLAB: "floor",
    ElementKind.CEILING: "ceiling",
    ElementKind.OPENING: "opening",
    ElementKind.WINDOW: "window",
    ElementKind.DOOR: "door",
}
"""Element kind to the key in ``deviation_tol`` (slabs are floors)."""


class CeilingSagReference(StrEnum):
    """Reference of the ceiling sag tolerance (Q2)."""

    FITTED = "fitted"
    HORIZONTAL = "horizontal"


class DeviationTolerances(BaseModel):
    """Tolerance of the signed point-to-surface deviation per element type."""

    model_config = ConfigDict(extra="forbid")

    wall: Pos
    floor: Pos
    ceiling: Pos
    opening: Pos
    window: Pos
    door: Pos


class WallDefaultThickness(BaseModel):
    """Assumed thickness of one-sided walls (origin assumed)."""

    model_config = ConfigDict(extra="forbid")

    interior: Pos
    exterior: Pos


class ToleranceColors(BaseModel):
    """Tolerance view colors (RGB 0..255)."""

    model_config = ConfigDict(extra="forbid")

    within: Rgb = (46, 160, 67)
    front: Rgb = (214, 39, 40)
    behind: Rgb = (31, 119, 180)

    def palette(self) -> NDArray[np.uint8]:
        """``(3, 3)`` array: rows within, front, behind."""
        return np.array([self.within, self.front, self.behind], dtype=np.uint8)


class ToleranceProfile(BaseModel):
    """One tolerance profile (tls, slam or photogrammetry)."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    provisional: bool = False
    note: str = ""
    planar_snap_tol: Pos
    ceiling_sag_tol: Pos
    opening_dimension_target: Pos
    coverage_radius: Pos
    coverage_min: Annotated[float, Field(ge=0, le=1)]
    snap_quantile: Annotated[float, Field(gt=0, le=1)]
    noise_warning_factor: Pos = 2.0
    ceiling_sag_reference: CeilingSagReference = CeilingSagReference.FITTED
    wall_default_thickness: WallDefaultThickness
    deviation_tol: DeviationTolerances
    colors: ToleranceColors = Field(default_factory=ToleranceColors)

    @model_validator(mode="after")
    def _check(self) -> ToleranceProfile:
        if self.coverage_radius > 1.0:
            raise ValueError("coverage_radius above 1 m is not plausible")
        return self

    def tolerance_for(self, kind: ElementKind | str) -> float:
        """Deviation tolerance for an element kind (slab uses the floor value)."""
        return float(getattr(self.deviation_tol, _kind_key(kind)))


def _kind_key(kind: ElementKind | str) -> str:
    try:
        element_kind: ElementKind | None = ElementKind(kind)
    except ValueError:
        element_kind = None
    if element_kind in _KIND_TO_KEY:
        return _KIND_TO_KEY[element_kind]
    if isinstance(kind, str) and kind in DeviationTolerances.model_fields:
        return kind
    raise ValueError(f"no deviation tolerance defined for element kind {kind!r}")


def default_tolerances_path() -> Path:
    """Locate ``tolerances.yaml``.

    Order: the ``SAMCLOUD_BIM_CONFIGS_DIR`` environment variable, then ``config/bim``.
    """
    candidates: list[Path] = []
    override = os.environ.get(CONFIGS_ENV)
    if override:
        candidates.append(Path(override) / TOLERANCES_NAME)
    candidates.append(TOLERANCES_PATH)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        f"tolerance file not found (searched: {searched}); set {CONFIGS_ENV} to the configs folder"
    )


def load_tolerances(path: Path | str | None = None, profile: str = "tls") -> ToleranceProfile:
    """Load one profile from the tolerance file (default ``configs/tolerances.yaml``)."""
    file = Path(path) if path is not None else default_tolerances_path()
    data = yaml.safe_load(file.read_text(encoding="utf-8"))
    profiles = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(profiles, dict) or profile not in profiles:
        known = sorted(profiles) if isinstance(profiles, dict) else []
        raise KeyError(f"tolerance profile {profile!r} not in {file} (known: {known})")
    return ToleranceProfile.model_validate({**profiles[profile], "name": profile})
