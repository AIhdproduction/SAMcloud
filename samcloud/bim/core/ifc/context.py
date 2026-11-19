"""Shared state of one IFC export and the export result."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import ifcopenshell

from samcloud.bim.core.geometry import SceneGeometry
from samcloud.bim.core.model import Scene

NAMESPACE = uuid.UUID("7a1c2b0e-52a1-5d6e-8f33-52616f6d6c6f")
"""Fixed namespace for deterministic GlobalIds of objects without a scene uuid."""

NOT_APPLICABLE = "not applicable"
UNREFERENCED = "unreferenced"


def class_of(entity: ifcopenshell.entity_instance) -> str:
    """IFC class name of an entity."""
    return str(entity.is_a())  # type: ignore[no-untyped-call]


def is_a(entity: ifcopenshell.entity_instance, name: str) -> bool:
    """True when the entity is of the IFC class (or a subtype)."""
    return bool(entity.is_a(name))  # type: ignore[no-untyped-call]


@dataclass(frozen=True)
class IfcExportResult:
    """Outcome of :func:`samcloud.bim.core.ifc.write_ifc`."""

    path: str
    counts: dict[str, int]
    """Number of IFC root entities per class, sorted by class name."""
    warnings: list[str]
    referenced: bool
    """True when IfcProjectedCRS and IfcMapConversion were written."""
    incomplete: bool
    """True when mandatory data is missing (MASTERPROMPT 10.1)."""
    flagged_elements: dict[str, list[str]]
    """Scene element id to the list of reasons it is flagged incomplete."""


@dataclass
class ExportContext:
    """Mutable state shared by the writer modules."""

    scene: Scene
    geometry: SceneGeometry
    file: ifcopenshell.file
    run_id: str | None
    ai_backend: str | None
    ai_model: str | None
    referenced: bool = False
    model_context: ifcopenshell.entity_instance | None = None
    body_context: ifcopenshell.entity_instance | None = None
    axis_context: ifcopenshell.entity_instance | None = None
    project: ifcopenshell.entity_instance | None = None
    sites: dict[str, ifcopenshell.entity_instance] = field(default_factory=dict)
    buildings: dict[str, ifcopenshell.entity_instance] = field(default_factory=dict)
    storeys: dict[str, ifcopenshell.entity_instance] = field(default_factory=dict)
    default_site_id: str = ""
    default_building_id: str = ""
    entities: dict[str, ifcopenshell.entity_instance] = field(default_factory=dict)
    """Scene element id to the IFC product (spaces and openings included)."""
    explicit_guids: set[int] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)
    flagged: dict[str, list[str]] = field(default_factory=dict)
    materials: dict[str, ifcopenshell.entity_instance] = field(default_factory=dict)
    types: dict[tuple[object, ...], ifcopenshell.entity_instance] = field(default_factory=dict)
    space_frames: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    """Space element id to the placement origin (x, y, z) used for its body."""

    @property
    def body(self) -> ifcopenshell.entity_instance:
        assert self.body_context is not None
        return self.body_context

    @property
    def axis(self) -> ifcopenshell.entity_instance:
        assert self.axis_context is not None
        return self.axis_context

    def warn(self, message: str) -> None:
        """Record a global warning."""
        self.warnings.append(message)

    def flag(self, element_id: str, message: str) -> None:
        """Flag an element as incomplete and record the reason (also a warning)."""
        self.flagged.setdefault(element_id, []).append(message)
        element = self.scene.elements.get(element_id)
        label = f"{element.kind.value} {element.name or element_id}" if element else element_id
        self.warnings.append(f"{label}: {message}")
