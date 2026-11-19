"""Scene container: project settings, spatial structure, types and elements."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from samcloud.bim.core.model.elements import (
    DoorElement,
    Element,
    MaterialLayer,
    OpeningElement,
    PropertyValue,
    Real,
    WallJoinElement,
    WindowElement,
    new_id,
)
from samcloud.bim.core.model.enums import (
    CeilingIfcClass,
    DataSource,
    ElementKind,
    ImageMode,
    IndoorDetail,
    Scope,
    VisibilityMode,
)

SCHEMA_VERSION = 1
"""Version of the scene JSON layout. Increase when the layout changes."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CrsInfo(_Strict):
    """Coordinate reference system of the project (taken from the input, see Q4)."""

    epsg: int | None = Field(default=None, gt=0)
    wkt: str | None = None
    name: str = ""
    vertical_datum: str | None = None
    source: Literal["input", "user", "none"] = "none"


class ProjectSettings(_Strict):
    """Project setup choices (section 3.0) and export settings."""

    data_source: DataSource = DataSource.SLAM
    visibility_mode: dict[str, VisibilityMode] = Field(default_factory=dict)
    scope: Scope = Scope.INDOOR_ONLY
    indoor_detail: IndoorDetail = IndoorDetail.STRUCTURE_ONLY
    image_mode: ImageMode = ImageMode.WITH_IMAGES
    tolerance_profile: str = "tls"
    class_map_path: str = ""
    origin_offset: tuple[Real, Real, Real] = (0.0, 0.0, 0.0)
    crs: CrsInfo | None = None
    ceiling_ifc_class: CeilingIfcClass = CeilingIfcClass.SLAB
    auto_lock_on_manual_edit: bool = False
    extract_stairs: bool = True
    """Stairs toggle of the structure_only detail level (section 3.0), default on."""


class Site(_Strict):
    """Site (IfcSite)."""

    id: str = Field(default_factory=new_id, min_length=1)
    name: str = ""


class Building(_Strict):
    """Building (IfcBuilding)."""

    id: str = Field(default_factory=new_id, min_length=1)
    name: str = ""
    site_id: str | None = None


class Storey(_Strict):
    """Building storey with its elevation in the local frame (m)."""

    id: str = Field(default_factory=new_id, min_length=1)
    name: str = ""
    elevation: Real = 0.0
    height: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    building_id: str | None = None


class ElementType(_Strict):
    """Shared type object (for example a wall type)."""

    id: str = Field(default_factory=new_id, min_length=1)
    kind: ElementKind
    name: str = ""
    ifc_type_class: str = ""
    predefined_type: str | None = None
    properties: dict[str, PropertyValue] = Field(default_factory=dict)
    layers: list[MaterialLayer] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> ElementType:
        if self.layers and self.kind not in (ElementKind.WALL, ElementKind.SLAB):
            raise ValueError("layers are only allowed on wall and slab types")
        return self


class Scene(_Strict):
    """The source of truth for geometry, IFC export and review."""

    schema_version: int = SCHEMA_VERSION
    settings: ProjectSettings = Field(default_factory=ProjectSettings)
    sites: dict[str, Site] = Field(default_factory=dict)
    buildings: dict[str, Building] = Field(default_factory=dict)
    storeys: dict[str, Storey] = Field(default_factory=dict)
    types: dict[str, ElementType] = Field(default_factory=dict)
    elements: dict[str, Element] = Field(default_factory=dict)
    derived: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_keys(self) -> Scene:
        collections: dict[str, dict[str, Any]] = {
            "sites": self.sites,
            "buildings": self.buildings,
            "storeys": self.storeys,
            "types": self.types,
            "elements": self.elements,
        }
        for label, items in collections.items():
            for key, item in items.items():
                if key != item.id:
                    raise ValueError(f"{label} key {key!r} differs from id {item.id!r}")
        return self

    def add(self, element: Element) -> Element:
        """Add an element, rejecting duplicate ids."""
        if element.id in self.elements:
            raise ValueError(f"duplicate element id {element.id!r}")
        self.elements[element.id] = element
        return element

    def get(self, element_id: str) -> Element:
        """Return the element with the given id (``KeyError`` if missing)."""
        return self.elements[element_id]

    def elements_of(self, kind: ElementKind) -> list[Element]:
        """Return all elements of a kind in insertion order."""
        return [element for element in self.elements.values() if element.kind == kind]

    def check_references(self) -> list[str]:
        """Return messages for dangling references between scene objects."""
        problems: list[str] = []

        def need(label: str, ref: str | None, pool: dict[str, Any]) -> None:
            if ref is not None and ref not in pool:
                problems.append(f"{label}: unknown id {ref!r}")

        for building in self.buildings.values():
            need(f"building {building.id} site_id", building.site_id, self.sites)
        for storey in self.storeys.values():
            need(f"storey {storey.id} building_id", storey.building_id, self.buildings)
        for element in self.elements.values():
            prefix = f"{element.kind.value} {element.id}"
            need(f"{prefix} storey_id", element.storey_id, self.storeys)
            need(f"{prefix} building_id", element.building_id, self.buildings)
            need(f"{prefix} site_id", element.site_id, self.sites)
            need(f"{prefix} type_id", element.type_id, self.types)
            for space_id in element.space_ids:
                self._need_kind(problems, f"{prefix} space_ids", space_id, ElementKind.SPACE)
            if isinstance(element, WallJoinElement):
                for field, ref in (
                    ("wall_a", element.params.wall_a),
                    ("wall_b", element.params.wall_b),
                ):
                    self._need_kind(problems, f"{prefix} {field}", ref, ElementKind.WALL)
            elif isinstance(element, OpeningElement):
                self._need_kind(
                    problems,
                    f"{prefix} host_wall_id",
                    element.params.host_wall_id,
                    ElementKind.WALL,
                )
            elif isinstance(element, WindowElement | DoorElement):
                self._need_kind(
                    problems, f"{prefix} opening_id", element.params.opening_id, ElementKind.OPENING
                )
        return problems

    def _need_kind(self, problems: list[str], label: str, ref: str, kind: ElementKind) -> None:
        target = self.elements.get(ref)
        if target is None:
            problems.append(f"{label}: unknown id {ref!r}")
        elif target.kind != kind:
            problems.append(f"{label}: {ref!r} is a {target.kind.value}, expected {kind.value}")

    def to_json(self) -> str:
        """Return deterministic JSON: sorted keys, 2-space indent, trailing newline."""
        data = self.model_dump(mode="json")
        return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n"

    def save(self, path: str | Path) -> None:
        """Write the scene as deterministic JSON (UTF-8, LF line endings)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_bytes(self.to_json().encode("utf-8"))
        os.replace(temporary, target)

    @classmethod
    def load(cls, path: str | Path) -> Scene:
        """Read and validate a scene JSON file."""
        return cls.model_validate_json(Path(path).read_bytes())
