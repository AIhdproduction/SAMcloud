"""Parametric elements of the scene model (docs/MASTERPROMPT.md sections 5 and 5.1).

Every element has typed ``params`` plus a ``param_origin`` map. Geometry is never stored;
it is generated from the parameters by the geometry module. All floats are float64.
"""

from __future__ import annotations

import math
import uuid
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from samcloud.bim.core.model.enums import (
    DoorOperation,
    ElementKind,
    JoinEnd,
    JoinType,
    ParamOrigin,
    Provenance,
    Side,
    Status,
    Tristate,
    WallFunction,
    WallJustification,
    WindowOperation,
)

LAYER_SUM_TOLERANCE = 1e-9
"""Maximum allowed difference between the layer thickness sum and the total thickness (m)."""

DEFAULT_MATERIAL = "Unspecified"
MIN_WALL_LENGTH = 1e-9
MIN_OUTLINE_AREA = 1e-12

Real = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Point2 = tuple[Real, Real]
Point3 = tuple[Real, Real, Real]


def new_id() -> str:
    """Return a new stable element id (uuid4 string)."""
    return str(uuid.uuid4())


class _Strict(BaseModel):
    """Base for all model classes: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class MaterialLayer(_Strict):
    """One layer of a layer set."""

    material: str = Field(min_length=1)
    thickness: Positive


def _check_outline(points: list[Point2], label: str) -> None:
    if len(points) < 3:
        raise ValueError(f"{label} needs at least 3 vertices, got {len(points)}")
    if points[0] == points[-1]:
        raise ValueError(f"{label} must be closed without repeating the first point")
    twice_area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        twice_area += x1 * y2 - x2 * y1
    if abs(twice_area) / 2.0 <= MIN_OUTLINE_AREA:
        raise ValueError(f"{label} has zero area")


def _resolve_layers(layers: list[MaterialLayer], thickness: float) -> list[MaterialLayer]:
    if not layers:
        return [MaterialLayer(material=DEFAULT_MATERIAL, thickness=thickness)]
    total = math.fsum(layer.thickness for layer in layers)
    if abs(total - thickness) > LAYER_SUM_TOLERANCE:
        raise ValueError(f"layer thickness sum {total!r} differs from thickness {thickness!r}")
    return layers


class ParamsBase(_Strict):
    """Base class of typed element parameters."""


class OpaqueParams(ParamsBase):
    """Free parameters of reserved element kinds that have no typed schema yet."""

    model_config = ConfigDict(extra="allow")


class WallParams(ParamsBase):
    """Parameters of a straight wall."""

    start: Point2
    end: Point2
    thickness: Positive
    height: Positive
    base_offset: Real = 0.0
    justification: WallJustification = WallJustification.CENTER
    layers: list[MaterialLayer] = Field(default_factory=list)
    function: WallFunction = WallFunction.UNKNOWN
    load_bearing: Tristate = Tristate.UNKNOWN

    @model_validator(mode="after")
    def _validate(self) -> WallParams:
        length = math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])
        if length <= MIN_WALL_LENGTH:
            raise ValueError("wall has zero length (start equals end)")
        self.layers = _resolve_layers(self.layers, self.thickness)
        return self


class WallJoinParams(ParamsBase):
    """Parameters of a join between two walls."""

    wall_a: str = Field(min_length=1)
    wall_b: str = Field(min_length=1)
    join_type: JoinType
    end_a: JoinEnd
    end_b: JoinEnd

    @model_validator(mode="after")
    def _validate(self) -> WallJoinParams:
        if self.wall_a == self.wall_b:
            raise ValueError("a wall cannot be joined to itself")
        return self


class SlabParams(ParamsBase):
    """Parameters of a floor slab."""

    outline: list[Point2]
    thickness: Positive
    top_elevation: Real
    openings: list[list[Point2]] = Field(default_factory=list)
    layers: list[MaterialLayer] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> SlabParams:
        _check_outline(self.outline, "slab outline")
        for index, opening in enumerate(self.openings):
            _check_outline(opening, f"slab opening {index}")
        self.layers = _resolve_layers(self.layers, self.thickness)
        return self


class CeilingParams(ParamsBase):
    """Parameters of a ceiling."""

    outline: list[Point2]
    elevation: Real
    plane_normal: Point3 | None = None
    plane_offset: Real | None = None
    sag: NonNegative = 0.0
    thickness: Positive = 0.1
    trim_to_walls: bool = True
    """Derive the effective outline from the walls of the storey: the ceiling ends at the inner
    wall faces; ``outline`` is the point-based envelope that stays where no wall bounds a side."""
    region_seed: Point2 | None = None
    """Plan point inside the piece this ceiling takes when the walls split the envelope (the
    largest piece when None)."""

    @model_validator(mode="after")
    def _validate(self) -> CeilingParams:
        _check_outline(self.outline, "ceiling outline")
        if (self.plane_normal is None) != (self.plane_offset is None):
            raise ValueError("plane_normal and plane_offset must be given together")
        if self.plane_normal is not None and math.hypot(*self.plane_normal) <= 1e-12:
            raise ValueError("plane_normal must not be the zero vector")
        return self


class OpeningParams(ParamsBase):
    """Parameters of a wall opening."""

    host_wall_id: str = Field(min_length=1)
    position_along_axis: NonNegative
    width: Positive
    height: Positive
    sill_height: NonNegative = 0.0
    reveal_depth: NonNegative = 0.0
    inside_side: Side = Side.LEFT


class WindowParams(ParamsBase):
    """Parameters of a window filling an opening."""

    opening_id: str = Field(min_length=1)
    operation: WindowOperation = WindowOperation.FIXED
    frame_depth: Positive | None = None
    sill_projection: NonNegative = 0.0
    sill_thickness: NonNegative = 0.0
    mullion_positions: list[NonNegative] = Field(default_factory=list)
    """Mullion bar centres in metres from the window's left edge (looking from the room); the panes lie between them."""


class DoorParams(ParamsBase):
    """Parameters of a door filling an opening."""

    opening_id: str = Field(min_length=1)
    operation: DoorOperation = DoorOperation.SINGLE_SWING_RIGHT
    swing_side: Side = Side.RIGHT
    frame_depth: Positive | None = None
    glazed: bool = False
    """The leaf carries a glass panel above a kick panel."""


class SpaceParams(ParamsBase):
    """Editable parameters of a space. Height, area and volume are derived."""

    name: str = ""
    number: str = ""
    seed: Point2 | None = None
    """A point inside the room in the local plan frame (same frame as the wall coordinates)."""
    outline: list[Point2] | None = None
    """Explicit room outline (project frame). When given the space geometry uses it directly
    instead of deriving the room from the wall network (rooms that come from a plan model)."""
    holes: list[list[Point2]] = Field(default_factory=list)
    """Holes of the explicit outline (only used together with ``outline``)."""

    @model_validator(mode="after")
    def _validate_outline(self) -> SpaceParams:
        if self.outline is not None:
            _check_outline(self.outline, "space outline")
        elif self.holes:
            raise ValueError("space holes need an outline")
        for index, hole in enumerate(self.holes):
            _check_outline(hole, f"space hole {index}")
        return self


class StairFlight(_Strict):
    """One straight flight of a stair, optionally followed by a landing in the same direction."""

    riser_count: int = Field(ge=2, le=100)
    landing_length: NonNegative = 0.0
    """Length of the landing after this flight along the stair direction (m); 0 means none."""
    landing_outline: list[Point2] | None = None
    """Optional explicit landing outline (plan, project frame); derived as a rectangle if None."""

    @model_validator(mode="after")
    def _validate(self) -> StairFlight:
        if self.landing_outline is not None:
            _check_outline(self.landing_outline, "landing outline")
        return self


class StairParams(ParamsBase):
    """Parameters of a straight stair (one or more flights in one direction with landings).

    ``start`` is the plan point at the foot of the first riser, centred across the width.
    ``direction`` is the travel direction in degrees (counter-clockwise from +x). The first
    riser rises from the base (storey elevation plus ``base_offset``). A flight of ``n`` risers
    has ``n - 1`` treads; the last riser arrives at the landing or the upper floor.
    """

    start: Point2
    direction: Real = 0.0
    base_offset: Real = 0.0
    width: Positive
    step_count: int = Field(ge=2, le=200)
    """Total number of risers over all flights."""
    riser_height: Positive
    tread_length: Positive
    flights: list[StairFlight] = Field(default_factory=list)
    nosing: NonNegative = 0.0
    landing_thickness: Positive = 0.15

    @model_validator(mode="after")
    def _validate(self) -> StairParams:
        if not self.flights:
            self.flights = [StairFlight(riser_count=self.step_count)]
        total = sum(flight.riser_count for flight in self.flights)
        if total != self.step_count:
            raise ValueError(f"flight riser counts sum to {total}, step_count is {self.step_count}")
        if self.flights[-1].landing_length > 0.0 or self.flights[-1].landing_outline is not None:
            raise ValueError("the last flight cannot be followed by a landing")
        return self


class PropertyValue(_Strict):
    """A property set value that is not geometry."""

    value: bool | float | str
    value_type: Literal["text", "number", "boolean"]
    origin: ParamOrigin = ParamOrigin.MANUAL
    pset: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> PropertyValue:
        value = self.value
        if self.value_type == "boolean":
            valid = isinstance(value, bool)
        elif self.value_type == "number":
            valid = isinstance(value, float | int) and not isinstance(value, bool)
            if valid and not math.isfinite(float(value)):
                valid = False
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(f"value {value!r} does not match value_type {self.value_type!r}")
        if self.value_type == "number":
            self.value = float(value)
        return self


class Metrics(_Strict):
    """Deviation and coverage metrics. Deviations are in meters."""

    deviation_rms: NonNegative | None = None
    deviation_p95: NonNegative | None = None
    deviation_max: NonNegative | None = None
    coverage: Unit | None = None
    point_count: int | None = Field(default=None, ge=0)
    confidence: Unit | None = None


class ElementBase(_Strict):
    """Fields shared by all elements (section 5)."""

    default_ifc_class: ClassVar[str] = ""
    default_predefined_type: ClassVar[str | None] = None

    id: str = Field(default_factory=new_id, min_length=1)
    kind: ElementKind
    ifc_class: str = ""
    predefined_type: str | None = None
    name: str = ""
    description: str = ""
    tag: str = ""
    type_id: str | None = None
    storey_id: str | None = None
    space_ids: list[str] = Field(default_factory=list)
    building_id: str | None = None
    site_id: str | None = None
    params: ParamsBase
    param_origin: dict[str, ParamOrigin] = Field(default_factory=dict)
    properties: dict[str, PropertyValue] = Field(default_factory=dict)
    provenance: Provenance = Provenance.SCRIPT
    locked: bool = False
    source_instance_ids: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    metrics: Metrics = Field(default_factory=Metrics)
    status: Status = Status.UNSURE
    notes: list[str] = Field(default_factory=list)
    ai_comments: list[str] = Field(default_factory=list)
    non_parametric: bool = False

    @model_validator(mode="after")
    def _validate_base(self) -> ElementBase:
        if not self.ifc_class:
            self.ifc_class = self.default_ifc_class
        if self.predefined_type is None:
            self.predefined_type = self.default_predefined_type
        if self.params.model_config.get("extra") != "allow":
            unknown = set(self.param_origin) - set(type(self.params).model_fields)
            if unknown:
                raise ValueError(f"param_origin has unknown parameters: {sorted(unknown)}")
        return self


class WallElement(ElementBase):
    """Straight wall."""

    default_ifc_class: ClassVar[str] = "IfcWall"
    default_predefined_type: ClassVar[str | None] = "STANDARD"
    kind: Literal[ElementKind.WALL] = ElementKind.WALL
    params: WallParams


class WallJoinElement(ElementBase):
    """Join between two walls."""

    default_ifc_class: ClassVar[str] = "IfcRelConnectsPathElements"
    kind: Literal[ElementKind.WALL_JOIN] = ElementKind.WALL_JOIN
    params: WallJoinParams


class SlabElement(ElementBase):
    """Floor slab."""

    default_ifc_class: ClassVar[str] = "IfcSlab"
    default_predefined_type: ClassVar[str | None] = "FLOOR"
    kind: Literal[ElementKind.SLAB] = ElementKind.SLAB
    params: SlabParams


class CeilingElement(ElementBase):
    """Ceiling. The exported IFC class follows ProjectSettings.ceiling_ifc_class."""

    default_ifc_class: ClassVar[str] = "IfcSlab"
    kind: Literal[ElementKind.CEILING] = ElementKind.CEILING
    params: CeilingParams


class OpeningElement(ElementBase):
    """Opening in a wall."""

    default_ifc_class: ClassVar[str] = "IfcOpeningElement"
    kind: Literal[ElementKind.OPENING] = ElementKind.OPENING
    params: OpeningParams


class WindowElement(ElementBase):
    """Window."""

    default_ifc_class: ClassVar[str] = "IfcWindow"
    kind: Literal[ElementKind.WINDOW] = ElementKind.WINDOW
    params: WindowParams


class DoorElement(ElementBase):
    """Door."""

    default_ifc_class: ClassVar[str] = "IfcDoor"
    kind: Literal[ElementKind.DOOR] = ElementKind.DOOR
    params: DoorParams


class SpaceElement(ElementBase):
    """Space (room)."""

    default_ifc_class: ClassVar[str] = "IfcSpace"
    kind: Literal[ElementKind.SPACE] = ElementKind.SPACE
    params: SpaceParams


class StairElement(ElementBase):
    """Straight stair with flights and landings (non-parametric fallback: ``non_parametric``)."""

    default_ifc_class: ClassVar[str] = "IfcStair"
    kind: Literal[ElementKind.STAIR] = ElementKind.STAIR
    params: StairParams


class FurnitureElement(ElementBase):
    """Reserved: furniture (typed parameters follow in M10)."""

    default_ifc_class: ClassVar[str] = "IfcFurniture"
    kind: Literal[ElementKind.FURNITURE] = ElementKind.FURNITURE
    params: OpaqueParams = Field(default_factory=OpaqueParams)


class MeshParams(ParamsBase):
    """Indexed triangular mesh in local scene coordinates."""

    vertices: list[Point3] = Field(default_factory=list)
    triangles: list[tuple[int, int, int]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> MeshParams:
        if not self.vertices and not self.triangles:
            return self
        if not self.vertices or not self.triangles:
            raise ValueError("mesh needs vertices and triangles")
        for triangle in self.triangles:
            if len(set(triangle)) != 3 or any(index < 0 or index >= len(self.vertices) for index in triangle):
                raise ValueError("mesh triangle has invalid vertex indices")
        return self


class TerrainParams(MeshParams):
    """Terrain surface mesh."""

    model_config = ConfigDict(extra="allow")


class TreeParams(ParamsBase):
    """Measured tree dimensions; omitted values are conservative stylised defaults."""

    trunk_xy: Point2 | None = None
    base_z: Real | None = None
    height: Positive | None = None
    crown_diameter: Positive | None = None
    trunk_diameter: Positive | None = None


class ShrubParams(ParamsBase):
    outline: list[Point2]
    base_z: Real | None = None
    height: Positive

    @model_validator(mode="after")
    def _validate(self) -> ShrubParams:
        _check_outline(self.outline, "shrub outline")
        if self.base_z is None:
            raise ValueError("shrub base_z is required")
        return self


class SiteWallParams(ParamsBase):
    footprint: list[Point2] | None = None
    axis: list[Point2] | None = None
    thickness: Positive | None = None
    base_z: Real | None = None
    top_z: Real = 1.0

    @model_validator(mode="after")
    def _validate(self) -> SiteWallParams:
        if self.footprint is None and (self.axis is None or self.thickness is None):
            raise ValueError("site wall needs a footprint or axis and thickness")
        if self.base_z is None:
            raise ValueError("site wall base_z is required")
        if self.footprint is not None:
            _check_outline(self.footprint, "site wall footprint")
        if self.axis is not None and len(self.axis) < 2:
            raise ValueError("site wall axis needs at least two points")
        if self.top_z <= self.base_z:
            raise ValueError("site wall top_z must exceed base_z")
        return self


class FenceParams(ParamsBase):
    axis: list[Point3]
    height: Positive

    @model_validator(mode="after")
    def _validate(self) -> FenceParams:
        if len(self.axis) < 2:
            raise ValueError("fence axis needs at least two points")
        return self


class PavementParams(MeshParams):
    kind: Literal["road", "sidewalk", "path"] = "road"
    thickness: Positive = 0.10


class CurbParams(ParamsBase):
    axis: list[Point3]
    width: Positive = 0.12
    height: Positive = 0.25
    curb_sweep: Literal["auto", "extrude_only"] = "auto"

    @model_validator(mode="after")
    def _validate(self) -> CurbParams:
        if len(self.axis) < 2:
            raise ValueError("curb axis needs at least two points")
        return self


class MarkingParams(ParamsBase):
    points: list[Point2 | Point3]
    closed: bool = False

    @model_validator(mode="after")
    def _validate(self) -> MarkingParams:
        if len(self.points) < (3 if self.closed else 2):
            raise ValueError("marking has too few points")
        return self


class ManholeParams(ParamsBase):
    center: Point3
    diameter: Positive
    type: str = ""


class VegetationAreaParams(ParamsBase):
    outline: list[Point2]
    base_z: Real | None = None
    height: NonNegative = 0.0

    @model_validator(mode="after")
    def _validate(self) -> VegetationAreaParams:
        _check_outline(self.outline, "vegetation outline")
        if self.base_z is None:
            raise ValueError("vegetation area base_z is required")
        return self


class BuildingParams(MeshParams):
    """Closed LoD2 envelope; face roles are ground=0, wall=1 and roof=2."""

    model_config = ConfigDict(extra="allow")

    face_roles: list[Literal[0, 1, 2]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_roles(self) -> BuildingParams:
        if self.face_roles and len(self.face_roles) != len(self.triangles):
            raise ValueError("building face_roles must match triangles")
        if not self.face_roles:
            self.face_roles = [1] * len(self.triangles)
        return self


class TerrainElement(ElementBase):
    """Terrain surface mesh."""

    default_ifc_class: ClassVar[str] = "IfcGeographicElement"
    default_predefined_type: ClassVar[str | None] = "TERRAIN"
    kind: Literal[ElementKind.TERRAIN] = ElementKind.TERRAIN
    params: TerrainParams = Field(default_factory=TerrainParams)
    non_parametric: bool = True


class BuildingElement(ElementBase):
    """Outdoor building LoD2 envelope."""

    default_ifc_class: ClassVar[str] = "IfcBuilding"
    kind: Literal[ElementKind.BUILDING] = ElementKind.BUILDING
    params: BuildingParams = Field(default_factory=BuildingParams)


class TreeElement(ElementBase):
    default_ifc_class: ClassVar[str] = "IfcGeographicElement"
    kind: Literal[ElementKind.TREE] = ElementKind.TREE
    params: TreeParams = Field(default_factory=TreeParams)

    @model_validator(mode="after")
    def _defaults(self) -> TreeElement:
        defaults: dict[str, object] = {"trunk_xy": (0.0, 0.0), "base_z": 0.0, "height": 5.0, "crown_diameter": 3.0, "trunk_diameter": 0.25}
        for name, value in defaults.items():
            if getattr(self.params, name) is None:
                setattr(self.params, name, value)
                self.param_origin.setdefault(name, ParamOrigin.ASSUMED)
        return self


class ShrubElement(ElementBase):
    default_ifc_class: ClassVar[str] = "IfcGeographicElement"
    kind: Literal[ElementKind.SHRUB] = ElementKind.SHRUB
    params: ShrubParams

    @model_validator(mode="after")
    def _set_base_z_origin(self) -> ShrubElement:
        self.param_origin.setdefault("base_z", ParamOrigin.MEASURED)
        return self


class SiteWallElement(ElementBase):
    default_ifc_class: ClassVar[str] = "IfcWall"
    kind: Literal[ElementKind.SITE_WALL] = ElementKind.SITE_WALL
    params: SiteWallParams


class FenceElement(ElementBase):
    default_ifc_class: ClassVar[str] = "IfcRailing"
    kind: Literal[ElementKind.FENCE] = ElementKind.FENCE
    params: FenceParams


class PavementElement(ElementBase):
    default_ifc_class: ClassVar[str] = "IfcCivilElement"
    kind: Literal[ElementKind.PAVEMENT] = ElementKind.PAVEMENT
    params: PavementParams


class CurbElement(ElementBase):
    default_ifc_class: ClassVar[str] = "IfcCivilElement"
    kind: Literal[ElementKind.CURB] = ElementKind.CURB
    params: CurbParams


class MarkingElement(ElementBase):
    default_ifc_class: ClassVar[str] = "IfcAnnotation"
    kind: Literal[ElementKind.MARKING] = ElementKind.MARKING
    params: MarkingParams


class ManholeElement(ElementBase):
    default_ifc_class: ClassVar[str] = "IfcDistributionChamberElement"
    default_predefined_type: ClassVar[str | None] = "MANHOLE"
    kind: Literal[ElementKind.MANHOLE] = ElementKind.MANHOLE
    params: ManholeParams


class VegetationAreaElement(ElementBase):
    default_ifc_class: ClassVar[str] = "IfcGeographicElement"
    kind: Literal[ElementKind.VEGETATION_AREA] = ElementKind.VEGETATION_AREA
    params: VegetationAreaParams

    @model_validator(mode="after")
    def _set_base_z_origin(self) -> VegetationAreaElement:
        self.param_origin.setdefault("base_z", ParamOrigin.MEASURED)
        return self


Element = Annotated[
    WallElement
    | WallJoinElement
    | SlabElement
    | CeilingElement
    | OpeningElement
    | WindowElement
    | DoorElement
    | SpaceElement
    | StairElement
    | FurnitureElement
    | TerrainElement
    | BuildingElement
    | TreeElement
    | ShrubElement
    | SiteWallElement
    | FenceElement
    | PavementElement
    | CurbElement
    | MarkingElement
    | ManholeElement
    | VegetationAreaElement,
    Field(discriminator="kind"),
]
"""Discriminated union over all element kinds."""

_ELEMENT_ADAPTER: TypeAdapter[Element] = TypeAdapter(Element)


def parse_element(data: Any) -> Element:
    """Validate a dict (or JSON-like structure) into the matching element class."""
    return _ELEMENT_ADAPTER.validate_python(data)


PARAMS_MODELS: dict[ElementKind, type[ParamsBase]] = {
    ElementKind.WALL: WallParams,
    ElementKind.WALL_JOIN: WallJoinParams,
    ElementKind.SLAB: SlabParams,
    ElementKind.CEILING: CeilingParams,
    ElementKind.OPENING: OpeningParams,
    ElementKind.WINDOW: WindowParams,
    ElementKind.DOOR: DoorParams,
    ElementKind.SPACE: SpaceParams,
    ElementKind.STAIR: StairParams,
    ElementKind.TERRAIN: TerrainParams,
    ElementKind.BUILDING: BuildingParams,
    ElementKind.TREE: TreeParams,
    ElementKind.SHRUB: ShrubParams,
    ElementKind.SITE_WALL: SiteWallParams,
    ElementKind.FENCE: FenceParams,
    ElementKind.PAVEMENT: PavementParams,
    ElementKind.CURB: CurbParams,
    ElementKind.MARKING: MarkingParams,
    ElementKind.MANHOLE: ManholeParams,
    ElementKind.VEGETATION_AREA: VegetationAreaParams,
}
