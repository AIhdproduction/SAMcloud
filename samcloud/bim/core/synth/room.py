"""Deterministic synthetic reference room: a parametric `Scene` without points."""

from __future__ import annotations

import math
import uuid
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from samcloud.bim.core.model import (
    Building,
    CeilingElement,
    CeilingParams,
    DoorElement,
    DoorOperation,
    DoorParams,
    Element,
    ElementKind,
    ElementType,
    JoinEnd,
    JoinType,
    MaterialLayer,
    OpeningElement,
    OpeningParams,
    ParamOrigin,
    Point2,
    ProjectSettings,
    Provenance,
    Scene,
    Side,
    Site,
    SlabElement,
    SlabParams,
    SpaceElement,
    SpaceParams,
    StairElement,
    StairFlight,
    StairParams,
    Status,
    Storey,
    WallElement,
    WallFunction,
    WallJoinElement,
    WallJoinParams,
    WallParams,
    WindowElement,
    WindowOperation,
    WindowParams,
)

_NAMESPACE = uuid.UUID("5d0c6a3e-7a7b-5c1e-9d1f-52616f6d6c6f")
"""Fixed namespace for deterministic ids (uuid5 of namespace and a stable name)."""

Pos = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNeg = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Real = Annotated[float, Field(allow_inf_nan=False)]


class RoomLayout(StrEnum):
    """Footprint layout of the synthetic room set."""

    RECTANGLE = "rectangle"
    TWO_ROOMS = "two_rooms"


class StairSpec(BaseModel):
    """A straight stair (optionally several flights with landings) placed in the room."""

    model_config = ConfigDict(extra="forbid")

    start: tuple[Real, Real] | None = None
    """Foot of the first riser, centred across the width; ``None`` places it near the north wall."""
    direction: Real = 0.0
    """Travel direction in degrees, counter-clockwise from +x."""
    width: Pos = 1.0
    step_count: int = Field(default=14, ge=2, le=200)
    riser_height: Pos = 0.18
    tread_length: Pos = 0.28
    flights: tuple[tuple[int, float], ...] = ()
    """Optional ``(riser_count, landing_length)`` per flight; the last landing length must be 0."""

    @model_validator(mode="after")
    def _validate(self) -> StairSpec:
        if self.flights:
            if sum(count for count, _ in self.flights) != self.step_count:
                raise ValueError("flight riser counts must sum to step_count")
            if self.flights[-1][1] != 0.0:
                raise ValueError("the last flight has no landing")
        return self


class RoomSpec(BaseModel):
    """Configuration of the synthetic room generator. Defaults are fully deterministic."""

    model_config = ConfigDict(extra="forbid")

    layout: RoomLayout = RoomLayout.RECTANGLE
    rectangle_size: tuple[Pos, Pos] = (5.0, 4.0)
    two_rooms_size: tuple[Pos, Pos] = (8.0, 5.0)
    storey_elevation: Real = 0.0
    storey_height: Pos = 2.8
    origin_offset: tuple[Real, Real, Real] = (2_600_000.0, 1_200_000.0, 400.0)
    exterior_thickness: Pos = 0.30
    partition_thickness: Pos = 0.12
    floor_thickness: Pos = 0.25
    ceiling_thickness: Pos = 0.20
    ceiling_sag: NonNeg = 0.0
    window_width: Pos = 1.2
    window_height: Pos = 1.4
    window_sill: NonNeg = 0.9
    window_operation: WindowOperation = WindowOperation.SIDE_HUNG_LEFT
    door_width: Pos = 0.9
    door_height: Pos = 2.1
    door_sill: NonNeg = 0.0
    door_operation: DoorOperation = DoorOperation.SINGLE_SWING_LEFT
    stair: StairSpec | None = None
    """Optional stair in the room (default none; existing scenes do not change)."""
    seed: int = 0
    """Seed for any randomness. The M1 scene generator uses none, so it only labels the run."""

    @model_validator(mode="after")
    def _validate(self) -> RoomSpec:
        if self.window_sill + self.window_height > self.storey_height:
            raise ValueError("window sill plus height exceeds the storey height")
        if self.door_sill + self.door_height > self.storey_height:
            raise ValueError("door sill plus height exceeds the storey height")
        return self

    @property
    def size(self) -> tuple[float, float]:
        """Footprint size (x, y) of the chosen layout."""
        if self.layout == RoomLayout.TWO_ROOMS:
            return self.two_rooms_size
        return self.rectangle_size


def _uid(name: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, name))


def _build_stair(
    spec: RoomSpec, stair: StairSpec, width: float, depth: float, storey_height: float | None
) -> StairElement:
    """Stair element for the room; raises ValueError if it does not fit inside the room."""
    start = stair.start if stair.start is not None else (0.7, depth - 0.9)
    flights = [
        StairFlight(riser_count=count, landing_length=landing) for count, landing in stair.flights
    ]
    params = StairParams(
        start=start,
        direction=stair.direction,
        width=stair.width,
        step_count=stair.step_count,
        riser_height=stair.riser_height,
        tread_length=stair.tread_length,
        flights=flights,
    )
    element = StairElement(id=_uid("stair"), name="Stair", tag="stair", params=params)
    margin = spec.exterior_thickness / 2.0 + 0.02
    radians = math.radians(stair.direction)
    d = (math.cos(radians), math.sin(radians))
    n = (-d[1], d[0])
    length = sum(
        (c - 1) * stair.tread_length + lan
        for c, lan in (stair.flights or ((stair.step_count, 0.0),))
    )
    for s_value, t_value in ((0, -1), (0, 1), (length, -1), (length, 1)):
        x = start[0] + d[0] * s_value + n[0] * t_value * stair.width / 2.0
        y = start[1] + d[1] * s_value + n[1] * t_value * stair.width / 2.0
        if not (margin <= x <= width - margin and margin <= y <= depth - margin):
            raise ValueError("the stair does not fit inside the room")
    return element


def build_room_scene(spec: RoomSpec | None = None) -> Scene:
    """Build the reference room scene for a spec. Same spec gives byte-identical JSON."""
    spec = spec or RoomSpec()
    width, depth = spec.size
    scene = Scene(settings=ProjectSettings(origin_offset=spec.origin_offset, crs=None))

    site = Site(id=_uid("site"), name="Site")
    building = Building(id=_uid("building"), name="Building", site_id=site.id)
    storey = Storey(
        id=_uid("storey"),
        name="Ground floor",
        elevation=spec.storey_elevation,
        height=spec.storey_height,
        building_id=building.id,
    )
    scene.sites[site.id] = site
    scene.buildings[building.id] = building
    scene.storeys[storey.id] = storey

    exterior_type = ElementType(
        id=_uid("type/wall/exterior"),
        kind=ElementKind.WALL,
        name="Exterior wall",
        ifc_type_class="IfcWallType",
        predefined_type="STANDARD",
        layers=[MaterialLayer(material="Unspecified", thickness=spec.exterior_thickness)],
    )
    interior_type = ElementType(
        id=_uid("type/wall/interior"),
        kind=ElementKind.WALL,
        name="Interior wall",
        ifc_type_class="IfcWallType",
        predefined_type="STANDARD",
        layers=[MaterialLayer(material="Unspecified", thickness=spec.partition_thickness)],
    )
    slab_type = ElementType(
        id=_uid("type/slab/floor"),
        kind=ElementKind.SLAB,
        name="Floor slab",
        ifc_type_class="IfcSlabType",
        predefined_type="FLOOR",
        layers=[MaterialLayer(material="Unspecified", thickness=spec.floor_thickness)],
    )
    window_type = ElementType(
        id=_uid("type/window"),
        kind=ElementKind.WINDOW,
        name="Window",
        ifc_type_class="IfcWindowType",
        predefined_type="WINDOW",
    )
    door_type = ElementType(
        id=_uid("type/door"),
        kind=ElementKind.DOOR,
        name="Door",
        ifc_type_class="IfcDoorType",
        predefined_type="DOOR",
    )
    for element_type in (exterior_type, interior_type, slab_type, window_type, door_type):
        scene.types[element_type.id] = element_type

    def place(element: Element, type_id: str | None = None) -> None:
        element.type_id = type_id
        element.storey_id = storey.id
        element.building_id = building.id
        element.site_id = site.id
        element.provenance = Provenance.SCRIPT
        element.status = Status.UNSURE
        element.param_origin = {
            name: ParamOrigin.TEMPLATE for name in type(element.params).model_fields
        }
        scene.add(element)

    def add_wall(key: str, name: str, start: Point2, end: Point2, exterior: bool) -> WallElement:
        wall = WallElement(
            id=_uid(f"wall/{key}"),
            name=name,
            tag=key,
            params=WallParams(
                start=start,
                end=end,
                thickness=spec.exterior_thickness if exterior else spec.partition_thickness,
                height=spec.storey_height,
                function=WallFunction.EXTERIOR if exterior else WallFunction.INTERIOR,
            ),
        )
        place(wall, exterior_type.id if exterior else interior_type.id)
        return wall

    corners: list[Point2] = [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)]
    names = ["south", "east", "north", "west"]
    walls: list[WallElement] = [
        add_wall(
            f"exterior_{name}",
            f"Exterior wall {name}",
            corners[index],
            corners[(index + 1) % 4],
            True,
        )
        for index, name in enumerate(names)
    ]

    def add_join(
        key: str,
        join_type: JoinType,
        wall_a: WallElement,
        end_a: JoinEnd,
        wall_b: WallElement,
        end_b: JoinEnd,
    ) -> None:
        join = WallJoinElement(
            id=_uid(f"join/{key}"),
            name=f"Join {key}",
            tag=key,
            params=WallJoinParams(
                wall_a=wall_a.id,
                wall_b=wall_b.id,
                join_type=join_type,
                end_a=end_a,
                end_b=end_b,
            ),
        )
        place(join)

    for index in range(4):
        add_join(
            f"L_{names[index]}_{names[(index + 1) % 4]}",
            JoinType.L,
            walls[index],
            JoinEnd.END,
            walls[(index + 1) % 4],
            JoinEnd.START,
        )

    partition: WallElement | None = None
    if spec.layout == RoomLayout.TWO_ROOMS:
        partition = add_wall(
            "partition", "Partition wall", (width / 2.0, 0.0), (width / 2.0, depth), False
        )
        add_join("T_partition_south", JoinType.T, partition, JoinEnd.START, walls[0], JoinEnd.PATH)
        add_join("T_partition_north", JoinType.T, partition, JoinEnd.END, walls[2], JoinEnd.PATH)

    half = spec.exterior_thickness / 2.0
    place(
        SlabElement(
            id=_uid("slab/floor"),
            name="Floor slab",
            tag="floor",
            params=SlabParams(
                outline=[
                    (-half, -half),
                    (width + half, -half),
                    (width + half, depth + half),
                    (-half, depth + half),
                ],
                thickness=spec.floor_thickness,
                top_elevation=spec.storey_elevation,
            ),
        ),
        slab_type.id,
    )
    ceiling_seeds: list[Point2 | None] = (
        [None]
        if spec.layout != RoomLayout.TWO_ROOMS
        else [(width / 4.0, depth / 2.0), (3.0 * width / 4.0, depth / 2.0)]
    )
    ceiling_ids: list[str] = []
    for index, ceiling_seed in enumerate(ceiling_seeds):
        # One envelope (inner face of the exterior walls); the walls cut it per room.
        ceiling_id = _uid("ceiling" if index == 0 else f"ceiling/{index + 1}")
        ceiling_ids.append(ceiling_id)
        place(
            CeilingElement(
                id=ceiling_id,
                name="Ceiling" if len(ceiling_seeds) == 1 else f"Ceiling {index + 1}",
                tag="ceiling" if index == 0 else f"ceiling/{index + 1}",
                params=CeilingParams(
                    outline=[
                        (half, half),
                        (width - half, half),
                        (width - half, depth - half),
                        (half, depth - half),
                    ],
                    elevation=spec.storey_elevation + spec.storey_height,
                    sag=spec.ceiling_sag,
                    thickness=spec.ceiling_thickness,
                    region_seed=ceiling_seed,
                ),
            )
        )

    def add_opening(
        key: str, host: WallElement, position: float, w: float, h: float, sill: float
    ) -> OpeningElement:
        length = math.dist(host.params.start, host.params.end)
        if position - w / 2.0 < 0.0 or position + w / 2.0 > length:
            raise ValueError(f"opening {key} does not fit into its host wall")
        if sill + h > host.params.height:
            raise ValueError(f"opening {key} exceeds the wall height")
        opening = OpeningElement(
            id=_uid(f"opening/{key}"),
            name=f"Opening {key}",
            tag=key,
            params=OpeningParams(
                host_wall_id=host.id,
                position_along_axis=position,
                width=w,
                height=h,
                sill_height=sill,
                inside_side=Side.LEFT,
            ),
        )
        place(opening)
        return opening

    def add_window(key: str, host: WallElement, position: float) -> None:
        opening = add_opening(
            key, host, position, spec.window_width, spec.window_height, spec.window_sill
        )
        place(
            WindowElement(
                id=_uid(f"window/{key}"),
                name=f"Window {key}",
                tag=key,
                params=WindowParams(opening_id=opening.id, operation=spec.window_operation),
            ),
            window_type.id,
        )

    def add_door(key: str, host: WallElement, position: float) -> None:
        opening = add_opening(
            key, host, position, spec.door_width, spec.door_height, spec.door_sill
        )
        place(
            DoorElement(
                id=_uid(f"door/{key}"),
                name=f"Door {key}",
                tag=key,
                params=DoorParams(
                    opening_id=opening.id,
                    operation=spec.door_operation,
                    swing_side=Side.LEFT,
                ),
            ),
            door_type.id,
        )

    south, east, north, west = walls
    if partition is None:
        add_window("window_south", south, width / 2.0)
        add_window("window_north", north, width / 2.0)
        add_door("door_west", west, depth / 2.0)
        rooms = [("Room 1", "001", (width / 2.0, depth / 2.0))]
        wall_rooms = {w.id: [0] for w in walls}
    else:
        add_window("window_south_1", south, width / 4.0)
        add_window("window_south_2", south, 3.0 * width / 4.0)
        add_door("door_west", west, depth / 2.0)
        add_door("door_partition", partition, depth / 2.0)
        rooms = [
            ("Room 1", "001", (width / 4.0, depth / 2.0)),
            ("Room 2", "002", (3.0 * width / 4.0, depth / 2.0)),
        ]
        wall_rooms = {
            south.id: [0, 1],
            north.id: [0, 1],
            west.id: [0],
            east.id: [1],
            partition.id: [0, 1],
        }

    if spec.stair is not None:
        place(_build_stair(spec, spec.stair, width, depth, storey.height))

    space_ids = [_uid(f"space/{number}") for _name, number, _seed in rooms]
    for name, number, seed in rooms:
        place(
            SpaceElement(
                id=_uid(f"space/{number}"),
                name=name,
                tag=number,
                params=SpaceParams(name=name, number=number, seed=seed),
            )
        )

    # Space membership (all ids are deterministic, so this is a pure post-pass).
    hosts: dict[str, list[str]] = {}
    for wall_id, indices in wall_rooms.items():
        hosts[wall_id] = [space_ids[i] for i in indices]
    openings: dict[str, list[str]] = {}
    for element in scene.elements.values():
        if isinstance(element, WallElement):
            element.space_ids = list(hosts[element.id])
        elif isinstance(element, OpeningElement):
            element.space_ids = list(hosts[element.params.host_wall_id])
            openings[element.id] = element.space_ids
        elif isinstance(element, CeilingElement) and len(ceiling_ids) == len(space_ids) > 1:
            element.space_ids = [space_ids[ceiling_ids.index(element.id)]]
        elif isinstance(element, SlabElement | CeilingElement):
            element.space_ids = list(space_ids)
    for element in scene.elements.values():
        if isinstance(element, WindowElement | DoorElement):
            element.space_ids = list(openings[element.params.opening_id])

    return scene
