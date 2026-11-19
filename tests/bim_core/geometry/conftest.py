"""Fixtures for geometry tests."""

from __future__ import annotations

from samcloud.bim.core.model import (
    CeilingElement,
    CeilingParams,
    JoinEnd,
    JoinType,
    OpeningElement,
    OpeningParams,
    Point2,
    Scene,
    SlabElement,
    SlabParams,
    SpaceElement,
    SpaceParams,
    Storey,
    WallElement,
    WallJoinElement,
    WallJoinParams,
    WallJustification,
    WallParams,
)


def make_wall(
    start: Point2,
    end: Point2,
    thickness: float = 0.2,
    height: float = 2.5,
    justification: WallJustification = WallJustification.CENTER,
    storey_id: str | None = None,
) -> WallElement:
    return WallElement(
        params=WallParams(
            start=start,
            end=end,
            thickness=thickness,
            height=height,
            justification=justification,
        ),
        storey_id=storey_id,
    )


def make_join(
    a: WallElement, end_a: JoinEnd, b: WallElement, end_b: JoinEnd, kind: JoinType
) -> WallJoinElement:
    return WallJoinElement(
        params=WallJoinParams(wall_a=a.id, wall_b=b.id, join_type=kind, end_a=end_a, end_b=end_b)
    )


def room_scene(dx: float = 0.0, dy: float = 0.0, thickness: float = 0.2) -> Scene:
    """5 m x 4 m room, center walls, L joins at all corners, floor, ceiling and one space."""
    scene = Scene()
    storey = Storey(name="EG", elevation=0.1, height=2.7)
    scene.storeys[storey.id] = storey
    corners = [(dx, dy), (dx + 5.0, dy), (dx + 5.0, dy + 4.0), (dx, dy + 4.0)]
    walls = [
        make_wall(corners[i], corners[(i + 1) % 4], thickness, 2.5, storey_id=storey.id)
        for i in range(4)
    ]
    for wall in walls:
        scene.add(wall)
    for i in range(4):
        scene.add(make_join(walls[i], JoinEnd.END, walls[(i + 1) % 4], JoinEnd.START, JoinType.L))
    scene.add(
        SlabElement(
            params=SlabParams(
                outline=[
                    (dx - 0.5, dy - 0.5),
                    (dx + 5.5, dy - 0.5),
                    (dx + 5.5, dy + 4.5),
                    (dx - 0.5, dy + 4.5),
                ],
                thickness=0.3,
                top_elevation=0.1,
            ),
            storey_id=storey.id,
        )
    )
    scene.add(
        CeilingElement(
            params=CeilingParams(
                outline=[(dx + 0.1, dy + 0.1), (dx + 4.9, dy + 0.1)]
                + [(dx + 4.9, dy + 3.9), (dx + 0.1, dy + 3.9)],
                elevation=2.9,
            ),
            storey_id=storey.id,
        )
    )
    scene.add(SpaceElement(params=SpaceParams(name="Room", number="1"), storey_id=storey.id))
    return scene


def add_opening(
    scene: Scene,
    wall: WallElement,
    position: float,
    width: float = 1.0,
    height: float = 2.0,
    sill: float = 0.0,
) -> OpeningElement:
    opening = OpeningElement(
        params=OpeningParams(
            host_wall_id=wall.id,
            position_along_axis=position,
            width=width,
            height=height,
            sill_height=sill,
        ),
        storey_id=wall.storey_id,
    )
    scene.add(opening)
    return opening
