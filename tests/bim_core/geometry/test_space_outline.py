"""Spaces with an explicit outline (rooms from a plan model) skip the room derivation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from samcloud.bim.core.geometry import compute_scene_geometry
from samcloud.bim.core.model import Scene, SpaceElement, SpaceParams, Storey

from .conftest import room_scene


def _scene_with_outline_space(outline, holes=None) -> tuple[Scene, SpaceElement]:
    scene = Scene()
    storey = Storey(name="EG", elevation=0.5, height=2.6)
    scene.storeys[storey.id] = storey
    space = SpaceElement(
        params=SpaceParams(name="Living", outline=outline, holes=holes or []), storey_id=storey.id
    )
    scene.add(space)
    return scene, space


def test_outline_space_has_geometry_without_any_wall() -> None:
    scene, space = _scene_with_outline_space([(0, 0), (4, 0), (4, 3), (0, 3)])

    geometry = compute_scene_geometry(scene)

    space_geometry = geometry.spaces[space.id]
    assert space_geometry.matched_by == "outline"
    assert space_geometry.net_area == pytest.approx(12.0)
    assert space_geometry.perimeter == pytest.approx(14.0)
    assert space_geometry.floor_z == pytest.approx(0.5)
    assert space_geometry.height == pytest.approx(2.6)  # storey height, no ceiling in the scene
    assert space.id not in geometry.issues or all("ambiguous" not in i for i in geometry.issues[space.id])


def test_outline_with_hole_reduces_net_area_only() -> None:
    scene, space = _scene_with_outline_space(
        [(0, 0), (4, 0), (4, 3), (0, 3)], holes=[[(1, 1), (1, 1.5), (1.5, 1.5), (1.5, 1)]]
    )

    space_geometry = compute_scene_geometry(scene).spaces[space.id]

    assert space_geometry.gross_area == pytest.approx(12.0)
    assert space_geometry.net_area == pytest.approx(12.0 - 0.25)
    assert len(space_geometry.holes) == 1


def test_outline_wins_over_the_wall_network() -> None:
    scene = room_scene()  # closed 5 x 4 room; its space is derived from the walls
    storey_id = next(iter(scene.storeys))
    # a second space with an explicit outline that does not match any wall derived polygon
    other = SpaceElement(
        params=SpaceParams(name="Elsewhere", outline=[(20, 20), (22, 20), (22, 21), (20, 21)]),
        storey_id=storey_id,
    )
    scene.add(other)

    geometry = compute_scene_geometry(scene)

    assert geometry.spaces[other.id].net_area == pytest.approx(2.0)
    derived = [g for key, g in geometry.spaces.items() if key != other.id]
    assert len(derived) == 1 and derived[0].matched_by == "single_room"
    assert derived[0].net_area == pytest.approx(4.8 * 3.8)


def test_holes_without_outline_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SpaceParams(holes=[[(0, 0), (1, 0), (1, 1)]])


def test_degenerate_outline_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SpaceParams(outline=[(0, 0), (1, 0), (2, 0)])
