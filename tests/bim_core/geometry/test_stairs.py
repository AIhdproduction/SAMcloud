"""Stair geometry: rise, run, step solids, landings and quantities by hand (WP5.3)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from samcloud.bim.core.geometry import StairGeometry, compute_scene_geometry
from samcloud.bim.core.model import Scene, StairElement, StairFlight, StairParams, Storey


def scene_with(
    params: StairParams, height: float | None = 2.8, elevation: float = 0.0
) -> tuple[Scene, StairGeometry]:
    scene = Scene()
    storey = Storey(id="s", name="L0", elevation=elevation, height=height)
    scene.storeys[storey.id] = storey
    stair = StairElement(id="stair", storey_id="s", params=params)
    scene.add(stair)
    return scene, compute_scene_geometry(scene).stairs["stair"]


def straight(**changes: Any) -> StairParams:
    data: dict[str, Any] = {
        "start": (1.0, 1.0),
        "width": 1.0,
        "step_count": 14,
        "riser_height": 0.18,
        "tread_length": 0.28,
    }
    data.update(changes)
    return StairParams(**data)


def test_single_flight_quantities_by_hand() -> None:
    _, geometry = scene_with(straight(direction=90.0))
    assert geometry.total_rise == pytest.approx(14 * 0.18)
    assert geometry.total_rise == pytest.approx(2.52)
    assert geometry.tread_count == 13
    assert geometry.total_run == pytest.approx(13 * 0.28)
    flight = geometry.flights[0]
    assert flight.profile[0] == (0.0, 0.0)
    assert flight.profile[1] == (pytest.approx(3.64), 0.0)
    assert max(z for _, z in flight.profile) == pytest.approx(13 * 0.18)
    # Area under the steps: sum over treads k of (k * riser) * tread, k = 1..13.
    expected_area = sum(k * 0.18 * 0.28 for k in range(1, 14))
    assert flight.gross_volume == pytest.approx(expected_area * 1.0)
    quantities = geometry.quantities
    assert quantities["NumberOfRiser"] == 14
    assert quantities["NumberOfTreads"] == 13
    assert quantities["RiserHeight"] == pytest.approx(0.18)
    assert quantities["TreadLength"] == pytest.approx(0.28)
    assert quantities["Width"] == pytest.approx(1.0)
    assert geometry.issues == ()
    assert len(flight.steps) == 13
    assert flight.steps[-1].z_top == pytest.approx(13 * 0.18)


def test_direction_and_base_offset_place_the_flight() -> None:
    _, geometry = scene_with(straight(direction=90.0, base_offset=0.1), elevation=3.0)
    assert geometry.base_z == pytest.approx(3.1)
    assert geometry.direction == (pytest.approx(0.0, abs=1e-12), pytest.approx(1.0))
    assert geometry.normal == (pytest.approx(-1.0), pytest.approx(0.0, abs=1e-12))
    assert geometry.top_z == pytest.approx(3.1 + 2.52)


def test_two_flights_with_a_landing() -> None:
    params = straight(
        step_count=16,
        flights=[StairFlight(riser_count=8, landing_length=1.0), StairFlight(riser_count=8)],
    )
    _, geometry = scene_with(params, height=3.0)
    first, second = geometry.flights
    assert first.rise == pytest.approx(1.44)
    assert second.origin[2] == pytest.approx(1.44)
    assert second.s_offset == pytest.approx(7 * 0.28 + 1.0)
    assert second.origin[0] == pytest.approx(1.0 + 7 * 0.28 + 1.0)
    assert geometry.total_length == pytest.approx(14 * 0.28 + 1.0)
    assert geometry.total_run == pytest.approx(14 * 0.28)
    assert geometry.tread_count == 14
    (landing,) = geometry.landings
    assert landing.area == pytest.approx(1.0)
    assert landing.top_z == pytest.approx(1.44)
    assert landing.bottom_z == pytest.approx(1.44 - 0.15)
    assert geometry.quantities["NumberOfRiser"] == 16
    volume = first.gross_volume + second.gross_volume + landing.volume
    assert geometry.gross_volume == pytest.approx(volume)


def test_explicit_landing_outline_is_used() -> None:
    outline = [(3.0, 0.0), (5.0, 0.0), (5.0, 2.0), (3.0, 2.0)]
    params = straight(
        step_count=8,
        flights=[
            StairFlight(riser_count=4, landing_length=2.0, landing_outline=outline),
            StairFlight(riser_count=4),
        ],
    )
    _, geometry = scene_with(params)
    assert geometry.landings[0].area == pytest.approx(4.0)


def test_rise_far_from_the_storey_height_is_an_issue() -> None:
    _, geometry = scene_with(straight(), height=2.0)
    assert any("storey height" in issue for issue in geometry.issues)
    _, fine = scene_with(straight(), height=2.6)
    assert fine.issues == ()


def test_uncomfortable_stair_violates_the_step_rule() -> None:
    _, geometry = scene_with(straight(riser_height=0.15, tread_length=0.15), height=None)
    assert any("step rule" in issue for issue in geometry.issues)


def test_geometry_is_part_of_the_derived_values() -> None:
    from samcloud.bim.core.model.graph import recompute

    scene, _ = scene_with(straight())
    recompute(scene)
    entry = scene.derived["elements"]["stair"]
    assert entry["quantities"]["NumberOfRiser"] == 14.0
    assert math.isclose(entry["quantities"]["TotalRise"], 2.52)
