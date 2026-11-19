"""Stair parameters: validation, defaults and the schema (WP5.3)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from samcloud.bim.core.model import (
    ElementKind,
    StairElement,
    StairFlight,
    StairParams,
    get_param_schema,
    parse_element,
)
from samcloud.bim.core.model.params import check_param_ranges


def params(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "start": (1.0, 2.0),
        "width": 1.0,
        "step_count": 14,
        "riser_height": 0.18,
        "tread_length": 0.28,
    }
    data.update(changes)
    return data


def test_single_flight_is_derived_from_the_step_count() -> None:
    stair = StairParams(**params())
    assert [f.riser_count for f in stair.flights] == [14]
    assert stair.direction == 0.0
    assert stair.base_offset == 0.0


def test_flights_must_add_up_to_the_step_count() -> None:
    with pytest.raises(ValidationError, match="sum to 15"):
        StairParams(**params(flights=[StairFlight(riser_count=7), StairFlight(riser_count=8)]))


def test_last_flight_has_no_landing() -> None:
    with pytest.raises(ValidationError, match="last flight"):
        StairParams(
            **params(
                step_count=14,
                flights=[
                    StairFlight(riser_count=7, landing_length=1.0),
                    StairFlight(riser_count=7, landing_length=0.5),
                ],
            )
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 0.0},
        {"riser_height": -0.1},
        {"tread_length": 0.0},
        {"step_count": 1},
        {"start": (float("nan"), 0.0)},
    ],
)
def test_invalid_values_are_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        StairParams(**params(**changes))


def test_landing_outline_is_validated() -> None:
    with pytest.raises(ValidationError):
        StairFlight(riser_count=4, landing_length=1.0, landing_outline=[(0, 0), (1, 0)])


def test_stair_is_a_parametric_kind_and_round_trips() -> None:
    element = parse_element({"kind": "stair", "params": params(direction=90.0)})
    assert isinstance(element, StairElement)
    assert element.ifc_class == "IfcStair"
    again = parse_element(element.model_dump(mode="json"))
    assert again.params == element.params


def test_unknown_param_is_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_element({"kind": "stair", "params": params(color="red")})


def test_schema_lists_every_param_and_ranges_are_checked() -> None:
    schema = get_param_schema(ElementKind.STAIR)
    names = {p["name"] for p in schema["params"]}
    assert names == set(StairParams.model_fields)
    assert check_param_ranges("stair", {"riser_height": 0.5}) != []
    assert check_param_ranges("stair", {"riser_height": 0.18, "width": 1.0}) == []
