"""Stair geometry: step solids per flight, landing slabs, rise, run and quantities.

A stair is a chain of straight flights in one direction, optionally separated by landings. The
frame of a stair has its origin at the foot of the first riser (``params.start``) at the base
elevation (storey elevation plus ``base_offset``); ``s`` runs along the travel direction and ``t``
across the width (left normal, centred). A flight of ``n`` risers has ``n - 1`` treads: tread
``k`` (1-based) lies at height ``k * riser_height`` above the flight base and the last riser
arrives at the flight top (landing or upper floor), so the flight run is ``(n - 1) * tread_length``.
The flight body is the solid under the stepped profile, from the flight base up. Landing slabs lie
between flights: top at the flight top, ``landing_thickness`` thick.

All numbers are float64 in project coordinates (same frame as walls and slabs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from samcloud.bim.core.geometry._math import Vec2, left_normal, signed_area
from samcloud.bim.core.model import StairElement

STOREY_RISE_TOLERANCE = 0.35
"""Allowed difference between total rise and storey height (m); covers the upper floor slab."""
COMFORT_RANGE = (0.50, 0.80)
"""Blondel rule: 2 * riser + tread should lie in this range (m)."""


@dataclass(frozen=True)
class StepGeometry:
    """One step: the block under its tread, in flight coordinates (s along the flight)."""

    index: int
    """1-based number of the tread within the flight."""
    s_start: float
    s_end: float
    z_bottom: float
    z_top: float
    """Absolute z of the tread surface."""


@dataclass(frozen=True)
class FlightGeometry:
    """One straight flight."""

    index: int
    riser_count: int
    tread_count: int
    riser_height: float
    tread_length: float
    origin: tuple[float, float, float]
    """Foot of the first riser (centre across the width), project coordinates."""
    s_offset: float
    """Distance of the flight foot from the stair start along the direction."""
    length: float
    """Run of the flight (m)."""
    rise: float
    top_z: float
    profile: tuple[Vec2, ...]
    """Side profile polygon in (s, z) relative to the flight foot (counter-clockwise)."""
    steps: tuple[StepGeometry, ...]
    gross_volume: float

    @property
    def quantities(self) -> dict[str, float]:
        """Qto_StairFlightBaseQuantities (Length, volumes) plus the counts of the flight."""
        return {
            "Length": self.length,
            "GrossVolume": self.gross_volume,
            "NetVolume": self.gross_volume,
            "NumberOfRiser": float(self.riser_count),
            "NumberOfTreads": float(self.tread_count),
            "RiserHeight": self.riser_height,
            "TreadLength": self.tread_length,
        }


@dataclass(frozen=True)
class LandingGeometry:
    """A landing slab between two flights."""

    index: int
    outline: tuple[Vec2, ...]
    """Plan outline, counter-clockwise, project coordinates."""
    top_z: float
    bottom_z: float
    area: float

    @property
    def volume(self) -> float:
        return self.area * (self.top_z - self.bottom_z)


@dataclass(frozen=True)
class StairGeometry:
    """Derived geometry of one stair."""

    element_id: str
    storey_id: str | None
    base_z: float
    start: Vec2
    direction: Vec2
    normal: Vec2
    width: float
    step_count: int
    tread_count: int
    riser_height: float
    tread_length: float
    flights: tuple[FlightGeometry, ...]
    landings: tuple[LandingGeometry, ...]
    total_rise: float
    total_run: float
    total_length: float
    issues: tuple[str, ...] = ()

    @property
    def top_z(self) -> float:
        return self.base_z + self.total_rise

    @property
    def gross_volume(self) -> float:
        return math.fsum(
            [f.gross_volume for f in self.flights] + [landing.volume for landing in self.landings]
        )

    @property
    def quantities(self) -> dict[str, float]:
        """Stair level values: counts (Pset_StairCommon), flight quantities and Raumlot values."""
        return {
            "NumberOfRiser": float(self.step_count),
            "NumberOfTreads": float(self.tread_count),
            "RiserHeight": self.riser_height,
            "TreadLength": self.tread_length,
            "Width": self.width,
            "Length": self.total_length,
            "TotalRise": self.total_rise,
            "TotalRun": self.total_run,
            "GrossVolume": self.gross_volume,
            "NetVolume": self.gross_volume,
        }


def _flight_profile(riser_count: int, riser: float, tread: float) -> tuple[Vec2, ...]:
    """Stepped side profile of a flight in (s, z), from the foot, counter-clockwise.

    Riser ``k`` stands at ``s = (k - 1) * tread`` from ``(k - 1) * riser`` to ``k * riser``;
    tread ``k`` lies at ``z = k * riser`` over ``s`` in ``[(k - 1) * tread, k * tread]``. The last
    riser (number ``n``) stands at the end of the run and arrives at the landing or floor above,
    so the solid ends at ``s = run`` with height ``(n - 1) * riser``.
    """
    n = riser_count
    run = (n - 1) * tread
    surface: list[Vec2] = [(0.0, 0.0)]
    for k in range(1, n):
        surface.append(((k - 1) * tread, k * riser))
        surface.append((k * tread, k * riser))
    return tuple(_clean([(0.0, 0.0), (run, 0.0), *reversed(surface[1:])]))


def _clean(points: list[Vec2]) -> list[Vec2]:
    """Remove consecutive duplicates and collinear points."""
    unique: list[Vec2] = []
    for point in points:
        if not unique or math.dist(unique[-1], point) > 1e-12:
            unique.append(point)
    if len(unique) > 1 and math.dist(unique[0], unique[-1]) <= 1e-12:
        unique.pop()
    changed = True
    while changed and len(unique) > 3:
        changed = False
        for index in range(len(unique)):
            a, b, c = unique[index - 1], unique[index], unique[(index + 1) % len(unique)]
            if abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) <= 1e-12:
                del unique[index]
                changed = True
                break
    return unique


def stair_frame(params_start: Vec2, direction_deg: float) -> tuple[Vec2, Vec2]:
    """Unit travel direction and left normal for a direction angle in degrees."""
    radians = math.radians(direction_deg)
    direction = (math.cos(radians), math.sin(radians))
    return direction, left_normal(direction)


def compute_stair_geometry(
    stair: StairElement, base_elevation: float, storey_height: float | None = None
) -> StairGeometry:
    """Flights, steps, landings, rise, run and quantities of a stair.

    ``base_elevation`` is the storey elevation; the stair base adds ``params.base_offset``.
    ``storey_height`` (when known) is used for the rise check.
    """
    params = stair.params
    start = (float(params.start[0]), float(params.start[1]))
    direction, normal = stair_frame(start, params.direction)
    riser, tread, width = params.riser_height, params.tread_length, params.width
    base_z = base_elevation + params.base_offset
    issues: list[str] = []

    flights: list[FlightGeometry] = []
    landings: list[LandingGeometry] = []
    s_cursor = 0.0
    z_cursor = base_z
    for number, spec in enumerate(params.flights):
        n = spec.riser_count
        run = (n - 1) * tread
        rise = n * riser
        origin = (
            start[0] + direction[0] * s_cursor,
            start[1] + direction[1] * s_cursor,
            z_cursor,
        )
        steps = tuple(
            StepGeometry(
                k,
                (k - 1) * tread,
                k * tread,
                z_cursor,
                z_cursor + k * riser,
            )
            for k in range(1, n)
        )
        profile = _flight_profile(n, riser, tread)
        area = abs(signed_area(profile))
        flights.append(
            FlightGeometry(
                index=number,
                riser_count=n,
                tread_count=n - 1,
                riser_height=riser,
                tread_length=tread,
                origin=origin,
                s_offset=s_cursor,
                length=run,
                rise=rise,
                top_z=z_cursor + rise,
                profile=profile,
                steps=steps,
                gross_volume=area * width,
            )
        )
        s_cursor += run
        z_cursor += rise
        is_last = number == len(params.flights) - 1
        if not is_last and (spec.landing_length > 0.0 or spec.landing_outline is not None):
            outline = _landing_outline(
                spec.landing_outline, start, direction, normal, s_cursor, spec.landing_length, width
            )
            landings.append(
                LandingGeometry(
                    index=len(landings),
                    outline=outline,
                    top_z=z_cursor,
                    bottom_z=z_cursor - params.landing_thickness,
                    area=abs(signed_area(outline)),
                )
            )
            s_cursor += spec.landing_length

    total_rise = params.step_count * riser
    tread_count = sum(flight.tread_count for flight in flights)
    total_run = math.fsum(flight.length for flight in flights)
    total_length = s_cursor
    blondel = 2.0 * riser + tread
    if not COMFORT_RANGE[0] <= blondel <= COMFORT_RANGE[1]:
        issues.append(
            f"stair step rule 2*riser+tread = {blondel:.3f} m is outside "
            f"{COMFORT_RANGE[0]:.2f}..{COMFORT_RANGE[1]:.2f} m"
        )
    if storey_height is not None and abs(total_rise - storey_height) > STOREY_RISE_TOLERANCE:
        issues.append(
            f"stair rise {total_rise:.3f} m differs from the storey height {storey_height:.3f} m "
            f"by more than {STOREY_RISE_TOLERANCE:.2f} m"
        )
    return StairGeometry(
        element_id=stair.id,
        storey_id=stair.storey_id,
        base_z=base_z,
        start=start,
        direction=direction,
        normal=normal,
        width=width,
        step_count=params.step_count,
        tread_count=tread_count,
        riser_height=riser,
        tread_length=tread,
        flights=tuple(flights),
        landings=tuple(landings),
        total_rise=total_rise,
        total_run=total_run,
        total_length=total_length,
        issues=tuple(issues),
    )


def _landing_outline(
    explicit: list[tuple[float, float]] | None,
    start: Vec2,
    direction: Vec2,
    normal: Vec2,
    s0: float,
    length: float,
    width: float,
) -> tuple[Vec2, ...]:
    if explicit is not None:
        ring = tuple((float(x), float(y)) for x, y in explicit)
        return ring if signed_area(ring) > 0 else tuple(reversed(ring))
    half = width / 2.0
    corners = []
    for s, t in ((s0, -half), (s0 + length, -half), (s0 + length, half), (s0, half)):
        corners.append(
            (
                start[0] + direction[0] * s + normal[0] * t,
                start[1] + direction[1] * s + normal[1] * t,
            )
        )
    return tuple(corners)


def plan_corners(geometry: StairGeometry) -> tuple[Vec2, Vec2, Vec2, Vec2]:
    """Plan rectangle around all flights and landings (start-left to start-right order)."""
    half = geometry.width / 2.0
    s_end = geometry.total_length
    corners = []
    for s, t in ((0.0, -half), (s_end, -half), (s_end, half), (0.0, half)):
        corners.append(
            (
                geometry.start[0] + geometry.direction[0] * s + geometry.normal[0] * t,
                geometry.start[1] + geometry.direction[1] * s + geometry.normal[1] * t,
            )
        )
    return (corners[0], corners[1], corners[2], corners[3])
