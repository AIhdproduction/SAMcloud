"""Slab and ceiling geometry (outline polygons with holes, areas, volumes, elevations).

``top_elevation`` (slab) and ``elevation`` (ceiling) are absolute z values in the local model
frame (the same frame as storey elevations). The slab occupies ``[top - thickness, top]``. The
ceiling elevation is the underside (bottom surface); the ceiling occupies ``[bottom, bottom +
thickness]``. Areas of tilted ceilings are plan (horizontal projection) areas and the volume is
plan area times the vertical thickness. ``sag`` is kept as a parameter only.

Exterior rings are returned counter-clockwise, holes clockwise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from shapely import make_valid
from shapely.geometry import Polygon
from shapely.ops import unary_union

from samcloud.bim.core.geometry._math import Vec2, oriented, ring_length, shifted, unshifted
from samcloud.bim.core.geometry.ceiling_trim import ceiling_pieces, select_piece
from samcloud.bim.core.geometry.walls import WallGeometry
from samcloud.bim.core.model import CeilingElement, SlabElement

MIN_AREA = 1e-12
PLANE_EPS = 1e-9


def local_polygon(
    exterior: tuple[Vec2, ...], holes: tuple[tuple[Vec2, ...], ...], origin: Vec2
) -> Polygon:
    """Shapely polygon from project coordinates, shifted by a local origin."""
    return Polygon(shifted(exterior, origin), [shifted(hole, origin) for hole in holes])


@dataclass(frozen=True)
class SlabGeometry:
    """Floor slab geometry in project coordinates."""

    element_id: str
    storey_id: str | None
    outline: tuple[Vec2, ...]
    holes: tuple[tuple[Vec2, ...], ...]
    thickness: float
    top_z: float
    bottom_z: float
    gross_area: float
    net_area: float
    perimeter: float
    issues: tuple[str, ...] = ()

    @property
    def gross_volume(self) -> float:
        return self.gross_area * self.thickness

    @property
    def net_volume(self) -> float:
        return self.net_area * self.thickness

    @property
    def quantities(self) -> dict[str, float]:
        """IFC4 Qto_SlabBaseQuantities names. Perimeter is that of the outer outline."""
        return {
            "Width": self.thickness,
            "Perimeter": self.perimeter,
            "GrossArea": self.gross_area,
            "NetArea": self.net_area,
            "GrossVolume": self.gross_volume,
            "NetVolume": self.net_volume,
        }

    def polygon(self, origin: Vec2 = (0.0, 0.0)) -> Polygon:
        """Net polygon (outline with holes) shifted by ``origin``."""
        return local_polygon(self.outline, self.holes, origin)


@dataclass(frozen=True)
class CeilingGeometry:
    """Ceiling geometry. ``outline_z`` is the underside z at each outline vertex."""

    element_id: str
    storey_id: str | None
    outline: tuple[Vec2, ...]
    outline_z: tuple[float, ...]
    thickness: float
    elevation: float
    plane_normal: tuple[float, float, float] | None
    plane_offset: float | None
    sag: float
    gross_area: float
    perimeter: float
    origin: Vec2 = (0.0, 0.0)
    issues: tuple[str, ...] = ()
    trimmed: bool = False
    """True when the outline was derived from the walls (``trim_to_walls``)."""
    wall_ids: tuple[str, ...] = ()
    """Walls of the storey the outline depends on (empty when not trimmed)."""
    piece_count: int = 1
    """Number of pieces the walls cut the envelope into (this ceiling takes one)."""

    @property
    def bottom_z(self) -> float:
        """Lowest underside z."""
        return min(self.outline_z)

    @property
    def top_z(self) -> float:
        """Highest upper surface z."""
        return max(self.outline_z) + self.thickness

    @property
    def net_area(self) -> float:
        return self.gross_area

    @property
    def gross_volume(self) -> float:
        return self.gross_area * self.thickness

    @property
    def net_volume(self) -> float:
        return self.net_area * self.thickness

    @property
    def quantities(self) -> dict[str, float]:
        """Same names as slabs (IFC4 Qto_SlabBaseQuantities); areas are plan areas."""
        return {
            "Width": self.thickness,
            "Perimeter": self.perimeter,
            "GrossArea": self.gross_area,
            "NetArea": self.net_area,
            "GrossVolume": self.gross_volume,
            "NetVolume": self.net_volume,
        }

    def bottom_z_at(self, x: float, y: float) -> float:
        """Underside z at a plan position (plane when given, else the elevation)."""
        if self.plane_normal is None or self.plane_offset is None:
            return self.elevation
        return _plane_z(self.plane_normal, self.plane_offset, x, y, self.origin)

    def polygon(self, origin: Vec2 = (0.0, 0.0)) -> Polygon:
        """Plan polygon shifted by ``origin``."""
        return local_polygon(self.outline, (), origin)


def _plane_z(
    normal: tuple[float, float, float], offset: float, x: float, y: float, origin: Vec2
) -> float:
    """z of the plane ``n . p = offset`` at (x, y), with the local origin subtracted first."""
    base = math.fsum((offset, -normal[0] * origin[0], -normal[1] * origin[1]))
    return math.fsum((base, -normal[0] * (x - origin[0]), -normal[1] * (y - origin[1]))) / normal[2]


def compute_slab_geometry(slab: SlabElement, origin: Vec2 = (0.0, 0.0)) -> SlabGeometry:
    """Slab outline with holes, areas, perimeter, volume inputs and z levels."""
    params = slab.params
    exterior = oriented([(x, y) for x, y in params.outline], True)
    holes = tuple(oriented([(x, y) for x, y in hole], False) for hole in params.openings)
    issues: list[str] = []
    shell = Polygon(shifted(exterior, origin))
    if not shell.is_valid:
        issues.append("slab outline is not a valid polygon (self-intersection)")
        shell = make_valid(shell)
    hole_polygons = []
    for index, hole in enumerate(holes):
        polygon = Polygon(shifted(hole, origin))
        if not polygon.is_valid:
            issues.append(f"slab opening {index} is not a valid polygon")
            polygon = make_valid(polygon)
        if not shell.buffer(1e-9).covers(polygon):
            issues.append(f"slab opening {index} is not inside the slab outline")
        hole_polygons.append(polygon)
    gross = float(shell.area)
    net = float(shell.difference(unary_union(hole_polygons)).area) if hole_polygons else gross
    return SlabGeometry(
        element_id=slab.id,
        storey_id=slab.storey_id,
        outline=exterior,
        holes=holes,
        thickness=params.thickness,
        top_z=params.top_elevation,
        bottom_z=params.top_elevation - params.thickness,
        gross_area=gross,
        net_area=net,
        perimeter=ring_length(exterior),
        issues=tuple(issues),
    )


def compute_ceiling_geometry(
    ceiling: CeilingElement,
    origin: Vec2 = (0.0, 0.0),
    walls: Sequence[WallGeometry] = (),
    rooms: Sequence[Polygon] = (),
) -> CeilingGeometry:
    """Ceiling outline, z at each vertex (plane or elevation), plan area and perimeter.

    With ``trim_to_walls`` and the walls of the storey (``walls``, ``rooms`` = their closed room
    polygons in the local frame) the outline is the piece of the envelope that ends at the inner
    wall faces (``samcloud.bim.core.geometry.ceiling_trim``); otherwise the stored outline is used.
    """
    params = ceiling.params
    outline = oriented([(x, y) for x, y in params.outline], True)
    issues: list[str] = []
    trimmed = False
    wall_ids: tuple[str, ...] = ()
    piece_count = 1
    storey_walls = [w for w in walls if w.storey_id == ceiling.storey_id]
    if params.trim_to_walls and storey_walls:
        envelope = Polygon(shifted(outline, origin))
        if not envelope.is_valid:
            envelope = make_valid(envelope).buffer(0)
        seed = None
        if params.region_seed is not None:
            seed = (params.region_seed[0] - origin[0], params.region_seed[1] - origin[1])
        pieces = ceiling_pieces(envelope, storey_walls, rooms, origin)
        piece = select_piece(pieces, seed)
        wall_ids = tuple(sorted(w.element_id for w in storey_walls))
        if piece is None:
            issues.append("ceiling outline vanishes when cut at the walls; envelope kept")
        else:
            outline = oriented([(float(x), float(y)) for x, y in piece.exterior.coords[:-1]], True)
            outline = unshifted(outline, origin)
            trimmed = True
            piece_count = len(pieces)
    normal = params.plane_normal
    offset = params.plane_offset
    if normal is not None and offset is not None:
        length = math.sqrt(math.fsum(c * c for c in normal))
        if abs(normal[2]) / length < PLANE_EPS:
            issues.append("ceiling plane is vertical; the elevation is used instead")
            normal = None
            offset = None
        elif normal[2] < 0.0:
            normal = (-normal[0], -normal[1], -normal[2])
            offset = -offset
    if normal is not None and offset is not None:
        z_values = tuple(_plane_z(normal, offset, x, y, origin) for x, y in outline)
    else:
        z_values = tuple(params.elevation for _ in outline)
    shell = Polygon(shifted(outline, origin))
    if not shell.is_valid:
        issues.append("ceiling outline is not a valid polygon (self-intersection)")
        shell = make_valid(shell)
    return CeilingGeometry(
        element_id=ceiling.id,
        storey_id=ceiling.storey_id,
        outline=outline,
        outline_z=z_values,
        thickness=params.thickness,
        elevation=params.elevation,
        plane_normal=normal,
        plane_offset=offset,
        sag=params.sag,
        gross_area=float(shell.area),
        perimeter=ring_length(outline),
        origin=origin,
        issues=tuple(issues),
        trimmed=trimmed,
        wall_ids=wall_ids,
        piece_count=piece_count,
    )
