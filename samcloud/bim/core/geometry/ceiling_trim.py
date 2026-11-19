"""Ceiling pieces derived from the walls of a storey (MASTERPROMPT 5.1, 7).

A ceiling stores a point-based *envelope* (its ``outline`` parameter). The effective ceiling ends
at the inner faces of the walls that bound it, like a ceiling modelled per room in ArchiCAD:

1. every closed interior polygon of the storey (room polygon, ``derive_storey_rooms``) that the
   envelope covers by at least ``ROOM_COVER_MIN`` is taken whole, so the ceiling follows the inner
   wall faces exactly, also where the points stop a few centimetres short of the wall;
2. the rest of the envelope loses the wall footprints and is cleaned of strips narrower than
   ``MIN_WIDTH`` (point hull overshoot behind the outer wall faces);
3. a side without a wall keeps the point-based edge.

The result is a list of pieces (largest first). The ceiling takes the piece that contains its
``region_seed`` (the nearest piece when the seed lies in no piece), else the largest one. All
inputs are walls and the envelope, so a change of a wall moves the ceiling through the dependency
graph (wall -> ceiling edges of ``DependencyGraph``). Everything is in the local frame.
"""

from __future__ import annotations

from collections.abc import Sequence

from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from samcloud.bim.core.geometry.walls import WallGeometry

CLOSING = 1e-6
"""Hairline gaps between wall footprints that are closed (m), as for the room polygons."""
MIN_WIDTH = 0.30
"""Strips of the envelope narrower than this between wall faces and edge are dropped (m)."""
ROOM_COVER_MIN = 0.5
"""Share of a room polygon that the envelope must cover for the room to be taken whole."""
MIN_PIECE_AREA = 1e-4
"""Smallest piece that is kept (m^2)."""
_MITRE = 10.0


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if isinstance(geometry, MultiPolygon) or hasattr(geometry, "geoms"):
        return [p for p in geometry.geoms if isinstance(p, Polygon) and not p.is_empty]
    return []


def _opened(geometry: BaseGeometry, width: float) -> BaseGeometry:
    """Morphological opening with mitre joins: removes strips narrower than ``width``."""
    radius = width / 2.0
    return geometry.buffer(-radius, join_style=2, mitre_limit=_MITRE).buffer(
        radius, join_style=2, mitre_limit=_MITRE
    )


def ceiling_pieces(
    envelope: Polygon,
    storey_walls: Sequence[WallGeometry],
    rooms: Sequence[Polygon],
    origin: tuple[float, float],
) -> list[Polygon]:
    """Pieces of the effective ceiling (local frame, largest first, holes dropped)."""
    if storey_walls:
        union = unary_union([Polygon(w.footprint_relative_to(origin)) for w in storey_walls])
        union = union.buffer(CLOSING, join_style=2, mitre_limit=_MITRE).buffer(
            -CLOSING, join_style=2, mitre_limit=_MITRE
        )
    else:
        union = Polygon()
    taken = [
        room
        for room in rooms
        if room.area > 0.0 and envelope.intersection(room).area >= ROOM_COVER_MIN * room.area
    ]
    base = envelope.difference(union)
    if taken:
        base = base.difference(unary_union(taken))
    if storey_walls:
        base = _opened(base, MIN_WIDTH)
    region = unary_union([*taken, *_polygons(base)])
    pieces = [
        Polygon(p.exterior).simplify(1e-9, preserve_topology=True)
        for p in _polygons(region)
        if p.area >= MIN_PIECE_AREA
    ]
    pieces = [p for p in pieces if isinstance(p, Polygon) and not p.is_empty]
    pieces.sort(key=lambda p: (-p.area, p.bounds))
    return pieces


def select_piece(pieces: Sequence[Polygon], seed: tuple[float, float] | None) -> Polygon | None:
    """The piece with the seed inside (nearest when none), else the largest piece."""
    if not pieces:
        return None
    if seed is None:
        return pieces[0]
    point = Point(seed)
    for piece in pieces:
        if piece.covers(point):
            return piece
    return min(pieces, key=lambda p: (p.distance(point), -p.area))
