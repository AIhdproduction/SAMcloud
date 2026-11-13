"""Reusable unit-scale CAD block definitions for outdoor and indoor plans."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import ezdxf


@dataclass(frozen=True, slots=True)
class SymbolDef:
    """Neutral unit block geometry, with an invisible ``LABEL`` attribute."""

    entities: tuple[dict[str, object], ...]
    label_visible: bool = False


def _polyline(points: list[tuple[float, float]], *, closed: bool = False) -> dict[str, object]:
    return {"type": "polyline", "points": [[x, y] for x, y in points], "closed": closed}


def _ellipse(cx: float, cy: float, rx: float, ry: float, count: int = 20) -> dict[str, object]:
    import math

    return _polyline([(cx + rx * math.cos(2.0 * math.pi * index / count), cy + ry * math.sin(2.0 * math.pi * index / count)) for index in range(count)], closed=True)


def _rounded_rectangle(left: float, bottom: float, right: float, top: float, radius: float) -> dict[str, object]:
    import math

    points: list[tuple[float, float]] = []
    for cx, cy, start in ((right - radius, top - radius, 0.0), (left + radius, top - radius, 90.0), (left + radius, bottom + radius, 180.0), (right - radius, bottom + radius, 270.0)):
        points.extend((cx + radius * math.cos(math.radians(start + offset)), cy + radius * math.sin(math.radians(start + offset))) for offset in (0.0, 30.0, 60.0))
    return _polyline(points, closed=True)


SYMBOLS: dict[str, SymbolDef] = {
    "TREE": SymbolDef((
        {"type": "circle", "center": [0.0, 0.0], "radius": 0.5},
        {"type": "circle", "center": [0.0, 0.0], "radius": 0.03},
        {"type": "line", "start": [-0.5, 0.0], "end": [0.5, 0.0]},
        {"type": "line", "start": [0.0, -0.5], "end": [0.0, 0.5]},
    )),
    "MANHOLE": SymbolDef((
        {"type": "circle", "center": [0.0, 0.0], "radius": 0.5},
        {"type": "line", "start": [-0.35, -0.35], "end": [0.35, 0.35]},
        {"type": "line", "start": [-0.35, 0.35], "end": [0.35, -0.35]},
    )),
    "POLE": SymbolDef((
        {"type": "circle", "center": [0.0, 0.0], "radius": 0.15},
        {"type": "circle", "center": [0.0, 0.0], "radius": 0.03},
    )),
    "SIGN": SymbolDef((
        {"type": "line", "start": [0.0, -0.5], "end": [0.0, 0.2]},
        _polyline([(-0.35, 0.2), (0.35, 0.2), (0.0, 0.7)], closed=True),
    )),
    "LIGHT": SymbolDef((
        {"type": "circle", "center": [0.0, 0.0], "radius": 0.2},
        {"type": "line", "start": [-0.5, 0.0], "end": [-0.25, 0.0]},
        {"type": "line", "start": [0.5, 0.0], "end": [0.25, 0.0]},
        {"type": "line", "start": [0.0, -0.5], "end": [0.0, -0.25]},
        {"type": "line", "start": [0.0, 0.5], "end": [0.0, 0.25]},
    )),
    "DOOR": SymbolDef((
        {"type": "line", "start": [0.0, 0.0], "end": [0.0, 1.0]},
        {"type": "arc", "center": [0.0, 0.0], "radius": 1.0, "start_angle": 0.0, "end_angle": 90.0},
    )),
    "WINDOW": SymbolDef((_polyline([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], closed=True), {"type": "line", "start": [0.0, 0.4], "end": [1.0, 0.4]}, {"type": "line", "start": [0.0, 0.6], "end": [1.0, 0.6]})),
    "WC": SymbolDef((_polyline([(-0.4, -0.5), (0.4, -0.5), (0.4, -0.25), (-0.4, -0.25)], closed=True), _ellipse(0.0, 0.15, 0.32, 0.5))),
    "WASHBASIN": SymbolDef((_rounded_rectangle(-0.45, -0.3, 0.45, 0.3, 0.12), _ellipse(0.0, 0.0, 0.28, 0.16))),
    "SINK": SymbolDef((_polyline([(-0.5, -0.35), (0.5, -0.35), (0.5, 0.35), (-0.5, 0.35)], closed=True), _polyline([(-0.35, -0.22), (0.35, -0.22), (0.35, 0.22), (-0.35, 0.22)], closed=True), {"type": "circle", "center": [0.0, 0.0], "radius": 0.05})),
    "BATHTUB": SymbolDef((_polyline([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], closed=True), _rounded_rectangle(-0.4, -0.38, 0.4, 0.38, 0.14), {"type": "circle", "center": [0.3, 0.0], "radius": 0.04})),
    "SHOWER": SymbolDef((_polyline([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], closed=True), {"type": "line", "start": [-0.5, -0.5], "end": [0.5, 0.5]}, {"type": "line", "start": [-0.5, 0.5], "end": [0.5, -0.5]}, {"type": "circle", "center": [0.0, 0.0], "radius": 0.05})),
    "STOVE": SymbolDef((_polyline([(-0.5, -0.35), (0.5, -0.35), (0.5, 0.35), (-0.5, 0.35)], closed=True), *tuple({"type": "circle", "center": [x, y], "radius": 0.12} for x in (-0.22, 0.22) for y in (-0.14, 0.14)))),
    "RADIATOR": SymbolDef((_polyline([(-0.5, -0.15), (0.5, -0.15), (0.5, 0.15), (-0.5, 0.15)], closed=True), *tuple({"type": "line", "start": [x, -0.15], "end": [x, 0.15]} for x in (-0.3, -0.1, 0.1, 0.3)))),
    "COLUMN": SymbolDef((_polyline([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], closed=True), {"type": "line", "start": [-0.5, -0.5], "end": [0.5, 0.5]})),
    "FURNITURE": SymbolDef((_polyline([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], closed=True),)),
}




# ---------------------------------------------------------------------------
# Metric interior-architecture catalogue.
#
# Every catalogue block is drawn in metres at its true standard size.  A block
# is inserted at one uniform scale (units per metre), never stretched.
#
# Origin convention for wall-mounted blocks (WC_WALL, WASHBASIN, BATHTUB,
# SHOWER, KITCHEN_SINK, COOKTOP, WARDROBE, BED_*, SOFA): (0, 0) lies exactly in
# the centre of the wall-mounting edge, +X runs along the wall and +Y points
# perpendicular into free space.  Snapping to a wall therefore only rotates and
# places the block.  TABLE_DINING and KITCHEN_ISLAND are free-standing and
# centred on (0, 0).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CatalogueEntry:
    """One standard size of a catalogue symbol (metres)."""

    kind: str
    width_m: float
    depth_m: float
    block: str
    layer: str
    wall_mounted: bool


def _rect(x0: float, y0: float, x1: float, y1: float) -> dict[str, object]:
    return _polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True)


def _rrect(x0: float, y0: float, x1: float, y1: float, radius: float, steps: int = 4) -> dict[str, object]:
    import math

    radius = max(0.0, min(radius, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
    points: list[tuple[float, float]] = []
    for cx, cy, start in ((x1 - radius, y1 - radius, 0.0), (x0 + radius, y1 - radius, 90.0), (x0 + radius, y0 + radius, 180.0), (x1 - radius, y0 + radius, 270.0)):
        for index in range(steps + 1):
            angle = math.radians(start + 90.0 * index / steps)
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return _polyline(points, closed=True)


def _circle(cx: float, cy: float, radius: float) -> dict[str, object]:
    return {"type": "circle", "center": [cx, cy], "radius": radius}


def _line(x0: float, y0: float, x1: float, y1: float) -> dict[str, object]:
    return {"type": "line", "start": [x0, y0], "end": [x1, y1]}


def _wc_wall(width: float, depth: float) -> tuple[dict[str, object], ...]:
    import math

    half = width / 2.0
    housing = 0.12
    centre_y = depth - 0.22
    outline: list[tuple[float, float]] = [(-half, housing), (-half, centre_y)]
    steps = 12
    outline.extend((half * math.cos(math.pi * (1.0 - index / steps)), centre_y + 0.22 * math.sin(math.pi * (1.0 - index / steps))) for index in range(1, steps))
    outline.extend([(half, centre_y), (half, housing)])
    return (
        _rect(-half, 0.0, half, housing),
        _rect(-0.08, 0.03, 0.08, 0.09),
        _polyline(outline, closed=True),
        _ellipse(0.0, centre_y, half * 0.75, 0.17, 24),
    )


def _washbasin(width: float, depth: float) -> tuple[dict[str, object], ...]:
    half = width / 2.0
    return (
        _rrect(-half, 0.0, half, depth, 0.06),
        _ellipse(0.0, depth * 0.54, width * 0.36, depth * 0.34, 24),
        _circle(0.0, 0.065, 0.015),
        _circle(0.0, depth * 0.54 + depth * 0.34 - 0.035, 0.012),
    )


def _bathtub(width: float, depth: float) -> tuple[dict[str, object], ...]:
    half = width / 2.0
    return (
        _rect(-half, 0.0, half, depth),
        _rrect(-half + 0.05, 0.05, half - 0.05, depth - 0.05, 0.16),
        _circle(half - 0.20, depth / 2.0, 0.035),
    )


def _shower(width: float, depth: float) -> tuple[dict[str, object], ...]:
    half = width / 2.0
    return (
        _rect(-half, 0.0, half, depth),
        _line(-half, 0.0, half, depth),
        _line(-half, depth, half, 0.0),
        _circle(0.0, depth / 2.0, 0.04),
    )


def _cooktop(width: float, depth: float) -> tuple[dict[str, object], ...]:
    half = width / 2.0
    return (
        _rect(-half, 0.0, half, depth),
        *tuple(_circle(sign * width * 0.25, y, 0.10) for sign in (-1.0, 1.0) for y in (depth * 0.30, depth * 0.72)),
    )


def _kitchen_sink(width: float, depth: float) -> tuple[dict[str, object], ...]:
    half = width / 2.0
    bowl_width = 0.45 if width < 0.95 else 0.42
    left = -half + 0.04
    entities: list[dict[str, object]] = [
        _rect(-half, 0.0, half, depth),
        _rrect(left, 0.11, left + bowl_width, depth - 0.05, 0.04),
        _circle(left + bowl_width / 2.0, 0.065, 0.02),
    ]
    cursor = left + bowl_width
    if width >= 0.95:
        entities.append(_rrect(cursor + 0.03, 0.11, cursor + 0.23, depth - 0.05, 0.03))
        cursor += 0.23
    for index in range(3):
        x = cursor + 0.08 + index * 0.025
        if x < half - 0.02:
            entities.append(_line(x, 0.10, x, depth - 0.06))
    return tuple(entities)


def _kitchen_island(width: float, depth: float) -> tuple[dict[str, object], ...]:
    half_w, half_d = width / 2.0, depth / 2.0
    return (_rect(-half_w, -half_d, half_w, half_d), _line(-half_w, half_d - 0.30, half_w, half_d - 0.30))


def _bed(width: float, depth: float) -> tuple[dict[str, object], ...]:
    half = width / 2.0
    count = 1 if width < 1.3 else 2
    pillow = (width - 0.20 - 0.05 * (count - 1)) / count
    entities: list[dict[str, object]] = [_rect(-half, 0.0, half, depth), _rect(-half, 0.0, half, 0.06)]
    for index in range(count):
        left = -half + 0.10 + index * (pillow + 0.05)
        entities.append(_rrect(left, 0.14, left + pillow, 0.46, 0.05))
    entities.append(_line(-half, depth * 0.45, half, depth * 0.45))
    return tuple(entities)


def _dining_table(width: float, depth: float) -> tuple[dict[str, object], ...]:
    half_w, half_d = width / 2.0, depth / 2.0
    seats = max(2, int(width // 0.60))
    pitch = width / seats
    entities: list[dict[str, object]] = [_rect(-half_w, -half_d, half_w, half_d)]
    for index in range(seats):
        x = -half_w + pitch * (index + 0.5)
        entities.append(_rrect(x - 0.21, half_d - 0.12, x + 0.21, half_d + 0.30, 0.04))
        entities.append(_rrect(x - 0.21, -half_d - 0.30, x + 0.21, -half_d + 0.12, 0.04))
    return tuple(entities)


def _sofa(width: float, depth: float) -> tuple[dict[str, object], ...]:
    half = width / 2.0
    arm = 0.15
    seats = 2 if width < 1.9 else 3
    seat_width = (width - 2.0 * arm) / seats
    entities: list[dict[str, object]] = [_rect(-half, 0.0, half, depth), _rect(-half, 0.0, half, 0.22), _rect(-half, 0.0, -half + arm, depth), _rect(half - arm, 0.0, half, depth)]
    entities.extend(_line(-half + arm + seat_width * index, 0.22, -half + arm + seat_width * index, depth) for index in range(1, seats))
    return tuple(entities)


def _wardrobe(width: float, depth: float) -> tuple[dict[str, object], ...]:
    half = width / 2.0
    doors = max(1, int(round(width / 0.50)))
    pitch = width / doors
    entities: list[dict[str, object]] = [_rect(-half, 0.0, half, depth)]
    for index in range(doors):
        left = -half + pitch * index
        if index:
            entities.append(_line(left, 0.0, left, depth))
        entities.append(_line(left, 0.0, left + pitch, depth))
    return tuple(entities)


def _catalogue_entries() -> dict[str, tuple[CatalogueEntry, ...]]:
    spec: dict[str, tuple[str, bool, tuple[tuple[float, float], ...]]] = {
        "WC_WALL": ("P-SANI", True, ((0.40, 0.56),)),
        "WASHBASIN": ("P-SANI", True, ((0.60, 0.48), (0.80, 0.50))),
        "BATHTUB": ("P-SANI", True, ((1.70, 0.75), (1.80, 0.80))),
        "SHOWER": ("P-SANI", True, ((0.90, 0.90), (1.00, 1.00), (1.20, 0.90))),
        "COOKTOP": ("A-EINB", True, ((0.58, 0.51), (0.80, 0.52))),
        "KITCHEN_SINK": ("A-EINB", True, ((0.86, 0.50), (1.00, 0.50))),
        "KITCHEN_ISLAND": ("A-EINB", False, ((1.80, 0.90), (2.40, 1.20))),
        "BED_SINGLE": ("A-FURN", True, ((0.90, 2.00), (1.00, 2.00))),
        "BED_DOUBLE": ("A-FURN", True, ((1.60, 2.00), (1.80, 2.00))),
        "TABLE_DINING": ("A-FURN", False, ((1.40, 0.80), (1.60, 0.90), (1.80, 0.90), (2.00, 1.00))),
        "SOFA": ("A-FURN", True, ((1.60, 0.90), (2.10, 0.90))),
        "WARDROBE": ("A-FURN", True, ((1.00, 0.60), (1.50, 0.60), (2.00, 0.60), (2.50, 0.60), (3.00, 0.60))),
    }
    return {
        kind: tuple(CatalogueEntry(kind, width, depth, f"{kind}_{round(width * 1000)}X{round(depth * 1000)}", layer, wall) for width, depth in sizes)
        for kind, (layer, wall, sizes) in spec.items()
    }


CATALOGUE: dict[str, tuple[CatalogueEntry, ...]] = _catalogue_entries()
_BUILDERS = {
    "WC_WALL": _wc_wall, "WASHBASIN": _washbasin, "BATHTUB": _bathtub, "SHOWER": _shower, "COOKTOP": _cooktop,
    "KITCHEN_SINK": _kitchen_sink, "KITCHEN_ISLAND": _kitchen_island, "BED_SINGLE": _bed, "BED_DOUBLE": _bed,
    "TABLE_DINING": _dining_table, "SOFA": _sofa, "WARDROBE": _wardrobe,
}


def _register_catalogue() -> None:
    for kind, entries in CATALOGUE.items():
        for entry in entries:
            SYMBOLS[entry.block] = SymbolDef(_BUILDERS[kind](entry.width_m, entry.depth_m))
        # The bare kind name is the default (first) catalogue size.  This
        # replaces the earlier unit-scale WASHBASIN / BATHTUB / SHOWER blocks.
        SYMBOLS[kind] = SYMBOLS[entries[0].block]


_register_catalogue()


def catalogue_entry_for_block(name: str) -> CatalogueEntry | None:
    """Return the catalogue entry of a size-specific block name."""
    key = str(name).upper()
    return next((entry for entries in CATALOGUE.values() for entry in entries if entry.block == key), None)


def symbol_extents(name: str) -> tuple[float, float, float, float]:
    """Return the (xmin, ymin, xmax, ymax) bounding box of a block in block units."""
    xs: list[float] = []
    ys: list[float] = []
    for entity in SYMBOLS[str(name).upper()].entities:
        kind = entity["type"]
        if kind == "line":
            for key in ("start", "end"):
                xs.append(float(entity[key][0]))  # type: ignore[index]
                ys.append(float(entity[key][1]))  # type: ignore[index]
        elif kind == "polyline":
            for point in entity["points"]:  # type: ignore[union-attr]
                xs.append(float(point[0]))
                ys.append(float(point[1]))
        elif kind in {"circle", "arc"}:
            cx, cy = entity["center"]  # type: ignore[misc]
            radius = float(entity["radius"])
            xs.extend((float(cx) - radius, float(cx) + radius))
            ys.extend((float(cy) - radius, float(cy) + radius))
    return min(xs), min(ys), max(xs), max(ys)


def add_symbol_blocks(doc: ezdxf.document.Drawing, names: Iterable[str] | None = None) -> None:
    """Create requested R2018 block definitions using ezdxf's current APIs.

    An existing block is always retained unchanged, including the legacy
    FURNITURE block supplied by the DXF writer.  Blocks created here receive a
    LABEL ATTDEF; our FURNITURE block additionally receives ``class`` and
    ``height`` ATTDEFs for writer compatibility.
    """
    requested = list(SYMBOLS) if names is None else [str(name).upper() for name in names]
    unknown = sorted(set(requested) - set(SYMBOLS))
    if unknown:
        raise KeyError(f"unknown symbol blocks: {', '.join(unknown)}")
    for name in requested:
        if name in doc.blocks:
            continue
        definition = SYMBOLS[name]
        block = doc.blocks.new(name)
        for entity in definition.entities:
            _add_entity(block, entity)
        flags = 0 if definition.label_visible else 1
        block.add_attdef("LABEL", (0.0, -0.65), "", height=0.08, dxfattribs={"flags": flags})
        if name == "FURNITURE":
            block.add_attdef("class", (0.0, -0.78), "", height=0.08)
            block.add_attdef("height", (0.0, -0.9), "", height=0.08)


def _add_entity(block: ezdxf.layouts.BlockLayout, entity: dict[str, object]) -> None:
    kind = str(entity["type"])
    if kind == "line":
        block.add_line(entity["start"], entity["end"])
    elif kind == "circle":
        block.add_circle(entity["center"], float(entity["radius"]))
    elif kind == "arc":
        block.add_arc(entity["center"], float(entity["radius"]), float(entity["start_angle"]), float(entity["end_angle"]))
    elif kind == "polyline":
        block.add_lwpolyline(entity["points"], close=bool(entity.get("closed", False)))
    else:
        raise ValueError(f"unsupported symbol entity type: {kind}")
