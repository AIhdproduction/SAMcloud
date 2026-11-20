"""Hole-free, gap-free stepped solids of one furniture item.

Measured height parts (sofa seat / back / arms, bed base / headboard ...) come from the per-instance height map; the exported
footprint is the (regularised) plan outline.  :func:`tile_parts_to_footprint` re-tiles the measured level regions onto that
footprint so that

* the parts together equal the footprint (no gaps, no interior rings): cells the scan did not see - because something lay on
  them, or the outline was regularised after the measurement - take the height level of the nearest measured cell
  (interpolation);
* a small higher island that does not touch the footprint edge is an object lying on the furniture (cushion, blanket, item), not part
  of it: it takes the level around it;
* tiny fragments are merged into the level they touch most.

All coordinates are plan-local units; ``units_per_metre`` converts the metric thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import Polygon
from shapely.ops import unary_union


@dataclass(frozen=True, slots=True)
class TileSettings:
    """Metric thresholds of the part tiling."""

    enabled: bool = True
    cell_m: float = 0.025
    # a level region smaller than this is a fragment of the level around it
    min_part_area_m2: float = 0.04
    # a higher region that does not touch the footprint edge and is smaller than this is an object lying on the item
    island_max_area_m2: float = 0.35
    island_max_share: float = 0.30
    # a footprint cell counts as "touching the edge" when the footprint ends within this distance of the region
    edge_margin_m: float = 0.04
    max_parts: int = 6


def _ring(polygon: Polygon) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in list(polygon.exterior.coords)[:-1]]


def tile_parts_to_footprint(parts: list[dict[str, Any]], footprint: Polygon, units_per_metre: float = 1.0, settings: TileSettings = TileSettings()) -> list[dict[str, Any]]:
    """Parts (``{"footprint", "z_max", ...}``) re-tiled onto ``footprint``; [] when the item should stay one plain box.

    Parts keep their height (``z_max``); their polygons become exact, non-overlapping tiles of the footprint.
    """
    if not settings.enabled or not parts or footprint is None or footprint.is_empty:
        return [dict(part) for part in parts]
    upm = max(float(units_per_metre), 1e-9)
    polygons: list[tuple[float, Polygon, dict[str, Any]]] = []
    for part in parts:
        try:
            polygon = Polygon(part["footprint"])
            polygon = polygon if polygon.is_valid else polygon.buffer(0)
        except (ValueError, TypeError, KeyError):
            continue
        if isinstance(polygon, Polygon) and not polygon.is_empty:
            polygons.append((float(part["z_max"]), polygon, part))
    if len(polygons) < 2:
        return [dict(part) for part in parts]
    polygons.sort(key=lambda item: item[0])
    step = settings.cell_m * upm
    minx, miny, maxx, maxy = footprint.bounds
    nx, ny = max(1, int(np.ceil((maxx - minx) / step))), max(1, int(np.ceil((maxy - miny) / step)))
    xs = minx + (np.arange(nx) + 0.5) * step
    ys = miny + (np.arange(ny) + 0.5) * step
    gx, gy = np.meshgrid(xs, ys)
    inside = shapely.contains_xy(footprint, gx.ravel(), gy.ravel()).reshape(gx.shape)
    if int(inside.sum()) < 4:
        return [dict(part) for part in parts]
    label = np.full(inside.shape, -1, dtype=np.int32)
    for index, (_, polygon, _) in enumerate(polygons):
        label[shapely.contains_xy(polygon, gx.ravel(), gy.ravel()).reshape(gx.shape)] = index  # ascending z: the higher part wins an overlap
    label[~inside] = -1
    if not (label >= 0).any():
        return []
    # unseen cells take the level of the nearest measured cell
    missing = inside & (label < 0)
    if missing.any():
        nearest = ndimage.distance_transform_edt(label < 0, return_distances=False, return_indices=True)
        label = np.where(inside, label[tuple(nearest)], -1)
    cell_area = step * step
    min_cells = max(1, int(np.ceil(settings.min_part_area_m2 * upm * upm / cell_area)))
    count_levels = len(polygons)
    # majority filter inside the footprint (single-cell noise on the level borders)
    for _ in range(2):
        counts = np.stack([ndimage.uniform_filter(((label == index) & inside).astype(float), size=5, mode="constant") for index in range(count_levels)], axis=2)
        smoothed = np.argmax(counts, axis=2).astype(np.int32)
        label = np.where(inside, smoothed, -1)
    footprint_cells = int(inside.sum())
    margin_cells = max(1, int(round(settings.edge_margin_m * upm / step)))
    outside_ring = ndimage.binary_dilation(~inside, iterations=margin_cells) | _border(inside.shape)
    for _ in range(4):
        changed = False
        for index in range(count_levels):
            regions, total = ndimage.label(label == index, structure=np.ones((3, 3), bool))
            for number in range(1, total + 1):
                region = regions == number
                size = int(region.sum())
                around = ndimage.binary_dilation(region, structure=np.ones((3, 3), bool), iterations=2) & ~region & inside
                neighbours = label[around]
                neighbours = neighbours[neighbours != index]
                if not len(neighbours):
                    continue
                fragment = size < min_cells
                touches_edge = bool((region & outside_ring).any())
                island_limit = min(settings.island_max_area_m2 * upm * upm / cell_area, settings.island_max_share * footprint_cells)
                lying = (not touches_edge) and size <= island_limit and index > int(np.bincount(neighbours).argmax()) and _enclosed_by_one(region, label, inside)
                if fragment or lying:
                    label[region] = int(np.bincount(neighbours).argmax())
                    changed = True
        if not changed:
            break
    present = sorted({int(value) for value in np.unique(label[inside]) if value >= 0})
    if len(present) < 2 or len(present) > settings.max_parts:
        return []
    z_levels = {index: polygons[index][0] for index in present}
    if len({round(z, 3) for z in z_levels.values()}) < 2:
        return []
    # Tiles from the highest level down: each level is the smoothed region minus the higher tiles, the lowest takes the rest of the
    # footprint, so the tiles partition the footprint exactly (no gap, no overlap, no interior ring).
    tiles: list[tuple[float, Polygon]] = []
    half = step / 2.0 * 1.001  # a hair of overlap: float noise must not leave hairline gaps between neighbouring cells
    higher: Polygon | None = None
    for position, index in enumerate(reversed(present)):
        lowest = position == len(present) - 1
        if lowest:
            region = footprint if higher is None else footprint.difference(higher)
        else:
            rows, cols = np.nonzero((label == index) & inside)
            boxes = shapely.box(xs[cols] - half, ys[rows] - half, xs[cols] + half, ys[rows] + half)
            region = shapely.union_all(boxes).buffer(0).intersection(footprint).buffer(0)
            region = region.simplify(0.02 * upm, preserve_topology=True).buffer(0)
            if higher is not None:
                region = region.difference(higher)
        region = region.buffer(0)
        pieces = [piece for piece in getattr(region, "geoms", [region]) if isinstance(piece, Polygon) and piece.area >= 0.25 * settings.min_part_area_m2 * upm * upm]
        # a tile is a filled outline (an enclosed pocket is part of it: solids carry no interior rings)
        filled = [Polygon(piece.exterior) for piece in pieces]
        for piece in filled:
            tiles.append((z_levels[index], piece))
        if filled:
            higher = unary_union(filled + ([higher] if higher is not None else []))
    if len({round(z, 3) for z, _ in tiles}) < 2:
        return []
    tiles = _absorb_leftovers(tiles, footprint, upm)
    template = {key: value for key, value in polygons[0][2].items() if key not in ("footprint", "z_max")}
    output = []
    for z_top, polygon in sorted(tiles, key=lambda item: (item[0], item[1].bounds)):
        output.append({**template, "footprint": _ring(polygon), "z_max": float(z_top)})
    return output


def _absorb_leftovers(tiles: list[tuple[float, Polygon]], footprint: Polygon, upm: float) -> list[tuple[float, Polygon]]:
    """Footprint pieces no tile covers (simplification / small-piece slivers) join the neighbouring tile they share most border with."""
    covered = unary_union([polygon for _, polygon in tiles])
    leftover = footprint.difference(covered)
    pieces = [piece for piece in getattr(leftover, "geoms", [leftover]) if isinstance(piece, Polygon) and piece.area > 1e-9 * upm * upm]
    if not pieces:
        return tiles
    tiles = list(tiles)
    for piece in pieces:
        grown = piece.buffer(1e-4 * upm)
        shared = [grown.intersection(polygon.exterior).length for _, polygon in tiles]
        if max(shared) <= 0.0:
            continue
        index = int(np.argmax(shared))
        merged = unary_union([tiles[index][1], piece]).buffer(1e-4 * upm).buffer(-1e-4 * upm)
        merged = max((g for g in getattr(merged, "geoms", [merged]) if isinstance(g, Polygon)), key=lambda g: g.area, default=None)
        if merged is not None:
            tiles[index] = (tiles[index][0], Polygon(merged.exterior))
    return tiles


def _border(shape: tuple[int, int]) -> np.ndarray:
    border = np.zeros(shape, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    return border


def _enclosed_by_one(region: np.ndarray, label: np.ndarray, inside: np.ndarray) -> bool:
    """True when (nearly) everything around the region belongs to a single other level."""
    ring = ndimage.binary_dilation(region, structure=np.ones((3, 3), bool), iterations=1) & ~region & inside
    around = label[ring]
    if not len(around):
        return False
    return float(np.bincount(around).max() / len(around)) >= 0.85


def parts_hole_report(parts: list[dict[str, Any]], footprint: Polygon, tolerance: float = 0.01) -> dict[str, Any]:
    """Diagnostics: share of the footprint the parts miss, interior rings (> 1e-5 area units, hairline slits ignored), overlaps (tests / eval script)."""
    polygons = [Polygon(part["footprint"]).buffer(0) for part in parts if len(part.get("footprint") or []) >= 3]
    if not polygons:
        return {"missing_share": 1.0, "rings": 0, "overlap_share": 0.0}
    union = unary_union(polygons)
    area = max(footprint.area, 1e-12)
    upm_sq = 1.0
    rings = sum(1 for piece in getattr(union, "geoms", [union]) for ring in getattr(piece, "interiors", []) if Polygon(ring).area > 1e-5 * upm_sq)
    overlap = sum(polygon.area for polygon in polygons) - union.area
    return {
        "missing_share": float(footprint.difference(union).area / area), "outside_share": float(union.difference(footprint).area / area),
        "rings": int(rings), "overlap_share": float(max(0.0, overlap) / area), "tolerance": tolerance,
    }
