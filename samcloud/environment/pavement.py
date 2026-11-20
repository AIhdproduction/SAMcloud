"""Pavement (road / sidewalk / path) and lawn polygons from a classified point set.

Each kind is rasterised into an occupancy grid, cleaned with morphological close/open,
vectorised and simplified. Everything is computed in the local metric frame (metres).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from rasterio import features
from rasterio.transform import Affine
from scipy import ndimage
from shapely import contains_xy
from shapely.geometry import Polygon, shape
from shapely.validation import make_valid

from samcloud.drone import contracts as c

LOGGER = logging.getLogger(__name__)

#: LAS codes that make up one pavement kind. Markings and manholes lie inside the road surface.
KIND_CODES: dict[str, tuple[int, ...]] = {
    "road": (c.ROAD, c.SEALED_TERRAIN, c.ROAD_MARKING, c.MANHOLE),
    "sidewalk": (c.SIDEWALK,),
    "path": (c.PATH,),
}
LAWN_CODES: tuple[int, ...] = (c.LOW_VEGETATION,)


@dataclass(frozen=True)
class RasterGrid:
    """Axis-aligned raster grid. Row 0 is the smallest Y, column 0 the smallest X."""

    x0: float
    y0: float
    resolution: float
    nx: int
    ny: int

    @property
    def transform(self) -> Affine:
        """Affine transform (col,row) -> (x,y) with a positive Y step."""
        return Affine(self.resolution, 0.0, self.x0, 0.0, self.resolution, self.y0)

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape (rows, cols)."""
        return (self.ny, self.nx)


def _cloud_bounds(ps: c.PointSet) -> tuple[float, float, float, float]:
    xy = ps.xyz[:, :2]
    return (float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max()))


def class_raster(
    ps: c.PointSet,
    codes,
    resolution: float = 0.10,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, RasterGrid]:
    """Occupancy raster of all points whose class is in ``codes``.

    ``bounds`` = (xmin, ymin, xmax, ymax); default is the extent of the whole point set so
    rasters of different kinds share one grid. Returns (bool mask of shape (ny, nx), grid).
    """
    if len(ps) == 0:
        raise ValueError("empty point set")
    xmin, ymin, xmax, ymax = bounds if bounds is not None else _cloud_bounds(ps)
    x0 = math.floor(xmin / resolution) * resolution
    y0 = math.floor(ymin / resolution) * resolution
    nx = int(math.floor((xmax - x0) / resolution)) + 1
    ny = int(math.floor((ymax - y0) / resolution)) + 1
    grid = RasterGrid(x0, y0, resolution, nx, ny)
    mask = np.zeros(grid.shape, dtype=bool)
    sel = np.isin(ps.classes, np.asarray(list(codes), dtype=ps.classes.dtype))
    if sel.any():
        xy = ps.xyz[sel][:, :2]
        col = np.floor((xy[:, 0] - x0) / resolution).astype(np.int64)
        row = np.floor((xy[:, 1] - y0) / resolution).astype(np.int64)
        ok = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
        mask[row[ok], col[ok]] = True
    return mask, grid


def _disk(radius_cells: int) -> np.ndarray:
    r = max(int(radius_cells), 1)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy) <= r * r


def _clean_mask(mask: np.ndarray, resolution: float, close_radius: float, open_radius: float) -> np.ndarray:
    """Close then open ``mask``; padded so that borders are not eroded."""
    rc = int(round(close_radius / resolution))
    ro = int(round(open_radius / resolution))
    pad = max(rc, ro) + 2
    padded = np.pad(mask, pad, mode="constant", constant_values=False)
    if rc > 0:
        padded = ndimage.binary_closing(padded, structure=_disk(rc), border_value=0)
    if ro > 0:
        padded = ndimage.binary_opening(padded, structure=_disk(ro), border_value=0)
    return padded[pad:-pad, pad:-pad]


def _explode(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        out: list[Polygon] = []
        for g in geom.geoms:
            out.extend(_explode(g))
        return out
    return []


def _drop_small_holes(poly: Polygon, min_area: float) -> Polygon:
    holes = [h for h in poly.interiors if Polygon(h).area >= min_area]
    return Polygon(poly.exterior, holes)


def _vectorise(mask: np.ndarray, grid: RasterGrid) -> list[Polygon]:
    if not mask.any():
        return []
    polys = []
    for geom, val in features.shapes(mask.astype(np.uint8), mask=mask, transform=grid.transform):
        if val:
            polys.append(shape(geom))
    return polys


def _kind_polygons(
    ps: c.PointSet,
    codes,
    *,
    resolution: float,
    min_area: float,
    close_radius: float,
    open_radius: float | None,
    simplify: float,
    bounds: tuple[float, float, float, float] | None,
) -> tuple[list[Polygon], list[dict]]:
    """Polygons for one class group; second result lists the dropped (too small) polygons."""
    open_r = close_radius / 2.0 if open_radius is None else open_radius
    mask, grid = class_raster(ps, codes, resolution, bounds)
    mask = _clean_mask(mask, resolution, close_radius, open_r)
    kept: list[Polygon] = []
    dropped: list[dict] = []
    for raw in _vectorise(mask, grid):
        # Occupied cells overshoot the true extent by half a cell on average: compensate.
        poly = raw.buffer(-resolution / 2.0, join_style="mitre", mitre_limit=2.0)
        for part in _explode(make_valid(poly)):
            part = _drop_small_holes(part, min_area)
            if simplify > 0:
                part = part.simplify(simplify, preserve_topology=True)
            for fixed in _explode(make_valid(part)):
                if fixed.area < min_area:
                    dropped.append({"area": float(fixed.area), "centroid": (fixed.centroid.x, fixed.centroid.y),
                                    "reason": f"area < min_area ({min_area} m2)"})
                else:
                    kept.append(fixed)
    return kept, dropped


def _points_inside(poly: Polygon, xy: np.ndarray) -> int:
    xmin, ymin, xmax, ymax = poly.bounds
    near = (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)
    sub = xy[near]
    if len(sub) == 0:
        return 0
    return int(contains_xy(poly, sub[:, 0], sub[:, 1]).sum())


def pavement_polygons(
    ps: c.PointSet,
    *,
    resolution: float = 0.10,
    min_area: float = 2.0,
    close_radius: float = 0.3,
    simplify: float = 0.03,
    open_radius: float | None = None,
    dropped: list[dict] | None = None,
) -> list[c.PavementPolygon]:
    """Clean 2D polygons per pavement kind (road, sidewalk, path).

    Road = ROAD + SEALED_TERRAIN + ROAD_MARKING + MANHOLE (so markings and manholes create no
    holes); sidewalk = SIDEWALK; path = PATH. Per kind: raster, close/open, vectorise,
    make_valid, simplify (topology preserving), drop < ``min_area`` m2. Holes smaller than
    ``min_area`` are filled. Dropped polygons are logged and, if ``dropped`` is given, appended
    to it as dicts with ``kind``, ``area``, ``centroid`` and ``reason``. Evidence.points is the
    number of kind points inside the polygon.
    """
    if len(ps) == 0:
        return []
    bounds = _cloud_bounds(ps)
    out: list[c.PavementPolygon] = []
    pid = 0
    for kind in c.PAVEMENT_KINDS:
        codes = KIND_CODES[kind]
        kept, drops = _kind_polygons(
            ps, codes, resolution=resolution, min_area=min_area, close_radius=close_radius,
            open_radius=open_radius, simplify=simplify, bounds=bounds,
        )
        for d in drops:
            d["kind"] = kind
            LOGGER.info("pavement %s polygon dropped: area %.2f m2 at (%.1f, %.1f)", kind, d["area"], *d["centroid"])
            if dropped is not None:
                dropped.append(d)
        xy = ps.xyz[np.isin(ps.classes, np.asarray(list(codes), dtype=ps.classes.dtype))][:, :2]
        for poly in kept:
            out.append(c.PavementPolygon(
                pavement_id=pid, kind=kind, polygon=poly,
                evidence=c.Evidence(points=_points_inside(poly, xy)), status="ok",
            ))
            pid += 1
    return out


def lawn_polygons(
    ps: c.PointSet,
    *,
    resolution: float = 0.10,
    min_area: float = 2.0,
    close_radius: float = 0.3,
    simplify: float = 0.03,
    open_radius: float | None = None,
    dropped: list[dict] | None = None,
) -> list[Polygon]:
    """Lawn area polygons (LOW_VEGETATION) built like :func:`pavement_polygons`."""
    if len(ps) == 0:
        return []
    kept, drops = _kind_polygons(
        ps, LAWN_CODES, resolution=resolution, min_area=min_area, close_radius=close_radius,
        open_radius=open_radius, simplify=simplify, bounds=_cloud_bounds(ps),
    )
    for d in drops:
        d["kind"] = "lawn"
        LOGGER.info("lawn polygon dropped: area %.2f m2", d["area"])
        if dropped is not None:
            dropped.append(d)
    return kept
