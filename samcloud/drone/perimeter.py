"""Area-of-interest (perimeter) from a KML polygon, used to limit the outdoor products."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from shapely import contains_xy
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import unary_union

from samcloud.drone.contracts import LocalFrame, PointSet

logger = logging.getLogger(__name__)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _ring(element: ET.Element) -> list[tuple[float, float]]:
    coords = next((c for c in element.iter() if _local(c.tag) == "coordinates"), None)
    out: list[tuple[float, float]] = []
    for token in (coords.text or "").split() if coords is not None else []:
        parts = token.split(",")
        if len(parts) >= 2:
            out.append((float(parts[0]), float(parts[1])))
    return out


def load_perimeter_kml(path) -> Polygon | MultiPolygon:
    """Parse all KML Polygons (also inside MultiGeometry; outer and inner rings) into one WGS84 (lon, lat) geometry."""
    root = ET.parse(str(path)).getroot()
    polygons: list[Polygon] = []
    for node in root.iter():
        if _local(node.tag) != "Polygon":
            continue
        outer = inner = None
        holes = []
        for child in node:
            kind = _local(child.tag)
            if kind == "outerBoundaryIs":
                outer = _ring(child)
            elif kind == "innerBoundaryIs":
                holes.append(_ring(child))
        if outer and len(outer) >= 3:
            polygons.append(Polygon(outer, [h for h in holes if len(h) >= 3]))
    if not polygons:
        raise ValueError(f"no Polygon found in KML {path}")
    merged = unary_union([p if p.is_valid else p.buffer(0) for p in polygons])
    return merged


def perimeter_to_local(poly_wgs84, frame: LocalFrame):
    """WGS84 lon/lat geometry -> EPSG:2056 -> local metric frame (XY only)."""
    from pyproj import Transformer
    from shapely.ops import transform

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{int(frame.epsg)}", always_xy=True)
    origin = np.asarray(frame.origin_lv95, float)

    def fn(x, y, z=None):
        e, n = tr.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(e) - origin[0], np.asarray(n) - origin[1]

    return transform(fn, poly_wgs84)


def clip_pointset(ps: PointSet, poly_local, buffer_m: float = 0.0) -> tuple[PointSet, np.ndarray]:
    """Keep points inside ``poly_local`` grown by ``buffer_m``; returns (subset, boolean mask)."""
    area = poly_local.buffer(buffer_m) if buffer_m else poly_local
    mask = np.asarray(contains_xy(area, ps.xyz[:, 0], ps.xyz[:, 1]), bool)
    return ps.subset(mask), mask


def find_project_perimeter(project_dir, override=None) -> Path | None:
    """Locate a perimeter KML: config override, then inputs/survey/*.kml, then inputs/*.kml."""
    project = Path(project_dir)
    if override:
        p = Path(override)
        p = p if p.is_absolute() else project / p
        return p if p.is_file() else None
    for folder in (project / "inputs" / "survey", project / "inputs"):
        found = sorted(folder.glob("*.kml")) if folder.is_dir() else []
        if found:
            return found[0]
    return None


def write_perimeter_geojson(poly_local, frame: LocalFrame, path) -> None:
    """Store the exact perimeter in LV95 as GeoJSON."""
    from shapely.affinity import translate

    lv95 = translate(poly_local, float(frame.origin_lv95[0]), float(frame.origin_lv95[1]))
    doc = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:2056"}},
           "features": [{"type": "Feature", "properties": {"name": "perimeter", "area_m2": float(lv95.area)},
                         "geometry": mapping(lv95)}]}
    Path(path).write_text(json.dumps(doc), encoding="utf-8")


def camera_positions_lv95(project_dir) -> np.ndarray:
    """Camera positions (N, 2) in LV95 E, N from ``products/gps_ref.txt`` (``name lat lon alt``, WGS84)."""
    from pyproj import Transformer

    path = Path(project_dir) / "products" / "gps_ref.txt"
    if not path.is_file():
        return np.zeros((0, 2))
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 4:
            try:
                rows.append((float(parts[-2]), float(parts[-3])))  # lon, lat
            except ValueError:
                continue
    if not rows:
        return np.zeros((0, 2))
    lonlat = np.asarray(rows, float)
    e, n = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True).transform(lonlat[:, 0], lonlat[:, 1])
    return np.column_stack([e, n])


def camera_hull_boundary(project_dir, frame: LocalFrame, buffer_m: float = 10.0) -> Polygon | None:
    """Convex hull of the camera positions in the local plan frame, buffered outward by ``buffer_m``."""
    from shapely.geometry import MultiPoint

    cams = camera_positions_lv95(project_dir)
    if len(cams) == 0:
        return None
    local = cams - np.asarray(frame.origin_lv95[:2], float)
    return MultiPoint(local).convex_hull.buffer(float(buffer_m))


def processing_boundary(project_dir, frame: LocalFrame, config: dict) -> tuple[Polygon | MultiPolygon | None, dict]:
    """Resolve the processing boundary from ``config['boundary']``; returns (local polygon or None, info)."""
    bcfg = (config or {}).get("boundary", {}) or {}
    mode = str(bcfg.get("mode", "camera_hull"))
    info: dict = {"mode": mode}
    if mode not in ("camera_hull", "perimeter", "intersection", "none"):
        raise ValueError(f"unknown boundary mode {mode!r}")
    if mode == "none":
        return None, info
    hull = per = None
    if mode in ("camera_hull", "intersection"):
        hull = camera_hull_boundary(project_dir, frame, float(bcfg.get("buffer_m", 10.0)))
        info["cameras"] = int(len(camera_positions_lv95(project_dir)))
    if mode in ("perimeter", "intersection"):
        kml = find_project_perimeter(project_dir, bcfg.get("perimeter_path"))
        if kml is not None:
            per = perimeter_to_local(load_perimeter_kml(kml), frame)
            if mode == "perimeter" and float(bcfg.get("perimeter_buffer_m", 2.0)):
                per = per.buffer(float(bcfg.get("perimeter_buffer_m", 2.0)))
            info["perimeter_source"] = str(kml)
    parts = [p for p in (hull, per) if p is not None]
    if not parts:
        info["warning"] = "no boundary source found; unclipped"
        return None, info
    poly = parts[0] if len(parts) == 1 else parts[0].intersection(parts[1])
    if poly.is_empty:
        info["warning"] = "boundary empty; unclipped"
        return None, info
    info["area_m2"] = round(float(poly.area), 2)
    return poly, info


def mask_geotiff_outside(path, poly_lv95) -> None:
    """Set raster cells outside ``poly_lv95`` (LV95 geometry) to the file's nodata (float rasters only)."""
    import rasterio
    from rasterio.features import geometry_mask

    with rasterio.open(path, "r+") as src:
        if src.nodata is None or not np.issubdtype(np.dtype(src.dtypes[0]), np.floating):
            return
        outside = geometry_mask([mapping(poly_lv95)], out_shape=src.shape, transform=src.transform, invert=False)
        for band in range(1, src.count + 1):
            data = src.read(band)
            data[outside] = src.nodata
            src.write(data, band)
