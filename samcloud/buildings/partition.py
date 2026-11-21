"""Split building footprints along roof-plane boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point as shapely_point, Polygon
from shapely.ops import nearest_points, polygonize, split, unary_union

from samcloud.drone.contracts import RoofPlane
from samcloud.ortho.vector.regularize import dominant_direction


@dataclass
class Partition:
    """Footprint cells and directed shared-edge adjacency."""

    cells: list[Polygon]
    adjacency: list[tuple[int, int, float]]
    footprint: Polygon | None = None  # reduced footprint when roofless parts were removed
    superstructures: list = field(default_factory=list)  # generalised chimneys/dormers/vents
    feature_geometry: object = None  # union of generalised superstructure regions


def _height(plane: RoofPlane, x: float, y: float) -> float:
    if abs(float(plane.normal[2])) < 1e-10:
        return float("nan")
    return -float(plane.normal[0] * x + plane.normal[1] * y + plane.d) / float(plane.normal[2])


def _lines(geometry: object) -> list[LineString]:
    if isinstance(geometry, LineString):
        return [geometry] if geometry.length > 1e-7 else []
    if isinstance(geometry, MultiLineString):
        return [line for line in geometry.geoms if line.length > 1e-7]
    if isinstance(geometry, GeometryCollection):
        answer: list[LineString] = []
        for item in geometry.geoms:
            answer.extend(_lines(item))
        return answer
    return []


def _extend(line: LineString, footprint: Polygon) -> LineString | None:
    coordinates = np.asarray(line.coords)
    if len(coordinates) < 2:
        return None
    vector = coordinates[-1] - coordinates[0]
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        return None
    vector /= length
    center = coordinates.mean(axis=0)
    extent = max(footprint.bounds[2] - footprint.bounds[0], footprint.bounds[3] - footprint.bounds[1]) * 3.0 + 1.0
    extended = LineString([tuple(center - vector * extent), tuple(center + vector * extent)])
    clipped = extended.intersection(footprint)
    candidates = _lines(clipped)
    return max(candidates, key=lambda item: item.length) if candidates else None


def _snap_direction(line: LineString, direction: float, tolerance: float = 10.0) -> LineString:
    points = np.asarray(line.coords)
    vector = points[-1] - points[0]
    angle = math.atan2(float(vector[1]), float(vector[0]))
    candidates = [direction + step * math.pi / 2.0 for step in range(4)]
    target = min(candidates, key=lambda item: abs((angle - item + math.pi / 2.0) % math.pi - math.pi / 2.0))
    error = abs((angle - target + math.pi / 2.0) % math.pi - math.pi / 2.0)
    if error > math.radians(tolerance):
        return line
    center = points.mean(axis=0)
    half = float(np.linalg.norm(vector)) * 0.5
    unit = np.array((math.cos(target), math.sin(target)))
    return LineString([tuple(center - unit * half), tuple(center + unit * half)])


def plane_lines(roof_planes: list[RoofPlane], footprint: Polygon) -> list[LineString]:
    """Return clipped ridge and height-step lines between adjacent roof planes."""
    if footprint.is_empty:
        return []
    result: list[LineString] = []
    direction = dominant_direction([footprint])
    for index, first in enumerate(roof_planes):
        if first.outline is None or first.outline.is_empty:
            continue
        for second in roof_planes[index + 1:]:
            if second.outline is None or second.outline.is_empty or first.outline.distance(second.outline) > 0.5:
                continue
            cross = np.cross(first.normal, second.normal)
            # Plane fits of two horizontal stepped roofs differ very slightly
            # because of measurement noise.  Treating them as intersecting
            # creates a line far outside the footprint and skips the outline
            # boundary that marks the actual height step.
            if np.linalg.norm(cross[:2]) > 1e-3:
                # One least-norm point on both 3-D planes, projected to XY.
                matrix = np.vstack((first.normal, second.normal))
                if np.linalg.matrix_rank(matrix) == 2:
                    point3d = np.linalg.lstsq(matrix, -np.array([first.d, second.d]), rcond=None)[0]
                    unit = cross[:2] / np.linalg.norm(cross[:2])
                    # The least-norm solution is the line point nearest the origin,
                    # which can be far outside the footprint in a georeferenced
                    # frame.  Anchor the line next to the footprint instead.
                    anchor = np.asarray(footprint.centroid.coords[0], dtype=float)
                    point = point3d[:2] + unit * float((anchor - point3d[:2]) @ unit)
                    raw = LineString([tuple(point - unit), tuple(point + unit)])
                    clipped = _extend(_snap_direction(raw, direction), footprint)
                    if clipped is not None:
                        result.append(clipped)
                    continue
            shared = first.outline.boundary.intersection(second.outline.boundary)
            for boundary in _lines(shared):
                midpoint = boundary.interpolate(0.5, normalized=True)
                jump = abs(_height(first, midpoint.x, midpoint.y) - _height(second, midpoint.x, midpoint.y))
                if jump > 0.5:
                    clipped = _extend(_snap_direction(boundary, direction), footprint)
                    if clipped is not None:
                        result.append(clipped)
            if shared.is_empty:
                first_point, second_point = nearest_points(first.outline, second.outline)
                delta = np.array((second_point.x - first_point.x, second_point.y - first_point.y))
                length = float(np.linalg.norm(delta))
                if length > 1e-8:
                    midpoint = (np.array((first_point.x, first_point.y)) + np.array((second_point.x, second_point.y))) * 0.5
                    tangent = np.array((-delta[1], delta[0])) / length
                    raw = LineString([tuple(midpoint - tangent), tuple(midpoint + tangent)])
                    jump = abs(_height(first, midpoint[0], midpoint[1]) - _height(second, midpoint[0], midpoint[1]))
                    if jump > 0.5:
                        clipped = _extend(_snap_direction(raw, direction), footprint)
                        if clipped is not None:
                            result.append(clipped)
    unique: list[LineString] = []
    for line in result:
        if not any(line.hausdorff_distance(old) < 1e-5 for old in unique):
            unique.append(line)
    return unique


def _adjacency(cells: list[Polygon]) -> list[tuple[int, int, float]]:
    result: list[tuple[int, int, float]] = []
    for first, cell in enumerate(cells):
        for second in range(first + 1, len(cells)):
            shared = cell.boundary.intersection(cells[second].boundary).length
            if shared > 1e-8:
                result.append((first, second, float(shared)))
                result.append((second, first, float(shared)))
    return result


def partition_footprint(footprint: Polygon, lines: list[LineString], *, min_cell_area: float = 0.5) -> Partition:
    """Polygonize an extended roof-line arrangement and return cells with adjacency."""
    if footprint.is_empty:
        return Partition([], [])
    extended = [_extend(line, footprint) for line in lines]
    # Let cutters cross the footprint boundary very slightly.  Clipping a line
    # to a noisy facade-derived footprint can leave its endpoints a few microns
    # short of the boundary, which prevents polygonize from creating cells.
    noded: list[LineString] = []
    for line in extended:
        if line is None:
            continue
        coordinates = np.asarray(line.coords, dtype=float)
        vector = coordinates[-1] - coordinates[0]
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            continue
        unit = vector / length
        noded.append(LineString([tuple(coordinates[0] - unit * 1e-4), tuple(coordinates[-1] + unit * 1e-4)]))
    geometry = unary_union([footprint.boundary, *noded])
    cells = [poly.intersection(footprint) for poly in polygonize(geometry)]
    cells = [cell for cell in cells if isinstance(cell, Polygon) and cell.area > 1e-8]
    if not cells:
        cells = [footprint]
    changed = True
    while changed:
        changed = False
        for index, cell in enumerate(cells):
            if cell.area >= min_cell_area or len(cells) == 1:
                continue
            candidates = [(cell.boundary.intersection(other.boundary).length, other_index) for other_index, other in enumerate(cells) if other_index != index]
            length, neighbour = max(candidates, default=(0.0, -1))
            if length <= 1e-8:
                continue
            merged = cell.union(cells[neighbour])
            if not isinstance(merged, Polygon):
                continue
            cells[neighbour] = merged
            cells.pop(index)
            changed = True
            break
    cells.sort(key=lambda item: (item.centroid.x, item.centroid.y))
    return Partition(cells, _adjacency(cells))


def _label_image(raster_xyz: np.ndarray, planes: list[RoofPlane], resolution: float, tolerance: float, margin: float, low_xyz: np.ndarray | None = None, min_none_area_m2: float = 0.8, footprint: Polygon | None = None, min_feature_area_m2: float = 6.0, min_step_m: float = 0.35):
    """Rasterize DSM cells into plane labels; return (labels, x0, y_top, feature_mask, features)."""
    from scipy import ndimage

    xyz = np.asarray(raster_xyz, dtype=float)
    residual = np.column_stack([np.abs(plane.distance(xyz)) for plane in planes])
    best = residual.argmin(axis=1)
    best[residual.min(axis=1) > max(tolerance * 2.0, 0.12)] = -1
    low = np.asarray(low_xyz, dtype=float) if low_xyz is not None and len(low_xyz) else np.empty((0, 3))
    if len(low):
        xyz = np.vstack((xyz, low))
        best = np.concatenate((best, np.full(len(low), len(planes), dtype=int)))
    x0, x1 = float(xyz[:, 0].min() - margin), float(xyz[:, 0].max() + margin)
    y0, y1 = float(xyz[:, 1].min() - margin), float(xyz[:, 1].max() + margin)
    columns = int(math.ceil((x1 - x0) / resolution)) + 1
    rows = int(math.ceil((y1 - y0) / resolution)) + 1
    image = np.full((rows, columns), -1, dtype=int)
    col = np.clip(np.rint((xyz[:, 0] - x0) / resolution).astype(int), 0, columns - 1)
    row = np.clip(np.rint((y1 - xyz[:, 1]) / resolution).astype(int), 0, rows - 1)
    image[row, col] = best
    none = len(planes)
    zimg = np.full(image.shape, np.nan)
    zimg[row, col] = xyz[:, 2]
    grid_x = x0 + np.arange(columns) * resolution
    grid_y = y1 - np.arange(rows) * resolution
    X, Y = np.meshgrid(grid_x, grid_y)
    interior = np.ones(image.shape, dtype=bool)
    if footprint is not None and not footprint.is_empty:
        from rasterio import features as rio_features
        from rasterio.transform import Affine as RioAffine

        inner = footprint.buffer(-0.5)
        if not inner.is_empty:
            interior = rio_features.rasterize(
                [(inner, 1)], out_shape=image.shape, transform=RioAffine(resolution, 0.0, x0 - resolution / 2.0, 0.0, -resolution, y1 + resolution / 2.0)
            ).astype(bool)
    pixel_area = resolution * resolution
    feature_mask = np.zeros(image.shape, dtype=bool)
    features: list[dict] = []

    def plane_z(plane: RoofPlane, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return -(plane.normal[0] * xs + plane.normal[1] * ys + plane.d) / plane.normal[2]

    # Large coherent regions that fit no roof plane (shrubs, clutter, ground) are
    # not roof; they leave the footprint, small misfits are absorbed by neighbours.
    data = np.zeros(image.shape, dtype=bool)
    data[row, col] = True
    poor, poor_count = ndimage.label((image == -1) & data)
    if poor_count:
        sizes = np.bincount(poor.ravel())
        for component in range(1, poor_count + 1):
            members = poor == component
            area = sizes[component] * pixel_area
            touches = bool((members & ~interior).any())
            if area >= 0.02 and area < min_feature_area_m2 and (not touches or area < min_none_area_m2):
                # a protruding blob (chimney, vent, antenna) inside the roof: generalised
                feature_mask |= members
                features.append({"kind": "protrusion", "area_m2": float(area), "x": float(X[members].mean()), "y": float(Y[members].mean()), "height_m": None})
            elif sizes[component] >= max(1, int(round(float(min_none_area_m2) / pixel_area))):
                image[members] = none

    def fill(values: np.ndarray) -> np.ndarray:
        missing = values < 0
        if not missing.any() or missing.all():
            return values
        index = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
        return values[index[0], index[1]]

    image = fill(image)
    count = len(planes) + 1
    votes = np.stack([ndimage.uniform_filter((image == label).astype(float), size=3, mode="nearest") for label in range(count)])
    image = votes.argmax(axis=0)
    minimum = max(1, int(round(1.0 / (resolution * resolution))))
    for _round in range(3):
        cleaned = image.copy()
        for label in range(count):
            components, total = ndimage.label(image == label)
            if total == 0:
                continue
            sizes = np.bincount(components.ravel())
            small = np.isin(components, np.flatnonzero(sizes < minimum)) & (components > 0)
            cleaned[small] = -1
        if (cleaned < 0).all():
            break
        image = fill(cleaned)
    # Generalise small or low-relief regions into the dominant surrounding plane.
    structure = np.ones((3, 3), dtype=bool)
    for _pass in range(3):
        changed = False
        for label in range(none):
            components, total = ndimage.label(image == label)
            for component in range(1, total + 1):
                members = components == component
                area = float(members.sum()) * pixel_area
                ring = ndimage.binary_dilation(members, structure=structure, iterations=2) & ~members
                around = image[ring]
                around = around[(around != label) & (around != none)]
                if not len(around):
                    continue
                host = int(np.bincount(around).argmax())
                small = area < min_feature_area_m2 and (area < 0.5 * min_feature_area_m2 or not (members & ~interior).any())
                signed = zimg[members] - plane_z(planes[host], X[members], Y[members])
                deviation = np.abs(signed[np.isfinite(signed)])
                low = len(deviation) > 0 and float(deviation.mean()) < 0.15 and float(np.percentile(deviation, 95)) < min_step_m
                if not (small or low):
                    continue
                image[members] = host
                changed = True
                if small:
                    feature_mask |= members
                    features.append({"kind": "small_region", "area_m2": area, "x": float(X[members].mean()), "y": float(Y[members].mean()),
                                     "height_m": float(deviation.max()) if len(deviation) else None})
        if not changed:
            break
    if True:
        roof = image != none
        # Opening removes thin roof bridges and corner contacts (non-manifold
        # pinches after dropping roofless parts); holes are filled with roof.
        roof = ndimage.binary_opening(roof, structure=np.ones((3, 3), dtype=bool))
        roof = ndimage.binary_fill_holes(roof)
        components, total = ndimage.label(roof)
        if total > 1:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            roof = components == int(sizes.argmax())
        valid = roof & (image != none)
        if valid.any():
            nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
            image = np.where(roof, image[nearest[0], nearest[1]], none)
        else:
            image = np.full_like(image, none)
    return image, x0, y1, feature_mask, features


def partition_from_raster(
    footprint: Polygon,
    raster_xyz: np.ndarray,
    planes: list[RoofPlane],
    *,
    resolution: float,
    tolerance: float = 0.08,
    simplify_m: float = 0.25,
    min_cell_area: float = 0.5,
    outline_simplify_m: float = 0.25,
    step_simplify_m: float = 0.1,
    merge_jump_m: float = 0.3,
    merge_area_m2: float = 8.0,
    min_feature_area_m2: float = 6.0,
    min_step_m: float = 0.35,
    low_xyz: np.ndarray | None = None,
    ridge_snap_m: float = 0.35,
) -> tuple[Partition, np.ndarray]:
    """Partition a footprint by DSM-cell plane labels; return cells and 1-based plane labels.

    Every DSM cell is assigned to the roof plane that fits it best, cleaned with a
    majority filter and a minimum region size, polygonized and simplified as one
    coverage so that shared region boundaries stay identical.  Heights of a cell
    therefore always come from its labelled plane.
    """
    import shapely
    from rasterio import features
    from rasterio.transform import Affine

    if footprint.is_empty or not planes or len(raster_xyz) == 0:
        return Partition([footprint], []), np.zeros(1, dtype=int)
    image, x0, y_top, feature_mask, feature_list = _label_image(
        raster_xyz, planes, resolution, tolerance, 1.0, low_xyz, footprint=footprint, min_feature_area_m2=min_feature_area_m2, min_step_m=min_step_m
    )
    transform = Affine(resolution, 0.0, x0 - resolution / 2.0, 0.0, -resolution, y_top + resolution / 2.0)
    feature_geometry = None
    if feature_mask.any():
        pieces_ = [shapely.geometry.shape(g) for g, v in features.shapes(feature_mask.astype(np.uint8), mask=feature_mask, transform=transform, connectivity=8) if int(v) == 1]
        feature_geometry = unary_union(pieces_).intersection(footprint) if pieces_ else None
    regions: list[tuple[Polygon, int]] = []
    for label in np.unique(image):
        mask = (image == label).astype(np.uint8)
        for geometry, value in features.shapes(mask, mask=mask.astype(bool), transform=transform, connectivity=4):
            if int(value) == 1:
                regions.append((shapely.geometry.shape(geometry), int(label)))
    polygons = [item[0] for item in regions]
    simplified = list(shapely.coverage_simplify(np.asarray(polygons, dtype=object), step_simplify_m)) if step_simplify_m > 0 else polygons
    none_label = len(planes) + 1
    roof_ids = [index for index, item in enumerate(regions) if item[1] + 1 < none_label]
    none_ids = [index for index, item in enumerate(regions) if item[1] + 1 == none_label]
    reduced = footprint
    if none_ids:
        # Roofless parts (terrace, courtyard, ground) leave the footprint.  A mitred
        # opening removes notches and corner contacts that cannot be extruded.
        reduced = footprint.difference(unary_union([simplified[index] for index in none_ids]))
        reduced = reduced.buffer(-0.1, join_style=2).buffer(0.1, join_style=2)
        pieces = [item for item in getattr(reduced, "geoms", [reduced]) if isinstance(item, Polygon) and not item.is_empty]
        if not pieces:
            return Partition([footprint], []), np.zeros(1, dtype=int)
        reduced = Polygon(max(pieces, key=lambda item: item.area).exterior).simplify(outline_simplify_m)
    # Node all region boundaries and the footprint outline together so that
    # neighbouring cells share exactly identical vertices (watertight extrusion).
    roof_polygons = [simplified[index] for index in roof_ids]
    if not roof_polygons:
        return Partition([footprint], []), np.zeros(1, dtype=int)
    grid = 0.05
    reduced = shapely.set_precision(reduced, grid)
    if not isinstance(reduced, Polygon) or reduced.is_empty:
        return Partition([footprint], []), np.zeros(1, dtype=int)
    # Snap every vertex to a common grid: near-duplicate vertices a few millimetres
    # apart would otherwise create micro edges that break the watertight shell.
    snapped = [shapely.set_precision(polygon.boundary, grid, mode="pointwise") for polygon in roof_polygons]
    noded = unary_union([reduced.boundary, *snapped])
    tree = shapely.STRtree(roof_polygons)
    cells: list[Polygon] = []
    labels: list[int] = []
    for piece in polygonize(noded):
        point = piece.representative_point()
        if not reduced.covers(point):
            continue
        owners = [index for index in tree.query(point) if roof_polygons[index].covers(point)]
        if not owners:
            owners = [int(tree.nearest(point))]
        if piece.area <= 1e-8:
            continue
        # Enclosed islands (chimney, dormer) make holes; slice the ring cell so every
        # cell is a simple polygon whose boundary the extruder can node.
        parts = [piece]
        for hole in list(piece.interiors):
            cutter_y = Polygon(hole).centroid.y
            bounds = piece.bounds
            cutter = LineString([(bounds[0] - 1.0, cutter_y), (bounds[2] + 1.0, cutter_y)])
            sliced: list[Polygon] = []
            for part in parts:
                if any(Polygon(h).equals(Polygon(hole)) for h in part.interiors):
                    sliced.extend(item for item in split(part, cutter).geoms if isinstance(item, Polygon))
                else:
                    sliced.append(part)
            parts = sliced
        for part in parts:
            cells.append(part)
            labels.append(regions[roof_ids[owners[0]]][1] + 1)
    if not cells:
        return Partition([footprint], []), np.zeros(1, dtype=int)
    changed = True
    while changed and len(cells) > 1:
        changed = False
        for index, cell in enumerate(cells):
            if cell.area >= min_cell_area:
                continue
            candidates = [(cell.boundary.intersection(other.boundary).length, other_index) for other_index, other in enumerate(cells) if other_index != index]
            length, neighbour = max(candidates, default=(0.0, -1))
            if length <= 1e-8:
                continue
            merged = cell.union(cells[neighbour])
            if not isinstance(merged, Polygon):
                continue
            cells[neighbour] = merged
            cells.pop(index)
            labels.pop(index)
            changed = True
            break
    # Small low-relief islands (rooflights, parapet strips) are details, not roof faces:
    # merge them into the neighbour with the longest shared boundary when the planes
    # differ by less than ``merge_jump_m`` over the island.
    changed = merge_jump_m > 0
    while changed and len(cells) > 1:
        changed = False
        for index, cell in enumerate(cells):
            width = 2.0 * cell.area / max(cell.length, 1e-9)
            thin = (width < 0.9 and cell.area < 15.0) or width < 0.5
            if cell.area >= merge_area_m2 and not thin:
                continue
            ranked = sorted(
                ((cell.boundary.intersection(other.boundary).length, other_index) for other_index, other in enumerate(cells) if other_index != index),
                reverse=True,
            )
            for length, neighbour in ranked:
                if length <= 1e-8:
                    break
                first, second = planes[labels[index] - 1], planes[labels[neighbour] - 1]
                shared = cell.boundary.intersection(cells[neighbour].boundary)
                samples = [np.asarray(point, dtype=float) for part_ in getattr(shared, "geoms", [shared]) for point in getattr(part_, "coords", [])]
                jump = max((abs(_plane_height(first, point) - _plane_height(second, point)) for point in samples), default=0.0)
                if jump >= (0.6 if thin else merge_jump_m):
                    continue
                merged = cell.union(cells[neighbour])
                if not isinstance(merged, Polygon) or merged.interiors:
                    continue
                cells[neighbour] = merged
                cells.pop(index)
                labels.pop(index)
                changed = True
                break
            if changed:
                break
    order = sorted(range(len(cells)), key=lambda item: (cells[item].centroid.x, cells[item].centroid.y))
    cells = [cells[item] for item in order]
    labels = [labels[item] for item in order]
    cells = clean_cells(cells, np.asarray(labels, dtype=int), planes, reduced, ridge_tolerance=simplify_m, step_tolerance=step_simplify_m) if ridge_snap_m > 0 else cells
    return Partition(cells, _adjacency(cells), reduced, feature_list, feature_geometry), np.asarray(labels, dtype=int)


def _plane_xy_line(first: RoofPlane, second: RoofPlane) -> tuple[float, float, float] | None:
    """Return ``a*x + b*y + c = 0`` where two roof planes have equal height."""
    n1, n2 = np.asarray(first.normal, dtype=float), np.asarray(second.normal, dtype=float)
    if abs(n1[2]) < 1e-9 or abs(n2[2]) < 1e-9:
        return None
    a = -(n1[0] / n1[2] - n2[0] / n2[2])
    b = -(n1[1] / n1[2] - n2[1] / n2[2])
    c = -(float(first.d) / n1[2] - float(second.d) / n2[2])
    return float(a), float(b), float(c)


def _plane_height(plane: RoofPlane, xy: np.ndarray) -> float:
    return -float(plane.normal[0] * xy[0] + plane.normal[1] * xy[1] + plane.d) / float(plane.normal[2])


def _key(xy) -> tuple[int, int]:
    return (round(float(xy[0]) * 1e5), round(float(xy[1]) * 1e5))


def _segment_hit(line: tuple[float, float, float], start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    """Intersection of ``a*x+b*y+c=0`` with a segment (parameter within a small margin)."""
    a, b, c = line
    f0, f1 = a * start[0] + b * start[1] + c, a * end[0] + b * end[1] + c
    if abs(f1 - f0) < 1e-12:
        return None
    t = f0 / (f0 - f1)
    if t < 0.0 or t > 1.0:
        return None
    return start + min(1.0, max(0.0, t)) * (end - start)


def _three_plane_point(planes3: list[RoofPlane]) -> np.ndarray | None:
    first, second, third = planes3
    l12, l23 = _plane_xy_line(first, second), _plane_xy_line(second, third)
    if l12 is None or l23 is None:
        return None
    matrix = np.array([[l12[0], l12[1]], [l23[0], l23[1]]])
    if abs(np.linalg.det(matrix)) < 1e-6:
        return None
    return np.linalg.solve(matrix, -np.array([l12[2], l23[2]]))


def clean_cells(
    cells: list[Polygon],
    labels: np.ndarray,
    planes: list[RoofPlane],
    footprint: Polygon,
    *,
    node_tolerance: float = 0.2,
    ridge_tolerance: float = 0.35,
    step_tolerance: float = 0.1,
    axis_tolerance_deg: float = 12.0,
) -> list[Polygon]:
    """Turn stair-stepped label regions into CAD-clean roof polygons.

    Junctions between two planes move onto the exact plane intersection line (and onto the
    footprint edge); three-plane junctions move to the common point; boundaries between two
    regions become single straight segments; near-axis step boundaries snap to the building
    axes.  Every vertex edit is checked against the owning cells and reverted when it would
    create an invalid polygon.  The whole clean-up is refused if the result is not extrudable.
    """
    if len(cells) < 2:
        return cells
    labels = np.asarray(labels, dtype=int)
    pos: dict[tuple[int, int], np.ndarray] = {}
    rings: list[list[tuple[int, int]]] = []
    for cell in cells:
        keys = []
        for point in np.asarray(cell.exterior.coords[:-1], dtype=float):
            pos.setdefault(_key(point), point)
            if not keys or keys[-1] != _key(point):
                keys.append(_key(point))
        rings.append(keys)
    outline = footprint.exterior

    def build() -> tuple[dict, dict, dict]:
        owners: dict = {}
        edges: dict = {}
        for index, ring in enumerate(rings):
            for position, vertex in enumerate(ring):
                owners.setdefault(vertex, set()).add(index)
                other = ring[(position + 1) % len(ring)]
                if other == vertex:
                    continue
                edges.setdefault(frozenset((vertex, other)), set()).add(index)
        neighbours: dict = {}
        for edge in edges:
            a, b = tuple(edge)
            neighbours.setdefault(a, set()).add(b)
            neighbours.setdefault(b, set()).add(a)
        return owners, edges, neighbours

    def on_outline(vertex) -> bool:
        return outline.distance(shapely_point(pos[vertex])) < 1e-4

    owners, edges, neighbours = build()

    def sound(touched) -> bool:
        """Touched cells stay valid and never overlap any other cell."""
        if any(len(rings[cell]) < 3 for cell in touched):
            return False
        shapes = {cell: Polygon([pos[v] for v in rings[cell]]) for cell in range(len(rings)) if len(rings[cell]) >= 3}
        for cell in touched:
            if not shapes[cell].is_valid:
                return False
            for other in range(len(rings)):
                if other != cell and other in shapes and shapes[cell].intersection(shapes[other]).area > 1e-6:
                    return False
        return True

    def move(vertex, target: np.ndarray) -> bool:
        old = pos[vertex].copy()
        pos[vertex] = np.asarray(target, dtype=float)
        if sound(owners[vertex]):
            return True
        pos[vertex] = old
        return False

    locked: set = set()
    owners, edges, neighbours = build()

    # 2. polylines between junctions become one straight segment
    def is_node(vertex) -> bool:
        return len(neighbours[vertex]) != 2

    visited: set = set()
    chains = []
    for edge, members in edges.items():
        if len(members) != 2 or edge in visited:
            continue
        a, b = tuple(edge)
        chain = [a, b]
        visited.add(edge)
        for end in (0, 1):
            while True:
                tail = chain[-1] if end == 0 else chain[0]
                if is_node(tail) or on_outline(tail):
                    break
                previous = chain[-2] if end == 0 else chain[1]
                following = [n for n in neighbours[tail] if n != previous]
                if not following or len(edges[frozenset((tail, following[0]))]) != 2:
                    break
                visited.add(frozenset((tail, following[0])))
                if end == 0:
                    chain.append(following[0])
                else:
                    chain.insert(0, following[0])
        chains.append(chain)
    def douglas_peucker(indices: list[int], points: list[np.ndarray], epsilon: float) -> set[int]:
        first, last = points[indices[0]], points[indices[-1]]
        chord = last - first
        length = float(np.linalg.norm(chord))
        if len(indices) <= 2:
            return set(indices)
        if length < 1e-9:
            distances = [float(np.linalg.norm(points[i] - first)) for i in indices[1:-1]]
        else:
            distances = [abs(float((points[i] - first)[0] * chord[1] - (points[i] - first)[1] * chord[0])) / length for i in indices[1:-1]]
        worst = int(np.argmax(distances))
        if distances[worst] <= epsilon:
            return {indices[0], indices[-1]}
        split = indices[1:-1][worst]
        cut = indices.index(split)
        return douglas_peucker(indices[: cut + 1], points, epsilon) | douglas_peucker(indices[cut:], points, epsilon)

    for chain in chains:
        if len(chain) < 3:
            continue
        points = [pos[v] for v in chain]
        indices = list(range(len(chain)))
        owners_here = sorted({int(labels[cell]) for cell in owners[chain[0]] & owners[chain[1]]}) if len(chain) > 1 else []
        epsilon = ridge_tolerance
        if len(owners_here) == 2:
            middle = (points[0] + points[-1]) / 2.0
            jump = abs(_plane_height(planes[owners_here[0] - 1], middle) - _plane_height(planes[owners_here[1] - 1], middle))
            epsilon = ridge_tolerance if jump < 0.3 else (0.25 if jump < 0.8 else step_tolerance)
        if len(owners_here) == 2:
            line = _plane_xy_line(planes[owners_here[0] - 1], planes[owners_here[1] - 1])
            if line is not None and math.hypot(line[0], line[1]) >= 0.04:
                norm = math.hypot(line[0], line[1])
                offsets = [(line[0] * point[0] + line[1] * point[1] + line[2]) / norm for point in points]
                if max(abs(value) for value in offsets) <= 0.5:
                    # a ridge, valley or hip: the boundary IS the plane intersection line
                    for vertex, value in zip(chain[1:-1], offsets[1:-1], strict=True):
                        move(vertex, pos[vertex] - value * np.array((line[0], line[1])) / norm)
                    points = [pos[v] for v in chain]
                    epsilon = 0.03
        keep = douglas_peucker(indices, points, epsilon)
        for index in indices[1:-1]:
            if index in keep:
                continue
            vertex = chain[index]
            members = set(owners[vertex])
            saved = {cell: list(rings[cell]) for cell in members}
            for cell in members:
                rings[cell] = [item for item in rings[cell] if item != vertex]
            if not sound(members):
                for cell, ring in saved.items():
                    rings[cell] = ring
    owners, edges, neighbours = build()

    # 1. junctions onto exact intersection lines / three-plane points
    for vertex in list(owners):
        around = {int(labels[cell]) for cell in owners[vertex]}
        point = pos[vertex]
        if not any(vertex in edge and len(members) == 2 for edge, members in edges.items()):
            continue
        if on_outline(vertex):
            if len(around) != 2:
                continue
            outline_neighbours = [next(iter(edge - {vertex})) for edge, members in edges.items() if vertex in edge and len(members) == 1]
            if len(outline_neighbours) != 2:
                continue
            u, v = pos[outline_neighbours[0]] - point, pos[outline_neighbours[1]] - point
            cosine = float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))
            if cosine > -0.94:  # a real footprint corner, never moved
                continue
            first, second = sorted(around)
            line = _plane_xy_line(planes[first - 1], planes[second - 1])
            if line is None or math.hypot(line[0], line[1]) < 0.04:
                continue
            hits = [hit for neighbour in outline_neighbours if (hit := _segment_hit(line, point, pos[neighbour])) is not None]
            hits = [hit for hit in hits if np.linalg.norm(hit - point) <= node_tolerance]
            if hits and move(vertex, min(hits, key=lambda hit: np.linalg.norm(hit - point))):
                locked.add(vertex)
        elif len(around) == 3 and len(neighbours[vertex]) == 3:
            target = _three_plane_point([planes[label - 1] for label in sorted(around)])
            if target is not None and np.linalg.norm(target - point) <= node_tolerance and move(vertex, target):
                locked.add(vertex)
    owners, edges, neighbours = build()

    # 3. near-axis step boundaries snap to the building axes; coordinates of such
    #    vertices are clustered so neighbouring edges become exactly collinear
    base = dominant_direction([footprint])
    rotate = np.array([[math.cos(base), math.sin(base)], [-math.sin(base), math.cos(base)]])
    wanted: dict[int, dict] = {0: {}, 1: {}}  # axis index -> vertex -> rotated coordinate
    for edge, members in edges.items():
        if len(members) != 2:
            continue
        a, b = tuple(edge)
        pair = sorted({int(labels[cell]) for cell in members})
        if len(pair) != 2 or on_outline(a) or on_outline(b):
            continue
        line = _plane_xy_line(planes[pair[0] - 1], planes[pair[1] - 1])
        if line is not None and math.hypot(line[0], line[1]) >= 0.04:
            norm = math.hypot(line[0], line[1])
            if max(abs(line[0] * pos[v][0] + line[1] * pos[v][1] + line[2]) / norm for v in (a, b)) < 0.2:
                continue  # a ridge: the intersection line owns this boundary
        vector = rotate @ (pos[b] - pos[a])
        angle = math.atan2(vector[1], vector[0])
        for index, reference in ((1, 0.0), (0, math.pi / 2)):
            if abs((angle - reference + math.pi / 2) % math.pi - math.pi / 2) <= math.radians(axis_tolerance_deg):
                for vertex in (a, b):
                    if vertex not in locked:
                        wanted[index][vertex] = float((rotate @ pos[vertex])[index])
                break
    targets: dict = {}
    for index, values in wanted.items():
        ordered = sorted(values.items(), key=lambda item: item[1])
        group: list = []
        for vertex, value in ordered + [(None, math.inf)]:
            if group and value - group[-1][1] > 0.2:
                level = float(np.mean([item[1] for item in group]))
                for member, _value in group:
                    targets.setdefault(member, {})[index] = level
                group = []
            if vertex is not None:
                group.append((vertex, value))
    for vertex, levels in targets.items():
        rv = rotate @ pos[vertex]
        for index, level in levels.items():
            rv[index] = level
        target = rotate.T @ rv
        if np.linalg.norm(target - pos[vertex]) <= 0.25:
            move(vertex, target)
    owners, edges, neighbours = build()

    # 4. drop vertices that are collinear in every owning ring
    for vertex in list(owners):
        if len(neighbours[vertex]) != 2:
            continue
        flags = []
        for cell in owners[vertex]:
            ring = rings[cell]
            if vertex not in ring:
                flags.append(False)
                continue
            position = ring.index(vertex)
            previous, here, following = pos[ring[position - 1]], pos[vertex], pos[ring[(position + 1) % len(ring)]]
            cross = float((here[0] - previous[0]) * (following[1] - here[1]) - (here[1] - previous[1]) * (following[0] - here[0]))
            flags.append(abs(cross) < 1e-7 and float(np.dot(here - previous, following - here)) > 0)
        if flags and all(flags):
            for cell in owners[vertex]:
                if len(rings[cell]) > 3:
                    rings[cell] = [item for item in rings[cell] if item != vertex]
    result = [Polygon([pos[v] for v in ring]) for ring in rings]
    if any((not polygon.is_valid) or polygon.is_empty for polygon in result):
        return cells
    from samcloud.buildings.extrude import _node_points, _split_ring, _triangulate_ring

    try:
        every = list(np.asarray(footprint.exterior.coords[:-1], dtype=float))
        every.extend(point for polygon in result for point in np.asarray(polygon.exterior.coords[:-1], dtype=float))
        nodes = _node_points(every)
        for polygon in result:
            _triangulate_ring(_split_ring(np.asarray(polygon.exterior.coords[:-1], dtype=float), nodes))
    except ValueError:
        return cells
    union = unary_union(result)
    if abs(sum(item.area for item in result) - union.area) > 1e-6 * max(1.0, footprint.area):
        return cells
    if abs(union.area - unary_union(cells).area) > 0.05 * footprint.area:
        return cells
    return result
