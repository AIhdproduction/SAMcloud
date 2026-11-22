"""Synthetic indoor scans with walls at arbitrary angles for the full clean-plan chain.

A scene is a set of room polygons (free floor) separated by wall bodies of a
known thickness, plus door cuts through interior walls.  The generator writes
the orthophoto rasters, a combined segmentation and a classified point cloud
(interior wall faces only, as an indoor scan sees them) with Gaussian noise
and furniture-like occlusion.  ``evaluate_plan`` compares a generated
``DxfPlanResult`` with the scene truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import cv2
import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union
import yaml

from samcloud.dxf.levels import Level
from samcloud.engine.ply_io import write_classified_ply
from samcloud.ortho.build import ortho_output_dir

STOREY = 2.50
DOOR_HEIGHT = 2.02


@dataclass(slots=True)
class Door:
    """A door through the wall that contains ``center`` (metres) along ``angle_deg``."""

    center: tuple[float, float]
    angle_deg: float
    width: float = 0.90
    depth: float = 0.60

    def cut(self) -> Polygon:
        piece = box(-self.width / 2.0, -self.depth / 2.0, self.width / 2.0, self.depth / 2.0)
        return affinity.translate(affinity.rotate(piece, self.angle_deg, origin=(0.0, 0.0)), *self.center)


@dataclass(slots=True)
class Scene:
    """Room polygons, interior walls (centreline + thickness) and doors."""

    name: str
    rooms: list[Polygon]
    interior_walls: list[tuple[LineString, float]] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)
    exterior_thickness: float = 0.30
    noise: float = 0.008
    occlusion: bool = True
    unscanned: list[Polygon] = field(default_factory=list)
    seed: int = 20261121

    def wall_body(self) -> Polygon:
        outline = unary_union([room.buffer(0.20, join_style=2) for room in self.rooms]).buffer(-0.20 + self.exterior_thickness, join_style=2)
        return outline.difference(unary_union(self.rooms))

    def faces(self) -> list[LineString]:
        segments: list[LineString] = []
        for room in self.rooms:
            coords = list(room.exterior.coords)
            segments.extend(LineString((first, second)) for first, second in zip(coords, coords[1:], strict=False))
        return segments


def rotated(polygon: Polygon, angle_deg: float, origin: tuple[float, float] = (0.0, 0.0)) -> Polygon:
    return affinity.rotate(polygon, angle_deg, origin=origin)


def split_by_wall(outline: Polygon, centreline: LineString, thickness: float) -> list[Polygon]:
    """Split a room outline into the rooms on both sides of a straight wall."""
    (x0, y0), (x1, y1) = centreline.coords[0], centreline.coords[-1]
    direction = np.asarray((x1 - x0, y1 - y0), dtype=float)
    direction /= np.linalg.norm(direction)
    far = LineString((np.asarray((x0, y0)) - direction * 50.0, np.asarray((x1, y1)) + direction * 50.0))
    body = far.buffer(thickness / 2.0, cap_style=2)
    rest = outline.difference(body)
    parts = list(rest.geoms) if hasattr(rest, "geoms") else [rest]
    return sorted((part for part in parts if part.area > 0.5), key=lambda part: part.centroid.x)


def write_scene(scene: Scene, project_dir: Path, *, units_per_metre: float = 1.0, pixel: float = 0.02) -> Level:
    """Write the rasters, segmentation and classified cloud of ``scene``."""
    s = float(units_per_metre)
    rng = np.random.default_rng(scene.seed)
    project_dir.mkdir(parents=True, exist_ok=True)
    level = Level("level_1", "1", 0.0, STOREY * s, 0, 1.0, False)
    ortho_dir = ortho_output_dir(project_dir, level)
    ortho_dir.mkdir(parents=True, exist_ok=True)
    body = scene.wall_body()
    cuts = unary_union([door.cut().intersection(body.buffer(0.01)) for door in scene.doors]) if scene.doors else Polygon()
    min_x, min_y, max_x, max_y = body.buffer(0.6).bounds
    width, height = int(math.ceil((max_x - min_x) / pixel)), int(math.ceil((max_y - min_y) / pixel))

    rows, cols = np.mgrid[0:height, 0:width]
    centre_x, centre_y = min_x + (cols + 0.5) * pixel, max_y - (rows + 0.5) * pixel

    def raster(geometry: object) -> np.ndarray:
        """Pixels whose centre lies inside ``geometry``."""
        if geometry.is_empty:
            return np.zeros((height, width), dtype=bool)
        return shapely.contains_xy(geometry, centre_x, centre_y)

    floor = raster(unary_union([*scene.rooms, cuts]))
    walls = raster(body.difference(cuts)) & ~floor
    door_pixels = raster(cuts) & ~raster(unary_union(scene.rooms))
    classes = np.zeros((height, width), dtype=np.uint8)
    classes[floor] = 102
    classes[walls] = 100
    classes[door_pixels] = 103
    for area in scene.unscanned:
        classes[raster(area)] = 0  # nothing measured there: no segmentation either
    dsm = np.zeros((height, width), dtype=np.float32)
    np.save(ortho_dir / "dsm.npy", dsm)
    np.save(ortho_dir / "dtm.npy", np.zeros_like(dsm))
    np.save(ortho_dir / "classes_pc.npy", classes)
    cv2.imwrite(str(ortho_dir / "ortho.png"), np.full((height, width, 3), 180, dtype=np.uint8))
    cv2.imwrite(str(ortho_dir / "valid.png"), np.full((height, width), 255, dtype=np.uint8))
    cv2.imwrite(str(ortho_dir / "view_count.png"), np.ones((height, width), dtype=np.uint16))
    stats = {"bounds": [min_x * s, min_y * s, (min_x + width * pixel) * s, max_y * s], "pixel_size": pixel * s, "width": width, "height": height, "frame": {"origin": [0.0, 0.0, 0.0], "scale": 1.0, "rotation": np.eye(3).tolist()}}
    stats["bounds"][1] = (max_y - height * pixel) * s
    (ortho_dir / "ortho.json").write_text(json.dumps(stats), encoding="utf-8")
    (ortho_dir / "fingerprint.txt").write_text(f"{scene.name}\n", encoding="utf-8")
    np.save(ortho_dir / "class_combined.npy", classes)
    np.save(ortho_dir / "conf_combined.npy", np.full_like(classes, 0.9, dtype=np.float16))
    cv2.imwrite(str(ortho_dir / "class_combined.png"), classes)
    (ortho_dir / "segment_combined.json").write_text(json.dumps({"source": "combined", "fingerprint": f"{scene.name}-seg", "ortho_fingerprint": scene.name, "model_profile": "sam3_indoor"}), encoding="utf-8")

    points: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    z_levels = np.arange(0.30, STOREY - 0.12, 0.06)
    door_cuts = [door.cut() for door in scene.doors]
    for face in scene.faces():
        length = face.length
        values = np.arange(0.01, length, 0.02)
        if not len(values):
            continue
        start = np.asarray(face.coords[0]); direction = (np.asarray(face.coords[-1]) - start) / length
        xy = start + values[:, None] * direction
        occluded = np.zeros(len(values), dtype=bool)
        if scene.occlusion and length > 1.5:
            begin = rng.uniform(0.2, length - 0.9)
            occluded = (values >= begin) & (values <= begin + 0.6)
        in_door = np.zeros(len(values), dtype=bool)
        for cut in door_cuts:
            in_door |= np.asarray([cut.contains(Point(p)) for p in xy])
        for z in z_levels:
            keep = ~(in_door & (z < DOOR_HEIGHT)) & ~(occluded & (z < 1.6))
            if keep.any():
                chosen = xy[keep]
                points.append(np.column_stack((chosen, np.full(len(chosen), z))))
                labels.append(np.full(len(chosen), 2, dtype=np.uint8))
    # Door jambs: the reveal faces inside the wall thickness.
    for cut in door_cuts:
        inside = cut.intersection(body)
        for part in getattr(inside, "geoms", [inside]):
            if part.is_empty or not isinstance(part, Polygon):
                continue
            ring = list(part.exterior.coords)
            for first, second in zip(ring, ring[1:], strict=False):
                segment = LineString((first, second))
                if segment.length < 0.05 or any(segment.centroid.distance(room.exterior) <= 0.01 for room in scene.rooms):
                    continue
                values = np.arange(0.01, segment.length, 0.02)
                xy = np.asarray(first) + values[:, None] * (np.asarray(second) - np.asarray(first)) / segment.length
                for z in z_levels[z_levels < DOOR_HEIGHT]:
                    points.append(np.column_stack((xy, np.full(len(xy), z))))
                    labels.append(np.full(len(xy), 2, dtype=np.uint8))
    walls_xyz = np.vstack(points)
    walls_xyz[:, :2] += rng.normal(0.0, scene.noise, (len(walls_xyz), 2))
    walls_xyz[:, 2] += rng.normal(0.0, 0.004, len(walls_xyz))
    surfaces = []
    for room in scene.rooms:
        rx0, ry0, rx1, ry1 = room.bounds
        gx, gy = np.meshgrid(np.arange(rx0, rx1, 0.06), np.arange(ry0, ry1, 0.06))
        candidates = np.column_stack((gx.ravel(), gy.ravel()))
        inside = np.asarray([room.buffer(-0.03).contains(Point(p)) for p in candidates]) if len(candidates) else np.zeros(0, dtype=bool)
        candidates = candidates[inside]
        surfaces.append((np.column_stack((candidates, np.full(len(candidates), STOREY))), 3))
        surfaces.append((np.column_stack((candidates, np.zeros(len(candidates)))), 4))
    xyz = np.vstack([walls_xyz, *(part for part, _code in surfaces)])
    scalar = np.concatenate([np.concatenate(labels), *(np.full(len(part), code, dtype=np.uint8) for part, code in surfaces)])
    for area in scene.unscanned:
        keep = ~shapely.contains_xy(area, xyz[:, 0], xyz[:, 1])
        xyz, scalar = xyz[keep], scalar[keep]
    xyz = xyz * s
    point_path = project_dir / "products" / "classified" / "classified.ply"
    point_path.parent.mkdir(parents=True, exist_ok=True)
    write_classified_ply(point_path, xyz.astype(np.float32), np.full((len(xyz), 3), 160, dtype=np.uint8), scalar)
    (project_dir / "project.samcloud.yml").write_text(yaml.safe_dump({"project": {"name": project_dir.name}, "settings": {"classification": {"class_set": "indoor", "engine": "sam3"}, "coordinate_systems": {}}}), encoding="utf-8")
    return level


# --- scenes -----------------------------------------------------------------------------------------------------


def rectangle_17() -> Scene:
    """Two rooms (thin 0.10 m partition with a door) in a rectangle rotated by 17 degrees."""
    outline = box(0.0, 0.0, 7.0, 4.5)
    partition = LineString(((3.0, 0.0), (3.0, 4.5)))
    rooms = [rotated(room, 17.0) for room in split_by_wall(outline, partition, 0.10)]
    wall = LineString([tuple(np.asarray(point)) for point in rotated(partition, 17.0, origin=(0.0, 0.0)).coords])
    door_centre = rotated(Point(3.0, 2.0), 17.0)
    return Scene("rect17", rooms, [(wall, 0.10)], [Door((door_centre.x, door_centre.y), 17.0 + 90.0)])


def slanted_30_45() -> Scene:
    """One room with a 45 degree wall and a wall 30 degrees off the y axis."""
    room = Polygon([(0.0, 0.0), (7.0, 0.0), (7.0, 2.0), (5.0, 4.0), (4.0 * math.tan(math.radians(30.0)), 4.0)])
    return Scene("slanted_30_45", [room])


def trapezoid() -> Scene:
    return Scene("trapezoid", [Polygon([(0.0, 0.0), (6.0, 0.0), (5.0, 4.0), (1.0, 4.0)])])


def l_room() -> Scene:
    return Scene("l_room", [Polygon([(0.0, 0.0), (6.0, 0.0), (6.0, 3.0), (3.0, 3.0), (3.0, 6.0), (0.0, 6.0)])])


def slanted_partition() -> Scene:
    """Two rooms joined by a slanted 0.12 m partition (11.3 degrees off the y axis) with a door."""
    outline = box(0.0, 0.0, 8.0, 5.0)
    partition = LineString(((3.5, 0.0), (4.5, 5.0)))
    rooms = split_by_wall(outline, partition, 0.12)
    angle = math.degrees(math.atan2(5.0, 1.0))
    centre = partition.interpolate(0.5, normalized=True)
    return Scene("slanted_partition", rooms, [(partition, 0.12)], [Door((centre.x, centre.y), angle)])


def thin_and_thick() -> Scene:
    """Three rooms: a thick 0.36 m axis partition and a thin 0.10 m partition 25 degrees off the y axis (noise 12 mm)."""
    outline = box(0.0, 0.0, 10.0, 5.0)
    thick = LineString(((3.0, 0.0), (3.0, 5.0)))
    offset = 5.0 * math.tan(math.radians(25.0))
    thin = LineString(((6.0, 0.0), (6.0 + offset, 5.0)))
    left, rest = split_by_wall(outline, thick, 0.36)
    middle, right = split_by_wall(rest, thin, 0.10)
    thick_centre = Point(3.0, 2.5)
    thin_centre = thin.interpolate(0.5, normalized=True)
    doors = [Door((thick_centre.x, thick_centre.y), 90.0), Door((thin_centre.x, thin_centre.y), math.degrees(math.atan2(5.0, offset)))]
    return Scene("thin_and_thick", [left, middle, right], [(thick, 0.36), (thin, 0.10)], doors, noise=0.012)


def thin_axis_partition() -> Scene:
    """Two rooms split by a thin 0.10 m axis partition (both faces only with ``axis_thin_wall_faces``)."""
    outline = box(0.0, 0.0, 7.0, 4.5)
    partition = LineString(((3.0, 0.0), (3.0, 4.5)))
    return Scene("thin_axis_partition", split_by_wall(outline, partition, 0.10), [(partition, 0.10)], [Door((3.0, 2.0), 90.0)])


def unscanned_corner() -> Scene:
    """Pentagon room whose corner between a slanted and an axis wall was never scanned (1.6 m square, no points, no classes)."""
    room = Polygon([(0.0, 0.0), (6.0, 0.0), (6.0, 3.0), (3.0, 5.0), (0.0, 5.0)])
    return Scene("unscanned_corner", [room], unscanned=[box(-0.8, 4.2, 0.8, 5.8)], occlusion=False)


SCENES = {factory.__name__: factory for factory in (rectangle_17, slanted_30_45, trapezoid, l_room, slanted_partition, thin_and_thick, thin_axis_partition, unscanned_corner)}


# --- evaluation ---------------------------------------------------------------------------------------------------


def _bodies(result: object) -> list[Polygon]:
    output = []
    for entity in result.layers.get("A-WALL", []):
        if entity.get("type") == "polyline" and entity.get("closed") and len(entity.get("points", [])) >= 3:
            polygon = Polygon(entity["points"], entity.get("holes", []) or [])
            output.append(polygon.buffer(0) if not polygon.is_valid else polygon)
    return output


def evaluate_plan(scene: Scene, result: object, *, units_per_metre: float = 1.0) -> dict[str, object]:
    """Compare the double-line walls and rooms of ``result`` with the scene truth (metres)."""
    s = float(units_per_metre)
    bodies = [affinity.scale(body, 1.0 / s, 1.0 / s, origin=(0.0, 0.0)) for body in _bodies(result)]
    walls = unary_union(bodies) if bodies else Polygon()
    boundary = walls.boundary
    door_zone = unary_union([door.cut().buffer(0.15) for door in scene.doors]) if scene.doors else Polygon()
    face_errors: list[float] = []
    angle_errors: list[float] = []
    missing_faces = 0
    for face in scene.faces():
        if face.length < 0.6:
            continue
        values = np.linspace(0.25, face.length - 0.25, max(3, int(face.length / 0.10)))
        samples = [face.interpolate(value) for value in values]
        samples = [sample for sample in samples if not door_zone.contains(sample)]
        distances = [boundary.distance(sample) for sample in samples]
        if not distances:
            continue
        if float(np.median(distances)) > 0.10:
            missing_faces += 1
            continue
        face_errors.append(float(np.percentile(distances, 90.0)))
        # Angle of the nearest boundary piece: fit to the closest boundary points.
        nearest = np.asarray([boundary.interpolate(boundary.project(sample)).coords[0] for sample in samples])
        if len(nearest) >= 3:
            centre = nearest.mean(axis=0)
            _values, vectors = np.linalg.eigh(np.cov((nearest - centre).T))
            fitted = vectors[:, -1]
            truth = np.asarray(face.coords[-1]) - np.asarray(face.coords[0])
            truth /= np.linalg.norm(truth)
            angle_errors.append(math.degrees(math.asin(min(1.0, abs(float(fitted[0] * truth[1] - fitted[1] * truth[0]))))))
    thickness_errors: list[float] = []
    for centreline, thickness in scene.interior_walls:
        direction = np.asarray(centreline.coords[-1]) - np.asarray(centreline.coords[0])
        direction /= np.linalg.norm(direction)
        normal = np.asarray((-direction[1], direction[0]))
        for value in np.linspace(0.6, centreline.length - 0.6, 9):
            centre = np.asarray(centreline.interpolate(value).coords[0])
            if door_zone.contains(Point(centre)):
                continue
            probe = LineString((centre - normal * 0.6, centre + normal * 0.6))
            cut = probe.intersection(walls)
            pieces = [piece for piece in getattr(cut, "geoms", [cut]) if not piece.is_empty and piece.distance(Point(centre)) <= 0.02]
            measured = sum(piece.length for piece in pieces)
            thickness_errors.append(abs(measured - thickness))
    rooms = []
    for entity in result.layers.get("A-AREA", []):
        if entity.get("type") == "polyline" and entity.get("closed"):
            rooms.append(affinity.scale(Polygon(entity["points"]), 1.0 / s, 1.0 / s, origin=(0.0, 0.0)))
    room_matches = []
    for truth in scene.rooms:
        best = max((candidate.intersection(truth).area / max(candidate.union(truth).area, 1e-9) for candidate in rooms if candidate.is_valid), default=0.0)
        room_matches.append(best)
    truth_body = scene.wall_body()
    stray = walls.difference(truth_body.buffer(0.05, join_style=2)).area if not walls.is_empty else 0.0
    return {
        "face_error_p90_max": max(face_errors, default=float("nan")),
        "angle_error_max": max(angle_errors, default=float("nan")),
        "missing_faces": missing_faces,
        "thickness_error_max": max(thickness_errors, default=0.0),
        "room_count": len(rooms),
        "room_iou_min": min(room_matches, default=0.0),
        "rooms_valid": all(room.is_valid and not room.is_empty for room in rooms),
        "stray_wall_area": float(stray),
        "wall_bodies": len(bodies),
    }
