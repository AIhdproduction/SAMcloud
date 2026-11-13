"""Small deterministic projects rendered by a physically consistent analytic ray caster."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import yaml

from samcloud.dxf.levels import Level
from samcloud.dxf.state import DxfProjectState, save_state
from samcloud.engine.ply_io import write_classified_ply

OUTDOOR_BOX = (2.5, 4.5, 2.5, 4.1)
TREE_CENTRE = np.array([6.1, 5.2, 2.3])
TABLE = (2.2, 3.7, 1.8, 2.8)


@dataclass(frozen=True, slots=True)
class SyntheticOrthoProject:
    project_dir: Path
    model_dir: Path
    level: Level | None
    mode: str


def _floor_colours(xy: np.ndarray) -> np.ndarray:
    """Return the shared 30 cm checker and smooth ground/floor colour function."""
    checker = (np.floor(xy[:, 0] / 0.3) + np.floor(xy[:, 1] / 0.3)) % 2
    return np.column_stack((75 + checker * 18 + xy[:, 0] * 5, 70 + xy[:, 1] * 8, 90 + checker * 12)).clip(0, 255).astype(np.uint8)


def make_outdoor_project(root: Path, *, opencv: bool = False, seed: int = 20261109) -> SyntheticOrthoProject:
    """Build the textured site, including full building and tree geometry."""
    project = root / ("outdoor_opencv" if opencv else "outdoor")
    cloud, colours, classes = _outdoor_cloud(seed)
    model = _write_model(project, cloud[:: max(1, len(cloud) // 400)], opencv=opencv, indoor=False)
    _write_cloud_and_project(project, cloud, colours, classes, "outdoor")
    return SyntheticOrthoProject(project, model, None, "outdoor")


def make_indoor_project(root: Path, *, seed: int = 20261109, floor_hole: tuple[float, float, float, float] | None = None, table_hole: tuple[float, float, float, float] | None = None) -> SyntheticOrthoProject:
    """Build a room, including its ceiling and all enclosing walls for z-buffer tests."""
    project = root / "indoor"
    cloud, colours, classes = _indoor_cloud(seed, floor_hole=floor_hole, table_hole=table_hole)
    model = _write_model(project, cloud[:: max(1, len(cloud) // 400)], opencv=False, indoor=True)
    _write_cloud_and_project(project, cloud, colours, classes, "indoor")
    level = Level("level_1", "1", 0.0, 2.6, int(np.count_nonzero(classes == 4)), 1.0, False)
    state = DxfProjectState(levels=[{"id": level.id, "name": level.name, "floor_z": 0.0, "ceiling_z": 2.6}])
    save_state(project, state)
    return SyntheticOrthoProject(project, model, level, "indoor")


def ground_truth_rgb(project: SyntheticOrthoProject, grid: object) -> np.ndarray:
    """Return analytic top-surface colours at orthophoto pixel centres."""
    rows, cols = np.indices((grid.height, grid.width))
    xy = grid.pixel_to_local(np.column_stack((cols.ravel(), rows.ravel())))
    rgb = _floor_colours(xy)
    if project.mode == "outdoor":
        roof = _rect_mask(xy, OUTDOOR_BOX)
        crown = np.sum((xy - TREE_CENTRE[:2]) ** 2, axis=1) <= 0.8**2
        rgb[roof] = [230, 25, 20]
        rgb[crown] = [82, 122, 78]
    else:
        table = _rect_mask(xy, TABLE)
        rgb[table] = [140, 80, 30]
    return rgb.reshape(grid.height, grid.width, 3)


def footprint_masks(project: SyntheticOrthoProject, grid: object) -> dict[str, np.ndarray]:
    """Return boolean footprints with names shared by end-to-end assertions."""
    rows, cols = np.indices((grid.height, grid.width))
    xy = grid.pixel_to_local(np.column_stack((cols.ravel(), rows.ravel())))
    result = {
        "box": _rect_mask(xy, OUTDOOR_BOX),
        "tree": np.sum((xy - TREE_CENTRE[:2]) ** 2, axis=1) <= 0.8**2,
        "table": _rect_mask(xy, TABLE),
    }
    return {name: value.reshape(grid.height, grid.width) for name, value in result.items()}


def _write_cloud_and_project(project: Path, xyz: np.ndarray, rgb: np.ndarray, scalar: np.ndarray, mode: str) -> None:
    destination = project / "products" / "classified" / "classified.ply"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_classified_ply(destination, xyz.astype(np.float32), rgb.astype(np.uint8), scalar.astype(np.uint8))
    document = {"project": {"name": project.name}, "settings": {"classification": {"class_set": mode, "engine": "sam3"}, "paths": {"image_directory": str(project / "inputs" / "images")}}, "workflow": {}}
    (project / "project.samcloud.yml").write_text(yaml.safe_dump(document), encoding="utf-8")


def _write_model(project: Path, points: np.ndarray, *, opencv: bool, indoor: bool) -> Path:
    """Write cameras and photos from the exact same rays used by the source renderer."""
    images = project / "inputs" / "images"
    images.mkdir(parents=True, exist_ok=True)
    model = project / "products" / "sparse" / "0"
    model.mkdir(parents=True, exist_ok=True)
    reconstruction = pycolmap.Reconstruction()
    params = np.array([450.0, 450.0, 320.0, 240.0, 0.05, -0.01, 0.001, -0.0005]) if opencv else np.array([300.0, 300.0, 320.0, 240.0]) if indoor else np.array([450.0, 450.0, 320.0, 240.0])
    camera = pycolmap.Camera(camera_id=1, model="OPENCV" if opencv else "PINHOLE", width=640, height=480, params=params)
    reconstruction.add_camera_with_trivial_rig(camera)
    centres, targets = _camera_layout(indoor)
    for image_id, (centre, target) in enumerate(zip(centres, targets, strict=True), 1):
        pose = _look_at(centre, target)
        name = f"view_{image_id:02d}.png"
        reconstruction.add_image_with_trivial_frame(pycolmap.Image(name=name, camera_id=1, image_id=image_id), pose)
        image = _render_photo(camera, pose, indoor)
        assert cv2.imwrite(str(images / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    for point in points[:400]:
        reconstruction.add_point3D(point, pycolmap.Track())
    reconstruction.write(model)
    return model


def _camera_layout(indoor: bool) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if indoor:
        centres = [np.array(value, dtype=np.float64) for value in ((1.0, 1.0, 1.5), (5.0, 1.0, 1.45), (1.0, 4.0, 1.55), (5.0, 4.0, 1.5), (3.0, 0.8, 1.5), (3.0, 4.2, 1.5), (0.8, 2.5, 1.45), (5.2, 2.5, 1.55))]
        targets = [np.array((3.0, 2.5, 0.0), dtype=np.float64) for _ in centres]
        return centres, targets
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    centres = [np.array((4.0 + 19.0 * np.cos(angle), 3.5 + 19.0 * np.sin(angle), 30.0 + 5.0 * (index % 3)), dtype=np.float64) for index, angle in enumerate(angles)]
    centres += [np.array((4.0, 3.5, 35.0)), np.array((4.4, 3.1, 28.0))]
    return centres, [np.array((4.0, 3.5, 0.0), dtype=np.float64) for _ in centres]


def _look_at(centre: np.ndarray, target: np.ndarray) -> pycolmap.Rigid3d:
    """Build COLMAP's world-to-camera pose: x right, y down and z forward."""
    forward = target - centre
    forward /= np.linalg.norm(forward)
    reference_up = np.array((0.0, 0.0, 1.0))
    right = np.cross(forward, reference_up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array((1.0, 0.0, 0.0))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    pose = pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), -rotation @ centre)
    projected = pycolmap.Camera(camera_id=99, model="PINHOLE", width=640, height=480, params=np.array([400.0, 400.0, 320.0, 240.0])).img_from_cam((rotation @ target - rotation @ centre)[None, :])[0]
    assert np.max(np.abs(projected - np.array((320.0, 240.0)))) < 1e-6
    return pose


def _render_photo(camera: pycolmap.Camera, pose: pycolmap.Rigid3d, indoor: bool) -> np.ndarray:
    """Cast pixel-centre rays after pycolmap undistortion, so OPENCV photos match their model."""
    u, v = np.meshgrid(np.arange(camera.width) + 0.5, np.arange(camera.height) + 0.5)
    cam_xy = camera.cam_from_img(np.column_stack((u.ravel(), v.ravel())))
    rays_camera = np.column_stack((cam_xy, np.ones(len(cam_xy))))
    rotation = pose.rotation.matrix()
    origin = -rotation.T @ np.asarray(pose.translation)
    rays_world = rays_camera @ rotation
    points, colours = _cast_scene(origin, rays_world, indoor)
    del points
    return colours.reshape(camera.height, camera.width, 3)


def _cast_scene(origin: np.ndarray, rays: np.ndarray, indoor: bool) -> tuple[np.ndarray, np.ndarray]:
    count = len(rays)
    best = np.full(count, np.inf)
    hits = np.zeros((count, 3))
    colours = np.tile(np.array([84, 116, 100], dtype=np.uint8), (count, 1))

    def accept(distance: np.ndarray, mask: np.ndarray, colour: np.ndarray | None = None, point_colours: np.ndarray | None = None) -> None:
        chosen = mask & (distance > 1e-7) & (distance < best)
        if not np.any(chosen):
            return
        best[chosen] = distance[chosen]
        hits[chosen] = origin + rays[chosen] * distance[chosen, None]
        if point_colours is not None:
            colours[chosen] = point_colours[chosen]
        elif colour is not None:
            colours[chosen] = colour

    floor_t = _plane_t(origin, rays, 2, 0.0)
    floor_points = origin + rays * floor_t[:, None]
    width, height = (6.0, 5.0) if indoor else (8.0, 7.0)
    floor_mask = (floor_points[:, 0] >= 0) & (floor_points[:, 0] <= width) & (floor_points[:, 1] >= 0) & (floor_points[:, 1] <= height)
    accept(floor_t, floor_mask, point_colours=_floor_colours(floor_points[:, :2]))
    if indoor:
        ceiling_t = _plane_t(origin, rays, 2, 2.6)
        ceiling_points = origin + rays * ceiling_t[:, None]
        accept(ceiling_t, (ceiling_points[:, 0] >= 0) & (ceiling_points[:, 0] <= 6) & (ceiling_points[:, 1] >= 0) & (ceiling_points[:, 1] <= 5), np.array([30, 60, 235], dtype=np.uint8))
        _room_walls(origin, rays, accept)
        _box(origin, rays, np.array([2.2, 1.8, 0.0]), np.array([3.7, 2.8, 0.75]), accept, indoor=True)
    else:
        _box(origin, rays, np.array([2.5, 2.5, 0.0]), np.array([4.5, 4.1, 2.0]), accept, indoor=False)
        _sphere_hit(origin, rays, accept)
        _cylinder_hit(origin, rays, accept)
    return hits, colours


def _plane_t(origin: np.ndarray, rays: np.ndarray, axis: int, value: float) -> np.ndarray:
    denominator = rays[:, axis]
    return np.where(np.abs(denominator) > 1e-10, (value - origin[axis]) / denominator, np.inf)


def _box(origin: np.ndarray, rays: np.ndarray, lower: np.ndarray, upper: np.ndarray, accept: object, *, indoor: bool) -> None:
    inv = np.divide(1.0, rays, out=np.full_like(rays, np.inf), where=np.abs(rays) > 1e-12)
    first = (lower - origin) * inv
    second = (upper - origin) * inv
    enter = np.maximum.reduce(np.minimum(first, second), axis=1)
    leave = np.minimum.reduce(np.maximum(first, second), axis=1)
    distance = np.where(enter > 1e-7, enter, leave)
    mask = leave >= np.maximum(enter, 0.0)
    hit = origin + rays * distance[:, None]
    if indoor:
        colour = np.tile(np.array([140, 80, 30], dtype=np.uint8), (len(rays), 1))
    else:
        colour = np.tile(np.array([230, 25, 20], dtype=np.uint8), (len(rays), 1))
        face = np.argmin(np.minimum(np.abs(hit - lower), np.abs(hit - upper)), axis=1)
        roof = np.abs(hit[:, 2] - upper[2]) < 1e-5
        colour[(face == 0) & ~roof] = [255, 150, 30]
        colour[(face == 1) & ~roof] = [220, 30, 180]
        colour[(face == 2) & ~roof] = [240, 220, 20]
        colour[(face == 0) & ~roof & (hit[:, 0] < lower[0] + 1e-5)] = [20, 200, 210]
    accept(distance, mask, point_colours=colour)


def _room_walls(origin: np.ndarray, rays: np.ndarray, accept: object) -> None:
    for axis, value, extent in ((0, 0.0, 5.0), (0, 6.0, 5.0), (1, 0.0, 6.0), (1, 5.0, 6.0)):
        distance = _plane_t(origin, rays, axis, value)
        point = origin + rays * distance[:, None]
        other = 1 - axis
        mask = (point[:, other] >= 0) & (point[:, other] <= extent) & (point[:, 2] >= 0) & (point[:, 2] <= 2.6)
        accept(distance, mask, np.array([130, 130, 130], dtype=np.uint8))


def _sphere_hit(origin: np.ndarray, rays: np.ndarray, accept: object) -> None:
    offset = origin - TREE_CENTRE
    b = 2.0 * rays @ offset
    discriminant = b * b - 4.0 * (offset @ offset - 0.8**2)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    distance = (-b - root) / 2.0
    accept(distance, discriminant >= 0, np.array([82, 122, 78], dtype=np.uint8))


def _cylinder_hit(origin: np.ndarray, rays: np.ndarray, accept: object) -> None:
    offset = origin[:2] - TREE_CENTRE[:2]
    a = np.sum(rays[:, :2] ** 2, axis=1)
    b = 2.0 * (rays[:, :2] @ offset)
    c = offset @ offset - 0.08**2
    discriminant = b * b - 4.0 * a * c
    distance = (-b - np.sqrt(np.maximum(discriminant, 0.0))) / np.where(a > 1e-12, 2.0 * a, 1.0)
    z = origin[2] + distance * rays[:, 2]
    accept(distance, (discriminant >= 0) & (z >= 0) & (z <= 1.5), np.array([100, 60, 25], dtype=np.uint8))


def _outdoor_cloud(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ground = _surface_xy(0, 8, 0, 7, 0.05, 0.0)
    building = _box_surface(np.array([2.5, 2.5, 0.0]), np.array([4.5, 4.1, 2.0]), 0.05)
    sphere = _sphere_surface(TREE_CENTRE, 0.8, 0.05)
    trunk = _cylinder_surface(TREE_CENTRE[:2], 0.08, 1.5, 0.05)
    xyz = np.vstack((ground, building, sphere, trunk))
    rgb = np.vstack((_floor_colours(ground[:, :2]), _building_colours(building), np.tile([82, 122, 78], (len(sphere), 1)), np.tile([100, 60, 25], (len(trunk), 1))))
    scalar = np.concatenate((np.full(len(ground), 1), np.full(len(building), 7), np.full(len(sphere) + len(trunk), 6))).astype(np.uint8)
    return xyz + rng.normal(0.0, 0.004, xyz.shape), rgb, scalar


def _indoor_cloud(seed: int, *, floor_hole: tuple[float, float, float, float] | None = None, table_hole: tuple[float, float, float, float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    floor = _surface_xy(0, 6, 0, 5, 0.04, 0.0)
    if floor_hole is not None:
        x0, y0, x1, y1 = floor_hole
        floor = floor[~((floor[:, 0] >= x0) & (floor[:, 0] <= x1) & (floor[:, 1] >= y0) & (floor[:, 1] <= y1))]
    ceiling = _surface_xy(0, 6, 0, 5, 0.04, 2.6)
    walls = _room_wall_cloud(0.04)
    table = _box_surface(np.array([2.2, 1.8, 0.0]), np.array([3.7, 2.8, 0.75]), 0.04)
    if table_hole is not None:
        x0, y0, x1, y1 = table_hole
        in_hole = (floor[:, 0] >= x0) & (floor[:, 0] <= x1) & (floor[:, 1] >= y0) & (floor[:, 1] <= y1)
        floor = floor[~in_hole]
        table_top_hole = (table[:, 0] >= x0) & (table[:, 0] <= x1) & (table[:, 1] >= y0) & (table[:, 1] <= y1) & np.isclose(table[:, 2], 0.75)
        table = table[~table_top_hole]
    xyz = np.vstack((floor, ceiling, walls, table))
    rgb = np.vstack((_floor_colours(floor[:, :2]), np.tile([30, 60, 235], (len(ceiling), 1)), np.tile([130, 130, 130], (len(walls), 1)), np.tile([140, 80, 30], (len(table), 1))))
    scalar = np.concatenate((np.full(len(floor), 4), np.full(len(ceiling), 3), np.full(len(walls), 2), np.full(len(table), 17))).astype(np.uint8)
    return xyz + rng.normal(0.0, 0.004, xyz.shape), rgb, scalar


def _surface_xy(x0: float, x1: float, y0: float, y1: float, spacing: float, z: float) -> np.ndarray:
    x, y = np.meshgrid(np.arange(x0 + spacing / 2, x1, spacing), np.arange(y0 + spacing / 2, y1, spacing))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))


def _box_surface(lower: np.ndarray, upper: np.ndarray, spacing: float) -> np.ndarray:
    faces = [_surface_xy(lower[0], upper[0], lower[1], upper[1], spacing, upper[2])]
    for x in (lower[0], upper[0]):
        y, z = np.meshgrid(np.arange(lower[1] + spacing / 2, upper[1], spacing), np.arange(lower[2] + spacing / 2, upper[2], spacing))
        faces.append(np.column_stack((np.full(y.size, x), y.ravel(), z.ravel())))
    for y in (lower[1], upper[1]):
        x, z = np.meshgrid(np.arange(lower[0] + spacing / 2, upper[0], spacing), np.arange(lower[2] + spacing / 2, upper[2], spacing))
        faces.append(np.column_stack((x.ravel(), np.full(x.size, y), z.ravel())))
    return np.vstack(faces)


def _building_colours(points: np.ndarray) -> np.ndarray:
    result = np.tile(np.array([230, 25, 20], dtype=np.uint8), (len(points), 1))
    result[np.isclose(points[:, 0], 2.5)] = [20, 200, 210]
    result[np.isclose(points[:, 0], 4.5)] = [220, 30, 180]
    result[np.isclose(points[:, 1], 2.5)] = [240, 220, 20]
    result[np.isclose(points[:, 1], 4.1)] = [255, 150, 30]
    return result


def _sphere_surface(centre: np.ndarray, radius: float, spacing: float) -> np.ndarray:
    theta = np.arange(spacing / radius / 2, np.pi, spacing / radius)
    phi = np.arange(0, 2 * np.pi, spacing / radius)
    theta, phi = np.meshgrid(theta, phi)
    return centre + radius * np.column_stack((np.sin(theta).ravel() * np.cos(phi).ravel(), np.sin(theta).ravel() * np.sin(phi).ravel(), np.cos(theta).ravel()))


def _cylinder_surface(centre: np.ndarray, radius: float, upper: float, spacing: float) -> np.ndarray:
    angle = np.arange(0, 2 * np.pi, spacing / radius)
    z = np.arange(spacing / 2, upper, spacing)
    angle, z = np.meshgrid(angle, z)
    return np.column_stack((centre[0] + radius * np.cos(angle).ravel(), centre[1] + radius * np.sin(angle).ravel(), z.ravel()))


def _room_wall_cloud(spacing: float) -> np.ndarray:
    parts: list[np.ndarray] = []
    for x in (0.0, 6.0):
        y, z = np.meshgrid(np.arange(spacing / 2, 5, spacing), np.arange(spacing / 2, 2.6, spacing))
        parts.append(np.column_stack((np.full(y.size, x), y.ravel(), z.ravel())))
    for y in (0.0, 5.0):
        x, z = np.meshgrid(np.arange(spacing / 2, 6, spacing), np.arange(spacing / 2, 2.6, spacing))
        parts.append(np.column_stack((x.ravel(), np.full(x.size, y), z.ravel())))
    return np.vstack(parts)


def _rect_mask(xy: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    return (xy[:, 0] >= bounds[0]) & (xy[:, 0] <= bounds[1]) & (xy[:, 1] >= bounds[2]) & (xy[:, 1] <= bounds[3])
