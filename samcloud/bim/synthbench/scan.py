"""Scan simulation: 360-degree stations ray-cast against the scene mesh (open3d), noise, dropout, label noise, photos."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely import contains_xy
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from . import geometry as geo
from .mesh import SceneMesh
from .spec import CLASS_CLUTTER, CLASS_FLOOR, CLASS_FURNITURE, CLASS_UNKNOWN, CLASS_WALL, SceneSpec

SKY = np.array([185, 205, 230], dtype=np.uint8)
LIGHT = np.array([0.35, -0.45, 0.82]) / np.linalg.norm([0.35, -0.45, 0.82])


@dataclass
class ScanResult:
    xyz: np.ndarray  # scene frame (float64)
    rgb: np.ndarray  # uint8
    cls: np.ndarray  # uint8 class index (after label noise)
    true_cls: np.ndarray  # uint8 class index before label noise
    instance: np.ndarray  # uint32
    element: np.ndarray  # int32 element index
    station: np.ndarray  # int16 station index of the first return kept
    stations: np.ndarray  # (k,3) scene frame
    stats: dict[str, Any]


@dataclass
class Photo:
    name: str
    centre: np.ndarray  # scene frame
    rotation: np.ndarray  # cam_from_scene rows: right, down, forward
    image: np.ndarray  # (h,w,3) uint8 RGB
    width: int
    height: int
    focal: float


def _raycasting_scene(mesh: SceneMesh):
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(mesh.vertices.astype(np.float32)), o3d.core.Tensor(mesh.faces.astype(np.uint32)))
    return scene


def _cast(scene, origins: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    rays = np.hstack((origins, dirs)).astype(np.float32)
    out = scene.cast_rays(o3d.core.Tensor(rays))
    t = out["t_hit"].numpy().astype(np.float64)
    prim = out["primitive_ids"].numpy().astype(np.int64)
    return t, prim


def scan_stations(spec: SceneSpec, rng: np.random.Generator) -> np.ndarray:
    rooms = geo.room_polygons(spec)
    obstacles = [poly for name, poly, _z in geo.blockers(spec) if not name.endswith("-sweep")]
    blocked = unary_union([poly.buffer(0.25) for poly in obstacles]) if obstacles else Polygon()
    spacing = spec.scan.station_spacing_m
    stations: list[tuple[float, float, float]] = []
    for room in rooms:
        if room["name"] in spec.scan.unvisited_rooms:
            continue
        area = room["polygon"].buffer(-0.45)
        if area.is_empty:
            continue
        minx, miny, maxx, maxy = area.bounds
        nx = max(1, int(math.ceil((maxx - minx) / spacing)))
        ny = max(1, int(math.ceil((maxy - miny) / spacing)))
        found = []
        for i in range(nx):
            for j in range(ny):
                x = minx + (i + 0.5) * (maxx - minx) / nx + rng.uniform(-0.15, 0.15)
                y = miny + (j + 0.5) * (maxy - miny) / ny + rng.uniform(-0.15, 0.15)
                p = Point(x, y)
                if area.contains(p) and not blocked.contains(p):
                    found.append((x, y))
        if not found:  # free sample anywhere in the room
            for _ in range(200):
                x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
                if area.contains(Point(x, y)) and not blocked.contains(Point(x, y)):
                    found.append((x, y))
                    break
        for x, y in found:
            stations.append((x, y, rng.uniform(1.45, 1.65)))
    if spec.scan.door_stations:
        visited = [r["polygon"] for r in rooms if r["name"] not in spec.scan.unvisited_rooms]
        free_visited = unary_union(visited) if visited else Polygon()
        for opening in spec.openings:
            if opening.kind not in ("door", "passage"):
                continue
            wall = spec.wall(opening.wall_id)
            c = geo.opening_centre(spec, opening)
            n = wall.normal()
            sides = [c + n * (wall.thickness / 2 + 0.3), c - n * (wall.thickness / 2 + 0.3)]
            if all(free_visited.contains(Point(*s)) for s in sides):
                stations.append((float(c[0]), float(c[1]), rng.uniform(1.45, 1.65)))
    return np.asarray(stations, dtype=np.float64).reshape(-1, 3)


def _sphere_dirs(step_deg: float, rng: np.random.Generator) -> np.ndarray:
    az = np.radians(np.arange(0.0, 360.0, step_deg) + rng.uniform(0, step_deg))
    el = np.radians(np.arange(-72.0, 88.0, step_deg))
    A, E = np.meshgrid(az, el)
    # thin out near the poles so the angular density is roughly uniform
    keep = rng.random(A.shape) < np.cos(E)
    A, E = A[keep], E[keep]
    return np.column_stack((np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)))


def _shade(mesh: SceneMesh, prim: np.ndarray, points: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    colour = mesh.face_colour()[prim].astype(np.float64)
    normals = mesh.face_normals()[prim]
    light = 0.62 + 0.38 * np.abs(normals @ LIGHT)
    rgb = colour * light[:, None]
    fe = mesh.face_element[prim]
    floor_idx = [i for i, e in enumerate(mesh.elements) if e.kind == "floor"]
    if floor_idx:
        on_floor = fe == floor_idx[0]
        checker = (np.floor(points[on_floor, 0] / 0.3) + np.floor(points[on_floor, 1] / 0.3)) % 2
        rgb[on_floor] *= (0.85 + 0.25 * checker)[:, None]
    if rng is not None:
        rgb += rng.normal(0, 4, rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def simulate_scan(spec: SceneSpec, mesh: SceneMesh, seed: int | None = None) -> ScanResult:
    rng = np.random.default_rng(spec.seed * 1009 + 17 if seed is None else seed)
    cfg = spec.scan
    scene = _raycasting_scene(mesh)
    stations = scan_stations(spec, rng)
    face_transparent = mesh.face_transparent()
    chunks: list[tuple[np.ndarray, ...]] = []
    rays_total = 0
    for index, origin in enumerate(stations):
        dirs = _sphere_dirs(cfg.angular_step_deg, rng)
        rays_total += len(dirs)
        t, prim = _cast(scene, np.repeat(origin[None, :], len(dirs), axis=0), dirs)
        hit = np.isfinite(t) & (t < cfg.max_range_m) & (prim >= 0)
        glass = np.zeros_like(hit)
        glass[hit] = face_transparent[prim[hit]]
        hit &= ~glass | (rng.random(len(hit)) < cfg.window_return)
        t, prim, d = t[hit], prim[hit], dirs[hit]
        t_noisy = t + rng.normal(0.0, cfg.noise_m, len(t))
        pts = origin + d * t_noisy[:, None] + rng.normal(0.0, cfg.noise_m / 3.0, (len(t), 3))
        chunks.append((pts, prim, np.full(len(t), index, dtype=np.int16), t))
    xyz = np.vstack([c[0] for c in chunks])
    prim = np.concatenate([c[1] for c in chunks])
    station = np.concatenate([c[2] for c in chunks])
    rng_dist = np.concatenate([c[3] for c in chunks])
    raw_count = len(xyz)
    # voxel thinning (keeps the closest return per voxel, as a fused multi-view cloud would)
    keys = np.floor(xyz / cfg.voxel_m).astype(np.int64)
    order = np.lexsort((rng_dist, keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    keep = np.sort(order[first])
    xyz, prim, station = xyz[keep], prim[keep], station[keep]
    element = mesh.face_element[prim].astype(np.int32)
    true_cls = mesh.face_class()[prim]
    instance = mesh.face_instance()[prim]
    # dropout patches (missing surfaces)
    mask = np.ones(len(xyz), dtype=bool)
    dropped_patches = 0
    for _ in range(cfg.dropout_patches):
        centre = xyz[rng.integers(len(xyz))]
        radius = rng.uniform(*cfg.dropout_radius_m)
        near = np.linalg.norm(xyz - centre, axis=1) < radius
        mask &= ~(near & (rng.random(len(xyz)) < 0.92))
        dropped_patches += int(near.sum())
    for poly in cfg.unscanned:
        mask &= ~contains_xy(Polygon(poly), xyz[:, 0], xyz[:, 1])
    xyz, element, true_cls, instance, station, prim = xyz[mask], element[mask], true_cls[mask], instance[mask], station[mask], prim[mask]
    cls = true_cls.copy()
    flipped = 0
    if cfg.label_noise > 0:
        flip = rng.random(len(cls)) < cfg.label_noise
        choices = np.array([CLASS_UNKNOWN, CLASS_CLUTTER, CLASS_WALL, CLASS_FLOOR, CLASS_FURNITURE], dtype=np.uint8)
        new = choices[rng.integers(len(choices), size=int(flip.sum()))]
        changed = new != cls[flip]
        idx = np.flatnonzero(flip)[changed]
        cls[idx] = new[changed]
        instance = instance.copy()
        instance[idx] = 0
        flipped = int(len(idx))
    rgb = _shade(mesh, prim, xyz, rng)
    stats = {
        "stations": int(len(stations)), "rays": int(rays_total), "raw_returns": int(raw_count), "points": int(len(xyz)),
        "dropout_points": int(dropped_patches), "label_flips": flipped, "noise_m": cfg.noise_m, "voxel_m": cfg.voxel_m,
    }
    return ScanResult(xyz, rgb, cls, true_cls, instance, element, station, stations, stats)


def look_at(centre: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """cam_from_frame rotation with COLMAP axes (x right, y down, z forward)."""
    f = forward / np.linalg.norm(forward)
    right = np.cross(f, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    return np.vstack((right, down, f))


def render_photos(spec: SceneSpec, mesh: SceneMesh, stations: np.ndarray) -> list[Photo]:
    """Pinhole photos (90 deg horizontal field of view) from every station, ``photos_per_station`` yaw steps."""
    cfg = spec.scan
    width, height = cfg.photo_size
    focal = width / 2.0
    scene = _raycasting_scene(mesh)
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    cam = np.column_stack(((u.ravel() - width / 2) / focal, (v.ravel() - height / 2) / focal, np.ones(u.size)))
    photos: list[Photo] = []
    for s_index, centre in enumerate(stations):
        for k in range(cfg.photos_per_station):
            yaw = 2 * math.pi * k / cfg.photos_per_station + 0.3 * s_index
            pitch = math.radians(-12.0)
            forward = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
            rot = look_at(centre, forward)
            dirs = cam @ rot
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            t, prim = _cast(scene, np.repeat(centre[None, :], len(dirs), axis=0), dirs)
            hit = np.isfinite(t) & (prim >= 0)
            image = np.tile(SKY, (len(dirs), 1))
            pts = centre + dirs[hit] * t[hit, None]
            image[hit] = _shade(mesh, prim[hit], pts)
            photos.append(Photo(f"st{s_index:02d}_{k}.jpg", centre.copy(), rot, image.reshape(height, width, 3), width, height, focal))
    return photos
