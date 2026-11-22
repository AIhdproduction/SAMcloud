"""Screen panels from the mask outlines of the SAM pass (``screen_masks.json.gz``), for screens the cloud does not show.

A dark screen has almost no dense points, but its four corners are sharp in the masks.  A panel is accepted only when
the same four corners agree in several photos taken from different places, form an upright rectangle, and the screen
prompts found it in enough of the photos that show it.  Position and size come only from those rays: no default sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon

from samcloud.bim.room_objects.evidence import SCREEN_LABELS, canonical
from samcloud.bim.room_objects.settings import ScreenFrameSettings


_STRICT_LABELS = ("monitor", "laptop")


@dataclass(slots=True)
class MaskView:
    eye: np.ndarray  # camera centre, metres above the floor
    axes: np.ndarray  # rows: camera x, y, z in the metric plan frame
    pinhole: np.ndarray
    size: tuple[int, int]
    masks: list[dict[str, Any]]  # label (canonical), conf, cut, outline


@dataclass(slots=True)
class Frame:
    corners: np.ndarray  # top left, top right, bottom right, bottom left, as seen from the front
    centre: np.ndarray
    u: np.ndarray
    v: np.ndarray
    normal: np.ndarray
    width: float
    height: float
    views: int
    places: int
    seen: int
    hit: int
    conf: float
    labels: list[str]


def load_mask_views(
    path: Path, cloud_xyz: np.ndarray, to_local: Any, *, units_per_metre: float, floor_z: float,
) -> list[MaskView]:
    """The recorded views in metres of the level; empty when the product is absent or left the frame of the cloud."""
    from scipy.spatial import cKDTree

    from samcloud.engine.screen_masks import read_screen_masks

    data = read_screen_masks(path) if Path(path).is_file() else None
    if not data:
        return []
    probe = np.asarray(data.get("frame_probe") or [], dtype=np.float64)
    if probe.ndim != 2 or not len(probe) or not len(cloud_xyz):
        return []
    distance = cKDTree(np.asarray(cloud_xyz, dtype=np.float64)).query(probe)[0]
    if float(np.median(distance)) > 0.05:
        return []  # the product was moved after the SAM pass: its cameras do not fit the cloud
    views: list[MaskView] = []
    for item in data.get("views") or []:
        try:
            matrix = np.asarray(item["world_to_cam"], dtype=np.float64)
            eye = -matrix[:, :3].T @ matrix[:, 3]
            local = np.asarray(to_local(np.vstack((eye, eye + matrix[:, :3]))), dtype=np.float64)
            axes = local[1:] - local[0]
            axes /= np.linalg.norm(axes, axis=1, keepdims=True)
            masks = [
                {**mask, "label": canonical(mask["label"]), "outline": np.asarray(mask["outline"], dtype=np.float64)}
                for mask in item["masks"]
            ]
            views.append(MaskView(
                eye=(local[0] - (0.0, 0.0, floor_z)) / units_per_metre, axes=axes,
                pinhole=np.asarray(item["pinhole"], dtype=np.float64), size=tuple(item["size"]), masks=masks,
            ))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return views


def _project(view: MaskView, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = (points - view.eye) @ view.axes.T
    depth = camera[..., 2]
    fx, fy, cx, cy = view.pinhole
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    return np.stack((fx * camera[..., 0] / safe + cx, fy * camera[..., 1] / safe + cy), axis=-1), depth


def _rays(view: MaskView, pixels: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = view.pinhole
    camera = np.column_stack(((pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy, np.ones(len(pixels))))
    world = camera @ view.axes
    return world / np.linalg.norm(world, axis=1, keepdims=True)


def _side_intersections(quad: np.ndarray, outline: np.ndarray) -> np.ndarray:
    """Corners where the lines along the middle of the four sides meet: rounded or hidden corners do not cut them."""
    import cv2

    ahead = np.roll(outline, -1, axis=0)
    steps = np.linspace(0.0, 1.0, 24, endpoint=False)[:, None, None]
    points = (outline + steps * (ahead - outline)).reshape(-1, 2)
    weights = np.tile(np.linalg.norm(ahead - outline, axis=1), len(steps))
    points = points[weights > 0.5]
    reach = 0.08 * float(np.linalg.norm(np.ptp(quad, axis=0)))
    lines = []
    for start, end in zip(quad, np.roll(quad, -1, axis=0)):
        along = end - start
        length = float(np.linalg.norm(along))
        if length < 1e-6:
            return quad
        along = along / length
        position = (points - start) @ along
        offset = (points - start) @ np.asarray((-along[1], along[0]))
        side = points[(position >= 0.15 * length) & (position <= 0.85 * length) & (np.abs(offset) <= reach)]
        if len(side) < 4:
            return quad
        dx, dy, x, y = cv2.fitLine(side.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.asarray((x, y), dtype=np.float64), np.asarray((dx, dy), dtype=np.float64)))
    corners = np.zeros((4, 2))
    for index in range(4):
        (first, a), (second, b) = lines[index - 1], lines[index]
        determinant = a[0] * b[1] - a[1] * b[0]
        if abs(determinant) < 0.15:
            return quad
        offset = second - first
        corners[index] = first + (offset[0] * b[1] - offset[1] * b[0]) / determinant * a
    if np.linalg.norm(corners - quad, axis=1).max() > 1.5 * reach:
        return quad
    return corners


def _quad(view: MaskView, outline: np.ndarray, settings: ScreenFrameSettings) -> np.ndarray | None:
    """The four corners of an outline that is a quadrilateral, ordered by the vertical of the room."""
    import cv2

    ring = outline.astype(np.float32).reshape(-1, 1, 2)
    if len(ring) < 4:
        return None
    hull = cv2.convexHull(ring)
    perimeter = cv2.arcLength(hull, True)
    quad = None
    for share in (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07):
        simple = cv2.approxPolyDP(hull, share * perimeter, True).reshape(-1, 2)
        if len(simple) <= 4:
            quad = simple.astype(np.float64) if len(simple) == 4 else None
            break
    if quad is None:
        return None
    area, hull_area = float(cv2.contourArea(quad.astype(np.float32))), float(cv2.contourArea(hull))
    own = float(cv2.contourArea(ring))
    if hull_area <= 0 or min(area, own) < settings.quad_fill * max(area, own, hull_area):
        return None
    quad = _side_intersections(quad, outline)
    middle = quad.mean(axis=0)
    point = view.eye + 2.0 * _rays(view, middle[None, :])[0]
    ends, depth = _project(view, np.vstack((point, point + (0.0, 0.0, 0.05))))
    up = ends[1] - ends[0]
    if depth.min() <= 0 or np.linalg.norm(up) < 1e-6:
        return None
    up /= np.linalg.norm(up)
    right = np.asarray((-up[1], up[0]))
    order = np.argsort(-((quad - middle) @ up))
    top, bottom = quad[order[:2]], quad[order[2:]]
    top, bottom = top[np.argsort(top @ right)], bottom[np.argsort(bottom @ right)]
    corners = np.vstack((top[0], top[1], bottom[1], bottom[0]))
    if min(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)) < settings.min_edge_px:
        return None
    return corners


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    inner = np.prod(np.clip(np.minimum(a.max(0), b.max(0)) - np.maximum(a.min(0), b.min(0)), 0, None))
    union = np.prod(np.ptp(a, axis=0)) + np.prod(np.ptp(b, axis=0)) - inner
    return float(inner / max(union, 1e-9))


def _candidates(views: list[MaskView], settings: ScreenFrameSettings) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for number, view in enumerate(views):
        taken: list[dict[str, Any]] = []
        for mask in sorted(view.masks, key=lambda item: -item["conf"]):
            if mask["cut"] or mask["label"] not in SCREEN_LABELS or mask["conf"] < settings.min_conf:
                continue
            corners = _quad(view, mask["outline"], settings)
            if corners is None:
                continue
            twin = next((item for item in taken if _bbox_iou(corners, item["corners"]) > 0.8), None)
            if twin is not None:  # one screen answers to several names in one photo
                twin["labels"].add(mask["label"])
                continue
            taken.append({
                "view": number, "corners": corners, "rays": _rays(view, corners), "conf": mask["conf"],
                "diag": float(np.linalg.norm(np.ptp(corners, axis=0))), "labels": {mask["label"]},
            })
        found.extend(taken)
    return found


def _rectangles(corners: np.ndarray, settings: ScreenFrameSettings) -> np.ndarray:
    """Which (M, 4, 3) corner sets are upright rectangles of a possible screen size."""
    top, bottom = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 3]
    left, right = corners[:, 0] - corners[:, 3], corners[:, 1] - corners[:, 2]
    across, upwards = 0.5 * (top + bottom), 0.5 * (left + right)
    width, height = np.linalg.norm(across, axis=1), np.linalg.norm(upwards, axis=1)
    safe_w, safe_h = np.maximum(width, 1e-6), np.maximum(height, 1e-6)
    good = (width >= settings.width_m[0]) & (width <= settings.width_m[1])
    good &= (height >= settings.height_m[0]) & (height <= settings.height_m[1])
    good &= (width / safe_h >= settings.aspect[0]) & (width / safe_h <= settings.aspect[1])
    good &= np.linalg.norm(top - bottom, axis=1) <= settings.rectangle_tolerance * safe_w
    good &= np.linalg.norm(left - right, axis=1) <= 1.5 * settings.rectangle_tolerance * safe_h
    good &= np.abs(across[:, 2]) <= settings.level_share * safe_w
    good &= np.linalg.norm(upwards[:, :2], axis=1) <= np.sin(np.radians(settings.max_tilt_deg)) * safe_h
    good &= upwards[:, 2] > 0
    good &= np.abs((across * upwards).sum(axis=1)) <= 0.15 * safe_w * safe_h
    return good


def _intersection(eyes: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """Least-squares point of rays (K, 3)."""
    projectors = np.eye(3) - rays[:, :, None] * rays[:, None, :]
    return np.linalg.solve(projectors.sum(axis=0), (projectors @ eyes[:, :, None]).sum(axis=0))[:, 0]


def _places(eyes: np.ndarray, apart: float) -> int:
    kept: list[np.ndarray] = []
    for eye in eyes:
        if all(np.linalg.norm(eye - other) >= apart for other in kept):
            kept.append(eye)
    return len(kept)


def _agreeing(
    hypotheses: np.ndarray, active: np.ndarray, eyes: np.ndarray, axes: np.ndarray, pinholes: np.ndarray, pixels: np.ndarray,
    tolerance: np.ndarray,
) -> np.ndarray:
    """(H, N): the mask N shows the four corners of hypothesis H, from its front."""
    camera = np.einsum("hcnk,njk->hcnj", hypotheses[:, :, None, :] - eyes[active][None, None], axes[active])
    depth = camera[..., 2]
    safe = np.where(depth < 1e-6, 1e-6, depth)
    u = pinholes[active, 0] * camera[..., 0] / safe + pinholes[active, 2]
    v = pinholes[active, 1] * camera[..., 1] / safe + pinholes[active, 3]
    wanted = pixels[active].transpose(1, 0, 2)[None]
    error = np.hypot(u - wanted[..., 0], v - wanted[..., 1]).max(axis=1)
    return (error <= tolerance[active]) & (depth.min(axis=1) > 0.2)


def fit_screen_frames(
    views: list[MaskView], rooms: list[Polygon], settings: ScreenFrameSettings,
) -> tuple[list[Frame], list[dict[str, Any]]]:
    """Panels several photos agree on, and why other agreeing groups were refused."""
    if not settings.enabled or not views:
        return [], []
    masks = _candidates(views, settings)
    if len(masks) < settings.min_views:
        return [], []
    eyes = np.asarray([views[item["view"]].eye for item in masks])
    axes = np.asarray([views[item["view"]].axes for item in masks])
    pinholes = np.asarray([views[item["view"]].pinhole for item in masks])
    pixels = np.asarray([item["corners"] for item in masks])
    rays = np.asarray([item["rays"] for item in masks])
    diagonal = np.asarray([item["diag"] for item in masks])
    tolerance = settings.corner_share * diagonal + settings.corner_px
    unused = np.ones(len(masks), dtype=bool)
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for seed in np.argsort(-np.asarray([item["conf"] for item in masks]) * np.sqrt(diagonal), kind="stable"):
        if not unused[seed]:
            continue
        others = np.flatnonzero(unused)
        others = others[np.linalg.norm(eyes[others] - eyes[seed], axis=1) >= settings.place_m]
        if len(others) < settings.min_views - 1:
            continue
        # closest points of the corner rays of the seed and of every other mask
        first, second = rays[seed][None], rays[others]
        between = eyes[seed] - eyes[others][:, None, :]
        cosine = (first * second).sum(axis=2)
        along, other = (first * between).sum(axis=2), (second * between).sum(axis=2)
        denominator = np.maximum(1.0 - cosine ** 2, 1e-9)
        s, t = (cosine * other - along) / denominator, (other - cosine * along) / denominator
        near, far = eyes[seed] + s[..., None] * first, eyes[others][:, None, :] + t[..., None] * second
        gap = np.linalg.norm(near - far, axis=2).max(axis=1)
        good = (gap <= settings.ray_gap_m) & (np.minimum(s, t).min(axis=1) > 0.2)
        good &= (np.maximum(s, t).max(axis=1) <= settings.max_distance_m)
        good &= denominator.min(axis=1) >= np.sin(np.radians(settings.min_parallax_deg)) ** 2
        corners = 0.5 * (near + far)
        good &= _rectangles(corners, settings)
        picked = np.flatnonzero(good)
        if not len(picked):
            continue
        picked = picked[np.argsort(gap[picked])[:settings.hypotheses]]
        active = np.flatnonzero(unused)
        agree = _agreeing(corners[picked], active, eyes, axes, pinholes, pixels, tolerance)
        best = int(np.argmax(agree.sum(axis=1)))
        members = active[agree[best]]
        if len(members) < settings.min_views:
            continue
        fitted = corners[picked[best]]
        for _ in range(2):
            refined = np.asarray([_intersection(eyes[members], rays[members, corner]) for corner in range(4)])
            again = active[_agreeing(refined[None], active, eyes, axes, pinholes, pixels, tolerance)[0]]
            if len(again) < settings.min_views or not _rectangles(refined[None], settings)[0]:
                break
            fitted, members = refined, again
        unused[members] = False
        groups.append((fitted, members))
    frames: list[Frame] = []
    refused: list[dict[str, Any]] = []
    for corners, members in groups:
        frame, problem = _judge(corners, [masks[index] for index in members], eyes[members], views, rooms, settings)
        if frame is None:
            refused.append({
                "centre": np.round(corners.mean(axis=0), 3).tolist(), "views": int(len(members)), "reason": f"photo frame: {problem}",
            })
        else:
            frames.append(frame)
    return frames, refused


def _judge(
    corners: np.ndarray, members: list[dict[str, Any]], eyes: np.ndarray, views: list[MaskView], rooms: list[Polygon],
    settings: ScreenFrameSettings,
) -> tuple[Frame | None, str | None]:
    centre = corners.mean(axis=0)
    across = 0.5 * (corners[1] - corners[0] + corners[2] - corners[3])
    upwards = 0.5 * (corners[0] - corners[3] + corners[1] - corners[2])
    width, height = float(np.linalg.norm(across)), float(np.linalg.norm(upwards))
    u = across / width
    v = upwards - (upwards @ u) * u
    v /= np.linalg.norm(v)
    normal = np.cross(u, v)
    places = _places(eyes, settings.place_m)
    if places < settings.min_places:
        return None, "seen from too few places"
    if np.abs((corners - centre) @ normal).max() > settings.flat_m:
        return None, "not flat"
    if corners[:, 2].min() < settings.min_bottom_m:
        return None, "at the floor"
    room = next((item for item in rooms if item.buffer(0.10).contains(Point(centre[:2]))), None)
    if rooms and room is None:
        return None, "outside the rooms"
    if room is not None and np.mean([room.buffer(0.10).contains(Point(eye[:2])) for eye in eyes]) < 0.5:
        return None, "only seen from another room (pane or mirror)"
    seen = hit = named = 0
    for view in views:
        ray = view.eye - centre
        distance = float(np.linalg.norm(ray))
        if distance > settings.support_distance_m or float(ray @ normal) < settings.support_cosine * distance:
            continue
        if room is not None and not room.contains(Point(view.eye[:2])):
            continue
        pixels, depth = _project(view, corners)
        if depth.min() <= 0.2 or pixels.min() < 8 or (pixels.max(axis=0) > np.asarray(view.size) - 8).any():
            continue
        seen += 1
        names = [mask for mask in view.masks if mask["label"] in SCREEN_LABELS and _bbox_iou(pixels, mask["outline"]) >= 0.5]
        hit += bool(names)
        named += any(mask["label"] in _STRICT_LABELS and mask["conf"] >= settings.monitor_conf for mask in names)
    if seen < settings.min_views or hit < settings.support_share * seen:
        return None, f"screen prompts in {hit} of {seen} photos that show it"
    if named < settings.monitor_share * seen:
        # "screen" and "television" also fire on pictures, mirrors and oven doors; the monitor name hardly ever does
        return None, f"a picture, mirror or pane: monitor prompts in only {named} of {seen} photos that show it"
    confidence = float(np.median([item["conf"] for item in members]))
    if confidence < settings.median_conf:
        return None, "weak masks"
    return Frame(
        corners, centre, u, v, normal, width, height, len(members), places, seen, int(hit), confidence,
        [name for item in members for name in sorted(item["labels"])],
    ), None
