"""Televisions, monitors and laptops: thin panels with measured centre, size, yaw and tilt.

A screen needs semantic evidence (its SAM fine term or a screen instance) AND a flat upright surface of a plausible
size.  The labelled part is completed only with measured coplanar furniture points (the bezel of a dark screen), so a
carrier is never swallowed.  Reflections in mirrors and panes, non-planar hits and slivers are rejected with a reason.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon

from samcloud.bim.room_objects.evidence import APPLIANCE, CLUTTER, FURNITURE, SCREEN_LABELS, RoomEvidence
from samcloud.bim.room_objects.geometry import (
    Mesh, box_mesh, fit_plane, hull_of, join_meshes, line_direction, nearest_line, panel_axes, reflection_of, upright_plane,
    voxel_components,
)
from samcloud.bim.room_objects.screen_frames import Frame, fit_screen_frames
from samcloud.bim.room_objects.settings import ReflectionSettings, ScreenFrameSettings, ScreenSettings

DISPLAY_NAMES = {"television": "TV", "monitor": "Monitor", "laptop": "Notebook", "screen": "Bildschirm"}


@dataclass(slots=True)
class Screen:
    semantic: str
    instance_id: int | None
    centre: np.ndarray  # middle of the front face
    u: np.ndarray
    v: np.ndarray
    normal: np.ndarray  # out of the front face
    width: float
    height: float
    thickness: float
    mounting: str  # wall | resting | standing
    tilt_deg: float
    mesh: Mesh
    footprint: Polygon
    points: int
    has_stand: bool

    @property
    def facing_deg(self) -> float:
        return math.degrees(math.atan2(float(self.normal[1]), float(self.normal[0]))) % 360.0


def _seed_mask(evidence: RoomEvidence, settings: ScreenSettings, rejected: list[dict[str, Any]]) -> np.ndarray:
    terms = evidence.term_ids(SCREEN_LABELS)
    has_term = np.isin(evidence.fine, terms) & (evidence.cls == FURNITURE) if len(terms) else np.zeros(len(evidence.cls), dtype=bool)
    seed = has_term.copy()
    for identifier in evidence.instance_ids(SCREEN_LABELS):
        member = (evidence.instance == identifier) & (evidence.cls == FURNITURE)
        count = int(member.sum())
        if not count:
            continue
        share = float(has_term[member].mean())
        if len(terms) and share < settings.min_term_share:
            rejected.append({"instance_id": int(identifier), "reason": "no screen term on the instance", "share": round(share, 3)})
            seed &= evidence.instance != identifier
            continue
        seed |= member
    return seed


def _patches(evidence: RoomEvidence, seeds: np.ndarray, settings: ScreenSettings, rejected: list[dict[str, Any]]) -> list[np.ndarray]:
    """Flat upright patches of the seed points (index arrays)."""
    points = evidence.xyz[seeds]
    labels, count = voxel_components(points, settings.link_m)
    # two monitors edge to edge are two instances in one component: every screen instance is fitted by itself
    owner = evidence.instance[seeds]
    owner = np.where(np.isin(owner, evidence.instance_ids(SCREEN_LABELS)), owner, 0)
    if len(seeds):
        labels = np.unique(np.column_stack((labels, owner)), axis=0, return_inverse=True)[1].reshape(-1)
        count = int(labels.max()) + 1
    patches: list[np.ndarray] = []
    for component in range(count):
        member = seeds[labels == component]
        total = len(member)
        if total < settings.min_points:
            continue
        rest, found = member, []
        for _ in range(4):
            if len(rest) < max(settings.min_points, 0.25 * total):
                break
            inlier = upright_plane(evidence.xyz[rest], settings.plane_tolerance_m, settings.max_tilt_deg)
            if inlier is None or inlier.sum() < settings.min_points:
                break
            found.append(rest[inlier])
            rest = rest[~inlier]
        covered = sum(len(patch) for patch in found)
        if covered < settings.min_planar_share * total:
            rejected.append({"centre": _rounded(evidence.xyz[member].mean(axis=0)), "points": total, "reason": "not flat"})
            continue
        patches.extend(found)
    return patches


def _merge_patches(evidence: RoomEvidence, patches: list[np.ndarray], settings: ScreenSettings) -> list[np.ndarray]:
    """Parts of one (possibly curved) panel become one: close, at one height, on one chord plane, bridged by points."""
    groups = sorted(patches, key=len, reverse=True)
    merged = True
    while merged:
        merged = False
        for first in range(len(groups)):
            for second in range(first + 1, len(groups)):
                if _same_panel(evidence, groups[first], groups[second], settings):
                    groups[first] = np.concatenate((groups[first], groups.pop(second)))
                    merged = True
                    break
            if merged:
                break
    return groups


def _same_panel(evidence: RoomEvidence, a: np.ndarray, b: np.ndarray, settings: ScreenSettings) -> bool:
    pa, pb = evidence.xyz[a], evidence.xyz[b]
    apart = np.maximum(pb.min(axis=0) - pa.max(axis=0), pa.min(axis=0) - pb.max(axis=0))
    if apart.max() > settings.merge_gap_m:
        return False
    together = np.vstack((pa, pb))
    centre, normal, rms = fit_plane(together)
    if rms > settings.curved_rms_m or abs(float(normal[2])) > math.sin(math.radians(settings.max_tilt_deg)):
        return False
    u, v, normal = panel_axes(normal)
    va, vb = np.percentile(pa @ v, [2, 98]), np.percentile(pb @ v, [2, 98])
    if min(va[1], vb[1]) - max(va[0], vb[0]) < 0.4 * min(va[1] - va[0], vb[1] - vb[0]):
        return False
    ua, ub = np.percentile(pa @ u, [2, 98]), np.percentile(pb @ u, [2, 98])
    if max(ua[1], ub[1]) - min(ua[0], ub[0]) > settings.width_m[1]:
        return False
    wanted = evidence.instance_ids(SCREEN_LABELS)
    ida, idb = _dominant_instance(evidence, a, wanted), _dominant_instance(evidence, b, wanted)
    if not (ida and idb and ida != idb):
        return True
    # halves of one large screen: the same tall extent, which two screens edge to edge do not share, or a portrait rest
    heights = (va[1] - va[0], vb[1] - vb[0])
    widths = (ua[1] - ua[0], ub[1] - ub[0])
    same_edges = abs(va[0] - vb[0]) <= 0.10 and abs(va[1] - vb[1]) <= 0.10
    if same_edges and min(heights) >= 0.70:
        return True
    if same_edges and min(width / height for width, height in zip(widths, heights)) < 0.8:
        return True
    # two instances: one only when one is a fragment or the panel bends (a curved monitor split by the association)
    if min(ua[1] - ua[0], ub[1] - ub[0]) < 0.30:
        return True
    cosine = abs(float(fit_plane(pa)[1] @ fit_plane(pb)[1]))
    return cosine < math.cos(math.radians(8.0))


def _dominant_instance(evidence: RoomEvidence, member: np.ndarray, among: np.ndarray | None = None) -> int | None:
    ids = evidence.instance[member]
    ids = ids[ids > 0] if among is None else ids[np.isin(ids, among)]
    if not len(ids) or len(ids) < 0.3 * len(member):
        return None
    values, counts = np.unique(ids, return_counts=True)
    return int(values[np.argmax(counts)])


def _main_span(values: np.ndarray, gap: float, minor_share: float = 0.20) -> tuple[float, float]:
    """Robust span of the values without small detached runs (the foot of a monitor, a stray blob beside it)."""
    ordered = np.sort(values)
    runs = np.split(ordered, np.flatnonzero(np.diff(ordered) > gap) + 1)
    largest = max(len(run) for run in runs)
    kept = np.concatenate([run for run in runs if len(run) >= min(minor_share * len(ordered), largest)])
    low, high = np.percentile(kept, [0.5, 99.5])
    return float(low), float(high)


def _panel_rows(
    along: np.ndarray, up: np.ndarray, width: tuple[float, float], height: tuple[float, float], share: float, step: float = 0.02,
) -> tuple[float, float]:
    """Top and bottom of the lines that span the panel; the narrow foot under it does not count."""
    keep = (along >= width[0]) & (along <= width[1]) & (up >= height[0]) & (up <= height[1])
    if keep.sum() < 10:
        return height
    along, up = along[keep], up[keep]
    count = max(int(math.ceil((height[1] - height[0]) / step)), 1)
    row = np.minimum(((up - height[0]) / step).astype(int), count - 1)
    wide = np.zeros(count, dtype=bool)
    for index in range(count):
        # two neighbouring lines together, sparse points never cover a single one
        values = along[np.abs(row - index) <= 1]
        if len(values) >= 4:
            first, last = np.percentile(values, [3, 97])
            wide[index] = last - first >= share * (width[1] - width[0])
    found = np.flatnonzero(wide)
    if not len(found) or (found[-1] - found[0] + 1) * step < max(0.14, 0.6 * (height[1] - height[0])):
        return height  # a foot is the smaller part: otherwise the lines are just sparsely measured
    low = height[0] if found[0] == 0 else height[0] + (found[0] + 1) * step
    high = height[1] if found[-1] == count - 1 else height[0] + found[-1] * step
    return float(low), float(high)


def _measured_rectangle(
    evidence: RoomEvidence, member: np.ndarray, centre: np.ndarray, u: np.ndarray, v: np.ndarray, normal: np.ndarray, rms: float,
    settings: ScreenSettings,
) -> tuple[float, float, float, float]:
    """(u0, u1, v0, v1): the labelled patch, completed by coplanar measured points connected to it."""
    relative = evidence.xyz[member] - centre
    u0, u1 = _main_span(relative @ u, 0.12)
    v0, v1 = _main_span(relative @ v, 0.06)
    v0, v1 = _panel_rows(relative @ u, relative @ v, (u0, u1), (v0, v1), settings.body_row_share)
    seed = (u0, u1, v0, v1)
    if settings.grow_m <= 0:
        return seed
    cell = settings.grow_cell_m
    a, b = relative @ u, relative @ v
    covered = (a >= u0) & (a < u1) & (b >= v0) & (b < v1)
    columns, rows = max(int(math.ceil((u1 - u0) / cell)), 1), max(int(math.ceil((v1 - v0) / cell)), 1)
    filled = np.zeros((columns, rows), dtype=bool)
    filled[
        np.minimum(((a[covered] - u0) / cell).astype(int), columns - 1),
        np.minimum(((b[covered] - v0) / cell).astype(int), rows - 1),
    ] = True
    # a densely labelled panel grows only into a measured frame, otherwise it would climb into what carries it
    dense = filled.mean() >= settings.complete_fill and _size_problem(u1 - u0, v1 - v0, settings) is None
    reach = settings.grow_m
    corners = np.asarray([centre + a * u + b * v for a in (u0 - reach, u1 + reach) for b in (v0 - reach, v1 + reach)])
    near = evidence.in_box(corners.min(axis=0) - 0.10, corners.max(axis=0) + 0.10, classes=(FURNITURE,))
    if not len(near):
        return seed
    relative = evidence.xyz[near] - centre
    a, b = relative @ u, relative @ v
    inside = np.abs(relative @ normal) <= max(settings.plane_tolerance_m, 1.5 * rms)
    inside &= (a >= u0 - reach) & (a <= u1 + reach) & (b >= v0 - reach) & (b <= v1 + reach)
    if not inside.any():
        return seed
    origin = (u0 - reach, v0 - reach)
    column = ((a[inside] - origin[0]) / cell).astype(int)
    row = ((b[inside] - origin[1]) / cell).astype(int)
    occupied = np.zeros((int((u1 - u0 + 2 * reach) / cell) + 2, int((v1 - v0 + 2 * reach) / cell) + 2), dtype=bool)
    occupied[column, row] = True
    seeded = np.zeros_like(occupied)
    seeded[
        int(reach / cell): int((u1 - u0 + reach) / cell) + 1,
        int(reach / cell): int((v1 - v0 + reach) / cell) + 1,
    ] = True
    linked, _count = ndimage.label(ndimage.binary_dilation(occupied | seeded, np.ones((3, 3))), np.ones((3, 3)))
    identifier = linked[seeded][0]
    # a dark screen: the measured bezel is a ring around the labelled middle and need not touch it
    seed_columns, seed_rows = np.nonzero(seeded)
    outside = occupied & ~ndimage.binary_dilation(seeded, np.ones((3, 3)))
    rings, _count = ndimage.label(ndimage.binary_dilation(outside, np.ones((3, 3))), np.ones((3, 3)))
    for number, window in enumerate(ndimage.find_objects(rings), start=1):
        if window is None:
            continue
        if window[0].start > seed_columns.min() or window[0].stop <= seed_columns.max():
            continue
        if window[1].start > seed_rows.min() or window[1].stop <= seed_rows.max():
            continue
        ring_columns, ring_rows = np.nonzero(outside & (rings == number))
        if not len(ring_columns):
            continue
        ring = (
            origin[0] + (ring_columns.min() + 0.5) * cell, origin[0] + (ring_columns.max() + 0.5) * cell,
            origin[1] + (ring_rows.min() + 0.5) * cell, origin[1] + (ring_rows.max() + 0.5) * cell,
        )
        plausible = _size_problem(ring[1] - ring[0], ring[3] - ring[2], settings) is None
        if plausible and _is_bezel(occupied, seeded, ring, origin, cell):
            return tuple(float(value) for value in ring)
    body = occupied & (linked == identifier)
    # an edge of the panel is a line of points; single stray points next to it do not move the edge

    def outwards(profile: np.ndarray, edge: float, start: float, step: int) -> float:
        """Walk from the labelled edge over measured lines; more than one empty line is the end of the panel."""
        position = int((edge - start) / cell)
        reached, empty = edge, 0
        position += step
        while 0 <= position < len(profile) and empty <= 1:
            if profile[position]:
                reached, empty = start + (position + 0.5) * cell, 0
            else:
                empty += 1
            position += step
        return reached

    columns, rows = body.sum(axis=1) >= 4, body.sum(axis=0) >= 4
    grown = (
        min(u0, outwards(columns, u0, origin[0], -1)), max(u1, outwards(columns, u1, origin[0], 1)),
        min(v0, outwards(rows, v0, origin[1], -1)), max(v1, outwards(rows, v1, origin[1], 1)),
    )
    if dense and not _is_bezel(occupied, seeded, grown, origin, cell):
        return seed
    grown = (
        grown[0], grown[1],
        *_panel_rows(a[inside], b[inside], grown[:2], grown[2:], settings.body_row_share),
    )
    for candidate in (grown, (grown[0], grown[1], v0, v1)):
        if _size_problem(candidate[1] - candidate[0], candidate[3] - candidate[2], settings) is None:
            return tuple(float(value) for value in candidate)
    return seed


def _is_bezel(
    occupied: np.ndarray, seeded: np.ndarray, grown: tuple[float, float, float, float], origin: tuple[float, float], cell: float,
) -> bool:
    """Whether the grown rectangle is a frame: its four edges are measured and the surface between is (almost) empty."""
    c0, c1 = (min(max(int((value - origin[0]) / cell), 0), occupied.shape[0] - 1) for value in grown[:2])
    r0, r1 = (min(max(int((value - origin[1]) / cell), 0), occupied.shape[1] - 1) for value in grown[2:])
    if c1 - c0 < 3 or r1 - r0 < 3:
        return False
    near = ndimage.binary_dilation(occupied, np.ones((3, 3)))
    sides = (near[c0, r0:r1 + 1], near[c1, r0:r1 + 1], near[c0:c1 + 1, r0], near[c0:c1 + 1, r1])
    if min(float(side.mean()) for side in sides) < 0.8:
        return False
    between = np.zeros_like(occupied)
    between[c0 + 2:c1 - 1, r0 + 2:r1 - 1] = True
    between &= ~ndimage.binary_dilation(seeded, np.ones((3, 3)))
    return bool(between.any()) and float(occupied[between].mean()) <= 0.3


def _size_problem(width: float, height: float, settings: ScreenSettings) -> str | None:
    if not settings.width_m[0] <= width <= settings.width_m[1]:
        return "width"
    if not settings.height_m[0] <= height <= settings.height_m[1]:
        return "height"
    aspect = width / height
    highest = settings.ultrawide_aspect if width <= 1.4 else settings.aspect[1]
    return None if settings.aspect[0] <= aspect <= highest else "aspect"


def _surface_below(
    evidence: RoomEvidence, base: np.ndarray, u: np.ndarray, normal: np.ndarray, width: float, settings: ScreenSettings,
) -> float | None:
    """Height of the horizontal surface under the bottom edge (desk, sideboard)."""
    reach = 0.5 * width + 0.25
    low = base - np.asarray((reach, reach, settings.stand_reach_m))
    high = base + np.asarray((reach, reach, 0.03))
    near = evidence.in_box(low, high, classes=(CLUTTER, FURNITURE, APPLIANCE))
    if len(near) < 20:
        return None
    flat_normal = np.asarray((normal[0], normal[1]))
    flat_normal /= max(float(np.linalg.norm(flat_normal)), 1e-12)
    relative = evidence.xyz[near] - base
    along, across = relative @ u, relative[:, :2] @ flat_normal
    keep = (np.abs(along) <= 0.5 * width) & (np.abs(across) <= 0.25)
    if keep.sum() < 20:
        return None
    z, along = evidence.xyz[near[keep], 2], along[keep]
    edges = np.arange(low[2], high[2] + 0.02, 0.02)
    counts, edges = np.histogram(z, bins=edges)
    best = int(np.argmax(counts))
    on_surface = (z >= edges[best] - 0.02) & (z <= edges[best + 1] + 0.02)
    bins = max(int(width / 0.10), 1)
    columns = np.unique(np.clip(((along[on_surface] + 0.5 * width) / width * bins).astype(int), 0, bins - 1))
    if len(columns) < 0.4 * bins:
        return None
    return float(np.median(z[on_surface]))


def _term_share(
    evidence: RoomEvidence, centre: np.ndarray, u: np.ndarray, v: np.ndarray, normal: np.ndarray, width: float, height: float,
) -> float:
    """Share of the cloud points on the panel that carry a screen term."""
    reach = np.abs(u) * 0.5 * width + np.abs(v) * 0.5 * height + 0.06
    near = evidence.in_box(centre - reach, centre + reach, cloud_only=True)
    relative = evidence.xyz[near] - centre
    on_panel = (
        (np.abs(relative @ u) <= 0.5 * width) & (np.abs(relative @ v) <= 0.5 * height) & (np.abs(relative @ normal) <= 0.05)
    )
    if not on_panel.any():
        return 0.0
    return float(np.mean(np.isin(evidence.fine[near[on_panel]], evidence.term_ids(SCREEN_LABELS))))


def _stand_mesh(
    evidence: RoomEvidence, base: np.ndarray, surface: float, u: np.ndarray, flat_normal: np.ndarray,
) -> Mesh | None:
    """The foot between surface and panel, only where points show it."""
    if base[2] - surface < 0.04:
        return None
    near = evidence.in_box(
        base - np.asarray((0.35, 0.35, base[2] - surface - 0.015)), base + np.asarray((0.35, 0.35, -0.01)),
        classes=(CLUTTER, FURNITURE),
    )
    if not len(near):
        return None
    relative = evidence.xyz[near] - base
    along, across = relative @ u, relative @ flat_normal
    keep = (np.abs(along) <= 0.15) & (across >= -0.15) & (across <= 0.02)
    if keep.sum() < 15:
        return None
    u0, u1 = np.percentile(along[keep], [5, 95])
    d0, d1 = np.percentile(across[keep], [5, 95])
    half_u, half_d = max(0.5 * (u1 - u0), 0.02), max(0.5 * (d1 - d0), 0.015)
    middle = base + 0.5 * (u0 + u1) * u + 0.5 * (d0 + d1) * flat_normal
    middle[2] = 0.5 * (surface + base[2])
    return box_mesh(middle, (u, flat_normal, np.asarray((0.0, 0.0, 1.0))), (half_u, half_d, 0.5 * (base[2] - surface)))


def _semantic(evidence: RoomEvidence, member: np.ndarray, width: float, height: float) -> str:
    votes: Counter[str] = Counter()
    ids, counts = np.unique(evidence.instance[member], return_counts=True)
    for identifier, count in zip(ids.tolist(), counts.tolist()):
        label = evidence.instance_labels.get(int(identifier))
        if label in SCREEN_LABELS:
            votes[label] += count
    terms, counts = np.unique(evidence.fine[member], return_counts=True)
    for term, count in zip(terms.tolist(), counts.tolist()):
        if term and evidence.fine_names[term - 1] in SCREEN_LABELS:
            votes[evidence.fine_names[term - 1]] += count
    precise = [(count, -SCREEN_LABELS.index(name), name) for name, count in votes.items() if name != "screen"]
    name = max(precise)[2] if precise else "screen"
    knows_monitors = "monitor" in evidence.fine_names or "monitor" in evidence.instance_labels.values()
    if name == "television" and (width < 1.0 or height < 0.6) and not knows_monitors:
        return "screen"  # a vocabulary without "monitor" calls every screen a television
    if name == "screen" and width >= 1.4:
        return "television"
    return name


def _rounded(values: Any) -> list[float]:
    return [round(float(value), 3) for value in values]


def _make_screen(
    evidence: RoomEvidence, member: np.ndarray, walls: list[LineString], rooms: list[Polygon], settings: ScreenSettings,
) -> tuple[Screen | None, str | None]:
    points = evidence.xyz[member]
    centre, normal, rms = fit_plane(points)
    if abs(float(normal[2])) > math.sin(math.radians(settings.max_tilt_deg)):
        return None, "not upright"
    gap, wall_at = nearest_line(centre[:2], walls)
    if wall_at is not None and gap <= settings.wall_gap_m and abs(float(normal[2])) <= math.sin(math.radians(3.0)):
        direction = line_direction(wall_at)
        across = np.asarray((-direction[1], direction[0], 0.0))
        if abs(float(across @ normal)) >= math.cos(math.radians(3.0)):
            normal = across  # a flush panel hangs parallel to its wall, the rest is depth noise of the picture
    u, v, normal = panel_axes(normal)
    u0, u1, v0, v1 = _measured_rectangle(evidence, member, centre, u, v, normal, rms, settings)
    width, height = u1 - u0, v1 - v0
    problem = _size_problem(width, height, settings)
    if problem:
        return None, problem
    front = centre + 0.5 * (u0 + u1) * u + 0.5 * (v0 + v1) * v
    if front[2] - 0.5 * height * float(v[2]) < -0.02:
        return None, "below the floor"
    if _dominant_instance(evidence, member, evidence.instance_ids(SCREEN_LABELS)) is None:
        if _term_share(evidence, front, u, v, normal, width, height) < settings.unlabelled_term_share:
            return None, "screen term on too little of the surface"
    return _panel_screen(evidence, member, front, u, v, normal, width, height, walls, rooms, settings)


def _panel_screen(
    evidence: RoomEvidence, member: np.ndarray, front: np.ndarray, u: np.ndarray, v: np.ndarray, normal: np.ndarray,
    width: float, height: float, walls: list[LineString], rooms: list[Polygon], settings: ScreenSettings,
    photo: Frame | None = None,
) -> tuple[Screen | None, str | None]:
    """The screen of a measured rectangle: facing, what carries it, stand and name."""
    room = next((item for item in rooms if item.buffer(0.10).contains(Point(front[:2]))), None)
    if rooms and room is None:
        return None, "outside the rooms (seen through or reflected in a pane)"
    distance, wall = nearest_line(front[:2], walls)
    parallel = False
    if wall is not None:
        parallel = abs(float(line_direction(wall) @ u[:2])) >= math.cos(math.radians(15.0))
    # the front looks into the room: away from the wall behind it, else towards the middle of the room
    if wall is not None and parallel and distance <= 0.5:
        foot = np.asarray(wall.interpolate(wall.project(Point(front[:2]))).coords[0])
        away = front[:2] - foot
    elif abs(float(normal[2])) >= math.sin(math.radians(3.0)):
        away = normal[:2] * math.copysign(1.0, float(normal[2]))  # a leaning screen faces upwards, to the viewer
    else:
        away = np.asarray(room.centroid.coords[0]) - front[:2] if room is not None else normal[:2]
    if photo is None and float(away @ normal[:2]) < 0:  # a photo frame already faces its cameras
        normal = -normal
    base = front - 0.5 * height * v
    surface = _surface_below(evidence, base, u, normal, width, settings)
    flat_normal = np.asarray((normal[0], normal[1], 0.0))
    flat_normal /= max(float(np.linalg.norm(flat_normal)), 1e-12)
    stand = None
    if surface is not None and base[2] - surface <= settings.stand_gap_m:
        stand = _stand_mesh(evidence, base, surface, u, flat_normal)
    flush = parallel and distance <= settings.wall_gap_m
    carried = surface is not None and (stand is not None or base[2] - surface <= 0.22)
    if flush and not (carried and width < 1.2):
        mounting, stand = "wall", None
    elif carried:
        mounting = "resting"
    elif flush or (parallel and distance <= 2 * settings.wall_gap_m):
        mounting = "wall"
    elif base[2] < 0.25:
        mounting = "standing"
    elif surface is not None:
        mounting = "resting"
    else:
        return None, "nothing carries it"
    thickness = settings.thickness_m
    if parallel and distance < thickness:
        thickness = max(distance, 0.01)  # never into the wall
    panel = box_mesh(front - 0.5 * thickness * normal, (u, v, normal), (0.5 * width, 0.5 * height, 0.5 * thickness))
    mesh = panel if stand is None else join_meshes((panel, stand))
    footprint = hull_of(mesh[0][:, :2])
    if footprint is None:
        return None, "degenerate"
    semantic = _semantic(evidence, member, width, height) if photo is None else _photo_semantic(photo)
    if semantic == "television" and mounting != "wall" and width < settings.small_screen_m:
        # an instance keeps the broad name "television" of the first vocabulary; the point terms are more precise
        if photo is not None:
            monitor = photo.labels.count("monitor") / max(photo.views, 1)
        else:
            monitor = float(np.mean(np.isin(evidence.fine[member], evidence.term_ids(("monitor",)))))
        semantic = "monitor" if monitor >= settings.monitor_term_share else "screen"
    return Screen(
        semantic=semantic, instance_id=_dominant_instance(evidence, member), centre=front, u=u, v=v,
        normal=normal, width=width, height=height, thickness=thickness, mounting=mounting,
        tilt_deg=math.degrees(math.asin(max(-1.0, min(1.0, float(normal[2]))))), mesh=mesh, footprint=footprint,
        points=len(member), has_stand=stand is not None,
    ), None


def _photo_semantic(photo: Frame) -> str:
    precise = Counter(name for name in photo.labels if name != "screen")
    name = max(precise, key=lambda item: (precise[item], -SCREEN_LABELS.index(item))) if precise else "screen"
    if photo.width >= 1.4 and name in ("screen", "monitor"):
        return "television"
    return name


def _with_photo_frames(
    evidence: RoomEvidence, measured: list[Screen], walls: list[LineString], rooms: list[Polygon], settings: ScreenSettings,
    frame_settings: ScreenFrameSettings, rejected: list[dict[str, Any]],
) -> tuple[list[Screen], list[dict[str, Any]]]:
    """Add what several photos agree on; it replaces a smaller or offset fragment measured in the cloud."""
    frames, refused = fit_screen_frames(evidence.screen_views, rooms, frame_settings)
    rejected.extend(refused)
    screens = list(measured)
    report: list[dict[str, Any]] = []
    placed: list[Frame] = []
    for photo in sorted(frames, key=lambda item: -item.views):
        # two fits of one panel, or a frame around two screens that the stronger single frames already explain
        if any(
            abs(float((photo.centre - other.centre) @ other.normal)) <= 0.20
            and abs(float((photo.centre - other.centre) @ other.u)) < 0.40 * (photo.width + other.width)
            and abs(float((photo.centre - other.centre) @ other.v)) < 0.40 * (photo.height + other.height)
            for other in placed
        ):
            rejected.append({"centre": _rounded(photo.centre), "points": 0, "reason": "photo frame: duplicate"})
            continue
        reach = np.abs(photo.u) * 0.5 * photo.width + np.abs(photo.v) * 0.5 * photo.height + 0.05
        near = evidence.in_box(photo.centre - reach, photo.centre + reach, classes=(FURNITURE,))
        relative = evidence.xyz[near] - photo.centre
        member = near[
            (np.abs(relative @ photo.normal) <= 0.06) & (np.abs(relative @ photo.u) <= 0.5 * photo.width)
            & (np.abs(relative @ photo.v) <= 0.5 * photo.height)
        ]
        screen, problem = _panel_screen(
            evidence, member, photo.centre, photo.u, photo.v, photo.normal, photo.width, photo.height, walls, rooms, settings,
            photo=photo,
        )
        if screen is None:
            rejected.append({"centre": _rounded(photo.centre), "points": len(member), "reason": f"photo frame: {problem}"})
            continue
        same = []
        for other in screens:
            offset = other.centre - photo.centre
            if (
                abs(float(offset @ photo.normal)) <= 0.22 and abs(float(offset @ photo.u)) <= 0.5 * photo.width
                and abs(float(offset @ photo.v)) <= 0.5 * photo.height + 0.10
            ):
                same.append(other)
        placed.append(photo)
        entry = {
            "centre": _rounded(photo.centre), "width_m": round(photo.width, 3), "height_m": round(photo.height, 3),
            "views": photo.views, "places": photo.places, "prompt_support": f"{photo.hit}/{photo.seen}",
        }
        if len(same) == 1:
            share = frame_settings.same_size_share
            if abs(same[0].width - photo.width) <= share * photo.width and abs(same[0].height - photo.height) <= share * photo.height:
                report.append({**entry, "used": "cloud panel agrees"})
                continue
        if same:
            screen.instance_id = max(same, key=lambda item: item.points).instance_id
            screen.points = max(screen.points, *(item.points for item in same))
            screens = [item for item in screens if all(item is not other for other in same)]
        screens.append(screen)
        report.append({**entry, "used": "replaces the cloud panel" if same else "added"})
    return screens, report


def detect_screens(
    evidence: RoomEvidence, *, rooms_m: list[Polygon], wall_lines_m: list[LineString], settings: ScreenSettings,
    reflection: ReflectionSettings, frames: ScreenFrameSettings | None = None,
) -> tuple[list[Screen], dict[str, Any]]:
    """All screens of one level, each once, and why the other candidates were rejected."""
    rejected: list[dict[str, Any]] = []
    if not settings.enabled:
        return [], {"rejected": rejected}
    seeds = np.flatnonzero(_seed_mask(evidence, settings, rejected))
    groups = _merge_patches(evidence, _patches(evidence, seeds, settings, rejected), settings)
    rooms = [room for room in rooms_m if not room.is_empty]
    screens: list[Screen] = []
    for member in groups:
        screen, problem = _make_screen(evidence, member, wall_lines_m, rooms, settings)
        if screen is None:
            rejected.append({"centre": _rounded(evidence.xyz[member].mean(axis=0)), "points": len(member), "reason": problem})
        else:
            screens.append(screen)
    photo_report: list[dict[str, Any]] = []
    if evidence.screen_views:
        screens, photo_report = _with_photo_frames(
            evidence, screens, wall_lines_m, rooms, settings, frames or ScreenFrameSettings(), rejected,
        )
    screens.sort(key=lambda item: -item.points)
    mirrored = reflection_of(
        [{"centre": item.centre, "size": max(item.width, item.height), "points": item.points} for item in screens],
        evidence.mirrors, reflection,
    )
    kept: list[Screen] = []
    for index, screen in enumerate(screens):
        if index in mirrored:
            rejected.append({"centre": _rounded(screen.centre), "points": screen.points, "reason": "mirror reflection"})
            continue
        if any(_duplicate(screen, other) for other in kept):
            rejected.append({"centre": _rounded(screen.centre), "points": screen.points, "reason": "duplicate"})
            continue
        kept.append(screen)
    return kept, {"rejected": rejected, "photo_frames": photo_report}


def _duplicate(screen: Screen, other: Screen) -> bool:
    """The same panel seen twice: parallel, less than 12 cm apart, covering most of the smaller one."""
    offset = screen.centre - other.centre
    if abs(float(offset @ other.normal)) > 0.12 or abs(float(screen.normal @ other.normal)) < math.cos(math.radians(25.0)):
        return False
    du, dv = abs(float(offset @ other.u)), abs(float(offset @ other.v))
    common_u = min(screen.width, other.width, 0.5 * (screen.width + other.width) - du)
    common_v = min(screen.height, other.height, 0.5 * (screen.height + other.height) - dv)
    if common_u <= 0 or common_v <= 0:
        return False
    return common_u * common_v >= 0.4 * min(screen.width * screen.height, other.width * other.height)
