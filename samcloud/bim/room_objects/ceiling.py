"""Objects on the ceiling: lamps, spots, detectors, ventilation grilles and unknown fixtures with evidence.

Seeds are the ceiling-equipment and HVAC points of the classified cloud and of the COLMAP-supported depth voxels, so
clouds classified before ceiling instances existed work as well; instance labels and fine terms, where present, only
make the type more precise.  Every object is attached to the ceiling measured around it (flat or sloped).

Not ceiling objects, each rejected with a reason: curtain rails and wall tops (long and narrow along a wall), lintels
(inside a wall), cabinet tops and wall clocks (do not reach the ceiling), glare, mirror images and noise.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

from samcloud.bim.room_objects.evidence import (
    APPLIANCE, CEILING, CEILING_EQUIPMENT, CEILING_LABELS, FURNITURE, HVAC, LIGHT_LABELS, VENT_LABEL, RoomEvidence,
)
from samcloud.bim.room_objects.geometry import (
    Mesh, ceiling_plane, line_direction, nearest_line, occupancy_outline, prism_mesh, rectangle_sides, reflection_of,
    regular_outline, voxel_components,
)
from samcloud.bim.room_objects.settings import CeilingSettings, ReflectionSettings

DISPLAY_NAMES = {
    "pendant light": "Pendelleuchte", "ceiling lamp": "Deckenleuchte", "spotlight": "Spot", "smoke detector": "Rauchmelder",
    "sprinkler": "Sprinklerkopf", "projector": "Beamer", "ventilation grille": "Lüftungsgitter", "ceiling fixture": "Deckenobjekt",
}
_SMALL = frozenset({"smoke detector", "sprinkler"})


@dataclass(slots=True)
class CeilingObject:
    semantic: str
    instance_id: int | None
    outline: Polygon
    shape: str
    plane: np.ndarray  # ceiling z = a x + b y + c
    drop: float
    top_drop: float
    z_min: float
    z_max: float
    mesh: Mesh
    points: int
    centre: np.ndarray
    size: float
    luminous: bool


def _plane_z(plane: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return plane[0] * xy[..., 0] + plane[1] * xy[..., 1] + plane[2]


def _candidates(evidence: RoomEvidence, settings: CeilingSettings) -> list[np.ndarray]:
    mask = np.isin(evidence.cls, (HVAC, CEILING_EQUIPMENT))
    ids = evidence.instance_ids((*CEILING_LABELS, VENT_LABEL))
    if len(ids):
        mask |= np.isin(evidence.instance, ids)
    low = settings.min_height_m
    if evidence.ceiling_m is not None:
        # a pendant or a sloped ceiling may reach further down, never below head height
        low = max(low, evidence.ceiling_m - settings.band_m - 0.7)
    mask &= evidence.xyz[:, 2] >= low
    seeds = np.flatnonzero(mask)
    labels, count = voxel_components(evidence.xyz[seeds], settings.link_m)
    if not count:
        return []
    order = np.argsort(labels, kind="stable")
    groups = np.split(seeds[order], np.cumsum(np.bincount(labels, minlength=count))[:-1])
    return [group for group in groups if len(group) >= settings.min_points]


def _local_ceiling(evidence: RoomEvidence, member: np.ndarray) -> np.ndarray | None:
    points = evidence.xyz[member]
    low, high = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    classes = (CEILING,)
    for reach in (0.75, 1.5, 2.5):
        near = evidence.in_box((*(low - reach), points[:, 2].min() - 0.10), (*(high + reach), math.inf), classes=classes)
        x, y = evidence.xyz[near, 0], evidence.xyz[near, 1]
        beside = (x < low[0] - 0.10) | (x > high[0] + 0.10) | (y < low[1] - 0.10) | (y > high[1] + 0.10)
        if beside.sum() >= 60:  # the ceiling beside the object, not its glare
            near = near[beside]
        plane = ceiling_plane(evidence.xyz[near]) if len(near) >= 30 else None
        if plane is not None:
            return plane
    return None if evidence.ceiling_m is None else np.asarray((0.0, 0.0, evidence.ceiling_m))


def _label_votes(evidence: RoomEvidence, member: np.ndarray) -> Counter[str]:
    votes: Counter[str] = Counter()
    wanted = (*CEILING_LABELS, VENT_LABEL)
    ids, counts = np.unique(evidence.instance[member], return_counts=True)
    for identifier, count in zip(ids.tolist(), counts.tolist()):
        label = evidence.instance_labels.get(int(identifier))
        if label in wanted:
            votes[label] += 2 * count  # an association over several views outweighs single point terms
    terms, counts = np.unique(evidence.fine[member], return_counts=True)
    for term, count in zip(terms.tolist(), counts.tolist()):
        if term and evidence.fine_names[term - 1] in wanted:
            votes[evidence.fine_names[term - 1]] += count
    return votes


def _semantic(
    votes: Counter[str], hvac_share: float, drop: float, size: float, luminous: bool, settings: CeilingSettings,
) -> str | None:
    best = votes.most_common(1)[0][0] if votes else None
    flat = drop <= settings.flat_drop_m
    if hvac_share >= 0.5 or (best == VENT_LABEL and not luminous):
        # lamp prompts fire on grilles as well, the HVAC class does not fire on lamps
        return VENT_LABEL if drop <= settings.vent_drop_m else None
    if best in _SMALL and size <= settings.detector_size_m and not luminous:
        return best
    if best == "projector" and not flat and size >= 0.15:
        return best
    if best == "pendant light" and not flat:
        return best
    if best == "spotlight" and (luminous or not flat):
        return best
    if luminous or (best in LIGHT_LABELS and not flat):
        # a flat dark cover that fired a lamp prompt stays an unknown fixture
        return "spotlight" if flat and size <= settings.glare_size_m else "ceiling lamp"
    return "ceiling fixture"


def _shade_circle(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """(centre, outer radius) of the lower rim of a round shade; glare at ceiling height does not widen it."""
    if len(points) < 30:
        return None
    xy = points[:, :2]
    offset = xy.mean(axis=0)
    local = xy - offset
    keep = np.ones(len(local), dtype=bool)
    centre, radius = np.zeros(2), 0.0
    for _ in range(4):  # algebraic circle fit, trimmed
        design = np.column_stack((2.0 * local[keep], np.ones(int(keep.sum()))))
        solution = np.linalg.lstsq(design, (local[keep] ** 2).sum(axis=1), rcond=None)[0]
        centre, radius = solution[:2], math.sqrt(max(float(solution[2] + solution[:2] @ solution[:2]), 0.0))
        residual = np.abs(np.linalg.norm(local - centre, axis=1) - radius)
        keep = residual <= max(2.5 * float(np.median(residual)), 0.01)
        if keep.sum() < 20:
            return None
    distance = np.linalg.norm(local - centre, axis=1)
    sectors = np.floor((np.arctan2(*(local[keep] - centre).T[::-1]) + math.pi) / (2 * math.pi) * 12).astype(int) % 12
    if len(np.unique(sectors)) < 8 or float(np.median(residual)) > 0.10 * radius:
        return None
    return centre + offset, float(np.percentile(distance[keep], 90))


def _body_mesh(
    points: np.ndarray, drops: np.ndarray, outline: Polygon, plane: np.ndarray, top_drop: float, drop: float,
    settings: CeilingSettings,
) -> Mesh | None:
    """Plate at the measured bottom plus, for a hanging body, the narrower part that joins it to the ceiling."""
    from samcloud.bim.room_objects.geometry import join_meshes

    def level(value: float):
        return lambda xy: _plane_z(plane, xy) - value

    upper = None
    if drop - top_drop > 0.09:
        split = 0.5 * (drop + top_drop)
        above = points[drops < split]
        if len(above) >= settings.min_points // 2:
            upper = occupancy_outline(above[:, :2], settings.cell_m, settings.density_share)
            if upper is not None:
                upper = upper.intersection(outline)
                upper = upper if isinstance(upper, Polygon) and upper.area < 0.8 * outline.area and upper.area > 1e-4 else None
    if upper is None:
        return prism_mesh(outline, level(drop), level(top_drop))
    body = prism_mesh(outline, level(drop), level(split))
    neck = prism_mesh(upper, level(split), level(top_drop))
    return None if body is None else body if neck is None else join_meshes((body, neck))


def _make_object(
    evidence: RoomEvidence, member: np.ndarray, rooms: list[Polygon], walls: list[LineString], windows: list[LineString],
    wall_area: Any, settings: CeilingSettings,
) -> tuple[CeilingObject | None, str | None]:
    plane = _local_ceiling(evidence, member)
    if plane is None:
        return None, "no ceiling measured"
    drops = _plane_z(plane, evidence.xyz[member]) - evidence.xyz[member, 2]
    in_room = (drops >= -settings.above_tolerance_m) & (drops <= settings.band_m + settings.pendant_gap_m)
    member, drops = member[in_room], drops[in_room]
    if len(member) < settings.min_points:
        return None, "too few points below the ceiling"
    points = evidence.xyz[member]
    outline = occupancy_outline(points[:, :2], settings.cell_m, settings.density_share)
    if outline is None or outline.area < settings.min_area_m2:
        return None, "too small"
    body = shapely.contains_xy(outline.buffer(settings.cell_m), points[:, 0], points[:, 1])
    if body.sum() < 0.5 * settings.min_points:
        return None, "too few points on the body"
    member, drops, points = member[body], drops[body], points[body]
    outline, shape = regular_outline(outline)
    long_side, short_side, axis = rectangle_sides(outline)
    if long_side > settings.max_size_m:
        return None, "too large"
    centre_xy = np.asarray(outline.centroid.coords[0])
    if rooms and not any(room.buffer(0.05).contains(Point(centre_xy)) for room in rooms):
        return None, "outside the rooms"
    if wall_area is not None and wall_area.intersection(outline).area > 0.3 * outline.area:
        return None, "inside a wall (lintel)"
    top, middle, bottom = (float(value) for value in np.percentile(drops, [3, 50, 95]))
    votes = _label_votes(evidence, member)
    best = votes.most_common(1)[0][0] if votes else None
    wall_distance, wall = nearest_line(centre_xy, walls)
    gap = settings.attach_gap_m
    if best == "pendant light":
        gap = settings.pendant_gap_m
    elif wall_distance >= settings.pendant_wall_m and len(member) >= settings.pendant_min_points:
        if best == "ceiling lamp":
            gap = settings.pendant_gap_m  # a shade on a thin neck or cord the scan did not see
        elif long_side >= 0.20:
            lamp = evidence.brightness[member[evidence.cls[member] == CEILING_EQUIPMENT]]
            lamp = lamp[np.isfinite(lamp)]
            if len(lamp) >= 0.5 * settings.pendant_min_points and float(np.percentile(lamp, 90)) >= settings.luminous_level:
                gap = settings.neck_gap_m  # the same for a lamp that is on but has no label
    if top > gap:
        return None, "does not reach the ceiling"
    along_wall = wall is not None and abs(float(line_direction(wall) @ axis)) >= math.cos(math.radians(15.0))
    narrow = long_side / max(short_side, 1e-6) >= settings.rail_aspect
    if narrow and along_wall and wall_distance <= settings.rail_wall_m:
        return None, "rail or wall top along a wall"
    if narrow or short_side <= settings.rail_piece_m < long_side:
        for window in windows:
            along_window = abs(float(line_direction(window) @ axis)) >= math.cos(math.radians(15.0))
            if not along_window:
                continue
            window_middle, direction = window.centroid, line_direction(window)
            half = 0.5 * window.length + settings.curtain_reach_m  # a curtain parks beside its window
            rail = LineString([
                (window_middle.x - half * direction[0], window_middle.y - half * direction[1]),
                (window_middle.x + half * direction[0], window_middle.y + half * direction[1]),
            ])
            if outline.distance(rail) <= settings.curtain_window_m:
                return None, "curtain rail at a window"
    # spots with a halo are flush; a shade that reaches well below the ceiling is not, whatever its glare does
    flat = middle <= settings.flat_drop_m and (bottom <= 0.15 or long_side <= settings.glare_size_m)
    if not flat and wall is not None and outline.distance(wall) <= settings.wall_touch_m and bottom > settings.vent_drop_m:
        return None, "on the wall, not on the ceiling"
    if not flat:
        x0, y0, x1, y1 = outline.bounds
        ceiling = float(_plane_z(plane, centre_xy))
        under = evidence.in_box(
            (x0, y0, ceiling - bottom - 0.25), (x1, y1, ceiling - bottom + 0.05),
            classes=(FURNITURE, APPLIANCE),
        )
        if len(under) and int(shapely.contains_xy(outline, evidence.xyz[under, 0], evidence.xyz[under, 1]).sum()) > len(member):
            return None, "top of furniture"
    if not flat and shape == "circle":
        rim = _shade_circle(points[drops >= 0.5 * bottom])
        if rim is not None and rim[1] < 0.5 * long_side:
            centre_xy, radius = rim
            outline = Point(centre_xy).buffer(radius, quad_segs=6)
            long_side = 2 * radius
    x0, y0, x1, y1 = outline.bounds
    height = float(_plane_z(plane, centre_xy))
    # the saturated middle of a lamp that is on is usually classified as ceiling: every cloud point on the body counts
    # a hanging lamp seen in pieces: its bright hub may lie a little beside the piece
    margin = 0.03 if flat else 0.15
    on_body = evidence.in_box(
        (x0 - margin, y0 - margin, height - bottom - 0.10), (x1 + margin, y1 + margin, height + 0.05), cloud_only=True,
    )
    on_body = on_body[shapely.contains_xy(outline.buffer(margin), evidence.xyz[on_body, 0], evidence.xyz[on_body, 1])]
    luminous = False
    hvac_share = float(np.mean(evidence.cls[member] == HVAC))
    # three readings: the whole body (saturated middle), and the equipment points alone (small bulbs in a dim halo)
    equipment = on_body[evidence.cls[on_body] == CEILING_EQUIPMENT]
    readings = [(member, 5), (equipment, 5)] + ([(on_body, 10)] if flat or hvac_share < 0.5 else [])
    for sample, least in readings:
        colours = evidence.brightness[sample]
        colours = colours[np.isfinite(colours)]
        luminous |= len(colours) >= least and float(np.percentile(colours, 90)) >= settings.luminous_level
    solidity = outline.area / outline.convex_hull.area
    unlabelled = not any(votes[name] for name in ("pendant light", "projector"))
    if flat and luminous and hvac_share < 0.5:
        core = occupancy_outline(points[:, :2], settings.cell_m, settings.spot_core_share)
        core_size = rectangle_sides(core)[0] if core is not None else math.inf
        if core_size <= settings.glare_size_m and core.area >= settings.min_area_m2 and long_side <= 2 * settings.glare_size_m:
            radius = math.sqrt(core.area / math.pi)
            if 2 * radius > settings.spot_max_m and not votes["spotlight"]:
                return None, "glare on the ceiling"
            outline, shape = core.centroid.buffer(radius, quad_segs=6), "circle"
            centre_xy, long_side = np.asarray(outline.centroid.coords[0]), 2 * radius
        elif (
            long_side > settings.glare_size_m and shape == "outline" and solidity < 0.85 and unlabelled
            and bottom <= settings.flat_drop_m + 0.01  # glare has no body below the ceiling
        ):
            return None, "glare on the ceiling"
    if flat and luminous and hvac_share < 0.5 and long_side > settings.glare_size_m and bottom > settings.flat_drop_m + 0.01:
        flat = False  # a lamp body under a glare that fills the ceiling around it
    semantic = _semantic(votes, hvac_share, middle if flat else bottom, long_side, luminous, settings)
    if semantic is None:
        return None, "HVAC points that are no grille"
    if flat:
        top_drop = 0.0
        drop = min(max(float(np.percentile(drops, 75)), settings.flat_depth_m), settings.flat_drop_m)
    else:
        # an unseen cord is not invented: only a top within reach of the ceiling is closed up to it
        top_drop = 0.0 if top <= settings.neck_gap_m else top
        drop = max(bottom, top_drop + settings.flat_depth_m)
    mesh = _body_mesh(points, drops, outline, plane, top_drop, drop, settings)
    if mesh is None:
        return None, "degenerate"
    ids = evidence.instance[member]
    ids = ids[ids > 0]
    instance_id = None
    if len(ids) >= 0.3 * len(member):
        values, counts = np.unique(ids, return_counts=True)
        instance_id = int(values[np.argmax(counts)])
    z = float(_plane_z(plane, centre_xy)) - 0.5 * (top_drop + drop)
    return CeilingObject(
        semantic=semantic, instance_id=instance_id, outline=outline, shape=shape, plane=plane, drop=drop, top_drop=top_drop,
        z_min=float(mesh[0][:, 2].min()), z_max=float(mesh[0][:, 2].max()), mesh=mesh, points=len(member),
        centre=np.asarray((*centre_xy, z)), size=long_side, luminous=luminous,
    ), None


def _unreported_reflections(
    found: list[CeilingObject], rooms: list[Polygon], walls: list[LineString], settings: ReflectionSettings,
) -> set[int]:
    """Reflections of lamps in a mirror the scan did not report.

    Clouds from before the mirror report still hold them: a lamp of the same size straight across a wall, in another
    room, with only a sliver of the points of the real one.  Two real lamps in twin rooms both carry their points.
    """
    weak: set[int] = set()
    for index, faint in enumerate(found):
        for real in found:
            if real is faint or not (real.luminous and faint.luminous) or min(real.size, faint.size) < 0.30:
                continue
            if faint.points > settings.twin_point_share * real.points:
                continue
            if abs(real.size - faint.size) > settings.twin_size_share * real.size:
                continue
            link = LineString((faint.centre[:2], real.centre[:2]))
            if link.length > 2 * settings.lateral_m:
                continue
            if any(room.contains(Point(faint.centre[:2])) and room.contains(Point(real.centre[:2])) for room in rooms):
                continue
            direction = line_direction(link)
            middle = link.interpolate(0.5, normalized=True)
            for wall in walls:
                across = abs(float(line_direction(wall) @ direction)) <= math.sin(math.radians(5.0))
                if across and wall.intersects(link) and wall.distance(middle) <= settings.twin_wall_m:
                    weak.add(index)
                    break
    return weak


def detect_ceiling_objects(
    evidence: RoomEvidence, *, rooms_m: list[Polygon], wall_lines_m: list[LineString], wall_area_m: Any,
    settings: CeilingSettings, reflection: ReflectionSettings, window_lines_m: list[LineString] | None = None,
) -> tuple[list[CeilingObject], dict[str, Any]]:
    """All ceiling objects of one level, each once, and why the other candidates were rejected."""
    rejected: list[dict[str, Any]] = []
    if not settings.enabled:
        return [], {"rejected": rejected}
    found: list[CeilingObject] = []
    for member in _candidates(evidence, settings):
        item, problem = _make_object(evidence, member, rooms_m, wall_lines_m, window_lines_m or [], wall_area_m, settings)
        if item is None:
            centre = [round(float(value), 3) for value in evidence.xyz[member].mean(axis=0)]
            rejected.append({"centre": centre, "points": len(member), "reason": problem})
        else:
            found.append(item)
    found.sort(key=lambda item: -item.points)
    mirrored = reflection_of(
        [{"centre": item.centre, "size": item.size, "points": item.points} for item in found], evidence.mirrors, reflection,
    )
    weak_twins = _unreported_reflections(found, rooms_m, wall_lines_m, reflection)
    kept: list[CeilingObject] = []
    for index, item in enumerate(found):
        reason = "mirror reflection" if index in mirrored else None
        if index in weak_twins:
            reason = "mirror reflection (weak twin across a wall)"
        if reason is None and any(
            other.outline.intersection(item.outline).area > 0.3 * min(other.outline.area, item.outline.area) for other in kept
        ):
            reason = "duplicate"
        if reason:
            rejected.append({"centre": [round(float(v), 3) for v in item.centre], "points": item.points, "reason": reason})
        else:
            kept.append(item)
    kept.sort(key=lambda item: (round(float(item.centre[0]), 2), round(float(item.centre[1]), 2)))
    return kept, {"rejected": rejected}
