"""Synthetic point clouds for the structure analysis: room floor / ceiling, wall heights, columns, beams, stairs.

Every scene is rotated off the axes, carries measurement noise and partial occlusion.  Positive cases measure the detected
dimensions; negative cases (furniture, tall cabinets, door frames, lamps, lowered ceilings, bulkheads, furniture with steps,
scattered mislabelled points) must not produce any element.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import LineString, Polygon, box

from samcloud.ortho.vector.structure.analysis import StructureAnalysis, analyse_structure
from samcloud.ortho.vector.structure.beams import detect_beams
from samcloud.ortho.vector.structure.columns import detect_columns
from samcloud.ortho.vector.structure.common import BEAM, CEILING, COLUMN, FLOOR, STAIRS, WALL, LevelPoints, StructureSettings
from samcloud.ortho.vector.structure.levels import room_levels, wall_heights
from samcloud.ortho.vector.structure.stairs import detect_stairs

THETA = math.radians(17.0)
H = 2.60
DENSITY = 900.0  # points per m2
NOISE = 0.006
SETTINGS = StructureSettings()


def rot(xy: np.ndarray, theta: float = THETA) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray(xy, dtype=float) @ np.asarray([[c, s], [-s, c]])


def rot_polygon(polygon: Polygon, theta: float = THETA) -> Polygon:
    return affinity.rotate(polygon, theta, origin=(0.0, 0.0), use_radians=True)


class Cloud:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.parts: list[np.ndarray] = []
        self.codes: list[np.ndarray] = []

    def _add(self, xyz: np.ndarray, code: int) -> None:
        self.parts.append(xyz)
        self.codes.append(np.full(len(xyz), code, dtype=np.uint16))

    def horizontal(self, x0: float, y0: float, x1: float, y1: float, z: float, code: int, keep=None) -> None:
        n = int(DENSITY * abs(x1 - x0) * abs(y1 - y0))
        xy = np.column_stack((self.rng.uniform(x0, x1, n), self.rng.uniform(y0, y1, n)))
        if keep is not None:
            xy = xy[keep(xy)]
        zz = z + self.rng.normal(0, NOISE, len(xy))
        self._add(np.column_stack((rot(xy), zz)), code)

    def sloped(self, x0: float, y0: float, x1: float, y1: float, z_of, code: int) -> None:
        n = int(DENSITY * abs(x1 - x0) * abs(y1 - y0))
        xy = np.column_stack((self.rng.uniform(x0, x1, n), self.rng.uniform(y0, y1, n)))
        zz = z_of(xy) + self.rng.normal(0, NOISE, n)
        self._add(np.column_stack((rot(xy), zz)), code)

    def vertical(self, a: tuple[float, float], b: tuple[float, float], z0: float, z1: float, code: int, keep=None) -> None:
        a_, b_ = np.asarray(a, float), np.asarray(b, float)
        length = float(np.linalg.norm(b_ - a_))
        n = int(DENSITY * length * (z1 - z0))
        t = self.rng.uniform(0, 1, n)
        z = self.rng.uniform(z0, z1, n)
        xy = a_ + np.outer(t, b_ - a_)
        normal = np.asarray((-(b_ - a_)[1], (b_ - a_)[0])) / max(length, 1e-9)
        xy = xy + np.outer(self.rng.normal(0, NOISE, n), normal)
        pts = np.column_stack((xy, z))
        if keep is not None:
            pts = pts[keep(pts)]
        self._add(np.column_stack((rot(pts[:, :2]), pts[:, 2])), code)

    def box_faces(self, x0: float, y0: float, x1: float, y1: float, z0: float, z1: float, code: int, faces=(0, 1, 2, 3), top: bool = False) -> None:
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for index in faces:
            self.vertical(corners[index], corners[(index + 1) % 4], z0, z1, code)
        if top:
            self.horizontal(x0, y0, x1, y1, z1, code)

    def cylinder(self, cx: float, cy: float, radius: float, z0: float, z1: float, code: int, arc=(0.0, 2 * math.pi)) -> None:
        n = int(DENSITY * radius * (arc[1] - arc[0]) * (z1 - z0))
        phi = self.rng.uniform(arc[0], arc[1], n)
        r = radius + self.rng.normal(0, NOISE, n)
        xy = np.column_stack((cx + r * np.cos(phi), cy + r * np.sin(phi)))
        self._add(np.column_stack((rot(xy), self.rng.uniform(z0, z1, n))), code)

    def scatter(self, n: int, x0: float, y0: float, x1: float, y1: float, z0: float, z1: float, code: int) -> None:
        xy = np.column_stack((self.rng.uniform(x0, x1, n), self.rng.uniform(y0, y1, n)))
        self._add(np.column_stack((rot(xy), self.rng.uniform(z0, z1, n))), code)

    def points(self, storey: float = H) -> LevelPoints:
        return LevelPoints(np.vstack(self.parts), np.concatenate(self.codes), storey)


def room_shell(cloud: Cloud, x1: float = 6.0, y1: float = 4.0, ceiling: float = H, occlude: bool = True) -> None:
    """Floor, ceiling and the four inner wall faces of a room (0, 0)-(x1, y1); a sofa-sized hole in the floor evidence."""
    holes = (lambda xy: ~((xy[:, 0] > 0.5) & (xy[:, 0] < 2.5) & (xy[:, 1] > 0.3) & (xy[:, 1] < 1.2))) if occlude else None
    cloud.horizontal(0, 0, x1, y1, 0.0, FLOOR, keep=holes)
    cloud.horizontal(0, 0, x1, y1, ceiling, CEILING)
    corners = [(0, 0), (x1, 0), (x1, y1), (0, y1)]
    for index in range(4):
        cloud.vertical(corners[index], corners[(index + 1) % 4], 0.0, ceiling, WALL,
                       keep=(lambda p: ~((p[:, 2] < 0.9) & (self_random(p) < 0.6))) if occlude else None)


def self_random(points: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-random value per point (occlusion of the lower wall by furniture)."""
    return (np.sin(points[:, 0] * 12.9898 + points[:, 1] * 78.233) * 43758.5453) % 1.0


def wall_bodies(x1: float = 6.0, y1: float = 4.0, t: float = 0.20) -> list[Polygon]:
    return [rot_polygon(item) for item in (box(-t, -t, x1 + t, 0), box(x1, -t, x1 + t, y1 + t), box(-t, y1, x1 + t, y1 + t), box(-t, -t, 0, y1 + t))]


def room(x1: float = 6.0, y1: float = 4.0) -> Polygon:
    return rot_polygon(box(0, 0, x1, y1))


# ------------------------------------------------------------------------------------------------ room floor / ceiling

def test_flat_ceiling_and_floor_are_measured_per_room() -> None:
    cloud = Cloud(1)
    room_shell(cloud, ceiling=2.47)
    cloud.box_faces(2.0, 2.0, 2.4, 2.4, 2.20, 2.47, CEILING, top=False)  # a pendant lamp mislabelled as ceiling
    levels = room_levels(cloud.points(), [room()], SETTINGS)
    assert levels[0].ceiling_source == "points"
    assert levels[0].ceiling_m == pytest.approx(2.47, abs=0.01)
    assert levels[0].floor_m == pytest.approx(0.0, abs=0.01)
    assert levels[0].ceiling_plane is None


def test_sloped_ceiling_gets_a_plane() -> None:
    cloud = Cloud(2)
    slope = math.radians(15.0)
    cloud.horizontal(0, 0, 6, 4, 0.0, FLOOR)
    cloud.sloped(0, 0, 6, 4, lambda xy: 2.0 + xy[:, 1] * math.tan(slope), CEILING)
    levels = room_levels(cloud.points(storey=2.6), [room()], SETTINGS)
    level = levels[0]
    assert level.ceiling_source == "plane"
    assert level.slope_deg == pytest.approx(15.0, abs=0.5)
    # the plane reproduces the true height at a corner of the (rotated) room
    corner = rot(np.asarray([[6.0, 4.0]]))
    assert level.ceiling_at(corner, 2.6)[0] == pytest.approx(2.0 + 4.0 * math.tan(slope), abs=0.03)


def test_lowered_room_ceiling_and_missing_ceiling() -> None:
    cloud = Cloud(3)
    room_shell(cloud, ceiling=2.30)
    cloud.horizontal(7, 0, 9, 4, 0.0, FLOOR)  # second room without any ceiling evidence
    rooms = [room(), rot_polygon(box(7, 0, 9, 4))]
    levels = room_levels(cloud.points(), rooms, SETTINGS)
    assert levels[0].ceiling_m == pytest.approx(2.30, abs=0.01)
    assert levels[1].ceiling_source == "level" and levels[1].ceiling_m is None


# ------------------------------------------------------------------------------------------------------- wall heights

def test_full_height_wall_reaches_the_room_ceiling() -> None:
    cloud = Cloud(4)
    room_shell(cloud, ceiling=2.52)
    points = cloud.points()
    levels = room_levels(points, [room()], SETTINGS)
    [height] = wall_heights(points, [(rot(np.asarray([[0.0, -0.1]]))[0], rot(np.asarray([[6.0, -0.1]]))[0], 0.2)], [room()], levels, H, SETTINGS)
    assert height.source == "room_ceiling"
    assert height.height_m == pytest.approx(2.52, abs=0.01)


def test_half_height_wall_with_free_space_above() -> None:
    cloud = Cloud(5)
    room_shell(cloud, x1=8.0)
    # parapet / knee wall across the room at x = 4 (0.15 m thick, 1.05 m high, both faces + top)
    cloud.box_faces(3.925, 0.0, 4.075, 4.0, 0.0, 1.05, WALL, faces=(1, 3), top=True)
    points = cloud.points()
    rooms = [rot_polygon(box(0, 0, 3.925, 4)), rot_polygon(box(4.075, 0, 8, 4))]
    levels = room_levels(points, rooms, SETTINGS)
    span = (rot(np.asarray([[4.0, 0.0]]))[0], rot(np.asarray([[4.0, 4.0]]))[0], 0.15)
    [height] = wall_heights(points, [span], rooms, levels, H, SETTINGS)
    assert height.source == "points"
    assert height.height_m == pytest.approx(1.05, abs=0.03)


def test_wall_hidden_above_by_cabinets_keeps_the_ceiling_height() -> None:
    cloud = Cloud(6)
    cloud.horizontal(0, 0, 6, 4, 0.0, FLOOR)
    cloud.horizontal(0, 0, 6, 4, H, CEILING)
    cloud.vertical((0, 0), (6, 0), 0.0, 1.5, WALL)  # wall points only up to 1.5 m ...
    cloud.vertical((0, 0.35), (6, 0.35), 1.5, H, 117)  # ... cabinet fronts (furniture class) hide the rest up to the ceiling
    points = cloud.points()
    levels = room_levels(points, [room()], SETTINGS)
    span = (rot(np.asarray([[0.0, -0.1]]))[0], rot(np.asarray([[6.0, -0.1]]))[0], 0.2)
    # widen the band so the cabinet fronts count as occupied space above the wall top
    [height] = wall_heights(points, [(span[0], span[1], 0.9)], [room()], levels, H, SETTINGS)
    assert height.source == "ceiling_occluded"
    assert height.height_m == pytest.approx(H, abs=0.02)


# ------------------------------------------------------------------------------------------------------------ columns

def _columns(cloud: Cloud, furniture: list[Polygon] | None = None, openings: list[Polygon] | None = None):
    points = cloud.points()
    rooms = [room(8.0, 6.0)]
    levels = room_levels(points, rooms, SETTINGS)
    from shapely.ops import unary_union

    return detect_columns(points, rooms, levels, unary_union(wall_bodies(8.0, 6.0)), openings or [], furniture or [], H, SETTINGS)


def test_rect_and_round_columns_are_detected_with_partial_occlusion() -> None:
    cloud = Cloud(7)
    room_shell(cloud, 8.0, 6.0)
    cloud.box_faces(2.0, 2.0, 2.30, 2.40, 0.0, H, COLUMN, faces=(0, 1, 2))  # one face hidden
    cloud.cylinder(5.5, 3.5, 0.18, 0.0, H, WALL, arc=(0.0, math.radians(240)))  # 240 deg visible, wall class
    columns, rejected = _columns(cloud)
    assert len(columns) == 2, rejected
    rect = next(item for item in columns if item.shape == "rect")
    circle = next(item for item in columns if item.shape == "circle")
    assert sorted((rect.width_m, rect.depth_m)) == pytest.approx([0.30, 0.40], abs=0.03)
    assert np.hypot(*(np.asarray(rect.center) - rot(np.asarray([[2.15, 2.20]]))[0])) < 0.03
    assert circle.width_m == pytest.approx(0.36, abs=0.03)
    assert np.hypot(*(np.asarray(circle.center) - rot(np.asarray([[5.5, 3.5]]))[0])) < 0.03
    assert rect.z_top_m == pytest.approx(H, abs=0.02)


def test_furniture_lamps_cabinets_and_door_frames_are_no_columns() -> None:
    cloud = Cloud(8)
    room_shell(cloud, 8.0, 6.0)
    cloud.box_faces(1.0, 4.5, 1.6, 5.1, 0.0, 2.0, WALL, top=True)  # tall cabinet mislabelled as wall, below the ceiling
    cloud.box_faces(3.0, 3.0, 3.6, 3.6, 0.0, H, 117)  # wardrobe to the ceiling, furniture class
    cloud.cylinder(5.0, 1.5, 0.05, 0.0, 1.5, WALL)  # floor lamp pole ...
    cloud.cylinder(5.0, 1.5, 0.25, 1.5, 1.9, WALL)  # ... and its shade
    cloud.cylinder(6.5, 4.0, 0.06, 0.0, H, WALL)  # thin pole: below the minimum section after noise? diameter 0.12 > 0.08 -> must be in furniture
    cloud.box_faces(4.0, 0.05, 4.10, 0.15, 0.0, 2.05, WALL)  # door frame jamb at the wall
    cloud.box_faces(1.5, 1.5, 2.0, 2.0, 0.0, H, WALL)  # built-in tall cabinet to the ceiling, standing in a furniture footprint
    furniture = [rot_polygon(box(1.45, 1.45, 2.05, 2.05)), rot_polygon(box(6.3, 3.8, 6.7, 4.2))]
    openings = [rot_polygon(box(3.6, -0.1, 4.6, 0.1))]
    columns, rejected = _columns(cloud, furniture, openings)
    assert columns == [], [item.evidence for item in columns]
    reasons = {item["reason"] for item in rejected}
    assert {"height", "in_furniture"} <= reasons


# -------------------------------------------------------------------------------------------------------------- beams

def _beam_scene(seed: int, extra=None, beam: bool = True) -> tuple[list, list]:
    cloud = Cloud(seed)
    width, x_beam, bottom = 0.30, 3.0, 2.25
    keep = (lambda xy: ~((xy[:, 0] > x_beam - width / 2) & (xy[:, 0] < x_beam + width / 2))) if beam else None
    cloud.horizontal(0, 0, 6, 4, 0.0, FLOOR)
    cloud.horizontal(0, 0, 6, 4, H, CEILING, keep=keep)
    corners = [(0, 0), (6, 0), (6, 4), (0, 4)]
    for index in range(4):
        cloud.vertical(corners[index], corners[(index + 1) % 4], 0.0, H, WALL)
    if beam:
        cloud.horizontal(x_beam - width / 2, 0, x_beam + width / 2, 4, bottom, BEAM)
        cloud.vertical((x_beam - width / 2, 0), (x_beam - width / 2, 4), bottom, H, CEILING)
        cloud.vertical((x_beam + width / 2, 0), (x_beam + width / 2, 4), bottom, H, CEILING)
    if extra is not None:
        extra(cloud)
    points = cloud.points()
    rooms = [room()]
    levels = room_levels(points, rooms, SETTINGS)
    from shapely.ops import unary_union

    return detect_beams(points, rooms, levels, unary_union(wall_bodies()), [], H, SETTINGS)


def test_beam_spanning_wall_to_wall_is_detected() -> None:
    beams, rejected = _beam_scene(9)
    assert len(beams) == 1, rejected
    item = beams[0]
    assert item.width_m == pytest.approx(0.30, abs=0.10)
    assert item.depth_m == pytest.approx(0.35, abs=0.03)
    assert item.z_bottom_m == pytest.approx(2.25, abs=0.02)
    length = float(np.hypot(*(np.asarray(item.axis[1]) - np.asarray(item.axis[0]))))
    assert length == pytest.approx(4.0, abs=0.15)
    direction = np.asarray(item.axis[1]) - np.asarray(item.axis[0])
    expected = rot(np.asarray([[0.0, 1.0]]))[0]
    assert abs(abs(float(direction @ expected)) / length - 1.0) < 0.01


def test_lowered_area_bulkhead_and_lamp_are_no_beams() -> None:
    def extras(cloud: Cloud) -> None:
        cloud.horizontal(0.5, 0.5, 2.5, 2.0, 2.30, CEILING)  # lowered ceiling area (2 x 1.5 m)
        cloud.horizontal(3.5, 3.6, 5.8, 4.0, 2.35, CEILING)  # bulkhead / duct box along the wall
        cloud.horizontal(4.5, 1.5, 4.8, 1.8, 2.40, CEILING)  # compact lamp / smoke detector

    # the original ceiling has holes where the extras hang lower
    cloud = Cloud(10)
    lowered = lambda xy: ~(((xy[:, 0] > 0.5) & (xy[:, 0] < 2.5) & (xy[:, 1] > 0.5) & (xy[:, 1] < 2.0))  # noqa: E731
                           | ((xy[:, 0] > 3.5) & (xy[:, 0] < 5.8) & (xy[:, 1] > 3.6))
                           | ((xy[:, 0] > 4.5) & (xy[:, 0] < 4.8) & (xy[:, 1] > 1.5) & (xy[:, 1] < 1.8)))
    cloud.horizontal(0, 0, 6, 4, 0.0, FLOOR)
    cloud.horizontal(0, 0, 6, 4, H, CEILING, keep=lowered)
    corners = [(0, 0), (6, 0), (6, 4), (0, 4)]
    for index in range(4):
        cloud.vertical(corners[index], corners[(index + 1) % 4], 0.0, H, WALL)
    extras(cloud)
    points = cloud.points()
    levels = room_levels(points, [room()], SETTINGS)
    from shapely.ops import unary_union

    beams, rejected = detect_beams(points, [room()], levels, unary_union(wall_bodies()), [], H, SETTINGS)
    assert beams == [], [item.evidence for item in beams]
    reasons = {item["reason"] for item in rejected}
    assert "along_wall" in reasons or "not_a_narrow_band" in reasons


# ------------------------------------------------------------------------------------------------------------- stairs

def _stair(cloud: Cloud, x0: float, y0: float, risers: int, riser: float, tread: float, width: float, code: int = STAIRS) -> None:
    for k in range(1, risers):
        z = k * riser
        s0 = x0 + (k - 1) * tread
        cloud.horizontal(s0, y0, s0 + tread, y0 + width, z, code)
        cloud.vertical((s0, y0), (s0, y0 + width), z - riser, z, code)
    s_last = x0 + (risers - 1) * tread
    cloud.vertical((s_last, y0), (s_last, y0 + width), (risers - 1) * riser, risers * riser, code)


def test_straight_stair_is_measured() -> None:
    cloud = Cloud(11)
    room_shell(cloud, 8.0, 6.0, occlude=False)
    _stair(cloud, 2.0, 1.0, risers=14, riser=0.18, tread=0.27, width=1.0)
    stairs, rejected = detect_stairs(cloud.points(storey=2.52), [], SETTINGS)
    assert len(stairs) == 1, rejected
    stair = stairs[0]
    assert stair.riser_count == 14
    assert stair.riser_height_m == pytest.approx(0.18, abs=0.006)
    assert stair.tread_depth_m == pytest.approx(0.27, abs=0.02)
    assert stair.width_m == pytest.approx(1.0, abs=0.05)
    assert stair.from_z_m == pytest.approx(0.0, abs=0.03)
    assert stair.to_z_m == pytest.approx(14 * 0.18, abs=0.05)
    direction = math.radians(stair.direction_deg)
    assert math.cos(direction - THETA) > 0.999  # travels along the rotated +x axis
    assert np.hypot(*(np.asarray(stair.start) - rot(np.asarray([[2.0, 1.5]]))[0])) < 0.06


def test_furniture_with_steps_and_scattered_points_are_no_stairs() -> None:
    cloud = Cloud(12)
    room_shell(cloud, 8.0, 6.0, occlude=False)
    _stair(cloud, 1.0, 4.0, risers=4, riser=0.35, tread=0.30, width=0.45)  # step shelf mislabelled as stairs
    cloud.scatter(600, 0.2, 0.2, 7.8, 5.8, 0.0, H, STAIRS)  # scattered mislabelled points (as on real scans)
    cloud.box_faces(5.0, 1.0, 6.0, 1.6, 0.0, 0.45, STAIRS, top=True)  # low bench mislabelled as stairs
    stairs, rejected = detect_stairs(cloud.points(), [], SETTINGS)
    assert stairs == [], [item.evidence for item in stairs]


# ---------------------------------------------------------------------------------------------- precision / recall

def test_precision_recall_over_random_scenes() -> None:
    """Ten random scenes (random rotation, position, size, occlusion): every real element found, nothing invented."""
    found = {"column": [0, 0, 0], "beam": [0, 0, 0], "stair": [0, 0, 0]}  # tp, fp, fn
    errors: dict[str, list[float]] = {"column": [], "beam": [], "stair": []}
    global THETA
    original = THETA
    try:
        for seed in range(10):
            rng = np.random.default_rng(100 + seed)
            THETA = float(rng.uniform(0, math.pi / 2))
            cloud = Cloud(200 + seed)
            room_shell(cloud, 9.0, 7.0)
            size = float(rng.uniform(0.25, 0.5))
            cx, cy = float(rng.uniform(2.0, 4.0)), float(rng.uniform(2.0, 5.0))
            cloud.box_faces(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2, 0.0, H, WALL if seed % 2 else COLUMN, faces=tuple(rng.choice(4, 3, replace=False)))
            cloud.box_faces(6.0, 5.5, 6.6, 6.1, 0.0, 1.9, 117, top=True)  # cabinet (negative)
            riser = float(rng.uniform(0.16, 0.19))
            _stair(cloud, 5.5, 1.0, risers=int(round(2.6 / riser)) - 1, riser=riser, tread=float(rng.uniform(0.25, 0.29)), width=float(rng.uniform(0.9, 1.2)))
            points = cloud.points()
            from shapely.ops import unary_union

            rooms = [room(9.0, 7.0)]
            analysis = analyse_structure(points, rooms, wall_bodies(9.0, 7.0), [], [rot_polygon(box(6.0, 5.5, 6.6, 6.1))], 1.0, 0.0, H, SETTINGS)
            truth = rot(np.asarray([[cx, cy]]))[0]
            hits = [c for c in analysis.columns if np.hypot(*(np.asarray(c.center) - truth)) < 0.10]
            found["column"][0] += min(len(hits), 1)
            found["column"][1] += len(analysis.columns) - min(len(hits), 1)
            found["column"][2] += 0 if hits else 1
            if hits:
                errors["column"].append(abs(max(hits[0].width_m, hits[0].depth_m) - size))
            found["stair"][0] += min(len(analysis.stairs), 1)
            found["stair"][1] += max(len(analysis.stairs) - 1, 0)
            found["stair"][2] += 0 if analysis.stairs else 1
            if analysis.stairs:
                errors["stair"].append(abs(analysis.stairs[0].riser_height_m - riser))
            found["beam"][1] += len(analysis.beams)
    finally:
        THETA = original
    for kind, (tp, fp, fn) in found.items():
        if kind == "beam":
            assert fp == 0
            continue
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        assert precision == 1.0 and recall >= 0.9, (kind, found[kind])
    assert max(errors["column"]) < 0.04
    assert max(errors["stair"]) < 0.01


def test_analysis_writes_plan_model_and_dxf_entities() -> None:
    from samcloud.bim.plan_model import BuildingPlan, PlanRoom, PlanWall

    cloud = Cloud(13)
    room_shell(cloud, 8.0, 6.0)
    cloud.box_faces(3.0, 3.0, 3.3, 3.3, 0.0, H, COLUMN)
    upm, floor_z = 2.0, 10.0  # local units: 2 per metre, level floor at local z 10
    analysis = analyse_structure(cloud.points(), [room(8.0, 6.0)], wall_bodies(8.0, 6.0), [], [], upm, floor_z, H, SETTINGS)
    layers: dict[str, list] = {"A-COLS": [], "S-BEAM": []}
    metrics = analysis.add_dxf_entities(layers)
    assert metrics["columns"] == 1
    assert {entity["type"] for entity in layers["A-COLS"]} == {"polyline", "hatch"}
    room_local = affinity.scale(room(8.0, 6.0), upm, upm, origin=(0, 0))
    a, b = rot(np.asarray([[0.0, -0.1]]))[0] * upm, rot(np.asarray([[8.0, -0.1]]))[0] * upm
    plan = BuildingPlan(
        project_name="p", level_id="l", level_name="1", units_per_metre=upm, frame={"origin": [0, 0, 0], "scale_factor": 1.0, "rotation": np.eye(3).tolist()},
        floor_z=floor_z, ceiling_z=floor_z + H * upm,
        walls=[PlanWall("W001", [a.tolist(), b.tolist()], 0.2)],
        rooms=[PlanRoom("R001", "Room", {"exterior": [list(p) for p in list(room_local.exterior.coords)[:-1]], "holes": []}, 48.0, [0.0, 0.0])],
    )
    analysis.apply_to_plan(plan)
    assert plan.rooms[0].ceiling_source == "points"
    assert plan.rooms[0].ceiling_z == pytest.approx(floor_z + H * upm, abs=0.02 * upm)
    assert plan.walls[0].height_source == "room_ceiling" and plan.walls[0].height_m == pytest.approx(H, abs=0.02)
    assert len(plan.columns) == 1 and plan.columns[0]["z_max"] == pytest.approx(floor_z + H * upm, abs=0.05 * upm)
    restored = BuildingPlan.from_dict(plan.to_dict())
    assert restored.columns == plan.columns and restored.rooms[0].ceiling_source == "points"
    assert isinstance(analysis, StructureAnalysis)


def test_column_and_beam_entities_write_a_valid_dxf() -> None:
    import ezdxf  # noqa: F401
    from samcloud.dxf.las_source import Frame
    from samcloud.dxf.layers import load_layer_config
    from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus
    from samcloud.dxf.writer import build_dxf_document
    from samcloud.ortho.vector.structure.beams import Beam
    from samcloud.ortho.vector.structure.columns import Column
    from samcloud.ortho.vector.structure.levels import RoomLevel
    from shapely.geometry import Point

    analysis = StructureAnalysis(units_per_metre=1.0, floor_z=0.0, level_ceiling_m=H, rooms=[room()], levels=[RoomLevel()])
    analysis.columns = [
        Column("rect", (1.0, 1.0), box(0.85, 0.85, 1.15, 1.15), 0.3, 0.3, 0.0, 0.0, H, 0.004, 0.9),
        Column("circle", (3.0, 1.0), Point(3.0, 1.0).buffer(0.2, quad_segs=8), 0.4, 0.4, 0.0, 0.0, H, 0.003, 0.9),
    ]
    analysis.beams = [Beam(((2.0, 0.0), (2.0, 4.0)), 0.3, 0.3, 2.3, H, LineString([(2, 0), (2, 4)]).buffer(0.15, cap_style=2), 0.8)]
    layers: dict[str, list] = {}
    analysis.add_dxf_entities(layers)
    definitions = {layer.name: layer for layer in load_layer_config().for_project_mode("indoor")}
    assert definitions["S-BEAM"].linetype == "DASHED"
    frame = Frame(np.zeros(3), 1.0, np.eye(3))
    result = DxfPlanResult(
        project_name="p", level_name="EG", mode="indoor", profile_name=None, scale_factor=1.0, frame=frame, settings_fingerprint="x",
        layers=layers, statuses={name: LayerStatus(name, "ok", None, len(items)) for name, items in layers.items()},
        layer_colors={"A-COLS": 7, "S-BEAM": 8}, background=BackgroundImage(None, None, (0.0, 0.0, 5.0, 5.0)), fingerprint_inputs={},
        layer_linetypes={"A-COLS": "CONTINUOUS", "S-BEAM": "DASHED"}, layer_hatches={}, layer_styles={}, blocks=[], variant="ortho",
    )
    document = build_dxf_document(result)
    modelspace = document.modelspace()
    assert len(modelspace.query('HATCH[layer=="A-COLS"]')) == 2
    assert len(modelspace.query('CIRCLE[layer=="A-COLS"]')) == 1
    assert len(modelspace.query('LWPOLYLINE[layer=="A-COLS"]')) == 1
    assert len(modelspace.query('LINE[layer=="S-BEAM"]')) == 2
    assert document.layers.get("S-BEAM").dxf.linetype == "DASHED"


def test_beam_resting_on_a_column_stays_one_beam_and_the_column_ends_below_it() -> None:
    cloud = Cloud(14)
    width, x_beam, bottom = 0.30, 3.0, 2.25
    cloud.horizontal(0, 0, 6, 4, 0.0, FLOOR)
    cloud.horizontal(0, 0, 6, 4, H, CEILING, keep=lambda xy: ~((xy[:, 0] > x_beam - width / 2) & (xy[:, 0] < x_beam + width / 2)))
    corners = [(0, 0), (6, 0), (6, 4), (0, 4)]
    for index in range(4):
        cloud.vertical(corners[index], corners[(index + 1) % 4], 0.0, H, WALL)
    # beam underside, interrupted above the column (hidden by it)
    cloud.horizontal(x_beam - width / 2, 0, x_beam + width / 2, 1.75, bottom, BEAM)
    cloud.horizontal(x_beam - width / 2, 2.25, x_beam + width / 2, 4, bottom, BEAM)
    cloud.vertical((x_beam - width / 2, 0), (x_beam - width / 2, 4), bottom, H, CEILING)
    cloud.vertical((x_beam + width / 2, 0), (x_beam + width / 2, 4), bottom, H, CEILING)
    cloud.box_faces(2.80, 1.80, 3.20, 2.20, 0.0, bottom, COLUMN)
    analysis = analyse_structure(cloud.points(), [room()], wall_bodies(), [], [], 1.0, 0.0, H, SETTINGS)
    assert len(analysis.beams) == 1, [beam.evidence for beam in analysis.beams]
    length = float(np.hypot(*(np.asarray(analysis.beams[0].axis[1]) - np.asarray(analysis.beams[0].axis[0]))))
    assert length == pytest.approx(4.0, abs=0.15)
    assert len(analysis.columns) == 1
    assert analysis.columns[0].z_top_m == pytest.approx(bottom, abs=0.03)


def test_gable_ceiling_gets_two_planes_meeting_at_the_ridge() -> None:
    cloud = Cloud(15)
    slope = math.radians(25.0)
    cloud.horizontal(0, 0, 6, 4, 0.0, FLOOR)
    cloud.sloped(0, 0, 6, 4, lambda xy: 2.0 + (2.0 - np.abs(xy[:, 1] - 2.0)) * math.tan(slope), CEILING)  # ridge along y = 2
    level = room_levels(cloud.points(storey=2.6), [room()], SETTINGS)[0]
    assert level.ceiling_source == "planes" and level.ceiling_envelope == "min"
    assert len(level.ceiling_planes) == 2
    for y, expected in ((0.5, 2.0 + 0.5 * math.tan(slope)), (2.0, 2.0 + 2.0 * math.tan(slope)), (3.6, 2.0 + 0.4 * math.tan(slope))):
        assert level.ceiling_at(rot(np.asarray([[3.0, y]])), 2.6)[0] == pytest.approx(expected, abs=0.03)
    slopes = sorted(math.degrees(math.atan(math.hypot(a, b))) for a, b, _ in level.ceiling_planes)
    assert slopes == pytest.approx([25.0, 25.0], abs=0.7)


def test_mono_pitch_and_flat_ceilings_stay_single_surfaces() -> None:
    cloud = Cloud(16)
    cloud.horizontal(0, 0, 6, 4, 0.0, FLOOR)
    cloud.sloped(0, 0, 6, 4, lambda xy: 2.0 + xy[:, 0] * math.tan(math.radians(12.0)), CEILING)
    mono = room_levels(cloud.points(storey=2.6), [room()], SETTINGS)[0]
    assert mono.ceiling_source == "plane" and mono.ceiling_planes is None
    flat_cloud = Cloud(17)
    room_shell(flat_cloud, ceiling=2.5)
    flat_cloud.horizontal(0.5, 0.5, 2.0, 1.5, 2.25, CEILING)  # a lowered area (suspended part) is no second plane
    flat = room_levels(flat_cloud.points(), [room()], SETTINGS)[0]
    assert flat.ceiling_source == "points" and flat.ceiling_planes is None


def test_smeared_wall_stub_door_post_and_labelled_instance_are_no_columns() -> None:
    """Unfused clouds smear wall faces and door frames: such floor-to-ceiling blobs must not become columns."""
    from samcloud.ortho.vector.structure.common import DOOR

    cloud = Cloud(18)
    room_shell(cloud, 8.0, 6.0)
    # wall smear: a wall-class blob reaching 0.45 m into the room from the wall face (its far part is beyond the candidate clearance)
    cloud.box_faces(3.0, 0.0, 3.5, 0.45, 0.0, H, WALL, faces=(1, 2, 3))
    # door post: mostly door-class points with some wall-class points, free-standing
    for _ in range(2):
        cloud.box_faces(5.0, 3.0, 5.15, 3.15, 0.0, H, DOOR)
    cloud.box_faces(5.0, 3.0, 5.15, 3.15, 0.0, H, WALL)
    # wall-class post inside a scan instance labelled as a shelf
    cloud.box_faces(2.0, 4.0, 2.3, 4.3, 0.0, H, WALL)
    instances = [{"label": "shelf", "hull": rot_polygon(box(1.9, 3.9, 2.4, 4.4))}]
    points = cloud.points()
    rooms = [room(8.0, 6.0)]
    levels = room_levels(points, rooms, SETTINGS)
    from shapely.ops import unary_union

    columns, rejected = detect_columns(points, rooms, levels, unary_union(wall_bodies(8.0, 6.0)), [], [], H, SETTINGS, instances)
    assert columns == [], [item.evidence for item in columns]
    assert {"furniture_points", "in_instance"} <= {item["reason"] for item in rejected}, rejected
    assert {"touches_wall", "attached_to_wall"} & {item["reason"] for item in rejected}, rejected


def test_free_standing_column_close_to_a_wall_is_still_found() -> None:
    cloud = Cloud(19)
    room_shell(cloud, 8.0, 6.0)
    cloud.box_faces(3.0, 0.25, 3.3, 0.55, 0.0, H, WALL, faces=(0, 1, 2, 3))  # 0.25 m in front of the wall face, free space between
    points = cloud.points()
    rooms = [room(8.0, 6.0)]
    from shapely.ops import unary_union

    columns, rejected = detect_columns(points, rooms, room_levels(points, rooms, SETTINGS), unary_union(wall_bodies(8.0, 6.0)), [], [], H, SETTINGS)
    assert len(columns) == 1, rejected
    assert sorted((columns[0].width_m, columns[0].depth_m)) == pytest.approx([0.30, 0.30], abs=0.03)
