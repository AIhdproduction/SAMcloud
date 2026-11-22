"""A wall body is the empty band between two measured faces that look away from each other."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from samcloud.bim.plan_model import BuildingPlan, PlanWall, load_refine_plan, save_plan_model
from samcloud.bim.scene_builder import _extension_blocked, _Wall
from samcloud.ortho.vector.ceiling import snap_exterior_faces_to_flucht
from samcloud.ortho.vector.door_rooms import door_bounded_rooms
from samcloud.ortho.vector.double_walls import DoubleWallSettings, WallEvidence, WallRun, _Face, build_double_walls
from samcloud.ortho.vector.indoor import Opening
from samcloud.ortho.vector.opening_bands import passage_reveals, run_band
from samcloud.ortho.vector.raster_grid import RasterGrid
from samcloud.ortho.vector.wall_lines import WallFace
from samcloud.ortho.vector.wall_sides import MeasuredWall, WallSideEvidence, apply_measured_bands, doorway_walls, two_faced_walls, wall_side_evidence

SETTINGS = DoubleWallSettings()
NEAR, FAR = 0.0037, 0.2086  # off every cell border
U0, U1 = 0.4137, 2.8871
FRAMES = [(0.0, 1.0), (0.37, 0.989), (17.3, 1.0), (93.1, 1.013)]


class _Cloud:
    """Storey rasters in a wall frame (u along, v across), turned and scaled, with scan gaps."""

    def __init__(self, angle: float = 0.37, upm: float = 1.0) -> None:
        self.upm = upm
        turn = math.radians(angle)
        self.u = np.asarray((math.cos(turn), math.sin(turn)))
        self.v = np.asarray((-self.u[1], self.u[0]))
        self.origin = np.asarray((4.0137, 4.0271))
        self.grid = RasterGrid(0.0, 8.0, 0.01, 800, 800)
        self.masks = {name: np.zeros((800, 800), dtype=bool) for name in ("ceiling", "floor", "high")}
        x, y = np.meshgrid(0.01 * (np.arange(800) + 0.5), 8.0 - 0.01 * (np.arange(800) + 0.5))
        self.along = ((x - self.origin[0]) * self.u[0] + (y - self.origin[1]) * self.u[1]) / upm
        self.across = ((x - self.origin[0]) * self.v[0] + (y - self.origin[1]) * self.v[1]) / upm
        self.random = np.random.default_rng(11)
        self.lines: list[list[tuple[float, float]]] = []

    def at(self, u: float, v: float) -> np.ndarray:
        return self.origin + (self.u * u + self.v * v) * self.upm

    def fill(self, name: str, v0: float, v1: float, u0: float = U0, u1: float = U1, seen: float = 0.8) -> None:
        inside = (self.along >= u0) & (self.along <= u1) & (self.across >= v0) & (self.across <= v1)
        self.masks[name][inside & (self.random.random(inside.shape) < seen)] = True

    def face(self, v: float, top: bool = True, u0: float = U0, u1: float = U1) -> None:
        self.lines.append([tuple(self.at(u0, v + 0.0011)), tuple(self.at(u1, v - 0.0013))])
        if top:
            self.fill("high", v - 0.012, v + 0.012, u0, u1)

    def room(self, v0: float, v1: float, floor: bool = True, ceiling: bool = True) -> None:
        for name, seen in (("floor", floor), ("ceiling", ceiling)):
            if seen:
                self.fill(name, v0, v1)

    def walls(self) -> list[MeasuredWall]:
        evidence = WallSideEvidence(self.grid, self.upm, self.masks["ceiling"], self.masks["floor"], self.masks["high"])
        return two_faced_walls(self.lines, evidence)


def _partition(angle: float = 0.37, upm: float = 1.0, **far_room: bool) -> _Cloud:
    cloud = _Cloud(angle, upm)
    cloud.face(NEAR)
    cloud.face(FAR)
    cloud.room(-0.6, NEAR - 0.004)
    cloud.room(FAR + 0.004, 0.8, **far_room)
    return cloud


# --------------------------------------------------------------------------
# two faces that look away from each other
# --------------------------------------------------------------------------

@pytest.mark.parametrize("angle,upm", FRAMES)
def test_wall_seen_from_both_rooms_is_proven(angle: float, upm: float) -> None:
    cloud = _partition(angle, upm)
    [wall] = cloud.walls()
    assert wall.thickness / upm == pytest.approx(FAR - NEAR, abs=0.004)
    assert float(np.dot(wall.towards, cloud.v)) > 0.99


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_direction_always_points_into_the_body(order: tuple[int, int]) -> None:
    cloud = _partition()
    cloud.lines = [cloud.lines[index] for index in order]
    [wall] = cloud.walls()
    middle = (np.asarray(wall.start) + np.asarray(wall.end)) / 2.0 + np.asarray(wall.towards) * wall.thickness / 2.0
    assert float((middle - cloud.origin) @ cloud.v) == pytest.approx((NEAR + FAR) / 2.0, abs=0.004)


@pytest.mark.parametrize("angle,upm", FRAMES)
def test_floor_hidden_under_a_bathtub_still_proves_the_wall(angle: float, upm: float) -> None:
    assert len(_partition(angle, upm, floor=False).walls()) == 1


def test_lintel_over_a_doorway_is_a_wall() -> None:
    cloud = _partition()
    cloud.fill("floor", NEAR, FAR, 1.4037, 2.4029)  # the doorway below the lintel
    assert len(cloud.walls()) == 1


def test_face_hidden_at_the_top_needs_the_floor_to_end_at_both_faces() -> None:
    cloud = _Cloud()
    cloud.face(NEAR)
    cloud.face(FAR, top=False)
    cloud.room(-0.6, NEAR - 0.004)
    cloud.room(FAR + 0.004, 0.8, ceiling=False)
    assert len(cloud.walls()) == 1


# --------------------------------------------------------------------------
# surfaces inside a room are no wall faces
# --------------------------------------------------------------------------

def test_low_cabinet_front_is_no_wall_face() -> None:
    cloud = _Cloud()
    cloud.face(-0.1813, top=False)
    cloud.face(NEAR)
    cloud.room(-0.8, -0.1853)
    cloud.fill("ceiling", -0.1853, NEAR - 0.004)  # seen above the cabinet
    assert cloud.walls() == []


def test_full_height_cabinet_front_is_no_wall_face() -> None:
    cloud = _Cloud()
    cloud.face(-0.1813)
    cloud.face(NEAR)  # the wall behind, seen beside the doors
    cloud.room(-0.8, -0.1853)
    assert cloud.walls() == []


def test_curtain_is_no_wall_face() -> None:
    cloud = _Cloud()
    cloud.face(-0.1427)
    cloud.face(NEAR)
    cloud.room(-0.8, NEAR - 0.004)
    assert cloud.walls() == []


def test_curtain_before_a_wall_with_a_room_behind_leaves_the_wall_alone() -> None:
    cloud = _partition()
    cloud.face(-0.0713)
    walls = cloud.walls()
    assert [sorted((wall.first, wall.second)) for wall in walls] == [[0, 1]]


@pytest.mark.parametrize("floor_behind", [True, False])
def test_shower_screen_is_no_wall_face(floor_behind: bool) -> None:
    cloud = _Cloud()
    cloud.face(-0.2317, top=False)
    cloud.face(NEAR)
    cloud.room(-0.8, -0.2357)
    cloud.room(-0.2357, NEAR - 0.004, floor=floor_behind)
    assert cloud.walls() == []


@pytest.mark.parametrize("depth", [0.0713, 0.1519])
@pytest.mark.parametrize("room_behind", [False, True])
def test_tiles_and_pre_wall_are_no_wall_face(depth: float, room_behind: bool) -> None:
    cloud = _Cloud()
    cloud.face(-depth, top=False)
    cloud.face(NEAR)
    cloud.room(-0.8, -depth - 0.004)
    cloud.fill("ceiling", -depth - 0.004, NEAR - 0.004)  # over the top of the installation
    if room_behind:
        cloud.room(NEAR + 0.12, 0.8)
    assert cloud.walls() == []


def test_mirror_double_surface_is_no_wall_face() -> None:
    cloud = _Cloud()
    cloud.face(NEAR)
    cloud.face(0.0927, top=False)
    cloud.room(-0.8, NEAR - 0.004)
    assert cloud.walls() == []


def test_floor_reflected_behind_a_mirror_is_no_second_room() -> None:
    cloud = _Cloud()
    cloud.face(NEAR)
    cloud.face(0.1131, top=False)
    cloud.room(-0.8, NEAR - 0.004)
    cloud.room(NEAR - 0.004, 0.8, ceiling=False)  # the reflection begins at the mirror
    assert cloud.walls() == []


@pytest.mark.parametrize("second_room", [False, True])
def test_double_registration_is_no_wall(second_room: bool) -> None:
    cloud = _Cloud()
    cloud.face(NEAR)
    cloud.face(0.0853)
    cloud.room(-0.8, NEAR - 0.004)
    if second_room:
        cloud.room(-0.72, 0.0813)  # the second copy of floor and ceiling
    assert cloud.walls() == []


def test_door_leaf_before_a_wall_is_no_wall_face() -> None:
    cloud = _Cloud()
    cloud.face(-0.1517, top=False)
    cloud.face(NEAR)
    cloud.room(-0.8, -0.1557)
    cloud.fill("floor", -0.1557, NEAR - 0.004)
    cloud.room(0.0137, 0.8)
    assert cloud.walls() == []


def test_free_space_between_parallel_walls_is_no_body() -> None:
    cloud = _Cloud()
    cloud.face(NEAR)
    cloud.face(FAR)
    cloud.room(-0.8, 0.8)
    assert cloud.walls() == []


def test_exterior_wall_with_an_unscanned_outside_keeps_its_rules() -> None:
    cloud = _Cloud()
    cloud.face(NEAR)
    cloud.face(FAR)
    cloud.room(-0.8, NEAR - 0.004)
    assert cloud.walls() == []


def test_too_thick_or_too_short_is_not_proven() -> None:
    thick = _Cloud()
    thick.face(NEAR)
    thick.face(0.3419)
    thick.room(-0.6, NEAR - 0.004)
    thick.room(0.3459, 0.8)
    assert thick.walls() == []
    short = _partition()
    short.lines[1] = [tuple(short.at(U0, FAR)), tuple(short.at(U0 + 0.31, FAR))]
    assert short.walls() == []


def test_evidence_needs_ceiling_floor_and_wall_heights() -> None:
    grid = RasterGrid(0.0, 0.5, 0.05, 10, 10)
    full = np.ones((10, 10), dtype=np.uint32)
    assert two_faced_walls([[(0, 0), (1, 0)]], None) == []
    assert wall_side_evidence(SimpleNamespace(ceiling_count=full, wall_bits=full), grid, 1.0, 16) is None
    vertical = SimpleNamespace(ceiling_count=full, floor_points=full > 0, wall_bits=full << np.uint32(13))
    evidence = wall_side_evidence(vertical, grid, 1.0, 16)
    assert evidence is not None and evidence.high.all() and evidence.ceiling.all()
    low = SimpleNamespace(ceiling_count=full, floor_points=full > 0, wall_bits=full << np.uint32(5))
    assert not wall_side_evidence(low, grid, 1.0, 16).high.any()


def test_both_faces_are_known_before_the_flucht_snap_joins_them() -> None:
    cloud = _partition(0.0)
    faces = [WallFace("line", np.empty((0, 2)), line=LineString(line)) for line in cloud.lines]
    assert len(cloud.walls()) == 1
    snap_exterior_faces_to_flucht(faces, np.zeros((800, 800), dtype=bool), cloud.grid, 1.0)
    offsets = [float((np.mean(np.asarray(face.line.coords), axis=0) - cloud.origin) @ cloud.v) for face in faces]
    assert abs(offsets[0] - offsets[1]) < 0.004  # what the wall stage alone gets to see


def _plane_points(cloud: _Cloud, v: float, count: int, u0: float = U0, u1: float = U1, low: float = 0.3131, high: float = 2.3517) -> np.ndarray:
    u, z = cloud.random.uniform(u0, u1, count), cloud.random.uniform(low, high, count)
    xy = cloud.origin + (np.outer(u, cloud.u) + np.outer(cloud.random.normal(v, 0.002, count), cloud.v)) * cloud.upm
    return np.column_stack((xy, z * cloud.upm))


@pytest.mark.parametrize("angle,upm", FRAMES)
def test_each_face_is_measured_again_on_the_overlap(angle: float, upm: float) -> None:
    cloud = _partition(angle, upm)
    cloud.lines[1] = [tuple(cloud.at(U0, FAR + 0.0271)), tuple(cloud.at(U1, FAR + 0.0271))]  # a long fit averaged two layers
    points = np.vstack((
        _plane_points(cloud, NEAR, 3000), _plane_points(cloud, FAR, 3000), _plane_points(cloud, FAR + 0.0373, 400),
    ))
    evidence = WallSideEvidence(cloud.grid, upm, cloud.masks["ceiling"], cloud.masks["floor"], cloud.masks["high"])
    [wall] = two_faced_walls(cloud.lines, evidence, points)
    assert float((np.asarray(wall.start) - cloud.origin) @ cloud.v) / upm == pytest.approx(NEAR, abs=0.003)
    assert wall.thickness / upm == pytest.approx(FAR - NEAR, abs=0.004)


def test_a_few_points_do_not_move_a_fitted_face() -> None:
    cloud = _partition()
    evidence = WallSideEvidence(cloud.grid, 1.0, cloud.masks["ceiling"], cloud.masks["floor"], cloud.masks["high"])
    few = _plane_points(cloud, FAR + 0.0311, 200, 1.4137, 1.7131)
    assert two_faced_walls(cloud.lines, evidence, few) == two_faced_walls(cloud.lines, evidence)


def test_doorway_that_fills_the_overlap_needs_the_ceiling_at_both_faces() -> None:
    cloud = _partition()
    cloud.fill("floor", NEAR, FAR)
    assert len(cloud.walls()) == 1
    one_sided = _partition(ceiling=False)
    one_sided.fill("floor", NEAR, FAR)
    assert one_sided.walls() == []


def _found_from_one_face(cloud: _Cloud, planes: tuple[tuple[float, float], ...], keep: int = 0) -> list[MeasuredWall]:
    points = np.vstack([_plane_points(cloud, value, 4000, high=top) for value, top in planes])
    evidence = WallSideEvidence(cloud.grid, cloud.upm, cloud.masks["ceiling"], cloud.masks["floor"], cloud.masks["high"])
    return two_faced_walls([cloud.lines[keep]], evidence, points)


@pytest.mark.parametrize("angle,upm", FRAMES)
@pytest.mark.parametrize("keep", [0, 1])
def test_second_plane_is_found_where_no_face_was_extracted(angle: float, upm: float, keep: int) -> None:
    cloud = _partition(angle, upm)
    [wall] = _found_from_one_face(cloud, ((NEAR, 2.3517), (FAR, 2.3517)), keep)
    start = float((np.asarray(wall.start) - cloud.origin) @ cloud.v) / upm
    other = start + wall.thickness / upm * float(np.dot(wall.towards, cloud.v))
    assert sorted((start, other)) == pytest.approx([NEAR, FAR], abs=0.004)
    assert wall.second == -1


@pytest.mark.parametrize("kind", ["cabinet", "full cabinet", "curtain", "screen", "pre-wall", "mirror", "double registration", "leaf", "drift"])
def test_searched_plane_must_pass_the_same_test(kind: str) -> None:
    cloud = _Cloud()
    front, top = -0.1813, 2.3517
    if kind in ("cabinet", "screen", "pre-wall", "mirror", "leaf"):
        top = 1.9131
    if kind == "mirror":
        front = 0.1131
    if kind == "double registration":
        front = 0.1253
    cloud.face(NEAR)
    cloud.face(front, top=top > 2.0)
    if kind == "mirror":
        cloud.room(-0.8, NEAR - 0.004)
    elif kind == "double registration":
        cloud.room(-0.8, NEAR - 0.004)
        cloud.room(-0.7, 0.1213)
    elif kind == "drift":
        cloud.room(-0.8, 0.8)
    else:
        cloud.room(-0.8, front - 0.004)
        if kind != "full cabinet":
            cloud.fill("ceiling", front - 0.004, NEAR - 0.004)
        if kind in ("screen", "leaf"):
            cloud.fill("floor", front - 0.004, NEAR - 0.004)
    for keep in (0, 1):
        assert _found_from_one_face(cloud, ((NEAR, 2.3517), (front, top)), keep) == []


@pytest.mark.parametrize("surface", ["floor", "ceiling"])
def test_searched_plane_never_encloses_free_space_of_a_room(surface: str) -> None:
    cloud = _partition()
    stretch = 0.52 if surface == "floor" else 0.265  # below what two extracted faces may show (doorway, ceiling edge)
    cloud.fill(surface, NEAR, FAR, U0, U0 + stretch * (U1 - U0))
    assert _found_from_one_face(cloud, ((NEAR, 2.3517), (FAR, 2.3517))) == []
    assert len(cloud.walls()) == 1


def test_drift_layers_without_a_hollow_between_are_no_partner() -> None:
    cloud = _partition()
    layers = tuple((value, 2.3517) for value in np.linspace(NEAR, FAR, 9))
    assert _found_from_one_face(cloud, layers) == []


# --------------------------------------------------------------------------
# doorway: both faces exist above the head only
# --------------------------------------------------------------------------

DOOR0, DOOR1 = 1.2137, 2.0971
HEAD = 2.013


def _doorway(angle: float = 0.37, upm: float = 1.0, planes: tuple[float, ...] = (NEAR, FAR), ceiling_between: bool = False, east: bool = True):
    stage = _Stage(angle, upm)
    cloud = stage.frame
    host = stage.run((U0, -0.2113), (DOOR0, -0.2113), 0.2678)  # built 0.21 m beside its wall
    cloud.fill("floor", -0.7, 0.8)
    cloud.fill("ceiling", -0.7, NEAR - 0.004)
    if east:
        cloud.fill("ceiling", FAR + 0.004, 0.8)
    if ceiling_between:
        cloud.fill("ceiling", NEAR - 0.004, FAR + 0.004)
    points = np.vstack([
        _plane_points(cloud, -0.2113, 150, DOOR0, DOOR1, 0.2, 1.7),  # below the head: the line the door was detected on
        *[_plane_points(cloud, value, 900 - 200 * number, DOOR0, DOOR1, 2.0913, 2.4431) for number, value in enumerate(planes)],
    ])
    centre = cloud.at((DOOR0 + DOOR1) / 2.0, -0.2113)
    door = Opening("DOOR", tuple(centre), (DOOR1 - DOOR0) * upm, math.atan2(cloud.u[1], cloud.u[0]), 0.2678 * upm, head_m=HEAD)
    evidence = WallSideEvidence(cloud.grid, upm, cloud.masks["ceiling"], cloud.masks["floor"], cloud.masks["high"])
    found = doorway_walls(stage.runs, stage.faces, [door], points, 2.48 * upm, evidence, upm, SETTINGS)
    return stage, host, found


@pytest.mark.parametrize("angle,upm", FRAMES)
def test_wall_seen_only_above_a_door_takes_both_planes(angle: float, upm: float) -> None:
    stage, host, found = _doorway(angle, upm)
    assert [index for index, _ in found] == [0]
    apply_measured_bands(stage.runs, stage.faces, [], upm, SETTINGS, found)
    assert stage.band(host) == pytest.approx((NEAR, FAR), abs=0.005)


def test_curtain_rail_or_screen_above_a_door_is_no_wall() -> None:
    assert _doorway(ceiling_between=True)[2] == []


def test_cabinet_above_a_door_without_a_room_behind_is_no_wall() -> None:
    assert _doorway(east=False)[2] == []


@pytest.mark.parametrize("planes", [(NEAR,), (NEAR, 0.0553)])
def test_one_plane_or_a_second_registration_above_a_door_is_no_wall(planes: tuple[float, ...]) -> None:
    assert _doorway(planes=planes)[2] == []


def test_doorway_needs_the_rasters_and_an_assumed_thickness() -> None:
    stage, host, found = _doorway()
    cloud = stage.frame
    door = Opening("DOOR", tuple(cloud.at((DOOR0 + DOOR1) / 2.0, -0.2113)), DOOR1 - DOOR0, math.atan2(cloud.u[1], cloud.u[0]), 0.2678, head_m=HEAD)
    points = np.vstack([_plane_points(cloud, value, 800, DOOR0, DOOR1, 2.0913, 2.4431) for value in (NEAR, FAR)])
    assert doorway_walls(stage.runs, stage.faces, [door], points, 2.48, None, 1.0, SETTINGS) == []
    host.evidence = "face"
    evidence = WallSideEvidence(cloud.grid, 1.0, cloud.masks["ceiling"], cloud.masks["floor"], cloud.masks["high"])
    assert doorway_walls(stage.runs, stage.faces, [door], points, 2.48, evidence, 1.0, SETTINGS) == []


# --------------------------------------------------------------------------
# the wall stage takes both faces
# --------------------------------------------------------------------------

class _Stage:
    def __init__(self, angle: float = 0.0, upm: float = 1.0) -> None:
        self.frame = _Cloud(angle, upm)
        self.upm = upm
        self.faces: list[_Face] = []
        self.runs: list[WallRun] = []

    def run(self, start: tuple[float, float], end: tuple[float, float], thickness: float, evidence: str = "exterior", side: int = 1) -> WallRun:
        face = _Face(len(self.faces), self.frame.at(*start), self.frame.at(*end), {"type": "polyline", "points": []}, side=side)
        self.faces.append(face)
        self.runs.append(WallRun(face.index, 0.0, face.length, thickness * self.upm, "interior", evidence))
        return self.runs[-1]

    def apply(self, *walls: tuple[float, float, float, float], openings: tuple[Opening, ...] = ()) -> dict[str, int]:
        measured = [
            MeasuredWall(0, 1, tuple(self.frame.at(u0, v0)), tuple(self.frame.at(u1, v0)), tuple(self.frame.v), (v1 - v0) * self.upm)
            for u0, u1, v0, v1 in (walls or ((U0, U1, NEAR, FAR),))
        ]
        return apply_measured_bands(self.runs, self.faces, measured, self.upm, SETTINGS, openings=openings)

    def door(self, u0: float, u1: float, v: float, kind: str = "DOOR") -> Opening:
        a, b = self.frame.at(u0, v), self.frame.at(u1, v)
        return Opening(kind, tuple((a + b) / 2.0), (u1 - u0) * self.upm, math.atan2(self.frame.u[1], self.frame.u[0]), 0.2373 * self.upm)

    def crossing_door(self, u: float, v0: float, v1: float) -> Opening:
        a, b = self.frame.at(u, v0), self.frame.at(u, v1)
        return Opening("passage", tuple((a + b) / 2.0), (v1 - v0) * self.upm, math.atan2(self.frame.v[1], self.frame.v[0]), 0.2373 * self.upm)

    def band(self, run: WallRun) -> tuple[float, float]:
        face = self.faces[run.face]
        low, high = run_band(run, face, face.point((run.start + run.end) / 2.0), self.frame.v)
        base = float((face.point((run.start + run.end) / 2.0) - self.frame.origin) @ self.frame.v)
        return (low + base) / self.upm, (high + base) / self.upm

    def end(self, run: WallRun, at_start: bool = False) -> float:
        face = self.faces[run.face]
        return float((face.point(run.start if at_start else run.end, run.thickness / 2.0) - self.frame.origin) @ self.frame.v) / self.upm


@pytest.mark.parametrize("angle,upm", FRAMES)
@pytest.mark.parametrize("evidence", ["room", "learned", "exterior", "partition", "face"])
def test_wall_on_the_wrong_side_of_a_measured_face_moves_between_both(angle: float, upm: float, evidence: str) -> None:
    stage = _Stage(angle, upm)
    run = stage.run((U0, FAR + 0.0017), (U1, FAR + 0.0017), 0.2373, evidence)
    assert stage.apply()["runs_changed"] == 1
    assert stage.band(run) == pytest.approx((NEAR, FAR), abs=1e-6)
    assert run.evidence == "face" and run.detected is not None and run.detected[1] == pytest.approx(0.2373 * upm)


@pytest.mark.parametrize("side", [1, -1])
def test_default_or_averaged_thickness_becomes_the_measured_one(side: int) -> None:
    stage = _Stage(17.3)
    run = stage.run((U0, NEAR if side > 0 else FAR + 0.0373), (U1, NEAR if side > 0 else FAR + 0.0373), 0.2851, "face-collinear", side)
    stage.apply()
    assert stage.band(run) == pytest.approx((NEAR, FAR), abs=1e-6)


def test_exact_band_is_left_untouched() -> None:
    stage = _Stage()
    run = stage.run((U0, NEAR + 0.0011), (U1, NEAR + 0.0011), FAR - NEAR, "face")
    assert stage.apply()["runs_changed"] == 0
    assert run.detected is None and len(stage.faces) == 1


def test_face_within_the_tolerance_stays_where_it_was_fitted() -> None:
    stage = _Stage(17.3)
    run = stage.run((U0, NEAR - 0.0013), (U1, NEAR - 0.0013), 0.2373)
    stage.apply()
    assert stage.band(run) == pytest.approx((NEAR - 0.0013, FAR), abs=1e-6)


def test_passage_reveal_reaches_the_line_the_passage_was_detected_on() -> None:
    passage = Opening("passage", (2.0137, 1.5271), 0.8913, 0.0, 0.2081)
    [beside] = passage_reveals([passage], [(-0.3873, -0.1813)], [True])
    assert beside.bounds == pytest.approx((2.0137 - 0.8913 / 2.0, 1.5271 - 0.3873, 2.0137 + 0.8913 / 2.0, 1.5271))
    [on_face] = passage_reveals([passage], [(0.0, 0.2081)], [True])
    assert on_face.bounds[1::2] == pytest.approx((1.5271, 1.5271 + 0.2081))


def test_piece_with_the_same_band_follows_and_other_bands_do_not() -> None:
    stage = _Stage()
    main = stage.run((U0, NEAR), (U1, NEAR), 0.2851)
    piece = stage.run((U1 + 0.91, NEAR), (U1 + 1.43, NEAR), 0.2851)
    own = stage.run((U1 + 0.93, 0.0913), (U1 + 1.41, 0.0913), 0.1213)
    niche = stage.run((U0 + 0.1, NEAR), (U0 + 0.7, NEAR), 0.1017, "niche")
    stage.apply()
    for run in (main, piece):
        assert stage.band(run) == pytest.approx((NEAR, FAR), abs=1e-6)
    assert stage.band(own) == pytest.approx((0.0913, 0.2126), abs=1e-6)
    assert stage.band(niche) == pytest.approx((NEAR, NEAR + 0.1017), abs=1e-6)


@pytest.mark.parametrize("evidence,moved", [("exterior", True), ("door-fill", True), ("room", False), ("face", False)])
def test_piece_beside_a_doorway_continues_the_wall_seen_above_it(evidence: str, moved: bool) -> None:
    stage = _Stage(17.3)
    run = stage.run((U0, NEAR), (1.5731, NEAR), 0.2678, evidence)
    assert stage.apply((1.4013, 2.5517, NEAR, FAR))["runs_changed"] == int(moved)


def test_parallel_wall_and_long_measured_wall_are_left_alone() -> None:
    stage = _Stage()
    other = stage.run((U0, 0.4731), (U1, 0.4731), 0.1513)
    long_wall = stage.run((-3.0, NEAR), (3.5, NEAR), 0.2513, "face")
    assert stage.apply((U0, U0 + 0.71, NEAR, FAR))["runs_changed"] == 0
    assert other.detected is None and long_wall.detected is None


def test_crossing_walls_keep_ending_on_the_moved_wall() -> None:
    stage = _Stage(17.3)
    stage.run((U0, FAR), (U1, FAR), 0.2373, "room")  # built above the wall
    above = stage.run((1.1337, 2.0), (1.1337, FAR + 0.2373), 0.1213, "room", side=1)  # ended on the old far face
    below = stage.run((1.9121, -1.5), (1.9121, FAR), 0.1213, "room", side=1)  # crosses the wall to its far face
    reaching = stage.run((2.3119, FAR + 0.2211), (2.3119, 2.0), 0.1213, "room")
    stage.apply()
    assert stage.end(above) == pytest.approx(FAR - 0.01, abs=1e-6)
    assert stage.end(reaching, at_start=True) == pytest.approx(FAR - 0.01, abs=1e-6)
    assert stage.end(below) == pytest.approx(FAR, abs=1e-6)


def test_crossing_wall_never_sticks_out_of_a_wall_that_got_thinner() -> None:
    stage = _Stage()
    stage.run((U0, NEAR), (U1, NEAR), 0.2851)
    crossing = stage.run((1.9121, -1.5), (1.9121, NEAR + 0.2851), 0.1213, "room")
    far_away = stage.run((U1 + 2.0, -1.5), (U1 + 2.0, NEAR + 0.2851), 0.1213, "room")
    stage.apply()
    assert stage.end(crossing) == pytest.approx(FAR, abs=1e-6)
    assert stage.end(far_away) == pytest.approx(NEAR + 0.2851, abs=1e-6)


@pytest.mark.parametrize("angle,upm", FRAMES)
def test_wall_seen_above_its_door_ends_on_the_jamb_of_the_crossing_passage(angle: float, upm: float) -> None:
    stage = _Stage(angle, upm)
    run = stage.run((U0, NEAR), (1.2137, NEAR), 0.2373)
    door = stage.door(1.2137, 2.0971, NEAR)
    passage = stage.crossing_door(2.2017, FAR + 0.0169, FAR + 0.9071)  # its jamb is 17 mm beside the face seen above the door
    stage.apply((1.0931, 2.1371, NEAR, FAR), openings=(door, passage))
    assert stage.band(run) == pytest.approx((NEAR, FAR + 0.0169), abs=1e-6)


def test_wall_seen_along_its_length_wins_over_a_jamb() -> None:
    stage = _Stage(17.3)
    run = stage.run((U0, NEAR), (U1, NEAR), 0.2851)
    door = stage.door(U1 - 0.1, U1 + 0.7873, NEAR)
    passage = stage.crossing_door(U1 + 0.9, FAR - 0.0331, FAR + 0.8571)
    stage.apply(openings=(door, passage))
    assert stage.band(run) == pytest.approx((NEAR, FAR), abs=1e-6)


def test_wall_across_a_hosted_doorway_ends_flush_with_the_moved_band() -> None:
    stage = _Stage()
    stage.run((U0, NEAR), (1.2137, NEAR), 0.2851)
    door = stage.door(1.2137, 2.0971, NEAR)
    short = stage.run((2.1713, -1.5), (2.1713, FAR - 0.0223), 0.1213, "room")  # the corner at the jamb, 22 mm short
    stub = stage.run((2.1213, 1.5), (2.1213, NEAR - 0.1311), 0.1213, "room")  # a stub the wall bodies trim at the band
    stage.apply((U0, 1.2137, NEAR, FAR), openings=(door,))
    assert stage.end(short) == pytest.approx(FAR, abs=1e-6)
    assert stage.end(stub) == pytest.approx(NEAR, abs=1e-6)


def test_wall_stage_builds_both_faces_and_keeps_what_was_detected(tmp_path) -> None:
    cloud = _partition(0.0)
    [wall] = cloud.walls()
    line = [cloud.at(U0, FAR).tolist(), cloud.at(U1, FAR).tolist()]
    room = Polygon([tuple(cloud.at(u, v)) for u, v in ((U0, -1.7), (U1, -1.7), (U1, NEAR), (U0, NEAR))])
    free = np.zeros((800, 800), dtype=bool)
    free[(cloud.along >= U0) & (cloud.along <= U1) & (cloud.across >= -1.7) & (cloud.across <= FAR - 0.01)] = True  # closing filled the body
    arguments = ([{"type": "polyline", "points": line, "closed": False}], [], [room], free, np.ones((800, 800), dtype=bool), cloud.grid, 1.0)
    settings = DoubleWallSettings(room_edge_walls=False, close_walls=False, face_test=False, visibility_test=False)
    plain = build_double_walls(*arguments, settings, evidence=WallEvidence())
    fixed = build_double_walls(*arguments, settings, evidence=WallEvidence(two_faced=[wall]))
    assert len(plain.pieces) == len(fixed.pieces) == 1

    def band(piece: dict) -> list[float]:
        return sorted(float((np.mean(piece[key], axis=0) - cloud.origin) @ cloud.v) for key in ("face", "face_b"))

    assert band(plain.pieces[0])[0] == pytest.approx(FAR, abs=0.004)  # with one face the body lands on the wrong side
    assert band(fixed.pieces[0]) == pytest.approx([NEAR, FAR], abs=0.004)
    piece, before = fixed.pieces[0], plain.pieces[0]
    # bit for bit: the pieces on both jambs of an opening tie, a rounding would flip the host
    assert np.array_equal(piece["detected_centreline"], (np.asarray(before["face"]) + np.asarray(before["face_b"])) / 2.0)
    assert piece["detected_thickness"] == pytest.approx(before["thickness"])

    stored = PlanWall(
        "W001", ((np.asarray(piece["face"]) + np.asarray(piece["face_b"])) / 2.0).tolist(), piece["thickness"],
        face_a=piece["face"], face_b=piece["face_b"], detected_centreline=piece["detected_centreline"], detected_thickness_m=piece["detected_thickness"],
    )
    save_plan_model(BuildingPlan(
        project_name="made_up", level_id="level_1", level_name="1", units_per_metre=1.0,
        frame={"origin": [0.0, 0.0, 0.0], "scale_factor": 1.0, "rotation": np.eye(3).tolist()}, floor_z=0.0, ceiling_z=2.5, walls=[stored],
    ), tmp_path)
    [scanned] = load_refine_plan(tmp_path).walls
    assert np.allclose(scanned.centreline, piece["detected_centreline"]) and scanned.thickness_m == pytest.approx(before["thickness"])


@pytest.mark.parametrize("past,blocked", [(-0.0213, False), (0.0013, False), (0.0371, True)])
def test_an_opening_reaching_the_axis_of_the_corner_wall_keeps_its_host(past: float, blocked: bool) -> None:
    host = _Wall("H", np.asarray((1.1137, 2.0371)), np.asarray((1.1137, 3.4813)), 0.2269, "interior", [])
    corner = _Wall("C", np.asarray((0.4131, 1.2273)), np.asarray((3.0517, 1.2273)), 0.1973, "interior", [])
    low = -(2.0371 - 1.2273) - past
    assert _extension_blocked(host, low, host.length, [host, corner], set()) is blocked


@pytest.mark.parametrize("upm", [1.0, 0.989])
def test_a_room_enclosed_by_another_is_not_counted_twice(upm: float) -> None:
    def rectangle(x0: float, y0: float, x1: float, y1: float) -> Polygon:
        return box(*(value * upm for value in (x0, y0, x1, y1)))

    inner = rectangle(3.1137, 3.0731, 5.0713, 4.8319)
    bodies = rectangle(2.9317, 2.8931, 5.2519, 5.0117).difference(inner).difference(rectangle(3.6131, 2.8, 4.4273, 3.1))
    width, thickness = (4.4273 - 3.6131) * upm, (3.0731 - 2.8931) * upm
    door = Opening("DOOR", ((3.6131 + 4.4273) / 2.0 * upm, (2.8931 + 3.0731) / 2.0 * upm), width, 0.0, thickness)
    outer = rectangle(0.5131, 0.5273, 7.4139, 7.3871).difference(rectangle(2.9317, 2.8931, 5.2519, 5.0117))
    outline = rectangle(0.3313, 0.3473, 7.5939, 7.5671).difference(rectangle(0.5131, 0.5273, 7.4139, 7.3871))
    rooms, _stats = door_bounded_rooms([outer, inner], [bodies, outline], [door], None, None, upm)
    assert len(rooms) == 2
    assert rooms[0].intersection(rooms[1]).area < 1e-6
    half_reveal = width * thickness / 2.0
    assert min(room.area for room in rooms) == pytest.approx(inner.area + half_reveal, rel=0.01)
    assert max(room.area for room in rooms) == pytest.approx(outer.area + half_reveal, rel=0.01)
