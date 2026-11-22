"""The wall above an opening is flush with, and as thick as, the wall it interrupts (plan, 2D symbols, IFC, 3D DXF)."""

from __future__ import annotations

import math
from pathlib import Path

import ezdxf
import ifcopenshell
import ifcopenshell.geom
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from samcloud.bim.core.geometry import compute_scene_geometry
from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.model import OpeningElement
from samcloud.bim.export import export_level
from samcloud.bim.opening_flush import check_opening_flush
from samcloud.bim.plan_model import BuildingPlan, PlanOpening, PlanRoom, PlanWall, _openings, load_bim_export_config, save_plan_model
from samcloud.bim.scene_builder import build_scene
from samcloud.dxf.writer3d import export_ifc_3d_dxf
from samcloud.ortho.vector.double_walls import DoubleWallSettings, WallRun, _Face, _wall_bodies, update_opening_symbols
from samcloud.ortho.vector.indoor import Opening, opening_entities
from samcloud.ortho.vector.opening_bands import apply_lintel_bands, flush_interrupted_runs, host_run, run_band

SETTINGS = DoubleWallSettings()
STOREY = 2.5
HEAD = 2.013
JAMB_A, JAMB_B = 2.013, 2.903  # off every grid line


# --------------------------------------------------------------------------
# wall stage: measured band above the door
# --------------------------------------------------------------------------

class _Stage:
    """Runs and openings in a wall frame (u along, v across) turned by ``angle`` and scaled by ``upm``."""

    def __init__(self, angle: float = 0.0, upm: float = 1.0) -> None:
        self.upm = upm
        self.angle = math.radians(angle)
        self.u = np.asarray((math.cos(self.angle), math.sin(self.angle)))
        self.v = np.asarray((-self.u[1], self.u[0]))
        self.origin = np.asarray((3.137, 1.271)) * upm
        self.faces: list[_Face] = []
        self.runs: list[WallRun] = []
        self.openings: list[Opening] = []

    def at(self, u: float, v: float) -> np.ndarray:
        return self.origin + (self.u * u + self.v * v) * self.upm

    def run(self, start: tuple[float, float], end: tuple[float, float], thickness: float, evidence: str, side: int = 1) -> WallRun:
        face = _Face(len(self.faces), self.at(*start), self.at(*end), {"type": "polyline", "points": [], "closed": False}, side=side)
        self.faces.append(face)
        run = WallRun(face.index, 0.0, face.length, thickness * self.upm, "exterior", evidence)
        self.runs.append(run)
        return run

    def opening(self, kind: str, low: float, high: float, v: float = 0.0, turn: float = 0.0, head: float | None = HEAD) -> Opening:
        direction = math.radians(turn)
        middle = (low + high) / 2.0
        centre = self.at(middle, v) if not turn else self.at(v, middle)
        item = Opening(
            kind, (float(centre[0]), float(centre[1])), (high - low) * self.upm, (self.angle + direction) % math.pi, 0.301 * self.upm,
            head_m=head, sill_m=0.0, height_source="points", leaf_detected=False,
        )
        self.openings.append(item)
        return item

    def band(self, run: WallRun) -> tuple[float, float]:
        """The run's band in the v coordinate (metres), at the middle of the door."""
        face = {item.index: item for item in self.faces}[run.face]
        low, high = run_band(run, face, self.at((JAMB_A + JAMB_B) / 2.0, 0.0), self.v)
        return low / self.upm, high / self.upm

    def scan(self, planes: list[float], *, z: tuple[float, float] = (HEAD + 0.06, 0.95 * STOREY), noise: float = 0.004, seed: int = 7) -> np.ndarray:
        rng = np.random.default_rng(seed)
        parts = []
        for plane in planes:
            u, height = rng.uniform(JAMB_A, JAMB_B, 900), rng.uniform(*z, 900)
            v = plane + rng.normal(0.0, noise, 900)
            xy = self.origin + (self.u[None, :] * u[:, None] + self.v[None, :] * v[:, None]) * self.upm
            parts.append(np.column_stack((xy, height * self.upm)))
        return np.vstack(parts)

    def measure(self, points: np.ndarray) -> dict[str, int]:
        return apply_lintel_bands(self.runs, self.faces, self.openings, points, STOREY * self.upm, self.upm, SETTINGS)


@pytest.mark.parametrize("upm", [1.0, 0.9891])
@pytest.mark.parametrize("angle", [0.0, 23.0, 91.3])
@pytest.mark.parametrize("truth", [0.117, 0.228, 0.364])
def test_measured_lintel_band_wins_over_the_assumed_thickness(truth: float, angle: float, upm: float) -> None:
    stage = _Stage(angle, upm)
    first = stage.run((0.0, 0.0), (JAMB_A, 0.0), 0.285, "exterior")
    second = stage.run((JAMB_B, 0.0), (5.07, 0.0), 0.285, "exterior")
    stage.opening("DOOR", JAMB_A, JAMB_B)

    stats = stage.measure(stage.scan([0.0, truth]))

    assert stats["measured"] == 1 and stats["runs_changed"] == 2
    for run in (first, second):  # the wall and its lintel: both pieces
        low, high = stage.band(run)
        assert low == pytest.approx(0.0, abs=1e-9)  # the measured face of the wall never moves
        assert high == pytest.approx(truth, abs=0.008)
        assert run.evidence == "lintel"


@pytest.mark.parametrize("evidence", ["face", "face-collinear", "face-beside", "room"])
def test_a_measured_thickness_is_not_replaced(evidence: str) -> None:
    stage = _Stage()
    run = stage.run((0.0, 0.0), (5.07, 0.0), 0.285, evidence)
    stage.opening("DOOR", JAMB_A, JAMB_B)
    assert stage.measure(stage.scan([0.0, 0.207]))["runs_changed"] == 0
    assert run.thickness == pytest.approx(0.285)


def test_one_seen_face_is_no_measurement() -> None:
    stage = _Stage()
    run = stage.run((0.0, 0.0), (5.07, 0.0), 0.285, "exterior")
    stage.opening("DOOR", JAMB_A, JAMB_B)
    assert stage.measure(stage.scan([0.0]))["measured"] == 0
    assert run.thickness == pytest.approx(0.285)


def test_a_flat_soffit_or_casing_top_is_no_second_face() -> None:
    stage = _Stage()
    run = stage.run((0.0, 0.0), (5.07, 0.0), 0.285, "learned")
    stage.opening("DOOR", JAMB_A, JAMB_B)
    points = np.vstack((stage.scan([0.0]), stage.scan([0.193], z=(HEAD + 0.052, HEAD + 0.079), seed=11)))
    assert stage.measure(points)["measured"] == 0
    assert run.thickness == pytest.approx(0.285)


def test_drift_layers_of_one_face_are_no_wall() -> None:
    stage = _Stage()
    run = stage.run((0.0, 0.0), (5.07, 0.0), 0.285, "exterior")
    stage.opening("DOOR", JAMB_A, JAMB_B)
    assert stage.measure(stage.scan([0.0, 0.027, 0.051, 0.078, 0.113], noise=0.013))["measured"] == 0
    assert run.thickness == pytest.approx(0.285)


def test_an_opening_without_a_measured_head_or_without_a_lintel_is_not_measured() -> None:
    for kind, head in (("DOOR", None), ("passage", HEAD), ("WINDOW", HEAD), ("DOOR", 2.41)):
        stage = _Stage()
        run = stage.run((0.0, 0.0), (5.07, 0.0), 0.285, "exterior")
        stage.opening(kind, JAMB_A, JAMB_B, head=head)
        assert stage.measure(stage.scan([0.0, 0.207]))["measured"] == 0
        assert run.thickness == pytest.approx(0.285)


@pytest.mark.parametrize("angle", [0.0, 31.0])
def test_the_measured_face_ends_on_the_jamb_of_the_crossing_wall(angle: float) -> None:
    stage = _Stage(angle)
    host = stage.run((0.0, 0.0), (JAMB_B, 0.0), 0.285, "exterior")
    stage.opening("DOOR", JAMB_A, JAMB_B)
    # a passage in the crossing wall at the door's end: its jamb is the end of that wall, 0.241 behind the host face
    stage.opening("passage", 0.241, 1.133, v=JAMB_B + 0.052, turn=90.0)

    stage.measure(stage.scan([0.0, 0.229]))

    assert stage.band(host)[1] == pytest.approx(0.241, abs=1e-6)


def test_a_corner_that_contradicts_the_lintel_keeps_the_wall() -> None:
    stage = _Stage()
    host = stage.run((0.0, 0.0), (JAMB_B, 0.0), 0.268, "exterior")
    stage.opening("DOOR", JAMB_A, JAMB_B)
    stage.opening("passage", 0.2675, 1.133, v=JAMB_B + 0.052, turn=90.0)
    assert stage.measure(stage.scan([0.0, 0.176]))["corner_conflicts"] == 1
    assert host.thickness == pytest.approx(0.268)


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_a_piece_built_between_the_jambs_moves_onto_its_two_measured_faces(side: float) -> None:
    stage = _Stage(12.0)
    # the opening line is a wall face; the built piece was centred on it
    host = stage.run((JAMB_A - 0.02, -0.1525), (JAMB_B + 0.02, -0.1525), 0.305, "door-host")
    stage.opening("DOOR", JAMB_A, JAMB_B)

    stage.measure(stage.scan([0.0, side * 0.236]))

    low, high = stage.band(host)
    assert sorted((low, high)) == pytest.approx(sorted((0.0, side * 0.236)), abs=0.008)
    assert host.evidence == "door-host"


@pytest.mark.parametrize("angle", [0.0, 23.0])
@pytest.mark.parametrize("strong_first", [True, False])
def test_unequal_neighbour_pieces_share_the_better_band(strong_first: bool, angle: float) -> None:
    stage = _Stage(angle)
    strong = (0.0, JAMB_A) if strong_first else (JAMB_B, 5.07)
    weak = (JAMB_B, 5.07) if strong_first else (0.0, JAMB_A)
    measured = stage.run((strong[0], 0.0), (strong[1], 0.0), 0.203, "face")
    learned = stage.run((weak[0], 0.221), (weak[1], 0.221), 0.237, "learned", side=-1)  # recorded from the other face
    stage.opening("DOOR", JAMB_A, JAMB_B)

    assert flush_interrupted_runs(stage.runs, stage.faces, stage.openings, 1.0, SETTINGS) == 1

    assert stage.band(measured) == pytest.approx((0.0, 0.203), abs=1e-9)
    assert stage.band(learned) == pytest.approx((0.0, 0.203), abs=1e-9)


def test_a_measured_reveal_and_a_niche_keep_their_step() -> None:
    for evidence, thickness in (("face", 0.331), ("niche", 0.147)):
        stage = _Stage()
        stage.run((0.0, 0.0), (JAMB_A, 0.0), 0.203, "face")
        other = stage.run((JAMB_B, 0.0), (5.07, 0.0), thickness, evidence)
        stage.opening("DOOR", JAMB_A, JAMB_B)
        assert flush_interrupted_runs(stage.runs, stage.faces, stage.openings, 1.0, SETTINGS) == 0
        assert other.thickness == pytest.approx(thickness)


@pytest.mark.parametrize("angle", [0.0, 23.0])
@pytest.mark.parametrize("kind", ["DOOR", "passage", "WINDOW"])
def test_2d_cut_jambs_and_symbols_stand_on_both_faces_of_the_host(kind: str, angle: float) -> None:
    stage = _Stage(angle)
    thickness = 0.163
    host = stage.run((0.0, 0.0), (JAMB_A, 0.0), thickness, "face")
    stage.run((JAMB_B, 0.0), (5.07, 0.0), thickness, "face")
    opening = stage.opening(kind, JAMB_A, JAMB_B)

    bands, slabs, cuts, _ = _wall_bodies(stage.runs, stage.faces, stage.openings, 1.0, SETTINGS)

    assert host_run(opening, stage.runs, {face.index: face for face in stage.faces}, 1.0) is not None
    normal = np.asarray((-math.sin(opening.axis_angle), math.cos(opening.axis_angle)))
    sign = float(normal @ stage.v)
    assert sorted(value * sign for value in slabs[0]) == pytest.approx(sorted(stage.band(host)), abs=1e-6)
    body = unary_union(bands).difference(unary_union(cuts))
    assert not body.intersects(LineString([tuple(stage.at((JAMB_A + JAMB_B) / 2.0, v)) for v in (-0.05, thickness + 0.05)]))
    for u in (JAMB_A - 0.03, JAMB_B + 0.03):  # the same two faces on both sides of the cut
        assert body.covers(Point(tuple(stage.at(u, 0.002)))) and body.covers(Point(tuple(stage.at(u, thickness - 0.002))))
        assert not body.covers(Point(tuple(stage.at(u, -0.002)))) and not body.covers(Point(tuple(stage.at(u, thickness + 0.002))))
    entities = opening_entities([opening])
    layers = {"A-DOOR": entities if kind != "WINDOW" else [], "A-WIND": entities if kind == "WINDOW" else []}
    update_opening_symbols(layers, [opening], slabs, 1.0, [True])
    for entity in entities:
        marker = np.asarray(entity["_opening_center"]) - stage.origin
        assert float(marker @ stage.v) == pytest.approx(thickness / 2.0, abs=1e-6)
        if entity.get("_jamb"):
            ends = sorted(float((np.asarray(entity[key]) - stage.origin) @ stage.v) for key in ("start", "end"))
            assert ends == pytest.approx((0.0, thickness), abs=1e-6)
        if entity.get("name") == "WINDOW":
            assert abs(float(entity["yscale"])) == pytest.approx(thickness, abs=1e-6)


# --------------------------------------------------------------------------
# plan model -> scene -> IFC / 3D DXF
# --------------------------------------------------------------------------

def _turn(angle: float):
    radians = math.radians(angle)
    u, v = np.asarray((math.cos(radians), math.sin(radians))), np.asarray((-math.sin(radians), math.cos(radians)))
    origin = np.asarray((1.117, 2.341))
    return lambda x, y: [float(value) for value in origin + u * x + v * y]


def _plan(
    pieces: list[tuple[tuple[float, float], tuple[float, float], float, str]], openings: list[tuple[str, float, float, float]],
    angle: float = 0.0, upm: float = 1.0, rooms: list[list[tuple[float, float]]] = (),
) -> BuildingPlan:
    """Openings are given as detected: ON a face (``v``) with a thickness estimate of their own."""
    turn = _turn(angle)

    def local(x: float, y: float) -> list[float]:
        return [value * upm for value in turn(x, y)]

    walls = [
        PlanWall(f"W{index:03d}", [local(*start), local(*end)], thickness, "interior", evidence)
        for index, (start, end, thickness, evidence) in enumerate(pieces, start=1)
    ]
    detected = [
        Opening(
            kind, tuple(local((low + high) / 2.0, v)), (high - low) * upm, math.radians(angle) % math.pi, 0.301 * upm,
            sill_m=0.91 if kind == "WINDOW" else 0.0, head_m=2.43 if kind == "passage" else HEAD, height_source="points", evidence="3d:points; backing=frame",
        )
        for kind, low, high, v in openings
    ]
    plan_rooms = [
        PlanRoom(f"R{index:03d}", f"Room {index}", {"exterior": [local(*point) for point in ring], "holes": []}, Polygon(ring).area, local(*Polygon(ring).representative_point().coords[0]))
        for index, ring in enumerate(rooms, start=1)
    ]
    return BuildingPlan(
        project_name="flush", level_id="level_1", level_name="1", units_per_metre=upm,
        frame={"origin": [0, 0, 0], "scale_factor": 1.0, "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        floor_z=0.0, ceiling_z=STOREY * upm, walls=walls, rooms=plan_rooms, openings=_openings(detected, walls, upm, load_bim_export_config()),
    )


def _check(plan: BuildingPlan):
    scene = build_scene(plan)
    geometry = compute_scene_geometry(scene)
    return scene, geometry, check_opening_flush(plan, scene, geometry)


def _straight(thickness: float, second: float | None = None, evidence: tuple[str, str] = ("face", "face")):
    return [
        ((0.0, 0.0), (JAMB_A, 0.0), thickness, evidence[0]),
        ((JAMB_B, 0.0), (5.07, 0.0), thickness if second is None else second, evidence[1]),
        ((0.0, -1.9), (0.0, 1.9), 0.2, "face"),
        ((5.07, -1.9), (5.07, 1.9), 0.2, "face"),
    ]


def _inside(meshes: list[tuple[np.ndarray, np.ndarray]], point: np.ndarray) -> bool:
    """Inside any closed mesh: a vertical ray crosses it an odd number of times."""
    for vertices, faces in meshes:
        triangles = vertices[faces]
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        flat = np.abs(area) > 1e-12
        a, b, c, area = a[flat], b[flat], c[flat], area[flat]
        w1 = ((point[0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (point[1] - a[:, 1]) * (c[:, 0] - a[:, 0])) / area
        w2 = ((b[:, 0] - a[:, 0]) * (point[1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (point[0] - a[:, 0])) / area
        hit = (w1 >= 0.0) & (w2 >= 0.0) & (w1 + w2 <= 1.0)
        z = a[hit, 2] + w1[hit] * (b[hit, 2] - a[hit, 2]) + w2[hit] * (c[hit, 2] - a[hit, 2])
        if np.count_nonzero(z > point[2]) % 2:
            return True
    return False


def _ifc_meshes(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    meshes = []
    model = ifcopenshell.open(str(path))  # the file must live as long as its entities are tessellated
    for wall in model.by_type("IfcWall"):
        shape = ifcopenshell.geom.create_shape(settings, wall).geometry
        meshes.append((np.asarray(shape.verts).reshape(-1, 3), np.asarray(shape.faces).reshape(-1, 3)))
    return meshes


def _dxf_meshes(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    meshes = []
    for mesh in ezdxf.readfile(path).modelspace().query("MESH"):
        if not mesh.dxf.layer.startswith("3D-IfcWall"):
            continue
        faces = [(face[0], face[i], face[i + 1]) for face in mesh.faces for i in range(1, len(face) - 1)]
        meshes.append((np.asarray(mesh.vertices, dtype=float), np.asarray(faces, dtype=int)))
    return meshes


@pytest.mark.parametrize("upm", [1.0, 0.9891])
@pytest.mark.parametrize("angle", [0.0, 17.0, 118.4])
@pytest.mark.parametrize("thickness", [0.097, 0.424])
def test_thin_thick_and_slanted_walls_have_flush_lintels_in_ifc_and_3d_dxf(tmp_path: Path, thickness: float, angle: float, upm: float) -> None:
    plan = _plan(_straight(thickness), [("DOOR", JAMB_A, JAMB_B, thickness / 2.0)], angle, upm)
    opening = plan.openings[0]
    assert opening.wall_thickness_m == pytest.approx(thickness)
    turn = _turn(angle)
    assert np.asarray(opening.center) / upm == pytest.approx(turn((JAMB_A + JAMB_B) / 2.0, 0.0), abs=1e-9)
    assert opening.width_m == pytest.approx(JAMB_B - JAMB_A) and opening.host_wall_id in ("W001", "W002")

    scene, _, result = _check(plan)

    assert result["ok"] and result["checked"] == 1 and not result["proven"], result
    ifc = tmp_path / "flush.ifc"
    write_ifc(scene, ifc)
    export_ifc_3d_dxf(ifc, tmp_path / "flush.dxf")
    for meshes in (_ifc_meshes(ifc), _dxf_meshes(tmp_path / "flush.dxf")):
        for u in (JAMB_A - 0.011, JAMB_A + 0.013, (JAMB_A + JAMB_B) / 2.0 + 0.007, JAMB_B - 0.013, JAMB_B + 0.011):
            z = (HEAD + STOREY) / 2.0
            for v, solid in ((thickness / 2.0 - 0.003, True), (thickness / 2.0 + 0.003, False), (-thickness / 2.0 + 0.003, True), (-thickness / 2.0 - 0.003, False)):
                assert _inside(meshes, np.asarray([*turn(u, v), z])) is solid, (u, v)


@pytest.mark.parametrize("angle", [0.0, 29.0])
@pytest.mark.parametrize("reach", [3.0, 3.152])  # the door ends at the crossing wall's face / reaches into it
def test_a_door_beyond_the_end_of_its_wall_piece(angle: float, reach: float) -> None:
    pieces = [((0.0, 0.0), (2.113, 0.0), 0.243, "face"), ((3.1, -1.87), (3.1, 2.11), 0.2, "face"), ((0.0, -1.87), (0.0, 2.11), 0.2, "face")]
    plan = _plan(pieces, [("DOOR", 2.113, reach, 0.1215)], angle)
    scene, _, result = _check(plan)
    assert result["ok"] and result["checked"] == 1, result
    host = next(scene.elements[e.params.host_wall_id] for e in scene.elements.values() if isinstance(e, OpeningElement))
    assert host.params.thickness == pytest.approx(0.243)


@pytest.mark.parametrize("angle", [0.0, 41.0])
def test_an_opening_far_out_in_a_gap_takes_the_band_of_the_pieces_on_its_line(angle: float) -> None:
    pieces = [((0.0, 0.0), (1.51, 0.0), 0.163, "face"), ((3.53, 0.0), (5.07, 0.0), 0.163, "face")]
    plan = _plan(pieces, [("DOOR", 2.117, 2.917, 0.0815)], angle)
    assert plan.openings[0].host_wall_id is None
    scene, _, result = _check(plan)
    assert result["checked"] == 1 and result["ok"], result
    host = next(scene.elements[e.params.host_wall_id] for e in scene.elements.values() if isinstance(e, OpeningElement))
    turn = _turn(angle)
    normal = np.subtract(turn(0.0, 1.0), turn(0.0, 0.0))
    assert host.params.thickness == pytest.approx(0.163)
    assert float(np.subtract(host.params.start, turn(0.0, 0.0)) @ normal) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("kind", ["DOOR", "passage", "WINDOW"])
def test_doors_passages_and_windows_are_flush(kind: str) -> None:
    _, _, result = _check(_plan(_straight(0.283), [(kind, JAMB_A, JAMB_B, 0.1415)]))
    assert result["ok"] and result["checked"] == 1, result


def test_unequal_neighbour_pieces_get_one_lintel_band() -> None:
    _, _, result = _check(_plan(_straight(0.203, 0.236, ("room", "learned")), [("DOOR", JAMB_A, JAMB_B, 0.1015)]))
    assert result["ok"], result


def test_a_measured_reveal_stays_and_is_listed() -> None:
    plan = _plan(_straight(0.203, 0.341), [("DOOR", JAMB_A, JAMB_B, 0.1015)])
    scene, _, result = _check(plan)
    assert result["ok"] and result["proven"], result
    assert {round(e.params.thickness, 3) for e in scene.elements.values() if type(e).__name__ == "WallElement"} >= {0.203, 0.341}


def test_chaining_wall_pieces_does_not_move_the_faces_of_a_host() -> None:
    pieces = [((0.0, 0.0), (3.07, 0.0), 0.237, "room"), ((3.07, 0.0), (9.03, 0.0), 0.213, "room")]
    _, _, result = _check(_plan(pieces, [("DOOR", 1.013, 1.903, 0.1185)]))
    assert result["ok"], result


def _step_plan(lintel_thickness: float) -> BuildingPlan:
    """A door wall meets the end of a crossing wall that ends at the jamb of its own passage."""
    face = 0.1185  # the measured face of the door wall
    middle = face - lintel_thickness / 2.0
    pieces = [
        ((0.0, middle), (JAMB_A, middle), lintel_thickness, "face"),
        ((JAMB_B + 0.1185, -1.093), (JAMB_B + 0.1185, 3.07), 0.237, "room"),
    ]
    plan = _plan(pieces, [("DOOR", JAMB_A, JAMB_B, face)])
    turn = _turn(0.0)
    plan.openings.append(PlanOpening(
        "O002", "passage", turn(JAMB_B + 0.1185, (-1.093 - 0.1185) / 2.0), 1.093 - 0.1185, math.pi / 2.0, 0.237, host_wall_id="W002", head_m=2.43,
    ))
    return plan


def test_the_qa_finds_a_lintel_thicker_than_the_wall_end_it_meets() -> None:
    _, _, stepped = _check(_step_plan(0.285))
    assert not stepped["ok"]
    step = next(item for item in stepped["steps"] if item["opening"] == "O001")
    assert step["step_m"] == pytest.approx(0.048, abs=0.002) and "end" in step["where"]
    _, _, flush = _check(_step_plan(0.237))
    assert flush["ok"], flush


def test_the_qa_finds_an_opening_that_is_not_in_the_band_of_its_wall() -> None:
    plan = _plan(_straight(0.253), [("DOOR", JAMB_A, JAMB_B, 0.1265)])
    plan.openings[0].center = [plan.openings[0].center[0], plan.openings[0].center[1] + 0.1265]
    _, _, result = _check(plan)
    assert [item["where"] for item in result["steps"]] == ["plan"]
    assert result["max_step_m"] == pytest.approx(0.1265, abs=1e-3)


def test_the_export_reports_steps_and_keeps_the_ceiling_out_of_the_lintel(tmp_path: Path) -> None:
    face = 0.1265
    pieces = [
        ((0.0, 0.0), (5.07, 0.0), 0.253, "face"), ((0.0, 3.07), (5.07, 3.07), 0.253, "face"),
        ((0.0, 0.0), (0.0, 3.07), 0.253, "face"), ((5.07, 0.0), (5.07, 3.07), 0.253, "face"),
    ]
    # the room owns the reveal of its door
    ring = [(face, face), (JAMB_A, face), (JAMB_A, -face), (JAMB_B, -face), (JAMB_B, face), (5.07 - face, face), (5.07 - face, 3.07 - face), (face, 3.07 - face)]
    plan = _plan(pieces, [("DOOR", JAMB_A, JAMB_B, -face)], rooms=[ring])
    project = tmp_path / "proj"
    save_plan_model(plan, project / "products" / "dxf" / "level_1" / "ortho")

    export_level(project, "level_1")

    import json

    qa = json.loads((project / "products" / "bim" / "level_1" / "qa.json").read_text(encoding="utf-8"))
    assert qa["opening_flush"]["ok"] and qa["summary"].startswith("IFC4 OK") and "opening steps 0" in qa["summary"], qa["summary"]
    assert qa["ceiling_wall_overlap_max_m2"] < 1e-4
    assert qa["room_area_check"]["rooms"][0]["ifc_m2"] == pytest.approx(Polygon(ring).area, abs=1e-3)  # the whole reveal is floor


@pytest.mark.parametrize("angle", [0.0, 34.0])
def test_a_lintel_inside_one_room_cuts_the_ceiling_around_it(angle: float) -> None:
    # a free-standing wall with a door: the room reaches across the doorway and all around the wall
    pieces = [((1.07, 0.0), (4.03, 0.0), 0.193, "face")]
    ring = [(-0.51, -1.53), (5.57, -1.53), (5.57, 1.49), (-0.51, 1.49)]
    plan = _plan(pieces, [("DOOR", JAMB_A, JAMB_B, 0.0965)], angle, rooms=[ring])
    scene = build_scene(plan)
    geometry = compute_scene_geometry(scene)
    turn = _turn(angle)
    ceiling = unary_union([Polygon(item.outline) for item in geometry.ceilings.values()])
    assert not ceiling.covers(Point(turn((JAMB_A + JAMB_B) / 2.0, 0.0)))
    assert ceiling.covers(Point(turn((JAMB_A + JAMB_B) / 2.0, 0.31))) and ceiling.covers(Point(turn(0.31, 0.007)))


def test_a_space_around_two_pillars_with_different_corner_counts_is_exported(tmp_path: Path) -> None:
    # reveals join spaces that the walls used to split: a space may keep several holes of any shape
    pieces = [
        ((0.0, 0.0), (5.07, 0.0), 0.253, "face"), ((0.0, 3.07), (5.07, 3.07), 0.253, "face"),
        ((0.0, 0.0), (0.0, 3.07), 0.253, "face"), ((5.07, 0.0), (5.07, 3.07), 0.253, "face"),
    ]
    plan = _plan(pieces, [], rooms=[[(0.1265, 0.1265), (4.9435, 0.1265), (4.9435, 2.9435), (0.1265, 2.9435)]])
    square = [[1.21, 1.13], [1.21, 1.52], [1.63, 1.52], [1.63, 1.13]]
    corner = [[2.71, 1.09], [2.71, 1.93], [3.02, 1.93], [3.02, 1.47], [3.58, 1.47], [3.58, 1.09]]
    turn = _turn(0.0)
    plan.rooms[0].polygon["holes"] = [[turn(*point) for point in hole] for hole in (square, corner)]
    path = tmp_path / "holes.ifc"

    write_ifc(build_scene(plan), path)

    model = ifcopenshell.open(str(path))
    planes = [item.ConnectionGeometry.SurfaceOnRelatingElement for item in model.by_type("IfcRelSpaceBoundary") if item.ConnectionGeometry]
    counts = [sorted(len(line.Points) for line in plane.InnerBoundaries) for plane in planes if plane.InnerBoundaries]
    assert counts and all(item == [5, 7] for item in counts)


@pytest.mark.parametrize("with_reveal", [False, True])
def test_a_passage_detected_on_one_face_still_joins_the_rooms_across_its_reveal(with_reveal: bool) -> None:
    from samcloud.ortho.vector.door_rooms import _merge_through_passages, opening_slab
    from samcloud.ortho.vector.opening_bands import passage_reveals

    thickness, half = 0.237, (JAMB_B - JAMB_A) / 2.0
    middle = (JAMB_A + JAMB_B) / 2.0
    hall = Polygon([(1.41, thickness), (3.49, thickness), (3.49, 1.43), (1.41, 1.43)])
    room = Polygon([(0.13, -3.07), (4.93, -3.07), (4.93, 0.0), (0.13, 0.0)])
    passage = Opening("passage", (middle, thickness), 2.0 * half, 0.0, 0.11)  # on the hall's face
    reveals = passage_reveals([passage], [(-thickness, 0.0)], [True]) if with_reveal else []

    rooms = [room, hall]
    merged = _merge_through_passages(rooms, [opening_slab(passage)], 0.06, reveals)

    assert merged == (1 if with_reveal else 0)
    if with_reveal:
        assert isinstance(rooms[0], Polygon) and rooms[1] is None  # one connected room, the reveal is its floor
        assert rooms[0].covers(Point(middle, thickness / 2.0))


@pytest.mark.parametrize("island_height", [None, 1.31])
def test_a_space_around_an_island_of_walls_has_no_ceiling_in_the_full_height_ones(island_height: float | None) -> None:
    pieces = [
        ((0.0, 0.0), (5.07, 0.0), 0.253, "face"), ((0.0, 4.07), (5.07, 4.07), 0.253, "face"),
        ((0.0, 0.0), (0.0, 4.07), 0.253, "face"), ((5.07, 0.0), (5.07, 4.07), 0.253, "face"),
        ((1.73, 2.017), (3.31, 2.017), 0.183, "face"),
    ]
    ring = [(0.1265, 0.1265), (4.9435, 0.1265), (4.9435, 3.9435), (0.1265, 3.9435)]
    plan = _plan(pieces, [], rooms=[ring])
    plan.walls[-1].height_m = island_height
    geometry = compute_scene_geometry(build_scene(plan))
    ceiling = unary_union([Polygon(item.outline) for item in geometry.ceilings.values()])
    turn = _turn(0.0)
    assert ceiling.covers(Point(turn(2.52, 2.017))) is (island_height is not None)
    assert ceiling.covers(Point(turn(2.52, 2.31))) and ceiling.covers(Point(turn(0.51, 2.017)))


def test_the_refinement_still_starts_from_the_detection_on_the_face(tmp_path: Path) -> None:
    from samcloud.bim.plan_model import load_refine_plan

    plan = _plan(_straight(0.237), [("DOOR", JAMB_A, JAMB_B, 0.1185)], 23.0)
    plan.source = {"scale": {"detector_units_per_metre": 0.9891}}
    save_plan_model(plan, tmp_path)

    seed = load_refine_plan(tmp_path).openings[0]

    turn = _turn(23.0)
    assert seed.center == pytest.approx(turn((JAMB_A + JAMB_B) / 2.0, 0.1185), abs=1e-12)
    assert seed.wall_thickness_m == pytest.approx(0.301 / 0.9891)  # its own estimate, in detector metres
    assert plan.openings[0].wall_thickness_m == pytest.approx(0.237)


def test_a_measured_band_does_not_flip_the_host_between_two_pieces_that_tie() -> None:
    from samcloud.bim.plan_model import _walls_as_detected

    stage = _Stage(23.0)
    first = stage.run((0.0, 0.0), (JAMB_A, 0.0), 0.285, "exterior")
    second = stage.run((JAMB_B, 0.0), (5.07, 0.0), 0.285, "exterior")
    door = stage.opening("DOOR", JAMB_A, JAMB_B)
    before = host_run(door, stage.runs, {face.index: face for face in stage.faces}, 1.0)

    stage.measure(stage.scan([0.0, 0.207]))

    assert host_run(door, stage.runs, {face.index: face for face in stage.faces}, 1.0) is before
    assert first.detected[1] == second.detected[1] == pytest.approx(0.285)
    walls = [PlanWall("W001", [[0.0, 0.1035], [JAMB_A, 0.1035]], 0.207)]
    detected = _walls_as_detected(walls, [{"detected_centreline": [[0.0, 0.1425], [JAMB_A, 0.1425]], "detected_thickness": 0.285}])
    assert detected[0].thickness_m == pytest.approx(0.285) and detected[0].centreline[0][1] == pytest.approx(0.1425)


@pytest.mark.parametrize("lean", [-21.0, 14.0, 24.0])
def test_a_door_against_a_slanted_crossing_wall_leaves_no_step_or_hole(lean: float) -> None:
    thickness = 0.217
    shift = math.tan(math.radians(lean))
    face = 2.903 + thickness / 2.0 * abs(shift)  # the crossing wall's near face meets the door wall beyond the door
    axis = face + 0.1 / math.cos(math.radians(lean))
    pieces = [
        ((0.0, 0.0), (JAMB_A, 0.0), thickness, "face"),
        ((axis - 1.7 * shift, -1.7), (axis + 1.9 * shift, 1.9), 0.2, "face"),
        ((0.0, -1.9), (0.0, 1.9), 0.2, "face"),
    ]
    scene, _, result = _check(_plan(pieces, [("DOOR", JAMB_A, JAMB_B, thickness / 2.0)], 17.0))
    assert result["checked"] == 1 and result["ok"], result
    assert scene.derived["samcloud_build"]["wall_overlaps"] == []


def test_the_qa_names_a_door_that_reaches_behind_a_crossing_walls_end() -> None:
    thickness = 0.207
    pieces = [
        ((0.0, 0.0), (5.07, 0.0), thickness, "face"),
        ((JAMB_B + 0.077, thickness / 2.0), (JAMB_B + 0.077, 2.2), 0.2, "face"),  # its side lies 23 mm inside the span
    ]
    _, _, result = _check(_plan(pieces, [("DOOR", JAMB_A, JAMB_B, thickness / 2.0)]))
    assert [item["opening"] for item in result["blocked_spans"]] == ["O001"]
    assert result["blocked_spans"][0]["overlap_m"] == pytest.approx(0.023, abs=0.006)
