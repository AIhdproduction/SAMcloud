"""3D furniture reconstruction from scan points (samcloud.bim.furniture_recon), IFC solids/mesh export and the 3D DXF.

All point clouds are synthetic: points sampled on the visible surfaces of known furniture (no floor-facing faces), with
noise, random dropout and occluded parts.
"""

from __future__ import annotations

import math

import ezdxf
import ifcopenshell
import ifcopenshell.geom
import numpy as np
import pytest
from shapely.geometry import Polygon, box as shapely_box

from samcloud.bim.furniture_recon import ReconSettings, SceneCloud, reconstruct_items
from samcloud.bim.furniture_recon.geometry import Mesh, frame_for_footprint, projected_outline
from samcloud.bim.furniture_recon.mesh import fit_mesh, is_watertight
from samcloud.bim.furniture_recon.parametric import fit_bed, fit_chair, fit_sofa, fit_table, surface_level
from samcloud.bim.plan_model import BuildingPlan, PlanFurniture

SETTINGS = ReconSettings()
NOISE = 0.004


# ----------------------------------------------------------------------------------------------------------------------
# synthetic scans
# ----------------------------------------------------------------------------------------------------------------------

def _rect(rng: np.random.Generator, a: np.ndarray, u: np.ndarray, v: np.ndarray, density: float) -> np.ndarray:
    area = float(np.linalg.norm(np.cross(u, v)))
    count = max(int(area * density), 3)
    s, t = rng.random(count), rng.random(count)
    return a + s[:, None] * u + t[:, None] * v


def box_points(rng: np.random.Generator, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, density: float = 4000.0, sides: bool = True, top: bool = True) -> np.ndarray:
    parts = []
    if top:
        parts.append(_rect(rng, np.array([x0, y0, z1]), np.array([x1 - x0, 0, 0]), np.array([0, y1 - y0, 0]), density))
    if sides:
        h = np.array([0, 0, z1 - z0])
        parts.append(_rect(rng, np.array([x0, y0, z0]), np.array([x1 - x0, 0, 0]), h, density))
        parts.append(_rect(rng, np.array([x0, y1, z0]), np.array([x1 - x0, 0, 0]), h, density))
        parts.append(_rect(rng, np.array([x0, y0, z0]), np.array([0, y1 - y0, 0]), h, density))
        parts.append(_rect(rng, np.array([x1, y0, z0]), np.array([0, y1 - y0, 0]), h, density))
    return np.vstack(parts)


def cylinder_points(rng: np.random.Generator, cx: float, cy: float, r: float, z0: float, z1: float, density: float = 4000.0, top: bool = True) -> np.ndarray:
    side_count = int(2 * math.pi * r * (z1 - z0) * density) + 3
    angle = rng.random(side_count) * 2 * math.pi
    side = np.column_stack((cx + r * np.cos(angle), cy + r * np.sin(angle), z0 + rng.random(side_count) * (z1 - z0)))
    if not top:
        return side
    top_count = int(math.pi * r * r * density) + 3
    radius = r * np.sqrt(rng.random(top_count))
    angle = rng.random(top_count) * 2 * math.pi
    cap = np.column_stack((cx + radius * np.cos(angle), cy + radius * np.sin(angle), np.full(top_count, z1)))
    return np.vstack((side, cap))


def scan(rng: np.random.Generator, *parts: np.ndarray, dropout: float = 0.3) -> np.ndarray:
    points = np.vstack(parts)
    points = points[rng.random(len(points)) >= dropout]
    return points + rng.normal(0.0, NOISE, points.shape)


def table_with_legs(rng: np.random.Generator, legs: int = 4, length: float = 1.2, width: float = 0.8, top: float = 0.74) -> np.ndarray:
    x0, x1, y0, y1 = -length / 2, length / 2, -width / 2, width / 2
    parts = [box_points(rng, x0, x1, y0, y1, top - 0.03, top)]
    corners = [(x0 + 0.05, y0 + 0.05), (x1 - 0.10, y0 + 0.05), (x1 - 0.10, y1 - 0.10), (x0 + 0.05, y1 - 0.10)]
    for x, y in corners[:legs]:
        parts.append(box_points(rng, x, x + 0.05, y, y + 0.05, 0.0, top - 0.03, top=False))
    return scan(rng, *parts)


FOOTPRINT_TABLE = shapely_box(-0.6, -0.4, 0.6, 0.4)


# ----------------------------------------------------------------------------------------------------------------------
# parametric families
# ----------------------------------------------------------------------------------------------------------------------

def test_table_with_four_legs_recovers_top_height_size_and_legs() -> None:
    points = table_with_legs(np.random.default_rng(1))
    solids = fit_table(points, FOOTPRINT_TABLE, SETTINGS, (0.0, 1.3))
    assert solids is not None
    top = next(s for s in solids if s.role == "top")
    assert top.z1 == pytest.approx(0.74, abs=0.03)
    assert top.kind == "box"
    assert top.sx == pytest.approx(1.2, abs=0.05) and top.sy == pytest.approx(0.8, abs=0.05)
    legs = [s for s in solids if s.role == "leg"]
    assert len(legs) == 4
    assert all(s.source == "measured" for s in legs)
    assert all(s.z0 == 0.0 and s.z1 == pytest.approx(top.z0, abs=1e-9) for s in legs)


def test_table_with_two_occluded_legs_completes_them_by_symmetry() -> None:
    points = table_with_legs(np.random.default_rng(2), legs=2)
    solids = fit_table(points, FOOTPRINT_TABLE, SETTINGS, (0.0, 1.3))
    legs = [s for s in solids or [] if s.role == "leg"]
    assert len(legs) == 4
    assert sum(s.source == "symmetry" for s in legs) == 2


def test_table_without_any_leg_evidence_gets_prior_legs_flagged() -> None:
    rng = np.random.default_rng(3)
    points = scan(rng, box_points(rng, -0.6, 0.6, -0.4, 0.4, 0.71, 0.74))
    solids = fit_table(points, FOOTPRINT_TABLE, SETTINGS, (0.0, 1.3))
    legs = [s for s in solids or [] if s.role == "leg"]
    assert len(legs) == 4 and all(s.source == "prior" for s in legs)


def test_round_pedestal_table_gives_cylinder_top_and_pedestal() -> None:
    rng = np.random.default_rng(4)
    points = scan(rng, cylinder_points(rng, 0.0, 0.0, 0.45, 0.69, 0.72), cylinder_points(rng, 0.0, 0.0, 0.06, 0.0, 0.69, top=False))
    footprint = Polygon([(0.45 * math.cos(a), 0.45 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 32, endpoint=False)])
    solids = fit_table(points, footprint, SETTINGS, (0.0, 1.3))
    assert solids is not None
    top = next(s for s in solids if s.role == "top")
    assert top.kind == "cylinder"
    assert top.radius == pytest.approx(0.45, abs=0.05)
    assert top.z1 == pytest.approx(0.72, abs=0.03)
    pedestals = [s for s in solids if s.role == "pedestal"]
    assert len(pedestals) == 1 and math.hypot(pedestals[0].cx, pedestals[0].cy) < 0.05


def test_floor_clutter_under_a_table_is_not_a_support() -> None:
    rng = np.random.default_rng(5)
    clutter = box_points(rng, -0.2, 0.1, -0.2, 0.1, 0.0, 0.18)  # a bag on the floor
    points = np.vstack((table_with_legs(rng), scan(rng, clutter)))
    solids = fit_table(points, FOOTPRINT_TABLE, SETTINGS, (0.0, 1.3))
    assert not [s for s in solids or [] if s.role == "pedestal"]
    assert len([s for s in solids or [] if s.role == "leg"]) == 4


def chair_points(rng: np.random.Generator, back: bool = True) -> np.ndarray:
    parts = [box_points(rng, -0.22, 0.22, -0.22, 0.22, 0.42, 0.46)]
    if back:
        parts.append(box_points(rng, -0.21, 0.21, 0.19, 0.22, 0.46, 0.90))
    for x, y in ((-0.21, -0.21), (0.17, -0.21), (0.17, 0.17), (-0.21, 0.17)):
        parts.append(box_points(rng, x, x + 0.04, y, y + 0.04, 0.0, 0.42, top=False))
    return scan(rng, *parts)


def test_chair_with_back_recovers_seat_back_and_legs() -> None:
    points = chair_points(np.random.default_rng(6))
    solids = fit_chair(points, shapely_box(-0.23, -0.23, 0.23, 0.23), SETTINGS, (0.0, 1.4))
    assert solids is not None
    seat = next(s for s in solids if s.role == "seat")
    assert seat.z1 == pytest.approx(0.46, abs=0.03)
    assert seat.sx == pytest.approx(0.44, abs=0.05) and seat.sy == pytest.approx(0.44, abs=0.05)
    back = [s for s in solids if s.role == "back"]
    assert len(back) == 1
    assert back[0].z1 == pytest.approx(0.90, abs=0.03)
    assert back[0].cy == pytest.approx(0.205, abs=0.04)  # at the rear edge
    assert len([s for s in solids if s.role == "leg"]) == 4


def test_chair_whose_seat_is_hidden_gets_prior_seat_and_measured_back() -> None:
    rng = np.random.default_rng(7)
    points = scan(rng, box_points(rng, -0.21, 0.21, 0.19, 0.22, 0.60, 0.90))  # only the back above a table edge is seen
    solids = fit_chair(points, shapely_box(-0.23, -0.23, 0.23, 0.23), SETTINGS, (0.0, 1.4))
    assert solids is not None
    seat = next(s for s in solids if s.role == "seat")
    assert seat.source == "prior"
    assert next(s for s in solids if s.role == "back").source == "measured"
    assert all(s.source == "prior" for s in solids if s.role == "leg")


def test_sofa_with_back_and_arms() -> None:
    rng = np.random.default_rng(8)
    parts = [
        box_points(rng, -1.0, 1.0, -0.45, 0.45, 0.0, 0.43),
        box_points(rng, -1.0, 1.0, 0.25, 0.45, 0.43, 0.80),  # back along the long edge
        box_points(rng, -1.0, -0.85, -0.45, 0.25, 0.43, 0.62),  # arms
        box_points(rng, 0.85, 1.0, -0.45, 0.25, 0.43, 0.62),
    ]
    points = scan(rng, *parts)
    solids = fit_sofa(points, shapely_box(-1.0, -0.45, 1.0, 0.45), SETTINGS, (0.0, 1.4))
    assert solids is not None
    base = next(s for s in solids if s.role == "base")
    assert base.z1 == pytest.approx(0.43, abs=0.03)
    backs = [s for s in solids if s.role == "back"]
    arms = [s for s in solids if s.role == "arm"]
    assert len(backs) == 1 and len(arms) == 2
    assert backs[0].z1 == pytest.approx(0.80, abs=0.03)
    assert backs[0].sx == pytest.approx(2.0, abs=0.08)
    assert all(arm.z1 == pytest.approx(0.62, abs=0.04) for arm in arms)
    assert all(arm.sy == pytest.approx(0.15, abs=0.05) for arm in arms)


def test_bed_with_headboard() -> None:
    rng = np.random.default_rng(9)
    points = scan(rng, box_points(rng, -1.0, 1.0, -0.8, 0.8, 0.0, 0.50), box_points(rng, 0.94, 1.0, -0.8, 0.8, 0.0, 1.0))
    solids = fit_bed(points, shapely_box(-1.0, -0.8, 1.0, 0.8), SETTINGS, (0.0, 1.6))
    assert solids is not None
    base = next(s for s in solids if s.role == "base_mattress")
    assert base.z1 == pytest.approx(0.50, abs=0.03)
    head = [s for s in solids if s.role == "headboard"]
    assert len(head) == 1
    assert head[0].z1 == pytest.approx(1.0, abs=0.03)
    assert head[0].cx == pytest.approx(0.97, abs=0.05)
    assert head[0].sx == pytest.approx(1.6, abs=0.08)


def test_bed_outline_swallowing_a_low_neighbour_is_cut_back_to_the_sleeping_surface() -> None:
    # The plan outline of the bed also holds a 0.45 m high bench along its long side (it joined the mask): 0.5 m deeper than the bed.
    rng = np.random.default_rng(11)
    points = scan(rng, box_points(rng, -1.0, 1.0, -0.9, 0.9, 0.0, 0.55), box_points(rng, -1.0, 0.0, 0.9, 1.4, 0.0, 0.25))
    outline = shapely_box(-1.0, -0.9, 1.0, 1.4)
    solids = fit_bed(points, outline, SETTINGS, (0.0, 1.6))
    assert solids is not None
    base = next(s for s in solids if s.role == "base_mattress")
    _x0, y0, _x1, y1 = base.outline().bounds
    assert y1 - y0 == pytest.approx(1.8, abs=0.12)  # the bed's own depth, not 2.3 m
    assert y1 == pytest.approx(0.93, abs=0.06) and y0 == pytest.approx(-0.9, abs=0.06)
    # a bed measured over its whole outline keeps it (no trimming without a gap of more than bed_trim_gap_m)
    rng = np.random.default_rng(12)
    whole = scan(rng, box_points(rng, -1.0, 1.0, -0.8, 0.8, 0.0, 0.50))
    base = next(s for s in fit_bed(whole, shapely_box(-1.0, -0.8, 1.0, 0.8), SETTINGS, (0.0, 1.6)) if s.role == "base_mattress")
    _x0, y0, _x1, y1 = base.outline().bounds
    assert y1 - y0 == pytest.approx(1.6, abs=0.01)


def test_surface_level_finds_the_top_not_the_faces_below() -> None:
    rng = np.random.default_rng(10)
    points = scan(rng, box_points(rng, -0.5, 0.5, -0.3, 0.3, 0.0, 0.74))
    level = surface_level(points, 0.2, 1.2)
    assert level is not None and level.top == pytest.approx(0.74, abs=0.02)


# ----------------------------------------------------------------------------------------------------------------------
# mesh
# ----------------------------------------------------------------------------------------------------------------------

def wc_points(rng: np.random.Generator) -> np.ndarray:
    bowl = cylinder_points(rng, 0.0, 0.0, 0.18, 0.30, 0.40)
    pedestal = cylinder_points(rng, 0.0, 0.05, 0.10, 0.0, 0.30, top=False)
    cistern = box_points(rng, -0.18, 0.18, 0.20, 0.36, 0.40, 0.80)
    return scan(rng, bowl, pedestal, cistern, dropout=0.4)


def test_wc_mesh_is_closed_bounded_and_matches_the_height() -> None:
    points = wc_points(np.random.default_rng(11))
    mesh = fit_mesh(points, z_bottom=0.0, z_top=float(np.percentile(points[:, 2], 99.5)), max_triangles=3000)
    assert mesh is not None
    assert is_watertight(mesh)
    assert mesh.volume() > 0  # outward normals
    assert len(mesh.faces) <= 3000
    assert float(mesh.vertices[:, 2].max()) == pytest.approx(0.80, abs=0.03)
    tri = mesh.vertices[mesh.faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    assert float(areas.min()) > 0.0
    # the pedestal is narrower than the bowl
    low = mesh.vertices[mesh.vertices[:, 2] < 0.2]
    high = mesh.vertices[(mesh.vertices[:, 2] > 0.32) & (mesh.vertices[:, 2] < 0.40)]
    assert np.ptp(low[:, 0]) < np.ptp(high[:, 0])


def test_mesh_drops_a_tap_above_a_bathtub_rim() -> None:
    rng = np.random.default_rng(12)
    tub = box_points(rng, -0.85, 0.85, -0.38, 0.38, 0.0, 0.55)
    tap = box_points(rng, -0.84, -0.80, -0.02, 0.02, 0.55, 0.70, density=1500)
    points = scan(rng, tub, tap)
    mesh = fit_mesh(points, z_bottom=0.0, z_top=float(np.percentile(points[:, 2], 99.5)))
    assert mesh is not None and is_watertight(mesh)
    assert float(mesh.vertices[:, 2].max()) <= 0.62


# ----------------------------------------------------------------------------------------------------------------------
# plan pipeline
# ----------------------------------------------------------------------------------------------------------------------

FINE_NAMES = ["table", "chair", "toilet", "kitchen cabinet", "shelf"]


def _cloud(*groups: tuple[np.ndarray, str]) -> SceneCloud:
    xyz = np.vstack([points for points, _ in groups])
    fine = np.concatenate([np.full(len(points), FINE_NAMES.index(name) + 1 if name else 0, dtype=np.int32) for points, name in groups])
    return SceneCloud(xyz, fine, np.zeros(len(xyz), dtype=np.int64), list(FINE_NAMES))


def _item(identifier: str, label: str, semantic: str, footprint: Polygon, **extra) -> PlanFurniture:
    ring = [[float(x), float(y)] for x, y in list(footprint.exterior.coords)[:-1]]
    defaults = {"instance_id": None, "catalogue_kind": None, "layer": "A-FURN", "default_height_m": 0.8}
    defaults.update(extra)
    return PlanFurniture(identifier, label=label, semantic_label=semantic, footprint=ring, **defaults)


def test_pipeline_table_and_chair_under_it_keep_their_own_points() -> None:
    rng = np.random.default_rng(13)
    table = table_with_legs(rng) + np.array([2.0, 1.0, 0.0])
    chair = chair_points(rng) * np.array([1.0, 1.0, 1.0]) + np.array([2.35, 1.2, 0.0])
    items = [
        _item("F001", "Tisch", "table", shapely_box(1.4, 0.6, 2.6, 1.4), z_min=0.0, z_max=0.74, z_source="instance_bbox"),
        _item("F002", "Stuhl", "chair", shapely_box(2.12, 0.97, 2.58, 1.43)),
    ]
    report = reconstruct_items(items, _cloud((table, "table"), (chair, "chair")), units_per_metre=1.0, floor_z=0.0, ceiling_z=2.5, settings=SETTINGS)
    table_item, chair_item = items
    assert table_item.shape_source == "parametric"
    roles = sorted(solid["role"] for solid in table_item.solids)
    assert roles.count("leg") == 4 and "top" in roles
    top = next(solid for solid in table_item.solids if solid["role"] == "top")
    assert top["z_max"] == pytest.approx(0.74, abs=0.03)
    assert top["part_source"] == "measured" and top["type"] == "box"
    assert chair_item.shape_source == "parametric"
    assert chair_item.z_source == "points"  # the default chair height became a measured one
    assert chair_item.z_max == pytest.approx(0.90, abs=0.03)
    assert report["heights_measured"] >= 1 and report["default_heights_left"] == 0
    for item in items:
        assert item.fit["after"]["coverage"] >= 0.6
        assert item.fit["after"]["mean_m"] <= 0.03
        assert item.fit["volume_ratio"] < 1.0


def test_pipeline_wc_becomes_a_mesh_in_plan_units() -> None:
    upm = 1.25  # local units per metre: the plan frame is not metric
    points = wc_points(np.random.default_rng(14))
    local = points.copy()
    local[:, :2] = local[:, :2] * upm + np.array([4.0, 3.0])
    local[:, 2] = local[:, 2] * upm + 0.5  # floor at local z 0.5
    items = [_item("F010", "WC", "WC", shapely_box(4.0 - 0.25 * upm, 3.0 - 0.25 * upm, 4.0 + 0.25 * upm, 3.0 + 0.45 * upm), catalogue_kind="WC_WALL", layer="P-SANI", floor_contact=True)]
    reconstruct_items(items, _cloud((local, "toilet")), units_per_metre=upm, floor_z=0.5, ceiling_z=0.5 + 2.5 * upm, settings=SETTINGS)
    wc = items[0]
    assert wc.shape_source == "mesh"
    vertices = np.asarray(wc.mesh["vertices"])
    assert (vertices[:, 2].max() - 0.5) / upm == pytest.approx(0.80, abs=0.03)
    assert (vertices[:, 2].min() - 0.5) / upm == pytest.approx(0.0, abs=1e-6)  # stands on the floor
    assert len(wc.mesh["faces"]) <= SETTINGS.mesh_max_triangles
    assert Polygon(wc.footprint).is_valid
    assert wc.fit["after"]["coverage"] > wc.fit["before"]["coverage"]


def test_pipeline_measures_the_worktop_height_of_a_kitchen_run() -> None:
    rng = np.random.default_rng(15)
    run = scan(rng, box_points(rng, 0.0, 2.4, 0.0, 0.6, 0.0, 0.92))
    footprint = shapely_box(0.0, 0.0, 2.4, 0.6)
    items = [
        _item("F020", "Unterschrank", "kitchen cabinet", footprint, catalogue_kind="kitchen_base", layer="A-EINB", default_height_m=0.86),
        _item("F021", "Arbeitsplatte", "worktop", footprint, catalogue_kind="worktop", layer="A-EINB", default_height_m=None, default_z_m=[0.86, 0.90]),
        _item("F022", "Spüle", "Spüle", shapely_box(0.4, 0.1, 1.0, 0.5), catalogue_kind="KITCHEN_SINK", layer="A-EINB", default_height_m=None, default_z_m=[0.80, 0.90]),
    ]
    reconstruct_items(items, _cloud((run, "kitchen cabinet")), units_per_metre=1.0, floor_z=0.0, ceiling_z=2.5, settings=SETTINGS)
    base, worktop, sink = items
    assert worktop.z_max == pytest.approx(0.92, abs=0.02)
    assert worktop.z_max - worktop.z_min == pytest.approx(SETTINGS.worktop_thickness_m, abs=1e-6)
    assert base.z_max == pytest.approx(worktop.z_min, abs=1e-6) and base.z_min == pytest.approx(0.0)
    assert sink.z_max == pytest.approx(worktop.z_max, abs=1e-6)
    assert all(item.z_source == "points" for item in items)
    assert sink.solids[0]["part_source"] == "prior"


def test_pipeline_keeps_the_footprint_body_without_evidence() -> None:
    rng = np.random.default_rng(16)
    stray = scan(rng, box_points(rng, 0.0, 0.05, 0.0, 0.05, 0.3, 0.32, density=500))
    items = [_item("F030", "Regal", "shelf", shapely_box(0.0, 0.0, 1.0, 0.4), z_min=0.0, z_max=1.8, z_source="instance_bbox")]
    footprint_before = [list(point) for point in items[0].footprint]
    reconstruct_items(items, _cloud((stray, "shelf")), units_per_metre=1.0, floor_z=0.0, ceiling_z=2.5, settings=SETTINGS)
    assert items[0].shape_source == "footprint"
    assert items[0].solids == [] and items[0].mesh is None
    assert items[0].footprint == footprint_before
    assert items[0].fit["reason"]


def test_projected_outline_of_a_table_is_its_top() -> None:
    points = table_with_legs(np.random.default_rng(17))
    solids = fit_table(points, FOOTPRINT_TABLE, SETTINGS, (0.0, 1.3))
    outline = projected_outline(solids or [], None)
    assert outline is not None
    assert outline.area == pytest.approx(1.2 * 0.8, rel=0.1)


def test_frame_for_footprint_is_aligned_to_the_long_side() -> None:
    footprint = Polygon([(0, 0), (2, 2), (1.5, 2.5), (-0.5, 0.5)])
    frame = frame_for_footprint(footprint, 1.0, 0.0)
    assert abs(abs(frame.angle) - math.pi / 4) < 1e-6
    local = frame.polygon_to_item(footprint)
    x0, y0, x1, y1 = local.bounds
    assert (x1 - x0) > (y1 - y0)


# ----------------------------------------------------------------------------------------------------------------------
# IFC + 3D DXF
# ----------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reconstructed_ifc(tmp_path_factory: pytest.TempPathFactory):
    from samcloud.bim.core.ifc.writer import write_ifc
    from samcloud.bim.core.qa.validate import validate_ifc
    from samcloud.bim.scene_builder import build_scene
    from test_bim_scene_builder import make_plan

    plan: BuildingPlan = make_plan()
    rng = np.random.default_rng(18)
    table = table_with_legs(rng) + np.array([2.0, 2.5, 0.0])
    round_table = scan(rng, cylinder_points(rng, 6.0, 2.5, 0.45, 0.69, 0.72), cylinder_points(rng, 6.0, 2.5, 0.06, 0.0, 0.69, top=False))
    wc = wc_points(rng) + np.array([6.2, 0.7, 0.0])
    floor = float(plan.floor_z)
    plan.furniture = [
        _item("F101", "Tisch", "table", shapely_box(1.4, 2.1, 2.6, 2.9), z_min=floor, z_max=floor + 0.74, z_source="instance_bbox"),
        _item("F102", "Tisch", "table", Polygon([(6.0 + 0.45 * math.cos(a), 2.5 + 0.45 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 32, endpoint=False)])),
        _item("F103", "WC", "WC", shapely_box(5.95, 0.45, 6.45, 1.15), catalogue_kind="WC_WALL", layer="P-SANI", floor_contact=True),
    ]
    shifted = [points + np.array([0.0, 0.0, floor]) for points in (table, round_table, wc)]
    reconstruct_items(plan.furniture, _cloud((shifted[0], "table"), (shifted[1], "table"), (shifted[2], "toilet")), units_per_metre=1.0, floor_z=floor, ceiling_z=plan.ceiling_z, settings=SETTINGS)
    scene = build_scene(plan)
    directory = tmp_path_factory.mktemp("recon_ifc")
    path = directory / "recon.ifc"
    export = write_ifc(scene, path)
    report = validate_ifc(path, scene=scene, export_result=export, reports_dir=directory)
    return plan, path, report


def test_reconstructed_furniture_ifc_validates(reconstructed_ifc) -> None:
    plan, path, report = reconstructed_ifc
    assert [item.shape_source for item in plan.furniture] == ["parametric", "parametric", "mesh"]
    assert not report.express_findings
    assert not report.ids_failures
    assert not report.reimport_mismatches


def test_ifc_bodies_are_part_solids_cylinders_and_a_tessellation(reconstructed_ifc) -> None:
    _plan, path, _report = reconstructed_ifc
    model = ifcopenshell.open(str(path))
    bodies = {}
    for product in model.by_type("IfcFurniture") + model.by_type("IfcSanitaryTerminal"):
        body = next(r for r in product.Representation.Representations if r.RepresentationIdentifier == "Body")
        bodies[product.Name] = body
        identifiers = {r.RepresentationIdentifier for r in product.Representation.Representations}
        assert {"Body", "Box"} <= identifiers
    square = bodies["Tisch 01"]
    assert square.RepresentationType == "SweptSolid" and len(square.Items) == 5  # top + 4 legs
    round_body = bodies["Tisch 02"]
    profiles = [item.SweptArea.is_a() for item in round_body.Items]
    assert "IfcCircleProfileDef" in profiles
    wc = bodies["WC 01"]
    assert wc.RepresentationType == "Tessellation" and wc.Items[0].is_a("IfcTriangulatedFaceSet")
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    table = next(p for p in model.by_type("IfcFurniture") if p.Name == "Tisch 01")
    verts = np.asarray(ifcopenshell.geom.create_shape(settings, table).geometry.verts).reshape(-1, 3)
    floor = float(_plan.floor_z)
    assert verts[:, 2].max() - floor == pytest.approx(0.74, abs=0.03)
    assert verts[:, 2].min() == pytest.approx(floor, abs=1e-3)


def test_3d_dxf_of_the_ifc_audits_clean(reconstructed_ifc, tmp_path) -> None:
    from samcloud.dxf.writer3d import export_ifc_3d_dxf

    _plan, path, _report = reconstructed_ifc
    target = tmp_path / "synthetic_1_3d.dxf"
    result = export_ifc_3d_dxf(path, target)
    document = ezdxf.readfile(target)
    assert not document.audit().has_errors
    meshes = list(document.modelspace().query("MESH"))
    assert len(meshes) == result.meshes > 0
    layers = {mesh.dxf.layer for mesh in meshes}
    assert {"3D-IfcWall", "3D-IfcFurniture", "3D-IfcSanitaryTerminal"} <= layers
    assert "3D-IfcOpeningElement" not in layers
    model = ifcopenshell.open(str(path))
    guids = {mesh.get_xdata("SAMCLOUD")[0][1] for mesh in meshes}
    assert {p.GlobalId for p in model.by_type("IfcFurniture")} <= guids


# ----------------------------------------------------------------------------------------------------------------------
# 2D DXF outlines
# ----------------------------------------------------------------------------------------------------------------------

def test_2d_dxf_redraws_reconstructed_polylines_but_not_inserts_or_kitchen(tmp_path) -> None:
    import json
    from dataclasses import dataclass

    from samcloud.bim.furniture_recon.dxf_outline import apply_reconstructed_outlines

    @dataclass(frozen=True)
    class _Result:
        layers: dict

    table_start = [[0.0, 0.0], [1.2, 0.0], [1.2, 0.8], [0.0, 0.8]]
    table_true = [[0.02, 0.01], [1.18, 0.01], [1.18, 0.79], [0.02, 0.79]]
    kitchen = [[3.0, 0.0], [5.0, 0.0], [5.0, 0.6], [3.0, 0.6]]
    plan = {"furniture": [
        {"label": "Tisch", "semantic_label": "table", "catalogue_kind": None, "shape_source": "parametric", "footprint": table_true, "fit": {"plan_footprint": table_start}},
        {"label": "Unterschrank", "semantic_label": "kitchen cabinet", "catalogue_kind": "kitchen_base", "shape_source": "parametric", "footprint": [[3.0, 0.0], [5.0, 0.0], [5.0, 0.62], [3.0, 0.62]], "fit": {"plan_footprint": kitchen}},
        {"label": "Regal", "semantic_label": "shelf", "catalogue_kind": None, "shape_source": "footprint", "footprint": [[7, 0], [8, 0], [8, 1], [7, 1]], "fit": {}},
    ]}
    (tmp_path / "plan_model.json").write_text(json.dumps(plan), encoding="utf-8")
    insert = {"type": "insert", "name": "WC_WALL", "point": [0.0, 0.0]}
    result = _Result({
        "A-FURN": [{"type": "polyline", "closed": True, "points": [list(p) for p in reversed(table_start)]}, insert],
        "A-EINB": [{"type": "polyline", "closed": True, "points": kitchen}],
        "A-WALL": [{"type": "polyline", "closed": True, "points": table_start}],
    })
    redrawn = apply_reconstructed_outlines(result, tmp_path)
    assert redrawn.layers["A-FURN"][0]["points"] == table_true
    assert redrawn.layers["A-FURN"][1] is insert
    assert redrawn.layers["A-EINB"][0]["points"] == kitchen
    assert redrawn.layers["A-WALL"][0]["points"] == table_start
    assert apply_reconstructed_outlines(result, tmp_path / "missing") is result


@pytest.mark.parametrize(
    ("label", "semantic", "ifc_class", "predefined", "object_type"),
    [
        ("Stehlampe", "floor lamp", "IfcLightFixture", "NOTDEFINED", "Stehlampe"),
        ("Kratzbaum", "cat tree", "IfcFurniture", "USERDEFINED", "CatTree"),
        ("Pflanze", "plant", "IfcBuildingElementProxy", "USERDEFINED", "Plant"),
        ("Sessel", "armchair", "IfcFurniture", "CHAIR", "Chair"),
        ("Esstisch", "dining table", "IfcFurniture", "TABLE", "Table"),
    ],
)
def test_ifc_classes_of_the_sam3_furniture_vocabulary(label: str, semantic: str, ifc_class: str, predefined: str, object_type: str) -> None:
    from samcloud.bim.furniture import load_furniture_mapping

    klass, resolved = load_furniture_mapping().resolve(label=label, semantic_label=semantic, catalogue_kind=None, layer="A-FURN")
    assert (klass.ifc_class, klass.predefined_type) == (ifc_class, predefined)
    assert resolved == object_type


def test_trestle_table_gives_two_panels_not_one_block() -> None:
    rng = np.random.default_rng(19)
    parts = [
        box_points(rng, -0.8, 0.8, -0.45, 0.45, 0.72, 0.75),
        box_points(rng, -0.70, -0.64, -0.35, 0.35, 0.0, 0.72, top=False),  # end panels
        box_points(rng, 0.64, 0.70, -0.35, 0.35, 0.0, 0.72, top=False),
        box_points(rng, -0.64, 0.64, -0.03, 0.03, 0.30, 0.36, top=False),  # stretcher between them
    ]
    points = scan(rng, *parts)
    solids = fit_table(points, shapely_box(-0.8, -0.45, 0.8, 0.45), SETTINGS, (0.0, 1.3))
    supports = [s for s in solids or [] if s.role != "top"]
    assert len(supports) == 2
    assert all(s.role == "panel" and s.sx < 0.15 and s.sy == pytest.approx(0.70, abs=0.08) for s in supports)


def test_table_cut_short_by_the_plan_outline_grows_to_its_own_points_only() -> None:
    rng = np.random.default_rng(20)
    table = table_with_legs(rng)  # 1.2 x 0.8 m top
    sofa = scan(rng, box_points(rng, -0.6, 0.6, 0.55, 1.4, 0.0, 0.45))  # neighbour of another family beside it
    items = [_item("F040", "Tisch", "table", shapely_box(-0.6, -0.4, 0.25, 0.4), z_min=0.0, z_max=0.74, z_source="instance_bbox")]
    names = list(FINE_NAMES) + ["sofa"]
    xyz = np.vstack((table, sofa))
    fine = np.concatenate((np.full(len(table), 1, np.int32), np.full(len(sofa), len(names), np.int32)))
    cloud = SceneCloud(xyz, fine, np.zeros(len(xyz), np.int64), names)
    reconstruct_items(items, cloud, units_per_metre=1.0, floor_z=0.0, ceiling_z=2.5, settings=SETTINGS)
    table_item = items[0]
    assert table_item.shape_source == "parametric"
    assert table_item.fit["grown_points"] > 0
    top = next(solid for solid in table_item.solids if solid["role"] == "top")
    assert sorted(top["dims_m"][:2]) == pytest.approx([0.8, 1.2], abs=0.06)
    assert Polygon(table_item.footprint).bounds[3] <= 0.45  # the sofa beside it is not swallowed


def test_kitchen_counter_item_stops_at_the_worktop_not_the_upper_cabinets() -> None:
    rng = np.random.default_rng(21)
    counter = scan(rng, box_points(rng, 0.0, 2.0, 0.0, 0.6, 0.0, 0.91))
    uppers = scan(rng, box_points(rng, 0.0, 2.0, 0.0, 0.35, 1.45, 2.15))
    items = [_item("F050", "fine_kitchen", "fine_kitchen", shapely_box(0.0, 0.0, 2.0, 0.6), catalogue_kind="fine_kitchen", layer="A-EINB", default_height_m=0.9)]
    reconstruct_items(items, _cloud((counter, "kitchen cabinet"), (uppers, "kitchen cabinet")), units_per_metre=1.0, floor_z=0.0, ceiling_z=2.5, settings=SETTINGS)
    assert items[0].z_max == pytest.approx(0.91, abs=0.03)
    assert items[0].z_min == pytest.approx(0.0)


def test_growth_never_absorbs_a_neighbour_of_the_same_label_with_its_own_instance() -> None:
    rng = np.random.default_rng(22)
    stand_a = scan(rng, box_points(rng, 0.0, 0.5, 0.0, 0.4, 0.0, 0.5))
    stand_b = scan(rng, box_points(rng, 0.52, 1.6, 0.0, 0.4, 0.0, 0.9))  # touching neighbour, same fine term, other instance
    names = ["tv stand"]
    xyz = np.vstack((stand_a, stand_b))
    cloud = SceneCloud(xyz, np.ones(len(xyz), np.int32), np.concatenate((np.full(len(stand_a), 37), np.full(len(stand_b), 29))).astype(np.int64), names)
    items = [
        _item("F060", "Sideboard", "tv stand", shapely_box(0.0, 0.0, 0.5, 0.4), instance_id=37, z_min=0.0, z_max=0.5, z_source="instance_bbox"),
        _item("F061", "Sideboard", "tv stand", shapely_box(1.0, 0.0, 1.6, 0.4), instance_id=29, z_min=0.0, z_max=0.9, z_source="instance_bbox"),
    ]
    reconstruct_items(items, cloud, units_per_metre=1.0, floor_z=0.0, ceiling_z=2.5, settings=SETTINGS)
    small = items[0]
    assert "grown_points" not in small.fit
    assert Polygon(small.footprint).bounds[2] <= 0.56
    assert small.z_max <= 0.55


def test_tv_on_a_tv_stand_is_not_part_of_the_stand() -> None:
    rng = np.random.default_rng(23)
    stand = box_points(rng, -0.8, 0.8, -0.2, 0.2, 0.0, 0.50)
    tv = box_points(rng, -0.55, 0.55, -0.04, 0.04, 0.52, 1.20)  # thin screen standing on it
    points = scan(rng, stand, tv)
    from samcloud.bim.furniture_recon.parametric import fit_box

    solids = fit_box(points, shapely_box(-0.8, -0.2, 0.8, 0.2), SETTINGS, hanging=False, low_piece=True)
    assert solids is not None and solids[0].z1 == pytest.approx(0.50, abs=0.03)
    tall = fit_box(points, shapely_box(-0.8, -0.2, 0.8, 0.2), SETTINGS, hanging=False, low_piece=False)
    assert tall[0].z1 > 1.1


def test_lamps_and_sanitary_fixtures_have_viewer_categories() -> None:
    from samcloud.bim.ui_data import category_of
    from samcloud.bim.viewer_export import category_for_ifc_class

    assert category_of("IfcLightFixture") == "appliances"
    assert category_of("IfcSanitaryTerminal") == "sanitary"
    for ifc_class in ("IfcLightFixture", "IfcSanitaryTerminal", "IfcElectricAppliance"):
        assert category_for_ifc_class(ifc_class) == "furniture"


def test_a_chair_the_plan_lost_is_added_from_its_scan_instance() -> None:
    from samcloud.bim.furniture_recon.discover import canonical_semantic_labels, discover_missing_items

    rng = np.random.default_rng(24)
    table = table_with_legs(rng)
    chair = chair_points(rng) + np.array([0.0, 0.75, 0.0])
    fragment = scan(rng, box_points(rng, 2.0, 2.3, 2.0, 2.3, 0.60, 0.85))  # a "chair" seen only above a table edge: not standing
    xyz = np.vstack((table, chair, fragment))
    instances = np.concatenate((np.full(len(table), 5), np.full(len(chair), 6), np.full(len(fragment), 7))).astype(np.int64)
    cloud = SceneCloud(xyz, np.zeros(len(xyz), np.int32), instances, list(FINE_NAMES))
    items = [_item("F001", "Tisch", "table", FOOTPRINT_TABLE, instance_id=5, z_min=0.0, z_max=0.74, z_source="instance_bbox"),
             _item("F002", "WC", "WC", shapely_box(5, 5, 5.4, 5.6), catalogue_kind="WC_WALL", layer="P-SANI")]
    added = discover_missing_items(items, cloud, {5: "table", 6: "chair", 7: "chair"}, units_per_metre=1.0, floor_z=0.0, item_factory=PlanFurniture)
    assert [item.instance_id for item in added] == [6]
    chair_item = added[0]
    assert chair_item.semantic_label == "chair" and chair_item.layer == "A-FURN" and chair_item.id == "F003"
    assert Polygon(chair_item.footprint).area == pytest.approx(0.44 * 0.44, rel=0.25)
    assert canonical_semantic_labels(items) == 1 and items[1].semantic_label == "toilet"


def test_a_fridge_cut_by_the_kitchen_row_takes_back_its_rest() -> None:
    from samcloud.bim.furniture_recon.discover import merge_split_appliances

    rng = np.random.default_rng(25)
    fridge = scan(rng, box_points(rng, 0.0, 0.6, 0.0, 0.7, 0.0, 1.95))
    cabinet = scan(rng, box_points(rng, 0.6, 1.2, 0.0, 0.7, 0.0, 2.1))
    xyz = np.vstack((fridge, cabinet))
    instances = np.concatenate((np.full(len(fridge), 7), np.zeros(len(cabinet)))).astype(np.int64)
    cloud = SceneCloud(xyz, np.zeros(len(xyz), np.int32), instances, list(FINE_NAMES))
    items = [
        _item("F003", "Kühlschrank", "refrigerator", shapely_box(0.0, 0.35, 0.6, 0.7), instance_id=7, z_min=0.0, z_max=1.95, z_source="points"),
        _item("F011", "Hochschrank", "tall unit", shapely_box(0.0, 0.0, 0.6, 0.35), catalogue_kind="tall_unit", layer="A-EINB", z_min=0.0, z_max=1.97, z_source="points"),
        _item("F012", "Hochschrank", "tall unit", shapely_box(0.6, 0.0, 1.2, 0.7), catalogue_kind="tall_unit", layer="A-EINB", z_min=0.0, z_max=2.1, z_source="points"),
    ]
    merged = merge_split_appliances(items, cloud, units_per_metre=1.0)
    assert [entry["removed"] for entry in merged] == ["F011"]
    assert [item.id for item in items] == ["F003", "F012"]
    assert Polygon(items[0].footprint).area == pytest.approx(0.42, rel=0.05)
