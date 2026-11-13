"""Outdoor surveyor / DGM site-plan standard: trunks, crowns, hedges, 3D edges, roofs, spot heights, contour labels.

All tests build real geometry, write a DXF and read it back with ezdxf.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import uuid

import ezdxf
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from samcloud.dxf.las_source import Frame
from samcloud.dxf.layers import load_layer_config
from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus
from samcloud.dxf.writer import export_plan
from samcloud.ortho.plan import _add_ortho_plan_layer_presentation, _layer_presentation
from samcloud.ortho.vector.buildings import RoofSettings, roof_height_entities, roof_heights
from samcloud.ortho.vector.contours import ContourLabelSettings, ContourSettings, contour_entities, label_major_contours
from samcloud.ortho.vector.raster_grid import RasterGrid, mask_to_polygons
from samcloud.ortho.vector.regularize import regularize_polygons
from samcloud.ortho.vector.spots import SpotSettings, format_spot_height, ground_grid_points, points_along, select_spots, spot_entities
from samcloud.ortho.vector.terrain import DtmSampler, drape_coordinates, drape_outline_heights
from samcloud.ortho.vector.trees import (
    TrunkCollector,
    TrunkSettings,
    detect_crowns,
    fit_trunk_diameter,
    tree_annotation_entities,
    tree_entities,
)
from samcloud.ortho.vector.vegetation import HedgeSettings, hedge_entities, max_turning_angle_deg

ORIGIN_Z = 440.0  # frame origin height: absolute = local + 440


@pytest.fixture
def work_dir(request: pytest.FixtureRequest) -> Path:
    root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    if root:
        path = Path(root) / f"outdoor_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return request.getfixturevalue("tmp_path_factory").mktemp("outdoor")


def _result(layers: dict[str, list[dict]], *, styles: dict | None = None, colors: dict | None = None, linetypes: dict | None = None, origin_z: float = 0.0) -> DxfPlanResult:
    frame = Frame(np.asarray([0.0, 0.0, origin_z], dtype=np.float64), 1.0, np.eye(3, dtype=np.float64))
    return DxfPlanResult(
        project_name="Out", level_name="site", mode="outdoor", profile_name="test", scale_factor=1.0, frame=frame,
        settings_fingerprint="x", layers=layers, statuses={name: LayerStatus(name, "ok", None, len(items)) for name, items in layers.items()},
        layer_colors=colors if colors is not None else {name: 7 for name in layers},
        background=BackgroundImage(None, None, (0.0, 0.0, 200.0, 200.0)),
        layer_linetypes=linetypes or {}, layer_styles=styles or {}, blocks=sorted({str(e["name"]) for items in layers.values() for e in items if e.get("type") == "insert"}), variant="ortho",
    )


def _texts(document: ezdxf.document.Drawing, layer: str | None = None) -> list[str]:
    return [entity.dxf.text for entity in document.modelspace() if entity.dxftype() == "TEXT" and (layer is None or entity.dxf.layer == layer)]


# 1. trunk circle fit --------------------------------------------------------------------------------------------


def _ring(radius: float, centre: tuple[float, float], count: int, sigma: float, rng: np.random.Generator, start: float = 0.0, span: float = 2.0 * math.pi) -> np.ndarray:
    angles = start + rng.uniform(0.0, span, count)
    return np.column_stack((centre[0] + (radius + rng.normal(0.0, sigma, count)) * np.cos(angles), centre[1] + (radius + rng.normal(0.0, sigma, count)) * np.sin(angles)))


def test_trunk_circle_fit_recovers_diameter_and_rejects_partial_arcs() -> None:
    rng = np.random.default_rng(3)
    centre = (12.0, 8.0)
    clutter = np.column_stack((rng.uniform(centre[0] - 0.9, centre[0] + 0.9, 40), rng.uniform(centre[1] - 0.9, centre[1] + 0.9, 40)))
    noisy = np.vstack((_ring(0.175, centre, 300, 0.004, rng), clutter))
    diameter = fit_trunk_diameter(noisy, centre)
    assert diameter is not None and diameter == pytest.approx(0.35, abs=0.02)
    for true_diameter in (0.12, 0.62, 1.1):
        fitted = fit_trunk_diameter(_ring(true_diameter / 2.0, centre, 400, 0.004, rng), centre)
        assert fitted == pytest.approx(true_diameter, abs=0.02)
    # A one-sided scan of the stem (90 degrees of arc) is not a measured trunk.
    assert fit_trunk_diameter(_ring(0.175, centre, 300, 0.004, rng, start=0.3, span=math.radians(90.0)), centre) is None
    # No / too few points, scattered points without a circle, an implausibly large circle: no diameter.
    assert fit_trunk_diameter(np.empty((0, 2)), centre) is None
    assert fit_trunk_diameter(noisy[:10], centre) is None
    assert fit_trunk_diameter(clutter, centre) is None
    assert fit_trunk_diameter(_ring(1.9, centre, 500, 0.004, rng), centre) is None
    # A cylinder that is not at the trunk position of the tree belongs to something else.
    assert fit_trunk_diameter(_ring(0.175, (centre[0] + 3.0, centre[1]), 300, 0.004, rng), centre) is None


def test_trunk_label_only_where_trunk_points_were_observed(work_dir: Path) -> None:
    rng = np.random.default_rng(5)
    # Two crowns of 5 m diameter / 7 m top: tree 1 has a scanned stem, tree 2 was seen from above only.
    from samcloud.ortho.vector.trees import Crown

    crowns = [Crown((20.0, 20.0), (20.0, 20.0), 5.0, 7.0, 19.6), Crown((40.0, 20.0), (40.0, 20.0), 4.0, 6.0, 12.6)]
    collector = TrunkCollector(np.asarray([c.trunk for c in crowns]), np.asarray([100.0, 100.0]), TrunkSettings())
    stem = _ring(0.16, (20.05, 19.95), 500, 0.004, rng)
    heights = rng.uniform(100.5, 101.7, len(stem))
    canopy = np.column_stack((rng.uniform(38.0, 42.0, 400), rng.uniform(18.0, 22.0, 400), rng.uniform(103.0, 106.0, 400)))  # tree 2: crown points only
    cloud = np.vstack((np.column_stack((stem, heights)), canopy))
    classes = np.full(len(cloud), 5, dtype=np.uint8)
    collector.add(cloud, classes, (4, 5, 95))
    # Ground points of the same class family below the stem band are not part of the cylinder.
    ground = np.column_stack((_ring(0.16, (20.05, 19.95), 100, 0.004, rng), np.full(100, 100.1)))
    collector.add(ground, np.full(100, 2, dtype=np.uint8), (4, 5, 95))
    diameters = collector.diameters()
    assert diameters[0] == pytest.approx(0.32, abs=0.02) and diameters[1] is None
    entities = tree_annotation_entities(crowns, diameters)
    path = export_plan(_result({"S-ANNO": entities}), work_dir, coordinates="local")
    texts = _texts(ezdxf.readfile(path), "S-ANNO")
    trunk_texts = [text for text in texts if text.startswith("St.")]
    assert len(trunk_texts) == 1 and re.fullmatch(r"St\. Ø 0\.3[0-4] m", trunk_texts[0])
    assert sum(text.startswith("DK") for text in texts) == 2  # both crowns labelled
    # Without any trunk measurement nothing about a trunk is written (no invented numbers).
    bare = export_plan(_result({"S-ANNO": tree_annotation_entities(crowns, None)}), work_dir / "bare", coordinates="local")
    assert not [text for text in _texts(ezdxf.readfile(bare)) if "St." in text or "Ø" in text]


# 2. crown labels -------------------------------------------------------------------------------------------------


def _cone_chm(grid: RasterGrid, centre: tuple[float, float], radius: float, peak: float) -> tuple[np.ndarray, np.ndarray]:
    cols, rows = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    xs, ys = grid.min_x + (cols + 0.5) * grid.pixel_size, grid.max_y - (rows + 0.5) * grid.pixel_size
    distance = np.hypot(xs - centre[0], ys - centre[1])
    mask = distance <= radius
    return np.where(mask, 2.5 + (peak - 2.5) * (1.0 - distance / radius), 0.0).astype(np.float32), mask


def test_crown_diameter_and_height_labels_match_the_chm(work_dir: Path) -> None:
    grid = RasterGrid(0.0, 40.0, 0.1, 400, 400)
    chm_a, mask_a = _cone_chm(grid, (10.0, 28.0), 2.4, 6.5)  # 4.8 m crown, 6.5 m top
    chm_b, mask_b = _cone_chm(grid, (28.0, 12.0), 3.5, 9.0)
    chm = np.maximum(chm_a, chm_b)
    crowns = detect_crowns(chm, mask_a | mask_b, grid, min_height=2.0)
    assert len(crowns) == 2
    crowns.sort(key=lambda crown: crown.trunk[0])
    assert crowns[0].diameter == pytest.approx(4.8, abs=0.3) and crowns[0].height == pytest.approx(6.5, abs=0.15)
    assert crowns[1].diameter == pytest.approx(7.0, abs=0.4) and crowns[1].height == pytest.approx(9.0, abs=0.15)
    layers = {"S-TREE": tree_entities(crowns), "S-ANNO": tree_annotation_entities(crowns, [None, None])}
    document = ezdxf.readfile(export_plan(_result(layers), work_dir, coordinates="local"))
    labels = [text for text in _texts(document, "S-ANNO") if text.startswith("DK")]
    assert len(labels) == 2
    parsed = sorted((float(m.group(1)), float(m.group(2))) for m in (re.fullmatch(r"DK (\d+\.\d) m  H (\d+\.\d) m", text) for text in labels))
    assert parsed[0] == (pytest.approx(crowns[0].diameter, abs=0.05), pytest.approx(6.5, abs=0.15))
    assert parsed[1] == (pytest.approx(crowns[1].diameter, abs=0.05), pytest.approx(9.0, abs=0.15))
    # The label sits next to its TREE block, outside the crown circle.
    block = next(entity for entity in document.modelspace() if entity.dxftype() == "INSERT" and entity.dxf.name == "TREE")
    first = next(entity for entity in document.modelspace() if entity.dxftype() == "TEXT" and entity.dxf.text.startswith("DK"))
    assert first.dxf.insert.x > block.dxf.insert.x


# 3. hedges ---------------------------------------------------------------------------------------------------------


def test_hedge_outline_is_closed_smooth_and_labelled(work_dir: Path) -> None:
    grid = RasterGrid(0.0, 20.0, 0.05, 400, 400)
    cols, rows = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    xs, ys = grid.min_x + (cols + 0.5) * grid.pixel_size, grid.max_y - (rows + 0.5) * grid.pixel_size
    # A 6.0 x 2.4 m hedge rotated by 20 degrees: its raster boundary is a staircase.
    angle = math.radians(20.0)
    u = (xs - 10.0) * math.cos(angle) + (ys - 10.0) * math.sin(angle)
    v = -(xs - 10.0) * math.sin(angle) + (ys - 10.0) * math.cos(angle)
    mask = ((u / 3.0) ** 2 + (v / 1.2) ** 2) <= 1.0
    rng = np.random.default_rng(9)
    mask &= rng.random(mask.shape) > 0.02  # scan speckle inside the patch
    chm = np.where(mask, 1.2, 0.0).astype(np.float32)
    raw = max(mask_to_polygons(mask, grid), key=lambda polygon: polygon.area)
    entities = hedge_entities(mask, chm, grid, HedgeSettings())
    outlines = [entity for entity in entities if entity.get("type") == "polyline"]
    assert len(outlines) == 1 and outlines[0]["closed"] is True
    points = outlines[0]["points"]
    assert len(points) <= 120 and len(points) < len(raw.exterior.coords) / 3  # no raster staircase
    assert max_turning_angle_deg(points) <= 40.0  # no 90 degree stair steps
    assert Polygon(points).is_valid
    assert Polygon(points).area == pytest.approx(math.pi * 3.0 * 1.2, rel=0.06)
    texts = [entity["text"] for entity in entities if entity.get("type") == "text"]
    assert texts == ["Gebüsch h=1.2 m"]
    document = ezdxf.readfile(export_plan(_result({"S-VEG": outlines, "S-ANNO": [entity for entity in entities if entity.get("type") == "text"]}), work_dir, coordinates="local"))
    closed = [entity for entity in document.modelspace() if entity.dxftype() == "LWPOLYLINE" and entity.dxf.layer == "S-VEG"]
    assert len(closed) == 1 and closed[0].closed
    assert "Gebüsch h=1.2 m" in _texts(document, "S-ANNO")
    # Too low (lawn) or too high (tree) vegetation is not a hedge; small patches are dropped.
    assert hedge_entities(mask, np.where(mask, 0.2, 0.0).astype(np.float32), grid) == []
    assert hedge_entities(mask, np.where(mask, 4.0, 0.0).astype(np.float32), grid) == []
    small = (xs - 5.0) ** 2 + (ys - 5.0) ** 2 <= 0.5**2
    assert hedge_entities(small, np.where(small, 1.0, 0.0).astype(np.float32), grid) == []


# 4. curbs and pavement edges as continuous 3D polylines ------------------------------------------------------------


def _sloped_terrain() -> tuple[DtmSampler, RasterGrid]:
    grid = RasterGrid(0.0, 60.0, 0.2, 300, 300)
    cols, rows = np.meshgrid(np.arange(300), np.arange(300))
    xs, ys = (cols + 0.5) * 0.2, 60.0 - (rows + 0.5) * 0.2
    return DtmSampler(10.0 + 0.05 * xs + 0.02 * ys, grid), grid


def test_curb_and_pavement_edges_are_continuous_3d_polylines_with_dtm_heights(work_dir: Path) -> None:
    sampler, _grid = _sloped_terrain()

    def terrain(x: float, y: float) -> float:
        return 10.0 + 0.05 * x + 0.02 * y

    curb_line = LineString([(5.0, 10.0), (20.0, 12.0), (40.0, 30.0), (52.0, 31.0)])  # one continuous edge
    curb = {"type": "polyline3d", "points": drape_coordinates(list(curb_line.coords), sampler), "closed": False}
    assert len(curb["points"]) > 4  # long segments are subdivided so the polyline follows the terrain
    assert all(point[2] == pytest.approx(terrain(point[0], point[1]), abs=1e-6) for point in curb["points"])
    pave = {"type": "polyline", "points": [[10.0, 40.0], [30.0, 40.0], [30.0, 55.0], [10.0, 55.0]], "closed": True, "holes": [[[15.0, 45.0], [20.0, 45.0], [20.0, 50.0], [15.0, 50.0]]]}
    drape_outline_heights(pave, sampler)
    layers = {"S-CURB": [curb], "S-PAVE": [pave]}
    for coordinates, offset in (("local", 0.0), ("original", ORIGIN_Z)):
        path = export_plan(_result(layers, origin_z=ORIGIN_Z), work_dir / coordinates, coordinates=coordinates)
        document = ezdxf.readfile(path)
        curbs = [entity for entity in document.modelspace() if entity.dxf.layer == "S-CURB"]
        edges = [entity for entity in document.modelspace() if entity.dxf.layer == "S-PAVE"]
        assert len(curbs) == 1 and curbs[0].dxftype() == "POLYLINE" and curbs[0].is_3d_polyline  # one polyline per continuous edge
        assert len(edges) == 2 and all(edge.dxftype() == "POLYLINE" and edge.is_3d_polyline and edge.is_closed for edge in edges)  # exterior + hole ring
        # z comes from the DTM (plus the frame height in original coordinates).
        for entity in (*curbs, *edges):
            for vertex in entity.vertices:
                location = vertex.dxf.location
                x, y = location.x, location.y
                assert location.z == pytest.approx(terrain(x, y) + offset, abs=1e-6)
    # Without terrain data the outline stays 2D: heights are never invented.
    empty = DtmSampler(np.full((300, 300), np.nan), _grid)
    flat = {"type": "polyline", "points": [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]], "closed": True, "holes": []}
    drape_outline_heights(flat, empty)
    assert "z" not in flat


# 5. buildings: rectangular outline plus ridge / eave heights ------------------------------------------------------------


def test_building_outline_is_rectangular_and_roof_text_uses_dsm_heights(work_dir: Path) -> None:
    grid = RasterGrid(0.0, 30.0, 0.1, 300, 300)
    cols, rows = np.meshgrid(np.arange(300), np.arange(300))
    xs, ys = (cols + 0.5) * 0.1, 30.0 - (rows + 0.5) * 0.1
    footprint = box(8.0, 10.0, 20.0, 18.0)  # 12 x 8 m
    inside = (xs >= 8.0) & (xs <= 20.0) & (ys >= 10.0) & (ys <= 18.0)
    # Gable roof along x: eave 5.0 m (local) at y = 10 / 18, ridge 8.0 m at y = 14.
    dsm = np.where(inside, 5.0 + 3.0 * (1.0 - np.abs(ys - 14.0) / 4.0), 1.0).astype(np.float32)
    # Raster-like, slightly ragged outline regularised to a rectangle.
    ragged = Polygon([(8.0, 10.0), (14.0, 10.04), (20.0, 10.0), (20.03, 14.0), (20.0, 18.0), (14.0, 17.96), (8.0, 18.0), (7.97, 14.0)])
    outline = regularize_polygons([ragged], angle_tolerance_deg=15.0)[0]
    assert len(outline.exterior.coords) - 1 == 4
    corners = np.asarray(outline.exterior.coords)[:-1]
    for index in range(4):
        first, second = corners[index - 1] - corners[index], corners[(index + 1) % 4] - corners[index]
        assert abs(math.degrees(math.acos(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))) - 90.0) < 0.5
    ridge, eave = roof_heights(footprint, dsm, grid, RoofSettings())
    assert ridge == pytest.approx(8.0, abs=0.1) and eave == pytest.approx(5.4, abs=0.45)  # eave ring runs 0.3-1.2 m inside the edge
    texts = roof_height_entities(footprint, ridge + ORIGIN_Z, eave + ORIGIN_Z)
    assert len(texts) == 1 and re.fullmatch(r"First \+\d+\.\d{2}  Traufe \+\d+\.\d{2}", texts[0]["text"])
    layers = {"S-BLDG": [{"type": "polyline", "points": [list(p) for p in corners], "closed": True, "holes": []}], "S-ANNO": texts}
    document = ezdxf.readfile(export_plan(_result(layers), work_dir, coordinates="local"))
    outline_entity = next(entity for entity in document.modelspace() if entity.dxftype() == "LWPOLYLINE")
    assert outline_entity.closed and len(list(outline_entity.get_points("xy"))) == 4
    text = _texts(document, "S-ANNO")[0]
    first_value, second_value = (float(value) for value in re.findall(r"\+(\d+\.\d{2})", text))
    assert first_value == pytest.approx(ridge + ORIGIN_Z, abs=0.005) and second_value == pytest.approx(eave + ORIGIN_Z, abs=0.005)
    # A flat roof gets a single height.
    flat = roof_height_entities(footprint, 447.2, 447.0)
    assert flat[0]["text"] == "Dach +447.20"


# 6. DGM spot elevations ------------------------------------------------------------------------------------------------


def test_spot_elevations_format_grid_spacing_and_priority_points(work_dir: Path) -> None:
    sampler, grid = _sloped_terrain()
    settings = SpotSettings()
    bounds = (0.0, 0.0, 60.0, 60.0)
    lattice = ground_grid_points(bounds, 17.5, lambda x, y: True)
    assert lattice and len(lattice) >= 4
    columns = sorted({round(x, 6) for x, _y in lattice})
    rows = sorted({round(y, 6) for _x, y in lattice})
    steps = np.diff(columns).tolist() + np.diff(rows).tolist()
    assert steps and all(15.0 <= step <= 20.0 for step in steps)  # regular ground grid with 15-20 m spacing
    blocked = ground_grid_points(bounds, 17.5, lambda x, y: not (x < 20.0))
    assert all(x >= 20.0 for x, _y in blocked)  # grid only where the ground is free (not under buildings / trees)
    manhole = (30.5, 30.5)
    building = box(30.0, 20.0, 36.0, 28.0)
    corners = list(building.exterior.coords)[:-1]
    curb_points = points_along([(5.0, 5.0), (55.0, 5.0)], settings.curb_spacing_m)
    groups = [("manhole", [manhole]), ("building_corner", corners), ("curb", curb_points), ("grid", lattice)]
    spots = select_spots(groups, settings)
    kinds = {spot.kind for spot in spots}
    assert {"manhole", "building_corner", "curb", "grid"} <= kinds
    assert any(math.dist((spot.x, spot.y), manhole) < 1e-9 and spot.kind == "manhole" for spot in spots)
    assert sum(spot.kind == "building_corner" for spot in spots) >= 3
    positions = np.asarray([(spot.x, spot.y) for spot in spots])
    distances = np.hypot(*(positions[:, None, :] - positions[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= settings.min_spacing_m - 1e-9  # minimum spacing respected
    local = [float(value) for value in sampler.sample(positions[:, 0], positions[:, 1])]
    absolute = [value + ORIGIN_Z for value in local]
    entities = spot_entities(spots, local, absolute, settings)
    document = ezdxf.readfile(export_plan(_result({"S-ELEV": entities}, origin_z=ORIGIN_Z), work_dir, coordinates="original"))
    spot_texts = _texts(document, "S-ELEV")
    assert len(spot_texts) == len(spots)
    assert all(re.fullmatch(r"\+\d{3}\.\d{2}", text) for text in spot_texts)  # e.g. +452.35
    manhole_text = next(entity for entity in document.modelspace() if entity.dxftype() == "TEXT" and entity.dxf.layer == "S-ELEV" and abs(entity.dxf.insert.y - manhole[1]) < 1e-6)
    assert float(manhole_text.dxf.text) == pytest.approx(10.0 + 0.05 * manhole[0] + 0.02 * manhole[1] + ORIGIN_Z, abs=0.005)
    crosses = [entity for entity in document.modelspace() if entity.dxftype() == "LINE" and entity.dxf.layer == "S-ELEV"]
    assert len(crosses) == 2 * len(spots)
    assert format_spot_height(452.349) == "+452.35" and format_spot_height(-0.35) == "-0.35"


# 7. major contour labels ---------------------------------------------------------------------------------------------------


def test_major_contours_are_labelled_with_their_absolute_elevation(work_dir: Path) -> None:
    grid = RasterGrid(0.0, 100.0, 0.2, 500, 500)
    cols, rows = np.meshgrid(np.arange(500), np.arange(500))
    xs, ys = (cols + 0.5) * 0.2, 100.0 - (rows + 0.5) * 0.2
    dtm = (10.15 + 0.06 * xs + 0.01 * ys).astype(np.float64)  # local 10.15 .. 17.15 -> absolute 450.15 .. 457.15
    settings = ContourSettings(min_length=2.0)
    entities = contour_entities(dtm, grid, settings, z_offset=ORIGIN_Z)
    major = [entity for entity in entities if entity["layer_role"] == "major"]
    assert sorted({entity["elevation"] for entity in major}) == [451.0, 452.0, 453.0, 454.0, 455.0, 456.0, 457.0]  # whole absolute metres
    assert all(point[2] == pytest.approx(entity["elevation"] - ORIGIN_Z) for entity in major for point in entity["points"])  # vertices keep local z
    label_settings = ContourLabelSettings(text_height_m=1.0, interval_m=60.0, min_length_m=20.0)
    labelled = label_major_contours(major, label_settings)
    texts = [entity for entity in labelled if entity.get("type") == "text"]
    lines = [entity for entity in labelled if entity.get("type") == "polyline3d"]
    lengths = {level: max(LineString([(p[0], p[1]) for p in entity["points"]]).length for entity in major if entity["elevation"] == level) for level in {entity["elevation"] for entity in major}}
    expected_labels = {str(int(level)) for level, length in lengths.items() if length >= label_settings.min_length_m}
    assert {"451", "452", "453", "454", "455", "456"} <= expected_labels
    assert {text["text"] for text in texts} == expected_labels  # every long major contour carries a label, short ones none
    for text in texts:
        assert -90.0 < text["rotation"] <= 90.0  # readable
        # The label belongs to the contour it sits on: its elevation matches the polyline pieces next to it.
        same = [LineString([(p[0], p[1]) for p in line["points"]]) for line in lines if line["elevation"] == text["elevation"]]
        nearest = min(piece.distance(__import__("shapely").geometry.Point(*text["point"][:2])) for piece in same)
        assert 0.2 <= nearest <= 2.0  # centred in the gap opened in the line
        assert float(text["text"]) == pytest.approx(text["elevation"])
    # The gap removes line length behind each label.
    original_length = sum(LineString([(p[0], p[1]) for p in entity["points"]]).length for entity in major)
    remaining = sum(LineString([(p[0], p[1]) for p in entity["points"]]).length for entity in lines)
    assert original_length - remaining >= len(texts) * 1.0
    # Short contours stay unlabelled.
    assert not [entity for entity in label_major_contours(major, ContourLabelSettings(min_length_m=10_000.0)) if entity.get("type") == "text"]
    path = export_plan(_result({"S-TOPO-MAJR": labelled}, origin_z=ORIGIN_Z), work_dir, coordinates="original")
    document = ezdxf.readfile(path)
    assert sorted(set(_texts(document, "S-TOPO-MAJR"))) == sorted(expected_labels)
    polylines = [entity for entity in document.modelspace() if entity.dxftype() == "POLYLINE" and entity.dxf.layer == "S-TOPO-MAJR"]
    assert polylines and all(float(round(vertex.dxf.location.z)) == pytest.approx(vertex.dxf.location.z, abs=1e-6) for entity in polylines for vertex in entity.vertices)  # absolute whole metres
    # Local 1 m contours are unchanged when no frame offset is given (previous behaviour).
    plain = contour_entities(dtm, grid, settings)
    assert {entity["elevation"] for entity in plain if entity["layer_role"] == "major"} == {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0}


# 8. layer lineweights --------------------------------------------------------------------------------------------------------


def test_outdoor_layer_lineweights_in_written_dxf(work_dir: Path) -> None:
    layers_config = load_layer_config().for_project_mode("outdoor")
    styles, colors, linetypes, hatches = _layer_presentation(layers_config)
    _add_ortho_plan_layer_presentation(styles, colors, linetypes, hatches)
    expected = {"S-BLDG": 50, "S-CURB": 35, "S-TREE": 25, "S-VEG": 25, "S-TOPO-MAJR": 25, "S-TOPO-MINR": 13, "S-ELEV": 18, "S-ANNO": 18}  # 0.50 / 0.35 / 0.25 / 0.25 / 0.25 / 0.13 / 0.18 / 0.18 mm
    layers = {name: [{"type": "line", "start": [0.0, float(index)], "end": [1.0, float(index)]}] for index, name in enumerate(expected)}
    document = ezdxf.readfile(export_plan(_result(layers, styles=styles, colors=colors, linetypes=linetypes), work_dir, coordinates="local"))
    assert {name: document.layers.get(name).dxf.lineweight for name in expected} == expected
    assert document.layers.get("S-TOPO-MAJR").rgb == (150, 60, 40)  # brown-red major contours kept


# plan stage on the synthetic outdoor scene ---------------------------------------------------------------------------------


def test_outdoor_plan_stage_writes_surveyor_elements(work_dir: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import test_ortho_plan as scene_tests
    from samcloud.ortho.plan import generate_clean_plan

    scene = scene_tests._build_synthetic_scene(work_dir / "site")
    result = generate_clean_plan(scene["project_dir"], level=None, mode="outdoor")
    stats = result.fingerprint_inputs["outdoor"]
    assert stats["trees"] == 2 and stats["trees_with_trunk"] == 0  # nadir-like cloud: no trunk points, so no trunk labels
    assert not [entity for entity in result.layers["S-ANNO"] if "St." in str(entity.get("text", ""))]
    assert [entity for entity in result.layers["S-ANNO"] if str(entity.get("text", "")).startswith("DK ")]
    assert stats["curb_polylines"] == len([entity for entity in result.layers["S-CURB"] if entity["type"] == "polyline3d"]) >= 1
    assert all(point[2] == point[2] for entity in result.layers["S-CURB"] for point in entity["points"])
    paved = [entity for entity in result.layers["S-PAVE"] + result.layers["S-WALK"] if entity.get("type") == "polyline"]
    assert paved and all(len(entity["z"]) == len(entity["points"]) for entity in paved)
    assert stats["spot_elevations"] >= 3 and stats["spots_by_kind"]["manhole"] >= 1 and stats["spots_by_kind"]["building_corner"] >= 1
    assert all(re.fullmatch(r"[+-]\d+\.\d{2}", entity["text"]) for entity in result.layers["S-ELEV"] if entity.get("type") == "text")
    assert stats["roof_texts"] == 1 and any(str(entity.get("text", "")).startswith(("Dach", "First")) for entity in result.layers["S-ANNO"])
    document = ezdxf.readfile(export_plan(result, work_dir / "dxf", coordinates="local"))
    assert document.layers.get("S-ELEV").dxf.lineweight == 18 and document.layers.get("S-CURB").dxf.lineweight == 35
    assert [entity for entity in document.modelspace() if entity.dxftype() == "POLYLINE" and entity.dxf.layer == "S-CURB"]
