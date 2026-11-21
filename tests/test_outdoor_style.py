"""Outdoor category style: one source (config/outdoor_style.yaml) for IFC, DXF and the app viewer."""

from __future__ import annotations

from pathlib import Path

import ezdxf
import ifcopenshell
import numpy as np

from samcloud.bim import ui_data
from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.ifc.outdoor import outdoor_category_of_product
from samcloud.bim.core.model import (
    BuildingElement, BuildingParams, CurbElement, CurbParams, FenceElement, FenceParams, ManholeElement,
    ManholeParams, MarkingElement, MarkingParams, OpeningElement, OpeningParams, PavementElement, PavementParams,
    Scene, ShrubElement, ShrubParams, SiteWallElement, SiteWallParams, TerrainElement, TerrainParams, TreeElement,
    TreeParams, VegetationAreaElement, VegetationAreaParams, WallElement, WallFunction, WallParams, WindowElement,
    WindowParams,
)
from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult
from samcloud.dxf.writer import build_dxf_document, ensure_outdoor_layers, outdoor_symbol_insert
from samcloud.outdoor_model.style import REQUIRED_CATEGORIES, category_style, layer_styles, load_outdoor_style


def _mesh() -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    return ([(0.0, 0.0, 0.01), (2.0, 0.1, 0.0), (0.1, 2.0, 0.02)], [(0, 1, 2)])


def _scene() -> Scene:
    vertices, triangles = _mesh()
    cube = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2)]
    cube_triangles = [(0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7), (4, 5, 6), (4, 6, 7)]
    scene = Scene()
    building = BuildingElement(params=BuildingParams(vertices=cube, triangles=cube_triangles, face_roles=[1] * 8 + [2, 2]))
    scene.add(building)
    wall = WallElement(building_id=building.id, params=WallParams(start=(0, 0), end=(2, 0), thickness=0.2, height=2, function=WallFunction.EXTERIOR))
    scene.add(wall)
    opening = OpeningElement(params=OpeningParams(host_wall_id=wall.id, position_along_axis=0.4, width=0.6, height=1))
    scene.add(opening)
    scene.add(WindowElement(params=WindowParams(opening_id=opening.id)))
    scene.add(TerrainElement(params=TerrainParams(vertices=vertices, triangles=triangles)))
    scene.add(TreeElement(params=TreeParams(trunk_xy=(1, 2), base_z=0, height=6, crown_diameter=4)))
    scene.add(ShrubElement(params=ShrubParams(outline=[(0, 0), (1, 0), (0, 1)], base_z=12, height=1)))
    scene.add(SiteWallElement(params=SiteWallParams(axis=[(0, 2), (2, 2)], thickness=0.2, base_z=12, top_z=13)))
    scene.add(FenceElement(params=FenceParams(axis=[(0, 3, 0), (2, 3, 0)], height=1)))
    for kind in ("road", "sidewalk", "path"):
        scene.add(PavementElement(params=PavementParams(vertices=vertices, triangles=triangles, kind=kind)))
    scene.add(CurbElement(params=CurbParams(axis=[(0, 4, 0), (2, 4, 0)])))
    scene.add(MarkingElement(params=MarkingParams(points=[(0, 5), (2, 5)])))
    scene.add(ManholeElement(params=ManholeParams(center=(1, 6, 0), diameter=0.6)))
    scene.add(VegetationAreaElement(params=VegetationAreaParams(outline=[(0, 7), (1, 7), (0, 8)], base_z=12)))
    return scene


def test_style_file_covers_every_category() -> None:
    styles = load_outdoor_style()
    assert set(REQUIRED_CATEGORIES) <= set(styles)
    for name in REQUIRED_CATEGORIES:
        style = styles[name]
        assert style["layer"] and len(style["rgb"]) == 3 and 1 <= style["aci"] <= 255
        assert 0.0 <= style["transparency"] < 1.0
    assert category_style("road")["layer"] == "S-PAVE"
    # one layer per category, no fallback / default layer
    assert len(layer_styles()) == len(styles)
    assert not {"0", "Sonstiges", "S-OTHER", "IGNORE"} & set(layer_styles())


def test_outdoor_ifc_every_product_styled_with_exactly_one_layer(tmp_path: Path) -> None:
    path = tmp_path / "outdoor.ifc"
    write_ifc(_scene(), path)
    file = ifcopenshell.open(str(path))
    styles = load_outdoor_style()
    layer_of_representation: dict[int, list[str]] = {}
    for layer in file.by_type("IfcPresentationLayerAssignment"):
        for item in layer.AssignedItems:
            layer_of_representation.setdefault(item.id(), []).append(layer.Name)
    seen: set[str] = set()
    for product in file.by_type("IfcProduct"):
        if product.Representation is None:
            continue
        category = outdoor_category_of_product(product)
        assert category is not None, product.is_a()
        style = styles[category]
        seen.add(category)
        for representation in product.Representation.Representations:
            assert layer_of_representation.get(representation.id()) == [style["layer"]], (product.is_a(), category)
            for item in representation.Items:
                targets = list(item.Elements) if item.is_a("IfcGeometricCurveSet") else [item]
                for target in targets:
                    (styled,) = target.StyledByItem
                    presentation = styled.Styles[0]
                    colour = (
                        presentation.Styles[0].SurfaceColour if presentation.is_a("IfcSurfaceStyle") else presentation.CurveColour
                    )
                    got = tuple(round(value * 255) for value in (colour.Red, colour.Green, colour.Blue))
                    assert got == style["rgb"], (category, got)
    assert {"terrain", "road", "sidewalk", "path", "lawn", "curb", "building_wall", "roof", "window", "site_wall",
            "fence", "tree", "shrub", "manhole", "marking"} <= seen
    names = {layer.Name for layer in file.by_type("IfcPresentationLayerAssignment")}
    assert names <= {style["layer"] for style in styles.values()}


def test_indoor_scene_gets_no_outdoor_layers(tmp_path: Path) -> None:
    scene = Scene()
    scene.add(WallElement(params=WallParams(start=(0, 0), end=(3, 0), thickness=0.2, height=2.5)))
    path = tmp_path / "indoor.ifc"
    write_ifc(scene, path)
    file = ifcopenshell.open(str(path))
    assert not file.by_type("IfcPresentationLayerAssignment")


def _dxf_result() -> DxfPlanResult:
    square = [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]]
    layers = {
        "S-PAVE": [{"type": "polyline", "points": square, "closed": True}],
        "S-VEG": [{"type": "polyline", "points": square, "closed": True}],
        "S-CURB": [{"type": "line", "start": [0, 0, 0], "end": [3, 0, 0]}],
        "S-TREE": [outdoor_symbol_insert("TREE", (5.0, 5.0, 1.0), {"HEIGHT": 6.0}, crown_radius=2.0)],
        "S-SHRUB": [outdoor_symbol_insert("SHRUB", (7.0, 5.0, 1.0), {"HEIGHT": 1.0}, crown_radius=0.7)],
        "S-MANH": [outdoor_symbol_insert("MANHOLE", (8.0, 8.0, 1.0), {"Z_DECKEL": 1.0, "TYP": "x"})],
        "S-TOPO-MAJR": [{"type": "polyline3d", "points": [[0, 0, 1], [3, 1, 1], [5, 3, 1]]}],
        "S-TOPO-TIN": [{"type": "3dface", "points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}],
    }
    return DxfPlanResult(
        "outdoor", "site", "outdoor", None, 1.0, Frame(np.zeros(3)), "test", layers, {},
        dict.fromkeys(layers, 7), BackgroundImage(None, None, (0.0, 0.0, 10.0, 10.0)),
    )


def test_outdoor_dxf_layers_use_style_colours_and_entities_are_bylayer(tmp_path: Path) -> None:
    document = build_dxf_document(_dxf_result())
    ensure_outdoor_layers(document, tin_frozen=True)
    path = tmp_path / "outdoor.dxf"
    document.saveas(path)
    document = ezdxf.readfile(path)
    styles = layer_styles()
    for name in _dxf_result().layers:
        layer = document.layers.get(name)
        assert tuple(layer.rgb) == styles[name]["rgb"], name
        assert layer.color == styles[name]["aci"], name
    assert document.layers.get("S-TOPO-TIN").is_frozen()
    for entity in document.modelspace():
        if entity.dxf.layer in styles:
            assert entity.dxf.color == 256 and not entity.dxf.hasattr("true_color"), (entity.dxftype(), entity.dxf.layer)
    for block_name in ("TREE", "SHRUB", "MANHOLE"):
        for entity in document.blocks.get(block_name):
            if entity.dxftype() != "ATTDEF":
                assert entity.dxf.color == 0, block_name  # BYBLOCK: the layer colour of the INSERT applies


def test_viewer_colours_equal_style_colours() -> None:
    styles = load_outdoor_style()
    mapping = {"terrain": "terrain", "pavements": "road", "sidewalks": "sidewalk", "curbs": "curb", "buildings": "building_wall",
               "roofs": "roof", "site_walls": "site_wall", "fences": "fence", "trees": "tree", "shrubs": "shrub",
               "manholes": "manhole", "markings": "marking"}
    for viewer_key, style_key in mapping.items():
        assert ui_data.OUTDOOR_COLORS[viewer_key] == styles[style_key]["hex"]
    assert set(ui_data.OUTDOOR_CATEGORY_ORDER) <= set(ui_data.OUTDOOR_COLORS)


def _grid_scene(with_evidence: bool = True) -> Scene:
    """3x1 strip of 6 TIN triangles; the right 2 cells (4 triangles) are lawn, the left cell is terrain."""
    vertices = [(x + 0.01 * y, y + 0.02 * x, 0.1 * x + 0.003 * y) for x in (0.0, 2.0, 4.0, 6.0) for y in (0.0, 2.0)]
    triangles = []
    for cell in range(3):
        a, b, c, d = 2 * cell, 2 * cell + 1, 2 * cell + 2, 2 * cell + 3
        triangles += [(a, c, b), (b, c, d)]
    evidence = {"lawn_triangle_indices": [2, 3, 4, 5]} if with_evidence else {}
    scene = Scene()
    scene.add(TerrainElement(params=TerrainParams(vertices=vertices, triangles=triangles), evidence=evidence))
    scene.add(VegetationAreaElement(id="lawn-1", tag="lawn", params=VegetationAreaParams(
        outline=[(1.9, -0.5), (6.5, -0.5), (6.5, 2.5), (1.9, 2.5)], base_z=0.3)))
    scene.add(VegetationAreaElement(id="lawn-2", tag="lawn", params=VegetationAreaParams(
        outline=[(20.0, 0.0), (21.0, 0.0), (21.0, 1.0)], base_z=0.3)))  # no TIN triangles: keeps its footprint
    return scene


def test_lawn_surface_is_cut_out_of_terrain(tmp_path: Path) -> None:
    for with_evidence in (True, False):
        path = tmp_path / f"lawn_{with_evidence}.ifc"
        write_ifc(_grid_scene(with_evidence), path)
        file = ifcopenshell.open(str(path))
        products = file.by_type("IfcGeographicElement")
        terrain = next(p for p in products if p.PredefinedType == "TERRAIN")
        lawns = [p for p in products if p.ObjectType == "Vegetationsfläche"]
        assert len(lawns) == 2  # one IFC element per vegetation area (qa_cad count rule)

        def triangle_count(product: ifcopenshell.entity_instance) -> int:
            return sum(len(item.CoordIndex) for rep in product.Representation.Representations for item in rep.Items
                       if item.is_a("IfcTriangulatedFaceSet"))

        surface = next(p for p in lawns if triangle_count(p) > 0)
        footprint = next(p for p in lawns if triangle_count(p) == 0)
        assert triangle_count(terrain) == 2 and triangle_count(surface) == 4  # no gap, no overlap
        assert footprint.Representation.Representations[0].RepresentationIdentifier == "FootPrint"
        assert outdoor_category_of_product(surface) == "lawn"
        layer_names = {layer.Name for layer in file.by_type("IfcPresentationLayerAssignment")}
        assert category_style("lawn")["layer"] in layer_names
        (styled,) = surface.Representation.Representations[0].Items[0].StyledByItem
        colour = styled.Styles[0].Styles[0].SurfaceColour
        assert tuple(round(v * 255) for v in (colour.Red, colour.Green, colour.Blue)) == category_style("lawn")["rgb"]


def test_terrain_builder_lists_lawn_triangles() -> None:
    from samcloud.drone import contracts as c
    from samcloud.outdoor_model.terrain import build_terrain_elements
    vertices = np.array([(0, 0, 0.0), (2, 0.1, 0.01), (0.1, 2, 0.02), (2.2, 2.1, 0.03), (4.1, 0.2, 0.01)], dtype=float)
    triangles = np.array([(0, 1, 2), (1, 3, 2), (1, 4, 3)])
    labels = np.array([c.TIN_LABELS["terrain"], c.TIN_LABELS["lawn"], c.TIN_LABELS["road"]], dtype=np.uint8)
    (element,), _qa = build_terrain_elements(c.LabeledTin(vertices, triangles, labels), config=c.load_drone_config())
    assert len(element.params.triangles) == 2
    assert element.evidence["lawn_triangle_indices"] == [1]


def test_viewer_has_path_and_lawn_categories() -> None:
    rows = [
        {"ifc_class": "IfcCivilElement", "name": "Pavement", "object_type": "Weg"},
        {"ifc_class": "IfcCivilElement", "name": "Pavement", "object_type": "Strasse"},
        {"ifc_class": "IfcGeographicElement", "name": "Rasen lawn-1", "object_type": "Vegetationsfläche", "predefined_type": "USERDEFINED"},
    ]
    _by_node, counts = ui_data.category_index({"elements": {f"g{i}": row for i, row in enumerate(rows)}}, True)
    assert counts == {"lawns": 1, "pavements": 1, "paths": 1}
    styles = load_outdoor_style()
    assert ui_data.OUTDOOR_COLORS["lawns"] == styles["lawn"]["hex"] and ui_data.OUTDOOR_COLORS["paths"] == styles["path"]["hex"]
    import json
    root = Path(__file__).resolve().parents[1] / "samcloud" / "app" / "locales"
    for lang in ("de", "en"):
        labels = json.loads((root / lang / "dxf.json").read_text(encoding="utf-8"))["ifc_outdoor"]
        assert {"lawns", "paths"} <= set(labels)
