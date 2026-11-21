"""Regression tests for IFC4 outdoor survey exports."""

from __future__ import annotations

import math
from pathlib import Path

import ifcopenshell
import pytest

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.ifc import shapes
from samcloud.bim.core.qa import validate_ifc
from samcloud.bim.core.model import (
    BuildingElement, BuildingParams, CrsInfo, CurbElement, CurbParams, FenceElement,
    FenceParams, ManholeElement, ManholeParams, MarkingElement, MarkingParams,
    ParamOrigin, PavementElement, PavementParams, Scene, ShrubElement, ShrubParams, SiteWallElement,
    SiteWallParams, TerrainElement, TerrainParams, TreeElement, TreeParams,
    VegetationAreaElement, VegetationAreaParams, WallElement, WallFunction, WallParams,
    WindowElement, WindowParams, OpeningElement, OpeningParams,
)


def _export(scene: Scene, path: Path) -> ifcopenshell.file:
    write_ifc(scene, path)
    return ifcopenshell.open(str(path))


def _mesh() -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    return ([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)], [(0, 1, 2)])


def _pset_names(product: ifcopenshell.entity_instance) -> set[str]:
    return {relation.RelatingPropertyDefinition.Name for relation in product.IsDefinedBy}


def test_triangulated_face_set_round_trip(tmp_path: Path) -> None:
    vertices, triangles = _mesh()
    file = ifcopenshell.file(schema="IFC4")
    item = shapes.triangulated_face_set(file, vertices, triangles)
    path = tmp_path / "mesh.ifc"
    file.write(str(path))
    reopened = ifcopenshell.open(str(path))
    face_set = reopened.by_type("IfcTriangulatedFaceSet")[0]
    assert len(face_set.Coordinates.CoordList) == len(vertices)
    assert len(face_set.CoordIndex) == len(triangles)


def test_terrain_is_site_contained_geographic_element(tmp_path: Path) -> None:
    vertices, triangles = _mesh()
    scene = Scene()
    scene.add(TerrainElement(params=TerrainParams(vertices=vertices, triangles=triangles)))
    file = _export(scene, tmp_path / "terrain.ifc")
    (terrain,) = file.by_type("IfcGeographicElement")
    assert terrain.PredefinedType == "TERRAIN"
    assert terrain.ContainedInStructure[0].RelatingStructure.is_a("IfcSite")


def test_curb_straight_segments_are_extrusions(tmp_path: Path) -> None:
    scene = Scene()
    scene.add(CurbElement(params=CurbParams(axis=[(0, 0, 0), (2, 0, 0), (3.88, 0.684, 0)])))
    file = _export(scene, tmp_path / "curb.ifc")
    solids = file.by_type("IfcExtrudedAreaSolid")
    assert len(solids) == 2
    assert not file.by_type("IfcFixedReferenceSweptAreaSolid")
    first_end = solids[0].Position.Location.Coordinates[0] + solids[0].Depth
    assert first_end >= 2.0


def test_curb_arc_switches_between_sweep_and_extrusion(tmp_path: Path) -> None:
    points = [(8 * math.cos(index * 0.1), 8 * math.sin(index * 0.1), 0.003 * (-1) ** index) for index in range(10)]
    auto = Scene()
    auto.add(CurbElement(params=CurbParams(axis=points)))
    auto_file = _export(auto, tmp_path / "arc.ifc")
    assert len(auto_file.by_type("IfcFixedReferenceSweptAreaSolid")) == 1
    only = Scene()
    only.add(CurbElement(params=CurbParams(axis=points, curb_sweep="extrude_only")))
    only_file = _export(only, tmp_path / "arc_extrude.ifc")
    assert not only_file.by_type("IfcFixedReferenceSweptAreaSolid")
    assert only_file.by_type("IfcExtrudedAreaSolid")


def test_pavement_mesh_is_closed_and_has_requested_thickness(tmp_path: Path) -> None:
    vertices, triangles = _mesh()
    scene = Scene()
    scene.add(PavementElement(params=PavementParams(vertices=vertices, triangles=triangles, thickness=0.10)))
    file = _export(scene, tmp_path / "pavement.ifc")
    face_set = file.by_type("IfcTriangulatedFaceSet")[0]
    edges: dict[tuple[int, int], int] = {}
    for face in face_set.CoordIndex:
        for a, b in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((a, b)))
            edges[edge] = edges.get(edge, 0) + 1
    assert set(edges.values()) == {2}
    z_values = [point[2] for point in face_set.Coordinates.CoordList]
    assert max(z_values) - min(z_values) == 0.10


def test_stylised_tree_triangle_count_and_assumed_parameter(tmp_path: Path) -> None:
    scene = Scene()
    tree = TreeElement(params=TreeParams(trunk_xy=(1, 2), base_z=0, height=6, crown_diameter=4))
    scene.add(tree)
    file = _export(scene, tmp_path / "tree.ifc")
    assert 60 <= sum(len(fs.CoordIndex) for fs in file.by_type("IfcTriangulatedFaceSet")) <= 120  # trunk + crown
    product = file.by_type("IfcGeographicElement")[0]
    pset = next(pset for pset in product.IsDefinedBy if pset.RelatingPropertyDefinition.Name == "SAMcloud_Herkunft")
    values = {item.Name: item.NominalValue.wrappedValue for item in pset.RelatingPropertyDefinition.HasProperties}
    assert values["Herkunft_trunk_diameter"] == "assumed"


def test_outdoor_categories_have_expected_classes_and_psets(tmp_path: Path) -> None:
    vertices, triangles = _mesh()
    scene = Scene()
    shrub = ShrubElement(
        params=ShrubParams(outline=[(0, 0), (1, 0), (0, 1)], base_z=12, height=1)
    )
    scene.add(shrub)
    scene.add(
        SiteWallElement(
            params=SiteWallParams(axis=[(0, 2), (2, 2)], thickness=.2, base_z=12, top_z=13)
        )
    )
    scene.add(FenceElement(params=FenceParams(axis=[(0, 3, 0), (2, 3, 0)], height=1)))
    scene.add(PavementElement(params=PavementParams(vertices=vertices, triangles=triangles)))
    scene.add(CurbElement(params=CurbParams(axis=[(0, 4, 0), (2, 4, 0)])))
    scene.add(MarkingElement(params=MarkingParams(points=[(0, 5), (2, 5)])))
    scene.add(ManholeElement(params=ManholeParams(center=(1, 6, 0), diameter=.6)))
    scene.add(
        VegetationAreaElement(
            params=VegetationAreaParams(outline=[(0, 7), (1, 7), (0, 8)], base_z=12)
        )
    )
    file = _export(scene, tmp_path / "categories.ifc")
    expected = {"IfcGeographicElement": ({"SAMcloud_Shrub", "SAMcloud_VegetationArea"}, {"Busch", "Vegetationsfläche"}), "IfcWall": ({"SAMcloud_SiteWall"}, {"Mauer"}), "IfcRailing": ({"SAMcloud_Fence"}, {"Zaun"}), "IfcCivilElement": ({"SAMcloud_Pavement", "SAMcloud_Curb"}, {"Strasse", "Randstein"}), "IfcAnnotation": ({"SAMcloud_Marking"}, {"Markierung"}), "IfcDistributionChamberElement": ({"SAMcloud_Manhole"}, {""})}
    for ifc_class, (psets, object_types) in expected.items():
        found = set().union(*(_pset_names(product) for product in file.by_type(ifc_class)))
        assert psets <= found
        assert object_types <= {product.ObjectType for product in file.by_type(ifc_class)}
    assert shrub.param_origin["base_z"] is ParamOrigin.MEASURED


def test_outdoor_outline_elements_require_a_measured_base_height() -> None:
    with pytest.raises(ValueError, match="shrub base_z is required"):
        ShrubParams(outline=[(0, 0), (1, 0), (0, 1)], height=1)
    with pytest.raises(ValueError, match="site wall base_z is required"):
        SiteWallParams(axis=[(0, 0), (1, 0)], thickness=.2)
    with pytest.raises(ValueError, match="vegetation area base_z is required"):
        VegetationAreaParams(outline=[(0, 0), (1, 0), (0, 1)])


def test_shrub_and_vegetation_footprint_use_their_base_height(tmp_path: Path) -> None:
    scene = Scene()
    scene.add(
        ShrubElement(
            params=ShrubParams(outline=[(0, 0), (1, 0), (0, 1)], base_z=42, height=2)
        )
    )
    scene.add(
        VegetationAreaElement(
            params=VegetationAreaParams(outline=[(2, 0), (3, 0), (2, 1)], base_z=42)
        )
    )
    file = _export(scene, tmp_path / "vegetation.ifc")

    shrub_mesh = file.by_type("IfcTriangulatedFaceSet")[0]
    assert {point[2] for point in shrub_mesh.Coordinates.CoordList} == {42.0, 44.0}
    vegetation = next(
        product
        for product in file.by_type("IfcGeographicElement")
        if product.ObjectType == "Vegetationsfläche"
    )
    representations = vegetation.Representation.Representations
    assert [(item.RepresentationIdentifier, item.RepresentationType) for item in representations] == [
        ("FootPrint", "GeometricCurveSet")
    ]
    assert not [item for item in representations if item.RepresentationIdentifier == "Body"]
    curve = representations[0].Items[0].Elements[0]
    assert {point.Coordinates[2] for point in curve.Points} == {42.0}


def test_outdoor_building_envelope_and_window_relations(tmp_path: Path) -> None:
    vertices = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2)]
    triangles = [(0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7), (4, 5, 6), (4, 6, 7)]
    scene = Scene()
    building = BuildingElement(params=BuildingParams(vertices=vertices, triangles=triangles, face_roles=[1] * 8 + [2, 2]))
    scene.add(building)
    wall = WallElement(building_id=building.id, params=WallParams(start=(0, 0), end=(2, 0), thickness=.2, height=2, function=WallFunction.EXTERIOR))
    scene.add(wall)
    opening = OpeningElement(params=OpeningParams(host_wall_id=wall.id, position_along_axis=.4, width=.6, height=1))
    scene.add(opening)
    scene.add(WindowElement(params=WindowParams(opening_id=opening.id)))
    file = _export(scene, tmp_path / "building.ifc")
    assert any(wall.ObjectType == "EXTERIOR" for wall in file.by_type("IfcWall"))
    assert len(file.by_type("IfcRoof")) == 1 and not any(slab.PredefinedType == "ROOF" for slab in file.by_type("IfcSlab"))  # one IfcRoof with its own body
    assert file.by_type("IfcRelVoidsElement") and file.by_type("IfcRelFillsElement")


def test_outdoor_wall_and_roof_pass_the_indoor_ids_gate(tmp_path: Path) -> None:
    vertices = [
        (0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),
        (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2),
    ]
    triangles = [
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
        (4, 5, 6), (4, 6, 7),
    ]
    scene = Scene()
    scene.add(
        BuildingElement(
            params=BuildingParams(vertices=vertices, triangles=triangles, face_roles=[1] * 8 + [2, 2])
        )
    )
    scene.add(
        SiteWallElement(
            params=SiteWallParams(axis=[(3, 0), (5, 0)], thickness=.2, base_z=4, top_z=5)
        )
    )
    result = write_ifc(scene, tmp_path / "outdoor_ids.ifc")
    report = validate_ifc(result.path, reports_dir=tmp_path / "reports")

    assert report.ids_failures == []
    applicability = {item["name"]: item["applicable"] for item in report.ids_specifications}
    assert applicability["IfcWall"] == 0
    assert applicability["IfcSlab floor"] == 0
    assert applicability["IfcSlab ceiling"] == 0


def test_outdoor_only_defaults_to_swiss_georeference(tmp_path: Path) -> None:
    vertices, triangles = _mesh()
    scene = Scene()
    scene.settings.origin_offset = (2600000.0, 1200000.0, 500.0)
    scene.add(TerrainElement(params=TerrainParams(vertices=vertices, triangles=triangles)))
    file = _export(scene, tmp_path / "referenced.ifc")
    conversion = file.by_type("IfcMapConversion")[0]
    projected = conversion.TargetCRS
    assert projected.Name == "EPSG:2056"
    assert projected.VerticalDatum == "LN02"
    assert max(abs(value) for point in file.by_type("IfcCartesianPointList3D")[0].CoordList for value in point) < 10000
