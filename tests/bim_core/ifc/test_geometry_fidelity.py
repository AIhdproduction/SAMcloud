"""The IFC bodies, placements, usages and quantities reproduce the scene geometry."""

from __future__ import annotations

from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as element_util
import ifcopenshell.util.placement
import ifcopenshell.util.shape
import numpy as np
import numpy.typing as npt
import pytest

from samcloud.bim.core.model import (
    ElementKind,
    SlabElement,
    SpaceElement,
    WallElement,
    WallJustification,
)
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene

from .conftest import Export, entity_by_tag, export_scene, validation_statements

SETTINGS = ifcopenshell.geom.settings()
TOLERANCE = 1e-6


def shape_vertices(
    element: ifcopenshell.entity_instance, *, world: bool
) -> npt.NDArray[np.float64]:
    """Vertices of the Body representation, in the local frame or in the world frame."""
    shape = ifcopenshell.geom.create_shape(SETTINGS, element)
    verts = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
    if not world:
        return verts
    matrix = np.array(shape.transformation.matrix, dtype=np.float64).reshape(4, 4).T
    return (matrix @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]


def world_box(element: ifcopenshell.entity_instance) -> tuple[list[float], list[float]]:
    verts = shape_vertices(element, world=True)
    return verts.min(axis=0).tolist(), verts.max(axis=0).tolist()


def volume(element: ifcopenshell.entity_instance) -> float:
    shape = ifcopenshell.geom.create_shape(SETTINGS, element)
    return float(ifcopenshell.util.shape.get_volume(shape.geometry))


def qto(element: ifcopenshell.entity_instance, name: str) -> dict[str, float]:
    sets = element_util.get_psets(element, qtos_only=True)
    assert name in sets, f"{name} missing on {element.Name}"
    return {key: value for key, value in sets[name].items() if key != "id"}


def pset(element: ifcopenshell.entity_instance, name: str) -> dict[str, object]:
    sets = element_util.get_psets(element, psets_only=True)
    assert name in sets, f"{name} missing on {element.Name}"
    return {key: value for key, value in sets[name].items() if key != "id"}


def test_wall_bodies_match_footprints(layout_export: Export) -> None:
    f = layout_export.file
    for scene_wall in layout_export.scene.elements_of(ElementKind.WALL):
        geometry = layout_export.geometry.walls[scene_wall.id]
        low, high = world_box(entity_by_tag(f, scene_wall.id))
        xs = [p[0] for p in geometry.footprint]
        ys = [p[1] for p in geometry.footprint]
        assert low == pytest.approx([min(xs), min(ys), geometry.base_z], abs=TOLERANCE)
        assert high == pytest.approx([max(xs), max(ys), geometry.top_z], abs=TOLERANCE)


def test_wall_quantities_equal_geometry(layout_export: Export) -> None:
    f = layout_export.file
    for scene_wall in layout_export.scene.elements_of(ElementKind.WALL):
        wall = entity_by_tag(f, scene_wall.id)
        geometry = layout_export.geometry.walls[scene_wall.id]
        values = qto(wall, "Qto_WallBaseQuantities")
        assert values == pytest.approx(geometry.quantities, abs=1e-12)
        assert values["Width"] == pytest.approx(scene_wall.params.thickness)  # type: ignore[attr-defined]
        common = pset(wall, "Pset_WallCommon")
        assert common["IsExternal"] is (scene_wall.params.function.value == "exterior")  # type: ignore[attr-defined]
        assert "LoadBearing" not in common
        assert common["Reference"] == scene_wall.tag


def test_wall_net_volume_matches_boolean_cut(layout_export: Export) -> None:
    """The openings really cut the wall bodies: boolean volume equals the geometry net volume."""
    f = layout_export.file
    cut_walls = 0
    for scene_wall in layout_export.scene.elements_of(ElementKind.WALL):
        geometry = layout_export.geometry.walls[scene_wall.id]
        assert volume(entity_by_tag(f, scene_wall.id)) == pytest.approx(
            geometry.net_volume, abs=1e-6
        )
        if geometry.net_volume < geometry.gross_volume - 1e-9:
            cut_walls += 1
    assert cut_walls >= 2


def test_openings_and_fillings_sit_in_their_walls(layout_export: Export) -> None:
    f = layout_export.file
    for scene_opening in layout_export.scene.elements_of(ElementKind.OPENING):
        geometry = layout_export.geometry.openings[scene_opening.id]
        corners = np.array(geometry.corners)
        low, high = world_box(entity_by_tag(f, scene_opening.id))
        assert low == pytest.approx(corners.min(axis=0).tolist(), abs=TOLERANCE)
        assert high == pytest.approx(corners.max(axis=0).tolist(), abs=TOLERANCE)
        values = qto(entity_by_tag(f, scene_opening.id), "Qto_OpeningElementBaseQuantities")
        assert values["Width"] == pytest.approx(geometry.width)
        assert values["Height"] == pytest.approx(geometry.height)
        assert values["Area"] == pytest.approx(geometry.area)
        assert values["Depth"] == pytest.approx(geometry.depth)
    for pool, name in (
        (layout_export.geometry.windows, "Qto_WindowBaseQuantities"),
        (layout_export.geometry.doors, "Qto_DoorBaseQuantities"),
    ):
        for element_id, filling in pool.items():
            product = entity_by_tag(f, element_id)
            low, high = world_box(product)
            box_low = np.array(filling.corners).min(axis=0)
            box_high = np.array(filling.corners).max(axis=0)
            # the frame lies inside the opening box (thinner than the wall)
            assert np.all(np.array(low) >= box_low - TOLERANCE)
            assert np.all(np.array(high) <= box_high + TOLERANCE)
            assert low[0] == pytest.approx(box_low[0], abs=TOLERANCE) or low[1] == pytest.approx(
                box_low[1], abs=TOLERANCE
            )
            assert low[2] == pytest.approx(box_low[2], abs=TOLERANCE)
            assert high[2] == pytest.approx(box_high[2], abs=TOLERANCE)
            values = qto(product, name)
            assert values["Width"] == pytest.approx(filling.width)
            assert values["Height"] == pytest.approx(filling.height)
            assert values["Area"] == pytest.approx(filling.area)


def test_slab_and_ceiling_geometry(layout_export: Export) -> None:
    f = layout_export.file
    for scene_slab in layout_export.scene.elements_of(ElementKind.SLAB):
        geometry = layout_export.geometry.slabs[scene_slab.id]
        slab = entity_by_tag(f, scene_slab.id)
        low, high = world_box(slab)
        assert low[2] == pytest.approx(geometry.bottom_z, abs=TOLERANCE)
        assert high[2] == pytest.approx(geometry.top_z, abs=TOLERANCE)
        assert slab.PredefinedType == "FLOOR"
        values = qto(slab, "Qto_SlabBaseQuantities")
        assert values["Depth"] == pytest.approx(geometry.thickness)
        assert values["GrossArea"] == pytest.approx(geometry.gross_area)
        assert values["Perimeter"] == pytest.approx(geometry.perimeter)
        usage = element_util.get_material(slab)
        assert usage.LayerSetDirection == "AXIS3" and usage.DirectionSense == "NEGATIVE"
        assert volume(slab) == pytest.approx(geometry.net_volume, abs=1e-6)
    for scene_ceiling in layout_export.scene.elements_of(ElementKind.CEILING):
        geometry_c = layout_export.geometry.ceilings[scene_ceiling.id]
        ceiling = entity_by_tag(f, scene_ceiling.id)
        low, high = world_box(ceiling)
        assert low[2] == pytest.approx(geometry_c.bottom_z, abs=TOLERANCE)
        assert high[2] == pytest.approx(geometry_c.top_z, abs=TOLERANCE)
        assert ceiling.is_a("IfcSlab")
        assert ceiling.PredefinedType == "USERDEFINED"
        assert ceiling.ObjectType == "CEILING"
        values = qto(ceiling, "Qto_SlabBaseQuantities")
        assert values["Depth"] == pytest.approx(geometry_c.thickness)
        assert values["NetArea"] == pytest.approx(geometry_c.net_area)
        usage = element_util.get_material(ceiling)
        assert usage.LayerSetDirection == "AXIS3" and usage.DirectionSense == "POSITIVE"


def test_space_bodies_and_quantities(layout_export: Export) -> None:
    f = layout_export.file
    for scene_space in layout_export.scene.elements.values():
        if not isinstance(scene_space, SpaceElement):
            continue
        geometry = layout_export.geometry.spaces[scene_space.id]
        space = next(s for s in f.by_type("IfcSpace") if s.Name == scene_space.params.number)
        low, high = world_box(space)
        xs = [p[0] for p in geometry.exterior]
        ys = [p[1] for p in geometry.exterior]
        assert low == pytest.approx([min(xs), min(ys), geometry.floor_z], abs=TOLERANCE)
        assert high == pytest.approx([max(xs), max(ys), geometry.ceiling_z], abs=TOLERANCE)
        assert qto(space, "Qto_SpaceBaseQuantities") == pytest.approx(
            geometry.quantities, abs=1e-12
        )
        assert volume(space) == pytest.approx(geometry.gross_volume, abs=1e-6)


def _boundary_polygon_world(
    space: ifcopenshell.entity_instance, boundary: ifcopenshell.entity_instance
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """World points of the outer boundary of the connection surface and its normal."""
    plane = boundary.ConnectionGeometry.SurfaceOnRelatingElement
    frame = ifcopenshell.util.placement.get_axis2placement(plane.BasisSurface.Position)
    space_matrix = ifcopenshell.util.placement.get_local_placement(space.ObjectPlacement)
    matrix = space_matrix @ frame
    points = np.array([(*p.Coordinates, 0.0) for p in plane.OuterBoundary.Points], dtype=np.float64)
    world = (matrix @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    return world, matrix[:3, 2]


def test_space_boundaries_lie_on_the_room_side_face_and_point_away(layout_export: Export) -> None:
    """Wall boundaries lie on the wall face towards the room, normals point into the wall."""
    from shapely.geometry import Point, Polygon

    f = layout_export.file
    for scene_space in layout_export.scene.elements.values():
        if not isinstance(scene_space, SpaceElement):
            continue
        geometry = layout_export.geometry.spaces[scene_space.id]
        room = Polygon(geometry.exterior)
        space = next(s for s in f.by_type("IfcSpace") if s.Name == scene_space.params.number)
        for boundary in space.BoundedBy:
            world, normal = _boundary_polygon_world(space, boundary)
            related = boundary.RelatedBuildingElement
            if related.is_a("IfcWall"):
                wall_geometry = layout_export.geometry.walls[
                    next(e.id for e in layout_export.scene.elements.values() if e.id == related.Tag)
                ]
                footprint = Polygon(wall_geometry.footprint)
                assert abs(normal[2]) < 1e-9
                # every boundary point lies on the wall boundary and on the room boundary
                for x, y, _z in world:
                    assert footprint.exterior.distance(Point(x, y)) < 1e-6
                    assert room.exterior.distance(Point(x, y)) < 2e-3
                # the normal points away from the room, into the wall
                centre = world.mean(axis=0)
                probe = Point(centre[0] + 0.01 * normal[0], centre[1] + 0.01 * normal[1])
                assert footprint.contains(probe)
                assert not room.contains(probe)
                # vertical extent: floor to ceiling
                assert world[:, 2].min() == pytest.approx(geometry.floor_z, abs=TOLERANCE)
                assert world[:, 2].max() == pytest.approx(geometry.ceiling_z, abs=TOLERANCE)
            else:
                heights = world[:, 2]
                assert np.ptp(heights) < 1e-9
                if related.PredefinedType == "FLOOR":
                    assert normal[2] == pytest.approx(-1.0)
                    assert heights[0] == pytest.approx(geometry.floor_z, abs=TOLERANCE)
                else:
                    assert normal[2] == pytest.approx(1.0)
                    assert heights[0] == pytest.approx(geometry.ceiling_z, abs=TOLERANCE)


@pytest.mark.parametrize(
    "justification",
    [WallJustification.CENTER, WallJustification.LEFT, WallJustification.RIGHT],
)
@pytest.mark.parametrize("layout", [RoomLayout.RECTANGLE, RoomLayout.TWO_ROOMS])
def test_layer_set_usage_offset_matches_body(
    tmp_path: Path, justification: WallJustification, layout: RoomLayout
) -> None:
    """Pitfall 3: OffsetFromReferenceLine equals the body offset along the local Y axis."""
    scene = build_room_scene(RoomSpec(layout=layout))
    for element in scene.elements.values():
        if isinstance(element, WallElement):
            element.params.justification = justification
    export = export_scene(scene, tmp_path / "model.ifc")
    assert validation_statements(export.file) == []
    for element in scene.elements_of(ElementKind.WALL):
        geometry = export.geometry.walls[element.id]
        wall = entity_by_tag(export.file, element.id)
        usage = element_util.get_material(wall)
        local = shape_vertices(wall, world=False)
        assert usage.OffsetFromReferenceLine == pytest.approx(geometry.offset_right)
        assert local[:, 1].min() == pytest.approx(usage.OffsetFromReferenceLine, abs=TOLERANCE)
        assert local[:, 1].max() == pytest.approx(
            usage.OffsetFromReferenceLine + geometry.thickness, abs=TOLERANCE
        )
        # mitre cuts move the corners at most one thickness beyond the trimmed axis ends
        assert -geometry.thickness - TOLERANCE <= local[:, 0].min() <= geometry.thickness
        assert geometry.length - geometry.thickness <= local[:, 0].max()
        assert local[:, 0].max() <= geometry.length + geometry.thickness + TOLERANCE
        low, high = world_box(wall)
        xs = [p[0] for p in geometry.footprint]
        ys = [p[1] for p in geometry.footprint]
        assert low[:2] == pytest.approx([min(xs), min(ys)], abs=TOLERANCE)
        assert high[:2] == pytest.approx([max(xs), max(ys)], abs=TOLERANCE)
        # axis representation: 2D curve from (0, 0) to (length, 0)
        axis = next(
            r for r in wall.Representation.Representations if r.RepresentationIdentifier == "Axis"
        )
        assert axis.RepresentationType == "Curve2D"
        points = [tuple(p) for p in axis.Items[0].Points.CoordList]
        assert np.allclose(points, [(0.0, 0.0), (geometry.length, 0.0)], atol=TOLERANCE)


def test_wall_body_profile_kinds(layout_export: Export) -> None:
    """Rectangle profile for square-ended walls, arbitrary profile for mitred or trimmed ones."""
    f = layout_export.file
    kinds = {}
    for scene_wall in layout_export.scene.elements_of(ElementKind.WALL):
        wall = entity_by_tag(f, scene_wall.id)
        body = next(
            r for r in wall.Representation.Representations if r.RepresentationIdentifier == "Body"
        )
        assert body.RepresentationType == "SweptSolid"
        (solid,) = body.Items
        assert solid.is_a("IfcExtrudedAreaSolid")
        kinds[scene_wall.tag] = solid.SweptArea.is_a()
    assert kinds["exterior_south"] == "IfcArbitraryClosedProfileDef"
    assert set(kinds.values()) <= {"IfcRectangleProfileDef", "IfcArbitraryClosedProfileDef"}


def test_wall_without_joins_uses_rectangle_profile(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    for key in [k for k, e in scene.elements.items() if e.kind == ElementKind.WALL_JOIN]:
        del scene.elements[key]
    export = export_scene(scene, tmp_path / "model.ifc")
    assert validation_statements(export.file) == []
    for wall in export.file.by_type("IfcWall"):
        body = next(
            r for r in wall.Representation.Representations if r.RepresentationIdentifier == "Body"
        )
        assert body.Items[0].SweptArea.is_a("IfcRectangleProfileDef")
    assert not export.file.by_type("IfcRelConnectsPathElements")


def test_slab_with_hole_validates_and_has_voids(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    slab = next(e for e in scene.elements.values() if isinstance(e, SlabElement))
    slab.params.openings = [[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]]
    export = export_scene(scene, tmp_path / "model.ifc")
    assert validation_statements(export.file) == []
    product = entity_by_tag(export.file, slab.id)
    geometry = export.geometry.slabs[slab.id]
    body = next(
        r for r in product.Representation.Representations if r.RepresentationIdentifier == "Body"
    )
    assert body.Items[0].SweptArea.is_a("IfcArbitraryProfileDefWithVoids")
    assert len(body.Items[0].SweptArea.InnerCurves) == 1
    assert qto(product, "Qto_SlabBaseQuantities")["NetArea"] == pytest.approx(geometry.net_area)
    assert volume(product) == pytest.approx(geometry.net_volume, abs=1e-6)
