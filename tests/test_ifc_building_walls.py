"""IFC building walls: one IfcWall per facade plane, exactly the LoD2 wall faces, no duplicate wall mesh."""
import numpy as np
import ifcopenshell
import ifcopenshell.geom

from samcloud.bim.core.ifc.writer import write_ifc
from samcloud.drone.contracts import BuildingModel, LocalFrame
from samcloud.outdoor_model.buildings import build_building_elements
from samcloud.outdoor_model.scene import assemble_scene

EAVE, RIDGE = 3.0, 5.0


def _gable_house_with_step() -> BuildingModel:
    """8 x 6 m gable house; the back facade has a 0.3 m stair-step at x = 4 (two planes + a riser)."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    roles: list[int] = []

    def vid(point):
        point = tuple(float(v) for v in point)
        if point not in vertices:
            vertices.append(point)
        return vertices.index(point)

    def polygon(points, role):
        ids = [vid(p) for p in points]
        for k in range(1, len(ids) - 1):
            faces.append((ids[0], ids[k], ids[k + 1]))
            roles.append(role)

    polygon([(0, 0, 0), (8, 0, 0), (8, 0, EAVE), (0, 0, EAVE)], 1)  # front
    polygon([(0, 6, 0), (0, 6, EAVE), (4, 6, EAVE), (4, 6, 0)], 1)  # back A
    polygon([(4, 6, 0), (4, 6, EAVE), (4, 6.3, EAVE), (4, 6.3, 0)], 1)  # riser
    polygon([(4, 6.3, 0), (4, 6.3, EAVE), (8, 6.3, EAVE), (8, 6.3, 0)], 1)  # back B
    polygon([(0, 0, 0), (0, 0, EAVE), (0, 3, RIDGE), (0, 6, EAVE), (0, 6, 0)], 1)  # gable west
    polygon([(8, 0, 0), (8, 6.3, 0), (8, 6.3, EAVE), (8, 3.15, RIDGE), (8, 0, EAVE)], 1)  # gable east
    polygon([(0, 0, EAVE), (8, 0, EAVE), (8, 3.15, RIDGE), (0, 3, RIDGE)], 2)  # roof front
    polygon([(0, 3, RIDGE), (8, 3.15, RIDGE), (8, 6.3, EAVE), (4, 6.3, EAVE), (4, 6, EAVE), (0, 6, EAVE)], 2)  # roof back
    return BuildingModel(1, "2.2", np.asarray(vertices, float), np.asarray(faces, int), np.asarray(roles), "ok")


def _world_vertices(product) -> np.ndarray:
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    shape = ifcopenshell.geom.create_shape(settings, product)
    return np.asarray(shape.geometry.verts, float).reshape(-1, 3)


def test_building_walls_are_exact_lod2_faces_per_facade_plane(tmp_path):
    model = _gable_house_with_step()
    elements, qa = build_building_elements([model], {}, {})
    assert not qa
    scene = assemble_scene(elements, frame=LocalFrame((2600000, 1200000, 500)))
    path = tmp_path / "walls.ifc"
    write_ifc(scene, path)
    ifc = ifcopenshell.open(str(path))

    walls = ifc.by_type("IfcWall")
    # planes: front, back A, back B (the riser merges into one of them), west gable, east gable
    assert len(walls) == 5
    assert not any("exterior wall" in (w.Name or "") for w in walls)
    back = [w for w in walls if abs(np.mean(_world_vertices(w)[:, 1]) - 6.0) < 0.5]
    assert 1 <= len(back) <= 2

    wall_ids = np.unique(model.faces[model.face_roles == 1].ravel())
    lod2 = model.vertices[wall_ids]
    bottom = float(model.vertices[:, 2].min())
    roof_edge = float(model.vertices[np.unique(model.faces[model.face_roles == 2].ravel())][:, 2].max())
    for wall in walls:
        verts = _world_vertices(wall)
        assert verts[:, 2].min() >= bottom - 1e-6
        assert verts[:, 2].max() <= roof_edge + 0.01
        for v in verts:  # every wall vertex is an LoD2 wall vertex (no extrusion beyond the mesh)
            assert np.min(np.linalg.norm(lod2 - v, axis=1)) < 1e-3
