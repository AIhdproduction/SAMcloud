import numpy as np
import trimesh
from shapely.geometry import Polygon

from samcloud.buildings.extrude import extrude_lod22
from samcloud.buildings.partition import Partition
from samcloud.buildings.validate import validate_model
from samcloud.drone.contracts import RoofPlane


def _flat(height, outline):
    return RoofPlane(np.array([0., 0., 1.]), -height, np.array([], dtype=int), 0., outline=Polygon(outline))


def _sloped(slope, intercept, outline):
    normal = np.array([-slope[0], -slope[1], 1.0])
    normal /= np.linalg.norm(normal)
    return RoofPlane(normal, -intercept / np.linalg.norm(np.array([-slope[0], -slope[1], 1.0])), np.array([], dtype=int), 0., outline=Polygon(outline))


def test_flat_annex_has_step_wall_and_is_watertight():
    footprint = Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])
    cells = [Polygon([(0, 0), (5, 0), (5, 6), (0, 6)]), Polygon([(5, 0), (10, 0), (10, 6), (5, 6)])]
    mesh = extrude_lod22(Partition(cells, [(0, 1, 6), (1, 0, 6)]), np.array([1, 2]), [_flat(6, cells[0].exterior.coords), _flat(7.5, cells[1].exterior.coords)], footprint, 0.)
    vertices, faces, roles = mesh
    report = validate_model(vertices, faces, np.array([[1, 1, 6], [8, 1, 7.5]]), footprint)
    assert report["status"] == "ok"
    walls = vertices[faces[roles == 1]]
    assert np.any(np.isclose(walls[:, :, 0], 5).all(axis=1))


def test_face_roles_have_expected_normals():
    footprint = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    vertices, faces, roles = extrude_lod22(Partition([footprint], []), np.array([1]), [_flat(5, footprint.exterior.coords)], footprint, 0.)
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    assert (normals[roles == 0, 2] < 0).all()
    assert (normals[roles == 2, 2] > 0).all()
    assert (np.abs(normals[roles == 1, 2]) < .1).all()


def test_gable_and_hip_extrusions_are_watertight_spheres():
    gable_footprint = Polygon([(0, 0), (10, 0), (10, 8), (0, 8)])
    gable_cells = [Polygon([(0, 0), (5, 0), (5, 8), (0, 8)]), Polygon([(5, 0), (10, 0), (10, 8), (5, 8)])]
    gable = extrude_lod22(
        Partition(gable_cells, [(0, 1, 8), (1, 0, 8)]),
        np.array([1, 2]),
        [_sloped((.6, 0.), 6., gable_cells[0].exterior.coords), _sloped((-.6, 0.), 12., gable_cells[1].exterior.coords)],
        gable_footprint,
        0.,
    )
    hip_footprint = Polygon([(0, 0), (12, 0), (12, 8), (0, 8)])
    hip_cells = [
        Polygon([(0, 0), (4, 4), (0, 8)]),
        Polygon([(12, 0), (12, 8), (8, 4)]),
        Polygon([(0, 0), (12, 0), (8, 4), (4, 4)]),
        Polygon([(0, 8), (4, 4), (8, 4), (12, 8)]),
    ]
    hip = extrude_lod22(
        Partition(hip_cells, [(0, 2, 4), (0, 3, 4), (1, 2, 4), (1, 3, 4), (2, 3, 4)]),
        np.array([1, 2, 3, 4]),
        [
            _sloped((.75, 0.), 6., hip_cells[0].exterior.coords),
            _sloped((-.75, 0.), 15., hip_cells[1].exterior.coords),
            _sloped((0., .75), 6., hip_cells[2].exterior.coords),
            _sloped((0., -.75), 12., hip_cells[3].exterior.coords),
        ],
        hip_footprint,
        0.,
    )
    for vertices, faces, _roles in (gable, hip):
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        assert mesh.is_watertight
        assert mesh.euler_number == 2
