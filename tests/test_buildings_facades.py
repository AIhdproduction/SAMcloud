import numpy as np
from shapely.geometry import LineString

from samcloud.buildings.facades import facade_surfaces
from samcloud.drone.contracts import BuildingModel, FacadePlane


def _gable_model():
    vertices = np.array([
        [0, 0, 0], [10, 0, 0], [10, 6, 0], [0, 6, 0],
        [0, 0, 3], [10, 0, 3], [10, 6, 3], [0, 6, 3],
        [5, 0, 5], [5, 6, 5],
    ], dtype=float)
    faces = np.array([
        [0, 1, 4], [1, 5, 4], [4, 5, 8],
        [3, 7, 2], [2, 7, 6], [7, 9, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 5], [2, 6, 5],
    ])
    return BuildingModel(7, "2.2", vertices, faces, np.ones(len(faces), dtype=np.uint8), "ok")


def _facade_planes():
    return [
        FacadePlane(np.array([0.0, -1.0, 0.0]), 0.0, np.empty(0, int), 0.0, segment=LineString([(0, 0), (10, 0)])),
        FacadePlane(np.array([1.0, 0.0, 0.0]), -10.0, np.empty(0, int), 0.0, segment=LineString([(10, 0), (10, 6)])),
        FacadePlane(np.array([0.0, 1.0, 0.0]), -6.0, np.empty(0, int), 0.0, segment=LineString([(10, 6), (0, 6)])),
        FacadePlane(np.array([-1.0, 0.0, 0.0]), 0.0, np.empty(0, int), 0.0, segment=LineString([(0, 6), (0, 0)])),
    ]


def test_facade_surfaces_keep_mesh_vertices_and_gable_polygon():
    model = _gable_model()
    surfaces = facade_surfaces(model, _facade_planes())

    assert len(surfaces) == 4
    assert all(surface.facade_plane is not None for surface in surfaces)
    assert all(abs(surface.polygon.bounds[0]) < 1e-9 for surface in surfaces)
    gables = [surface for surface in surfaces if surface.z_max == 5.0]
    assert len(gables) == 2
    assert all(surface.polygon.bounds[3] == 5.0 for surface in gables)
    for surface in surfaces:
        for u, z in np.asarray(surface.polygon.exterior.coords):
            xy = surface.origin_xy + surface.direction_xy * u
            distances = np.linalg.norm(model.vertices - np.array([xy[0], xy[1], z]), axis=1)
            assert distances.min() < 0.001
