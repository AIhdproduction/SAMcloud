import numpy as np
from shapely.geometry import Polygon

from samcloud.buildings.extrude import extrude_lod1
from samcloud.buildings.validate import validate_model


def test_open_mesh_is_rejected_as_not_watertight():
    footprint = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    vertices, faces, _roles = extrude_lod1(footprint, 5., 0.)
    report = validate_model(vertices, faces[:-1], np.array([[2, 2, 5.]]), footprint)
    assert report["status"] == "failed"
    assert report["reason"] == "not watertight"
