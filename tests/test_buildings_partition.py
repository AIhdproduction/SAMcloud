import numpy as np
from shapely.geometry import LineString, Polygon

from samcloud.buildings.partition import partition_footprint, plane_lines
from samcloud.drone.contracts import RoofPlane


def _plane(normal, d, outline):
    normal = np.asarray(normal, dtype=float)
    return RoofPlane(normal / np.linalg.norm(normal), d / np.linalg.norm(normal), np.array([], dtype=int), 0., outline=Polygon(outline))


def test_gable_partition_has_two_cells_and_symmetric_adjacency():
    footprint = Polygon([(0, 0), (10, 0), (10, 8), (0, 8)])
    planes = [_plane([-.6, 0, 1], -6, [(0, 0), (5, 0), (5, 8), (0, 8)]), _plane([.6, 0, 1], -12, [(5, 0), (10, 0), (10, 8), (5, 8)])]
    result = partition_footprint(footprint, plane_lines(planes, footprint))
    assert len(result.cells) == 2
    assert {(a, b) for a, b, _ in result.adjacency} == {(0, 1), (1, 0)}


def test_l_shape_partition_drops_slivers():
    footprint = Polygon([(0, 0), (8, 0), (8, 4), (4, 4), (4, 8), (0, 8)])
    result = partition_footprint(footprint, [LineString([(2, 0), (2, 8)]), LineString([(0, 3), (8, 3)])], min_cell_area=.5)
    assert len(result.cells) >= 3
    assert min(cell.area for cell in result.cells) >= .5


def test_height_jump_creates_boundary_line():
    footprint = Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])
    planes = [_plane([0, 0, 1], -6, [(0, 0), (5, 0), (5, 6), (0, 6)]), _plane([0, 0, 1], -7.5, [(5, 0), (10, 0), (10, 6), (5, 6)])]
    assert plane_lines(planes, footprint)
