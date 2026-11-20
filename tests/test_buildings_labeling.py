import time

import numpy as np
from shapely.geometry import Polygon

from samcloud.buildings.labeling import label_cells
from samcloud.buildings.partition import Partition
from samcloud.drone.contracts import RoofPlane


def _plane(slope, intercept, support, inliers):
    normal = np.array((-slope[0], -slope[1], 1.0))
    normal /= np.linalg.norm(normal)
    return RoofPlane(normal, -intercept / np.sqrt(1 + slope[0] ** 2 + slope[1] ** 2), np.asarray(inliers), 0.01, outline=Polygon(support))


def _gable():
    cells = [Polygon([(0, 0), (5, 0), (5, 8), (0, 8)]), Polygon([(5, 0), (10, 0), (10, 8), (5, 8)])]
    points = np.array([[x, y, 6 + .6 * x] for x in np.linspace(.1, 4.9, 12) for y in np.linspace(.1, 7.9, 10)] + [[x, y, 12 - .6 * x] for x in np.linspace(5.1, 9.9, 12) for y in np.linspace(.1, 7.9, 10)])
    planes = [_plane((.6, 0), 6, list(cells[0].exterior.coords), np.arange(120)), _plane((-.6, 0), 12, list(cells[1].exterior.coords), np.arange(120, 240))]
    return Partition(cells, [(0, 1, 8), (1, 0, 8)]), planes, points


def test_gable_labels_all_cells_with_two_planes():
    partition, planes, points = _gable()
    result = label_cells(partition, planes, points, buffer=2.5, complexity_factor=.7, time_limit=5)
    assert set(result.labels) == {1, 2}
    assert not np.any(result.labels == 0)


def test_label_pruning_and_large_case_performance():
    cells, adjacency, points, planes = [], [], [], []
    for row in range(1):
        for column in range(500):
            index = row * 500 + column
            cells.append(Polygon([(column, row), (column + 1, row), (column + 1, row + 1), (column, row + 1)]))
            if column:
                adjacency.extend([(index - 1, index, 1), (index, index - 1, 1)])
            if row:
                adjacency.extend([(index - 500, index, 1), (index, index - 500, 1)])
    for label in range(20):
        xs = np.linspace(label * 25 + .1, label * 25 + .9, 20)
        support = np.array([[x, y, 5 + .02 * label] for x in xs for y in np.linspace(.1, .9, 20)])
        offset = len(points)
        points.extend(support)
        planes.append(_plane((0, 0), 5 + .02 * label, [(label * 25, 0), (label * 25 + 1, 0), (label * 25 + 1, 1), (label * 25, 1)], np.arange(offset, offset + len(support))))
    partition = Partition(cells, adjacency)
    point_array = np.asarray(points)
    started = time.perf_counter()
    pruned = label_cells(partition, planes, point_array, buffer=2.5, complexity_factor=.7, time_limit=60)
    elapsed = time.perf_counter() - started
    unpruned = label_cells(partition, planes, point_array, buffer=1e6, complexity_factor=.7, time_limit=60)
    assert elapsed < 60 and pruned.seconds < 60
    assert all(0 in item for item in pruned.candidates_per_cell)
    for cell, options in zip(cells, pruned.candidates_per_cell, strict=False):
        for label in options[1:]:
            assert planes[label - 1].outline.distance(cell) <= 2.5 + 1e-9
    assert pruned.seconds * 3 < unpruned.seconds
    assert pruned.energy <= unpruned.energy * 1.01


def test_tiny_time_limit_sets_timeout():
    partition, planes, points = _gable()
    assert label_cells(partition, planes, points, buffer=2.5, complexity_factor=.7, time_limit=.0).timed_out
