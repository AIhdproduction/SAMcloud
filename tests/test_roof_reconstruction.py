"""DSM-driven roof reconstruction: planes, partition labelling, plausibility."""

from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import Polygon

from samcloud.buildings.extrude import extrude_lod22
from samcloud.buildings.partition import partition_from_raster
from samcloud.buildings.reconstruct import _plausibility, reconstruct_building
from samcloud.buildings.roof_planes import detect_raster_roof_planes
from samcloud.buildings.validate import roof_height_rmse
from samcloud.drone.contracts import BUILDING, PointSet, load_drone_config

RESOLUTION = 0.1
GROUND = 0.0


def _sampler(x, y=None):
    if y is None:
        return np.zeros(len(np.atleast_2d(np.asarray(x, dtype=float))))
    return GROUND


def _walls(footprint: Polygon, top: float, rng) -> PointSet:
    coordinates = list(footprint.exterior.coords)
    points, normals = [], []
    for start, end in zip(coordinates[:-1], coordinates[1:], strict=False):
        start, end = np.asarray(start), np.asarray(end)
        length = float(np.linalg.norm(end - start))
        t, z = np.meshgrid(np.linspace(0.02, 0.98, max(10, int(length * 8))), np.linspace(0.1, top - 0.2, 24))
        xy = start + t.ravel()[:, None] * (end - start)
        points.append(np.c_[xy, z.ravel()] + rng.normal(0, 0.01, (len(xy), 3)))
        edge = end - start
        normal = np.array((-edge[1], edge[0], 0.0))
        normal /= np.linalg.norm(normal)
        normals.append(np.tile(normal, (len(xy), 1)))
    xyz = np.vstack(points)
    return PointSet(xyz, np.full(len(xyz), BUILDING, dtype=np.uint16), normals=np.vstack(normals))


def _raster(footprint: Polygon, surface, rng, noise=0.03) -> np.ndarray:
    minx, miny, maxx, maxy = footprint.bounds
    x, y = np.meshgrid(np.arange(minx + RESOLUTION / 2, maxx, RESOLUTION), np.arange(miny + RESOLUTION / 2, maxy, RESOLUTION))
    x, y = x.ravel(), y.ravel()
    inside = np.array([footprint.covers(Polygon([(a, b), (a + 1e-6, b), (a, b + 1e-6)]).centroid) for a, b in zip(x, y, strict=True)])
    x, y = x[inside], y[inside]
    return np.c_[x, y, surface(x, y) + rng.normal(0, noise, len(x))]


def _gable(x, y):
    return 5.0 + 0.5 * np.minimum(x, 12.0 - x)


def _hip(x, y):
    return 5.0 + 0.5 * np.minimum.reduce([x, 12.0 - x, y, 8.0 - y])


L_FOOTPRINT = Polygon([(0, 0), (14, 0), (14, 6), (6, 6), (6, 12), (0, 12)])


def _l_gables(x, y):
    main = 5.0 + 0.5 * (3.0 - np.abs(y - 3.0))
    wing = 5.0 + 0.5 * (3.0 - np.abs(x - 3.0))
    in_wing = (x <= 6.0) & (y >= 6.0)
    return np.where(in_wing, np.maximum(wing, np.where(y <= 6.0, main, -np.inf)), np.where(y <= 6.0, np.where(x <= 6.0, np.maximum(main, wing), main), main))


def _reconstruct(footprint, surface, seed, noise=0.03, blob=None):
    rng = np.random.default_rng(seed)
    raster = _raster(footprint, surface, rng, noise)
    if blob is not None:
        centre, size, height = blob
        mask = (np.abs(raster[:, 0] - centre[0]) < size / 2) & (np.abs(raster[:, 1] - centre[1]) < size / 2)
        raster[mask, 2] += height
    cloud = _walls(footprint, float(raster[:, 2].max()), rng)
    model = reconstruct_building(
        cloud, 1, dtm_sampler=_sampler, config=load_drone_config(), dsm_roof_points=raster, dsm_cell_area_m2=RESOLUTION * RESOLUTION
    )
    return model, raster


def _roof_faces(model):
    mesh = trimesh.Trimesh(model.vertices, model.faces, process=False)
    return mesh, np.flatnonzero(mesh.face_normals[:, 2] > 0.1)


def test_gable_with_chimney_gives_two_planes_horizontal_ridge_low_rmse():
    footprint = Polygon([(0, 0), (12, 0), (12, 8), (0, 8)])
    model, raster = _reconstruct(footprint, _gable, 11, blob=((3.0, 4.0), 0.4, 0.8))
    assert model.status == "ok", model.reason
    planes = model.heights["roof_planes"]
    assert len(planes) == 2
    assert all(abs(item["normal"][1]) < 0.02 for item in planes)
    mesh, roof = _roof_faces(model)
    assert mesh.is_watertight
    vertices = np.asarray(model.vertices)
    ridge = vertices[np.abs(vertices[:, 0] - 6.0) < 0.05]
    ridge = ridge[ridge[:, 2] > 7.0]
    assert len(ridge) >= 2 and np.ptp(ridge[:, 2]) < 0.03
    clean = raster[~((np.abs(raster[:, 0] - 3.0) < 0.5) & (np.abs(raster[:, 1] - 4.0) < 0.5))]
    assert roof_height_rmse(vertices, np.asarray(model.faces)[roof], clean, footprint) <= 0.05


def test_hip_roof_has_four_planes():
    footprint = Polygon([(0, 0), (12, 0), (12, 8), (0, 8)])
    rng = np.random.default_rng(5)
    planes = detect_raster_roof_planes(_raster(footprint, _hip, rng), min_plane_area_m2=4.0, inlier_tolerance_m=0.08, cell_area_m2=RESOLUTION**2)
    assert len(planes) == 4
    model, _raster_points = _reconstruct(footprint, _hip, 5)
    assert model.status == "ok", model.reason
    assert len(model.heights["roof_planes"]) == 4
    assert trimesh.Trimesh(model.vertices, model.faces, process=False).is_watertight


def test_l_shaped_house_with_two_gables_is_watertight():
    model, _raster_points = _reconstruct(L_FOOTPRINT, _l_gables, 7)
    assert model.status == "ok", model.reason
    assert len(model.heights["roof_planes"]) >= 4
    assert trimesh.Trimesh(model.vertices, model.faces, process=False).is_watertight
    assert model.heights["partition_cells"] >= 4


def test_flat_roof_with_parapet_noise_is_one_flat_plane():
    footprint = Polygon([(0, 0), (10, 0), (10, 7), (0, 7)])

    def flat(x, y):
        rim = (np.minimum.reduce([x, 10 - x, y, 7 - y]) < 0.15) * 0.3
        return 6.0 + rim

    model, _raster_points = _reconstruct(footprint, flat, 3)
    assert model.status == "ok", model.reason
    planes = model.heights["roof_planes"]
    assert len(planes) == 1
    assert planes[0]["normal"][2] > 0.999


def test_narrow_tall_strip_is_rejected_by_plausibility():
    strip = Polygon([(0, 0), (8, 0), (8, 1.2), (0, 1.2)])
    config = load_drone_config()["buildings"]["plausibility"]
    raster = _raster(strip, lambda x, y: np.full_like(x, 3.0), np.random.default_rng(2))
    decision = _plausibility(strip, raster, GROUND, config, cut_by_boundary=False)
    assert decision["decision"] == "rejected"
    assert "minimum width" in decision["reason"]
    model, _unused = _reconstruct(strip, lambda x, y: np.full_like(x, 3.0), 2)
    assert model.status == "failed"
    assert model.reason.startswith("plausibility rejected")


def test_cell_heights_come_from_labelled_planes_never_footprint_corners():
    footprint = Polygon([(0, 0), (12, 0), (12, 8), (0, 8)])
    raster = _raster(footprint, _gable, np.random.default_rng(13))
    planes = detect_raster_roof_planes(raster, min_plane_area_m2=4.0, inlier_tolerance_m=0.08, cell_area_m2=RESOLUTION**2)
    partition, labels = partition_from_raster(footprint, raster, planes, resolution=RESOLUTION)
    assert len(partition.cells) >= 2 and (labels > 0).all()
    vertices, faces, roles = extrude_lod22(partition, labels, planes, partition.footprint or footprint, GROUND)
    roof = np.flatnonzero(np.asarray(roles) == 2)
    assert len(roof)
    checked = 0
    for face in faces[roof]:
        triangle = vertices[face]
        centre = triangle[:, :2].mean(axis=0)
        owner = next(index for index, cell in enumerate(partition.cells) if cell.buffer(1e-6).contains(Polygon(triangle[:, :2]).representative_point()))
        plane = planes[int(labels[owner]) - 1]
        expected = -(triangle[:, :2] @ np.asarray(plane.normal[:2]) + plane.d) / plane.normal[2]
        assert np.abs(triangle[:, 2] - expected).max() < 0.01, (centre, triangle[:, 2], expected)
        checked += 1
    assert checked == len(roof)


def test_noisy_gable_is_two_planar_polygons_and_one_ridge_with_few_faces():
    footprint = Polygon([(0, 0), (12, 0), (12, 8), (0, 8)])
    model, _raster_points = _reconstruct(footprint, _gable, 21)
    assert model.status == "ok", model.reason
    vertices, faces = np.asarray(model.vertices), np.asarray(model.faces)
    assert len(faces) < 40
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    assert mesh.is_watertight
    roof = faces[np.asarray(model.face_roles) == 2]
    # every roof triangle lies on one of exactly two planes, and the planes meet in one ridge
    normals = np.unique(np.round(mesh.face_normals[np.asarray(model.face_roles) == 2], 3), axis=0)
    assert len(normals) == 2
    ridge = vertices[(np.abs(vertices[:, 0] - 6.0) < 0.25) & (vertices[:, 2] > 7.5)]
    assert len(ridge) == 2 and np.ptp(ridge[:, 2]) < 0.03
    assert abs(np.ptp(ridge[:, 1]) - 8.0) < 0.1
    assert len(roof) <= 8


def test_chimney_and_dormer_are_generalised_into_two_planes_and_reported():
    footprint = Polygon([(0, 0), (12, 0), (12, 8), (0, 8)])
    rng = np.random.default_rng(31)
    raster = _raster(footprint, _gable, rng)
    chimney = (np.abs(raster[:, 0] - 3.0) < 0.5) & (np.abs(raster[:, 1] - 4.0) < 0.5)
    dormer = (np.abs(raster[:, 0] - 9.0) < 1.0) & (np.abs(raster[:, 1] - 4.0) < 0.75)
    raster[chimney, 2] += 1.5
    raster[dormer, 2] += 1.0
    cloud = _walls(footprint, float(raster[:, 2].max()), rng)
    model = reconstruct_building(cloud, 1, dtm_sampler=_sampler, config=load_drone_config(), dsm_roof_points=raster, dsm_cell_area_m2=RESOLUTION**2)
    assert model.status == "ok", model.reason
    assert len(model.heights["roof_planes"]) == 2
    assert len(model.faces) < 40
    superstructures = model.heights["superstructures"]
    assert superstructures["count"] >= 2
    assert 3.5 < superstructures["area_m2"] < 6.0
    assert model.heights["rmse_lod2"] <= 0.05
    assert model.heights["rmse_all"] > model.heights["rmse_lod2"]
    assert trimesh.Trimesh(model.vertices, model.faces, process=False).is_watertight
