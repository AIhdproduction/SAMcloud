"""Signed deviation, coverage, status, derived storage and tolerance colors."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from numpy.typing import NDArray

from samcloud.bim.core.geometry import SceneGeometry, compute_scene_geometry
from samcloud.bim.core.model import CeilingElement, Metrics, Scene, SlabElement, Status, WallElement
from samcloud.bim.core.model.graph import recompute
from samcloud.bim.core.qa import (
    DeviationSettings,
    ToleranceColors,
    apply_metrics,
    compute_deviation,
    element_faces,
    face_coverage,
    load_tolerances,
    signed_distances,
    status_from_metrics,
    tolerance_colors,
)
from samcloud.bim.core.synth.room import build_room_scene

TLS = load_tolerances(profile="tls")
SETTINGS = DeviationSettings(coverage_radius=TLS.coverage_radius, coverage_min=TLS.coverage_min)


@pytest.fixture(scope="module")
def scene() -> Scene:
    return build_room_scene()


@pytest.fixture(scope="module")
def geometry(scene: Scene) -> SceneGeometry:
    return compute_scene_geometry(scene)


def _wall(scene: Scene, name: str) -> str:
    return str(next(e.id for e in scene.elements.values() if e.name == name))


def _first(scene: Scene, cls: type[SlabElement]) -> str:
    return str(next(e.id for e in scene.elements.values() if isinstance(e, cls)))


def _south_left_points(offsets: NDArray[np.float64], x0: float = 0.3) -> NDArray[np.float64]:
    """Points over the left (room side) face of the south wall (y = 0.15), left of the window."""
    n = len(offsets)
    x = np.linspace(x0, 1.8, n)
    z = np.linspace(0.2, 0.8, n)
    return np.column_stack((x, 0.15 + offsets, z))


def test_sign_convention_on_a_wall(scene: Scene, geometry: SceneGeometry) -> None:
    wall_id = _wall(scene, "Exterior wall south")
    faces = element_faces(geometry, wall_id)
    assert len(faces) == 2
    normals = {face.label: tuple(face.normal) for face in faces}
    assert normals["wall_left"] == pytest.approx((0.0, 1.0, 0.0))  # toward the room
    assert normals["wall_right"] == pytest.approx((0.0, -1.0, 0.0))  # toward the outside

    pts = np.array(
        [
            [1.0, 0.16, 1.0],  # 1 cm in front of the room side face
            [1.0, 0.14, 1.0],  # 1 cm behind it (inside the wall body)
            [1.0, -0.17, 1.0],  # 2 cm in front of the outside face
            [1.0, -0.14, 1.0],  # 1 cm behind the outside face
        ]
    )
    signed, nearest = signed_distances(pts, faces)
    assert signed == pytest.approx([0.01, -0.01, 0.02, -0.01])
    assert [faces[i].label for i in nearest] == [
        "wall_left",
        "wall_left",
        "wall_right",
        "wall_right",
    ]


def test_points_in_a_window_opening_are_far_from_the_wall(
    scene: Scene, geometry: SceneGeometry
) -> None:
    faces = element_faces(geometry, _wall(scene, "Exterior wall south"))
    inside_window = np.array([[2.5, 0.15, 1.6]])  # on the face plane, inside the opening
    signed, _ = signed_distances(inside_window, faces)
    # window cut: s 1.9..3.1, z 0.9..2.3; the nearest cut edge is 0.6 m away in plan
    assert abs(signed[0]) == pytest.approx(0.6, abs=1e-9)


def test_rms_p95_max_on_known_offsets(scene: Scene, geometry: SceneGeometry) -> None:
    offsets = np.arange(1, 101) * 0.0005 * np.where(np.arange(100) % 2 == 0, 1.0, -1.0)
    pts = _south_left_points(offsets)
    result = compute_deviation(
        geometry, _wall(scene, "Exterior wall south"), pts, TLS.tolerance_for("wall"), SETTINGS
    )
    assert result.signed_distance == pytest.approx(offsets, abs=1e-12)
    metrics = result.metrics
    assert metrics.point_count == 100
    assert metrics.deviation_max == pytest.approx(0.05)
    assert metrics.deviation_rms == pytest.approx(float(np.sqrt(np.mean(offsets**2))))
    assert metrics.deviation_p95 == pytest.approx(float(np.quantile(np.abs(offsets), 0.95)))
    assert metrics.deviation_p95 == pytest.approx(0.04753, abs=1e-5)


def test_zero_points_give_no_deviation_and_a_warning(scene: Scene, geometry: SceneGeometry) -> None:
    result = compute_deviation(
        geometry, _wall(scene, "Exterior wall south"), np.empty((0, 3)), 0.02, SETTINGS
    )
    assert result.metrics.point_count == 0
    assert result.metrics.deviation_p95 is None
    assert result.metrics.coverage == 0.0
    assert result.status is Status.WARNING


def test_coverage_of_a_half_covered_floor(scene: Scene, geometry: SceneGeometry) -> None:
    floor_id = _first(scene, SlabElement)
    rng = np.random.default_rng(5)
    # slab outline is x -0.15..5.15, y -0.15..4.15, top at z = 0; cover x up to 2.5
    xs, ys = np.meshgrid(np.arange(-0.15, 2.5, 0.03), np.arange(-0.15, 4.16, 0.03))
    pts = np.column_stack((xs.ravel(), ys.ravel(), rng.normal(0.0, 0.001, xs.size)))
    result = compute_deviation(geometry, floor_id, pts, 0.02, SETTINGS)
    expected = (2.5 + 0.05 + 0.15) / 5.3
    coverage = result.metrics.coverage
    p95 = result.metrics.deviation_p95
    assert coverage is not None and p95 is not None
    assert coverage == pytest.approx(expected, abs=0.03)
    assert 0.45 < coverage < 0.56
    assert p95 < 0.003
    assert result.status is Status.OK


def test_half_covered_wall_face_and_max_over_faces(scene: Scene, geometry: SceneGeometry) -> None:
    wall_id = _wall(scene, "Exterior wall north")
    faces = element_faces(geometry, wall_id)
    left = next(f for f in faces if f.label == "wall_left")
    right = next(f for f in faces if f.label == "wall_right")
    # points on the outer (right) face of the north wall over its full length and height
    xs, zs = np.meshgrid(np.arange(-0.1, 5.1, 0.04), np.arange(0.0, 2.8, 0.04))
    y = right.origin[1]
    pts = np.column_stack((xs.ravel(), np.full(xs.size, y), zs.ravel()))
    assert face_coverage(right, pts, 0.05) > 0.95
    assert face_coverage(left, pts, 0.05) == 0.0
    result = compute_deviation(geometry, wall_id, pts, 0.02, SETTINGS)
    assert result.coverage_per_face[faces.index(right)] > 0.95
    assert result.metrics.coverage == pytest.approx(max(result.coverage_per_face))


def test_tilted_ceiling_sign_and_normal(scene: Scene) -> None:
    scene = scene.model_copy(deep=True)
    ceiling = next(e for e in scene.elements.values() if isinstance(e, CeilingElement))
    ceiling.params.plane_normal = (0.0, 0.05, 1.0)
    ceiling.params.plane_offset = 2.5
    geo = compute_scene_geometry(scene)
    ceiling_geo = geo.ceilings[ceiling.id]
    faces = element_faces(geo, ceiling.id)
    assert len(faces) == 1
    assert faces[0].normal[2] < 0
    tilt = np.array([0.0, 0.05, 1.0]) / np.linalg.norm([0.05, 1.0])
    assert faces[0].normal == pytest.approx(-tilt)

    xy = np.array([[1.0, 1.0], [2.0, 3.0], [4.0, 2.0]])
    z_plane = np.array([ceiling_geo.bottom_z_at(x, y) for x, y in xy])
    assert not np.allclose(z_plane, z_plane[0])
    below = np.column_stack((xy, z_plane - 0.01))  # room side: in front
    above = np.column_stack((xy, z_plane + 0.01))  # inside the slab: behind
    signed_below, _ = signed_distances(below, faces)
    signed_above, _ = signed_distances(above, faces)
    cos_tilt = float(np.cos(np.arctan(0.05)))
    assert signed_below == pytest.approx(0.01 * cos_tilt, abs=1e-12)
    assert signed_above == pytest.approx(-0.01 * cos_tilt, abs=1e-12)


def test_observer_flips_the_sign(scene: Scene, geometry: SceneGeometry) -> None:
    slab_id = next(iter(geometry.slabs))
    pts = np.array([[2.0, 2.0, 0.01]])
    above, _ = signed_distances(pts, element_faces(geometry, slab_id))
    below, _ = signed_distances(pts, element_faces(geometry, slab_id, observer=(2.0, 2.0, -3.0)))
    assert above[0] == pytest.approx(0.01)
    assert below[0] == pytest.approx(-0.01)


def test_window_and_door_use_the_opening_faces(scene: Scene, geometry: SceneGeometry) -> None:
    window_id = _wall(scene, "Window window_south")
    opening = geometry.openings[geometry.windows[window_id].opening_id]
    u, v, w = (np.array(a) for a in (opening.u_axis, opening.v_axis, opening.w_axis))
    center = np.array(opening.origin) + u * opening.width / 2 + v * opening.height / 2
    pts = np.array([center - w * 0.03, center + w * (opening.depth + 0.02)])
    signed, _ = signed_distances(pts, element_faces(geometry, window_id))
    assert signed == pytest.approx([0.03, 0.02])
    result = compute_deviation(geometry, window_id, pts, TLS.tolerance_for("window"), SETTINGS)
    assert result.metrics.deviation_max == pytest.approx(0.03)
    assert len(element_faces(geometry, _wall(scene, "Door door_west"))) == 2


def test_element_without_geometry_raises(geometry: SceneGeometry) -> None:
    with pytest.raises(KeyError):
        element_faces(geometry, "missing")


@pytest.mark.parametrize(
    ("p95", "coverage", "expected"),
    [
        (0.0, 1.0, Status.OK),
        (0.02, 0.3, Status.OK),  # both boundaries inclusive
        (0.0201, 0.9, Status.WARNING),
        (0.04, 0.9, Status.WARNING),  # exactly 2 x tolerance
        (0.0401, 0.9, Status.FAIL),
        (0.01, 0.2999, Status.WARNING),  # coverage too low
        (0.04, 0.1, Status.WARNING),
        (0.05, 0.1, Status.FAIL),  # above 2 x tolerance, coverage irrelevant
    ],
)
def test_status_rules_at_the_boundaries(p95: float, coverage: float, expected: Status) -> None:
    metrics = Metrics(deviation_p95=p95, coverage=coverage, point_count=10)
    assert status_from_metrics(metrics, 0.02, 0.3) is expected


def test_status_unsure_only_by_flags_and_missing_p95() -> None:
    metrics = Metrics(deviation_p95=0.001, coverage=1.0, point_count=10)
    assert status_from_metrics(metrics, 0.02, 0.3) is Status.OK
    assert status_from_metrics(metrics, 0.02, 0.3, ["ai answered unsure"]) is Status.UNSURE
    assert status_from_metrics(Metrics(), 0.02, 0.3) is Status.WARNING


def test_apply_metrics_writes_only_derived_values(scene: Scene) -> None:
    scene = scene.model_copy(deep=True)
    geo = compute_scene_geometry(scene)
    wall = next(e for e in scene.elements.values() if isinstance(e, WallElement))
    params_before = wall.params.model_dump()
    origin_before = dict(wall.param_origin)
    result = compute_deviation(geo, wall.id, _south_left_points(np.full(50, 0.004)), 0.02, SETTINGS)
    recompute(scene)
    apply_metrics(scene, [result])

    entry = scene.derived["elements"][wall.id]
    assert entry["status"] == result.status.value
    assert entry["metrics"]["deviation_max"] == pytest.approx(0.004)
    assert entry["metrics"]["point_count"] == 50
    assert entry["tolerance"] == 0.02
    assert wall.params.model_dump() == params_before
    assert wall.param_origin == origin_before
    assert wall.metrics == Metrics()

    recompute(scene, [wall.id])  # recompute keeps the foreign keys
    entry = scene.derived["elements"][wall.id]
    assert entry["status"] == result.status.value
    assert "metrics" in entry and "quantities" in entry and "geometry" in entry

    unknown = dataclasses.replace(result, element_id="missing")
    with pytest.raises(KeyError):
        apply_metrics(scene, {"missing": unknown})


def test_tolerance_colors() -> None:
    colors = ToleranceColors()
    distances = np.array([0.0, 0.02, -0.02, 0.0201, -0.0201, 0.5, -0.5, np.nan])
    rgb = tolerance_colors(distances, 0.02, colors)
    assert rgb.dtype == np.uint8
    assert rgb.shape == (8, 3)
    within, front, behind = colors.within, colors.front, colors.behind
    assert [tuple(row) for row in rgb[:3]] == [within] * 3
    assert [tuple(row) for row in rgb[3:6:2]] == [front] * 2
    assert [tuple(row) for row in rgb[4:7:2]] == [behind] * 2
    assert tuple(rgb[7]) == (128, 128, 128)
    custom = ToleranceColors(within=(0, 255, 0), front=(255, 0, 0), behind=(0, 0, 255))
    rgb = tolerance_colors(np.array([0.0, 0.1, -0.1]), 0.02, custom)
    assert rgb.tolist() == [[0, 255, 0], [255, 0, 0], [0, 0, 255]]
    assert (
        tolerance_colors(distances, 0.02).tolist()
        == tolerance_colors(distances, 0.02, ToleranceColors()).tolist()
    )
