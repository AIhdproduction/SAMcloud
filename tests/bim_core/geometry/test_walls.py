"""Wall geometry: justification, join cuts, footprints, quantities."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from samcloud.bim.core.geometry import compute_scene_geometry, compute_wall_geometries, face_offsets
from samcloud.bim.core.model import JoinEnd, JoinType, WallJustification

from .conftest import make_join, make_wall, room_scene


def _poly(footprint: tuple[tuple[float, float], ...]) -> Polygon:
    return Polygon(footprint)


def test_face_offsets_conventions() -> None:
    assert face_offsets(WallJustification.CENTER, 0.2) == (-0.1, 0.1)
    assert face_offsets(WallJustification.LEFT, 0.2) == (-0.2, 0.0)
    assert face_offsets(WallJustification.RIGHT, 0.2) == (0.0, 0.2)


@pytest.mark.parametrize(
    ("justification", "y_min", "y_max"),
    [
        (WallJustification.CENTER, -0.1, 0.1),
        (WallJustification.LEFT, -0.2, 0.0),
        (WallJustification.RIGHT, 0.0, 0.2),
    ],
)
def test_justification_faces(justification: WallJustification, y_min: float, y_max: float) -> None:
    wall = make_wall((0.0, 0.0), (4.0, 0.0), 0.2, 2.5, justification)
    walls, _ = compute_wall_geometries([wall])
    geometry = walls[wall.id]
    assert geometry.direction == (1.0, 0.0)
    assert geometry.normal == (0.0, 1.0)
    ys = sorted(p[1] for p in geometry.footprint)
    assert ys[0] == pytest.approx(y_min, abs=1e-12)
    assert ys[-1] == pytest.approx(y_max, abs=1e-12)
    assert geometry.footprint == ((0.0, y_min), (4.0, y_min), (4.0, y_max), (0.0, y_max))
    assert _poly(geometry.footprint).exterior.is_ccw


def test_wall_without_joins_quantities() -> None:
    wall = make_wall((0.0, 0.0), (4.0, 0.0), 0.2, 2.5)
    wall.params.base_offset = 0.05
    geometry = compute_wall_geometries([wall], storey_elevations={})[0][wall.id]
    q = geometry.quantities
    assert q["Length"] == pytest.approx(4.0)
    assert q["Width"] == pytest.approx(0.2)
    assert q["Height"] == pytest.approx(2.5)
    assert q["GrossSideArea"] == pytest.approx(10.0)
    assert q["NetSideArea"] == pytest.approx(10.0)
    assert q["GrossFootprintArea"] == pytest.approx(0.8)
    assert q["GrossVolume"] == pytest.approx(2.0)
    assert q["NetVolume"] == pytest.approx(2.0)
    assert geometry.base_z == pytest.approx(0.05)
    assert geometry.top_z == pytest.approx(2.55)


def test_room_l_joins_no_overlap_no_gap() -> None:
    scene = room_scene()
    geometry = compute_scene_geometry(scene)
    footprints = [_poly(w.footprint) for w in geometry.walls.values()]
    union = unary_union(footprints)
    assert sum(f.area for f in footprints) == pytest.approx(union.area, abs=1e-9)
    assert union.area == pytest.approx(5.2 * 4.2 - 4.8 * 3.8, abs=1e-9)
    assert isinstance(union, Polygon)
    assert len(union.interiors) == 1
    outer = Polygon(union.exterior)
    inner = Polygon(union.interiors[0])
    assert outer.area == pytest.approx(5.2 * 4.2, abs=1e-9)
    assert inner.area == pytest.approx(4.8 * 3.8, abs=1e-9)
    assert outer.bounds == pytest.approx((-0.1, -0.1, 5.1, 4.1), abs=1e-9)
    assert inner.bounds == pytest.approx((0.1, 0.1, 4.9, 3.9), abs=1e-9)
    lengths = sorted(w.length for w in geometry.walls.values())
    assert lengths == pytest.approx([4.0, 4.0, 5.0, 5.0], abs=1e-9)
    assert not geometry.issues


def test_l_join_with_reversed_wall_direction_is_symmetric() -> None:
    a = make_wall((0.0, 0.0), (5.0, 0.0))
    b = make_wall((5.0, 4.0), (5.0, 0.0))  # points toward the corner: END meets END
    join = make_join(a, JoinEnd.END, b, JoinEnd.END, JoinType.L)
    walls, issues = compute_wall_geometries([a, b], [join])
    assert not issues
    union = unary_union([_poly(w.footprint) for w in walls.values()])
    assert union.area == pytest.approx(5.1 * 0.2 + 3.9 * 0.2, abs=1e-9)
    assert walls[a.id].length == pytest.approx(5.0)
    assert walls[b.id].length == pytest.approx(4.0)


def test_t_join_trims_to_near_face() -> None:
    long_wall = make_wall((0.0, 0.0), (8.0, 0.0), 0.3)
    partition = make_wall((4.0, 0.0), (4.0, 5.0), 0.12)
    join = make_join(partition, JoinEnd.START, long_wall, JoinEnd.PATH, JoinType.T)
    walls, issues = compute_wall_geometries([long_wall, partition], [join])
    assert not issues
    part = walls[partition.id]
    assert part.axis_start == pytest.approx((4.0, 0.15))
    assert part.axis_end == pytest.approx((4.0, 5.0))
    assert part.length == pytest.approx(4.85)
    assert part.footprint[0] == pytest.approx((4.0 - 0.06, 0.15)) or part.footprint[
        0
    ] == pytest.approx((4.0 + 0.06, 0.15))
    assert walls[long_wall.id].footprint == ((0.0, -0.15), (8.0, -0.15), (8.0, 0.15), (0.0, 0.15))
    union = unary_union([_poly(w.footprint) for w in walls.values()])
    assert union.area == pytest.approx(8.0 * 0.3 + 4.85 * 0.12, abs=1e-9)


def test_t_join_from_the_other_side_and_extension() -> None:
    long_wall = make_wall((0.0, 0.0), (8.0, 0.0), 0.3)
    # partition starts short of the wall on the north side, END abuts the path
    partition = make_wall((3.0, 5.0), (3.0, 0.5), 0.1)
    join = make_join(long_wall, JoinEnd.PATH, partition, JoinEnd.END, JoinType.T)
    walls, _ = compute_wall_geometries([long_wall, partition], [join])
    part = walls[partition.id]
    assert part.axis_end == pytest.approx((3.0, 0.15))
    assert part.length == pytest.approx(4.85)
    # a partition ending south of the wall is extended up to the south face
    south = make_wall((6.0, -5.0), (6.0, -0.5), 0.1)
    join2 = make_join(south, JoinEnd.END, long_wall, JoinEnd.PATH, JoinType.T)
    walls2, _ = compute_wall_geometries([long_wall, south], [join2])
    assert walls2[south.id].axis_end == pytest.approx((6.0, -0.15))


def test_join_issues_are_reported_and_ends_stay_square() -> None:
    a = make_wall((0.0, 0.0), (4.0, 0.0))
    b = make_wall((4.0, 0.0), (8.0, 0.0))  # collinear
    c = make_wall((4.0, 0.0), (4.0, 3.0))
    bad_parallel = make_join(a, JoinEnd.END, b, JoinEnd.START, JoinType.L)
    bad_path = make_join(a, JoinEnd.PATH, c, JoinEnd.START, JoinType.L)
    first = make_join(a, JoinEnd.END, c, JoinEnd.START, JoinType.L)
    second = make_join(a, JoinEnd.END, b, JoinEnd.PATH, JoinType.T)
    unknown = make_join(
        a, JoinEnd.END, make_wall((9.0, 9.0), (9.0, 10.0)), JoinEnd.START, JoinType.L
    )
    walls, issues = compute_wall_geometries(
        [a, b, c], [bad_parallel, bad_path, first, second, unknown]
    )
    assert "parallel" in issues[bad_parallel.id][0]
    assert "start or end" in issues[bad_path.id][0]
    assert first.id not in issues
    assert "already cut" in issues[second.id][0]
    assert "unknown wall" in issues[unknown.id][0]
    assert walls[b.id].footprint[0] == (4.0, -0.1)


def test_acute_corner_mitre_has_no_overlap() -> None:
    a = make_wall((0.0, 0.0), (6.0, 0.0), 0.2)
    b = make_wall((0.0, 0.0), (3.0, 3.0), 0.2)
    join = make_join(a, JoinEnd.START, b, JoinEnd.START, JoinType.L)
    walls, issues = compute_wall_geometries([a, b], [join])
    assert not issues
    pa, pb = (_poly(w.footprint) for w in walls.values())
    assert pa.intersection(pb).area == pytest.approx(0.0, abs=1e-12)
    assert pa.union(pb).area == pytest.approx(pa.area + pb.area, abs=1e-12)
    assert walls[a.id].length == pytest.approx(6.0)
    assert walls[b.id].length == pytest.approx(3.0 * math.sqrt(2.0))
