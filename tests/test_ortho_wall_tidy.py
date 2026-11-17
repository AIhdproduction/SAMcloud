from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString

from samcloud.ortho.vector.indoor import Opening
from samcloud.ortho.vector.wall_lines import WallFace, break_faces_at_openings
from samcloud.ortho.vector.wall_tidy import tidy_wall_entities

SCALES = (1.0, 0.8)


def _wall(start, end, scale=1.0, **meta):
    return {"type": "polyline", "points": [[start[0] * scale, start[1] * scale], [end[0] * scale, end[1] * scale]], "closed": False, **meta}


def _door(x, y, width, scale=1.0, angle=0.0):
    return Opening("DOOR", (x * scale, y * scale), width * scale, angle, 0.2 * scale)


def _face(start, end, scale=1.0):
    return WallFace("line", np.empty((0, 2)), line=LineString([(start[0] * scale, start[1] * scale), (end[0] * scale, end[1] * scale)]))


def _x_span(entity):
    xs = [point[0] for point in entity["points"]]
    return min(xs), max(xs)


# 1. break_faces_at_openings -------------------------------------------------

@pytest.mark.parametrize("scale", SCALES)
def test_break_cuts_face_starting_inside_span_but_far_from_centre(scale):
    opening = _door(8.2, 11.1, 0.70, scale)  # span x 7.85 .. 8.55
    face = _face((8.37, 11.1), (10.7, 11.1), scale)  # starts 0.17 m from the centre
    result = break_faces_at_openings([face], [opening], 0.15 * scale)
    assert len(result) == 1
    xs = [point[0] for point in result[0].line.coords]
    assert min(xs) >= 8.55 * scale - 1e-9
    assert max(xs) == pytest.approx(10.7 * scale)


@pytest.mark.parametrize("scale", SCALES)
def test_break_ignores_far_and_perpendicular_faces(scale):
    opening = _door(8.2, 11.1, 0.70, scale)
    far = _face((0.0, 11.1), (5.0, 11.1), scale)
    offset = _face((7.0, 12.0), (9.5, 12.0), scale)
    perpendicular = _face((8.2, 9.0), (8.2, 13.0), scale)
    result = break_faces_at_openings([far, offset, perpendicular], [opening], 0.15 * scale)
    assert len(result) == 3
    for original, kept in zip((far, offset, perpendicular), result, strict=True):
        assert list(kept.line.coords) == list(original.line.coords)


def test_break_keeps_working_case_centred_on_face():
    opening = _door(2.0, 0.0, 0.9)
    result = break_faces_at_openings([_face((0.0, 0.0), (4.0, 0.0))], [opening], 0.15)
    spans = sorted((min(p[0] for p in face.line.coords), max(p[0] for p in face.line.coords)) for face in result)
    assert spans == [pytest.approx((0.0, 1.55)), pytest.approx((2.45, 4.0))]


# 2. merge ---------------------------------------------------------------------

@pytest.mark.parametrize("scale", SCALES)
def test_merge_overlapping_collinear_segments(scale):
    entities = [
        _wall((0.0, 0.0), (3.0, 0.0), scale, _source="a"),
        _wall((2.0, 0.005), (5.0, 0.005), scale, _source="b"),
    ]
    result, stats = tidy_wall_entities(entities, [], scale)
    assert len(result) == 1
    assert stats["walls_merged"] == 1
    low, high = _x_span(result[0])
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high == pytest.approx(5.0 * scale, abs=1e-9)
    ys = [point[1] for point in result[0]["points"]]
    assert all(0.0 <= y <= 0.005 * scale + 1e-9 for y in ys)


@pytest.mark.parametrize("scale", SCALES)
def test_merge_swallows_contained_fragment_and_keeps_longest_metadata(scale):
    entities = [
        _wall((0.0, 0.0), (4.0, 0.0), scale, _source="long"),
        _wall((1.0, 0.002), (1.08, 0.002), scale, _source="fragment"),
    ]
    result, stats = tidy_wall_entities(entities, [], scale)
    assert len(result) == 1
    assert result[0]["_source"] == "long"
    assert LineString(result[0]["points"]).length == pytest.approx(4.0 * scale, abs=1e-9)
    assert stats["walls_merged"] == 1


@pytest.mark.parametrize("scale", SCALES)
def test_parallel_wall_face_ten_centimetres_away_is_not_merged(scale):
    entities = [_wall((0.0, 0.0), (4.0, 0.0), scale), _wall((0.0, 0.10), (4.0, 0.10), scale)]
    result, stats = tidy_wall_entities(entities, [], scale)
    assert len(result) == 2
    assert stats["walls_merged"] == 0


def test_non_two_point_polylines_and_arcs_pass_through():
    polyline = {"type": "polyline", "points": [[0, 0], [1, 0], [1, 1]], "closed": False}
    arc = {"type": "arc", "center": [0.0, 0.0], "radius": 1.0, "start_angle": 0.0, "end_angle": 90.0}
    result, stats = tidy_wall_entities([polyline, arc], [], 1.0)
    assert result == [polyline, arc]
    assert stats["walls_merged"] == 0


# 3. merge never bridges openings ---------------------------------------------

@pytest.mark.parametrize("scale", SCALES)
def test_merge_never_bridges_door_opening(scale):
    entities = [_wall((0.0, 0.0), (2.0, 0.0), scale), _wall((2.9, 0.0), (5.0, 0.0), scale)]
    opening = _door(2.45, 0.0, 0.9, scale)
    result, _ = tidy_wall_entities(entities, [opening], scale)
    assert len(result) == 2
    spans = sorted(_x_span(entity) for entity in result)
    assert spans[1][0] - spans[0][1] >= 0.9 * scale - 1e-9


@pytest.mark.parametrize("scale", SCALES)
def test_overlapping_duplicate_inside_door_is_recut(scale):
    # the duplicate face starts inside the door span; merging must not close it
    entities = [_wall((2.9, 0.0), (6.0, 0.0), scale), _wall((2.5, 0.0), (5.0, 0.0), scale)]
    opening = _door(2.45, 0.0, 0.9, scale)  # span 2.0 .. 2.9
    result, stats = tidy_wall_entities(entities, [opening], scale)
    assert stats["walls_merged"] == 1
    assert len(result) == 1
    low, high = _x_span(result[0])
    assert low >= 2.9 * scale - 1e-9
    assert high == pytest.approx(6.0 * scale)


# 4. joints --------------------------------------------------------------------

def _l_corner(gap, scale):
    horizontal = _wall((-3.0, 0.0), (0.0, 0.0), scale)
    vertical = _wall((gap, gap), (gap, 3.0), scale)  # starts gap above the horizontal line, gap beside its end
    return [horizontal, vertical]


@pytest.mark.parametrize("scale", SCALES)
def test_l_corner_with_twenty_centimetre_gap_is_closed(scale):
    result, stats = tidy_wall_entities(_l_corner(0.20, scale), [], scale)
    assert stats["walls_joints_closed"] == 1
    assert stats["walls_extension_m"] == pytest.approx(0.40, abs=1e-6)
    horizontal, vertical = result
    end = np.asarray(horizontal["points"][1])
    start = np.asarray(vertical["points"][0])
    assert np.linalg.norm(end - start) < 1e-6
    assert tuple(end) == pytest.approx((0.20 * scale, 0.0), abs=1e-9)


@pytest.mark.parametrize("scale", SCALES)
def test_gap_of_forty_five_centimetres_stays_open(scale):
    entities = [_wall((-3.0, 0.0), (0.0, 0.0), scale), _wall((0.45, 0.45), (0.45, 3.0), scale)]
    result, stats = tidy_wall_entities(entities, [], scale)
    assert stats["walls_joints_closed"] == 0
    assert result == entities


@pytest.mark.parametrize("scale", SCALES)
def test_t_joint_extension_to_body_is_closed_without_moving_other_wall(scale):
    # the vertical ends 0.2 above the horizontal body (forward is -y)
    entities = [_wall((-3.0, 0.0), (3.0, 0.0), scale), _wall((0.0, 1.0), (0.0, 0.2), scale)]
    result, stats = tidy_wall_entities(entities, [], scale)
    assert stats["walls_joints_closed"] == 1
    assert result[1]["points"][1] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert result[0]["points"] == entities[0]["points"]


@pytest.mark.parametrize("scale", SCALES)
def test_joint_is_not_closed_when_extension_enters_opening_span(scale):
    entities = _l_corner(0.20, scale)
    opening = _door(0.05, 0.0, 0.70, scale)  # span x -0.30 .. 0.40 on the horizontal wall line
    result, stats = tidy_wall_entities(entities, [opening], scale)
    assert stats["walls_joints_closed"] == 0
    # the horizontal wall is cut by the door and the vertical wall is untouched
    assert _x_span(result[0])[1] <= -0.30 * scale + 1e-9
    assert result[1]["points"] == entities[1]["points"]


@pytest.mark.parametrize("scale", SCALES)
def test_parallel_segments_across_gap_are_not_joined(scale):
    entities = [_wall((0.0, 0.0), (2.0, 0.0), scale), _wall((2.2, 0.0), (4.0, 0.0), scale)]
    result, stats = tidy_wall_entities(entities, [], scale)
    assert stats["walls_joints_closed"] == 0
    assert result == entities


# 5. fragments -----------------------------------------------------------------

@pytest.mark.parametrize("scale", SCALES)
def test_fragment_covered_by_longer_wall_disappears(scale):
    entities = [_wall((0.0, 0.0), (4.0, 0.0), scale), _wall((2.0, 0.025), (2.04, 0.025), scale)]
    result, stats = tidy_wall_entities(entities, [], scale)
    assert len(result) == 1
    assert stats["walls_merged"] + stats["walls_fragments_dropped"] == 1
