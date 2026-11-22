"""Open door leaves fitted as wall faces: only the free-standing panel is removed; walls, frames and the wall a leaf is hinged at stay."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString

from samcloud.ortho.vector.door_leaf_faces import DoorLeafFaceSettings, drop_door_leaf_faces, leaf_face_mask
from samcloud.ortho.vector.raster_grid import RasterGrid
from samcloud.ortho.vector.wall_lines import WallFace

UNITS = pytest.mark.parametrize("upm", [1.0, 1.132])
STOREY = 2.6
ON = DoorLeafFaceSettings(enabled=True)


def _grid(upm: float) -> RasterGrid:
    return RasterGrid(min_x=-1.0 * upm, max_y=6.0 * upm, pixel_size=0.01 * upm, width=800, height=800)


def _paint(mask: np.ndarray, grid: RasterGrid, x0: float, y0: float, x1: float, y1: float, value: bool = True) -> None:
    r0, c0 = (int(round(v)) for v in grid.local_to_pixel(x0, y1))
    r1, c1 = (int(round(v)) for v in grid.local_to_pixel(x1, y0))
    mask[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1] = value


def _slab(rng, x0, y0, x1, y1, z0, z1, count):
    """Points (x, y, z) on a vertical slab with 5 mm noise across."""
    t = rng.random(count)
    xy = np.column_stack((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)) + rng.normal(0.0, 0.005, (count, 2))
    return np.column_stack((xy, rng.uniform(z0, z1, count)))


def _room_floor(grid: RasterGrid, upm: float) -> np.ndarray:
    """Free floor of a room y > 0 (the wall along y = 0, outside below it)."""
    free = np.zeros((grid.height, grid.width), dtype=bool)
    _paint(free, grid, -1.0 * upm, 0.15 * upm, 7.0 * upm, 6.0 * upm)
    return free


BAND = (0.30 * STOREY, 0.95 * STOREY)


@UNITS
def test_free_standing_leaf_panel_is_removed_but_walls_stay(upm):
    rng = np.random.default_rng(3)
    grid = _grid(upm)
    doors = np.zeros((grid.height, grid.width), dtype=bool)
    # wall along y = 0 with a 0.9 m doorway (x 1.0 .. 1.9); the leaf stands open 90 deg into the room at the hinge x = 1.0
    _paint(doors, grid, 0.98 * upm, 0.15 * upm, 1.02 * upm, 1.0 * upm)
    wall = np.vstack((_slab(rng, 0.0, 0.0, 1.0 * upm, 0.0, BAND[0] * upm, BAND[1] * upm, 3000), _slab(rng, 1.9 * upm, 0.0, 4.0 * upm, 0.0, BAND[0] * upm, BAND[1] * upm, 6000)))
    leaf = _slab(rng, 1.0 * upm, 0.15 * upm, 1.0 * upm, 1.0 * upm, BAND[0] * upm, 2.05 * upm, 1500)
    points = np.vstack((wall, leaf))
    faces = [
        WallFace("line", wall[:3000, :2], line=LineString([(0.0, 0.0), (1.0 * upm, 0.0)])),
        WallFace("line", wall[3000:, :2], line=LineString([(1.9 * upm, 0.0), (4.0 * upm, 0.0)])),
        WallFace("line", leaf[:, :2], line=LineString([(1.0 * upm, 0.15 * upm), (1.0 * upm, 1.0 * upm)])),
    ]
    free = _room_floor(grid, upm)
    kept, dropped = drop_door_leaf_faces(faces, points, doors, grid, STOREY * upm, upm, ON, free_space=free)
    assert len(kept) == 2 and all(abs(face.line.coords[0][1]) < 1e-9 for face in kept)
    assert len(dropped) == 1 and 0.6 <= dropped[0]["panel_m"] <= 0.9
    # disabled: nothing changes
    assert len(drop_door_leaf_faces(faces, points, doors, grid, STOREY * upm, upm, DoorLeafFaceSettings(enabled=False), free_space=free)[0]) == 3


@UNITS
def test_face_continuing_into_the_wall_keeps_its_wall_part_and_the_wall_cells(upm):
    """v12 case: the leaf face runs on past the hinge into the wall (that stretch reaches the ceiling): only the panel goes."""
    rng = np.random.default_rng(5)
    grid = _grid(upm)
    doors = np.zeros((grid.height, grid.width), dtype=bool)
    _paint(doors, grid, 0.98 * upm, 0.15 * upm, 1.02 * upm, 0.95 * upm)
    leaf = _slab(rng, 1.0 * upm, 0.15 * upm, 1.0 * upm, 0.95 * upm, BAND[0] * upm, 2.0 * upm, 1500)
    wall_end = _slab(rng, 1.0 * upm, -0.15 * upm, 1.0 * upm, 0.12 * upm, BAND[0] * upm, BAND[1] * upm, 600)  # the face's stretch inside the wall
    crossing = _slab(rng, -1.0 * upm, 0.0, 0.85 * upm, 0.0, BAND[0] * upm, BAND[1] * upm, 3000)  # the wall the door sits in (its face ends at the jamb)
    face_points = np.vstack((leaf, wall_end))
    faces = [
        WallFace("line", face_points[:, :2], line=LineString([(1.0 * upm, -0.15 * upm), (1.0 * upm, 0.95 * upm)])),
        WallFace("line", crossing[:, :2], line=LineString([(-1.0 * upm, 0.0), (0.85 * upm, 0.0)])),
    ]
    free = _room_floor(grid, upm)
    kept, dropped = drop_door_leaf_faces(faces, np.vstack((face_points, crossing)), doors, grid, STOREY * upm, upm, ON, free_space=free)
    assert len(dropped) == 1
    (x0, y0), (x1, y1) = dropped[0]["line"]
    assert min(y0, y1) >= 0.10 * upm  # the panel starts beyond the wall stretch
    rests = [face for face in kept if face.line.coords[0][0] == pytest.approx(1.0 * upm)]
    assert len(rests) == 1 and max(rests[0].line.coords[0][1], rests[0].line.coords[1][1]) <= 0.25 * upm  # the wall part stays a face
    erased = leaf_face_mask(dropped, grid, upm, kept_faces=kept, settings=ON)
    row, col = (int(round(v)) for v in grid.local_to_pixel(1.0 * upm, 0.05 * upm))
    assert not erased[row, col]  # cells at the wall are never erased
    row, col = (int(round(v)) for v in grid.local_to_pixel(1.0 * upm, 0.70 * upm))
    assert erased[row, col]  # the panel cells are


@UNITS
def test_leaf_needs_free_floor_on_both_sides(upm):
    """A door-class face with the room on one side only (a closed leaf / door panel in an outer wall) is no free-standing leaf."""
    rng = np.random.default_rng(7)
    grid = _grid(upm)
    doors = np.zeros((grid.height, grid.width), dtype=bool)
    _paint(doors, grid, 1.0 * upm, -0.02 * upm, 1.9 * upm, 0.02 * upm)
    panel = _slab(rng, 1.0 * upm, 0.0, 1.9 * upm, 0.0, BAND[0] * upm, 2.0 * upm, 1500)
    faces = [WallFace("line", panel[:, :2], line=LineString([(1.0 * upm, 0.0), (1.9 * upm, 0.0)]))]
    kept, dropped = drop_door_leaf_faces(faces, panel, doors, grid, STOREY * upm, upm, ON, free_space=_room_floor(grid, upm))
    assert len(kept) == 1 and not dropped


@UNITS
def test_panel_on_the_line_of_a_collinear_wall_is_kept(upm):
    """A leaf-like stretch lying on the extension of a real wall face (a frame / closed leaf in a wall gap) is no stub."""
    rng = np.random.default_rng(8)
    grid = _grid(upm)
    doors = np.zeros((grid.height, grid.width), dtype=bool)
    _paint(doors, grid, 2.0 * upm, 2.98 * upm, 2.9 * upm, 3.02 * upm)
    panel = _slab(rng, 2.0 * upm, 3.0 * upm, 2.9 * upm, 3.0 * upm, BAND[0] * upm, 2.0 * upm, 1500)
    wall = _slab(rng, 3.1 * upm, 3.0 * upm, 5.0 * upm, 3.0 * upm, BAND[0] * upm, BAND[1] * upm, 3000)
    faces = [
        WallFace("line", panel[:, :2], line=LineString([(2.0 * upm, 3.0 * upm), (2.9 * upm, 3.0 * upm)])),
        WallFace("line", wall[:, :2], line=LineString([(3.1 * upm, 3.0 * upm), (5.0 * upm, 3.0 * upm)])),
    ]
    free = np.ones((grid.height, grid.width), dtype=bool)
    kept, dropped = drop_door_leaf_faces(faces, np.vstack((panel, wall)), doors, grid, STOREY * upm, upm, ON, free_space=free)
    assert len(kept) == 2 and not dropped


@UNITS
def test_two_leaves_standing_in_line_are_both_removed(upm):
    """Two open leaves of neighbouring doors 17 cm apart, both free-standing: neither protects the other."""
    rng = np.random.default_rng(9)
    grid = _grid(upm)
    doors = np.zeros((grid.height, grid.width), dtype=bool)
    _paint(doors, grid, 1.62 * upm, 0.2 * upm, 1.66 * upm, 1.2 * upm)
    _paint(doors, grid, 1.79 * upm, 1.3 * upm, 1.83 * upm, 2.2 * upm)
    first = _slab(rng, 1.64 * upm, 0.2 * upm, 1.64 * upm, 1.2 * upm, BAND[0] * upm, 2.0 * upm, 1500)
    second = _slab(rng, 1.81 * upm, 1.3 * upm, 1.81 * upm, 2.2 * upm, BAND[0] * upm, 2.0 * upm, 1500)
    faces = [
        WallFace("line", first[:, :2], line=LineString([(1.64 * upm, 0.2 * upm), (1.64 * upm, 1.2 * upm)])),
        WallFace("line", second[:, :2], line=LineString([(1.81 * upm, 1.3 * upm), (1.81 * upm, 2.2 * upm)])),
    ]
    free = np.ones((grid.height, grid.width), dtype=bool)
    kept, dropped = drop_door_leaf_faces(faces, np.vstack((first, second)), doors, grid, STOREY * upm, upm, ON, free_space=free)
    assert not kept and len(dropped) == 2


@UNITS
def test_short_wall_with_a_door_frame_reaching_the_ceiling_stays(upm):
    """A 0.6 m stub between a door frame and a corner: door-class cells along it, but its points reach the ceiling."""
    rng = np.random.default_rng(4)
    grid = _grid(upm)
    doors = np.zeros((grid.height, grid.width), dtype=bool)
    _paint(doors, grid, 0.0, -0.02 * upm, 0.6 * upm, 0.02 * upm)
    stub = _slab(rng, 0.0, 0.0, 0.6 * upm, 0.0, BAND[0] * upm, BAND[1] * upm, 2000)
    faces = [WallFace("line", stub[:, :2], line=LineString([(0.0, 0.0), (0.6 * upm, 0.0)]))]
    kept, dropped = drop_door_leaf_faces(faces, stub, doors, grid, STOREY * upm, upm, ON, free_space=np.ones((grid.height, grid.width), dtype=bool))
    assert len(kept) == 1 and not dropped


@UNITS
def test_leaf_lying_open_against_a_wall_keeps_that_wall(upm):
    """The leaf swung open flat against a wall hides it up to the door head; the wall above reaches the ceiling a frame depth behind."""
    rng = np.random.default_rng(6)
    grid = _grid(upm)
    doors = np.zeros((grid.height, grid.width), dtype=bool)
    _paint(doors, grid, 2.0 * upm, 0.18 * upm, 2.9 * upm, 0.22 * upm)
    leaf = _slab(rng, 2.0 * upm, 0.20 * upm, 2.9 * upm, 0.20 * upm, BAND[0] * upm, 2.0 * upm, 1500)
    wall_above = _slab(rng, 2.0 * upm, 0.0, 2.9 * upm, 0.0, 2.05 * upm, BAND[1] * upm, 400)
    faces = [WallFace("line", leaf[:, :2], line=LineString([(2.0 * upm, 0.20 * upm), (2.9 * upm, 0.20 * upm)]))]
    kept, dropped = drop_door_leaf_faces(faces, np.vstack((leaf, wall_above)), doors, grid, STOREY * upm, upm, ON, free_space=np.ones((grid.height, grid.width), dtype=bool))
    assert len(kept) == 1 and not dropped
