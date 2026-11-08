"""Depth-Anything multi-view fill: only confirmed prior pixels become points."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from test_plane_fill import HOLE, SIZE, WALL_Y, _depth_for, _views, _wall_points

from samcloud.colmap.depth_fill import fill_from_depth_prior, neighbor_views
from samcloud.colmap.plane_fill import write_colmap_array

WINDOW = 5


def _inside_hole(xyz: np.ndarray, inset: float = 0.05) -> np.ndarray:
    return (
        (xyz[:, 0] > HOLE[0] + inset) & (xyz[:, 0] < HOLE[1] - inset)
        & (xyz[:, 2] > HOLE[2] + inset) & (xyz[:, 2] < HOLE[3] - inset)
    )


def _scene(tmp_path: Path, *, behind_in_neighbours: bool = False):
    """COLMAP measured nothing on the textureless patch; the prior sees the wall everywhere.

    With ``behind_in_neighbours`` the patch is really an opening: every view
    except the first reliably measures the room behind it.
    """
    stereo = tmp_path / "stereo"
    (stereo / "depth_maps").mkdir(parents=True)
    views = _views()
    priors = {}
    for view in views:
        photometric, geometric = _depth_for(view, None)
        priors[view.name] = photometric  # wall depth everywhere, 0.3 % noise
        if behind_in_neighbours and view.index > 0:
            _photometric, geometric = _depth_for(view, 0.6)
        write_colmap_array(stereo / "depth_maps" / f"{view.name}.geometric.bin", geometric)
    existing = _wall_points(np.random.default_rng(7))
    return stereo, views, priors, existing


SETTINGS = {"max_neighbor_distance_m": 5.0, "max_view_angle_deg": 60.0, "spacing_m": 0.01}


def test_prior_pixels_confirmed_by_other_views_fill_the_textureless_patch(tmp_path: Path) -> None:
    stereo, views, priors, existing = _scene(tmp_path)

    xyz, rgb, cls, votes, report = fill_from_depth_prior(views, stereo, priors.get, existing, SETTINGS)

    inside = _inside_hole(xyz)
    assert np.count_nonzero(inside) > 500
    assert np.all(np.abs(xyz[:, 1] - WALL_Y) < 0.05)  # on the wall, nowhere else
    assert np.all(votes >= 2) and report["added_points"] == len(xyz)
    # measured voxels are never duplicated
    assert not np.any(np.all(np.abs(xyz[:, None, :] - existing[None, :200, :]) < 1e-9, axis=2))


def test_points_are_not_laid_on_a_regular_grid(tmp_path: Path) -> None:
    stereo, views, priors, existing = _scene(tmp_path)

    xyz, *_ = fill_from_depth_prior(views, stereo, priors.get, existing, SETTINGS)

    fractional = np.mod(xyz[_inside_hole(xyz)][:, 0], 0.01)
    assert np.std(fractional) > 0.002  # positions follow pixels, not a 1 cm lattice


def test_an_opening_seen_through_by_other_views_stays_open(tmp_path: Path) -> None:
    stereo, views, priors, existing = _scene(tmp_path, behind_in_neighbours=True)

    xyz, _rgb, _cls, _votes, report = fill_from_depth_prior(views, stereo, priors.get, existing, SETTINGS)

    assert not np.any(_inside_hole(xyz))
    assert report["rejected_free_space"] > 0


def test_window_pixels_are_never_used(tmp_path: Path) -> None:
    from test_plane_fill import _labels_for

    stereo, views, priors, existing = _scene(tmp_path)
    labels = {view.name: _labels_for(view, WINDOW) for view in views}

    xyz, _rgb, cls, _votes, _report = fill_from_depth_prior(
        views, stereo, priors.get, existing, SETTINGS, label_reader=labels.get, exclude_label_ids=(WINDOW,),
    )

    assert not np.any(_inside_hole(xyz))
    assert not np.any(cls == WINDOW)


def test_classes_come_from_the_reference_sam_labels(tmp_path: Path) -> None:
    from test_plane_fill import WALL, _labels_for

    stereo, views, priors, existing = _scene(tmp_path)
    labels = {view.name: _labels_for(view, WALL) for view in views}

    _xyz, _rgb, cls, _votes, _report = fill_from_depth_prior(
        views, stereo, priors.get, existing, SETTINGS, label_reader=labels.get,
    )

    assert len(cls) and np.all(cls == WALL)


def test_neighbours_look_in_a_similar_direction() -> None:
    views = _views()
    turned = views[3]
    turned_rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]) @ turned.rotation  # 90 deg yaw
    views[3] = type(turned)(turned.index, turned.name, turned_rotation, -turned_rotation @ turned.center,
                            turned.fx, turned.fy, turned.cx, turned.cy, SIZE, SIZE)

    neighbours = neighbor_views(views, {"neighbors": 8, "max_view_angle_deg": 60.0, "max_neighbor_distance_m": 5.0})

    assert 3 not in neighbours[0] and neighbours[3] == []
    assert neighbours[0] == [1, 2]


def test_grazing_surfaces_are_skipped() -> None:
    from samcloud.colmap.depth_fill import _incidence_mask

    view = _views()[0]
    v, u = np.mgrid[0:SIZE, 0:SIZE].astype(float) + 0.5
    xn = (u - view.cx) / view.fx
    facing = np.full((SIZE, SIZE), 2.0, dtype=np.float32)  # plane facing the camera
    # plane z = 1 + 6 x in camera coordinates: seen at ~80 deg near the image centre
    denominator = 1.0 - 6.0 * xn
    grazing = np.where(denominator > 0.05, 1.0 / np.maximum(denominator, 0.05), 0.0).astype(np.float32)
    centre = (np.abs(xn) < 0.04) & (np.abs(v - view.cy) < 40)

    assert _incidence_mask(facing, view, 75.0)[10:-10, 10:-10].mean() > 0.95
    assert _incidence_mask(grazing, view, 75.0)[centre].mean() < 0.1


@pytest.mark.parametrize("behind", [False, True])
def test_gpu_and_numpy_paths_give_the_same_points(tmp_path: Path, behind: bool) -> None:
    pytest.importorskip("torch")
    stereo, views, priors, existing = _scene(tmp_path, behind_in_neighbours=behind)
    results = {}
    for device in ("numpy", "cpu"):  # "cpu" runs the torch implementation (as on a GPU)
        xyz, _rgb, cls, votes, report = fill_from_depth_prior(views, stereo, priors.get, existing,
                                                              {**SETTINGS, "device": device})
        results[device] = (xyz, cls, votes, report)
    a, b = results["numpy"], results["cpu"]
    assert abs(len(a[0]) - len(b[0])) <= max(2, len(a[0]) // 1000)
    assert abs(a[3]["accepted"] - b[3]["accepted"]) <= max(2, a[3]["accepted"] // 1000)
    if len(a[0]) and len(b[0]):
        assert np.allclose(a[0].mean(axis=0), b[0].mean(axis=0), atol=1e-3)
