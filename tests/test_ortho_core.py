"""End-to-end behavioural tests for the true orthophoto core."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import uuid

import cv2
import numpy as np
import pycolmap
import pytest
import torch
import yaml

from ortho_synthetic import make_indoor_project, make_outdoor_project, footprint_masks, ground_truth_rgb
from samcloud.dxf.cancel import CancelToken, Cancelled
from samcloud.dxf.levels import Level
from samcloud.ortho import build_orthophoto
from samcloud.ortho.build import OrthoError, load_ortho_result, ortho_fingerprint, select_sparse_model
from samcloud.ortho.cameras import OrthoCamera, max_undistorted_radius, project_points
from samcloud.ortho.config import OrthoConfig
from samcloud.ortho.dsm import build_dsm
from samcloud.ortho.grid import grid_for_bounds
from samcloud.ortho.zbuffer import build_zbuffer


@pytest.fixture(scope="module")
def work_dir(request: pytest.FixtureRequest) -> Path:
    """Return an isolated directory for this test session."""
    configured_root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    if configured_root:
        path = Path(configured_root) / f"ortho_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True)
        return path
    return request.getfixturevalue("tmp_path_factory").mktemp("ortho")


def _config() -> OrthoConfig:
    base = OrthoConfig()
    return replace(base, pixel_size_outdoor=0.1, pixel_size_indoor=0.05, tile_size=256, blend=replace(base.blend, top_k=1, feather_px=2))


@pytest.fixture(scope="module")
def outdoor(work_dir: Path) -> tuple[object, object]:
    project = make_outdoor_project(work_dir)
    return project, build_orthophoto(project.project_dir, level=None, config=_config(), device="cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def outdoor_opencv(work_dir: Path) -> tuple[object, object]:
    project = make_outdoor_project(work_dir, opencv=True)
    return project, build_orthophoto(project.project_dir, level=None, config=_config(), device="cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def indoor(work_dir: Path) -> tuple[object, object]:
    project = make_indoor_project(work_dir)
    return project, build_orthophoto(project.project_dir, level=project.level, config=_config(), device="cuda" if torch.cuda.is_available() else "cpu")


def _rgba(result: object) -> np.ndarray:
    image = cv2.imread(str(result.ortho_path), cv2.IMREAD_UNCHANGED)
    assert image is not None
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)


def _xy_grid(grid: object) -> np.ndarray:
    rows, cols = np.indices((grid.height, grid.width))
    return grid.pixel_to_local(np.column_stack((cols.ravel(), rows.ravel()))).reshape(grid.height, grid.width, 2)


def _box_distance(xy: np.ndarray) -> np.ndarray:
    return np.maximum.reduce((2.5 - xy[..., 0], xy[..., 0] - 4.5, 2.5 - xy[..., 1], xy[..., 1] - 4.1, np.zeros(xy.shape[:2])))


def _outdoor_ground_mask(project: object, result: object) -> np.ndarray:
    masks = footprint_masks(project, result.grid)
    return (_box_distance(_xy_grid(result.grid)) >= 0.5) & ~masks["tree"]


def test_projection_matches_pycolmap() -> None:
    models = [("SIMPLE_PINHOLE", [800, 320, 240]), ("PINHOLE", [800, 810, 320, 240]), ("SIMPLE_RADIAL", [800, 320, 240, 0.01]), ("RADIAL", [800, 320, 240, 0.01, -0.002]), ("OPENCV", [800, 810, 320, 240, 0.01, -0.002, 0.001, 0.0005]), ("FULL_OPENCV", [800, 810, 320, 240, 0.01, -0.002, 0.001, 0.0005, 0.0002, 0.001, -0.0002, 0.00001])]
    points = np.array(((0.2, 0.1, 3.0), (-0.3, 0.15, 4.0), (0.1, -0.2, 2.0)), dtype=np.float64)
    for model, parameters in models:
        reference = pycolmap.Camera.create_from_model_name(1, model, 800.0, 640, 480)
        reference.params = parameters
        camera = OrthoCamera("test", None, np.eye(3), np.zeros(3), model, np.asarray(parameters, dtype=np.float64), 640, 480, np.zeros(3))
        u, v, _depth, valid = project_points(camera, torch.as_tensor(points))
        assert valid.all().item()
        assert np.max(np.abs(np.column_stack((u.numpy(), v.numpy())) - reference.img_from_cam(points))) < 1e-3


def test_rig_camera_pose_matches_pycolmap_projection(work_dir: Path) -> None:
    model = work_dir / "rig_model"
    model.mkdir()
    reconstruction = pycolmap.Reconstruction()
    camera = pycolmap.Camera(camera_id=1, model="PINHOLE", width=640, height=480, params=np.array([500.0, 500.0, 320.0, 240.0]))
    reconstruction.add_camera_with_trivial_rig(camera)
    pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.diag([1.0, -1.0, -1.0])), np.array([0.0, 0.0, 10.0]))
    reconstruction.add_image_with_trivial_frame(pycolmap.Image(name="rig.png", camera_id=1, image_id=1), pose)
    reconstruction.write(model)
    image = pycolmap.Reconstruction(model).image(1)
    matrix = image.cam_from_world().matrix()
    point = np.array([0.2, -0.1, 0.0])
    ours = OrthoCamera("rig.png", None, matrix[:, :3], matrix[:, 3], "PINHOLE", np.array([500.0, 500.0, 320.0, 240.0]), 640, 480, image.projection_center())
    u, v, _depth, valid = project_points(ours, torch.as_tensor(point[None, :], dtype=torch.float64))
    assert valid.item()
    assert np.max(np.abs(np.array((u.item(), v.item())) - image.project_point(point))) < 1e-6


def test_outdoor_ortho_matches_ground_texture(outdoor: tuple[object, object]) -> None:
    project, result = outdoor
    rgba = _rgba(result)
    ground = _outdoor_ground_mask(project, result)
    assert np.mean(rgba[ground, 3] > 0) > 0.95
    assert np.abs(rgba[ground, :3].astype(np.float64) - ground_truth_rgb(project, result.grid)[ground]).mean() < 12.0


def test_occlusion_box_does_not_smear_onto_ground(outdoor: tuple[object, object]) -> None:
    project, result = outdoor
    rgba = _rgba(result)
    masks = footprint_masks(project, result.grid)
    distance = _box_distance(_xy_grid(result.grid))
    ring = (distance >= 0.3) & (distance <= 2.0) & ~masks["tree"]
    red = (rgba[..., 0] > 150) & (rgba[..., 1] < 80) & (rgba[..., 2] < 80)
    assert np.mean(red[ring]) < 0.01
    assert np.mean(red[masks["box"]]) > 0.90


def test_distorted_opencv_photos_give_same_texture_accuracy(outdoor_opencv: tuple[object, object]) -> None:
    project, result = outdoor_opencv
    rgba = _rgba(result)
    ground = _outdoor_ground_mask(project, result)
    assert np.mean(rgba[ground, 3] > 0) > 0.95
    assert np.abs(rgba[ground, :3].astype(np.float64) - ground_truth_rgb(project, result.grid)[ground]).mean() < 12.0


def test_indoor_level_ortho_excludes_ceiling(indoor: tuple[object, object]) -> None:
    project, result = indoor
    rgba = _rgba(result)
    valid = rgba[..., 3] > 0
    blue = (rgba[..., 2] > 150) & (rgba[..., 0] < 80)
    masks = footprint_masks(project, result.grid)
    xy = _xy_grid(result.grid)
    floor = ~masks["table"] & (xy[..., 0] >= 0.3) & (xy[..., 0] <= 5.7) & (xy[..., 1] >= 0.3) & (xy[..., 1] <= 4.7)
    assert np.mean(blue[valid]) < 0.005
    assert np.abs(rgba[floor, :3].astype(np.float64) - ground_truth_rgb(project, result.grid)[floor]).mean() < 15.0
    assert float(np.nanmax(np.load(result.dsm_path))) <= project.level.ceiling_z - _config().indoor.cut_below_ceiling + 1e-3


def _floor_hole_mask(result: object, hole: tuple[float, float, float, float]) -> np.ndarray:
    """Use the hole interior so point jitter at its analytic boundary is excluded."""
    xy = _xy_grid(result.grid)
    x0, y0, x1, y1 = hole
    margin = 0.05
    return (xy[..., 0] >= x0 + margin) & (xy[..., 0] <= x1 - margin) & (xy[..., 1] >= y0 + margin) & (xy[..., 1] <= y1 - margin)


def test_indoor_surface_fill_colours_floor_hole_from_photos(work_dir: Path) -> None:
    hole = (4.2, 0.9, 5.0, 1.5)
    project = make_indoor_project(work_dir / "surface_fill_photo", floor_hole=hole)
    result = build_orthophoto(project.project_dir, level=project.level, config=_config(), device="cuda" if torch.cuda.is_available() else "cpu")
    rgba = _rgba(result)
    validity = cv2.imread(str(result.validity_path), cv2.IMREAD_UNCHANGED)
    assert validity is not None
    mask = _floor_hole_mask(result, hole)
    assert np.mean(validity[mask] == 3) >= 0.95
    assert np.mean(rgba[mask, 3] == 255) >= 0.95
    assert np.abs(rgba[mask, :3].astype(np.float64) - ground_truth_rgb(project, result.grid)[mask]).mean() < 15.0


def test_indoor_surface_fill_disabled_keeps_hole(work_dir: Path) -> None:
    hole = (4.2, 0.9, 5.0, 1.5)
    project = make_indoor_project(work_dir / "surface_fill_disabled", floor_hole=hole)
    base = _config()
    config = replace(base, dsm=replace(base.dsm, surface_fill_indoor=False))
    result = build_orthophoto(project.project_dir, level=project.level, config=config, device="cuda" if torch.cuda.is_available() else "cpu")
    rgba = _rgba(result)
    assert np.mean(rgba[_floor_hole_mask(result, hole), 3] == 0) >= 0.95


def test_outdoor_unaffected_by_surface_fill(outdoor: tuple[object, object]) -> None:
    _project, result = outdoor
    validity = cv2.imread(str(result.validity_path), cv2.IMREAD_UNCHANGED)
    assert validity is not None
    assert not np.any(validity == 3)


def test_large_unscanned_area_not_filled(work_dir: Path) -> None:
    hole = (4.2, 0.9, 5.0, 1.5)
    project = make_indoor_project(work_dir / "surface_fill_large", floor_hole=hole)
    base = _config()
    config = replace(base, dsm=replace(base.dsm, surface_fill_max_hole_m2=0.2))
    result = build_orthophoto(project.project_dir, level=project.level, config=config, device="cuda" if torch.cuda.is_available() else "cpu")
    validity = cv2.imread(str(result.validity_path), cv2.IMREAD_UNCHANGED)
    assert validity is not None
    assert not np.any(validity[_floor_hole_mask(result, hole)] == 3)


def test_surface_fill_interpolates_table_top_hole(work_dir: Path) -> None:
    hole = (2.85, 2.20, 3.05, 2.40)
    project = make_indoor_project(work_dir / "surface_fill_table", table_hole=hole)
    base = _config()
    config = replace(base, dsm=replace(base.dsm, hole_fill_max_cells=0))
    result = build_orthophoto(project.project_dir, level=project.level, config=config, device="cuda" if torch.cuda.is_available() else "cpu")
    mask = _floor_hole_mask(result, hole)
    validity = cv2.imread(str(result.validity_path), cv2.IMREAD_UNCHANGED)
    assert validity is not None
    assert np.all(validity[mask] == 3)
    dsm = np.load(result.dsm_path)
    assert np.max(np.abs(dsm[mask] - 0.75)) <= 0.02
    rgba = _rgba(result)
    assert np.abs(rgba[mask, :3].astype(np.float64) - ground_truth_rgb(project, result.grid)[mask]).mean() < 15.0


def test_indoor_occlusion_uses_full_geometry_including_ceiling(indoor: tuple[object, object], monkeypatch: pytest.MonkeyPatch) -> None:
    project, default = indoor
    import samcloud.ortho.colorize as colour_module
    original = colour_module.build_zbuffer
    def without_ceiling(camera: object, points: torch.Tensor, config: object, *, indoor: bool) -> tuple[torch.Tensor, float]:
        return original(camera, points[points[:, 2] < 1.0], config, indoor=indoor)
    monkeypatch.setattr(colour_module, "build_zbuffer", without_ceiling)
    baseline = cv2.imread(str(default.view_count_path), cv2.IMREAD_UNCHANGED)
    changed = build_orthophoto(project.project_dir, level=project.level, config=_config(), device="cuda" if torch.cuda.is_available() else "cpu")
    altered = cv2.imread(str(changed.view_count_path), cv2.IMREAD_UNCHANGED)
    assert baseline is not None and altered is not None
    assert np.count_nonzero(altered != baseline) > 0
    # Removing the upper enclosure lets far-side cameras incorrectly see floor cells through it.


def test_georeferenced_frame_bounds_and_world_file() -> None:
    from samcloud.dxf.las_source import Frame
    angle = np.deg2rad(20.0)
    frame = Frame(np.array([2_600_000.0, 1_200_000.0, 500.0]), 1.0, np.array(((np.cos(angle), -np.sin(angle), 0.0), (np.sin(angle), np.cos(angle), 0.0), (0.0, 0.0, 1.0))))
    grid = grid_for_bounds((0.0, 0.0, 10.0, 4.0), 0.5, 100)
    world = grid.original_world_file(frame)
    pixel = np.array([3.0, 2.0])
    expected = frame.to_original(np.array([[*grid.pixel_to_local(pixel), 0.0]]))[0, :2]
    actual = np.array((world[4] + world[0] * pixel[0] + world[2] * pixel[1], world[5] + world[1] * pixel[0] + world[3] * pixel[1]))
    assert np.max(np.abs(expected - actual)) < 1e-5


def test_affine_fallback_to_memmap(work_dir: Path) -> None:
    x, y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
    local = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    raw = local.copy(); raw[:, 0] += 0.002 * local[:, 0] ** 2
    grid = grid_for_bounds((0.0, 0.0, 10.0, 10.0), 0.2, 100)
    dsm = build_dsm(raw, local, np.full(len(raw), 2, dtype=np.uint8), grid, None, OrthoConfig(), work_dir / "memmap_work")
    assert dsm.xyz_method == "memmap"
    assert (work_dir / "memmap_work" / "top_colmap_xyz.npy").is_file()
    expected = raw[(np.isclose(raw[:, 0], 0.0)) & (np.isclose(raw[:, 1], 10.0))][0]
    assert np.allclose(dsm.colmap_xyz[0, 0], expected, atol=1e-5)


def test_pixel_size_capped_to_max_side() -> None:
    grid = grid_for_bounds((0.0, 0.0, 1000.0, 10.0), 0.01, 100)
    assert grid.width == 100 and grid.height <= 100 and grid.pixel_size >= 10.0


def test_dsm_hole_fill_and_validity_mask(work_dir: Path) -> None:
    grid = grid_for_bounds((0.0, 0.0, 3.0, 3.0), 1.0, 10)
    points = np.array([[*grid.pixel_to_local(np.array([[col, row]]))[0], 1.0] for row in range(3) for col in range(3) if (row, col) != (1, 1)])
    dsm = build_dsm(points, points, np.full(len(points), 2, dtype=np.uint8), grid, None, OrthoConfig(), work_dir / "holes")
    assert dsm.validity[1, 1] == 2 and np.isfinite(dsm.dsm[1, 1])


def test_wrong_sparse_model_is_rejected_and_correct_one_selected(outdoor: tuple[object, object]) -> None:
    project, _result = outdoor
    wrong = project.project_dir / "products" / "sparse" / "1_wrong"
    shutil.copytree(project.model_dir, wrong)
    model = pycolmap.Reconstruction(wrong)
    for point in model.points3D.values(): point.xyz = point.xyz + np.array([5.0, 0.0, 0.0])
    model.write(wrong)
    (project.project_dir / "products" / "sparse" / "final_model.txt").write_text("1_wrong", encoding="utf-8")
    cloud = np.asarray([point.xyz for point in pycolmap.Reconstruction(project.model_dir).points3D.values()])
    selected, _distance = select_sparse_model(project.project_dir, cloud)
    assert selected.name == "0"
    with pytest.raises(OrthoError, match="distance"):
        select_sparse_model(project.project_dir, cloud + 50.0, _config())


def test_missing_photos_listed_and_too_many_missing_raise(outdoor: tuple[object, object], work_dir: Path) -> None:
    project, _result = outdoor
    copied = work_dir / "missing_photos_fresh"; copied.mkdir()
    shutil.copy2(project.project_dir / "project.samcloud.yml", copied / "project.samcloud.yml")
    shutil.copytree(project.project_dir / "products" / "classified", copied / "products" / "classified")
    shutil.copytree(project.project_dir / "products" / "sparse", copied / "products" / "sparse")
    image_dir = copied / "inputs" / "images"; image_dir.mkdir(parents=True)
    originals = sorted((project.project_dir / "inputs" / "images").glob("*.png"))
    for source in originals[:7]: shutil.copy2(source, image_dir / source.name)
    config_path = copied / "project.samcloud.yml"; document = yaml.safe_load(config_path.read_text(encoding="utf-8")); document["settings"]["paths"]["image_directory"] = str(image_dir); config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = build_orthophoto(copied, level=None, config=replace(_config(), pixel_size_outdoor=0.103), device="cuda" if torch.cuda.is_available() else "cpu")
    assert len(json.loads((result.output_dir / "ortho.json").read_text(encoding="utf-8"))["missing_photos"]) == 3
    too_many = work_dir / "too_many_missing_project"; too_many.mkdir()
    shutil.copy2(project.project_dir / "project.samcloud.yml", too_many / "project.samcloud.yml")
    shutil.copytree(project.project_dir / "products" / "classified", too_many / "products" / "classified")
    shutil.copytree(project.project_dir / "products" / "sparse", too_many / "products" / "sparse")
    sparse_only = too_many / "inputs" / "images"; sparse_only.mkdir(parents=True)
    for source in originals[:4]: shutil.copy2(source, sparse_only / source.name)
    too_many_config = too_many / "project.samcloud.yml"; document["settings"]["paths"]["image_directory"] = str(sparse_only); too_many_config.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(OrthoError): build_orthophoto(too_many, level=None, config=replace(_config(), pixel_size_outdoor=0.104), device="cuda" if torch.cuda.is_available() else "cpu")


def test_cancel_leaves_no_partial_outputs(outdoor: tuple[object, object]) -> None:
    project, complete = outdoor
    before = {name: (path.read_bytes(), path.stat().st_mtime_ns) for name, path in (("png", complete.ortho_path), ("json", complete.output_dir / "ortho.json"))}
    token = CancelToken()
    def progress(_fraction: float, stage: str) -> None:
        if "Colouring" in stage: token.cancel()
    with pytest.raises(Cancelled): build_orthophoto(project.project_dir, level=None, config=replace(_config(), pixel_size_outdoor=0.105), progress=progress, cancel=token, device="cuda" if torch.cuda.is_available() else "cpu")
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == before[name] for name, path in (("png", complete.ortho_path), ("json", complete.output_dir / "ortho.json")))
    assert not list(complete.output_dir.glob("*.tmp"))


def test_load_ortho_result_never_computes(outdoor: tuple[object, object], monkeypatch: pytest.MonkeyPatch) -> None:
    project, complete = outdoor
    import samcloud.ortho.build as build_module
    monkeypatch.setattr(build_module, "build_dsm", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not compute")))
    monkeypatch.setattr(build_module, "colorize", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not compute")))
    monkeypatch.setattr(build_module, "read_cloud", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not compute")))
    loaded = load_ortho_result(complete.output_dir)
    assert loaded is not None and loaded.grid == complete.grid
    assert load_ortho_result(project.project_dir / "missing") is None


def test_fingerprint_changes_with_settings_and_level(outdoor: tuple[object, object]) -> None:
    project, _result = outdoor
    point = project.project_dir / "products" / "classified" / "classified.ply"
    first = ortho_fingerprint(project.project_dir, level=None, config=_config(), point_source=point, sparse_model=project.model_dir)
    level = Level("level_1", "1", 0.0, 2.6, 1, 1.0, False)
    second = ortho_fingerprint(project.project_dir, level=level, config=replace(_config(), pixel_size_outdoor=0.2), point_source=point, sparse_model=project.model_dir)
    assert first != second


def test_project_points_rejects_fold_back_outside_fov() -> None:
    params = np.array([1000.0, 1000.0, 500.0, 400.0, -0.25, 0.0, 0.0, 0.0], dtype=np.float64)
    camera = OrthoCamera(
        name="test_opencv",
        image_path=None,
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
        model_name="OPENCV",
        params=params,
        width=1000,
        height=800,
        centre=np.zeros(3, dtype=np.float64),
    )
    try:
        ref_cam = pycolmap.Camera(model="OPENCV", width=1000, height=800, params=params)
    except TypeError:
        ref_cam = pycolmap.Camera(camera_id=1, model="OPENCV", width=1000, height=800, params=params)

    # Point (1.97, 0.013, 1.0): undistorted r~1.97, folds to r_d~0.06, i.e. near the image centre. Assert valid is False.
    far_pt = torch.tensor([[1.97, 0.013, 1.0]], dtype=torch.float64)
    _u_far, _v_far, _depth_far, valid_far = project_points(camera, far_pt)
    assert not valid_far.item()

    # Point (0.21, 0.13, 1.0): assert valid is True, and u, v match pycolmap.Camera(...).img_from_cam within 1e-3 px.
    in_pt = torch.tensor([[0.21, 0.13, 1.0]], dtype=torch.float64)
    u_in, v_in, _depth_in, valid_in = project_points(camera, in_pt)
    assert valid_in.item()
    try:
        expected_uv = ref_cam.img_from_cam(np.array([[0.21, 0.13, 1.0]], dtype=np.float64))
    except Exception:
        expected_uv = ref_cam.img_from_cam(np.array([[0.21, 0.13]], dtype=np.float64))
    expected_uv = np.asarray(expected_uv, dtype=np.float64).reshape(-1)
    assert np.allclose([u_in.item(), v_in.item()], expected_uv[:2], atol=1e-3)

    # A point just inside the image corner, e.g. pixel (990.3, 790.7) undistorted via cam_from_img and lifted to z=1: assert valid is True.
    corner_pixel = np.array([[990.3, 790.7]], dtype=np.float64)
    try:
        corner_undist = ref_cam.cam_from_img(corner_pixel)
    except Exception:
        corner_undist = ref_cam.cam_from_img(corner_pixel[0])
    if isinstance(corner_undist, list):
        corner_arr = np.asarray(corner_undist[0], dtype=np.float64).reshape(-1)
    else:
        corner_arr = np.asarray(corner_undist, dtype=np.float64).reshape(-1)
    if len(corner_arr) >= 3:
        corner_xy = corner_arr[:2] / corner_arr[2]
    else:
        corner_xy = corner_arr[:2]
    corner_point = torch.tensor([[corner_xy[0], corner_xy[1], 1.0]], dtype=torch.float64)
    _u_c, _v_c, _depth_c, valid_corner = project_points(camera, corner_point)
    assert valid_corner.item()


def test_project_points_real_gruett_camera_fold_back() -> None:
    # Real camera on gruettistrasse:
    # OPENCV, fx=3701.9, fy=3705.0, cx=2661.2, cy=1969.9, k1=-0.10233, k2=-0.01633, p1=-0.00026, p2=-0.00022, 5280x3956
    params = np.array([3701.9, 3705.0, 2661.2, 1969.9, -0.10233, -0.01633, -0.00026, -0.00022], dtype=np.float64)
    camera = OrthoCamera(
        name="gruett",
        image_path=None,
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
        model_name="OPENCV",
        params=params,
        width=5280,
        height=3956,
        centre=np.zeros(3, dtype=np.float64),
    )
    try:
        ref_cam = pycolmap.Camera(model="OPENCV", width=5280, height=3956, params=params)
    except TypeError:
        ref_cam = pycolmap.Camera(camera_id=1, model="OPENCV", width=5280, height=3956, params=params)

    # Point at 65° off-axis along x: (tan(65°), 0.0137, 1). Assert valid is False.
    x_65 = float(np.tan(np.deg2rad(65.0)))
    point_65 = torch.tensor([[x_65, 0.0137, 1.0]], dtype=torch.float64)
    _u_65, _v_65, _depth_65, valid_65 = project_points(camera, point_65)
    assert not valid_65.item()

    # Point at 25° off-axis (tan(25°), 0.11, 1): assert valid is True and it matches pycolmap within 1e-2 px.
    x_25 = float(np.tan(np.deg2rad(25.0)))
    point_25 = torch.tensor([[x_25, 0.11, 1.0]], dtype=torch.float64)
    u_25, v_25, _depth_25, valid_25 = project_points(camera, point_25)
    assert valid_25.item()

    try:
        expected_25 = ref_cam.img_from_cam(np.array([[x_25, 0.11, 1.0]], dtype=np.float64))
    except Exception:
        expected_25 = ref_cam.img_from_cam(np.array([[x_25, 0.11]], dtype=np.float64))
    expected_25 = np.asarray(expected_25, dtype=np.float64).reshape(-1)
    assert np.allclose([u_25.item(), v_25.item()], expected_25[:2], atol=1e-2)


def test_zbuffer_ignores_points_outside_fov() -> None:
    params = np.array([1000.0, 1000.0, 500.0, 400.0, -0.25, 0.0, 0.0, 0.0], dtype=np.float64)
    camera = OrthoCamera(
        name="test_zbuf",
        image_path=None,
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
        model_name="OPENCV",
        params=params,
        width=1000,
        height=800,
        centre=np.zeros(3, dtype=np.float64),
    )
    pt_far = np.array([1.97, 0.013, 1.0], dtype=np.float64) * 5.0
    pt_near = np.array([0.0137, 0.0091, 10.0], dtype=np.float64)
    points = torch.as_tensor(np.vstack((pt_far, pt_near)), dtype=torch.float32)

    config = OrthoConfig()
    zbuf, factor = build_zbuffer(camera, points, config, indoor=False)

    # Hand-compute distorted pixel of the fold-back point without fix:
    x_fold = 1.97
    y_fold = 0.013
    r2_fold = x_fold * x_fold + y_fold * y_fold
    radial_fold = 1.0 - 0.25 * r2_fold
    u_fold = factor * (1000.0 * (x_fold * radial_fold) + 500.0)
    v_fold = factor * (1000.0 * (y_fold * radial_fold) + 400.0)
    col_fold = int(np.floor(u_fold))
    row_fold = int(np.floor(v_fold))

    # Assert the z-buffer is +inf at the pixel where the fold-back point would land without the fix
    assert torch.isinf(zbuf[row_fold, col_fold]).item()

    # Hand-compute distorted pixel of the on-axis point:
    x_near = 0.0137 / 10.0
    y_near = 0.0091 / 10.0
    r2_near = x_near * x_near + y_near * y_near
    radial_near = 1.0 - 0.25 * r2_near
    u_near = factor * (1000.0 * (x_near * radial_near) + 500.0)
    v_near = factor * (1000.0 * (y_near * radial_near) + 400.0)
    col_near = int(np.floor(u_near))
    row_near = int(np.floor(v_near))

    # Assert finite (~10) near the image centre region of the on-axis point
    val_near = zbuf[row_near, col_near].item()
    assert np.isfinite(val_near)
    assert np.isclose(val_near, 10.0, atol=1.0)


def test_dsm_ray_occluded_detects_canopy_above_ray() -> None:
    from samcloud.ortho.colorize import dsm_ray_occluded

    grid = grid_for_bounds((0.0, 0.0, 20.0, 20.0), 0.1, 16000)
    dsm = torch.zeros((grid.height, grid.width), dtype=torch.float32)
    cols = np.arange(grid.width)
    local_x = grid.pixel_to_local(np.column_stack((cols, np.zeros_like(cols))))[:, 0]
    canopy_cols = (local_x >= 8.0) & (local_x <= 10.0)
    dsm[:, canopy_cols] = 5.0

    cfg = replace(
        OrthoConfig(),
        blend=replace(
            OrthoConfig().blend,
            ray_check_enabled=True,
            ray_check_samples=4,
            ray_check_distance=15.0,
            ray_check_clearance=0.15,
        ),
    )
    surface_xy = torch.tensor([[5.03, 10.07]], dtype=torch.float32)
    surface_z = torch.tensor([0.0], dtype=torch.float32)

    cam_low = torch.tensor([15.04, 10.02, 3.0], dtype=torch.float32)
    assert dsm_ray_occluded(dsm, grid, surface_xy, surface_z, cam_low, cfg).item() is True

    cam_high = torch.tensor([15.04, 10.02, 40.0], dtype=torch.float32)
    assert dsm_ray_occluded(dsm, grid, surface_xy, surface_z, cam_high, cfg).item() is False


def test_ray_check_disabled_by_default() -> None:
    from samcloud.ortho.config import load_ortho_config

    assert OrthoConfig().blend.ray_check_enabled is False
    assert load_ortho_config().blend.ray_check_enabled is False


def test_fingerprint_independent_of_path_spelling(outdoor: tuple[object, object], monkeypatch: pytest.MonkeyPatch) -> None:
    project, _result = outdoor
    project_dir = project.project_dir
    point_source = (project_dir / "products" / "classified" / "classified.ply").resolve()
    sparse_model = project.model_dir

    first = ortho_fingerprint(
        project_dir,
        level=None,
        point_source=point_source,
        sparse_model=sparse_model,
    )

    monkeypatch.chdir(point_source.parent)
    second = ortho_fingerprint(
        project_dir,
        level=None,
        point_source=Path(point_source.name),
        sparse_model=sparse_model,
    )

    assert first == second
