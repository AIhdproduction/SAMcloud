"""Orthophoto fine labels taken from the point cloud (fine_term) instead of a second SAM pass."""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import numpy as np
from plyfile import PlyData, PlyElement
import pytest
from shapely.geometry import Polygon
import yaml

from samcloud.dxf.las_source import Frame
from samcloud.dxf.levels import Level
from samcloud.ortho.build import ortho_fingerprint, ortho_output_dir
from samcloud.ortho.config import OrthoConfig
from samcloud.ortho.dsm import build_dsm, read_cloud
from samcloud.ortho.grid import grid_for_bounds
from samcloud.ortho.outputs import write_outputs
from samcloud.ortho.plan import generate_clean_plan
from samcloud.ortho.segment import segment_ortho
from test_ortho_plan import _build_indoor_wall_face_scene
from test_ortho_segment import _FakeLabeler, _level, _make_ortho, _project

WALL, CEILING, FLOOR, FURNITURE = 2, 3, 4, 17  # scalar ids of the indoor sam3 profile (codes 100, 101, 102, 117)


@pytest.fixture
def work_dir(request: pytest.FixtureRequest) -> Path:
    configured_root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    if configured_root:
        path = Path(configured_root) / f"point_fine_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return request.getfixturevalue("tmp_path_factory").mktemp("point_fine")


def _write_ply(path: Path, xyz: np.ndarray, scalar: np.ndarray, fine: np.ndarray | None) -> None:
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("scalar_class", "u1")]
    if fine is not None:
        fields.append(("fine_term", "u1"))
    data = np.zeros(len(xyz), dtype=fields)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"] = data["green"] = data["blue"] = 150
    data["scalar_class"] = scalar
    if fine is not None:
        data["fine_term"] = fine
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def _column(x: float, y: float, heights: tuple[float, ...], scalar: int, fine: int) -> list[tuple[float, float, float, int, int]]:
    return [(x, y, z, scalar, fine) for z in heights]


def _synthetic_project(root: Path, *, with_fine: bool) -> tuple[Path, Path]:
    project = root / ("fine" if with_fine else "plain")
    project.mkdir(parents=True)
    (project / "project.samcloud.yml").write_text(yaml.safe_dump({"project": {"name": "pf"}, "settings": {"classification": {"class_set": "indoor", "engine": "sam3"}}}), encoding="utf-8")
    rows: list[tuple[float, float, float, int, int]] = []
    rows += _column(0.525, 0.525, (0.0, 0.3, 0.4, 0.5), FURNITURE, 1)  # sofa pixel
    rows += _column(0.525, 0.525, (0.0,), FLOOR, 0)
    rows += _column(1.525, 0.525, (0.0, 0.3, 0.4), FURNITURE, 2)  # bed pixel
    rows += _column(1.525, 0.525, (0.35, 0.45, 0.55, 0.65, 0.75), WALL, 1)  # wall points must not outvote
    rows += _column(2.525, 0.525, (0.3, 0.6, 0.9, 1.2, 1.5), WALL, 1)  # wall only: ignored entirely
    rows += _column(2.525, 1.525, (0.0, 0.1), FLOOR, 0)
    rows += _column(0.525, 1.525, (1.0, 1.1), FURNITURE, 1)  # tie with bed term: the higher point wins
    rows += _column(0.525, 1.525, (0.2, 0.3), FURNITURE, 2)
    array = np.asarray(rows, dtype=np.float64)
    _write_ply(project / "products" / "classified" / "classified.ply", array[:, :3], array[:, 3].astype(np.uint8), array[:, 4].astype(np.uint8) if with_fine else None)
    if with_fine:
        labels = {"labels": ["sofa", "bed"], "ids": [1, 2], "aliases": {}, "class_set": "indoor", "property": "fine_term", "none": 0, "schema": 1}
        (project / "products" / "classified" / "fine_labels.json").write_text(json.dumps(labels), encoding="utf-8")
    return project, project / "products" / "classified" / "classified.ply"


def _rasterise(project: Path, ply: Path) -> tuple[object, object, tuple[np.ndarray, ...]]:
    frame = Frame(np.zeros(3), 1.0, np.eye(3))
    level = Level("level_1", "1", 0.0, 2.6, 0, 1.0, False)
    cloud = read_cloud(project, ply, frame)
    grid = grid_for_bounds((0.0, 0.0, 4.0, 2.0), 0.05, 4096)
    return level, grid, cloud


def test_ortho_reads_point_fine_term_ply(work_dir: Path) -> None:
    project, ply = _synthetic_project(work_dir, with_fine=True)
    level, grid, cloud = _rasterise(project, ply)
    raw, local, classes, rgb, fine = cloud
    assert fine.dtype == np.uint8 and set(np.unique(fine)) == {0, 1, 2}

    result = build_dsm(raw, local, classes, grid, level, OrthoConfig(), work_dir / "work", rgb=rgb, fine_terms=fine)

    def pixel(x: float, y: float) -> tuple[int, int]:
        return int(np.floor((grid.max_y - y) / grid.pixel_size)), int(np.floor((x - grid.min_x) / grid.pixel_size))

    sofa, bed, wall_only, tie = pixel(0.525, 0.525), pixel(1.525, 0.525), pixel(2.525, 0.525), pixel(0.525, 1.525)
    assert result.fine_pc.dtype == np.uint8 and result.fine_pc.shape == (grid.height, grid.width)
    assert result.fine_pc[sofa] == 1  # sofa
    assert result.fine_pc[bed] == 2  # bed, although more wall points carry term 1 in the same column
    assert result.fine_pc[wall_only] == 0
    assert result.fine_pc[tie] == 1  # two sofa points above two bed points: ties go to the higher point
    assert np.count_nonzero(result.fine_pc) == 3
    assert result.fine_pc_confidence.dtype == np.float16
    assert float(result.fine_pc_confidence[sofa]) == pytest.approx(1.0, abs=1e-3)  # points inside the floor margin do not count
    assert float(result.fine_pc_confidence[bed]) == pytest.approx(1.0, abs=1e-3)  # wall points are not part of the column vote
    assert float(result.fine_pc_confidence[tie]) == pytest.approx(0.5, abs=1e-3)
    assert result.fine_pc_confidence[wall_only] == 0

    output = work_dir / "out"
    dsm = result
    paths = write_outputs(output, rgba=np.zeros((grid.height, grid.width, 4), dtype=np.uint8), dsm=dsm.dsm, dtm=dsm.dtm, classes=dsm.classes, validity=dsm.validity, view_count=np.ones((grid.height, grid.width), dtype=np.uint8), metadata={}, fingerprint="x", local_world_file=grid.local_world_file(), original_world_file=None, jpeg_sides=(64,), jpeg_quality=80, fine_pc=dsm.fine_pc, fine_pc_confidence=dsm.fine_pc_confidence, fine_labels_source=ply.with_name("fine_labels.json"))
    assert np.array_equal(np.load(output / "fine_pc.npy"), result.fine_pc)
    assert np.load(output / "fine_pc_conf.npy").dtype == np.float16
    assert json.loads((output / "fine_pc_labels.json").read_text(encoding="utf-8"))["labels"] == ["sofa", "bed"]
    assert paths["fine_pc_labels"].name == "fine_pc_labels.json"


def test_ortho_fingerprint_follows_fine_term_and_labels(work_dir: Path) -> None:
    project, ply = _synthetic_project(work_dir, with_fine=True)
    model = work_dir / "model"
    model.mkdir()
    level = Level("level_1", "1", 0.0, 2.6, 0, 1.0, False)
    frame = Frame(np.zeros(3), 1.0, np.eye(3))
    config = OrthoConfig()

    def fingerprint() -> str:
        return ortho_fingerprint(project, level=level, config=config, frame=frame, point_source=ply, sparse_model=model)

    baseline = fingerprint()
    assert fingerprint() == baseline
    labels = ply.with_name("fine_labels.json")
    payload = json.loads(labels.read_text(encoding="utf-8"))
    payload["labels"][0] = "couch"
    labels.write_text(json.dumps(payload), encoding="utf-8")
    renamed = fingerprint()
    assert renamed != baseline
    array = PlyData.read(str(ply), mmap=False)["vertex"].data
    stat = ply.stat()
    changed = np.array(array)
    changed["fine_term"][0] = 2 if changed["fine_term"][0] != 2 else 1
    PlyData([PlyElement.describe(changed, "vertex")], text=False).write(str(ply))
    os.utime(ply, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # identical mtime: only the content hash can tell
    assert fingerprint() != renamed


def test_plan_prefers_point_fine_labels(work_dir: Path) -> None:
    project_dir = work_dir / "plan_point_fine"
    level, _room = _build_indoor_wall_face_scene(project_dir, units_per_metre=1.0)
    ortho_dir = ortho_output_dir(project_dir, level)
    classes = np.load(ortho_dir / "class_combined.npy")
    height, width = classes.shape
    table = classes == 117
    columns = np.nonzero(table.any(axis=0))[0]
    split = columns[0] + int(round(0.68 * (columns[-1] - columns[0] + 1)))
    fine = np.zeros_like(classes)
    fine[table & (np.arange(width)[None, :] < split)] = 1  # bed
    fine[table & (np.arange(width)[None, :] >= split)] = 2  # shelf
    assert np.count_nonzero(fine == 1) > 0 and np.count_nonzero(fine == 2) > 0
    np.save(ortho_dir / "fine_pc.npy", fine)
    np.save(ortho_dir / "fine_pc_conf.npy", np.where(fine > 0, 0.95, 0.0).astype(np.float16))
    (ortho_dir / "fine_pc_labels.json").write_text(json.dumps({"labels": ["bed", "shelf"], "aliases": {}, "schema": 1}), encoding="utf-8")
    # The old ortho-SAM raster says "toilet": it must be ignored while point labels exist.
    np.save(ortho_dir / "fine_class.npy", np.where(table, 1, 0).astype(np.uint8))
    np.save(ortho_dir / "fine_conf.npy", np.where(table, 0.95, 0.0).astype(np.float16))
    (ortho_dir / "fine_labels.json").write_text(json.dumps(["toilet"]), encoding="utf-8")

    result = generate_clean_plan(project_dir, level=level, mode="indoor")

    table = Polygon([(2.1, 2.5), (3.6, 2.5), (3.6, 3.5), (2.1, 3.5)])
    labels = sorted(
        str(entity.get("_label")) for entity in result.layers["A-FURN"]
        if entity.get("type") == "polyline" and entity.get("_label") and table.contains(Polygon(entity["points"]).centroid)
    )
    assert labels == ["Bett", "Regal"]  # (a 1.0 m deep "nightstand" would be implausible: the size rule drops that label)
    assert indoor_metric(result)["fine_label_source"] == "pointcloud"
    assert result.fingerprint_inputs["fine_label_source"] == "pointcloud"


def indoor_metric(result: object) -> dict[str, object]:
    from samcloud.ortho.plan import indoor_wall_metrics

    return indoor_wall_metrics(result)


def test_plan_falls_back_to_ortho_sam_fine_labels(work_dir: Path) -> None:
    project_dir = work_dir / "plan_ortho_sam_fine"
    level, _room = _build_indoor_wall_face_scene(project_dir, units_per_metre=1.0)
    result = generate_clean_plan(project_dir, level=level, mode="indoor")
    assert indoor_metric(result)["fine_label_source"] == "ortho_sam"
    plain = work_dir / "plan_no_fine"
    level, _room = _build_indoor_wall_face_scene(plain, units_per_metre=1.0)
    np.save(ortho_output_dir(plain, level) / "fine_class.npy", np.zeros_like(np.load(ortho_output_dir(plain, level) / "class_combined.npy")))
    result = generate_clean_plan(plain, level=level, mode="indoor")
    assert indoor_metric(result)["fine_label_source"] == "none"


def test_ortho_fine_sam_skipped_with_point_labels(work_dir: Path) -> None:
    project = _project(work_dir, "indoor")
    level = _level()
    rgba = np.full((3, 3, 4), 255, dtype=np.uint8)
    point = np.full((3, 3), 100, dtype=np.uint8)
    point[1, 1] = 117
    root = _make_ortho(project, rgba, point, level=level)
    fine_prompt_calls: list[list[str]] = []
    coarse_calls: list[list[str]] = []

    def factory(**kwargs: object) -> _FakeLabeler:
        names = list(kwargs["class_names"])
        (fine_prompt_calls if names[0] == "toilet" else coarse_calls).append(names)
        return _FakeLabeler(1, 0.9) if names[0] == "toilet" else _FakeLabeler(0, 0.1)

    # Without point labels the fine prompt pass runs.
    first = segment_ortho(project, level=level, source="combined", labeler_factory=factory)
    assert len(fine_prompt_calls) == 1 and len(coarse_calls) == 1
    assert first.fine_classes_path is not None
    assert first.metadata["fine_label_source"] == "ortho_sam"

    # With point labels the fine prompt pass is skipped, the coarse pass is unchanged.
    fine_pc = np.zeros((3, 3), dtype=np.uint8)
    fine_pc[1, 1] = 1
    np.save(root / "fine_pc.npy", fine_pc)
    np.save(root / "fine_pc_conf.npy", np.where(fine_pc > 0, 0.9, 0.0).astype(np.float16))
    (root / "fine_pc_labels.json").write_text(json.dumps({"labels": ["sofa"]}), encoding="utf-8")
    (root / "fine_class.npy").unlink()
    (root / "fine_conf.npy").unlink()
    (root / "fine_labels.json").unlink()
    second = segment_ortho(project, level=level, source="combined", labeler_factory=factory)
    assert len(fine_prompt_calls) == 1 and len(coarse_calls) == 2
    assert second.fine_classes_path is None
    assert second.metadata["fine_label_source"] == "pointcloud"
    assert not (root / "fine_class.npy").exists()

    # An all-zero point raster does not count as point labels.
    np.save(root / "fine_pc.npy", np.zeros((3, 3), dtype=np.uint8))
    segment_ortho(project, level=level, source="combined", labeler_factory=factory)
    assert len(fine_prompt_calls) == 2


def test_ortho_without_fine_term_unchanged(work_dir: Path) -> None:
    project, ply = _synthetic_project(work_dir, with_fine=False)
    level, grid, cloud = _rasterise(project, ply)
    raw, local, classes, rgb, fine = cloud
    assert not np.any(fine) and len(fine) == len(raw)

    new = build_dsm(raw, local, classes, grid, level, OrthoConfig(), work_dir / "new", rgb=rgb, fine_terms=fine)
    old = build_dsm(raw, local, classes, grid, level, OrthoConfig(), work_dir / "old", rgb=rgb)  # pre-fine-term call
    for name in ("dsm", "dtm", "classes", "validity", "footprint", "normals", "top_rgb"):
        assert np.array_equal(getattr(new, name), getattr(old, name), equal_nan=True), name
    assert not np.any(new.fine_pc) and not np.any(new.fine_pc_confidence)

    paths = write_outputs(work_dir / "out", rgba=np.zeros((grid.height, grid.width, 4), dtype=np.uint8), dsm=new.dsm, dtm=new.dtm, classes=new.classes, validity=new.validity, view_count=np.ones((grid.height, grid.width), dtype=np.uint8), metadata={}, fingerprint="x", local_world_file=grid.local_world_file(), original_world_file=None, jpeg_sides=(64,), jpeg_quality=80, fine_pc=new.fine_pc, fine_pc_confidence=new.fine_pc_confidence, fine_labels_source=ply.with_name("fine_labels.json"))
    assert "fine_pc_labels" not in paths  # no vocabulary exists, so none is copied
    assert not np.any(np.load(work_dir / "out" / "fine_pc.npy"))
    # A project whose ortho has no point labels keeps using the fallback sources.
    assert not (work_dir / "out" / "fine_pc_labels.json").exists()
