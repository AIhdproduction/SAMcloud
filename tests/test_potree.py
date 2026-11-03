from pathlib import Path
import struct
import subprocess

import laspy
import numpy as np
import pytest

from samcloud.app.application import potree_file
from samcloud.app.state import state
from samcloud.viewer.potree import (
    _ply_points,
    ensure_potree,
    potree_available,
    write_viewer_las,
)


def _read_ply_header_compat(source: Path):
    """Parse binary little-endian PLY headers supporting float, uchar, and int properties."""
    with source.open("rb") as file:
        header = bytearray()
        while b"end_header" not in header:
            line = file.readline()
            if not line:
                raise ValueError("PLY header is incomplete")
            header.extend(line)
        header_text = header.decode("ascii")
    lines = [line.strip().split() for line in header_text.splitlines()]
    if len(lines) < 3 or lines[0] != ["ply"] or lines[1][:2] != ["format", "binary_little_endian"]:
        raise ValueError("Only binary little-endian PLY files are supported")
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for parts in lines:
        if parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
        elif parts[:2] == ["element"]:
            in_vertex = False
        elif in_vertex and parts[:1] == ["property"] and len(parts) == 3:
            properties.append((parts[1], parts[2]))
    type_map = {
        "float": "<f4",
        "float32": "<f4",
        "uchar": "u1",
        "uint8": "u1",
        "int": "<i4",
        "int32": "<i4",
    }
    if not vertex_count or not properties or any(kind not in type_map for kind, _ in properties):
        raise ValueError("PLY does not contain a supported vertex element")
    dtype = np.dtype([(name, type_map[kind]) for kind, name in properties], align=False)
    offsets = {name: index for index, (_, name) in enumerate(properties)}
    return vertex_count, len(properties), dtype, offsets


def test_write_viewer_las_round_trip(tmp_path: Path) -> None:
    xyz = np.array(
        [
            [12.3456, -7.0012, 905.9681],
            [0.1234, 45.6789, -12.3456],
            [-100.5432, 200.6789, 300.9876],
        ],
        dtype=float,
    )
    rgb = np.array(
        [
            [255, 0, 128],
            [10, 20, 30],
            [200, 210, 220],
        ],
        dtype=np.uint8,
    )
    classes = np.array([0, 5, 16], dtype=int)
    las_path = tmp_path / "test.las"

    write_viewer_las(las_path, xyz, rgb, classes)
    assert las_path.is_file()

    las = laspy.read(las_path)
    read_xyz = np.column_stack((las.x, las.y, las.z))
    assert np.all(np.abs(read_xyz - xyz) <= 0.0006)

    read_rgb = np.column_stack((las.red, las.green, las.blue))
    assert np.array_equal(read_rgb, rgb.astype(np.uint16) * 257)

    read_classes = np.asarray(las.classification)
    assert np.array_equal(read_classes, classes)


def test_ply_points_reads_scalar_class(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ply_path = tmp_path / "classified.ply"
    header = (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 3\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"property uchar red\n"
        b"property uchar green\n"
        b"property uchar blue\n"
        b"property int scalar_class\n"
        b"end_header\n"
    )
    record_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("scalar_class", "<i4"),
        ],
        align=False,
    )
    data = np.zeros(3, dtype=record_dtype)
    data["x"] = [1.5, 2.5, 3.5]
    data["y"] = [4.5, 5.5, 6.5]
    data["z"] = [7.5, 8.5, 9.5]
    data["red"] = [10, 40, 70]
    data["green"] = [20, 50, 80]
    data["blue"] = [30, 60, 90]
    data["scalar_class"] = [1, 5, 12]
    ply_path.write_bytes(header + data.tobytes())

    monkeypatch.setattr("samcloud.viewer.potree._read_ply_header", _read_ply_header_compat)
    xyz, rgb, classes = _ply_points(ply_path)

    assert xyz.shape == (3, 3)
    assert np.allclose(xyz, [[1.5, 4.5, 7.5], [2.5, 5.5, 8.5], [3.5, 6.5, 9.5]])
    assert rgb.shape == (3, 3)
    assert rgb.dtype == np.uint8
    assert np.array_equal(rgb, [[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    assert classes is not None
    assert classes.shape == (3,)
    assert np.issubdtype(classes.dtype, np.integer)
    assert np.array_equal(classes, [1, 5, 12])


def test_ensure_potree_converts_and_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not potree_available():
        pytest.skip("Potree or PotreeConverter is not installed")

    project = tmp_path / "project"
    fused_ply = project / "products" / "dense" / "fused.ply"
    fused_ply.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(5)
    n_points = 2000
    xyz = rng.uniform(-10.0, 10.0, size=(n_points, 3)).astype("<f4")
    rgb = rng.integers(0, 256, size=(n_points, 3), dtype=np.uint8)
    header = (
        f"ply\n"
        f"format binary_little_endian 1.0\n"
        f"element vertex {n_points}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property uchar red\n"
        f"property uchar green\n"
        f"property uchar blue\n"
        f"end_header\n"
    ).encode("ascii")
    record_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
        align=False,
    )
    records = np.empty(n_points, dtype=record_dtype)
    records["x"] = xyz[:, 0]
    records["y"] = xyz[:, 1]
    records["z"] = xyz[:, 2]
    records["red"] = rgb[:, 0]
    records["green"] = rgb[:, 1]
    records["blue"] = rgb[:, 2]
    fused_ply.write_bytes(header + records.tobytes())

    output = ensure_potree(project, "dense")
    assert output.is_dir()
    assert (output / "metadata.json").is_file()

    def _fail_converter(*args, **kwargs):
        raise AssertionError("PotreeConverter should not be re-run on cache hit")

    monkeypatch.setattr("samcloud.viewer.potree.subprocess.run", _fail_converter)
    cached_output = ensure_potree(project, "dense")
    assert cached_output == output
    assert (cached_output / "metadata.json").is_file()


def test_potree_route_rejects_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    octree_path = project / "products" / "viewer" / "potree" / "dense" / "octree.bin"
    octree_path.parent.mkdir(parents=True, exist_ok=True)
    octree_path.write_bytes(b"octree binary payload")

    monkeypatch.setattr(state, "active_project_directory", project)

    res_traversal = potree_file("dense", "../../x")
    assert res_traversal.status_code == 400

    res_valid = potree_file("dense", "octree.bin")
    assert res_valid.status_code == 200
