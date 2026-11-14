from pathlib import Path
import struct
import subprocess
from threading import Thread
import time

import laspy
import numpy as np
import pytest

from samcloud.app.application import potree_file
from samcloud.app.state import state
from samcloud.viewer.potree import (
    _potree_paths,
    _ply_points,
    ensure_potree,
    potree_available,
    write_viewer_las,
)
from samcloud.viewer import potree


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


def test_ensure_potree_serializes_concurrent_conversions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    source = project / "products" / "classified" / "classified.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"ply")
    calls: list[list[str]] = []
    results: list[Path] = []
    errors: list[Exception] = []

    monkeypatch.setattr(potree, "potree_available", lambda: True)
    monkeypatch.setattr(potree, "_ply_points", lambda _path: (np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), None))

    def fake_write_las(path, *_args) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"las")

    monkeypatch.setattr(potree, "write_viewer_las", fake_write_las)

    def slow_converter(command, **_kwargs):
        calls.append(command)
        time.sleep(0.1)
        output = Path(command[command.index("-o") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "metadata.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(potree.subprocess, "run", slow_converter)

    def convert() -> None:
        try:
            results.append(ensure_potree(project, "dense"))
        except Exception as error:
            errors.append(error)

    threads = [Thread(target=convert), Thread(target=convert)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    expected = project.resolve() / "products" / "viewer" / "potree" / "dense"
    assert not errors
    assert len(calls) == 1
    assert results == [expected, expected]


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



def test_potree_status_reports_building_then_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from threading import Event
    from samcloud.viewer.potree import potree_status, start_potree_build

    potree._POTREE_BUILD_STATES.clear()
    project = tmp_path / "project"
    source = project / "products" / "classified" / "classified.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"ply")

    monkeypatch.setattr(potree, "potree_available", lambda: True)
    monkeypatch.setattr(potree, "_ply_points", lambda _path: (np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), None))

    def fake_write_las(path, *_args) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"las")

    monkeypatch.setattr(potree, "write_viewer_las", fake_write_las)

    release_converter = Event()
    converter_started = Event()

    def fake_converter(command, **_kwargs):
        converter_started.set()
        release_converter.wait(timeout=5)
        output = Path(command[command.index("-o") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "metadata.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(potree.subprocess, "run", fake_converter)

    start_potree_build(project, "dense")
    assert converter_started.wait(timeout=2)
    assert potree_status(project, "dense")["state"] == "building"

    release_converter.set()

    deadline = time.time() + 5
    state_str = ""
    while time.time() < deadline:
        status = potree_status(project, "dense")
        state_str = status["state"]
        if state_str == "ready":
            break
        time.sleep(0.05)

    assert state_str == "ready"


def test_start_potree_build_runs_one_conversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from threading import Event
    from samcloud.viewer.potree import potree_status, start_potree_build

    potree._POTREE_BUILD_STATES.clear()
    project = tmp_path / "project"
    source = project / "products" / "classified" / "classified.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"ply")

    monkeypatch.setattr(potree, "potree_available", lambda: True)
    monkeypatch.setattr(potree, "_ply_points", lambda _path: (np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), None))

    def fake_write_las(path, *_args) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"las")

    monkeypatch.setattr(potree, "write_viewer_las", fake_write_las)

    calls: list[list[str]] = []
    converter_started = Event()
    release_converter = Event()

    def fake_converter(command, **_kwargs):
        calls.append(command)
        converter_started.set()
        release_converter.wait(timeout=5)
        output = Path(command[command.index("-o") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "metadata.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(potree.subprocess, "run", fake_converter)

    start_potree_build(project, "dense")
    assert converter_started.wait(timeout=2)
    start_potree_build(project, "dense")

    release_converter.set()

    deadline = time.time() + 5
    state_str = ""
    while time.time() < deadline:
        status = potree_status(project, "dense")
        state_str = status["state"]
        if state_str == "ready":
            break
        time.sleep(0.05)

    assert state_str == "ready"
    assert len(calls) == 1


def test_potree_status_reports_error_when_converter_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.viewer.potree import potree_status, start_potree_build

    potree._POTREE_BUILD_STATES.clear()
    project = tmp_path / "project"
    source = project / "products" / "classified" / "classified.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"ply")

    monkeypatch.setattr(potree, "potree_available", lambda: True)
    monkeypatch.setattr(potree, "_ply_points", lambda _path: (np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), None))

    def fake_write_las(path, *_args) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"las")

    monkeypatch.setattr(potree, "write_viewer_las", fake_write_las)

    def failing_converter(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")

    monkeypatch.setattr(potree.subprocess, "run", failing_converter)

    start_potree_build(project, "dense")

    deadline = time.time() + 5
    status: dict[str, object] = {}
    while time.time() < deadline:
        status = potree_status(project, "dense")
        if status.get("state") != "building":
            break
        time.sleep(0.05)

    assert status.get("state") == "error"
    assert "boom" in str(status.get("message", ""))


def test_viewer_config_contains_status_url() -> None:
    from samcloud.app.potree_viewer import _viewer_config

    assert _viewer_config("dense", "Dense", [[0, 0, 0]])["status_url"] == "/api/potree/dense/status"


def test_write_viewer_las_preserves_instance_ids_and_omits_invalid_values(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    rgb = np.full((3, 3), 20, dtype=np.uint8)
    with_instances = tmp_path / "instances.las"
    write_viewer_las(with_instances, xyz, rgb, instance_ids=np.array([12, 0, 65536]))
    assert np.array_equal(np.asarray(laspy.read(with_instances).point_source_id), [12, 0, 0])
    assert "above 65535" in caplog.text

    without_instances = tmp_path / "no_instances.las"
    write_viewer_las(without_instances, xyz, rgb)
    assert np.array_equal(np.asarray(laspy.read(without_instances).point_source_id), [0, 0, 0])


def test_dense_octree_signature_tracks_instance_property(tmp_path: Path) -> None:
    source = tmp_path / "project" / "products" / "classified" / "classified.ply"
    source.parent.mkdir(parents=True)
    base_header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    ).encode("ascii")
    source.write_bytes(base_header + np.zeros(1, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")]).tobytes())
    without_instances = _potree_paths(tmp_path / "project", "dense")[3]

    instance_header = base_header.replace(b"end_header", b"property uint instance_id\nend_header")
    source.write_bytes(instance_header + np.zeros(1, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("instance_id", "<u4")]).tobytes())
    with_instances = _potree_paths(tmp_path / "project", "dense")[3]

    assert "instance_ids" not in without_instances
    assert with_instances["instance_ids"] is True
    assert without_instances != with_instances


def test_signature_without_instances_keeps_legacy_format(tmp_path: Path) -> None:
    source = tmp_path / "project" / "products" / "classified" / "classified.ply"
    source.parent.mkdir(parents=True)
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    ).encode("ascii")
    source.write_bytes(header + np.zeros(1, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")]).tobytes())
    signature = _potree_paths(tmp_path / "project", "dense")[3]
    assert set(signature) == {"source", "size", "mtime_ns", "encoding"}


def test_potree_converter_gets_separate_attribute_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    potree._POTREE_BUILD_STATES.clear()
    project = tmp_path / "project"
    source = project / "products" / "classified" / "classified.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"ply")
    monkeypatch.setattr(potree, "potree_available", lambda: True)
    monkeypatch.setattr(potree, "_ply_points", lambda _path: (np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), None))
    monkeypatch.setattr(potree, "write_viewer_las", lambda path, *_a: (path.parent.mkdir(parents=True, exist_ok=True), path.write_bytes(b"las")))
    seen: list[list[str]] = []

    def fake_converter(command, **_kwargs):
        seen.append(list(command))
        output = Path(command[command.index("-o") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "metadata.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(potree.subprocess, "run", fake_converter)
    potree.ensure_potree(project, "dense")
    command = seen[0]
    index = command.index("--attributes")
    assert command[index + 1:index + 4] == ["rgb", "classification", "point source id"]
    assert not any("," in token for token in command[index + 1:])



def test_write_viewer_las_writes_user_data_from_fine_terms(tmp_path: Path) -> None:
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    rgb = np.full((3, 3), 20, dtype=np.uint8)
    with_terms = tmp_path / "fine.las"
    write_viewer_las(with_terms, xyz, rgb, None, None, np.array([0, 3, 7]))
    assert np.array_equal(np.asarray(laspy.read(with_terms).user_data), [0, 3, 7])
    without = tmp_path / "none.las"
    write_viewer_las(without, xyz, rgb)
    assert np.array_equal(np.asarray(laspy.read(without).user_data), [0, 0, 0])


def _tiny_ply(source: Path, with_fine: bool) -> None:
    source.parent.mkdir(parents=True, exist_ok=True)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")] + ([("fine_term", "u1")] if with_fine else [])
    names = {"<f4": "float", "u1": "uchar"}
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n" + "".join(
        f"property {names[kind]} {name}\n" for name, kind in fields) + "end_header\n"
    source.write_bytes(header.encode("ascii") + np.zeros(1, dtype=fields).tobytes())


def test_signature_tracks_fine_terms_only_when_present(tmp_path: Path) -> None:
    source = tmp_path / "project" / "products" / "classified" / "classified.ply"
    _tiny_ply(source, False)
    plain = _potree_paths(tmp_path / "project", "dense")[3]
    assert set(plain) == {"source", "size", "mtime_ns", "encoding"}
    _tiny_ply(source, True)
    fine = _potree_paths(tmp_path / "project", "dense")[3]
    assert fine["fine_terms"] is True
    assert potree._ply_fine_terms(source).tolist() == [0]


def test_potree_converter_requests_user_data_attribute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    potree._POTREE_BUILD_STATES.clear()
    project = tmp_path / "project"
    source = project / "products" / "classified" / "classified.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"ply")
    monkeypatch.setattr(potree, "potree_available", lambda: True)
    monkeypatch.setattr(potree, "_ply_points", lambda _path: (np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), None))
    monkeypatch.setattr(potree, "write_viewer_las", lambda path, *_a: (path.parent.mkdir(parents=True, exist_ok=True), path.write_bytes(b"las")))
    seen: list[list[str]] = []

    def fake_converter(command, **_kwargs):
        seen.append(list(command))
        output = Path(command[command.index("-o") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "metadata.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(potree.subprocess, "run", fake_converter)
    potree.ensure_potree(project, "dense")
    command = seen[0]
    assert command[command.index("--attributes") + 1:][-1] == "user data"
