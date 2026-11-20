"""End-to-end coverage for the resumable drone environment pipeline."""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import laspy
import numpy as np
import pytest
from pyproj import Transformer

from samcloud.app.potree_viewer import node_group, read_glb_document
from samcloud.drone.contracts import BuildingInstance, load_drone_config
from samcloud.drone.run_environment import run_environment


_CODES = [0, 2, 11, 73, 74, 3, 5, 6, 104, 75, 95, 144]
_INTERNAL = {code: index for index, code in enumerate(_CODES)}
_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)


def _georef() -> dict:
    """Same ENU-to-ECEF georeference fixture used by the drone frame tests."""
    lat, lon = np.radians(47.134), np.radians(8.755)
    east = [-np.sin(lon), np.cos(lon), 0.0]
    north = [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)]
    up = [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    rotation = np.array([east, north, up]).T
    origin = np.array(_TO_ECEF.transform(8.755, 47.134, 450.0))
    return {"scale": 1.0, "rotation": rotation.tolist(), "translation": origin.tolist(),
            "pairs": 5, "residual_m": {"mean": 0.0, "p95": 0.0, "max": 0.0}}


def _write_ply(path: Path, xyz: np.ndarray, classes: np.ndarray, normals: np.ndarray) -> None:
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
             ("r", "u1"), ("g", "u1"), ("b", "u1"), ("c", "u1")]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(xyz)
    header += "property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\n"
    header += "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar scalar_class\nend_header\n"
    data = np.zeros(len(xyz), dtype=dtype)
    data["x"], data["y"], data["z"] = xyz.T
    data["nx"], data["ny"], data["nz"] = normals.T
    data["r"], data["g"], data["b"] = 120, 170, 110
    data["c"] = classes
    path.write_bytes(header.encode("ascii") + data.tobytes())


def _append(parts, points, code, normal=(0.0, 0.0, 1.0)) -> None:
    parts.append((np.asarray(points, dtype=np.float32), np.full(len(points), _INTERNAL[code], np.uint8),
                  np.tile(np.asarray(normal, np.float32), (len(points), 1))))


def _synthetic_project(tmp_path: Path, *, points: int = 100_000) -> Path:
    """A dense, small survey with every outdoor feature used by this pipeline."""
    rng = np.random.default_rng(19)
    project = tmp_path / "project"
    classified = project / "products" / "classified"
    classified.mkdir(parents=True)
    (project / "products" / "georeference.json").write_text(json.dumps(_georef()), encoding="utf-8")
    (project / "config").mkdir()
    prompts = [[str(code)] for code in _CODES[1:]]
    (project / "config" / "classes.json").write_text(json.dumps({"las_codes": {"outdoor": _CODES}, "prompts": {"outdoor": prompts}}), encoding="utf-8")

    parts = []
    slope = lambda xy: 0.015 * xy[:, 0] + 0.008 * xy[:, 1]
    n_ground = max(35_000, points - 41_000)
    xy = rng.uniform([-25, -20], [25, 25], (n_ground, 2)); _append(parts, np.c_[xy, slope(xy)], 2)
    xy = rng.uniform([-25, -2.5], [25, 3.0], (13_000, 2)); _append(parts, np.c_[xy, slope(xy)], 11)
    xy = rng.uniform([-25, 3.12], [25, 5.0], (7_000, 2)); _append(parts, np.c_[xy, slope(xy) + .12], 73)
    x = np.linspace(-24, 24, 1_200); y = 3.05 + rng.normal(0, .015, len(x)); xy = np.c_[x, y]; _append(parts, np.c_[xy, slope(xy) + .12], 74)
    xy = rng.uniform([-25, 6], [25, 24], (6_000, 2)); _append(parts, np.c_[xy, slope(xy)], 3)

    # 10 x 8 m gable house: roof, facades and a window patch at a 1 m sill.
    x = rng.uniform(-5, 5, 9_000); y = rng.uniform(9, 17, 9_000); z = 3.0 + 2.5 * (1.0 - np.abs(x) / 5.0)
    _append(parts, np.c_[x, y, z + slope(np.c_[x, y])], 144, (-.45, 0, .89))
    for fixed, axis, normal in ((-5.0, 1, (-1, 0, 0)), (5.0, 1, (1, 0, 0)), (9.0, 0, (0, -1, 0)), (17.0, 0, (0, 1, 0))):
        other = rng.uniform(9, 17, 1_200) if axis == 1 else rng.uniform(-5, 5, 1_200)
        height = rng.uniform(.1, 3.0, 1_200)
        xy = np.c_[np.full(1_200, fixed), other] if axis == 1 else np.c_[other, np.full(1_200, fixed)]
        _append(parts, np.c_[xy, slope(xy) + height], 6, normal)
    wx = rng.uniform(-1.4, 1.4, 600); wy = np.full(600, 9.0); xy = np.c_[wx, wy]
    _append(parts, np.c_[xy, slope(xy) + rng.uniform(1.0, 2.2, 600)], 104, (0, -1, 0))

    theta = rng.uniform(0, 2 * np.pi, 3_500); radius = rng.uniform(0, 2.3, 3_500) ** .5 * 2.3
    xy = np.c_[15 + radius * np.cos(theta), 14 + radius * np.sin(theta)]
    _append(parts, np.c_[xy, slope(xy) + 3.0 + 4.0 * (1 - radius / 2.3) + rng.normal(0, .15, len(xy))], 5)
    theta = rng.uniform(0, 2 * np.pi, 500); radius = np.sqrt(rng.uniform(0, 1, 500)) * .18
    xy = np.c_[15 + radius * np.cos(theta), 14 + radius * np.sin(theta)]
    _append(parts, np.c_[xy, slope(xy) + rng.uniform(0, 3.2, 500)], 95)
    theta = rng.uniform(0, 2 * np.pi, 500); radius = np.sqrt(rng.uniform(0, 1, 500)) * .35
    xy = np.c_[0 + radius * np.cos(theta), radius * np.sin(theta)]
    _append(parts, np.c_[xy, slope(xy) + .02], 75)

    xyz, classes, normals = (np.concatenate(values) for values in zip(*parts, strict=True))
    assert 100_000 <= len(xyz) <= 200_000
    _write_ply(classified / "classified.ply", xyz, classes, normals)
    top1 = np.full(len(xyz), .9, np.float16)
    np.savez(classified / "classified_scores.npz", top1_conf=top1, top2_conf=np.full(len(xyz), .4, np.float16),
             view_count=np.full(len(xyz), 5, np.uint16))
    return project


def _fast_config() -> dict:
    config = load_drone_config()
    config["terrain"].update({"ground_filter": "pmf", "dtm_resolution_m": .35, "dsm_resolution_m": .35})
    config["terrain"]["tin"]["thin_spacing_m"] = .75
    return config


def test_run_environment_end_to_end_resume_glb_and_cli(tmp_path: Path) -> None:
    project = _synthetic_project(tmp_path)
    started = time.perf_counter()
    result = run_environment(project, config=_fast_config(), resume=False)
    first_runtime = time.perf_counter() - started
    print(f"first end-to-end runtime: {first_runtime:.2f}s")
    assert first_runtime < 180
    statuses = {name: item["status"] for name, item in result["steps"].items()}
    assert all(status == "done" for status in statuses.values()), statuses

    out = project / "products" / "environment"
    for name in ("outdoor.ifc", "outdoor.dxf", "environment.glb", "classified.laz", "dtm.tif", "dsm.tif",
                 "environment_report.json", "environment_report.md", "qa_cad_report.json", "qa_cad_report.md"):
        assert (out / name).is_file(), name
    qa = json.loads((out / "qa_cad_report.json").read_text(encoding="utf-8"))
    checks = {item["rule"]: item["ok"] for item in qa}
    for category in ("tree", "manhole", "road", "sidewalk", "curb", "building"):
        assert checks[f"15_count_{category}"] is True

    import ifcopenshell
    import ezdxf
    assert ifcopenshell.open(str(out / "outdoor.ifc")).by_type("IfcMapConversion")
    assert min(entity.dxf.insert[0] for entity in ezdxf.readfile(out / "outdoor.dxf").modelspace() if entity.dxftype() == "INSERT") > 2_000_000
    las = laspy.read(out / "classified.laz")
    assert {"confidence", "margin", "view_count"}.issubset(las.point_format.extra_dimension_names)

    document = read_glb_document(out / "environment.glb")
    accepted = {"terrain", "dsm", "building", "vegetation", "pavement", "civil", "other"}
    mesh_nodes = [node for node in document["nodes"] if "mesh" in node]
    assert mesh_nodes and all(node.get("extras", {}).get("category") in accepted for node in mesh_nodes)
    assert all(node_group(node["name"], node["extras"]) == node["extras"]["category"] for node in mesh_nodes)
    assert any(node["extras"]["category"] == "dsm" for node in mesh_nodes)
    assert document["asset"]["extras"]["origin_lv95"]

    ifc_mtime = (out / "outdoor.ifc").stat().st_mtime_ns
    resumed_at = time.perf_counter(); resumed = run_environment(project, config=_fast_config(), resume=True)
    assert time.perf_counter() - resumed_at < first_runtime * .2
    assert (out / "outdoor.ifc").stat().st_mtime_ns == ifc_mtime
    assert all(item["reason"] == "resume: completed checkpoint" for item in resumed["steps"].values())

    targeted = run_environment(project, config=_fast_config(), steps=["buildings"], resume=True)
    rerun = {name for name, item in targeted["steps"].items() if item["status"] == "done"}
    assert {"buildings", "scene", "export", "qa", "report"}.issubset(rerun)
    assert "site" not in rerun
    command = [sys.executable, "-m", "samcloud.drone.run_environment", "--project", str(project), "--steps", "report"]
    assert subprocess.run(command, cwd=Path.cwd(), env={**os.environ, "SAMCLOUD_TEST_WORKDIR": str(tmp_path)}, check=False).returncode == 0


def test_run_environment_isolates_a_building_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _synthetic_project(tmp_path)
    config = _fast_config()
    run_environment(project, config=config, resume=False)
    out = project / "products" / "environment"
    cache = out / "cache"
    with (cache / "load.pkl").open("rb") as stream:
        points, _frame = pickle.load(stream)
    indices = np.flatnonzero(np.isin(points.classes, [6, 144]))
    with (cache / "semantics.pkl").open("wb") as stream:
        pickle.dump({"instances": [BuildingInstance(1, indices)]}, stream)

    import samcloud.buildings.reconstruct as reconstruct
    monkeypatch.setattr(reconstruct, "reconstruct_building", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("forced reconstruction error")))
    result = run_environment(project, config=config, steps=["buildings"], resume=True)
    assert result["steps"]["buildings"]["status"] == "failed"
    assert "forced reconstruction error" in result["steps"]["buildings"]["error"]
    for name in ("terrain", "vegetation", "pavement"):
        assert result["steps"][name]["status"] == "done"
    assert result["steps"]["scene"]["status"] in {"done", "skipped"}
    if result["steps"]["scene"]["status"] == "done":
        assert (out / "environment.glb").is_file()
    report = json.loads((out / "environment_report.json").read_text(encoding="utf-8"))
    assert "forced reconstruction error" in report["steps"]["buildings"]["error"]
