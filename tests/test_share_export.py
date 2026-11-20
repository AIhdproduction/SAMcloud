"""Static share package: thinning, octree, legend, markers and privacy scan."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from samcloud.drone.contracts import PointSet
from samcloud.share import export
from samcloud.viewer import potree as local_potree

needs_converter = pytest.mark.skipif(not local_potree.POTREE_CONVERTER.is_file(), reason="PotreeConverter missing")

# internal class indices used by the synthetic cloud (outdoor class set) and their shares
_INTERNAL = np.array([1, 2, 3, 5], dtype=np.uint8)
_SHARES = np.array([0.55, 0.25, 0.15, 0.05])


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, internal: np.ndarray) -> None:
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("c", "u1")]
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar scalar_class\nend_header\n")
    data = np.zeros(len(xyz), dtype=dtype)
    data["x"], data["y"], data["z"] = xyz.T
    data["r"], data["g"], data["b"] = rgb.T
    data["c"] = internal
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii") + data.tobytes())


def make_project(root: Path, name: str = "synthetic", points: int = 300_000, *, refinement: str = "completed",
                 with_cloud: bool = True) -> Path:
    """A minimal project with a classified cloud (noisy, no grid ties), one AprilTag and a project yml."""
    project = (Path(os.path.realpath(root)) / name).resolve()
    for sub in ("products/classified", "reports", "control_points"):
        (project / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    if with_cloud:
        xyz = rng.uniform([0.0, 0.0, 0.0], [60.0, 40.0, 8.0], (points, 3)) + rng.normal(0, 0.013, (points, 3))
        internal = rng.choice(_INTERNAL, size=points, p=_SHARES)
        rgb = rng.integers(0, 255, (points, 3), dtype=np.uint8)
        _write_ply(project / "products" / "classified" / "classified.ply", xyz, rgb, internal)
        np.savez(project / "products" / "classified" / "classified_scores.npz",
                 top1_conf=rng.uniform(0.4, 1.0, points).astype(np.float16))
    report = {"tags": [{"id": 7, "center_xyz": [10.0, 12.0, 1.0], "status": "ok", "size_mm_after": 100.0,
                        "corners_xyz": [[9.9, 11.9, 1.0], [10.1, 11.9, 1.0], [10.1, 12.1, 1.0], [9.9, 12.1, 1.0]]}]}
    (project / "reports" / "apriltag_report.json").write_text(json.dumps(report), encoding="utf-8")
    document = {"project": {"name": "Testprojekt Muster", "updated_at": "2026-01-01T00:00:00+00:00"},
                "settings": {"classification": {"class_set": "outdoor"}},
                "workflow": {"refinement": refinement}}
    (project / "project.samcloud.yml").write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    return project


def _noop(fraction: float, message: str = "") -> None:
    return None


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the whole export once (≈ 300k points) and share the result between the tests of this module."""
    if not local_potree.POTREE_CONVERTER.is_file():
        pytest.skip("PotreeConverter missing")
    project = make_project(tmp_path_factory.mktemp("share_export"))
    package_dir = project / "products" / "share" / "package"
    started = time.perf_counter()
    points, original = export.prepare_points(project, 100_000, _noop, lambda: None)
    metadata = export.build_octree(points, project / "products" / "share" / "work", package_dir / "pointcloud",
                                   _noop, lambda: None)
    legend = export.assemble_package(project, package_dir, points=points, original_points=original, metadata=metadata)
    return {"project": project, "package": package_dir, "points": points, "original": original, "legend": legend,
            "metadata": metadata, "seconds": time.perf_counter() - started}


def _proportions(classes: np.ndarray) -> dict[int, float]:
    values, counts = np.unique(classes, return_counts=True)
    return {int(v): c / len(classes) for v, c in zip(values, counts)}


def test_thinning_keeps_class_proportions_and_limit() -> None:
    rng = np.random.default_rng(3)
    n = 200_000
    classes = rng.choice(np.array([2, 11, 73, 3], dtype=np.uint16), size=n, p=_SHARES)
    xyz = rng.uniform(0, 100, (n, 3)) + rng.normal(0, 0.011, (n, 3))
    cloud = PointSet(xyz=xyz, classes=classes, rgb=rng.integers(0, 255, (n, 3), dtype=np.uint8),
                     confidence=rng.uniform(0, 1, n).astype(np.float32))
    thinned = export.thin_points(cloud, 40_000)
    assert 20_000 <= len(thinned) <= 40_000
    before, after = _proportions(cloud.classes), _proportions(thinned.classes)
    assert set(before) == set(after), "every class survives"
    for code, share in before.items():
        assert abs(after[code] - share) <= 0.05
    assert thinned.rgb is not None and thinned.confidence is not None and len(thinned.rgb) == len(thinned)


def test_thinning_returns_small_cloud_unchanged() -> None:
    rng = np.random.default_rng(4)
    cloud = PointSet(xyz=rng.uniform(0, 5, (500, 3)), classes=np.full(500, 2, dtype=np.uint16))
    assert export.thin_points(cloud, 1000) is cloud


def test_rare_class_is_never_dropped() -> None:
    rng = np.random.default_rng(5)
    n = 60_000
    classes = np.full(n, 2, dtype=np.uint16)
    classes[:7] = 99
    cloud = PointSet(xyz=rng.uniform(0, 50, (n, 3)) + 0.0137, classes=classes)
    thinned = export.thin_points(cloud, 5_000)
    assert len(thinned) <= 5_000
    assert int(np.count_nonzero(thinned.classes == 99)) >= 1


@needs_converter
def test_package_contains_viewer_octree_legend_and_markers(package: dict) -> None:
    root: Path = package["package"]
    for relative in ("index.html", "js/share_viewer.js", "css/share.css", "lib/potree/potree.js",
                     "lib/potree/workers/2.0/DecoderWorker_brotli.js", "data/legend.json", "data/markers.json",
                     "pointcloud/metadata.json", "pointcloud/hierarchy.bin", "pointcloud/octree.bin"):
        assert (root / relative).is_file(), relative
    html = (root / "index.html").read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html, "no CDN or external references"
    legend = json.loads((root / "data" / "legend.json").read_text(encoding="utf-8"))
    assert legend["project_name"] == "Testprojekt Muster" and legend["generator"] == "SAMcloud"
    assert legend["points"] == len(package["points"]) <= 100_000
    names = {entry["code"]: entry["name"] for entry in legend["classes"]}
    assert names[2] == "Boden"  # German name, LAS code 2 (internal index 1)
    assert all(entry["color"].startswith("#") and len(entry["color"]) == 7 for entry in legend["classes"])
    assert sum(entry["count"] for entry in legend["classes"]) == legend["points"]
    markers = json.loads((root / "data" / "markers.json").read_text(encoding="utf-8"))["markers"]
    assert [m["name"] for m in markers] == ["TAG7"] and markers[0]["kind"] == "apriltag"
    assert set(markers[0]) <= {"name", "kind", "state_color", "coordinates", "corners"}


@needs_converter
def test_octree_has_rgb_classification_and_intensity(package: dict) -> None:
    metadata = package["metadata"]
    attributes = {attribute["name"] for attribute in metadata["attributes"]}
    assert {"position", "rgb", "classification", "intensity"} <= attributes
    assert metadata["points"] == len(package["points"])
    assert metadata["encoding"] == "BROTLI"
    assert not (package["project"] / "products" / "share" / "work" / "share.las").exists()


@needs_converter
def test_package_privacy_scan_is_clean_and_detects_violations(package: dict, tmp_path: Path) -> None:
    root: Path = package["package"]
    assert export.scan_package(root) == []
    size_mb = export.package_size_mb(root)
    print(f"synthetic package: {size_mb:.2f} MB, {package['seconds']:.1f} s")
    assert size_mb < 60
    dirty = tmp_path / "dirty"
    (dirty / "data").mkdir(parents=True)
    (dirty / "photo.jpg").write_bytes(b"\xff\xd8")
    (dirty / "data" / "a.json").write_text('{"p": "C:\\\\Users\\\\someone\\\\proj\\\\x"}', encoding="utf-8")
    (dirty / "data" / "b.json").write_text('{"p": "D:/SAMcloud/projects/x"}', encoding="utf-8")
    (dirty / "data" / "c.txt").write_text("see gps_ref.txt", encoding="utf-8")
    problems = "\n".join(export.scan_package(dirty))
    assert "image file: photo.jpg" in problems
    assert "absolute path: data/a.json" in problems and "absolute path: data/b.json" in problems
    assert "gps_ref reference: data/c.txt" in problems


@needs_converter
def test_package_has_no_forbidden_files_or_gps_data(package: dict) -> None:
    root: Path = package["package"]
    suffixes = {path.suffix.lower() for path in root.rglob("*")
                if path.is_file() and not path.relative_to(root).as_posix().startswith(export.RESOURCE_PREFIX)}
    assert not suffixes & {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".ply", ".las", ".laz", ".yml", ".yaml"}
    for path in root.rglob("*"):
        assert "gps_ref" not in path.name and "georeference" not in path.name
        if path.suffix in {".json", ".html", ".js"}:
            assert str(package["project"]) not in path.read_text(encoding="utf-8", errors="replace")
