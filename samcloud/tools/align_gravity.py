"""Apply panorama gravity alignment to an already completed project."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData

from samcloud.colmap.gravity import GravityAlignmentError, align_sparse_model
from samcloud.core.sparse_model import resolve_sparse_model
from samcloud.core.storage import ensure_available


def _backup_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.pre_gravity{path.suffix}")


def _rotate_ply(path: Path, rotation: np.ndarray, translation: np.ndarray) -> bool:
    """Back up one PLY, rotate coordinates and normals, and retain gzip copies."""
    compressed = path.with_name(path.name + ".gz")
    backup = _backup_path(path)
    was_compressed = compressed.exists()
    if not path.exists() and not was_compressed:
        return False
    if was_compressed:
        shutil.copy2(compressed, backup.with_name(backup.name + ".gz"))
    ensure_available(path)
    shutil.copy2(path, backup)
    # plyfile defaults to a copy-on-write memory map for binary PLY files.
    # Windows keeps that mapping open, so writing this same path would fail.
    # The rotation already needs the complete vertex element in memory; disable
    # mmap explicitly to detach it from the source file before overwriting it.
    ply = PlyData.read(str(path), mmap=False)
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    required = {"x", "y", "z"}
    if not required.issubset(names):
        raise ValueError(f"PLY has no xyz vertex properties: {path}")
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(float)
    xyz = (rotation @ xyz.T).T + translation
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for normal_names in (("nx", "ny", "nz"), ("normal_x", "normal_y", "normal_z")):
        if set(normal_names).issubset(names):
            normals = np.column_stack([vertex[name] for name in normal_names]).astype(float)
            normals = (rotation @ normals.T).T
            for index, name in enumerate(normal_names):
                vertex[name] = normals[:, index]
    ply.write(str(path))
    if was_compressed:
        with path.open("rb") as source, gzip.open(compressed, "wb", compresslevel=3) as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    return True


def align_project(project: Path) -> dict[str, object]:
    """Align final sparse and displayed dense/classified point clouds in place."""
    project = Path(project).resolve()
    products = project / "products"
    sparse = products / "sparse"
    source = resolve_sparse_model(sparse)
    if source is None:
        raise GravityAlignmentError(f"No usable final sparse model found in {sparse}")
    destination = sparse / "0_gravity"
    if source == destination:
        raise GravityAlignmentError("Project already selects the gravity-aligned sparse model")
    report = align_sparse_model(source, destination)
    (sparse / "final_model.txt").write_text(destination.name + "\n", encoding="utf-8")
    rotation = np.asarray(report["rotation"], dtype=float)
    translation = np.asarray(report["translation"], dtype=float)
    rotated = [
        path.relative_to(project).as_posix()
        for path in (
            products / "dense" / "fused.ply",
            products / "classified" / "classified_raw.ply",
            products / "classified" / "classified.ply",
        )
        if _rotate_ply(path, rotation, translation)
    ]
    report["rotated_point_clouds"] = rotated
    report_path = project / "reports" / "gravity_alignment.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Align a completed panorama project to gravity")
    parser.add_argument("--project", required=True, type=Path, help="Project directory containing products/")
    args = parser.parse_args()
    report = align_project(args.project)
    print(
        f"Gravity alignment wrote {report['aligned_model']} from {report['frame_count']} rig frames "
        f"(spread {report['up_spread_deg_rms']:.3f} degrees)."
    )


if __name__ == "__main__":
    main()
