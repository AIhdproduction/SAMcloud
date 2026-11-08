"""Safe storage compaction helpers for completed SAMcloud projects."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path

from samcloud.core.sparse_model import resolve_sparse_model


DEFAULT_STORAGE = {
    "keep_control_point_inputs": True,
    "keep_classification_inputs": True,
    "keep_refinement_inputs": True,
    # Keeps depth maps, images, SAM label rasters and Depth-Anything priors
    # (~40-60 GB for an indoor 360 project) so step 6 can be re-run in minutes.
    "keep_processing_cache": False,
}


def storage_settings(project_settings: dict) -> dict:
    """Return storage settings with safe defaults for older projects."""
    if isinstance(project_settings, dict):
        flat = {
            key: bool(project_settings[key])
            for key in DEFAULT_STORAGE
            if key in project_settings
        }
        if flat:
            return {**DEFAULT_STORAGE, **flat}
    source = project_settings.get("settings", project_settings) if isinstance(project_settings, dict) else {}
    configured = source.get("storage", {}) if isinstance(source, dict) else {}
    return {**DEFAULT_STORAGE, **configured} if isinstance(configured, dict) else dict(DEFAULT_STORAGE)


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def _temporary_path(path: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    return Path(temporary_name)


def compress_file(path: Path, *, level: int = 3) -> Path:
    """Gzip a file atomically and retain the compressed copy for future re-runs."""
    path = Path(path)
    compressed = path.with_name(path.name + ".gz")
    if not path.exists():
        if compressed.exists():
            return compressed
        raise FileNotFoundError(path)
    temporary = _temporary_path(compressed)
    try:
        with path.open("rb") as source, temporary.open("wb") as file:
            with gzip.GzipFile(fileobj=file, mode="wb", compresslevel=level) as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            file.flush()
            os.fsync(file.fileno())
        if temporary.stat().st_size <= 0:
            raise OSError(f"Compressed file is empty: {compressed}")
        os.replace(temporary, compressed)
        path.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return compressed


def ensure_available(path: Path) -> Path:
    """Restore a compressed re-run input when the uncompressed path is needed."""
    path = Path(path)
    if path.exists():
        return path
    compressed = path.with_name(path.name + ".gz")
    if not compressed.exists():
        raise FileNotFoundError(path)
    temporary = _temporary_path(path)
    try:
        with gzip.open(compressed, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _validated_generated_path(path: Path, products_dir: Path) -> tuple[Path, Path]:
    products = Path(products_dir).resolve()
    candidate = Path(path)
    resolved = candidate.resolve()
    if products.name != "products" or not (products.parent / "project.samcloud.yml").is_file():
        raise ValueError("Generated paths require a project's products directory")
    try:
        resolved.relative_to(products)
    except ValueError as error:
        raise ValueError("Refusing to remove a path outside products") from error
    if resolved == products:
        raise ValueError("Refusing to remove the products directory itself")
    if candidate.is_symlink() or os.path.realpath(candidate) != str(resolved):
        raise ValueError("Refusing to remove a symlink or junction")
    return resolved, products


def remove_generated(path: Path, products_dir: Path) -> int:
    """Remove one validated generated file or directory and return freed bytes."""
    target, _products = _validated_generated_path(path, products_dir)
    if not target.exists():
        return 0
    freed = _size(target)
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return freed


def _remove(path: Path, products_dir: Path, deleted: list[str]) -> int:
    if not path.exists() and not path.is_symlink():
        return 0
    freed = remove_generated(path, products_dir)
    if freed or not path.exists():
        deleted.append(str(path.relative_to(products_dir)))
    return freed


def release_after_fusion(dense_dir: Path) -> int:
    """Release COLMAP's regenerable depth, normal, and consistency caches."""
    dense_dir = Path(dense_dir)
    products = dense_dir.parent
    freed = 0
    for name in ("depth_maps", "normal_maps", "consistency_graphs"):
        path = dense_dir / "stereo" / name
        if path.exists() or path.is_symlink():
            freed += remove_generated(path, products)
    return freed


def ensure_text_model(binary_dir: Path, text_dir: Path) -> Path:
    """Regenerate a COLMAP text model only when it is not already available."""
    binary_dir, text_dir = Path(binary_dir), Path(text_dir)
    required = ("cameras.txt", "images.txt", "points3D.txt")
    if all((text_dir / name).is_file() for name in required):
        return text_dir
    binary_required = ("cameras.bin", "images.bin", "points3D.bin")
    if not all((binary_dir / name).is_file() for name in binary_required):
        return text_dir
    import pycolmap

    text_dir.mkdir(parents=True, exist_ok=True)
    pycolmap.Reconstruction(binary_dir).write_text(text_dir)
    return text_dir


def compact_project(products_dir: Path, settings: dict) -> dict:
    """Compact a completed project without touching final deliverables."""
    products = Path(products_dir).resolve()
    _validated_generated_path(products / "dense", products)
    settings = storage_settings(settings)
    before_bytes = _size(products)
    deleted: list[str] = []
    compressed: list[str] = []
    freed = 0
    dense = products / "dense"
    sparse = products / "sparse"
    classified = products / "classified"

    for path in (dense / "stereo", dense / "images", dense / "sparse_txt", dense / "sparse_pre_gcp"):
        freed += _remove(path, products, deleted)
    for path in dense.glob("sparse_txt_*"):
        freed += _remove(path, products, deleted)
    for path in dense.glob("sparse_gcp*"):
        freed += _remove(path, products, deleted)
    for path in dense.glob("run-colmap-*.sh"):
        freed += _remove(path, products, deleted)

    final_model = resolve_sparse_model(sparse)
    for path in sparse.iterdir() if sparse.is_dir() else ():
        if path.is_dir() and path != final_model and path.name != "pre_gcp":
            freed += _remove(path, products, deleted)
    freed += _remove(sparse / "pre_gcp_txt", products, deleted)
    freed += _remove(classified / "classified_alt.ply", products, deleted)
    potree = products / "viewer" / "potree"
    for path in potree.glob("*.las") if potree.is_dir() else ():
        freed += _remove(path, products, deleted)

    keep_or_remove = (
        ("keep_control_point_inputs", (sparse / "pre_gcp", products / "database.db")),
        ("keep_classification_inputs", (dense / "fused.ply", dense / "fused.ply.vis", dense / "sparse")),
        ("keep_refinement_inputs", (classified / "classified_raw.ply", classified / "classified_scores.npz")),
    )
    for option, paths in keep_or_remove:
        if settings[option]:
            for path in paths:
                if path.is_file():
                    original_size = path.stat().st_size
                    compressed_path = compress_file(path)
                    freed += max(0, original_size - compressed_path.stat().st_size)
                    compressed.append(str(compressed_path.relative_to(products)))
        else:
            for path in paths:
                freed += _remove(path, products, deleted)

    after_bytes = _size(products)
    result = {
        "freed_bytes": freed,
        "deleted": deleted,
        "compressed": compressed,
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
    }
    report = products.parent / "reports" / "storage_summary.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Storage compaction freed {freed / 1024 ** 2:.1f} MB.")
    return result
