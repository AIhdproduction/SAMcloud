"""Resolve the final sparse reconstruction for project products."""

from __future__ import annotations

from pathlib import Path


def resolve_sparse_model(sparse_directory: Path) -> Path | None:
    """Return the preferred sparse model directory with non-empty points."""
    sparse_directory = Path(sparse_directory)
    candidates = [sparse_directory / "gcp_adjusted"]
    configured = _configured_final_model(sparse_directory)
    if configured is not None:
        candidates.append(configured)
    candidates.extend(sparse_directory / name for name in ("0_geo", "0"))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _has_points(candidate):
            return candidate

    try:
        remaining = sorted(
            (path for path in sparse_directory.iterdir() if path.is_dir() and path not in seen),
            key=lambda path: path.name,
        )
    except OSError:
        return None
    return next((path for path in remaining if _has_points(path)), None)


def sparse_points_path(sparse_directory: Path) -> Path | None:
    """Return the preferred non-empty ``points3D.bin`` file, if available."""
    model = resolve_sparse_model(sparse_directory)
    return model / "points3D.bin" if model is not None else None


def _configured_final_model(sparse_directory: Path) -> Path | None:
    try:
        configured = (sparse_directory / "final_model.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    configured_path = Path(configured)
    if not configured or configured_path.is_absolute() or configured_path.name != configured:
        return None
    return sparse_directory / configured_path


def _has_points(model_directory: Path) -> bool:
    points = model_directory / "points3D.bin"
    try:
        return points.is_file() and points.stat().st_size > 0
    except OSError:
        return False
