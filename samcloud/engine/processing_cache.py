"""Processing cache: keep a run's intermediates so step 6 can be recomputed in minutes.

Enabled per project (storage setting ``keep_processing_cache``). After SAM the
classified cloud and its scores are copied (step 6 rewrites them in place) and
the dense workspace is hard-linked into ``<project>/processing_cache``: depth
and normal maps, undistorted images, the dense model, the fused cloud and the
SAM label rasters; the Depth-Anything priors follow once they exist. Files that
are only ever deleted are hard links, so the cache costs no extra space until
the normal clean-up removes the originals.

``--refine-from-cache`` then reruns workflow step 6 (and the geometric
refinement) from this state without COLMAP or SAM.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

CACHE_DIRECTORY = "processing_cache"
SAM_CLOUD = "classified_after_sam.ply"
SAM_SCORES = "classified_scores_after_sam.npz"
MANIFEST = "cache.json"
DENSE_TREES = ("stereo", "images", "sparse", "sparse_txt", "sam_labels")
DENSE_FILES = ("fused.ply", "fused.ply.vis")


def cache_directory(project_directory: Path) -> Path:
    return Path(project_directory) / CACHE_DIRECTORY


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:  # other volume or no hard-link support
        shutil.copy2(source, target)


def _link_tree(source: Path, target: Path) -> int:
    count = 0
    for path in source.rglob("*"):
        if path.is_file():
            _link_or_copy(path, target / path.relative_to(source))
            count += 1
    return count


def save_after_sam(
    project_directory: Path, dense_dir: Path, classified_raw: Path, scores_path: Path | None,
    *, class_set: str, capture_type: str | None,
) -> dict:
    """Store the SAM result and the dense workspace; replaces an older cache."""
    cache = cache_directory(project_directory)
    if cache.exists():
        shutil.rmtree(cache)
    (cache / "dense").mkdir(parents=True)
    shutil.copy2(classified_raw, cache / SAM_CLOUD)
    if scores_path is not None and Path(scores_path).is_file():
        shutil.copy2(scores_path, cache / SAM_SCORES)
    counts = {}
    for name in DENSE_FILES:
        if (dense_dir / name).is_file():
            _link_or_copy(dense_dir / name, cache / "dense" / name)
    for name in DENSE_TREES:
        if (dense_dir / name).is_dir():
            counts[name] = _link_tree(dense_dir / name, cache / "dense" / name)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "class_set": class_set,
        "capture_type": capture_type,
        "files": counts,
        "depth_prior": False,
    }
    (cache / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def save_depth_priors(project_directory: Path, dense_dir: Path) -> int:
    """Add the aligned Depth-Anything priors to an existing cache."""
    cache = cache_directory(project_directory)
    manifest_path = cache / MANIFEST
    if not manifest_path.is_file() or not (dense_dir / "depth_prior").is_dir():
        return 0
    count = _link_tree(dense_dir / "depth_prior", cache / "dense" / "depth_prior")
    if (dense_dir / "depth_prior.json").is_file():
        shutil.copy2(dense_dir / "depth_prior.json", cache / "dense" / "depth_prior.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["depth_prior"] = count > 0
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return count


def load_manifest(project_directory: Path) -> dict | None:
    """Return the cache manifest when step 6 can be recomputed from it, else None."""
    cache = cache_directory(project_directory)
    manifest_path = cache / MANIFEST
    if not manifest_path.is_file() or not (cache / SAM_CLOUD).is_file():
        return None
    if not (cache / "dense" / "stereo" / "depth_maps").is_dir() or not (cache / "dense" / "sparse").is_dir():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError:
        return None
