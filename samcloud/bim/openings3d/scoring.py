"""SAM3 window / door / wall / furniture score maps per photograph, cached on disk.

The maps are produced by the repo's SAM3 head-fusion labeler (``HeadFusionLabeler._score_tile``, the
same path the point classification uses) and stored as uint8 (classes, R, R) arrays under
``<project>/products/bim/openings_cache/scores_<fingerprint>/``.  The folder name carries the
fingerprint of everything that changes the maps (prompts, class list, resolution, model file), so a
changed setting writes a new folder and never reuses stale maps; nothing is ever deleted here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from samcloud.dxf.cancel import Cancelled
from samcloud.ortho.cameras import OrthoCamera

from .config import Settings

_LOGGER = logging.getLogger(__name__)
ProgressFn = Callable[[float, str], None]
CACHE_ROOT_NAME = "openings_cache"
MANIFEST_NAME = "index.json"


class Scorer(Protocol):
    """Returns per-class probabilities (C, H, W) in [0, 1] for a BGR image."""

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray: ...


def cache_root(project_root: Path | str) -> Path:
    return Path(project_root) / "products" / "bim" / CACHE_ROOT_NAME


def _model_stamp() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[3] / "models" / "sam3.pt"
    try:
        stat = path.stat()
        return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        return {"name": path.name}


def scores_fingerprint(config: Settings) -> str:
    """Hash of every setting that changes the score maps."""
    payload = {"sam": config.sam.to_dict(), "model": _model_stamp(), "version": 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def scores_directory(project_root: Path | str, config: Settings) -> Path:
    return cache_root(project_root) / f"scores_{scores_fingerprint(config)}"


class Sam3Scorer:
    """Lazy SAM3 head-fusion scorer (loads the model on first use; needs CUDA)."""

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._labeler: Any = None

    def _ensure(self) -> Any:
        if self._labeler is None:
            from samcloud.engine.sam_labeler import HeadFusionLabeler

            self._labeler = HeadFusionLabeler(list(self._config.sam.classes), prompts=[list(group) for group in self._config.sam.prompts])
        return self._labeler

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        labeler = self._ensure()
        resolution = int(self._config.sam.map_resolution)
        with torch.no_grad():
            probs = labeler._score_tile(image_bgr)  # (C, H, W) float
            probs = functional.interpolate(probs[None].float(), (resolution, resolution), mode="area")[0]
        return probs.clamp(0, 1).cpu().numpy()


def _image_stamp(camera: OrthoCamera) -> list[int]:
    stat = Path(camera.image_path).stat()  # type: ignore[arg-type]
    return [int(stat.st_size), int(stat.st_mtime_ns)]


def _load_manifest(directory: Path) -> dict[str, list[int]]:
    try:
        return {str(key): list(value) for key, value in json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8")).items()}
    except (OSError, ValueError):
        return {}


def _write_manifest(directory: Path, manifest: dict[str, list[int]]) -> None:
    target = directory / MANIFEST_NAME
    temporary = directory / f".{MANIFEST_NAME}.tmp"
    temporary.write_text(json.dumps(manifest), encoding="utf-8")
    os.replace(temporary, target)


def score_images(
    keys: list[str],
    cameras: dict[str, OrthoCamera],
    directory: Path,
    config: Settings,
    scorer: Scorer,
    *,
    progress: ProgressFn | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict[str, Path]:
    """Make sure a score map exists for every key; returns ``{key: .npy path}``. Cached maps are reused."""
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(directory)
    paths: dict[str, Path] = {}
    pending = []
    for key in keys:
        path = directory / f"{key}.npy"
        paths[key] = path
        if path.is_file() and manifest.get(key) == _image_stamp(cameras[key]):
            continue
        pending.append(key)
    started = time.time()
    for index, key in enumerate(pending):
        if cancel is not None and cancel():
            _write_manifest(directory, manifest)
            raise Cancelled("Opening refinement cancelled")
        image = cv2.imread(str(cameras[key].image_path))
        if image is None:
            raise RuntimeError(f"Could not read image {cameras[key].image_path}")
        probs = scorer(image)
        temporary = directory / f".{key}.tmp.npy"
        np.save(temporary, (np.clip(probs, 0.0, 1.0) * 255.0).round().astype(np.uint8))
        os.replace(temporary, paths[key])
        manifest[key] = _image_stamp(cameras[key])
        if progress is not None:
            progress((index + 1) / max(len(pending), 1), f"Scoring photographs {index + 1}/{len(pending)}")
        if (index + 1) % 25 == 0:
            _write_manifest(directory, manifest)
    if pending:
        _write_manifest(directory, manifest)
        _LOGGER.info("openings3d: scored %d photographs in %.0f s (%d cached)", len(pending), time.time() - started, len(keys) - len(pending))
    return paths


def load_score_map(path: Path) -> np.ndarray:
    """uint8 (classes, R, R)."""
    return np.load(path)
