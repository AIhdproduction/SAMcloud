"""Depth-Anything-V2 depth priors aligned to the COLMAP reconstruction (indoor).

COLMAP measures depth only where there is texture; white walls, ceilings and
floors stay empty. Depth-Anything-V2 (metric, Hypersim indoor model) predicts a
dense depth for every pixel of a photo, but in its own, slightly wrong scale.
Each prediction is therefore aligned per view to COLMAP's own filtered
(geometric) depth map with a robust linear fit ``colmap ~= scale * da + shift``
on the pixels both measured. A view whose fit does not agree with COLMAP within
``max_fit_error`` is not used at all.

The aligned depth maps are cached as float16 ``.npy`` files at depth-map
resolution in ``dense/depth_prior/`` and read by the point cleaning and the
plane fill. They are temporary and removed with the other dense caches.

Model code: https://github.com/DepthAnything/Depth-Anything-V2 (Apache-2.0),
expected in ``third_party/Depth-Anything-V2``. Weights (``models/
depth_anything_v2``): Large/Base are CC-BY-NC-4.0, Small is Apache-2.0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from samcloud.colmap.plane_fill import View, read_colmap_array
from samcloud.core.paths import MODELS_DIRECTORY, PROJECT_ROOT

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "small": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "base": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
CODE_DIRECTORY = PROJECT_ROOT / "third_party" / "Depth-Anything-V2" / "metric_depth"
WEIGHTS_DIRECTORY = MODELS_DIRECTORY / "depth_anything_v2"

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "model": "large",
    "input_size": 518,
    "max_depth": 20.0,
    "min_fit_pixels": 2000,
    "fit_inlier_error": 0.05,
    "max_fit_error": 0.06,
}


def weights_path(model: str) -> Path:
    encoder = MODEL_CONFIGS[model]["encoder"]
    return WEIGHTS_DIRECTORY / f"depth_anything_v2_metric_hypersim_{encoder}.pth"


def availability(model: str) -> str | None:
    """Return None when code and weights are present, otherwise a short reason."""
    if model not in MODEL_CONFIGS:
        return f"unknown Depth-Anything-V2 model '{model}'"
    if not (CODE_DIRECTORY / "depth_anything_v2" / "dpt.py").is_file():
        return f"Depth-Anything-V2 code missing in {CODE_DIRECTORY}"
    if not weights_path(model).is_file():
        return f"Depth-Anything-V2 weights missing: {weights_path(model)}"
    return None


class DepthAnythingEstimator:
    """Metric indoor Depth-Anything-V2 on the GPU (falls back to CPU)."""

    def __init__(self, model: str = "large", *, input_size: int = 518, max_depth: float = 20.0) -> None:
        import torch

        reason = availability(model)
        if reason:
            raise FileNotFoundError(reason)
        if str(CODE_DIRECTORY) not in sys.path:
            sys.path.insert(0, str(CODE_DIRECTORY))
        from depth_anything_v2.dpt import DepthAnythingV2

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_size = int(input_size)
        network = DepthAnythingV2(**{**MODEL_CONFIGS[model], "max_depth": float(max_depth)})
        network.load_state_dict(torch.load(str(weights_path(model)), map_location="cpu"))
        self.network = network.to(self.device).eval()

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return an HxW float32 depth map (metres in the model's scale)."""
        return np.asarray(self.network.infer_image(image_bgr, self.input_size), dtype=np.float32)


def align_to_colmap(
    prediction: np.ndarray, colmap_depth: np.ndarray, settings: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, float]]:
    """Fit ``colmap ~= scale * prediction + shift`` robustly; None when it does not fit."""
    import cv2

    height, width = colmap_depth.shape[:2]
    if prediction.shape[:2] != (height, width):
        prediction = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_LINEAR)
    valid = (colmap_depth > 0) & (prediction > 0)
    info: dict[str, float] = {"pixels": float(np.count_nonzero(valid))}
    if info["pixels"] < int(settings["min_fit_pixels"]):
        return None, info
    target = colmap_depth[valid].astype(np.float64)
    source = prediction[valid].astype(np.float64)
    scale, shift = float(np.median(target / source)), 0.0
    inliers = np.ones(len(target), dtype=bool)
    for _ in range(4):
        relative = np.abs(scale * source + shift - target) / target
        inliers = relative <= float(settings["fit_inlier_error"]) * 2
        if np.count_nonzero(inliers) < int(settings["min_fit_pixels"]) // 2:
            return None, info
        design = np.column_stack((source[inliers], np.ones(np.count_nonzero(inliers))))
        scale, shift = (float(value) for value in np.linalg.lstsq(design, target[inliers], rcond=None)[0])
        if scale <= 0:
            return None, info
    error = float(np.median(np.abs(scale * source + shift - target) / target))
    info.update({"scale": scale, "shift": shift, "median_error": error,
                 "inlier_share": float(np.mean(np.abs(scale * source + shift - target) / target
                                               <= float(settings["fit_inlier_error"])))})
    if error > float(settings["max_fit_error"]):
        return None, info
    aligned = (scale * prediction + shift).astype(np.float32)
    aligned[aligned <= 0] = 0.0
    return aligned, info


def prior_path(prior_dir: Path, name: str) -> Path:
    return Path(prior_dir) / f"{name}.npy"


def _apply_fit(prediction: np.ndarray, shape: tuple[int, int], scale: float, shift: float) -> np.ndarray:
    import cv2

    height, width = shape
    if prediction.shape[:2] != (height, width):
        prediction = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_LINEAR)
    aligned = (scale * prediction + shift).astype(np.float32)
    aligned[aligned <= 0] = 0.0
    return aligned


def _global_check(aligned: np.ndarray, colmap_depth: np.ndarray, settings: dict[str, Any]) -> bool:
    """A view aligned with the global fit must not contradict COLMAP where COLMAP measured."""
    valid = (colmap_depth > 0) & (aligned > 0)
    if np.count_nonzero(valid) < 50:
        return True
    error = np.abs(aligned[valid] - colmap_depth[valid]) / colmap_depth[valid]
    return float(np.median(error)) <= 1.5 * float(settings["max_fit_error"])


def compute_depth_priors(
    dense_dir: Path,
    views: list[View],
    settings: dict[str, Any] | None = None,
    *,
    estimator: Any | None = None,
    image_reader: Callable[[Path], np.ndarray | None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Predict, align and cache one depth prior per dense view.

    Pass 1 fits every view to its own COLMAP depth. Views with too little
    COLMAP depth (typically white walls) get the median fit of all good views
    in pass 2, provided they do not contradict the COLMAP pixels they have.
    """
    import cv2

    settings = {**DEFAULTS, **(settings or {})}
    dense_dir = Path(dense_dir)
    prior_dir = dense_dir / "depth_prior"
    prior_dir.mkdir(parents=True, exist_ok=True)
    if estimator is None:
        estimator = DepthAnythingEstimator(
            str(settings["model"]), input_size=int(settings["input_size"]), max_depth=float(settings["max_depth"]),
        )
    reader = image_reader or (lambda path: cv2.imread(str(path), cv2.IMREAD_COLOR))
    total = 2 * len(views)
    fits, errors, retry = [], [], []
    skipped = 0

    def save(view: View, aligned: np.ndarray) -> None:
        target = prior_path(prior_dir, view.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, aligned.astype(np.float16))

    for number, view in enumerate(views, start=1):
        geometric = dense_dir / "stereo" / "depth_maps" / f"{view.name}.geometric.bin"
        image = reader(dense_dir / "images" / view.name)
        if image is None or not geometric.is_file():
            skipped += 1
        else:
            aligned, info = align_to_colmap(estimator.predict(image), read_colmap_array(geometric), settings)
            if aligned is None:
                retry.append(view)
            else:
                save(view, aligned)
                fits.append((info["scale"], info["shift"]))
                errors.append(info["median_error"])
        if progress is not None:
            progress(number, total)
    global_fit = None
    fallback = 0
    if fits:
        global_fit = (float(np.median([fit[0] for fit in fits])), float(np.median([fit[1] for fit in fits])))
        for number, view in enumerate(retry, start=len(views) + 1):
            geometric = read_colmap_array(dense_dir / "stereo" / "depth_maps" / f"{view.name}.geometric.bin")
            image = reader(dense_dir / "images" / view.name)
            aligned = _apply_fit(estimator.predict(image), geometric.shape[:2], *global_fit)
            if _global_check(aligned, geometric, settings):
                save(view, aligned)
                fallback += 1
            else:
                skipped += 1
            if progress is not None:
                progress(number, total)
    else:
        skipped += len(retry)
    if progress is not None:
        progress(total, total)
    report = {
        "model": settings["model"],
        "views": len(views),
        "aligned_views": len(fits),
        "global_fit_views": fallback,
        "skipped_views": skipped,
        "global_fit": global_fit,
        "median_fit_error": round(float(np.median(errors)), 4) if errors else None,
    }
    (dense_dir / "depth_prior.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def prior_reader(dense_dir: Path) -> Callable[[str], np.ndarray | None]:
    """Return name -> aligned prior depth (float32) or None."""
    prior_dir = Path(dense_dir) / "depth_prior"

    def read(name: str) -> np.ndarray | None:
        path = prior_path(prior_dir, name)
        return np.load(path).astype(np.float32) if path.is_file() else None

    return read
