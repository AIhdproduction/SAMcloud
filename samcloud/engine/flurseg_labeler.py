"""FlurSeg engine: own EoMT-DINOv3 semantic segmentation model as an outdoor alternative to SAM3.

The labeler has the same interface as :class:`sam_labeler.HeadFusionLabeler`
(``load_image`` and ``score_points``), so classify.py reuses the SegEarth-OV3
weighted multi-view voting, depth check, score export and step 6 unchanged.

Inference follows the training repository (flurseg ``predict.py``): RGB / 255,
ImageNet mean/std, sliding window (window = model image size, stride half a
window) with Hann blending, per-class scores from query class probabilities x
mask sigmoids. The scores are normalised to sum to one per pixel and summed into
the mapped SAMcloud classes (``config/flurseg.yml``); FlurSeg classes without a
SAMcloud counterpart ("other") keep their share as "no class".
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.nn import functional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "flurseg.yml"

_MODEL_CACHE: dict[tuple[str, str], torch.nn.Module] = {}


def load_config(path: Path | None = None) -> dict:
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def model_directory(config: dict) -> Path:
    directory = Path(config.get("model", {}).get("directory", ""))
    return directory if directory.is_absolute() else PROJECT_ROOT / directory


def availability(config: dict | None = None) -> str | None:
    """Return why FlurSeg cannot run, or None when model files and libraries are present."""
    config = config if config is not None else load_config()
    directory = model_directory(config)
    for name in ("config.json", "model.safetensors"):
        if not (directory / name).is_file():
            return f"FlurSeg model file missing: {directory / name}"
    try:
        import transformers  # noqa: F401
    except ImportError:
        return "FlurSeg needs the 'transformers' package (pip install transformers==5.22.1 scipy)"
    return None


def model_labels(directory: Path) -> list[str]:
    """FlurSeg class names in model output order (``id2label`` of config.json)."""
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    id2label = {int(key): value for key, value in config.get("id2label", {}).items()}
    return [id2label[index] for index in range(len(id2label))]


def build_mapping(flurseg_labels: list[str], class_names: list[str], class_mapping: dict) -> np.ndarray:
    """(num_samcloud_classes, num_flurseg_classes) 0/1 matrix that sums FlurSeg scores into SAMcloud classes."""
    short = [name.split(",", 1)[0].strip() for name in class_names]
    matrix = np.zeros((len(class_names), len(flurseg_labels)), dtype=np.float32)
    for source_index, source in enumerate(flurseg_labels):
        if source not in class_mapping:
            raise ValueError(f"config/flurseg.yml: class_mapping has no entry for FlurSeg class '{source}'")
        target = class_mapping[source]
        if target is None:
            continue
        if target not in short:
            raise ValueError(f"config/flurseg.yml: '{source}' maps to unknown SAMcloud class '{target}'")
        matrix[short.index(target), source_index] = 1.0
    return matrix


def semantic_scores(output, target_size: tuple[int, int]) -> torch.Tensor:
    """Per-class semantic scores (B, C, H, W) from EoMT query logits.

    The query masks are combined at their native resolution (a quarter of the
    window) and only the C class maps are upsampled - 100 query masks at full
    window size cost far more memory and time; the labels agree to 99.9 %.
    """
    classes = output.class_queries_logits.float().softmax(-1)[..., :-1]
    low = torch.einsum("bqc,bqhw->bchw", classes, output.masks_queries_logits.float().sigmoid())
    return functional.interpolate(low, size=target_size, mode="bilinear", align_corners=False)


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    last = max(length - window, 0)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def sliding_window_probabilities(model, image: torch.Tensor, window: int, stride: int, batch_size: int = 4,
                                 half: bool = False) -> torch.Tensor:
    """Hann-blended per-class probabilities (C, H, W) for a normalised (3, H, W) image on the model device.

    ``half`` runs the network under float16 autocast (CUDA only; about twice as
    fast on RTX GPUs, labels agree to 99.9 % with float32).
    """
    _, height, width = image.shape
    device = image.device
    hann = torch.outer(torch.hann_window(window, device=device), torch.hann_window(window, device=device)) + 1e-6
    boxes = [(top, left) for top in _window_starts(height, window, stride) for left in _window_starts(width, window, stride)]
    score_sum = None
    weights = torch.zeros((height, width), device=device)
    with torch.inference_mode():
        for start in range(0, len(boxes), max(1, batch_size)):
            batch_boxes = boxes[start:start + max(1, batch_size)]
            tiles = []
            for top, left in batch_boxes:
                tile = image[:, top:top + window, left:left + window]
                tiles.append(functional.pad(tile, (0, window - tile.shape[-1], 0, window - tile.shape[-2])))
            with torch.autocast("cuda", dtype=torch.float16, enabled=half and device.type == "cuda"):
                output = model(pixel_values=torch.stack(tiles))
            scores = semantic_scores(output, (window, window))
            if score_sum is None:
                score_sum = torch.zeros((scores.shape[1], height, width), device=device)
            for tile_scores, (top, left) in zip(scores, batch_boxes):
                bottom, right = min(top + window, height), min(left + window, width)
                blend = hann[: bottom - top, : right - left]
                score_sum[:, top:bottom, left:right] += tile_scores[:, : bottom - top, : right - left] * blend
                weights[top:bottom, left:right] += blend
    scores = score_sum / weights.clamp_min(1e-6)
    return scores / scores.sum(dim=0, keepdim=True).clamp_min(1e-6)


def _load_model(directory: Path, device: str):
    key = (str(directory), device)
    if key not in _MODEL_CACHE:
        from transformers import EomtDinov3ForUniversalSegmentation

        _MODEL_CACHE[key] = EomtDinov3ForUniversalSegmentation.from_pretrained(str(directory)).eval().to(device)
    return _MODEL_CACHE[key]


class FlurSegLabeler:
    """Score projected points with the FlurSeg model; drop-in for HeadFusionLabeler."""

    def __init__(self, class_names, unknown_id=0, config_path=None, device=None, model=None):
        config = load_config(config_path)
        self.class_names = list(class_names)
        self.unknown_id = unknown_id
        model_config = config.get("model", {})
        self.directory = model_directory(config)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            reason = availability(config)
            if reason:
                raise RuntimeError(reason)
            model = _load_model(self.directory, self.device)
            labels = model_labels(self.directory)
        else:
            labels = list(getattr(model, "flurseg_labels"))
        self.model = model
        self.flurseg_labels = labels
        self.window = int(model_config.get("window_size", 768))
        image_size = getattr(getattr(model, "config", None), "image_size", None)
        if image_size is not None and int(image_size) != self.window:
            raise ValueError(f"FlurSeg window_size {self.window} must equal the model image size {image_size}")
        self.stride = int(model_config.get("stride", self.window // 2))
        self.batch_size = int(model_config.get("batch_size", 4))
        self.half = str(model_config.get("precision", "fp16")).lower() == "fp16"
        preprocessor = self.directory / "preprocessor_config.json"
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if preprocessor.is_file():
            values = json.loads(preprocessor.read_text(encoding="utf-8"))
            mean, std = values.get("image_mean", mean), values.get("image_std", std)
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device).view(3, 1, 1)
        mapping = build_mapping(labels, self.class_names, config.get("class_mapping", {}))
        self.target_columns = np.flatnonzero(mapping.any(axis=1))
        self.mapping = torch.from_numpy(mapping[self.target_columns]).to(self.device)

    @staticmethod
    def load_image(image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Could not read image {image_path}")
        height, width = img.shape[:2]
        return img, width, height

    def _target_probabilities(self, image_bgr: np.ndarray) -> torch.Tensor:
        """Probabilities of the mapped SAMcloud classes (T, H, W) for a whole image."""
        rgb = torch.from_numpy(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).to(self.device)
        tensor = (rgb.permute(2, 0, 1).float() / 255.0 - self.mean) / self.std
        probs = sliding_window_probabilities(self.model, tensor, self.window, self.stride, self.batch_size, self.half)
        del tensor
        return torch.einsum("tc,chw->thw", self.mapping, probs)

    def score_points(self, image_path, u, v, image=None, label_threshold=None):
        """(N, num_classes) probabilities at integer (u, v); with label_threshold also an HxW label raster."""
        if image is None:
            image, width, height = self.load_image(image_path)
        else:
            height, width = image.shape[:2]
        u = np.asarray(u, dtype=np.int64)
        v = np.asarray(v, dtype=np.int64)
        if u.shape != v.shape:
            raise ValueError("u and v must have the same shape")
        if np.any((u < 0) | (u >= width) | (v < 0) | (v >= height)):
            raise ValueError("score_points requires in-bounds pixel coordinates")
        target = self._target_probabilities(image)
        scores = np.zeros((u.size, len(self.class_names)), dtype=np.float32)
        if u.size:
            sampled = target[:, torch.from_numpy(v).to(self.device), torch.from_numpy(u).to(self.device)]
            scores[:, self.target_columns] = sampled.transpose(0, 1).float().cpu().numpy()
        if label_threshold is None:
            return scores
        best, index = target.max(dim=0)
        lookup = torch.from_numpy(self.target_columns.astype(np.int64) + 1).to(self.device)
        labels = torch.where(best >= float(label_threshold), lookup[index], torch.zeros_like(index))
        return scores, labels.cpu().numpy().astype(np.uint8)
