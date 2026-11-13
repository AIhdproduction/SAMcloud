"""Bounded LRU access to RGB source photographs."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import torch


class PhotoLoader:
    """Load original photos as device-local RGB uint8 tensors.

    Sampling uses ``grid_sample(..., align_corners=False)``: image pixel centre
    ``(u, v)`` is represented by normalized coordinates
    ``((u + 0.5) * 2 / width - 1, (v + 0.5) * 2 / height - 1)``.
    """

    def __init__(self, device: str, capacity: int = 4) -> None:
        self.device = torch.device(device)
        self.capacity = max(1, capacity)
        self._images: OrderedDict[Path, torch.Tensor] = OrderedDict()

    def get(self, path: Path) -> torch.Tensor:
        key = Path(path)
        cached = self._images.pop(key, None)
        if cached is not None:
            self._images[key] = cached
            return cached
        image = cv2.imread(str(key), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Photo could not be read: {key}")
        tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(self.device)
        self._images[key] = tensor
        if len(self._images) > self.capacity:
            self._images.popitem(last=False)
        return tensor
