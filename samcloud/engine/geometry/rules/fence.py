"""Geometric plausibility check for terrain and roof-supported fences."""

import numpy as np

from .base import at_most


def passes(features, cfg):
    """Pass low fences or roof railings close to their horizontal support."""
    hls = features.get("hls", np.full(len(features["hag"]), np.nan))
    threshold = cfg.get("max_height_above_support")
    near_support = np.zeros(len(hls), dtype=bool) if threshold is None else (
        np.isfinite(hls) & (hls <= threshold)
    )
    return at_most(features["hag"], cfg.get("max_hag")) | near_support
