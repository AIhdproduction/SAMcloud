"""Geometric plausibility check for the low vegetation class."""

import numpy as np

from .base import at_most


def passes(features, cfg):
    """Pass low vegetation near terrain or near a local horizontal roof support."""
    hls = features.get("hls", np.full(len(features["hag"]), np.nan))
    threshold = cfg.get("max_height_above_support")
    near_support = np.zeros(len(hls), dtype=bool) if threshold is None else (
        np.isfinite(hls) & (hls <= threshold)
    )
    return at_most(features["hag"], cfg.get("max_hag")) | near_support
