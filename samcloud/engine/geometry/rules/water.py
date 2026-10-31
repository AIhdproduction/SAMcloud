"""Geometric plausibility check for the water class."""

import numpy as np

from .base import basic_hag


def passes(features, cfg):
    """Pass low, horizontal water using the ``max_hag`` and ``min_normal_z`` geometry.yml keys."""
    normal = ~np.isfinite(features["normal_z"]) | (features["normal_z"] >= cfg.get("min_normal_z", -float("inf")))
    return basic_hag(features, cfg) & normal
