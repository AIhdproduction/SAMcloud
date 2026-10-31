"""Geometric plausibility check for the sidewalk class."""

import numpy as np

from .base import at_most, basic_hag


def passes(features, cfg):
    """Pass low, horizontal and smooth sidewalks using ``max_hag``, ``min_normal_z`` and ``max_roughness``."""
    normal = ~np.isfinite(features["normal_z"]) | (features["normal_z"] >= cfg.get("min_normal_z", -float("inf")))
    return basic_hag(features, cfg) & at_most(features["roughness"], cfg.get("max_roughness")) & normal
