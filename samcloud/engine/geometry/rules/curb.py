"""Geometric plausibility check for the curb class."""

import numpy as np

from .base import at_least, basic_hag


def passes(features, cfg):
    """Pass low curbs only when valid road or sidewalk points are adjacent."""
    adjacent = features.get("adjacent_road_sidewalk", np.zeros(len(features["hag"])))
    return basic_hag(features, cfg) & at_least(adjacent, cfg.get("min_adjacent_points", 3))
