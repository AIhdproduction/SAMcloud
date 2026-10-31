"""Geometric plausibility checks for the building class."""

import numpy as np

from .base import at_least, at_most


def passes(features, cfg):
    """Pass elevated buildings or low vertical facades using ``min_hag`` and ``or_vertical``."""
    vertical = cfg.get("or_vertical", {})
    main = at_least(features["hag"], cfg.get("min_hag"))
    alternate = (
        at_most(features["normal_z"], vertical.get("max_normal_z"))
        & at_least(features["hag"], vertical.get("min_hag"))
    )
    return main | alternate


def flat_roof_passes(features, cfg):
    """Convert strong roofs always and ambiguous roofs only at low SAM confidence."""
    hag = features["hag"]
    normal_z = features["normal_z"]
    planarity = features["planarity"]
    geometry = (
        np.isfinite(hag)
        & np.isfinite(normal_z)
        & np.isfinite(planarity)
        & (hag >= cfg.get("min_hag", -np.inf))
        & (normal_z >= cfg.get("min_normal_z", -np.inf))
        & (planarity >= cfg.get("min_planarity", -np.inf))
    )
    strong = hag >= cfg.get("strong_min_hag", np.inf)
    confidence = features.get("confidence", np.full(len(hag), np.nan))
    ambiguous = ~np.isfinite(confidence) | (confidence < cfg.get("ambiguous_max_confidence", 1.0))
    return geometry & (strong | ambiguous)
