"""Geometric plausibility check for the overhead power line class."""

from .base import at_least, at_most


def passes(features, cfg):
    """Pass elevated, non-planar or sparse cable points."""
    return at_least(features["hag"], cfg.get("min_hag")) & at_most(
        features["planarity"], cfg.get("max_planarity")
    )
