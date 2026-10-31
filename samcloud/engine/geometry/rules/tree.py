"""Geometric plausibility check for the tree class."""

from .base import basic_hag


def passes(features, cfg):
    """Pass elevated canopy points using the ``min_hag`` geometry.yml key."""
    return basic_hag(features, cfg)
