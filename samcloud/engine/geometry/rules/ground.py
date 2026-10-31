"""Geometric plausibility check for the ground class."""

from .base import basic_hag


def passes(features, cfg):
    """Pass terrain-height points using the ``max_hag`` geometry.yml key."""
    return basic_hag(features, cfg)
