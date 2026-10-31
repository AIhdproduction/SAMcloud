"""Geometric plausibility check for the construction container class."""

from .base import basic_hag


def passes(features, cfg):
    """Pass container-height points using the ``max_hag`` geometry.yml key."""
    return basic_hag(features, cfg)
