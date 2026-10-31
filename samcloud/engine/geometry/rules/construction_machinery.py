"""Geometric plausibility check for the construction machinery class."""

from .base import basic_hag


def passes(features, cfg):
    """Pass machinery-height points using the ``max_hag`` geometry.yml key."""
    return basic_hag(features, cfg)
