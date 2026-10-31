"""Geometric plausibility check for the traffic sign class."""

from .base import basic_hag


def passes(features, cfg):
    """Pass sign-height points using the ``max_hag`` geometry.yml key."""
    return basic_hag(features, cfg)
