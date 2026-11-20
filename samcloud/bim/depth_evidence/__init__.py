"""Compact, labelled depth evidence for BIM consumers.

The product is intentionally independent from ``processing_cache``.  It can be
queried by plan coordinates after the dense reconstruction has been removed.
"""

from .core import DepthEvidence, build_depth_product, load_settings

__all__ = ["DepthEvidence", "build_depth_product", "load_settings"]
