"""Assistant domains: one adapter per project type (indoor plan, outdoor site, ...), chosen per project level.

See :mod:`samcloud.assistant.domains.base` for the interface and :mod:`samcloud.assistant.domains.registry` for the choice.
"""

from samcloud.assistant.domains.base import (
    TIER_ELEMENTS,
    TIER_FULL,
    TIERS,
    AssistantDomain,
    HistoryBackend,
    ToolDef,
    UnavailableDomain,
    normalize_tier,
)
from samcloud.assistant.domains.registry import domain_for, register_domain, resolve_tier, unsupported_reason

__all__ = ["AssistantDomain", "HistoryBackend", "TIERS", "TIER_ELEMENTS", "TIER_FULL", "ToolDef", "UnavailableDomain",
           "domain_for", "normalize_tier", "register_domain", "resolve_tier", "unsupported_reason"]
