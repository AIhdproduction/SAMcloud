"""Which assistant domain a project level gets, and which tool tier is active.

Domains register a matcher and a factory (:func:`register_domain`); :func:`domain_for` asks them in priority order
(lowest first).  Built in:

* ``outdoor`` (priority 10): drone / environment-chain projects, detected like
  :attr:`samcloud.app.dxf_controller.DxfPageController.uses_environment_chain` (outdoor mode and
  ``products/environment/state.json`` or ``acquisition.type: drone_gps``).  Until the outdoor domain module
  (``samcloud.assistant.domains.outdoor``) exists, a placeholder answers ``assistant.agent.outdoor_pending``; that module
  only has to call ``register_domain("outdoor", ...)`` on import to replace it.
* ``indoor`` (priority 20): the level has a ``plan_model.json``.

Without a match :func:`unsupported_reason` tells why: an outdoor (ortho) project without the environment chain needs it
(``assistant.agent.needs_environment``), anything else has no plan yet (``assistant.agent.no_plan``).

``mode`` is the project mode of the page (``"indoor"`` / ``"outdoor"``) when known; without it a level id ``"site"``
means outdoor.
"""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from samcloud.assistant.domains.base import TIER_ELEMENTS, TIER_FULL, AssistantDomain, UnavailableDomain

LOGGER = logging.getLogger(__name__)

OUTDOOR_SITE_LEVEL = "site"
REASON_NO_PLAN = "assistant.agent.no_plan"
REASON_NEEDS_ENVIRONMENT = "assistant.agent.needs_environment"
REASON_OUTDOOR_PENDING = "assistant.agent.outdoor_pending"
REASON_OUTDOOR_NO_SCENE = "assistant.agent.outdoor_no_scene"
DOMAIN_REASONS: tuple[str, ...] = (REASON_NO_PLAN, REASON_NEEDS_ENVIRONMENT, REASON_OUTDOOR_PENDING, REASON_OUTDOOR_NO_SCENE)
"""Reason keys that come from the project / domain (not from the assistant settings)."""

PRIORITY_OUTDOOR = 10
PRIORITY_INDOOR = 20
_PLUGIN_MODULES: tuple[str, ...] = ("samcloud.assistant.domains.indoor", "samcloud.assistant.domains.outdoor")

Matcher = Callable[[Path, str, "str | None"], bool]
Factory = Callable[[Path, str, "str | None"], AssistantDomain]


@dataclass(frozen=True)
class _Entry:
    kind: str
    factory: Factory
    matcher: Matcher
    priority: int


_REGISTRY: dict[str, _Entry] = {}
_LOCK = threading.RLock()
_LOADED = False


def register_domain(kind: str, factory: Factory, matcher: Matcher, *, priority: int | None = None) -> None:
    """Register (or replace) the domain ``kind``: ``matcher(project, level_id, mode) -> bool`` decides, ``factory(project,
    level_id, mode)`` builds the domain.  Without ``priority`` a replaced kind keeps its place (new kinds go last)."""
    with _LOCK:
        if priority is None:
            previous = _REGISTRY.get(kind)
            priority = previous.priority if previous is not None else 100
        _REGISTRY[str(kind)] = _Entry(str(kind), factory, matcher, int(priority))


def registered_kinds() -> list[str]:
    _ensure_loaded()
    with _LOCK:
        return [entry.kind for entry in sorted(_REGISTRY.values(), key=lambda e: e.priority)]


# ---------------------------------------------------------------------------------------------------- project tests

def is_outdoor(level_id: str, mode: str | None = None) -> bool:
    """Outdoor level: the page says so, or (mode unknown) the level is the outdoor ``site``."""
    if mode is not None:
        return str(mode) == "outdoor"
    return str(level_id) == OUTDOOR_SITE_LEVEL


def uses_environment_chain(project: Path | str, level_id: str = OUTDOOR_SITE_LEVEL, mode: str | None = None) -> bool:
    """Drone / environment-chain project (same rule as ``DxfPageController.uses_environment_chain``)."""
    if not is_outdoor(level_id, mode):
        return False
    project = Path(project)
    if (project / "products" / "environment" / "state.json").is_file():
        return True
    project_file = project / "project.samcloud.yml"
    if not project_file.is_file():
        return False
    try:
        from samcloud.core.configuration import load_yaml

        data = load_yaml(project_file)
    except Exception:  # noqa: BLE001 - an unreadable project file means "not a drone project"
        return False
    if not isinstance(data, dict):
        return False
    return str((data.get("acquisition") or {}).get("type", "")) == "drone_gps"


# ------------------------------------------------------------------------------------------------ outdoor placeholder

class OutdoorPendingDomain(UnavailableDomain):
    """Placeholder for drone / environment projects until the outdoor domain is implemented."""

    kind = "outdoor"
    reason_key = REASON_OUTDOOR_PENDING


def _outdoor_matcher(project: Path, level_id: str, mode: str | None) -> bool:
    return uses_environment_chain(project, level_id, mode)


def _outdoor_placeholder(project: Path, level_id: str, mode: str | None) -> AssistantDomain:
    return OutdoorPendingDomain(project, level_id, mode=mode)


def _ensure_loaded() -> None:
    """Register the built-in domains once; plug-in modules register themselves on import (a missing one is skipped)."""
    global _LOADED
    with _LOCK:
        if _LOADED:
            return
        _LOADED = True
        if "outdoor" not in _REGISTRY:
            register_domain("outdoor", _outdoor_placeholder, _outdoor_matcher, priority=PRIORITY_OUTDOOR)
        for name in _PLUGIN_MODULES:
            try:
                module = importlib.import_module(name)
            except ModuleNotFoundError as error:
                if error.name != name:
                    LOGGER.warning("assistant: domain module %s not loaded (%s)", name, error.name)
                continue
            except Exception as error:  # noqa: BLE001 - a broken domain module must not break the others
                LOGGER.warning("assistant: domain module %s failed (%s)", name, type(error).__name__)
                continue
            register = getattr(module, "register", None)
            if callable(register):
                register()
            elif name.endswith(".indoor") and "indoor" not in _REGISTRY:
                register_domain("indoor", module.factory, module.matches, priority=PRIORITY_INDOOR)


# ---------------------------------------------------------------------------------------------------------- lookups

def domain_for(project: Path | str, level_id: str, *, mode: str | None = None) -> AssistantDomain | None:
    """The domain of a project level, or None (see :func:`unsupported_reason`)."""
    _ensure_loaded()
    project = Path(project)
    with _LOCK:
        entries = sorted(_REGISTRY.values(), key=lambda e: e.priority)
    for entry in entries:
        try:
            if entry.matcher(project, str(level_id), mode):
                return entry.factory(project, str(level_id), mode)
        except Exception as error:  # noqa: BLE001 - a broken matcher must not hide the other domains
            LOGGER.warning("assistant: domain %s check failed (%s)", entry.kind, type(error).__name__)
    return None


def unsupported_reason(project: Path | str, level_id: str, *, mode: str | None = None) -> str | None:
    """Reason key why the level cannot be edited (None when a domain is available)."""
    domain = domain_for(project, level_id, mode=mode)
    if domain is not None:
        ok, reason = domain.availability()
        return None if ok else (reason or REASON_NO_PLAN)
    return reason_without_domain(project, level_id, mode=mode)


def reason_without_domain(project: Path | str, level_id: str, *, mode: str | None = None) -> str:
    """Reason key for a level no domain matched."""
    del project
    return REASON_NEEDS_ENVIRONMENT if is_outdoor(level_id, mode) else REASON_NO_PLAN


# ------------------------------------------------------------------------------------------------------------- tiers

def requested_tier(settings: Any) -> str:
    """``"full"`` when full control is on for the settings (see ``effective_full_control``), else ``"elements"``."""
    from samcloud.assistant.settings import effective_full_control

    try:
        return TIER_FULL if effective_full_control(settings) else TIER_ELEMENTS
    except Exception:  # noqa: BLE001 - odd settings objects fall back to the default tier
        return TIER_ELEMENTS


def resolve_tier(domain: AssistantDomain | None, settings: Any) -> str:
    """The tool tier a turn really uses: the requested one when the domain supports it, else ``"elements"``."""
    tier = requested_tier(settings)
    if domain is None or tier not in tuple(getattr(domain, "supported_tiers", (TIER_ELEMENTS,))):
        return TIER_ELEMENTS
    return tier


def _reset_for_tests() -> None:
    """Forget every registration (tests only); the built-ins are registered again on the next lookup."""
    global _LOADED
    with _LOCK:
        _REGISTRY.clear()
        _LOADED = False


__all__ = ["DOMAIN_REASONS", "OutdoorPendingDomain", "REASON_NEEDS_ENVIRONMENT", "REASON_NO_PLAN", "REASON_OUTDOOR_NO_SCENE",
           "REASON_OUTDOOR_PENDING",
           "domain_for", "is_outdoor", "reason_without_domain", "register_domain", "registered_kinds", "requested_tier",
           "resolve_tier", "unsupported_reason", "uses_environment_chain"]
