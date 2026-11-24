"""Configuration of the processing protocol (``config/protocol.yaml``) with safe defaults."""

from __future__ import annotations

import copy
import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "thresholds": {
        "control_rms_cm": {"green": 2.0, "yellow": 5.0},
        "pass_rms_cm": {"green": 2.0, "yellow": 5.0},
        "reprojection_px": {"green": 1.0, "yellow": 2.0},
        "registered_pct": {"green": 95, "yellow": 80},
        "apriltag_deviation_pct": {"green": 0.5, "yellow": 1.5},
    },
    "overlap": {"max_cells": 120},
    "density": {"cell_m": 0.25},
    "sections": {"camera_list": False, "class_details": True},
}


def _merge(base: dict[str, Any], override: Any) -> dict[str, Any]:
    result = copy.deepcopy(base)
    if not isinstance(override, dict):
        return result
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        elif key in result and type(value) is not type(result[key]) and not (
            isinstance(value, (int, float)) and isinstance(result[key], (int, float))
        ):
            continue  # wrong type: keep the default
        else:
            result[key] = value
    return result


def load_protocol_config() -> dict[str, Any]:
    """``config/protocol.yaml`` merged over the built-in defaults; a missing or broken file gives the defaults."""
    try:
        from samcloud.core.configuration import load_global_config

        return _merge(DEFAULTS, load_global_config("protocol.yaml"))
    except Exception as error:  # noqa: BLE001 - the protocol must work without its config
        LOGGER.warning("protocol.yaml not usable, using defaults: %s", error)
        return copy.deepcopy(DEFAULTS)
