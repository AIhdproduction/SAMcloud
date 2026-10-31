"""Rule registry and NaN-tolerant feature predicates."""

from __future__ import annotations

from typing import Callable

import numpy as np


Rule = Callable[[dict[str, np.ndarray], dict], np.ndarray]


def at_most(values: np.ndarray, threshold: float | None) -> np.ndarray:
    """Pass values below a threshold and NaN values without geometric evidence."""
    if threshold is None:
        return np.ones(len(values), dtype=bool)
    return ~np.isfinite(values) | (values <= threshold)


def at_least(values: np.ndarray, threshold: float | None) -> np.ndarray:
    """Pass values above a threshold and NaN values without geometric evidence."""
    if threshold is None:
        return np.ones(len(values), dtype=bool)
    return ~np.isfinite(values) | (values >= threshold)


def basic_hag(features: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    """Apply optional minimum and maximum HAG plausibility bounds."""
    return at_least(features["hag"], cfg.get("min_hag")) & at_most(features["hag"], cfg.get("max_hag"))


def _load_rules() -> dict[str, Rule]:
    """Import class modules only after shared predicates are defined."""
    from . import building, construction_container, construction_machinery, curb, fence, ground, low_vegetation
    from . import overhead_power_line, road, sidewalk, traffic_sign, tree, vehicle, water
    return {
        "ground": ground.passes,
        "road": road.passes,
        "sidewalk": sidewalk.passes,
        "curb": curb.passes,
        "water": water.passes,
        "low vegetation": low_vegetation.passes,
        "tree": tree.passes,
        "building": building.passes,
        "vehicle": vehicle.passes,
        "fence": fence.passes,
        "overhead power line": overhead_power_line.passes,
        "traffic sign": traffic_sign.passes,
        "construction container": construction_container.passes,
        "construction machinery": construction_machinery.passes,
    }


RULES = _load_rules()


def passes_for_class(class_name: str, features: dict[str, np.ndarray], config: dict) -> np.ndarray:
    """Return a pass mask; classes without a configured rule deliberately pass."""
    rule = RULES.get(class_name)
    if rule is None:
        return np.ones(len(features["hag"]), dtype=bool)
    return rule(features, config.get(class_name, {}))
