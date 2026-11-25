"""Short display labels of the outdoor elements ("B9", "S3", "Z12"): one letter code per category, numbered by position.

The real element ids are uuids (trees) or order-dependent ``fence-N`` style ids, both unreadable in a chat.  Every
element of a listed category gets a short label: the category code plus a number counted in a deterministic order
(north to south, then west to east, by the element's plan anchor; the id breaks ties).  The same labels appear in the
element list the model sees (``ref``), the chips, the proposal summary and the change list, and the edit tools accept
them as targets besides the real id (resolved to the id before an op is applied and stored, so the history only holds
real ids).

Labels are stable as long as the elements of a category keep their positions (a recompute of the same data gives the
same labels); adding, deleting or moving an element of a category may renumber that category.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

LABEL_CODES: dict[str, str] = {
    "tree": "B", "manhole": "S", "fence": "Z", "site_wall": "M", "curb": "R", "pavement": "P", "building": "G",
    "terrain": "T", "wall": "F", "shrub": "ST", "marking": "MK", "vegetation_area": "V",
}
"""Category -> label code (letters; German initials: Baum, Schacht, Zaun, Mauer, Randstein, ...)."""
_CODE_KIND = {code: kind for kind, code in LABEL_CODES.items()}
LABEL_PATTERN = re.compile(r"^\s*(ST|MK|[BSZMRPGTFV])\s*-?\s*(\d{1,5})\s*$", re.IGNORECASE)
_GRID_M = 0.1  # anchors are compared on a 10 cm grid so float noise does not reorder elements


def _kind(element: Any) -> str:
    kind = getattr(element, "kind", "")
    return str(getattr(kind, "value", kind))


def _anchor(element: Any) -> tuple[float, float] | None:
    """A cheap plan point (local metres) of an element: trunk, cover centre, mean of the axis / outline / mesh points."""
    p = element.params
    kind = _kind(element)
    try:
        if kind == "tree":
            return float(p.trunk_xy[0]), float(p.trunk_xy[1])
        if kind == "manhole":
            return float(p.center[0]), float(p.center[1])
        for name in ("axis", "footprint", "outline", "points", "vertices"):
            points = getattr(p, name, None)
            if points:
                xy = np.asarray([(float(q[0]), float(q[1])) for q in points], dtype=np.float64)
                return float(xy[:, 0].mean()), float(xy[:, 1].mean())
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    try:
        from samcloud.assistant.domains.outdoor_edits import element_anchor

        return element_anchor(element)
    except Exception:  # noqa: BLE001 - an element without a position is numbered last
        return None


def _order_key(element: Any) -> tuple[float, float, str]:
    anchor = _anchor(element)
    if anchor is None or not all(math.isfinite(v) for v in anchor):
        return math.inf, math.inf, str(element.id)
    return -round(anchor[1] / _GRID_M), round(anchor[0] / _GRID_M), str(element.id)


def display_labels(scene: Any) -> dict[str, str]:
    """``{element id: short label}`` of every element of a labelled category."""
    groups: dict[str, list[Any]] = {}
    for element in scene.elements.values():
        kind = _kind(element)
        if kind in LABEL_CODES:
            groups.setdefault(kind, []).append(element)
    labels: dict[str, str] = {}
    for kind, elements in groups.items():
        code = LABEL_CODES[kind]
        for number, element in enumerate(sorted(elements, key=_order_key), start=1):
            labels[str(element.id)] = f"{code}{number}"
    return labels


def display_label(scene: Any, element_id: str | None) -> str | None:
    """The short label of one element (None for unknown ids / unlabelled categories)."""
    if element_id is None:
        return None
    return display_labels(scene).get(str(element_id))


def is_label(text: Any) -> bool:
    return isinstance(text, str) and bool(LABEL_PATTERN.match(text))


def normalize_label(text: str) -> str | None:
    """``"b 9"`` / ``"B-9"`` -> ``"B9"``; None when ``text`` is no label."""
    match = LABEL_PATTERN.match(str(text or ""))
    return None if match is None else f"{match.group(1).upper()}{int(match.group(2))}"


def resolve_label(scene: Any, text: Any) -> str | None:
    """The element id of a short label (or of a real id) in ``scene``; None when nothing matches."""
    if text is None:
        return None
    if str(text) in scene.elements:
        return str(text)
    label = normalize_label(str(text))
    if label is None:
        return None
    for element_id, candidate in display_labels(scene).items():
        if candidate == label:
            return element_id
    return None


__all__ = ["LABEL_CODES", "LABEL_PATTERN", "display_label", "display_labels", "is_label", "normalize_label", "resolve_label"]
