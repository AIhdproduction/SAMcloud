"""Door invariants of the indoor model, checked after the opening refinement and in the IFC builder.

Rules (Domenic, generic, never project specific):

1. A door is never a wall: every door / passage has an opening in a wall (a host) and no wall body stands in its span.
2. The frame decides where a door is: a door is backed by frame evidence (jamb strip / head strip), a lintel or a closed leaf measured at
   the wall gap -- never only by an open leaf lying along a wall.
3. A candidate with frame evidence is never dropped by window / leaf / overlap filters (the refinement keeps it, at worst without a leaf).

``refinement_violations`` works on the refined openings (``openings_3d.json``); ``plan_violations`` works on the final plan model that the
IFC builder reads.  Both return a list of ``{"rule", "opening", "reason"}`` dicts; an empty list means the invariants hold.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

# classification reasons the refinement writes into ``evidence`` ("<kind>: <reason>") that count as measured backing of a door
_BACKED_MARKERS = ("frame (", "closed leaf", "lintel (")
_UNMEASURED_MARKERS = ("plan gap only",)


def _get(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def door_backing(opening: Any) -> str | None:
    """'frame' | 'closed leaf' | 'lintel' | 'plan gap' for a door that has evidence at its gap, None when it has none."""
    evidence = str(_get(opening, "evidence", "") or "")
    stamped = re.search(r"backing=(frame|closed leaf|lintel|plan gap)", evidence)
    if stamped:
        return stamped.group(1)
    if _get(opening, "frame_jambs") or _get(opening, "head_frame"):
        return "frame"
    for marker, label in (("frame (", "frame"), ("closed leaf", "closed leaf"), ("lintel (", "lintel")):
        if marker in evidence:
            return label
    if any(marker in evidence for marker in _UNMEASURED_MARKERS):
        return "plan gap"
    return None


def refinement_violations(openings: Iterable[Any]) -> list[dict[str, Any]]:
    """Invariants 1/2 on refined openings: a host wall, and measured evidence for every door that came from the wall-plane measurement."""
    found: list[dict[str, Any]] = []
    for opening in openings:
        kind = str(_get(opening, "kind", ""))
        if kind not in ("door", "passage"):
            continue
        identifier = str(_get(opening, "id", ""))
        if not _get(opening, "host_wall_id"):
            found.append({"rule": "door_has_host", "opening": identifier, "reason": f"{kind} without a host wall"})
        if kind == "door" and door_backing(opening) is None:
            found.append({"rule": "door_has_frame_or_lintel", "opening": identifier, "reason": "door without frame, closed-leaf or lintel evidence at its gap (leaf-only evidence is no door)"})
    return found


def plan_violations(plan: Any, tolerance_m: float = 0.05, max_share: float = 0.5) -> list[dict[str, Any]]:
    """Invariants on the final plan: hosts exist, doors are backed, and no OTHER wall stands in a door/passage span.

    ``plan`` is a :class:`samcloud.bim.plan_model.BuildingPlan` (or anything with the same attributes).  The wall bodies of the plan are
    uncut (the IFC voids the host wall at the opening), so the host wall is expected to cover the span; what must never happen is a
    wall piece that is not the host (an open door leaf modelled as a wall, a wall closing the gap next to the host) covering more than
    ``max_share`` of it.  The span is the opening rectangle (width x wall thickness) shrunk by ``tolerance_m``; a wall that merely meets
    the jamb at a T-junction covers only a sliver of it.
    """
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union

    upm = float(_get(plan, "units_per_metre", 1.0)) or 1.0
    walls = {}
    for wall in _get(plan, "walls", []):
        line = _get(wall, "centreline")
        if line and len(line) >= 2:
            half = max(float(_get(wall, "thickness_m", 0.2)) * upm / 2.0, 1e-6)
            walls[str(_get(wall, "id"))] = LineString([tuple(line[0]), tuple(line[-1])]).buffer(half, cap_style=2)
    found: list[dict[str, Any]] = []
    for opening in _get(plan, "openings", []):
        kind = str(_get(opening, "kind", ""))
        if kind not in ("door", "passage"):
            continue
        identifier = str(_get(opening, "id", ""))
        host = _get(opening, "host_wall_id")
        if host is None and kind == "door":
            found.append({"rule": "door_has_host", "opening": identifier, "reason": "door without a host wall"})
        elif host is not None and str(host) not in walls:
            found.append({"rule": "door_has_host", "opening": identifier, "reason": f"host wall {host} is not in the plan"})
        if kind == "door" and _get(opening, "confidence") is not None and door_backing(opening) is None:
            found.append({"rule": "door_has_frame_or_lintel", "opening": identifier, "reason": "measured door without frame, closed-leaf or lintel evidence"})
        others = [polygon for key, polygon in walls.items() if key != (None if host is None else str(host))]
        if not others:
            continue
        angle = float(_get(opening, "axis_angle", 0.0))
        axis = (math.cos(angle), math.sin(angle))
        normal = (-axis[1], axis[0])
        centre = _get(opening, "center")
        half = float(_get(opening, "width_m")) * upm / 2.0 - tolerance_m * upm
        depth = max(float(_get(opening, "wall_thickness_m", 0.2)) * upm / 2.0 - tolerance_m * upm, 0.02 * upm)
        if half <= 0.0:
            continue
        cx, cy = float(centre[0]), float(centre[1])
        span = Polygon([
            (cx - axis[0] * half - normal[0] * depth, cy - axis[1] * half - normal[1] * depth),
            (cx + axis[0] * half - normal[0] * depth, cy + axis[1] * half - normal[1] * depth),
            (cx + axis[0] * half + normal[0] * depth, cy + axis[1] * half + normal[1] * depth),
            (cx - axis[0] * half + normal[0] * depth, cy - axis[1] * half + normal[1] * depth),
        ])
        share = float(span.intersection(unary_union(others)).area) / max(float(span.area), 1e-12)
        if share > max_share:
            found.append({"rule": "wall_in_door", "opening": identifier, "reason": f"a wall other than the host covers {100.0 * share:.0f} % of the {kind} span"})
    return found
