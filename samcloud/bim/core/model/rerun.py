"""Re-run policy (MASTERPROMPT section 5; docs/DATA_MODEL.md, "Re-run policy").

A new extraction run produces script elements for a project that may already hold a reviewed
model. :func:`merge_rerun` matches the new elements to the existing ones and applies the
difference through a :class:`PatchEngine` as one patch group, so that ``undo_group`` restores the
scene exactly.

Matching is one-to-one and deterministic (greedy by best score, ties broken by the order of the
existing and the new elements) and runs per kind in dependency order: walls, slabs, ceilings and
spaces by geometric overlap, then openings (same matched host, overlap of the interval along the
wall axis), windows and doors (their matched opening), wall joins (their matched walls).
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import Polygon

from samcloud.bim.core.geometry import compute_scene_geometry
from samcloud.bim.core.geometry.walls import face_offsets
from samcloud.bim.core.model.elements import (
    CeilingElement,
    DoorElement,
    Element,
    OpeningElement,
    SlabElement,
    SpaceElement,
    WallElement,
    WallJoinElement,
    WindowElement,
    parse_element,
)
from samcloud.bim.core.model.enums import ElementKind, Provenance
from samcloud.bim.core.model.patches import AuthorKind, PatchEngine
from samcloud.bim.core.model.scene import ElementType, Scene

REMOVAL_REASON = "not found in re-run"
_NAMESPACE = uuid.UUID("0b7e55f4-3a54-5f0e-8e0c-7d1a52756e72")
_TYPE_GOVERNED_PARAMS = ("layers", "function", "operation")
"""A manual origin on one of these marks the existing type as set by hand."""
_LAYER_COUPLED = ("thickness", "layers")
"""Wall and slab thickness and layers change together (the layer sum must equal thickness)."""


class RerunSettings(BaseModel):
    """Thresholds of the re-run matching. Scores are IoU values in [0, 1]."""

    model_config = ConfigDict(extra="forbid")

    wall_iou_min: float = Field(default=0.5, ge=0.0, le=1.0)
    """Minimum IoU of the wall footprints (axis and thickness, joins ignored)."""
    wall_angle_max_deg: float = Field(default=10.0, ge=0.0, le=90.0)
    """Maximum difference of the axis directions in degrees (direction sign is ignored)."""
    slab_iou_min: float = Field(default=0.5, ge=0.0, le=1.0)
    ceiling_iou_min: float = Field(default=0.5, ge=0.0, le=1.0)
    space_iou_min: float = Field(default=0.5, ge=0.0, le=1.0)
    """Minimum IoU of the derived space polygons. Without geometry a space matches on an
    identical non-empty number (or name) with score ``space_iou_min``."""
    opening_overlap_min: float = Field(default=0.5, ge=0.0, le=1.0)
    """Minimum IoU of the opening intervals along the axis of the same host wall."""
    author: str = "samcloud rerun"


@dataclass(frozen=True)
class RerunMatch:
    """One matched pair."""

    kind: str
    existing_id: str
    new_id: str
    score: float
    protected: bool
    """Locked or manual: absorbed the new result but was not modified."""
    updated_params: tuple[str, ...] = ()
    kept_manual_params: tuple[str, ...] = ()
    """Parameters whose new value differs but whose manual origin was kept."""
    type_updated: bool = False


@dataclass
class RerunResult:
    """Outcome of a re-run merge."""

    group_id: str
    matches: list[RerunMatch] = field(default_factory=list)
    removed_ids: list[str] = field(default_factory=list)
    added_ids: list[str] = field(default_factory=list)
    added_type_ids: list[str] = field(default_factory=list)
    kept_unmatched_ids: list[str] = field(default_factory=list)
    """Unmatched old elements that were kept (manual, locked, AI, or referenced by those)."""
    id_map: dict[str, str] = field(default_factory=dict)
    """New element id to the final id in the scene (existing id or the id of the added element)."""
    patch_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def kept_manual_params(self) -> dict[str, tuple[str, ...]]:
        """Existing id to its manual parameters that differed from the new result."""
        return {m.existing_id: m.kept_manual_params for m in self.matches if m.kept_manual_params}

    @property
    def changed(self) -> bool:
        return bool(self.patch_ids)


# ------------------------------------------------------------------------------ geometry


def _iou(a: Polygon, b: Polygon) -> float:
    if a.is_empty or b.is_empty or not a.is_valid or not b.is_valid:
        return 0.0
    union = a.union(b).area
    if union <= 0.0:
        return 0.0
    return float(a.intersection(b).area / union)


def _wall_direction(wall: WallElement) -> tuple[float, float]:
    dx = wall.params.end[0] - wall.params.start[0]
    dy = wall.params.end[1] - wall.params.start[1]
    length = math.hypot(dx, dy)
    return (dx / length, dy / length)


def _wall_footprint(wall: WallElement) -> Polygon:
    direction = _wall_direction(wall)
    normal = (-direction[1], direction[0])
    right, left = face_offsets(wall.params.justification, wall.params.thickness)
    sx, sy = wall.params.start
    ex, ey = wall.params.end
    return Polygon(
        [
            (sx + right * normal[0], sy + right * normal[1]),
            (ex + right * normal[0], ey + right * normal[1]),
            (ex + left * normal[0], ey + left * normal[1]),
            (sx + left * normal[0], sy + left * normal[1]),
        ]
    )


def _angle_deg(a: WallElement, b: WallElement) -> float:
    da, db = _wall_direction(a), _wall_direction(b)
    cosine = abs(da[0] * db[0] + da[1] * db[1])
    return math.degrees(math.acos(min(1.0, cosine)))


def _outline(element: SlabElement | CeilingElement) -> Polygon:
    return Polygon(element.params.outline)


def _interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    overlap = min(a[1], b[1]) - max(a[0], b[0])
    union = max(a[1], b[1]) - min(a[0], b[0])
    if overlap <= 0.0 or union <= 0.0:
        return 0.0
    return overlap / union


def _space_polygons(elements: Iterable[Element], scene: Scene) -> dict[str, Polygon]:
    """Space polygons in project coordinates, empty when the geometry cannot be derived."""
    try:
        temp = Scene(
            settings=scene.settings.model_copy(deep=True),
            storeys=dict(scene.storeys),
            types=dict(scene.types),
            elements={e.id: e for e in elements},
        )
        geometry = compute_scene_geometry(temp)
        return {key: space.polygon() for key, space in geometry.spaces.items()}
    except Exception:
        return {}


def _same_storey(a: Element, b: Element) -> bool:
    return a.storey_id is None or b.storey_id is None or a.storey_id == b.storey_id


def _greedy(
    pairs: list[tuple[float, int, int]],
) -> list[tuple[int, int, float]]:
    """One-to-one assignment, best score first; ties by (existing index, new index)."""
    used_old: set[int] = set()
    used_new: set[int] = set()
    chosen: list[tuple[int, int, float]] = []
    for score, i, j in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if i in used_old or j in used_new:
            continue
        used_old.add(i)
        used_new.add(j)
        chosen.append((i, j, score))
    return chosen


# ---------------------------------------------------------------------------- the merge


def _is_protected(element: Element) -> bool:
    return element.locked or element.provenance == Provenance.MANUAL


def _references(element: Element) -> list[str]:
    refs = list(element.space_ids)
    if isinstance(element, WallJoinElement):
        refs += [element.params.wall_a, element.params.wall_b]
    elif isinstance(element, OpeningElement):
        refs.append(element.params.host_wall_id)
    elif isinstance(element, WindowElement | DoorElement):
        refs.append(element.params.opening_id)
    return refs


def _json_pointer(*tokens: str) -> str:
    return "/" + "/".join(t.replace("~", "~0").replace("/", "~1") for t in tokens)


def merge_rerun(
    scene: Scene,
    new_elements: Sequence[Element],
    new_types: Sequence[ElementType],
    engine: PatchEngine,
    *,
    settings: RerunSettings | None = None,
    run_id: str,
) -> RerunResult:
    """Merge the elements of a new run into ``scene`` through ``engine`` (one patch group).

    ``new_elements`` carry their own fresh ids and reference each other by those ids;
    ``storey_id``, ``building_id`` and ``site_id`` refer to the existing scene (an unknown id is
    replaced by the only storey, building or site of the scene, else cleared). Elements of kinds
    without a matching rule (stairs, furniture, terrain, buildings) are not matched: new ones are
    added, old ones stay untouched. Nothing is applied when nothing changed.
    """
    cfg = settings or RerunSettings()
    group_id = f"rerun-{run_id}"
    result = RerunResult(group_id=group_id)
    if engine.scene is not scene:
        raise ValueError("engine must be bound to the scene that is merged")

    old_by_kind: dict[ElementKind, list[Element]] = {}
    for element in scene.elements.values():
        old_by_kind.setdefault(ElementKind(element.kind), []).append(element)
    new_by_kind: dict[ElementKind, list[Element]] = {}
    for element in new_elements:
        new_by_kind.setdefault(ElementKind(element.kind), []).append(element)
    new_ids = {e.id for e in new_elements}
    if len(new_ids) != len(new_elements):
        raise ValueError("new_elements contain duplicate ids")

    match_of: dict[str, str] = {}  # new id -> old id
    score_of: dict[str, float] = {}
    new_to_old_wall = match_of  # alias for readability (same dict, filled kind by kind)

    def olds(kind: ElementKind) -> list[Element]:
        return old_by_kind.get(kind, [])

    def news(kind: ElementKind) -> list[Element]:
        return new_by_kind.get(kind, [])

    def record(
        chosen: list[tuple[int, int, float]], old: list[Element], new: list[Element]
    ) -> None:
        for i, j, score in chosen:
            match_of[new[j].id] = old[i].id
            score_of[new[j].id] = score

    # walls
    old_w, new_w = olds(ElementKind.WALL), news(ElementKind.WALL)
    old_fp = [_wall_footprint(w) for w in old_w if isinstance(w, WallElement)]
    new_fp = [_wall_footprint(w) for w in new_w if isinstance(w, WallElement)]
    pairs: list[tuple[float, int, int]] = []
    for i, ow in enumerate(old_w):
        for j, nw in enumerate(new_w):
            if not (isinstance(ow, WallElement) and isinstance(nw, WallElement)):
                continue
            if not _same_storey(ow, nw) or _angle_deg(ow, nw) > cfg.wall_angle_max_deg:
                continue
            score = _iou(old_fp[i], new_fp[j])
            if score >= cfg.wall_iou_min and score > 0.0:
                pairs.append((score, i, j))
    record(_greedy(pairs), old_w, new_w)

    # slabs and ceilings
    for kind, minimum in (
        (ElementKind.SLAB, cfg.slab_iou_min),
        (ElementKind.CEILING, cfg.ceiling_iou_min),
    ):
        old_k, new_k = olds(kind), news(kind)
        pairs = []
        for i, oe in enumerate(old_k):
            for j, ne in enumerate(new_k):
                if not (
                    isinstance(oe, SlabElement | CeilingElement)
                    and isinstance(ne, SlabElement | CeilingElement)
                ):
                    continue
                if not _same_storey(oe, ne):
                    continue
                score = _iou(_outline(oe), _outline(ne))
                if score >= minimum and score > 0.0:
                    pairs.append((score, i, j))
        record(_greedy(pairs), old_k, new_k)

    # spaces
    old_s, new_s = olds(ElementKind.SPACE), news(ElementKind.SPACE)
    if old_s and new_s:
        old_poly = _space_polygons(scene.elements.values(), scene)
        new_poly = _space_polygons(new_elements, scene)
        pairs = []
        for i, os_ in enumerate(old_s):
            for j, ns in enumerate(new_s):
                if not (isinstance(os_, SpaceElement) and isinstance(ns, SpaceElement)):
                    continue
                if not _same_storey(os_, ns):
                    continue
                if os_.id in old_poly and ns.id in new_poly:
                    score = _iou(old_poly[os_.id], new_poly[ns.id])
                else:
                    same = (os_.params.number and os_.params.number == ns.params.number) or (
                        os_.params.name and os_.params.name == ns.params.name
                    )
                    score = cfg.space_iou_min if same else 0.0
                if score >= cfg.space_iou_min and score > 0.0:
                    pairs.append((score, i, j))
        record(_greedy(pairs), old_s, new_s)

    # openings: same matched host, overlap of the interval along the axis
    old_o, new_o = olds(ElementKind.OPENING), news(ElementKind.OPENING)
    pairs = []
    for i, oo in enumerate(old_o):
        for j, no in enumerate(new_o):
            if not (isinstance(oo, OpeningElement) and isinstance(no, OpeningElement)):
                continue
            if new_to_old_wall.get(no.params.host_wall_id) != oo.params.host_wall_id:
                continue
            a = (oo.params.position_along_axis, oo.params.position_along_axis + oo.params.width)
            b = (no.params.position_along_axis, no.params.position_along_axis + no.params.width)
            score = _interval_iou(a, b)
            if score >= cfg.opening_overlap_min and score > 0.0:
                pairs.append((score, i, j))
    record(_greedy(pairs), old_o, new_o)

    # windows and doors: via the matched opening
    for kind in (ElementKind.WINDOW, ElementKind.DOOR):
        old_k, new_k = olds(kind), news(kind)
        pairs = []
        for i, of in enumerate(old_k):
            for j, nf in enumerate(new_k):
                if not (
                    isinstance(of, WindowElement | DoorElement)
                    and isinstance(nf, WindowElement | DoorElement)
                ):
                    continue
                if match_of.get(nf.params.opening_id) == of.params.opening_id:
                    pairs.append((1.0, i, j))
        record(_greedy(pairs), old_k, new_k)

    # joins: via the matched walls
    old_j, new_j = olds(ElementKind.WALL_JOIN), news(ElementKind.WALL_JOIN)
    pairs = []
    for i, oj in enumerate(old_j):
        for j, nj in enumerate(new_j):
            if not (isinstance(oj, WallJoinElement) and isinstance(nj, WallJoinElement)):
                continue
            mapped = {match_of.get(nj.params.wall_a), match_of.get(nj.params.wall_b)}
            if mapped == {oj.params.wall_a, oj.params.wall_b}:
                pairs.append((1.0, i, j))
    record(_greedy(pairs), old_j, new_j)

    # ---------------------------------------------------------------- types
    type_map: dict[str, str] = {}
    types_to_add: list[ElementType] = []
    for new_type in new_types:
        existing = next(
            (
                t
                for t in scene.types.values()
                if t.kind == new_type.kind and t.name == new_type.name
            ),
            None,
        )
        if existing is not None:
            type_map[new_type.id] = existing.id
            continue
        final = new_type.id
        if final in scene.types:
            final = str(uuid.uuid5(_NAMESPACE, f"{run_id}/type/{new_type.id}"))
        type_map[new_type.id] = final
        types_to_add.append(new_type.model_copy(update={"id": final}))
        result.added_type_ids.append(final)

    def final_type(type_id: str | None) -> str | None:
        if type_id is None:
            return None
        if type_id in type_map:
            return type_map[type_id]
        return type_id if type_id in scene.types else None

    def only(pool: dict[str, Any], ref: str | None) -> str | None:
        if ref is not None and ref in pool:
            return ref
        return next(iter(pool)) if len(pool) == 1 else None

    # ---------------------------------------------------------------- id map
    id_map: dict[str, str] = dict(match_of)
    for element in new_elements:
        if element.id in id_map:
            continue
        final = element.id
        if final in scene.elements:
            final = str(uuid.uuid5(_NAMESPACE, f"{run_id}/{element.id}"))
        id_map[element.id] = final
    result.id_map = id_map

    # ---------------------------------------------------------- build the add patch
    before = scene.model_dump(mode="json")
    old_dump: dict[str, Any] = before["elements"]

    def remapped(element: Element) -> dict[str, Any]:
        data = element.model_dump(mode="json")
        params = data["params"]
        if isinstance(element, WallJoinElement):
            params["wall_a"] = id_map.get(params["wall_a"], params["wall_a"])
            params["wall_b"] = id_map.get(params["wall_b"], params["wall_b"])
        elif isinstance(element, OpeningElement):
            params["host_wall_id"] = id_map.get(params["host_wall_id"], params["host_wall_id"])
        elif isinstance(element, WindowElement | DoorElement):
            params["opening_id"] = id_map.get(params["opening_id"], params["opening_id"])
        data["space_ids"] = [id_map[s] for s in data["space_ids"] if s in id_map]
        data["type_id"] = final_type(data["type_id"])
        data["storey_id"] = only(scene.storeys, data["storey_id"])
        data["building_id"] = only(scene.buildings, data["building_id"])
        data["site_id"] = only(scene.sites, data["site_id"])
        return data

    add_ops: list[dict[str, Any]] = []
    for new_type in types_to_add:
        add_ops.append(
            {
                "op": "add",
                "path": _json_pointer("types", new_type.id),
                "value": new_type.model_dump(mode="json"),
            }
        )
    for element in new_elements:
        if element.id in match_of:
            continue
        data = remapped(element)
        data["id"] = id_map[element.id]
        data["provenance"] = Provenance.SCRIPT.value
        data["locked"] = False
        value = parse_element(data).model_dump(mode="json")
        add_ops.append(
            {"op": "add", "path": _json_pointer("elements", value["id"]), "value": value}
        )
        result.added_ids.append(value["id"])

    # ------------------------------------------------------- build the update patch
    update_ops: list[dict[str, Any]] = []
    for element in new_elements:
        old_id = match_of.get(element.id)
        if old_id is None:
            continue
        old = scene.elements[old_id]
        data = remapped(element)
        if _is_protected(old):
            result.matches.append(
                RerunMatch(
                    kind=old.kind.value,
                    existing_id=old_id,
                    new_id=element.id,
                    score=score_of[element.id],
                    protected=True,
                )
            )
            continue
        old_data = old_dump[old_id]
        new_params: dict[str, Any] = data["params"]
        if isinstance(old, WallJoinElement):
            # the join may list its walls in the other order: swap the ends along with them
            if new_params["wall_a"] != old.params.wall_a:
                new_params["wall_a"], new_params["wall_b"] = (
                    new_params["wall_b"],
                    new_params["wall_a"],
                )
                new_params["end_a"], new_params["end_b"] = (
                    new_params["end_b"],
                    new_params["end_a"],
                )
        old_params: dict[str, Any] = old_data["params"]
        old_origin: dict[str, str] = old_data["param_origin"]
        new_origin: dict[str, str] = data["param_origin"]
        manual = {k for k, v in old_origin.items() if v == "manual"}
        if manual & set(_LAYER_COUPLED):
            manual |= set(_LAYER_COUPLED)
        updated: list[str] = []
        kept: list[str] = []
        base = ["elements", old_id]
        keys = [k for k in new_params if k in old_params]
        for key in keys:
            differs = new_params[key] != old_params[key]
            if key in manual:
                if differs:
                    kept.append(key)
                continue
            if key in ("wall_a", "wall_b", "host_wall_id", "opening_id"):
                continue  # identity of the match: references stay as they are
            if differs:
                update_ops.append(
                    {
                        "op": "replace",
                        "path": _json_pointer(*base, "params", key),
                        "value": new_params[key],
                    }
                )
                updated.append(key)
            if key in new_origin and old_origin.get(key) != new_origin[key]:
                update_ops.append(
                    {
                        "op": "add",
                        "path": _json_pointer(*base, "param_origin", key),
                        "value": new_origin[key],
                    }
                )
                if key not in updated:
                    updated.append(key)
        for field_name in ("metrics", "source_instance_ids", "status"):
            if data[field_name] != old_data[field_name]:
                update_ops.append(
                    {
                        "op": "replace",
                        "path": _json_pointer(*base, field_name),
                        "value": data[field_name],
                    }
                )
        type_updated = False
        type_manual = bool(manual & set(_TYPE_GOVERNED_PARAMS))
        if (
            data["type_id"] != old_data["type_id"]
            and data["type_id"] is not None
            and not type_manual
        ):
            update_ops.append(
                {
                    "op": "replace",
                    "path": _json_pointer(*base, "type_id"),
                    "value": data["type_id"],
                }
            )
            type_updated = True
        result.matches.append(
            RerunMatch(
                kind=old.kind.value,
                existing_id=old_id,
                new_id=element.id,
                score=score_of[element.id],
                protected=False,
                updated_params=tuple(sorted(updated)),
                kept_manual_params=tuple(sorted(kept)),
                type_updated=type_updated,
            )
        )

    # ------------------------------------------------------- build the removal patch
    matched_old = set(match_of.values())
    removal: set[str] = {
        e.id
        for e in scene.elements.values()
        if e.id not in matched_old
        and e.provenance == Provenance.SCRIPT
        and not e.locked
        and e.kind in _MATCHED_KINDS
    }
    # fixpoint: dependents of removed elements go too; protected dependents keep their target
    pinned: set[str] = set()
    changed = True
    while changed:
        changed = False
        for element in scene.elements.values():
            if element.id in removal:
                continue
            refs = [r for r in _references(element) if r in removal]
            if not refs:
                continue
            keeper = _is_protected(element) or element.id in pinned
            structural = _is_structural_dependent(element, refs)
            if not keeper and structural:
                removal.add(element.id)
                result.notes.append(
                    f"{element.kind.value} {element.id} removed with its removed host "
                    f"{refs[0]} ({REMOVAL_REASON})"
                )
            elif keeper:
                removal -= set(refs)
                pinned |= set(refs)
                result.notes.append(
                    f"{refs[0]} kept: referenced by protected {element.kind.value} {element.id}"
                )
            else:
                continue  # only a space membership of an editable element: space_ids is edited
            changed = True

    remove_ops: list[dict[str, Any]] = []
    for element in scene.elements.values():
        if element.id in removal:
            continue
        if any(s in removal for s in element.space_ids):
            remove_ops.append(
                {
                    "op": "replace",
                    "path": _json_pointer("elements", element.id, "space_ids"),
                    "value": [s for s in element.space_ids if s not in removal],
                }
            )
    for element in scene.elements.values():
        if element.id in removal:
            remove_ops.append({"op": "remove", "path": _json_pointer("elements", element.id)})
            result.removed_ids.append(element.id)
    result.kept_unmatched_ids = [
        e.id for e in scene.elements.values() if e.id not in matched_old and e.id not in removal
    ]

    # ----------------------------------------------------------------- apply
    for ops, reason in (
        (add_ops, f"re-run {run_id}: add new elements"),
        (update_ops, f"re-run {run_id}: update matched elements"),
        (remove_ops, f"re-run {run_id}: {REMOVAL_REASON}"),
    ):
        if not ops:
            continue
        record_ = engine.apply(ops, cfg.author, AuthorKind.SCRIPT, reason=reason, group_id=group_id)
        result.patch_ids.append(record_.id)
    return result


_MATCHED_KINDS = frozenset(
    {
        ElementKind.WALL,
        ElementKind.WALL_JOIN,
        ElementKind.SLAB,
        ElementKind.CEILING,
        ElementKind.OPENING,
        ElementKind.WINDOW,
        ElementKind.DOOR,
        ElementKind.SPACE,
    }
)


def _is_structural_dependent(element: Element, refs: list[str]) -> bool:
    """True when ``element`` references one of ``refs`` other than through ``space_ids``."""
    structural: list[str] = []
    if isinstance(element, WallJoinElement):
        structural = [element.params.wall_a, element.params.wall_b]
    elif isinstance(element, OpeningElement):
        structural = [element.params.host_wall_id]
    elif isinstance(element, WindowElement | DoorElement):
        structural = [element.params.opening_id]
    return any(r in structural for r in refs)
