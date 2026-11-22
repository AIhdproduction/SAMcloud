"""Score pipeline outputs (plan_model.json, IFC, DXF, qa.json) of one bench case against its ground truth.

All comparisons happen in the *scene frame* in true metres:

* plan coordinates (local units) -> ``Frame.to_original`` -> world -> ``SceneSpec.to_scene``
* plan ``*_m`` values are local units divided by ``units_per_metre``; true metres = value * upm / frame scale
  (the scale error ``upm / scale - 1`` of the delivered metric values is reported separately as ``scale_error``)
* IFC coordinates are local / upm, so local = IFC * upm.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from .furniture import label_group
from .geometry import polygon_from_list

FURNITURE_IFC = ("IfcFurniture", "IfcFurnishingElement", "IfcSanitaryTerminal", "IfcElectricAppliance", "IfcBuildingElementProxy",
                 "IfcSystemFurnitureElement", "IfcDiscreteAccessory", "IfcFlowTerminal")
LABEL_ALIASES = {"couch": "sofa", "fridge": "refrigerator", "bookcase": "shelf", "cabinet": "kitchen cabinet", "hob": "stove",
                 "cooktop": "stove", "wc": "toilet", "basin": "washbasin"}


def _nan() -> float:
    return float("nan")


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else _nan()


def _ratio(a: float, b: float) -> float:
    return float(a) / float(b) if b else _nan()


def _f1(p: float, r: float) -> float:
    if not (np.isfinite(p) and np.isfinite(r)) or p + r == 0:
        return _nan() if not (np.isfinite(p) and np.isfinite(r)) else 0.0
    return 2 * p * r / (p + r)


class PlanFrame:
    """Coordinate conversion between plan local units and the scene frame."""

    def __init__(self, plan: dict[str, Any], truth: dict[str, Any]) -> None:
        frame = plan.get("frame") or {}
        self.origin = np.asarray(frame.get("origin", [0, 0, 0]), dtype=float)
        self.scale = float(frame.get("scale_factor", 1.0) or 1.0)
        self.rotation = np.asarray(frame.get("rotation", np.eye(3)), dtype=float)
        self.upm = float(plan.get("units_per_metre") or 1.0)
        self.floor_z = float(plan.get("floor_z") or 0.0)
        yaw = math.radians(float(truth.get("world_yaw_deg", 0.0)))
        self.world_rot = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
        self.world_offset = np.asarray(truth.get("world_offset", [0, 0, 0]), dtype=float)

    def local_to_scene(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        world = (pts @ self.rotation) / self.scale + self.origin
        return (world - self.world_offset) @ self.world_rot

    def xy(self, pts: Iterable, z: float | None = None) -> np.ndarray:
        arr = np.asarray(list(pts), dtype=float).reshape(-1, 2)
        zz = np.full((len(arr), 1), self.floor_z if z is None else z)
        return self.local_to_scene(np.hstack((arr, zz)))[:, :2]

    def z(self, z_local: float) -> float:
        return float(self.local_to_scene(np.array([[0.0, 0.0, z_local]]))[0, 2] - self.local_to_scene(np.array([[0.0, 0.0, self.floor_z]]))[0, 2])

    def metres(self, value_m: float | None) -> float:
        return _nan() if value_m is None else float(value_m) * self.upm / self.scale

    def angle(self, axis_angle: float) -> float:
        d = np.array([[math.cos(axis_angle), math.sin(axis_angle), 0.0]]) @ self.rotation
        d = d @ self.world_rot
        return float(math.atan2(d[0, 1], d[0, 0]))

    def ifc_to_scene(self, pts: np.ndarray) -> np.ndarray:
        return self.local_to_scene(np.asarray(pts, dtype=float) * self.upm)

    def polygon(self, data: Any) -> Polygon:
        if isinstance(data, dict):
            ext = self.xy(data.get("exterior", []))
            holes = [self.xy(h) for h in data.get("holes", []) or [] if len(h) >= 3]
            return Polygon(ext, holes).buffer(0) if len(ext) >= 3 else Polygon()
        pts = self.xy(data)
        return Polygon(pts).buffer(0) if len(pts) >= 3 else Polygon()


# ---------------------------------------------------------------------------------------------------------------
# walls
# ---------------------------------------------------------------------------------------------------------------

def _seg(p0, p1) -> tuple[np.ndarray, np.ndarray, float]:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    return p0, d / max(length, 1e-12), length


def _room_side(gt_wall: dict[str, Any], rooms: list[Polygon]) -> list[int]:
    p0, d, length = _seg(gt_wall["p0"], gt_wall["p1"])
    n = np.array([-d[1], d[0]])
    mid = p0 + d * length / 2
    sides = []
    for s in (1, -1):
        probe = Point(*(mid + n * s * (gt_wall["thickness"] / 2 + 0.15)))
        if any(r.contains(probe) for r in rooms):
            sides.append(s)
    return sides


def wall_metrics(gt: dict[str, Any], pred_walls: list[dict[str, Any]], rooms_gt: list[Polygon]) -> dict[str, Any]:
    gt_walls = gt["walls"]
    openings_by_wall: dict[str, list[tuple[float, float]]] = {}
    for o in gt["openings"]:
        if o["kind"] in ("door", "passage"):
            w = next(w for w in gt_walls if w["id"] == o["host"])
            p0, d, _ = _seg(w["p0"], w["p1"])
            s = float(np.dot(np.asarray(o["center"]) - p0, d))
            openings_by_wall.setdefault(o["host"], []).append((s - o["width"] / 2, s + o["width"] / 2))
    assignments: list[dict[str, Any] | None] = []
    for p in pred_walls:
        q0, qd, qlen = _seg(p["p0"], p["p1"])
        best = None
        for g in gt_walls:
            g0, gd, glen = _seg(g["p0"], g["p1"])
            ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(qd, gd))))))
            if ang > 8.0:
                continue
            gn = np.array([-gd[1], gd[0]])
            offs = [float(np.dot(q - g0, gn)) for q in (q0, q0 + qd * qlen)]
            offset = float(np.mean(offs))
            limit = max(0.5 * (g["thickness"] + p["thickness"]), 0.15) + 0.10
            if abs(offset) > limit:
                continue
            s0, s1 = sorted(float(np.dot(q - g0, gd)) for q in (q0, q0 + qd * qlen))
            lo, hi = max(s0, -g["thickness"] / 2), min(s1, glen + g["thickness"] / 2)
            overlap = hi - lo
            if overlap <= 0.1 * qlen or overlap <= 0.05:
                continue
            score = (overlap, -abs(offset))
            if best is None or score > best["score"]:
                best = {"gt": g["id"], "angle": ang, "offset": offset, "overlap": overlap, "interval": (max(s0, 0.0), min(s1, glen)), "score": score}
        assignments.append(best)
    per_gt = []
    total_pred_len = sum(p["length"] for p in pred_walls)
    matched_len = sum(a["overlap"] for a in assignments if a)
    for g in gt_walls:
        g0, gd, glen = _seg(g["p0"], g["p1"])
        gn = np.array([-gd[1], gd[0]])
        ivals = sorted(a["interval"] for a in assignments if a and a["gt"] == g["id"])
        covered = _interval_union(ivals)
        excused = _interval_union(sorted(openings_by_wall.get(g["id"], [])))
        required = glen - _interval_overlap(excused, [(0.0, glen)])
        cov_len = _interval_overlap(covered, [(0.0, glen)]) - _interval_overlap(_interval_intersect(covered, excused), [(0.0, glen)])
        coverage = min(1.0, cov_len / required) if required > 0 else 1.0
        members = [(p, a) for p, a in zip(pred_walls, assignments) if a and a["gt"] == g["id"]]
        sides = _room_side(g, rooms_gt)
        face_err = []
        for p, a in members:
            weight = a["overlap"]
            for s in sides:
                gt_face = s * g["thickness"] / 2
                pred_face = a["offset"] + s * p["thickness"] / 2
                face_err.append((abs(pred_face - gt_face), weight))
        per_gt.append({
            "id": g["id"], "exterior": g["exterior"], "length": glen, "thickness": g["thickness"], "coverage": coverage,
            "pieces": len(members),
            "centre_offset": _wmean([(abs(a["offset"]), a["overlap"]) for _p, a in members]),
            "face_error": _wmean(face_err),
            "angle_error": _wmean([(a["angle"], a["overlap"]) for _p, a in members]),
            "thickness_error": _wmean([(abs(p["thickness"] - g["thickness"]), a["overlap"]) for p, a in members]),
            "thickness_signed": _wmean([(p["thickness"] - g["thickness"], a["overlap"]) for p, a in members]),
        })
    detected = [w for w in per_gt if w["coverage"] >= 0.5]
    precise_preds = [a for p, a in zip(pred_walls, assignments) if a and a["overlap"] >= 0.5 * p["length"]]
    interior = [w for w in per_gt if not w["exterior"]]
    topology = wall_topology(gt, pred_walls)
    return {
        "gt_count": len(gt_walls), "pred_count": len(pred_walls),
        "recall": _ratio(len(detected), len(gt_walls)),
        "precision": _ratio(len(precise_preds), len(pred_walls)),
        "precision_length": _ratio(matched_len, total_pred_len),
        "length_coverage": _mean(w["coverage"] for w in per_gt),
        "length_coverage_interior": _mean(w["coverage"] for w in interior),
        "centre_offset_m": _mean(w["centre_offset"] for w in per_gt),
        "face_error_m": _mean(w["face_error"] for w in per_gt),
        "angle_error_deg": _mean(w["angle_error"] for w in per_gt),
        "thickness_error_m": _mean(w["thickness_error"] for w in per_gt),
        "thickness_error_interior_m": _mean(w["thickness_error"] for w in interior),
        "thickness_error_exterior_m": _mean(w["thickness_error"] for w in per_gt if w["exterior"]),
        "missed": [w["id"] for w in per_gt if w["coverage"] < 0.5],
        "false_walls": [p["id"] for p, a in zip(pred_walls, assignments) if a is None],
        "per_wall": per_gt,
        "assign": {p["id"]: (a["gt"] if a else None) for p, a in zip(pred_walls, assignments)},
        **topology,
    }


def _wmean(pairs: list[tuple[float, float]]) -> float:
    pairs = [(v, w) for v, w in pairs if np.isfinite(v) and w > 0]
    if not pairs:
        return _nan()
    return float(sum(v * w for v, w in pairs) / sum(w for _v, w in pairs))


def _interval_union(ivals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(ivals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _interval_intersect(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for x0, x1 in a:
        for y0, y1 in b:
            lo, hi = max(x0, y0), min(x1, y1)
            if hi > lo:
                out.append((lo, hi))
    return _interval_union(out)


def _interval_overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    return float(sum(hi - lo for lo, hi in _interval_intersect(a, b)))


def _wall_rect(p: dict[str, Any]) -> Polygon:
    return LineString([p["p0"], p["p1"]]).buffer(max(p["thickness"], 0.02) / 2, cap_style="flat")


def wall_topology(gt: dict[str, Any], pred_walls: list[dict[str, Any]]) -> dict[str, Any]:
    rects = [_wall_rect(p) for p in pred_walls]
    free_ends = 0
    opening_edges = []
    for o in gt["openings"]:
        a = o["axis_angle"]
        d = np.array([math.cos(a), math.sin(a)])
        c = np.asarray(o["center"])
        opening_edges += [c + d * o["width"] / 2, c - d * o["width"] / 2]
    for i, p in enumerate(pred_walls):
        for end in (np.asarray(p["p0"]), np.asarray(p["p1"])):
            pt = Point(*end)
            touching = any(j != i and rects[j].distance(pt) < pred_walls[j]["thickness"] / 2 + 0.08 for j in range(len(rects)))
            near_opening = any(np.linalg.norm(end - e) < 0.35 for e in opening_edges)
            if not touching and not near_opening:
                free_ends += 1
    overlaps = 0
    for i in range(len(pred_walls)):
        for j in range(i + 1, len(pred_walls)):
            _a0, da, _la = _seg(pred_walls[i]["p0"], pred_walls[i]["p1"])
            _b0, db, _lb = _seg(pred_walls[j]["p0"], pred_walls[j]["p1"])
            if abs(float(np.dot(da, db))) < math.cos(math.radians(10)):
                continue
            if rects[i].intersection(rects[j]).area > 0.02:
                overlaps += 1
    union = unary_union(rects) if rects else Polygon()
    gaps = 0
    for junction in gt.get("junctions", []):
        if union.distance(Point(*junction["point"])) > 0.10:
            gaps += 1
    return {"free_ends": free_ends, "parallel_overlaps": overlaps, "junction_gaps": gaps, "junctions": len(gt.get("junctions", []))}


# ---------------------------------------------------------------------------------------------------------------
# rooms / openings / furniture
# ---------------------------------------------------------------------------------------------------------------

def _hungarian(score: np.ndarray, minimum: float) -> list[tuple[int, int, float]]:
    if score.size == 0:
        return []
    rows, cols = linear_sum_assignment(-score)
    return [(int(r), int(c), float(score[r, c])) for r, c in zip(rows, cols) if score[r, c] >= minimum]


def _iou(a: Polygon, b: Polygon) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    union = a.union(b).area
    return float(inter / union) if union > 0 else 0.0


def room_metrics(gt_rooms: list[dict[str, Any]], pred_rooms: list[dict[str, Any]]) -> dict[str, Any]:
    gpolys = [polygon_from_list(r["polygon"]) for r in gt_rooms]
    ppolys = [r["polygon"] for r in pred_rooms]
    score = np.array([[_iou(g, p) for p in ppolys] for g in gpolys]).reshape(len(gpolys), len(ppolys))
    matches = _hungarian(score, 0.3)
    per = []
    for gi, room in enumerate(gt_rooms):
        m = next((x for x in matches if x[0] == gi), None)
        pred = pred_rooms[m[1]] if m else None
        per.append({
            "name": room["name"], "gt_area": room["area"], "iou": m[2] if m else 0.0,
            "pred_area_true": float(ppolys[m[1]].area) if m else _nan(),
            "pred_area_field": pred["area_m2"] if pred else _nan(),
            "area_error_rel": (float(ppolys[m[1]].area) - room["area"]) / room["area"] if m else _nan(),
            "area_field_error_rel": (pred["area_m2"] - room["area"]) / room["area"] if pred else _nan(),
            "ceiling_gt": room["ceiling_h"], "ceiling_pred": pred.get("ceiling_h") if pred else None,
        })
    return {
        "gt_count": len(gt_rooms), "pred_count": len(pred_rooms), "matched": len(matches),
        "count_ok": len(gt_rooms) == len(pred_rooms),
        "mean_iou": _mean(r["iou"] for r in per),
        "area_error_rel": _mean(abs(r["area_error_rel"]) for r in per),
        "area_field_error_rel": _mean(abs(r["area_field_error_rel"]) for r in per),
        "ceiling_error_m": _mean(abs(r["ceiling_pred"] - r["ceiling_gt"]) for r in per if r["ceiling_pred"] is not None),
        "ceiling_reported": sum(1 for r in per if r["ceiling_pred"] is not None),
        "per_room": per,
    }


def opening_metrics(gt_openings: list[dict[str, Any]], pred_openings: list[dict[str, Any]], wall_assign: dict[str, str | None]) -> dict[str, Any]:
    score = np.zeros((len(gt_openings), len(pred_openings)))
    for gi, g in enumerate(gt_openings):
        a = g["axis_angle"]
        d = np.array([math.cos(a), math.sin(a)])
        n = np.array([-d[1], d[0]])
        gcen = np.asarray(g["center"], dtype=float)
        for pi, p in enumerate(pred_openings):
            delta = np.asarray(p["center"], dtype=float) - gcen
            if abs(float(np.dot(delta, n))) > g["thickness"] / 2 + 0.35:
                continue
            pa = float(p.get("angle", a))
            if abs(math.sin(pa - a)) > math.sin(math.radians(15)):
                continue
            s = float(np.dot(delta, d))
            lo, hi = max(-g["width"] / 2, s - p["width"] / 2), min(g["width"] / 2, s + p["width"] / 2)
            inter = max(0.0, hi - lo)
            union = max(g["width"] / 2, s + p["width"] / 2) - min(-g["width"] / 2, s - p["width"] / 2)
            iou = inter / union if union > 0 else 0.0
            if iou > 0.1 or float(np.linalg.norm(delta)) < 0.3:
                score[gi, pi] = max(iou, 0.11)
    matches = _hungarian(score, 0.1)
    pairs = [(gt_openings[g], pred_openings[p]) for g, p, _s in matches]
    result: dict[str, Any] = {"gt_count": len(gt_openings), "pred_count": len(pred_openings), "matched_any_kind": len(pairs)}
    for kind in ("door", "window", "passage"):
        g_k = sum(1 for o in gt_openings if o["kind"] == kind)
        p_k = sum(1 for o in pred_openings if o["kind"] == kind)
        tp = sum(1 for g, p in pairs if g["kind"] == kind and p["kind"] == kind)
        result[kind] = {"gt": g_k, "pred": p_k, "tp": tp, "precision": _ratio(tp, p_k), "recall": _ratio(tp, g_k)}
    result["detection_precision"] = _ratio(len(pairs), len(pred_openings))
    result["detection_recall"] = _ratio(len(pairs), len(gt_openings))
    result["kind_accuracy"] = _ratio(sum(1 for g, p in pairs if g["kind"] == p["kind"]), len(pairs))
    result["width_error_m"] = _mean(abs(p["width"] - g["width"]) for g, p in pairs)
    result["sill_error_m"] = _mean(abs(p["sill"] - g["sill"]) for g, p in pairs if g["kind"] == "window" and p["kind"] == "window")
    result["head_error_m"] = _mean(abs(p["head"] - g["head"]) for g, p in pairs)
    result["center_error_m"] = _mean(float(np.linalg.norm(np.subtract(p["center"], g["center"]))) for g, p in pairs)
    hosts = [(g, p) for g, p in pairs if p.get("host")]
    result["host_correct"] = _ratio(sum(1 for g, p in hosts if wall_assign.get(p["host"]) == g["host"]), len(pairs))
    result["confusions"] = [f"{g['id']}:{g['kind']}->{p['kind']}" for g, p in pairs if g["kind"] != p["kind"]]
    matched_g = {g for g, _p, _s in matches}
    matched_p = {p for _g, p, _s in matches}
    result["missed"] = [f"{o['id']}:{o['kind']} w{o['width']:.2f}" for i, o in enumerate(gt_openings) if i not in matched_g]
    result["false"] = [f"{o['id']}:{o['kind']} w{o['width']:.2f}" for i, o in enumerate(pred_openings) if i not in matched_p]
    result["pairs"] = [{"gt": g["id"], "pred": p["id"], "gt_kind": g["kind"], "pred_kind": p["kind"], "gt_width": g["width"], "pred_width": p["width"],
                        "gt_sill": g["sill"], "pred_sill": p["sill"], "gt_head": g["head"], "pred_head": p["head"]} for g, p in pairs]
    return result


def _canon(label: str | None) -> str:
    text = (label or "").strip().lower()
    return LABEL_ALIASES.get(text, text)


def furniture_metrics(gt_items: list[dict[str, Any]], pred_items: list[dict[str, Any]]) -> dict[str, Any]:
    gpolys = [polygon_from_list(g["footprint"]) for g in gt_items]
    ppolys = [p["footprint"] for p in pred_items]
    def zov(g: dict[str, Any], p: dict[str, Any]) -> bool:
        return min(g["z1"], p["z1"]) - max(g["z0"], p["z0"]) > 0.02

    # stacked items (sink on worktop, upper cabinets over the base run) overlap in plan only: down-weight them
    score = np.array([[_iou(gpolys[i], ppolys[j]) * (1.0 if zov(gt_items[i], pred_items[j]) else 0.3) for j in range(len(ppolys))]
                      for i in range(len(gpolys))]).reshape(len(gpolys), len(ppolys))
    matches = _hungarian(score, 0.10)
    matches = [(gi, pi, _iou(gpolys[gi], ppolys[pi])) for gi, pi, _s in matches]
    pairs = []
    for gi, pi, iou in matches:
        g, p = gt_items[gi], pred_items[pi]
        inter = gpolys[gi].intersection(ppolys[pi]).area
        zo = max(0.0, min(g["z1"], p["z1"]) - max(g["z0"], p["z0"]))
        vg = gpolys[gi].area * (g["z1"] - g["z0"])
        vp = ppolys[pi].area * max(p["z1"] - p["z0"], 1e-6)
        vi = inter * zo
        iou3 = vi / (vg + vp - vi) if vg + vp - vi > 0 else 0.0
        pairs.append({
            "gt": g["id"], "pred": p["id"], "gt_label": g["label"], "pred_label": p["label"], "variant": g.get("variant", ""),
            "iou2d": iou, "iou3d": float(iou3), "label_ok": _canon(p["label"]) == _canon(g["label"]),
            "group_ok": label_group(_canon(p["label"])) == g["group"],
            "height_error": abs((p["z1"] - p["z0"]) - (g["z1"] - g["z0"])), "top_error": abs(p["z1"] - g["z1"]),
            "z_source": p.get("z_source", ""), "parts": p.get("parts", 0),
        })
    # separation of touching / close objects (one-to-one)
    def covers(gi: int, pi: int) -> bool:
        gp, pp = gpolys[gi], ppolys[pi]
        return not pp.is_empty and zov(gt_items[gi], pred_items[pi]) and gp.intersection(pp).area > 0.3 * min(gp.area, pp.area)

    cover_gt = [[pi for pi in range(len(ppolys)) if covers(gi, pi)] for gi in range(len(gpolys))]
    cover_pred = [[gi for gi in range(len(gpolys)) if covers(gi, pi)] for pi in range(len(ppolys))]
    merged = sorted({gt_items[gi]["id"] for pi, gl in enumerate(cover_pred) if len(gl) > 1 for gi in gl})
    split = [gt_items[gi]["id"] for gi, pl in enumerate(cover_gt) if len(pl) > 1]
    one_to_one = [gi for gi, pl in enumerate(cover_gt) if len(pl) == 1 and len(cover_pred[pl[0]]) == 1]
    close = [gi for gi, gp in enumerate(gpolys) if any(gj != gi and gp.distance(o) < 0.05 for gj, o in enumerate(gpolys)
                                                         if (gt_items[gj]["z1"] > gt_items[gi]["z0"] and gt_items[gi]["z1"] > gt_items[gj]["z0"]))]
    close_ok = [gi for gi in close if gi in one_to_one]
    matched_g = {gi for gi, _pi, _s in matches}
    by_variant: dict[str, list[float]] = {}
    for gi, g in enumerate(gt_items):
        m = next((x for x in pairs if x["gt"] == g["id"]), None)
        by_variant.setdefault(g.get("variant") or g["label"], []).append(m["iou2d"] if m else 0.0)
    return {
        "gt_count": len(gt_items), "pred_count": len(pred_items), "matched": len(pairs),
        "precision": _ratio(len(pairs), len(pred_items)), "recall": _ratio(len(pairs), len(gt_items)),
        "label_accuracy": _ratio(sum(p["label_ok"] for p in pairs), len(pairs)),
        "group_accuracy": _ratio(sum(p["group_ok"] for p in pairs), len(pairs)),
        "footprint_iou": _mean(p["iou2d"] for p in pairs), "box3d_iou": _mean(p["iou3d"] for p in pairs),
        "height_error_m": _mean(p["height_error"] for p in pairs), "top_error_m": _mean(p["top_error"] for p in pairs),
        "one_to_one_rate": _ratio(len(one_to_one), len(gt_items)), "merged": merged, "split": split,
        "close_pairs_separated": _ratio(len(close_ok), len(close)), "close_count": len(close),
        "missed": [f"{g['id']}:{g['variant'] or g['label']}" for gi, g in enumerate(gt_items) if gi not in matched_g],
        "false": [f"{p['id']}:{p['label']}" for pi, p in enumerate(pred_items) if pi not in {x[1] for x in matches}],
        "iou_by_variant": {k: float(np.mean(v)) for k, v in sorted(by_variant.items())},
        "pairs": pairs,
    }


_SCREEN_NAMES = frozenset({"television", "monitor", "screen", "laptop", "computer monitor", "tv"})
_HANGING_VARIANTS = frozenset({"wall_shelf", "tv_wall", "monitor"})


def object_metrics(
    gt: dict[str, Any], adapted: dict[str, Any], plan: dict[str, Any], frame: PlanFrame, furniture: dict[str, Any],
) -> dict[str, Any]:
    """Room objects: shelf / screen bottom edges, screens, standing pieces wrongly lifted, ceiling objects."""
    gt_items, pred_items = gt["furniture"], adapted["furniture"]
    by_gt = {item["id"]: item for item in gt_items}
    by_pred = {item["id"]: item for item in pred_items}
    bottom, lifted = [], []
    for pair in furniture.get("pairs", []):
        g, p = by_gt[pair["gt"]], by_pred[pair["pred"]]
        if g.get("variant") in _HANGING_VARIANTS:
            bottom.append(abs(p["z0"] - g["z0"]))
        elif g["z0"] < 0.05 and p["z0"] > 0.15:
            lifted.append(g["id"])
    hanging = [item for item in gt_items if item.get("variant") in _HANGING_VARIANTS]

    def matched(truth: list[dict[str, Any]], found: list[dict[str, Any]], reach: float) -> list[tuple[int, int, float]]:
        score = np.array([
            [max(0.0, 1.0 - float(np.linalg.norm(np.subtract(g["xy"], p["xy"]))) / reach) for p in found] for g in truth
        ]).reshape(len(truth), len(found))
        return _hungarian(score, 1e-6)

    truth_screens = [
        {**item, "xy": polygon_from_list(item["footprint"]).centroid.coords[0]}
        for item in gt_items if str(item["label"]).lower() in _SCREEN_NAMES
    ]
    found_screens = [
        {**item, "xy": item["footprint"].centroid.coords[0]}
        for item in pred_items if str(item["label"]).lower() in _SCREEN_NAMES and not item["footprint"].is_empty
    ]
    screens = matched(truth_screens, found_screens, 0.30)
    sizes = []
    for gi, pi, _score in screens:
        g, p = truth_screens[gi], found_screens[pi]
        widths = []
        for polygon in (polygon_from_list(g["footprint"]), p["footprint"]):
            corners = np.asarray(polygon.minimum_rotated_rectangle.exterior.coords)[:3]
            widths.append(float(max(np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[1]))))
        sizes.append(max(abs(widths[0] - widths[1]), abs((g["z1"] - g["z0"]) - (p["z1"] - p["z0"]))))

    truth_ceiling = [{**item, "xy": item["center"]} for item in gt.get("ceiling_objects", [])]
    found_ceiling = []
    for item in plan.get("ceiling_objects") or []:
        polygon = frame.polygon(item.get("footprint") or [])
        if polygon.is_empty:
            continue
        found_ceiling.append({
            "xy": polygon.centroid.coords[0], "label": item.get("semantic_label"), "z0": frame.z(float(item["z_min"])),
        })
    ceiling = matched(truth_ceiling, found_ceiling, 0.30)

    def share(part: int, whole: int) -> float | None:
        return _ratio(part, whole) if whole else None

    return {
        "bottom_error_m": _mean(bottom) if bottom else None,
        "bottom_within_3cm": share(sum(value <= 0.03 for value in bottom), len(hanging)),
        "lifted_standing": lifted,
        "screen_recall": share(len(screens), len(truth_screens)),
        "screen_precision": share(len(screens), len(found_screens)),
        "screen_false": len(found_screens) - len(screens),
        "screen_size_error_m": _mean(sizes) if sizes else None,
        "ceiling_recall": share(len(ceiling), len(truth_ceiling)),
        "ceiling_precision": share(len(ceiling), len(found_ceiling)),
        "ceiling_false": len(found_ceiling) - len(ceiling),
        "ceiling_label_accuracy": share(
            sum(truth_ceiling[gi]["label"] == found_ceiling[pi]["label"] for gi, pi, _s in ceiling), len(ceiling),
        ),
        "ceiling_bottom_error_m": _mean(
            abs(truth_ceiling[gi]["z0"] - found_ceiling[pi]["z0"]) for gi, pi, _s in ceiling
        ) if ceiling else None,
    }


def _item_centroid(frame: PlanFrame, item: dict[str, Any]) -> np.ndarray | None:
    for key in ("footprint", "polygon", "outline"):
        if item.get(key):
            poly = frame.polygon(item[key])
            if not poly.is_empty:
                return np.array([poly.centroid.x, poly.centroid.y])
    for key in ("center", "centre", "position"):
        if item.get(key) is not None:
            return frame.xy([item[key][:2]])[0]
    for a, b in (("p0", "p1"), ("start", "end")):
        if item.get(a) is not None and item.get(b) is not None:
            pts = frame.xy([item[a][:2], item[b][:2]])
            return pts.mean(axis=0)
    if item.get("centreline"):
        return frame.xy([p[:2] for p in item["centreline"]]).mean(axis=0)
    return None


def structure_metrics(gt: dict[str, Any], plan: dict[str, Any], frame: PlanFrame, ifc_products: list[dict[str, Any]] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ifc_map = {"columns": ("IfcColumn",), "beams": ("IfcBeam",), "stairs": ("IfcStair", "IfcStairFlight")}
    for key in ("columns", "beams", "stairs"):
        gts = gt.get(key, [])
        gpolys = [polygon_from_list(g["footprint"]).buffer(0.4) for g in gts]
        preds = [c for c in (_item_centroid(frame, item) for item in plan.get(key) or []) if c is not None]
        tp = sum(1 for gp in gpolys if any(gp.contains(Point(*c)) for c in preds))
        ifc_cents = [p["centroid"][:2] for p in (ifc_products or []) if p["type"] in ifc_map[key]]
        if key == "stairs":  # one IfcStair may aggregate flights; count distinct
            ifc_cents = ifc_cents[:]
        tp_ifc = sum(1 for gp in gpolys if any(gp.contains(Point(*c)) for c in ifc_cents))
        out[key] = {"gt": len(gts), "plan": len(preds), "plan_tp": tp, "plan_recall": _ratio(tp, len(gts)), "plan_precision": _ratio(tp, len(preds)),
                    "ifc": len(ifc_cents), "ifc_tp": tp_ifc, "ifc_recall": _ratio(tp_ifc, len(gts))}
    return out


# ---------------------------------------------------------------------------------------------------------------
# plan adapter
# ---------------------------------------------------------------------------------------------------------------

def adapt_plan(plan: dict[str, Any], frame: PlanFrame) -> dict[str, Any]:
    walls = []
    for w in plan.get("walls", []):
        cl = w.get("centreline") or []
        if len(cl) < 2:
            continue
        pts = frame.xy([cl[0], cl[-1]])
        walls.append({"id": w["id"], "p0": pts[0], "p1": pts[1], "thickness": frame.metres(w.get("thickness_m") or 0.0),
                      "length": float(np.linalg.norm(pts[1] - pts[0])), "kind": w.get("kind", "")})
    rooms = []
    for r in plan.get("rooms", []):
        poly = frame.polygon(r.get("polygon") or {})
        ceiling = None
        if r.get("ceiling_z") is not None:
            ceiling = frame.z(float(r["ceiling_z"]))
        rooms.append({"id": r.get("id"), "name": r.get("name"), "polygon": poly, "area_m2": float(r.get("area_m2") or 0.0), "ceiling_h": ceiling})
    openings = []
    for o in plan.get("openings", []):
        c = frame.xy([o["center"]])[0]
        openings.append({"id": o["id"], "kind": o.get("kind"), "center": c, "width": frame.metres(o.get("width_m")),
                         "sill": frame.metres(o.get("sill_m") or 0.0), "head": frame.metres(o.get("head_m")), "host": o.get("host_wall_id"),
                         "angle": frame.angle(float(o.get("axis_angle") or 0.0))})
    furniture = []
    for f in plan.get("furniture", []):
        poly = frame.polygon(f.get("footprint") or [])
        if f.get("z_min") is not None and f.get("z_max") is not None:
            z0, z1 = frame.z(float(f["z_min"])), frame.z(float(f["z_max"]))
        elif f.get("default_z_m"):
            z0, z1 = (frame.metres(v) for v in f["default_z_m"])
        else:
            z0, z1 = 0.0, frame.metres(f.get("default_height_m") or 0.8)
        furniture.append({"id": f["id"], "label": f.get("semantic_label") or f.get("label"), "footprint": poly, "z0": z0, "z1": z1,
                          "z_source": f.get("z_source", ""), "parts": len(f.get("parts") or []), "instance_id": f.get("instance_id")})
    return {"walls": walls, "rooms": rooms, "openings": openings, "furniture": furniture}


# ---------------------------------------------------------------------------------------------------------------
# IFC / DXF
# ---------------------------------------------------------------------------------------------------------------

def ifc_products(path: Path, frame: PlanFrame) -> list[dict[str, Any]]:
    import ifcopenshell
    import ifcopenshell.geom

    model = ifcopenshell.open(str(path))
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    products = []
    iterator = ifcopenshell.geom.iterator(settings, model, 1)
    if iterator.initialize():
        while True:
            shape = iterator.get()
            try:
                element = model.by_guid(shape.guid)
                verts = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)
                faces = np.asarray(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
                if len(verts) and len(faces):
                    scene_v = frame.ifc_to_scene(verts)
                    products.append({"guid": shape.guid, "type": element.is_a(), "name": getattr(element, "Name", None) or "",
                                     "vertices": scene_v, "faces": faces, "centroid": scene_v.mean(axis=0),
                                     "bbox": (scene_v.min(axis=0), scene_v.max(axis=0))})
            except Exception:  # noqa: BLE001 - a broken product must not stop the evaluation
                pass
            if not iterator.next():
                break
    return products


def _sample(products: list[dict[str, Any]], count_per_m2: float, rng: np.random.Generator) -> np.ndarray:
    from .mesh import sample_surface

    parts = []
    for p in products:
        tri = p["vertices"][p["faces"]]
        area = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
        parts.append(sample_surface(p["vertices"], p["faces"], int(min(150_000, max(30, area * count_per_m2))), rng))
    return np.vstack(parts) if parts else np.zeros((0, 3))


def surface_metrics(products: list[dict[str, Any]], samples: dict[str, np.ndarray], rooms: list[Polygon], height: float, tau: float = 0.05) -> dict[str, Any]:
    from shapely import contains_xy

    rng = np.random.default_rng(0)
    inside = unary_union([r.buffer(0.02) for r in rooms]) if rooms else Polygon()
    groups = {
        "wall": ("IfcWall", "IfcWallStandardCase"), "floor": ("IfcSlab",), "ceiling": ("IfcCovering",),
        "furniture": FURNITURE_IFC, "column": ("IfcColumn",), "beam": ("IfcBeam",), "stair": ("IfcStair", "IfcStairFlight", "IfcSlab_stair"),
    }
    out: dict[str, Any] = {}
    for kind, types in groups.items():
        gt = samples.get(kind)
        if gt is None or len(gt) == 0:
            continue
        chosen = [p for p in products if p["type"] in types]
        if kind == "floor":
            chosen = [p for p in chosen if p["centroid"][2] < height / 2]
        if not chosen:
            out[kind] = {"completeness": 0.0, "accuracy_median_m": _nan(), "chamfer_m": _nan(), "products": 0}
            continue
        pred = _sample(chosen, 400.0, rng)
        pred_in = pred[contains_xy(inside, pred[:, 0], pred[:, 1])] if len(pred) else pred
        if kind == "wall":
            pred_in = pred_in[(pred_in[:, 2] > 0.01) & (pred_in[:, 2] < height - 0.01)]
        d_gt = cKDTree(pred).query(gt, k=1)[0] if len(pred) else np.full(len(gt), np.inf)
        d_pred = cKDTree(gt).query(pred_in, k=1)[0] if len(pred_in) else np.zeros(0)
        out[kind] = {
            "completeness": float(np.mean(d_gt < tau)), "accuracy": float(np.mean(d_pred < tau)) if len(d_pred) else _nan(),
            "gt_to_ifc_median_m": float(np.median(d_gt)), "ifc_to_gt_median_m": float(np.median(d_pred)) if len(d_pred) else _nan(),
            "chamfer_m": float(0.5 * (np.mean(np.minimum(d_gt, 1.0)) + (np.mean(np.minimum(d_pred, 1.0)) if len(d_pred) else 1.0))),
            "products": len(chosen),
        }
    return out


def furniture_chamfer(products: list[dict[str, Any]], gt_items: list[dict[str, Any]], gt_mesh: dict[str, Any]) -> dict[str, Any]:
    """Match IFC furniture products to GT items by 3D box overlap and compute the symmetric Chamfer distance per pair."""
    from .mesh import sample_surface

    rng = np.random.default_rng(1)
    furn = [p for p in products if p["type"] in FURNITURE_IFC]
    vertices, faces, face_element, elements = gt_mesh["vertices"], gt_mesh["faces"], gt_mesh["face_element"], gt_mesh["elements"]
    index_of = {e["id"]: i for i, e in enumerate(elements)}

    def box_iou(a, b) -> float:
        lo = np.maximum(a[0], b[0])
        hi = np.minimum(a[1], b[1])
        inter = float(np.prod(np.clip(hi - lo, 0, None)))
        va, vb = float(np.prod(a[1] - a[0])), float(np.prod(b[1] - b[0]))
        return inter / (va + vb - inter) if va + vb - inter > 0 else 0.0

    gboxes = []
    for g in gt_items:
        poly = polygon_from_list(g["footprint"])
        minx, miny, maxx, maxy = poly.bounds
        gboxes.append((np.array([minx, miny, g["z0"]]), np.array([maxx, maxy, g["z1"]])))
    score = np.array([[box_iou(gb, p["bbox"]) for p in furn] for gb in gboxes]).reshape(len(gboxes), len(furn))
    matches = _hungarian(score, 0.05)
    pairs = []
    for gi, pi, iou in matches:
        eidx = index_of.get(gt_items[gi]["id"])
        if eidx is None:
            continue
        sel = faces[face_element == eidx]
        gpts = sample_surface(vertices, sel, 3000, rng)
        ppts = sample_surface(furn[pi]["vertices"], furn[pi]["faces"], 3000, rng)
        d1 = cKDTree(ppts).query(gpts, k=1)[0]
        d2 = cKDTree(gpts).query(ppts, k=1)[0]
        pairs.append({"gt": gt_items[gi]["id"], "variant": gt_items[gi].get("variant", ""), "ifc_type": furn[pi]["type"], "box_iou": iou,
                      "chamfer_m": float(0.5 * (d1.mean() + d2.mean())), "gt_to_ifc_p90_m": float(np.percentile(d1, 90))})
    return {"ifc_furniture": len(furn), "matched": len(pairs), "chamfer_m": _mean(p["chamfer_m"] for p in pairs),
            "chamfer_by_variant": {v: _mean(p["chamfer_m"] for p in pairs if p["variant"] == v) for v in sorted({p["variant"] for p in pairs})},
            "ifc_classes": {t: sum(1 for p in furn if p["type"] == t) for t in sorted({p["type"] for p in furn})}, "pairs": pairs}


def ifc_junction_gaps(truth: dict[str, Any], products: list[dict[str, Any]]) -> int:
    """GT wall junctions not covered (10 cm) by the plan projection of any IFC wall."""
    from shapely.geometry import MultiPoint

    hulls = [MultiPoint(p["vertices"][:, :2]).convex_hull for p in products if p["type"] in ("IfcWall", "IfcWallStandardCase")]
    union = unary_union(hulls) if hulls else Polygon()
    return sum(1 for j in truth.get("junctions", []) if union.distance(Point(*j["point"])) > 0.10)


def dxf_metrics(path: Path) -> dict[str, Any]:
    import ezdxf

    doc = ezdxf.readfile(str(path))
    auditor = doc.audit()
    seen: dict[tuple, int] = {}
    duplicates = 0
    open_closed = 0
    layers: dict[str, int] = {}
    for e in doc.modelspace():
        layer = e.dxf.layer
        layers[layer] = layers.get(layer, 0) + 1
        key: tuple | None = None
        if e.dxftype() == "LINE":
            a, b = sorted([(round(e.dxf.start.x, 3), round(e.dxf.start.y, 3)), (round(e.dxf.end.x, 3), round(e.dxf.end.y, 3))])
            key = ("LINE", layer, a, b)
        elif e.dxftype() == "LWPOLYLINE":
            pts = tuple((round(float(p[0]), 3), round(float(p[1]), 3)) for p in e.get_points("xy"))
            key = ("LWPOLYLINE", layer, pts)
            if layer in ("A-WALL", "A-AREA", "A-FURN") and not e.closed and len(pts) > 2 and pts[0] != pts[-1]:
                open_closed += 1
        elif e.dxftype() == "INSERT":
            key = ("INSERT", layer, e.dxf.name, (round(e.dxf.insert.x, 3), round(e.dxf.insert.y, 3)))
        if key is not None:
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 1:
                duplicates += 1
    return {"audit_errors": len(auditor.errors), "audit_fixes": len(auditor.fixes), "duplicates": duplicates,
            "open_polylines_expected_closed": open_closed, "entities": sum(layers.values()), "layers": layers}


# ---------------------------------------------------------------------------------------------------------------
# case evaluation
# ---------------------------------------------------------------------------------------------------------------

def find_outputs(project: Path) -> dict[str, Path | None]:
    dxf_levels = sorted((project / "products" / "dxf").glob("*/ortho/plan_model.json"))
    plan = dxf_levels[0] if dxf_levels else None
    level = plan.parent.parent.name if plan else "level_1"
    ifcs = sorted((project / "products" / "bim" / level).glob("*.ifc"))
    dxfs = sorted((project / "products" / "dxf" / "export").glob("*_clean.dxf"))
    qa = project / "products" / "bim" / level / "qa.json"
    return {"plan": plan, "ifc": ifcs[0] if ifcs else None, "dxf": dxfs[0] if dxfs else None, "qa": qa if qa.is_file() else None,
            "run_metrics": project / "reports" / "dxf_run_metrics.json", "auto": project / "reports" / "auto_dxf.json", "level_count": len(dxf_levels)}


def evaluate_case(case_dir: Path, project: Path | None = None) -> dict[str, Any]:
    case_dir = Path(case_dir)
    truth = json.loads((case_dir / "gt" / "ground_truth.json").read_text(encoding="utf-8"))
    if project is None:
        project = case_dir / truth["name"]
    out = find_outputs(project)
    result: dict[str, Any] = {"name": truth["name"], "family": truth["family"], "seed": truth["seed"], "features": truth.get("features", []),
                              "scan": truth.get("scan", {}), "errors": []}
    auto = _read_json(out["auto"]) if out["auto"] and Path(out["auto"]).is_file() else {}
    run = _read_json(out["run_metrics"]) if out["run_metrics"] and Path(out["run_metrics"]).is_file() else {}
    result["pipeline"] = {"status": auto.get("status", "missing"), "reason": auto.get("reason", ""), "auto_seconds": auto.get("total_seconds"),
                          "run_seconds": run.get("total_seconds"), "steps": run.get("step_seconds", {}), "levels_detected": out["level_count"]}
    if out["plan"] is None:
        result["errors"].append("no plan_model.json")
        result["summary"] = summarise(result)
        return result
    plan = json.loads(Path(out["plan"]).read_text(encoding="utf-8"))
    frame = PlanFrame(plan, truth)
    adapted = adapt_plan(plan, frame)
    rooms_gt = [polygon_from_list(r["polygon"]) for r in truth["rooms"]]
    storey_true = (float(plan["ceiling_z"]) - float(plan["floor_z"])) / frame.scale if plan.get("ceiling_z") is not None else _nan()
    result["frame"] = {"units_per_metre": frame.upm, "scale_error": frame.upm / frame.scale - 1.0, "storey_height_pred": storey_true,
                       "storey_height_gt": truth["storey_height"], "max_height_gt": truth["max_height"]}
    result["walls"] = wall_metrics(truth, adapted["walls"], rooms_gt)
    result["rooms"] = room_metrics(truth["rooms"], adapted["rooms"])
    result["openings"] = opening_metrics(truth["openings"], adapted["openings"], result["walls"]["assign"])
    result["furniture"] = furniture_metrics(truth["furniture"], adapted["furniture"])
    result["objects"] = object_metrics(truth, adapted, plan, frame, result["furniture"])
    products = None
    if out["ifc"] is not None:
        try:
            products = ifc_products(Path(out["ifc"]), frame)
            samples = dict(np.load(case_dir / "gt" / "class_samples.npz"))
            result["surfaces"] = surface_metrics(products, samples, rooms_gt, truth["max_height"])
            mesh = np.load(case_dir / "gt" / "mesh.npz")
            gt_mesh = {"vertices": mesh["vertices"].astype(float), "faces": mesh["faces"], "face_element": mesh["face_element"],
                       "elements": json.loads(str(mesh["elements"]))}
            result["furniture_chamfer"] = furniture_chamfer(products, truth["furniture"], gt_mesh)
            result["ifc_counts"] = {t: sum(1 for p in products if p["type"] == t) for t in sorted({p["type"] for p in products})}
        except Exception as error:  # noqa: BLE001
            result["errors"].append(f"IFC evaluation failed: {type(error).__name__}: {error}")
    else:
        result["errors"].append("no IFC")
    result["structure"] = structure_metrics(truth, plan, frame, products)
    if products is not None:
        result["walls"]["ifc_junction_gaps"] = ifc_junction_gaps(truth, products)
    if out["qa"] is not None:
        qa = _read_json(out["qa"])
        result["qa"] = {"summary": qa.get("summary"), "express_findings": qa.get("express_findings"), "ids_failures": qa.get("ids_failures"),
                        "reimport_ok": qa.get("reimport_ok"), "wall_overlaps": qa.get("wall_overlap_count"),
                        "ifc4_ok": str(qa.get("summary", "")).startswith("IFC4 OK")}
    if out["dxf"] is not None:
        try:
            result["dxf"] = dxf_metrics(Path(out["dxf"]))
        except Exception as error:  # noqa: BLE001
            result["errors"].append(f"DXF evaluation failed: {type(error).__name__}: {error}")
    else:
        result["errors"].append("no DXF")
    result["summary"] = summarise(result)
    return result


def summarise(r: dict[str, Any]) -> dict[str, Any]:
    w, ro, o, f = r.get("walls", {}), r.get("rooms", {}), r.get("openings", {}), r.get("furniture", {})
    s = r.get("structure", {})
    surf = r.get("surfaces", {})
    obj = r.get("objects", {})
    return {
        "shelf_bottom_err": obj.get("bottom_error_m"), "shelf_bottom_3cm": obj.get("bottom_within_3cm"),
        "lifted": len(obj["lifted_standing"]) if "lifted_standing" in obj else None,
        "screen_R": obj.get("screen_recall"), "screen_P": obj.get("screen_precision"),
        "screen_size_err": obj.get("screen_size_error_m"),
        "ceil_R": obj.get("ceiling_recall"), "ceil_P": obj.get("ceiling_precision"),
        "ceil_false": obj.get("ceiling_false"), "screen_false": obj.get("screen_false"),
        "ceil_label_acc": obj.get("ceiling_label_accuracy"), "ceil_bottom_err": obj.get("ceiling_bottom_error_m"),
        "status": r.get("pipeline", {}).get("status"),
        "runtime_s": r.get("pipeline", {}).get("auto_seconds"),
        "scale_error": r.get("frame", {}).get("scale_error"),
        "wall_P": w.get("precision"), "wall_R": w.get("recall"), "wall_cov": w.get("length_coverage"),
        "wall_face_err": w.get("face_error_m"), "wall_ctr_err": w.get("centre_offset_m"), "wall_ang_err": w.get("angle_error_deg"),
        "wall_thk_err": w.get("thickness_error_m"), "free_ends": w.get("free_ends"), "overlaps": w.get("parallel_overlaps"),
        "junction_gaps": w.get("junction_gaps"), "ifc_junction_gaps": w.get("ifc_junction_gaps"),
        "rooms_gt": ro.get("gt_count"), "rooms_pred": ro.get("pred_count"), "room_iou": ro.get("mean_iou"), "room_area_err": ro.get("area_error_rel"),
        "room_area_field_err": ro.get("area_field_error_rel"),
        "door_P": o.get("door", {}).get("precision"), "door_R": o.get("door", {}).get("recall"),
        "win_P": o.get("window", {}).get("precision"), "win_R": o.get("window", {}).get("recall"),
        "pass_P": o.get("passage", {}).get("precision"), "pass_R": o.get("passage", {}).get("recall"),
        "open_width_err": o.get("width_error_m"), "open_sill_err": o.get("sill_error_m"), "open_head_err": o.get("head_error_m"),
        "open_host_ok": o.get("host_correct"),
        "furn_P": f.get("precision"), "furn_R": f.get("recall"), "furn_label_acc": f.get("label_accuracy"), "furn_group_acc": f.get("group_accuracy"),
        "furn_iou2d": f.get("footprint_iou"), "furn_iou3d": f.get("box3d_iou"), "furn_h_err": f.get("height_error_m"),
        "furn_1to1": f.get("one_to_one_rate"), "furn_close_sep": f.get("close_pairs_separated"),
        "furn_chamfer": r.get("furniture_chamfer", {}).get("chamfer_m"),
        "col_R": s.get("columns", {}).get("plan_recall") if s.get("columns", {}).get("gt") else None,
        "beam_R": s.get("beams", {}).get("plan_recall") if s.get("beams", {}).get("gt") else None,
        "stair_R": s.get("stairs", {}).get("plan_recall") if s.get("stairs", {}).get("gt") else None,
        "col_R_ifc": s.get("columns", {}).get("ifc_recall") if s.get("columns", {}).get("gt") else None,
        "beam_R_ifc": s.get("beams", {}).get("ifc_recall") if s.get("beams", {}).get("gt") else None,
        "stair_R_ifc": s.get("stairs", {}).get("ifc_recall") if s.get("stairs", {}).get("gt") else None,
        "wall_completeness": surf.get("wall", {}).get("completeness"), "wall_chamfer": surf.get("wall", {}).get("chamfer_m"),
        "floor_completeness": surf.get("floor", {}).get("completeness"), "ceiling_completeness": surf.get("ceiling", {}).get("completeness"),
        "furn_completeness": surf.get("furniture", {}).get("completeness"),
        "ifc4_ok": r.get("qa", {}).get("ifc4_ok"), "express": r.get("qa", {}).get("express_findings"), "ids_fail": r.get("qa", {}).get("ids_failures"),
        "dxf_audit_err": r.get("dxf", {}).get("audit_errors"), "dxf_dup": r.get("dxf", {}).get("duplicates"), "dxf_open": r.get("dxf", {}).get("open_polylines_expected_closed"),
    }


def _read_json(path: Path | str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
