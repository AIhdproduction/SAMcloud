"""Generic diff of two plan_model.json files (real-data regression checks).

Both plans are mapped to their source (world) coordinates with their own frame, so a changed alignment or level does
not show up as a wall shift.  Lengths are reported in world units of the point cloud (metres for scaled projects);
``*_m`` fields are converted the same way (value * units_per_metre / scale).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon


class _Frame:
    def __init__(self, plan: dict[str, Any]) -> None:
        frame = plan.get("frame") or {}
        self.origin = np.asarray(frame.get("origin", [0, 0, 0]), dtype=float)
        self.scale = float(frame.get("scale_factor", 1.0) or 1.0)
        self.rotation = np.asarray(frame.get("rotation", np.eye(3)), dtype=float)
        self.upm = float(plan.get("units_per_metre") or 1.0)
        self.floor_z = float(plan.get("floor_z") or 0.0)

    def xy(self, pts) -> np.ndarray:
        arr = np.asarray(pts, dtype=float).reshape(-1, 2)
        local = np.hstack((arr, np.full((len(arr), 1), self.floor_z)))
        return ((local @ self.rotation) / self.scale + self.origin)[:, :2]

    def z(self, value: float) -> float:
        return float(value) / self.scale

    def m(self, value: float | None) -> float | None:
        return None if value is None else float(value) * self.upm / self.scale

    def poly(self, data: Any) -> Polygon:
        if isinstance(data, dict):
            ext = self.xy(data.get("exterior", []))
            holes = [self.xy(h) for h in data.get("holes", []) or [] if len(h) >= 3]
            return Polygon(ext, holes).buffer(0) if len(ext) >= 3 else Polygon()
        pts = self.xy(data)
        return Polygon(pts).buffer(0) if len(pts) >= 3 else Polygon()


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "max": None, "p90": None}
    arr = np.abs(np.asarray(values, dtype=float))
    return {"mean": float(arr.mean()), "max": float(arr.max()), "p90": float(np.percentile(arr, 90))}


def _walls(plan: dict[str, Any], frame: _Frame) -> list[dict[str, Any]]:
    out = []
    for w in plan.get("walls", []):
        cl = w.get("centreline") or []
        if len(cl) < 2:
            continue
        p = frame.xy([cl[0], cl[-1]])
        d = p[1] - p[0]
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            continue
        out.append({"id": w["id"], "p0": p[0], "p1": p[1], "d": d / length, "length": length, "thickness": frame.m(w.get("thickness_m") or 0.0), "kind": w.get("kind")})
    return out


def compare_walls(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    used_b: set[int] = set()
    score = np.zeros((len(a), len(b)))
    detail: dict[tuple[int, int], dict[str, float]] = {}
    for i, wa in enumerate(a):
        n = np.array([-wa["d"][1], wa["d"][0]])
        for j, wb in enumerate(b):
            cos = abs(float(np.dot(wa["d"], wb["d"])))
            angle = math.degrees(math.acos(min(1.0, cos)))
            if angle > 8.0:
                continue
            offs = [float(np.dot(q - wa["p0"], n)) for q in (wb["p0"], wb["p1"])]
            shift = float(np.mean(offs))
            if abs(shift) > 0.5 * ((wa["thickness"] or 0.2) + (wb["thickness"] or 0.2)) + 0.10:
                continue
            s = sorted(float(np.dot(q - wa["p0"], wa["d"])) for q in (wb["p0"], wb["p1"]))
            overlap = min(wa["length"], s[1]) - max(0.0, s[0])
            if overlap <= 0.05:
                continue
            score[i, j] = overlap / max(wa["length"], wb["length"])
            detail[(i, j)] = {"shift": shift, "angle": angle, "overlap": overlap}
    if score.size:
        rows, cols = linear_sum_assignment(-score)
        for i, j in zip(rows, cols):
            if score[i, j] > 0:
                d = detail[(int(i), int(j))]
                pairs.append({"base": a[i]["id"], "new": b[j]["id"], "shift": d["shift"], "angle_deg": d["angle"],
                              "thickness_change": (b[j]["thickness"] or 0) - (a[i]["thickness"] or 0),
                              "length_change": b[j]["length"] - a[i]["length"], "overlap_share": float(score[i, j])})
                used_b.add(int(j))
    used_a = {p["base"] for p in pairs}
    moved = [p for p in pairs if abs(p["shift"]) > 0.02 or p["angle_deg"] > 0.5 or abs(p["length_change"]) > 0.05 or abs(p["thickness_change"]) > 0.01]
    return {
        "base_count": len(a), "new_count": len(b), "matched": len(pairs),
        "base_length": float(sum(w["length"] for w in a)), "new_length": float(sum(w["length"] for w in b)),
        "shift": _stats([p["shift"] for p in pairs]), "angle_deg": _stats([p["angle_deg"] for p in pairs]),
        "thickness_change": _stats([p["thickness_change"] for p in pairs]), "length_change": _stats([p["length_change"] for p in pairs]),
        "unchanged": len(pairs) - len(moved), "changed": [p for p in moved],
        "only_base": [w["id"] for w in a if w["id"] not in used_a], "only_new": [w["id"] for j, w in enumerate(b) if j not in used_b],
    }


def compare_openings(a: dict[str, Any], b: dict[str, Any], fa: _Frame, fb: _Frame) -> dict[str, Any]:
    oa = [(o, fa.xy([o["center"]])[0]) for o in a.get("openings", [])]
    ob = [(o, fb.xy([o["center"]])[0]) for o in b.get("openings", [])]
    dist = np.array([[float(np.linalg.norm(ca - cb)) for _o, cb in ob] for _p, ca in oa]).reshape(len(oa), len(ob))
    pairs = []
    if dist.size:
        rows, cols = linear_sum_assignment(np.where(dist < 0.5, dist, 1e3))
        for i, j in zip(rows, cols):
            if dist[i, j] < 0.5:
                x, y = oa[i][0], ob[j][0]
                pairs.append({"base": x["id"], "new": y["id"], "base_kind": x.get("kind"), "new_kind": y.get("kind"), "shift": float(dist[i, j]),
                              "width_change": (fb.m(y.get("width_m")) or 0) - (fa.m(x.get("width_m")) or 0),
                              "sill_change": (fb.m(y.get("sill_m")) or 0) - (fa.m(x.get("sill_m")) or 0),
                              "head_change": (fb.m(y.get("head_m")) or 0) - (fa.m(x.get("head_m")) or 0)})
    kinds = lambda plan: {k: sum(1 for o in plan.get("openings", []) if o.get("kind") == k) for k in ("door", "window", "passage")}  # noqa: E731
    mb, mn = {p["base"] for p in pairs}, {p["new"] for p in pairs}
    return {"base_kinds": kinds(a), "new_kinds": kinds(b), "matched": len(pairs),
            "kind_changes": [f"{p['base']}:{p['base_kind']}->{p['new_kind']}" for p in pairs if p["base_kind"] != p["new_kind"]],
            "width_change": _stats([p["width_change"] for p in pairs]), "shift": _stats([p["shift"] for p in pairs]),
            "only_base": [f"{o['id']}:{o.get('kind')}" for o, _c in oa if o["id"] not in mb], "only_new": [f"{o['id']}:{o.get('kind')}" for o, _c in ob if o["id"] not in mn],
            "pairs": pairs}


def _iou(p: Polygon, q: Polygon) -> float:
    if p.is_empty or q.is_empty:
        return 0.0
    union = p.union(q).area
    return float(p.intersection(q).area / union) if union > 0 else 0.0


def compare_polys(items_a: list[tuple[str, str, Polygon, dict]], items_b: list[tuple[str, str, Polygon, dict]], minimum: float) -> dict[str, Any]:
    score = np.array([[_iou(pa, pb) for _i, _n, pb, _d in items_b] for _j, _m, pa, _e in items_a]).reshape(len(items_a), len(items_b))
    pairs = []
    if score.size:
        rows, cols = linear_sum_assignment(-score)
        for i, j in zip(rows, cols):
            if score[i, j] >= minimum:
                pairs.append((items_a[i], items_b[j], float(score[i, j])))
    mb, mn = {p[0][0] for p in pairs}, {p[1][0] for p in pairs}
    return {"pairs": pairs, "only_base": [f"{i}:{n}" for i, n, _p, _d in items_a if i not in mb], "only_new": [f"{i}:{n}" for i, n, _p, _d in items_b if i not in mn]}


def compare_plans(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    fa, fb = _Frame(base), _Frame(new)
    result: dict[str, Any] = {"frames": {"base_upm": fa.upm, "new_upm": fb.upm, "same_frame": bool(np.allclose(fa.rotation, fb.rotation) and np.allclose(fa.origin, fb.origin))}}
    result["walls"] = compare_walls(_walls(base, fa), _walls(new, fb))
    result["openings"] = compare_openings(base, new, fa, fb)
    rooms = compare_polys([(r["id"], r.get("name", ""), fa.poly(r.get("polygon") or {}), r) for r in base.get("rooms", [])],
                          [(r["id"], r.get("name", ""), fb.poly(r.get("polygon") or {}), r) for r in new.get("rooms", [])], 0.3)
    result["rooms"] = {"base_count": len(base.get("rooms", [])), "new_count": len(new.get("rooms", [])), "matched": len(rooms["pairs"]),
                       "pairs": [{"base": a[0], "new": b[0], "base_name": a[1], "new_name": b[1], "iou": iou, "area_base": float(a[2].area), "area_new": float(b[2].area),
                                  "area_change_rel": (b[2].area - a[2].area) / a[2].area if a[2].area else None} for a, b, iou in rooms["pairs"]],
                       "only_base": rooms["only_base"], "only_new": rooms["only_new"]}
    furn = compare_polys([(f["id"], f.get("semantic_label") or f.get("label", ""), fa.poly(f.get("footprint") or []), f) for f in base.get("furniture", [])],
                         [(f["id"], f.get("semantic_label") or f.get("label", ""), fb.poly(f.get("footprint") or []), f) for f in new.get("furniture", [])], 0.1)

    def height(frame: _Frame, item: dict[str, Any]) -> float | None:
        if item.get("z_min") is None or item.get("z_max") is None:
            return None
        return frame.z(item["z_max"]) - frame.z(item["z_min"])

    fpairs = [{"base": a[0], "new": b[0], "base_label": a[1], "new_label": b[1], "iou": iou,
               "height_change": (height(fb, b[3]) - height(fa, a[3])) if height(fa, a[3]) is not None and height(fb, b[3]) is not None else None,
               "parts_base": len(a[3].get("parts") or []), "parts_new": len(b[3].get("parts") or [])} for a, b, iou in furn["pairs"]]
    result["furniture"] = {"base_count": len(base.get("furniture", [])), "new_count": len(new.get("furniture", [])), "matched": len(fpairs),
                           "mean_iou": float(np.mean([p["iou"] for p in fpairs])) if fpairs else None,
                           "label_changes": [f"{p['base']}:{p['base_label']}->{p['new_label']}" for p in fpairs if p["base_label"] != p["new_label"]],
                           "height_change": _stats([p["height_change"] for p in fpairs if p["height_change"] is not None]),
                           "only_base": furn["only_base"], "only_new": furn["only_new"], "pairs": fpairs}
    result["structure"] = {k: {"base": len(base.get(k) or []), "new": len(new.get(k) or [])} for k in ("columns", "beams", "stairs")}
    result["new_keys"] = sorted(set(new.keys()) - set(base.keys()))
    return result


def summary_text(result: dict[str, Any]) -> str:
    w, o, r, f = result["walls"], result["openings"], result["rooms"], result["furniture"]

    def s(x: dict[str, Any], key: str = "max") -> str:
        v = x.get(key)
        return "-" if v is None else f"{v:.3f}"

    lines = [
        f"walls: base {w['base_count']} / new {w['new_count']}, matched {w['matched']}, unchanged {w['unchanged']}, changed {len(w['changed'])}; "
        f"shift mean {s(w['shift'], 'mean')} max {s(w['shift'])} m, angle max {s(w['angle_deg'])} deg, thickness change max {s(w['thickness_change'])} m; "
        f"only base {w['only_base']} only new {w['only_new']}",
        f"openings: base {o['base_kinds']} new {o['new_kinds']}, matched {o['matched']}, kind changes {o['kind_changes']}, width change max {s(o['width_change'])} m; "
        f"only base {o['only_base']} only new {o['only_new']}",
        "rooms: base {} / new {}, matched {}; ".format(r["base_count"], r["new_count"], r["matched"])
        + ", ".join(f"{p['base_name']}->{p['new_name']} IoU {p['iou']:.3f} area {p['area_base']:.2f}->{p['area_new']:.2f}" for p in r["pairs"])
        + f"; only base {r['only_base']} only new {r['only_new']}",
        f"furniture: base {f['base_count']} / new {f['new_count']}, matched {f['matched']}, mean IoU {f['mean_iou'] if f['mean_iou'] is None else round(f['mean_iou'], 3)}, "
        f"label changes {len(f['label_changes'])}, height change max {s(f['height_change'])} m; only base {f['only_base']} only new {f['only_new']}",
        f"structure: {result['structure']}; new keys {result['new_keys']}; same frame {result['frames']['same_frame']}",
    ]
    return "\n".join(lines)


def compare_files(base: Path, new: Path) -> dict[str, Any]:
    a = json.loads(Path(base).read_text(encoding="utf-8"))
    b = json.loads(Path(new).read_text(encoding="utf-8"))
    return compare_plans(a, b)
