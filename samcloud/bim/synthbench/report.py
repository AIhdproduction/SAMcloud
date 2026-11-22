"""Aggregate per-case ``metrics.json`` files into ``report.json`` and ``report.md``."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

COLUMNS = [
    ("status", "status", "s"), ("runtime_s", "t[s]", ".0f"), ("scale_error", "scale", "+.3f"),
    ("wall_P", "wall P", ".2f"), ("wall_R", "wall R", ".2f"), ("wall_cov", "cov", ".2f"), ("wall_face_err", "face[m]", ".3f"),
    ("wall_ang_err", "ang[deg]", ".2f"), ("wall_thk_err", "thk[m]", ".3f"), ("free_ends", "free", "d"), ("overlaps", "ovl", "d"),
    ("junction_gaps", "gapP", "d"), ("ifc_junction_gaps", "gapIFC", "d"),
    ("rooms_gt", "rm gt", "d"), ("rooms_pred", "rm pr", "d"), ("room_iou", "rm IoU", ".2f"), ("room_area_err", "area err", ".3f"),
    ("door_R", "door R", ".2f"), ("door_P", "door P", ".2f"), ("win_R", "win R", ".2f"), ("win_P", "win P", ".2f"),
    ("pass_R", "pass R", ".2f"), ("open_width_err", "w err", ".2f"), ("open_host_ok", "host", ".2f"),
    ("furn_R", "furn R", ".2f"), ("furn_P", "furn P", ".2f"), ("furn_label_acc", "label", ".2f"), ("furn_iou2d", "fIoU", ".2f"),
    ("furn_iou3d", "3D IoU", ".2f"), ("furn_h_err", "h err", ".2f"), ("furn_1to1", "1:1", ".2f"), ("furn_chamfer", "chamf", ".3f"),
    ("shelf_bottom_err", "bottom[m]", ".3f"), ("shelf_bottom_3cm", "bottom<3cm", ".2f"), ("lifted", "lifted", "d"),
    ("screen_R", "scr R", ".2f"), ("screen_P", "scr P", ".2f"), ("screen_false", "scr false", "d"),
    ("screen_size_err", "scr size", ".3f"), ("ceil_R", "clg R", ".2f"), ("ceil_P", "clg P", ".2f"),
    ("ceil_false", "clg false", "d"), ("ceil_label_acc", "clg type", ".2f"), ("ceil_bottom_err", "clg z", ".3f"),
    ("col_R", "col R", ".2f"), ("beam_R", "beam R", ".2f"), ("stair_R", "stair R", ".2f"),
    ("wall_completeness", "wall cpl", ".2f"), ("furn_completeness", "furn cpl", ".2f"),
    ("ifc4_ok", "IFC ok", "s"), ("express", "expr", "d"), ("ids_fail", "IDS", "d"), ("dxf_audit_err", "dxf err", "d"),
]

# metrics where lower is better (used only for wording in the aggregate table)
LOWER_BETTER = {"runtime_s", "wall_face_err", "wall_ctr_err", "wall_ang_err", "wall_thk_err", "free_ends", "overlaps", "junction_gaps", "ifc_junction_gaps",
                "room_area_err", "room_area_field_err", "open_width_err", "open_sill_err", "open_head_err", "furn_h_err", "furn_chamfer", "express", "ids_fail",
                "shelf_bottom_err", "lifted", "screen_false", "screen_size_err", "ceil_false", "ceil_bottom_err",
                "dxf_audit_err", "dxf_dup", "dxf_open", "wall_chamfer"}


def _fmt(value: Any, spec: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if spec == "s":
        return str(value)
    if spec == "d":
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value)
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def load_cases(out_dir: Path) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(Path(out_dir).glob("*/metrics.json")):
        try:
            cases.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return cases


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({k for c in cases for k in (c.get("summary") or {}).keys()})
    agg: dict[str, Any] = {}
    for key in keys:
        values = [c["summary"].get(key) for c in cases if c.get("summary")]
        nums = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None and math.isfinite(float(v))]
        bools = [v for v in values if isinstance(v, bool)]
        if bools and not nums:
            agg[key] = f"{sum(bools)}/{len(bools)}"
        elif nums:
            agg[key] = {"mean": float(np.mean(nums)), "min": float(np.min(nums)), "max": float(np.max(nums)), "n": len(nums)}
    # micro-averaged detection numbers
    micro: dict[str, Any] = {}
    for kind in ("door", "window", "passage"):
        gt = sum(c.get("openings", {}).get(kind, {}).get("gt", 0) for c in cases)
        pred = sum(c.get("openings", {}).get(kind, {}).get("pred", 0) for c in cases)
        tp = sum(c.get("openings", {}).get(kind, {}).get("tp", 0) for c in cases)
        micro[kind] = {"gt": gt, "pred": pred, "tp": tp, "precision": tp / pred if pred else None, "recall": tp / gt if gt else None}
    fg = sum(c.get("furniture", {}).get("gt_count", 0) for c in cases)
    fp = sum(c.get("furniture", {}).get("pred_count", 0) for c in cases)
    fm = sum(c.get("furniture", {}).get("matched", 0) for c in cases)
    micro["furniture"] = {"gt": fg, "pred": fp, "tp": fm, "precision": fm / fp if fp else None, "recall": fm / fg if fg else None}
    wg = sum(c.get("walls", {}).get("gt_count", 0) for c in cases)
    wm = sum(round(c.get("walls", {}).get("recall", 0) * c.get("walls", {}).get("gt_count", 0)) for c in cases if c.get("walls", {}).get("recall") is not None)
    micro["walls"] = {"gt": wg, "detected": wm, "recall": wm / wg if wg else None}
    rg = sum(c.get("rooms", {}).get("gt_count", 0) for c in cases)
    rc = sum(1 for c in cases if c.get("rooms", {}).get("count_ok"))
    micro["rooms"] = {"gt": rg, "scenes_with_correct_count": rc, "scenes": len(cases)}
    for key in ("columns", "beams", "stairs"):
        gt = sum(c.get("structure", {}).get(key, {}).get("gt", 0) for c in cases)
        tp = sum(c.get("structure", {}).get(key, {}).get("plan_tp", 0) for c in cases)
        tpi = sum(c.get("structure", {}).get(key, {}).get("ifc_tp", 0) for c in cases)
        micro[key] = {"gt": gt, "plan_tp": tp, "ifc_tp": tpi}
    variants: dict[str, list[float]] = {}
    for c in cases:
        for variant, iou in (c.get("furniture", {}).get("iou_by_variant") or {}).items():
            variants.setdefault(variant, []).append(iou)
    chamfer: dict[str, list[float]] = {}
    for c in cases:
        for variant, value in (c.get("furniture_chamfer", {}).get("chamfer_by_variant") or {}).items():
            if value is not None:
                chamfer.setdefault(variant, []).append(value)
    return {"metrics": agg, "micro": micro, "furniture_iou_by_variant": {k: float(np.mean(v)) for k, v in sorted(variants.items())},
            "furniture_chamfer_by_variant": {k: float(np.mean(v)) for k, v in sorted(chamfer.items())}}


def failures(cases: list[dict[str, Any]]) -> list[str]:
    lines = []
    for c in cases:
        name = c.get("name")
        pipe = c.get("pipeline", {})
        if pipe.get("status") != "completed":
            lines.append(f"{name}: pipeline status {pipe.get('status')} - {pipe.get('reason')}")
        for error in c.get("errors", []):
            lines.append(f"{name}: {error}")
        for line in (c.get("run", {}) or {}).get("log_errors", [])[:6]:
            if "[ERROR]" in line or "Error" in line:
                lines.append(f"{name}: log: {line[:220]}")
        w = c.get("walls", {})
        if w.get("missed"):
            lines.append(f"{name}: walls missed {w['missed']}")
        if w.get("false_walls"):
            lines.append(f"{name}: false walls {w['false_walls']}")
        o = c.get("openings", {})
        if o.get("missed"):
            lines.append(f"{name}: openings missed {o['missed']}")
        if o.get("false"):
            lines.append(f"{name}: false openings {o['false']}")
        if o.get("confusions"):
            lines.append(f"{name}: opening kind confusions {o['confusions']}")
        f = c.get("furniture", {})
        if f.get("missed"):
            lines.append(f"{name}: furniture missed {f['missed']}")
        if f.get("merged"):
            lines.append(f"{name}: furniture merged {f['merged']}")
        r = c.get("rooms", {})
        if r and not r.get("count_ok"):
            lines.append(f"{name}: rooms {r.get('pred_count')} predicted vs {r.get('gt_count')} GT")
    return lines


def markdown(cases: list[dict[str, Any]], agg: dict[str, Any], meta: dict[str, Any] | None) -> str:
    out = []
    meta = meta or {}
    out.append(f"# Synthetic indoor bench - set `{meta.get('set', '?')}`\n")
    out.append(f"- run: {meta.get('started_at', '?')} -> {meta.get('finished_at', '?')}, total {meta.get('total_seconds', '?')} s, jobs {meta.get('jobs', '?')}")
    out.append(f"- working tree at run time: `git diff --shortstat`: {meta.get('git_diff_shortstat', '?')}")
    out.append(f"- cases: {', '.join(meta.get('cases', [c.get('name', '?') for c in cases]))}\n")
    out.append("## Per scene\n")
    header = "| scene | " + " | ".join(label for _k, label, _f in COLUMNS) + " |"
    out.append(header)
    out.append("|" + "---|" * (len(COLUMNS) + 1))
    for c in cases:
        s = c.get("summary") or {}
        out.append(f"| {c.get('name')} | " + " | ".join(_fmt(s.get(k), f) for k, _l, f in COLUMNS) + " |")
    out.append("\n## Aggregated (mean over scenes; min..max)\n")
    out.append("| metric | mean | min | max | n |")
    out.append("|---|---|---|---|---|")
    for key, value in agg["metrics"].items():
        if isinstance(value, dict):
            arrow = " (lower better)" if key in LOWER_BETTER else ""
            out.append(f"| {key}{arrow} | {value['mean']:.3f} | {value['min']:.3f} | {value['max']:.3f} | {value['n']} |")
        else:
            out.append(f"| {key} | {value} | | | |")
    out.append("\n## Micro-averaged detection (all scenes pooled)\n")
    out.append("| element | GT | pred | TP | precision | recall |")
    out.append("|---|---|---|---|---|---|")
    for kind in ("door", "window", "passage", "furniture"):
        m = agg["micro"][kind]
        out.append(f"| {kind} | {m['gt']} | {m['pred']} | {m['tp']} | {_fmt(m['precision'], '.2f')} | {_fmt(m['recall'], '.2f')} |")
    w = agg["micro"]["walls"]
    out.append(f"| walls | {w['gt']} | - | {w['detected']} | - | {_fmt(w['recall'], '.2f')} |")
    for key in ("columns", "beams", "stairs"):
        m = agg["micro"][key]
        out.append(f"| {key} (plan / IFC) | {m['gt']} | - | {m['plan_tp']} / {m['ifc_tp']} | - | - |")
    rm = agg["micro"]["rooms"]
    out.append(f"\nRooms: correct room count in {rm['scenes_with_correct_count']}/{rm['scenes']} scenes ({rm['gt']} GT rooms).\n")
    out.append("## Furniture footprint IoU by variant (0 = missed)\n")
    out.append(", ".join(f"{k} {v:.2f}" for k, v in agg["furniture_iou_by_variant"].items()))
    out.append("\n\n## Furniture Chamfer (IFC vs GT mesh, m) by variant\n")
    out.append(", ".join(f"{k} {v:.3f}" for k, v in agg["furniture_chamfer_by_variant"].items()))
    out.append("\n\n## Failures / deviations per scene\n")
    for line in failures(cases):
        out.append(f"- {line}")
    out.append("\nColumn legend: scale = units_per_metre - 1 (metric scale error of exported *_m values and IFC); face = wall face position error; "
               "gapP / gapIFC = GT wall junctions not covered by plan wall pieces / IFC walls; cov = wall length coverage (door gaps excused); "
               "1:1 = furniture one-to-one separation rate; chamf = mean symmetric Chamfer IFC furniture vs GT mesh; cpl = share of GT surface within 5 cm of IFC.")
    return "\n".join(out) + "\n"


def build_report(out_dir: Path, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if meta is None and (out_dir / "run_meta.json").is_file():
        meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    cases = load_cases(out_dir)
    agg = aggregate(cases)
    report = {"meta": meta or {}, "aggregate": agg, "cases": {c.get("name"): c.get("summary") for c in cases}, "failures": failures(cases)}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=lambda v: None), encoding="utf-8")
    (out_dir / "report.md").write_text(markdown(cases, agg, meta), encoding="utf-8")
    return report
