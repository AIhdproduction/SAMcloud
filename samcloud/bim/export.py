"""IFC4 export of one level: plan_model.json -> Scene -> IFC -> validation -> qa.json / qa.html.

Output folder: ``<project>/products/bim/<level_key>/``

* ``<project_name>_<level_key>.ifc``
* ``qa.json`` / ``qa.html`` (the validation chain of ``samcloud.bim.core.qa`` plus SAMcloud checks:
  wall overlap after joins, ceiling-vs-wall overlap, consistency with plan_model.json)
* ``<stem>.validation.json|html`` and ``<stem>.ids.html`` (the raw validation reports)
"""

from __future__ import annotations

import html
import json
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from shapely.geometry import Polygon

from samcloud.bim.core.geometry import compute_scene_geometry
from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.model import ElementKind
from samcloud.bim.core.qa.validate import validate_ifc
from samcloud.bim.plan_model import BuildingPlan, load_plan_model
from samcloud.bim.scene_builder import build_scene

BIM_PRODUCT_DIRECTORY = ("products", "bim")


@dataclass
class BimExportResult:
    ifc_path: Path
    qa_json_path: Path
    qa_html_path: Path
    counts: dict[str, int]
    ids_failures: int
    express_findings: int
    warnings: list[str] = field(default_factory=list)
    openings_consistency: dict[str, Any] = field(default_factory=dict)
    openings_consistency_errors: list[dict[str, Any]] = field(default_factory=list)


def bim_output_dir(project_root: Path | str, level_key: str) -> Path:
    return Path(project_root).joinpath(*BIM_PRODUCT_DIRECTORY, str(level_key))


def plan_model_file(project_root: Path | str, level_key: str) -> Path:
    """Location of the ortho ``plan_model.json`` of a level (``clean_plan_cache_dir``, read only)."""
    from samcloud.ortho.plan import clean_plan_cache_dir

    level = None if level_key == "site" else SimpleNamespace(id=str(level_key))
    return clean_plan_cache_dir(project_root, level) / "plan_model.json"  # type: ignore[arg-type]


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "project"


def export_level(
    project_root: Path,
    level_key: str,
    *,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], None] | None = None,
) -> BimExportResult:
    """Export one level to IFC4 and validate it.  ``cancel`` raises to abort between stages."""
    started = time.perf_counter()

    def stage(fraction: float, text: str) -> None:
        if cancel is not None:
            cancel()
        if progress is not None:
            progress(fraction, text)

    project_root = Path(project_root)
    stage(0.0, "load plan model")
    source = plan_model_file(project_root, level_key)
    if not source.is_file():
        raise FileNotFoundError(f"plan_model.json of level {level_key!r} not found: {source}")
    plan = load_plan_model(source)

    stage(0.10, "build scene")
    scene = build_scene(plan)

    stage(0.30, "write IFC")
    out_dir = bim_output_dir(project_root, level_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    ifc_path = out_dir / f"{_safe_name(plan.project_name)}_{_safe_name(str(level_key))}.ifc"
    export = write_ifc(scene, ifc_path)

    stage(0.55, "validate IFC")
    report = validate_ifc(ifc_path, scene=scene, export_result=export, reports_dir=out_dir)

    stage(0.85, "write QA report")
    qa = _qa_payload(plan, scene, export, report, started, ifc_path)
    qa_json = out_dir / "qa.json"
    qa_html = out_dir / "qa.html"
    qa_json.write_text(json.dumps(qa, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    qa_html.write_text(_qa_html(qa), encoding="utf-8")
    stage(1.0, "done")
    return BimExportResult(
        ifc_path=ifc_path,
        qa_json_path=qa_json,
        qa_html_path=qa_html,
        counts=dict(export.counts),
        ids_failures=len(report.ids_failures),
        express_findings=len(report.express_findings),
        warnings=list(qa["warnings"]),
        openings_consistency=dict(qa["openings_consistency"]),
        openings_consistency_errors=list(qa["openings_consistency"]["errors"]),
    )


# --------------------------------------------------------------------------------------
# QA
# --------------------------------------------------------------------------------------

ROOM_AREA_TOLERANCE = 0.03  # relative; door reveals and wall joins make the IFC space a little smaller than the plan room
ROOM_AREA_ALLOWANCE_M2 = 0.25  # absolute allowance per room (a small bath loses relatively more to its door reveals)


def check_room_areas(build: dict[str, Any]) -> dict[str, Any]:
    """Per room and in total: the IfcSpace area must match the plan room within 3 % (+ 0.25 m2 per room).

    A room that the wall clipping split (only its largest piece is written), shrank or dropped is listed; ``ok`` is False then.
    """
    rooms = list(build.get("room_areas") or [])
    errors: list[dict[str, Any]] = []
    for item in rooms:
        plan_area, ifc_area = float(item["plan_m2"]), float(item["ifc_m2"])
        if abs(ifc_area - plan_area) > max(ROOM_AREA_TOLERANCE * plan_area, ROOM_AREA_ALLOWANCE_M2):
            errors.append({"room": item["id"], "name": item["name"], "plan_m2": round(plan_area, 3), "ifc_m2": round(ifc_area, 3), "reason": "IfcSpace area differs from the plan room"})
    plan_total = sum(float(item["plan_m2"]) for item in rooms)
    ifc_total = sum(float(item["ifc_m2"]) for item in rooms)
    if plan_total > 0.0 and abs(ifc_total - plan_total) > max(ROOM_AREA_TOLERANCE * plan_total, ROOM_AREA_ALLOWANCE_M2):
        errors.append({"room": "total", "name": "all rooms", "plan_m2": round(plan_total, 3), "ifc_m2": round(ifc_total, 3), "reason": "total IfcSpace area differs from the plan"})
    return {"ok": not errors, "errors": errors, "tolerance": ROOM_AREA_TOLERANCE, "rooms": [{"id": i["id"], "name": i["name"], "plan_m2": round(float(i["plan_m2"]), 3), "ifc_m2": round(float(i["ifc_m2"]), 3)} for i in rooms]}


def _consistency(plan: BuildingPlan, scene: Any, counts: dict[str, int]) -> dict[str, Any]:
    build = scene.derived.get("samcloud_build", {})
    openings_with_host = sum(1 for o in plan.openings if o.host_wall_id)
    kinds = Counter(e.kind for e in scene.elements.values())
    doors = counts.get("IfcDoor", 0)
    windows = counts.get("IfcWindow", 0)
    return {
        "walls": {"plan_pieces": len(plan.walls), "ifc_walls": counts.get("IfcWall", 0), "merged_duplicates": build.get("duplicates_merged", 0), "chained": build.get("chained", 0), "bridged": build.get("bridged", 0)},
        "spaces": {"plan_rooms": len(plan.rooms), "ifc_spaces": counts.get("IfcSpace", 0)},
        "doors": {"plan_with_host": sum(1 for o in plan.openings if o.kind == "door" and o.host_wall_id), "plan_total": sum(1 for o in plan.openings if o.kind == "door"), "ifc": doors},
        "windows": {"plan_with_host": sum(1 for o in plan.openings if o.kind == "window" and o.host_wall_id), "plan_total": sum(1 for o in plan.openings if o.kind == "window"), "ifc": windows},
        "openings": {"plan_with_host": openings_with_host, "plan_total": len(plan.openings), "ifc_opening_elements": counts.get("IfcOpeningElement", 0)},
        "furniture": {"plan": len(plan.furniture), "ifc": kinds.get(ElementKind.FURNITURE, 0) and build.get("furniture_out", 0)},
        "room_area_m2": {"plan": round(build.get("room_area_plan_m2", 0.0), 3), "ifc": round(build.get("room_area_scene_m2", 0.0), 3)},
    }


_OPENING_KINDS = ("door", "window", "passage")


def ifc_opening_ids(ifc_path: Path | str) -> dict[str, str]:
    """Plan opening ids found in an IFC file, mapped to their kind (``door`` / ``window`` / ``passage``).

    Doors and windows are named ``Door <id>`` / ``Window <id>``; a void without filling is a passage
    (``Passage opening <id>``).
    """
    import ifcopenshell

    model = ifcopenshell.open(str(ifc_path))
    found: dict[str, str] = {}
    for ifc_class, kind in (("IfcDoor", "door"), ("IfcWindow", "window")):
        for element in model.by_type(ifc_class):
            if element.Name:
                found[str(element.Name).split()[-1]] = kind
    for element in model.by_type("IfcOpeningElement"):
        if not element.Name or element.HasFillings:
            continue
        found.setdefault(str(element.Name).split()[-1], "passage")
    return found


def check_openings_consistency(
    plan: BuildingPlan, ifc_path: Path | str, skipped: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Compare doors / windows / passages of the plan model with the written IFC.

    Every plan opening that is missing in the IFC (or has another kind) is an error with its id and the
    reason (``skipped`` carries the scene builder's reasons).
    """
    reasons = {item["id"]: item.get("reason", "") for item in (skipped or [])}
    in_ifc = ifc_opening_ids(ifc_path)
    plan_counts = {kind: sum(1 for o in plan.openings if o.kind == kind) for kind in _OPENING_KINDS}
    ifc_counts = {kind: sum(1 for k in in_ifc.values() if k == kind) for kind in _OPENING_KINDS}
    errors: list[dict[str, Any]] = []
    plan_ids = {o.id: o for o in plan.openings}
    for opening in sorted(plan.openings, key=lambda o: o.id):
        found = in_ifc.get(opening.id)
        if found is None:
            errors.append({"id": opening.id, "kind": opening.kind, "reason": reasons.get(opening.id) or "missing in IFC"})
        elif found != opening.kind:
            errors.append({"id": opening.id, "kind": opening.kind, "reason": f"exported as {found}"})
    for ident, kind in sorted(in_ifc.items()):
        if ident not in plan_ids:
            errors.append({"id": ident, "kind": kind, "reason": "in IFC but not in plan model"})
    return {"plan": plan_counts, "ifc": ifc_counts, "ok": not errors, "errors": errors}


def _qa_payload(
    plan: BuildingPlan, scene: Any, export: Any, report: Any, started: float, ifc_path: Path | str | None = None
) -> dict[str, Any]:
    geometry = compute_scene_geometry(scene)
    wall_union = [Polygon(g.footprint) for g in geometry.walls.values()]
    ceiling_overlap = 0.0
    for ceiling in geometry.ceilings.values():
        shell = Polygon(ceiling.outline)
        for wall in wall_union:
            if shell.intersects(wall):
                ceiling_overlap = max(ceiling_overlap, float(shell.intersection(wall).area))
    # overlap of one ceiling with the union of walls (several small overlaps add up)
    union_overlap = 0.0
    if wall_union:
        from shapely.ops import unary_union

        merged = unary_union(wall_union)
        for ceiling in geometry.ceilings.values():
            union_overlap = max(union_overlap, float(Polygon(ceiling.outline).intersection(merged).area))
    build = dict(scene.derived.get("samcloud_build", {}))
    counts = dict(export.counts)
    warnings = list(dict.fromkeys(list(export.warnings) + list(build.get("warnings", []))))
    consistency = _consistency(plan, scene, counts)
    consistency["furniture"]["ifc"] = counts.get("IfcFurniture", 0) + sum(
        counts.get(name, 0)
        for name in ("IfcElectricAppliance", "IfcSanitaryTerminal", "IfcAudioVisualAppliance", "IfcLightFixture", "IfcSpaceHeater", "IfcBuildingElementProxy")
    )
    opening_check = check_openings_consistency(plan, ifc_path or export.path, build.get("skipped_openings", []))
    for error in opening_check["errors"]:
        warnings.append(f"openings_consistency: {error['kind']} {error['id']}: {error['reason']}")
    area_check = check_room_areas(build)
    for error in area_check["errors"]:
        warnings.append(f"room_areas: {error['room']} {error['name']}: plan {error['plan_m2']} m2 vs IFC {error['ifc_m2']} m2")
    ids_n, express_n = len(report.ids_failures), len(report.express_findings)
    reimport_ok = report.reimport_checked and not report.reimport_mismatches
    overlaps = build.get("wall_overlaps", [])
    summary = (
        f"IFC4 {'OK' if report.ok and not overlaps and area_check['ok'] else 'with findings'}: "
        f"{counts.get('IfcWall', 0)} walls, {counts.get('IfcSpace', 0)} spaces, "
        f"{counts.get('IfcDoor', 0)} doors, {counts.get('IfcWindow', 0)} windows, "
        f"{consistency['furniture']['ifc']} furniture items; "
        f"IDS failures {ids_n}, EXPRESS findings {express_n}, re-import {'ok' if reimport_ok else 'MISMATCH'}, "
        f"wall overlaps {len(overlaps)}, ceiling-wall overlap {union_overlap:.6f} m2, "
        f"openings plan-vs-IFC {'consistent' if opening_check['ok'] else str(len(opening_check['errors'])) + ' ERRORS'}, "
        f"room areas plan-vs-IFC {'consistent' if area_check['ok'] else str(len(area_check['errors'])) + ' MISMATCHES'}."
    )
    return {
        "ids_failures": ids_n,
        "express_findings": express_n,
        "reimport_ok": bool(reimport_ok),
        "elements": sum(counts.values()),
        "counts": counts,
        "warnings": warnings,
        "summary": summary,
        "unreferenced": bool(report.unreferenced),
        "wall_overlaps": overlaps,
        "wall_overlap_count": len(overlaps),
        "ceiling_wall_overlap_max_m2": union_overlap,
        "ceiling_single_wall_overlap_max_m2": ceiling_overlap,
        "build": {k: v for k, v in build.items() if k not in ("warnings", "wall_overlaps", "skipped_openings")},
        "consistency": consistency,
        "openings_consistency": opening_check,
        "room_area_check": area_check,
        "openings_consistency_errors": opening_check["errors"],
        "ids_failure_details": [failure.to_dict() for failure in report.ids_failures[:100]],
        "express_finding_details": list(report.express_findings[:100]),
        "reimport_mismatches": [item.to_dict() for item in report.reimport_mismatches[:100]],
        "ifc_file": Path(export.path).name,
        "runtime_s": round(time.perf_counter() - started, 3),
    }


def _qa_html(qa: dict[str, Any]) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    def table(rows: list[tuple[str, object]]) -> str:
        return "<table>" + "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows) + "</table>"

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>IFC QA</title>",
        "<style>body{font-family:sans-serif;margin:24px;max-width:960px}th{text-align:left;padding-right:16px}"
        "td,th{padding:2px 8px;border-bottom:1px solid #ddd}.bad{color:#b00020}.ok{color:#1b7f3b}</style></head><body>",
        f"<h1>IFC QA - {esc(qa['ifc_file'])}</h1>",
        f"<p class='{'ok' if qa['ids_failures'] == 0 and qa['express_findings'] == 0 and qa['reimport_ok'] else 'bad'}'>{esc(qa['summary'])}</p>",
        "<h2>Checks</h2>",
        table([
            ("IDS failures", qa["ids_failures"]),
            ("EXPRESS findings", qa["express_findings"]),
            ("Re-import", "ok" if qa["reimport_ok"] else "mismatch"),
            ("Wall overlaps > 5 % (after joins)", qa["wall_overlap_count"]),
            ("Largest ceiling/wall overlap (m2)", f"{qa['ceiling_wall_overlap_max_m2']:.6f}"),
            ("Openings plan vs IFC", "consistent" if qa["openings_consistency"]["ok"] else f"{len(qa['openings_consistency']['errors'])} errors"),
            ("Georeferenced", "no" if qa["unreferenced"] else "yes"),
            ("Runtime (s)", qa["runtime_s"]),
        ]),
        "<h2>Element counts (IFC roots)</h2>",
        table(sorted(qa["counts"].items())),
        "<h2>Consistency with plan_model.json</h2>",
        table([(k, json.dumps(v)) for k, v in qa["consistency"].items()]),
        "<h2>Openings consistency (plan model vs IFC)</h2>",
        table([
            ("doors plan / IFC", f"{qa['openings_consistency']['plan']['door']} / {qa['openings_consistency']['ifc']['door']}"),
            ("windows plan / IFC", f"{qa['openings_consistency']['plan']['window']} / {qa['openings_consistency']['ifc']['window']}"),
            ("passages plan / IFC", f"{qa['openings_consistency']['plan']['passage']} / {qa['openings_consistency']['ifc']['passage']}"),
        ]),
        "<h2>Scene build</h2>",
        table(sorted(qa["build"].items())),
    ]
    for title, key in (("Opening consistency errors", "openings_consistency_errors"), ("Wall overlaps", "wall_overlaps"), ("IDS failures", "ids_failure_details"), ("EXPRESS findings", "express_finding_details"), ("Re-import mismatches", "reimport_mismatches")):
        if qa[key]:
            parts.append(f"<h2>{title}</h2><ul>" + "".join(f"<li>{esc(json.dumps(item, ensure_ascii=False))}</li>" for item in qa[key]) + "</ul>")
    parts.append(f"<h2>Warnings ({len(qa['warnings'])})</h2><ul>" + "".join(f"<li>{esc(w)}</li>" for w in qa["warnings"]) + "</ul>")
    parts.append("</body></html>")
    return "".join(parts)
