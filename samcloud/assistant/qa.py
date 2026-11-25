"""QA of an assistant proposal against the automatic baseline plan.

The proposal's plan goes through the real export chain (``build_scene`` -> ``write_ifc`` -> ``validate_ifc``: IDS,
EXPRESS, re-import with the scene) and the SAMcloud plan checks (``plan_violations``, openings plan-vs-IFC, room
areas, wall overlaps after joins, openings flush with their wall) plus a check that every opening stays on its host
wall.  Real projects already have findings, so only findings that the baseline does not have count: ``ok`` is True
when there is no new finding.  Findings are normalised to comparable dicts (``check``, ``subject``, ``detail``);
magnitudes are kept in ``detail`` and ignored by the comparison.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from samcloud.bim.plan_model import BuildingPlan

LOGGER = logging.getLogger(__name__)

OPENING_OFF_WALL_M = 0.02  # an opening centre may sit this far outside its host wall band
_FIXED_TIMESTAMP = datetime(2000, 1, 1, tzinfo=timezone.utc)
_BASELINE_CACHE: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
_BASELINE_CACHE_SIZE = 4
_CACHE_LOCK = threading.Lock()


@dataclass
class QaResult:
    ok: bool
    new_findings: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    ifc_path: Path | None
    baseline_findings: int = 0
    summary: str = ""
    runtime_s: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "new_findings": self.new_findings, "findings": self.findings,
                "ifc_path": None if self.ifc_path is None else str(self.ifc_path), "baseline_findings": self.baseline_findings,
                "summary": self.summary, "runtime_s": self.runtime_s, "counts": dict(self.counts)}


def _plan_hash(plan: BuildingPlan) -> str:
    return hashlib.sha256(json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _strip_numbers(text: str) -> str:
    return re.sub(r"#\d+", "#", re.sub(r"-?\d+(\.\d+)?(e-?\d+)?", "N", str(text)))


def _finding(check: str, subject: str, detail: Any = None) -> dict[str, Any]:
    return {"check": check, "subject": subject, "detail": detail}


def finding_key(finding: dict[str, Any]) -> str:
    return f"{finding.get('check')}|{finding.get('subject')}"


def _opening_wall_findings(plan: BuildingPlan) -> list[dict[str, Any]]:
    from samcloud.assistant.edits import find_element, opening_overhang_m, wall_band
    from shapely.geometry import Point

    upm = plan.units_per_metre or 1.0
    findings = []
    for opening in plan.openings:
        if not opening.host_wall_id:
            continue
        host = find_element(plan, opening.host_wall_id)
        if host is None or host[0] != "wall":
            findings.append(_finding("opening_host_missing", opening.id, {"host": opening.host_wall_id}))
            continue
        overhang = opening_overhang_m(plan, opening) or 0.0
        if overhang > OPENING_OFF_WALL_M:
            findings.append(_finding("opening_outside_wall", opening.id, {"overhang_m": round(overhang, 3)}))
        distance = wall_band(host[1], upm).distance(Point(opening.center[0], opening.center[1])) / upm
        if distance > OPENING_OFF_WALL_M:
            findings.append(_finding("opening_off_wall", opening.id, {"distance_m": round(distance, 3)}))
    return findings


def plan_findings(plan: BuildingPlan, work_dir: Path | str, name: str = "proposal") -> tuple[list[dict[str, Any]], Path | None, dict[str, Any]]:
    """(normalised findings, IFC path, raw QA payload) of one plan through the export chain into ``work_dir``."""
    from samcloud.bim.core.ifc import write_ifc
    from samcloud.bim.core.qa.validate import validate_ifc
    from samcloud.bim.export import _qa_payload
    from samcloud.bim.scene_builder import build_scene

    started = time.perf_counter()
    directory = Path(work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    findings: list[dict[str, Any]] = []
    try:
        scene = build_scene(plan)
        ifc_path = directory / f"{name}.ifc"
        export = write_ifc(scene, ifc_path, timestamp=_FIXED_TIMESTAMP)
        report = validate_ifc(ifc_path, scene=scene, export_result=export, reports_dir=directory / f"{name}_reports")
        qa = _qa_payload(plan, scene, export, report, started, ifc_path)
    except Exception as error:  # noqa: BLE001 - a plan the export chain cannot handle is itself the finding
        LOGGER.warning("assistant QA: export chain failed (%s)", type(error).__name__)
        findings.append(_finding("export_failed", type(error).__name__, str(error)[:300]))
        return findings + _opening_wall_findings(plan), None, {}
    for failure in qa.get("ids_failure_details") or []:
        findings.append(_finding("ids", f"{failure.get('specification')}|{failure.get('requirement')}|{failure.get('ifc_class')}|{failure.get('tag') or failure.get('name')}",
                                 failure.get("reason")))
    for item in qa.get("express_finding_details") or []:
        klass = re.search(r"Ifc\w+", str(item.get("instance", "")))
        findings.append(_finding("express", f"{item.get('type')}|{item.get('attribute')}|{klass.group(0) if klass else ''}|{_strip_numbers(item.get('message', ''))[:120]}",
                                 item.get("message")))
    for item in qa.get("reimport_mismatches") or []:
        findings.append(_finding("reimport", f"{item.get('check')}|{item.get('element_id')}|{_strip_numbers(item.get('message', ''))[:120]}", item.get("message")))
    for item in qa.get("wall_overlaps") or []:
        findings.append(_finding("wall_overlap", "|".join(sorted([str(item.get("a")), str(item.get("b"))])), {"area_m2": item.get("area_m2")}))
    for item in (qa.get("openings_consistency") or {}).get("errors") or []:
        findings.append(_finding("opening_consistency", f"{item.get('kind')}|{item.get('id')}", item.get("reason")))
    for item in (qa.get("door_invariants") or {}).get("violations") or []:
        findings.append(_finding("door_invariant", f"{item.get('rule')}|{item.get('opening')}", item.get("reason")))
    flush = qa.get("opening_flush") or {}
    for item in flush.get("steps") or []:
        findings.append(_finding("opening_flush", f"{item.get('kind')}|{item.get('opening')}|{item.get('where')}", {"step_m": item.get("step_m")}))
    for item in flush.get("blocked_spans") or []:
        findings.append(_finding("opening_span", f"{item.get('kind')}|{item.get('opening')}|{item.get('wall')}", {"overlap_m": item.get("overlap_m")}))
    for item in (qa.get("room_area_check") or {}).get("errors") or []:
        findings.append(_finding("room_area", f"{item.get('room')}", {"plan_m2": item.get("plan_m2"), "ifc_m2": item.get("ifc_m2")}))
    findings.extend(_opening_wall_findings(plan))
    return findings, ifc_path, qa


def _baseline(baseline: BuildingPlan, work_dir: Path) -> list[dict[str, Any]]:
    key = _plan_hash(baseline)
    with _CACHE_LOCK:
        cached = _BASELINE_CACHE.get(key)
        if cached is not None:
            _BASELINE_CACHE.move_to_end(key)
            return cached
    findings, _path, _qa = plan_findings(baseline, work_dir, "baseline")
    with _CACHE_LOCK:
        _BASELINE_CACHE[key] = findings
        while len(_BASELINE_CACHE) > _BASELINE_CACHE_SIZE:
            _BASELINE_CACHE.popitem(last=False)
    return findings


def run_qa(plan: BuildingPlan, *, baseline: BuildingPlan | None, work_dir: Path) -> QaResult:
    """QA of ``plan``; only findings that ``baseline`` does not have count (``ok`` = no new finding)."""
    started = time.perf_counter()
    work_dir = Path(work_dir)
    findings, ifc_path, qa = plan_findings(plan, work_dir, "proposal")
    base_findings = _baseline(baseline, work_dir) if baseline is not None else []
    available = Counter(finding_key(item) for item in base_findings)
    new: list[dict[str, Any]] = []
    for item in findings:
        key = finding_key(item)
        if available[key] > 0:
            available[key] -= 1
        else:
            new.append(item)
    summary = qa.get("summary", "") if qa else "export chain failed"
    return QaResult(not new, new, findings, ifc_path, len(base_findings), summary, round(time.perf_counter() - started, 3),
                    dict(qa.get("counts") or {}) if qa else {})
