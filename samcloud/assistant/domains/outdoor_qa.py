"""QA of an outdoor assistant proposal: the real CAD checks of the drone chain on IFC + DXF of the edited Scene.

The edited Scene goes through the same builders as the chain (``run_environment.build_scene_outputs`` with the
proposal as ``edited_scene`` and ``write_cad_exports``) into a temporary IFC and DXF, then ``qa_cad.check_dxf``,
``check_ifc`` and ``count_consistency`` run on them (the manual marking counted with its category,
:func:`~samcloud.assistant.domains.outdoor_pipeline.adjust_manual_qa`).  Real projects already fail some rules, so the
same QA runs once on the base Scene (cached per base and checkpoint version) and only new findings block: rules that
pass there and fail for the proposal, and rules that already fail but got worse (new offenders, a grown count gap; see
:func:`compare`): ``ok`` is True without a new finding.  It takes about as long as the chain QA (the
IFC check dominates, ~20 s on a real site).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from samcloud.assistant.qa import QaResult

LOGGER = logging.getLogger(__name__)

_BASELINE_CACHE: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
_BASELINE_CACHE_SIZE = 4
_LOCK = threading.Lock()
_CHECKPOINTS = ("vegetation.pkl", "pavement.pkl", "tin.pkl", "buildings.pkl", "site.pkl", "terrain.pkl", "boundary.pkl")


def frame_of(scene: Any) -> Any:
    """The chain's ``LocalFrame`` of a Scene (translation ``settings.origin_offset``, EPSG / datum from ``settings.crs``)."""
    from samcloud.drone.contracts import LocalFrame

    crs = scene.settings.crs
    epsg = int(crs.epsg) if crs is not None and crs.epsg else 2056
    datum = (crs.vertical_datum if crs is not None and crs.vertical_datum else "LN02")
    return LocalFrame(tuple(float(v) for v in scene.settings.origin_offset), epsg, datum)


def scene_counts(scene: Any) -> dict[str, int]:
    """Category counts of a Scene for ``count_consistency`` (as ``_step_qa``)."""
    from samcloud.outdoor_model import buildings, civil, pavement, vegetation

    elements = list(scene.elements.values())
    counts: dict[str, int] = {}
    for getter in (buildings.category_counts, civil.category_counts, pavement.category_counts, vegetation.category_counts):
        counts.update(getter(elements))
    return counts


def cad_qa(project: Path | str, scene: Any, out_dir: Path, *, config: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], Path, Path]:
    """(rule results, IFC path, DXF path) of the CAD QA of ``scene`` written into ``out_dir``."""
    from samcloud.assistant.domains.outdoor_pipeline import adjust_manual_qa
    from samcloud.assistant.domains.outdoor_scene import cache_dir
    from samcloud.drone import run_environment as env
    from samcloud.drone.contracts import load_drone_config
    from samcloud.drone.qa_cad import check_dxf, check_ifc, count_consistency

    cfg = config if config is not None else load_drone_config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = frame_of(scene)
    built, dxf_result, _qa = env.build_scene_outputs(project, cfg, cache_dir(project), edited_scene=scene, frame=frame)
    ifc_path, dxf_path = out_dir / "outdoor.ifc", out_dir / "outdoor.dxf"
    env.write_cad_exports(built, dxf_result, frame, cfg, ifc_path, dxf_path)
    results = check_dxf(dxf_path, cfg) + check_ifc(ifc_path, cfg) + count_consistency(scene_counts(built), dxf_path, ifc_path)
    return adjust_manual_qa(results, dxf_path, ifc_path), ifc_path, dxf_path


def _baseline_key(project: Path | str, base: Any) -> str:
    from samcloud.assistant.domains.outdoor_scene import cache_dir, scene_semantic_hash

    digest = hashlib.sha256(scene_semantic_hash(base).encode("ascii"))
    for name in _CHECKPOINTS:
        try:
            stat = (cache_dir(project) / name).stat()
            digest.update(f"{name}:{stat.st_mtime_ns}:{stat.st_size}".encode("ascii"))
        except OSError:
            digest.update(f"{name}:-".encode("ascii"))
    return digest.hexdigest()


def baseline_results(project: Path | str, base: Any, work_dir: Path) -> list[dict[str, Any]]:
    """CAD QA results of the base Scene (memory and ``work_dir/baseline_<key>.json`` cache)."""
    key = _baseline_key(project, base)
    with _LOCK:
        if key in _BASELINE_CACHE:
            _BASELINE_CACHE.move_to_end(key)
            return _BASELINE_CACHE[key]
    stored = Path(work_dir) / f"baseline_{key[:16]}.json"
    results = None
    if stored.is_file():
        try:
            results = json.loads(stored.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            results = None
    if results is None:
        results = cad_qa(project, base, Path(work_dir) / "baseline")[0]
        try:
            stored.write_text(json.dumps(results, ensure_ascii=False, default=str), encoding="utf-8")
        except OSError as error:
            LOGGER.info("assistant: QA baseline not cached (%s)", type(error).__name__)
    with _LOCK:
        _BASELINE_CACHE[key] = results
        while len(_BASELINE_CACHE) > _BASELINE_CACHE_SIZE:
            _BASELINE_CACHE.popitem(last=False)
    return results


def _finding(result: dict[str, Any], new_offenders: list[str] | None = None, reason: str | None = None) -> dict[str, Any]:
    detail: dict[str, Any] = {"value": result.get("value"), "offenders": list(result.get("offenders") or [])[:5]}
    if new_offenders:
        detail["new_offenders"] = list(new_offenders)[:5]
    if reason:
        detail["reason"] = reason
    return {"check": str(result.get("rule")), "subject": "outdoor", "detail": detail}


def offender_key(text: Any) -> str:
    """An offender text without run-specific parts: DXF handles / IFC entity numbers (``#1A2``, ``#537``) and decimal
    magnitudes (deviations, heights) are dropped, element names / tags and integer counts stay."""
    key = re.sub(r"#[0-9A-Fa-f]+", "#", str(text))
    key = re.sub(r"-?\d+\.\d+(e-?\d+)?", "N", key)
    return re.sub(r"\s+", " ", key).strip()


def _count_gaps(result: dict[str, Any]) -> dict[str, float]:
    """``{source: |count - scene count|}`` of a count rule (``15_count_*``)."""
    value = result.get("value")
    if not isinstance(value, dict) or not isinstance(value.get("scene"), (int, float)):
        return {}
    scene = float(value["scene"])
    return {str(src): abs(float(n) - scene) for src, n in value.items()
            if src != "scene" and isinstance(n, (int, float)) and not isinstance(n, bool)}


def compare(results: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(failing findings of ``results``, the new ones).

    New is a failing rule that passes in ``baseline`` or does not exist there, and also a rule that already fails in the
    baseline but got worse: a count rule (``15_count_*``) whose DXF / IFC gap to the scene count grew, any other rule with
    an offender the baseline does not have (compared by :func:`offender_key`, so element names / tags count, handles do not).
    """
    before = {str(r.get("rule")): r for r in baseline}
    failing = [r for r in results if not r.get("ok")]
    new: list[dict[str, Any]] = []
    for result in failing:
        rule = str(result.get("rule"))
        old = before.get(rule)
        if old is None or old.get("ok"):
            new.append(_finding(result, reason="new failure"))
            continue
        if rule.startswith("15_count_"):
            gaps, old_gaps = _count_gaps(result), _count_gaps(old)
            grown = [f"{src}: {old_gaps.get(src, 0.0):g} -> {gap:g}" for src, gap in gaps.items() if gap > old_gaps.get(src, 0.0) + 1e-9]
            if grown:
                new.append(_finding(result, grown, "count gap grew"))
            continue
        known = {offender_key(text) for text in old.get("offenders") or []}
        added = [str(text) for text in result.get("offenders") or [] if offender_key(text) not in known]
        if added:
            new.append(_finding(result, added, "new offenders"))
    return [_finding(r) for r in failing], new


def run_outdoor_qa(project: Path | str, proposal_scene: Any, base_scene: Any, *, work_dir: Path) -> QaResult:
    """QA of an edited Scene against its base (see the module docstring)."""
    started = time.monotonic()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        baseline = baseline_results(project, base_scene, work_dir)
        results, ifc_path, _dxf = cad_qa(project, proposal_scene, work_dir / "proposal")
    except Exception as error:  # noqa: BLE001 - a broken QA must block the proposal, not crash the turn
        LOGGER.warning("assistant: outdoor QA failed (%s: %s)", type(error).__name__, error)
        finding = {"check": "qa_failed", "subject": "outdoor", "detail": f"{type(error).__name__}: {str(error)[:200]}"}
        return QaResult(False, [finding], [finding], None, 0, "QA could not run", round(time.monotonic() - started, 2))
    findings, new = compare(results, baseline)
    baseline_failing = sum(1 for r in baseline if not r.get("ok"))
    summary = (f"{len(results)} CAD rules, {len(findings)} failing, {len(new)} new" if new
               else f"{len(results)} CAD rules, no new finding ({baseline_failing} known)")
    return QaResult(not new, new, findings, ifc_path, baseline_failing, summary, round(time.monotonic() - started, 2),
                    {"rules": len(results), "failing": len(findings), "new": len(new)})


__all__ = ["baseline_results", "cad_qa", "compare", "frame_of", "run_outdoor_qa", "scene_counts"]
