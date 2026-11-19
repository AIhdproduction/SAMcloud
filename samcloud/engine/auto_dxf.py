"""Automatic final stage "DXF/IFC export" (workflow step 8) after a successful processing run.

The stage runs the same background jobs the DXF/IFC page starts (``samcloud.dxf.jobs``): indoor projects detect
their storeys first when none are stored, then ``run_all`` runs once per storey; outdoor projects run ``run_all``
once for the site. Every problem is recorded (``reports/auto_dxf.json``, a WARNING in ``reports/run_summary.txt``,
``reports/dxf_run_metrics.json`` written by the jobs) and never marks the processing run failed.

Time accounting: the pipeline finalises ``run_metrics*.json`` before this stage starts, so its ``total_seconds``
never contains the DXF/IFC time; the home page adds ``dxf_run_metrics.json`` on top.

Standalone, for a finished project::

    python -m samcloud.engine.auto_dxf --project <project directory>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGE_TEXT = "DXF/IFC export"
"""English progress text of the stage (translated for display through ``workflow.stage.dxf_ifc``)."""

WARNING_MARKER = "[auto DXF/IFC]"
REPORT_NAME = "auto_dxf.json"


def auto_dxf_enabled(settings: dict[str, Any] | None) -> bool:
    """Project setting ``workflow.auto_dxf``; a missing key (older projects) means ON."""
    workflow = (settings or {}).get("workflow", {}) if isinstance(settings, dict) else {}
    value = workflow.get("auto_dxf", True) if isinstance(workflow, dict) else True
    return bool(value)


def pipeline_run_completed(project_dir: Path, not_before: float | None = None) -> bool:
    """True when reports/run_metrics.json says "Completed" (and, with ``not_before``, was written by this run)."""
    try:
        data = json.loads((Path(project_dir) / "reports" / "run_metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or data.get("status") != "Completed":
        return False
    if not_before is None:
        return True
    try:
        started = datetime.fromisoformat(str(data.get("started_at"))).timestamp()
    except ValueError:
        return False
    return started >= not_before - 1.0


def select_levels(controller: Any) -> list[str | None]:
    """Run targets: the checked storeys for indoor projects (as the review page offers them), ``[None]`` outdoors."""
    if controller.mode == "indoor":
        return [str(level["id"]) for level in controller.checked_levels()]
    return [None]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_progress(progress_file: Path | None, stage: str, detail: str = "") -> None:
    """Update progress.json in place: steps 1-6 stay complete (100 %), only stage text and detail change."""
    if progress_file is None:
        return
    payload = _read_json(progress_file)
    payload.update({"percent": 100.0, "stage": stage, "detail": detail, "stage_key": None, "updated_at": time.time()})
    payload.setdefault("started_at", time.time())
    try:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def _add_warning(project_dir: Path, message: str) -> None:
    """Replace the automatic-stage warning in run_summary.txt (and the run's timestamped copy)."""
    reports = Path(project_dir) / "reports"
    run_id = _read_json(reports / "run_metrics.json").get("run_id")
    names = ["run_summary.txt"] + ([f"run_summary_{run_id}.txt"] if run_id else [])
    for name in names:
        path = reports / name
        if not path.is_file():
            continue
        try:
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if WARNING_MARKER not in line]
            while lines and not lines[-1].strip():
                lines.pop()
            if message:
                lines += ["", f"WARNING {WARNING_MARKER} {message}"]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass


def _combine_metrics(project_dir: Path, runs: list[dict[str, Any]], levels: list[str | None]) -> None:
    """Several storeys overwrite dxf_run_metrics.json one after another; leave one record covering all of them."""
    if len(runs) < 2:
        return
    from samcloud.dxf.jobs import write_dxf_run_metrics

    steps: dict[str, float] = {}
    for run in runs:
        for name, seconds in (run.get("step_seconds") or {}).items():
            steps[name] = round(steps.get(name, 0.0) + float(seconds), 3)
    all_done = all(run.get("status") == "completed" for run in runs) and len(runs) == len(levels)
    write_dxf_run_metrics(project_dir, {
        "action": "run_all",
        "level": None,
        "levels": [run.get("level") for run in runs],
        "status": "completed" if all_done else "failed",
        "started_at": runs[0].get("started_at"),
        "finished_at": runs[-1].get("finished_at"),
        "total_seconds": round(sum(float(run.get("total_seconds") or 0.0) for run in runs), 3),
        "step_seconds": steps,
    })


def _run_job(runner: Any, action: str, las: Path, mode: str, source: str, level: str | None) -> int:
    """Start one DXF job exactly like the page (DxfJobRunner) and wait for it; returns the exit code."""
    options = {"source": source, "variant": "ortho"} if action == "run_all" else {}
    process = runner.start(action, las=las, level=level, mode=mode, force=True, **options)
    return int(process.wait())


def run_auto_dxf(project_dir: Path | str, *, progress_file: Path | str | None = None) -> dict[str, Any]:
    """Run the final DXF/IFC stage for a finished project and return the recorded result (never raises)."""
    project_dir = Path(project_dir).resolve()
    progress_path = Path(progress_file) if progress_file else project_dir / "reports" / "progress.json"
    started = time.monotonic()
    result: dict[str, Any] = {
        "status": "failed", "reason": "", "mode": None, "levels": [], "failed_levels": [],
        "started_at": datetime.now(timezone.utc).isoformat(), "total_seconds": 0.0,
    }
    runs: list[dict[str, Any]] = []
    levels: list[str | None] = []
    try:
        from samcloud.app.dxf_controller import DxfPageController
        from samcloud.dxf.jobs import dxf_run_metrics_path

        controller = DxfPageController(project_dir)
        las = controller.las_path
        result["mode"] = controller.mode
        if las is None or not Path(las).is_file():
            result.update(status="skipped", reason="no classified point cloud found")
        elif controller.blocked_message is not None:
            result.update(status="skipped", reason=controller.blocked_message)
        elif controller.profile_required:
            result.update(status="skipped", reason="the model profile of the point cloud is unknown (choose it on the DXF/IFC page)")
        else:
            mode, source, runner = controller.mode, controller.state.segment_source, controller.runner
            _write_progress(progress_path, STAGE_TEXT, "")
            levels = select_levels(controller)
            if mode == "indoor" and not levels:
                _write_progress(progress_path, STAGE_TEXT, "storeys")
                code = _run_job(runner, "prepare_levels", las, mode, source, None)
                controller.refresh_state()
                levels = select_levels(controller)
                if code != 0 or not levels:
                    result["reason"] = "detecting storeys failed" if code != 0 else "no storeys detected"
            if levels:
                for index, level in enumerate(levels, start=1):
                    label = level if level is not None else "site"
                    _write_progress(progress_path, STAGE_TEXT, f"{label} ({index}/{len(levels)})")
                    code = _run_job(runner, "run_all", las, mode, source, level)
                    runs.append(_read_json(dxf_run_metrics_path(project_dir)))
                    result["levels"].append(label)
                    if code != 0:
                        result["failed_levels"].append(label)
                _combine_metrics(project_dir, runs, levels)
                if result["failed_levels"]:
                    result["reason"] = "DXF/IFC export failed for: " + ", ".join(result["failed_levels"])
                else:
                    result["status"] = "completed"
    except Exception as error:  # the stage must never fail the processing run
        result.update(status="failed", reason=f"{type(error).__name__}: {error}")
    result["total_seconds"] = round(time.monotonic() - started, 3)
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    try:
        (project_dir / "reports").mkdir(parents=True, exist_ok=True)
        (project_dir / "reports" / REPORT_NAME).write_text(json.dumps(result, indent=2), encoding="utf-8")
    except OSError:
        pass
    if result["status"] == "completed":
        _add_warning(project_dir, "")
    else:
        verb = "skipped" if result["status"] == "skipped" else "failed"
        _add_warning(project_dir, f"DXF/IFC export {verb}: {result['reason']}. See products/dxf/job.log.")
    _write_progress(progress_path, "Complete", "")
    return result


def run_after_pipeline(project_dir: Path | str, *, progress_file: Path | str | None = None,
                       not_before: float | None = None) -> dict[str, Any] | None:
    """Pipeline hook: run the stage only when this run finished with status "Completed"."""
    if not pipeline_run_completed(Path(project_dir), not_before):
        return None
    print(f"\n{STAGE_TEXT} (automatic): started", flush=True)
    result = run_auto_dxf(project_dir, progress_file=progress_file)
    print(f"{STAGE_TEXT} (automatic): {result['status']} in {result['total_seconds']:.0f} s"
          + (f" - {result['reason']}" if result["reason"] else ""), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the automatic DXF/IFC export stage for a finished project")
    parser.add_argument("--project", required=True, type=Path, help="project directory")
    args = parser.parse_args(argv)
    result = run_auto_dxf(args.project)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"completed", "skipped"} else 1


if __name__ == "__main__":
    sys.exit(main())
