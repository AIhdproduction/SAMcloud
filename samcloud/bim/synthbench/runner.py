"""Run bench cases end to end: generate -> real pipeline (``python -m samcloud.engine.auto_dxf``) -> evaluate."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .generate import generate_case
from .metrics import evaluate_case
from .scenes import make_scene, scene_list

REPO_ROOT = Path(__file__).resolve().parents[3]


def git_diff_stat() -> str:
    try:
        out = subprocess.run(["git", "diff", "--shortstat"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
        return out.stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f"unavailable ({error})"


def pipeline_errors(project: Path, limit: int = 40) -> list[str]:
    """ERROR / WARNING lines and tracebacks of the DXF job log (products/dxf/job.log)."""
    log = project / "products" / "dxf" / "job.log"
    if not log.is_file():
        return []
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    picked: list[str] = []
    in_tb = False
    for line in lines:
        if "Traceback (most recent call last)" in line:
            in_tb = True
        if in_tb or "[ERROR]" in line or "[WARNING]" in line:
            picked.append(line.rstrip())
        if in_tb and re.match(r"^\w+(Error|Exception)\b", line):
            in_tb = False
    return picked[:6] + picked[-(limit - 6):] if len(picked) > limit else picked


def run_pipeline(project: Path, log_path: Path, timeout_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    command = [sys.executable, "-m", "samcloud.engine.auto_dxf", "--project", str(project)]
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout_s)
        code = completed.returncode
        status = "ok" if code == 0 else "failed"
    except subprocess.TimeoutExpired:
        code, status = -1, "timeout"
    return {"returncode": code, "status": status, "seconds": round(time.perf_counter() - started, 2), "log_errors": pipeline_errors(project)}


def run_case(family: str, seed: int, out_dir: Path, *, timeout_s: float = 1200, regenerate: bool = False, skip_pipeline: bool = False) -> dict[str, Any]:
    spec = make_scene(family, seed)
    case_dir = Path(out_dir) / spec.name
    project = case_dir / spec.name
    record: dict[str, Any] = {"name": spec.name, "family": family, "seed": seed}
    try:
        if regenerate or not (case_dir / "generate.json").is_file():
            record["generate"] = generate_case(family, seed, case_dir, spec)
        else:
            record["generate"] = json.loads((case_dir / "generate.json").read_text(encoding="utf-8"))
        if not skip_pipeline:
            record["run"] = run_pipeline(project, case_dir / "auto_dxf.log", timeout_s)
        started = time.perf_counter()
        metrics = evaluate_case(case_dir, project)
        metrics["run"] = record.get("run", {})
        metrics["generate"] = {k: v for k, v in record["generate"].items() if k not in ("project",)}
        metrics["evaluate_seconds"] = round(time.perf_counter() - started, 2)
        (case_dir / "metrics.json").write_text(json.dumps(_clean(metrics), indent=1), encoding="utf-8")
        record["metrics"] = metrics
    except Exception as error:  # noqa: BLE001 - one broken case must not stop the set
        import traceback

        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
        (case_dir).mkdir(parents=True, exist_ok=True)
        (case_dir / "bench_error.txt").write_text(record["traceback"], encoding="utf-8")
    return record


def run_set(set_name: str, out_dir: Path, *, jobs: int = 1, timeout_s: float = 1200, regenerate: bool = False, skip_pipeline: bool = False) -> dict[str, Any]:
    from .report import build_report

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = scene_list(set_name)
    started = time.perf_counter()
    meta = {"set": set_name, "started_at": datetime.now(timezone.utc).isoformat(), "git_diff_shortstat": git_diff_stat(), "cases": [f"{f}:{s}" for f, s in cases],
            "jobs": jobs}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        records = list(pool.map(lambda c: run_case(c[0], c[1], out_dir, timeout_s=timeout_s, regenerate=regenerate, skip_pipeline=skip_pipeline), cases))
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["total_seconds"] = round(time.perf_counter() - started, 1)
    meta["bench_errors"] = {r["name"]: r["error"] for r in records if "error" in r}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return build_report(out_dir, meta)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "tolist"):
        return _clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
