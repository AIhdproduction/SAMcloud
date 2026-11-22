"""CLI of the synthetic indoor bench.

    python -m samcloud.bim.synthbench run --set dev --out <dir> [--jobs 2] [--timeout 1200] [--regenerate]
    python -m samcloud.bim.synthbench run --set rect_basic:101,dense:107 --out <dir>
    python -m samcloud.bim.synthbench report --out <dir>
    python -m samcloud.bim.synthbench generate --set dev --out <dir>
    python -m samcloud.bim.synthbench evaluate --case <dir>/<scene>
    python -m samcloud.bim.synthbench compare-plan <baseline plan_model.json> <new plan_model.json> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m samcloud.bim.synthbench", description="Synthetic indoor DXF/IFC bench")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="generate, run the real pipeline and score a scene set")
    run.add_argument("--set", default="dev", help="dev | val | objects | objects_val | family:seed[,family:seed...]")
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--timeout", type=float, default=1200.0, help="pipeline timeout per scene in seconds")
    run.add_argument("--regenerate", action="store_true", help="rebuild scenes even when the case folder exists")
    run.add_argument("--skip-pipeline", action="store_true", help="only (re)evaluate existing pipeline outputs")
    rep = sub.add_parser("report", help="aggregate metrics.json files of an output folder")
    rep.add_argument("--out", required=True, type=Path)
    gen = sub.add_parser("generate", help="only generate projects + ground truth")
    gen.add_argument("--set", default="dev")
    gen.add_argument("--out", required=True, type=Path)
    ev = sub.add_parser("evaluate", help="score one case folder")
    ev.add_argument("--case", required=True, type=Path)
    cmp_ = sub.add_parser("compare-plan", help="diff two plan_model.json files")
    cmp_.add_argument("base", type=Path)
    cmp_.add_argument("new", type=Path)
    cmp_.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.command == "run":
        from .runner import run_set

        report = run_set(args.set, args.out, jobs=args.jobs, timeout_s=args.timeout, regenerate=args.regenerate, skip_pipeline=args.skip_pipeline)
        print((Path(args.out) / "report.md").read_text(encoding="utf-8"))
        return 0 if not report["meta"].get("bench_errors") else 1
    if args.command == "report":
        from .report import build_report

        build_report(args.out)
        print((Path(args.out) / "report.md").read_text(encoding="utf-8"))
        return 0
    if args.command == "generate":
        from .generate import generate_case
        from .scenes import make_scene, scene_list

        for family, seed in scene_list(args.set):
            spec = make_scene(family, seed)
            info = generate_case(family, seed, Path(args.out) / spec.name, spec)
            print(spec.name, json.dumps({k: info[k] for k in ("points", "instances", "photos", "generate_seconds", "scene_problems")}))
        return 0
    if args.command == "evaluate":
        from .metrics import evaluate_case
        from .runner import _clean

        result = _clean(evaluate_case(args.case))
        (Path(args.case) / "metrics.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps(result.get("summary", {}), indent=1))
        return 0
    if args.command == "compare-plan":
        from .compare import compare_files, summary_text
        from .runner import _clean

        result = compare_files(args.base, args.new)
        if args.json:
            args.json.write_text(json.dumps(_clean(result), indent=1), encoding="utf-8")
        print(summary_text(result))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
