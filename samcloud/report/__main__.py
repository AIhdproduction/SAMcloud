"""CLI: ``python -m samcloud.report <project_dir> [--lang de|en] [--camera-list] [--no-camera-list] [--no-class-details] [--json-only]``."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path


def _page_count(path: Path) -> int:
    try:
        return len(re.findall(rb"/Type\s*/Page(?![a-zA-Z])", path.read_bytes()))
    except OSError:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m samcloud.report", description="Create the SAMcloud processing protocol (PDF)")
    parser.add_argument("project_dir")
    parser.add_argument("--lang", choices=("de", "en"), default=None, help="report language (default: app language)")
    parser.add_argument("--camera-list", action="store_true", help="add the per-image camera list (off by default)")
    parser.add_argument("--no-camera-list", action="store_true", help="omit the per-image camera list")
    parser.add_argument("--no-class-details", action="store_true", help="omit the per-class detail tables")
    parser.add_argument("--json-only", action="store_true", help="only write reports/protocol_data.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    project = Path(args.project_dir)
    if not project.is_dir():
        print(f"Project folder not found: {project}", file=sys.stderr)
        return 1
    try:
        from samcloud.report.collect import collect_protocol_data, write_protocol_data
        from samcloud.report.generate import generate_protocol
        from samcloud.report.i18n_text import normalize_language

        if args.json_only:
            data = collect_protocol_data(project, language=normalize_language(args.lang))
            print(write_protocol_data(project, data))
            return 0
        seen: set[tuple[str, int]] = set()

        def progress(fraction: float, phase: str) -> None:
            marker = (phase, int(fraction * 10))
            if marker not in seen:
                seen.add(marker)
                print(f"[{int(round(fraction * 100)):3d}%] {phase}", flush=True)

        options = {"camera_list": False if args.no_camera_list else (True if args.camera_list else None), "class_details": False if args.no_class_details else None}
        path = generate_protocol(project, language=args.lang, options=options, progress=progress)
        print(path)
        print(f"pages: {_page_count(path)}")
        return 0
    except Exception as error:  # noqa: BLE001 - CLI reports every failure as exit code 1
        logging.getLogger(__name__).error("protocol failed: %s", error, exc_info=True)
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
