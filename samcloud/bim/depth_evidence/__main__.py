"""Build compact depth evidence for an existing project."""

from __future__ import annotations

import argparse
import json

from .core import build_depth_product


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SAMcloud compact depth evidence")
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    report = build_depth_product(args.project)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report.get("written") else 1


if __name__ == "__main__":
    raise SystemExit(main())
