"""IFC export job logic (export + viewer assets), freshness rules and a thin CLI.

``python -m samcloud.bim.jobs export --project <dir> --level <key>`` runs the same code path as the
``export_ifc`` DXF job but without ``job_status.json``. Exit codes: 0 ok, 1 failed, 2 cancelled.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Literal

from samcloud.dxf.cancel import Cancelled

ProgressFn = Callable[[float, str], None]
IfcExportStatus = Literal["missing", "fresh", "stale"]

EXPORT_MANIFEST = "export.json"
EXPORT_SPAN = 0.85  # share of the job's progress taken by the IFC/QA export; the rest is the viewer export


def _export_level(*args: Any, **kwargs: Any) -> Any:
    """Lazily delegate to samcloud.bim.export.export_level (heavy imports, monkeypatchable)."""
    from samcloud.bim import export as bim_export

    return bim_export.export_level(*args, **kwargs)


def bim_output_dir(project_root: Path | str, level_key: str) -> Path:
    return Path(project_root) / "products" / "bim" / level_key


def plan_model_file(project_root: Path | str, level_key: str) -> Path:
    return Path(project_root) / "products" / "dxf" / level_key / "ortho" / "plan_model.json"


def plan_model_sha256(project_root: Path | str, level_key: str) -> str | None:
    path = plan_model_file(project_root, level_key)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def ifc_export_status(project_root: Path | str, level_key: str) -> IfcExportStatus:
    """'missing' without a manifest or outputs, 'fresh' when it matches plan_model.json, else 'stale'."""
    directory = bim_output_dir(project_root, level_key)
    manifest_path = directory / EXPORT_MANIFEST
    if not manifest_path.is_file():
        return "missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = [directory / str(manifest[key]) for key in ("ifc", "glb", "props")]
    except (OSError, ValueError, KeyError, TypeError):
        return "stale"
    if not all(path.is_file() for path in outputs):
        return "missing"
    if manifest.get("source") == "environment":
        # Outdoor exports published by samcloud.drone.publish: no plan model, the IFC hash decides.
        try:
            digest = hashlib.sha256((directory / str(manifest["ifc"])).read_bytes()).hexdigest()
        except OSError:
            return "stale"
        return "fresh" if manifest.get("ifc_sha256") == digest else "stale"
    current = plan_model_sha256(project_root, level_key)
    if current is None or manifest.get("plan_model_sha256") != current:
        return "stale"
    return "fresh"


def _write_manifest(directory: Path, data: dict[str, object]) -> None:
    target = directory / EXPORT_MANIFEST
    temporary = directory / f".{EXPORT_MANIFEST}.tmp"
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def run_ifc_export(
    project_root: Path | str,
    level_key: str,
    progress: ProgressFn | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Export the level's IFC (+QA) and viewer assets, then write export.json. Returns the manifest.

    ``cancel`` returns True when the job should stop; Cancelled is raised between stages.
    """
    from samcloud.bim import viewer_export

    root = Path(project_root)
    report = progress or (lambda _fraction, _stage: None)

    def check() -> None:
        if cancel is not None and cancel():
            raise Cancelled("IFC export cancelled")

    def raising_cancel() -> None:
        check()

    check()
    # Re-apply the AI assistant's accepted edits before the plan is hashed and read (no-op without edits.json, never raises).
    from samcloud.dxf.assistant_hooks import ensure_applied_safely

    assistant_report = ensure_applied_safely(root, level_key, "ifc_export")
    plan_hash = plan_model_sha256(root, level_key)
    if plan_hash is None:
        raise RuntimeError(f"plan_model.json not found for level '{level_key}'. Generate the clean plan first.")
    report(0.0, "Exporting IFC")
    result = _export_level(
        root,
        level_key,
        progress=lambda fraction, stage: report(max(0.0, min(1.0, float(fraction))) * EXPORT_SPAN, stage),
        cancel=raising_cancel,
    )
    check()
    ifc_path = Path(result.ifc_path)
    out_dir = ifc_path.parent
    report(EXPORT_SPAN, "Creating viewer model")
    assets = viewer_export.export_viewer_assets(ifc_path, out_dir)
    check()
    manifest: dict[str, object] = {
        "plan_model_sha256": plan_hash,
        "ifc": ifc_path.name,
        "glb": assets.glb_path.name,
        "props": assets.props_path.name,
        "finished": datetime.now(timezone.utc).isoformat(),
        "counts": dict(result.counts),
        "viewer_counts": dict(assets.counts),
        "ids_failures": int(result.ids_failures),
        "express_findings": int(result.express_findings),
        "warnings": list(result.warnings),
    }
    if assistant_report is not None:
        manifest["assistant"] = assistant_report
    _write_manifest(out_dir, manifest)
    report(1.0, "IFC export complete")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAMcloud IFC export")
    parser.add_argument("action", choices=["export"])
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--level", required=True, type=str)
    args = parser.parse_args(argv)

    def progress(fraction: float, stage: str) -> None:
        print(f"[{int(fraction * 100):3d}%] {stage}", flush=True)

    try:
        manifest = run_ifc_export(args.project.resolve(), args.level, progress)
    except Cancelled as error:
        print(f"Cancelled: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"Failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
