"""Checkpoint-based vertical correction for drone-product georeferencing.

Checkpoint residuals follow ``dH = measured - surveyed``.  The stored target
height offset is therefore ``-mean(dH)`` so corrected export heights match the
surveyed datum.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def estimate_height_shift(
    checkpoint_json: Path | str,
    *,
    exclude: tuple[str, ...] | list[str] = (),
    min_points: int = 3,
    max_spread_m: float = 0.05,
) -> dict[str, Any]:
    """Estimate a safe vertical target-height offset from successful checkpoints.

    The estimate only uses points with status ``ok``.  Its spread is the range
    of the included dH values; a wider range requires a separate GCP adjustment.
    """
    source = Path(checkpoint_json)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Checkpoint file is not valid JSON: {source}") from error
    points = payload.get("points")
    if not isinstance(points, list):
        raise ValueError(f"Checkpoint file has no points list: {source}")

    excluded = set(exclude)
    used: list[tuple[str, float]] = []
    for point in points:
        if not isinstance(point, dict) or point.get("status") != "ok":
            continue
        name = str(point.get("name", ""))
        if name in excluded:
            continue
        try:
            delta_h = float(point["dH"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Checkpoint {name or '<unnamed>'} has no valid dH") from error
        if not np.isfinite(delta_h):
            raise ValueError(f"Checkpoint {name or '<unnamed>'} has non-finite dH")
        used.append((name, delta_h))

    if len(used) < min_points:
        raise ValueError(
            f"Height correction needs at least {min_points} successful checkpoints; found {len(used)}. "
            "A GCP adjustment is required."
        )

    values = np.asarray([value for _, value in used], dtype=float)
    spread_m = float(values.max() - values.min())
    if spread_m > max_spread_m:
        raise ValueError(
            f"Checkpoint height spread {spread_m:.4f} m exceeds {max_spread_m:.4f} m. "
            "A GCP adjustment is required."
        )

    mean_dh_m = float(values.mean())
    rows = []
    for index, (name, delta_h) in enumerate(used):
        others_mean = float(np.delete(values, index).mean())
        rows.append({"name": name, "dH_m": delta_h, "loo_residual_m": delta_h - others_mean})
    loo_max_m = max(abs(float(row["loo_residual_m"])) for row in rows)
    return {
        "checkpoint_file": str(source),
        "points": rows,
        "mean_dH_m": mean_dh_m,
        "std_dH_m": float(values.std()),
        "spread_m": spread_m,
        "max_loo_residual_m": loo_max_m,
        "height_offset_m": -mean_dh_m,
    }


def apply_height_shift(project_dir: Path | str, shift: dict[str, Any] | float) -> Path:
    """Copy-backup and apply a checkpoint-derived target-height offset."""
    project = Path(project_dir)
    georeference_path = project / "products" / "georeference.json"
    if not georeference_path.is_file():
        raise FileNotFoundError(f"Georeference file not found: {georeference_path}")
    try:
        georeference = json.loads(georeference_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Georeference file is not valid JSON: {georeference_path}") from error

    if isinstance(shift, dict):
        height_offset_m = float(shift["height_offset_m"])
        source_file = shift.get("checkpoint_file")
        used_points = [str(row["name"]) for row in shift.get("points", [])]
    else:
        height_offset_m = float(shift)
        source_file = None
        used_points = []
    if not np.isfinite(height_offset_m):
        raise ValueError("Height offset must be finite")

    timestamp = datetime.now(timezone.utc)
    backup = georeference_path.with_name(f"georeference.json.bak_{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}")
    shutil.copy2(georeference_path, backup)
    georeference["height_offset_m"] = height_offset_m
    georeference["height_offset_source"] = {
        "file": str(source_file) if source_file is not None else None,
        "points_used": used_points,
        "date": timestamp.isoformat(),
    }
    georeference_path.write_text(json.dumps(georeference, indent=2) + "\n", encoding="utf-8")
    return georeference_path


def revert_height_shift(project_dir: Path | str) -> Path:
    """Restore the newest copy-backup of a project's georeference without removing it."""
    georeference_path = Path(project_dir) / "products" / "georeference.json"
    backups = list(georeference_path.parent.glob("georeference.json.bak_*"))
    if not backups:
        raise FileNotFoundError(f"No georeference backup found beside: {georeference_path}")
    latest = max(backups, key=lambda path: path.name)
    shutil.copy2(latest, georeference_path)
    return georeference_path


def _print_estimate(estimate: dict[str, Any]) -> None:
    print("| Punkt | dH | LOO-Residuum |")
    print("|---|---:|---:|")
    for point in estimate["points"]:
        print(f"| {point['name']} | {point['dH_m']:+.4f} m | {point['loo_residual_m']:+.4f} m |")
    print()
    print(f"Höhenverschiebung: {estimate['height_offset_m']:+.4f} m")
    print(f"Mittelwert dH: {estimate['mean_dH_m']:+.4f} m; max. LOO-Residuum: {estimate['max_loo_residual_m']:.4f} m")


def main(argv: list[str] | None = None) -> None:
    """Print a checkpoint estimate and optionally apply it to the project georeference."""
    parser = argparse.ArgumentParser(description="Höhenkorrektur aus unabhängigen Checkpoints schätzen.")
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--exclude", nargs="*", default=(), metavar="PUNKT")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    estimate = estimate_height_shift(args.checkpoints, exclude=args.exclude)
    _print_estimate(estimate)
    if args.apply:
        destination = apply_height_shift(args.project, estimate)
        print(f"Höhenkorrektur in {destination} gespeichert.")


if __name__ == "__main__":
    main()
