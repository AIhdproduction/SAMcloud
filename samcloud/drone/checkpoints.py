"""Independent checkpoint accuracy evaluation of an RTK-georeferenced drone project.

No bundle adjustment is run and no surveyed point enters the model: the surveyed points are projected
into the RTK-aligned sparse model, their targets are detected in the ORIGINAL (distorted) images,
triangulated with the model's cameras/poses, converted to LV95/LN02 and compared with the survey.

Sign convention: ``d = measured - surveyed`` (positive dH means the photogrammetric point lies above
the survey).  The survey file is a ``name,E,N,H,`` list in LV95 / LN02 (EPSG:2056+5728).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from samcloud.control_points.targets import detect_target, load_settings
from samcloud.control_points.auto_detect import (
    PredictedObservation,
    assign_detection,
    predict_observations,
    triangulate_observations,
)

LOGGER = logging.getLogger(__name__)

SURVEY_CRS = "EPSG:2056+5728"
CROP_PX = 64
MAX_CROPS_PER_POINT = 4
MAX_PRED_CROPS = 3
INLIER_PX = 3.0


def evaluate_checkpoints_model(
    reconstruction: Any,
    survey: dict[str, np.ndarray],
    image_paths: dict[str, Path],
    *,
    to_model: Callable[[np.ndarray], np.ndarray],
    from_model: Callable[[np.ndarray], np.ndarray],
    target_type: str = "cross",
    settings: dict[str, Any] | None = None,
    window_px: float = 150.0,
    refine_px: float = 25.0,
    max_plausible_m: float = 1.0,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Core evaluation on a loaded reconstruction.

    ``survey`` maps point name -> (E, N, H); ``to_model`` / ``from_model`` convert survey <-> model frame
    (both take and return (N, 3) arrays).  Nothing here modifies the reconstruction.  Points deviating more than
    ``max_plausible_m`` (3D) from the survey get status ``implausible`` (most likely a false detection) and are
    excluded from the totals.
    """
    settings = dict(settings or {})
    names = list(survey)
    model_xyz = dict(zip(names, to_model(np.asarray([survey[n] for n in names], dtype=float)), strict=True))

    stage1_pred = {n: predict_observations(reconstruction, model_xyz[n], 0.03, 0.05, settings) for n in names}
    stage1_pred = {n: [PredictedObservation(p.image, p.u, p.v, window_px) for p in items] for n, items in stage1_pred.items()}
    stage1 = _detect_stage(image_paths, stage1_pred, window_px, target_type, settings)

    rough: dict[str, np.ndarray] = {}
    for name in names:
        try:
            rough[name], _ = triangulate_observations(reconstruction, stage1[name], inlier_px=INLIER_PX * 3)
        except ValueError:
            continue
    stage2_pred = {n: predict_observations(reconstruction, xyz, 0.03, 0.05, settings) for n, xyz in rough.items()}
    stage2 = _detect_stage(image_paths, stage2_pred, refine_px, target_type, settings)

    points: list[dict[str, Any]] = []
    accepted_obs: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        entry: dict[str, Any] = {
            "name": name, "survey": [float(v) for v in survey[name]],
            "images_predicted": len(stage1_pred[name]), "images_detected": 0,
            "status": "not_found", "reason": "",
        }
        observations = stage2.get(name, [])
        entry["images_detected"] = len(observations)
        if not stage1_pred[name]:
            entry["reason"] = "Punkt liegt in keinem Bild"
        elif len(observations) == 0:
            entry["reason"] = "Target in keinem Bild erkannt"
        elif len(observations) < 2:
            entry["status"], entry["reason"] = "few_views", "nur 1 Detektion (mind. 2 noetig)"
        else:
            try:
                xyz, errors = triangulate_observations(reconstruction, observations, inlier_px=INLIER_PX)
            except ValueError as error:
                entry["status"], entry["reason"] = "few_views", f"Triangulation fehlgeschlagen: {error}"
            else:
                inliers = [(o, float(e)) for o, e in zip(_usable(reconstruction, observations), errors, strict=False) if e <= INLIER_PX]
                entry["images_inliers"] = len(inliers)
                if len(inliers) < 2:
                    entry["status"], entry["reason"] = "few_views", "weniger als 2 konsistente Detektionen"
                else:
                    measured = np.asarray(from_model(xyz[None, :]), dtype=float)[0]
                    delta = measured - np.asarray(survey[name], dtype=float)
                    rms = float(np.sqrt(np.mean([e ** 2 for _, e in inliers])))
                    entry.update(
                        status="ok", reason="", measured=[float(v) for v in measured],
                        dE=float(delta[0]), dN=float(delta[1]), dH=float(delta[2]),
                        d2D=float(np.hypot(delta[0], delta[1])), d3D=float(np.linalg.norm(delta)),
                        reproj_rms_px=rms,
                    )
                    accepted_obs[name] = [dict(o, reprojection_px=e) for o, e in inliers]
                    if entry["d3D"] > max_plausible_m:
                        entry["status"] = "implausible"
                        entry["reason"] = (f"3D-Abweichung {entry['d3D']:.2f} m > {max_plausible_m:g} m: "
                                           "vermutlich Fehldetektion, nicht in Summen")
        points.append(entry)

    result = {
        "target_type": target_type, "convention": "d = gemessen - Soll (Vermessung)",
        "points": points, "totals": _totals(points),
    }
    if out_dir is not None:
        _write_crops(Path(out_dir) / "checkpoint_crops", points, accepted_obs, stage1_pred, image_paths)
    return result


def _detect_stage(
    image_paths: dict[str, Path], predictions: dict[str, list[PredictedObservation]], radius_px: float,
    target_type: str, settings: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Detect targets in predicted windows (like ``auto_detect.detect_stage`` but robust against detector errors)."""
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in predictions}
    by_image: dict[str, dict[str, tuple[float, float, float]]] = {}
    for name, items in predictions.items():
        for item in items:
            by_image.setdefault(item.image, {})[name] = (item.u, item.v, radius_px)
    detection_settings = load_settings(settings)
    for image_name, windows in by_image.items():
        path = image_paths.get(image_name)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path is not None else None
        if image is None:
            continue
        for name, (u, v, radius) in windows.items():
            try:
                found = detect_target(image, (u, v), radius, target_type, settings=detection_settings)
            except (cv2.error, AttributeError, ValueError) as error:  # degenerate blobs inside the detector
                LOGGER.debug("detector failed for %s in %s: %s", name, image_name, error)
                continue
            if found is not None and assign_detection(found, windows, name):
                results[name].append({"image": image_name, "u": found.u, "v": found.v, "score": found.score})
    return results


def _usable(reconstruction: Any, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Observations in the same order/filter as ``triangulate_observations`` uses for its error list."""
    return [o for o in observations if reconstruction.find_image_with_name(str(o["image"])) is not None]


def _totals(points: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [p for p in points if p["status"] == "ok"]
    totals: dict[str, Any] = {"n_points": len(points), "n_ok": len(ok)}
    if not ok:
        return totals
    for key in ("dE", "dN", "dH", "d2D", "d3D"):
        totals[f"rms_{key[1:]}"] = float(np.sqrt(np.mean([p[key] ** 2 for p in ok])))
    for key in ("dE", "dN", "dH"):
        totals[f"bias_{key[1:]}"] = float(np.mean([p[key] for p in ok]))
    return totals


def _crop(image: np.ndarray, u: float, v: float, size: int) -> np.ndarray:
    """Square crop (padded with black) centred on pixel coordinates (u, v)."""
    half = size // 2
    cx, cy = int(round(u - 0.5)), int(round(v - 0.5))
    out = np.zeros((size, size) + image.shape[2:], dtype=image.dtype)
    x0, y0 = max(cx - half, 0), max(cy - half, 0)
    x1, y1 = min(cx + half, image.shape[1]), min(cy + half, image.shape[0])
    if x1 > x0 and y1 > y0:
        out[y0 - (cy - half):y1 - (cy - half), x0 - (cx - half):x1 - (cx - half)] = image[y0:y1, x0:x1]
    return out


def _write_crops(
    directory: Path, points: list[dict[str, Any]], accepted: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[PredictedObservation]], image_paths: dict[str, Path],
) -> None:
    """64 px crops around accepted detections; wider predicted windows for points that were not found."""
    directory.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str, float, float, int, str]] = []
    for entry in points:
        name = entry["name"]
        if entry["name"] in accepted:
            best = sorted(accepted[name], key=lambda o: o["reprojection_px"])[:MAX_CROPS_PER_POINT]
            jobs += [(name, o["image"], o["u"], o["v"], CROP_PX, "") for o in best]
        else:
            central = sorted(predictions[name], key=lambda p: abs(p.u - 2640) + abs(p.v - 1980))[:MAX_PRED_CROPS]
            jobs += [(name, p.image, p.u, p.v, 2 * CROP_PX + 72, "pred_") for p in central]
    for name, image_name, u, v, size, prefix in jobs:
        path = image_paths.get(image_name)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path is not None else None
        if image is None:
            continue
        stem = Path(image_name).stem
        cv2.imwrite(str(directory / f"{prefix}{name}_{stem}.png"), _crop(image, u, v, size))


def render_markdown(result: dict[str, Any], title: str) -> str:
    """German Markdown report with the per-point table and totals."""
    lines = [f"# Checkpoint-Auswertung: {title}", "",
             f"Targettyp: {result['target_type']}; Konvention: {result['convention']}. Werte in Metern.", "",
             "| Punkt | Status | dE | dN | dH | d2D | d3D | Bilder (erkannt/vorhergesagt) | Reproj. RMS px |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in result["points"]:
        views = f"{p['images_detected']}/{p['images_predicted']}"
        if "dE" in p:
            status = "ok" if p["status"] == "ok" else f"{p['status']} ({p['reason']})"
            lines.append(f"| {p['name']} | {status} | {p['dE']:+.3f} | {p['dN']:+.3f} | {p['dH']:+.3f} | {p['d2D']:.3f} | {p['d3D']:.3f} | {views} | {p['reproj_rms_px']:.2f} |")
        else:
            lines.append(f"| {p['name']} | {p['status']} ({p['reason']}) | - | - | - | - | - | {views} | - |")
    t = result["totals"]
    lines += ["", f"Auswertbare Punkte: {t['n_ok']} von {t['n_points']}."]
    if t["n_ok"]:
        lines += ["", "| Summe | E | N | H | 2D | 3D |", "|---|---|---|---|---|---|",
                  f"| RMS | {t['rms_E']:.3f} | {t['rms_N']:.3f} | {t['rms_H']:.3f} | {t['rms_2D']:.3f} | {t['rms_3D']:.3f} |",
                  f"| Bias (Mittel) | {t['bias_E']:+.3f} | {t['bias_N']:+.3f} | {t['bias_H']:+.3f} | | |"]
    return "\n".join(lines) + "\n"


def _image_map(image_dir: Path, names: list[str]) -> dict[str, Path]:
    """COLMAP image name (relative path) -> file below ``image_dir``; unique bare names as fallback."""
    mapping: dict[str, Path] = {}
    missing = []
    for name in names:
        candidate = image_dir / name
        (mapping.__setitem__(name, candidate) if candidate.is_file() else missing.append(name))
    if missing:
        by_name: dict[str, list[Path]] = {}
        for path in image_dir.rglob("*"):
            if path.is_file():
                by_name.setdefault(path.name, []).append(path)
        for name in missing:
            hits = by_name.get(Path(name).name, [])
            if len(hits) == 1:
                mapping[name] = hits[0]
    return mapping


def _project_settings(project_dir: Path) -> tuple[dict[str, Any], Path]:
    """control_points settings and image directory of the project yml."""
    import yaml

    config = yaml.safe_load((project_dir / "project.samcloud.yml").read_text(encoding="utf-8")) or {}
    settings = dict((config.get("settings") or {}).get("control_points") or {})
    image_dir = ((config.get("acquisition") or {}).get("sources") or {}).get("image_directory") \
        or (config.get("paths") or {}).get("image_directory")
    if not image_dir:
        raise ValueError("project yml has no acquisition.sources.image_directory")
    return settings, Path(image_dir)


def evaluate_checkpoints(
    project_dir: Path | str, control_points_csv: Path | str, *,
    target_type: str = "cross", out_dir: Path | str | None = None, name: str = "baseline",
) -> dict[str, Any]:
    """Evaluate surveyed points as pure checkpoints against the RTK-aligned sparse model of a project."""
    import pycolmap

    from samcloud.control_points.georeference import coordinates_to_ecef
    from samcloud.control_points.repository import read_control_points
    from samcloud.core.sparse_model import resolve_sparse_model
    from samcloud.drone.frames import enu_to_lv95, project_georeference

    project = Path(project_dir)
    products = project / "products"
    model_dir = resolve_sparse_model(products / "sparse")
    if model_dir is None:
        raise FileNotFoundError(f"no sparse model under {products / 'sparse'}")
    settings, image_dir = _project_settings(project)
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    georef = project_georeference(project)
    scale = float(georef["scale"])
    rotation = np.asarray(georef["rotation"], dtype=float)
    translation = np.asarray(georef["translation"], dtype=float)

    table = read_control_points(Path(control_points_csv))
    survey = {str(r.name): np.array([r.x, r.y, r.z], dtype=float) for r in table.itertuples(index=False)}

    def to_model(xyz: np.ndarray) -> np.ndarray:
        ecef = coordinates_to_ecef(xyz, SURVEY_CRS)
        return (rotation.T @ ((ecef - translation) / scale).T).T

    names = [reconstruction.image(i).name for i in reconstruction.reg_image_ids()]
    result = evaluate_checkpoints_model(
        reconstruction, survey, _image_map(image_dir, names),
        to_model=to_model, from_model=lambda xyz: enu_to_lv95(xyz, georef),
        target_type=target_type, settings=settings,
        out_dir=Path(out_dir) if out_dir else project / "reports",
    )
    result.update(project=str(project), model=str(model_dir), points_file=str(control_points_csv), name=name)
    destination = Path(out_dir) if out_dir else project / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"checkpoints_{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (destination / f"checkpoints_{name}.md").write_text(render_markdown(result, name), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Checkpoint accuracy of an RTK-aligned drone project (no adjustment).")
    parser.add_argument("--project", required=True)
    parser.add_argument("--points", required=True)
    parser.add_argument("--target", default="cross", choices=["cross", "circle"])
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    result = evaluate_checkpoints(args.project, args.points, target_type=args.target, out_dir=args.out_dir, name=args.name)
    print(render_markdown(result, args.name))


if __name__ == "__main__":
    main()
