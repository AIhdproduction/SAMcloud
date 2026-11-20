"""Exterior door and window detection on reconstructed LoD2 facades."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from samcloud.bim.openings3d.plane_entry import detect_openings_on_plane
from samcloud.buildings.facades import facade_surfaces
from samcloud.drone.contracts import Evidence, Opening, PointSet, QAEntry, load_drone_config


def _opening_config(config) -> dict:
    if config is None:
        return load_drone_config()["openings"]
    return config.get("openings", config)


def _sample_dtm(dtm_sampler, x: float, y: float) -> float:
    try:
        return float(dtm_sampler(x, y))
    except TypeError:
        return float(dtm_sampler((x, y)))


def _reference_height(surface, u0: float, u1: float, dtm_sampler, strip_m: float) -> float | None:
    samples: list[float] = []
    for u in np.linspace(u0, u1, 5):
        for offset in np.linspace(max(0.03, strip_m / 6.0), strip_m, 3):
            xy = surface.origin_xy + surface.direction_xy * u + surface.normal_xy * offset
            try:
                value = _sample_dtm(dtm_sampler, float(xy[0]), float(xy[1]))
            except (TypeError, ValueError, ArithmeticError):
                continue
            if np.isfinite(value):
                samples.append(value)
    return None if not samples else float(np.median(samples))


def _overlaps(a: dict, b: dict) -> bool:
    return min(a["u1"], b["u1"]) > max(a["u0"], b["u0"]) and min(a["z1"], b["z1"]) > max(a["z0"], b["z0"])


def _qa(building_id: int, facade_index: int, ordinal: int, reason: str, candidate: dict) -> QAEntry:
    return QAEntry("opening", f"{building_id}:{facade_index}:{ordinal}", reason, {
        "kind": candidate["kind"], "u0": round(candidate["u0"], 3), "u1": round(candidate["u1"], 3),
        "z0": round(candidate["z0"], 3), "z1": round(candidate["z1"], 3),
    })


def detect_building_openings(ps: PointSet, model, facades, *, dtm_sampler, config=None) -> tuple[list[Opening], list[QAEntry]]:
    """Return plausible outdoor openings and QA entries for every rejected candidate."""
    cfg = _opening_config(config)
    surfaces = facade_surfaces(model, facades)
    points = np.asarray(ps.xyz, dtype=float)
    labels = np.asarray(ps.classes)
    view_count = None if ps.view_count is None else np.asarray(ps.view_count)
    openings: list[Opening] = []
    qa: list[QAEntry] = []
    for surface in surfaces:
        candidates = detect_openings_on_plane(points, labels, view_count, surface, config=cfg)
        accepted: list[dict] = []
        for ordinal, candidate in enumerate(candidates, start=1):
            width, height = candidate["u1"] - candidate["u0"], candidate["z1"] - candidate["z0"]
            area = width * height
            if not candidate["inside"]:
                qa.append(_qa(model.building_id, surface.facade_index, ordinal, "opening extends outside facade polygon", candidate))
                continue
            if not float(cfg["min_area_m2"]) <= area <= float(cfg["max_area_m2"]):
                qa.append(_qa(model.building_id, surface.facade_index, ordinal, f"area {area:.2f} m2 outside configured range", candidate))
                continue
            aspect = width / max(height, 1e-9)
            if not 0.2 <= aspect <= 5.0:
                qa.append(_qa(model.building_id, surface.facade_index, ordinal, f"aspect ratio {aspect:.2f} outside 0.2-5", candidate))
                continue
            if candidate["views"] < int(cfg["min_views"]):
                qa.append(_qa(model.building_id, surface.facade_index, ordinal, f"only {candidate['views']} views, need {cfg['min_views']}", candidate))
                continue
            reference = _reference_height(surface, candidate["u0"], candidate["u1"], dtm_sampler, float(cfg["reference_strip_m"]))
            if reference is None:
                qa.append(_qa(model.building_id, surface.facade_index, ordinal, "no valid DTM reference in facade strip", candidate))
                continue
            candidate["reference_z"] = reference
            clearance = candidate["z0"] - reference
            candidate["status"] = "ok"
            candidate["reason"] = ""
            if candidate["kind"] == "door":
                if abs(clearance) > float(cfg["door_bottom_tolerance_m"]):
                    if clearance >= float(cfg["window_min_sill_m"]):
                        candidate["kind"] = "window"
                        candidate["status"] = "warning"
                        candidate["reason"] = f"floating door bottom {clearance:.2f} m above DTM; reclassified as window"
                    else:
                        qa.append(_qa(model.building_id, surface.facade_index, ordinal, f"floating door bottom {clearance:.2f} m above DTM", candidate))
                        continue
            elif clearance < float(cfg["window_min_sill_m"]):
                qa.append(_qa(model.building_id, surface.facade_index, ordinal, f"window sill {clearance:.2f} m below minimum", candidate))
                continue
            accepted.append(candidate)
        kept: list[dict] = []
        for candidate in sorted(accepted, key=lambda item: item["score"], reverse=True):
            if any(_overlaps(candidate, other) for other in kept):
                qa.append(_qa(model.building_id, surface.facade_index, len(qa) + 1, "overlaps stronger opening", candidate))
                continue
            kept.append(candidate)
        for candidate in sorted(kept, key=lambda item: (item["u0"], item["z0"])):
            evidence = Evidence(
                views=int(candidate["views"]), points=int(candidate["points"]),
                notes={"median_views": round(candidate["median_views"], 2), "coverage": round(candidate["coverage"], 3), "reference_z": round(candidate["reference_z"], 3)},
            )
            openings.append(Opening(
                building_id=model.building_id, facade_index=surface.facade_index, kind=candidate["kind"],
                u0=candidate["u0"], u1=candidate["u1"], z0=candidate["z0"], z1=candidate["z1"],
                evidence=evidence, status=candidate["status"], reason=candidate["reason"],
            ))
    return openings, qa
