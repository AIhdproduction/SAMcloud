"""Synthetic evaluation: extracted scene against the reference scene (spec 14.1, WP5.4).

``evaluate_against_reference(extracted, reference)`` matches elements by kind and overlap and
reports, per kind, detection precision and recall, and per quantity the error statistics (mean,
95th percentile, maximum, all absolute errors in metres, square metres or as stated).

Matching (greedy, largest overlap first, one to one):

- walls: footprints (from ``compute_scene_geometry``) overlap by at least ``min_overlap`` of the
  smaller footprint and the axes differ by less than 15 degrees;
- slabs, ceilings, spaces: plan polygons overlap by at least ``min_overlap`` of the smaller one;
- openings: the plan centres lie within half the reference width and the vertical ranges overlap.
  Windows and doors are scored per kind: precision and recall of the kind use only matched
  openings of that kind in both scenes, ``classification_accuracy`` is the share of matched
  openings whose kind (window, door, opening only) agrees.

Errors: wall axis endpoints (Euclidean distance of the trimmed axis ends, direction aligned),
``axis_offset`` (perpendicular distance of the axis midpoints), thickness, height; slab top
elevation and area; ceiling elevation (lowest underside) and area (slab and ceiling areas count
only the part not covered by wall footprints, because the reference slab runs under the walls);
opening width, height, sill height and plan position; space net area (also relative).
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon
from shapely.ops import unary_union

from samcloud.bim.core.geometry import SceneGeometry, compute_scene_geometry
from samcloud.bim.core.model import ElementKind, Scene

MIN_OVERLAP = 0.3
MAX_AXIS_ANGLE_DEG = 15.0
REFERENCE_SCENE_PATH = Path("scene") / "reference_scene.json"
"""Location of the generator's reference scene inside a project."""


@dataclass(frozen=True)
class ErrorStats:
    """Absolute error statistics of one quantity."""

    count: int
    mean: float
    p95: float
    max: float

    @classmethod
    def of(cls, values: list[float]) -> ErrorStats:
        if not values:
            return cls(0, 0.0, 0.0, 0.0)
        arr = np.abs(np.asarray(values, dtype=np.float64))
        return cls(len(arr), float(arr.mean()), float(np.quantile(arr, 0.95)), float(arr.max()))

    def to_dict(self) -> dict[str, float | int]:
        return {"count": self.count, "mean": self.mean, "p95": self.p95, "max": self.max}


@dataclass(frozen=True)
class Detection:
    """Detection counts of one kind. Without references recall is 1, without extractions 1."""

    reference: int
    extracted: int
    matched: int

    @property
    def precision(self) -> float:
        return self.matched / self.extracted if self.extracted else 1.0

    @property
    def recall(self) -> float:
        return self.matched / self.reference if self.reference else 1.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "reference": self.reference,
            "extracted": self.extracted,
            "matched": self.matched,
            "precision": self.precision,
            "recall": self.recall,
        }


@dataclass
class EvaluationReport:
    """Result of :func:`evaluate_against_reference`."""

    detection: dict[str, Detection] = field(default_factory=dict)
    errors: dict[str, ErrorStats] = field(default_factory=dict)
    """Keys ``<kind>.<quantity>``, for example ``wall.thickness`` or ``opening.width``."""
    classification_accuracy: float | None = None
    """Share of matched openings with the same kind (window, door, opening only)."""
    unmatched_reference: dict[str, list[str]] = field(default_factory=dict)
    unmatched_extracted: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection": {k: v.to_dict() for k, v in sorted(self.detection.items())},
            "errors": {k: v.to_dict() for k, v in sorted(self.errors.items())},
            "classification_accuracy": self.classification_accuracy,
            "unmatched_reference": {
                k: sorted(v) for k, v in sorted(self.unmatched_reference.items())
            },
            "unmatched_extracted": {
                k: sorted(v) for k, v in sorted(self.unmatched_extracted.items())
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _greedy(scores: dict[tuple[str, str], float]) -> list[tuple[str, str]]:
    """One to one assignment, best score first (ties by ids, deterministic)."""
    used_a: set[str] = set()
    used_b: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for (a, b), _ in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        if a not in used_a and b not in used_b:
            used_a.add(a)
            used_b.add(b)
            pairs.append((a, b))
    return pairs


def _overlap_scores(
    ref: dict[str, Polygon], ext: dict[str, Polygon], min_overlap: float
) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for rid, rp in ref.items():
        for eid, ep in ext.items():
            if rp.is_empty or ep.is_empty or not rp.intersects(ep):
                continue
            inter = rp.intersection(ep).area
            share = inter / min(rp.area, ep.area)
            if share >= min_overlap:
                scores[(rid, eid)] = inter
    return scores


def _record(
    report: EvaluationReport,
    kind: str,
    ref_ids: list[str],
    ext_ids: list[str],
    pairs: list[tuple[str, str]],
) -> None:
    report.detection[kind] = Detection(len(ref_ids), len(ext_ids), len(pairs))
    matched_ref = {a for a, _ in pairs}
    matched_ext = {b for _, b in pairs}
    report.unmatched_reference[kind] = [i for i in ref_ids if i not in matched_ref]
    report.unmatched_extracted[kind] = [i for i in ext_ids if i not in matched_ext]


def _evaluate_walls(
    report: EvaluationReport, ext: SceneGeometry, ref: SceneGeometry, min_overlap: float
) -> list[tuple[str, str]]:
    def angle(geo: Any) -> float:
        return math.atan2(geo.direction[1], geo.direction[0])

    scores = _overlap_scores(
        {k: Polygon(v.footprint) for k, v in ref.walls.items()},
        {k: Polygon(v.footprint) for k, v in ext.walls.items()},
        min_overlap,
    )
    for key in list(scores):
        diff = abs(angle(ref.walls[key[0]]) - angle(ext.walls[key[1]])) % math.pi
        if math.degrees(min(diff, math.pi - diff)) > MAX_AXIS_ANGLE_DEG:
            del scores[key]
    pairs = _greedy(scores)
    _record(report, "wall", list(ref.walls), list(ext.walls), pairs)
    endpoint: list[float] = []
    offset: list[float] = []
    thickness: list[float] = []
    height: list[float] = []
    for rid, eid in pairs:
        r, e = ref.walls[rid], ext.walls[eid]
        r0, r1 = np.array(r.axis_start), np.array(r.axis_end)
        e0, e1 = np.array(e.axis_start), np.array(e.axis_end)
        if float((r1 - r0) @ (e1 - e0)) < 0:
            e0, e1 = e1, e0
        endpoint += [float(np.linalg.norm(e0 - r0)), float(np.linalg.norm(e1 - r1))]
        normal = np.array(r.normal)
        offset.append(float((0.5 * (e0 + e1) - 0.5 * (r0 + r1)) @ normal))
        thickness.append(e.thickness - r.thickness)
        height.append(e.height - r.height)
    report.errors["wall.axis_endpoint"] = ErrorStats.of(endpoint)
    report.errors["wall.axis_offset"] = ErrorStats.of(offset)
    report.errors["wall.thickness"] = ErrorStats.of(thickness)
    report.errors["wall.height"] = ErrorStats.of(height)
    return pairs


def _free_area(polygon: Polygon, geo: SceneGeometry) -> float:
    """Area of ``polygon`` not covered by wall footprints (slabs run under the walls)."""
    walls = unary_union([Polygon(w.footprint) for w in geo.walls.values()])
    return float(polygon.difference(walls).area)


def _evaluate_slabs(
    report: EvaluationReport, ext: SceneGeometry, ref: SceneGeometry, min_overlap: float
) -> None:
    pairs = _greedy(
        _overlap_scores(
            {k: Polygon(v.outline, v.holes) for k, v in ref.slabs.items()},
            {k: Polygon(v.outline, v.holes) for k, v in ext.slabs.items()},
            min_overlap,
        )
    )
    _record(report, "slab", list(ref.slabs), list(ext.slabs), pairs)
    report.errors["slab.top"] = ErrorStats.of(
        [ext.slabs[e].top_z - ref.slabs[r].top_z for r, e in pairs]
    )
    report.errors["slab.area"] = ErrorStats.of(
        [
            _free_area(Polygon(ext.slabs[e].outline, ext.slabs[e].holes), ext)
            - _free_area(Polygon(ref.slabs[r].outline, ref.slabs[r].holes), ref)
            for r, e in pairs
        ]
    )
    cpairs = _greedy(
        _overlap_scores(
            {k: Polygon(v.outline) for k, v in ref.ceilings.items()},
            {k: Polygon(v.outline) for k, v in ext.ceilings.items()},
            min_overlap,
        )
    )
    _record(report, "ceiling", list(ref.ceilings), list(ext.ceilings), cpairs)
    report.errors["ceiling.elevation"] = ErrorStats.of(
        [ext.ceilings[e].bottom_z - ref.ceilings[r].bottom_z for r, e in cpairs]
    )
    report.errors["ceiling.area"] = ErrorStats.of(
        [
            _free_area(Polygon(ext.ceilings[e].outline), ext)
            - _free_area(Polygon(ref.ceilings[r].outline), ref)
            for r, e in cpairs
        ]
    )


def _evaluate_spaces(
    report: EvaluationReport, ext: SceneGeometry, ref: SceneGeometry, min_overlap: float
) -> None:
    pairs = _greedy(
        _overlap_scores(
            {k: Polygon(v.exterior, v.holes) for k, v in ref.spaces.items()},
            {k: Polygon(v.exterior, v.holes) for k, v in ext.spaces.items()},
            min_overlap,
        )
    )
    _record(report, "space", list(ref.spaces), list(ext.spaces), pairs)
    report.errors["space.area"] = ErrorStats.of(
        [ext.spaces[e].net_area - ref.spaces[r].net_area for r, e in pairs]
    )
    report.errors["space.area_rel"] = ErrorStats.of(
        [
            (ext.spaces[e].net_area - ref.spaces[r].net_area) / ref.spaces[r].net_area
            for r, e in pairs
            if ref.spaces[r].net_area > 0
        ]
    )


def _filling_kinds(scene: Scene) -> dict[str, str]:
    """Kind per opening id: ``window``, ``door`` or ``opening`` (no filling)."""
    kinds = {e.id: "opening" for e in scene.elements_of(ElementKind.OPENING)}
    for element in scene.elements.values():
        if element.kind in (ElementKind.WINDOW, ElementKind.DOOR):
            opening_id = getattr(element.params, "opening_id", None)
            if opening_id in kinds:
                kinds[opening_id] = element.kind.value
    return kinds


def _centre(geo: Any) -> NDArray[np.float64]:
    origin = np.array(geo.origin[:2])
    return np.asarray(origin + np.array(geo.u_axis[:2]) * geo.width / 2.0)


def _evaluate_openings(
    report: EvaluationReport,
    ext_scene: Scene,
    ref_scene: Scene,
    ext: SceneGeometry,
    ref: SceneGeometry,
) -> None:
    scores: dict[tuple[str, str], float] = {}
    for rid, r in ref.openings.items():
        rc = _centre(r)
        for eid, e in ext.openings.items():
            dist = float(np.linalg.norm(_centre(e) - rc))
            z_overlap = min(r.z_max, e.z_max) - max(r.z_min, e.z_min)
            if dist <= r.width / 2.0 and z_overlap > 0:
                scores[(rid, eid)] = 1.0 / (1.0 + dist)
    pairs = _greedy(scores)
    ref_kinds, ext_kinds = _filling_kinds(ref_scene), _filling_kinds(ext_scene)
    _record(report, "opening", list(ref.openings), list(ext.openings), pairs)
    for kind in ("window", "door"):
        ref_ids = [i for i, k in ref_kinds.items() if k == kind]
        ext_ids = [i for i, k in ext_kinds.items() if k == kind]
        good = [(r, e) for r, e in pairs if ref_kinds[r] == kind and ext_kinds[e] == kind]
        _record(report, kind, ref_ids, ext_ids, good)
    if pairs:
        report.classification_accuracy = sum(ref_kinds[r] == ext_kinds[e] for r, e in pairs) / len(
            pairs
        )
    report.errors["opening.width"] = ErrorStats.of(
        [ext.openings[e].width - ref.openings[r].width for r, e in pairs]
    )
    report.errors["opening.height"] = ErrorStats.of(
        [ext.openings[e].height - ref.openings[r].height for r, e in pairs]
    )
    report.errors["opening.sill"] = ErrorStats.of(
        [ext.openings[e].z_min - ref.openings[r].z_min for r, e in pairs]
    )
    report.errors["opening.position"] = ErrorStats.of(
        [
            float(np.linalg.norm(_centre(ext.openings[e]) - _centre(ref.openings[r])))
            for r, e in pairs
        ]
    )


_POINT_PARAMS = ("start", "end", "seed")
_RING_PARAMS = ("outline", "landing_outline")


def frame_shift(extracted: Scene, reference: Scene) -> tuple[float, float, float]:
    """Translation (metres, float64) from reference local coordinates to the extracted frame.

    Absolute position = local + ``settings.origin_offset``, so the shift is the difference of the
    two origin offsets (reference minus extracted).
    """
    r = reference.settings.origin_offset
    e = extracted.settings.origin_offset
    return (
        float(np.float64(r[0]) - np.float64(e[0])),
        float(np.float64(r[1]) - np.float64(e[1])),
        float(np.float64(r[2]) - np.float64(e[2])),
    )


def _to_frame(reference: Scene, shift: tuple[float, float, float]) -> Scene:
    """The reference scene moved by ``shift`` (metres) into another local frame.

    Moves the plan points (wall ends, outlines, openings of slabs, space seeds, stair starts)
    and the elevations of storeys, slabs and ceilings; quantities along a wall or relative to a
    storey are frame independent.
    """
    dx, dy, dz = shift
    data = reference.model_dump(mode="json")

    def move(point: list[float]) -> list[float]:
        return [point[0] + dx, point[1] + dy]

    for storey in data["storeys"].values():
        storey["elevation"] += dz
    for element in data["elements"].values():
        params = element["params"]
        for name in _POINT_PARAMS:
            if params.get(name) is not None:
                params[name] = move(params[name])
        for name in _RING_PARAMS:
            if params.get(name) is not None:
                params[name] = [move(point) for point in params[name]]
        if params.get("openings"):
            params["openings"] = [[move(p) for p in ring] for ring in params["openings"]]
        for name in ("top_elevation", "elevation"):
            if name in params and element["kind"] in ("slab", "ceiling"):
                params[name] += dz
        if element["kind"] == "ceiling" and params.get("plane_offset") is not None:
            params["plane_offset"] += dz
    return Scene.model_validate(data)


def evaluate_against_reference(
    extracted_scene: Scene, reference_scene: Scene, min_overlap: float = MIN_OVERLAP
) -> EvaluationReport:
    """Compare an extracted scene with the generator's reference scene (see module docstring).

    The scenes may use different origin offsets: the reference is translated into the frame of
    the extracted scene first.
    """
    shift = frame_shift(extracted_scene, reference_scene)
    if any(abs(v) > 1e-9 for v in shift):
        reference_scene = _to_frame(reference_scene, shift)
    ext = compute_scene_geometry(extracted_scene)
    ref = compute_scene_geometry(reference_scene)
    report = EvaluationReport()
    _evaluate_walls(report, ext, ref, min_overlap)
    _evaluate_slabs(report, ext, ref, min_overlap)
    _evaluate_spaces(report, ext, ref, min_overlap)
    _evaluate_openings(report, extracted_scene, reference_scene, ext, ref)
    return report


def evaluation_section(project: str | Path, scene: Scene | None = None) -> dict[str, Any] | None:
    """The ``evaluation`` section of the QA report, or None without a reference scene.

    Reads ``<project>/scene/reference_scene.json`` (written by the synthetic generator) and
    compares it with ``scene`` (default: ``scene/scene.json`` of the project). The dict is
    ``EvaluationReport.to_dict()`` plus ``reference`` (file name). ``render_evaluation_html``
    renders it.
    """
    root = Path(project)
    reference_path = root / REFERENCE_SCENE_PATH
    if not reference_path.is_file():
        return None
    extracted = scene if scene is not None else Scene.load(root / "scene" / "scene.json")
    reference = Scene.load(reference_path)
    data = evaluate_against_reference(extracted, reference).to_dict()
    data["reference"] = REFERENCE_SCENE_PATH.as_posix()
    shift = frame_shift(extracted, reference)
    if any(abs(v) > 1e-9 for v in shift):
        data["frame_shift"] = list(shift)
    return data


def render_evaluation_html(section: dict[str, Any]) -> str:
    """HTML fragment (heading and tables, no CSS) of an evaluation section."""

    def esc(value: Any) -> str:
        return html.escape(str(value))

    detection_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{v['reference']}</td><td>{v['extracted']}</td>"
        f"<td>{v['matched']}</td><td>{v['precision']:.3f}</td><td>{v['recall']:.3f}</td></tr>"
        for k, v in section["detection"].items()
    )
    error_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{v['count']}</td><td>{v['mean']:.4f}</td>"
        f"<td>{v['p95']:.4f}</td><td>{v['max']:.4f}</td></tr>"
        for k, v in section["errors"].items()
    )
    accuracy = section.get("classification_accuracy")
    text = "" if accuracy is None else f"<p>Opening classification accuracy: {accuracy:.3f}</p>"
    return (
        "<h2>Evaluation against the reference scene</h2>"
        f"<small>reference {esc(section.get('reference', ''))}</small>{text}"
        "<table><tr><th>kind</th><th>reference</th><th>extracted</th><th>matched</th>"
        f"<th>precision</th><th>recall</th></tr>{detection_rows}</table>"
        "<table><tr><th>quantity</th><th>n</th><th>mean</th><th>p95</th><th>max</th></tr>"
        f"{error_rows}</table>"
    )


__all__ = [
    "Detection",
    "ErrorStats",
    "EvaluationReport",
    "evaluate_against_reference",
    "evaluation_section",
    "render_evaluation_html",
]
