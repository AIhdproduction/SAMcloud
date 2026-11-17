"""Furniture shape recognition: the simplest outline that explains the measured footprint (MDL-style choice).

Every candidate outline is scored against the object's footprint evidence (the traced instance mask) on one
raster in the wall frame:

    cost = (1 - IoU) + complexity penalty

Candidates (all along the wall axes unless noted):

* ``rect``       the IoU-optimal wall-axis rectangle (edges moved by coordinate descent on a summed-area table);
* ``rect_free``  the minimum rotated rectangle, only for an object turned clearly off the wall axes;
* ``L``          a rectangle minus one corner notch (= union of two rectangles: corner sofa, L desk);
* ``U``          a rectangle minus a notch in the middle of one side (= union of three rectangles);
* ``circle`` / ``ellipse`` for round tables, stools and similar (only for object classes that can be round);
* extra candidates supplied by the caller (e.g. a rectilinear simplification) with their own complexity.

The complexity penalty is counted in extra rectangles and depends on the object class: a sofa may well be a
corner sofa, a bed or a dining table is a rectangle.  A notch must be large enough to be furniture geometry
(both sides at least ``min_notch_m`` and a share of the bounding box), never trace noise, and every arm keeps
``min_arm_m``.  All lengths are metric and scaled by ``units_per_metre``.

Sofas additionally get CAD detail from the height profile: backrest and armrest bands (raised above the seat)
become inner lines, the seat is what remains; a corner module of an L / U sofa is marked by continuing the
inner corner across the seat.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import math
from typing import Any

import cv2
import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import linemerge, unary_union


@dataclass(frozen=True, slots=True)
class FurnitureShapeSettings:
    """Metric thresholds of the shape choice (config ``plan_polish:`` keys ``furniture_shape_*``)."""

    resolution_m: float = 0.02
    min_notch_m: float = 0.25
    min_notch_share: float = 0.06
    min_arm_m: float = 0.25
    snap_deg: float = 15.0
    penalty_free_rotation: float = 0.02
    penalty_default: float = 0.07
    penalty_sofa: float = 0.035
    penalty_desk: float = 0.06
    penalty_rectangular: float = 0.25
    penalty_ellipse: float = 0.03
    min_shape_iou: float = 0.85  # an L / U / round outline must explain the evidence at least this well
    min_free_iou: float = 0.80  # a rotated rectangle must explain the evidence at least this well
    min_rect_iou: float = 0.70  # a rectangle replaces a rectilinear trace only when it explains the evidence this well
    descent_rounds: int = 8
    # sofa detail from heights
    sofa_back_min_m: float = 0.12
    sofa_back_max_depth_m: float = 0.40
    sofa_back_min_depth_m: float = 0.10
    sofa_edge_min_m: float = 0.40
    sofa_sample_m: float = 0.04


SOFA_LABELS = frozenset({"sofa", "couch", "sessel", "armchair"})
_DESK_LABELS = frozenset({"schreibtisch", "desk"})
_RECTANGULAR_LABELS = frozenset({
    "bett", "bed", "mattress", "esstisch", "dining table", "tisch", "table", "couchtisch", "coffee table", "nachttisch",
    "nightstand", "stuhl", "chair", "waschmaschine", "washing machine", "kühlschrank", "refrigerator", "tv",
})
_ROUND_LABELS = frozenset({
    "tisch", "table", "couchtisch", "coffee table", "esstisch", "dining table", "pflanze", "plant", "stehlampe", "floor lamp",
    "hocker", "stool",
})


@dataclass(slots=True)
class ShapeCandidate:
    kind: str
    polygon: Polygon
    iou: float
    complexity: float
    cost: float


@dataclass(slots=True)
class ShapeChoice:
    """The chosen outline plus every scored candidate (kind -> (IoU, cost))."""

    kind: str
    polygon: Polygon
    iou: float
    boundary_m: float
    cost: float
    scores: dict[str, tuple[float, float]] = field(default_factory=dict)


def class_penalty(label: str | None, settings: FurnitureShapeSettings = FurnitureShapeSettings()) -> float:
    """Penalty per extra rectangle for an object class (display or semantic label)."""
    key = str(label or "").split(" h=")[0].strip().lower()
    if key in SOFA_LABELS:
        return settings.penalty_sofa
    if key in _DESK_LABELS:
        return settings.penalty_desk
    if key in _RECTANGULAR_LABELS:
        return settings.penalty_rectangular
    return settings.penalty_default


def is_sofa(label: str | None) -> bool:
    return str(label or "").split(" h=")[0].strip().lower() in {"sofa", "couch"}


# --------------------------------------------------------------------------------------------- evidence raster

@dataclass(slots=True)
class _Frame:
    """Evidence raster in the wall frame: (u, v) = local coordinates rotated by -theta about ``centre``."""

    theta: float
    centre: tuple[float, float]
    u0: float
    v0: float
    res: float
    mask: np.ndarray  # rows = v, cols = u
    table: np.ndarray  # summed-area table, shape (rows + 1, cols + 1)

    @property
    def total(self) -> float:
        return float(self.table[-1, -1])

    def count(self, r0: Any, r1: Any, c0: Any, c1: Any) -> Any:
        t = self.table
        return t[r1, c1] - t[r0, c1] - t[r1, c0] + t[r0, c0]

    def to_local(self, polygon: Polygon) -> Polygon:
        shifted = affinity.translate(polygon, self.u0, self.v0)
        return affinity.rotate(shifted, self.theta, origin=self.centre, use_radians=True)

    def to_frame(self, geometry: Any) -> Any:
        rotated = affinity.rotate(geometry, -self.theta, origin=self.centre, use_radians=True)
        return affinity.translate(rotated, -self.u0, -self.v0)

    def pixel_box(self, r0: int, r1: int, c0: int, c1: int) -> Polygon:
        return box(c0 * self.res, r0 * self.res, c1 * self.res, r1 * self.res)

    def rasterize(self, polygon: Polygon) -> np.ndarray:
        """Mask of a local-coordinate polygon on this frame's raster (pixel centres inside)."""
        local = self.to_frame(polygon)
        rows, cols = self.mask.shape
        xs = (np.arange(cols) + 0.5) * self.res
        ys = (np.arange(rows) + 0.5) * self.res
        gx, gy = np.meshgrid(xs, ys)
        return shapely.contains_xy(local, gx, gy)


def _frame_of(evidence: Polygon, theta: float, res: float, pad_px: int) -> _Frame | None:
    centre = evidence.centroid
    rotated = affinity.rotate(evidence, -theta, origin=(centre.x, centre.y), use_radians=True)
    min_u, min_v, max_u, max_v = rotated.bounds
    while ((max_u - min_u) / res) * ((max_v - min_v) / res) > 250_000:
        res *= 1.5
    u0, v0 = min_u - pad_px * res, min_v - pad_px * res
    cols = int(math.ceil((max_u - min_u) / res)) + 2 * pad_px + 1
    rows = int(math.ceil((max_v - min_v) / res)) + 2 * pad_px + 1
    xs = u0 + (np.arange(cols) + 0.5) * res
    ys = v0 + (np.arange(rows) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    mask = shapely.contains_xy(rotated, gx, gy)
    if not mask.any():
        return None
    table = np.zeros((rows + 1, cols + 1), dtype=np.float64)
    table[1:, 1:] = np.cumsum(np.cumsum(mask, axis=0), axis=1)
    return _Frame(theta, (float(centre.x), float(centre.y)), u0, v0, res, mask, table)


def _iou(frame: _Frame, inside: float, area: float) -> float:
    union = area + frame.total - inside
    return float(inside / union) if union > 0.0 else 0.0


def _iou_mask(frame: _Frame, candidate: np.ndarray) -> float:
    inside = float(np.count_nonzero(candidate & frame.mask))
    return _iou(frame, inside, float(np.count_nonzero(candidate)))


# --------------------------------------------------------------------------------------------- rectangle / L / U

def _trimmed_bounds(frame: _Frame, share: float = 0.01) -> tuple[int, int, int, int]:
    rows = frame.mask.sum(axis=1).astype(float)
    cols = frame.mask.sum(axis=0).astype(float)

    def span(profile: np.ndarray) -> tuple[int, int]:
        cumulative = np.cumsum(profile) / max(float(profile.sum()), 1e-9)
        low = int(np.searchsorted(cumulative, share))
        high = int(np.searchsorted(cumulative, 1.0 - share)) + 1
        return low, max(high, low + 1)

    r0, r1 = span(rows)
    c0, c1 = span(cols)
    return r0, r1, c0, c1


def _best_rect(frame: _Frame, rounds: int) -> tuple[tuple[int, int, int, int], float]:
    rows, cols = frame.mask.shape
    r0, r1, c0, c1 = _trimmed_bounds(frame)
    best = _iou(frame, float(frame.count(r0, r1, c0, c1)), float((r1 - r0) * (c1 - c0)))
    for _ in range(rounds):
        changed = False
        for name in ("r0", "r1", "c0", "c1"):
            if name == "r0":
                values = np.arange(0, r1)
                inside, area = frame.count(values, r1, c0, c1), (r1 - values) * (c1 - c0)
            elif name == "r1":
                values = np.arange(r0 + 1, rows + 1)
                inside, area = frame.count(r0, values, c0, c1), (values - r0) * (c1 - c0)
            elif name == "c0":
                values = np.arange(0, c1)
                inside, area = frame.count(r0, r1, values, c1), (r1 - r0) * (c1 - values)
            else:
                values = np.arange(c0 + 1, cols + 1)
                inside, area = frame.count(r0, r1, c0, values), (r1 - r0) * (values - c0)
            scores = inside / np.maximum(area + frame.total - inside, 1e-9)
            k = int(np.argmax(scores))
            if scores[k] > best + 1e-12:
                best = float(scores[k])
                value = int(values[k])
                r0, r1, c0, c1 = (value, r1, c0, c1) if name == "r0" else (r0, value, c0, c1) if name == "r1" else (r0, r1, value, c1) if name == "c0" else (r0, r1, c0, value)
                changed = True
        if not changed:
            break
    return (r0, r1, c0, c1), best


def _notch_valid(r0: int, r1: int, c0: int, c1: int, notch: tuple[int, int, int, int], limits: tuple[int, int, float], sides: int = 0) -> bool:
    """``sides`` = number of outer sides the notch must touch (2 = a corner notch / L, 1 = a side notch / U, 0 = any)."""
    min_notch, min_arm, min_share = limits
    nr0, nr1, nc0, nc1 = notch
    nh, nw = nr1 - nr0, nc1 - nc0
    if nh < min_notch or nw < min_notch or nh * nw < min_share * (r1 - r0) * (c1 - c0):
        return False
    if nr0 < r0 or nr1 > r1 or nc0 < c0 or nc1 > c1:
        return False
    touching = int(nr0 == r0) + int(nr1 == r1) + int(nc0 == c0) + int(nc1 == c1)
    if touching == 0 or (sides and touching != sides):
        return False  # a notch opens to the outside (a hole is not a furniture outline) and keeps its kind
    # every arm keeps min_arm: the notch never cuts the shape through
    for low, high, n_low, n_high in ((r0, r1, nr0, nr1), (c0, c1, nc0, nc1)):
        if (high - low) - (n_high - n_low) < min_arm:
            return False
        if n_low > low and n_low - low < min_arm:
            return False
        if n_high < high and high - n_high < min_arm:
            return False
    return True


def _shape_iou(frame: _Frame, outer: tuple[int, int, int, int], notch: tuple[int, int, int, int]) -> float:
    r0, r1, c0, c1 = outer
    nr0, nr1, nc0, nc1 = notch
    inside = float(frame.count(r0, r1, c0, c1) - frame.count(nr0, nr1, nc0, nc1))
    area = float((r1 - r0) * (c1 - c0) - (nr1 - nr0) * (nc1 - nc0))
    return _iou(frame, inside, area)


def _corner_notch_search(frame: _Frame, outer: tuple[int, int, int, int], corner: int, limits: tuple[int, int, float]) -> tuple[tuple[int, int, int, int], float] | None:
    """Best notch at one corner of ``outer`` (0 = low-low, 1 = low-high, 2 = high-low, 3 = high-high in (row, col))."""
    r0, r1, c0, c1 = outer
    min_notch, min_arm, min_share = limits
    rows = np.arange(r0 + min_arm, r1 - min_notch + 1) if corner >= 2 else np.arange(r0 + min_notch, r1 - min_arm + 1)
    cols = np.arange(c0 + min_arm, c1 - min_notch + 1) if corner in (1, 3) else np.arange(c0 + min_notch, c1 - min_arm + 1)
    if len(rows) == 0 or len(cols) == 0:
        return None
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    if corner >= 2:
        nr0, nr1 = rr, np.full_like(rr, r1)
    else:
        nr0, nr1 = np.full_like(rr, r0), rr
    if corner in (1, 3):
        nc0, nc1 = cc, np.full_like(cc, c1)
    else:
        nc0, nc1 = np.full_like(cc, c0), cc
    notch_inside = frame.count(nr0, nr1, nc0, nc1)
    notch_area = (nr1 - nr0) * (nc1 - nc0)
    outer_inside = float(frame.count(r0, r1, c0, c1))
    outer_area = float((r1 - r0) * (c1 - c0))
    inside = outer_inside - notch_inside
    area = outer_area - notch_area
    scores = inside / np.maximum(area + frame.total - inside, 1e-9)
    scores = np.where(notch_area >= min_share * outer_area, scores, -1.0)
    k = np.unravel_index(int(np.argmax(scores)), scores.shape)
    if scores[k] < 0.0:
        return None
    return (int(nr0[k]), int(nr1[k]), int(nc0[k]), int(nc1[k])), float(scores[k])


def _side_notch_search(frame: _Frame, outer: tuple[int, int, int, int], side: int, limits: tuple[int, int, float]) -> tuple[tuple[int, int, int, int], float] | None:
    """Best notch in the middle of one side (0 = low rows, 1 = high rows, 2 = low cols, 3 = high cols)."""
    r0, r1, c0, c1 = outer
    min_notch, min_arm, min_share = limits
    along = (c0, c1) if side < 2 else (r0, r1)
    depth_lo, depth_hi = (r0, r1) if side < 2 else (c0, c1)
    starts = np.arange(along[0] + min_arm, along[1] - min_arm - min_notch + 1)
    if len(starts) == 0:
        return None
    a, b = np.meshgrid(starts, np.arange(along[0] + min_arm + min_notch, along[1] - min_arm + 1), indexing="ij")
    valid_ab = (b - a) >= min_notch
    best: tuple[tuple[int, int, int, int], float] | None = None
    outer_inside = float(frame.count(r0, r1, c0, c1))
    outer_area = float((r1 - r0) * (c1 - c0))
    for depth in range(min_notch, (depth_hi - depth_lo) - min_arm + 1):
        if side in (0, 2):
            d0, d1 = depth_lo, depth_lo + depth
        else:
            d0, d1 = depth_hi - depth, depth_hi
        if side < 2:
            notch_inside = frame.count(d0, d1, a, b)
        else:
            notch_inside = frame.count(a, b, d0, d1)
        notch_area = (b - a) * depth
        inside = outer_inside - notch_inside
        area = outer_area - notch_area
        scores = inside / np.maximum(area + frame.total - inside, 1e-9)
        scores = np.where(valid_ab & (notch_area >= min_share * outer_area), scores, -1.0)
        k = np.unravel_index(int(np.argmax(scores)), scores.shape)
        if scores[k] >= 0.0 and (best is None or scores[k] > best[1]):
            notch = (d0, d1, int(a[k]), int(b[k])) if side < 2 else (int(a[k]), int(b[k]), d0, d1)
            best = (notch, float(scores[k]))
    return best


def _refine_notched(frame: _Frame, outer: tuple[int, int, int, int], notch: tuple[int, int, int, int], limits: tuple[int, int, float], rounds: int, sides: int = 0) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], float]:
    """Coordinate descent over the outer rectangle and the notch edges; edges shared by both move together."""
    rows, cols = frame.mask.shape
    outer_l, notch_l = list(outer), list(notch)
    best = _shape_iou(frame, tuple(outer_l), tuple(notch_l))  # type: ignore[arg-type]
    for _ in range(rounds):
        changed = False
        for which, index in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)):
            limit = rows if index < 2 else cols
            for value in range(0, limit + 1):
                trial_outer, trial_notch = list(outer_l), list(notch_l)
                target = trial_outer if which == 0 else trial_notch
                old = target[index]
                if value == old:
                    continue
                target[index] = value
                if which == 0 and notch_l[index] == outer_l[index]:
                    trial_notch[index] = value  # the notch stays open to the side it opens to
                if not _notch_valid(*trial_outer, tuple(trial_notch), limits, sides) or trial_outer[0] >= trial_outer[1] or trial_outer[2] >= trial_outer[3]:  # type: ignore[arg-type]
                    continue
                score = _shape_iou(frame, tuple(trial_outer), tuple(trial_notch))  # type: ignore[arg-type]
                if score > best + 1e-9:
                    best, outer_l, notch_l = score, trial_outer, trial_notch
                    changed = True
        if not changed:
            break
    return tuple(outer_l), tuple(notch_l), best  # type: ignore[return-value]


def _notched_polygon(frame: _Frame, outer: tuple[int, int, int, int], notch: tuple[int, int, int, int]) -> Polygon | None:
    shape = frame.pixel_box(*outer).difference(frame.pixel_box(*notch))
    parts = [part for part in getattr(shape, "geoms", [shape]) if isinstance(part, Polygon) and part.area > 0.0]
    if len(parts) != 1:
        return None
    return orient(parts[0].simplify(1e-9 * frame.res), 1.0)


# --------------------------------------------------------------------------------------------- round shapes

def _sample_ring(polygon: Polygon, count: int) -> np.ndarray:
    ring = polygon.exterior
    return np.asarray([ring.interpolate(k / count, normalized=True).coords[0] for k in range(count)], dtype=float)


def _circle(polygon: Polygon) -> Polygon | None:
    points = _sample_ring(polygon, 240)
    solution = np.linalg.lstsq(np.c_[2 * points, np.ones(len(points))], (points**2).sum(axis=1), rcond=None)[0]
    centre = solution[:2]
    radius_sq = float(solution[2] + centre @ centre)
    if radius_sq <= 0.0:
        return None
    return Point(centre).buffer(math.sqrt(radius_sq), quad_segs=24)


def _ellipse(polygon: Polygon) -> Polygon | None:
    points = _sample_ring(polygon, 240).astype(np.float32)
    try:
        (cx, cy), (first, second), angle = cv2.fitEllipse(points.reshape(-1, 1, 2))
    except cv2.error:
        return None
    if min(first, second) <= 1e-9:
        return None
    unit = Point(float(cx), float(cy)).buffer(1.0, quad_segs=24)
    return affinity.rotate(affinity.scale(unit, first / 2.0, second / 2.0, origin=(cx, cy)), float(angle), origin=(cx, cy))


# --------------------------------------------------------------------------------------------- the choice

def fit_furniture_shape(
    evidence: Polygon,
    theta: float,
    units_per_metre: float,
    label: str | None = None,
    settings: FurnitureShapeSettings = FurnitureShapeSettings(),
    extra_candidates: Iterable[tuple[str, Polygon, float]] = (),
) -> ShapeChoice | None:
    """Choose the outline with the lowest ``(1 - IoU) + penalty`` against the footprint evidence."""
    upm = max(float(units_per_metre), 1e-9)
    evidence = _largest(evidence.buffer(0)) if evidence is not None else None
    if evidence is None or evidence.area <= 0.0:
        return None
    res = settings.resolution_m * upm
    frame = _frame_of(evidence, theta, res, 2)
    if frame is None:
        return None
    res = frame.res
    penalty = class_penalty(label, settings)
    candidates: list[ShapeCandidate] = []

    def add(kind: str, polygon: Polygon | None, complexity: float, iou: float | None = None) -> None:
        if polygon is None or polygon.is_empty or polygon.area <= 0.0:
            return
        value = iou if iou is not None else _iou_mask(frame, frame.rasterize(polygon))
        candidates.append(ShapeCandidate(kind, polygon, value, complexity, (1.0 - value) + complexity * penalty))

    outer, rect_iou = _best_rect(frame, settings.descent_rounds)
    add("rect", frame.to_local(frame.pixel_box(*outer)), 0.0, rect_iou)
    free = evidence.minimum_rotated_rectangle
    if isinstance(free, Polygon) and free.area > 0.0:
        corners = np.asarray(free.exterior.coords[:4], dtype=float)
        edge = corners[1] - corners[0]
        deviation = abs(((math.degrees(math.atan2(edge[1], edge[0])) - math.degrees(theta) + 45.0) % 90.0) - 45.0)
        if deviation > settings.snap_deg:
            candidates.append(ShapeCandidate("rect_free", free, iou := _iou_mask(frame, frame.rasterize(free)), 0.0, (1.0 - iou) + settings.penalty_free_rotation))
    limits = (
        max(2, int(math.ceil(settings.min_notch_m * upm / res))),
        max(2, int(math.ceil(settings.min_arm_m * upm / res))),
        settings.min_notch_share,
    )
    full = _trimmed_bounds(frame, 0.005)
    for start in {full, outer}:
        best_l: tuple[float, tuple[int, int, int, int], tuple[int, int, int, int]] | None = None
        for corner in range(4):
            found = _corner_notch_search(frame, start, corner, limits)
            if found is not None and (best_l is None or found[1] > best_l[0]):
                best_l = (found[1], start, found[0])
        if best_l is not None:
            l_outer, l_notch, l_iou = _refine_notched(frame, best_l[1], best_l[2], limits, settings.descent_rounds, 2)
            shape = _notched_polygon(frame, l_outer, l_notch)
            if shape is not None:
                add("L", frame.to_local(shape), 1.0, l_iou)
        best_u: tuple[float, tuple[int, int, int, int], tuple[int, int, int, int]] | None = None
        for side in range(4):
            found = _side_notch_search(frame, start, side, limits)
            if found is not None and (best_u is None or found[1] > best_u[0]):
                best_u = (found[1], start, found[0])
        if best_u is not None:
            u_outer, u_notch, u_iou = _refine_notched(frame, best_u[1], best_u[2], limits, settings.descent_rounds, 1)
            shape = _notched_polygon(frame, u_outer, u_notch)
            if shape is not None:
                add("U", frame.to_local(shape), 2.0, u_iou)
    key = str(label or "").split(" h=")[0].strip().lower()
    if key in _ROUND_LABELS:
        circle = _circle(evidence)
        add("circle", circle, 0.0)
        ellipse = _ellipse(evidence)
        if ellipse is not None:
            iou = _iou_mask(frame, frame.rasterize(ellipse))
            candidates.append(ShapeCandidate("ellipse", ellipse, iou, 0.0, (1.0 - iou) + settings.penalty_ellipse))
    for kind, polygon, complexity in extra_candidates:
        add(kind, polygon, complexity)
    if not candidates:
        return None
    # keep the best of each kind (two starts for L / U)
    by_kind: dict[str, ShapeCandidate] = {}
    for item in candidates:
        if item.kind not in by_kind or item.cost < by_kind[item.kind].cost:
            by_kind[item.kind] = item
    # Noisy evidence inflates the gain of any detailed outline: a notched or round outline must also explain the
    # footprint well on its own, else the plain rectangle is kept.
    eligible = [
        item for item in by_kind.values()
        if item.kind == "rect" or item.iou >= (settings.min_free_iou if item.kind == "rect_free" else settings.min_shape_iou)
    ]
    chosen = min(eligible or list(by_kind.values()), key=lambda item: (item.cost, item.complexity))
    return ShapeChoice(
        chosen.kind, chosen.polygon, chosen.iou, boundary_distance(chosen.polygon, evidence) / upm, chosen.cost,
        {kind: (round(item.iou, 4), round(item.cost, 4)) for kind, item in by_kind.items()},
    )


def boundary_distance(first: Polygon, second: Polygon, samples: int = 200) -> float:
    """Mean symmetric distance between two outlines (units of the polygons)."""
    a = [first.exterior.interpolate(k / samples, normalized=True) for k in range(samples)]
    b = [second.exterior.interpolate(k / samples, normalized=True) for k in range(samples)]
    return 0.5 * (float(np.mean([second.exterior.distance(p) for p in a])) + float(np.mean([first.exterior.distance(p) for p in b])))


# --------------------------------------------------------------------------------------------- sofa detail

@dataclass(slots=True)
class SofaDetail:
    lines: list[LineString]
    seat: Polygon | None
    raised_edges: int


def sofa_details(
    shape: Polygon,
    theta: float,
    units_per_metre: float,
    height_of: Callable[[np.ndarray], np.ndarray],
    settings: FurnitureShapeSettings = FurnitureShapeSettings(),
) -> SofaDetail:
    """Backrest / armrest lines of a sofa outline from its height profile, plus corner-module lines.

    ``height_of`` returns the height above the floor (units) at local (x, y) points, NaN where unknown.  An edge
    of the outline is raised (backrest or armrest) when the band right inside it stands at least
    ``sofa_back_min_m`` above the seat; the band depth is where the profile falls half-way to the seat.
    """
    upm = max(float(units_per_metre), 1e-9)
    shape = orient(shape, 1.0)
    empty = SofaDetail([], None, 0)
    step = settings.sofa_sample_m * upm
    if shape.area <= 0.0:
        return empty
    # seat level: heights deeper than the deepest plausible backrest inside the outline
    core = shape.buffer(-settings.sofa_back_max_depth_m * upm, join_style=2)
    interior = core if not core.is_empty and core.area >= 0.05 * shape.area else shape.buffer(-0.5 * settings.sofa_back_min_depth_m * upm, join_style=2)
    seat_samples = _grid_points(interior, step)
    if len(seat_samples) < 4:
        return empty
    seat_heights = np.asarray(height_of(seat_samples), dtype=float)
    seat_heights = seat_heights[np.isfinite(seat_heights)]
    if len(seat_heights) < 0.3 * len(seat_samples):
        return empty
    seat = float(np.percentile(seat_heights, 40))
    coords = list(shape.exterior.coords)[:-1]
    strips: list[Polygon] = []
    depths: list[float] = []
    for index, start in enumerate(coords):
        end = coords[(index + 1) % len(coords)]
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        length = float(np.hypot(*(b - a)))
        if length < settings.sofa_edge_min_m * upm:
            continue
        tangent = (b - a) / length
        normal = np.asarray((-tangent[1], tangent[0]))  # left of a CCW ring = inside
        along = np.arange(0.15 * length, 0.85 * length + 1e-9, step)
        across = np.arange(0.5 * step, settings.sofa_back_max_depth_m * upm * 1.4, step)
        points = a[None, None, :] + along[None, :, None] * tangent + across[:, None, None] * normal
        flat = points.reshape(-1, 2)
        inside = shapely.contains_xy(shape, flat[:, 0], flat[:, 1]).reshape(points.shape[:2])
        heights = np.asarray(height_of(flat), dtype=float).reshape(points.shape[:2])
        heights[~inside] = np.nan
        if np.isfinite(heights).sum() < 0.3 * heights.size:
            continue
        with np.errstate(all="ignore"):
            profile = np.nanmedian(heights, axis=1)
        if not np.isfinite(profile[:3]).any():
            continue
        top = float(np.nanmax(profile[: max(3, int(round(0.20 * upm / step)))]))
        if top - seat < settings.sofa_back_min_m * upm:
            continue
        half = seat + 0.5 * (top - seat)
        below = np.flatnonzero(np.nan_to_num(profile, nan=seat) < half)
        below = below[across[below] >= 0.04 * upm] if len(below) else below
        depth = float(across[below[0]]) if len(below) else settings.sofa_back_max_depth_m * upm
        depth = float(np.clip(depth, settings.sofa_back_min_depth_m * upm, settings.sofa_back_max_depth_m * upm))
        strips.append(Polygon([tuple(a), tuple(b), tuple(b + normal * depth), tuple(a + normal * depth)]))
        depths.append(depth)
    if not strips:
        return empty
    seat_area = _largest(shape.difference(unary_union(strips)))
    lines: list[LineString] = []
    modules: list[LineString] = []
    outline = shape.exterior.buffer(0.01 * upm)
    if seat_area is not None:
        inner = seat_area.exterior.difference(outline)
        lines.extend(_line_parts(inner))
        # corner module of an L / U sofa: continue the edges meeting at a reflex corner across the seat
        for corner in _reflex_corners(shape):
            for direction in corner[1]:
                ray = LineString([corner[0], tuple(np.asarray(corner[0]) + direction * 10.0 * upm)])
                cut = ray.intersection(seat_area)
                for part in _line_parts(cut):
                    if part.length >= 0.10 * upm:
                        modules.append(part)
    merged = linemerge(unary_union(lines + modules)) if lines or modules else GeometryCollection()
    output = [line for line in _line_parts(merged) if line.length >= 0.10 * upm]
    output = [LineString([line.coords[0], line.coords[-1]]) if _straight(line) else line for line in output]
    label_area = seat_area
    if seat_area is not None and modules:
        label_area = _largest(seat_area.difference(unary_union(modules).buffer(0.005 * upm))) or seat_area
    return SofaDetail(output, label_area, len(strips))


def _straight(line: LineString) -> bool:
    start, end = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
    chord = float(np.hypot(*(end - start)))
    return chord > 0.0 and line.length <= chord * 1.0001


def _reflex_corners(shape: Polygon) -> list[tuple[tuple[float, float], list[np.ndarray]]]:
    """Reflex vertices of a CCW outline with the directions of the two edges continued into the shape."""
    coords = [np.asarray(point, dtype=float) for point in list(orient(shape, 1.0).exterior.coords)[:-1]]
    output: list[tuple[tuple[float, float], list[np.ndarray]]] = []
    count = len(coords)
    for index in range(count):
        previous, vertex, following = coords[index - 1], coords[index], coords[(index + 1) % count]
        first, second = vertex - previous, following - vertex
        cross = float(first[0] * second[1] - first[1] * second[0])
        if cross >= 0.0:
            continue
        incoming = first / max(float(np.hypot(*first)), 1e-12)
        outgoing = second / max(float(np.hypot(*second)), 1e-12)
        output.append(((float(vertex[0]), float(vertex[1])), [incoming, -outgoing]))
    return output


def _grid_points(polygon: Any, step: float) -> np.ndarray:
    if polygon is None or polygon.is_empty:
        return np.zeros((0, 2))
    min_x, min_y, max_x, max_y = polygon.bounds
    xs = np.arange(min_x + step / 2.0, max_x, step)
    ys = np.arange(min_y + step / 2.0, max_y, step)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((0, 2))
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack((gx.ravel(), gy.ravel()))
    return points[shapely.contains_xy(polygon, points[:, 0], points[:, 1])]


def _line_parts(geometry: Any) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, (MultiLineString, GeometryCollection)):
        output: list[LineString] = []
        for member in geometry.geoms:
            output.extend(_line_parts(member))
        return output
    return []


def _largest(geometry: Any) -> Polygon | None:
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, Polygon):
        return geometry if geometry.area > 0.0 else None
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        parts = [part for part in (_largest(member) for member in geometry.geoms) if part is not None]
        return max(parts, key=lambda part: part.area) if parts else None
    return None
