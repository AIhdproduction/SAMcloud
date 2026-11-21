"""Straight lines and circular arcs for pavement/lawn outlines and curb axes.

Real road, sidewalk and curb edges are straight or circular.  :func:`fit_chain` turns a noisy
polyline into a sequence of lines and arcs (a "bulge polyline": vertices plus one DXF bulge per
segment).  :func:`fit_surfaces` applies it to a whole planar partition: the outlines of all
surface polygons are noded into edge chains, every chain is fitted ONCE and both neighbours reuse
it, so no gaps or overlaps appear.  Curb axes take the geometry of the outline edge they lie on
(the curb is the master edge, neighbours share it) and are fitted on their own only when no such
edge exists.

Bulge convention (DXF): ``bulge = tan(sweep / 4)``, positive for counter-clockwise arcs.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge
from shapely.validation import make_valid

LOGGER = logging.getLogger(__name__)

_EPS = 1e-9
_DEFAULTS = {"min_radius_m": 1.0, "max_radius_m": 200.0, "tolerance_m": 0.05, "sample_m": 0.10,
             "tangent_deg": 10.0, "merge_deg": 3.0, "sagitta_m": 0.01, "curb_match_m": 0.12}


def fit_settings(config: Mapping | None) -> dict:
    """``edges.fit`` of the drone config merged over the built-in defaults."""
    out = dict(_DEFAULTS)
    node = (config or {}).get("edges", {}) if isinstance(config, Mapping) else {}
    out.update(node.get("fit", {}) if isinstance(node, Mapping) else {})
    return out


# ---------------------------------------------------------------------------------- bulge geometry


@dataclass
class BulgeLine:
    """Vertices ``pts`` (n, 2) and one bulge per segment (n - 1)."""

    pts: np.ndarray
    bulges: np.ndarray
    report: dict = field(default_factory=dict)

    @property
    def closed(self) -> bool:
        return len(self.pts) > 2 and bool(np.allclose(self.pts[0], self.pts[-1]))

    @property
    def arcs(self) -> int:
        return int((np.abs(self.bulges) > 1e-9).sum())

    @property
    def lines(self) -> int:
        return int((np.abs(self.bulges) <= 1e-9).sum())

    def reversed(self) -> "BulgeLine":
        return BulgeLine(self.pts[::-1].copy(), -self.bulges[::-1].copy(), dict(self.report))


def arc_geometry(p0, p1, bulge: float) -> tuple[np.ndarray, float, float, float]:
    """Centre, radius, start angle and signed sweep of the arc ``p0 -> p1`` with ``bulge``."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    chord = p1 - p0
    length = float(np.hypot(*chord))
    theta = 4.0 * math.atan(abs(bulge))
    radius = length / (2.0 * math.sin(theta / 2.0)) if theta > 1e-12 else float("inf")
    left = np.array([-chord[1], chord[0]]) / max(length, _EPS)
    sign = 1.0 if bulge > 0 else -1.0
    centre = (p0 + p1) / 2.0 + sign * left * radius * math.cos(theta / 2.0)
    start = math.atan2(*(p0 - centre)[::-1])
    return centre, radius, start, sign * theta


def bulge_for(p0, p1, radius: float, ccw: bool) -> float:
    """Bulge of the minor arc between two points with ``radius``."""
    chord = float(np.hypot(*(np.asarray(p1, float) - np.asarray(p0, float))))
    sweep = 2.0 * math.asin(min(1.0, chord / (2.0 * max(radius, _EPS))))
    return (1.0 if ccw else -1.0) * math.tan(sweep / 4.0)


def densify(line: BulgeLine, sagitta: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Dense XY polyline of ``line`` (arcs with sagitta <= ``sagitta``) and the segment index of each dense edge."""
    out = [line.pts[0]]
    seg: list[int] = []
    for k, bulge in enumerate(line.bulges):
        p0, p1 = line.pts[k], line.pts[k + 1]
        if abs(bulge) <= 1e-9:
            out.append(p1)
            seg.append(k)
            continue
        centre, radius, a0, sweep = arc_geometry(p0, p1, bulge)
        step = 2.0 * math.acos(max(-1.0, min(1.0, 1.0 - sagitta / radius))) if radius > sagitta else abs(sweep)
        n = max(1, int(math.ceil(abs(sweep) / max(step, 1e-6))))
        for m in range(1, n + 1):
            ang = a0 + sweep * m / n
            out.append(p1 if m == n else centre + radius * np.array([math.cos(ang), math.sin(ang)]))
            seg.append(k)
    return np.asarray(out, dtype=float), np.asarray(seg, dtype=np.int64)


def segment_lengths(line: BulgeLine) -> np.ndarray:
    lengths = np.hypot(*(line.pts[1:] - line.pts[:-1]).T)
    for k, bulge in enumerate(line.bulges):
        if abs(bulge) > 1e-9:
            _c, radius, _a, sweep = arc_geometry(line.pts[k], line.pts[k + 1], bulge)
            lengths[k] = radius * abs(sweep)
    return lengths


def point_at(line: BulgeLine, distance: float) -> tuple[np.ndarray, int, float]:
    """Point at arc-length ``distance``: ``(xy, segment, fraction within segment)``."""
    lengths = segment_lengths(line)
    cum = np.r_[0.0, np.cumsum(lengths)]
    distance = float(np.clip(distance, 0.0, cum[-1]))
    k = int(min(np.searchsorted(cum, distance, side="right") - 1, len(lengths) - 1))
    frac = 0.0 if lengths[k] <= _EPS else (distance - cum[k]) / lengths[k]
    return _segment_point(line, k, frac), k, frac


def _segment_point(line: BulgeLine, k: int, frac: float) -> np.ndarray:
    p0, p1, bulge = line.pts[k], line.pts[k + 1], float(line.bulges[k])
    if abs(bulge) <= 1e-9:
        return p0 + (p1 - p0) * frac
    centre, radius, a0, sweep = arc_geometry(p0, p1, bulge)
    ang = a0 + sweep * frac
    return centre + radius * np.array([math.cos(ang), math.sin(ang)])


def sub_line(line: BulgeLine, start: float, end: float) -> BulgeLine:
    """Part of ``line`` between the arc-length parameters ``start`` < ``end``."""
    lengths = segment_lengths(line)
    cum = np.r_[0.0, np.cumsum(lengths)]
    start, end = float(np.clip(start, 0, cum[-1])), float(np.clip(end, 0, cum[-1]))
    pts, bulges = [], []
    for k, length in enumerate(lengths):
        lo, hi = max(start, cum[k]), min(end, cum[k + 1])
        if hi - lo <= 1e-6 or length <= _EPS:
            continue
        f0, f1 = (lo - cum[k]) / length, (hi - cum[k]) / length
        a, b = _segment_point(line, k, f0), _segment_point(line, k, f1)
        bulge = float(line.bulges[k])
        if abs(bulge) > 1e-9:
            _c, _r, _a0, sweep = arc_geometry(line.pts[k], line.pts[k + 1], bulge)
            bulge = math.copysign(math.tan(abs(sweep) * (f1 - f0) / 4.0), bulge)
        if not pts:
            pts.append(a)
        pts.append(b)
        bulges.append(bulge)
    if not pts:
        a = point_at(line, start)[0]
        return BulgeLine(np.vstack((a, a)), np.zeros(1))
    return BulgeLine(np.asarray(pts), np.asarray(bulges))


def join_lines(parts: Sequence[BulgeLine]) -> BulgeLine:
    pts = [parts[0].pts[0]]
    bulges: list[float] = []
    for part in parts:
        pts.extend(part.pts[1:])
        bulges.extend(part.bulges)
    return BulgeLine(np.asarray(pts), np.asarray(bulges))


# ---------------------------------------------------------------------------------- primitive fits


def _resample(xy: np.ndarray, step: float) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)[:, :2]
    keep = np.r_[True, np.hypot(*np.diff(xy, axis=0).T) > _EPS]
    xy = xy[keep]
    if len(xy) < 2:
        return xy
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    n = max(1, int(math.ceil(d[-1] / step)))
    t = np.linspace(0.0, d[-1], n + 1)
    out = np.column_stack((np.interp(t, d, xy[:, 0]), np.interp(t, d, xy[:, 1])))
    out[0], out[-1] = xy[0], xy[-1]
    return out


def _fit_line(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centre = p.mean(axis=0)
    u = np.linalg.svd(p - centre, full_matrices=False)[2][0]
    if np.dot(p[-1] - p[0], u) < 0:
        u = -u
    dev = float(np.abs((p - centre) @ np.array([-u[1], u[0]])).max())
    return centre, u, dev


def _fit_circle(p: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = p.mean(axis=0)
    q = p - mean
    a = np.column_stack((2.0 * q, np.ones(len(q))))
    sol = np.linalg.lstsq(a, (q ** 2).sum(axis=1), rcond=None)[0]
    centre = sol[:2].copy()
    radius = math.sqrt(max(sol[2] + centre @ centre, 1e-12))
    for _ in range(6):  # Gauss-Newton on the geometric residual
        diff = q - centre
        dist = np.maximum(np.hypot(*diff.T), 1e-12)
        jac = np.column_stack((-diff / dist[:, None], -np.ones(len(q))))
        step = np.linalg.lstsq(jac, -(dist - radius), rcond=None)[0]
        centre, radius = centre + step[:2], radius + step[2]
        if np.abs(step).max() < 1e-7:
            break
    dev = float(np.abs(np.hypot(*(q - centre).T) - radius).max())
    return centre + mean, float(abs(radius)), dev


def _sweep(p: np.ndarray, centre: np.ndarray) -> float:
    ang = np.unwrap(np.arctan2(p[:, 1] - centre[1], p[:, 0] - centre[0]))
    return float(ang[-1] - ang[0])


class _Fitter:
    """Greedy line/arc segmentation of one resampled chain."""

    def __init__(self, p: np.ndarray, cfg: dict, tolerance: float) -> None:
        self.p = p
        self.n = len(p)
        self.cfg = cfg
        self.tol = tolerance
        self.rmin, self.rmax = float(cfg["min_radius_m"]), float(cfg["max_radius_m"])
        self.arc_min = max(8, int(round(0.8 / max(float(cfg["sample_m"]), 1e-3))))

    def line_ok(self, i: int, j: int) -> bool:
        return j - i < 1 or _fit_line(self.p[i:j + 1])[2] <= self.tol

    def arc_ok(self, i: int, j: int) -> bool:
        if j - i + 1 < self.arc_min:
            return False
        centre, radius, dev = _fit_circle(self.p[i:j + 1])
        if dev > self.tol or not (self.rmin <= radius <= self.rmax):
            return False
        sweep = abs(_sweep(self.p[i:j + 1], centre))
        return 0.03 <= sweep <= 2.6

    def _extend(self, ok, i: int, minimum: int) -> int:
        lo = i + minimum
        if lo >= self.n or not ok(i, lo):
            return -1
        step = max(minimum, 4)
        hi = lo
        while True:
            nxt = min(hi + step, self.n - 1)
            if nxt == hi:
                return hi
            if ok(i, nxt):
                hi = nxt
                step *= 2
                continue
            lo, bad = hi, nxt
            break
        while bad - lo > 1:
            mid = (lo + bad) // 2
            if ok(i, mid):
                lo = mid
            else:
                bad = mid
        return lo

    def segment(self) -> list[list]:
        runs: list[list] = []
        i = 0
        while i < self.n - 1:
            jl = max(self._extend(self.line_ok, i, 1), i + 1)
            ja = self._extend(self.arc_ok, i, self.arc_min - 1)
            if ja >= 0 and (ja - i) > 1.15 * (jl - i) + 2:
                runs.append(["arc", i, ja])
                i = ja
            else:
                runs.append(["line", i, jl])
                i = jl
        merged: list[list] = []
        merge_cos = math.cos(math.radians(float(self.cfg["merge_deg"])))
        for run in runs:
            if merged and run[0] == "line" and merged[-1][0] == "line":
                a = self.p[merged[-1][2]] - self.p[merged[-1][1]]
                b = self.p[run[2]] - self.p[run[1]]
                cos = float(a @ b) / max(float(np.hypot(*a) * np.hypot(*b)), _EPS)
                if cos >= merge_cos and self.line_ok(merged[-1][1], run[2]):
                    merged[-1][2] = run[2]
                    continue
            merged.append(run)
        return merged


def _intersect_lines(c1, u1, c2, u2):
    den = u1[0] * u2[1] - u1[1] * u2[0]
    if abs(den) < 1e-6:
        return None
    t = ((c2[0] - c1[0]) * u2[1] - (c2[1] - c1[1]) * u2[0]) / den
    return c1 + t * u1


def _line_circle(c, u, centre, radius):
    d = c - centre
    b = float(d @ u)
    disc = b * b - (float(d @ d) - radius * radius)
    if disc < 0:
        return []
    root = math.sqrt(disc)
    return [c + (-b - root) * u, c + (-b + root) * u]


def _circle_circle(c1, r1, c2, r2):
    d = float(np.hypot(*(c2 - c1)))
    if d < _EPS or d > r1 + r2 or d < abs(r1 - r2):
        return []
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = math.sqrt(max(r1 * r1 - a * a, 0.0))
    base = c1 + a * (c2 - c1) / d
    perp = np.array([-(c2 - c1)[1], (c2 - c1)[0]]) / d
    return [base + h * perp, base - h * perp]


def _project(prim, point: np.ndarray) -> np.ndarray:
    if prim["kind"] == "line":
        return prim["c"] + float((point - prim["c"]) @ prim["u"]) * prim["u"]
    off = point - prim["c"]
    return prim["c"] + prim["r"] * off / max(float(np.hypot(*off)), _EPS)


def _junction(a, b, sample: np.ndarray, reach: float = 1.0) -> np.ndarray:
    cands: list[np.ndarray] = []
    if a["kind"] == "line" and b["kind"] == "line":
        hit = _intersect_lines(a["c"], a["u"], b["c"], b["u"])
        cands = [] if hit is None else [hit]
    elif a["kind"] == "line":
        cands = _line_circle(a["c"], a["u"], b["c"], b["r"])
    elif b["kind"] == "line":
        cands = _line_circle(b["c"], b["u"], a["c"], a["r"])
    else:
        cands = _circle_circle(a["c"], a["r"], b["c"], b["r"])
    cands = [c for c in cands if float(np.hypot(*(c - sample))) <= reach]
    if cands:
        return min(cands, key=lambda c: float(np.hypot(*(c - sample))))
    return 0.5 * (_project(a, sample) + _project(b, sample))


def _dense_distance(line: BulgeLine, samples: np.ndarray, sagitta: float = 0.002) -> float:
    dense, _seg = densify(line, sagitta)
    if len(dense) < 2:
        return float("inf")
    return float(shapely.distance(shapely.points(samples), LineString(dense)).max())


def _build_chain(p: np.ndarray, runs: list[list], cfg: dict, tolerance: float) -> BulgeLine:
    """Refit the primitives of ``runs`` on their core samples and join them at exact junctions."""
    trim = max(2, int(round(1.0 / max(float(cfg["sample_m"]), 1e-3))))
    prims = []
    for index, (kind, i, j) in enumerate(runs):
        lo = i + (trim if index > 0 else 0)
        hi = j - (trim if index < len(runs) - 1 else 0)
        if hi - lo < (8 if kind == "arc" else 3):
            lo, hi = i, j
        core = p[lo:hi + 1]
        if kind == "line" or len(core) < 4:
            c, u, _dev = _fit_line(p[lo:hi + 1] if len(core) >= 2 else p[i:j + 1])
            prims.append({"kind": "line", "c": c, "u": u, "i": i, "j": j})
        else:
            c, r, _dev = _fit_circle(core)
            sweep = _sweep(p[i:j + 1], c)
            prims.append({"kind": "arc", "c": c, "r": r, "ccw": sweep > 0, "i": i, "j": j})
    # tangent fillet for line - arc - line: exact tangent continuity
    tangent = math.radians(float(cfg["tangent_deg"]))
    fillets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(1, len(prims) - 1):
        a, m, b = prims[k - 1], prims[k], prims[k + 1]
        if (a["kind"], m["kind"], b["kind"]) != ("line", "arc", "line"):
            continue
        v = _intersect_lines(a["c"], a["u"], b["c"], b["u"])
        if v is None:
            continue
        theta = math.atan2(a["u"][0] * b["u"][1] - a["u"][1] * b["u"][0], float(a["u"] @ b["u"]))
        if (theta > 0) != m["ccw"] or not (math.radians(2) < abs(theta) < math.radians(170)):
            continue
        sweep = abs(_sweep(p[m["i"]:m["j"] + 1], m["c"]))
        if abs(abs(theta) - sweep) > tangent:
            continue
        d = m["r"] * math.tan(abs(theta) / 2.0)
        fillets[k] = (v - a["u"] * d, v + b["u"] * d)
    pts = [p[0]]
    for k in range(len(prims) - 1):
        if k in fillets or (k + 1) in fillets:
            pts.append(fillets[k][1] if k in fillets else fillets[k + 1][0])
        else:
            pts.append(_junction(prims[k], prims[k + 1], p[prims[k]["j"]]))
    pts.append(p[-1])
    pts = np.asarray(pts)
    bulges = np.zeros(len(prims))
    for k, prim in enumerate(prims):
        if prim["kind"] == "arc":
            bulges[k] = bulge_for(pts[k], pts[k + 1], prim["r"], prim["ccw"])
    chain = BulgeLine(pts, bulges)
    chain.report["tangent_fillets"] = len(fillets)
    if fillets and _dense_distance(chain, p) > tolerance:
        # the exact tangent construction left the tolerance: use the plain junction chain
        pts2 = [p[0]]
        for k in range(len(prims) - 1):
            pts2.append(_junction(prims[k], prims[k + 1], p[prims[k]["j"]]))
        pts2.append(p[-1])
        pts2 = np.asarray(pts2)
        for k, prim in enumerate(prims):
            if prim["kind"] == "arc":
                bulges[k] = bulge_for(pts2[k], pts2[k + 1], prim["r"], prim["ccw"])
        chain = BulgeLine(pts2, bulges)
        chain.report["tangent_fillets"] = 0
    return chain


def _drop_degenerate(chain: BulgeLine, min_len: float = 1e-4) -> BulgeLine:
    keep_pts, keep_b = [chain.pts[0]], []
    for k, b in enumerate(chain.bulges):
        if float(np.hypot(*(chain.pts[k + 1] - keep_pts[-1]))) < min_len and k < len(chain.bulges) - 1:
            continue
        keep_pts.append(chain.pts[k + 1])
        keep_b.append(b)
    return BulgeLine(np.asarray(keep_pts), np.asarray(keep_b), chain.report)


def tangent_gap_deg(chain: BulgeLine) -> float:
    """Largest direction change (degrees below 45) at a vertex that joins a line and an arc or two arcs."""
    worst = 0.0

    def heading(k: int, at_end: bool) -> float:
        p0, p1, b = chain.pts[k], chain.pts[k + 1], float(chain.bulges[k])
        if abs(b) <= 1e-9:
            d = p1 - p0
            return math.atan2(d[1], d[0])
        centre, _r, a0, sweep = arc_geometry(p0, p1, b)
        ang = a0 + sweep if at_end else a0
        sign = 1.0 if sweep > 0 else -1.0
        return ang + sign * math.pi / 2.0

    for k in range(len(chain.bulges) - 1):
        if abs(chain.bulges[k]) <= 1e-9 and abs(chain.bulges[k + 1]) <= 1e-9:
            continue
        diff = abs(math.degrees((heading(k + 1, False) - heading(k, True) + math.pi) % (2 * math.pi) - math.pi))
        if diff < 45.0:  # larger changes are real corners, not meant to be tangent
            worst = max(worst, diff)
    return worst


def fit_chain(xy, *, closed: bool = False, free_start: bool = False, config: Mapping | None = None) -> BulgeLine:
    """Fit lines and circular arcs to the polyline ``xy`` (first and last vertex stay exactly).

    Greedy longest-run segmentation (a line while every sample stays within ``edges.fit.tolerance_m``,
    an arc while a circle with radius in ``[min_radius_m, max_radius_m]`` fits), then every primitive is
    refitted on its core samples and joined at intersections; ``line - arc - line`` becomes an exactly
    tangent fillet.  The result's ``report`` holds ``max_dev_m``, ``tangent_gap_deg`` and vertex counts;
    if the tolerance cannot be met the polyline falls back to straight segments.
    """
    cfg = fit_settings(config)
    raw = np.asarray(xy, dtype=float)[:, :2]
    raw = raw[np.r_[True, np.hypot(*np.diff(raw, axis=0).T) > _EPS]] if len(raw) > 1 else raw
    ring_start = 0
    if closed and free_start and len(raw) > 4:
        h = max(float(cfg["sample_m"]), 0.05)
        body = _resample(raw, h)[:-1]
        k = max(2, int(round(0.5 / h)))
        n = len(body)
        a = body - np.roll(body, k, axis=0)
        b = np.roll(body, -k, axis=0) - body
        turn = np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(axis=1)))
        ring_start = int(np.argmax(turn)) if n else 0
        body = np.roll(body, -ring_start, axis=0)
        raw = np.vstack((body, body[:1]))
    before = len(raw)
    if len(raw) < 3:
        chain = BulgeLine(raw.copy(), np.zeros(max(len(raw) - 1, 0)))
        chain.report.update(vertices_before=before, vertices_after=len(raw), max_dev_m=0.0, tangent_gap_deg=0.0)
        return chain
    tolerance = float(cfg["tolerance_m"])
    p = _resample(raw, float(cfg["sample_m"]))
    chain = None
    for factor in (0.8, 0.55, 0.35):
        runs = _Fitter(p, cfg, tolerance * factor).segment()
        candidate = _drop_degenerate(_build_chain(p, runs, cfg, tolerance))
        dev = _dense_distance(candidate, p)
        if dev <= tolerance:
            chain = candidate
            break
    fallback = chain is None
    if fallback:
        simplified = shapely.simplify(LineString(raw), tolerance * 0.5)
        pts = np.asarray(simplified.coords, dtype=float)
        pts[0], pts[-1] = raw[0], raw[-1]
        chain = BulgeLine(pts, np.zeros(len(pts) - 1))
        dev = _dense_distance(chain, p)
    chain.report.update(vertices_before=before, vertices_after=len(chain.pts), max_dev_m=round(dev, 5),
                        tangent_gap_deg=round(tangent_gap_deg(chain), 3), fallback=fallback)
    return chain


# ---------------------------------------------------------------------------------- partition fit


@dataclass
class FittedCurb:
    curb_id: object
    line: BulgeLine
    z: np.ndarray            # height at each vertex of ``line``
    dense_xyz: np.ndarray    # dense axis, arcs at sagitta <= edges.fit.sagitta_m
    shared: bool
    status: str = "ok"


@dataclass
class EdgeFit:
    """Result of :func:`fit_surfaces`."""

    regions: dict[str, list[list[Polygon]]]
    chains: list[BulgeLine]
    curbs: list[FittedCurb]
    report: dict
    sagitta: float = 0.01
    _tree: cKDTree | None = None
    _owner: np.ndarray | None = None
    _dense: list | None = None

    def _build_index(self) -> None:
        if self._tree is not None:
            return
        self._dense = [densify(c, self.sagitta) for c in self.chains]
        xy = [d[0] for d in self._dense]
        self._owner = np.concatenate([np.column_stack((np.full(len(d), k), np.arange(len(d)))) for k, d in enumerate(xy)]) if xy else np.empty((0, 2), int)
        self._tree = cKDTree(np.vstack(xy)) if xy else None

    def bulge_ring(self, ring_xy, tolerance: float = 6e-3) -> BulgeLine:
        """Re-express a (dense) ring that follows the fitted chains as lines and arcs."""
        ring = np.asarray(ring_xy, dtype=float)[:, :2]
        if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        self._build_index()
        plain = BulgeLine(np.vstack((ring, ring[:1])), np.zeros(len(ring)))
        if self._tree is None or len(ring) < 3:
            return plain
        dist, hit = self._tree.query(ring)
        keep = dist <= tolerance
        if keep.sum() < 3:
            return plain
        ring, hit = ring[keep], hit[keep]
        owner = self._owner[hit]
        change = np.r_[True, (owner[1:] != owner[:-1]).any(axis=1)] if len(owner) > 1 else np.array([True])
        ring, owner = ring[change], owner[change]
        count = len(ring)
        if count < 3:
            return plain
        edges: list[tuple] = []  # (start index, end index, kind key, signed sweep)
        for a in range(count):
            b = (a + 1) % count
            (ca, ia), (cb, ib) = owner[a], owner[b]
            key, sweep = None, 0.0
            if ca == cb:
                dense, seg = self._dense[ca]
                n = len(dense)
                closed = np.allclose(dense[0], dense[-1])
                step = ib - ia
                if closed and abs(step) == n - 1:
                    step = -1 if step > 0 else 1
                if abs(step) == 1:
                    s = int(seg[min(ia, ib)])
                    chain = self.chains[ca]
                    b_s = float(chain.bulges[s])
                    if abs(b_s) > 1e-9:
                        centre = arc_geometry(chain.pts[s], chain.pts[s + 1], b_s)[0]
                        va, vb = ring[a] - centre, ring[b] - centre
                        sweep = math.atan2(va[0] * vb[1] - va[1] * vb[0], float(va @ vb))
                        key = (int(ca), s, 1 if step > 0 else -1)
            edges.append((a, b, key, sweep))
        pts, bulges = [], []
        a = 0
        # start at an edge that is not the continuation of the previous one
        first = next((k for k in range(count) if edges[k][2] is None or edges[k - 1][2] != edges[k][2]), 0)
        order = [(first + k) % count for k in range(count)]
        k = 0
        while k < count:
            e = edges[order[k]]
            total, last = e[3], k
            if e[2] is not None:
                while last + 1 < count and edges[order[last + 1]][2] == e[2]:
                    last += 1
                    total += edges[order[last]][3]
            start = ring[e[0]]
            pts.append(start)
            bulges.append(math.tan(total / 4.0) if e[2] is not None and abs(total) > 1e-9 else 0.0)
            k = last + 1
        pts.append(pts[0])
        return BulgeLine(np.asarray(pts), np.asarray(bulges))


def _vertex_count(geom) -> int:
    return int(shapely.get_num_coordinates(geom))


def _explode(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [p for g in getattr(geom, "geoms", []) for p in _explode(g)]


def _curb_xyz(curb) -> np.ndarray:
    return np.asarray(getattr(curb, "points", curb), dtype=float)


def _interp_z(xyz: np.ndarray, xy: np.ndarray) -> np.ndarray:
    base = LineString(xyz[:, :2])
    if base.length <= _EPS or xyz.shape[1] < 3:
        return np.full(len(xy), float(xyz[0, 2]) if xyz.shape[1] > 2 else 0.0)
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xyz[:, :2], axis=0).T))]
    along = shapely.line_locate_point(base, shapely.points(xy))
    return np.interp(along, cum, xyz[:, 2])


def _match_curb(curb_xy: np.ndarray, chains: list[BulgeLine], dense: list[np.ndarray], match: float) -> BulgeLine | None:
    """Curb geometry taken from the outline chains it lies on, or ``None`` when it is not covered."""
    base = LineString(curb_xy)
    if base.length < 0.3:
        return None
    samples = shapely.line_interpolate_point(base, np.linspace(0.0, base.length, max(int(base.length / 0.1), 2) + 1))
    s = np.linspace(0.0, base.length, len(samples))
    pieces = []
    for k, chain in enumerate(chains):
        line = LineString(dense[k])
        near = shapely.distance(samples, line) <= match
        if near.sum() < 3:
            continue
        # longest contiguous run along the curb
        idx = np.flatnonzero(near)
        splits = np.flatnonzero(np.diff(idx) > 3) + 1
        runs = np.split(idx, splits)
        best = max(runs, key=len)
        if (best[-1] - best[0]) * 0.1 < 0.3:
            continue
        a = float(shapely.line_locate_point(line, samples[best[0]]))
        b = float(shapely.line_locate_point(line, samples[best[-1]]))
        scale = segment_lengths(chain).sum() / max(line.length, _EPS)
        part = sub_line(chain, min(a, b) * scale, max(a, b) * scale)
        if a > b:
            part = part.reversed()
        pieces.append((s[best[0]], s[best[-1]], part))
    if not pieces:
        return None
    pieces.sort(key=lambda t: t[0])
    covered = 0.0
    for lo, hi, _p in pieces:
        covered += hi - lo
    if covered < 0.9 * base.length:
        return None
    for (_lo1, _hi1, p1), (_lo2, _hi2, p2) in zip(pieces[:-1], pieces[1:]):
        if float(np.hypot(*(p1.pts[-1] - p2.pts[0]))) > 0.25:
            return None
    joined = join_lines([p for _lo, _hi, p in pieces])
    # pin the ends to the curb ends so the element keeps its length
    return joined


def prune_for_cad(line: BulgeLine, *, min_segment: float = 0.08, collinear: float = 0.015) -> BulgeLine:
    """Drop vertices that only add noise to a line/arc chain (for curb axes, qa_cad rule 5).

    A vertex goes when it joins two straight segments within ``collinear`` of their chord, or when
    it leaves a straight segment shorter than ``min_segment``.  Arcs are never changed.
    """
    pts, bulges = [p for p in line.pts], [float(b) for b in line.bulges]
    changed = True
    while changed and len(pts) > 2:
        changed = False
        for i in range(1, len(pts) - 1):
            a, b, cc = pts[i - 1], pts[i], pts[i + 1]
            if bulges[i - 1] != 0.0 or bulges[i] != 0.0:
                short = min(float(np.hypot(*(b - a))), float(np.hypot(*(cc - b)))) < min_segment
                if not short:
                    continue
                # a tiny straight piece next to an arc: absorb it into the arc's neighbour by moving the vertex
                if bulges[i - 1] == 0.0 and float(np.hypot(*(b - a))) < min_segment and i - 1 > 0:
                    del pts[i], bulges[i - 1]
                    changed = True
                    break
                continue
            chord = cc - a
            length = float(np.hypot(*chord))
            dev = float(np.hypot(*(b - a))) if length < 1e-12 else abs(chord[0] * (b - a)[1] - chord[1] * (b - a)[0]) / length
            if dev <= collinear or min(float(np.hypot(*(b - a))), float(np.hypot(*(cc - b)))) < min_segment:
                del pts[i], bulges[i]
                changed = True
                break
    return BulgeLine(np.asarray(pts), np.asarray(bulges), dict(line.report))


def fit_surfaces(
    regions: Mapping[str, Sequence[Polygon]],
    curbs: Sequence = (),
    *,
    config: Mapping | None = None,
) -> EdgeFit:
    """Fit all outline edges of a planar partition once, rebuild the polygons and fit the curbs.

    ``regions`` maps a class name (road, sidewalk, path, lawn) to its polygons.  The result's
    ``regions[name][i]`` lists the fitted polygons that replace input polygon ``i`` (several if it
    split, none if it vanished).  If rebuilding changes any class area by more than 2 % the input is
    returned unchanged (``report["fallback"]``).
    """
    cfg = fit_settings(config)
    sag = float(cfg["sagitta_m"])
    grid = 1e-3
    originals: list[tuple[str, int, Polygon]] = []
    for name, polys in regions.items():
        for i, poly in enumerate(polys):
            if poly is None or poly.is_empty:
                continue
            for part in _explode(make_valid(shapely.set_precision(poly, grid))):
                originals.append((name, i, part))
    identity = EdgeFit({n: [[p] if p is not None and not p.is_empty else [] for p in polys] for n, polys in regions.items()},
                       [], [], {"fallback": "no outlines"}, sag)
    if not originals:
        return identity
    boundary = shapely.union_all([p.boundary for _n, _i, p in originals])
    boundary = shapely.set_precision(boundary, grid)
    merged = linemerge(boundary)
    lines = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
    tree = shapely.STRtree(lines)
    chains: list[BulgeLine] = []
    for index, line in enumerate(lines):
        xy = np.asarray(line.coords, dtype=float)
        closed = len(xy) > 3 and np.allclose(xy[0], xy[-1])
        touching = len(tree.query(line, predicate="intersects")) if closed else 0
        chains.append(fit_chain(xy, closed=closed, free_start=closed and touching <= 1, config=config))
    dense = [densify(c, sag)[0] for c in chains]
    # node the fitted chains (fits of neighbouring chains may cross slightly), then build the faces
    noded = shapely.union_all([LineString(d) for d in dense])
    faces = [f for f in shapely.polygonize([noded]).geoms if f.area > 1e-6]
    originals_tree = shapely.STRtree([p for _n, _i, p in originals])
    fitted: dict[str, list[list[Polygon]]] = {n: [[] for _ in polys] for n, polys in regions.items()}
    area_new: dict[str, float] = {n: 0.0 for n in regions}
    area_old: dict[str, float] = {n: 0.0 for n in regions}
    for name, _i, poly in originals:
        area_old[name] += poly.area
    for face in faces:
        rp = face.representative_point()
        hits = originals_tree.query(rp, predicate="within")
        if len(hits) == 0:
            continue
        name, i, _p = originals[int(hits[0])]
        fitted[name][i].append(face)
        area_new[name] += face.area
    worst = max((abs(area_new[n] - area_old[n]) / area_old[n] for n in area_old if area_old[n] > 0), default=0.0)
    if worst > 0.02:
        LOGGER.warning("edge fitting changed a class area by %.1f %%; keeping the unfitted outlines", worst * 100)
        identity.report = {"fallback": f"area change {worst * 100:.1f} %",
                           "area_old": {k: round(v, 2) for k, v in area_old.items()}, "area_new": {k: round(v, 2) for k, v in area_new.items()},
                           "faces": len(faces), "chains": len(chains)}
        return identity
    result = EdgeFit(fitted, chains, [], {}, sag)
    result.report = _report(regions, fitted, chains, dense)
    result.curbs = _fit_curbs(curbs, chains, dense, cfg, config)
    result.report["curbs"] = {"count": len(result.curbs), "shared": sum(c.shared for c in result.curbs),
                              "independent": sum(not c.shared for c in result.curbs),
                              "arcs": int(sum(c.line.arcs for c in result.curbs)),
                              "lines": int(sum(c.line.lines for c in result.curbs)),
                              "max_dev_m": round(max((c.line.report.get("max_dev_m", 0.0) for c in result.curbs), default=0.0), 4)}
    return result


def _report(regions, fitted, chains, dense) -> dict:
    per_class: dict[str, dict] = {}
    for name, polys in regions.items():
        before = sum(_vertex_count(p) for p in polys if p is not None and not p.is_empty)
        faces = [f for group in fitted[name] for f in group]
        after = sum(_vertex_count(f) for f in faces)
        per_class[name] = {"polygons": len(faces), "vertices_before": int(before), "vertices_after_dense": int(after)}
    arcs, lines, devs, gaps = 0, 0, [], []
    for chain in chains:
        arcs += chain.arcs
        lines += chain.lines
        devs.append(chain.report.get("max_dev_m", 0.0))
        gaps.append(chain.report.get("tangent_gap_deg", 0.0))
    return {"classes": per_class, "chains": len(chains), "arcs": arcs, "lines": lines,
            "vertices_fitted": int(sum(len(c.pts) for c in chains)),
            "max_dev_m": round(float(max(devs, default=0.0)), 4),
            "max_tangent_gap_deg": round(float(max(gaps, default=0.0)), 3),
            "fallback_chains": int(sum(bool(c.report.get("fallback")) for c in chains))}


def _fit_curbs(curbs: Sequence, chains: list[BulgeLine], dense: list[np.ndarray], cfg: dict, config) -> list[FittedCurb]:
    out: list[FittedCurb] = []
    sag = float(cfg["sagitta_m"])
    for number, curb in enumerate(curbs):
        xyz = _curb_xyz(curb)
        if xyz.ndim != 2 or len(xyz) < 2:
            continue
        shared_line = _match_curb(xyz[:, :2], chains, dense, float(cfg["curb_match_m"])) if chains else None
        if shared_line is not None:
            line, shared = shared_line, True
            line.report.update(max_dev_m=round(_dense_distance(line, _resample(xyz[:, :2], 0.1)), 5))
        else:
            line, shared = fit_chain(xyz[:, :2], config=config), False
        pruned = prune_for_cad(line)
        if _dense_distance(pruned, _resample(xyz[:, :2], 0.1)) <= max(line.report.get("max_dev_m", 0.0), float(cfg["tolerance_m"])) + 1e-9:
            pruned.report = dict(line.report)
            line = pruned
        pts_dense, _seg = densify(line, sag)
        z = _interp_z(xyz, pts_dense)
        vz = _interp_z(xyz, line.pts)
        out.append(FittedCurb(getattr(curb, "curb_id", number), line, vz, np.column_stack((pts_dense, z)), shared,
                              str(getattr(curb, "status", "ok"))))
    return out


__all__ = ["prune_for_cad", "BulgeLine", "EdgeFit", "FittedCurb", "arc_geometry", "bulge_for", "densify", "fit_chain",
           "fit_settings", "fit_surfaces", "join_lines", "sub_line", "tangent_gap_deg"]
