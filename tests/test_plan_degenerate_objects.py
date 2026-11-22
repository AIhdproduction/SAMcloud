"""Degenerate furniture shapes never crash the clean plan (synthetic-bench crashes, 2026-11-21)."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box


def test_drop_short_edges_on_a_polygon_smaller_than_the_minimum_edge() -> None:
    from samcloud.ortho.vector.footprints import _drop_short_edges

    tiny = Polygon([(0.0, 0.0), (0.013, 0.0), (0.013, 0.011), (0.0, 0.011)])
    assert _drop_short_edges(tiny, 0.04) is None  # was: ValueError (polygon from < 3 points)
    triangle = Polygon([(0.0, 0.0), (0.02, 0.0), (0.0, 0.5)])
    assert _drop_short_edges(triangle, 0.04) is None


def test_drop_short_edges_keeps_a_normal_polygon_and_removes_a_short_edge() -> None:
    from samcloud.ortho.vector.footprints import _drop_short_edges

    shape = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 0.6), (0.99, 0.61), (0.0, 0.6)])
    cleaned = _drop_short_edges(shape, 0.04)
    assert cleaned is not None and len(cleaned.exterior.coords) - 1 == 4
    assert cleaned.area == pytest.approx(shape.area, rel=0.02)


def test_regularised_returns_none_for_a_degenerate_trace() -> None:
    from samcloud.ortho.vector.footprints import _regularised

    sliver = Polygon([(0.0, 0.0), (0.03, 0.0), (0.03, 0.025), (0.0, 0.025)])
    assert _regularised(sliver, 0.0, 1.0) is None or _regularised(sliver, 0.0, 1.0).is_valid


def test_merge_lines_handles_zero_one_and_many_pieces() -> None:
    from samcloud.ortho.vector.furniture_shapes import _merge_lines

    assert _merge_lines([]).is_empty
    single = _merge_lines([LineString([(0.0, 0.0), (1.0, 0.0)])])
    assert isinstance(single, LineString) and single.length == pytest.approx(1.0)
    joined = _merge_lines([LineString([(0.0, 0.0), (1.0, 0.0)]), LineString([(1.0, 0.0), (1.0, 1.0)])])
    assert joined.length == pytest.approx(2.0) and joined.geom_type == "LineString"


def test_sofa_with_a_single_backrest_line_does_not_crash() -> None:
    """One raised edge gives ONE seat line: ``linemerge`` used to reject that single LineString."""
    from samcloud.ortho.vector.furniture_shapes import sofa_details

    shape = box(0.0, 0.0, 2.0, 0.9)

    def height_of(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return np.where(points[:, 1] >= 0.68, 0.80, 0.42)

    detail = sofa_details(shape, 0.0, 1.0, height_of)
    assert detail.raised_edges == 1
    assert len(detail.lines) == 1
    line = detail.lines[0]
    assert abs(line.coords[0][1] - line.coords[-1][1]) < 1e-6  # the backrest line runs along the long edge
    assert detail.seat is not None and detail.seat.area < shape.area


def test_bridge_barriers_survives_a_geos_topology_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GEOS 'Ring edge missing' in the stub overlay (seen on v12 at 1.0 units per metre) retries snapped, never raises."""
    from shapely.errors import GEOSException

    from samcloud.ortho.vector import plan_polish

    door, window = box(0.0, 0.0, 0.9, 0.2), box(1.15, 0.0, 2.15, 0.2)  # 25 cm jamb between two spans of one wall
    expected = plan_polish.bridge_barriers([door, window], 0.30, 0.20)
    assert len(expected) == 1 and expected[0].area == pytest.approx(0.25 * 0.2, rel=1e-6)

    real_union = plan_polish.unary_union
    calls = {"n": 0}

    def flaky_union(geometries: object) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise GEOSException("TopologyException: Ring edge missing at 0.9 0.2")
        return real_union(geometries)

    monkeypatch.setattr(plan_polish, "unary_union", flaky_union)
    retried = plan_polish.bridge_barriers([door, window], 0.30, 0.20)
    assert len(retried) == 1 and retried[0].area == pytest.approx(expected[0].area, rel=1e-3)

    def broken_union(geometries: object) -> object:
        raise GEOSException("TopologyException: Ring edge missing")

    monkeypatch.setattr(plan_polish, "unary_union", broken_union)
    assert plan_polish.bridge_barriers([door, window], 0.30, 0.20) == []  # skipped, not raised


def test_object_step_skips_a_failing_object_and_counts_it(caplog: pytest.LogCaptureFixture) -> None:
    from samcloud.dxf.cancel import Cancelled
    from samcloud.ortho.plan import _object_step

    failures: dict[str, int] = {}

    def boom() -> None:
        raise ValueError("A linearring requires at least 4 coordinates.")

    with caplog.at_level(logging.WARNING):
        assert _object_step(failures, "fit_footprint", "sofa", boom) is None
        assert _object_step(failures, "solid_parts", "sofa", boom, []) == []
        assert _object_step(failures, "fit_footprint", "chair", lambda: 7) == 7
    assert failures == {"fit_footprint": 1, "solid_parts": 1}
    assert "object skipped" in caplog.text

    def cancelled() -> None:
        raise Cancelled("stop")

    with pytest.raises(Cancelled):
        _object_step(failures, "fit_footprint", "sofa", cancelled)
