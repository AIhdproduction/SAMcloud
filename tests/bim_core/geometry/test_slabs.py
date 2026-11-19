"""Slab and ceiling geometry."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from samcloud.bim.core.geometry import compute_ceiling_geometry, compute_slab_geometry
from samcloud.bim.core.model import CeilingElement, CeilingParams, SlabElement, SlabParams

OUTLINE = [(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (0.0, 5.0)]


def test_slab_with_hole() -> None:
    slab = SlabElement(
        params=SlabParams(
            outline=OUTLINE,
            thickness=0.25,
            top_elevation=0.0,
            openings=[[(1.0, 1.0), (2.0, 1.0), (2.0, 2.5), (1.0, 2.5)]],
        )
    )
    g = compute_slab_geometry(slab)
    assert g.gross_area == pytest.approx(30.0)
    assert g.net_area == pytest.approx(30.0 - 1.5)
    assert g.perimeter == pytest.approx(22.0)
    assert g.top_z == 0.0
    assert g.bottom_z == pytest.approx(-0.25)
    q = g.quantities
    assert q["Width"] == pytest.approx(0.25)
    assert q["GrossVolume"] == pytest.approx(7.5)
    assert q["NetVolume"] == pytest.approx(28.5 * 0.25)
    assert g.issues == ()
    assert g.polygon().area == pytest.approx(28.5)


def test_slab_orientation_is_normalized_and_bad_hole_reported() -> None:
    slab = SlabElement(
        params=SlabParams(
            outline=list(reversed(OUTLINE)),
            thickness=0.2,
            top_elevation=1.0,
            openings=[[(5.0, 4.0), (7.0, 4.0), (7.0, 6.0), (5.0, 6.0)]],
        )
    )
    g = compute_slab_geometry(slab)
    assert Polygon(g.outline).exterior.is_ccw
    assert any("not inside the slab outline" in m for m in g.issues)
    assert g.net_area == pytest.approx(30.0 - 1.0)


def test_flat_ceiling() -> None:
    ceiling = CeilingElement(params=CeilingParams(outline=OUTLINE, elevation=2.7, thickness=0.1))
    g = compute_ceiling_geometry(ceiling)
    assert g.bottom_z == pytest.approx(2.7)
    assert g.top_z == pytest.approx(2.8)
    assert g.gross_area == pytest.approx(30.0)
    assert g.quantities["GrossVolume"] == pytest.approx(3.0)
    assert g.bottom_z_at(3.0, 2.0) == 2.7


def test_ceiling_with_tilted_plane() -> None:
    # plane z = 2.5 + 0.1 x  ->  0.1 x - z = -2.5  ->  n = (0.1, 0, -1), d = -2.5
    ceiling = CeilingElement(
        params=CeilingParams(
            outline=OUTLINE,
            elevation=0.0,
            plane_normal=(0.1, 0.0, -1.0),
            plane_offset=-2.5,
            sag=0.02,
            thickness=0.1,
        )
    )
    g = compute_ceiling_geometry(ceiling)
    assert g.outline_z == pytest.approx((2.5, 3.1, 3.1, 2.5), abs=1e-12)
    assert g.bottom_z == pytest.approx(2.5)
    assert g.top_z == pytest.approx(3.2)
    assert g.bottom_z_at(3.0, 4.0) == pytest.approx(2.8, abs=1e-12)
    assert g.sag == 0.02
    assert g.issues == ()


def test_ceiling_vertical_plane_falls_back_to_elevation() -> None:
    ceiling = CeilingElement(
        params=CeilingParams(
            outline=OUTLINE, elevation=2.6, plane_normal=(1.0, 0.0, 0.0), plane_offset=1.0
        )
    )
    g = compute_ceiling_geometry(ceiling)
    assert g.bottom_z == pytest.approx(2.6)
    assert "vertical" in g.issues[0]
