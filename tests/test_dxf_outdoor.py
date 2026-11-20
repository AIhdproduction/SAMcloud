"""Outdoor DXF extensions: layers, symbol blocks, new entity types, local origin, TIN guard."""

from __future__ import annotations

from pathlib import Path

import ezdxf
import numpy as np
import pytest

from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import BackgroundImage, DxfPlanResult
from samcloud.dxf.writer import (
    assert_no_tin_outside,
    build_dxf_document,
    contours_as_splines,
    ensure_outdoor_layers,
    outdoor_symbol_insert,
)

NEW_LAYERS = ("S-BLDG-ROOF", "S-BLDG-WIND", "S-BLDG-DOOR", "S-SHRUB", "S-PATH", "S-TOPO-TIN", "S-TOPO-MAJR", "S-TOPO-MINR", "S-ELEV", "S-ANNO-GRENZE")


def _result(layers: dict[str, list[dict]], bounds=(0.0, 0.0, 10.0, 10.0)) -> DxfPlanResult:
    return DxfPlanResult(
        "outdoor", "site", "outdoor", None, 1.0, Frame(np.zeros(3)), "test",
        layers, {}, dict.fromkeys(layers, 7), BackgroundImage(None, None, bounds),
    )


def _roundtrip(document: ezdxf.document.Drawing, tmp_path: Path) -> ezdxf.document.Drawing:
    path = tmp_path / "out.dxf"
    document.saveas(path)
    return ezdxf.readfile(path)


def _attribs(insert) -> dict[str, str]:
    return {attribute.dxf.tag: attribute.dxf.text for attribute in insert.attribs}


def test_new_layers_exist_and_tin_is_frozen() -> None:
    document = ezdxf.new("R2018")
    created = ensure_outdoor_layers(document)
    assert set(created) == set(NEW_LAYERS)
    for name in NEW_LAYERS:
        assert name in document.layers
    assert document.layers.get("S-TOPO-TIN").is_frozen()
    assert not document.layers.get("S-TOPO-TIN").is_on()
    assert not document.layers.get("S-PATH").is_frozen()
    assert document.layers.get("S-SHRUB").dxf.color == 94
    other = ezdxf.new("R2018")
    ensure_outdoor_layers(other, tin_frozen=False)
    assert not other.layers.get("S-TOPO-TIN").is_frozen()


def test_topo_and_elev_layers_get_yaml_style() -> None:
    document = ezdxf.new("R2018")
    ensure_outdoor_layers(document, {"S-TOPO-MAJR", "S-TOPO-MINR", "S-ELEV"})
    expected = {"S-TOPO-MAJR": (30, 25), "S-TOPO-MINR": (74, 13), "S-ELEV": (3, 18)}
    for name, (color, weight) in expected.items():
        layer = document.layers.get(name)
        assert layer.dxf.color == color and layer.dxf.lineweight == weight
        assert not layer.is_frozen() and layer.is_on()


def test_tree_insert_attributes_and_scale(tmp_path: Path) -> None:
    full = outdoor_symbol_insert("tree", (5.0, 5.0, 1.2), {"HEIGHT": 8.4, "CROWN_DIA": 5.0, "TRUNK_DIA": 0.35}, crown_radius=2.5)
    partial = outdoor_symbol_insert("TREE", (7.0, 7.0, 1.0), {"HEIGHT": 6.0}, crown_radius=None)
    document = build_dxf_document(_result({"S-TREE": [full, partial]}))
    assert "TREE" in document.blocks
    loaded = _roundtrip(document, tmp_path)
    inserts = [entity for entity in loaded.modelspace() if entity.dxftype() == "INSERT"]
    assert len(inserts) == 2
    first, second = inserts
    assert _attribs(first) == {"HEIGHT": "8.40", "CROWN_DIA": "5.00", "TRUNK_DIA": "0.35"}
    assert first.dxf.xscale == pytest.approx(2.5) and first.dxf.yscale == pytest.approx(2.5)
    assert _attribs(second) == {"HEIGHT": "6.00", "CROWN_DIA": "n/a", "TRUNK_DIA": "n/a"}
    assert second.dxf.xscale == pytest.approx(1.0)
    assert {"S-TREE"} == {entity.dxf.layer for entity in inserts}


def test_manhole_insert_has_cover_height_and_type(tmp_path: Path) -> None:
    manhole = outdoor_symbol_insert("manhole", (3.0, 4.0, 431.256), {"Z_DECKEL": 431.256, "TYP": "Schacht"})
    unknown = outdoor_symbol_insert("MANHOLE", (6.0, 4.0, 0.0), {"Z_DECKEL": None})
    shrub = outdoor_symbol_insert("shrub", (1.0, 1.0, 0.0), {}, crown_radius=0.8)
    document = build_dxf_document(_result({"S-MANH": [manhole, unknown], "S-SHRUB": [shrub]}))
    loaded = _roundtrip(document, tmp_path)
    inserts = {entity.dxf.handle: entity for entity in loaded.modelspace() if entity.dxftype() == "INSERT"}
    attribute_sets = [_attribs(entity) for entity in inserts.values()]
    assert {"Z_DECKEL": "431.26", "TYP": "Schacht"} in attribute_sets
    assert {"Z_DECKEL": "n/a", "TYP": "n/a"} in attribute_sets
    assert {"HEIGHT": "n/a"} in attribute_sets
    assert "MANHOLE" in loaded.blocks and "SHRUB" in loaded.blocks
    assert "S-SHRUB" in loaded.layers


def test_closed_lwpolyline_with_elevation_round_trip(tmp_path: Path) -> None:
    entity = {"type": "lwpolyline", "closed": True, "elevation": 432.5, "points": [[0.0, 0.0], [4.1, 0.3], [4.0, 3.7], [0.2, 3.9]]}
    document = build_dxf_document(_result({"S-BLDG-ROOF": [entity]}))
    loaded = _roundtrip(document, tmp_path)
    polyline = next(e for e in loaded.modelspace() if e.dxftype() == "LWPOLYLINE")
    assert polyline.closed
    assert polyline.dxf.elevation == pytest.approx(432.5)
    assert list(polyline.get_points("xy"))[1] == pytest.approx((4.1, 0.3))
    assert polyline.dxf.layer == "S-BLDG-ROOF"
    assert not [e for e in loaded.modelspace() if e.dxftype() == "HATCH"]
    hatched = build_dxf_document(_result({"S-BLDG-ROOF": [{**entity, "hatch": True}]}))
    assert [e for e in hatched.modelspace() if e.dxftype() == "HATCH"]


def test_spline_conversion_keeps_z_and_closedness(tmp_path: Path) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 9)
    ring = [[float(5 + 2 * np.cos(a)), float(5 + 2 * np.sin(a)), 431.37] for a in angles[:-1]]
    closed = {"type": "polyline3d", "layer": "S-TOPO-MAJR", "closed": True, "points": [*ring, ring[0]], "elevation": 431.0}
    open_line = {"type": "polyline3d", "layer": "S-TOPO-MINR", "closed": False, "points": [[0.0, 0.1, 430.5], [1.0, 0.7, 430.5], [2.1, 0.9, 430.5]]}
    converted = contours_as_splines([closed, open_line, {"type": "line", "start": [0, 0], "end": [1, 1]}])
    assert converted[0]["type"] == "spline" and converted[0]["closed"] is True
    assert converted[0]["layer"] == "S-TOPO-MAJR" and converted[0]["elevation"] == 431.0
    assert len(converted[0]["fit_points"]) == len(ring)
    assert converted[1]["closed"] is False and converted[2]["type"] == "line"
    document = build_dxf_document(_result({"S-TOPO-MAJR": [converted[0]], "S-TOPO-MINR": [converted[1]]}))
    loaded = _roundtrip(document, tmp_path)
    splines = {e.dxf.layer: e for e in loaded.modelspace() if e.dxftype() == "SPLINE"}
    assert splines["S-TOPO-MAJR"].closed and not splines["S-TOPO-MINR"].closed
    assert all(point[2] == pytest.approx(431.37) for point in splines["S-TOPO-MAJR"].fit_points)
    assert all(point[2] == pytest.approx(430.5) for point in splines["S-TOPO-MINR"].fit_points)


def test_3dface_only_allowed_on_tin_layer(tmp_path: Path) -> None:
    good = {"type": "3dface", "points": [[0.0, 0.0, 1.1], [2.0, 0.1, 1.3], [1.0, 1.9, 1.7]]}
    bad = {"type": "3dface", "points": [[5.0, 5.0, 2.1], [7.0, 5.1, 2.3], [6.0, 6.9, 2.7]]}
    clean = build_dxf_document(_result({"S-TOPO-TIN": [good]}))
    assert assert_no_tin_outside(clean) == []
    assert clean.layers.get("S-TOPO-TIN").is_frozen()
    dirty = _roundtrip(build_dxf_document(_result({"S-TOPO-TIN": [good], "S-BLDG-ROOF": [bad]})), tmp_path)
    offenders = assert_no_tin_outside(dirty)
    assert len(offenders) == 1 and offenders[0].dxf.layer == "S-BLDG-ROOF"
    assert len(assert_no_tin_outside(dirty, allowed_layer="S-BLDG-ROOF")) == 1


def test_lv95_precision_round_trip(tmp_path: Path) -> None:
    east, north, height = 2_680_123.456, 1_230_456.789, 432.109
    layers = {
        "S-MARK": [{"type": "point", "point": [east, north, height]}],
        "S-PATH": [{"type": "lwpolyline", "closed": True, "elevation": height, "points": [[east, north], [east + 3.3, north + 0.4], [east + 2.9, north + 4.7]]}],
    }
    loaded = _roundtrip(build_dxf_document(_result(layers, bounds=(east, north, east + 10.0, north + 10.0))), tmp_path)
    point = next(e for e in loaded.modelspace() if e.dxftype() == "POINT")
    assert point.dxf.location.x == pytest.approx(east, abs=1e-6)
    assert point.dxf.location.y == pytest.approx(north, abs=1e-6)
    assert point.dxf.location.z == pytest.approx(height, abs=1e-6)
    polyline = next(e for e in loaded.modelspace() if e.dxftype() == "LWPOLYLINE")
    assert list(polyline.get_points("xy"))[0] == pytest.approx((east, north), abs=1e-6)


def test_local_origin_offsets_coordinates_and_records_offset(tmp_path: Path) -> None:
    east, north = 2_680_123.456, 1_230_456.789
    layers = {
        "S-MARK": [{"type": "point", "point": [east, north, 432.109]}],
        "S-TREE": [outdoor_symbol_insert("tree", (east + 2.0, north + 1.5, 431.0), {"HEIGHT": 7.0}, crown_radius=2.0)],
    }
    result = _result(layers, bounds=(east - 5.2, north - 3.7, east + 10.0, north + 10.0))
    default = build_dxf_document(result)
    assert next(e for e in default.modelspace() if e.dxftype() == "POINT").dxf.location.x == pytest.approx(east, abs=1e-6)
    shifted = _roundtrip(build_dxf_document(result, local_origin=True), tmp_path)
    offset = tuple(shifted.header["$INSBASE"])
    assert offset == pytest.approx((np.floor(east - 5.2), np.floor(north - 3.7), 0.0), abs=1e-9)
    point = next(e for e in shifted.modelspace() if e.dxftype() == "POINT")
    assert point.dxf.location.x + offset[0] == pytest.approx(east, abs=1e-6)
    assert point.dxf.location.y + offset[1] == pytest.approx(north, abs=1e-6)
    assert abs(point.dxf.location.x) < 100.0
    insert = next(e for e in shifted.modelspace() if e.dxftype() == "INSERT")
    assert insert.dxf.insert.x + offset[0] == pytest.approx(east + 2.0, abs=1e-6)
    note = next(e for e in shifted.modelspace() if e.dxftype() == "TEXT" and e.dxf.layer == "X-NOTE")
    assert "local origin offset" in note.dxf.text
    explicit = build_dxf_document(result, local_origin=True, origin_offset=(east, north, 400.0))
    assert tuple(explicit.header["$INSBASE"]) == pytest.approx((east, north, 400.0))
    moved = next(e for e in explicit.modelspace() if e.dxftype() == "POINT")
    assert moved.dxf.location.z == pytest.approx(32.109, abs=1e-6)
