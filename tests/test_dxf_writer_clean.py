"""Unit and behavioural tests for DXF writer, result, and layers extensions."""

from __future__ import annotations

import os
from pathlib import Path
import uuid

import ezdxf
from ezdxf.enums import TextEntityAlignment
import numpy as np
import pytest

from samcloud.dxf.las_source import Frame
from samcloud.dxf.result import (
    BackgroundImage,
    DxfPlanResult,
    LayerStatus,
    result_from_dict,
    result_to_dict,
    vector_payload,
    visible_entity_indices,
)
from samcloud.dxf.writer import export_plan


@pytest.fixture
def work_dir(request: pytest.FixtureRequest) -> Path:
    """Return an isolated directory for this test session."""
    configured_root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    if configured_root:
        path = Path(configured_root) / f"dxf_clean_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return request.getfixturevalue("tmp_path_factory").mktemp("dxf_clean")


def _make_result(
    *,
    layers: dict[str, list[dict]] | None = None,
    layer_colors: dict[str, int] | None = None,
    layer_linetypes: dict[str, str] | None = None,
    layer_hatches: dict[str, bool] | None = None,
    layer_styles: dict[str, dict] | None = None,
    blocks: list[str] | None = None,
    variant: str = "pointcloud",
    frame: Frame | None = None,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
    project_name: str = "TestProject",
    level_name: str = "EG",
) -> DxfPlanResult:
    if layers is None:
        layers = {}
    if layer_colors is None:
        layer_colors = {name: 7 for name in layers}
    if frame is None:
        frame = Frame(np.zeros(3, dtype=np.float64), 1.0, np.eye(3, dtype=np.float64))
    statuses = {
        name: LayerStatus(name, "ok", None, len(entities))
        for name, entities in layers.items()
    }
    background = BackgroundImage(None, None, bounds)
    return DxfPlanResult(
        project_name=project_name,
        level_name=level_name,
        mode="indoor",
        profile_name="test_profile",
        scale_factor=frame.scale_factor,
        frame=frame,
        settings_fingerprint="fingerprint_test",
        layers=layers,
        statuses=statuses,
        layer_colors=layer_colors,
        background=background,
        fingerprint_inputs={},
        layer_linetypes=layer_linetypes or {},
        layer_hatches=layer_hatches or {},
        layer_styles=layer_styles or {},
        blocks=blocks or [],
        variant=variant,
    )


def test_layer_true_color_and_lineweight_written(work_dir: Path) -> None:
    result = _make_result(
        layers={"S-TOPO-MAJR": [{"type": "line", "start": [0.0, 0.0], "end": [1.0, 1.0]}]},
        layer_colors={"S-TOPO-MAJR": 30},
        layer_styles={"S-TOPO-MAJR": {"true_color": [150, 60, 40], "lineweight": 35}},
    )
    dxf_path = export_plan(result, work_dir, "local")
    doc = ezdxf.readfile(dxf_path)
    layer = doc.layers.get("S-TOPO-MAJR")
    assert layer.rgb == (150, 60, 40)
    assert layer.dxf.lineweight == 35


def test_polyline3d_written_with_z(work_dir: Path) -> None:
    frame = Frame(np.array([2_600_000.0, 1_200_000.0, 450.0], dtype=np.float64), 1.0, np.eye(3, dtype=np.float64))
    entity = {
        "type": "polyline3d",
        "points": [[10.0, 20.0, 5.0], [15.0, 25.0, 6.0]],
        "closed": False,
    }
    result = _make_result(
        layers={"S-TOPO-MAJR": [entity]},
        layer_colors={"S-TOPO-MAJR": 30},
        frame=frame,
    )

    # Local mode: coordinates and z preserved as given
    local_path = export_plan(result, work_dir / "local", "local")
    local_doc = ezdxf.readfile(local_path)
    local_poly = next(e for e in local_doc.modelspace() if e.dxftype() == "POLYLINE")
    assert local_poly.is_3d_polyline
    local_pts = list(local_poly.points())
    assert local_pts[0].x == pytest.approx(10.0)
    assert local_pts[0].y == pytest.approx(20.0)
    assert local_pts[0].z == pytest.approx(5.0)
    assert local_pts[1].x == pytest.approx(15.0)
    assert local_pts[1].y == pytest.approx(25.0)
    assert local_pts[1].z == pytest.approx(6.0)

    # Original mode: Frame origin offset (2.6e6, 1.2e6, 450) applied
    orig_path = export_plan(result, work_dir / "orig", "original")
    orig_doc = ezdxf.readfile(orig_path)
    orig_poly = next(e for e in orig_doc.modelspace() if e.dxftype() == "POLYLINE")
    assert orig_poly.is_3d_polyline
    orig_pts = list(orig_poly.points())
    assert orig_pts[0].x == pytest.approx(10.0 + 2_600_000.0)
    assert orig_pts[0].y == pytest.approx(20.0 + 1_200_000.0)
    assert orig_pts[0].z == pytest.approx(5.0 + 450.0)
    assert orig_pts[1].x == pytest.approx(15.0 + 2_600_000.0)
    assert orig_pts[1].y == pytest.approx(25.0 + 1_200_000.0)
    assert orig_pts[1].z == pytest.approx(6.0 + 450.0)


def test_text_alignment_height_rotation(work_dir: Path) -> None:
    entity = {
        "type": "text",
        "text": "Room 101",
        "point": [12.0, 34.0],
        "height": 0.35,
        "rotation": 45.0,
        "align": "MIDDLE_CENTER",
    }
    result = _make_result(
        layers={"A-ANNO": [entity]},
        layer_colors={"A-ANNO": 1},
    )
    dxf_path = export_plan(result, work_dir, "local")
    doc = ezdxf.readfile(dxf_path)
    text = next(e for e in doc.modelspace() if e.dxftype() == "TEXT" and e.dxf.layer == "A-ANNO")
    assert text.dxf.text == "Room 101"
    assert text.dxf.height == pytest.approx(0.35)
    assert text.dxf.rotation == pytest.approx(45.0)
    align, p1, _p2 = text.get_placement()
    assert align == TextEntityAlignment.MIDDLE_CENTER
    assert p1.x == pytest.approx(12.0)
    assert p1.y == pytest.approx(34.0)


def test_symbol_blocks_defined_and_door_insert_readback(work_dir: Path) -> None:
    door_entity = {
        "type": "insert",
        "name": "DOOR",
        "point": [2.5, 3.5],
        "xscale": 0.85,
        "yscale": 0.85,
        "rotation": 90.0,
    }
    result = _make_result(
        layers={"A-DOOR": [door_entity]},
        layer_colors={"A-DOOR": 2},
        blocks=["DOOR", "WINDOW", "TREE"],
    )
    dxf_path = export_plan(result, work_dir, "local")
    doc = ezdxf.readfile(dxf_path)
    assert "DOOR" in doc.blocks
    assert "WINDOW" in doc.blocks
    assert "TREE" in doc.blocks
    insert = next(e for e in doc.modelspace() if e.dxftype() == "INSERT" and e.dxf.name == "DOOR")
    assert insert.dxf.xscale == pytest.approx(0.85)
    assert insert.dxf.yscale == pytest.approx(0.85)
    assert insert.dxf.rotation == pytest.approx(90.0)


def test_original_coordinates_scale_and_rotation(work_dir: Path) -> None:
    theta = np.radians(30.0)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    R = np.array([
        [cos_t, sin_t, 0.0],
        [-sin_t, cos_t, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    origin = np.array([500.0, 600.0, 100.0], dtype=np.float64)
    frame = Frame(origin, 2.0, R)

    # Derive the local-x yaw independently through Frame's public transform.
    original_origin = frame.to_original(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))[0]
    original_x = frame.to_original(np.array([[1.0, 0.0, 0.0]], dtype=np.float64))[0]
    expected_yaw = float(np.degrees(np.arctan2(*(original_x - original_origin)[1::-1])))
    assert expected_yaw == pytest.approx(30.0)

    insert_entity = {
        "type": "insert",
        "name": "DOOR",
        "point": [10.0, 20.0],
        "xscale": 0.9,
        "yscale": 0.9,
        "rotation": 15.0,
    }
    text_entity = {
        "type": "text",
        "text": "Label",
        "point": [15.0, 25.0],
        "height": 0.4,
        "rotation": 20.0,
        "align": "MIDDLE_CENTER",
    }
    circle_entity = {
        "type": "circle",
        "center": [30.0, 40.0],
        "radius": 1.6,
    }
    result = _make_result(
        layers={"A-SYMB": [insert_entity, text_entity, circle_entity]},
        layer_colors={"A-SYMB": 4},
        blocks=["DOOR"],
        frame=frame,
    )

    dxf_path = export_plan(result, work_dir, "original")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Insert: xscale halves (0.9 -> 0.45), rotation increases by expected_yaw
    insert = next(e for e in msp if e.dxftype() == "INSERT")
    assert insert.dxf.xscale == pytest.approx(0.45)
    assert insert.dxf.rotation == pytest.approx(15.0 + expected_yaw)

    # Text: height halves (0.4 -> 0.2), rotation increases by expected_yaw
    text = next(e for e in msp if e.dxftype() == "TEXT" and e.dxf.layer == "A-SYMB")
    assert text.dxf.height == pytest.approx(0.2)
    assert text.dxf.rotation == pytest.approx(20.0 + expected_yaw)

    # Circle: radius halves (1.6 -> 0.8)
    circle = next(e for e in msp if e.dxftype() == "CIRCLE")
    assert circle.dxf.radius == pytest.approx(0.8)


def test_hatch_pattern_from_layer_styles(work_dir: Path) -> None:
    poly_entity = {
        "type": "polyline",
        "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        "closed": True,
    }
    result = _make_result(
        layers={"A-WALL": [poly_entity]},
        layer_colors={"A-WALL": 7},
        layer_hatches={"A-WALL": True},
        layer_styles={"A-WALL": {"hatch_pattern": "ANSI31", "hatch_scale": 0.5}},
    )
    dxf_path = export_plan(result, work_dir, "local")
    doc = ezdxf.readfile(dxf_path)
    hatch = next(e for e in doc.modelspace() if e.dxftype() == "HATCH")
    assert hatch.dxf.pattern_name == "ANSI31"
    assert hatch.dxf.pattern_scale == pytest.approx(0.5)
    assert hatch.dxf.solid_fill == 0


def test_old_result_dict_without_new_fields_loads(work_dir: Path) -> None:
    original_result = _make_result(
        layers={"A-WALL": [{"type": "line", "start": [0.0, 0.0], "end": [5.0, 5.0]}]},
        layer_colors={"A-WALL": 7},
    )
    payload = result_to_dict(original_result)
    # Remove the new fields to simulate an old cached result.json
    payload.pop("layer_styles", None)
    payload.pop("blocks", None)
    payload.pop("variant", None)

    loaded = result_from_dict(payload)
    assert loaded.layer_styles == {}
    assert loaded.blocks == []
    assert loaded.variant == "pointcloud"

    # Export must succeed
    dxf_path = export_plan(loaded, work_dir, "local")
    doc = ezdxf.readfile(dxf_path)
    assert "A-WALL" in doc.layers
    assert not doc.audit().has_errors


def test_vector_payload_handles_text_and_polyline3d() -> None:
    poly3d = {
        "type": "polyline3d",
        "points": [[1.0, 2.0, 10.0], [3.0, 4.0, 20.0], [5.0, 6.0, 30.0]],
        "closed": True,
    }
    text_entity = {
        "type": "text",
        "text": "Sample",
        "point": [7.0, 8.0],
        "height": 0.25,
        "rotation": 30.0,
        "align": "MIDDLE_CENTER",
        "layer_role": "annotation",
    }
    result = _make_result(
        layers={"TEST_LAYER": [poly3d, text_entity]},
        layer_colors={"TEST_LAYER": 1},
    )
    payload = vector_payload(result)
    layer_data = payload["layers"]["TEST_LAYER"]
    entities = layer_data["entities"]
    assert len(entities) == 2

    # Polyline3d transformed to 2D polyline (z dropped)
    poly2d = entities[0]
    assert poly2d["type"] == "polyline"
    assert poly2d["closed"] is True
    assert poly2d["points"] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    # Text entity stripped to {type, text, point, height, rotation}
    browser_text = entities[1]
    assert browser_text == {
        "type": "text",
        "text": "Sample",
        "point": [7.0, 8.0],
        "height": 0.25,
        "rotation": 30.0,
    }

    # Bounds and grid correctly populated
    bounds = layer_data["bounds"]
    assert bounds is not None
    assert bounds[0] == pytest.approx(1.0)
    assert bounds[1] == pytest.approx(2.0)
    assert bounds[2] == pytest.approx(7.0)
    assert bounds[3] == pytest.approx(8.0)
    assert len(layer_data["grid"]) > 0

    # Test visible_cells / visible_entity_indices with viewport covering text and polyline
    all_indices = visible_entity_indices(layer_data, (-10.0, -10.0, 20.0, 20.0))
    assert all_indices == [0, 1]

    # Viewport covering only text point
    text_indices = visible_entity_indices(layer_data, (6.9, 7.9, 7.1, 8.1))
    assert 1 in text_indices


def test_xnote_variant_appended(work_dir: Path) -> None:
    # Default variant "pointcloud" does not append variant text
    default_result = _make_result(
        variant="pointcloud",
        layers={"A-WALL": [{"type": "line", "start": [0.0, 0.0], "end": [1.0, 1.0]}]},
    )
    doc_default = ezdxf.readfile(export_plan(default_result, work_dir / "default", "local"))
    note_default = next(e for e in doc_default.modelspace() if e.dxftype() == "TEXT" and e.dxf.layer == "X-NOTE")
    assert "| variant" not in note_default.dxf.text

    # Custom variant appends "| variant <variant>"
    ortho_result = _make_result(
        variant="ortho",
        layers={"A-WALL": [{"type": "line", "start": [0.0, 0.0], "end": [1.0, 1.0]}]},
    )
    doc_ortho = ezdxf.readfile(export_plan(ortho_result, work_dir / "ortho", "local"))
    note_ortho = next(e for e in doc_ortho.modelspace() if e.dxftype() == "TEXT" and e.dxf.layer == "X-NOTE")
    assert "| variant ortho" in note_ortho.dxf.text
