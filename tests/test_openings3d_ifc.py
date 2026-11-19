"""IFC windows/doors from refined openings: measured sill/head, partitioning, frame + mullion + glass items."""

from __future__ import annotations

import math
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as element_util
import numpy as np
import pytest

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.qa.validate import validate_ifc
from samcloud.bim.plan_model import PlanOpening, BuildingPlan
from samcloud.bim.scene_builder import build_scene
from test_bim_scene_builder import FLOOR_Z, make_plan


def _plan() -> BuildingPlan:
    plan = make_plan()

    def window(opening_id: str, x: float, width: float, mullions: list[float], sill: float = 0.85, head: float = 2.05) -> PlanOpening:
        return PlanOpening(
            opening_id, "window", [x, 0.0], width, 0.0, 0.3, host_wall_id="W001", sill_m=sill, head_m=head, height_source="photo+points",
            mullion_count=len(mullions), mullion_u=mullions, confidence=0.8,
        )

    plan.openings = [
        PlanOpening("O001", "door", [4.0, 2.5], 0.9, math.pi / 2, 0.2, host_wall_id="W005", sill_m=0.0, head_m=2.13, height_source="photo+points", confidence=0.9),
        PlanOpening("O003", "door", [4.0, 1.2], 0.9, math.pi / 2, 0.2, host_wall_id="W005", sill_m=0.0, head_m=2.2, height_source="points", glazed=True, confidence=0.7),
        window("O010", 1.0, 0.8, []),
        window("O011", 2.4, 1.2, [0.6]),
        window("O012", 4.2, 1.5, [0.5, 1.0], sill=0.7, head=2.2),
        window("O013", 6.4, 1.8, [0.45, 0.9, 1.35]),
    ]
    return plan


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory):
    scene = build_scene(_plan())
    path = tmp_path_factory.mktemp("ifc_openings") / "windows.ifc"
    result = write_ifc(scene, path)
    return scene, result, ifcopenshell.open(str(path)), path


def _window(file: ifcopenshell.file, opening_id: str):
    return next(w for w in file.by_type("IfcWindow") if w.Name == f"Window {opening_id}")


def _door(file: ifcopenshell.file, opening_id: str):
    return next(d for d in file.by_type("IfcDoor") if d.Name == f"Door {opening_id}")


def _items(product) -> list:
    return list(product.Representation.Representations[0].Items)


def _bbox(entity) -> np.ndarray:
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    verts = np.array(ifcopenshell.geom.create_shape(settings, entity).geometry.verts).reshape(-1, 3)
    return np.array([verts.min(axis=0), verts.max(axis=0)])


@pytest.mark.parametrize(
    ("opening_id", "partitioning", "user_defined", "items"),
    [
        ("O010", "SINGLE_PANEL", None, 2),  # lining + 1 pane
        ("O011", "DOUBLE_PANEL_VERTICAL", None, 4),  # lining + 1 bar + 2 panes
        ("O012", "TRIPLE_PANEL_VERTICAL", None, 6),  # lining + 2 bars + 3 panes
        ("O013", "USERDEFINED", "4_PANEL_VERTICAL", 8),  # lining + 3 bars + 4 panes
    ],
)
def test_window_partitioning_and_geometry_items(built, opening_id: str, partitioning: str, user_defined: str | None, items: int) -> None:
    _, _, file, _ = built
    window = _window(file, opening_id)
    assert window.PartitioningType == partitioning
    assert window.UserDefinedPartitioningType == user_defined
    window_type = element_util.get_type(window)
    assert window_type.PartitioningType == partitioning
    assert window_type.UserDefinedPartitioningType == user_defined
    assert len(_items(window)) == items
    panels = [p for p in window_type.HasPropertySets if p.is_a("IfcWindowPanelProperties")]
    assert len(panels) == int(items / 2) if opening_id != "O010" else len(panels) == 1


def test_window_panel_positions_follow_the_pane_order(built) -> None:
    _, _, file, _ = built
    panels = [p for p in element_util.get_type(_window(file, "O012")).HasPropertySets if p.is_a("IfcWindowPanelProperties")]
    assert [p.PanelPosition for p in panels] == ["LEFT", "MIDDLE", "RIGHT"]


def test_window_sill_and_head_are_the_measured_ones(built) -> None:
    _, _, file, _ = built
    for opening_id, sill, head in (("O010", 0.85, 2.05), ("O012", 0.7, 2.2)):
        box = _bbox(_window(file, opening_id))
        assert box[0][2] == pytest.approx(FLOOR_Z + sill, abs=2e-3)
        assert box[1][2] == pytest.approx(FLOOR_Z + head, abs=2e-3)
        assert _window(file, opening_id).OverallHeight == pytest.approx(head - sill, abs=1e-6)
    qto = element_util.get_pset(_window(file, "O012"), "Qto_WindowBaseQuantities")
    assert qto["Height"] == pytest.approx(1.5, abs=1e-6) and qto["Width"] == pytest.approx(1.5, abs=1e-6)


def test_mullion_bars_sit_at_the_measured_positions(built) -> None:
    _, _, file, _ = built
    window = _window(file, "O012")
    # items: lining, bar, bar, pane, pane, pane; bar placement x is the profile centre in the opening frame (from the left edge)
    bars = _items(window)[1:3]
    positions = sorted(float(bar.SweptArea.Position.Location.Coordinates[0]) for bar in bars)
    assert positions == pytest.approx([0.5, 1.0], abs=1e-6)


def test_glass_items_are_styled_translucent_and_frame_is_opaque(built) -> None:
    _, _, file, _ = built
    window = _window(file, "O011")
    frame, bar, left, right = _items(window)

    def transparency(item) -> float | None:
        styled = next(iter(item.StyledByItem or ()), None)
        if styled is None:
            return None
        style = styled.Styles[0]
        if style.is_a("IfcPresentationStyleAssignment"):
            style = style.Styles[0]
        return float(style.Styles[0].Transparency or 0.0)

    assert transparency(left) == pytest.approx(0.65) and transparency(right) == pytest.approx(0.65)
    assert transparency(frame) == pytest.approx(0.0) and transparency(bar) == pytest.approx(0.0)


def test_door_head_and_glazed_door_has_a_glass_panel(built) -> None:
    _, _, file, _ = built
    plain, glazed = _door(file, "O001"), _door(file, "O003")
    assert _bbox(plain)[1][2] == pytest.approx(FLOOR_Z + 2.13, abs=2e-3)
    assert _bbox(glazed)[1][2] == pytest.approx(FLOOR_Z + 2.2, abs=2e-3)
    assert len(_items(plain)) == 2  # lining + leaf
    assert len(_items(glazed)) == 3  # lining + kick panel + glass
    glass = _items(glazed)[2]
    assert next(iter(glass.StyledByItem)) is not None
    assert element_util.get_pset(glazed, "Raumlot_User")["Glazed"] is True
    assert element_util.get_pset(plain, "Raumlot_User")["Glazed"] is False


def test_export_stays_valid_with_ids_and_express(built, tmp_path: Path) -> None:
    scene, result, _, path = built
    report = validate_ifc(path, scene=scene, export_result=result, reports_dir=tmp_path)
    assert report.express_findings == []
    assert report.ids_failures == []
    assert report.reimport_checked and report.reimport_mismatches == []


def test_scene_mullions_are_mirrored_when_the_wall_runs_against_the_axis() -> None:
    plan = _plan()
    # same window, axis pointing the other way (angle pi): positions are measured from the opposite edge
    window = next(o for o in plan.openings if o.id == "O011")
    window.axis_angle = math.pi
    window.mullion_u = [0.3]
    scene = build_scene(plan)
    element = next(e for e in scene.elements.values() if e.name == "Window O011")
    assert element.params.mullion_positions == pytest.approx([0.9])


def test_old_plan_models_with_only_a_count_get_evenly_spaced_bars() -> None:
    plan = make_plan()  # O002 has mullion_count=2 and no positions
    scene = build_scene(plan)
    element = next(e for e in scene.elements.values() if e.name == "Window O002")
    assert element.params.mullion_positions == pytest.approx([0.4, 0.8])


def test_wall_stub_consumed_by_join_cuts_is_not_written_as_a_zero_size_profile(tmp_path: Path) -> None:
    """Real v11 plan: a 16 cm stub disappears in its join cut; the export must stay EXPRESS-clean."""
    from samcloud.bim.plan_model import load_plan_model

    plan = load_plan_model(Path(__file__).parent / "fixtures" / "plan_model_stub_wall.json")
    scene = build_scene(plan)
    path = tmp_path / "stub.ifc"
    result = write_ifc(scene, path)
    report = validate_ifc(path, scene=scene, export_result=result, reports_dir=tmp_path)
    assert report.express_findings == []
    assert report.ids_failures == []
    # the scene builder now drops such stubs (with their joins) before the writer sees them
    assert any("shorter than its join cuts, dropped" in w for w in scene.derived["samcloud_build"]["warnings"])
    assert not any("no positive extent" in str(item) for item in result.warnings)
    file = ifcopenshell.open(str(path))
    assert len(file.by_type("IfcWindow")) >= 2 and len(file.by_type("IfcDoor")) >= 5
