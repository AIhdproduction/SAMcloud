"""Combined multi-storey IFC: one storey per level, slab opening where the stair of the level below arrives."""

from __future__ import annotations

import copy
from pathlib import Path

import ifcopenshell
import pytest

from samcloud.bim.building_export import export_building_ifc
from samcloud.bim.plan_model import save_plan_model

from test_structure_ifc import make_plan


def _write_levels(root: Path, count: int) -> None:
    ground = make_plan()
    save_plan_model(ground, root / "products" / "dxf" / "level_1" / "ortho")
    if count > 1:
        upper = copy.deepcopy(ground)
        upper.level_id, upper.level_name = "level_2", "OG"
        shift = 2.52 * upper.units_per_metre  # the ground floor stair rises 14 x 0.18 m
        upper.floor_z += shift
        upper.ceiling_z += shift
        for room in upper.rooms:
            room.floor_z = None if room.floor_z is None else room.floor_z + shift
            room.ceiling_z = None if room.ceiling_z is None else room.ceiling_z + shift
            room.ceiling_plane = None if room.ceiling_plane is None else [room.ceiling_plane[0], room.ceiling_plane[1], room.ceiling_plane[2] + shift]
        upper.columns, upper.beams, upper.stairs = [], [], []
        save_plan_model(upper, root / "products" / "dxf" / "level_2" / "ortho")


def test_single_level_project_writes_no_building_ifc(tmp_path: Path) -> None:
    _write_levels(tmp_path, 1)
    assert export_building_ifc(tmp_path) is None
    assert not (tmp_path / "products" / "bim" / "building").exists()


def test_two_levels_give_two_storeys_and_a_stair_opening(tmp_path: Path) -> None:
    _write_levels(tmp_path, 2)
    summary = export_building_ifc(tmp_path)
    assert summary is not None and summary["written"], summary
    assert summary["slab_openings_at_stairs"] == 1
    model = ifcopenshell.open(str(tmp_path / "products" / "bim" / "building" / f"{tmp_path.name}.ifc"))
    storeys = sorted(model.by_type("IfcBuildingStorey"), key=lambda item: item.Elevation)
    assert len(storeys) == 2
    assert storeys[1].Elevation - storeys[0].Elevation == pytest.approx(2.52, abs=1e-3)
    assert len(model.by_type("IfcBuilding")) == 1
    assert len(model.by_type("IfcStair")) == 1
    profiles = [item for item in model.by_type("IfcArbitraryProfileDefWithVoids")]
    assert profiles, "the upper floor slab carries the stair opening as a void"
