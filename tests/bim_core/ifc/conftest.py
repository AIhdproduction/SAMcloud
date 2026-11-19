"""Fixtures and helpers for the IFC writer tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import ifcopenshell
import ifcopenshell.validate
import pytest

from samcloud.bim.core.geometry import SceneGeometry, compute_scene_geometry
from samcloud.bim.core.ifc import IfcExportResult, write_ifc
from samcloud.bim.core.model import CrsInfo, Scene
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene

FIXED_TIME = datetime.fromisoformat("2026-01-01T12:00:00+00:00")


@dataclass
class Export:
    """One exported scene with its geometry and the re-opened IFC file."""

    scene: Scene
    geometry: SceneGeometry
    result: IfcExportResult
    file: ifcopenshell.file
    path: Path


def export_scene(scene: Scene, path: Path, **kwargs: Any) -> Export:
    """Export a scene with a fixed time stamp and re-open the file."""
    result = write_ifc(scene, path, timestamp=FIXED_TIME, **kwargs)
    return Export(scene, compute_scene_geometry(scene), result, ifcopenshell.open(str(path)), path)


def validation_statements(file: ifcopenshell.file) -> list[dict[str, Any]]:
    """All findings of ifcopenshell.validate with the EXPRESS rules (empty means valid)."""
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(file, logger, express_rules=True)
    return list(logger.statements)


def swiss_crs() -> CrsInfo:
    """EPSG:2056 with the LN02 height system."""
    return CrsInfo(epsg=2056, vertical_datum="LN02", name="CH1903+ / LV95", source="input")


def entity_by_tag(file: ifcopenshell.file, element_id: str) -> ifcopenshell.entity_instance:
    """The IfcElement whose Tag is the scene element uuid."""
    matches = [e for e in file.by_type("IfcElement") if e.Tag == element_id]
    assert len(matches) == 1
    return matches[0]


@pytest.fixture(scope="module", params=[RoomLayout.RECTANGLE, RoomLayout.TWO_ROOMS])
def layout_export(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Export:
    """Both synthetic layouts, exported with a Swiss CRS."""
    layout: RoomLayout = request.param
    scene = build_room_scene(RoomSpec(layout=layout))
    scene.settings.crs = swiss_crs()
    path = tmp_path_factory.mktemp(f"ifc_{layout.value}") / "model.ifc"
    return export_scene(scene, path, run_id="run-1", ai_backend="openrouter", ai_model="test-model")


@pytest.fixture(scope="module")
def rectangle_export(tmp_path_factory: pytest.TempPathFactory) -> Export:
    """Rectangle layout without CRS (unreferenced)."""
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    path = tmp_path_factory.mktemp("ifc_rect") / "model.ifc"
    return export_scene(scene, path)
