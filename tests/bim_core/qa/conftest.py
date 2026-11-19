"""Fixtures for the QA tests: exported synthetic layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from samcloud.bim.core.ifc import IfcExportResult, write_ifc
from samcloud.bim.core.model import CrsInfo, Scene
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene


@dataclass
class Exported:
    scene: Scene
    result: IfcExportResult
    path: Path


@pytest.fixture(scope="module", params=[RoomLayout.RECTANGLE, RoomLayout.TWO_ROOMS])
def exported(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Exported:
    """Both synthetic layouts, exported with a Swiss CRS."""
    layout: RoomLayout = request.param
    scene = build_room_scene(RoomSpec(layout=layout))
    scene.settings.crs = CrsInfo(epsg=2056, vertical_datum="LN02", source="input")
    path = tmp_path_factory.mktemp(f"qa_{layout.value}") / "model.ifc"
    result = write_ifc(scene, path, run_id="run-1")
    return Exported(scene, result, path)


@pytest.fixture(scope="module")
def unreferenced(tmp_path_factory: pytest.TempPathFactory) -> Exported:
    """Rectangle layout without a CRS."""
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    path = tmp_path_factory.mktemp("qa_unreferenced") / "model.ifc"
    return Exported(scene, write_ifc(scene, path), path)
