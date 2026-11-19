"""End-to-end smoke test of the ported BIM core: Scene -> IFC -> IDS validation."""

from __future__ import annotations

from pathlib import Path

import ifcopenshell

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.qa import validate_ifc
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene


def test_room_scene_to_valid_ifc(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE))
    path = tmp_path / "room.ifc"
    result = write_ifc(scene, path, run_id="smoke")
    assert path.is_file()

    model = ifcopenshell.open(str(path))
    for ifc_class in ("IfcWall", "IfcDoor", "IfcWindow", "IfcSpace", "IfcSlab"):
        assert len(model.by_type(ifc_class)) > 0, ifc_class
    assert "SAMcloud" in str(model.header.file_name.originating_system)

    report = validate_ifc(path, scene=scene, export_result=result, reports_dir=tmp_path / "reports")
    assert report.ids_specifications
    assert len(report.ids_failures) == 0
    assert len(report.express_findings) == 0
