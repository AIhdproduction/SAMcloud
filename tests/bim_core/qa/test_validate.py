from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import ifcopenshell
import ifcopenshell.api.pset
import ifcopenshell.api.root

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.model import CeilingIfcClass, ElementKind
from samcloud.bim.core.qa import ValidationReport, validate_ifc
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene

from .conftest import Exported


def _pset(entity: ifcopenshell.entity_instance, name: str) -> ifcopenshell.entity_instance:
    for relation in entity.IsDefinedBy:
        definition = relation.RelatingPropertyDefinition
        if definition.Name == name:
            return definition
    raise AssertionError(f"{name} not found")


def _broken_copy(source: Path, tmp_path: Path, edit: Callable[[ifcopenshell.file], None]) -> Path:
    file = ifcopenshell.open(str(source))
    edit(file)
    target = tmp_path / "broken.ifc"
    file.write(str(target))
    return target


def test_both_layouts_pass_all_checks(exported: Exported, tmp_path: Path) -> None:
    report = validate_ifc(
        exported.path, scene=exported.scene, export_result=exported.result, reports_dir=tmp_path
    )

    assert report.express_findings == []
    assert report.ids_failures == []
    assert report.reimport_mismatches == []
    assert report.reimport_checked
    assert report.ok
    assert not report.incomplete
    assert not report.unreferenced
    assert all(item["status"] for item in report.ids_specifications)
    assert sum(item["applicable"] for item in report.ids_specifications) > 0


def test_unreferenced_export_passes_and_is_flagged(unreferenced: Exported, tmp_path: Path) -> None:
    report = validate_ifc(
        unreferenced.path,
        scene=unreferenced.scene,
        export_result=unreferenced.result,
        reports_dir=tmp_path,
    )

    assert report.ok
    assert report.unreferenced


def test_covering_ceiling_variant_passes(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec(layout=RoomLayout.TWO_ROOMS))
    scene.settings.ceiling_ifc_class = CeilingIfcClass.COVERING
    result = write_ifc(scene, tmp_path / "covering.ifc")

    report = validate_ifc(
        result.path, scene=scene, export_result=result, reports_dir=tmp_path / "reports"
    )

    assert result.counts["IfcCovering"] == 2  # one ceiling per room
    assert report.ok, report.to_json()
    covering = next(
        item for item in report.ids_specifications if item["name"] == "IfcCovering ceiling"
    )
    assert covering["applicable"] == 2 and covering["status"]


def test_reports_are_written(exported: Exported, tmp_path: Path) -> None:
    report = validate_ifc(exported.path, scene=exported.scene, reports_dir=tmp_path / "reports")

    assert report.json_path is not None and report.html_path is not None
    document = json.loads(Path(report.json_path).read_text(encoding="utf-8"))
    assert document["ok"] is True
    assert document == report.to_dict()
    assert "IFC validation" in Path(report.html_path).read_text(encoding="utf-8")
    assert report.ids_html_path is not None and Path(report.ids_html_path).exists()


def test_reports_default_next_to_the_file(unreferenced: Exported) -> None:
    report = validate_ifc(unreferenced.path)

    assert Path(report.json_path or "").parent == unreferenced.path.parent
    assert not report.reimport_checked


def test_missing_pset_and_quantity_are_reported_by_ids(
    unreferenced: Exported, tmp_path: Path
) -> None:
    removed: dict[str, str] = {}

    def edit(file: ifcopenshell.file) -> None:
        wall = file.by_type("IfcWall")[0]
        removed["tag"] = wall.Tag
        ifcopenshell.api.pset.remove_pset(file, product=wall, pset=_pset(wall, "Pset_WallCommon"))
        window = file.by_type("IfcWindow")[0]
        quantities = _pset(window, "Qto_WindowBaseQuantities")
        quantities.Quantities = tuple(q for q in quantities.Quantities if q.Name != "Height")
        space = file.by_type("IfcSpace")[0]
        ifcopenshell.api.pset.remove_pset(file, product=space, pset=_pset(space, "Raumlot_Quality"))

    broken = _broken_copy(unreferenced.path, tmp_path, edit)
    report = validate_ifc(broken, scene=unreferenced.scene, reports_dir=tmp_path / "reports")

    assert not report.ok
    assert report.incomplete
    wall_failures = [f for f in report.ids_failures if f.specification == "IfcWall"]
    assert wall_failures
    assert {f.tag for f in wall_failures} == {removed["tag"]}
    assert any("Pset_WallCommon" in f.requirement for f in wall_failures)
    assert all(f.global_id and f.reason for f in report.ids_failures)
    assert any(
        f.specification == "IfcWindow" and "Height" in f.requirement for f in report.ids_failures
    )
    assert any(
        f.specification == "IfcSpace" and "Raumlot_Quality" in f.requirement
        for f in report.ids_failures
    )
    assert any(
        m.check == "quantity" and "Qto_WindowBaseQuantities.Height" in m.message
        for m in report.reimport_mismatches
    )


def test_changed_wall_width_is_found_by_reimport_only(
    unreferenced: Exported, tmp_path: Path
) -> None:
    def edit(file: ifcopenshell.file) -> None:
        quantities = _pset(file.by_type("IfcWall")[0], "Qto_WallBaseQuantities")
        for quantity in quantities.Quantities:
            if quantity.Name == "Width":
                quantity.LengthValue = quantity.LengthValue + 0.01

    broken = _broken_copy(unreferenced.path, tmp_path, edit)
    report = validate_ifc(broken, scene=unreferenced.scene, reports_dir=tmp_path / "reports")

    assert report.ids_failures == []
    assert not report.ok
    mismatches = [m for m in report.reimport_mismatches if m.check == "quantity"]
    assert len(mismatches) == 1
    assert "Qto_WallBaseQuantities.Width" in mismatches[0].message
    assert mismatches[0].element_id in unreferenced.scene.elements


def test_broken_relations_and_types_are_found(unreferenced: Exported, tmp_path: Path) -> None:
    def edit(file: ifcopenshell.file) -> None:
        file.remove(file.by_type("IfcRelVoidsElement")[0])
        window = file.by_type("IfcWindow")[0]
        fill = next(
            r for r in file.by_type("IfcRelFillsElement") if r.RelatedBuildingElement == window
        )
        file.remove(fill)
        wall = file.by_type("IfcWall")[0]
        typed = next(r for r in file.by_type("IfcRelDefinesByType") if wall in r.RelatedObjects)
        file.remove(typed)

    broken = _broken_copy(unreferenced.path, tmp_path, edit)
    report = validate_ifc(broken, scene=unreferenced.scene, reports_dir=tmp_path / "reports")

    checks = {m.check for m in report.reimport_mismatches}
    assert {"relation", "type"} <= checks
    messages = " ".join(m.message for m in report.reimport_mismatches)
    assert "does not void its host wall" in messages
    assert "does not fill its opening" in messages
    assert "has no type object" in messages


def test_missing_element_and_count_are_found(unreferenced: Exported, tmp_path: Path) -> None:
    def edit(file: ifcopenshell.file) -> None:
        ifcopenshell.api.root.remove_product(file, product=file.by_type("IfcDoor")[0])

    broken = _broken_copy(unreferenced.path, tmp_path, edit)
    report = validate_ifc(broken, scene=unreferenced.scene, reports_dir=tmp_path / "reports")

    checks = {m.check for m in report.reimport_mismatches}
    assert {"count", "element"} <= checks
    assert not report.ok


def test_wrong_georeference_is_found(exported: Exported, tmp_path: Path) -> None:
    def edit(file: ifcopenshell.file) -> None:
        file.by_type("IfcMapConversion")[0].Eastings += 1.0

    broken = _broken_copy(exported.path, tmp_path, edit)
    report = validate_ifc(broken, scene=exported.scene, reports_dir=tmp_path / "reports")

    assert [m.check for m in report.reimport_mismatches] == ["georeference"]


def test_express_findings_are_collected(unreferenced: Exported, tmp_path: Path) -> None:
    def edit(file: ifcopenshell.file) -> None:
        file.by_type("IfcWall")[0].GlobalId = "too-short"

    broken = _broken_copy(unreferenced.path, tmp_path, edit)
    report: ValidationReport = validate_ifc(broken, reports_dir=tmp_path / "reports")

    assert report.express_findings
    assert not report.ok
    assert {"level", "message", "instance"} <= report.express_findings[0].keys()


def test_incomplete_flag_comes_from_the_export_result(tmp_path: Path) -> None:
    scene = build_room_scene(RoomSpec())
    wall = next(e for e in scene.elements.values() if e.kind == ElementKind.WALL)
    wall.storey_id = None
    result = write_ifc(scene, tmp_path / "nostorey.ifc")

    report = validate_ifc(
        result.path, scene=scene, export_result=result, reports_dir=tmp_path / "r"
    )

    assert result.incomplete
    assert report.incomplete
    assert report.warnings == result.warnings


def test_removed_description_is_caught_by_ids_and_reimport(
    unreferenced: Exported, tmp_path: Path
) -> None:
    removed: dict[str, str] = {}

    def edit(file: ifcopenshell.file) -> None:
        wall = file.by_type("IfcWall")[0]
        removed["tag"] = wall.Tag
        wall.Description = None
        file.by_type("IfcDoor")[0].ObjectType = " "

    broken = _broken_copy(unreferenced.path, tmp_path, edit)
    report = validate_ifc(broken, scene=unreferenced.scene, reports_dir=tmp_path / "reports")

    assert not report.ok
    assert any(
        f.specification == "IfcWall" and f.tag == removed["tag"] and "Description" in f.requirement
        for f in report.ids_failures
    )
    messages = [m.message for m in report.reimport_mismatches if m.check == "identity"]
    assert any("IfcWall" in m and "Description is empty" in m for m in messages)
    assert any("IfcDoor" in m and "ObjectType is empty" in m for m in messages)


def test_intact_export_has_no_identity_mismatch(exported: Exported, tmp_path: Path) -> None:
    report = validate_ifc(exported.path, scene=exported.scene, reports_dir=tmp_path)

    assert not [m for m in report.reimport_mismatches if m.check == "identity"]
