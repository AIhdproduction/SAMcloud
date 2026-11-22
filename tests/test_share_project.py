"""Backend data contracts for complete project shares."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ezdxf
import pytest

from samcloud.share import export, project
from test_share_export import make_project


def _layout(root: Path, *, indoor: bool = False) -> Path:
    directory = root / "demo"
    (directory / "products" / "classified").mkdir(parents=True)
    (directory / "products" / "classified" / "classified.ply").write_bytes(b"ply")
    key, suffix = ("level_1", "EG") if indoor else ("site", "Site")
    bim = directory / "products" / "bim" / key
    bim.mkdir(parents=True)
    (bim / "model.glb").write_bytes(b"glTF")
    (bim / "x.ifc").write_text("IFC", encoding="utf-8")
    (bim / "props.json").write_text(json.dumps({"elements": {"id": {"ifc_class": "IfcRoof"}}}), encoding="utf-8")
    dxf = directory / "products" / "dxf" / "export"
    dxf.mkdir(parents=True)
    doc = ezdxf.new(); doc.modelspace().add_line((0, 0), (1, 1)); doc.saveas(dxf / f"demo_{suffix}_clean.dxf")
    return directory


def test_project_share_available_for_drone_and_indoor(tmp_path: Path) -> None:
    drone = _layout(tmp_path)
    assert project.project_share_available(drone)
    assert project.project_share_available(_layout(tmp_path / "indoor", indoor=True))
    (drone / "products" / "classified" / "classified.ply").unlink()
    assert not project.project_share_available(drone)


def test_project_share_unavailable_without_dxf_or_glb(tmp_path: Path) -> None:
    no_dxf = _layout(tmp_path / "a")
    for path in (no_dxf / "products" / "dxf" / "export").glob("*_clean.dxf"):
        path.unlink()
    assert not project.project_share_available(no_dxf)
    no_glb = _layout(tmp_path / "b")
    (no_glb / "products" / "bim" / "site" / "model.glb").unlink()
    assert not project.project_share_available(no_glb)


def test_find_project_outputs_keeps_unpaired_views(tmp_path: Path) -> None:
    directory = _layout(tmp_path)
    extra = directory / "products" / "dxf" / "export" / "demo_Only_clean.dxf"
    extra.write_bytes(b"empty")
    lonely = directory / "products" / "bim" / "level_2"
    lonely.mkdir(); (lonely / "model.glb").write_bytes(b"glTF")
    levels = project.find_project_outputs(directory)
    site = next(level for level in levels if level.key == "site")
    assert site.dxf and site.ifc_dir
    assert sum(level.dxf is not None for level in levels) == 2
    assert sum(level.ifc_dir is not None for level in levels) == 2


def test_dxf_to_json_flattens_curves_text_and_blocks(tmp_path: Path) -> None:
    path = tmp_path / "source.dxf"
    doc = ezdxf.new(); doc.layers.add("red", color=1); doc.layers.add("frozen", color=3); doc.layers.get("frozen").freeze()
    doc.layers.add("custom").rgb = (4, 5, 6)
    msp = doc.modelspace(); msp.add_line((2665000.3, 1211000.7), (2665001.0, 1211000.7), dxfattribs={"layer": "red"})
    msp.add_arc((2665001, 1211001), 0.3, 0, 180, dxfattribs={"layer": "red"}); msp.add_circle((2665002, 1211001), 0.2, dxfattribs={"layer": "custom"})
    msp.add_lwpolyline([(2665000.3, 1211002, 0.5), (2665001, 1211002)], format="xyb", dxfattribs={"layer": "red"})
    msp.add_text("plain", dxfattribs={"insert": (2665000.3, 1211000.7), "height": 0.1, "layer": "frozen"}); msp.add_mtext("multi", dxfattribs={"insert": (2665000.4, 1211000.8), "char_height": 0.1})
    block = doc.blocks.new("tag"); block.add_line((0, 0), (1, 0)); block.add_attdef("NOTE", insert=(0, 0), height=0.2)
    insert = msp.add_blockref("tag", (2665003, 1211003), dxfattribs={"layer": "red"}); insert.add_attrib("NOTE", "attribute", (2665003, 1211003), dxfattribs={"height": 0.2})
    doc.saveas(path)
    data = project.dxf_to_json(path)
    assert data["origin"] == pytest.approx([2665000.3, 1211000.7]) and data["bounds"][2] > 0
    # extent: x 2665000.3 .. 2665004.0 (block line end), y 1211000.7 .. 1211003.0 (insert / attrib)
    assert data["bounds"] == [0, 0, round((2665004.0 - 2665000.3) * 1000), round((1211003.0 - 1211000.7) * 1000)]
    rows = [row for layer in data["layers"] for row in layer["polylines"]]
    arc = [row for row in rows if abs(row[0] - 1000) <= 1 and abs(row[1] - 300) <= 1]
    assert arc and len(arc[0]) // 2 > 2, "the ARC is flattened to a polyline with more than two points"
    block_line = [2700, 2300, 3700, 2300]
    assert any(row == block_line or row == block_line[2:] + block_line[:2] for row in rows), "INSERT geometry at the transformed position"
    strings = {text["s"] for layer in data["layers"] for text in layer["texts"]}
    assert {"plain", "multi", "attribute"} <= strings
    assert all(isinstance(value, int) for layer in data["layers"] for row in layer["polylines"] for value in row)
    assert max(len(row) for layer in data["layers"] for row in layer["polylines"]) > 4
    assert any(text["s"] == "attribute" for layer in data["layers"] for text in layer["texts"])
    assert next(layer for layer in data["layers"] if layer["name"] == "frozen")["visible"] is False
    assert next(layer for layer in data["layers"] if layer["name"] == "red")["color"] == "#ff0000"
    assert next(layer for layer in data["layers"] if layer["name"] == "custom")["color"] == "#040506"


def test_ifc_view_json_has_contract_keys_for_both_languages(tmp_path: Path) -> None:
    directory = _layout(tmp_path) / "products" / "bim" / "site"
    for lang in ("de", "en"):
        data = project.ifc_view_json("site", directory, lang)
        assert {"key", "label", "outdoor", "order", "labels", "counts", "hidden", "colors", "by_node"} <= set(data)
        assert isinstance(data["hidden"], list) and data["by_node"] == {"id": "roofs"}
        assert data["labels"]["roofs"]


def test_scan_package_accepts_nested_potree_resources_and_glb(tmp_path: Path) -> None:
    root = tmp_path / "package"; (root / "pc" / "lib" / "potree" / "resources").mkdir(parents=True); (root / "data").mkdir()
    (root / "pc" / "lib" / "potree" / "resources" / "icon.png").write_bytes(b"png")
    (root / "data" / "model.glb").write_bytes(b"glTF")
    assert export.scan_package(root) == []
    (root / "elsewhere.jpg").write_bytes(b"jpg"); (root / "data" / "bad.json").write_text('{"p":"C:/private/x"}', encoding="utf-8")
    assert len(export.scan_package(root)) == 2


NL, CRLF = chr(10), chr(13) + chr(10)


def _fake_converter(root: Path) -> Path:
    """A stand-in for PotreeConverter: writes a minimal octree (metadata.json, hierarchy.bin, octree.bin)."""
    script = root / "fake_converter.py"
    lines = [
        "import json, pathlib, sys",
        "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])",
        "out.mkdir(parents=True, exist_ok=True)",
        "(out / 'metadata.json').write_text(json.dumps({'boundingBox': {'min': [0, 0, 0], 'max': [1, 1, 1]}}))",
        "(out / 'hierarchy.bin').write_bytes(b'h')",
        "(out / 'octree.bin').write_bytes(b'o')",
    ]
    script.write_text(NL.join(lines) + NL, encoding="utf-8")
    launcher = root / "fake_converter.bat"
    launcher.write_text(f'@echo off{CRLF}"{sys.executable}" "{script}" %*{CRLF}', encoding="ascii", newline="")
    return launcher


def test_build_project_package_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    template = tmp_path / "template"
    (template / "css").mkdir(parents=True); (template / "js").mkdir()
    (template / "index.html").write_text("<html><head></head><body></body></html>", encoding="utf-8")
    (template / "css" / "register.css").write_text("body{}", encoding="utf-8")
    (template / "js" / "register.js").write_text("// register", encoding="utf-8")
    monkeypatch.setattr(project, "PROJECT_TEMPLATE_DIRECTORY", template)

    directory = make_project(tmp_path, "demo", points=400)
    dxf_dir = directory / "products" / "dxf" / "export"; dxf_dir.mkdir(parents=True)
    drawing = ezdxf.new(); drawing.modelspace().add_line((10.5, 20.5), (11.5, 21.5)); drawing.saveas(dxf_dir / "demo_Site_clean.dxf")
    bim = directory / "products" / "bim" / "site"; bim.mkdir(parents=True)
    (bim / "model.glb").write_bytes(b"glTF")
    (bim / "props.json").write_text(json.dumps({"elements": {"id": {"ifc_class": "IfcRoof"}}}), encoding="utf-8")
    (bim / "export.json").write_text("{}", encoding="utf-8")
    (bim / "x_site.ifc").write_text("IFC", encoding="utf-8")

    package = tmp_path / "out" / "package"
    result = project.build_project_package(
        directory, package, config={"octree_encoding": "BROTLI"}, progress_for_phase=lambda index: (lambda fraction, message="": None),
        check=lambda: None, max_points=200, converter=_fake_converter(tmp_path))

    assert result["levels"] == ["site"] and set(result["views"]) == {"dxf", "pc", "ifc"}
    assert "</head>" in (package / "index.html").read_text(encoding="utf-8")
    pc_index = (package / "pc" / "index.html").read_text(encoding="utf-8")
    assert "../js/register.js" in pc_index and "../css/register.css" in pc_index
    assert (package / "lib" / "three" / "three.module.js").is_file()
    data = json.loads((package / "data" / "project.json").read_text(encoding="utf-8"))
    assert set(data) == {"version", "name", "created_at", "lang", "generator", "points", "views", "default_level", "levels"}
    assert data["default_level"] == "site" and data["points"] == result["points"] > 0
    assert set(data["levels"][0]) == {"key", "label", "dxf", "ifc"}
    assert data["levels"][0]["ifc"] == "data/ifc/site/" and data["levels"][0]["dxf"] == "data/dxf/site.json"
    assert (package / "data" / "dxf" / "site.json").is_file()
    assert (package / "data" / "ifc" / "site" / "model.glb").is_file() and (package / "data" / "ifc" / "site" / "view.json").is_file()
    names = [path.name.lower() for path in package.rglob("*") if path.is_file()]
    assert not [name for name in names if name in {"props.json", "export.json", "qa.json"} or name.endswith(".ifc")]
    assert export.scan_package(package) == []


def test_assemble_package_register_flag_controls_register_tags(tmp_path: Path) -> None:
    directory = make_project(tmp_path, "reg", points=300)
    points, original = export.prepare_points(directory, 200, lambda fraction, message="": None, lambda: None)
    outputs = {}
    for flag in (False, True):
        target = tmp_path / f"pkg_{flag}"
        export.assemble_package(directory, target, points=points, original_points=original, metadata={}, register=flag)
        outputs[flag] = (target / "index.html").read_text(encoding="utf-8")
    assert "register.js" not in outputs[False] and "register.css" not in outputs[False]
    assert outputs[True].count("register.js") == 1 and outputs[True].count("register.css") == 1
