"""DXF preview renderer (outdoor DXF/IFC page): rendering, frozen layers, cache, legend, page helper."""

from __future__ import annotations

import os
import time
from pathlib import Path

import ezdxf
import numpy as np
import yaml
from PIL import Image

from samcloud.app.dxf_controller import DxfPageController
from samcloud.drone import dxf_preview

RED, GREEN, BLUE, FROZEN = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)


def _write_dxf(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    for name, rgb in (("L-RED", RED), ("L-GREEN", GREEN), ("L-BLUE", BLUE), ("L-FROZEN", FROZEN)):
        layer = doc.layers.add(name)
        layer.rgb = rgb
    doc.layers.get("L-FROZEN").freeze()
    msp = doc.modelspace()
    rng = np.random.default_rng(3)
    for index, (name, y0) in enumerate((("L-RED", 0.0), ("L-GREEN", 30.0), ("L-BLUE", 60.0), ("L-FROZEN", 90.0))):
        for k in range(3):
            x = 3.7 * k + 1.13 * index
            msp.add_line((x, y0 + rng.uniform(0.3, 0.9)), (x + 40.3, y0 + 10.17 + rng.uniform(0.1, 0.8)),
                         dxfattribs={"layer": name, "lineweight": 100})
    doc.saveas(path)
    return path


def _has_colour(image: Path, rgb: tuple[int, int, int], tolerance: int = 40) -> bool:
    pixels = np.asarray(Image.open(image).convert("RGB")).astype(int)
    return bool((np.abs(pixels - np.array(rgb)).sum(axis=2) < tolerance).any())


def test_render_draws_visible_layers_and_hides_frozen(tmp_path: Path) -> None:
    dxf = _write_dxf(tmp_path / "plan.dxf")
    out = dxf_preview.render_dxf_preview(dxf, tmp_path / "preview.png", size_px=600)
    assert out.is_file() and out.stat().st_size > 1000
    for rgb in (RED, GREEN, BLUE):
        assert _has_colour(out, rgb), rgb
    assert not _has_colour(out, FROZEN)
    assert not (tmp_path / dxf_preview.LAYER_DIR / "L-FROZEN.png").exists()
    sizes = {Image.open(p).size for p in (tmp_path / dxf_preview.LAYER_DIR).glob("*.png")}
    assert sizes == {Image.open(out).size}


def test_cache_rerenders_only_when_dxf_is_newer(tmp_path: Path) -> None:
    dxf = _write_dxf(tmp_path / "plan.dxf")
    out = tmp_path / "preview.png"
    dxf_preview.render_dxf_preview(dxf, out, size_px=300)
    assert dxf_preview.preview_is_fresh(dxf, out)
    first = out.stat().st_mtime_ns
    time.sleep(0.05)
    dxf_preview.render_dxf_preview(dxf, out, size_px=300)
    assert out.stat().st_mtime_ns == first  # cache hit
    old = time.time() - 100
    for stale in (out, out.with_name(dxf_preview.META_NAME)):
        os.utime(stale, (old, old))
    os.utime(dxf, (old + 50, old + 50))  # DXF newer than the preview
    assert not dxf_preview.preview_is_fresh(dxf, out)
    dxf_preview.render_dxf_preview(dxf, out, size_px=300)
    assert out.stat().st_mtime_ns > int(old * 1e9) + 10**9
    assert dxf_preview.preview_is_fresh(dxf, out)


def test_legend_lists_layers_with_colours_and_counts(tmp_path: Path) -> None:
    dxf = _write_dxf(tmp_path / "plan.dxf")
    legend = {item["name"]: item for item in dxf_preview.layer_legend(dxf)}
    assert set(legend) == {"L-RED", "L-GREEN", "L-BLUE"}  # frozen layer excluded
    assert legend["L-RED"]["color"] == "#ff0000"
    assert legend["L-GREEN"]["color"] == "#00ff00"
    assert all(item["count"] == 3 for item in legend.values())
    meta = dxf_preview.load_preview_meta(dxf_preview.render_dxf_preview(dxf, tmp_path / "preview.png", size_px=300))
    assert meta is not None and {item["name"] for item in meta["layers"]} == set(legend)
    assert all(item["file"] for item in meta["layers"])


def _project(root: Path, name: str, *, drone: bool) -> Path:
    project = (root / name).resolve()
    project.mkdir(parents=True)
    document: dict = {"project": {"name": name}, "settings": {"classification": {"class_set": "outdoor" if drone else "indoor"}}, "workflow": {}}
    if drone:
        document["acquisition"] = {"type": "drone_gps"}
    (project / "project.samcloud.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    export = project / "products" / "dxf" / "export"
    export.mkdir(parents=True)
    _write_dxf(export / f"{name}_Site_clean.dxf")
    return project


def test_page_helper_returns_preview_path_for_outdoor_only(tmp_path: Path) -> None:
    outdoor = DxfPageController(_project(tmp_path, "outdoor_proj", drone=True))
    path = outdoor.dxf_preview_path()
    assert path is not None and path.name == "preview.png" and path.parent.name == "export"
    indoor = DxfPageController(_project(tmp_path, "indoor_proj", drone=False))
    assert indoor.dxf_preview_path() is None
    empty = _project(tmp_path, "outdoor_empty", drone=True)
    for dxf in (empty / "products" / "dxf" / "export").glob("*.dxf"):
        dxf.rename(dxf.with_suffix(".bak"))
    assert DxfPageController(empty).dxf_preview_path() is None


def test_dxf_preview_card_keys_resolve_in_de_and_en() -> None:
    import re

    from samcloud.app import i18n

    source = (Path(__file__).resolve().parents[1] / "samcloud" / "app" / "pages" / "dxf.py").read_text(encoding="utf-8")
    keys = set(re.findall(r't\("([^"]*dxf_preview[^"]*)"', source))
    assert keys, "no dxf_preview keys found in the page"
    for lang in ("de", "en"):
        i18n.set_language(lang)
        for key in keys:
            assert key.startswith("dxf.dxf_preview."), key
            assert i18n.has_translation(key), (lang, key)
            assert i18n.t(key, error="x") != key, (lang, key)


def test_black_layer_is_drawn_white_on_dark_background(tmp_path: Path) -> None:
    doc = ezdxf.new("R2018")
    black = doc.layers.add("S-ELEV")
    black.rgb = (0, 0, 0)
    aci = doc.layers.add("L-ACI7")
    aci.color = 7
    msp = doc.modelspace()
    msp.add_line((0, 0), (40.3, 10.1), dxfattribs={"layer": "S-ELEV", "lineweight": 100})
    msp.add_line((0, 20), (40.3, 30.1), dxfattribs={"layer": "L-ACI7", "lineweight": 100})
    dxf = tmp_path / "black.dxf"
    doc.saveas(dxf)
    out = dxf_preview.render_dxf_preview(dxf, tmp_path / "preview.png", size_px=400)
    layer_png = tmp_path / dxf_preview.LAYER_DIR / "S-ELEV.png"
    assert _has_colour(layer_png, (255, 255, 255)) or (np.asarray(Image.open(layer_png).convert("RGBA"))[..., :3].min(axis=2) > 200).any()
    assert _has_colour(out, (255, 255, 255))
    legend = {item["name"]: item["color"] for item in dxf_preview.layer_legend(dxf)}
    assert legend["S-ELEV"] == "#ffffff"


# ------------------------------------------------------------------------------------------- vector (SVG) preview

import re
import xml.etree.ElementTree as ET

import pytest

SVG_NS = "{http://www.w3.org/2000/svg}"
LV95_X, LV95_Y = 2_699_800.0, 1_221_200.0


def _write_entity_dxf(path: Path, *, origin: tuple[float, float] = (0.0, 0.0)) -> Path:
    ox, oy = origin
    doc = ezdxf.new("R2018")
    for name, rgb in (("S-LINE", RED), ("S-POLY", GREEN), ("S-ROUND", BLUE), ("S-TXT", (255, 255, 0)),
                      ("S-TREE", (0, 200, 90)), ("S-TIN", FROZEN)):
        doc.layers.add(name).rgb = rgb
    doc.layers.get("S-TIN").freeze()
    block = doc.blocks.new("TREEB")
    block.add_circle((0, 0), 1.0)
    block.add_attdef("KRONE", (1.2, -1.2), height=0.2, text="KRONE=?")
    hidden = block.add_attdef("HEIGHT", (0, -0.1), height=0.05, text="h")
    hidden.dxf.flags = 1
    msp = doc.modelspace()
    msp.add_line((ox, oy), (ox + 10.0, oy + 5.0), dxfattribs={"layer": "S-LINE"})
    msp.add_lwpolyline([(ox, oy + 20, 0, 0, 0), (ox + 10, oy + 20, 0, 0, 0.5), (ox + 10, oy + 30, 0, 0, 0),
                        (ox, oy + 30, 0, 0, 0)], format="xyseb", close=True, dxfattribs={"layer": "S-POLY"})
    msp.add_polyline3d([(ox, oy + 40, 1), (ox + 5, oy + 41, 2), (ox + 10, oy + 40, 3)], dxfattribs={"layer": "S-POLY"})
    msp.add_spline([(ox, oy + 50), (ox + 3, oy + 55), (ox + 6, oy + 50), (ox + 9, oy + 56)], dxfattribs={"layer": "S-POLY"})
    msp.add_circle((ox + 20, oy + 5), 2.0, dxfattribs={"layer": "S-ROUND"})
    msp.add_arc((ox + 20, oy + 20), 3.0, 10, 200, dxfattribs={"layer": "S-ROUND"})
    msp.add_text("Contour 902.0", height=0.25, dxfattribs={"layer": "S-TXT", "insert": (ox + 30, oy + 5), "rotation": 30})
    insert = msp.add_blockref("TREEB", (ox + 40, oy + 10), dxfattribs={"layer": "S-TREE"})
    insert.add_auto_attribs({"KRONE": "KRONE=1.99 m", "HEIGHT": "HIDDEN"})
    msp.add_3dface([(ox, oy), (ox + 1, oy), (ox + 1, oy + 1), (ox, oy + 1)], dxfattribs={"layer": "S-TIN"})
    doc.saveas(path)
    return path


def _groups(svg: Path) -> dict[str, ET.Element]:
    root = ET.parse(svg).getroot()
    return {g.get("data-layer"): g for g in root.iter(f"{SVG_NS}g") if g.get("data-layer")}


def test_svg_has_one_group_per_layer_with_all_entity_types(tmp_path: Path) -> None:
    dxf = _write_entity_dxf(tmp_path / "plan.dxf")
    svg = dxf_preview.render_dxf_svg(dxf, tmp_path / "preview.svg")
    root = ET.parse(svg).getroot()  # valid XML
    assert root.tag == f"{SVG_NS}svg" and root.get("viewBox")
    groups = _groups(svg)
    assert set(groups) == {"S-LINE", "S-POLY", "S-ROUND", "S-TXT", "S-TREE"}  # frozen S-TIN hidden
    assert groups["S-LINE"].get("stroke") == "#ff0000"
    assert len(groups["S-LINE"].findall(f"{SVG_NS}path")) == 1
    assert len(groups["S-POLY"].findall(f"{SVG_NS}path")) == 3  # LWPOLYLINE with bulge, POLYLINE 3D, SPLINE
    assert any(p.get("d", "").endswith("Z") for p in groups["S-POLY"].findall(f"{SVG_NS}path"))
    assert len(groups["S-ROUND"].findall(f"{SVG_NS}circle")) == 1 and len(groups["S-ROUND"].findall(f"{SVG_NS}path")) == 1
    texts = [element.text for element in groups["S-TXT"].findall(f"{SVG_NS}text")]
    assert texts == ["Contour 902.0"]
    assert "rotate(-30" in groups["S-TXT"].find(f"{SVG_NS}text").get("transform")
    tree = groups["S-TREE"]
    assert len(tree.findall(f"{SVG_NS}circle")) == 1  # block geometry exploded
    assert [element.text for element in tree.findall(f"{SVG_NS}text")] == ["KRONE=1.99 m"]  # invisible attribute skipped
    assert "S-TIN" not in svg.read_text(encoding="utf-8")
    meta = dxf_preview.load_svg_meta(svg)
    assert meta is not None and {item["name"] for item in meta["layers"]} == set(groups)


def test_svg_includes_frozen_layer_when_requested(tmp_path: Path) -> None:
    dxf = _write_entity_dxf(tmp_path / "plan.dxf")
    svg = dxf_preview.render_dxf_svg(dxf, tmp_path / "preview.svg", frozen_layers_hidden=False, force=True)
    assert "S-TIN" in _groups(svg)


def test_svg_uses_local_origin_for_lv95_coordinates(tmp_path: Path) -> None:
    dxf = _write_entity_dxf(tmp_path / "plan.dxf", origin=(LV95_X, LV95_Y))
    svg = dxf_preview.render_dxf_svg(dxf, tmp_path / "preview.svg")
    text = svg.read_text(encoding="utf-8")
    viewbox = [float(v) for v in ET.parse(svg).getroot().get("viewBox").split()]
    assert all(abs(v) < 10_000 for v in viewbox)
    values = [float(n) for attr in re.findall(r'(?:d|x|y|cx|cy)="([^"]*)"', text) for n in re.findall(r"-?\d+(?:\.\d+)?", attr)]
    assert values and max(abs(v) for v in values) < 10_000
    meta = dxf_preview.load_svg_meta(svg)
    assert meta is not None and abs(meta["origin"][0] - LV95_X) < 100 and abs(meta["origin"][1] - LV95_Y) < 100


def test_svg_cache_rebuilds_only_when_dxf_is_newer(tmp_path: Path) -> None:
    dxf = _write_entity_dxf(tmp_path / "plan.dxf")
    out = tmp_path / "preview.svg"
    dxf_preview.render_dxf_svg(dxf, out)
    assert dxf_preview.svg_is_fresh(dxf, out)
    first = out.stat().st_mtime_ns
    time.sleep(0.05)
    dxf_preview.render_dxf_svg(dxf, out)
    assert out.stat().st_mtime_ns == first  # cache hit
    old = time.time() - 100
    for stale in (out, out.with_name(dxf_preview.SVG_META_NAME)):
        os.utime(stale, (old, old))
    os.utime(dxf, (old + 50, old + 50))
    assert not dxf_preview.svg_is_fresh(dxf, out)
    dxf_preview.render_dxf_svg(dxf, out)
    assert out.stat().st_mtime_ns > int(old * 1e9) + 10**9
    assert dxf_preview.svg_is_fresh(dxf, out)


def test_svg_path_helper_and_controller(tmp_path: Path) -> None:
    outdoor = DxfPageController(_project(tmp_path, "outdoor_svg", drone=True))
    svg = outdoor.dxf_preview_svg_path()
    assert svg is not None and svg.name == "preview.svg" and svg.parent.name == "export"
    assert dxf_preview.svg_preview_path(tmp_path / "preview.png") == tmp_path / "preview.svg"


def _page_route() -> None:
    from samcloud.app.pages import dxf as dxf_page

    dxf_page.render()


_page_route.__module__ = "tests.dxf_preview_route"


@pytest.mark.anyio
async def test_page_builds_card_with_svg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nicegui import ui
    from nicegui.testing.user_simulation import user_simulation

    from samcloud.app.pages import dxf as dxf_page
    from samcloud.app.state import state as app_state

    from dxf_synthetic import write_outdoor_scene

    project = _project(tmp_path, "outdoor_page", drone=True)
    write_outdoor_scene(project / "products" / "classified" / "classified.las")
    dxf = next((project / "products" / "dxf" / "export").glob("*.dxf"))
    _write_entity_dxf(dxf)
    dxf_preview.render_dxf_svg(dxf, dxf.parent / "preview.svg")
    monkeypatch.setattr(app_state, "active_project_directory", project)
    monkeypatch.setattr(dxf_page, "_RUNNERS", {})
    async with user_simulation(root=lambda: None) as user:
        ui.page("/dxf")(_page_route)
        await user.open("/dxf")
        await user.should_see(marker="dxf-preview-card")
        await user.should_see(marker="dxf-preview-svg")
        await user.should_see(marker="dxf-preview-layer-S-TREE")
        await user.should_not_see(marker="dxf-preview-spinner")
        response = dxf_page.dxf_preview_file("preview.svg")  # served as SVG, never re-rendered
        assert getattr(response, "media_type", "") == "image/svg+xml"
