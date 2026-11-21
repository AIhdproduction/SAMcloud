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
