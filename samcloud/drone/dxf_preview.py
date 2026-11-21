"""Raster preview of a published outdoor DXF for the DXF/IFC page.

``render_dxf_preview`` draws the DXF with ezdxf's drawing add-on (matplotlib Agg backend) once per visible layer
into transparent PNGs of identical size and extent (``preview_layers/``), then alpha-composites them over the
background into ``preview.png``. The page stacks the layer images, so layers can be toggled without re-rendering.
Frozen/off layers (for example ``S-TOPO-TIN``) are never drawn. The result is cached next to the DXF and only
re-rendered when the DXF is newer.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import ezdxf
from ezdxf import bbox as ezbbox
from ezdxf.colors import DXF_DEFAULT_COLORS, int2rgb
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import Configuration
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.properties import LayoutProperties
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

logger = logging.getLogger(__name__)

PREVIEW_NAME = "preview.png"
META_NAME = "preview.json"
LAYER_DIR = "preview_layers"
META_VERSION = 1
DEFAULT_BACKGROUND = "#0d1b2a"


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _background_rgb(background: str) -> tuple[int, int, int]:
    text = background.lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def _is_dark(rgb: tuple[int, int, int]) -> bool:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 128


def _is_near_black(rgb: tuple[int, int, int]) -> bool:
    # luminance < 0.15, but saturated colours (pure blue has luma 0.11) stay as they are
    return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255.0 < 0.15 and max(rgb) < 100


def _visible(rgb: tuple[int, int, int], dark: bool) -> tuple[int, int, int]:
    """On a dark background near-black colours are drawn white (the usual ACI-7 behaviour)."""
    return (255, 255, 255) if dark and _is_near_black(rgb) else rgb


def _lighten_black_in_doc(doc: Any, dark: bool) -> None:
    """In-memory only: recolour near-black layers/entities so the drawing add-on renders them white."""
    if not dark:
        return
    for layer in doc.layers:
        try:
            if layer.rgb is not None:
                if _is_near_black(tuple(layer.rgb)):  # type: ignore[arg-type]
                    layer.rgb = (255, 255, 255)
            elif 0 < abs(int(layer.color)) < 256 and abs(int(layer.color)) != 7                     and _is_near_black(tuple(int2rgb(DXF_DEFAULT_COLORS[abs(int(layer.color))]))):
                layer.color = 7
        except Exception:
            continue
    spaces = [doc.modelspace()] + [block for block in doc.blocks if not block.name.startswith("*")]
    for space in spaces:
        for entity in space:
            try:
                if entity.dxf.hasattr("true_color"):
                    if _is_near_black(tuple(int2rgb(entity.dxf.true_color))):  # type: ignore[arg-type]
                        entity.rgb = (255, 255, 255)
                else:
                    aci = int(entity.dxf.get("color", 256))
                    if 0 < aci < 256 and aci != 7 and _is_near_black(tuple(int2rgb(DXF_DEFAULT_COLORS[aci]))):
                        entity.dxf.color = 7
            except Exception:
                continue


def _aci_rgb(aci: int, dark: bool) -> tuple[int, int, int]:
    if aci == 7:  # ezdxf (like CAD) draws ACI 7 white on dark and black on light backgrounds
        return (255, 255, 255) if dark else (0, 0, 0)
    return _visible(tuple(int2rgb(DXF_DEFAULT_COLORS[aci])), dark) if 0 < aci < 256 else (200, 200, 200)  # type: ignore[arg-type]


def _layer_color(layer: Any, dark: bool) -> tuple[int, int, int]:
    rgb = layer.rgb
    return _visible(tuple(rgb), dark) if rgb is not None else _aci_rgb(abs(int(layer.color)), dark)  # type: ignore[arg-type]


def _entity_color(entity: Any, fallback: tuple[int, int, int], dark: bool) -> tuple[int, int, int]:
    """Explicit entity colour (true colour or ACI) or the layer colour when the entity is BYLAYER."""
    try:
        if entity.dxf.hasattr("true_color"):
            return _visible(tuple(int2rgb(entity.dxf.true_color)), dark)  # type: ignore[arg-type]
        aci = int(entity.dxf.get("color", 256))
    except Exception:
        return fallback
    return _aci_rgb(aci, dark) if 0 < aci < 256 else fallback


def _visible_layers(doc: Any) -> list[str]:
    return [layer.dxf.name for layer in doc.layers if not layer.is_frozen() and not layer.is_off()]


def layer_legend(dxf_path: Path | str, background: str = DEFAULT_BACKGROUND) -> list[dict[str, Any]]:
    """Layers of the DXF that carry entities and are not frozen/off: ``{name, color (#rrggbb), count}``.

    The colour is the most common explicit entity colour of the layer, else the layer's own colour."""
    doc = ezdxf.readfile(str(dxf_path))
    return _legend_from_doc(doc, _is_dark(_background_rgb(background)))


def _legend_from_doc(doc: Any, dark: bool) -> list[dict[str, Any]]:
    visible = set(_visible_layers(doc))
    counts: Counter[str] = Counter()
    colors: dict[str, Counter[tuple[int, int, int]]] = {}
    layer_rgb = {layer.dxf.name: _layer_color(layer, dark) for layer in doc.layers}
    for entity in doc.modelspace():
        name = entity.dxf.layer
        if name not in visible:
            continue
        counts[name] += 1
        colors.setdefault(name, Counter())[_entity_color(entity, layer_rgb.get(name, (200, 200, 200)), dark)] += 1
    legend = []
    for name in sorted(counts):
        legend.append({"name": name, "color": _hex(colors[name].most_common(1)[0][0]), "count": counts[name]})
    return legend


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def layer_file(layers_dir: Path, name: str) -> Path:
    """Path of a layer's transparent PNG inside ``preview_layers/``."""
    return layers_dir / f"{_safe_name(name)}.png"


def preview_is_fresh(dxf_path: Path | str, out_path: Path | str) -> bool:
    """True when ``out_path`` and its metadata exist and are not older than the DXF."""
    dxf_path, out_path = Path(dxf_path), Path(out_path)
    meta = out_path.with_name(META_NAME)
    try:
        return (out_path.stat().st_mtime_ns >= dxf_path.stat().st_mtime_ns
                and meta.stat().st_mtime_ns >= dxf_path.stat().st_mtime_ns)
    except OSError:
        return False


def load_preview_meta(out_path: Path | str) -> dict[str, Any] | None:
    """Metadata written next to the preview (extent, size, layers with colours/counts/files); None if absent."""
    meta = Path(out_path).with_name(META_NAME)
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("version") == META_VERSION else None


def _extent(doc: Any, names: set[str]) -> tuple[float, float, float, float] | None:
    entities = [e for e in doc.modelspace() if e.dxf.layer in names]
    if not entities:
        return None
    box = ezbbox.extents(entities, fast=True)
    if not box.has_data:
        return None
    return float(box.extmin.x), float(box.extmin.y), float(box.extmax.x), float(box.extmax.y)


def _render_layer(doc: Any, name: str, extent: tuple[float, float, float, float], size: tuple[int, int],
                  background: str, target: Path) -> None:
    width, height = size
    dpi = 100
    figure = Figure(figsize=((width + 0.25) / dpi, (height + 0.25) / dpi), dpi=dpi)
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_axis_off()
    context = RenderContext(doc)
    backend = MatplotlibBackend(axes)
    properties = LayoutProperties.from_layout(doc.modelspace())
    properties.set_colors(background)
    Frontend(context, backend, Configuration()).draw_layout(
        doc.modelspace(), layout_properties=properties, filter_func=lambda entity: entity.dxf.layer == name)
    axes.set_aspect("auto")
    axes.set_xlim(extent[0], extent[2])
    axes.set_ylim(extent[1], extent[3])
    figure.set_size_inches((width + 0.25) / dpi, (height + 0.25) / dpi)  # the backend may have resized the figure
    axes.set_position((0, 0, 1, 1))
    figure.patch.set_alpha(0.0)
    axes.patch.set_alpha(0.0)
    figure.savefig(str(target), dpi=dpi, transparent=True, format="png")


def render_dxf_preview(dxf_path: Path | str, out_path: Path | str, *, size_px: int = 2400,
                       frozen_layers_hidden: bool = True, background: str = DEFAULT_BACKGROUND,
                       force: bool = False) -> Path:
    """Render ``dxf_path`` to ``out_path`` (PNG) plus per-layer transparent PNGs and ``preview.json``.

    The cache is honoured: nothing is re-rendered while ``out_path`` is newer than the DXF. With
    ``frozen_layers_hidden`` (default) frozen/off layers are skipped; otherwise they are drawn as well."""
    dxf_path, out_path = Path(dxf_path), Path(out_path)
    if not force and preview_is_fresh(dxf_path, out_path) and load_preview_meta(out_path) is not None:
        return out_path
    doc = ezdxf.readfile(str(dxf_path))
    dark = _is_dark(_background_rgb(background))
    _lighten_black_in_doc(doc, dark)
    legend = _legend_from_doc(doc, dark)
    if not frozen_layers_hidden:
        legend = _legend_with_frozen(doc, legend, dark)
    names = {item["name"] for item in legend}
    extent = _extent(doc, names)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if extent is None:
        Image.new("RGB", (16, 16), _background_rgb(background)).save(out_path)
        _write_meta(out_path, dxf_path, (0.0, 0.0, 1.0, 1.0), (16, 16), background, [])
        return out_path
    pad = 0.02 * max(extent[2] - extent[0], extent[3] - extent[1], 1e-6)
    extent = (extent[0] - pad, extent[1] - pad, extent[2] + pad, extent[3] + pad)
    span_x, span_y = extent[2] - extent[0], extent[3] - extent[1]
    scale = size_px / max(span_x, span_y)
    size = (max(16, round(span_x * scale)), max(16, round(span_y * scale)))
    layers_dir = out_path.parent / LAYER_DIR
    layers_dir.mkdir(parents=True, exist_ok=True)
    composite = Image.new("RGBA", size, _background_rgb(background) + (255,))
    for item in legend:
        target = layer_file(layers_dir, item["name"])
        _render_layer(doc, item["name"], extent, size, background, target)
        item["file"] = target.name
        with Image.open(target) as layer_image:
            composite.alpha_composite(layer_image.convert("RGBA"))
    composite.convert("RGB").save(out_path, optimize=False)
    _write_meta(out_path, dxf_path, extent, size, background, legend)
    return out_path


def _legend_with_frozen(doc: Any, legend: list[dict[str, Any]], dark: bool) -> list[dict[str, Any]]:
    """Add frozen/off layers with entities to the legend (only used when they should be drawn)."""
    have = {item["name"] for item in legend}
    layer_rgb = {layer.dxf.name: _layer_color(layer, dark) for layer in doc.layers}
    counts = Counter(e.dxf.layer for e in doc.modelspace() if e.dxf.layer not in have)
    for layer in doc.layers:
        layer.thaw()
        layer.on()
    extra = [{"name": n, "color": _hex(layer_rgb.get(n, (200, 200, 200))), "count": c} for n, c in sorted(counts.items())]
    return sorted(legend + extra, key=lambda item: item["name"])


def _write_meta(out_path: Path, dxf_path: Path, extent: tuple[float, float, float, float],
                size: tuple[int, int], background: str, legend: list[dict[str, Any]]) -> None:
    meta = {"version": META_VERSION, "dxf": dxf_path.name, "extent": list(extent), "width": size[0],
            "height": size[1], "background": background, "layers": legend}
    out_path.with_name(META_NAME).write_text(json.dumps(meta), encoding="utf-8")
