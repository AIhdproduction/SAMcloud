"""Raster preview of a published outdoor DXF for the DXF/IFC page.

``render_dxf_preview`` draws the DXF with ezdxf's drawing add-on (matplotlib Agg backend) once per visible layer
into transparent PNGs of identical size and extent (``preview_layers/``), then alpha-composites them over the
background into ``preview.png``. The page stacks the layer images, so layers can be toggled without re-rendering.
Frozen/off layers (for example ``S-TOPO-TIN``) are never drawn. The result is cached next to the DXF and only
re-rendered when the DXF is newer.
"""

from __future__ import annotations

import json
import math
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


# --------------------------------------------------------------------------------------------------------------
# Vector (SVG) preview: stays sharp at any zoom level, text is real <text>
# --------------------------------------------------------------------------------------------------------------

SVG_NAME = "preview.svg"
SVG_META_NAME = "preview_svg.json"
SVG_META_VERSION = 1
SVG_SIMPLIFY_M = 0.01  # polylines are simplified to 1 cm
SVG_FLATTEN_M = 0.01
SVG_TEXT_EM = 0.72  # DXF text height is the cap height, SVG font-size the em size
SVG_STROKE_PX = 1.0
_SVG_PATH_TYPES = {"LINE", "LWPOLYLINE", "POLYLINE", "SPLINE", "ARC", "ELLIPSE", "3DFACE"}
_SVG_STYLE = (
    ".samdxfp-svg path,.samdxfp-svg circle{fill:none;stroke-width:%s;vector-effect:non-scaling-stroke;"
    "stroke-linecap:round;stroke-linejoin:round}"
    ".samdxfp-svg text{stroke:none;font-family:Arial,Helvetica,sans-serif;white-space:pre}"
) % SVG_STROKE_PX
_ANCHOR = {0: "start", 1: "middle", 2: "end", 3: "middle", 4: "middle", 5: "middle"}
_BASELINE = {0: "alphabetic", 1: "alphabetic", 2: "central", 3: "hanging"}
_MTEXT_ALIGN = {1: (0, 3), 2: (1, 3), 3: (2, 3), 4: (0, 2), 5: (1, 2), 6: (2, 2), 7: (0, 1), 8: (1, 1), 9: (2, 1)}


def svg_preview_path(png_path: Path | str) -> Path:
    """``preview.svg`` next to the raster ``preview.png``."""
    return Path(png_path).with_name(SVG_NAME)


def svg_is_fresh(dxf_path: Path | str, svg_path: Path | str) -> bool:
    """True when the SVG and its metadata exist and are not older than the DXF."""
    dxf_path, svg_path = Path(dxf_path), Path(svg_path)
    meta = svg_path.with_name(SVG_META_NAME)
    try:
        newest = dxf_path.stat().st_mtime_ns
        return svg_path.stat().st_mtime_ns >= newest and meta.stat().st_mtime_ns >= newest
    except OSError:
        return False


def load_svg_meta(svg_path: Path | str) -> dict[str, Any] | None:
    """Metadata of the SVG preview (viewBox, origin, layers with colours/counts); None if absent."""
    try:
        data = json.loads(Path(svg_path).with_name(SVG_META_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("version") == SVG_META_VERSION else None


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _num(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return _background_rgb(color)


def _entity_polylines(entity: Any) -> list[tuple[list[tuple[float, float]], bool]]:
    """Flattened ``(points, closed)`` lists of a path-like entity in DXF coordinates."""
    from ezdxf import path as ezpath

    kind = entity.dxftype()
    if kind == "LINE":
        return [([(entity.dxf.start.x, entity.dxf.start.y), (entity.dxf.end.x, entity.dxf.end.y)], False)]
    if kind == "3DFACE":
        pts = [entity.dxf.get(f"vtx{i}") for i in range(4) if entity.dxf.hasattr(f"vtx{i}")]
        return [([(p.x, p.y) for p in pts], True)] if len(pts) >= 3 else []
    result = []
    for sub in ezpath.make_path(entity).sub_paths():
        points = [(v.x, v.y) for v in sub.flattening(SVG_FLATTEN_M)]
        if len(points) >= 2:
            result.append((points, bool(sub.is_closed)))
    return result


def _simplify(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 4:
        return points
    from shapely.geometry import LineString

    simplified = list(LineString(points).simplify(SVG_SIMPLIFY_M, preserve_topology=False).coords)
    return [(p[0], p[1]) for p in simplified] if len(simplified) >= 2 else points


def _path_data(points: list[tuple[float, float]], closed: bool, origin: tuple[float, float]) -> str:
    ox, oy = origin
    local = [(x - ox, oy - y) for x, y in points]
    if closed and len(local) > 2 and local[0] == local[-1]:
        local = local[:-1]
    parts: list[str] = []
    last: tuple[str, str] | None = None
    for x, y in local:
        pair = (_num(x), _num(y))
        if pair == last:
            continue
        parts.append(("M" if not parts else "L") + f"{pair[0]} {pair[1]}")
        last = pair
    if len(parts) < 2:
        return ""
    return "".join(parts) + ("Z" if closed else "")


def _text_lines(entity: Any) -> list[str]:
    try:
        text = entity.plain_text()
    except Exception:
        text = str(entity.dxf.get("text", ""))
    text = str(text).replace("\r", "")
    return text.split("\n") if text.strip() else []


def _text_element(entity: Any, origin: tuple[float, float], color: str) -> str:
    """One SVG ``<text>`` for TEXT / ATTRIB / MTEXT; empty string when there is nothing to show."""
    lines = _text_lines(entity)
    if not lines:
        return ""
    dxf = entity.dxf
    if entity.dxftype() == "MTEXT":
        insert = dxf.insert
        height = float(dxf.get("char_height", 1.0))
        if dxf.hasattr("text_direction"):
            rotation = math.degrees(math.atan2(dxf.text_direction.y, dxf.text_direction.x))
        else:
            rotation = float(dxf.get("rotation", 0.0))
        halign, valign = _MTEXT_ALIGN.get(int(dxf.get("attachment_point", 1)), (0, 3))
    else:
        halign, valign = int(dxf.get("halign", 0)), int(dxf.get("valign", 0))
        insert = dxf.align_point if (halign or valign) and dxf.hasattr("align_point") else dxf.insert
        height = float(dxf.get("height", 1.0))
        rotation = float(dxf.get("rotation", 0.0))
    if height <= 0:
        return ""
    x, y = insert.x - origin[0], origin[1] - insert.y
    size = height / SVG_TEXT_EM
    attrs = (f'x="{_num(x)}" y="{_num(y)}" font-size="{size:.3f}" fill="{color}" '
             f'text-anchor="{_ANCHOR.get(halign, "start")}" dominant-baseline="{_BASELINE.get(valign, "alphabetic")}"')
    if abs(rotation) > 1e-6:
        attrs += f' transform="rotate({-rotation:.2f} {_num(x)} {_num(y)})"'
    if len(lines) == 1:
        return f"<text {attrs}>{_esc(lines[0])}</text>"
    spans = "".join(f'<tspan x="{_num(x)}" dy="{"0" if i == 0 else _num(size * 1.2)}">{_esc(line)}</tspan>'
                    for i, line in enumerate(lines))
    return f"<text {attrs}>{spans}</text>"


def _attrib_hidden(attrib: Any) -> bool:
    return bool(int(attrib.dxf.get("flags", 0)) & 1) or bool(attrib.dxf.get("invisible", 0))


def _entity_svg(entity: Any, origin: tuple[float, float], layer_rgb: tuple[int, int, int], dark: bool,
                depth: int = 0) -> list[str]:
    """SVG elements of one entity; INSERTs are exploded (block children take the INSERT's colour when BYBLOCK /
    layer 0) and only visible attributes are written."""
    kind = entity.dxftype()
    rgb = _entity_color(entity, layer_rgb, dark)
    stroke = "" if rgb == layer_rgb else f' stroke="{_hex(rgb)}"'
    out: list[str] = []
    if kind == "INSERT":
        if depth > 4:
            return out
        try:
            children = list(entity.virtual_entities())
        except Exception:
            children = []
        for child in children:
            if child.dxftype() not in ("ATTDEF", "ATTRIB"):
                out.extend(_entity_svg(child, origin, rgb, dark, depth + 1))
        for attrib in entity.attribs:
            if _attrib_hidden(attrib):
                continue
            element = _text_element(attrib, origin, _hex(_entity_color(attrib, rgb, dark)))
            if element:
                out.append(element)
        return out
    if kind in ("TEXT", "MTEXT", "ATTRIB"):
        element = _text_element(entity, origin, _hex(rgb))
        return [element] if element else []
    if kind == "CIRCLE":
        cx, cy = entity.dxf.center.x - origin[0], origin[1] - entity.dxf.center.y
        return [f'<circle cx="{_num(cx)}" cy="{_num(cy)}" r="{_num(entity.dxf.radius)}"{stroke}/>']
    if kind in _SVG_PATH_TYPES:
        try:
            polylines = _entity_polylines(entity)
        except Exception:
            return out
        for points, closed in polylines:
            data = _path_data(_simplify(points), closed, origin)
            if data:
                out.append(f'<path d="{data}"{stroke}/>')
    return out


def render_dxf_svg(dxf_path: Path | str, out_path: Path | str, *, frozen_layers_hidden: bool = True,
                   background: str = DEFAULT_BACKGROUND, force: bool = False) -> Path:
    """Render ``dxf_path`` to a vector ``out_path`` (SVG, one ``<g data-layer>`` per layer) plus ``preview_svg.json``.

    Coordinates are translated to a local origin (``origin`` in the metadata) so LV95 values do not hurt float
    precision in the browser; y points down. Stroke widths do not scale (``vector-effect``), text is real
    ``<text>``. Cached: nothing is rebuilt while the SVG is newer than the DXF."""
    dxf_path, out_path = Path(dxf_path), Path(out_path)
    if not force and svg_is_fresh(dxf_path, out_path) and load_svg_meta(out_path) is not None:
        return out_path
    doc = ezdxf.readfile(str(dxf_path))
    dark = _is_dark(_background_rgb(background))
    _lighten_black_in_doc(doc, dark)
    legend = _legend_from_doc(doc, dark)
    if not frozen_layers_hidden:
        legend = _legend_with_frozen(doc, legend, dark)
    names = {item["name"] for item in legend}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extent = _extent(doc, names) or (0.0, 0.0, 1.0, 1.0)
    origin = (float(math.floor(extent[0])), float(math.ceil(extent[3])))
    layer_color = {item["name"]: _hex_to_rgb(item["color"]) for item in legend}
    groups: dict[str, list[str]] = {name: [] for name in names}
    for entity in doc.modelspace():
        name = entity.dxf.layer
        if name in groups:
            groups[name].extend(_entity_svg(entity, origin, layer_color[name], dark))
    pad = 0.02 * max(extent[2] - extent[0], extent[3] - extent[1], 1e-6)
    viewbox = [extent[0] - origin[0] - pad, origin[1] - extent[3] - pad,
               extent[2] - extent[0] + 2 * pad, extent[3] - extent[1] + 2 * pad]
    body = "".join(
        f'<g data-layer="{_esc(item["name"])}" stroke="{item["color"]}" fill="none">' + "".join(groups[item["name"]]) + "</g>"
        for item in legend)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" class="samdxfp-svg" '
           f'viewBox="{" ".join(_num(v) for v in viewbox)}" preserveAspectRatio="xMidYMid meet" width="100%" height="100%">'
           f"<style>{_SVG_STYLE}</style>{body}</svg>")
    out_path.write_text(svg, encoding="utf-8")
    for item in legend:
        item["file"] = SVG_NAME
    meta = {"version": SVG_META_VERSION, "dxf": dxf_path.name, "origin": list(origin),
            "viewbox": viewbox, "background": background, "layers": legend}
    out_path.with_name(SVG_META_NAME).write_text(json.dumps(meta), encoding="utf-8")
    return out_path
