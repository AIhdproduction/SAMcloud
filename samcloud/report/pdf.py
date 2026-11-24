"""PDF of the processing protocol (reportlab Platypus): ``build_protocol_pdf(data, out_path, ...)``.

A4 portrait, title page, table of contents, 12 sections. Each section is rendered in its own try/except: a failing
section shows "no data" and logs a warning, the PDF is always produced. Values stay in source units in the data
layer; errors are converted to centimetres here. Traffic lights always carry a symbol (check, !, cross) next to
the tint, so colour is never the only signal.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any, Callable
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

from samcloud.report.charts import CHART_FILES
from samcloud.report.config import load_protocol_config
from samcloud.report.i18n_text import Translator, fmt_duration, fmt_int, fmt_num
from samcloud.report.textutil import scrub, short_name, summarize_value

LOGGER = logging.getLogger(__name__)

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
AVAIL_W = PAGE_W - 2 * MARGIN
INK, MUTED, RULE, ZEBRA, HEAD_BG = "#0b0b0b", "#52514e", "#d4d3cf", "#f5f5f4", "#ebeae6"
TINT = {"good": "#e1f3e1", "warn": "#fdf1d0", "bad": "#f7dcdc"}
SYMBOL = {"good": "✓", "warn": "!", "bad": "✗"}
SECTION_ORDER = ("title", "summary", "import_", "alignment", "control_points", "densify", "refine",
                 "classification", "refinement", "products", "runtimes", "appendix")
NOT_RUN_WITH_BODY = {"control_points", "refine", "refinement"}
MAX_LIST = 60
MAX_WARNING_LINES = 200
MAX_TABLE_ROWS = 200

_FONT_LOCK = threading.Lock()
_FONTS: tuple[str, str, str] | None = None


# ------------------------------------------------------------------------------------- basics


def _d(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _l(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _fonts() -> tuple[str, str, str]:
    """(regular, bold, italic) font names: DejaVu Sans from matplotlib's bundled fonts, else Helvetica."""
    global _FONTS
    with _FONT_LOCK:
        if _FONTS is not None:
            return _FONTS
        try:
            import matplotlib

            directory = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
            pdfmetrics.registerFont(TTFont("DejaVuSans", str(directory / "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(directory / "DejaVuSans-Bold.ttf")))
            pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", str(directory / "DejaVuSans-Oblique.ttf")))
            pdfmetrics.registerFont(TTFont("DejaVuSans-BoldOblique", str(directory / "DejaVuSans-BoldOblique.ttf")))
            pdfmetrics.registerFontFamily("DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold",
                                          italic="DejaVuSans-Oblique", boldItalic="DejaVuSans-BoldOblique")
            _FONTS = ("DejaVuSans", "DejaVuSans-Bold", "DejaVuSans-Oblique")
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("DejaVu fonts not available, using Helvetica: %s", error)
            _FONTS = ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique")
        return _FONTS


def app_version() -> str:
    try:
        import samcloud

        return str(getattr(samcloud, "__version__", "") or "")
    except Exception:  # noqa: BLE001
        return ""


class _Ctx:
    """Everything a section renderer needs: translator, formatting, styles, chart lookup, thresholds."""

    def __init__(self, data: Any, language: str, options: dict[str, Any], chart_dir: Path | None) -> None:
        self.data = data
        self.tr = Translator(language)
        self.language = self.tr.language
        self.options = options
        self.chart_dir = Path(chart_dir) if chart_dir else None
        config = load_protocol_config()
        self.thresholds = _d(config.get("thresholds"))
        sections_cfg = _d(config.get("sections"))
        self.camera_list = bool(options.get("camera_list", sections_cfg.get("camera_list", False)))
        self.class_details = bool(options.get("class_details", sections_cfg.get("class_details", True)))
        self.sections: dict[str, Any] = _d(getattr(data, "sections", None))
        regular, bold, italic = _fonts()
        self.font, self.bold, self.italic = regular, bold, italic
        base = dict(fontName=regular, textColor=colors.HexColor(INK))
        self.styles = {
            "body": ParagraphStyle("body", fontSize=8.5, leading=11.5, spaceAfter=3, **base),
            "small": ParagraphStyle("small", fontSize=7, leading=9, textColor=colors.HexColor(MUTED), fontName=regular, spaceAfter=3),
            "tiny": ParagraphStyle("tiny", fontSize=6.5, leading=8, textColor=colors.HexColor(MUTED), fontName=regular),
            "cell": ParagraphStyle("cell", fontSize=7.5, leading=9.5, alignment=TA_LEFT, **base),
            "cell_r": ParagraphStyle("cell_r", fontSize=7.5, leading=9.5, alignment=TA_RIGHT, **base),
            "head": ParagraphStyle("head", fontSize=7.5, leading=9.5, alignment=TA_LEFT, fontName=bold, textColor=colors.HexColor(INK)),
            "head_r": ParagraphStyle("head_r", fontSize=7.5, leading=9.5, alignment=TA_RIGHT, fontName=bold, textColor=colors.HexColor(INK)),
            "h1": ParagraphStyle("h1", fontSize=15, leading=19, fontName=bold, textColor=colors.HexColor(INK), spaceBefore=4,
                                 spaceAfter=7, keepWithNext=1),
            "h1_plain": ParagraphStyle("h1_plain", fontSize=15, leading=19, fontName=bold, textColor=colors.HexColor(INK), spaceAfter=10),
            "h2": ParagraphStyle("h2", fontSize=10, leading=13, fontName=bold, textColor=colors.HexColor(INK), spaceBefore=8,
                                 spaceAfter=3, keepWithNext=1),
            "title": ParagraphStyle("title", fontSize=26, leading=31, fontName=bold, textColor=colors.HexColor(INK), spaceAfter=4),
            "subtitle": ParagraphStyle("subtitle", fontSize=15, leading=19, fontName=regular, textColor=colors.HexColor(MUTED), spaceAfter=14),
            "toc": ParagraphStyle("toc", fontSize=9.5, leading=15, fontName=regular, textColor=colors.HexColor(INK)),
        }

    # formatting ---------------------------------------------------------------------------------
    def n(self, value: Any, decimals: int = 2) -> str:
        return fmt_num(self.tr, value, decimals)

    def i(self, value: Any) -> str:
        return fmt_int(self.tr, value)

    def cm(self, value_m: Any, decimals: int = 2) -> str:
        number = _num(value_m)
        return "–" if number is None else fmt_num(self.tr, number * 100.0, decimals)

    def duration(self, seconds: Any) -> str:
        number = _num(seconds)
        if number is None:
            return "–"
        return f"{fmt_duration(number)} ({fmt_num(self.tr, number, 0 if number >= 100 else 1)} s)"

    def yes_no(self, value: Any) -> str:
        return self.tr("status.yes") if value else self.tr("status.no")

    def lookup(self, prefix: str, value: Any) -> str:
        return self.tr.name(prefix, value)

    # not-recorded handling ------------------------------------------------------------------------
    def missing(self, data: dict[str, Any], key: str) -> str:
        """Placeholder for a None value: "not recorded" when the collector listed it, else a dash."""
        applicable = _d(data.get("not_applicable"))
        code = applicable.get(key) or applicable.get(key.split(".", 1)[0])
        if code:
            return self.tr(f"na.{code}") if self.tr.has(f"na.{code}") else self.tr("na.not_applicable")
        return self.tr("status.not_recorded") if key in _l(data.get("not_recorded")) else "–"

    def value(self, data: dict[str, Any], key: str, formatter: Callable[[Any], str]) -> str:
        """``data[key]`` (dotted for one nested level) formatted, or the not-recorded placeholder."""
        raw: Any = data
        for part in key.split("."):
            raw = _d(raw).get(part) if isinstance(raw, dict) else None
        if raw is None:
            return self.missing(data, key)
        return formatter(raw)

    # thresholds / lights --------------------------------------------------------------------------
    def thr(self, name: str) -> tuple[float, float] | None:
        entry = _d(self.thresholds.get(name))
        green, yellow = _num(entry.get("green")), _num(entry.get("yellow"))
        return (green, yellow) if green is not None and yellow is not None else None

    def level_upper(self, name: str, value: Any) -> str | None:
        number, limits = _num(value), self.thr(name)
        if number is None or limits is None:
            return None
        return "good" if number <= limits[0] else "warn" if number <= limits[1] else "bad"

    def level_lower(self, name: str, value: Any) -> str | None:
        number, limits = _num(value), self.thr(name)
        if number is None or limits is None:
            return None
        return "good" if number >= limits[0] else "warn" if number >= limits[1] else "bad"

    def lit(self, text: str, level: str | None) -> str:
        return f"{SYMBOL[level]} {text}" if level else text

    # flowables -------------------------------------------------------------------------------------
    def p(self, text: Any, style: str = "body") -> Paragraph:
        return Paragraph(escape(scrub(text)), self.styles[style])

    def h2(self, text: str) -> Paragraph:
        return Paragraph(escape(text), self.styles["h2"])

    def small(self, text: str) -> Paragraph:
        return self.p(text, "small")

    def more(self, count: int) -> Paragraph:
        return self.small(self.tr("common.more", count=self.i(count)))

    def table(self, header: list[str], rows: list[list[Any]], weights: list[float], aligns: str,
              lights: dict[tuple[int, int], str] | None = None, repeat: int = 1) -> Table:
        """Table with a bold header row (units in the header text), zebra rows, wrapped cells and tinted light cells."""

        def cell(value: Any, column: int, head: bool = False) -> Any:
            if not isinstance(value, str):
                return value
            right = aligns[column:column + 1] == "r"
            style = ("head_r" if right else "head") if head else ("cell_r" if right else "cell")
            return Paragraph(escape(scrub(value)), self.styles[style])

        body = [[cell(c, j) for j, c in enumerate(row)] for row in rows]
        total = sum(weights) or 1.0
        widths = [AVAIL_W * w / total for w in weights]
        table = Table([[cell(h, j, True) for j, h in enumerate(header)], *body], colWidths=widths, repeatRows=repeat)
        commands: list[Any] = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(HEAD_BG)),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#9a9992")),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor(RULE)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
            ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ]
        if len(rows) > 1:
            commands.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(ZEBRA)]))
        for (row_index, column), level in (lights or {}).items():
            if level in TINT:
                commands.append(("BACKGROUND", (column, row_index + 1), (column, row_index + 1), colors.HexColor(TINT[level])))
        table.setStyle(TableStyle(commands))
        return table

    def kv(self, rows: list[tuple[str, str, str | None]], with_rating: bool = False) -> Table:
        """Two-column key/value table; with ``with_rating`` a third column carries the traffic-light label."""
        lights: dict[tuple[int, int], str] = {}
        body: list[list[Any]] = []
        for index, (label, value, level) in enumerate(rows):
            if with_rating:
                body.append([label, value, f"{SYMBOL[level]} {self.tr(f'light.{level}')}" if level else ""])
                if level:
                    lights[(index, 2)] = level
                    lights[(index, 1)] = level
            else:
                body.append([label, value])
        if with_rating:
            header = [self.tr("common.parameter"), self.tr("common.value"), self.tr("common.rating")]
            return self.table(header, body, [5, 3.2, 1.8], "lrl", lights)
        return self.table([self.tr("common.parameter"), self.tr("common.value")], body, [5, 5], "ll", lights)

    def chart(self, key: str, max_height_mm: float = 150) -> Image | None:
        if self.chart_dir is None or key not in CHART_FILES:
            return None
        path = self.chart_dir / CHART_FILES[key]
        if not path.is_file():
            return None
        try:
            from PIL import Image as PilImage

            with PilImage.open(path) as picture:
                width, height = picture.size
            if width <= 0 or height <= 0:
                return None
            scale = min(AVAIL_W / width, (max_height_mm * mm) / height)
            return Image(str(path), width=width * scale, height=height * scale)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("chart %s not usable in the PDF: %s", path, error)
            return None

    def titled_chart(self, title: str, key: str, max_height_mm: float = 150) -> list[Any]:
        image = self.chart(key, max_height_mm)
        return [self.h2(title), image, Spacer(1, 4)] if image is not None else []

    def no_data(self) -> Paragraph:
        return self.p(self.tr("status.no_data"), "body")


# ------------------------------------------------------------------------------------ sections


def sec_summary(ctx: _Ctx, sec: Any) -> list[Any]:
    data = _d(sec.data)
    tr = ctx.tr
    control = ctx.sections.get("control_points")
    without = (getattr(control, "status", None) == "not_run"
               and _d(getattr(control, "data", None)).get("reason") == "without_control_points")
    registered, total = _num(data.get("images_registered")), _num(data.get("images_total"))
    pct = registered / total * 100.0 if registered is not None and total else None
    reg_text = (f"{ctx.i(registered)} / {ctx.i(total)} ({ctx.n(pct, 1)} %)" if registered is not None
                else ctx.missing(data, "images_registered"))
    rows: list[tuple[str, str, str | None]] = [
        (tr("summary.registered"), reg_text, ctx.level_lower("registered_pct", pct)),
        (tr("summary.reproj"), ctx.value(data, "reprojection_error_px", ctx.n), ctx.level_upper("reprojection_px", data.get("reprojection_error_px"))),
    ]
    for prefix, threshold in (("rms_pass", "pass_rms_cm"), ("rms_control", "control_rms_cm")):
        if without and all(data.get(f"{prefix}_{axis}_m") is None for axis in ("x", "y", "z", "3d")):
            rows.append((tr(f"summary.{prefix}", axis="X/Y/Z/3D"), tr("status.without_control_points"), None))
            continue
        for axis in ("x", "y", "z", "3d"):
            key = f"{prefix}_{axis}_m"
            raw = data.get(key)
            if raw is None and without:
                text = tr("status.without_control_points")
            else:
                text = ctx.value(data, key, lambda v: ctx.cm(v))
            level = ctx.level_upper(threshold, _num(raw) * 100.0 if _num(raw) is not None else None) if axis == "3d" else None
            rows.append((tr(f"summary.{prefix}", axis=axis.upper()), text, level))
    rows.append((tr("summary.scale"), ctx.value(data, "apriltag_scale_factor", lambda v: ctx.n(v, 4)), None))
    rows.append((tr("summary.tag_dev"), ctx.value(data, "apriltag_max_deviation_pct", lambda v: ctx.n(v, 2)),
                 ctx.level_upper("apriltag_deviation_pct", data.get("apriltag_max_deviation_pct"))))
    rows.append((tr("summary.dense"), ctx.value(data, "dense_points", ctx.i), None))
    rows.append((tr("summary.filtered"), ctx.value(data, "filtered_points", ctx.i), None))
    rows.append((tr("summary.total_time"), ctx.value(data, "total_seconds", ctx.duration), None))
    return [ctx.kv(rows, with_rating=True), Spacer(1, 3), ctx.small(tr("light.legend"))]


def sec_import(ctx: _Ctx, sec: Any) -> list[Any]:
    data = _d(sec.data)
    tr = ctx.tr
    rows: list[tuple[str, str, str | None]] = [
        (tr("import.source"), ctx.value(data, "source", lambda v: ctx.lookup("capture", v)), None),
        (tr("import.images"), ctx.value(data, "image_count", ctx.i), None),
    ]
    if data.get("frame_count") is not None or data.get("source") in ("video", "panorama"):
        rows.append((tr("import.frames"), ctx.value(data, "frame_count", ctx.i), None))
        rows.append((tr("import.interval"), ctx.value(data, "frame_interval_s", lambda v: ctx.n(v, 2)), None))
    cube = _d(data.get("panorama_to_cubemap"))
    if cube:
        rows.append((tr("import.cubemap"), tr("import.cubemap_value", panoramas=ctx.i(cube.get("panoramas")),
                                              faces=ctx.i(cube.get("faces")), size=ctx.i(cube.get("face_size"))), None))
    camera = data.get("camera")
    if isinstance(camera, dict):
        name = " ".join(str(camera.get(k)) for k in ("make", "model") if camera.get(k))
        rows.append((tr("import.camera"), name or "–", None))
        rows.append((tr("import.focal"), ctx.n(_num(camera.get("focal_mm")), 2), None))
        rows.append((tr("import.focal35"), str(camera.get("focal_35mm") or "–"), None))
        sensor = _d(camera.get("sensor"))
        if sensor:
            rows.append((tr("import.sensor"), f"{sensor.get('pixel_width', '–')} × {sensor.get('pixel_height', '–')}", None))
    elif "camera" in _l(data.get("not_recorded")):
        rows.append((tr("import.camera"), tr("status.not_recorded"), None))
    gps = _d(data.get("gps"))
    if gps and any(v is not None for v in gps.values()):
        rows.append((tr("import.gps"), ctx.lookup("gps", gps.get("mode")) if gps.get("mode") is not None else "–", None))
        if gps.get("rtk") is not None:
            rows.append((tr("import.rtk"), ctx.yes_no(gps.get("rtk")), None))
        rows.append((tr("import.gps_error"), ctx.n(_num(gps.get("max_error_m")), 2), None))
    discarded = _l(data.get("discarded_frames"))
    count = data.get("discarded_count")
    if count is not None and data.get("source") in ("video", "panorama"):
        rows.append((tr("import.discarded"), ctx.i(count), None))
    story: list[Any] = [ctx.kv(rows)]
    if discarded:
        table_rows = []
        for item in discarded[:15]:
            item = _d(item)
            table_rows.append([short_name(item.get("kept_frame")) or "–", ctx.n(_num(item.get("time_seconds")), 2),
                               ctx.n(_num(item.get("sharpness")), 1), ctx.n(_num(item.get("kept_sharpness")), 1)])
        story += [ctx.h2(tr("import.discarded_table")),
                  ctx.table([tr("import.kept_frame"), tr("import.time"), tr("import.sharpness"), tr("import.kept_sharpness")],
                            table_rows, [5, 1.5, 1.5, 1.5], "lrrr")]
        if len(discarded) > 15:
            story.append(ctx.more(len(discarded) - 15))
    return story


def _safe_block(ctx: _Ctx, name: str, build: Callable[[], list[Any]]) -> list[Any]:
    try:
        return build()
    except Exception as error:  # noqa: BLE001 - one block never removes the whole section
        LOGGER.warning("protocol block %s failed: %s", name, error, exc_info=True)
        return []


def sec_alignment(ctx: _Ctx, sec: Any) -> list[Any]:
    data = _d(sec.data)
    tr = ctx.tr
    story: list[Any] = []

    def key_table() -> list[Any]:
        registered, total = _num(data.get("registered")), _num(data.get("total"))
        pct = _num(data.get("registered_pct"))
        rows: list[tuple[str, str, str | None]] = [
            (tr("align.registered"), f"{ctx.i(registered)} / {ctx.i(total)}" if registered is not None else ctx.missing(data, "registered"), None),
            (tr("align.registered_pct"), ctx.value(data, "registered_pct", lambda v: ctx.n(v, 1)), ctx.level_lower("registered_pct", pct)),
            (tr("align.unregistered"), ctx.i(len(_l(data.get("unregistered_images")))), None),
            (tr("align.submodels"), ctx.value(data, "submodels", ctx.i), None),
            (tr("align.tie_points"), ctx.value(data, "tie_points", ctx.i), None),
            (tr("align.track"), ctx.value(data, "mean_track_length", lambda v: ctx.n(v, 2)), None),
            (tr("align.reproj"), ctx.value(data, "reprojection_error_px", ctx.n), ctx.level_upper("reprojection_px", data.get("reprojection_error_px"))),
            (tr("align.matching"), str(data.get("matching_strategy") or ctx.missing(data, "matching_strategy")), None),
            (tr("align.camera_model"), str(data.get("camera_model") or ctx.missing(data, "camera_model")), None),
        ]
        return [ctx.kv(rows, with_rating=True)]

    def passes() -> list[Any]:
        rows = []
        for item in _l(data.get("tiepoint_filter_passes")):
            item = _d(item)
            before, after = _d(item.get("before")), _d(item.get("after_filter"))
            rows.append([str(item.get("pass", "")), ctx.i(_num(before.get("points"))), ctx.i(_num(after.get("points"))),
                         ctx.n(_num(before.get("mean_reprojection_error")), 3), ctx.n(_num(after.get("mean_reprojection_error")), 3)])
        if not rows:
            return []
        return [ctx.h2(tr("align.passes")),
                ctx.table([tr("align.pass"), tr("align.points_before"), tr("align.points_after"), tr("align.reproj_before"),
                           tr("align.reproj_after")], rows, [1, 2, 2, 2, 2], "lrrrr")]

    def calibration() -> list[Any]:
        cameras = _l(data.get("cameras"))
        rows = []
        for camera in cameras[:12]:
            camera = _d(camera)
            params = _d(camera.get("params"))
            focal = _num(params.get("focal_px"))
            if focal is None:
                focal = _num(params.get("fx"))
            other = "; ".join(f"{k}={ctx.n(_num(v), 5)}" for k, v in params.items()
                              if k not in ("fx", "fy", "cx", "cy", "focal_px") and _num(v) is not None)
            rows.append([str(camera.get("camera_id", "")), str(camera.get("model") or "–"),
                         f"{ctx.i(_num(camera.get('width')))} × {ctx.i(_num(camera.get('height')))}", ctx.n(focal, 2),
                         ctx.n(_num(params.get("cx")), 2), ctx.n(_num(params.get("cy")), 2), other or "–"])
        if not rows:
            return []
        out = [ctx.h2(tr("align.calib")),
               ctx.table([tr("align.cam_id"), tr("align.model"), tr("align.size"), tr("align.focal"), tr("align.cx"),
                          tr("align.cy"), tr("align.other")], rows, [1.4, 1.8, 2.2, 1.8, 1.5, 1.5, 3.2], "llrrrrl")]
        if len(cameras) > 12:
            out.append(ctx.more(len(cameras) - 12))
        return out

    def unregistered() -> list[Any]:
        names = [short_name(n) for n in _l(data.get("unregistered_images"))]
        if not names:
            return []
        shown = ", ".join(names[:MAX_LIST])
        out = [ctx.h2(tr("align.unregistered_list")), ctx.p(shown, "small")]
        if len(names) > MAX_LIST:
            out.append(ctx.more(len(names) - MAX_LIST))
        return out

    story += _safe_block(ctx, "alignment.key", key_table)
    story += _safe_block(ctx, "alignment.passes", passes)
    story += _safe_block(ctx, "alignment.calibration", calibration)
    story += ctx.titled_chart(tr("align.camera_map"), "camera_map", 110)
    story += ctx.titled_chart(tr("align.overlap"), "overlap", 110)
    story += _safe_block(ctx, "alignment.unregistered", unregistered)
    return story


def sec_control_points(ctx: _Ctx, sec: Any) -> list[Any]:
    data = _d(sec.data)
    tr = ctx.tr
    story: list[Any] = []
    if data.get("reason") == "without_control_points":
        story.append(ctx.p(tr("cp.without")))

    def point_table() -> list[Any]:
        points = [_d(p) for p in _l(data.get("points"))]
        if not points:
            return []
        rows = []
        for item in points[:MAX_TABLE_ROWS]:
            role = item.get("role")
            rows.append([str(item.get("id", "")), ctx.lookup("cp", item.get("type") or "target") if item.get("type") in ("target", "apriltag") else str(item.get("type") or "–"),
                         tr("cp.control") if role == "control" else tr("cp.pass") if role == "pass" else "–",
                         ctx.cm(item.get("error_x_m")), ctx.cm(item.get("error_y_m")), ctx.cm(item.get("error_z_m")),
                         ctx.cm(item.get("error_3d_m")), ctx.i(_num(item.get("images")))])
        out = [ctx.h2(tr("cp.points")),
               ctx.table([tr("common.id"), tr("cp.type"), tr("common.role"), tr("cp.ex"), tr("cp.ey"), tr("cp.ez"), tr("cp.e3d"),
                          tr("cp.images")], rows, [2, 1.8, 2.2, 1.7, 1.7, 1.7, 1.9, 1.5], "lllrrrrr")]
        if len(points) > MAX_TABLE_ROWS:
            out.append(ctx.more(len(points) - MAX_TABLE_ROWS))
        return out

    def rms_table() -> list[Any]:
        rows, lights = [], {}
        for index, (role, key, threshold) in enumerate((("cp.pass", "rms_pass", "pass_rms_cm"), ("cp.control", "rms_control", "control_rms_cm"))):
            block = _d(data.get(key))
            if not block:
                continue
            total = _num(block.get("3d_m"))
            level = ctx.level_upper(threshold, total * 100.0 if total is not None else None)
            rows.append([tr(role), ctx.cm(block.get("x_m")), ctx.cm(block.get("y_m")), ctx.cm(block.get("z_m")),
                         ctx.cm(block.get("xy_m")), ctx.lit(ctx.cm(total), level), ctx.n(_num(block.get("mean_reprojection_px")), 2),
                         ctx.i(_num(block.get("count")))])
            if level:
                lights[(len(rows) - 1, 5)] = level
        if not rows:
            return []
        return [ctx.h2(tr("cp.rms")),
                ctx.table([tr("common.role"), tr("cp.rms_x"), tr("cp.rms_y"), tr("cp.rms_z"), tr("cp.rms_xy"), tr("cp.rms_3d"),
                           tr("cp.rms_reproj"), tr("cp.rms_count")], rows, [2.2, 1.5, 1.5, 1.5, 1.5, 1.9, 1.8, 1.5], "lrrrrrrr", lights),
                ctx.small(tr("light.legend"))]

    def shift() -> list[Any]:
        value = _num(data.get("height_shift_m"))
        out: list[Any] = []
        if value is not None:
            applied = data.get("height_shift_applied")
            note = tr("cp.applied") if applied else tr("cp.not_applied") if applied is not None else ""
            out.append(ctx.kv([(tr("cp.height_shift_value"), f"{ctx.n(value, 3)} {note}".strip(), None)]))
        compare = _d(data.get("gcp_vs_shift"))
        rows = []
        for key, label in (("pre_gcp", "cp.pre_gcp"), ("post_gcp", "cp.post_gcp")):
            block = _d(compare.get(key))
            if block:
                rows.append([tr(label), ctx.cm(block.get("rms_e_m")), ctx.cm(block.get("rms_n_m")), ctx.cm(block.get("rms_h_m")),
                             ctx.cm(block.get("rms_2d_m")), ctx.cm(block.get("rms_3d_m")), ctx.cm(block.get("bias_h_m")),
                             ctx.i(_num(block.get("n_points")))])
        if rows:
            out += [ctx.h2(tr("cp.gcp_vs_shift")),
                    ctx.table([tr("cp.stage"), tr("cp.rms_e"), tr("cp.rms_n"), tr("cp.rms_h"), tr("cp.rms_2d"), tr("cp.rms_3d"),
                               tr("cp.bias_h"), tr("cp.n_points")], rows, [3.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4], "lrrrrrrr")]
        return out

    def checkpoints() -> list[Any]:
        out: list[Any] = []
        for name, entry in _d(data.get("checkpoints")).items():
            pts = [_d(p) for p in _l(_d(entry).get("points"))]
            if not pts:
                continue
            rows = [[str(p.get("name", "")), ctx.cm(p.get("d_e_m")), ctx.cm(p.get("d_n_m")), ctx.cm(p.get("d_h_m")),
                     ctx.cm(p.get("d_3d_m")), str(p.get("status") or "–")] for p in pts[:50]]
            out += [ctx.h2(f"{tr('cp.checkpoints')}: {name}"),
                    ctx.table([tr("common.id"), tr("cp.d_e"), tr("cp.d_n"), tr("cp.d_h"), tr("cp.d_3d"), tr("common.status")],
                              rows, [2, 1.8, 1.8, 1.8, 1.8, 1.8], "lrrrrl")]
            if len(pts) > 50:
                out.append(ctx.more(len(pts) - 50))
        return out

    def apriltag() -> list[Any]:
        april = _d(data.get("apriltag"))
        if not april:
            return []
        out: list[Any] = [ctx.h2(tr("cp.tags")), ctx.kv([
            (tr("cp.tag_mode"), str(april.get("mode") or "–"), None),
            (tr("cp.tag_factor"), ctx.n(_num(april.get("scale_factor")), 4), None),
            (tr("cp.tag_applied"), ctx.n(_num(april.get("scale_applied")), 4), None)])]
        rows, lights = [], {}
        for tag in _l(april.get("tags")):
            tag = _d(tag)
            deviation = _num(tag.get("deviation_pct"))
            level = ctx.level_upper("apriltag_deviation_pct", abs(deviation) if deviation is not None else None)
            rows.append([str(tag.get("id", "")), ctx.i(_num(tag.get("images"))), ctx.n(_num(tag.get("size_mm_before")), 1),
                         ctx.n(_num(tag.get("size_mm_after")), 1), ctx.lit(ctx.n(deviation, 2), level)])
            if level:
                lights[(len(rows) - 1, 4)] = level
        if rows:
            out += [Spacer(1, 4), ctx.table([tr("cp.tag_id"), tr("cp.tag_images"), tr("cp.tag_before"), tr("cp.tag_after"), tr("cp.tag_dev")],
                                            rows, [1.2, 1.5, 2.2, 2.2, 2], "lrrrr", lights)]
        warnings = [str(w) for w in _l(april.get("warnings"))]
        if warnings:
            out += [ctx.h2(tr("cp.tag_warnings")), *[ctx.small(w) for w in warnings[:20]]]
        return out

    def excluded() -> list[Any]:
        rows = []
        for item in _l(data.get("points")):
            item = _d(item)
            reason = item.get("excluded_reason")
            if reason:
                rows.append([str(item.get("id", "")), tr("cp.reason_disabled") if reason == "disabled" else tr.reason(reason)])
        if not rows:
            return []
        return [ctx.h2(tr("cp.excluded")), ctx.table([tr("common.id"), tr("common.reason")], rows[:MAX_TABLE_ROWS], [2, 8], "ll")]

    def gate() -> list[Any]:
        info = _d(data.get("gate"))
        if info.get("passed") is None:
            return []
        text = tr("cp.gate_passed") if info.get("passed") else tr("cp.gate_failed")
        reasons = "; ".join(str(r) for r in _l(info.get("reasons")))
        return [ctx.kv([(tr("cp.gate"), ctx.lit(text, "good" if info.get("passed") else "bad") + (f" ({reasons})" if reasons else ""), None)])]

    for name, builder in (("points", point_table), ("rms", rms_table), ("shift", shift), ("checkpoints", checkpoints),
                          ("apriltag", apriltag), ("excluded", excluded), ("gate", gate)):
        story += _safe_block(ctx, f"control_points.{name}", builder)
    story += ctx.titled_chart(tr("cp.vectors"), "error_vectors", 120)
    return story


def sec_densify(ctx: _Ctx, sec: Any) -> list[Any]:
    data = _d(sec.data)
    tr = ctx.tr
    settings = _d(data.get("settings"))
    density = _num(data.get("density_pts_m2"))
    rows: list[tuple[str, str, str | None]] = [
        (tr("densify.points"), ctx.value(data, "points", ctx.i), None),
        (tr("densify.area"), ctx.value(data, "area_m2", lambda v: ctx.n(v, 1)), None),
        (tr("densify.density"), ctx.n(density, 1) if density is not None else (tr.reason(data.get("density_reason")) if data.get("density_reason") else ctx.missing(data, "density_pts_m2")), None),
    ]
    if data.get("density_cloud"):
        rows.append((tr("densify.cloud"), str(data.get("density_cloud")), None))
    rows += [
        (tr("densify.max_size"), ctx.i(_num(settings.get("max_image_size"))), None),
        (tr("densify.iterations"), ctx.i(_num(settings.get("num_iterations"))), None),
    ]
    if settings.get("size_reason"):
        rows.append((tr("densify.size_reason"), tr.reason(settings.get("size_reason")), None))
    rows.append((tr("densify.excluded"), ctx.i(len(_l(settings.get("excluded_images")))), None))
    rows.append((tr("densify.seconds"), ctx.value(data, "seconds", ctx.duration), None))
    return [ctx.kv(rows)]


def sec_refine(ctx: _Ctx, sec: Any) -> list[Any]:
    data = _d(sec.data)
    tr = ctx.tr
    rows: list[tuple[str, str, str | None]] = [(tr("refine.executed"), ctx.yes_no(data.get("executed")), None)]
    if data.get("skipped_reason"):
        rows.append((tr("refine.reason"), tr.reason(data.get("skipped_reason")), None))
    metrics = _d(data.get("metrics"))
    if metrics:
        rows += [
            (tr("refine.unknown_before"), ctx.i(_num(metrics.get("unknown_before"))), None),
            (tr("refine.unknown_after"), ctx.i(_num(metrics.get("unknown_after"))), None),
            (tr("refine.set_unknown"), ctx.i(_num(metrics.get("set_to_unknown"))), None),
            (tr("refine.smoothed"), ctx.i(_num(metrics.get("smoothed_points"))), None),
            (tr("refine.runtime"), ctx.duration(metrics.get("runtime_seconds")), None),
        ]
    return [ctx.kv(rows)]


def sec_classification(ctx: _Ctx, sec: Any) -> list[Any]:
    data = _d(sec.data)
    tr = ctx.tr
    thresholds = _d(data.get("thresholds"))
    total, unknown = _num(data.get("total_points")), _num(data.get("unknown_points"))
    unknown_text = ctx.i(unknown)
    if unknown is not None and total:
        unknown_text += f" ({ctx.n(unknown / total * 100.0, 1)} %)"
    rows: list[tuple[str, str, str | None]] = [
        (tr("cls.model"), str(data.get("model") or ctx.missing(data, "model")), None),
        (tr("cls.engine"), str(data.get("engine") or ctx.missing(data, "engine")), None),
        (tr("cls.thr_bg"), ctx.n(_num(thresholds.get("background")), 2), None),
        (tr("cls.thr_query"), ctx.n(_num(thresholds.get("query")), 2), None),
        (tr("cls.total"), ctx.i(total), None),
        (tr("cls.unknown"), unknown_text, None),
    ]
    histogram = _d(data.get("confidence_histogram"))
    if _num(histogram.get("without_own_score")) is not None:
        rows.append((tr("cls.no_own_score"), ctx.i(_num(histogram.get("without_own_score"))), None))
    if _num(histogram.get("mean_conf")) is not None:
        rows.append((tr("cls.mean_conf"), ctx.n(_num(histogram.get("mean_conf")), 3), None))
    story: list[Any] = [ctx.kv(rows)]
    classes = [_d(c) for c in _l(data.get("classes"))]
    if ctx.class_details and classes:
        table_rows = [[ctx.tr.cls(c.get("name")), ctx.i(_num(c.get("points"))), ctx.n(_num(c.get("share_pct")), 1),
                       ctx.i(_num(c.get("instances"))) if c.get("instances") is not None else "–"] for c in classes[:MAX_TABLE_ROWS]]
        story += [ctx.h2(tr("cls.classes")), ctx.table([tr("cls.class"), tr("cls.points"), tr("cls.share"), tr("cls.instances")],
                                                        table_rows, [4, 2.2, 1.6, 2], "lrrr")]
        percentiles = _d(data.get("per_class_percentiles"))
        if percentiles:
            prow = [[ctx.tr.cls(name), ctx.n(_num(_d(v).get("p10")), 2), ctx.n(_num(_d(v).get("p50")), 2), ctx.n(_num(_d(v).get("p90")), 2),
                     ctx.i(_num(_d(v).get("points")))] for name, v in list(percentiles.items())[:MAX_TABLE_ROWS]]
            story += [ctx.h2(tr("cls.percentiles")), ctx.table([tr("cls.class"), tr("cls.p10"), tr("cls.p50"), tr("cls.p90"), tr("cls.points")],
                                                               prow, [4, 1.5, 1.5, 1.5, 2.2], "lrrrr")]
    story += ctx.titled_chart(tr("cls.chart"), "classes", 150)
    story += ctx.titled_chart(tr("cls.hist"), "confidence", 90)
    if histogram and _num(histogram.get("without_own_score")) is not None:
        story.append(ctx.small(tr("cls.hist_note")))
    return story


def _mio(ctx: _Ctx, value: Any) -> str:
    number = _num(value)
    return ctx.n(number / 1_000_000.0, 1) if number is not None else "\u2013"


def sec_refinement(ctx: _Ctx, sec: Any) -> list[Any]:
    """Top-level stages as rows; surface-fusion removals as indented sub-rows (counted once); mirror detection shows
    only the mirror count; depth evidence is a hint, not points added to the cloud."""
    data = _d(sec.data)
    tr = ctx.tr
    stages = [_d(stage) for stage in _l(data.get("stages"))]
    top = [stage for stage in stages if "." not in str(stage.get("stage"))]
    subs = [stage for stage in stages if "." in str(stage.get("stage"))]
    rows: list[list[Any]] = []

    def sub_rows() -> list[list[Any]]:
        return [[f"\u21b3 {tr.stage(sub.get('stage'))}", "\u2013", "\u2013", ctx.i(_num(sub.get("removed"))), "\u2013", "\u2013", ""]
                for sub in subs]

    placed = False
    for stage in top:
        name = str(stage.get("stage"))
        removed, added, note = stage.get("removed"), stage.get("added"), ""
        if stage.get("skipped"):
            note = f"{tr('ref.skipped')}: {tr.reason(stage.get('reason'))}" if stage.get("reason") else tr("ref.skipped")
        elif name == "mirror_detection":
            removed = added = None
            note = tr("ref.mirrors", n=ctx.i(_num(stage.get("detected")))) if _num(stage.get("detected")) is not None else ""
        elif name == "depth_evidence":
            note = tr("ref.evidence", count=_mio(ctx, added))
            added = None
        rows.append([tr.stage(name), ctx.i(_num(stage.get("before"))), ctx.i(_num(stage.get("after"))), ctx.i(_num(removed)),
                     ctx.i(_num(added)), ctx.n(_num(stage.get("seconds")), 1), note])
        if name == "surface_fusion":
            rows += sub_rows()
            placed = True
    if subs and not placed:
        rows += sub_rows()
    story: list[Any] = []
    if rows:
        story.append(ctx.table([tr("ref.stage"), tr("ref.before"), tr("ref.after"), tr("ref.removed"), tr("ref.added"),
                                tr("ref.seconds"), tr("ref.note")], rows[:MAX_TABLE_ROWS], [3.5, 1.5, 1.5, 1.5, 1.9, 1.2, 3.1], "lrrrrrl"))
    if _num(data.get("final_points")) is not None:
        story += [Spacer(1, 4), ctx.kv([(tr("ref.final"), ctx.i(_num(data.get("final_points"))), None)])]
    story += ctx.titled_chart(tr("ref.chart"), "refinement", 120)
    return story


def sec_products(ctx: _Ctx, sec: Any) -> list[Any]:
    data = _d(sec.data)
    tr = ctx.tr
    story: list[Any] = []
    files = [_d(f) for f in _l(data.get("files"))]
    if files:
        rows = [[str(f.get("path", "")), ctx.lookup("kind", f.get("kind")), ctx.n(_num(f.get("size_bytes")) / 1_000_000 if _num(f.get("size_bytes")) is not None else None, 1),
                 str(f.get("crs") or "–")] for f in files[:MAX_TABLE_ROWS]]
        story.append(ctx.table([tr("prod.file"), tr("prod.kind"), tr("prod.size"), tr("prod.crs")], rows, [5.4, 2.6, 1.3, 1.5], "llrl"))
    qa = _d(data.get("dxf_qa"))
    if qa:
        story += [ctx.h2(tr("prod.qa")), ctx.p(tr("prod.qa_summary", passed=ctx.i(_num(qa.get("passed"))), failed=ctx.i(_num(qa.get("failed")))))]
        rules, lights = [], {}
        for rule in _l(qa.get("rules")):
            rule = _d(rule)
            level = "good" if rule.get("ok") is True else "bad" if rule.get("ok") is False else None
            rules.append([tr.name("qa_rule", rule.get("rule", "")), ctx.lit(tr("prod.ok") if rule.get("ok") else "–" if rule.get("ok") is None else tr("status.no"), level),
                          summarize_value(rule.get("value"), ctx.n, lambda n: tr("qa.entries", n=ctx.i(n))),
                          summarize_value(rule.get("threshold"), ctx.n, lambda n: tr("qa.entries", n=ctx.i(n)))])
            if level:
                lights[(len(rules) - 1, 1)] = level
        if rules:
            story.append(ctx.table([tr("common.rule"), tr("common.result"), tr("prod.value"), tr("prod.threshold")],
                                   rules[:MAX_TABLE_ROWS], [3.4, 1.5, 2.4, 3], "llll", lights))
    return story


def sec_runtimes(ctx: _Ctx, sec: Any) -> list[Any]:
    data = _d(sec.data)
    tr = ctx.tr
    story: list[Any] = []
    steps = [_d(s) for s in _l(data.get("steps"))]
    if steps:
        rows = [[tr.step(s.get("name", "")), ctx.n(_num(s.get("seconds")), 1), fmt_duration(s.get("seconds"))] for s in steps[:MAX_TABLE_ROWS]]
        story.append(ctx.table([tr("run.step"), tr("run.seconds"), tr("run.duration")], rows, [6, 1.8, 1.8], "lrr"))
    story += [Spacer(1, 4), ctx.kv([(tr("run.total"), ctx.value(data, "total_seconds", ctx.duration), None)])]
    hardware = _d(data.get("hardware"))
    if hardware:
        rows = [(tr("run.gpu"), str(hardware.get("gpu_name") or "–"), None),
                (tr("run.vram"), ctx.n(_num(hardware.get("vram_gb")), 1), None),
                (tr("run.ram"), ctx.n(_num(hardware.get("ram_gb")), 1), None),
                (tr("run.cpu"), str(hardware.get("cpu") or "–"), None),
                (tr("run.cores"), ctx.i(_num(hardware.get("cpu_count"))), None),
                (tr("run.os"), str(hardware.get("os") or "–"), None)]
        story += [ctx.h2(tr("run.hardware")), ctx.kv(rows)]
    story += ctx.titled_chart(tr("run.chart"), "runtimes", 200)
    return story


def sec_appendix(ctx: _Ctx, sec: Any) -> list[Any]:
    data = _d(sec.data)
    tr = ctx.tr
    story: list[Any] = []
    images = [_d(i) for i in _l(data.get("cameras_per_image"))]
    if ctx.camera_list and images:
        def short(name: Any) -> str:
            text = short_name(name)
            return text if len(text) <= 38 else "…" + text[-37:]

        cells = [[short(i.get("name")), str(i.get("camera_id", "")), ctx.n(_num(i.get("reprojection_rms_px")), 3)] for i in images]
        half = (len(cells) + 1) // 2
        left, right = cells[:half], cells[half:]
        header = [tr("app.image"), tr("app.camera"), tr("app.rms")] * 2
        rows = [[*left[k], *(right[k] if k < len(right) else ["", "", ""])] for k in range(half)]
        group = AVAIL_W / 2
        widths = [group * 0.62, group * 0.14, group * 0.24] * 2
        table = Table([header, *rows], colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), ctx.bold), ("FONTNAME", (0, 1), (-1, -1), ctx.font), ("FONTSIZE", (0, 0), (-1, -1), 6.3),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(HEAD_BG)),
            ("ALIGN", (2, 0), (2, -1), "RIGHT"), ("ALIGN", (5, 0), (5, -1), "RIGHT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(ZEBRA)]),
            ("LINEAFTER", (2, 0), (2, -1), 0.4, colors.HexColor(RULE)),
            ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        story += [ctx.h2(f"{tr('app.cameras')} ({ctx.i(len(images))})"), table]
    if not ctx.camera_list and images:
        story.append(ctx.small(tr("app.camera_hint")))
    warnings = [str(w) for w in _l(data.get("warnings"))]
    if not warnings:
        story += [ctx.h2(tr("app.warnings")), ctx.p(tr("app.no_warnings"))]
    if warnings:
        story.append(ctx.h2(tr("app.warnings")))
        story += [ctx.p(line, "tiny") for line in warnings[:MAX_WARNING_LINES]]
        if len(warnings) > MAX_WARNING_LINES:
            story.append(ctx.more(len(warnings) - MAX_WARNING_LINES))
    return story


BODIES: dict[str, Callable[[_Ctx, Any], list[Any]]] = {
    "summary": sec_summary, "import_": sec_import, "alignment": sec_alignment, "control_points": sec_control_points,
    "densify": sec_densify, "refine": sec_refine, "classification": sec_classification, "refinement": sec_refinement,
    "products": sec_products, "runtimes": sec_runtimes, "appendix": sec_appendix,
}


def _render_section(ctx: _Ctx, key: str, number: int) -> list[Any]:
    tr = ctx.tr
    heading = Paragraph(f"{number}&nbsp;&nbsp;{escape(tr(f'section.{key}'))}", ctx.styles["h1"])
    sec = ctx.sections.get(key)
    try:
        status = getattr(sec, "status", None)
        source = getattr(sec, "source", None)
        warnings = [str(w) for w in _l(getattr(sec, "warnings", None))]
        body: list[Any] = []
        if sec is None or status in ("no_data", "error") or status not in ("ok", "not_run"):
            body.append(ctx.no_data())
        else:
            if status == "not_run":
                body.append(ctx.p(tr("status.not_run")))
            if status == "ok" or key in NOT_RUN_WITH_BODY:
                body += BODIES[key](ctx, sec)
        if source in ("derived", "mixed") and status in ("ok", "not_run"):
            if body:
                try:
                    body[-1].keepWithNext = True  # the note stays on the page of the table it belongs to
                except Exception:  # noqa: BLE001
                    pass
            body.append(ctx.small(tr(f"status.{source}")))
        for warning in warnings[:10]:
            body.append(ctx.small(warning))
        return [heading, *body]
    except Exception as error:  # noqa: BLE001 - the PDF is always produced
        LOGGER.warning("protocol section %s could not be rendered: %s", key, error, exc_info=True)
        return [heading, ctx.no_data()]


def _title_page(ctx: _Ctx) -> list[Any]:
    tr = ctx.tr
    sec = ctx.sections.get("title")
    data = _d(getattr(sec, "data", None))
    project = _d(getattr(ctx.data, "project", None))
    name = str(data.get("project_name") or project.get("name") or "")
    story: list[Any] = [Spacer(1, 22 * mm), ctx.p(tr("doc.title"), "title"), ctx.p(name, "subtitle")]
    try:
        crs = data.get("crs")
        version = " ".join(str(v) for v in (data.get("samcloud_version"), data.get("git_hash")) if v)
        rows = [
            (tr("title.project"), name or "–", None),
            (tr("title.date"), str(data.get("date") or ctx.missing(data, "date")), None),
            (tr("title.version"), version or ctx.missing(data, "samcloud_version"), None),
            (tr("title.capture"), ctx.lookup("capture", data["capture_type"]) if data.get("capture_type") else ctx.missing(data, "capture_type"), None),
            (tr("title.class_set"), ctx.lookup("classset", data["class_set"]) if data.get("class_set") else ctx.missing(data, "class_set"), None),
            (tr("title.crs"), tr("title.crs_local") if str(crs).upper() == "LOCAL" else str(crs or ctx.missing(data, "crs")), None),
            (tr("title.control_mode"), tr("title.mode_with") if data.get("control_points_mode") == "with" else tr("title.mode_without")
             if data.get("control_points_mode") == "without" else "–", None),
        ]
        story.append(ctx.kv(rows))
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("title table failed: %s", error)
    image = ctx.chart("overview", 105)
    if image is not None:
        story += [Spacer(1, 8 * mm), ctx.h2(tr("title.overview")), image]
    return story


class _Doc(BaseDocTemplate):
    def afterFlowable(self, flowable: Any) -> None:  # noqa: N802 - reportlab hook
        if isinstance(flowable, Paragraph) and flowable.style.name == "h1":
            text = flowable.getPlainText().replace("\xa0", " ")
            key = "section_" + (text.split(" ", 1)[0] if text else "x")
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, 0, 0)
            self.notify("TOCEntry", (0, text, self.page, key))


def _canvas_class(ctx: _Ctx, project_name: str, generated: str, version: str) -> type:
    tr = ctx.tr

    class NumberedCanvas(canvas.Canvas):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._saved_states: list[dict[str, Any]] = []

        def showPage(self) -> None:  # noqa: N802
            self._saved_states.append(dict(self.__dict__))
            self._startPage()

        def save(self) -> None:
            total = len(self._saved_states)
            for state in self._saved_states:
                self.__dict__.update(state)
                self._chrome(total)
                super().showPage()
            super().save()

        def _chrome(self, total: int) -> None:
            if self._pageNumber <= 1:
                return
            self.saveState()
            self.setStrokeColor(colors.HexColor(RULE))
            self.setLineWidth(0.5)
            self.setFillColor(colors.HexColor(MUTED))
            self.setFont(ctx.font, 7.5)
            top = PAGE_H - 11 * mm
            self.drawString(MARGIN, top, project_name[:80])
            self.drawRightString(PAGE_W - MARGIN, top, tr("doc.title"))
            self.line(MARGIN, top - 2.5 * mm, PAGE_W - MARGIN, top - 2.5 * mm)
            self.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
            left = tr("doc.generated", date=generated) + (f"  ·  {tr('doc.app', version=version)}" if version else "")
            self.drawString(MARGIN, 9.5 * mm, left)
            self.drawRightString(PAGE_W - MARGIN, 9.5 * mm, tr("doc.page", page=self._pageNumber, total=total))
            self.restoreState()

    return NumberedCanvas


def _generated_date(data: Any, language: str) -> str:
    from datetime import datetime

    raw = str(getattr(data, "generated_at", "") or "")
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        moment = datetime.now()
    return moment.strftime("%d.%m.%Y" if language == "de" else "%Y-%m-%d")


def build_protocol_pdf(data: Any, out_path: Path, *, language: str, options: dict[str, Any] | None = None,
                       chart_dir: Path | None = None, progress: Callable[[float, str], None] | None = None) -> Path:
    """Write the protocol PDF to ``out_path`` (parent folders are created) and return the path."""
    options = dict(options or {})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = _Ctx(data, language, options, chart_dir)
    tr = ctx.tr
    project = _d(getattr(data, "project", None))
    name = str(project.get("name") or _d(getattr(ctx.sections.get("title"), "data", None)).get("project_name") or "")
    story: list[Any] = []
    try:
        story += _title_page(ctx)
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("title page failed: %s", error, exc_info=True)
        story += [ctx.p(tr("doc.title"), "title"), ctx.p(name, "subtitle")]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("toc0", parent=ctx.styles["toc"], leftIndent=0, firstLineIndent=0)]
    toc.dotsMinLevel = 0
    story += [NextPageTemplate("main"), PageBreak(), Paragraph(escape(tr("doc.toc")), ctx.styles["h1_plain"]), toc, PageBreak()]
    sections = [key for key in SECTION_ORDER if key != "title"]
    for index, key in enumerate(sections):
        if progress:
            progress(index / len(sections), key)
        part = _render_section(ctx, key, index + 2)
        if index:
            story.append(Spacer(1, 6))
        story += part
    version = app_version()
    generated = _generated_date(data, ctx.language)
    frame = Frame(MARGIN, MARGIN, AVAIL_W, PAGE_H - 2 * MARGIN, id="frame", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc = _Doc(str(out_path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
               title=f"{tr('doc.title')} - {name}".strip(" -"), author="SAMcloud", subject=tr("doc.title"),
               pageCompression=0 if options.get("uncompressed") else 1)
    doc.addPageTemplates([PageTemplate(id="title", frames=[frame]), PageTemplate(id="main", frames=[frame])])
    doc.multiBuild(story, canvasmaker=_canvas_class(ctx, name, generated, version))
    if progress:
        progress(1.0, "done")
    return out_path
