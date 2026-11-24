"""Charts of the processing protocol (matplotlib Figure API only, no pyplot state: safe in worker threads).

Every public ``*_chart`` function takes section data, an output directory and a translator, returns the PNG path
or ``None`` when data is missing, and never raises (errors are logged as warnings). ``build_all_charts`` writes the
whole set under the fixed file names of ``CHART_FILES``; ``samcloud.report.pdf`` looks for those names.
"""

from __future__ import annotations

import functools
import logging
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from samcloud.report.i18n_text import Translator, fmt_duration, fmt_int, fmt_num

LOGGER = logging.getLogger(__name__)

CHART_FILES = {
    "overview": "overview.png", "camera_map": "camera_map.png", "overlap": "overlap.png",
    "error_vectors": "error_vectors.png", "classes": "classes.png", "confidence": "confidence.png",
    "refinement": "refinement.png", "runtimes": "runtimes.png",
}
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0dc"
RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
OVERVIEW_MAX_PX = 1600
DPI = 150
WIDTH_CM = 17.0


def sequential_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("samcloud_blue", RAMP)


def _safe(function: Callable[..., Path | None]) -> Callable[..., Path | None]:
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Path | None:
        try:
            return function(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - a chart never stops the protocol
            LOGGER.warning("chart %s failed: %s", function.__name__, error, exc_info=True)
            return None

    return wrapper


# --------------------------------------------------------------------------------------- helpers


def _f(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _d(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _l(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _figure(height_cm: float, width_cm: float = WIDTH_CM) -> Figure:
    figure = Figure(figsize=(width_cm / 2.54, height_cm / 2.54), dpi=DPI, facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _style(ax: Any, grid_axis: str | None = "both") -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=7.5, colors=MUTED, length=2.5, width=0.6)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_color(MUTED)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)


def _colourbar(figure: Figure, mappable: Any, ax: Any, label: str) -> None:
    bar = figure.colorbar(mappable, ax=ax, fraction=0.04, pad=0.02)
    bar.outline.set_linewidth(0.4)
    bar.outline.set_edgecolor(GRID)
    bar.ax.tick_params(labelsize=7, colors=MUTED, length=2, width=0.5)
    bar.set_label(label, size=8, color=MUTED)


def _save(figure: Figure, out_dir: Path, key: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / CHART_FILES[key]
    try:
        figure.tight_layout()
    except Exception:  # noqa: BLE001 - layout is cosmetic
        LOGGER.debug("tight_layout failed for %s", key)
    figure.savefig(path, dpi=DPI, facecolor="white")
    return path


def _thousands_formatter(tr: Translator) -> FuncFormatter:
    return FuncFormatter(lambda value, _pos: fmt_int(tr, value))


def _thousands_axis(ax: Any, tr: Translator, axis: str = "x") -> None:
    """Thousands separators and at most ~5 ticks so large numbers do not run into each other."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(_thousands_formatter(tr))
    target.set_major_locator(MaxNLocator(nbins=5))


def _nice_number(value: float) -> float:
    """Largest 1/2/5 x 10^n not above ``value`` (> 0)."""
    exponent = math.floor(math.log10(value))
    for factor in (5, 2, 1):
        candidate = factor * 10 ** exponent
        if candidate <= value:
            return candidate
    return 10 ** exponent


ABSOLUTE_LIMIT = 100000.0


def _frame_offset(alignment: dict[str, Any]) -> tuple[float, float]:
    """Centre of the cameras when the model frame holds projected (absolute) coordinates, else (0, 0)."""
    alignment = _d(alignment)
    xs, ys, _ = _camera_xy(alignment, raw=True)
    if not xs:
        grid = _overlap_array(alignment, raw=True)
        return (grid[1], grid[2]) if grid and max(abs(grid[1]), abs(grid[2])) >= ABSOLUTE_LIMIT else (0.0, 0.0)
    if max(max(abs(v) for v in xs), max(abs(v) for v in ys)) < ABSOLUTE_LIMIT:
        return 0.0, 0.0
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _axis_labels(ax: Any, alignment: dict[str, Any], tr: Translator) -> None:
    relative = _frame_offset(alignment) != (0.0, 0.0)
    ax.set_xlabel(tr("chart.x_rel" if relative else "chart.x_m"))
    ax.set_ylabel(tr("chart.y_rel" if relative else "chart.y_m"))


def _camera_xy(alignment: dict[str, Any], raw: bool = False) -> tuple[list[float], list[float], list[float | None]]:
    xs: list[float] = []
    ys: list[float] = []
    rms: list[float | None] = []
    for image in _l(alignment.get("images")):
        centre = _d(image).get("center_xyz")
        if not isinstance(centre, (list, tuple)) or len(centre) < 2:
            continue
        x, y = _f(centre[0]), _f(centre[1])
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        rms.append(_f(_d(image).get("reprojection_rms_px")))
    if not raw:
        ox, oy = _frame_offset(alignment)
        xs, ys = [x - ox for x in xs], [y - oy for y in ys]
    return xs, ys, rms


def _overlap_array(alignment: dict[str, Any], raw: bool = False) -> tuple[np.ndarray, float, float, float] | None:
    overlap = _d(alignment.get("overlap"))
    counts = overlap.get("counts")
    origin = overlap.get("origin_xy")
    cell = _f(overlap.get("cell_size"))
    if not counts or cell is None or cell <= 0 or not isinstance(origin, (list, tuple)) or len(origin) < 2:
        return None
    array = np.asarray(counts, dtype=float)
    if array.ndim != 2 or array.size == 0:
        return None
    x0, y0 = _f(origin[0]), _f(origin[1])
    if x0 is None or y0 is None:
        return None
    if not raw:
        ox, oy = _frame_offset(alignment)
        x0, y0 = x0 - ox, y0 - oy
    return array, x0, y0, cell


# ---------------------------------------------------------------------------------- overview image


def _load_overview(path: Path) -> Any | None:
    """Downscaled RGB PIL image of a TIFF/PNG/JPG (large GeoTIFFs are read through rasterio overviews)."""
    from PIL import Image

    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio
            from rasterio.enums import Resampling

            with rasterio.open(path) as dataset:
                scale = max(dataset.width, dataset.height) / OVERVIEW_MAX_PX
                width = max(1, int(round(dataset.width / max(scale, 1.0))))
                height = max(1, int(round(dataset.height / max(scale, 1.0))))
                bands = min(dataset.count, 4)
                data = dataset.read(list(range(1, bands + 1)), out_shape=(bands, height, width),
                                    resampling=Resampling.average).astype(float)
            if bands == 1:
                data = np.repeat(data, 3, axis=0)
                bands = 3
            alpha = data[3] / 255.0 if bands == 4 else None
            rgb = data[:3]
            if rgb.max() > 255 or rgb.min() < 0:
                low, high = np.percentile(rgb, 1), np.percentile(rgb, 99)
                rgb = np.clip((rgb - low) / max(high - low, 1e-9) * 255.0, 0, 255)
            if alpha is not None:
                rgb = rgb * alpha + 255.0 * (1 - alpha)
            image = Image.fromarray(np.clip(np.moveaxis(rgb, 0, -1), 0, 255).astype("uint8"), "RGB")
            return image
        except Exception as error:  # noqa: BLE001 - fall through to PIL
            LOGGER.info("overview %s not readable with rasterio (%s), trying PIL", path.name, error)
    with Image.open(path) as source:
        source.draft("RGB", (OVERVIEW_MAX_PX * 2, OVERVIEW_MAX_PX * 2))
        image = source.convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, image).convert("RGB")
    image.thumbnail((OVERVIEW_MAX_PX, OVERVIEW_MAX_PX))
    return image


@_safe
def overview_chart(title: dict[str, Any], alignment: dict[str, Any], project_dir: Any, out_dir: Path,
                   tr: Translator) -> Path | None:
    """Project overview: the ortho/hillshade image when present, else a top view of cameras and overlap footprint."""
    relative = _d(title).get("overview_image")
    if relative and project_dir:
        path = Path(project_dir) / str(relative)
        # colour_source.png is a label mask (values 0..2, almost black); fall back to the real ortho previews next to it.
        candidates = [path] + [path.with_name(name) for name in ("preview.jpg", "ortho_4096.jpg", "ortho.png")
                               if path.name == "colour_source.png"]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                image = _load_overview(candidate)
                if image is None or (np.asarray(image.convert("L").resize((64, 64))).max() < 24 and candidate is not candidates[-1]):
                    continue  # nearly black: not a photo
                target = Path(out_dir) / CHART_FILES["overview"]
                target.parent.mkdir(parents=True, exist_ok=True)
                image.save(target, format="PNG")
                return target
            except Exception as error:  # noqa: BLE001
                LOGGER.warning("overview image %s not usable: %s", candidate, error)
    xs, ys, _rms = _camera_xy(_d(alignment))
    grid = _overlap_array(_d(alignment))
    if not xs and grid is None:
        return None
    figure = _figure(9.5)
    ax = figure.add_subplot(111)
    if grid is not None:
        array, x0, y0, cell = grid
        occupied = np.ma.masked_where(array <= 0, np.ones_like(array))
        ax.imshow(occupied, origin="lower", cmap=LinearSegmentedColormap.from_list("foot", ["#e6e5e1", "#e6e5e1"]),
                  extent=[x0, x0 + array.shape[1] * cell, y0, y0 + array.shape[0] * cell], interpolation="nearest")
    if xs:
        ax.scatter(xs, ys, s=5, color=BLUE, linewidths=0, label=tr("chart.cameras"))
    ax.set_aspect("equal", adjustable="datalim")
    _axis_labels(ax, alignment, tr)
    _style(ax)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=BLUE, markersize=4, label=tr("chart.cameras"))]
    if grid is not None:
        handles.append(Line2D([0], [0], marker="s", linestyle="", color="#e6e5e1", markersize=6, label=tr("chart.footprint")))
    ax.legend(handles=handles, fontsize=7.5, frameon=False, labelcolor=MUTED, loc="best")
    return _save(figure, out_dir, "overview")


# ------------------------------------------------------------------------------------ alignment


@_safe
def camera_map_chart(alignment: dict[str, Any], out_dir: Path, tr: Translator) -> Path | None:
    """Top view of the camera centres, coloured by the reprojection RMS (px)."""
    xs, ys, rms = _camera_xy(_d(alignment))
    if not xs:
        return None
    figure = _figure(9.5)
    ax = figure.add_subplot(111)
    values = np.array([np.nan if v is None else v for v in rms], dtype=float)
    if np.isfinite(values).any():
        vmax = float(np.nanpercentile(values, 98))
        mappable = ax.scatter(xs, ys, c=values, s=7, cmap=sequential_cmap(), vmin=0.0, vmax=max(vmax, 1e-6), linewidths=0)
        _colourbar(figure, mappable, ax, tr("chart.px"))
    else:
        ax.scatter(xs, ys, s=7, color=BLUE, linewidths=0)
    ax.set_aspect("equal", adjustable="datalim")
    _axis_labels(ax, alignment, tr)
    _style(ax)
    return _save(figure, out_dir, "camera_map")


@_safe
def overlap_chart(alignment: dict[str, Any], out_dir: Path, tr: Translator) -> Path | None:
    """Image overlap heatmap (number of images per cell); empty cells stay white."""
    grid = _overlap_array(_d(alignment))
    if grid is None:
        return None
    array, x0, y0, cell = grid
    masked = np.ma.masked_where(array <= 0, array)
    if masked.count() == 0:
        return None
    cmap = sequential_cmap().copy()
    cmap.set_bad(alpha=0.0)
    aspect = array.shape[0] / max(array.shape[1], 1)
    figure = _figure(9.5, width_cm=max(8.0, min(WIDTH_CM, 9.5 / max(aspect, 0.05) + 3.5)))
    ax = figure.add_subplot(111)
    mappable = ax.imshow(masked, origin="lower", cmap=cmap, vmin=1, interpolation="nearest",
                         extent=[x0, x0 + array.shape[1] * cell, y0, y0 + array.shape[0] * cell])
    _colourbar(figure, mappable, ax, tr("chart.images"))
    _axis_labels(ax, alignment, tr)
    _style(ax)
    return _save(figure, out_dir, "overlap")


# -------------------------------------------------------------------------------- control points


@_safe
def error_vector_chart(control: dict[str, Any], out_dir: Path, tr: Translator) -> Path | None:
    """Horizontal error vectors of pass (circle, blue) and control (square, orange) points with a scale arrow in cm."""
    points = []
    for item in _l(_d(control).get("local_points_xy")):
        item = _d(item)
        x, y = _f(item.get("x_m")), _f(item.get("y_m"))
        if x is None or y is None:
            continue
        points.append((str(item.get("name")), item.get("role") == "control", x, y, _f(item.get("err_x_m")), _f(item.get("err_y_m"))))
    if not points:
        return None
    xs = [p[2] for p in points]
    ys = [p[3] for p in points]
    extent = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    errors = [math.hypot(p[4], p[5]) for p in points if p[4] is not None and p[5] is not None]
    max_error = max(errors) if errors else 0.0
    scale = 0.12 * extent / max_error if max_error > 1e-9 else 1.0
    figure = _figure(10.5)
    ax = figure.add_subplot(111)
    for name, is_control, x, y, ex, ey in points:
        colour, marker = (ORANGE, "s") if is_control else (BLUE, "o")
        ax.plot([x], [y], marker=marker, color=colour, markersize=5, markeredgewidth=0, linestyle="")
        if ex is not None and ey is not None and max_error > 1e-9:
            ax.annotate("", xy=(x + ex * scale, y + ey * scale), xytext=(x, y),
                        arrowprops={"arrowstyle": "-|>", "color": colour, "linewidth": 1.0, "shrinkA": 0, "shrinkB": 0,
                                    "mutation_scale": 7})
        dx = -1 if (ex or 0.0) > 0 else 1  # label on the side opposite to the arrow
        dy = -1 if (ey or 0.0) > 0 else 1
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(7 * dx, 7 * dy), fontsize=7, color=MUTED,
                    ha="left" if dx > 0 else "right", va="bottom" if dy > 0 else "top")
    pad = 0.18 * extent
    x_low, x_high = min(xs) - pad, max(xs) + pad
    y_low, y_high = min(ys) - pad, max(ys) + pad
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    if max_error > 1e-9:
        nice_cm = _nice_number(max_error * 100.0)
        length = nice_cm / 100.0 * scale
        base_x, base_y = x_low + 0.04 * (x_high - x_low), y_low + 0.06 * (y_high - y_low)
        ax.annotate("", xy=(base_x + length, base_y), xytext=(base_x, base_y),
                    arrowprops={"arrowstyle": "-|>", "color": TEXT, "linewidth": 1.2, "shrinkA": 0, "shrinkB": 0,
                                "mutation_scale": 8})
        ax.text(base_x, base_y + 0.02 * (y_high - y_low), tr("chart.scale", value=fmt_num(tr, nice_cm, 1 if nice_cm < 1 else 0)),
                fontsize=7.5, color=TEXT, va="bottom")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(tr("chart.x_m"))
    ax.set_ylabel(tr("chart.y_m"))
    _style(ax)
    ax.legend(handles=[Line2D([0], [0], marker="o", linestyle="", color=BLUE, markersize=5, label=tr("chart.pass")),
                       Line2D([0], [0], marker="s", linestyle="", color=ORANGE, markersize=5, label=tr("chart.control"))],
              fontsize=7.5, frameon=False, labelcolor=MUTED, loc="upper right")
    return _save(figure, out_dir, "error_vectors")


# ------------------------------------------------------------------------------- classification


@_safe
def class_chart(classification: dict[str, Any], out_dir: Path, tr: Translator, max_classes: int = 20) -> Path | None:
    """Horizontal bars: points per class, sorted descending, share in percent as direct label."""
    data = _d(classification)
    total = _f(data.get("total_points"))
    rows = []
    for item in _l(data.get("classes")):
        item = _d(item)
        points = _f(item.get("points"))
        if points is None:
            continue
        share = _f(item.get("share_pct"))
        if share is None and total:
            share = points / total * 100.0
        rows.append((tr.cls(item.get("name")), points, share))
    if not rows:
        return None
    rows.sort(key=lambda row: -row[1])
    rows = rows[:max_classes]
    figure = _figure(max(4.0, 1.6 + 0.55 * len(rows)))
    ax = figure.add_subplot(111)
    positions = list(range(len(rows)))
    ax.barh(positions, [r[1] for r in rows], color=BLUE, height=0.6)
    ax.set_yticks(positions)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    biggest = max(r[1] for r in rows) or 1.0
    ax.set_xlim(0, biggest * 1.18)
    for position, (_name, points, share) in zip(positions, rows):
        label = tr("chart.share", value=fmt_num(tr, share, 1)) if share is not None else ""
        ax.text(points + biggest * 0.01, position, label, va="center", fontsize=7.5, color=MUTED)
    _thousands_axis(ax, tr, "x")
    ax.set_xlabel(tr("chart.points"))
    _style(ax, "x")
    return _save(figure, out_dir, "classes")


@_safe
def confidence_chart(classification: dict[str, Any], out_dir: Path, tr: Translator) -> Path | None:
    """Histogram of the SAM confidence (log scale when the counts span more than two orders of magnitude)."""
    histogram = _d(_d(classification).get("confidence_histogram"))
    edges = [_f(v) for v in _l(histogram.get("edges"))]
    counts = [_f(v) for v in _l(histogram.get("counts"))]
    if len(edges) < 2 or len(counts) != len(edges) - 1 or any(v is None for v in edges + counts):
        return None
    edges_a, counts_a = np.asarray(edges, dtype=float), np.asarray(counts, dtype=float)
    if counts_a.sum() <= 0:
        return None
    figure = _figure(6.5)
    ax = figure.add_subplot(111)
    ax.bar(edges_a[:-1], counts_a, width=np.diff(edges_a), align="edge", color=BLUE, edgecolor="white", linewidth=0.4)
    positive = counts_a[counts_a > 0]
    log = positive.max() / positive.min() > 100
    if log:
        ax.set_yscale("log")
        ax.set_ylabel(tr("chart.points_log"))
    else:
        _thousands_axis(ax, tr, "y")
        ax.set_ylabel(tr("chart.points"))
    ax.set_xlabel(tr("chart.conf"))
    ax.set_xlim(edges_a[0], edges_a[-1])
    _style(ax, "y")
    return _save(figure, out_dir, "confidence")


# ---------------------------------------------------------------------------- refinement/runtimes


@_safe
def refinement_chart(refinement: dict[str, Any], out_dir: Path, tr: Translator) -> Path | None:
    """Removed (blue) and added (orange) points per step-6 stage."""
    rows = []
    for stage in _l(_d(refinement).get("stages")):
        stage = _d(stage)
        name = str(stage.get("stage"))
        if "." in name or stage.get("skipped"):
            continue  # sub-removals are part of their parent stage
        removed, added = _f(stage.get("removed")), _f(stage.get("added"))
        if name == "mirror_detection":
            removed = None  # counted once, inside surface fusion
        if name == "depth_evidence":
            added = None  # evidence points are not part of the cloud
        if not removed and not added:
            continue
        rows.append((tr.stage(name), removed or 0.0, added or 0.0))
    if not rows:
        return None
    figure = _figure(max(4.5, 2.2 + 1.0 * len(rows)))
    ax = figure.add_subplot(111)
    positions = np.arange(len(rows))
    height = 0.34
    ax.barh(positions - height / 2, [r[1] for r in rows], height=height, color=BLUE, label=tr("chart.removed"))
    ax.barh(positions + height / 2, [r[2] for r in rows], height=height, color=ORANGE, label=tr("chart.added"))
    ax.set_yticks(positions)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    biggest = max(max(r[1], r[2]) for r in rows) or 1.0
    ax.set_xlim(0, biggest * 1.2)
    for position, (_n, removed, added) in zip(positions, rows):
        for offset, value in ((-height / 2, removed), (height / 2, added)):
            if value:
                ax.text(value + biggest * 0.01, position + offset, fmt_int(tr, value), va="center", fontsize=7, color=MUTED)
    _thousands_axis(ax, tr, "x")
    ax.set_xlabel(tr("chart.points"))
    _style(ax, "x")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=MUTED, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              borderaxespad=0.2)
    return _save(figure, out_dir, "refinement")


@_safe
def runtime_chart(runtimes: dict[str, Any], out_dir: Path, tr: Translator, max_steps: int = 40) -> Path | None:
    """Seconds per step, horizontal bars in pipeline order (first step on top)."""
    rows = []
    for step in _l(_d(runtimes).get("steps")):
        step = _d(step)
        seconds = _f(step.get("seconds"))
        if seconds is not None and seconds > 0:
            rows.append((tr.step(step.get("name")), seconds))
    if not rows:
        return None
    rows = rows[:max_steps]
    figure = _figure(max(4.0, 1.4 + 0.42 * len(rows)))
    ax = figure.add_subplot(111)
    positions = list(range(len(rows)))
    ax.barh(positions, [r[1] for r in rows], color=BLUE, height=0.62)
    ax.set_yticks(positions)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    biggest = max(r[1] for r in rows)
    ax.set_xlim(0, biggest * 1.18)
    for position, (_name, seconds) in zip(positions, rows):
        ax.text(seconds + biggest * 0.01, position, fmt_duration(seconds), va="center", fontsize=7, color=MUTED)
    _thousands_axis(ax, tr, "x")
    ax.set_xlabel(tr("chart.seconds"))
    _style(ax, "x")
    return _save(figure, out_dir, "runtimes")


# -------------------------------------------------------------------------------------- all


def build_all_charts(data: Any, chart_dir: Path, language: str | None = None,
                     progress: Callable[[float, str], None] | None = None) -> dict[str, Path]:
    """Render every chart of a ``ProtocolData`` into ``chart_dir``; returns {chart key: png path} for the ones made."""
    tr = Translator(language)
    sections = getattr(data, "sections", {}) or {}

    def section_data(key: str) -> dict[str, Any]:
        section = sections.get(key)
        return _d(getattr(section, "data", None))

    project_dir = _d(getattr(data, "project", None)).get("path")
    title, alignment, control = section_data("title"), section_data("alignment"), section_data("control_points")
    jobs: list[tuple[str, Callable[[], Path | None]]] = [
        ("overview", lambda: overview_chart(title, alignment, project_dir, chart_dir, tr)),
        ("camera_map", lambda: camera_map_chart(alignment, chart_dir, tr)),
        ("overlap", lambda: overlap_chart(alignment, chart_dir, tr)),
        ("error_vectors", lambda: error_vector_chart(control, chart_dir, tr)),
        ("classes", lambda: class_chart(section_data("classification"), chart_dir, tr)),
        ("confidence", lambda: confidence_chart(section_data("classification"), chart_dir, tr)),
        ("refinement", lambda: refinement_chart(section_data("refinement"), chart_dir, tr)),
        ("runtimes", lambda: runtime_chart(section_data("runtimes"), chart_dir, tr)),
    ]
    result: dict[str, Path] = {}
    for index, (key, job) in enumerate(jobs):
        if progress:
            progress(index / len(jobs), key)
        path = job()
        if path is not None:
            result[key] = path
    if progress:
        progress(1.0, "charts")
    return result
