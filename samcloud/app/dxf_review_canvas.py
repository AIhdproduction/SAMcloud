"""DXF review canvas: cached background image and DXF vectors in one world transform (NiceGUI Vue component)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from nicegui import ui

from samcloud.app.dxf_controller import REVIEW_BACKGROUND, DxfPageController

ReviewTool = Literal["pan", "measure"]

PAYLOAD_ROUTE = "/api/dxf/payload"
BACKGROUND_ROUTE = "/api/dxf/background"
ORTHO_ROUTE = "/api/dxf/ortho"


def level_key(level_id: str | None) -> str:
    """URL key of a review target: the level id, or 'site' for the outdoor plan."""
    return "site" if level_id is None else str(level_id)


def level_from_key(key: str) -> str | None:
    """Inverse of level_key."""
    return None if key == "site" else key


def payload_url(controller: DxfPageController, level_id: str | None, variant: str = "pointcloud") -> str:
    """Browser URL of the cached vector payload; the version token busts the browser cache after regeneration."""
    if variant == "pointcloud":
        return f"{PAYLOAD_ROUTE}/{quote(level_key(level_id))}?v={controller.cache_version(level_id)}"
    return f"{PAYLOAD_ROUTE}/{quote(level_key(level_id))}?variant={quote(variant)}&v={controller.plan_version(level_id, variant)}"


def background_url(controller: DxfPageController, level_id: str | None, source: str) -> str | None:
    """Browser URL of the cached background PNG ('rgb' or 'classes'), or None if not cached."""
    if controller.background_path(level_id, source) is None:
        return None
    return f"{BACKGROUND_ROUTE}/{quote(level_key(level_id))}/{source}?v={controller.cache_version(level_id)}"


def ortho_preview_url(controller: DxfPageController, level_id: str | None) -> str | None:
    """Browser URL of the cached orthophoto preview JPEG, or None when no preview exists."""
    if controller.ortho_file(level_id, "preview") is None:
        return None
    return f"{ORTHO_ROUTE}/{quote(level_key(level_id))}/preview?v={controller.ortho_version(level_id)}"


def review_payload(project_directory: Path, key: str, variant: str = "pointcloud") -> dict[str, Any] | None:
    """Vector payload for the HTTP route; reads only the cache (never generates)."""
    controller = DxfPageController(project_directory)
    return controller.plan_payload(level_from_key(key), variant)


def review_background(project_directory: Path, key: str, source: str) -> Path | None:
    """Background PNG path for the HTTP route; reads only the cache (never generates)."""
    controller = DxfPageController(project_directory)
    return controller.background_path(level_from_key(key), source)


class DxfReviewCanvas(ui.element, component="dxf_review_canvas.js"):
    """Image layer and DXF layer in one pan/zoom canvas with grid-index culling.

    Python -> JS (props): ``payload-url`` and optional ``compare-payload-url`` (JSON from
    samcloud.dxf.result.vector_payload),
    ``image-url`` (north-up PNG), ``image-bounds`` ([min_x, min_y, max_x, max_y] local metres),
    ``image-opacity``, ``dxf-opacity`` and ``compare-opacity`` (0..1, canvas globalAlpha),
    ``hidden-layers`` (layer names), ``tool`` ("pan" | "measure"), ``background`` ([r, g, b]).
    Method ``fit()`` resets the view.

    JS -> Python (events): ``measure`` {p1: [x, y], p2: [x, y], distance}, ``loaded`` {entities, layers}.
    Holding the key "H" hides the DXF layer while held (client side only).
    """

    def __init__(self) -> None:
        super().__init__()
        self._props["payload-url"] = ""
        self._props["compare-payload-url"] = ""
        self._props["image-url"] = ""
        self._props["image-bounds"] = [0.0, 0.0, 1.0, 1.0]
        self._props["image-opacity"] = 1.0
        self._props["dxf-opacity"] = 1.0
        self._props["compare-opacity"] = 1.0
        self._props["hidden-layers"] = []
        self._props["tool"] = "pan"
        self._props["background"] = list(REVIEW_BACKGROUND)

    def show(
        self,
        payload: str,
        image: str | None,
        bounds: tuple[float, float, float, float],
        compare_payload: str | None = None,
    ) -> None:
        """Load a cached review target: vector payload URL, background URL and its world bounds."""
        self._props["payload-url"] = payload
        self._props["compare-payload-url"] = compare_payload or ""
        self._props["image-url"] = image or ""
        self._props["image-bounds"] = [float(value) for value in bounds]
        self.update()

    def set_image(self, image: str | None) -> None:
        """Switch the background source (RGB or class colours)."""
        self._props["image-url"] = image or ""
        self.update()

    def set_opacity(self, image_opacity: float, dxf_opacity: float) -> None:
        """Set both layer opacities in the range 0..1."""
        self._props["image-opacity"] = max(0.0, min(1.0, float(image_opacity)))
        self._props["dxf-opacity"] = max(0.0, min(1.0, float(dxf_opacity)))
        self.update()

    def set_compare_opacity(self, opacity: float) -> None:
        """Set opacity of the second vector overlay in compare mode."""
        self._props["compare-opacity"] = max(0.0, min(1.0, float(opacity)))
        self.update()

    def set_hidden_layers(self, names: list[str]) -> None:
        """Hide the given DXF layers in the canvas."""
        self._props["hidden-layers"] = sorted(str(name) for name in names)
        self.update()

    def set_tool(self, tool: ReviewTool) -> None:
        """Switch between pan/zoom and the two-point measure tool."""
        if tool not in ("pan", "measure"):
            raise ValueError(f"Unknown review tool '{tool}'")
        self._props["tool"] = tool
        self.update()

    def fit(self) -> None:
        """Fit the cached plan into the canvas."""
        self.run_method("fit")
