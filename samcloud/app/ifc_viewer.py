"""IFC 3D viewer: glTF model with per-category visibility, picking and a horizontal section (NiceGUI Vue component)."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from nicegui import ui

from samcloud.app.dxf_controller import REVIEW_BACKGROUND
from samcloud.app.i18n import t

MODEL_ROUTE = "/api/bim"


def model_url(level_key: str, version: str | int) -> str:
    """Browser URL of the level's GLB; the version token (mtime) busts the browser cache."""
    return f"{MODEL_ROUTE}/{quote(level_key)}/model.glb?v={version}"


class IfcViewer(ui.element, component="ifc_viewer.js"):
    """three.js view of ``model.glb`` (one node per IFC element, node.name = GlobalId).

    Python -> JS (props): ``src`` (GLB url), ``hidden-classes`` (category keys to hide), ``class-by-node``
    (GlobalId -> category key), ``category-colors`` (category -> hex colour, outdoor only), ``clip-height`` (model Y height in metres, None = no section),
    ``background`` ([r, g, b]), ``selected`` (GlobalId to highlight), ``strings`` (translated texts).
    Method ``fit()`` reframes the camera.

    JS -> Python (events): ``ready`` {elements, min_height, max_height, up_converted},
    ``pick`` {global_id | None, ifc_class}, ``error`` {message}.

    AI assistant (only set by the page while the assistant is active; never set otherwise): ``pick-enabled`` (``pick``
    also carries ``ctrl`` and ``point`` = hit point, or the intersection with the floor plane ``pick-floor``, in the
    model frame: glTF Y up, metres, ``origin_offset_yup`` subtracted; the page then leaves the highlight to Python),
    ``selected-many`` (GlobalIds highlighted like ``selected``), ``highlight`` ({GlobalId: "#rrggbb"} preview colours)
    and ``boxes`` ([{min: [x, y, z], max: [x, y, z], color}] in the model frame).
    """

    def __init__(self) -> None:
        super().__init__()
        self._props["src"] = ""
        self._props["hidden-classes"] = []
        self._props["class-by-node"] = {}
        self._props["clip-height"] = None
        self._props["category-colors"] = {}
        self._props["background"] = list(REVIEW_BACKGROUND)
        self._props["selected"] = ""
        self._props["strings"] = {
            "hint": t("ifc.viewer.hint"),
            "loading": t("ifc.viewer.loading"),
            "load_failed": t("ifc.viewer.load_failed"),
            "no_model": t("ifc.viewer.no_model"),
        }

    def show(self, src: str, class_by_node: dict[str, str]) -> None:
        self._props["src"] = src
        self._props["class-by-node"] = dict(class_by_node)
        self.update()

    def set_hidden_classes(self, names: Any) -> None:
        self._props["hidden-classes"] = sorted(str(name) for name in names)
        self.update()

    def set_category_colors(self, colors: dict[str, str]) -> None:
        """Category key -> '#rrggbb' override of the GLB material colour (empty = keep the model colours)."""
        self._props["category-colors"] = {str(key): str(value) for key, value in colors.items()}
        self.update()

    def set_clip_height(self, height: float | None) -> None:
        self._props["clip-height"] = None if height is None else float(height)
        self.update()

    def set_selected(self, global_id: str | None) -> None:
        self._props["selected"] = global_id or ""
        self.update()

    def fit(self) -> None:
        self.run_method("fit")

    def set_pick_enabled(self, enabled: bool, floor_height: float | None = None) -> None:
        """Assistant picking: ``pick`` adds ``ctrl`` and ``point``; ``floor_height`` = model Y of the floor plane."""
        self._props["pick-enabled"] = bool(enabled)
        if floor_height is not None:
            self._props["pick-floor"] = float(floor_height)
        self.update()

    def set_selected_many(self, global_ids: list[str]) -> None:
        """Highlight several elements (the assistant selection)."""
        self._props["selected-many"] = [str(value) for value in global_ids or [] if value]
        self.update()

    def set_highlight(self, colors: dict[str, str]) -> None:
        """GlobalId -> '#rrggbb' tint of existing elements (assistant preview: removed red, moved amber)."""
        self._props["highlight"] = {str(key): str(value) for key, value in (colors or {}).items() if key}
        self.update()

    def set_boxes(self, boxes: list[dict[str, Any]]) -> None:
        """Translucent boxes ``{min, max, color}`` in the model frame (assistant preview of added / moved elements)."""
        self._props["boxes"] = [{"min": [float(v) for v in box["min"]], "max": [float(v) for v in box["max"]],
                                 "color": str(box.get("color") or "#4bc692")} for box in boxes or []]
        self.update()
