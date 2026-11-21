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
