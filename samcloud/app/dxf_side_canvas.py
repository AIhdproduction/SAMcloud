"""Interactive side-view canvas for indoor levels (NiceGUI custom Vue component)."""

from __future__ import annotations

from typing import Any, Literal

from nicegui import ui

SideTool = Literal["pan", "add", "measure"]


class DxfSideCanvas(ui.element, component="dxf_side_canvas.js"):
    """Orthographic side view PNG with level bands, draggable floor/ceiling lines and a measure tool.

    Python -> JS (props): ``image`` (PNG data URL or ""), ``bounds`` ([h_min, h_max, z_min, z_max] in
    metres), ``levels`` (list of {id, name, floor_z, ceiling_z, checked}), ``tool`` ("pan" | "add" |
    "measure"), ``axis-label`` (horizontal axis name). Method ``fit()`` resets the view.

    JS -> Python (events): ``level_move`` {id, edge: "floor" | "ceiling", z}, ``add_level`` {z},
    ``measure`` {p1: [h, z], p2: [h, z], distance}.
    """

    def __init__(self) -> None:
        super().__init__()
        self._props["image"] = ""
        self._props["bounds"] = [0.0, 1.0, 0.0, 1.0]
        self._props["levels"] = []
        self._props["tool"] = "pan"
        self._props["axis-label"] = "Y"

    def set_image(self, data_url: str, bounds: tuple[float, float, float, float], axis_label: str) -> None:
        """Show a rendered side view; bounds are (horizontal min, horizontal max, z min, z max)."""
        self._props["image"] = data_url
        self._props["bounds"] = [float(value) for value in bounds]
        self._props["axis-label"] = axis_label
        self.update()

    def set_levels(self, levels: list[dict[str, Any]]) -> None:
        """Replace the drawn level bands."""
        self._props["levels"] = [
            {
                "id": str(level["id"]),
                "name": str(level.get("name", level["id"])),
                "floor_z": float(level["floor_z"]),
                "ceiling_z": None if level.get("ceiling_z") is None else float(level["ceiling_z"]),
                "checked": bool(level.get("checked", True)),
            }
            for level in levels
        ]
        self.update()

    def set_tool(self, tool: SideTool) -> None:
        """Switch the pointer tool: pan/drag lines, add a level by click, or two-point measure."""
        if tool not in ("pan", "add", "measure"):
            raise ValueError(f"Unknown side view tool '{tool}'")
        self._props["tool"] = tool
        self.update()

    def fit(self) -> None:
        """Fit the whole side view into the canvas."""
        self.run_method("fit")
