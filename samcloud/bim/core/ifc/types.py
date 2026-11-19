"""Type objects (IfcWallType, IfcSlabType, IfcCoveringType, IfcWindowType, IfcDoorType).

Types are deduplicated: one per distinct key (scene type, layer set or dimensions, operation), so
the file never contains one type per occurrence. Layered types carry an IfcMaterialLayerSet;
occurrences get an IfcMaterialLayerSetUsage from ``type.assign_type``. Window and door types
carry the lining and panel property entities in ``HasPropertySets`` (the ifcopenshell functions
``geometry.add_window_representation`` and ``add_door_representation`` do not create them).
Types have no RepresentationMaps; geometry is stored per occurrence.
"""

from __future__ import annotations

from collections.abc import Sequence

import ifcopenshell
import ifcopenshell.api.material
import ifcopenshell.api.root

from samcloud.bim.core.ifc.context import ExportContext
from samcloud.bim.core.ifc.ids import set_key_guid
from samcloud.bim.core.model import (
    DoorElement,
    DoorOperation,
    MaterialLayer,
    Side,
    WindowElement,
    WindowOperation,
)

Entity = ifcopenshell.entity_instance

LINING_THICKNESS = 0.05
PANEL_DEPTH = 0.04
DEFAULT_FRAME_DEPTH = 0.06
MULLION_WIDTH = 0.04
"""Face width of a mullion bar (m)."""

WINDOW_PANEL_OPERATION: dict[WindowOperation, str] = {
    WindowOperation.FIXED: "FIXEDCASEMENT",
    WindowOperation.SIDE_HUNG_LEFT: "SIDEHUNGLEFTHAND",
    WindowOperation.SIDE_HUNG_RIGHT: "SIDEHUNGRIGHTHAND",
    WindowOperation.TOP_HUNG: "TOPHUNG",
    WindowOperation.BOTTOM_HUNG: "BOTTOMHUNG",
    WindowOperation.TILT_AND_TURN: "OTHEROPERATION",
    WindowOperation.SLIDING: "SLIDINGHORIZONTAL",
    WindowOperation.OTHER: "OTHEROPERATION",
}
"""Scene window operation to IfcWindowPanelOperationEnum (tilt and turn has no hand: OTHER)."""


def window_partitioning(pane_count: int) -> tuple[str, str | None]:
    """Pane count to (IfcWindowTypePartitioningEnum, UserDefinedPartitioningType) for vertical mullions."""
    if pane_count <= 1:
        return "SINGLE_PANEL", None
    if pane_count == 2:
        return "DOUBLE_PANEL_VERTICAL", None
    if pane_count == 3:
        return "TRIPLE_PANEL_VERTICAL", None
    return "USERDEFINED", f"{pane_count}_PANEL_VERTICAL"


def window_pane_count(element: WindowElement) -> int:
    return len(element.params.mullion_positions) + 1


def door_operation(element: DoorElement) -> tuple[str, str | None]:
    """Scene door operation to (IfcDoorTypeOperationEnum, UserDefinedOperationType)."""
    operation = element.params.operation
    if operation == DoorOperation.SINGLE_SWING_LEFT:
        return "SINGLE_SWING_LEFT", None
    if operation == DoorOperation.SINGLE_SWING_RIGHT:
        return "SINGLE_SWING_RIGHT", None
    if operation == DoorOperation.DOUBLE_SWING:
        return "DOUBLE_DOOR_SINGLE_SWING", None
    if operation == DoorOperation.SLIDING:
        side = element.params.swing_side
        return ("SLIDING_TO_LEFT" if side == Side.LEFT else "SLIDING_TO_RIGHT"), None
    return "USERDEFINED", operation.value


def door_panel_operation(operation: str) -> str:
    """IfcDoorPanelOperationEnum for an IfcDoorTypeOperationEnum value."""
    if operation.startswith(("SINGLE_SWING", "DOUBLE_DOOR_SINGLE")):
        return "SWINGING"
    if "SLIDING" in operation:
        return "SLIDING"
    return "NOTDEFINED"


def material(ctx: ExportContext, name: str) -> Entity:
    """IfcMaterial by name (created once)."""
    if name not in ctx.materials:
        ctx.materials[name] = ifcopenshell.api.material.add_material(ctx.file, name=name)
    return ctx.materials[name]


def layer_set(ctx: ExportContext, name: str, layers: Sequence[MaterialLayer]) -> Entity:
    """IfcMaterialLayerSet with one layer per scene layer (thickness in metres)."""
    f = ctx.file
    result = ifcopenshell.api.material.add_material_set(
        f, name=name, set_type="IfcMaterialLayerSet"
    )
    for item in layers:
        layer = ifcopenshell.api.material.add_layer(
            f, layer_set=result, material=material(ctx, item.material)
        )
        ifcopenshell.api.material.edit_layer(
            f, layer=layer, attributes={"LayerThickness": float(item.thickness)}
        )
    return result


def _layer_key(layers: Sequence[MaterialLayer]) -> tuple[tuple[str, float], ...]:
    return tuple((layer.material, float(layer.thickness)) for layer in layers)


def mm_text(value: float) -> str:
    """Millimetres of a length in metres as short text."""
    return f"{round(value * 1000.0, 1):g}"


def layered_type(
    ctx: ExportContext,
    *,
    ifc_class: str,
    predefined_type: str,
    base_name: str,
    scene_type_id: str | None,
    layers: Sequence[MaterialLayer],
    element_type: str | None = None,
) -> Entity:
    """IfcWallType, IfcSlabType or IfcCoveringType with a layer set, one per distinct key."""
    key: tuple[object, ...] = (
        ifc_class,
        predefined_type,
        scene_type_id,
        _layer_key(layers),
        element_type,
    )
    if key in ctx.types:
        return ctx.types[key]
    total = sum(layer.thickness for layer in layers)
    name = f"{base_name} {mm_text(total)}"
    entity = ifcopenshell.api.root.create_entity(
        ctx.file, ifc_class=ifc_class, predefined_type=predefined_type, name=name
    )
    if element_type is not None:
        entity.ElementType = element_type
    ifcopenshell.api.material.assign_material(
        ctx.file,
        products=[entity],
        type="IfcMaterialLayerSet",
        material=layer_set(ctx, name, layers),
    )
    set_key_guid(ctx, entity, f"type/{key!r}")
    ctx.types[key] = entity
    return entity


def window_type(
    ctx: ExportContext, element: WindowElement, width: float, height: float, base_name: str
) -> Entity:
    """IfcWindowType (partitioning by mullion count) with lining and one panel property per pane, one per distinct key."""
    params = element.params
    frame_depth = params.frame_depth or DEFAULT_FRAME_DEPTH
    panes = window_pane_count(element)
    partitioning, user_defined = window_partitioning(panes)
    key: tuple[object, ...] = (
        "IfcWindowType",
        element.type_id,
        params.operation,
        round(width, 6),
        round(height, 6),
        round(frame_depth, 6),
        panes,
    )
    if key in ctx.types:
        return ctx.types[key]
    f = ctx.file
    name = f"{base_name} {mm_text(width)}x{mm_text(height)} {params.operation.value}"
    if panes > 1:
        name += f" {partitioning.lower()}" if user_defined is None else f" {panes}-pane"
    entity = ifcopenshell.api.root.create_entity(
        f, ifc_class="IfcWindowType", predefined_type="WINDOW", name=name
    )
    entity.PartitioningType = partitioning
    if user_defined is not None:
        entity.UserDefinedPartitioningType = user_defined
    entity.ParameterTakesPrecedence = False
    lining = f.create_entity(
        "IfcWindowLiningProperties",
        GlobalId="0",
        Name="Lining",
        LiningDepth=float(frame_depth),
        LiningThickness=LINING_THICKNESS,
    )
    panels = []
    for pane in range(panes):
        position = "MIDDLE" if panes == 1 else ("LEFT" if pane == 0 else ("RIGHT" if pane == panes - 1 else "MIDDLE"))
        panels.append(
            f.create_entity(
                "IfcWindowPanelProperties",
                GlobalId="0",
                Name=f"Panel {pane + 1}",
                OperationType=WINDOW_PANEL_OPERATION[params.operation],
                PanelPosition=position,
                FrameDepth=float(min(frame_depth, PANEL_DEPTH)),
                FrameThickness=LINING_THICKNESS / 2.0,
            )
        )
    for index, item in enumerate((lining, *panels)):
        set_key_guid(ctx, item, f"type/{key!r}/props/{index}")
    entity.HasPropertySets = [lining, *panels]
    set_key_guid(ctx, entity, f"type/{key!r}")
    ctx.types[key] = entity
    return entity


def door_type(
    ctx: ExportContext, element: DoorElement, width: float, height: float, base_name: str
) -> Entity:
    """IfcDoorType with lining and panel properties, one per distinct key."""
    params = element.params
    frame_depth = params.frame_depth or DEFAULT_FRAME_DEPTH
    operation, user_defined = door_operation(element)
    key: tuple[object, ...] = (
        "IfcDoorType",
        element.type_id,
        operation,
        user_defined,
        round(width, 6),
        round(height, 6),
        round(frame_depth, 6),
    )
    if key in ctx.types:
        return ctx.types[key]
    f = ctx.file
    name = f"{base_name} {mm_text(width)}x{mm_text(height)} {operation}"
    entity = ifcopenshell.api.root.create_entity(
        f, ifc_class="IfcDoorType", predefined_type="DOOR", name=name
    )
    entity.OperationType = operation
    entity.ParameterTakesPrecedence = False
    if user_defined is not None:
        entity.UserDefinedOperationType = user_defined
    lining = f.create_entity(
        "IfcDoorLiningProperties",
        GlobalId="0",
        Name="Lining",
        LiningDepth=float(frame_depth),
        LiningThickness=LINING_THICKNESS,
    )
    panel = f.create_entity(
        "IfcDoorPanelProperties",
        GlobalId="0",
        Name="Panel",
        PanelDepth=float(min(frame_depth, PANEL_DEPTH)),
        PanelOperation=door_panel_operation(operation),
        PanelWidth=1.0,
        PanelPosition="MIDDLE",
    )
    for index, item in enumerate((lining, panel)):
        set_key_guid(ctx, item, f"type/{key!r}/props/{index}")
    entity.HasPropertySets = [lining, panel]
    set_key_guid(ctx, entity, f"type/{key!r}")
    ctx.types[key] = entity
    return entity
