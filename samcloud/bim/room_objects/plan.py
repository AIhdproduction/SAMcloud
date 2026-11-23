"""Write detected screens and ceiling objects into the plan model (metres in, local plan units out)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Polygon

from samcloud.bim.room_objects import ceiling, screens
from samcloud.bim.room_objects.evidence import SCREEN_LABELS, RoomEvidence, canonical
from samcloud.bim.room_objects.extent import discover_wall_shelves
from samcloud.bim.room_objects.geometry import enclosed_spaces
from samcloud.bim.room_objects.settings import RoomObjectSettings

CEILING_LAYER = "A-CLNG-OBJ"
_CARRIER_AREA_M2 = 0.35  # a plan outline larger than this is no screen seen from above
_SCREEN_NAMES = frozenset({*SCREEN_LABELS, "fernseher", "bildschirm", "notebook"})


def _mesh_dict(evidence: RoomEvidence, mesh: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    return {"vertices": evidence.vertices_local(mesh[0]), "faces": mesh[1].tolist()}


def _is_screen(item: Any) -> bool:
    return any(canonical(text) in _SCREEN_NAMES for text in (item.semantic_label, item.label))


def add_room_objects(
    items: list[Any], evidence: RoomEvidence, *, rooms: list[Polygon], wall_lines: list[LineString], wall_area: Any,
    settings: RoomObjectSettings, item_factory: Callable[..., Any], window_lines: list[LineString] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Append the screens to ``items`` and return (ceiling objects as plan dicts, report).

    ``rooms``, ``wall_lines`` and ``wall_area`` are in local plan units.  A plan item that already stood for a found
    screen is replaced by the measured panel, so every screen is present once; a plan "screen" the scan rejected as
    such (not flat, no screen term) is dropped.
    """
    scale = 1.0 / evidence.units_per_metre

    def metric(shape: Any) -> Any:
        return affinity.scale(shape, scale, scale, origin=(0.0, 0.0))

    rooms_m = [metric(room) for room in rooms]
    lines_m = [metric(line) for line in wall_lines]
    area_m = None if wall_area is None or wall_area.is_empty else metric(wall_area)

    shelves = discover_wall_shelves(items, evidence, lines_m, settings.extent, item_factory)
    # a screen or a lamp in a room the plan did not recognise is still inside the building: the walls close that space in.
    # The spaces only place objects; no room is written.
    spaces = enclosed_spaces(
        rooms_m, lines_m, area_m, face_gap_m=settings.screens.space_face_gap_m,
        min_area_m2=settings.screens.space_min_area_m2, min_width_m=settings.screens.space_min_width_m,
    ) if rooms_m else []
    found_screens, screen_report = screens.detect_screens(
        evidence, rooms_m=rooms_m + spaces, wall_lines_m=lines_m, settings=settings.screens, reflection=settings.reflection,
        frames=settings.screen_frames, wall_area_m=area_m,
    )
    replaced: list[str] = []
    renamed: list[str] = []
    refused = {entry["instance_id"] for entry in screen_report["rejected"] if "instance_id" in entry}
    for item in list(items):
        if not _is_screen(item):
            continue
        try:
            outline = metric(Polygon(item.footprint)).buffer(0.05)
        except (TypeError, ValueError):
            continue
        same = any(
            (item.instance_id is not None and item.instance_id == screen.instance_id)
            or outline.intersection(screen.footprint).area >= 0.5 * screen.footprint.area
            for screen in found_screens
        )
        if same:
            largest = max(screen.footprint.area for screen in found_screens)
            if outline.buffer(-0.05).area > max(_CARRIER_AREA_M2, 4.0 * largest):
                # the desk or sideboard under the screen, named after the screen: it stays, only the name goes
                item.label, item.semantic_label, item.instance_id = "Möbel", "furniture", None
                renamed.append(item.id)
                continue
            items.remove(item)
            replaced.append(item.id)
        elif item.instance_id is not None and item.instance_id in refused:
            items.remove(item)
            replaced.append(item.id)
    number = max([int(item.id[1:]) for item in items if str(item.id)[1:].isdigit()] + [len(items) + len(replaced)]) + 1
    added: list[dict[str, Any]] = []
    for screen in found_screens:
        vertices = screen.mesh[0]
        item = item_factory(
            id=f"F{number:03d}", instance_id=screen.instance_id, label=screens.DISPLAY_NAMES[screen.semantic],
            semantic_label=screen.semantic, catalogue_kind=None, layer="A-FURN",
            footprint=evidence.xy_local(screen.footprint.exterior.coords[:-1]),
            z_min=evidence.z_local(float(vertices[:, 2].min())), z_max=evidence.z_local(float(vertices[:, 2].max())),
            z_source="points", floor_contact=screen.mounting == "standing", mounting=screen.mounting,
            facing_deg=screen.facing_deg, mesh=_mesh_dict(evidence, screen.mesh), shape_source="mesh",
            fit={
                "added": "screen", "model": "panel", "points": screen.points, "width_m": round(screen.width, 3),
                "height_m": round(screen.height, 3), "thickness_m": round(screen.thickness, 3),
                "tilt_deg": round(screen.tilt_deg, 2), "stand": screen.has_stand,
                "centre": evidence.vertices_local(screen.centre[None, :])[0],
            },
        )
        number += 1
        items.append(item)
        added.append({"id": item.id, "label": screen.semantic, "width_m": round(screen.width, 3), "height_m": round(screen.height, 3)})

    found, ceiling_report = ceiling.detect_ceiling_objects(
        evidence, rooms_m=rooms_m + spaces, wall_lines_m=lines_m, wall_area_m=area_m, settings=settings.ceiling,
        reflection=settings.reflection, window_lines_m=[metric(line) for line in window_lines or []],
    )
    objects: list[dict[str, Any]] = []
    for index, entry in enumerate(found, start=1):
        a, b, c = (float(value) for value in entry.plane)
        objects.append({
            "id": f"C{index:03d}", "semantic_label": entry.semantic, "label": ceiling.DISPLAY_NAMES[entry.semantic],
            "layer": CEILING_LAYER, "mounting": "ceiling", "instance_id": entry.instance_id,
            "footprint": evidence.xy_local(entry.outline.exterior.coords[:-1]),
            "z_min": evidence.z_local(entry.z_min), "z_max": evidence.z_local(entry.z_max), "z_source": "points",
            "shape": entry.shape, "shape_source": "mesh", "mesh": _mesh_dict(evidence, entry.mesh),
            # ceiling z = a x + b y + c in local plan units, as PlanRoom.ceiling_plane
            "ceiling_plane": [a, b, evidence.z_local(c)],
            "drop_m": round(entry.drop, 3), "points": entry.points, "luminous": bool(entry.luminous),
        })
    report = {
        "wall_shelves_added": shelves, "screens": added,
        "screen_spaces_without_room": [
            {"area_m2": round(space.area, 2), "centre": [round(value, 3) for value in space.centroid.coords[0]]} for space in spaces
        ], "screen_items_replaced": replaced, "screen_carriers_renamed": renamed, "screens_rejected": screen_report["rejected"],
        "ceiling_objects": len(objects), "ceiling_rejected": ceiling_report["rejected"],
    }
    return objects, report
