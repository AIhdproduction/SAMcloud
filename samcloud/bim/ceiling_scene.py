"""Ceiling objects of the plan model as scene elements (lamps, detectors, grilles, unknown fixtures)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from shapely.geometry import Polygon

from samcloud.bim.core.model import FurnitureElement, Scene
from samcloud.bim.core.model.elements import OpaqueParams
from samcloud.bim.core.model.scene import Storey
from samcloud.bim.furniture import FurnitureMapping

CEILING_LAYER = "A-CLNG-OBJ"


def add_ceiling_objects(
    plan: Any, scene: Scene, storey: Storey, building_id: str, uid: Callable[[str], str],
    room_polygons: list[tuple[str, Polygon, Polygon]], mapping: FurnitureMapping, warnings: list[str],
) -> int:
    """Add ``plan.ceiling_objects`` and return how many were added.

    The objects keep their measured height: they follow the ceiling (also a sloped one, which rises above the storey
    height), so nothing is clipped at the storey top or pulled to the floor.
    """
    upm = plan.units_per_metre or 1.0
    numbers: dict[str, int] = {}
    added = 0
    for entry in plan.ceiling_objects:
        try:
            vertices = np.asarray(entry["mesh"]["vertices"], dtype=float) / upm
            faces = [[int(v) for v in face] for face in entry["mesh"]["faces"]]
            polygon = Polygon(np.asarray(entry["footprint"], dtype=float) / upm)
            z0, z1 = float(vertices[:, 2].min()), float(vertices[:, 2].max())
            usable = polygon.is_valid and polygon.area > 1e-4 and z1 - z0 >= 0.01 and len(faces) >= 4
        except (KeyError, TypeError, ValueError, IndexError):
            usable = False
        if not usable:
            warnings.append(f"ceiling object {entry.get('id')}: unusable geometry, skipped")
            continue
        label, semantic = str(entry.get("label") or ""), str(entry.get("semantic_label") or "ceiling fixture")
        klass, object_type = mapping.resolve(label=label, semantic_label=semantic, layer=CEILING_LAYER)
        if klass.key == mapping.default and "ceiling_proxy" in mapping.classes:
            klass = mapping.classes["ceiling_proxy"]
            object_type = klass.object_type or object_type
        point = polygon.representative_point()
        space = next((sid for sid, clipped, original in room_polygons if original.contains(point) or clipped.contains(point)), None)
        name = label or object_type
        numbers[name] = numbers.get(name, 0) + 1
        ring = [[float(x), float(y)] for x, y in polygon.exterior.coords[:-1]]
        scene.add(
            FurnitureElement(
                id=uid(str(entry.get("id"))),
                name=f"{name} {numbers[name]:02d}",
                tag=str(entry.get("id")),
                ifc_class=klass.ifc_class,
                predefined_type=klass.predefined_type,
                storey_id=storey.id,
                building_id=building_id,
                space_ids=[space] if space else [],
                params=OpaqueParams(
                    footprint=ring,
                    z_bottom=z0,
                    z_top=z1,
                    parts=[{"footprint": ring, "z_bottom": z0, "z_top": z1}],
                    mesh={"vertices": vertices.tolist(), "faces": faces},
                    mounting="ceiling",
                    object_type=object_type,
                    colour=list(klass.colour),
                    label=label,
                    semantic_label=semantic,
                    instance_id=entry.get("instance_id"),
                    confidence=entry.get("confidence"),
                    z_source=str(entry.get("z_source") or "points"),
                    layer=CEILING_LAYER,
                    catalogue_kind=None,
                    class_key=klass.key,
                    facing_deg=None,
                    back_wall_id=None,
                    shape_source="mesh",
                ),
            )
        )
        added += 1
    return added
