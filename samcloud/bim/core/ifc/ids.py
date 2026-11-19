"""Deterministic IFC GlobalIds (same scene gives the same ids on every export)."""

from __future__ import annotations

import uuid

import ifcopenshell
import ifcopenshell.guid

from samcloud.bim.core.ifc.context import NAMESPACE, ExportContext, class_of


def guid_from_uuid(value: uuid.UUID) -> str:
    """22 character IFC GUID of a uuid."""
    return str(ifcopenshell.guid.compress(value.hex))


def uuid_of(text: str) -> uuid.UUID:
    """The uuid of an element id; ids that are not uuids are hashed (uuid5)."""
    try:
        return uuid.UUID(text)
    except ValueError:
        return uuid.uuid5(NAMESPACE, text)


def set_guid(ctx: ExportContext, entity: ifcopenshell.entity_instance, key: str) -> None:
    """Give an IfcRoot entity the GlobalId derived from a scene id (uuid) or a stable key."""
    entity.GlobalId = guid_from_uuid(uuid_of(key))
    ctx.explicit_guids.add(entity.id())


def set_key_guid(ctx: ExportContext, entity: ifcopenshell.entity_instance, key: str) -> None:
    """Give an IfcRoot entity a GlobalId derived from a stable text key (uuid5)."""
    entity.GlobalId = guid_from_uuid(uuid.uuid5(NAMESPACE, key))
    ctx.explicit_guids.add(entity.id())


def finalize_guids(ctx: ExportContext) -> None:
    """Replace the random GlobalIds of API-created entities by deterministic ones.

    Entities are visited in file order (creation order is deterministic) and numbered per class.
    """
    counters: dict[str, int] = {}
    roots = sorted(ctx.file.by_type("IfcRoot"), key=lambda entity: entity.id())
    for entity in roots:
        if entity.id() in ctx.explicit_guids:
            continue
        ifc_class = class_of(entity)
        counters[ifc_class] = counters.get(ifc_class, 0) + 1
        entity.GlobalId = guid_from_uuid(
            uuid.uuid5(NAMESPACE, f"auto/{ifc_class}/{counters[ifc_class]}")
        )


def normalize_relation_order(ctx: ExportContext) -> None:
    """Sort the related objects of relationships by entity id.

    Several ifcopenshell API functions store related objects through a Python set, whose
    iteration order is not stable between runs. Sorting makes the file content deterministic.
    """
    for relation in ctx.file.by_type("IfcRelationship"):
        for attribute in ("RelatedObjects", "RelatedElements"):
            if not hasattr(relation, attribute):
                continue
            items = getattr(relation, attribute)
            if items and len(items) > 1:
                setattr(relation, attribute, sorted(items, key=lambda item: item.id()))
