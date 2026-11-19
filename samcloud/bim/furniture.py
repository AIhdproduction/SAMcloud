"""Furniture mapping for the IFC export: plan labels to IFC4 class, predefined type and style.

The table lives in ``config/bim_furniture.yaml``.  :meth:`FurnitureMapping.resolve` maps one
``PlanFurniture`` (catalogue kind, semantic label, label, layer) to a :class:`FurnitureClass`.
IFC4 notes: the predefined type sits on the occurrence (``IfcFurniture.PredefinedType`` exists in
IFC4 ADD2) and on the type object (``IfcFurnitureType``); USERDEFINED entries carry the user type in
``ObjectType`` / ``ElementType``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_FALLBACK_HEIGHT_M = 0.8


@dataclass(frozen=True)
class FurnitureClass:
    """One entry of ``classes`` in bim_furniture.yaml."""

    key: str
    ifc_class: str
    predefined_type: str
    object_type: str | None = None
    colour: tuple[float, float, float] = (0.65, 0.65, 0.65)


@dataclass(frozen=True)
class FurnitureMapping:
    classes: dict[str, FurnitureClass]
    by_catalogue_kind: dict[str, str]
    by_label: dict[str, str]
    by_layer: dict[str, str]
    default: str
    fallback_height_m: float = DEFAULT_FALLBACK_HEIGHT_M

    def resolve(
        self,
        *,
        label: str = "",
        semantic_label: str = "",
        catalogue_kind: str | None = None,
        layer: str = "",
    ) -> tuple[FurnitureClass, str]:
        """Return ``(class, ObjectType)``.

        ObjectType is the class's own ``object_type`` for USERDEFINED entries, the plan label for
        the generic proxy, and the title-cased predefined type otherwise.
        """
        key = self._class_key(label, semantic_label, catalogue_kind, layer)
        cls = self.classes[key]
        if cls.object_type:
            return cls, cls.object_type
        if cls.key == self.default or cls.predefined_type in ("NOTDEFINED", "USERDEFINED"):
            return cls, (label or semantic_label or cls.key).strip() or cls.key
        return cls, _title(cls.predefined_type)

    def _class_key(self, label: str, semantic_label: str, catalogue_kind: str | None, layer: str) -> str:
        if catalogue_kind:
            hit = self.by_catalogue_kind.get(catalogue_kind) or self.by_catalogue_kind.get(catalogue_kind.lower())
            if hit:
                return hit
        for text in (semantic_label, label):
            hit = self.by_label.get(str(text or "").strip().lower())
            if hit:
                return hit
        return self.by_layer.get(layer) or self.default


def _title(value: str) -> str:
    return value.replace("_", " ").title().replace(" ", "")


def load_furniture_mapping(path: Path | str | None = None) -> FurnitureMapping:
    """Load and validate ``config/bim_furniture.yaml`` (every reference must exist in ``classes``)."""
    if path is None:
        from samcloud.core.paths import CONFIG_DIRECTORY

        path = Path(CONFIG_DIRECTORY) / "bim_furniture.yaml"
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    classes: dict[str, FurnitureClass] = {}
    for key, value in (raw.get("classes") or {}).items():
        colour = tuple(float(c) for c in value.get("colour", (0.65, 0.65, 0.65)))
        classes[str(key)] = FurnitureClass(
            key=str(key),
            ifc_class=str(value["ifc_class"]),
            predefined_type=str(value["predefined_type"]),
            object_type=value.get("object_type"),
            colour=(colour[0], colour[1], colour[2]),
        )
    default = str(raw.get("default", "proxy"))
    if default not in classes:
        raise ValueError(f"default furniture class {default!r} is not defined in {path}")
    tables = {
        name: {str(k).lower() if name != "by_layer" else str(k): str(v) for k, v in (raw.get(name) or {}).items()}
        for name in ("by_catalogue_kind", "by_label", "by_layer")
    }
    # catalogue kinds keep their case (upper-case catalogue keys, lower-case einb kinds)
    tables["by_catalogue_kind"] = {str(k): str(v) for k, v in (raw.get("by_catalogue_kind") or {}).items()}
    for name, table in tables.items():
        for source, target in table.items():
            if target not in classes:
                raise ValueError(f"{name}[{source!r}] points to unknown furniture class {target!r}")
    return FurnitureMapping(
        classes=classes,
        by_catalogue_kind=tables["by_catalogue_kind"],
        by_label=tables["by_label"],
        by_layer=tables["by_layer"],
        default=default,
        fallback_height_m=float(raw.get("fallback_height_m", DEFAULT_FALLBACK_HEIGHT_M)),
    )
