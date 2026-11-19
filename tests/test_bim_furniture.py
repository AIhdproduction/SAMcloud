"""config/bim_furniture.yaml against the IFC4 schema and the resolution rules."""

from __future__ import annotations

import ifcopenshell
import pytest

from samcloud.bim.furniture import load_furniture_mapping

SCHEMA = ifcopenshell.schema_by_name("IFC4")
MAPPING = load_furniture_mapping()


def _enum_items(declaration_name: str, attribute: str) -> tuple[str, ...]:
    declaration = SCHEMA.declaration_by_name(declaration_name)
    for attr in declaration.all_attributes():
        if attr.name() == attribute:
            type_decl = attr.type_of_attribute()
            if type_decl.as_named_type() is not None:
                type_decl = type_decl.as_named_type().declared_type()
            enumeration = type_decl.as_enumeration_type() or type_decl.declared_type().as_enumeration_type()
            return tuple(enumeration.enumeration_items())
    raise AssertionError(f"{declaration_name} has no attribute {attribute}")


@pytest.mark.parametrize("key", sorted(MAPPING.classes))
def test_every_class_is_valid_ifc4(key: str) -> None:
    entry = MAPPING.classes[key]
    declaration = SCHEMA.declaration_by_name(entry.ifc_class)
    assert declaration.is_abstract() is False
    # a building element / furnishing, not a type or relationship
    chain = []
    parent = declaration
    while parent is not None:
        chain.append(parent.name())
        parent = parent.supertype()
    assert "IfcElement" in chain, f"{entry.ifc_class} is not an IfcElement"
    # the type object exists and carries the same enumeration
    type_name = f"{entry.ifc_class}Type"
    SCHEMA.declaration_by_name(type_name)
    occurrence_items = _enum_items(entry.ifc_class, "PredefinedType")
    type_items = _enum_items(type_name, "PredefinedType")
    assert entry.predefined_type in occurrence_items, (entry.ifc_class, entry.predefined_type)
    assert entry.predefined_type in type_items, (type_name, entry.predefined_type)
    if entry.predefined_type == "USERDEFINED":
        assert entry.object_type, f"{key}: USERDEFINED needs an object_type"
    assert len(entry.colour) == 3 and all(0.0 <= c <= 1.0 for c in entry.colour)


def test_expected_table_rows() -> None:
    expected = {
        "chair": ("IfcFurniture", "CHAIR"),
        "sofa": ("IfcFurniture", "SOFA"),
        "table": ("IfcFurniture", "TABLE"),
        "desk": ("IfcFurniture", "DESK"),
        "bed": ("IfcFurniture", "BED"),
        "shelf": ("IfcFurniture", "SHELF"),
        "refrigerator": ("IfcElectricAppliance", "REFRIGERATOR"),
        "cooker": ("IfcElectricAppliance", "ELECTRICCOOKER"),
        "dishwasher": ("IfcElectricAppliance", "DISHWASHER"),
        "washing_machine": ("IfcElectricAppliance", "WASHINGMACHINE"),
        "dryer": ("IfcElectricAppliance", "TUMBLEDRYER"),
        "microwave": ("IfcElectricAppliance", "MICROWAVE"),
        "sink": ("IfcSanitaryTerminal", "SINK"),
        "washbasin": ("IfcSanitaryTerminal", "WASHHANDBASIN"),
        "toilet": ("IfcSanitaryTerminal", "TOILETPAN"),
        "bathtub": ("IfcSanitaryTerminal", "BATH"),
        "shower": ("IfcSanitaryTerminal", "SHOWER"),
        "tv": ("IfcAudioVisualAppliance", "DISPLAY"),  # IFC4 has no TELEVISION literal
        "lamp": ("IfcLightFixture", "NOTDEFINED"),
        "radiator": ("IfcSpaceHeater", "RADIATOR"),
        "proxy": ("IfcBuildingElementProxy", "NOTDEFINED"),
    }
    for key, (ifc_class, predefined) in expected.items():
        assert (MAPPING.classes[key].ifc_class, MAPPING.classes[key].predefined_type) == (ifc_class, predefined), key
    for key, object_type in {
        "wardrobe": "Wardrobe", "cabinet": "Cabinet", "kitchen_unit": "KitchenUnit",
        "tall_unit": "TallUnit", "wall_cabinet": "WallCabinet",
    }.items():
        entry = MAPPING.classes[key]
        assert (entry.ifc_class, entry.predefined_type, entry.object_type) == ("IfcFurniture", "USERDEFINED", object_type)


def test_television_is_not_an_ifc4_literal() -> None:
    assert "TELEVISION" not in _enum_items("IfcAudioVisualAppliance", "PredefinedType")


def test_all_references_point_to_defined_classes() -> None:
    for table in (MAPPING.by_catalogue_kind, MAPPING.by_label, MAPPING.by_layer):
        assert set(table.values()) <= set(MAPPING.classes)
    assert MAPPING.default in MAPPING.classes


@pytest.mark.parametrize(
    ("kwargs", "ifc_class", "predefined", "object_type"),
    [
        ({"label": "Stuhl", "semantic_label": "chair", "layer": "A-FURN"}, "IfcFurniture", "CHAIR", "Chair"),
        ({"label": "Möbel", "semantic_label": "bed", "layer": "A-FURN"}, "IfcFurniture", "BED", "Bed"),
        ({"label": "Schrank", "semantic_label": "wardrobe", "layer": "A-FURN"}, "IfcFurniture", "USERDEFINED", "Wardrobe"),
        ({"label": "Kommode", "semantic_label": "chest of drawers", "layer": "A-FURN"}, "IfcFurniture", "USERDEFINED", "Cabinet"),
        ({"label": "upper_cabinet", "semantic_label": "upper_cabinet", "catalogue_kind": "upper_cabinet", "layer": "A-OBEN"}, "IfcFurniture", "USERDEFINED", "WallCabinet"),
        ({"label": "tall_unit", "semantic_label": "tall_unit", "catalogue_kind": "tall_unit", "layer": "A-EINB"}, "IfcFurniture", "USERDEFINED", "TallUnit"),
        ({"label": "Herd", "semantic_label": "Herd", "catalogue_kind": "COOKTOP", "layer": "A-EINB"}, "IfcElectricAppliance", "ELECTRICCOOKER", "Electriccooker"),
        ({"label": "Spüle", "semantic_label": "Spüle", "catalogue_kind": "KITCHEN_SINK", "layer": "A-EINB"}, "IfcSanitaryTerminal", "SINK", "Sink"),
        ({"label": "WC", "semantic_label": "WC", "catalogue_kind": "WC_WALL", "layer": "P-SANI"}, "IfcSanitaryTerminal", "TOILETPAN", "Toiletpan"),
        ({"label": "Lavabo", "semantic_label": "Lavabo", "catalogue_kind": "WASHBASIN", "layer": "P-SANI"}, "IfcSanitaryTerminal", "WASHHANDBASIN", "Washhandbasin"),
        ({"label": "Badewanne", "semantic_label": "Badewanne", "catalogue_kind": "BATHTUB", "layer": "P-SANI"}, "IfcSanitaryTerminal", "BATH", "Bath"),
        ({"label": "Sideboard", "semantic_label": "tv stand", "layer": "A-FURN"}, "IfcFurniture", "USERDEFINED", "Cabinet"),
        ({"label": "Möbel", "semantic_label": "cushion", "layer": "A-FURN"}, "IfcBuildingElementProxy", "NOTDEFINED", "Möbel"),
        ({"label": "niche", "semantic_label": "niche", "catalogue_kind": "niche", "layer": "A-EINB"}, "IfcBuildingElementProxy", "NOTDEFINED", "niche"),
    ],
)
def test_resolution_of_real_plan_labels(kwargs: dict, ifc_class: str, predefined: str, object_type: str) -> None:
    klass, resolved_object_type = MAPPING.resolve(**kwargs)

    assert (klass.ifc_class, klass.predefined_type) == (ifc_class, predefined)
    assert resolved_object_type == object_type
