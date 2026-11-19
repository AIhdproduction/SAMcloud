"""Generate the SAMcloud IDS file (IDS 1.0) from the mandatory attribute list of MASTERPROMPT 10.1.

One specification per element type, driven by the tables below. Only what the IFC writer is
specified to emit is required (docs/IFC_MAPPING.md, section "Implementation notes (WP1.5)").

Not expressible in IDS 1.0 and therefore checked by the re-import test of
:mod:`samcloud.bim.core.qa.validate` instead: the type object (IfcRelDefinesByType), the material layer set
and first level space boundaries (IDS has no facet for these relations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ifctester import ids

from samcloud.core.paths import CONFIG_DIRECTORY

IDS_PATH = CONFIG_DIRECTORY / "bim" / "samcloud.ids"
"""Checked-in IDS file in the repository (``config/bim/samcloud.ids``)."""

IDS_VERSION = "1.0"
IDS_DATE = "2026-11-07"
IDS_AUTHOR = "samcloud@example.com"

LENGTH = "IFCLENGTHMEASURE"
AREA = "IFCAREAMEASURE"
VOLUME = "IFCVOLUMEMEASURE"
BOOLEAN = "IFCBOOLEAN"
COUNT = "IFCCOUNTMEASURE"
POSITIVE_LENGTH = "IFCPOSITIVELENGTHMEASURE"

SCAN_PSET = "SAMcloud_Scan"
"""Property set of furniture / equipment from the scan (not a ``Pset_`` set)."""
SCAN_PROPERTIES: dict[str, str | None] = {
    "Label": None,
    "ZSource": None,
    "Layer": None,
    "Length": LENGTH,
    "Width": LENGTH,
    "Height": LENGTH,
    "ZMin": LENGTH,
}
"""Always written; InstanceId, Confidence and CatalogueKind exist only when known."""
FURNITURE_CLASSES: tuple[tuple[str, str], ...] = (
    ("IfcFurniture", "IFCFURNITURE"),
    ("IfcElectricAppliance", "IFCELECTRICAPPLIANCE"),
    ("IfcSanitaryTerminal", "IFCSANITARYTERMINAL"),
    ("IfcAudioVisualAppliance", "IFCAUDIOVISUALAPPLIANCE"),
    ("IfcLightFixture", "IFCLIGHTFIXTURE"),
    ("IfcSpaceHeater", "IFCSPACEHEATER"),
    ("IfcBuildingElementProxy", "IFCBUILDINGELEMENTPROXY"),
)

PropertyTable = dict[str, dict[str, str | None]]
"""Property set name to {property name: IFC data type or None for any type}."""

# Raumlot_Quality: written on every exported element (RunId, backend and model id carry a
# placeholder when unknown, MASTERPROMPT 10.1 point 5).
QUALITY: dict[str, str | None] = {
    "ElementId": None,
    "Provenance": None,
    "RunId": None,
    "AiBackend": None,
    "AiModelId": None,
    "ToleranceProfile": None,
    "DataSource": None,
    "Locked": None,
    "Status": None,
    "Incomplete": None,
    "ParameterOrigins": None,
}

# Raumlot_Location: written on every exported element, "not applicable" for absent levels.
LOCATION: dict[str, str | None] = {
    "SiteName": None,
    "BuildingName": None,
    "StoreyName": None,
    "StoreyElevation": None,
    "SpaceNames": None,
    "SpaceNumbers": None,
    "RoomAssignment": None,
    "ReferenceX": None,
    "ReferenceY": None,
    "ReferenceZ": None,
    "BBoxMinX": None,
    "BBoxMinY": None,
    "BBoxMinZ": None,
    "BBoxMaxX": None,
    "BBoxMaxY": None,
    "BBoxMaxZ": None,
    "CrsName": None,
    "HeightSystem": None,
}

STOREY_RELATION = "IFCRELCONTAINEDINSPATIALSTRUCTURE"
HOST_RELATION = "IFCRELVOIDSELEMENT IFCRELFILLSELEMENT"
"""The only spelling ifctester accepts for the void/fill chain (opening or filling to host)."""


@dataclass(frozen=True)
class PartOfRule:
    """A PartOf requirement."""

    name: str
    relation: str


@dataclass(frozen=True)
class TypeSpec:
    """Requirements for one element type."""

    name: str
    ifc_class: str
    predefined_type: str | None = None
    attributes: tuple[tuple[str, str | None], ...] = ()
    """(attribute name, required value or None for any non-empty value)."""
    properties: PropertyTable = field(default_factory=dict)
    part_of: tuple[PartOfRule, ...] = ()
    raumlot: bool = True
    """Require Raumlot_Quality and Raumlot_Location."""
    description: str = ""


_COMMON_ATTRIBUTES: tuple[tuple[str, str | None], ...] = (
    ("Name", None),
    ("Description", None),
    ("ObjectType", None),
    ("Tag", None),
)
_IN_STOREY = (PartOfRule("IFCBUILDINGSTOREY", STOREY_RELATION),)
_SILL: dict[str, str | None] = {"SillHeight": None}
_SLAB_QUANTITIES: dict[str, str | None] = {
    "Depth": LENGTH,
    "Perimeter": LENGTH,
    "GrossArea": AREA,
    "NetArea": AREA,
    "GrossVolume": VOLUME,
    "NetVolume": VOLUME,
}

SPECS: tuple[TypeSpec, ...] = (
    TypeSpec(
        name="IfcWall",
        ifc_class="IFCWALL",
        attributes=(*_COMMON_ATTRIBUTES, ("PredefinedType", None)),
        properties={
            "Pset_WallCommon": {"Reference": None, "IsExternal": BOOLEAN},
            "Qto_WallBaseQuantities": {
                "Length": LENGTH,
                "Width": LENGTH,
                "Height": LENGTH,
                "GrossFootprintArea": AREA,
                "GrossSideArea": AREA,
                "NetSideArea": AREA,
                "GrossVolume": VOLUME,
                "NetVolume": VOLUME,
            },
        },
        part_of=_IN_STOREY,
        description="Walls: identity, dimensions, quantities, location, quality.",
    ),
    TypeSpec(
        name="IfcSlab floor",
        ifc_class="IFCSLAB",
        predefined_type="FLOOR",
        attributes=_COMMON_ATTRIBUTES,
        properties={
            "Pset_SlabCommon": {"Reference": None},
            "Qto_SlabBaseQuantities": dict(_SLAB_QUANTITIES),
        },
        part_of=_IN_STOREY,
        description="Floor slabs.",
    ),
    TypeSpec(
        name="IfcSlab ceiling",
        ifc_class="IFCSLAB",
        predefined_type="USERDEFINED",
        attributes=(
            ("Name", None),
            ("Description", None),
            ("ObjectType", "CEILING"),
            ("Tag", None),
        ),
        properties={
            "Pset_SlabCommon": {"Reference": None},
            "Qto_SlabBaseQuantities": dict(_SLAB_QUANTITIES),
        },
        part_of=_IN_STOREY,
        description="Ceilings exported as IfcSlab (USERDEFINED, ObjectType CEILING).",
    ),
    TypeSpec(
        name="IfcCovering ceiling",
        ifc_class="IFCCOVERING",
        predefined_type="CEILING",
        attributes=_COMMON_ATTRIBUTES,
        properties={
            "Pset_CoveringCommon": {"Reference": None},
            "Qto_CoveringBaseQuantities": {"Width": LENGTH, "GrossArea": AREA, "NetArea": AREA},
        },
        part_of=_IN_STOREY,
        description="Ceilings exported as IfcCovering (ceiling_ifc_class covering).",
    ),
    TypeSpec(
        name="IfcOpeningElement",
        ifc_class="IFCOPENINGELEMENT",
        attributes=(*_COMMON_ATTRIBUTES, ("PredefinedType", None)),
        properties={
            "Pset_OpeningElementCommon": {"Reference": None},
            "Qto_OpeningElementBaseQuantities": {
                "Width": LENGTH,
                "Height": LENGTH,
                "Depth": LENGTH,
                "Area": AREA,
            },
            "Raumlot_Quality": dict(_SILL),
        },
        part_of=(PartOfRule("IFCWALL", HOST_RELATION),),
        description="Openings: voiding a host wall, dimensions, sill height.",
    ),
    TypeSpec(
        name="IfcWindow",
        ifc_class="IFCWINDOW",
        attributes=(*_COMMON_ATTRIBUTES, ("PredefinedType", None), ("PartitioningType", None)),
        properties={
            "Pset_WindowCommon": {"Reference": None, "IsExternal": BOOLEAN},
            "Qto_WindowBaseQuantities": {"Width": LENGTH, "Height": LENGTH, "Area": AREA},
            "Raumlot_Quality": dict(_SILL),
        },
        part_of=(*_IN_STOREY, PartOfRule("IFCWALL", HOST_RELATION)),
        description="Windows: filling an opening of a host wall, dimensions, sill height.",
    ),
    TypeSpec(
        name="IfcDoor",
        ifc_class="IFCDOOR",
        attributes=(*_COMMON_ATTRIBUTES, ("PredefinedType", None)),
        properties={
            "Pset_DoorCommon": {"Reference": None, "IsExternal": BOOLEAN},
            "Qto_DoorBaseQuantities": {"Width": LENGTH, "Height": LENGTH, "Area": AREA},
            "Raumlot_Quality": dict(_SILL),
        },
        part_of=(*_IN_STOREY, PartOfRule("IFCWALL", HOST_RELATION)),
        description="Doors: filling an opening of a host wall, dimensions, sill height.",
    ),
    TypeSpec(
        name="IfcSpace",
        ifc_class="IFCSPACE",
        attributes=(
            ("Name", None),
            ("Description", None),
            ("ObjectType", None),
            ("LongName", None),
            ("PredefinedType", None),
        ),
        properties={
            "Pset_SpaceCommon": {"Reference": None, "IsExternal": BOOLEAN},
            "Qto_SpaceBaseQuantities": {
                "Height": LENGTH,
                "GrossFloorArea": AREA,
                "NetFloorArea": AREA,
                "GrossPerimeter": LENGTH,
                "GrossVolume": VOLUME,
                "NetVolume": VOLUME,
            },
        },
        part_of=(PartOfRule("IFCBUILDINGSTOREY", "IFCRELAGGREGATES"),),
        description="Spaces (zones): name, number, area, volume, perimeter, height.",
    ),
    TypeSpec(
        name="IfcStair",
        ifc_class="IFCSTAIR",
        attributes=(*_COMMON_ATTRIBUTES, ("PredefinedType", None)),
        properties={
            "Pset_StairCommon": {
                "Reference": None,
                "NumberOfRiser": COUNT,
                "NumberOfTreads": COUNT,
                "RiserHeight": POSITIVE_LENGTH,
                "TreadLength": POSITIVE_LENGTH,
            },
            "Raumlot_Quality": {
                "StairWidth": LENGTH,
                "TotalRise": LENGTH,
                "TotalRun": LENGTH,
            },
        },
        part_of=_IN_STOREY,
        description="Stairs: step counts, riser height, tread length, width, rise and run.",
    ),
    *(
        TypeSpec(
            name=name,
            ifc_class=ifc_class,
            attributes=(("Name", None), ("Description", None), ("ObjectType", None), ("PredefinedType", None)),
            properties={SCAN_PSET: dict(SCAN_PROPERTIES)},
            part_of=_IN_STOREY,
            raumlot=False,
            description=f"{name}: scanned furniture / equipment with name, type, storey and SAMcloud_Scan.",
        )
        for name, ifc_class in FURNITURE_CLASSES
    ),
    TypeSpec(
        name="IfcBuildingStorey",
        ifc_class="IFCBUILDINGSTOREY",
        attributes=(("Name", None), ("Elevation", None)),
        part_of=(PartOfRule("IFCBUILDING", "IFCRELAGGREGATES"),),
        raumlot=False,
        description="Storeys: name, elevation, aggregated into the building.",
    ),
)
"""One specification per element type (MASTERPROMPT 10.1)."""


def _build_specification(spec: TypeSpec) -> ids.Specification:
    result = ids.Specification(
        name=spec.name,
        ifcVersion=["IFC4"],
        minOccurs=0,
        description=spec.description or None,
    )
    result.applicability.append(
        ids.Entity(name=spec.ifc_class, predefinedType=spec.predefined_type)
    )
    for attribute, value in spec.attributes:
        result.requirements.append(ids.Attribute(name=attribute, value=value))
    for rule in spec.part_of:
        result.requirements.append(ids.PartOf(name=rule.name, relation=rule.relation))

    tables: PropertyTable = {pset: dict(props) for pset, props in spec.properties.items()}
    if spec.raumlot:
        tables.setdefault("Raumlot_Quality", {}).update(QUALITY)
        tables["Raumlot_Location"] = dict(LOCATION)
    for pset, props in tables.items():
        for name, data_type in props.items():
            result.requirements.append(
                ids.Property(propertySet=pset, baseName=name, dataType=data_type)
            )
    return result


def build_ids() -> ids.Ids:
    """Build the IDS object (deterministic: fixed title, version, author and date)."""
    document = ids.Ids(
        title="SAMcloud export gate",
        description=(
            "Mandatory attributes of every SAMcloud IFC export (MASTERPROMPT 10.1). "
            "Generated by samcloud.bim.core.qa.ids_spec; do not edit by hand."
        ),
        author=IDS_AUTHOR,
        version=IDS_VERSION,
        date=IDS_DATE,
        purpose="Export gate",
        milestone="M1",
    )
    for spec in SPECS:
        document.specifications.append(_build_specification(spec))
    return document


def write_ids(path: str | Path = IDS_PATH) -> Path:
    """Write the IDS XML to ``path`` (default ``config/bim/samcloud.ids``), validated against the XSD."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not build_ids().to_xml(str(target)):
        raise ValueError(f"generated IDS {target} does not validate against the IDS schema")
    return target
