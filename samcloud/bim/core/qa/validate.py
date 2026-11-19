"""Validation chain after every IFC export (MASTERPROMPT 10.2).

1. ``ifcopenshell.validate`` with the EXPRESS rules (all findings are collected),
2. IfcTester against ``config/bim/samcloud.ids`` (per element missing attributes),
3. a re-import test: the IFC is reopened and compared with the scene (counts, dimensions,
   quantities, relations, types, units, georeference) at a tolerance of 1e-6 m.

The result is a :class:`ValidationReport`, also written as HTML and JSON next to the IFC file (or
into a given reports directory).
"""

from __future__ import annotations

import html
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ifcopenshell
import ifcopenshell.util.element as element_util
import ifcopenshell.util.unit
import ifcopenshell.validate
from ifctester import ids as ifc_ids
from ifctester import reporter as ids_reporter

from samcloud.bim.core.geometry import SceneGeometry, compute_scene_geometry
from samcloud.bim.core.ifc.context import IfcExportResult, class_of, is_a
from samcloud.bim.core.ifc.ids import guid_from_uuid, uuid_of
from samcloud.bim.core.model import Element, ElementKind, Scene
from samcloud.bim.core.qa.ids_spec import IDS_PATH

TOLERANCE = 1e-6
"""Re-import comparison tolerance (m, m2, m3)."""

Entity = ifcopenshell.entity_instance


@dataclass(frozen=True)
class IdsFailure:
    """One unmet IDS requirement of one element."""

    specification: str
    global_id: str
    tag: str | None
    name: str | None
    ifc_class: str
    requirement: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "specification": self.specification,
            "global_id": self.global_id,
            "tag": self.tag,
            "name": self.name,
            "ifc_class": self.ifc_class,
            "requirement": self.requirement,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ReimportMismatch:
    """One difference between the reopened IFC and the scene."""

    check: str
    """Category: count, element, quantity, relation, type, unit, georeference."""
    element_id: str
    """Scene element id, or an empty string for file-level checks."""
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"check": self.check, "element_id": self.element_id, "message": self.message}


@dataclass
class ValidationReport:
    """Result of :func:`validate_ifc`."""

    ifc_path: str
    ids_path: str
    express_findings: list[dict[str, Any]] = field(default_factory=list)
    ids_failures: list[IdsFailure] = field(default_factory=list)
    ids_specifications: list[dict[str, Any]] = field(default_factory=list)
    reimport_mismatches: list[ReimportMismatch] = field(default_factory=list)
    reimport_checked: bool = False
    """True when a scene was given and the re-import comparison ran."""
    incomplete: bool = False
    """The export is incomplete (mandatory data missing or an IDS failure, MASTERPROMPT 10.1)."""
    unreferenced: bool = False
    warnings: list[str] = field(default_factory=list)
    html_path: str | None = None
    json_path: str | None = None
    ids_html_path: str | None = None

    @property
    def ok(self) -> bool:
        """No EXPRESS finding, no IDS failure and no re-import mismatch."""
        return not (self.express_findings or self.ids_failures or self.reimport_mismatches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "incomplete": self.incomplete,
            "unreferenced": self.unreferenced,
            "ifc_path": self.ifc_path,
            "ids_path": self.ids_path,
            "express_findings": self.express_findings,
            "ids_specifications": self.ids_specifications,
            "ids_failures": [failure.to_dict() for failure in self.ids_failures],
            "reimport_checked": self.reimport_checked,
            "reimport_mismatches": [item.to_dict() for item in self.reimport_mismatches],
            "warnings": self.warnings,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_html(self) -> str:
        return _render_html(self)


# ----------------------------------------------------------------------------- EXPRESS


def _express_findings(file: ifcopenshell.file) -> list[dict[str, Any]]:
    logger = ifcopenshell.validate.json_logger()  # type: ignore[no-untyped-call]
    ifcopenshell.validate.validate(file, logger, express_rules=True)
    findings: list[dict[str, Any]] = []
    for statement in logger.statements:
        findings.append(
            {
                "level": str(statement.get("level", "")),
                "type": str(statement.get("type", "")),
                "message": str(statement.get("message", "")),
                "instance": str(statement.get("instance", "")),
                "attribute": str(statement.get("attribute", "")),
            }
        )
    return findings


# --------------------------------------------------------------------------------- IDS


def _ids_check(
    file: ifcopenshell.file, ids_path: Path, html_target: Path | None
) -> tuple[list[IdsFailure], list[dict[str, Any]], str | None]:
    document = ifc_ids.open(str(ids_path), validate=True)
    document.validate(file)
    result = ids_reporter.Json(document).report()
    failures: list[IdsFailure] = []
    summaries: list[dict[str, Any]] = []
    for spec in result["specifications"]:
        summaries.append(
            {
                "name": spec["name"],
                "status": bool(spec["status"]),
                "applicable": spec["total_applicable"],
                "applicable_failed": spec["total_applicable_fail"],
            }
        )
        for requirement in spec["requirements"]:
            for entity in requirement["failed_entities"]:
                failures.append(
                    IdsFailure(
                        specification=str(spec["name"]),
                        global_id=str(entity.get("global_id") or ""),
                        tag=entity.get("tag"),
                        name=entity.get("name"),
                        ifc_class=str(entity.get("class") or ""),
                        requirement=str(requirement["description"]),
                        reason=str(entity.get("reason") or ""),
                    )
                )
    html_path: str | None = None
    if html_target is not None:
        reporter = ids_reporter.Html(document)
        reporter.report()
        reporter.to_file(str(html_target))
        html_path = str(html_target)
    return failures, summaries, html_path


# ------------------------------------------------------------------------- re-import


def _close(expected: float, actual: object) -> bool:
    return (
        isinstance(actual, int | float)
        and not isinstance(actual, bool)
        and math.isfinite(float(actual))
        and abs(float(actual) - expected) <= TOLERANCE
    )


@dataclass(frozen=True)
class _QuantityRule:
    """Compare geometry quantity ``source`` with property ``target`` of ``pset``."""

    pset: str
    target: str
    source: str


def _rules(pset: str, names: tuple[str, ...], **renamed: str) -> list[_QuantityRule]:
    rules = [_QuantityRule(pset, name, name) for name in names]
    rules.extend(_QuantityRule(pset, target, source) for target, source in renamed.items())
    return rules


_WALL_RULES = _rules(
    "Qto_WallBaseQuantities",
    (
        "Length",
        "Width",
        "Height",
        "GrossFootprintArea",
        "GrossSideArea",
        "NetSideArea",
        "GrossVolume",
        "NetVolume",
    ),
)
_OPENING_RULES = [
    *_rules("Qto_OpeningElementBaseQuantities", ("Width", "Height", "Area")),
    _QuantityRule("Raumlot_Quality", "SillHeight", "SillHeight"),
]
_FILLING_RULES = {
    "window": [
        *_rules("Qto_WindowBaseQuantities", ("Width", "Height", "Area")),
        _QuantityRule("Raumlot_Quality", "SillHeight", "SillHeight"),
    ],
    "door": [
        *_rules("Qto_DoorBaseQuantities", ("Width", "Height", "Area")),
        _QuantityRule("Raumlot_Quality", "SillHeight", "SillHeight"),
    ],
}
_SLAB_RULES = _rules(
    "Qto_SlabBaseQuantities",
    ("Perimeter", "GrossArea", "NetArea", "GrossVolume", "NetVolume"),
    Depth="Width",
)
_COVERING_RULES = _rules("Qto_CoveringBaseQuantities", ("Width", "GrossArea", "NetArea"))
_SPACE_RULES = _rules(
    "Qto_SpaceBaseQuantities",
    (
        "Height",
        "GrossFloorArea",
        "NetFloorArea",
        "GrossPerimeter",
        "GrossVolume",
        "NetVolume",
    ),
)

_STAIR_RULES = [
    *_rules("Pset_StairCommon", ("NumberOfRiser", "NumberOfTreads", "RiserHeight", "TreadLength")),
    _QuantityRule("Raumlot_Quality", "StairWidth", "Width"),
    _QuantityRule("Raumlot_Quality", "TotalRise", "TotalRise"),
    _QuantityRule("Raumlot_Quality", "TotalRun", "TotalRun"),
]

_ELEMENT_CLASSES: dict[ElementKind, str] = {
    ElementKind.WALL: "IfcWall",
    ElementKind.SLAB: "IfcSlab",
    ElementKind.CEILING: "IfcSlab or IfcCovering",
    ElementKind.OPENING: "IfcOpeningElement",
    ElementKind.WINDOW: "IfcWindow",
    ElementKind.DOOR: "IfcDoor",
    ElementKind.SPACE: "IfcSpace",
    ElementKind.STAIR: "IfcStair",
}
_TYPED_KINDS = frozenset(
    {
        ElementKind.WALL,
        ElementKind.SLAB,
        ElementKind.CEILING,
        ElementKind.WINDOW,
        ElementKind.DOOR,
        ElementKind.STAIR,
    }
)


def _guid(scene_id: str) -> str:
    return guid_from_uuid(uuid_of(scene_id))


def _ifc_counts(file: ifcopenshell.file) -> Counter[ElementKind]:
    counts: Counter[ElementKind] = Counter()
    slabs = [
        slab
        for slab in file.by_type("IfcSlab", include_subtypes=False)
        if slab.PredefinedType != "LANDING"
    ]
    ceiling_slabs = [slab for slab in slabs if _is_ceiling(slab)]
    coverings = [
        covering
        for covering in file.by_type("IfcCovering", include_subtypes=False)
        if covering.PredefinedType == "CEILING"
    ]
    counts[ElementKind.WALL] = len(file.by_type("IfcWall", include_subtypes=False))
    counts[ElementKind.CEILING] = len(ceiling_slabs) + len(coverings)
    counts[ElementKind.SLAB] = len(slabs) - len(ceiling_slabs)
    counts[ElementKind.OPENING] = len(file.by_type("IfcOpeningElement"))
    counts[ElementKind.WINDOW] = len(file.by_type("IfcWindow", include_subtypes=False))
    counts[ElementKind.DOOR] = len(file.by_type("IfcDoor", include_subtypes=False))
    counts[ElementKind.SPACE] = len(file.by_type("IfcSpace"))
    counts[ElementKind.STAIR] = len(file.by_type("IfcStair", include_subtypes=False))
    return counts


def _is_ceiling(slab: Entity) -> bool:
    return bool(slab.PredefinedType == "USERDEFINED" and slab.ObjectType == "CEILING")


class _Reimport:
    """Collects mismatches of one re-import comparison."""

    def __init__(self, file: ifcopenshell.file, scene: Scene, geometry: SceneGeometry) -> None:
        self.file = file
        self.scene = scene
        self.geometry = geometry
        self.mismatches: list[ReimportMismatch] = []
        self.by_guid: dict[str, Entity] = {
            str(entity.GlobalId): entity for entity in file.by_type("IfcRoot")
        }

    def add(self, check: str, element_id: str, message: str) -> None:
        self.mismatches.append(ReimportMismatch(check, element_id, message))

    def run(self) -> list[ReimportMismatch]:
        self._units()
        self._georeference()
        self._counts()
        for element in self.scene.elements.values():
            if element.kind not in _ELEMENT_CLASSES:
                continue
            entity = self.by_guid.get(_guid(element.id))
            if entity is None:
                self.add(
                    "element",
                    element.id,
                    f"{element.kind.value} {element.name!r} is missing in the IFC",
                )
                continue
            self._identity(element, entity)
            self._quantities(element, entity)
            self._relations(element, entity)
            if element.kind in _TYPED_KINDS and element_util.get_type(entity) is None:
                self.add("type", element.id, f"{class_of(entity)} has no type object")
        return self.mismatches

    def _units(self) -> None:
        scale = ifcopenshell.util.unit.calculate_unit_scale(self.file)
        if abs(scale - 1.0) > 1e-12:
            self.add("unit", "", f"length unit scale is {scale}, expected 1.0 (metres)")

    def _georeference(self) -> None:
        conversions = self.file.by_type("IfcMapConversion")
        crs = self.scene.settings.crs
        referenced = crs is not None and (crs.epsg is not None or bool(crs.wkt))
        if not referenced:
            if conversions:
                self.add("georeference", "", "IfcMapConversion present but the scene has no CRS")
            return
        if len(conversions) != 1:
            self.add("georeference", "", f"expected one IfcMapConversion, found {len(conversions)}")
            return
        conversion = conversions[0]
        offset = self.scene.settings.origin_offset
        actual = (conversion.Eastings, conversion.Northings, conversion.OrthogonalHeight)
        for axis, expected, value in zip("XYZ", offset, actual, strict=True):
            if not _close(expected, value):
                self.add(
                    "georeference",
                    "",
                    f"IfcMapConversion {axis} offset {value!r} differs from the origin offset "
                    f"{expected}",
                )

    def _counts(self) -> None:
        expected: Counter[ElementKind] = Counter(
            element.kind for element in self.scene.elements.values()
        )
        actual = _ifc_counts(self.file)
        for kind in _ELEMENT_CLASSES:
            if expected[kind] != actual[kind]:
                self.add(
                    "count",
                    "",
                    f"{kind.value}: scene has {expected[kind]}, IFC has {actual[kind]} "
                    f"({_ELEMENT_CLASSES[kind]})",
                )

    def _identity(self, element: Element, entity: Entity) -> None:
        for attribute in ("Description", "ObjectType"):
            value = getattr(entity, attribute, None)
            if not isinstance(value, str) or not value.strip():
                self.add(
                    "identity",
                    element.id,
                    f"{class_of(entity)} {element.name!r}: {attribute} is empty",
                )

    # ------------------------------------------------------------------ quantities

    def _rules_for(self, element: Element, entity: Entity) -> list[_QuantityRule]:
        match element.kind:
            case ElementKind.WALL:
                return _WALL_RULES
            case ElementKind.OPENING:
                return _OPENING_RULES
            case ElementKind.WINDOW:
                return _FILLING_RULES["window"]
            case ElementKind.DOOR:
                return _FILLING_RULES["door"]
            case ElementKind.SLAB:
                return _SLAB_RULES
            case ElementKind.CEILING:
                return _COVERING_RULES if is_a(entity, "IfcCovering") else _SLAB_RULES
            case ElementKind.SPACE:
                return _SPACE_RULES
            case ElementKind.STAIR:
                return _STAIR_RULES
            case _:
                return []

    def _quantities(self, element: Element, entity: Entity) -> None:
        expected = self.geometry.quantities_of(element.id)
        psets: dict[str, dict[str, Any]] = element_util.get_psets(entity)
        for rule in self._rules_for(element, entity):
            if rule.source not in expected:
                continue
            value = psets.get(rule.pset, {}).get(rule.target)
            if value is None:
                self.add(
                    "quantity",
                    element.id,
                    f"{class_of(entity)} {element.name!r}: {rule.pset}.{rule.target} is missing",
                )
            elif not _close(float(expected[rule.source]), value):
                self.add(
                    "quantity",
                    element.id,
                    f"{class_of(entity)} {element.name!r}: {rule.pset}.{rule.target} is {value!r}, "
                    f"the scene gives {float(expected[rule.source])!r}",
                )

    # -------------------------------------------------------------------- relations

    def _relations(self, element: Element, entity: Entity) -> None:
        label = f"{class_of(entity)} {element.name!r}"
        params: Any = element.params
        if element.kind == ElementKind.OPENING:
            host = element_util.get_voided_element(entity)
            if host is None or str(host.GlobalId) != _guid(params.host_wall_id):
                self.add("relation", element.id, f"{label} does not void its host wall")
        elif element.kind in (ElementKind.WINDOW, ElementKind.DOOR):
            opening = element_util.get_filled_void(entity)
            if opening is None or str(opening.GlobalId) != _guid(params.opening_id):
                self.add("relation", element.id, f"{label} does not fill its opening")
        if element.kind == ElementKind.STAIR:
            self._stair_parts(element, entity, label)
        if element.storey_id is None or element.kind == ElementKind.OPENING:
            return
        container = (
            element_util.get_aggregate(entity)
            if element.kind == ElementKind.SPACE
            else element_util.get_container(entity)
        )
        if container is None or str(container.GlobalId) != _guid(element.storey_id):
            self.add("relation", element.id, f"{label} is not placed in its storey")

    def _stair_parts(self, element: Element, entity: Entity, label: str) -> None:
        """Flights and landings of a stair are aggregated into the IfcStair."""
        geometry = self.geometry.stairs.get(element.id)
        if geometry is None:
            return
        parts = element_util.get_parts(entity)
        flights = [part for part in parts if is_a(part, "IfcStairFlight")]
        landings = [part for part in parts if is_a(part, "IfcSlab")]
        if len(flights) != len(geometry.flights):
            self.add(
                "relation",
                element.id,
                f"{label} aggregates {len(flights)} flights, the scene has {len(geometry.flights)}",
            )
        if len(landings) != len(geometry.landings):
            self.add(
                "relation",
                element.id,
                f"{label} aggregates {len(landings)} landings, "
                f"the scene has {len(geometry.landings)}",
            )
        risers = sorted(int(part.NumberOfRisers or 0) for part in flights)
        expected = sorted(flight.riser_count for flight in geometry.flights)
        if risers != expected:
            self.add("quantity", element.id, f"{label} flight risers {risers}, scene {expected}")


def reimport_check(
    file: ifcopenshell.file, scene: Scene, geometry: SceneGeometry | None = None
) -> list[ReimportMismatch]:
    """Compare a reopened IFC file with the scene (counts, dimensions, relations, types)."""
    return _Reimport(file, scene, geometry or compute_scene_geometry(scene)).run()


# ------------------------------------------------------------------------------ report


def _render_html(report: ValidationReport) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    def table(headers: list[str], rows: list[list[object]]) -> str:
        if not rows:
            return "<p>none</p>"
        head = "".join(f"<th>{esc(header)}</th>" for header in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>" for row in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    status = "OK" if report.ok else "FAILED"
    css = "ok" if report.ok else "bad"
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8"><title>IFC validation</title>',
        "<style>body{font-family:sans-serif;margin:2em}table{border-collapse:collapse}"
        "td,th{border:1px solid #999;padding:2px 8px;text-align:left}"
        ".ok{color:#070}.bad{color:#b00}</style></head><body>",
        f'<h1>IFC validation: <span class="{css}">{status}</span></h1>',
        f"<p>File: {esc(report.ifc_path)}<br>IDS: {esc(report.ids_path)}</p>",
        "<p>"
        f"Incomplete export: {esc(report.incomplete)}. "
        f"Unreferenced (no georeference): {esc(report.unreferenced)}. "
        f"Re-import compared with a scene: {esc(report.reimport_checked)}.</p>",
        f"<h2>EXPRESS rules ({len(report.express_findings)} findings)</h2>",
        table(
            ["level", "message", "instance", "attribute"],
            [
                [item["level"], item["message"], item["instance"], item["attribute"]]
                for item in report.express_findings
            ],
        ),
        "<h2>IDS specifications</h2>",
        table(
            ["specification", "passed", "applicable", "failed"],
            [
                [item["name"], item["status"], item["applicable"], item["applicable_failed"]]
                for item in report.ids_specifications
            ],
        ),
        f"<h2>IDS failures ({len(report.ids_failures)})</h2>",
        table(
            ["specification", "GlobalId", "Tag", "name", "requirement", "reason"],
            [
                [f.specification, f.global_id, f.tag, f.name, f.requirement, f.reason]
                for f in report.ids_failures
            ],
        ),
        f"<h2>Re-import mismatches ({len(report.reimport_mismatches)})</h2>",
        table(
            ["check", "element", "message"],
            [[m.check, m.element_id, m.message] for m in report.reimport_mismatches],
        ),
        "<h2>Warnings</h2>",
        table(["message"], [[warning] for warning in report.warnings]),
        "</body></html>",
    ]
    return "\n".join(parts)


def _write_reports(report: ValidationReport, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.validation.json"
    html_path = directory / f"{stem}.validation.html"
    report.json_path = str(json_path)
    report.html_path = str(html_path)
    json_path.write_text(report.to_json() + "\n", encoding="utf-8")
    html_path.write_text(report.to_html(), encoding="utf-8")


def validate_ifc(
    ifc_path: str | Path,
    ids_path: str | Path = IDS_PATH,
    scene: Scene | None = None,
    *,
    export_result: IfcExportResult | None = None,
    reports_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> ValidationReport:
    """Run the validation chain on an IFC file and write the HTML and JSON reports.

    ``scene`` enables the re-import test. ``export_result`` (from :func:`samcloud.bim.core.ifc.write_ifc`)
    contributes the incomplete and unreferenced flags and the writer warnings. Reports go to
    ``reports_dir`` (default: next to the IFC file) as ``<stem>.validation.html``,
    ``<stem>.validation.json`` and ``<stem>.ids.html``.
    """
    path = Path(ifc_path)
    ids_file = Path(ids_path)
    directory = Path(reports_dir) if reports_dir is not None else path.parent
    directory.mkdir(parents=True, exist_ok=True)

    def note(message: str) -> None:
        if progress is not None:
            progress(message)

    file = ifcopenshell.open(str(path))
    report = ValidationReport(ifc_path=str(path), ids_path=str(ids_file))
    note("express rules")
    report.express_findings = _express_findings(file)
    note("ids")
    failures, summaries, ids_html = _ids_check(file, ids_file, directory / f"{path.stem}.ids.html")
    report.ids_failures = failures
    report.ids_specifications = summaries
    report.ids_html_path = ids_html
    if scene is not None:
        note("re-import")
        report.reimport_checked = True
        report.reimport_mismatches = reimport_check(file, scene)
    if export_result is not None:
        report.warnings = list(export_result.warnings)
        report.unreferenced = not export_result.referenced
    report.incomplete = bool(
        (export_result is not None and export_result.incomplete) or report.ids_failures
    )
    _write_reports(report, directory, path.stem)
    return report


def summarize(report: ValidationReport) -> Mapping[str, int]:
    """Number of findings per check (for printing)."""
    return {
        "express": len(report.express_findings),
        "ids": len(report.ids_failures),
        "reimport": len(report.reimport_mismatches),
    }
