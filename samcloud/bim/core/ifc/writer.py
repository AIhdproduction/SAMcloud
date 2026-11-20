"""Entry point of the IFC export: :func:`write_ifc`."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

import ifcopenshell
import ifcopenshell.api.project

from samcloud.bim.core import BIM_CORE_VERSION as __version__
from samcloud.bim.core.geometry import compute_scene_geometry
from samcloud.bim.core.ifc.context import UNREFERENCED, ExportContext, IfcExportResult, class_of
from samcloud.bim.core.ifc.elements import (
    write_ceilings,
    write_fillings,
    write_joins,
    write_openings,
    write_slabs,
    write_walls,
)
from samcloud.bim.core.ifc.outdoor import write_outdoor
from samcloud.bim.core.ifc.georef import crs_label, write_georeferencing
from samcloud.bim.core.ifc.ids import finalize_guids, normalize_relation_order
from samcloud.bim.core.ifc.properties import write_properties
from samcloud.bim.core.ifc.scan import write_furniture, write_project_frame
from samcloud.bim.core.ifc.spaces import write_space_boundaries, write_spaces
from samcloud.bim.core.ifc.spatial import create_project, create_spatial_tree
from samcloud.bim.core.ifc.stairs import write_stairs
from samcloud.bim.core.model import ElementKind, Scene

SUPPORTED_KINDS = frozenset(
    {
        ElementKind.WALL,
        ElementKind.WALL_JOIN,
        ElementKind.SLAB,
        ElementKind.CEILING,
        ElementKind.OPENING,
        ElementKind.WINDOW,
        ElementKind.DOOR,
        ElementKind.SPACE,
        ElementKind.STAIR,
        ElementKind.FURNITURE,
        ElementKind.TERRAIN,
        ElementKind.BUILDING,
        ElementKind.TREE,
        ElementKind.SHRUB,
        ElementKind.SITE_WALL,
        ElementKind.FENCE,
        ElementKind.PAVEMENT,
        ElementKind.CURB,
        ElementKind.MARKING,
        ElementKind.MANHOLE,
        ElementKind.VEGETATION_AREA,
    }
)
"""Kinds the M1 writer exports; reserved kinds are skipped with a warning."""


def _flag_geometry_issues(ctx: ExportContext) -> None:
    """Flag every exported element that has geometry issues (wall joins report themselves)."""
    for element in ctx.scene.elements.values():
        if element.kind not in SUPPORTED_KINDS:
            ctx.flag(
                element.id,
                f"element kind {element.kind.value!r} is not exported by this writer version",
            )
            continue
        if element.kind == ElementKind.WALL_JOIN:
            continue
        for issue in ctx.geometry.issues.get(element.id, ()):
            if element.kind == ElementKind.SPACE and "ambiguous" in issue:
                continue
            ctx.flag(element.id, issue)


def _set_header(
    ctx: ExportContext, path: Path, timestamp: datetime | None, incomplete: bool
) -> None:
    header = ctx.file.header
    stamp = (timestamp or datetime.now().astimezone()).isoformat(timespec="seconds")
    notes = [f"SAMcloud {__version__} export"]
    notes.append(
        "georeference: " + (crs_label(ctx) or UNREFERENCED)
        if ctx.referenced
        else "georeference: unreferenced (no IfcMapConversion)"
    )
    if incomplete:
        notes.append("export incomplete: mandatory data missing, see Raumlot_Quality")
    header.file_description.description = tuple(notes)
    header.file_name.name = path.name
    header.file_name.time_stamp = stamp
    header.file_name.author = ("SAMcloud",)
    header.file_name.organization = ("",)
    header.file_name.originating_system = f"SAMcloud {__version__}"
    header.file_name.authorization = ""


def write_ifc(
    scene: Scene,
    path: str | Path,
    *,
    run_id: str | None = None,
    ai_backend: str | None = None,
    ai_model: str | None = None,
    timestamp: datetime | None = None,
) -> IfcExportResult:
    """Write the scene as an IFC4 file that ArchiCAD imports as native elements.

    Geometry and quantities come from :func:`samcloud.bim.core.geometry.compute_scene_geometry`. The same
    scene gives the same file content except for the header time stamp, which can be fixed with
    ``timestamp``. Missing mandatory data does not stop the export: the element is flagged
    (``Raumlot_Quality.Incomplete``), listed in the result and the export is marked incomplete.
    """
    target = Path(path)
    geometry = compute_scene_geometry(scene)
    file = ifcopenshell.api.project.create_file(version="IFC4")
    ctx = ExportContext(
        scene=scene,
        geometry=geometry,
        file=file,
        run_id=run_id,
        ai_backend=ai_backend,
        ai_model=ai_model,
    )
    for problem in scene.check_references():
        if " building_id: unknown id " in problem:
            building_id = problem.rsplit("'", 2)[1]
            if scene.elements.get(building_id, None) is not None and scene.elements[building_id].kind == ElementKind.BUILDING:
                continue
        ctx.warn(f"scene reference: {problem}")

    create_project(ctx)
    write_project_frame(ctx)
    write_georeferencing(ctx)
    create_spatial_tree(ctx)
    _flag_geometry_issues(ctx)
    write_walls(ctx)
    write_joins(ctx)
    write_slabs(ctx)
    write_ceilings(ctx)
    write_stairs(ctx)
    write_openings(ctx)
    write_fillings(ctx)
    write_outdoor(ctx)
    write_spaces(ctx)
    write_space_boundaries(ctx)
    write_properties(ctx)
    write_furniture(ctx)
    normalize_relation_order(ctx)
    finalize_guids(ctx)

    incomplete = bool(ctx.flagged)
    _set_header(ctx, target, timestamp, incomplete)
    target.parent.mkdir(parents=True, exist_ok=True)
    file.write(str(target))

    counts = Counter(class_of(entity) for entity in file.by_type("IfcRoot"))
    return IfcExportResult(
        path=str(target),
        counts=dict(sorted(counts.items())),
        warnings=list(ctx.warnings),
        referenced=ctx.referenced,
        incomplete=incomplete,
        flagged_elements={key: list(value) for key, value in ctx.flagged.items()},
    )
