"""Georeferencing: IfcProjectedCRS and IfcMapConversion from the stored origin offset."""

from __future__ import annotations

import ifcopenshell
import ifcopenshell.api.georeference

from samcloud.bim.core.ifc.context import ExportContext
from samcloud.bim.core.model import CrsInfo, ElementKind


def crs_label(ctx: ExportContext) -> str | None:
    """EPSG code or name of the CRS when the export is referenced, else None."""
    crs = ctx.georef_crs or ctx.scene.settings.crs
    if crs is None:
        return None
    if crs.epsg is not None:
        return f"EPSG:{crs.epsg}"
    if crs.wkt:
        return crs.name or "WKT"
    return None


def write_georeferencing(ctx: ExportContext) -> None:
    """Write IfcProjectedCRS and IfcMapConversion when a CRS (EPSG or WKT) is set.

    The origin offset is the map position of the local origin: Eastings, Northings and
    OrthogonalHeight. Rotation 0, scale 1, metre CRS unit. The geometric contexts must exist
    before this runs (otherwise ``add_georeferencing`` creates nothing).
    """
    crs = ctx.scene.settings.crs
    if crs is None and any(
        element.kind in {
            ElementKind.TERRAIN, ElementKind.BUILDING, ElementKind.TREE, ElementKind.SHRUB,
            ElementKind.SITE_WALL, ElementKind.FENCE, ElementKind.PAVEMENT, ElementKind.CURB,
            ElementKind.MARKING, ElementKind.MANHOLE, ElementKind.VEGETATION_AREA,
        }
        for element in ctx.scene.elements.values()
    ):
        crs = CrsInfo(epsg=2056, vertical_datum="LN02", name="CH1903+ / LV95", source="none")
    ctx.georef_crs = crs
    label = crs_label(ctx)
    if crs is None or label is None:
        ctx.referenced = False
        if crs is not None:
            ctx.warn("crs is set but has neither an EPSG code nor WKT: exported as unreferenced")
        return
    f = ctx.file
    ifcopenshell.api.georeference.add_georeferencing(f, ifc_class="IfcMapConversion", name=label)
    length_unit = next(u for u in f.by_type("IfcSIUnit") if u.UnitType == "LENGTHUNIT")
    projected: dict[str, object] = {"MapUnit": length_unit}
    if crs.wkt:
        projected["Description"] = crs.wkt
    if crs.vertical_datum:
        projected["VerticalDatum"] = crs.vertical_datum
    offset = ctx.scene.settings.origin_offset
    ifcopenshell.api.georeference.edit_georeferencing(
        f,
        projected_crs=projected,
        coordinate_operation={
            "Eastings": float(offset[0]),
            "Northings": float(offset[1]),
            "OrthogonalHeight": float(offset[2]),
            "XAxisAbscissa": 1.0,
            "XAxisOrdinate": 0.0,
            "Scale": 1.0,
        },
    )
    ctx.referenced = True
