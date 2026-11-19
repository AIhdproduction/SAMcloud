"""IFC4 export of the parametric scene (docs/IFC_MAPPING.md, MASTERPROMPT section 10)."""

from __future__ import annotations

from samcloud.bim.core.ifc.context import IfcExportResult
from samcloud.bim.core.ifc.writer import write_ifc

__all__ = ["IfcExportResult", "write_ifc"]
