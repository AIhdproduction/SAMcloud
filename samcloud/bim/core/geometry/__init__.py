"""Single source of geometry and quantities, built from element parameters.

Pure, deterministic functions in float64 without IFC imports. See docs/GEOMETRY.md.
"""

from __future__ import annotations

from samcloud.bim.core.geometry.openings import (
    CORNER_ORDER,
    FillingGeometry,
    OpeningGeometry,
    apply_openings,
    compute_filling_geometry,
    compute_opening_geometry,
)
from samcloud.bim.core.geometry.scene_geometry import SceneGeometry, compute_scene_geometry
from samcloud.bim.core.geometry.slabs import (
    CeilingGeometry,
    SlabGeometry,
    compute_ceiling_geometry,
    compute_slab_geometry,
)
from samcloud.bim.core.geometry.spaces import (
    RoomPolygon,
    SpaceGeometry,
    compute_space_geometries,
    derive_storey_rooms,
)
from samcloud.bim.core.geometry.stairs import (
    FlightGeometry,
    LandingGeometry,
    StairGeometry,
    StepGeometry,
    compute_stair_geometry,
)
from samcloud.bim.core.geometry.walls import (
    OpeningCut,
    WallGeometry,
    compute_wall_geometries,
    face_offsets,
)

__all__ = [
    "CORNER_ORDER",
    "CeilingGeometry",
    "FillingGeometry",
    "FlightGeometry",
    "LandingGeometry",
    "OpeningCut",
    "OpeningGeometry",
    "RoomPolygon",
    "SceneGeometry",
    "SlabGeometry",
    "SpaceGeometry",
    "StairGeometry",
    "StepGeometry",
    "WallGeometry",
    "apply_openings",
    "compute_ceiling_geometry",
    "compute_filling_geometry",
    "compute_opening_geometry",
    "compute_scene_geometry",
    "compute_slab_geometry",
    "compute_space_geometries",
    "compute_stair_geometry",
    "compute_wall_geometries",
    "derive_storey_rooms",
    "face_offsets",
]
