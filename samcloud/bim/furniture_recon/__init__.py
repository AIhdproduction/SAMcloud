"""Realistic 3D furniture from the scan: per-item points -> class-specific parametric parts or a compact mesh.

Public entry points:

* :func:`reconstruct_items` - reconstruct ``PlanFurniture`` items from a :class:`SceneCloud` (in place).
* :func:`reconstruct_from_project_files` - the plan_model hook (loads ``classified.ply`` beside ``instances.json``).

Results are written to the optional ``PlanFurniture`` fields ``solids`` (vertical prisms: box / cylinder / extrusion in
plan-local units), ``mesh`` (closed triangle mesh, plan-local units), ``shape_source`` (parametric | mesh | footprint)
and ``fit`` (point counts, point-to-surface distances, coverage, volume ratio, part sources).
"""

from samcloud.bim.furniture_recon.cloud import SceneCloud, load_scene_cloud
from samcloud.bim.furniture_recon.pipeline import reconstruct_from_project_files, reconstruct_items
from samcloud.bim.furniture_recon.settings import ReconSettings, load_recon_settings

__all__ = [
    "ReconSettings", "SceneCloud", "load_recon_settings", "load_scene_cloud", "reconstruct_from_project_files", "reconstruct_items",
]
