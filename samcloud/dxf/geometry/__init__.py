"""Pure geometry operations used by the GUI-free DXF pipeline."""

from samcloud.dxf.geometry.features import circles_from_clusters, furniture_inserts, points_from_cluster, skeleton_lines, tree_entities
from samcloud.dxf.geometry.raster import Raster, polygons_from_points, rasterize_points
from samcloud.dxf.geometry.walls import bridge_collinear, opening_entities, opening_specs, wall_lines, wall_polygons

__all__ = [
    "Raster",
    "circles_from_clusters",
    "furniture_inserts",
    "opening_entities",
    "points_from_cluster",
    "polygons_from_points",
    "rasterize_points",
    "skeleton_lines",
    "tree_entities",
    "wall_lines",
    "bridge_collinear",
    "wall_polygons",
    "opening_specs",
]
