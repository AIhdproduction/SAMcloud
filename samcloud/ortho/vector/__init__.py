"""Pure raster-to-CAD vector geometry public API."""

from .areas import AreaParams, build_partition, clean_label_raster, partition_entities
from .indoor import Opening, centred_catalogue_insert, match_catalogue, snap_insert_to_wall, cut_openings, detect_openings, fit_block_insert, furniture_entities, guess_fixture, opening_entities, stair_entities, wall_polygons_from_section
from .lines import boundary_lines, trace_lines
from .raster_grid import RasterGrid, mask_to_polygons
from .regularize import dominant_direction, regularize_polygon, regularize_polygons
from .symbols import CATALOGUE, SYMBOLS, SymbolDef, add_symbol_blocks, symbol_extents
from .trees import Crown, detect_crowns, tree_entities

__all__ = ["CATALOGUE", "centred_catalogue_insert", "match_catalogue", "snap_insert_to_wall", "symbol_extents", "AreaParams", "Crown", "Opening", "RasterGrid", "SYMBOLS", "SymbolDef", "add_symbol_blocks", "boundary_lines", "build_partition", "clean_label_raster", "cut_openings", "detect_crowns", "detect_openings", "dominant_direction", "fit_block_insert", "furniture_entities", "guess_fixture", "mask_to_polygons", "opening_entities", "partition_entities", "regularize_polygon", "regularize_polygons", "stair_entities", "trace_lines", "tree_entities", "wall_polygons_from_section"]
