"""Shared data contracts and config loader for the drone outdoor pipeline.

All geometry is in the project's local metric frame (metres, Z up).
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from shapely.geometry import LineString, Polygon

from samcloud.core.paths import CONFIG_DIRECTORY

logger = logging.getLogger(__name__)

# LAS class codes used by the outdoor chain (match config/classes.json + cyclone_codes.yaml)
GROUND = 2
LOW_VEGETATION = 3
MEDIUM_VEGETATION = 4
TREE = 5
BUILDING = 6
ROAD = 11
SEALED_TERRAIN = 71
SITE_WALL = 72
SIDEWALK = 73
CURB = 74
MANHOLE = 75
FENCE = 77
ROAD_MARKING = 82
PATH = 83
SHRUB = 94
TREE_TRUNK = 95
DOOR = 103
WINDOW = 104
ROOF_INCLINED = 144
ROOF_FLAT = 145

TIN_LABELS: dict[str, int] = {"terrain": 0, "road": 1, "sidewalk": 2, "path": 3, "lawn": 4, "step": 5}  # step: 1 cm vertical faces of terrain/curb steps (samcloud/terrain/steps.py)
PAVEMENT_KINDS = ("road", "sidewalk", "path")
Status = Literal["ok", "warning", "failed", "rejected"]


@dataclass(frozen=True)
class LocalFrame:
    """Local metric frame -> LV95 (translation only)."""

    origin_lv95: tuple[float, float, float]  # E, N, H(LN02) of local (0,0,0)
    epsg: int = 2056
    vertical_datum: str = "LN02"

    def to_lv95(self, xyz: np.ndarray) -> np.ndarray:
        """Local (...,3) -> LV95 E, N, H."""
        return np.asarray(xyz, dtype=np.float64) + np.asarray(self.origin_lv95, dtype=np.float64)

    def from_lv95(self, enh: np.ndarray) -> np.ndarray:
        """LV95 E, N, H (...,3) -> local."""
        return np.asarray(enh, dtype=np.float64) - np.asarray(self.origin_lv95, dtype=np.float64)


@dataclass
class Evidence:
    views: int = 0
    points: int = 0
    rmse: float | None = None
    notes: dict = field(default_factory=dict)


_POINTSET_FIELDS = ("xyz", "classes", "confidence", "margin", "view_count", "normals", "rgb", "adjusted")


@dataclass
class PointSet:
    """One classified cloud in local coords."""

    xyz: np.ndarray                       # (N,3) float64
    classes: np.ndarray                   # (N,) uint16 LAS codes
    confidence: np.ndarray | None = None  # (N,) float32 top1_conf
    margin: np.ndarray | None = None      # (N,) float32 top1-top2
    view_count: np.ndarray | None = None  # (N,) uint16
    normals: np.ndarray | None = None     # (N,3)
    rgb: np.ndarray | None = None         # (N,3) uint8
    adjusted: np.ndarray | None = None    # (N,) bool, point moved by a filter

    def subset(self, mask_or_idx) -> "PointSet":
        """Return a new PointSet with the rows selected by a bool mask or index array."""
        sel = np.asarray(mask_or_idx)
        kwargs = {}
        for name in _POINTSET_FIELDS:
            arr = getattr(self, name)
            kwargs[name] = None if arr is None else arr[sel]
        return PointSet(**kwargs)

    def __len__(self) -> int:
        return int(len(self.xyz))


@dataclass
class Plane:
    normal: np.ndarray       # (3,) unit
    d: float                 # n.x + d = 0
    inliers: np.ndarray      # point indices into the PointSet it was fitted on
    rmse: float

    def distance(self, xyz: np.ndarray) -> np.ndarray:
        """Signed distance of points (...,3) to the plane."""
        return np.asarray(xyz, dtype=np.float64) @ np.asarray(self.normal, dtype=np.float64) + self.d


@dataclass
class FacadePlane(Plane):
    building_id: int = -1
    segment: LineString | None = None   # 2D footprint trace (local XY)
    z_min: float = 0.0
    z_max: float = 0.0


@dataclass
class RoofPlane(Plane):
    building_id: int = -1
    outline: Polygon | None = None      # 2D projection of the plane's support
    slope_deg: float = 0.0


@dataclass
class BuildingInstance:
    building_id: int
    point_indices: np.ndarray
    footprint: Polygon | None = None
    facades: list[FacadePlane] = field(default_factory=list)
    roof_planes: list[RoofPlane] = field(default_factory=list)
    status: Status = "ok"
    reason: str = ""


@dataclass
class BuildingModel:
    building_id: int
    lod: Literal["2.2", "1.3", "1.2"]
    vertices: np.ndarray          # (V,3) local
    faces: np.ndarray             # (F,3) int, outward normals
    face_roles: np.ndarray        # (F,) uint8: 0 ground, 1 wall, 2 roof
    status: Literal["ok", "lod1_fallback", "failed"]
    reason: str = ""
    rmse: float | None = None
    coverage: float | None = None
    heights: dict = field(default_factory=dict)   # eave, ridge, ground, etc.


@dataclass
class Opening:
    building_id: int
    facade_index: int
    kind: Literal["window", "door"]
    u0: float                     # along facade segment (m from segment start)
    u1: float
    z0: float                     # absolute heights (local Z)
    z1: float
    evidence: Evidence = field(default_factory=Evidence)
    status: Status = "ok"
    reason: str = ""


@dataclass
class PavementPolygon:
    pavement_id: int | str
    kind: Literal["road", "sidewalk", "path"]
    polygon: Polygon              # 2D local, valid
    evidence: Evidence = field(default_factory=Evidence)
    status: Status = "ok"
    reason: str = ""


@dataclass
class CurbAxis:
    curb_id: int
    points: np.ndarray            # (M,3) local, DP-thinned; z = curb top
    evidence: Evidence = field(default_factory=Evidence)
    status: Status = "ok"
    reason: str = ""
    foot_z: np.ndarray | None = None   # (M,) road-side surface height per vertex (None = not analysed)
    road_side: int = 0                 # +1 road on the left of the axis direction, -1 right, 0 unknown
    gutter_width: float = 0.0          # gutter stone width (m); 0 = none
    gutter_measured: bool = False      # True when the width was measured, False when assumed
    flush: bool = False                # mostly flush with the surrounding surface
    flush_stations: int = 0            # stations lowered to the surface


@dataclass
class TreeInstance:
    tree_id: int
    trunk_xy: tuple[float, float]
    base_z: float
    height_m: float | None        # None = not measured, never guessed
    crown_diameter_m: float | None
    trunk_diameter_m: float | None
    point_indices: np.ndarray
    evidence: Evidence = field(default_factory=Evidence)
    status: Status = "ok"
    reason: str = ""
    # Crown shape fit (defaults keep old pickles / positional constructions valid).
    crown_shape: str = "ellipsoid"          # sphere|ellipsoid|cone|umbrella|cylinder
    crown_shape_status: str = "assumed"     # "measured" (fitted) | "assumed" (too few crown points)
    crown_base_z: float | None = None       # local Z of the crown base = trunk top
    crown_top_z: float | None = None
    crown_offset_xy: tuple[float, float] = (0.0, 0.0)   # crown centre minus trunk position
    trunk_measured: bool = False            # True only for a circle fit that passed all trunk gates
    crown_fit_residual: float | None = None  # normalised profile residual of the chosen shape


@dataclass
class ShrubInstance:
    shrub_id: int
    outline: Polygon
    height_m: float | None
    evidence: Evidence = field(default_factory=Evidence)
    status: Status = "ok"
    reason: str = ""


@dataclass
class LabeledTin:
    vertices: np.ndarray          # (V,3) local
    triangles: np.ndarray         # (T,3) int
    labels: np.ndarray            # (T,) uint8, values from TIN_LABELS

    def _filter(self, keep: np.ndarray) -> "LabeledTin":
        tris = np.asarray(self.triangles)[keep]
        used, inverse = np.unique(tris.ravel(), return_inverse=True)
        return LabeledTin(
            vertices=np.asarray(self.vertices)[used],
            triangles=inverse.reshape(tris.shape).astype(np.asarray(self.triangles).dtype, copy=False),
            labels=np.asarray(self.labels)[keep],
        )

    @staticmethod
    def _codes(names: tuple[str, ...]) -> list[int]:
        unknown = [n for n in names if n not in TIN_LABELS]
        if unknown:
            raise KeyError(f"unknown TIN label(s): {unknown}")
        return [TIN_LABELS[n] for n in names]

    def select(self, *label_names: str) -> "LabeledTin":
        """Keep triangles with these labels (vertices reindexed)."""
        return self._filter(np.isin(self.labels, self._codes(label_names)))

    def without(self, *label_names: str) -> "LabeledTin":
        """Keep triangles whose label is not in label_names (vertices reindexed)."""
        return self._filter(~np.isin(self.labels, self._codes(label_names)))


@dataclass
class QAEntry:
    category: str                 # e.g. "opening", "tree", "building"
    element_id: str
    reason: str
    values: dict = field(default_factory=dict)


_DEFAULT_CONFIG: dict = {
    "class_codes": {"road_marking": 82, "path": 83, "custom_slots_start": 200},
    "boundary": {"mode": "camera_hull", "buffer_m": 10.0, "perimeter_path": None, "perimeter_buffer_m": 2.0},
    "terrain": {
        "dsm_resolution_m": 0.10,
        "dtm_resolution_m": 0.25,
        "ground_filter": "auto",
        "csf": {"cloth_resolution_m": 0.5, "rigidness": 2, "class_threshold_m": 0.15,
                "iterations": 500, "time_step": 0.65},
        "clean": {"enabled": True, "window_m": 5.0, "max_bump_m": 0.15, "max_bump_area_m2": 25.0, "tall_bump_m": 1.0, "tall_area_factor": 4.0,
                  "grow_fraction": 0.25, "grow_margin_m": 1.0, "steep_slope": 1.0, "wall_min_length_m": 2.0,
                  "protect_dilate_m": 0.5, "median_cells": 3},
        "smooth": {"enabled": True, "window_m": 2.0, "iterations": 3, "robust_scale_m": 0.08, "max_change_m": 0.06},
        "deviation": {"max_rms_pavement_m": 0.04, "max_rms_terrain_m": 0.06, "max_points": 400000},
        "tin": {
            "constraint_buffer_m": 0.05, "min_angle_deg": 20, "max_steiner_per_edge": 50,
            "tolerance_m": 0.03, "tolerance_pavement_m": 0.02, "tolerance_terrain_m": 0.04, "initial_spacing_m": 2.0, "support_spacing_m": 0.50, "max_vertices": 60000,
            "max_refinement_rounds": 2,
        },
        "lawn_ground_percentile": 20.0, "lawn_ground_min_points": 4, "lawn_ground_max_lowering_m": 0.25,
        "manhole_ring_segments": 16, "manhole_below_m": 0.001,
        "step": {"enabled": True, "min_slope_deg": 60.0, "min_length_m": 1.0, "probe_m": 0.15, "min_jump_m": 0.04,
                 "offset_m": 0.01, "min_run_m": 0.5, "sample_m": 0.25, "dtm_features": True, "wall_min_jump_m": 0.25},
        "flush": {"enabled": True, "blend_m": 1.0, "probe_m": 0.15, "sample_m": 0.25, "step_clear_m": 0.05, "qa_tol_m": 0.01},
    },
    "buildings": {
        "ransac_distance_m": 0.05,
        "region_growing": {"k": 16, "epsilon_m": 0.08, "min_points": 200, "normal_angle_deg": 10},
        "label_pruning_buffer_m": 2.5,
        "complexity_factor": 0.7,
        "max_seconds_per_building": 60,
        "roof": {"min_plane_area_m2": 4.0, "inlier_tolerance_m": 0.08, "min_roof_height_m": 1.5, "region_simplify_m": 0.35, "step_simplify_m": 0.12, "outline_simplify_m": 0.3, "min_region_area_m2": 1.0,
                 "min_feature_area_m2": 6.0, "min_step_m": 0.35, "merge_normal_deg": 4.0, "merge_offset_m": 0.15},
        "dxf": {"coplanar_deg": 2.0},
        "base": {"mode": "terrain", "embed_m": 0.02, "max_edge_m": 2.0, "follow_tol_m": 0.03},
        "plausibility": {"min_width_m": 2.5, "min_area_m2": 10.0, "min_height_m": 2.5, "min_plane_area_m2": 4.0},
        "regularize": {
            "angle_tol_deg": 12, "merge_offset_m": 0.3, "min_edge_m": 0.5,
            "ridge_tol_deg": 5, "flat_slope_deg": 5, "eave_tol_m": 0.15,
            "min_footprint_iou": 0.92, "max_rmse_increase_m": 0.03,
        },
    },
    "openings": {
        "min_views": 3, "min_area_m2": 0.3, "max_area_m2": 12.0,
        "door_bottom_tolerance_m": 0.20, "window_min_sill_m": 0.40, "reference_strip_m": 0.5,
    },
    "vegetation": {
        "tree_min_height_m": 2.0, "tree_min_points": 200,
        "crown_triangles": {"min": 60, "max": 120},
        "default_trunk_diameter_m": 0.30,
        "crown": {"min_crown_points": 150, "slices": 12, "min_slice_points": 15, "base_radius_fraction": 0.2,
                  "sphere_aspect_tol": 0.15, "margin": 0.15, "cone_rho": -0.6, "cylinder_aspect": 1.8,
                  "cylinder_cv": 0.15, "envelope_percentile": 90.0, "umbrella_aspect": 0.6},
        "trunk": {"max_rms_m": 0.03, "min_points": 20, "min_diameter_m": 0.05, "max_diameter_m": 2.0,
                  "band_m": [0.5, 1.6], "search_radius_m": 1.0},
    },
    "pavement": {
        "min_thickness_m": 0.10, "snap_tolerance_m": 0.01, "max_gap_overlap_cm2": 1.0,
        "snap_search_m": 0.15, "gap_tolerance_m": 0.15, "min_area_m2": 2.0,
        "outline": {"close_m": 0.25, "open_m": 0.10, "min_area_m2": 2.0,
                    "min_hole_m2": 1.0, "simplify_m": 0.05},
    },
    "surfaces": {"clean": {"close_m": 0.4, "open_m": 0.2,
                           "min_area_m2": {"lawn": 4.0, "sidewalk": 2.0, "path": 2.0},
                           "min_hole_m2": 2.0, "simplify_m": 0.10, "min_width_m": 0.6}},
    "edges": {"enabled": True,
              "fit": {"min_radius_m": 1.0, "max_radius_m": 200.0, "tolerance_m": 0.05, "sample_m": 0.10,
                      "tangent_deg": 10.0, "merge_deg": 3.0, "sagitta_m": 0.01, "curb_match_m": 0.12},
              "dxf_curb_flat_z_m": 0.05,
              "image": {"enabled": True, "cell_m": 0.03, "z_percentile": 0.7, "hole_fill_cells": 3, "search_m": 0.30,
                        "step_m": 0.02, "window_m": 0.06, "along_m": 0.6, "sigma_m": 0.03, "weight_color": 1.0,
                        "weight_height": 1.0, "min_contrast": 5.0, "prior_per_m": 1.0, "smooth_cost": 1.5, "max_step": 3,
                        "passes": 2, "pass2_search_m": 0.10, "fit_tolerance_m": 0.03, "straight": True, "min_segment_m": 1.5, "inlier_m": 0.04, "merge_deg": 4.0, "snap_deg": 6.0,
                        "min_arc_radius_m": 2.0, "ransac_trials": 150, "hough": True, "hough_parallel_deg": 8.0,
                        "hough_min_len_m": 1.0, "min_support": 0.60, "agree_m": 0.04,
                        "max_area_change": 0.05, "min_edge_m": 0.5, "zigzag_dev_m": 0.05, "node_shift_m": 0.45,
                        "min_chain_m": 1.0, "side_offset_m": 0.15, "merge_gap_samples": 8},
              "clean": {"enabled": True, "spike_m": 0.25, "spike_protrude_m": 0.10, "line_tol_m": 0.08, "min_segment_m": 1.5,
                        "merge_gap_samples": 40, "max_shift_m": 0.25, "max_area_change": 0.05}},
    "curb": {
        "min_length_m": 0.3, "dp_tolerance_m": 0.015, "section_width_m": 0.12, "section_height_m": 0.25,
        "arc_fit_max_residual_m": 0.01, "arc_max_radius_m": 50.0, "arc_min_points": 4,
        "dedupe_offset_m": 0.5, "dedupe_angle_deg": 10.0, "dedupe_overlap": 0.5,
        "gutter_width_m": 0.30, "gutter_thickness_m": 0.12, "gutter_min_band_m": 0.15, "gutter_z_tol_m": 0.03,
        "embed_m": 0.15, "flush_max_m": 0.03, "flush_min_run": 3, "default_reveal_m": 0.12, "max_reveal_m": 0.35,
        "section_halfwidth_m": 0.6, "section_step_m": 0.25, "dxf_flat_tol_m": 0.003, "surface_probe_m": 0.06, "gutter_station_m": 0.25, "gutter_flat_tol_m": 0.01, "gutter_max_cross_m": 0.05, "gutter_min_run_m": 0.5, "gutter_pavement_reach_m": 0.3,
    },
    "site": {
        "fence": {"eps_m": 0.3, "step_m": 0.25, "min_points": 4, "min_station_points": 2, "branch_offset_m": 0.3,
                  "top_percentile": 95.0, "smooth_window": 5, "dp_m": 0.02, "thickness_m": 0.04, "min_height_m": 0.2, "min_length_m": 0.5, "bottom_percentile": 3.0, "bottom_smooth_window": 9,
                  "bottom_dp_m": 0.03, "ground_snap_m": 0.15, "terrain_radius_m": 0.5, "merge_gap_m": 1.0,
                  "merge_angle_deg": 15.0, "merge_height_diff_m": 0.3},
        "wall": {"eps_m": 0.75, "step_m": 0.4, "min_points": 4, "min_station_points": 3, "branch_offset_m": 0.8,
                 "top_percentile": 95.0, "smooth_window": 5, "dp_m": 0.02, "min_thickness_m": 0.1,
                 "top_variation_m": 0.10, "terrain_radius_m": 0.5},
    },
    "manhole": {
        "thickness_m": 0.05,
        "standard_diameters_m": [0.27, 0.42, 0.62, 0.745, 0.785, 0.85, 0.92, 1.05],
        "standard_labels": ["Schieberkappe", "Hydrantenkappe", "Deckel LW600-625", "Rahmen LW625 / Trottoirdeckel",
                            "Rahmen LW600 (Typ Viatop)", "Rahmen LW600 (Typ Pamrex)", "Rahmen LW800", "Betonuntersatz"],
        "snap_tol_m": 0.08, "snap_tol_small_m": 0.04,
    },
    "dxf": {
        "local_origin": False, "hatch": False, "contour_major_m": 1.0,
        "contour_minor_m": "auto", "spline_contours": True, "tin_layer_frozen": True,
        "spot_spacing_m": 10.0, "spot_heights": False, "contour_label_spacing_m": 10.0, "contour_label_height_m": 0.25, "contour_building_gap_m": 0.10,
        "block_text_height_m": 0.2,
        "fence_bottom_line": False,
    },
    "ifc": {"curb_sweep": "auto", "epsg": 2056, "vertical_datum": "LN02"},
    "cloud_filter": {
        "tile_size_m": 10.0, "halo_m": 1.0,
        "outliers": {"nb_neighbors": 20, "std_ratio": 2.0, "radius": None, "min_points": None},
        "planar": {"class_codes": [6, 144, 145], "sigma_factor": 1.0, "curvature_keep": 0.05, "radius": 0.3,
                   "min_inlier_fraction": 0.95, "max_rms_m": 0.015},
        "ground": {"class_codes": [2, 11, 71, 73, 83, 82], "radius": 0.5, "normal_jump_deg": 12.0,
                   "curb_clearance_m": 0.15, "curb_code": 74, "sigma_factor": 1.0,
                   "min_inlier_fraction": 0.95, "max_rms_m": 0.015},
    },
    "environment": {"working_voxel_m": "auto", "max_working_points": 8000000, "min_working_voxel_m": 0.03,
                    "clip_point_cloud": True},
    "semantics": {"plane_k": 24, "plane_chunk_size": 500000},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_drone_config(path=None) -> dict:
    """Load drone config; the file is deep-merged over the built-in defaults."""
    cfg_path = Path(path) if path is not None else Path(CONFIG_DIRECTORY) / "drone_outdoor.yaml"
    if not cfg_path.exists():
        logger.warning("drone config %s not found, using built-in defaults", cfg_path)
        return copy.deepcopy(_DEFAULT_CONFIG)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return _deep_merge(_DEFAULT_CONFIG, data)
