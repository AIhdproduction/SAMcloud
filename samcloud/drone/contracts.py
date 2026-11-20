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

TIN_LABELS: dict[str, int] = {"terrain": 0, "road": 1, "sidewalk": 2, "path": 3, "lawn": 4}
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
    pavement_id: int
    kind: Literal["road", "sidewalk", "path"]
    polygon: Polygon              # 2D local, valid
    evidence: Evidence = field(default_factory=Evidence)
    status: Status = "ok"
    reason: str = ""


@dataclass
class CurbAxis:
    curb_id: int
    points: np.ndarray            # (M,3) local, DP-thinned
    evidence: Evidence = field(default_factory=Evidence)
    status: Status = "ok"
    reason: str = ""


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
        "tin": {"constraint_buffer_m": 0.05, "min_angle_deg": 20, "max_steiner_per_edge": 50},
    },
    "buildings": {
        "ransac_distance_m": 0.05,
        "region_growing": {"k": 16, "epsilon_m": 0.08, "min_points": 200, "normal_angle_deg": 10},
        "label_pruning_buffer_m": 2.5,
        "complexity_factor": 0.7,
        "max_seconds_per_building": 60,
    },
    "openings": {
        "min_views": 3, "min_area_m2": 0.3, "max_area_m2": 12.0,
        "door_bottom_tolerance_m": 0.20, "window_min_sill_m": 0.40, "reference_strip_m": 0.5,
    },
    "vegetation": {
        "tree_min_height_m": 2.0, "tree_min_points": 200,
        "crown_triangles": {"min": 60, "max": 120},
        "default_trunk_diameter_m": 0.30,
    },
    "pavement": {
        "min_thickness_m": 0.10, "snap_tolerance_m": 0.01, "max_gap_overlap_cm2": 1.0,
        "snap_search_m": 0.15, "gap_tolerance_m": 0.15, "min_area_m2": 2.0,
    },
    "curb": {
        "min_length_m": 0.3, "dp_tolerance_m": 0.015, "section_width_m": 0.12, "section_height_m": 0.25,
        "arc_fit_max_residual_m": 0.01, "arc_max_radius_m": 50.0, "arc_min_points": 4,
    },
    "dxf": {
        "local_origin": False, "hatch": False, "contour_major_m": 1.0,
        "contour_minor_m": "auto", "spline_contours": True, "tin_layer_frozen": True,
        "spot_spacing_m": 10.0,
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
