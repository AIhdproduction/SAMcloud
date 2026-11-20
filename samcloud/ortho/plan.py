"""Clean DXF plan generation from true orthophoto, segmentation, and elevation rasters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Literal

import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
import shapely
from shapely import affinity, coverage_is_valid, line_merge, make_valid, unary_union
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box

from samcloud.classification.profiles import load_cyclone_codes
from samcloud.bim.plan_model import build_plan_model, save_plan_model
from samcloud.core.configuration import load_yaml
from samcloud.core.paths import CONFIG_DIRECTORY
from samcloud.dxf.cancel import CancelToken, Cancelled, ProgressCallback, check_cancel, report_progress
from samcloud.dxf.las_source import Frame, iter_source_chunks
from samcloud.dxf.layers import LayerConfig, LayerDefinition, load_layer_config
from samcloud.dxf.levels import Level
from samcloud.dxf.pipeline import dxf_entity_count, save_cached_result
from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus
from samcloud.ortho.build import OrthoError, load_ortho_result, ortho_output_dir
from samcloud.ortho.segment import load_segment_result
from samcloud.ortho.vector.areas import AreaParams, build_partition, clean_label_raster, partition_entities
from samcloud.ortho.vector.contours import (
    GroundDtmAccumulator,
    ContourLabelSettings,
    contour_entities,
    fill_dtm,
    ground_codes_from_config,
    label_major_contours,
    load_contour_settings,
)
from samcloud.ortho.vector.indoor import (
    Opening,
    cut_openings,
    detect_openings,
    fit_block_insert,
    centred_catalogue_insert,
    furniture_entities,
    guess_fixture,
    opening_entities,
    snap_insert_to_wall,
    stair_entities,
)
from samcloud.ortho.vector.symbols import CATALOGUE, catalogue_entry_for_block
from samcloud.ortho.vector.buildings import RoofSettings, roof_height_entities, roof_heights
from samcloud.ortho.vector.spots import SpotSettings, ground_grid_points, points_along, select_spots, spot_entities
from samcloud.ortho.vector.terrain import DtmSampler, drape_coordinates, drape_outline_heights
from samcloud.ortho.vector.vegetation import HedgeSettings, hedge_entities
from samcloud.ortho.vector.kitchen_run import KitchenRunSettings, appliance_insert, build_kitchen_run
from samcloud.ortho.vector.kitchen_layout import counter_lines, kitchen_cad_layout, tall_unit_entities
from samcloud.ortho.vector.upper_cabinets import UpperCabinetSettings, detect_upper_cabinets, upper_band_mask
from samcloud.bim.furniture_footprint import TightFootprintSettings, instance_point_sets
from samcloud.ortho.vector.upper_points import UpperPointSettings, detect_upper_cabinets_from_points, load_class_points, measured_z_band
from samcloud.bim.depth_evidence.plan_use import PlanDepth, open_plan_depth, plan_depth_fingerprint
from samcloud.ortho.vector.room_names import RoomNameSettings, name_halls, name_rooms, room_label_text, tag_for_name
from samcloud.ortho.vector.builtin_fronts import _absorb_einb, _absorb_furniture, resolve_builtin_fronts, tall_furniture_mask
from samcloud.ortho.vector.niche import NicheSettings, content_mask, resolve_recessed_niches
from samcloud.ortho.vector.door_zones import DoorZoneSettings, drop_misplaced_builtins
from samcloud.ortho.vector.footprints import _NON_FOOTPRINT_INSTANCE_LABELS, ISLAND_LABELS, FootprintFit, FurnitureSolidSettings, IslandSettings, clutter_rectangle, detect_kitchen_island, fit_footprint, furniture_solid_parts, height_step_details, object_instances, clip_to_room, refine_dining_instances, refit_clipped, wall_axis_rectangle, wall_band_of
from samcloud.ortho.vector.wall_lines import (
    WallFace,
    break_faces_at_openings,
    extend_faces_for_window_gaps,
    extract_axis_wall_faces,
    extract_wall_faces,
    face_buffer_polygons,
    face_entities,
)
from samcloud.ortho.vector.wall_tidy import tidy_wall_entities
from samcloud.ortho.vector.double_walls import DoubleWallSettings, TallUnitEvidence, WallEvidence, bound_rooms_by_walls, build_double_walls, update_opening_symbols
from samcloud.ortho.vector.labels import class_display_name, label_entities, load_label_settings, resolve_label_overlaps
from samcloud.ortho.vector.door_rooms import DoorRoomSettings, door_bounded_rooms
from samcloud.ortho.vector.sanitary_facing import SanitaryFacingSettings, orient_sanitary_inserts
from samcloud.ortho.vector.plan_polish import PlanPolishSettings, fit_kitchen_to_walls, place_object_labels, regularize_furniture, rooms_from_wall_cells
from samcloud.ortho.vector.lines import boundary_lines, trace_lines
from samcloud.ortho.vector.raster_grid import RasterGrid, mask_to_polygons
from samcloud.ortho.vector.regularize import regularize_polygons
from samcloud.ortho.vector.trees import TrunkCollector, TrunkSettings, detect_crowns, tree_annotation_entities, tree_entities
from samcloud.ortho.vector.vertical import VerticalEvidence, VerticalSettings, detect_door_leaf, extract_builtins, filter_builtins_to_rooms, tidy_builtins, vertical_evidence
from samcloud.ortho.vector.ceiling import CeilingSettings, ceiling_rooms, ceiling_wall_pieces, snap_exterior_faces_to_flucht

_LOGGER = logging.getLogger(__name__)


_MAX_WALL_THICKNESS_M: float = 0.30


@dataclass(frozen=True, slots=True)
class CleanPlanSettings:
    """Settings controlling clean plan vectorisation, simplification, and feature filters."""

    segment_source: str = "combined"
    label_min_area_outdoor: float = 20.0
    label_min_area_indoor: float = 0.3
    building_angle_tolerance_deg: float = 15.0
    line_gap_max: float = 1.5
    line_angle_tol_deg: float = 20.0
    line_min_length: float = 2.0
    courtyard_min_area: float = 25.0
    tree_min_height: float = 2.0
    point_object_min_area: float = 0.02
    contours: bool = True
    nominal_storey_height: float = 2.5
    indoor_wall_band: tuple[float, float] = (0.72, 0.94)
    indoor_section_band: tuple[float, float] = (0.36, 0.52)
    indoor_sill_band: tuple[float, float] = (0.08, 0.32)
    indoor_lintel_band: tuple[float, float] = (0.82, 0.94)
    indoor_face_min_length: float = 0.30
    indoor_face_ransac_residual: float = 0.02
    footprint_min_height: float = 0.05
    footprint_max_ceiling_fraction: float = 0.97
    footprint_step_height: float = 0.10
    indoor_vertical: VerticalSettings = VerticalSettings()
    indoor_ceiling: CeilingSettings = CeilingSettings()
    indoor_island: IslandSettings = IslandSettings()
    indoor_niche: NicheSettings = NicheSettings()
    indoor_door_zone: DoorZoneSettings = DoorZoneSettings()
    indoor_upper_cabinets: UpperCabinetSettings = UpperCabinetSettings()
    indoor_upper_points: UpperPointSettings = UpperPointSettings()
    indoor_tight_footprint: TightFootprintSettings = TightFootprintSettings()
    indoor_room_names: RoomNameSettings = RoomNameSettings()
    indoor_kitchen_run: KitchenRunSettings = KitchenRunSettings()
    indoor_furniture_solids: FurnitureSolidSettings = FurnitureSolidSettings()
    walls: DoubleWallSettings = DoubleWallSettings()
    plan_polish: PlanPolishSettings = PlanPolishSettings()
    indoor_door_rooms: DoorRoomSettings = DoorRoomSettings()
    outdoor_trunk: TrunkSettings = TrunkSettings()
    outdoor_hedges: HedgeSettings = HedgeSettings()
    outdoor_spots: SpotSettings = SpotSettings()
    outdoor_contour_labels: ContourLabelSettings = ContourLabelSettings()
    outdoor_roof: RoofSettings = RoofSettings()


def load_clean_plan_settings(path: Path | str | None = None) -> CleanPlanSettings:
    """Load clean plan settings from the clean_plan section of ortho_plan.yaml."""
    config_path = Path(path) if path is not None else CONFIG_DIRECTORY / "ortho_plan.yaml"
    if not config_path.is_file():
        return CleanPlanSettings()
    data = load_yaml(config_path)
    if not isinstance(data, dict):
        return CleanPlanSettings()
    section = data.get("clean_plan")
    if not isinstance(section, dict):
        return CleanPlanSettings()
    bands = section.get("indoor_bands", {})
    if not isinstance(bands, dict):
        bands = {}
    def band(name: str, default: tuple[float, float]) -> tuple[float, float]:
        value = bands.get(name, default)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return default
        return float(value[0]), float(value[1])
    raw_vertical = data.get("indoor_vertical", {})
    if not isinstance(raw_vertical, dict):
        raw_vertical = {}
    defaults = VerticalSettings()
    vertical = VerticalSettings(**{
        name: type(getattr(defaults, name))(raw_vertical.get(name, getattr(defaults, name)))
        for name in VerticalSettings.__dataclass_fields__
    })
    raw_ceiling = data.get("indoor_ceiling", {})
    if not isinstance(raw_ceiling, dict):
        raw_ceiling = {}
    ceiling_defaults = CeilingSettings()
    ceiling = CeilingSettings(**{
        name: type(getattr(ceiling_defaults, name))(raw_ceiling.get(name, getattr(ceiling_defaults, name)))
        for name in CeilingSettings.__dataclass_fields__
    })
    def section_of(name: str, cls: type) -> Any:
        raw = data.get(name, {})
        raw = raw if isinstance(raw, dict) else {}
        defaults_ = cls()
        return cls(**{field: type(getattr(defaults_, field))(raw.get(field, getattr(defaults_, field))) for field in cls.__dataclass_fields__})
    return CleanPlanSettings(
        segment_source=str(section.get("segment_source", "combined")),
        label_min_area_outdoor=float(section.get("label_min_area_outdoor", 20.0)),
        label_min_area_indoor=float(section.get("label_min_area_indoor", 0.3)),
        building_angle_tolerance_deg=float(section.get("building_angle_tolerance_deg", 15.0)),
        line_gap_max=float(section.get("line_gap_max", 1.5)),
        line_angle_tol_deg=float(section.get("line_angle_tol_deg", 20.0)),
        line_min_length=float(section.get("line_min_length", 2.0)),
        courtyard_min_area=float(section.get("courtyard_min_area", 25.0)),
        tree_min_height=float(section.get("tree_min_height", 2.0)),
        point_object_min_area=float(section.get("point_object_min_area", 0.02)),
        contours=bool(section.get("contours", True)),
        nominal_storey_height=float(section.get("nominal_storey_height", 2.5)),
        indoor_wall_band=band("wall", (0.72, 0.94)),
        indoor_section_band=band("section", (0.36, 0.52)),
        indoor_sill_band=band("sill", (0.08, 0.32)),
        indoor_lintel_band=band("lintel", (0.82, 0.94)),
        indoor_face_min_length=float(section.get("indoor_face_min_length", 0.30)),
        indoor_face_ransac_residual=float(section.get("indoor_face_ransac_residual", 0.02)),
        footprint_min_height=float(section.get("footprint_min_height", 0.05)),
        footprint_max_ceiling_fraction=float(section.get("footprint_max_ceiling_fraction", 0.97)),
        footprint_step_height=float(section.get("footprint_step_height", 0.10)),
        indoor_vertical=vertical,
        indoor_ceiling=ceiling,
        indoor_island=section_of("indoor_island", IslandSettings),
        indoor_niche=section_of("indoor_niche", NicheSettings),
        indoor_door_zone=section_of("indoor_door_zone", DoorZoneSettings),
        indoor_upper_cabinets=section_of("indoor_upper_cabinets", UpperCabinetSettings),
        indoor_upper_points=section_of("indoor_upper_points", UpperPointSettings),
        indoor_tight_footprint=section_of("indoor_tight_footprint", TightFootprintSettings),
        indoor_room_names=section_of("indoor_room_names", RoomNameSettings),
        indoor_kitchen_run=section_of("indoor_kitchen_run", KitchenRunSettings),
        indoor_furniture_solids=section_of("indoor_furniture_solids", FurnitureSolidSettings),
        walls=section_of("walls", DoubleWallSettings),
        plan_polish=section_of("plan_polish", PlanPolishSettings),
        indoor_door_rooms=section_of("indoor_door_rooms", DoorRoomSettings),
        outdoor_trunk=section_of("outdoor_trunk", TrunkSettings),
        outdoor_hedges=section_of("outdoor_hedges", HedgeSettings),
        outdoor_spots=section_of("outdoor_spots", SpotSettings),
        outdoor_contour_labels=section_of("outdoor_contour_labels", ContourLabelSettings),
        outdoor_roof=section_of("outdoor_roof", RoofSettings),
    )


def clean_plan_cache_dir(project_dir: Path | str, level: Level | None) -> Path:
    """Return the on-disk cache directory for the clean orthophoto DXF plan."""
    return Path(project_dir) / "products" / "dxf" / ("site" if level is None else level.id) / "ortho"


def _available_clean_plan_settings(project: Path, level: Level | None, settings: CleanPlanSettings | None) -> CleanPlanSettings:
    """Use cached point-cloud segmentation when it is the only available source."""
    selected = settings or load_clean_plan_settings()
    if settings is None and load_segment_result(project, level, selected.segment_source) is None:
        pointcloud = replace(selected, segment_source="pointcloud")
        if load_segment_result(project, level, pointcloud.segment_source) is not None:
            return pointcloud
    return selected


def clean_plan_fingerprint(
    project_dir: Path | str,
    level: Level | None,
    settings: CleanPlanSettings | None = None,
) -> str:
    """Fingerprint inputs, settings, and the wall/opening code that produces pass 1."""
    project = Path(project_dir)
    plan_settings = _available_clean_plan_settings(project, level, settings)
    seg = load_segment_result(project, level, plan_settings.segment_source)
    if seg is None:
        ortho = load_ortho_result(ortho_output_dir(project, level))
        if ortho is None:
            raise OrthoError("Build orthophoto first")
        raise OrthoError("Run segmentation first")

    layer_config = load_layer_config()
    layers_data = [
        {
            "name": lyr.name,
            "project_mode": lyr.project_mode,
            "codes": list(lyr.codes),
            "geometry": lyr.geometry,
            "min_area": lyr.min_area,
            "simplify_tolerance": lyr.simplify_tolerance,
            "color": lyr.color,
            "hatch": lyr.hatch,
        }
        for lyr in layer_config.layers
    ]
    ortho_plan_file = CONFIG_DIRECTORY / "ortho_plan.yaml"
    ortho_plan_content = ortho_plan_file.read_text(encoding="utf-8") if ortho_plan_file.is_file() else ""

    payload = {
        "segment_fingerprint": seg.fingerprint,
        "layers": layers_data,
        "ortho_plan_yaml": ortho_plan_content,
        "settings": asdict(plan_settings),
        "pass1_code": _wall_pipeline_code_fingerprint(),
    }
    instances_path = project / "products" / "classified" / "instances.json"
    if instances_path.is_file():
        payload["instances_json"] = hashlib.sha256(instances_path.read_bytes()).hexdigest()
    depth_key = plan_depth_fingerprint(project)
    if depth_key is not None:  # only when depth is used: projects without a product keep their fingerprint
        payload["depth_in_plan"] = depth_key
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _wall_pipeline_code_fingerprint() -> str:
    """Return a deterministic digest of code that can change a pass-1 wall plan.

    A cached pure pass 1 must not survive a code deployment: refined openings are
    based on that geometry.  Settings alone are therefore not a sufficient key.
    """
    package = Path(__file__).resolve().parents[1]
    roots = (package / "ortho", package / "bim" / "openings3d")
    digest = hashlib.sha256()
    files = sorted(
        (path for root in roots if root.is_dir() for path in root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(package).as_posix(),
    )
    for path in files:
        digest.update(path.relative_to(package).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def generate_clean_plan(
    project_dir: Path,
    *,
    level: Level | None,
    mode: str,
    settings: CleanPlanSettings | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
    point_source: Path | None = None,
) -> DxfPlanResult:
    """Generate the clean DXF plan from orthophoto rasters, segmentation, and contours."""
    if mode == "indoor":
        if level is None:
            raise ValueError("Indoor clean plan generation requires a level")
        return _generate_indoor_clean_plan(
            Path(project_dir), level, settings=settings, progress=progress, cancel=cancel, point_source=point_source,
        )
    if mode != "outdoor":
        raise ValueError(f"Unsupported project mode: {mode}")

    check_cancel(cancel)
    project = Path(project_dir)
    plan_settings = _available_clean_plan_settings(project, level, settings)

    # Step 1: Verify and load ortho + segmentation
    report_progress(progress, 0.05, "Loading orthophoto and segmentation")
    ortho_dir = ortho_output_dir(project, level)
    ortho = load_ortho_result(ortho_dir)
    if ortho is None:
        raise OrthoError("Build orthophoto first")

    segment = load_segment_result(project, level, plan_settings.segment_source)
    if segment is None:
        raise OrthoError("Run segmentation first")

    check_cancel(cancel)

    # Grid and Frame reconstruction
    bounds = ortho.stats["bounds"]
    pixel_size = float(ortho.stats["pixel_size"])
    width = int(ortho.stats["width"])
    height = int(ortho.stats["height"])
    grid = RasterGrid(
        min_x=float(bounds[0]),
        max_y=float(bounds[3]),
        pixel_size=pixel_size,
        width=width,
        height=height,
    )
    frame_dict = ortho.stats["frame"]
    frame = Frame(
        origin=np.asarray(frame_dict["origin"], dtype=np.float64),
        scale_factor=float(frame_dict["scale"]),
        rotation=np.asarray(frame_dict["rotation"], dtype=np.float64),
    )

    segment_classes = np.load(segment.classes_path, allow_pickle=False).astype(np.uint8, copy=False)
    dsm = np.load(ortho.dsm_path, allow_pickle=False).astype(np.float32, copy=False)
    dtm = np.load(ortho.dtm_path, allow_pickle=False).astype(np.float32, copy=False)

    layer_config = load_layer_config()
    outdoor_layers = layer_config.for_project_mode("outdoor")

    output_layers: dict[str, list[dict[str, Any]]] = {lyr.name: [] for lyr in outdoor_layers}
    output_layers["S-TOPO-MAJR"] = []
    output_layers["S-TOPO-MINR"] = []
    output_layers["S-ELEV"] = []
    output_layers["S-ANNO"] = []

    # Step 2: Group layers by geometry type
    report_progress(progress, 0.20, "Extracting area partitions and regularizing buildings")
    check_cancel(cancel)

    polygon_layers = [lyr for lyr in outdoor_layers if lyr.geometry == "polygon"]
    label_raster = np.zeros((grid.height, grid.width), dtype=np.uint8)
    class_params: dict[int, AreaParams] = {}
    class_to_layer: dict[int, str] = {}
    tolerances: dict[str, float] = {}

    for idx, lyr in enumerate(polygon_layers, start=1):
        class_to_layer[idx] = lyr.name
        tolerances[lyr.name] = lyr.simplify_tolerance
        class_params[idx] = AreaParams(
            min_area=lyr.min_area,
            courtyard_min_area=plan_settings.courtyard_min_area,
        )
        mask = np.isin(segment_classes, lyr.codes)
        label_raster[mask] = idx

    cleaned_labels = clean_label_raster(label_raster, class_params=class_params, grid=grid)
    partition = build_partition(cleaned_labels, grid, class_to_layer=class_to_layer, tolerances=tolerances)

    # Regularize S-BLDG and clip neighbours to keep shared borders
    regularized_bldgs: list[Polygon] = []
    if "S-BLDG" in partition and partition["S-BLDG"]:
        regularized_bldgs = regularize_polygons(
            partition["S-BLDG"],
            angle_tolerance_deg=plan_settings.building_angle_tolerance_deg,
        )
        regularized_bldgs = [_rectify_nearly_rectangular_building(poly) for poly in regularized_bldgs]
        _clip_neighbours_to_regularized_buildings(partition, regularized_bldgs)

    # Convert partition polygons to neutral entities
    partition_entities_dict = partition_entities(partition)
    for lyr_name, ents in partition_entities_dict.items():
        if lyr_name in output_layers:
            output_layers[lyr_name].extend(ents)

    check_cancel(cancel)

    # Step 2b: tree crowns, then one point-cloud pass for the ground DTM and the trunk cylinders.
    report_progress(progress, 0.30, "Detecting tree crowns and streaming terrain points")
    tree_layer = next((lyr for lyr in outdoor_layers if lyr.geometry == "tree"), None)
    chm = dsm - dtm
    crowns: list[Any] = []
    if tree_layer is not None:
        tree_mask = np.isin(segment_classes, tree_layer.codes)
        # The segmentation supplies the crown outline while CHM validates that a
        # component is a tree.  Keep the full classified crown during watershed;
        # clipping it to CHM >= min_height otherwise shrinks wide, low-edge crowns
        # below the minimum-diameter filter.
        crown_chm = np.where(tree_mask, np.maximum(chm, plan_settings.tree_min_height), chm)
        crowns = detect_crowns(crown_chm, tree_mask, grid, min_height=plan_settings.tree_min_height)
    raster_sampler = DtmSampler(dtm, grid)
    resolved_point_source = point_source or _find_point_source(project, ortho.stats)
    filled_dtm: np.ndarray | None = None
    dtm_grid: RasterGrid | None = None
    trunk_diameters: list[float | None] | None = None
    contour_settings = load_contour_settings()
    if (plan_settings.contours or crowns) and resolved_point_source and resolved_point_source.is_file():
        ground_codes = ground_codes_from_config()
        grid_bounds = (
            grid.min_x,
            grid.max_y - grid.height * grid.pixel_size,
            grid.min_x + grid.width * grid.pixel_size,
            grid.max_y,
        )
        acc = GroundDtmAccumulator(bounds=grid_bounds, resolution=contour_settings.resolution)
        collector = None
        if crowns and tree_layer is not None:
            trunk_xy = np.asarray([crown.trunk for crown in crowns], dtype=float)
            collector = TrunkCollector(trunk_xy, raster_sampler.sample(trunk_xy[:, 0], trunk_xy[:, 1]), plan_settings.outdoor_trunk)
        for chunk in iter_source_chunks(resolved_point_source, 200_000, cancel=cancel, stage="Streaming terrain and trunk points"):
            check_cancel(cancel)
            if len(chunk) == 0:
                continue
            xyz = np.column_stack((chunk.x, chunk.y, chunk.z))
            xyz_local = frame.to_local(xyz)
            classes_chunk = np.asarray(chunk.classification, dtype=np.uint8)
            acc.add(xyz_local, classes_chunk, ground_codes)
            if collector is not None and tree_layer is not None:
                collector.add(xyz_local, classes_chunk, tuple(tree_layer.codes))
        raw_dtm, dtm_grid = acc.finish(contour_settings)
        filled_dtm = fill_dtm(
            raw_dtm,
            dtm_grid,
            max_gap=contour_settings.max_gap,
            smoothing_sigma_m=contour_settings.smoothing_sigma_m,
        )
        if collector is not None:
            trunk_diameters = collector.diameters()
    terrain = DtmSampler(filled_dtm, dtm_grid) if filled_dtm is not None and dtm_grid is not None and np.isfinite(filled_dtm).any() else raster_sampler
    z_offset = float(frame.origin[2])
    z_scale = float(frame.scale_factor) if frame.scale_factor else 1.0

    def absolute_heights(local_z: Any) -> np.ndarray:
        """Absolute (CRS) height of local heights: local / scale + origin z."""
        return np.atleast_1d(np.asarray(local_z, dtype=np.float64)) / z_scale + z_offset

    # Pavement and walk outlines are continuous 3D polylines (terrain breaklines): the 2D outline stays
    # in ``points``; per-vertex DTM heights in ``z`` / ``hole_z`` make the writer emit one 3D polyline per ring.
    for edge_layer in ("S-PAVE", "S-WALK"):
        for entity in output_layers.get(edge_layer, []):
            if entity.get("type") == "polyline" and entity.get("closed") and len(entity.get("points", [])) >= 3:
                drape_outline_heights(entity, terrain)

    check_cancel(cancel)

    # Step 3: Skeleton and line layers
    report_progress(progress, 0.45, "Tracing curb and skeleton line networks")
    line_layers = [lyr for lyr in outdoor_layers if lyr.geometry in ("line", "skeleton")]
    for lyr in line_layers:
        check_cancel(cancel)
        mask = np.isin(segment_classes, lyr.codes)
        lines: list[LineString] = []
        if mask.any():
            lines = trace_lines(
                mask,
                grid,
                min_length=plan_settings.line_min_length,
                gap_max=plan_settings.line_gap_max,
                angle_tol_deg=plan_settings.line_angle_tol_deg,
                smooth_iterations=2,
                simplify_tolerance=lyr.simplify_tolerance,
            )

        if lyr.name == "S-CURB":
            b_lines = boundary_lines(
                partition,
                "S-PAVE",
                "S-WALK",
                min_length=plan_settings.line_min_length,
                simplify_tolerance=lyr.simplify_tolerance,
            )
            all_curbs = list(lines) + list(b_lines)
            if all_curbs:
                # ``line_merge`` interprets a Python list as an array-like geometry
                # and returns an ndarray. Unite first so a single shared boundary is
                # retained as a geometry on every Shapely 2.x build.
                merged = line_merge(unary_union(all_curbs))
                curb_candidates = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
                lines = [c for c in curb_candidates if isinstance(c, LineString) and c.length >= plan_settings.line_min_length]

        for line in lines:
            if lyr.name == "S-CURB":
                # Curbs are continuous 3D breaklines: one polyline per continuous edge, z from the DTM.
                draped = drape_coordinates(list(line.coords), terrain)
                if all(np.isfinite(point[2]) for point in draped):
                    output_layers[lyr.name].append({"type": "polyline3d", "points": draped, "closed": False, "_breakline": True})
                    continue
            output_layers[lyr.name].append({
                "type": "polyline",
                "points": [[float(x), float(y)] for x, y in line.coords],
                "closed": False,
            })

    check_cancel(cancel)

    # Step 4: Tree layer
    report_progress(progress, 0.60, "Detecting tree crowns and point objects")
    hedge_count = 0
    if tree_layer is not None:
        output_layers[tree_layer.name].extend(tree_entities(crowns))
        output_layers["S-ANNO"].extend(tree_annotation_entities(crowns, trunk_diameters, text_height=plan_settings.outdoor_trunk.text_height_m))
        # Shrubs and hedges: vegetation below tree height outside the detected crowns, as smooth outlines.
        vegetation_codes = tuple(sorted({code for lyr in outdoor_layers if lyr.name in {"S-VEG", tree_layer.name} for code in lyr.codes}))
        crown_discs = np.zeros(segment_classes.shape, dtype=np.uint8)
        for crown in crowns:
            row, col = grid.local_to_pixel(float(crown.center[0]), float(crown.center[1]))
            cv2.circle(crown_discs, (int(round(col)), int(round(row))), max(1, int(round(0.5 * crown.diameter / grid.pixel_size))), 1, -1)
        hedge_mask = np.isin(segment_classes, vegetation_codes) & ~crown_discs.astype(bool)
        hedge_items = hedge_entities(hedge_mask, chm, grid, replace(plan_settings.outdoor_hedges, max_height_m=min(plan_settings.outdoor_hedges.max_height_m, plan_settings.tree_min_height)))
        for item in hedge_items:
            if item.get("type") == "text":
                output_layers["S-ANNO"].append(item)
            else:
                output_layers["S-VEG"].append(item)
                hedge_count += 1

    # Step 5: Circle and point layers (manhole, pole, sign, light)
    point_layers = [lyr for lyr in outdoor_layers if lyr.geometry in ("circle", "point")]
    cyclone_registry = load_cyclone_codes()
    for lyr in point_layers:
        check_cancel(cancel)
        mask = np.isin(segment_classes, lyr.codes)
        if not mask.any():
            continue
        _num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for idx, component_slice in enumerate(ndimage.find_objects(labels), start=1):
            if component_slice is None:
                continue
            area = float(stats[idx, cv2.CC_STAT_AREA]) * (grid.pixel_size ** 2)
            if area < plan_settings.point_object_min_area:
                continue
            comp_mask = labels[component_slice] == idx
            comp_codes = segment_classes[component_slice][comp_mask]
            dominant_code = int(np.bincount(comp_codes).argmax()) if len(comp_codes) else (lyr.codes[0] if lyr.codes else 0)
            class_info = cyclone_registry.get(dominant_code, {})
            class_name_str = str(class_info.get("name", "")).lower()

            block_name = _resolve_point_block(lyr.name, class_name_str)
            display = class_display_name(class_info.get("name", lyr.name))
            entity = fit_block_insert(
                comp_mask,
                grid,
                block_name,
                display,
                row_offset=component_slice[0].start,
                col_offset=component_slice[1].start,
            )
            output_layers[lyr.name].append(entity)

    check_cancel(cancel)

    # Step 6: Contours (absolute levels), contour labels and DGM spot elevations
    contour_label_count = 0
    if plan_settings.contours and filled_dtm is not None and dtm_grid is not None:
        report_progress(progress, 0.75, "Extracting terrain elevation contours")
        c_entities = contour_entities(
            filled_dtm,
            dtm_grid,
            settings=contour_settings,
            exclude=regularized_bldgs,
            z_offset=z_offset,
            z_scale=z_scale,
        )
        for ent in c_entities:
            target_layer = "S-TOPO-MAJR" if ent.get("layer_role") == "major" else "S-TOPO-MINR"
            output_layers[target_layer].append(ent)
        building_union = unary_union(regularized_bldgs) if regularized_bldgs else None
        output_layers["S-TOPO-MAJR"] = label_major_contours(output_layers["S-TOPO-MAJR"], plan_settings.outdoor_contour_labels, avoid=building_union)
        contour_label_count = sum(1 for ent in output_layers["S-TOPO-MAJR"] if ent.get("type") == "text")

    report_progress(progress, 0.82, "Selecting DGM spot elevations")
    spot_groups: list[tuple[str, list[tuple[float, float]]]] = []
    spot_settings = plan_settings.outdoor_spots
    spot_groups.append(("manhole", [(float(ent["point"][0]), float(ent["point"][1])) for ent in output_layers.get("S-MANH", []) if ent.get("type") == "insert"]))
    roof_settings = plan_settings.outdoor_roof
    main_buildings = [polygon for polygon in regularized_bldgs if polygon.area >= roof_settings.min_area_m2]
    spot_groups.append(("building_corner", [(float(x), float(y)) for polygon in main_buildings for x, y in list(polygon.exterior.coords)[:-1]]))
    spot_groups.append(("curb", [point for ent in output_layers.get("S-CURB", []) if ent.get("type") == "polyline3d" for point in points_along([(p[0], p[1]) for p in ent["points"]], spot_settings.curb_spacing_m)]))
    ground_lookup = np.isin(segment_classes, tuple(code for code in ground_codes_from_config() if code != 9))
    blocked = np.zeros(segment_classes.shape, dtype=np.uint8)
    for polygon in regularized_bldgs:
        cv2.fillPoly(blocked, [np.asarray([[round(grid.local_to_pixel(x, y)[1]), round(grid.local_to_pixel(x, y)[0])] for x, y in polygon.exterior.coords], dtype=np.int32)], 1)
    for crown in crowns:
        row, col = grid.local_to_pixel(float(crown.center[0]), float(crown.center[1]))
        cv2.circle(blocked, (int(round(col)), int(round(row))), max(1, int(round(0.5 * crown.diameter / grid.pixel_size))), 1, -1)

    def is_ground(x: float, y: float) -> bool:
        row, col = (int(round(value)) for value in grid.local_to_pixel(x, y))
        return bool(0 <= row < grid.height and 0 <= col < grid.width and ground_lookup[row, col] and not blocked[row, col] and np.isfinite(dsm[row, col]))

    spot_groups.append(("grid", ground_grid_points((grid.min_x, grid.max_y - grid.height * grid.pixel_size, grid.min_x + grid.width * grid.pixel_size, grid.max_y), spot_settings.grid_spacing_m, is_ground)))
    spots = select_spots(spot_groups, spot_settings)
    spot_local = terrain.sample(np.asarray([spot.x for spot in spots]), np.asarray([spot.y for spot in spots])) if spots else np.empty(0)
    output_layers["S-ELEV"].extend(spot_entities(spots, [float(value) for value in spot_local], [float(value) for value in absolute_heights(spot_local)], spot_settings))

    check_cancel(cancel)

    # Step 7: Smart text labels
    report_progress(progress, 0.90, "Placing polygon text annotations")
    label_items: list[tuple[Polygon, str]] = []
    for lyr in polygon_layers:
        polygons = partition.get(lyr.name, [])
        if not polygons:
            continue
        dominant_label = _dominant_layer_display_name(lyr, segment_classes, segment.metadata)
        for poly in polygons:
            if poly.area >= plan_settings.label_min_area_outdoor:
                label_items.append((poly, dominant_label))

    lbl_settings = replace(load_label_settings(), min_area=plan_settings.label_min_area_outdoor)
    text_entities = label_entities(label_items, lbl_settings)
    output_layers["S-ANNO"].extend(text_entities)
    roof_texts = 0
    for polygon in main_buildings:
        roof = roof_heights(polygon, dsm, grid, plan_settings.outdoor_roof)
        if roof is None:
            continue
        ridge_abs, eave_abs = (float(value) for value in absolute_heights(list(roof)))
        output_layers["S-ANNO"].extend(roof_height_entities(polygon, ridge_abs, eave_abs, plan_settings.outdoor_roof))
        roof_texts += 1

    check_cancel(cancel)

    # Step 8: Build and persist DxfPlanResult
    report_progress(progress, 0.95, "Assembling DXF plan result")

    # Layer styling, colors, and hatches
    layer_colors: dict[str, int] = {}
    layer_linetypes: dict[str, str] = {}
    layer_hatches: dict[str, bool] = {}
    layer_styles: dict[str, dict[str, Any]] = {}

    for lyr in outdoor_layers:
        layer_colors[lyr.name] = lyr.color
        layer_linetypes[lyr.name] = lyr.linetype
        layer_hatches[lyr.name] = lyr.hatch
        style: dict[str, Any] = {}
        if lyr.true_color is not None:
            style["true_color"] = lyr.true_color
        if lyr.lineweight is not None:
            style["lineweight"] = lyr.lineweight
        if lyr.hatch_pattern is not None:
            style["hatch_pattern"] = lyr.hatch_pattern
        if lyr.hatch_scale is not None:
            style["hatch_scale"] = lyr.hatch_scale
        if style:
            layer_styles[lyr.name] = style

    ortho_plan_data = load_yaml(CONFIG_DIRECTORY / "ortho_plan.yaml") or {}
    for lyr_name, lyr_info in ortho_plan_data.get("layers", {}).items():
        if not isinstance(lyr_info, dict):
            continue
        if lyr_name not in layer_colors:
            layer_colors[lyr_name] = int(lyr_info.get("color", 7))
        if lyr_name not in layer_linetypes:
            layer_linetypes[lyr_name] = str(lyr_info.get("linetype", "CONTINUOUS"))
        if lyr_name not in layer_hatches:
            layer_hatches[lyr_name] = False
        style = layer_styles.setdefault(lyr_name, {})
        if "true_color" in lyr_info:
            style["true_color"] = tuple(int(c) for c in lyr_info["true_color"])
        if "lineweight" in lyr_info:
            style["lineweight"] = int(lyr_info["lineweight"])

    blocks = sorted(list({str(ent["name"]) for ents in output_layers.values() for ent in ents if ent.get("type") == "insert"}))

    statuses: dict[str, LayerStatus] = {}
    for lyr_name, ents in output_layers.items():
        cnt = dxf_entity_count(ents)
        statuses[lyr_name] = LayerStatus(
            name=lyr_name,
            state="ok" if cnt > 0 else "empty",
            reason=None if cnt > 0 else "no geometry survived filtering",
            entity_count=cnt,
        )

    fingerprint = clean_plan_fingerprint(project, level, plan_settings)
    background = BackgroundImage(
        rgb_png=None,
        class_png=None,
        bounds=(grid.min_x, grid.max_y - grid.height * grid.pixel_size, grid.min_x + grid.width * grid.pixel_size, grid.max_y),
    )

    result = DxfPlanResult(
        project_name=project.name,
        level_name="site" if level is None else level.name,
        mode="outdoor",
        profile_name=segment.metadata.get("model_profile") if segment.metadata else None,
        scale_factor=frame.scale_factor,
        frame=frame,
        settings_fingerprint=fingerprint,
        layers=output_layers,
        statuses=statuses,
        layer_colors=layer_colors,
        background=background,
        layer_linetypes=layer_linetypes,
        layer_hatches=layer_hatches,
        layer_styles=layer_styles,
        blocks=blocks,
        variant="ortho",
        fingerprint_inputs={"outdoor": {
            "trees": len(crowns),
            "trees_with_trunk": sum(value is not None for value in (trunk_diameters or [])),
            "hedges": hedge_count,
            "curb_polylines": sum(1 for ent in output_layers.get("S-CURB", []) if ent.get("type") == "polyline3d"),
            "pavement_edge_polylines": sum(1 + len(ent.get("hole_z", [])) for name in ("S-PAVE", "S-WALK") for ent in output_layers.get(name, []) if ent.get("z") is not None),
            "spot_elevations": sum(1 for ent in output_layers["S-ELEV"] if ent.get("type") == "text"),
            "spots_by_kind": {kind: sum(1 for ent in output_layers["S-ELEV"] if ent.get("type") == "text" and ent.get("_spot") == kind) for kind in ("manhole", "building_corner", "curb", "grid")},
            "contour_labels": contour_label_count,
            "roof_texts": roof_texts,
            "terrain_source": "point_cloud_dtm" if terrain is not raster_sampler else "ortho_dtm",
        }},
    )

    cache_dir = clean_plan_cache_dir(project, level)
    save_cached_result(result, cache_dir)
    report_progress(progress, 1.0, "Clean DXF plan generation complete")
    return result


def _rectify_nearly_rectangular_building(polygon: Polygon) -> Polygon:
    """Remove raster notches from a footprint that is demonstrably rectangular.

    The generic regulariser preserves meaningful non-orthogonal or concave edges.
    A near-rectangular raster footprint can still retain one one-pixel notch,
    however, which produces a fifth corner.  The minimum rotated rectangle is
    appropriate only when it changes the measured footprint by less than 12%.
    """
    rectangle = polygon.minimum_rotated_rectangle
    if rectangle.is_empty or rectangle.area <= 0.0:
        return polygon
    ratio = polygon.area / rectangle.area
    return rectangle if ratio >= 0.88 else polygon


def _generate_indoor_clean_plan(
    project: Path,
    level: Level,
    *,
    settings: CleanPlanSettings | None,
    progress: ProgressCallback | None,
    cancel: CancelToken | None,
    point_source: Path | None,
) -> DxfPlanResult:
    """Generate a level plan from section occupancy, segmentation, and DSM data."""
    check_cancel(cancel)
    plan_settings = _available_clean_plan_settings(project, level, settings)
    report_progress(progress, 0.05, "Loading indoor orthophoto and segmentation")
    ortho = load_ortho_result(ortho_output_dir(project, level))
    if ortho is None:
        raise OrthoError("Build orthophoto first")
    segment = load_segment_result(project, level, plan_settings.segment_source)
    if segment is None:
        raise OrthoError("Run segmentation first")

    bounds = ortho.stats["bounds"]
    grid = RasterGrid(
        min_x=float(bounds[0]), max_y=float(bounds[3]), pixel_size=float(ortho.stats["pixel_size"]),
        width=int(ortho.stats["width"]), height=int(ortho.stats["height"]),
    )
    frame_data = ortho.stats["frame"]
    frame = Frame(
        origin=np.asarray(frame_data["origin"], dtype=np.float64), scale_factor=float(frame_data["scale"]),
        rotation=np.asarray(frame_data["rotation"], dtype=np.float64),
    )
    classes = np.load(segment.classes_path, allow_pickle=False).astype(np.uint8, copy=False)
    confidence = np.load(segment.confidence_path, allow_pickle=False).astype(np.float32, copy=False)
    dsm = np.load(ortho.dsm_path, allow_pickle=False).astype(np.float32, copy=False)
    if classes.shape != (grid.height, grid.width) or dsm.shape != classes.shape:
        raise OrthoError("Indoor segmentation and elevation rasters do not match the orthophoto grid")
    fine_classes, fine_confidence, fine_names, fine_label_source = _load_fine_labels(segment, classes.shape)
    instance_raster = _load_instance_raster(segment, classes.shape)
    instance_labels = _load_instance_labels(project, point_source) if instance_raster is not None else {}
    indoor_layers = load_layer_config().for_project_mode("indoor")
    output_layers: dict[str, list[dict[str, Any]]] = {layer.name: [] for layer in indoor_layers}
    output_layers["A-FURN-DETL"] = []
    output_layers["A-EINB"] = []
    output_layers["A-OBEN"] = []
    output_layers["A-AREA"] = []
    output_layers["A-ANNO"] = []

    report_progress(progress, 0.18, "Building indoor section occupancy")
    units_per_metre, bands = _indoor_scale_and_bands(level, plan_settings)
    # Optional compact depth evidence (COLMAP-supported voxels for faces, free space for occupancy); None = point cloud only.
    plan_depth = open_plan_depth(project, frame, level.floor_z, units_per_metre)
    wall_evidence, section, low, high, wall_points, vertical = _section_occupancy(
        project, ortho.stats, point_source, frame, grid, level, cancel, bands, units_per_metre, plan_settings.indoor_vertical, plan_depth,
    )
    structural_codes = _indoor_structural_codes(indoor_layers)
    if not wall_evidence.any():
        # A plan can still be reviewed when its classified source is unavailable;
        # segmentation is a conservative fallback, not a replacement for the
        # measured section where a source is present.
        wall_evidence = np.isin(classes, structural_codes)
    if not section.any():
        section = wall_evidence.copy()
        low = wall_evidence.copy()
        high = wall_evidence.copy()

    report_progress(progress, 0.35, "Extracting walls and openings")
    free_space = np.isin(classes, (102,))
    if not free_space.any():
        free_space = ~section
    closed_floor = _closed_floor_mask(free_space, grid, units_per_metre)
    furniture_layers = [layer for layer in indoor_layers if layer.geometry == "furniture"]
    furniture_codes = tuple(code for layer in furniture_layers for code in layer.codes)
    ceiling_furniture_mask = (np.isin(classes, furniture_codes) | (fine_classes > 0)) & ~wall_evidence
    ceiling_furniture_polygons = [polygon for polygon in mask_to_polygons(ceiling_furniture_mask, grid) if polygon.area >= 0.05 * units_per_metre**2]
    if not len(wall_points):
        # This fallback preserves reviewability for legacy raster-only jobs;
        # normal generation always uses the streamed upper-band measurements.
        rows, cols = np.nonzero(wall_evidence)
        raster_points = np.asarray([grid.pixel_to_local(float(row), float(col)) for row, col in zip(rows, cols, strict=True)], dtype=float)
        wall_points = np.column_stack((np.repeat(raster_points, 3, axis=0), np.tile(np.asarray((0.30, 0.60, 0.90)) * units_per_metre, len(raster_points))))
    faces = extract_axis_wall_faces(wall_points, grid, units_per_metre=units_per_metre, passable=free_space)
    theta = _indoor_wall_axis([face.line for face in faces if face.kind == "line" and face.line is not None])
    ceiling_height = bands["wall"][1] / 0.94
    rooms_data = ceiling_rooms(
        vertical.ceiling_count if vertical.ceiling_count is not None else np.zeros_like(section),
        vertical.soffit if vertical.soffit is not None else np.zeros_like(section), wall_evidence, grid, theta, units_per_metre, plan_settings.indoor_ceiling,
    )
    ceiling_faces, ceiling_edges_rejected = ceiling_wall_pieces(
        rooms_data.polygons, faces, free_space | closed_floor,
        vertical.ceiling_count if vertical.ceiling_count is not None else np.zeros_like(section),
        vertical.ceiling_z if vertical.ceiling_z is not None else np.full(section.shape, -np.inf, dtype=np.float32),
        vertical.soffit if vertical.soffit is not None else np.zeros_like(section), wall_evidence,
        grid, ceiling_height, units_per_metre, plan_settings.indoor_ceiling, furniture=ceiling_furniture_polygons, theta=theta,
    )
    faces.extend(ceiling_faces)
    snap_exterior_faces_to_flucht(faces, free_space | closed_floor, grid, units_per_metre)
    structural_walls = face_buffer_polygons(faces, 0.015 * units_per_metre)
    door_codes = next((layer.codes for layer in indoor_layers if layer.name == "A-DOOR"), ())
    door_mask = np.isin(classes, door_codes)
    window_codes = next((layer.codes for layer in indoor_layers if layer.name == "A-WIND"), ())
    window_mask = np.isin(classes, window_codes) if window_codes else np.zeros_like(section)
    openings = detect_openings(section, low, high, structural_walls, grid, door_mask=door_mask, door_min_width=0.6 * units_per_metre, door_max_width=1.3 * units_per_metre, window_min_width=0.4 * units_per_metre, window_max_width=3.0 * units_per_metre)
    openings = _snap_openings_to_faces(openings, faces, 0.30 * units_per_metre)
    openings = [opening for opening in openings if _opening_on_faces(opening, faces, 0.02 * units_per_metre)]
    openings = [opening for opening in openings if opening.kind != "WINDOW" or _window_is_exterior(opening, free_space, grid, units_per_metre)]
    # A section gap with wall above and below is only a window when the
    # segmentation also sees window evidence there; furniture occlusion leaves
    # the same gap signature and must not cut the wall face.
    openings = [opening for opening in openings if opening.kind != "WINDOW" or _opening_class_fraction(opening, window_mask, grid, units_per_metre) >= 0.05]
    openings.extend(_class_window_openings(window_mask, faces, openings, grid, units_per_metre))
    openings.extend(_collinear_face_gap_openings(faces, free_space | closed_floor, high, door_mask, classes != 0, grid, units_per_metre))
    openings = _snap_openings_to_faces(openings, faces, 0.30 * units_per_metre)
    openings = _normalise_door_openings(openings, door_mask, high, window_mask, free_space | closed_floor, grid, units_per_metre)
    opening_candidate_count = len(openings)
    openings = _deduplicate_plan_openings(openings, 0.15 * units_per_metre)
    doors_deduplicated = opening_candidate_count - len(openings)
    openings = [opening for opening in openings if not _opening_into_void(opening, classes != 0, grid, units_per_metre)]
    openings, openings_3d_info = _apply_refined_openings(openings, faces, project, level, units_per_metre, clean_plan_fingerprint(project, level, plan_settings))
    windows = [opening for opening in openings if opening.kind == "WINDOW"]
    faces, wall_extended_pieces, wall_extended_length = extend_faces_for_window_gaps(faces, windows, wall_evidence | vertical.full, grid, units_per_metre, passable=free_space | closed_floor)
    enriched: list[Opening] = []
    for opening in openings:
        if opening.kind != "DOOR" or (opening.confidence is not None and opening.leaf_detected and opening.hinge_side is not None):
            enriched.append(opening)  # windows, passages and doors whose open leaf the 3D refinement measured
            continue
        leaf = detect_door_leaf(opening, vertical, grid, ceiling_height, units_per_metre)
        enriched.append(replace(opening, leaf_detected=leaf is not None) if leaf is None else replace(opening, leaf_detected=True, leaf_angle_deg=leaf.angle_deg, leaf_direction_deg=leaf.direction_angle_deg, leaf_mode=leaf.mode, hinge_side=leaf.hinge_side, swing_side=leaf.swing_side))
    openings = enriched
    door_leaf_lines = _door_leaf_lines(openings)
    faces = break_faces_at_openings(faces, openings, 0.15 * units_per_metre)
    output_layers["A-WALL"].extend(face_entities(faces))
    # Room outlines and names are written after fixtures and furniture are known
    # (semantic names such as Bad / Küche need them).
    sorted_rooms = sorted(rooms_data.polygons, key=lambda item: (-item.representative_point().y, item.representative_point().x))
    face_geometry_for_metrics = unary_union([face.geometry() for face in faces]) if faces else LineString()
    wall_evidence_coverage = float(np.mean([face_geometry_for_metrics.distance(Point(point[:2])) <= 0.05 * units_per_metre for point in wall_points])) if len(wall_points) else 0.0
    opening_symbols = opening_entities(openings, free_space, grid=grid)
    for entity in opening_symbols:
        if (entity.get("type") == "insert" and entity.get("name") == "WINDOW") or entity.get("_opening_kind") == "WINDOW":
            entity["_on_face"] = True
            output_layers["A-WIND"].append(entity)
        else:
            entity["_on_face"] = True
            output_layers["A-DOOR"].append(entity)

    report_progress(progress, 0.54, "Extracting stairs and fixtures")
    stair_layer = next((layer for layer in indoor_layers if layer.name == "A-STAI"), None)
    if stair_layer is not None:
        output_layers["A-STAI"].extend(
            stair_entities(np.isin(classes, stair_layer.codes), grid, min_area=stair_layer.min_area * units_per_metre**2, tread_depth=0.28 * units_per_metre)
        )
    cyclone_codes = load_cyclone_codes()
    wall_lines = [face.line for face in faces if getattr(face, "kind", "") == "line" and getattr(face, "line", None) is not None]
    theta = _indoor_wall_axis(wall_lines)
    view_count = cv2.imread(str(ortho.view_count_path), cv2.IMREAD_UNCHANGED) if ortho.view_count_path.is_file() else None
    if view_count is not None and view_count.shape != classes.shape:
        view_count = None
    outer_band = cv2.dilate(wall_evidence.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(round(0.10 * units_per_metre / grid.pixel_size))) + 1,) * 2)).astype(bool)
    # Preserve the established fixture path.  It has reliable dimension-based
    # sanitation recognition even when SAM did not assign a fine fixture label.
    legacy_fixture_mask = (dsm - level.floor_z >= 0.05 * units_per_metre) & (np.isin(classes, furniture_codes) | (fine_classes > 0)) & (classes != 180) & ~wall_evidence & ~outer_band
    fixture_coverage = np.zeros_like(legacy_fixture_mask, dtype=bool)
    sanitary_codes = next((layer.codes for layer in furniture_layers if layer.name == "P-SANI"), ())
    fixtures_rejected: dict[str, int] = {}
    catalogue_inserts: list[dict[str, object]] = []
    kitchen_islands = 0
    if instance_raster is not None and instance_labels:
        for placed, sani_mask, sani_slice, _sani_id in _sanitary_instance_inserts(instance_raster, instance_labels, grid, wall_lines, units_per_metre):
            fixture_coverage[sani_slice] |= sani_mask[sani_slice]
            if _insert_duplicates_existing(placed, output_layers):
                continue
            output_layers[str(placed["_catalogue"]["layer"]) if "_catalogue" in placed else "P-SANI"].append(placed)  # type: ignore[index]
            if "_catalogue" in placed:
                catalogue_inserts.append(placed)
    for component, component_slice, fine_label, code in _furniture_instances(legacy_fixture_mask, classes, dsm, level.floor_z, fine_classes, fine_confidence, fine_names, grid, units_per_metre):
        local_grid = _grid_for_slice(grid, component_slice)
        display = _furniture_display_name(fine_label, cyclone_codes.get(code, {}))
        insert = fit_block_insert(component, local_grid, "FURNITURE", display, units_per_metre=units_per_metre)
        height = float(np.nanmax(dsm[component_slice][component]) - level.floor_z)
        fixture, aligned, rejected = _fixture_candidate(code, insert, height, fine_label, units_per_metre, faces, sanitary_codes, classes, fine_classes, fine_names, grid, allow_coarse_sani=True)
        if fixture == "FURNITURE":
            if rejected:
                fixtures_rejected[rejected] = fixtures_rejected.get(rejected, 0) + 1
            continue
        if fixture in {"STOVE", "SINK"}:
            continue  # kitchen appliances are placed on the kitchen run, only where a label was observed
        legacy_instance_id = _dominant_instance(instance_raster[component_slice][component]) if instance_raster is not None else 0
        if instance_raster is not None and instance_labels and legacy_instance_id == 0:
            legacy_instance_id = _dominant_instance_nearby(instance_raster, component_slice, component, grid.pixel_size, 0.15 * units_per_metre)
        if instance_raster is not None and instance_labels and legacy_instance_id > 0 and instance_labels.get(legacy_instance_id):
            # The dominant instance is labelled: sanitary ones were placed from the instance footprint above,
            # everything else (window, door, furniture) must never receive a sanitary block.
            continue
        placed = _catalogue_fixture(fixture, insert, wall_lines, units_per_metre)
        if placed is None and fixture in _FIXTURE_KINDS:
            continue
        if instance_raster is not None:
            display = _fixture_display_name(display, fixture, instance_labels.get(legacy_instance_id))
        if placed is not None:
            placed["attributes"] = {"LABEL": display}
            if legacy_instance_id > 0:
                placed["attributes"]["INSTANCE"] = str(legacy_instance_id)
            placed["_wall_aligned"] = aligned
            if _insert_duplicates_existing(placed, output_layers):
                fixture_coverage[component_slice] |= component
                continue
            output_layers[str(placed["_catalogue"]["layer"])].append(placed)  # type: ignore[index]
            catalogue_inserts.append(placed)
        else:
            insert["name"] = fixture
            insert["attributes"] = {"LABEL": display}
            if legacy_instance_id > 0:
                insert["attributes"]["INSTANCE"] = str(legacy_instance_id)
            insert["_wall_aligned"] = aligned
            output_layers["P-SANI" if code in sanitary_codes else "A-FURN"].append(insert)
        fixture_coverage[component_slice] |= component
    objects_dsm = dsm
    depth_filled_cells = 0
    if plan_depth is not None and plan_depth.settings.get("plan_height_fill", True):
        # Unmeasured cells (no point above the floor) take the COLMAP-supported furniture top of the depth product; measured cells keep the cloud.
        depth_top = plan_depth.top_height_map(grid, z_low=0.05 * units_per_metre, z_high=0.97 * ceiling_height)
        unmeasured = (~np.isfinite(dsm) | (dsm - level.floor_z < 0.05 * units_per_metre)) & np.isfinite(depth_top)
        # Only holes of / beside objects the cloud measured: a depth-only island must not become a new object.
        measured_objects = np.isfinite(dsm) & (dsm - level.floor_z >= 0.05 * units_per_metre)
        reach = max(1, int(round(plan_depth.settings.get("plan_height_fill_reach_m", 0.15) * units_per_metre / grid.pixel_size)))
        near_objects = cv2.dilate(measured_objects.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))).astype(bool)
        unmeasured &= near_objects
        depth_filled_cells = int(unmeasured.sum())
        if depth_filled_cells:
            objects_dsm = dsm.copy()
            objects_dsm[unmeasured] = (depth_top[unmeasured] + level.floor_z).astype(np.float32)
    instances, footprint_metrics, height_map = object_instances(
        objects_dsm, level.floor_z, classes, fine_classes, fine_confidence, fine_names, grid, units_per_metre, ceiling_height, wall_lines,
        free_space | closed_floor, outer_band | wall_evidence, view_count, instance_raster, instance_labels,
    )
    footprint_metrics["depth_in_plan"] = plan_depth is not None
    footprint_metrics["depth_height_cells_filled"] = depth_filled_cells
    labelled_instances = [item for item in instances if item.has_fine_label]
    instances = [
        item for item in instances
        if item.has_fine_label or not any(
            float(np.count_nonzero(item.mask & owner.mask)) / max(float(item.mask.sum()), 1.0) > 0.40
            or bool(owner.mask[int(round(np.nonzero(item.mask)[0].mean())), int(round(np.nonzero(item.mask)[1].mean()))])
            for owner in labelled_instances
        )
    ]
    wall_polylines = [LineString(entity["points"]) for entity in output_layers["A-WALL"] if entity.get("type") == "polyline" and len(entity.get("points", [])) >= 2]
    room_union = unary_union(list(rooms_data.polygons)) if rooms_data.polygons else None
    furniture_wall_band = wall_band_of(wall_polylines, units_per_metre)
    instances = refine_dining_instances(instances, classes, grid, height_map, theta, units_per_metre, footprint_metrics, wall_polylines, room_union)
    footprint_kinds: dict[str, int] = {}
    footprint_dev90_max = 0.0
    free_footprint_labels: list[str] = []
    kitchen_einb_from_fine = 0
    alignment_names = {"bed", "wardrobe", "washing machine", "refrigerator", "fridge", "desk", "shelf"}
    fine_kitchen_labels = {"kitchen counter", "kitchen cabinet"}
    pending_kitchen: list[tuple[Any, str, str]] = []
    furniture_id = 0  # links a furniture outline / insert with its "Name h=.." text (plan polish moves or drops both)
    furniture_evidence: dict[int, Polygon] = {}  # traced footprint per outline: the plan polish picks its shape against it
    traced_chairs = [item.polygon for item in instances if str(item.label or "").strip().lower() == "chair" and item.polygon is not None]
    kitchen_zone = _kitchen_zone(rooms_data.polygons, fine_classes, fine_names, grid, catalogue_inserts, units_per_metre, plan_settings.indoor_island.zone_radius_m)
    for instance in instances:
        if float(np.mean(fixture_coverage[instance.mask])) >= 0.50:
            continue
        display = _furniture_display_name(instance.label, cyclone_codes.get(instance.code, {}))
        component_slice = _mask_slice(instance.mask)
        component = instance.mask[component_slice]
        local_grid = _grid_for_slice(grid, component_slice)
        island = detect_kitchen_island(instance, grid, theta, units_per_metre, wall_lines, kitchen_zone, plan_settings.indoor_island)
        if island is not None:
            output_layers["A-EINB"].extend([
                {"type": "polyline", "points": _polygon_points(island.polygon), "closed": True, "holes": [], "_einb_kind": "island", "_island_length_m": island.length_m, "_island_depth_m": island.depth_m, "_island_reason": island.reason},
                {"type": "text", "text": "Kücheninsel", "point": list(island.polygon.centroid.coords[0]), "height": 0.10 * units_per_metre, "rotation": math.degrees(island.angle_rad) % 180.0},
            ])
            for appliance in _island_appliances(island, fine_classes, fine_names, grid, units_per_metre):
                if not _insert_duplicates_existing(appliance, output_layers):
                    output_layers["A-EINB"].append(appliance)
                    catalogue_inserts.append(appliance)
            kitchen_islands += 1
            continue
        label_key = str(instance.label or "").strip().lower()
        if label_key in fine_kitchen_labels or label_key in _KITCHEN_APPLIANCE_KINDS:
            # Counters and labelled appliances form one kitchen run (built after the loop).
            pending_kitchen.append((instance, label_key, display))
            continue
        insert = fit_block_insert(component, local_grid, "FURNITURE", display, units_per_metre=units_per_metre)
        fixture, aligned, rejected = _fixture_candidate(instance.code, insert, instance.height, instance.label, units_per_metre, faces, sanitary_codes, classes, fine_classes, fine_names, grid)
        owner_id = instance.instance_id
        if fixture != "FURNITURE" and instance_raster is not None and instance_labels and owner_id <= 0:
            owner_id = _dominant_instance_nearby(instance_raster, component_slice, component, grid.pixel_size, 0.15 * units_per_metre)
        if fixture != "FURNITURE" and instance_raster is not None and instance_labels and owner_id > 0 and instance_labels.get(owner_id):
            if _sanitary_instance_kind(instance_labels.get(owner_id)) is not None:
                continue  # placed from its instance footprint in the sanitary pre-pass
            fixture = "FURNITURE"
        if fixture != "FURNITURE":
            if instance_raster is not None:
                display = _fixture_display_name(display, fixture, None)
            placed = _catalogue_fixture(fixture, insert, wall_lines, units_per_metre)
            if placed is None and fixture in _FIXTURE_KINDS:
                fixture = "FURNITURE"
            elif placed is not None:
                placed["attributes"] = {"LABEL": display}
                if instance.instance_id > 0:
                    placed["attributes"]["INSTANCE"] = str(instance.instance_id)
                placed["_wall_aligned"] = aligned
                if not _insert_duplicates_existing(placed, output_layers):
                    output_layers[str(placed["_catalogue"]["layer"])].append(placed)  # type: ignore[index]
                    catalogue_inserts.append(placed)
                continue
            else:
                insert["name"] = fixture
                insert["attributes"] = {"LABEL": display}
                if instance.instance_id > 0:
                    insert["attributes"]["INSTANCE"] = str(instance.instance_id)
                insert["_wall_aligned"] = aligned
                output_layers["P-SANI" if instance.code in sanitary_codes else "A-FURN"].append(insert)
                continue
        if rejected:
            fixtures_rejected[rejected] = fixtures_rejected.get(rejected, 0) + 1
        area = float(instance.mask.sum()) * grid.pixel_size**2
        touches_wall = any(line.distance(Point(insert["point"])) <= 0.15 * units_per_metre for line in wall_lines)
        kitchen_counter = not instance.has_fine_label and ((0.75 * units_per_metre <= instance.height <= 1.0 * units_per_metre and area >= 0.4 * units_per_metre**2 and touches_wall) or (instance.solidity >= 0.6 and area >= 0.3 * units_per_metre**2))
        clutter = not instance.has_fine_label and not kitchen_counter and (instance.code == 180 or instance.solidity < 0.6 or instance.height < 0.25 * units_per_metre)
        if clutter and area < 0.15 * units_per_metre**2:
            continue
        if instance.polygon is not None:
            fit = FootprintFit("rect", instance.polygon, 0.0, 1.0)
        else:
            fit = fit_footprint(instance.mask, grid, height_map, theta, units_per_metre, fine_label=instance.has_fine_label, semantic_label=instance.label)
        if fit is None:
            continue
        if clutter:
            display = "Objekt"
            rectangle = clutter_rectangle(instance.mask, grid, theta, units_per_metre, free_space | closed_floor)
            if rectangle is None:
                continue
            footprint_metrics["footprints_clutter_rectangles"] += 1
            fit = FootprintFit("rect", rectangle, fit.dev90, fit.iou)
        aligned = (
            fit.kind in {"rect", "rect_free"}
            and (str(instance.label or "").strip().lower() in alignment_names or str(instance.label or "").strip().lower() == "table")
            and _align_insert_to_wall(insert, faces, 1.0 * units_per_metre, 12.0, 0.05 * units_per_metre)
        )
        if instance.polygon is None:
            # Furniture never crosses a wall line; it is clipped to the room only when it mostly lies inside one.
            mostly_inside = room_union is not None and fit.polygon.intersection(room_union).area >= 0.5 * fit.polygon.area
            clipped_polygon = clip_to_room(fit.polygon, furniture_wall_band, room_union if mostly_inside else None, units_per_metre)
            if clipped_polygon is None:
                continue
            if clipped_polygon is not fit.polygon:
                was_rectangle = len(fit.polygon.exterior.coords) - 1 == 4
                refit = refit_clipped(clipped_polygon, fit.polygon, wall_polylines, units_per_metre, force_rectangle=was_rectangle)
                if refit is None:
                    continue
                fit = FootprintFit("rect" if was_rectangle else fit.kind, refit, fit.dev90, fit.iou, fit.radius)
        metric_height = instance.height / max(units_per_metre, 1e-9)
        # A table with traced chairs keeps its own rectangle: the dining block would draw stock chairs instead.
        own_chairs = any(chair.distance(fit.polygon) <= 0.30 * units_per_metre for chair in traced_chairs)
        furniture_insert = _furniture_catalogue_insert(str(instance.label or "").strip().lower(), fit, wall_lines, units_per_metre) if fit.kind in {"rect", "rect_free"} and not clutter and not own_chairs else None
        if furniture_insert is not None and not _insert_duplicates_existing(furniture_insert, output_layers):
            furniture_insert["attributes"] = {"LABEL": display}
            insert_parts = furniture_solid_parts(instance, height_map, grid, units_per_metre, fit.polygon, plan_settings.indoor_furniture_solids)
            if insert_parts:
                furniture_insert["_solid_parts"] = [
                    {"footprint": part["footprint"], "z_min": float(level.floor_z), "z_max": float(level.floor_z) + float(part["z_max_m"]) * units_per_metre}
                    for part in insert_parts
                ]
            if instance.instance_id > 0:
                furniture_insert["attributes"]["INSTANCE"] = str(instance.instance_id)
            furniture_id += 1
            furniture_insert["_furn_id"] = furniture_id
            output_layers["A-FURN"].append(furniture_insert)
            catalogue_inserts.append(furniture_insert)
            output_layers["A-ANNO"].append({
                "type": "text", "text": _furniture_label_text(display, label_key, metric_height), "point": list(fit.polygon.representative_point().coords[0]),
                "height": (0.08 if fit.polygon.area < units_per_metre**2 else 0.12) * units_per_metre, "rotation": 0.0, "_furn_id": furniture_id,
            })
            continue
        footprint_kinds[fit.kind] = footprint_kinds.get(fit.kind, 0) + 1
        if fit.kind == "free":
            free_footprint_labels.append(display)
        footprint_dev90_max = max(footprint_dev90_max, fit.dev90)
        furniture_id += 1
        solid_parts = furniture_solid_parts(instance, height_map, grid, units_per_metre, fit.polygon, plan_settings.indoor_furniture_solids)
        output_layers["A-FURN"].append({
            "type": "polyline", "points": _polygon_points(fit.polygon), "closed": True, "holes": [], "_furn_id": furniture_id,
            "_hausdorff": fit.dev90, "_label": display, "_footprint_kind": fit.kind,
            "_footprint_dev90": fit.dev90, "_wall_aligned": aligned, "_semantic_label": str(instance.label or display),
            **({"_instance_id": int(instance.instance_id)} if instance.instance_id > 0 else {}),
            **({"_solid_parts": [
                {"footprint": part["footprint"], "z_min": float(level.floor_z), "z_max": float(level.floor_z) + float(part["z_max_m"]) * units_per_metre}
                for part in solid_parts
            ]} if solid_parts else {}),
        })
        if not clutter:
            measured = _footprint_evidence(component, local_grid, units_per_metre)
            if measured is not None:
                furniture_evidence[furniture_id] = measured
        detail_labels = {"sofa", "armchair", "bed", "table", "desk", "kitchen counter", "wardrobe", "shelf", "bathtub"}
        if str(instance.label or "").strip().lower() in detail_labels:
            for line in height_step_details(instance, height_map, grid, theta, units_per_metre, fit.polygon):
                output_layers["A-FURN-DETL"].append({"type": "polyline", "points": [[float(x), float(y)] for x, y in line.coords], "closed": False})
        metric_height = instance.height / max(units_per_metre, 1e-9)
        output_layers["A-ANNO"].append({
            "type": "text", "text": _furniture_label_text(display, label_key, metric_height), "point": list(fit.polygon.representative_point().coords[0]),
            "height": (0.08 if fit.polygon.area < units_per_metre**2 else 0.12) * units_per_metre, "rotation": 0.0, "_furn_id": furniture_id,
        })

    kitchen_einb_from_fine += _emit_kitchen_run(
        pending_kitchen, output_layers, catalogue_inserts, grid, height_map, theta, wall_lines, units_per_metre,
        footprint_metrics, plan_settings.indoor_kitchen_run,
    )
    footprint_metrics["kitchen_einb_from_fine"] = kitchen_einb_from_fine
    footprint_metrics["kitchen_islands"] = kitchen_islands
    _absorb_chairs_at_dining_tables(output_layers, units_per_metre)
    furniture_polygons = [Polygon(entity["points"]) for entity in output_layers["A-FURN"] if entity.get("type") == "polyline" and entity.get("closed") and len(entity.get("points", [])) >= 3]
    counter_top = (height_map >= 0.70 * units_per_metre) & (height_map <= 1.05 * units_per_metre) & (vertical.full | vertical.counter)
    kitchen_names = {"sink", "stove", "refrigerator", "fridge", "kitchen", "washbasin", "toilet", "wc", "bathtub", "shower"}
    kitchen_ids = [index + 1 for index, name in enumerate(fine_names) if str(name).strip().lower() in kitchen_names]
    kitchen_evidence = np.isin(fine_classes, kitchen_ids) | np.isin(classes, sanitary_codes)
    einb_entities, einb_rejected, einb_polygons = extract_builtins(
        vertical, grid, theta, wall_lines, wall_evidence, furniture_polygons, counter_top, units_per_metre,
        kitchen_evidence=kitchen_evidence, settings=plan_settings.indoor_vertical,
    )
    einb_entities, door_leaf_rejected = _exclude_door_leaves_from_builtins(einb_entities, door_leaf_lines, units_per_metre)
    einb_rejected["door_leaf"] = einb_rejected.get("door_leaf", 0) + door_leaf_rejected
    einb_entities, einb_polygons, einb_room_rejected = filter_builtins_to_rooms(
        einb_entities, rooms_data.polygons, wall_lines, furniture_polygons, free_space | closed_floor, grid, units_per_metre,
    )
    for reason, count in einb_room_rejected.items():
        einb_rejected[reason] = einb_rejected.get(reason, 0) + count
    if einb_polygons:
        output_layers["A-FURN"] = [
            entity for entity in output_layers["A-FURN"]
            if not (
                entity.get("type") == "polyline" and entity.get("closed") and len(entity.get("points", [])) >= 3
                and any(Polygon(entity["points"]).intersection(polygon).area >= 0.50 * max(Polygon(entity["points"]).area, 1e-9) for polygon in einb_polygons)
            )
        ]
    output_layers["A-EINB"].extend(einb_entities)
    # A wall line that stands at the front of a tall built-in unit is not the wall: move it behind the unit.
    tall_mask = tall_furniture_mask(fine_classes, fine_names, classes, height_map, structural_codes, grid, units_per_metre)
    # A recessed niche (open wardrobe / coat niche) is resolved first: the wall follows the niche and what lies behind its back wall is dropped.
    niche_stats = resolve_recessed_niches(
        output_layers, rooms_data.polygons,
        content=content_mask(tall_mask, height_map, units_per_metre, plan_settings.indoor_niche),
        free_floor=free_space | closed_floor, fine_classes=fine_classes, fine_names=fine_names, grid=grid, units_per_metre=units_per_metre,
        wall_points=wall_points, settings=plan_settings.indoor_niche,
        exclude=kitchen_evidence,
    )
    footprint_metrics["niches"] = niche_stats["niches"]
    footprint_metrics["niche_count"] = niche_stats["niche_count"]
    builtin_front_stats = resolve_builtin_fronts(
        output_layers, rooms_data.polygons,
        tall=tall_mask,
        free_floor=free_space | closed_floor, fine_classes=fine_classes, fine_names=fine_names, grid=grid, units_per_metre=units_per_metre,
        exclude=niche_stats["strips"],
    )
    footprint_metrics["builtin_fronts"] = builtin_front_stats["builtin_fronts"]
    # Upper cabinets: hanging units in the 1.80-2.20 m band above a kitchen run stay dashed on A-OBEN.
    upper_settings = plan_settings.indoor_upper_cabinets
    upper_polygons = detect_upper_cabinets(
        _kitchen_runs(output_layers["A-EINB"]), upper_band_mask(vertical.bitmask, ceiling_height, units_per_metre, upper_settings),
        vertical.full | vertical.counter, wall_evidence, grid, wall_lines, units_per_metre, upper_settings,
    )
    # Kitchen-class points of the cloud (+ COLMAP depth voxels): the measured band gives every upper cabinet its height, and hanging units
    # without a base run below them (above a tall appliance) are found from the points alone.
    upper_points = np.empty((0, 3), dtype=float)
    row_points = np.empty((0, 3), dtype=float)
    instance_points: dict[int, np.ndarray] = {}
    upper_bands: list[tuple[float, float] | None] = [None] * len(upper_polygons)
    point_uppers: list[tuple[Polygon, tuple[float, float]]] = []
    if plan_settings.indoor_upper_points.enabled:
        all_points, all_classes, all_instances = load_class_points(point_source or _find_point_source(project, ortho.stats), frame, level.floor_z, min_z=0.0)
        high_points = all_points[:, 2] >= 0.9 * units_per_metre if len(all_points) else np.zeros(0, dtype=bool)
        cloud_points, cloud_classes, _cloud_instances = all_points[high_points], all_classes[high_points], all_instances[high_points]
        row_points = all_points
        # Hanging kitchen units are class 18 (appliances / kitchen); furniture-class shelves and TV units must not become upper cabinets.
        kitchen_cloud = cloud_points[cloud_classes == 18] if len(cloud_points) else cloud_points
        upper_points = kitchen_cloud
        depth_points = None
        kitchen_depth = np.empty((0, 3), dtype=float)
        if plan_depth is not None:
            depth_points, depth_classes = plan_depth.furniture_points(colmap_only=True)
            keep_depth = depth_points[:, 2] >= 0.9 * units_per_metre
            depth_points, depth_classes = depth_points[keep_depth], depth_classes[keep_depth]
            kitchen_depth = depth_points[depth_classes == 18]
            all_depth, _ = plan_depth.furniture_points(colmap_only=True)
            if len(all_depth):
                row_points = np.vstack((row_points, all_depth)) if len(row_points) else all_depth
            upper_points = np.vstack((kitchen_cloud, kitchen_depth)) if len(kitchen_cloud) else kitchen_depth
        upper_bands = [measured_z_band(upper_points, polygon, units_per_metre) for polygon in upper_polygons]
        if plan_settings.indoor_tight_footprint.enabled:
            instance_points = instance_point_sets(cloud_points, _cloud_instances, None if plan_depth is None else depth_points, units_per_metre)
        if kitchen_zone is not None:
            point_uppers = detect_upper_cabinets_from_points(
                upper_points, wall_lines, kitchen_zone.buffer(plan_settings.indoor_door_zone.kitchen_zone_margin_m * units_per_metre), upper_polygons,
                units_per_metre, plan_settings.indoor_upper_points,
            )
    output_layers["A-OBEN"].extend(
        {"type": "polyline", "points": _polygon_points(polygon), "closed": True, "holes": [], "_upper_cabinet": True, **({"_z_range_m": [band[0], band[1]], "_z_source": "points"} if band else {})}
        for polygon, band in zip(upper_polygons, upper_bands, strict=True)
    )
    output_layers["A-OBEN"].extend(
        {"type": "polyline", "points": _polygon_points(polygon), "closed": True, "holes": [], "_upper_cabinet": True, "_z_range_m": [band[0], band[1]], "_z_source": "points", "_upper_from_points": True}
        for polygon, band in point_uppers
    )
    footprint_metrics["upper_cabinets"] = len(upper_polygons) + len(point_uppers)
    footprint_metrics["upper_cabinets_from_points"] = len(point_uppers)
    footprint_metrics["builtin_furniture_absorbed"] = builtin_front_stats["builtin_furniture_absorbed"]
    footprint_metrics.update(drop_misplaced_builtins(output_layers, openings, door_leaf_lines, kitchen_zone, units_per_metre, plan_settings.indoor_door_zone))

    double_walls_done = False
    plan_wall_pieces: list[dict[str, Any]] | None = None
    plan_wall_bodies: list[Polygon] = []
    tall_units: list[tuple[Polygon, str]] = []
    if plan_settings.walls.double_line:
        # Double-line walls: every wall gets both faces with a measured or learned
        # thickness; the room outlines then follow the inner faces.
        report_progress(progress, 0.70, "Building double-line walls")
        # A candidate the refinement classified as no opening still shapes the wall faces like the opening of the first pass did
        # (measured faces are trimmed around it); the wall stage then closes its span (``close_spans``).
        ghosts = [Opening("DOOR", tuple(span["center"]), float(span["width"]), float(span["axis_angle"]), float(span.get("thickness", 0.25 * units_per_metre))) for span in openings_3d_info.get("closed") or ()]
        output_layers["A-WALL"], wall_tidy_stats = tidy_wall_entities(output_layers["A-WALL"], [*openings, *ghosts], units_per_metre)
        footprint_metrics.update(wall_tidy_stats)
        wall_evidence_rasters = WallEvidence(
            wall_bits=vertical.wall_bits, all_bits=vertical.bitmask, bins=plan_settings.indoor_vertical.bins, storey_height=ceiling_height,
            section=section, low=low, floor=free_space, door=door_mask, cameras=_level_camera_centres(project, ortho.stats, frame, level, ceiling_height),
            depth_points=None if plan_depth is None else plan_depth.wall_points(0.30 * ceiling_height, 0.95 * ceiling_height),
        )
        double = build_double_walls(
            output_layers["A-WALL"], openings, rooms_data.polygons, free_space | closed_floor, classes != 0, grid, units_per_metre,
            _wall_settings_for_pass(plan_settings.walls, openings_3d_info),
            furniture=ceiling_furniture_mask | tall_mask, evidence=wall_evidence_rasters,
            builtins=[Polygon(entity["points"]) for entity in output_layers["A-EINB"] if entity.get("type") == "polyline" and entity.get("closed") and entity.get("_einb_kind") == "builtin_front" and len(entity.get("points", [])) >= 3],
            tall_units=tall_unit_evidence(fine_classes, fine_names, classes, height_map, units_per_metre),
            close_spans=[*(openings_3d_info.get("closed") or ()), *(openings_3d_info.get("fills") or ())],
        )
        tall_units = list(double.tall_units)
        plan_wall_pieces, plan_wall_bodies = list(double.pieces), list(double.closed_polygons)
        output_layers["A-WALL"] = double.entities
        output_layers["A-WALL-PATT"] = double.hatches
        footprint_metrics.update(double.stats)
        footprint_metrics["double_wall_openings_changed"] = _apply_opening_changes(output_layers, double.openings_changed, free_space, grid)
        openings = list(double.openings)
        footprint_metrics["double_wall_symbols_updated"] = update_opening_symbols(output_layers, openings, double.opening_slabs, units_per_metre)
        if plan_settings.walls.bound_rooms:
            bounded_rooms, bound_stats = bound_rooms_by_walls(list(sorted_rooms), double.closed_polygons, units_per_metre, plan_settings.walls, barriers=double.opening_cuts)
            sorted_rooms = sorted(bounded_rooms, key=lambda item: (-item.representative_point().y, item.representative_point().x))
            footprint_metrics.update(bound_stats)
        double_walls_done = True
        polish = plan_settings.plan_polish
        if polish.enabled:
            # Clean architectural plan: rooms = wall cells (an open-plan space is one room), regular furniture
            # outlines that never reach into a wall.
            polish_walls = unary_union(list(double.closed_polygons)) if double.closed_polygons else None
            if polish.cell_rooms:
                cell_rooms, cell_stats = rooms_from_wall_cells(
                    list(sorted_rooms), double.closed_polygons, double.opening_cuts, free_space | closed_floor, grid, units_per_metre, theta, polish,
                )
                if cell_rooms:
                    sorted_rooms = sorted(cell_rooms, key=lambda item: (-item.representative_point().y, item.representative_point().x))
                footprint_metrics.update(cell_stats)
                # Rooms go from door to door: the floor no wall cell encloses (hall, strips, pockets behind furniture) is
                # handed to the room it is open to; only walls and door lines split the interior.
                door_rooms, door_stats = door_bounded_rooms(
                    list(sorted_rooms), double.closed_polygons, openings, free_space | closed_floor, grid, units_per_metre, plan_settings.indoor_door_rooms,
                    furniture=ceiling_furniture_mask | tall_mask, support=classes != 0, wall_pieces=double.pieces, theta=theta,
                )
                if door_rooms:
                    sorted_rooms = sorted(door_rooms, key=lambda item: (-item.representative_point().y, item.representative_point().x))
                footprint_metrics.update(door_stats)
            if polish.furniture:
                footprint_metrics.update(regularize_furniture(
                    output_layers, list(sorted_rooms), polish_walls, theta, units_per_metre, polish, _insert_footprint,
                    evidence_of=lambda entity: furniture_evidence.get(entity.get("_furn_id")),
                    height_of=_height_sampler(height_map, grid),
                ))

    report_progress(progress, 0.72, "Labelling enclosed room areas")
    tagged_points = _collect_room_tags(output_layers, units_per_metre)
    room_names = name_rooms(sorted_rooms, tagged_points, units_per_metre, plan_settings.indoor_room_names)
    room_names = name_halls(sorted_rooms, room_names, [(opening.center, opening.axis_angle) for opening in openings if opening.kind in {"DOOR", "passage"}], units_per_metre, plan_settings.indoor_room_names)
    generic_number = 0
    room_name_texts: list[str] = []
    for polygon, decision in zip(sorted_rooms, room_names, strict=True):
        if not decision.keep:
            continue
        generic_number += 1
        title = decision.name or "Zimmer"
        room_name_texts.append(title)
        output_layers["A-AREA"].append({"type": "polyline", "points": _polygon_points(polygon), "closed": True, "holes": []})
        output_layers["A-ANNO"].append({"type": "text", "text": room_label_text(title, decision.area_m2), "point": list(polygon.representative_point().coords[0]), "height": 0.12 * units_per_metre, "rotation": 0.0, "_room_name": title})
    footprint_metrics["room_names"] = room_name_texts
    footprint_metrics["rooms_dropped_tiny"] = sum(not decision.keep for decision in room_names)
    room_count = _add_room_labels(
        output_layers["A-ANNO"], section, grid, plan_settings.label_min_area_indoor * units_per_metre**2,
        units_per_metre=units_per_metre, tagged_points=tagged_points, name_settings=plan_settings.indoor_room_names,
    )
    output_layers["A-EINB"], einb_tidy = tidy_builtins(output_layers["A-EINB"], output_layers["A-ANNO"], units_per_metre)
    footprint_metrics["einb_before_tidy"] = einb_tidy["before"]
    footprint_metrics["einb_after_tidy"] = einb_tidy["after"]
    # Tall built-in kitchen units resolved by the wall stage: outline + cross + KS/HS (after the A-EINB tidy,
    # which would merge two neighbouring units); traced furniture and built-ins inside them are absorbed.
    footprint_metrics["tall_units_absorbed"] = _emit_tall_units(output_layers, tall_units, units_per_metre, _height_sampler(height_map, grid))
    if double_walls_done and plan_settings.plan_polish.enabled:
        kept_rooms = [polygon for polygon, decision in zip(sorted_rooms, room_names, strict=True) if decision.keep]
        # The kitchen run ends at a tall unit like at a wall.
        kitchen_walls = unary_union([polish_walls, *(shape for shape, _ in tall_units)]) if tall_units and polish_walls is not None else polish_walls
        footprint_metrics.update(orient_sanitary_inserts(
            output_layers, polish_walls, units_per_metre,
            SanitaryFacingSettings(
                enabled=plan_settings.plan_polish.sanitary_facing, max_gap_m=plan_settings.plan_polish.sanitary_back_gap_m,
                tie_m=plan_settings.plan_polish.sanitary_tie_m, snap_angle_deg=plan_settings.plan_polish.sanitary_snap_deg,
            ),
        ))
        if plan_settings.plan_polish.kitchen:
            footprint_metrics.update(fit_kitchen_to_walls(output_layers, kept_rooms, kitchen_walls, theta, units_per_metre, plan_settings.plan_polish))
        if plan_settings.plan_polish.kitchen and plan_settings.plan_polish.kitchen_layout:
            footprint_metrics.update(kitchen_cad_layout(
                output_layers, kept_rooms, kitchen_walls, theta, units_per_metre, _height_sampler(height_map, grid),
                _kitchen_appliance_points(fine_classes, fine_names, instance_raster, instance_labels, height_map, grid, units_per_metre),
            ))
        footprint_metrics.update(place_object_labels(output_layers, kept_rooms, units_per_metre, theta, _insert_footprint, plan_settings.plan_polish))
    if not double_walls_done:
        output_layers["A-WALL"], wall_tidy_stats = tidy_wall_entities(output_layers["A-WALL"], openings, units_per_metre)
        footprint_metrics.update(wall_tidy_stats)
    label_obstacles = [footprint for entities in output_layers.values() for entity in entities if entity.get("type") == "insert" and (footprint := _insert_footprint(entity)) is not None]
    if double_walls_done and plan_settings.plan_polish.enabled:
        # Room names also keep clear of traced furniture and built-in outlines.
        label_obstacles += [
            Polygon(entity["points"]).buffer(0) for name in ("A-FURN", "A-EINB") for entity in output_layers.get(name, [])
            if entity.get("type") == "polyline" and entity.get("closed") and len(entity.get("points", [])) >= 3
        ]
    footprint_metrics.update(resolve_label_overlaps(
        output_layers,
        label_obstacles,
        [polygon for polygon, decision in zip(sorted_rooms, room_names, strict=True) if decision.keep],
        units_per_metre, load_label_settings(),
    ))
    if double_walls_done and plan_settings.plan_polish.enabled and plan_settings.plan_polish.kitchen and plan_settings.plan_polish.kitchen_layout:
        # CAD kitchen: the counter is drawn by its front and end lines (its back is the wall face).
        footprint_metrics["kitchen_counter_lines"] = counter_lines(output_layers, kitchen_walls, units_per_metre)
    styles, colors, linetypes, hatches = _layer_presentation(indoor_layers)
    _add_ortho_plan_layer_presentation(styles, colors, linetypes, hatches)
    statuses = {
        name: LayerStatus(name, "ok" if dxf_entity_count(entities) else "empty", None if dxf_entity_count(entities) else "no geometry survived filtering", dxf_entity_count(entities))
        for name, entities in output_layers.items()
    }
    if room_count == 0:
        statuses["A-ANNO"] = LayerStatus("A-ANNO", "empty" if not output_layers["A-ANNO"] else "ok", "walls do not enclose a room" if not output_layers["A-ANNO"] else None, dxf_entity_count(output_layers["A-ANNO"]))
    blocks = sorted({str(entity["name"]) for entities in output_layers.values() for entity in entities if entity.get("type") == "insert"})
    footprint_metrics["catalogue_inserts"] = [
        {"block": str(item["name"]), "kind": item["_catalogue"]["kind"], "layer": item["_catalogue"]["layer"], "wall_snapped": item["_catalogue"]["wall_snapped"], "scale": float(item["xscale"]) / max(units_per_metre, 1e-9)}  # type: ignore[index]
        for item in catalogue_inserts if any(item is entity for entities in output_layers.values() for entity in entities)
    ]
    floor_polygons = [polygon for polygon in mask_to_polygons(closed_floor, grid) if polygon.area >= 0.5 * units_per_metre**2]
    result = DxfPlanResult(
        project_name=project.name, level_name=level.name, mode="indoor", profile_name=(segment.metadata or {}).get("model_profile"),
        scale_factor=frame.scale_factor, frame=frame, settings_fingerprint=clean_plan_fingerprint(project, level, plan_settings),
        layers=output_layers, statuses=statuses, layer_colors=colors,
        background=BackgroundImage(None, None, (grid.min_x, grid.max_y - grid.height * grid.pixel_size, grid.min_x + grid.width * grid.pixel_size, grid.max_y)),
        fingerprint_inputs={"indoor_floor_polygons": [_polygon_points(polygon) for polygon in floor_polygons], "units_per_metre": units_per_metre, "windows_on_interior_walls": 0, "wall_evidence_coverage": wall_evidence_coverage, "ceiling_coverage": float(np.mean(rooms_data.coverage_mask[free_space])) if free_space.any() else 0.0, "rooms": len(rooms_data.polygons), "rooms_skipped": rooms_data.skipped, "ceiling_wall_pieces": len(ceiling_faces), "ceiling_wall_length": float(sum(face.length for face in ceiling_faces) / max(units_per_metre, 1e-9)), "ceiling_edges_rejected": ceiling_edges_rejected, "fine_label_source": fine_label_source, "footprint_kinds": footprint_kinds, "footprint_dev90_max": footprint_dev90_max, "free_footprint_labels": free_footprint_labels, "fixtures_rejected": fixtures_rejected, "einb_rejected": einb_rejected, "doors_total": sum(item.kind == "DOOR" for item in openings), "doors_with_leaf": sum(item.kind == "DOOR" and item.hinge_side is not None for item in openings), "doors_with_leaf_flat": sum(item.kind == "DOOR" and item.leaf_mode == "flat" for item in openings), "doors_opening_only": sum(item.kind == "DOOR" and item.hinge_side is None for item in openings), "passages": sum(item.kind == "passage" for item in openings), "doors_deduplicated": doors_deduplicated, "windows_snapped": len(windows), "wall_extended_pieces": wall_extended_pieces, "wall_extended_length": wall_extended_length / max(units_per_metre, 1e-9), **footprint_metrics},
        layer_linetypes=linetypes, layer_hatches=hatches, layer_styles=styles, blocks=blocks, variant="ortho",
    )
    metrics = indoor_wall_metrics(result)
    statuses["A-WALL"] = LayerStatus(
        "A-WALL", "ok" if output_layers["A-WALL"] else "empty",
        "; ".join(f"{name}={value:.3f}" if isinstance(value, float) else f"{name}={value}" for name, value in {"units_per_metre": units_per_metre, "wall_faces": len(faces), **metrics}.items()),
        dxf_entity_count(output_layers["A-WALL"]),
    )
    result = replace(result, statuses=statuses)
    cache_dir = clean_plan_cache_dir(project, level)
    save_cached_result(result, cache_dir)
    try:
        save_plan_model(build_plan_model(
            project_name=project.name, level=level, frame=frame, units_per_metre=units_per_metre, layers=output_layers, openings=openings,
            rooms=[(polygon, decision.name or "Zimmer") for polygon, decision in zip(sorted_rooms, room_names, strict=True) if decision.keep],
            wall_pieces=plan_wall_pieces, wall_bodies=plan_wall_bodies, result_fingerprint=result.settings_fingerprint,
            instances_path=project / "products" / "classified" / "instances.json", source_extra={"openings_3d": openings_3d_info} if openings_3d_info else None,
            instance_points=instance_points, tight_settings=plan_settings.indoor_tight_footprint, depth=plan_depth, class_points=row_points,
        ), cache_dir)
        _store_pass1_snapshot(cache_dir, openings_3d_info, result.settings_fingerprint)
    except Exception:
        _LOGGER.warning("plan_model.json sidecar could not be written; DXF unaffected; IFC export will be stale", exc_info=True)
    report_progress(progress, 1.0, "Indoor clean DXF plan generation complete")
    return result


def pass1_snapshot_is_current(cache_dir: Path, fingerprint: str) -> bool:
    """True when ``plan_model_pass1.json`` is a PURE pass-1 plan (no refined openings applied) of this clean-plan fingerprint."""
    from samcloud.bim.plan_model import pass1_plan_path

    snapshot = pass1_plan_path(cache_dir)
    if not snapshot.is_file():
        return False
    try:
        source = json.loads(snapshot.read_text(encoding="utf-8")).get("source") or {}
    except (OSError, ValueError):
        return False
    return source.get("result_fingerprint") == fingerprint and not (source.get("openings_3d") or {}).get("applied")


def _store_pass1_snapshot(cache_dir: Path, openings_3d_info: dict[str, Any], fingerprint: str) -> None:
    """Keep the plan of the pass computed WITHOUT refined openings as the stable seed of the opening refinement.

    A pass that applied refined openings never replaces the snapshot (refining again never starts from refined
    openings). Every other pass is a pure pass 1 and becomes the snapshot; the hook only applies refined openings
    while a pure snapshot of the same fingerprint exists, so a snapshot can never contain refined openings.
    """
    from samcloud.bim.plan_model import save_pass1_snapshot

    if openings_3d_info.get("applied") and pass1_snapshot_is_current(cache_dir, fingerprint):
        return
    save_pass1_snapshot(cache_dir)


def _apply_refined_openings(openings: list[Opening], faces: list[WallFace], project: Path, level: Level, units_per_metre: float, fingerprint: str) -> tuple[list[Opening], dict[str, Any]]:
    """Hook: swap the 2D openings for the 3D-refined ones when a matching openings_3d.json exists (never fatal).

    Refined openings are applied only when a pure pass-1 snapshot of the current fingerprint exists; otherwise this pass
    is the unrefined pass 1 (it becomes the snapshot) and the empty sha makes run_all regenerate the plan once the
    refinement has run on that snapshot.
    """
    try:
        from samcloud.bim.openings3d.apply import apply_refined_openings
        from samcloud.bim.openings3d.model import openings_3d_sha256

        cache_directory = clean_plan_cache_dir(project, level)
        if openings_3d_sha256(cache_directory) is not None and not pass1_snapshot_is_current(cache_directory, fingerprint):
            return openings, {"sha256": "", "applied": False, "reason": "no current pass-1 snapshot yet"}
        face_lines = [face.line for face in faces if face.kind == "line" and face.line is not None]
        return apply_refined_openings(openings, cache_directory, face_lines, units_per_metre)
    except Exception:  # noqa: BLE001 - refinement is an optional improvement
        _LOGGER.warning("refined openings could not be applied; using the 2D openings", exc_info=True)
        return openings, {}


def _indoor_scale_and_bands(level: Level, settings: CleanPlanSettings) -> tuple[float, dict[str, tuple[float, float]]]:
    """Return local-units-per-metre and occupancy bands for a level.

    A known ceiling lets local 360 projects use storey-height fractions.  This
    is an approximation for unscaled projects, documented in the wall status.
    """
    if level.ceiling_z is None:
        return 1.0, {"wall": (1.8, 2.35), "section": (0.9, 1.3), "sill": (0.2, 0.8), "lintel": (2.0, 2.35)}
    height = float(level.ceiling_z - level.floor_z)
    if height <= 0.0:
        return 1.0, {"wall": (1.8, 2.35), "section": (0.9, 1.3), "sill": (0.2, 0.8), "lintel": (2.0, 2.35)}
    return height / settings.nominal_storey_height, {
        "wall": (height * settings.indoor_wall_band[0], height * settings.indoor_wall_band[1]),
        "section": (height * settings.indoor_section_band[0], height * settings.indoor_section_band[1]),
        "sill": (height * settings.indoor_sill_band[0], height * settings.indoor_sill_band[1]),
        "lintel": (height * settings.indoor_lintel_band[0], height * settings.indoor_lintel_band[1]),
    }


def _section_occupancy(project: Path, stats: dict[str, Any], point_source: Path | None, frame: Frame, grid: RasterGrid, level: Level, cancel: CancelToken | None, bands: dict[str, tuple[float, float]], units_per_metre: float, vertical_settings: VerticalSettings = VerticalSettings(), depth: PlanDepth | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, VerticalEvidence]:
    """Rasterise section and vertical occupancy in a single point-cloud stream.

    ``depth`` is accepted for API symmetry; the depth product does not enter the section rasters (it refines faces and thickness later,
    in the double-wall stage, where its COLMAP voxels can be tested against measured faces).
    """
    source = point_source or _find_point_source(project, stats)
    wall_counts = np.zeros((grid.height, grid.width), dtype=np.uint16)
    wall_min_z = np.full((grid.height, grid.width), np.inf, dtype=np.float32)
    wall_max_z = np.full((grid.height, grid.width), -np.inf, dtype=np.float32)
    wall_height_bins = np.zeros((grid.height, grid.width), dtype=np.uint16)
    section = np.zeros_like(wall_counts, dtype=bool)
    low = np.zeros_like(section)
    high = np.zeros_like(section)
    measured_points: list[np.ndarray] = []
    vertical_bits = np.zeros((grid.height, grid.width), dtype=np.uint32)
    # Height bins of wall-class (structural) points only: a wall face must be backed by wall points
    # over most of the storey height, a furniture front or upper cabinet is not.
    wall_bits = np.zeros((grid.height, grid.width), dtype=np.uint32)
    ceiling_count = np.zeros((grid.height, grid.width), dtype=np.uint16)
    ceiling_z = np.full((grid.height, grid.width), -np.inf, dtype=np.float32)
    soffit = np.zeros((grid.height, grid.width), dtype=bool)
    height = bands["wall"][1] / 0.94
    if source is None or not source.is_file():
        return section, section, low, high, np.empty((0, 2), dtype=float), _with_ceiling_evidence(vertical_bits, height, vertical_settings, ceiling_count, ceiling_z, soffit)
    structural = np.asarray((100, 103, 104, 108), dtype=np.uint8)
    wall_low, wall_high = 0.30 * height, 0.95 * height
    for chunk in iter_source_chunks(source, 200_000, cancel=cancel, stage="Streaming indoor section points"):
        check_cancel(cancel)
        classes = np.asarray(chunk.classification, dtype=np.uint8)
        keep = np.isin(classes, structural) & ~np.isin(classes, (101, 116))
        local_all = frame.to_local(np.column_stack((chunk.x, chunk.y, chunk.z)))
        z_all = local_all[:, 2] - level.floor_z
        rows_all = np.floor((grid.max_y - local_all[:, 1]) / grid.pixel_size).astype(np.int64)
        cols_all = np.floor((local_all[:, 0] - grid.min_x) / grid.pixel_size).astype(np.int64)
        inside_all = (rows_all >= 0) & (rows_all < grid.height) & (cols_all >= 0) & (cols_all < grid.width)
        vertical_range = inside_all & (z_all >= vertical_settings.floor_fraction * units_per_metre) & (z_all <= vertical_settings.ceiling_fraction * height)
        if vertical_range.any():
            bins = np.clip((z_all[vertical_range] / max(height, 1e-9) * vertical_settings.bins).astype(np.int32), 0, vertical_settings.bins - 1)
            np.bitwise_or.at(vertical_bits, (rows_all[vertical_range], cols_all[vertical_range]), (np.uint32(1) << bins.astype(np.uint32)))
            structural_range = vertical_range & keep
            if structural_range.any():
                wall_bin = np.clip((z_all[structural_range] / max(height, 1e-9) * vertical_settings.bins).astype(np.int32), 0, vertical_settings.bins - 1)
                np.bitwise_or.at(wall_bits, (rows_all[structural_range], cols_all[structural_range]), (np.uint32(1) << wall_bin.astype(np.uint32)))
        ceiling_points = inside_all & (classes == 101) & (z_all >= 0.95 * height) & (z_all <= 1.08 * height)
        if ceiling_points.any():
            np.add.at(ceiling_count, (rows_all[ceiling_points], cols_all[ceiling_points]), 1)
            np.maximum.at(ceiling_z, (rows_all[ceiling_points], cols_all[ceiling_points]), z_all[ceiling_points])
        soffit_points = inside_all & np.isin(classes, (100, 103)) & (z_all >= 0.80 * height) & (z_all <= 0.97 * height)
        if soffit_points.any():
            soffit[rows_all[soffit_points], cols_all[soffit_points]] = True
        if not keep.any():
            continue
        local = local_all[keep]
        z = z_all[keep]
        wall_band = (z >= wall_low) & (z <= wall_high)
        if wall_band.any():
            measured_points.append(np.column_stack((local[wall_band, :2], z[wall_band])))
        rows = rows_all[keep]
        cols = cols_all[keep]
        inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
        rows, cols, z = rows[inside], cols[inside], z[inside]
        wall_band = (z >= wall_low) & (z <= wall_high)
        np.add.at(wall_counts, (rows[wall_band], cols[wall_band]), 1)
        if wall_band.any():
            np.minimum.at(wall_min_z, (rows[wall_band], cols[wall_band]), z[wall_band])
            np.maximum.at(wall_max_z, (rows[wall_band], cols[wall_band]), z[wall_band])
            height_bins = np.clip(np.floor(z[wall_band] / max(0.10 * height, 1e-9)).astype(np.int32), 0, 15)
            np.bitwise_or.at(wall_height_bins, (rows[wall_band], cols[wall_band]), (np.uint16(1) << height_bins).astype(np.uint16))
        section[rows[(z >= bands["section"][0]) & (z <= bands["section"][1])], cols[(z >= bands["section"][0]) & (z <= bands["section"][1])]] = True
        low[rows[(z >= bands["sill"][0]) & (z <= bands["sill"][1])], cols[(z >= bands["sill"][0]) & (z <= bands["sill"][1])]] = True
        high[rows[(z >= bands["lintel"][0]) & (z <= bands["lintel"][1])], cols[(z >= bands["lintel"][0]) & (z <= bands["lintel"][1])]] = True
    spans = wall_max_z - wall_min_z
    bin_counts = np.unpackbits(wall_height_bins.view(np.uint8).reshape(grid.height, grid.width, 2), axis=2).sum(axis=2)
    wall_cells = (wall_counts > 0) & ((spans >= 0.25 * height) | (bin_counts >= 3))
    points = np.vstack(measured_points) if measured_points else np.empty((0, 3), dtype=float)
    return wall_cells, section, low, high, points, _with_ceiling_evidence(vertical_bits, height, vertical_settings, ceiling_count, ceiling_z, soffit, wall_bits)


def _with_ceiling_evidence(bits: np.ndarray, height: float, settings: VerticalSettings, ceiling_count: np.ndarray, ceiling_z: np.ndarray, soffit: np.ndarray, wall_bits: np.ndarray | None = None) -> VerticalEvidence:
    """Attach ceiling rasters (and the wall-class height bins) to the vertical-evidence return value."""
    evidence = vertical_evidence(bits, height, settings)
    return VerticalEvidence(evidence.full, evidence.counter, evidence.top_height, evidence.bitmask, ceiling_count, ceiling_z, soffit, wall_bits)


def _level_camera_centres(project: Path, stats: dict[str, Any], frame: Frame, level: Level, storey_height: float) -> np.ndarray | None:
    """Plan positions of the cameras registered on this level (for the wall visibility test)."""
    model = stats.get("sparse_model_used") or (stats.get("source_paths") or {}).get("sparse_model")
    if not model or not Path(model).exists():
        return None
    try:
        from samcloud.ortho.cameras import load_cameras

        cameras = load_cameras(Path(model), project)
    except Exception:  # noqa: BLE001 - the visibility test is optional evidence
        return None
    if not cameras:
        return None
    centres = frame.to_local(np.asarray([camera.centre for camera in cameras], dtype=np.float64))
    height = centres[:, 2] - float(level.floor_z)
    keep = (height >= -0.2 * storey_height) & (height <= 1.2 * storey_height)
    if not keep.any():
        return None
    return np.unique(np.round(centres[keep, :2], 3), axis=0)


def _apply_opening_changes(layers: dict[str, list[dict[str, Any]]], changes: list[tuple[Any, Any]], free_space: np.ndarray, grid: RasterGrid) -> int:
    """Redraw the symbols of openings the wall builder corrected, removed or added."""
    count = 0
    for old, new in changes:
        if old is not None:
            centre = np.asarray(old.center, dtype=float)
            for layer in ("A-DOOR", "A-WIND"):
                layers[layer] = [
                    entity for entity in layers.get(layer, [])
                    if entity.get("_opening_center") is None or float(np.linalg.norm(np.asarray(entity["_opening_center"][:2], dtype=float) - centre)) > 1e-6
                ]
        if new is not None:
            for entity in opening_entities([new], free_space, grid=grid):
                entity["_on_face"] = True
                layers["A-WIND" if (entity.get("type") == "insert" and entity.get("name") == "WINDOW") or entity.get("_opening_kind") == "WINDOW" else "A-DOOR"].append(entity)
        count += 1
    return count


def _indoor_structural_codes(layers: tuple[LayerDefinition, ...]) -> tuple[int, ...]:
    return tuple(code for layer in layers if layer.name in {"A-WALL", "A-DOOR", "A-WIND", "A-COLS"} for code in layer.codes)


def _load_fine_labels(segment: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Load point-cloud fine labels first, retaining ortho SAM as the compatibility fallback."""
    point_paths = (
        Path(segment.output_dir) / "fine_pc.npy",
        Path(segment.output_dir) / "fine_pc_conf.npy",
        Path(segment.output_dir) / "fine_pc_labels.json",
    )
    point = _read_fine_label_rasters(*point_paths, shape)
    if point is not None and np.any(point[0]):
        labels, confidence, names = point
        return labels, confidence, names, "pointcloud"
    if segment.fine_classes_path is None or segment.fine_confidence_path is None or segment.fine_labels_path is None:
        return np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.float32), [], "none"
    ortho = _read_fine_label_rasters(segment.fine_classes_path, segment.fine_confidence_path, segment.fine_labels_path, shape)
    if ortho is None:
        return np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.float32), [], "none"
    labels, confidence, names = ortho
    return labels, confidence, names, "ortho_sam" if np.any(labels) else "none"


def _load_instance_raster(segment: Any, shape: tuple[int, int]) -> np.ndarray | None:
    """Load the optional point-cloud instance raster without changing legacy jobs."""
    path = Path(segment.output_dir) / "instance_pc.npy"
    if not path.is_file():
        return None
    try:
        values = np.load(path, allow_pickle=False).astype(np.uint32, copy=False)
    except (OSError, ValueError, TypeError):
        return None
    return values if values.shape == shape else None


def _load_instance_labels(project: Path, point_source: Path | None) -> dict[int, str]:
    """Read canonical per-instance labels from instances.json; a missing or malformed file gives no labels."""
    candidates = [Path(project) / "products" / "classified" / "instances.json"]
    if point_source is not None:
        candidates.append(Path(point_source).with_name("instances.json"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            records = json.loads(path.read_text(encoding="utf-8")).get("instances", [])
            return {int(item["id"]): str(item.get("label", "")).strip().lower() for item in records if isinstance(item, dict) and int(item.get("id", 0)) > 0}
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return {}


def _read_fine_label_rasters(labels_path: Path, confidence_path: Path, names_path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    if not all(path.is_file() for path in (labels_path, confidence_path, names_path)):
        return None
    try:
        labels = np.load(labels_path, allow_pickle=False).astype(np.uint8, copy=False)
        confidence = np.load(confidence_path, allow_pickle=False).astype(np.float32, copy=False)
        payload = json.loads(names_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if labels.shape != shape or confidence.shape != shape:
        return None
    if isinstance(payload, dict):
        raw_names = payload.get("labels", [])
        aliases = {str(key).strip().lower(): str(value).strip().lower() for key, value in dict(payload.get("aliases", {}) or {}).items()}
    elif isinstance(payload, list):
        raw_names, aliases = payload, {}
    else:
        return None
    names = [_canonical_fine_label(name, aliases) for name in raw_names]
    if int(labels.max(initial=0)) > len(names):
        return None
    return labels, confidence, names


def _canonical_fine_label(value: object, aliases: dict[str, str]) -> str:
    name = str(value).strip().lower()
    seen: set[str] = set()
    while name in aliases and name not in seen:
        seen.add(name)
        name = aliases[name]
    return name


def _majority_fine_label(component: np.ndarray, fine_classes: np.ndarray, fine_confidence: np.ndarray, names: list[str]) -> str | None:
    eligible = component & (fine_confidence >= 0.4) & (fine_classes > 0)
    if not eligible.any():
        return None
    identifier = int(np.bincount(fine_classes[eligible], minlength=len(names) + 1).argmax())
    return names[identifier - 1] if 1 <= identifier <= len(names) else None


FINE_LABEL_GERMAN_NAMES: dict[str, str] = {
    "sofa": "Sofa", "couch": "Sofa", "armchair": "Sessel", "bed": "Bett", "mattress": "Bett",
    "nightstand": "Nachttisch", "table": "Tisch", "coffee table": "Couchtisch", "dining table": "Esstisch",
    "desk": "Schreibtisch", "chair": "Stuhl", "wardrobe": "Schrank", "chest of drawers": "Kommode",
    "shelf": "Regal", "cabinet": "Schrank", "bookcase": "Bücherregal", "sideboard": "Sideboard", "tv stand": "Sideboard",
    "television": "TV", "cat tree": "Kratzbaum", "plant": "Pflanze", "floor lamp": "Stehlampe",
    "cushion": "Kissen", "pillow": "Kissen", "blanket": "Decke", "duvet": "Decke", "clothes": "Kleidung",
    "refrigerator": "Kühlschrank", "washing machine": "Waschmaschine", "kitchen counter": "Küche",
    "kitchen cabinet": "Küche", "stove": "Herd", "kitchen sink": "Spüle", "toilet": "WC",
    "washbasin": "Lavabo", "bathtub": "Badewanne", "shower tray": "Dusche", "radiator": "Heizkörper",
}


_FIXTURE_KIND_GERMAN = {"WC": "WC", "WASHBASIN": "Lavabo", "BATHTUB": "Badewanne", "SHOWER": "Dusche"}


def _dominant_instance(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.uint32)
    values = values[values > 0]
    if len(values) == 0:
        return 0
    ids, counts = np.unique(values, return_counts=True)
    return int(ids[int(np.argmax(counts))])


_SANITARY_INSTANCE_KINDS = {"bathtub": "BATHTUB", "bath tub": "BATHTUB", "washbasin": "WASHBASIN", "wash basin": "WASHBASIN", "toilet": "WC", "wc": "WC", "shower tray": "SHOWER", "shower": "SHOWER"}


def _sanitary_instance_kind(label: str | None) -> str | None:
    """Catalogue fixture kind for a canonical sanitary instance label, else None."""
    return _SANITARY_INSTANCE_KINDS.get(str(label or "").strip().lower())


def _sanitary_instance_inserts(
    instance_raster: np.ndarray, instance_labels: dict[int, str], grid: RasterGrid, wall_lines: list[LineString], units_per_metre: float,
    *, min_area_m2: float = 0.08,
) -> list[tuple[dict[str, Any], np.ndarray, tuple[slice, slice], int]]:
    """One catalogue block per sanitary instance group (touching same-kind instances merge), sized from the instance footprint."""
    results: list[tuple[dict[str, Any], np.ndarray, tuple[slice, slice], int]] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for kind in ("BATHTUB", "WASHBASIN", "WC", "SHOWER"):
        ids = sorted(identifier for identifier, label in instance_labels.items() if _sanitary_instance_kind(label) == kind)
        present = [identifier for identifier in ids if np.any(instance_raster == identifier)]
        if not present:
            continue
        union = np.isin(instance_raster, present)
        count, components = cv2.connectedComponents(cv2.dilate(union.astype(np.uint8), kernel), connectivity=8)
        for index in range(1, count):
            mask = (components == index) & union
            if float(mask.sum()) * grid.pixel_size**2 < min_area_m2 * units_per_metre**2:
                continue
            member_ids = np.unique(instance_raster[mask])
            member_ids = member_ids[member_ids > 0]
            dominant = _dominant_instance(instance_raster[mask])
            component_slice = _mask_slice(mask)
            component = mask[component_slice]
            display = _FIXTURE_KIND_GERMAN.get(kind, kind)
            insert = fit_block_insert(component, _grid_for_slice(grid, component_slice), "FURNITURE", display, units_per_metre=units_per_metre)
            placed = _catalogue_fixture(kind, insert, wall_lines, units_per_metre)
            if placed is None:
                insert["name"] = kind
                placed = insert
            placed["attributes"] = {"LABEL": display, "INSTANCE": str(dominant)}
            placed["_wall_aligned"] = False
            results.append((placed, mask, component_slice, dominant))
    return results


def _dominant_instance_nearby(instance_raster: np.ndarray, component_slice: tuple[slice, slice], component: np.ndarray, pixel_size: float, radius: float) -> int:
    """Dominant instance within ``radius`` of a component that itself carries no instance pixels (0 when none)."""
    pad = max(1, int(math.ceil(radius / max(pixel_size, 1e-9))))
    rows, cols = component_slice
    r0, r1 = max(0, rows.start - pad), min(instance_raster.shape[0], rows.stop + pad)
    c0, c1 = max(0, cols.start - pad), min(instance_raster.shape[1], cols.stop + pad)
    padded = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    padded[rows.start - r0: rows.stop - r0, cols.start - c0: cols.stop - c0] = component
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    near = cv2.dilate(padded.astype(np.uint8), kernel).astype(bool)
    return _dominant_instance(instance_raster[r0:r1, c0:c1][near])


def _fixture_display_name(display: str, fixture: str, instance_label: str | None) -> str:
    """Specific German name for a sanitary/catalogue insert whose fine label was missing (only with instance data)."""
    if display != "Möbel":
        return display
    label = str(instance_label or "").strip().lower()
    if label and label not in _NON_FOOTPRINT_INSTANCE_LABELS:
        named = _furniture_display_name(label, {})
        if named != "Möbel":
            return named
    return _FIXTURE_KIND_GERMAN.get(fixture, display)


def _furniture_display_name(fine_label: str | None, class_info: dict[str, Any]) -> str:
    """Use German semantic names for furniture and never expose Cyclone custom codes."""
    label = str(fine_label or "").strip().lower()
    if label:
        return FINE_LABEL_GERMAN_NAMES.get(label, class_display_name(label))
    return "Möbel"


def _fixture_candidate(code: int, insert: dict[str, object], local_height: float, fine_label: str | None, units_per_metre: float, faces: list[Any], sanitary_codes: tuple[int, ...], classes: np.ndarray, fine_classes: np.ndarray, fine_names: list[str], grid: RasterGrid, *, allow_coarse_sani: bool = False) -> tuple[str, bool, str | None]:
    """Accept fixture inserts only when their metric and spatial evidence agrees."""
    metric_length = float(insert["xscale"]) / max(units_per_metre, 1e-9)
    metric_width = float(insert["yscale"]) / max(units_per_metre, 1e-9)
    metric_height = float(local_height) / max(units_per_metre, 1e-9)
    fixture = guess_fixture(code, metric_length, metric_width, metric_height, fine_label)
    if fixture == "FURNITURE":
        # A single P-SANI connected component can include the adjacent wall or
        # counter return.  Its coarse semantic is explicit, so retain one
        # wall-aligned sanitary insert instead of recasting it as furniture.
        if allow_coarse_sani and code in sanitary_codes and not fine_label:
            aligned = _align_insert_to_wall(insert, faces, 1.0 * units_per_metre, 12.0, 0.05 * units_per_metre)
            if aligned:
                return "BATHTUB", True, None
        return fixture, False, None
    aliases = {"toilet", "wc", "bathtub", "washbasin", "shower", "stove", "sink", "radiator", "column"}
    is_dimension_guess = not (fine_label and fine_label.strip().lower() in aliases)
    if not is_dimension_guess:
        return fixture, _align_insert_to_wall(insert, faces, 1.0 * units_per_metre, 12.0, 0.05 * units_per_metre), None
    if code not in sanitary_codes and not _near_fixture_evidence(insert["point"], classes, fine_classes, fine_names, sanitary_codes, grid, units_per_metre):
        return "FURNITURE", False, "context"
    aligned = _align_insert_to_wall(insert, faces, 1.0 * units_per_metre, 12.0, 0.05 * units_per_metre)
    if not aligned:
        return "FURNITURE", False, "unaligned"
    if fixture == "BATHTUB" and not 0.35 <= metric_height <= 0.70:
        return "FURNITURE", False, "height"
    return fixture, True, None


def _near_fixture_evidence(point: object, classes: np.ndarray, fine_classes: np.ndarray, fine_names: list[str], sanitary_codes: tuple[int, ...], grid: RasterGrid, units_per_metre: float) -> bool:
    """Return whether a dimension-only fixture has local sanitary semantic support."""
    x, y = (float(value) for value in point)  # type: ignore[arg-type]
    row, column = grid.local_to_pixel(x, y)
    radius = max(1, int(math.ceil(1.5 * units_per_metre / grid.pixel_size)))
    r0, r1 = max(0, int(row) - radius), min(classes.shape[0], int(row) + radius + 1)
    c0, c1 = max(0, int(column) - radius), min(classes.shape[1], int(column) + radius + 1)
    coarse = classes[r0:r1, c0:c1]
    if np.isin(coarse, sanitary_codes).any():
        return True
    aliases = {"toilet", "wc", "bathtub", "washbasin", "shower", "stove", "sink", "radiator", "column"}
    identifiers = [index + 1 for index, name in enumerate(fine_names) if str(name).strip().lower() in aliases]
    return bool(identifiers) and np.isin(fine_classes[r0:r1, c0:c1], identifiers).any()


def _indoor_wall_axis(lines: list[LineString]) -> float:
    """Use the longest measured face as the dominant indoor wall direction."""
    if not lines:
        return 0.0
    line = max(lines, key=lambda item: item.length)
    first, second = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
    return math.atan2(float(second[1] - first[1]), float(second[0] - first[0])) % (math.pi / 2.0)


def _mask_slice(mask: np.ndarray) -> tuple[slice, slice]:
    """Return the tight rectangular raster slice around a non-empty mask."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        raise ValueError("mask must contain foreground pixels")
    return slice(int(rows.min()), int(rows.max()) + 1), slice(int(cols.min()), int(cols.max()) + 1)


def _furniture_instances(
    object_mask: np.ndarray,
    classes: np.ndarray,
    dsm: np.ndarray,
    floor_z: float,
    fine_classes: np.ndarray,
    fine_confidence: np.ndarray,
    fine_names: list[str],
    grid: RasterGrid,
    units_per_metre: float,
) -> list[tuple[np.ndarray, tuple[slice, slice], str | None, int]]:
    """Split coarse furniture components into reliable semantic instances."""
    _count, coarse, _stats, _centres = cv2.connectedComponentsWithStats(object_mask.astype(np.uint8), connectivity=8)
    output: list[tuple[np.ndarray, tuple[slice, slice], str | None, int]] = []
    for identifier, source_slice in enumerate(ndimage.find_objects(coarse), start=1):
        if source_slice is None:
            continue
        component = coarse[source_slice] == identifier
        assigned = np.zeros_like(component, dtype=bool)
        local_fine = fine_classes[source_slice]
        local_confidence = fine_confidence[source_slice]
        radius = max(1, int(round(0.05 * units_per_metre / grid.pixel_size)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        for fine_identifier in np.unique(local_fine[component & (local_confidence >= 0.4)]):
            if fine_identifier <= 0 or fine_identifier > len(fine_names):
                continue
            region = component & (local_fine == fine_identifier) & (local_confidence >= 0.4)
            region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
            region = ndimage.binary_fill_holes(region) & component
            if float(np.count_nonzero(region)) * grid.pixel_size**2 < 0.10 * units_per_metre**2:
                continue
            label_name = fine_names[int(fine_identifier) - 1]
            pieces = _split_touching_label_instances(region, grid.pixel_size, units_per_metre) if label_name.strip().lower() in {"chair", "armchair"} else [region]
            for piece in pieces:
                area = float(np.count_nonzero(piece)) * grid.pixel_size**2
                if area < 0.10 * units_per_metre**2:
                    continue
                output.append((piece, source_slice, label_name, _dominant_code(classes[source_slice], piece)))
                assigned |= piece
        remainder = component & ~assigned
        if float(np.count_nonzero(remainder)) * grid.pixel_size**2 >= 0.30 * units_per_metre**2:
            _remainder_count, remainder_labels = cv2.connectedComponents(remainder.astype(np.uint8), connectivity=8)
            for remainder_identifier, remainder_slice in enumerate(ndimage.find_objects(remainder_labels), start=1):
                if remainder_slice is None:
                    continue
                piece = remainder_labels[remainder_slice] == remainder_identifier
                area = float(np.count_nonzero(piece)) * grid.pixel_size**2
                local_dsm = dsm[source_slice][remainder_slice]
                polygon = max(mask_to_polygons(piece, _grid_for_slice(grid, (slice(0, piece.shape[0]), slice(0, piece.shape[1])))), key=lambda item: item.area, default=None)
                solidity = 0.0 if polygon is None else polygon.area / max(polygon.convex_hull.area, 1e-9)
                if area < 0.30 * units_per_metre**2 or float(np.nanmax(local_dsm[piece]) - floor_z) < 0.25 * units_per_metre or solidity < 0.6:
                    continue
                row_start = source_slice[0].start + remainder_slice[0].start
                col_start = source_slice[1].start + remainder_slice[1].start
                output.append((piece, (slice(row_start, row_start + piece.shape[0]), slice(col_start, col_start + piece.shape[1])), None, _dominant_code(classes[source_slice][remainder_slice], piece)))
    return output


def _split_touching_label_instances(region: np.ndarray, pixel_size: float, units_per_metre: float) -> list[np.ndarray]:
    """Use DSM-independent distance markers before watershed separates touching chairs."""
    distance = ndimage.distance_transform_edt(region)
    threshold = max(1.0, 0.12 * units_per_metre / pixel_size)
    maxima = (distance >= ndimage.maximum_filter(distance, size=5)) & (distance >= threshold) & region
    markers, count = ndimage.label(maxima)
    centres = [np.mean(np.argwhere(markers == identifier), axis=0) * pixel_size for identifier in range(1, count + 1)]
    if count <= 1 or any(math.dist(first, second) < 0.35 * units_per_metre for index, first in enumerate(centres) for second in centres[index + 1:]):
        return [region]
    labels = watershed(-distance, markers, mask=region)
    pieces = [labels == identifier for identifier in range(1, int(labels.max()) + 1)]
    return [piece for piece in pieces if 0.12 * units_per_metre**2 <= float(np.count_nonzero(piece)) * pixel_size**2 <= 0.8 * units_per_metre**2] or [region]


def _dominant_code(classes: np.ndarray, mask: np.ndarray) -> int:
    values = np.asarray(classes)[np.asarray(mask, dtype=bool)]
    return int(np.bincount(values, minlength=256).argmax()) if len(values) else 0


def _align_insert_to_wall(insert: dict[str, object], faces: list[Any], maximum_distance: float, angle_tolerance_deg: float, flush_gap: float) -> bool:
    """Rotate a rectangular fixture to a nearby measured wall and make it flush."""
    if not faces:
        return False
    point = np.asarray(insert["point"], dtype=float)
    candidates = [face for face in faces if getattr(face, "kind", "") == "line" and getattr(face, "line", None) is not None]
    if not candidates:
        return False
    face = min(candidates, key=lambda candidate: candidate.line.distance(Point(point)))
    distance = float(face.line.distance(Point(point)))
    if distance > maximum_distance:
        return False
    axis = np.asarray(face.line.coords[-1]) - np.asarray(face.line.coords[0])
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    current = math.radians(float(insert["rotation"]))
    current_axis = np.asarray((math.cos(current), math.sin(current)))
    cross = float(axis[0] * current_axis[1] - axis[1] * current_axis[0])
    difference = abs(math.degrees(math.asin(float(np.clip(abs(cross), 0.0, 1.0)))))
    if difference > angle_tolerance_deg:
        return False
    insert["rotation"] = math.degrees(math.atan2(axis[1], axis[0])) % 180.0
    projected = np.asarray(face.line.interpolate(face.line.project(Point(point))).coords[0])
    normal = point - projected
    normal_length = float(np.linalg.norm(normal))
    if normal_length > 1e-9 and abs(normal_length - float(insert["yscale"]) / 2.0) <= flush_gap:
        insert["point"] = (projected + normal / normal_length * float(insert["yscale"]) / 2.0).tolist()
    return True


def _rotate_rectangular_outline_to_wall(entities: list[dict[str, object]], faces: list[Any], grid: RasterGrid, component: np.ndarray, units_per_metre: float) -> None:
    """Keep true outlines untouched unless their measured footprint is rectangular."""
    polygons = mask_to_polygons(component, grid)
    if not polygons:
        return
    raw = max(polygons, key=lambda polygon: polygon.area)
    if raw.area / max(raw.minimum_rotated_rectangle.area, 1e-9) < 0.85:
        return
    # The vector footprint remains its traced boundary.  The matching block
    # insert is the wall-aligned representation for fixtures and sanitation.
    return


def _clean_wall_evidence(mask: np.ndarray, grid: RasterGrid, units_per_metre: float) -> np.ndarray:
    """Close short sampling holes and discard non-architectural upper-band specks."""
    pixels = max(1, int(round(0.03 * units_per_metre / grid.pixel_size)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    cleaned = cv2.morphologyEx(np.asarray(mask, dtype=np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    count, labels, stats, _centres = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=8)
    minimum_pixels = 0.3 * units_per_metre**2 / grid.pixel_size**2
    return np.isin(labels, [identifier for identifier in range(1, count) if stats[identifier, cv2.CC_STAT_AREA] >= minimum_pixels])


def _rectangular_wall_polygons(polygons: list[Polygon]) -> list[Polygon]:
    """Represent every exported wall as an independent simple rectangle.

    Orthogonal L and T arrangements remain multiple touching rectangles rather
    than one unioned raster contour, preserving measured wall thicknesses.
    """
    rectangles: list[Polygon] = []
    for polygon in polygons:
        if polygon.is_empty or polygon.area <= 1e-8:
            continue
        candidate = polygon.minimum_rotated_rectangle
        if candidate.area <= 1e-8:
            continue
        # Thick section regions are accepted only when they are already solid
        # and straight; thin faces always arrive here as fitted rectangles.
        if polygon.area / candidate.area < 0.80:
            continue
        rectangles.append(candidate)
    return rectangles


def _wall_axis(polygon: Polygon) -> np.ndarray:
    coordinates = list(polygon.minimum_rotated_rectangle.exterior.coords)
    edges = [np.asarray(coordinates[index + 1]) - np.asarray(coordinates[index]) for index in range(4)]
    vector = max(edges, key=lambda edge: float(np.hypot(*edge)))
    return vector / max(float(np.hypot(*vector)), 1e-9)


def _parallel_wall_cluster(first: Polygon, second: Polygon, distance_limit: float) -> bool:
    first_axis, second_axis = _wall_axis(first), _wall_axis(second)
    angle = abs(math.asin(float(np.clip(abs(first_axis[0] * second_axis[1] - first_axis[1] * second_axis[0]), 0.0, 1.0))))
    if angle > math.radians(3.0):
        return False
    origin = np.asarray(first.centroid.coords[0])
    normal = np.asarray((-first_axis[1], first_axis[0]))
    offset = abs(float(np.dot(np.asarray(second.centroid.coords[0]) - origin, normal)))
    if offset > distance_limit:
        return False
    first_values = [float(np.dot(np.asarray(point) - origin, first_axis)) for point in first.exterior.coords]
    second_values = [float(np.dot(np.asarray(point) - origin, first_axis)) for point in second.exterior.coords]
    overlap = max(0.0, min(max(first_values), max(second_values)) - max(min(first_values), min(second_values)))
    return overlap >= 0.30 * min(max(first_values) - min(first_values), max(second_values) - min(second_values))


def _merge_parallel_wall_rectangles(walls: list[Polygon], distance_limit: float, max_thickness: float | None = None) -> list[Polygon]:
    """Collapse overlapping parallel wall observations into one outer-face wall."""
    remaining = set(range(len(walls)))
    merged: list[Polygon] = []
    while remaining:
        seed = remaining.pop()
        cluster = {seed}
        changed = True
        while changed:
            changed = False
            for candidate in list(remaining):
                if any(_parallel_wall_cluster(walls[current], walls[candidate], distance_limit) for current in cluster):
                    remaining.remove(candidate)
                    cluster.add(candidate)
                    changed = True
        if len(cluster) == 1:
            merged.append(walls[seed])
            continue
        members = [walls[index] for index in cluster]
        axis = _wall_axis(members[0])
        origin = np.asarray(members[0].centroid.coords[0])
        normal = np.asarray((-axis[1], axis[0]))
        along = [float(np.dot(np.asarray(point) - origin, axis)) for wall in members for point in wall.exterior.coords]
        across = [float(np.dot(np.asarray(point) - origin, normal)) for wall in members for point in wall.exterior.coords]
        min_across, max_across = min(across), max(across)
        if max_thickness is not None and (max_across - min_across) > max_thickness:
            strongest = max(members, key=lambda member: member.area)
            strongest_across = [float(np.dot(np.asarray(point) - origin, normal)) for point in strongest.exterior.coords]
            a0, a1 = min(strongest_across), max(strongest_across)
            if a1 - a0 > max_thickness:
                center = 0.5 * (a0 + a1)
                a0 = center - 0.5 * max_thickness
                a1 = center + 0.5 * max_thickness
            min_across, max_across = a0, a1
        corners = [origin + axis * coordinate + normal * offset for coordinate, offset in ((min(along), min_across), (max(along), min_across), (max(along), max_across), (min(along), max_across))]
        merged.append(Polygon([tuple(point) for point in corners]))
    return merged


def _weak_boundary_walls(floor_mask: np.ndarray, evidence: np.ndarray, classes: np.ndarray, grid: RasterGrid, walls: list[Polygon], units_per_metre: float) -> list[Polygon]:
    """Add only evidence-supported long exterior floor-boundary runs as walls."""
    floors = [polygon for polygon in mask_to_polygons(_closed_floor_mask(floor_mask, grid, units_per_metre), grid) if polygon.area >= 0.5 * units_per_metre**2]
    if not floors or not evidence.any():
        return []
    evidence_geometry = unary_union([polygon.buffer(0.25 * units_per_metre) for polygon in mask_to_polygons(evidence, grid)])
    result: list[Polygon] = []
    existing = unary_union(walls) if walls else None
    thickness = 0.15 * units_per_metre
    for floor in floors:
        coordinates = list(floor.exterior.simplify(0.08 * units_per_metre, preserve_topology=False).coords)
        for first, second in zip(coordinates, coordinates[1:], strict=False):
            line = LineString((first, second))
            if line.length < 1.0 * units_per_metre or evidence_geometry.intersection(line).length < 0.40 * line.length:
                continue
            axis = np.asarray(second) - np.asarray(first)
            axis /= max(float(np.hypot(*axis)), 1e-9)
            normal = np.asarray((-axis[1], axis[0]))
            midpoint = np.asarray(line.interpolate(0.5, normalized=True).coords[0])
            outward = normal
            if floor.contains(Point(*(midpoint + normal * 0.08 * units_per_metre))):
                outward = -normal
            candidate = Polygon([first, second, tuple(np.asarray(second) + outward * thickness), tuple(np.asarray(first) + outward * thickness)])
            if existing is not None and candidate.distance(existing) <= 0.08 * units_per_metre:
                continue
            if _polygon_class_fraction(candidate.buffer(0.05 * units_per_metre), classes, grid, (117, 118, 119, 120, 121)) > 0.30:
                continue
            result.append(candidate)
    return result


def _polygon_class_fraction(polygon: Polygon, classes: np.ndarray, grid: RasterGrid, codes: tuple[int, ...]) -> float:
    mask = _polygon_to_mask(polygon, grid)
    if not mask.any():
        return 0.0
    return float(np.count_nonzero(mask & np.isin(classes, codes)) / np.count_nonzero(mask))


def _polygon_to_mask(polygon: Polygon, grid: RasterGrid) -> np.ndarray:
    mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
    points = np.asarray([[round(grid.local_to_pixel(x, y)[1]), round(grid.local_to_pixel(x, y)[0])] for x, y in polygon.exterior.coords], dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def _filter_wall_stubs(walls: list[Polygon], classes: np.ndarray, grid: RasterGrid, units_per_metre: float) -> list[Polygon]:
    result: list[Polygon] = []
    for index, wall in enumerate(walls):
        axis = _wall_axis(wall)
        length = max(float(np.ptp([np.dot(np.asarray(point), axis) for point in wall.exterior.coords])), 0.0)
        touches = any(index != other_index and wall.distance(other) <= 0.3 * units_per_metre for other_index, other in enumerate(walls))
        furniture_fraction = _polygon_class_fraction(wall, classes, grid, (117, 118, 119, 120, 121))
        if (length < 0.6 * units_per_metre and not touches) or furniture_fraction > 0.30:
            continue
        result.append(wall)
    return result


def _window_is_exterior(opening: Any, free_space: np.ndarray, grid: RasterGrid, units_per_metre: float) -> bool:
    axis = np.asarray((math.cos(opening.axis_angle), math.sin(opening.axis_angle)))
    normal = np.asarray((-axis[1], axis[0]))
    samples = np.linspace(0.20, 1.0, 5) * units_per_metre
    side_has_floor: list[bool] = []
    for sign in (1.0, -1.0):
        values = []
        for distance in samples:
            point = np.asarray(opening.center) + normal * distance * sign
            row, col = grid.local_to_pixel(float(point[0]), float(point[1]))
            row, col = int(round(row)), int(round(col))
            values.append(0 <= row < grid.height and 0 <= col < grid.width and bool(free_space[row, col]))
        side_has_floor.append(any(values))
    return not all(side_has_floor)


def _wall_polygons_from_result(result: DxfPlanResult) -> list[Polygon]:
    polygons: list[Polygon] = []
    for entity in result.layers.get("A-WALL", []):
        if entity.get("type") != "polyline" or not entity.get("closed"):
            continue
        points = entity.get("points", [])
        if len(points) >= 3:
            polygon = Polygon(points, entity.get("holes", []))
            if polygon.is_valid and polygon.area > 1e-9:
                polygons.append(polygon)
    return polygons


def indoor_wall_metrics(result: DxfPlanResult) -> dict[str, Any]:
    """Measure exported indoor face lines and true furniture outlines."""
    wall_entities = result.layers.get("A-WALL", [])
    lines: list[LineString] = []
    rms_values: list[float] = []
    max_values: list[float] = []
    snapped_length = 0.0
    for entity in wall_entities:
        if entity.get("type") == "polyline" and entity.get("closed") and entity.get("_double_wall") and len(entity.get("points", [])) >= 3:
            for ring in [entity["points"], *entity.get("holes", [])]:
                ring_line = LineString([*ring, ring[0]])
                lines.append(ring_line)
                rms_values.append(0.0)
                max_values.append(0.0)
                snapped_length += ring_line.length
            continue
        if entity.get("type") == "polyline" and not entity.get("closed") and len(entity.get("points", [])) >= 2:
            line = LineString(entity["points"])
        elif entity.get("type") == "arc":
            center = entity.get("center", (0.0, 0.0)); radius = float(entity.get("radius", 0.0))
            start, end = math.radians(float(entity.get("start_angle", 0.0))), math.radians(float(entity.get("end_angle", 0.0)))
            angles = np.linspace(start, end, max(8, int(abs(end - start) * radius / 0.02)))
            line = LineString([(float(center[0]) + radius * math.cos(angle), float(center[1]) + radius * math.sin(angle)) for angle in angles])
        else:
            continue
        lines.append(line)
        rms_values.append(float(entity.get("_rms", 0.0)))
        max_values.append(float(entity.get("_max_residual", 0.0)))
        if entity.get("_axis_snapped"):
            snapped_length += line.length
    face_geometry = unary_union(lines) if lines else LineString()
    total_length = sum(line.length for line in lines)
    floor_polygons = [Polygon(points) for points in result.fingerprint_inputs.get("indoor_floor_polygons", []) if len(points) >= 3]
    units_per_metre = float(result.fingerprint_inputs.get("units_per_metre", 1.0))
    if floor_polygons and not face_geometry.is_empty:
        floor_area = unary_union(floor_polygons)
        boundary = unary_union([polygon.exterior for polygon in floor_polygons])
        coverage = 0.0 if boundary.length <= 1e-9 else float(boundary.intersection(face_geometry.buffer(0.05 * units_per_metre)).length / boundary.length)
        inside = face_geometry.intersection(floor_area.buffer(-0.02 * units_per_metre)).length / max(total_length, 1e-9)
    else:
        coverage, inside = 0.0, 0.0
    door_entities = [entity for entity in result.layers.get("A-DOOR", []) if entity.get("type") == "insert" and entity.get("name") == "DOOR"]
    doors_off_wall = sum(not bool(entity.get("_on_face", face_geometry.distance(Point(entity.get("_opening_center", entity["point"]))) <= 0.15 * units_per_metre)) for entity in door_entities)
    furniture = [entity for entity in result.layers.get("A-FURN", []) if entity.get("type") == "polyline" and entity.get("closed")]
    labels: dict[str, int] = {}
    for entity in furniture:
        name = str(entity.get("_label", "Object"))
        labels[name] = labels.get(name, 0) + 1
    aligned = sum(bool(entity.get("_wall_aligned")) for layer in ("A-FURN", "P-SANI") for entity in result.layers.get(layer, []))
    einb = result.layers.get("A-EINB", [])
    einb_segments = sum(entity.get("type") == "polyline" and not entity.get("closed") for entity in einb)
    einb_counter_polygons = sum(entity.get("type") == "polyline" and bool(entity.get("closed")) for entity in einb)
    return {
        "wall_face_rms_max": float(max(rms_values, default=0.0)),
        "wall_face_max_residual": float(max(max_values, default=0.0)),
        "wall_boundary_coverage": float(coverage),
        "wall_evidence_coverage": float(result.fingerprint_inputs.get("wall_evidence_coverage", 0.0)),
        "wall_axis_fraction": float(snapped_length / max(total_length, 1e-9)),
        "faces_inside_free_space": float(inside),
        "doors_off_wall": int(doors_off_wall),
        "furniture_outline_hausdorff_max": float(max((float(entity.get("_hausdorff", 0.0)) for entity in furniture), default=0.0)),
        "furniture_convex_hull_count": 0,
        "fixtures_wall_aligned": int(aligned),
        "furniture_instances_by_label": labels,
        "footprint_kinds": dict(result.fingerprint_inputs.get("footprint_kinds", {})),
        "footprint_dev90_max": float(result.fingerprint_inputs.get("footprint_dev90_max", 0.0)),
        "footprints_noise_dropped": int(result.fingerprint_inputs.get("footprints_noise_dropped", 0)),
        "footprints_clutter_rectangles": int(result.fingerprint_inputs.get("footprints_clutter_rectangles", 0)),
        "footprints_floor_view_median": float(result.fingerprint_inputs.get("footprints_floor_view_median", 0.0)),
        "fine_instances_by_label": dict(result.fingerprint_inputs.get("fine_instances_by_label", {})),
        "cushions_merged": int(result.fingerprint_inputs.get("cushions_merged", 0)),
        "clothes_absorbed": int(result.fingerprint_inputs.get("clothes_absorbed", 0)),
        "nightstands_split": int(result.fingerprint_inputs.get("nightstands_split", 0)),
        "kitchen_einb_from_fine": int(result.fingerprint_inputs.get("kitchen_einb_from_fine", 0)),
        "fine_label_source": str(result.fingerprint_inputs.get("fine_label_source", "none")),
        "free_footprint_labels": list(result.fingerprint_inputs.get("free_footprint_labels", [])),
        "fixtures_rejected": dict(result.fingerprint_inputs.get("fixtures_rejected", {})),
        "ceiling_coverage": float(result.fingerprint_inputs.get("ceiling_coverage", 0.0)),
        "rooms": int(result.fingerprint_inputs.get("rooms", 0)),
        "rooms_skipped": int(result.fingerprint_inputs.get("rooms_skipped", 0)),
        "ceiling_wall_pieces": int(result.fingerprint_inputs.get("ceiling_wall_pieces", 0)),
        "ceiling_wall_length": float(result.fingerprint_inputs.get("ceiling_wall_length", 0.0)),
        "ceiling_edges_rejected": dict(result.fingerprint_inputs.get("ceiling_edges_rejected", {})),
        "einb_segments": int(einb_segments),
        "einb_counter_polygons": int(einb_counter_polygons),
        "einb_rejected": dict(result.fingerprint_inputs.get("einb_rejected", {})),
        "doors_total": int(result.fingerprint_inputs.get("doors_total", 0)),
        "doors_with_leaf": int(result.fingerprint_inputs.get("doors_with_leaf", 0)),
        "doors_with_leaf_flat": int(result.fingerprint_inputs.get("doors_with_leaf_flat", 0)),
        "doors_opening_only": int(result.fingerprint_inputs.get("doors_opening_only", 0)),
        "passages": int(result.fingerprint_inputs.get("passages", 0)),
        "doors_deduplicated": int(result.fingerprint_inputs.get("doors_deduplicated", 0)),
        "windows_snapped": int(result.fingerprint_inputs.get("windows_snapped", 0)),
        "wall_extended_pieces": int(result.fingerprint_inputs.get("wall_extended_pieces", 0)),
        "wall_extended_length": float(result.fingerprint_inputs.get("wall_extended_length", 0.0)),
    }


def _closed_floor_mask(mask: np.ndarray, grid: RasterGrid, units_per_metre: float) -> np.ndarray:
    pixels = max(1, int(round(0.10 * units_per_metre / grid.pixel_size)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.morphologyEx(np.asarray(mask, dtype=np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def _opening_on_wall(opening: Any, walls: list[Polygon], tolerance: float) -> bool:
    point = Point(opening.center)
    return bool(walls) and min(wall.boundary.distance(point) for wall in walls) <= tolerance


def _opening_class_fraction(opening: Any, mask: np.ndarray, grid: RasterGrid, units_per_metre: float) -> float:
    """Share of the opening footprint (width x 0.30 m across) covered by ``mask``."""
    if not mask.any():
        return 0.0
    axis = np.asarray((math.cos(opening.axis_angle), math.sin(opening.axis_angle)))
    normal = np.asarray((-axis[1], axis[0]))
    along = np.linspace(-opening.width / 2.0, opening.width / 2.0, max(2, int(math.ceil(opening.width / grid.pixel_size)) + 1))
    across = np.linspace(-0.15 * units_per_metre, 0.15 * units_per_metre, max(2, int(math.ceil(0.30 * units_per_metre / grid.pixel_size)) + 1))
    a, n = np.meshgrid(along, across)
    xs = opening.center[0] + a * axis[0] + n * normal[0]
    ys = opening.center[1] + a * axis[1] + n * normal[1]
    cols = np.round((xs - grid.min_x) / grid.pixel_size - 0.5).astype(int)
    rows = np.round((grid.max_y - ys) / grid.pixel_size - 0.5).astype(int)
    inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    if not inside.any():
        return 0.0
    return float(mask[rows[inside], cols[inside]].mean())


def _snap_openings_to_faces(openings: list[Opening], faces: list[WallFace], tolerance: float) -> list[Opening]:
    """Project candidates to the nearest compatible infinite measured Flucht."""
    lines = [face.line for face in faces if face.kind == "line" and face.line is not None]
    snapped: list[Opening] = []
    for opening in openings:
        centre = np.asarray(opening.center, dtype=float)
        original_axis = np.asarray((math.cos(opening.axis_angle), math.sin(opening.axis_angle)))
        matches: list[tuple[float, float, LineString, np.ndarray, np.ndarray]] = []
        for line in lines:
            start, end = np.asarray(line.coords[0], dtype=float), np.asarray(line.coords[-1], dtype=float)
            axis = end - start
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            angle = abs(math.degrees(math.asin(float(np.clip(abs(np.cross(axis, original_axis)), 0.0, 1.0)))))
            if opening.kind == "WINDOW" and angle > 5.0:
                continue
            normal = np.asarray((-axis[1], axis[0]))
            offset = abs(float(np.dot(centre - start, normal)))
            if offset > tolerance:
                continue
            projected = start + axis * float(np.dot(centre - start, axis))
            end_distance = min(float(np.linalg.norm(projected - start)), float(np.linalg.norm(projected - end)))
            matches.append((offset, end_distance, line, axis, projected))
        if not matches:
            snapped.append(opening)
            continue
        # A partial face often terminates at the opening; prefer that end over a
        # parallel but unrelated face that happens to lie marginally closer.
        _offset, _end_distance, line, axis, projected = min(matches, key=lambda item: (item[1], item[0]))
        width = float(opening.width)
        if opening.kind == "WINDOW":
            width = _clamp_window_width(projected, axis, width, lines)
        default_thickness = tolerance * (0.25 / 0.30) if opening.kind == "WINDOW" else opening.wall_thickness
        snapped.append(replace(opening, center=(float(projected[0]), float(projected[1])), width=width, axis_angle=math.atan2(float(axis[1]), float(axis[0])) % math.pi, wall_thickness=max(opening.wall_thickness, default_thickness)))
    return snapped


def _clamp_window_width(centre: np.ndarray, axis: np.ndarray, width: float, lines: list[LineString]) -> float:
    """Prevent a window span from consuming an existing face crossing its jamb."""
    limits: list[float] = []
    normal = np.asarray((-axis[1], axis[0]))
    for line in lines:
        start, end = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
        direction = end - start
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        if abs(float(np.dot(direction, axis))) < 0.25:
            continue
        for point in (start, end):
            if abs(float(np.dot(point - centre, normal))) <= 0.03:
                limits.append(abs(float(np.dot(point - centre, axis))) * 2.0)
    permissible = [value for value in limits if value >= 0.40]
    return min([width, *permissible]) if permissible else width


def _collinear_face_gap_openings(faces: list[WallFace], passable: np.ndarray, lintel: np.ndarray, door_mask: np.ndarray, valid: np.ndarray, grid: RasterGrid, units_per_metre: float) -> list[Opening]:
    """Turn passable 0.60-1.30 m collinear face gaps into door candidates."""
    lines = [face.line for face in faces if face.kind == "line" and face.line is not None]
    candidates: list[Opening] = []
    for index, first in enumerate(lines):
        axis = np.asarray(first.coords[-1]) - np.asarray(first.coords[0])
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        normal = np.asarray((-axis[1], axis[0]))
        origin = np.asarray(first.coords[0])
        for second in lines[index + 1:]:
            second_axis = np.asarray(second.coords[-1]) - np.asarray(second.coords[0])
            second_axis /= max(float(np.linalg.norm(second_axis)), 1e-12)
            if abs(math.degrees(math.asin(float(np.clip(abs(np.cross(axis, second_axis)), 0.0, 1.0))))) > 5.0:
                continue
            if abs(float(np.dot(np.asarray(second.coords[0]) - origin, normal))) > 0.05 * units_per_metre:
                continue
            first_endpoints = [np.asarray(point, dtype=float) for point in first.coords]
            second_endpoints = [np.asarray(point, dtype=float) for point in second.coords]
            left_point, right_point = min(((left, right) for left in first_endpoints for right in second_endpoints), key=lambda pair: float(np.linalg.norm(pair[0] - pair[1])))
            gap = float(np.linalg.norm(right_point - left_point))
            if not 0.60 * units_per_metre <= gap <= 1.30 * units_per_metre:
                continue
            segment = LineString((tuple(left_point), tuple(right_point)))
            if not _line_passable_doorway(segment, normal, passable, grid, units_per_metre):
                continue
            centre = (left_point + right_point) / 2.0
            probe = Opening("DOOR", (float(centre[0]), float(centre[1])), gap, math.atan2(float(axis[1]), float(axis[0])), 0.25 * units_per_metre, evidence="face-gap")
            class_fraction = _opening_class_fraction(probe, door_mask, grid, units_per_metre)
            lintel_fraction = _opening_class_fraction(probe, lintel, grid, units_per_metre)
            kind = "DOOR" if class_fraction > 0.0 or lintel_fraction >= 0.50 else "passage"
            candidate = replace(probe, kind=kind, class_evidence=class_fraction, lintel_evidence=lintel_fraction)
            if not _opening_into_void(candidate, valid, grid, units_per_metre):
                candidates.append(candidate)
    return candidates


def _line_passable_doorway(line: LineString, normal: np.ndarray, passable: np.ndarray, grid: RasterGrid, units_per_metre: float) -> bool:
    samples = [line.interpolate(value, normalized=True) for value in np.linspace(0.15, 0.85, 5)]
    values = []
    for point in samples:
        values.append(all(_raster_bool(passable, grid, point.x + sign * normal[0] * 0.30 * units_per_metre, point.y + sign * normal[1] * 0.30 * units_per_metre) for sign in (-1.0, 1.0)))
    return bool(values) and float(np.mean(values)) >= 0.50


def _normalise_door_openings(openings: list[Opening], door_mask: np.ndarray, lintel: np.ndarray, window_mask: np.ndarray, passable: np.ndarray, grid: RasterGrid, units_per_metre: float) -> list[Opening]:
    """Keep doors only with evidence and passages only with floor on both sides."""
    output: list[Opening] = []
    for opening in openings:
        if opening.kind not in {"DOOR", "passage"}:
            output.append(opening)
            continue
        class_fraction = _opening_class_fraction(opening, door_mask, grid, units_per_metre)
        lintel_fraction = _opening_class_fraction(opening, lintel, grid, units_per_metre)
        window_fraction = _opening_class_fraction(opening, window_mask, grid, units_per_metre)
        both_sides = _opening_has_free_floor_both_sides(opening, passable, grid, units_per_metre)
        in_door_range = 0.60 * units_per_metre <= opening.width <= 1.30 * units_per_metre
        # Window segmentation wins over trace leakage from a nearby door class.
        # The latter is common at sill edges and is not meaningful door evidence.
        if opening.kind == "DOOR" and window_fraction >= 0.05 and class_fraction < 0.05 and lintel_fraction < 0.50:
            output.append(replace(opening, kind="WINDOW", wall_thickness=max(opening.wall_thickness, 0.25 * units_per_metre), evidence="class-104", class_evidence=0.0, lintel_evidence=0.0))
            continue
        if opening.kind == "DOOR" and (not in_door_range or (class_fraction <= 0.0 and lintel_fraction < 0.50)):
            if in_door_range and window_fraction >= 0.05:
                output.append(replace(opening, kind="WINDOW", wall_thickness=max(opening.wall_thickness, 0.25 * units_per_metre), evidence="class-104", class_evidence=0.0, lintel_evidence=0.0))
            elif in_door_range and both_sides:
                output.append(replace(opening, kind="passage", class_evidence=class_fraction, lintel_evidence=lintel_fraction))
            continue
        if opening.kind == "passage" and not both_sides:
            continue
        output.append(replace(opening, class_evidence=class_fraction, lintel_evidence=lintel_fraction))
    return output


def _opening_has_free_floor_both_sides(opening: Opening, passable: np.ndarray, grid: RasterGrid, units_per_metre: float) -> bool:
    axis = np.asarray((math.cos(opening.axis_angle), math.sin(opening.axis_angle)))
    normal = np.asarray((-axis[1], axis[0]))
    values = []
    for position in np.linspace(-0.35, 0.35, 5) * opening.width:
        centre = np.asarray(opening.center) + axis * position
        values.append(all(_raster_bool(passable, grid, float(centre[0] + sign * normal[0] * 0.30 * units_per_metre), float(centre[1] + sign * normal[1] * 0.30 * units_per_metre)) for sign in (-1.0, 1.0)))
    return bool(values) and float(np.mean(values)) >= 0.50


def _opening_into_void(opening: Opening, valid: np.ndarray, grid: RasterGrid, units_per_metre: float) -> bool:
    if opening.kind not in {"DOOR", "passage"}:
        return False
    axis = np.asarray((math.cos(opening.axis_angle), math.sin(opening.axis_angle)))
    normal = np.asarray((-axis[1], axis[0]))
    sides = []
    for sign in (-1.0, 1.0):
        values = [_raster_bool(valid, grid, opening.center[0] + sign * normal[0] * distance, opening.center[1] + sign * normal[1] * distance) for distance in np.linspace(0.15, 0.60, 4) * units_per_metre]
        sides.append(any(values))
    return not any(sides)


def _raster_bool(mask: np.ndarray, grid: RasterGrid, x: float, y: float) -> bool:
    row, col = grid.local_to_pixel(float(x), float(y))
    row, col = int(round(row)), int(round(col))
    return 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1] and bool(mask[row, col])


def _wall_settings_for_pass(settings: Any, openings_3d_info: dict[str, Any]) -> Any:
    """Wall settings of this plan pass: with the refined openings applied the 3D openings are final (``refined_openings``)."""
    return replace(settings, refined_openings=True) if openings_3d_info.get("applied") else settings


def _door_leaf_lines(openings: list[Opening]) -> list[LineString]:
    """Reconstruct measured leaves so A-EINB cannot redraw them as built-ins."""
    lines: list[LineString] = []
    for opening in openings:
        if opening.kind == "DOOR" and opening.leaf_hinge is not None and opening.leaf_free is not None:
            lines.append(LineString((tuple(opening.leaf_hinge), tuple(opening.leaf_free))))  # measured leaf of the 3D refinement
            continue
        if opening.kind != "DOOR" or opening.leaf_angle_deg is None or opening.hinge_side is None:
            continue
        axis = np.asarray((math.cos(opening.axis_angle), math.sin(opening.axis_angle)))
        hinge = np.asarray(opening.center) - axis * opening.width / 2.0 if opening.hinge_side == "left" else np.asarray(opening.center) + axis * opening.width / 2.0
        direction_angle = opening.leaf_direction_deg if opening.leaf_direction_deg is not None else opening.leaf_angle_deg
        direction = np.asarray((math.cos(math.radians(direction_angle)), math.sin(math.radians(direction_angle))))
        lines.append(LineString((tuple(hinge), tuple(hinge + direction * opening.width))))
    return lines


def _exclude_door_leaves_from_builtins(entities: list[dict[str, Any]], leaves: list[LineString], units_per_metre: float) -> tuple[list[dict[str, Any]], int]:
    if not leaves:
        return entities, 0
    kept: list[dict[str, Any]] = []
    rejected = 0
    for entity in entities:
        if entity.get("type") != "polyline" or entity.get("closed") or len(entity.get("points", [])) < 2:
            kept.append(entity)
            continue
        line = LineString(entity["points"])
        if any(line.distance(leaf) <= 0.07 * units_per_metre and line.intersection(leaf.buffer(0.04 * units_per_metre)).length >= 0.50 * line.length for leaf in leaves):
            rejected += 1
            continue
        kept.append(entity)
    return kept, rejected


def _deduplicate_plan_openings(openings: list[Opening], window_distance: float = 0.0) -> list[Opening]:
    """Deduplicate door candidates after snapping them to a shared Flucht."""
    result: list[Opening] = []
    for opening in sorted(openings, key=lambda item: (-item.class_evidence, -item.lintel_evidence, item.kind == "passage")):
        duplicate = next((existing for existing in result if opening.kind in {"DOOR", "passage"} and existing.kind in {"DOOR", "passage"} and _same_opening_span(opening, existing)), None)
        if duplicate is None and opening.kind == "WINDOW" and window_distance > 0.0:
            duplicate = next((existing for existing in result if existing.kind == "WINDOW" and math.dist(opening.center, existing.center) < window_distance and _same_opening_span(opening, existing)), None)
        if duplicate is None:
            result.append(opening)
        elif opening.kind == "DOOR" and duplicate.kind == "passage":
            result[result.index(duplicate)] = opening
    return result


def _same_opening_span(first: Opening, second: Opening) -> bool:
    axis = np.asarray((math.cos(first.axis_angle), math.sin(first.axis_angle)))
    other_axis = np.asarray((math.cos(second.axis_angle), math.sin(second.axis_angle)))
    if abs(math.degrees(math.asin(float(np.clip(abs(np.cross(axis, other_axis)), 0.0, 1.0))))) > 5.0:
        return False
    normal = np.asarray((-axis[1], axis[0]))
    delta = np.asarray(second.center) - np.asarray(first.center)
    return abs(float(np.dot(delta, normal))) <= max(0.05, first.wall_thickness, second.wall_thickness) and (abs(float(np.dot(delta, axis))) < 0.5 * max(first.width, second.width) or abs(float(np.dot(delta, axis))) < (first.width + second.width) / 2.0)


def _class_window_openings(mask: np.ndarray, faces: list[Any], existing: list[Any], grid: RasterGrid, units_per_metre: float) -> list[Any]:
    """Windows seen by the segmentation on a wall face (glass often leaves no section gap)."""
    if not mask.any() or not faces:
        return []
    geometries = [face.geometry() for face in faces if face.kind == "line" and face.line is not None]
    if not geometries:
        return []
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(round(0.05 * units_per_metre / grid.pixel_size))) + 1,) * 2))
    count, components, stats, _centres = cv2.connectedComponentsWithStats(closed, connectivity=8)
    minimum_area = 0.02 * units_per_metre**2 / grid.pixel_size**2
    openings: list[Any] = []
    for identifier in range(1, count):
        if stats[identifier, cv2.CC_STAT_AREA] < minimum_area:
            continue
        rows, cols = np.nonzero(components == identifier)
        points = np.column_stack((grid.min_x + (cols + 0.5) * grid.pixel_size, grid.max_y - (rows + 0.5) * grid.pixel_size))
        centre = Point(points.mean(axis=0))
        face = min(geometries, key=lambda geometry: geometry.distance(centre))
        if face.distance(centre) > 0.25 * units_per_metre:
            continue
        start, end = np.asarray(face.coords[0]), np.asarray(face.coords[-1])
        axis = (end - start) / max(float(np.linalg.norm(end - start)), 1e-12)
        projections = (points - start) @ axis
        width = float(np.ptp(projections)) + grid.pixel_size
        if not 0.4 * units_per_metre <= width <= 3.0 * units_per_metre:
            continue
        middle = start + axis * float((projections.min() + projections.max()) / 2.0)
        if any(math.dist(tuple(middle), other.center) <= max(width, other.width) / 2.0 for other in [*existing, *openings]):
            continue
        openings.append(Opening("WINDOW", (float(middle[0]), float(middle[1])), width, math.atan2(float(axis[1]), float(axis[0])), max(2.0 * grid.pixel_size, 0.10 * units_per_metre)))
    return openings


def _opening_on_faces(opening: Any, faces: list[Any], tolerance: float) -> bool:
    """Accept openings only when their measured centre is on a face line."""
    point = Point(opening.center)
    geometries = [face.geometry() for face in faces]
    return bool(geometries) and min(geometry.distance(point) for geometry in geometries) <= tolerance


def _wall_quality(walls: list[Polygon]) -> tuple[float, float]:
    if not walls:
        return 0.0, 0.0
    angles: list[float] = []
    lengths: list[float] = []
    vertices: list[int] = []
    for wall in walls:
        coordinates = list(wall.exterior.coords)
        vertices.append(max(0, len(coordinates) - 1))
        for first, second in zip(coordinates, coordinates[1:], strict=False):
            dx, dy = second[0] - first[0], second[1] - first[1]
            length = math.hypot(dx, dy)
            if length > 1e-9:
                angles.append(math.atan2(dy, dx) % (math.pi / 2.0))
                lengths.append(length)
    if not lengths:
        return 0.0, float(np.median(vertices))
    axis = angles[int(np.argmax(lengths))]
    aligned = sum(length for angle, length in zip(angles, lengths, strict=True) if abs((angle - axis + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0) <= math.radians(3.0))
    return aligned / sum(lengths), float(np.median(vertices))


def _add_room_labels(
    target: list[dict[str, Any]], wall_mask: np.ndarray, grid: RasterGrid, minimum_area: float, *,
    units_per_metre: float = 1.0,
    tagged_points: list[tuple[tuple[float, float], str]] | None = None,
    name_settings: RoomNameSettings = RoomNameSettings(),
) -> int:
    """Label enclosed rooms of a wall mask.

    With ``tagged_points`` (fixture and furniture tags with positions) each room
    is named from the fixtures inside it, e.g. ``Küche  12.4 m²``; without them
    the generic ``Room 1 (12.4 m²)`` text is kept.
    """
    _count, labels, stats, _centres = cv2.connectedComponentsWithStats((~wall_mask).astype(np.uint8), connectivity=4)
    rooms: list[tuple[Polygon, float]] = []
    for identifier, component_slice in enumerate(ndimage.find_objects(labels), start=1):
        if component_slice is None:
            continue
        component = labels[component_slice] == identifier
        if (
            (component_slice[0].start == 0 and component[0].any())
            or (component_slice[0].stop == grid.height and component[-1].any())
            or (component_slice[1].start == 0 and component[:, 0].any())
            or (component_slice[1].stop == grid.width and component[:, -1].any())
        ):
            continue
        area = float(stats[identifier, cv2.CC_STAT_AREA]) * grid.pixel_size**2
        if area < max(2.0, minimum_area):
            continue
        polygons = [polygon for polygon in mask_to_polygons(component, _grid_for_slice(grid, component_slice)) if polygon.area >= area * 0.9]
        if polygons:
            rooms.append((max(polygons, key=lambda polygon: polygon.area), area))
    labels_settings = replace(load_label_settings(), min_area=minimum_area)
    ordered = sorted(rooms, key=lambda item: item[0].centroid.x)
    decisions = name_rooms([polygon for polygon, _area in ordered], tagged_points, units_per_metre, name_settings) if tagged_points is not None else None
    for index, (polygon, area) in enumerate(ordered, start=1):
        decision = None if decisions is None else decisions[index - 1]
        if decision is not None and decision.name:
            text = room_label_text(decision.name, decision.area_m2)
        else:
            text = f"Room {index} ({area:.1f} m²)"
        target.extend(label_entities([(polygon, text)], labels_settings))
    return len(rooms)


_FIXTURE_KINDS: dict[str, tuple[str, ...]] = {
    "WC": ("WC_WALL",), "WASHBASIN": ("WASHBASIN",), "BATHTUB": ("BATHTUB",), "SHOWER": ("SHOWER",),
    "STOVE": ("COOKTOP",), "SINK": ("KITCHEN_SINK",),
}
_APPLIANCE_TOLERANCE = 0.30
_FURNITURE_KINDS: dict[str, tuple[str, ...]] = {
    "bed": ("BED_SINGLE", "BED_DOUBLE"), "sofa": ("SOFA",), "wardrobe": ("WARDROBE",),
    "table": ("TABLE_DINING",), "dining table": ("TABLE_DINING",),
}
_FURNITURE_TOLERANCE = 0.15


def _catalogue_fixture(fixture: str, insert: dict[str, object], wall_lines: list[LineString], units_per_metre: float) -> dict[str, object] | None:
    """Replace a measured fixture insert by a uniform-scale catalogue block, snapped flush to the wall."""
    kinds = _FIXTURE_KINDS.get(fixture)
    if kinds is None:
        return None
    tolerance = _APPLIANCE_TOLERANCE if fixture in {"STOVE", "SINK"} else None
    snapped = snap_insert_to_wall(insert, kinds, wall_lines, units_per_metre, max_gap_m=0.30, tolerance=tolerance)
    return snapped or centred_catalogue_insert(insert, kinds, units_per_metre, tolerance=tolerance)


def _furniture_catalogue_insert(label: str, fit: Any, wall_lines: list[LineString], units_per_metre: float) -> dict[str, object] | None:
    """Catalogue block for a rectangular furniture footprint (bed, sofa, wardrobe, dining table) when it matches a standard size."""
    kinds = _FURNITURE_KINDS.get(label)
    if kinds is None:
        return None
    rectangle = fit.polygon.minimum_rotated_rectangle
    corners = np.asarray(rectangle.exterior.coords[:4], dtype=float)
    edges = [corners[(index + 1) % 4] - corners[index] for index in range(4)]
    lengths = [float(np.hypot(*edge)) for edge in edges[:2]]
    long_index = int(np.argmax(lengths))
    measured: dict[str, object] = {
        "type": "insert", "name": "FURNITURE", "point": [float(rectangle.centroid.x), float(rectangle.centroid.y)],
        "rotation": math.degrees(math.atan2(float(edges[long_index][1]), float(edges[long_index][0]))) % 180.0,
        "xscale": lengths[long_index], "yscale": lengths[1 - long_index], "attributes": {},
    }
    if kinds == ("TABLE_DINING",):
        return centred_catalogue_insert(measured, kinds, units_per_metre, tolerance=_FURNITURE_TOLERANCE, centred_origin=True)
    return snap_insert_to_wall(measured, kinds, wall_lines, units_per_metre, max_gap_m=0.15, tolerance=_FURNITURE_TOLERANCE)


_APPLIANCE_LABELS: dict[str, tuple[str, ...]] = {
    "COOKTOP": ("stove", "cooktop", "hob", "oven"),
    "KITCHEN_SINK": ("kitchen sink", "sink"),
    "REFRIGERATOR": ("refrigerator", "fridge"),
}


def _kitchen_appliance_points(
    fine_classes: np.ndarray, fine_names: list[str], instance_raster: np.ndarray | None, instance_labels: dict[int, str],
    height_map: np.ndarray, grid: RasterGrid, units_per_metre: float,
) -> dict[str, np.ndarray]:
    """Local points of appliance labels (fine label or instance label); cooktop / sink only at counter height."""
    heights = np.nan_to_num(np.asarray(height_map, dtype=np.float32), nan=-1.0)
    counter = (heights >= 0.75 * units_per_metre) & (heights <= 1.05 * units_per_metre)
    output: dict[str, np.ndarray] = {}
    for kind, names in _APPLIANCE_LABELS.items():
        ids = [index + 1 for index, name in enumerate(fine_names) if str(name).strip().lower() in names]
        mask = np.isin(fine_classes, ids) if ids else np.zeros(fine_classes.shape, dtype=bool)
        if instance_raster is not None and instance_labels and not mask.any():
            owners = [identifier for identifier, label in instance_labels.items() if str(label).strip().lower() in names]
            if owners:
                mask |= np.isin(instance_raster, owners)
        if kind != "REFRIGERATOR":
            mask &= counter
        rows, cols = np.nonzero(mask)
        if len(rows):
            output[kind] = np.column_stack((grid.min_x + (cols + 0.5) * grid.pixel_size, grid.max_y - (rows + 0.5) * grid.pixel_size))
    return output


def _footprint_evidence(component: np.ndarray, local_grid: RasterGrid, units_per_metre: float) -> Polygon | None:
    """Traced footprint of one object mask (largest part, 2 cm closing, holes filled)."""
    polygons = mask_to_polygons(np.asarray(component, dtype=bool), local_grid)
    if not polygons:
        return None
    measured = max(polygons, key=lambda item: item.area).buffer(0.02 * units_per_metre).buffer(-0.02 * units_per_metre)
    parts = [measured] if isinstance(measured, Polygon) else [part for part in getattr(measured, "geoms", ()) if isinstance(part, Polygon)]
    parts = [part for part in parts if part.area > 0.0]
    if not parts:
        return None
    return Polygon(max(parts, key=lambda item: item.area).exterior).simplify(0.005 * units_per_metre, preserve_topology=True)


def _height_sampler(height_map: np.ndarray, grid: RasterGrid) -> Callable[[np.ndarray], np.ndarray]:
    """Height above the floor (units) at local points; NaN outside the raster or where nothing was measured."""
    heights = np.asarray(height_map, dtype=np.float64)

    def sample(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        cols = np.floor((values[:, 0] - grid.min_x) / grid.pixel_size).astype(np.int64)
        rows = np.floor((grid.max_y - values[:, 1]) / grid.pixel_size).astype(np.int64)
        inside = (rows >= 0) & (rows < heights.shape[0]) & (cols >= 0) & (cols < heights.shape[1])
        output = np.full(len(values), np.nan)
        output[inside] = heights[rows[inside], cols[inside]]
        return output

    return sample


def _insert_footprint(insert: dict[str, object]) -> Polygon | None:
    """Plan footprint of a catalogue insert (body outline, no chairs)."""
    entry = catalogue_entry_for_block(str(insert["name"]))
    if entry is None:
        return None
    scale = float(insert["xscale"])
    y0, y1 = (0.0, entry.depth_m) if entry.wall_mounted else (-entry.depth_m / 2.0, entry.depth_m / 2.0)
    body = box(-entry.width_m / 2.0 * scale, y0 * scale, entry.width_m / 2.0 * scale, y1 * scale)
    body = affinity.rotate(body, float(insert["rotation"]), origin=(0.0, 0.0))
    return affinity.translate(body, float(insert["point"][0]), float(insert["point"][1]))  # type: ignore[index]


def _insert_duplicates_existing(candidate: dict[str, object], layers: dict[str, list[dict[str, Any]]]) -> bool:
    """True when a catalogue insert of the same kind already covers (mostly) the same footprint."""
    footprint = _insert_footprint(candidate)
    if footprint is None:
        return False
    kind = candidate["_catalogue"]["kind"]  # type: ignore[index]
    for entities in layers.values():
        for entity in entities:
            if entity.get("type") != "insert" or "_catalogue" not in entity or entity["_catalogue"]["kind"] != kind:
                continue
            other = _insert_footprint(entity)
            if other is not None and footprint.intersection(other).area / max(min(footprint.area, other.area), 1e-12) >= 0.30:
                return True
    return False


_KITCHEN_APPLIANCE_KINDS: dict[str, tuple[str, ...]] = {
    "stove": ("COOKTOP",), "cooktop": ("COOKTOP",), "oven": ("COOKTOP",),
    "sink": ("KITCHEN_SINK",), "kitchen sink": ("KITCHEN_SINK",),
}


def _emit_kitchen_run(
    pending: list[tuple[Any, str, str]],
    layers: dict[str, list[dict[str, Any]]],
    catalogue_inserts: list[dict[str, object]],
    grid: RasterGrid,
    height_map: np.ndarray,
    theta: float,
    wall_lines: list[LineString],
    units_per_metre: float,
    metrics: dict[str, Any],
    settings: KitchenRunSettings,
) -> int:
    """Write the kitchen run (straight / L / U polygon on A-EINB) and its appliance inserts.

    Cooktop and sink blocks are placed only where a fine label observed that appliance.
    Without a wall-based run the counters fall back to individually fitted footprints.
    """
    if not pending:
        return 0
    union = np.zeros(pending[0][0].mask.shape, dtype=bool)
    for instance, _label, _display in pending:
        union |= instance.mask
    run = build_kitchen_run(union, grid, wall_lines, units_per_metre, settings, height_map)
    emitted = 0
    if run is not None:
        metrics["kitchen_run_blobs_absorbed"] = _absorb_blobs_in_counter(layers, run.pieces)
        legs = [[[float(x), float(y)] for x, y in list(leg.polygon().exterior.coords)[:-1]] for leg in run.legs]
        for piece in run.pieces:
            layers["A-EINB"].append({"type": "polyline", "points": _polygon_points(piece), "closed": True, "holes": [], "_einb_kind": "fine_kitchen", "_kitchen_run": run.shape, "_run_legs": legs})
            layers["A-EINB"].append({"type": "text", "text": "Küche", "point": list(piece.representative_point().coords[0]), "height": 0.10 * units_per_metre, "rotation": 0.0})
            emitted += 1
        for instance, label_key, display in pending:
            kinds = _KITCHEN_APPLIANCE_KINDS.get(label_key)
            if kinds is None:
                continue
            rows, cols = np.nonzero(instance.mask)
            appliance_points = np.asarray([grid.pixel_to_local(float(row), float(col)) for row, col in zip(rows, cols, strict=True)])
            placed = appliance_insert(run, kinds, appliance_points, units_per_metre, display)
            if placed is not None and not _insert_duplicates_existing(placed, layers):
                layers["A-EINB"].append(placed)
                catalogue_inserts.append(placed)
        metrics["kitchen_run_shape"] = run.shape
        return emitted
    for instance, label_key, display in pending:
        fit = fit_footprint(instance.mask, grid, height_map, theta, units_per_metre, fine_label=True, semantic_label=instance.label)
        if fit is None:
            continue
        layers["A-EINB"].extend([
            {"type": "polyline", "points": _polygon_points(fit.polygon), "closed": True, "holes": [], "_einb_kind": "fine_kitchen"},
            {"type": "text", "text": "Küche" if label_key in {"kitchen counter", "kitchen cabinet"} else display, "point": list(fit.polygon.representative_point().coords[0]), "height": 0.10 * units_per_metre, "rotation": 0.0, "_height_m": instance.height / max(units_per_metre, 1e-9)},
        ])
        emitted += 1
    return emitted


def _absorb_blobs_in_counter(layers: dict[str, list[dict[str, Any]]], pieces: list[Polygon], share: float = 0.50) -> int:
    """Remove traced A-FURN blobs (objects standing on the counter) that lie >= ``share`` inside the counter footprint."""
    counter = unary_union(pieces)
    removed: list[Polygon] = []
    kept: list[dict[str, Any]] = []
    for entity in layers.get("A-FURN", []):
        if entity.get("type") == "polyline" and entity.get("closed") and len(entity.get("points", [])) >= 3:
            blob = Polygon(entity["points"]).buffer(0)
            if not blob.is_empty and blob.area > 0.0 and blob.intersection(counter).area >= share * blob.area:
                removed.append(blob)
                continue
        kept.append(entity)
    if not removed:
        return 0
    layers["A-FURN"] = kept
    zone = unary_union(removed).buffer(1e-6)
    for name in ("A-ANNO", "A-FURN-DETL"):
        if name in layers:
            layers[name] = [
                entity for entity in layers[name]
                if not (
                    (name == "A-ANNO" and entity.get("type") == "text" and " h=" in str(entity.get("text", "")) and zone.covers(Point(entity["point"][:2])))
                    or (name == "A-FURN-DETL" and entity.get("type") == "polyline" and len(entity.get("points", [])) >= 2 and zone.covers(LineString(entity["points"])))
                )
            ]
    return len(removed)


def _kitchen_zone(rooms: list[Polygon], fine_classes: np.ndarray, fine_names: list[str], grid: RasterGrid, inserts: list[dict[str, object]], units_per_metre: float, radius_m: float) -> Any | None:
    """Rooms (or a disc around the evidence) that contain kitchen fine labels or kitchen appliances."""
    names = {str(name).strip().lower() for name in ISLAND_LABELS}
    ids = [index + 1 for index, name in enumerate(fine_names) if str(name).strip().lower() in names]
    points: list[Point] = []
    if ids:
        rows, cols = np.nonzero(np.isin(fine_classes, ids))
        step = max(1, len(rows) // 400)
        points.extend(Point(*grid.pixel_to_local(float(row), float(col))) for row, col in zip(rows[::step], cols[::step], strict=True))
    points.extend(Point(float(item["point"][0]), float(item["point"][1])) for item in inserts if item["_catalogue"]["kind"] in {"COOKTOP", "KITCHEN_SINK"})  # type: ignore[index]
    if not points:
        return None
    containing = [room for room in rooms if any(room.contains(point) for point in points)]
    if containing:
        return unary_union(containing)
    return unary_union([point.buffer(radius_m * units_per_metre) for point in points])


def _island_appliances(island: Any, fine_classes: np.ndarray, fine_names: list[str], grid: RasterGrid, units_per_metre: float) -> list[dict[str, object]]:
    """Cooktop / sink inserts embedded in an island, from fine-label regions inside its polygon."""
    output: list[dict[str, object]] = []
    spec = {"cooktop": ("COOKTOP",), "stove": ("COOKTOP",), "sink": ("KITCHEN_SINK",), "kitchen sink": ("KITCHEN_SINK",)}
    minx, miny, maxx, maxy = island.polygon.bounds
    r0, c0 = grid.local_to_pixel(minx, maxy)
    r1, c1 = grid.local_to_pixel(maxx, miny)
    r0, r1 = max(0, int(r0) - 1), min(grid.height, int(r1) + 2)
    c0, c1 = max(0, int(c0) - 1), min(grid.width, int(c1) + 2)
    direction = np.asarray((math.cos(island.angle_rad), math.sin(island.angle_rad)))
    across = np.asarray((-direction[1], direction[0]))
    for index, name in enumerate(fine_names):
        kinds = spec.get(str(name).strip().lower())
        if kinds is None:
            continue
        rows, cols = np.nonzero(fine_classes[r0:r1, c0:c1] == index + 1)
        points = np.asarray([grid.pixel_to_local(float(r + r0), float(c + c0)) for r, c in zip(rows, cols, strict=True)])
        points = np.asarray([point for point in points if island.polygon.buffer(0.02 * units_per_metre).contains(Point(float(point[0]), float(point[1])))]) if len(points) else points
        if len(points) < max(6, int(0.04 * units_per_metre**2 / grid.pixel_size**2)):
            continue
        u, v = points @ direction, points @ across
        centre = direction * (u.min() + u.max()) / 2.0 + across * (v.min() + v.max()) / 2.0
        measured: dict[str, object] = {"type": "insert", "name": "FURNITURE", "point": centre.tolist(), "rotation": math.degrees(island.angle_rad) % 180.0, "xscale": float(u.max() - u.min() + grid.pixel_size), "yscale": float(v.max() - v.min() + grid.pixel_size), "attributes": {"LABEL": str(name)}}
        placed = centred_catalogue_insert(measured, kinds, units_per_metre, tolerance=0.40)
        if placed is not None:
            placed["_island_appliance"] = True
            output.append(placed)
    return output


def _furniture_label_text(display: str, label_key: str, height: float) -> str:
    """Furniture label "Name h=0.76"; a chair is a seat-sized box and gets its name only ("Stuhl")."""
    return display if label_key == "chair" else f"{display} h={height:.2f}"


TALL_UNIT_FINE_LABELS = frozenset({"refrigerator", "fridge", "oven"})
TALL_UNIT_IF_HIGH_FINE_LABELS = frozenset({"kitchen cabinet"})


# Fine labels of kitchen units: a wall run lying on them without wall-class points is a cabinet front (double_walls._cabinet_wall_runs).
CABINET_FINE_LABELS = frozenset({"kitchen cabinet", "kitchen counter", "stove", "refrigerator", "kitchen sink", "oven", "dishwasher", "microwave oven"})


def tall_unit_evidence(
    fine_classes: np.ndarray, fine_names: list[str], classes: np.ndarray, height_map: np.ndarray, units_per_metre: float,
    tall_m: float = 1.6, low_m: float = 1.5,
) -> TallUnitEvidence:
    """Rasters for the tall built-in unit test of the wall stage: kitchen cabinets standing at full height
    (>= ``tall_m``) and fridges / ovens are unit evidence; observed surfaces between 0.05 m and ``low_m`` are
    low surfaces (counter tops, furniture) that never lie inside a floor-to-ceiling unit."""
    names = [str(name).strip().lower() for name in fine_names]
    heights = np.nan_to_num(np.asarray(height_map, dtype=float), nan=0.0)
    fridge_ids = [index + 1 for index, name in enumerate(names) if name in TALL_UNIT_FINE_LABELS]
    high_ids = [index + 1 for index, name in enumerate(names) if name in TALL_UNIT_IF_HIGH_FINE_LABELS]
    fridge = np.isin(fine_classes, fridge_ids) & (heights >= 0.5 * units_per_metre)
    front = fridge | (np.isin(fine_classes, high_ids) & (heights >= tall_m * units_per_metre))
    low = (np.asarray(classes) != 0) & (heights > 0.05 * units_per_metre) & (heights < low_m * units_per_metre)
    cabinet_ids = [index + 1 for index, name in enumerate(names) if name in CABINET_FINE_LABELS]
    cabinet = np.isin(fine_classes, cabinet_ids) if cabinet_ids else np.zeros(np.shape(fine_classes), dtype=bool)
    return TallUnitEvidence(front=front, fridge=fridge, low=low, cabinet=cabinet)


def _measured_unit_height_m(shape: Polygon, height_of: Callable[[np.ndarray], np.ndarray] | None, units_per_metre: float) -> float | None:
    """Height of a tall unit in metres: the 90th percentile of the measured surface heights over its footprint (None below 1.6 m or unmeasured)."""
    if height_of is None or shape.is_empty:
        return None
    step = 0.04 * units_per_metre
    x0, y0, x1, y1 = shape.bounds
    xs, ys = np.meshgrid(np.arange(x0 + step / 2, x1, step), np.arange(y0 + step / 2, y1, step))
    inside = shapely.contains_xy(shape, xs, ys)
    if not inside.any():
        return None
    values = np.asarray(height_of(np.column_stack((xs[inside], ys[inside]))), dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 10:
        return None
    height = float(np.percentile(values, 90)) / max(units_per_metre, 1e-9)
    return round(height, 2) if height >= 1.6 else None


def _emit_tall_units(layers: dict[str, list[dict[str, Any]]], units: list[tuple[Polygon, str]], units_per_metre: float, height_of: Callable[[np.ndarray], np.ndarray] | None = None) -> int:
    """Write the tall units of the wall stage to A-EINB (outline, cross, KS/HS); returns absorbed objects."""
    absorbed = 0
    for shape, _label in units:
        layers["A-EINB"], count = _absorb_einb(layers.get("A-EINB", []), shape)
        absorbed += count + _absorb_furniture(layers, shape, shape)
    for shape, label in units:
        meta: dict[str, Any] = {"_wall_stage": True}
        measured = _measured_unit_height_m(shape, height_of, units_per_metre)
        if measured is not None:
            meta["_height_m"] = measured
        layers["A-EINB"].extend(tall_unit_entities(shape, label, units_per_metre, **meta))
    return absorbed


def _absorb_chairs_at_dining_tables(layers: dict[str, list[dict[str, Any]]], units_per_metre: float) -> None:
    """Drop traced chairs that the TABLE_DINING block already draws around its table."""
    zones = []
    for entity in layers.get("A-FURN", []):
        if entity.get("type") == "insert" and str(entity.get("name", "")).startswith("TABLE_DINING"):
            footprint = _insert_footprint(entity)
            if footprint is not None:
                zones.append(footprint.buffer(0.30 * units_per_metre))
    if not zones:
        return
    chair_labels = {"Stuhl", "Chair", "Sessel"}
    dropped: list[list[float]] = []
    kept: list[dict[str, Any]] = []
    for entity in layers["A-FURN"]:
        if entity.get("type") == "polyline" and entity.get("closed") and entity.get("_label") in chair_labels and len(entity.get("points", [])) >= 3:
            centroid = Polygon(entity["points"]).centroid
            if any(zone.contains(centroid) for zone in zones):
                dropped.append([centroid.x, centroid.y])
                continue
        kept.append(entity)
    layers["A-FURN"] = kept
    if dropped:
        layers["A-ANNO"] = [
            entity for entity in layers["A-ANNO"]
            if not (entity.get("type") == "text" and str(entity.get("text", "")).split(" h=")[0] in chair_labels and any(math.dist(entity["point"][:2], point) <= 0.05 * units_per_metre for point in dropped))
        ]


def _kitchen_runs(einb: list[dict[str, Any]]) -> list[Polygon]:
    """Closed A-EINB polygons that belong to the kitchen (fine kitchen counter, island or Küche label)."""
    texts = [entity for entity in einb if entity.get("type") == "text"]
    runs: list[Polygon] = []
    for entity in einb:
        if entity.get("type") != "polyline" or not entity.get("closed") or len(entity.get("points", [])) < 3:
            continue
        polygon = Polygon(entity["points"]).buffer(0)
        if not isinstance(polygon, Polygon) or polygon.is_empty:
            continue
        if entity.get("_run_legs"):
            runs.extend(Polygon(leg) for leg in entity["_run_legs"])
            continue
        kitchen = entity.get("_einb_kind") in {"fine_kitchen", "island"} or any(
            str(text.get("text", "")).split(" h=")[0].strip() in {"Küche", "Kücheninsel"}
            and (text.get("_einb_owner") == entity.get("_einb_id") and entity.get("_einb_id") is not None or polygon.distance(Point(text["point"])) <= 1e-6)
            for text in texts
        )
        if kitchen:
            runs.append(polygon)
    return runs


def _collect_room_tags(layers: dict[str, list[dict[str, Any]]], units_per_metre: float) -> list[tuple[tuple[float, float], str]]:
    """Semantic fixture / furniture tags with a position inside the room they stand in."""
    tagged: list[tuple[tuple[float, float], str]] = []
    for name in ("P-SANI", "A-EINB", "A-FURN"):
        for entity in layers.get(name, []):
            if entity.get("type") == "insert":
                tag = tag_for_name(str(entity["name"]))
                footprint = _insert_footprint(entity)
                if tag and footprint is not None:
                    centre = footprint.centroid
                    tagged.append(((float(centre.x), float(centre.y)), tag))
            elif entity.get("type") == "polyline" and entity.get("closed") and len(entity.get("points", [])) >= 3:
                label = entity.get("_label") or ("Küche" if entity.get("_einb_kind") in {"fine_kitchen", "island"} else None)
                tag = tag_for_name(str(label)) if label else None
                if tag:
                    centre = Polygon(entity["points"]).centroid
                    tagged.append(((float(centre.x), float(centre.y)), tag))
            elif entity.get("type") == "text" and name == "A-EINB":
                tag = tag_for_name(str(entity.get("text", "")))
                if tag:
                    tagged.append(((float(entity["point"][0]), float(entity["point"][1])), tag))
    for entity in layers.get("A-ANNO", []):
        if entity.get("type") == "text" and " h=" in str(entity.get("text", "")):
            tag = tag_for_name(str(entity["text"]))
            if tag:
                tagged.append(((float(entity["point"][0]), float(entity["point"][1])), tag))
    return tagged


def _grid_for_slice(grid: RasterGrid, component_slice: tuple[slice, slice]) -> RasterGrid:
    row_slice, col_slice = component_slice
    return RasterGrid(
        min_x=grid.min_x + col_slice.start * grid.pixel_size,
        max_y=grid.max_y - row_slice.start * grid.pixel_size,
        pixel_size=grid.pixel_size,
        width=col_slice.stop - col_slice.start,
        height=row_slice.stop - row_slice.start,
    )


def _polygon_points(polygon: Polygon) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in list(polygon.exterior.coords)[:-1]]


def _ring_points(coordinates: Any) -> list[list[float]]:
    return [[float(x), float(y)] for x, y, *_rest in list(coordinates)[:-1]]


def _layer_presentation(layers: tuple[LayerDefinition, ...]) -> tuple[dict[str, dict[str, Any]], dict[str, int], dict[str, str], dict[str, bool]]:
    styles: dict[str, dict[str, Any]] = {}
    colors: dict[str, int] = {}
    linetypes: dict[str, str] = {}
    hatches: dict[str, bool] = {}
    for layer in layers:
        colors[layer.name] = layer.color
        linetypes[layer.name] = layer.linetype
        hatches[layer.name] = layer.hatch
        style = {key: value for key, value in {"true_color": layer.true_color, "lineweight": layer.lineweight, "hatch_pattern": layer.hatch_pattern, "hatch_scale": layer.hatch_scale}.items() if value is not None}
        if style:
            styles[layer.name] = style
    return styles, colors, linetypes, hatches


def _add_ortho_plan_layer_presentation(styles: dict[str, dict[str, Any]], colors: dict[str, int], linetypes: dict[str, str], hatches: dict[str, bool]) -> None:
    data = load_yaml(CONFIG_DIRECTORY / "ortho_plan.yaml") or {}
    for name, values in data.get("layers", {}).items():
        if not isinstance(values, dict):
            continue
        colors.setdefault(name, int(values.get("color", 7)))
        linetypes.setdefault(name, str(values.get("linetype", "CONTINUOUS")))
        hatches.setdefault(name, False)
        style = styles.setdefault(name, {})
        if "true_color" in values:
            style["true_color"] = tuple(int(value) for value in values["true_color"])
        if "lineweight" in values:
            style["lineweight"] = int(values["lineweight"])
        if "hatch_pattern" in values:
            style["hatch_pattern"] = str(values["hatch_pattern"])
        if "hatch_scale" in values:
            style["hatch_scale"] = float(values["hatch_scale"])


def _clip_neighbours_to_regularized_buildings(
    partition: dict[str, list[Polygon]],
    regularized_buildings: list[Polygon],
) -> None:
    """Clip neighbouring partition polygons against regularised building footprints.

    Design decision for shared borders:
    When building footprints are regularised, their geometry snaps to dominant orthogonal
    orientations. To prevent overlaps and maintain exact shared planar boundaries:
    1. Any area freed up where the original building footprint shrank is assigned to the
       adjacent neighbour sharing the boundary.
    2. Neighbour polygons are clipped with a boolean difference against the regularised
       building footprint.
    3. The inner boundary of enclosing neighbours matches the regularised building boundary
       exactly vertex-for-vertex, satisfying ``shapely.coverage_is_valid``.
    """
    orig_bldgs = partition.get("S-BLDG", [])
    if not orig_bldgs or not regularized_buildings:
        partition["S-BLDG"] = regularized_buildings
        return

    bldg_orig_union = unary_union([make_valid(p) for p in orig_bldgs])
    bldg_reg_union = unary_union([make_valid(p) for p in regularized_buildings])
    partition["S-BLDG"] = list(regularized_buildings)

    recipients: list[tuple[float, str, int]] = []
    for layer_name, polygons in partition.items():
        if layer_name == "S-BLDG":
            continue
        for index, polygon in enumerate(polygons):
            shared = polygon.boundary.intersection(bldg_orig_union.boundary).length
            if shared > 1e-7:
                recipients.append((shared, layer_name, index))
    recipient = max(recipients, default=(0.0, "", -1))

    for layer_name, polygons in list(partition.items()):
        if layer_name == "S-BLDG" or not polygons:
            continue
        updated_polys: list[Polygon] = []
        for index, poly in enumerate(polygons):
            if not poly.intersects(bldg_reg_union) and not poly.intersects(bldg_orig_union):
                updated_polys.append(poly)
                continue

            current = make_valid(poly)
            if (layer_name, index) == (recipient[1], recipient[2]):
                # A regularised building can shrink away from a jagged source
                # boundary.  The adjoining area must receive that exact gap
                # before the new building is subtracted, otherwise the old hole
                # remains and coverage validation correctly reports a mismatch.
                current = make_valid(current.union(bldg_orig_union))

            diff = current.difference(bldg_reg_union)
            if diff.is_empty:
                continue

            geoms = list(diff.geoms) if hasattr(diff, "geoms") else [diff]
            for geom in geoms:
                if isinstance(geom, Polygon) and geom.area > 1e-4:
                    updated_polys.append(geom)

        partition[layer_name] = updated_polys


def _resolve_point_block(layer_name: str, class_name_str: str) -> str:
    """Determine the block name for point objects, checking light/lamp class names."""
    if "light" in class_name_str or "lamp" in class_name_str:
        return "LIGHT"
    if layer_name == "S-MANH":
        return "MANHOLE"
    if layer_name == "S-POLE":
        return "POLE"
    if layer_name == "S-SIGN":
        return "SIGN"
    return "POLE"


def _dominant_layer_display_name(
    layer_def: LayerDefinition,
    segment_classes: np.ndarray,
    metadata: dict[str, Any] | None,
) -> str:
    """Return the display label for an area layer using the dominant class name."""
    if not layer_def.codes:
        return class_display_name(layer_def.name)

    dominant_code = layer_def.codes[0]
    if len(layer_def.codes) > 1:
        dominant_code = max(layer_def.codes, key=lambda c: int(np.count_nonzero(segment_classes == c)))

    cyclone_info = load_cyclone_codes().get(dominant_code)
    if cyclone_info and "name" in cyclone_info:
        return class_display_name(cyclone_info["name"])

    return class_display_name(layer_def.name)


def _find_point_source(project: Path, stats: dict[str, Any]) -> Path | None:
    """Locate point cloud source from ortho metadata or project products."""
    source_paths = stats.get("source_paths", {})
    if "point_source" in source_paths:
        candidate = Path(source_paths["point_source"])
        if candidate.is_file():
            return candidate

    classified_dir = project / "products" / "classified"
    las_candidate = classified_dir / "classified.las"
    if las_candidate.is_file():
        return las_candidate
    ply_candidate = classified_dir / "classified.ply"
    if ply_candidate.is_file():
        return ply_candidate
    return None
