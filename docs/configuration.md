# Configuration reference

SAMcloud separates global defaults from the settings saved in each project.
Global configuration files are stored in `config/`; project-specific choices
are written to `projects/<project-name>/project.samcloud.yml`. A new project
receives a copy of `defaults.yml`; only the settings pages and the project
dialog change values afterwards.

| File | Section |
|---|---|
| `defaults.yml` | [project defaults](#defaultsyml) |
| `colmap.yml` | [COLMAP, dense, step 6](#colmapyml) |
| `export_crs.yml` | [LAS export CRS presets](#export_crsyml) |
| `sam.yml`, `flurseg.yml`, `classes.json` | [classification](#samyml) |
| `geometry.yml` | [geometric refinement](#geometryyml) |
| `viewer.yml` | [viewer](#vieweryml) |
| `dxf_export.yaml`, `dxf_layers.yaml`, `cyclone_codes.yaml`, `model_profiles/` | [DXF export and Cyclone codes](#dxf_exportyaml) |
| `ortho.yaml`, `ortho_segment.yaml`, `ortho_plan.yaml` | [orthophoto and clean plan](#orthoyaml) |

## `defaults.yml`

| Setting | Meaning |
|---|---|
| `application.projects_directory` | Local folder holding reopenable SAMcloud projects. |
| `project.image_storage` | `external_reference` keeps source photos in their original location. |
| `coordinate_systems.input_crs` | CRS of image GPS data, normally `EPSG:4326` for drone EXIF. |
| `coordinate_systems.working_crs` | Internal coordinate frame. `LOCAL` is valid before georeferencing. |
| `coordinate_systems.export_crs` | Default LAS export preset key from `export_crs.yml`. |
| `workflow.use_gpu` | Enables GPU use for supported COLMAP and SAM3 stages. |
| `workflow.control_points_mode` | `none` runs through classification automatically; `after_alignment` pauses for manual control points; `automatic` detects targets and pauses only when the quality gate fails. |
| `control_points.crs` | CRS for imported GCP coordinates; default `EPSG:2056+5728` is LV95 horizontal coordinates with LN02 height. |
| `control_points.min_points` | Minimum usable GCP count required to resume processing; default three. |
| `control_points.min_observations_per_point` | Images in which a point must be marked; default two. |
| `control_points.max_reprojection_px` | Maximum RMS reprojection error accepted while triangulating a marked GCP. |
| `control_points.min_triangulation_angle_deg` | Minimum view angle accepted for a triangulated GCP. |
| `control_points.max_residual_warning_m` | Residual threshold shown as a warning in the GCP report. |
| `control_points.detection.*` | Automatic target detection (`samcloud/control_points/targets.py`): `target_type` (`cross` or `circle`), `polarity` (`bright`, `dark`, or `auto` for both), `min_contrast`, `min_score`, `max_saturation` (HSV limit for white targets), `ambiguity_ratio` (runner-up must score below this share of the best). |
| `control_points.min_projections` | Images a `pass` point needs in the automatic quality gate (`control` points need two). |
| `control_points.max_marker_reprojection_px` | Marker error limit of the automatic gate (error to the estimated marker position, as in Agisoft). |
| `control_points.merge_tolerance_m` | A target counts as `accepted` when its triangulated position lies within this distance (m) of the corrected prediction (at least 1 m for `control` points); otherwise it needs review. |
| `control_points.search_radius_min_px`, `search_radius_max_px` | Bounds of the pixel search window around the predicted target position in the first pass. |
| `control_points.stage2_radius_px` | Search radius of the second pass, after the prediction has been refined. |
| `control_points.observation_outlier_factor` | An image click whose pixel residual exceeds this factor times the median residual of the marker is disabled. |
| `control_points.max_gate_iterations` | Maximum detect-adjust-gate rounds of the automatic mode. |
| `apriltags.enabled` | Search AprilTags for scale. The project dialog sets it per project (on for Indoor, off for Outdoor). |
| `apriltags.outlier_pct` | A tag whose scale factor deviates more than this percentage from the median is ignored (default 3). |
| `apriltags.last_printed_id` | Highest tag ID printed from this project; the print dialog starts at the next ID (-1 = none). |
| `classification.geometry_refinement` | Switches the geometric refinement on or off. |
| `classification.class_set` | `outdoor` or `indoor` class set. |
| `classification.engine` | `sam3` or `flurseg`. FlurSeg only applies to the outdoor class set. |
| `classification.min_votes`, `tile_size`, `tile_min_overlap`, `mode`, `background_threshold`, `query_threshold`, `threshold_sweep` | Per-project copies of the SAM3 inference settings of `sam.yml`. |
| `classification.instances.*` | Instance association settings, see [`sam.yml`](#samyml). |
| `acquisition.panorama.frame_interval_seconds` | Default fixed interval used to extract frames from a 360 degree MP4 video. |
| `acquisition.panorama.cubemap_face_size` | Resolution of the perspective cubemap views generated for COLMAP; `0` uses the native size (panorama width / 4), larger values are capped at native. |
| `acquisition.panorama.supported_image_extensions`, `supported_video_extensions` | Accepted panorama file types (`.jpg`, `.jpeg`; `.mp4`). |
| `viewer.sparse_cleaning.*` | Display-only clean-up of the sparse cloud (the COLMAP model is not changed): `min_track_length`, `max_reprojection_error`, `min_triangulation_angle_deg`, `max_camera_distance_factor`, `min_neighbors`, `min_points` (smaller clouds are shown unfiltered), `min_keep_ratio` (never hide more than 80 %). |

The keys `application.language`, `project.keep_intermediate_files`,
`workflow.alignment_mode`, `workflow.georeferencing_mode`,
`workflow.export_format`, `control_points.detection.secondary_target_type`, and
`control_points.continue_automatically` are stored with each project, but no
code reads them at the moment. A `storage` block (what to keep after a run) and
the `refinement` block (Depth-Anything and surface fusion switches) are added
to the project file by the dialog and the settings page; they are not part of
`defaults.yml`.

## `colmap.yml`

This file defines camera assumptions, matching, RTK/GPS accuracy, control-point
requirements, dense reconstruction, and the step 6 stages. It is split into
sections.

| Section | Content |
|---|---|
| `camera` | `camera_model` (`AUTO` resolves per capture type), `grouping` (cameras are grouped per flight by sequence number), `max_image_size`. |
| `features` | COLMAP SIFT extraction, see below. |
| `alignment` | `matching_strategy` (`auto`), `spatial_max_num_neighbors`, `spatial_max_distance_m`, `use_gpu`, `mapper` (`pose_prior` uses the per-image RTK/GPS covariances), `gravity_align_panorama`. |
| `tiepoint_filter` | Tie point filter passes (`max_reproj_error`, `min_tri_angle`, `min_track_len`) applied before the GPS alignment or control points. |
| `georeferencing` | RTK and standard GPS accuracy limits and prior standard deviations; `control_point_minimum_observations` is two because a point must be marked in at least two images. |
| `dense_reconstruction` | PatchMatch and fusion settings plus the step 6 blocks below. |

The `features` section controls COLMAP SIFT extraction. `max_num_features` remains
the limit for every capture type. For `panorama_360` only,
`panorama_peak_threshold` (default `0.003`) and `panorama_edge_threshold` (default
`20`) are passed to COLMAP to retain sufficient cubemap-face features; they do not
affect `drone_gps` or `camera` projects.

`alignment.gravity_align_panorama` defaults to `true`. For horizon-levelled
`panorama_360` inputs, SAMcloud derives gravity from the calibrated cubemap rig,
rotates the final sparse model so the up face points to `+Z`, and uses that model
for dense reconstruction. The resulting report records the arbitrary scale,
rig-pose spread, and sparse horizontal-plane cross-check. When AprilTags
georeference the model, this step is skipped.

### Dense reconstruction and step 6

`dense_reconstruction.max_image_size: auto` derives the dense image size from
the image size with `auto.ratio`, `minimum`, and `maximum`. `patch_match`
overrides COLMAP's PatchMatch settings (empty means COLMAP defaults);
`fusion` holds the fusion thresholds. The following blocks belong to
[workflow step 6](../README.md#improve--filter-step-6):

| Block | Applies to | Purpose |
|---|---|---|
| `point_cleaning` | all capture types | removes points that at least `min_free_views` cameras see through (`free_space_margin`, `support_tolerance`, `free_ratio`); inactive when indoor surface fusion's own visibility check runs |
| `depth_fill` | indoor | Depth-Anything gap fill confirmed by neighbouring views (`neighbors`, `tolerance`, `min_baseline_m`, `max_depth`, `max_incidence_deg`, `min_agreeing_views`); `local_alignment` and `use_sparse` realign the prior per image tile; `units_per_metre: auto` derives the scale of unscaled indoor projects from the storey height versus `nominal_storey_height_m`. The options `use_edges`, `halo_filter`, `anchor_rule`, and `plane_completion` are off (no measured gain). |
| `surface_fusion` | indoor, switchable per project | TSDF fusion of all depth maps (`voxel_m`, `truncation_m`, `geometric_weight`, `prior_weight`), `mode: snap` (default) or `extract`, plane snapping (`plane_classes`, `plane_snap_m`, `plane_merge_m`, Manhattan directions), visibility check, `depth_edge_filter`, `mirror_filter`, optional `hidden_filter` and `behind_wall_filter` (both off), and class coherence for wall, ceiling, floor, furniture, door, window. |
| `superpoint_vote` | indoor | majority vote of broad classes inside geometry/colour superpoints (`majority_share`, `protect_min_points`, `relabel_classes`) |
| `plane_fill` | indoor | grid fill of fitted planes, disabled |
| `capture_profiles` | per capture type | `drone_gps`, `camera`, and `panorama_360` each hold overrides for the blocks above; the 360 profile sets `patch_match: {window_step: 2}` |

Inline comments in `config/colmap.yml` record why individual values were chosen.

## `export_crs.yml`

This file defines the LAS export CRS choices shown in the settings page. Each
preset has a stored key, display label, PROJ-readable target, height-system
description, and optional `required_grids`. `LOCAL`, `EPSG:4326`, and
`EPSG:2056` are the standard presets. Add other CRSs here and install every
listed geoid or transformation grid yourself in the PyProj data directory;
SAMcloud rejects missing grids and never uses ballpark transformations. Restart
SAMcloud after changing this file.

## `sam.yml`

This file controls the SAM3 weight location (`model.weights`, default
`models/sam3.pt`, excluded from Git), prompt confidence threshold, tile size,
overlap, and multi-view voting threshold. `inference.mode` selects the default
`segearth_ov3` head-fusion path; `legacy` remains available for comparison.
For SegEarth-OV3, `background_threshold` is the per-point background threshold,
`query_threshold` filters decoder queries, and `threshold_sweep` can list A/B
thresholds. The paper does not publish a fixed tau, so choose it per dataset.
`output` selects which products are written (`write_ply`, `write_las`).

The `instances` block controls the per-point object instances
([instances.md](instances.md)); the project copy lives under
`classification.instances` in `defaults.yml`:

| Key | Meaning |
|---|---|
| `enabled` | Compute `instance_id`; `--no-instances` on the pipeline overrides it. |
| `query_threshold` | Minimum SAM3 query confidence of an instance mask. |
| `mask_threshold` | Mask probability that counts as inside the object. |
| `min_mask_pixels` | Smallest mask kept per view. |
| `merge_overlap` | Two same-class per-view masks are linked when they share at least this fraction of the smaller mask's points. |
| `min_shared_points` | Minimum number of shared points for two masks to be linked. |
| `min_views` | Distinct views in which a point must belong to the winning object. |
| `min_points` | Smallest instance kept after refinement. |
| `split_radius_m` | Linking radius (m) used to split a linked mask group into spatially connected objects. |
| `debug_ply` | Also write `classified_instances.ply` with one colour per instance. |

## `flurseg.yml`

FlurSeg is the project's own EoMT-DINOv3-L semantic segmentation model (trained on
SAM3 pseudo-labels; published at <https://huggingface.co/AIhdproduction/Flurseg>). In
the project settings it can replace SAM3 for the **outdoor** class set
(`classification.engine: flurseg`); indoor projects always use SAM3. The file holds
the model folder (`model.directory`), the sliding window (`window_size` 768,
`stride` 384), `batch_size`, `precision` (`fp16` or `fp32`), and `class_mapping`
from the 11 FlurSeg classes to the SAMcloud outdoor classes (`other` maps to
unknown). FlurSeg uses the same weighted multi-view voting, confidence scores
and step 6 as SAM3. It needs `transformers` and `scipy` (in `requirements.txt`).

"Reclassify" on the dashboard (pipeline `--reclassify`) re-runs only the
classification and step 6 on the kept dense cloud, recreating the undistorted
images if compaction released them. This allows a SAM3/FlurSeg comparison on
exactly the same points.

The FlurSeg model folder is not part of the repository (`models/` is ignored).
See the README for the download. In the Grütistrasse comparison it is about 15x
faster per image than SAM3, with 93 % agreement in FlurSeg's 11-class taxonomy.

### Project class customisation (SAM3 only)

When a project is created with SAM3 (outdoor or indoor), the "Klassen anpassen" panel lets you
edit the prompts of the standard classes and add own classes (English name, German label,
prompts, colour). The result is written to the project's own `config/classes.json`, which the
pipeline (`--classes-json`), the LAS export and the dense viewer then use instead of the global
file. Standard classes keep their ids and names because step 6 and the geometry rules address
them by name; own classes are appended and get free LAS codes from 64. FlurSeg's classes are
fixed by its training and cannot be customised.

## `classes.json`

This file provides separate ordered prompt lists for `outdoor` and `indoor`
classification. The project setting `classification.class_set` chooses one of
them before a SAM3 run. The LAS exporter maps the selected IDs to
Cyclone-compatible LAS codes, including standard ASPRS codes where available
and custom codes for project-specific classes.

| Key | Meaning |
|---|---|
| `unknown_class_id`, `unknown_class_name` | ID 0, *Unknown/noise*. |
| `prompts`, `indoor`, `outdoor` | Ordered prompt list per class set (comma-separated synonyms per class; every prompt is sent to SAM3 on its own). |
| `las_codes`, `colors`, `labels_de` | LAS code, viewer colour, and German label per class and set. |
| `suppress` | Prompts per class set whose winning points become Unknown/noise. Indoor: curtains. |
| `mirror` | Mirror prompts per class set, used by the mirror detection of step 6. |
| `instance_classes` | Classes that receive instance IDs (walls, floors and ceilings are never listed). |
| `instance_label_aliases` | Maps a SAM prompt text to its canonical object label before instances are merged. |
| `instance_labels_de` | German object names for the viewer card and the DXF labels. |
| `fine_terms`, `fine_aliases`, `fine_term_classes` | Fine vocabulary (sofa, bed, ...) stored per point as `fine_term`, aliases, and the class each term belongs to. |

## `geometry.yml`

Optional geometric post-processing for georeferenced outdoor classifications.
It builds a robust terrain model, measures height above ground and local
flatness, and applies per-class plausibility rules. When SegEarth-OV3 is used,
the pipeline also stores `classified_scores.npz` and supplies its top-one and
top-two confidence scores to refinement. `confidence.protect_above` protects
high-confidence SAM labels, `dtm_min_confidence` excludes uncertain ground from
terrain estimation, and `second_best_min` enables a confident SAM second choice
before neighbour reassignment. The `support` block controls the local horizontal
surface used to keep low vegetation and roof railings on roofs. `flat_roof.strong_min_hag` always
converts tall, planar roofs, while its ambiguous band respects SAM confidence.
The curb rule requires nearby valid road or sidewalk points; the overhead power
line rule rejects planar elevated patches. It is enabled by default through
`classification.geometry_refinement` and can be switched off in the settings;
`classified_raw.ply` remains next to the refined output for A/B comparison.
Thresholds are in metres and should be tuned against a representative project.

## `viewer.yml`

| Key | Meaning |
|---|---|
| `viewer.engine` | `potree`; the canvas preview is used when Potree or PotreeConverter is missing. |
| `viewer.point_budget` | Potree point budget (default 5 000 000). |
| `viewer.default_display_mode` | `rgb` or class colours on opening. |
| `viewer.allow_measurements`, `allow_clipping_volumes` | Viewer features. |
| `classification.default_visibility`, `allow_isolation`, `allow_rgb_switch` | Class visibility defaults and controls. |

Per-project class colours are saved with the project.

## `dxf_export.yaml`

Global defaults for the DXF export (`samcloud/dxf/config.py` validates every
key). Project-specific choices made on the DXF page (mode, profile, scale,
alignment, levels, cell size, layer overrides, view options, plan variant) are
stored in `<project>/products/dxf/state.json` and never change this file.

| Key | Meaning |
|---|---|
| `mode_detection.indoor_share` | Share of wall/ceiling/floor points (codes 100–102) above which a LAS without provenance counts as indoor. |
| `loading.chunk_size` | Points read per LAS chunk; bounds memory use for large files. |
| `preview.max_points` | Maximum points kept in the voxel preview (`preview.npz`). |
| `preview.voxel_size` | Fixed preview voxel size in metres; `null` derives it from extent and `max_points`. |
| `plausibility.indoor_min_extent`, `indoor_max_extent` | Indoor extent range in metres; outside it the preview shows a scale warning. |
| `plausibility.outdoor_min_extent`, `outdoor_max_extent` | The same range for outdoor projects. |
| `alignment.ransac_iterations`, `ransac_distance` | RANSAC iterations and inlier distance (m) for the floor plane used to estimate gravity. |
| `alignment.min_floor_points` | Floor points needed for the RANSAC path; otherwise surface normals are used. |
| `alignment.normal_sample_size`, `normal_neighbors` | Sample size and neighbourhood for the normal-based fallback and wall yaw. |
| `alignment.align_wall_yaw` | Also estimate the yaw that puts walls on the X/Y axes (can be switched off per project on the DXF page). |
| `levels.bin_size`, `gaussian_sigma` | Height histogram bin (m) and smoothing used to find floor and ceiling peaks. |
| `levels.min_point_count`, `min_prominence` | Minimum support and peak prominence for a detected surface. |
| `levels.floor_point_tolerance` | Height tolerance (m) when counting points that support a floor. |
| `levels.min_floor_to_ceiling`, `max_floor_to_ceiling` | Accepted storey height range in metres. |
| `levels.stair_cluster_eps`, `stair_cluster_min_samples` | DBSCAN settings that group stair points before assigning them to a level. |
| `geometry.indoor_cell_size`, `outdoor_cell_size` | Raster cell size (m) for plan extraction; a project can override it on the DXF page. |
| `geometry.closing_kernel` | Morphological closing kernel (cells) before polygons are traced. |
| `geometry.simplify_tolerance` | Default polygon/line simplification tolerance (m). |
| `geometry.wall_direction_tolerance_degrees` | Angle tolerance for treating wall pieces as parallel/collinear. |
| `geometry.wall_min_thickness`, `wall_max_thickness` | Wall thickness range (m); two faces closer than the maximum become one wall. |
| `geometry.wall_merge_gap`, `wall_corner_distance` | Gaps (m) closed between wall pieces and at corners. |
| `geometry.wall_min_length` | Shortest wall segment kept (m). |
| `geometry.wall_opening_bridge` | Largest gap (m) bridged between collinear wall pieces when looking for openings. |
| `geometry.opening_max_distance`, `opening_cluster_eps` | Maximum door/window distance to a wall and clustering radius (m). |
| `geometry.skeleton_min_length` | Shortest centre-line kept for skeleton layers such as road markings and fences (m). |
| `geometry.indoor_default_height` | Storey height (m) used when a level has no ceiling. |
| `geometry.vector_grid_size` | Cells per side of the spatial grid that the review canvas uses to cull DXF entities. |
| `geometry.fill_under_canopy` | Fill ground below tree canopies in outdoor plans. |

## `dxf_layers.yaml`

`layers` is a list; each entry maps Cyclone classification codes to one DXF
layer. Layer names must be unique (except `IGNORE`). Per-layer `mode` and
`z_band` can be overridden per project on the DXF page.

| Key | Meaning |
|---|---|
| `project_mode` | `indoor` or `outdoor`: the projects this layer applies to. Older entries write the project mode into `mode` instead; they are mapped automatically. |
| `mode` | Extraction mode: `cut_band` (points within `z_band` above the level floor), `footprint` (all points projected), `top_surface`, `ground_surface`, or `ignore`. |
| `codes` | Cyclone classification codes collected into this layer (must exist in `config/cyclone_codes.yaml`). |
| `name` | DXF layer name, for example `A-WALL` or `S-GRND`. |
| `color` | AutoCAD colour index (ACI, 1–255). |
| `true_color` | Optional `[r, g, b]` colour instead of the ACI colour. |
| `lineweight` | Optional DXF lineweight (hundredths of a millimetre). |
| `linetype` | DXF linetype, normally `CONTINUOUS`. |
| `geometry` | Vectorisation: `wall`, `opening`, `polygon`, `line`, `furniture`, `skeleton`, `circle`, `tree`, `point` or `ignore`. |
| `z_band` | `[low, high]` in metres above the level floor; required for `cut_band`. |
| `min_area` | Smallest polygon kept (m²). |
| `simplify_tolerance` | Simplification tolerance for this layer (m). |
| `hatch` | Add a solid hatch to closed polygons. |

## `cyclone_codes.yaml` and `model_profiles/`

`cyclone_codes.yaml` lists the Cyclone 3DR / ASPRS classification codes and
their names. A model profile (`model_profiles/flurseg_v1.yaml`,
`legacy_v1.yaml`; the SAM3 profiles are built from `classes.json`) maps model
class IDs to Cyclone codes and states which project modes it supports and which
LAS point states are used for never-classified and unclassified points. The
LAS sidecar `*.las.profile.json` records the profile so the DXF page knows how
to read the codes. See [cyclone_classes.md](cyclone_classes.md).

## `ortho.yaml`

Defaults of the true orthophoto (DXF page job **Build orthophoto**,
[ortho_plan.md](ortho_plan.md)). Lengths are in DXF-local metres.

| Key | Meaning |
|---|---|
| `pixel_size_outdoor`, `pixel_size_indoor` | Orthophoto pixel size (m): 0.02 outdoor, 0.01 indoor. |
| `max_side` | Longest allowed orthophoto side in pixels. |
| `indoor.*` | Indoor cut: `cut_below_ceiling`, `cut_fallback_above_floor`, `floor_margin`, `ceiling_codes`. |
| `dsm.*` | Height raster: `cell_outdoor`, `cell_indoor`, hole-fill limits, and indoor `surface_fill*` (closing floor gaps from the scanned-room footprint, `laplace` or `floor_plane` method). |
| `zbuffer.*` | Visibility test per camera: voxel sizes, `max_long_side`, relative and absolute tolerances, filter radius, `cache_mb`. |
| `blend.*` | Blending of the best views per cell: `top_k`, `candidates`, `consistency_threshold`, incidence limits (outdoor 60°, indoor 82°), `feather_px`, weights, nadir penalty, and the optional 2.5D `ray_check_*` (off). |
| `tile_size`, `jpeg_long_sides`, `jpeg_quality` | Output tiling and JPEG export sizes (default 8000 and 4096 px). |
| `affine_max_residual`, `max_model_cloud_distance` | Consistency checks between camera model and cloud. |

## `ortho_segment.yaml`

Semantic segmentation of a cached orthophoto (job **Segment**): `chunk_px` and
`chunk_overlap_px` tiling, `min_image_confidence`, `pointcloud_confidence`,
`high_threshold`, `low_threshold`, the Cyclone `code_groups` that are treated as
high and low priority per mode, and `fine_allowed_codes` / `fine_prompts` for
the indoor fine labels (toilet, sofa, stove, ...).

## `ortho_plan.yaml`

Settings of the clean plan (job **Generate clean plan**,
[ortho_plan.md](ortho_plan.md)). Top-level blocks:

| Block | Purpose |
|---|---|
| `contours` | Outdoor DTM contours: `resolution`, `major_interval` (1 m), `minor_interval` (0.25 m), smoothing, `ground_codes`. |
| `labels` | Smart text labels: height factors, minimum area, shrink step. |
| `layers` | Style (colour, lineweight, linetype) of plan layers such as `S-TOPO-MAJR`, `S-TOPO-MINR`, `S-ANNO`, `S-ELEV`, `A-ANNO`, `A-FURN-DETL`, `A-EINB`, `A-AREA`, `A-OBEN`. |
| `indoor_vertical`, `indoor_ceiling` | Vertical-surface evidence for wall faces and built-ins, and room boundaries from the ceiling. |
| `clean_plan` | General limits: `segment_source`, label minimum areas, building angle tolerance, line gaps, tree height, `nominal_storey_height`, the indoor height bands (`wall`, `section`, `sill`, `lintel`) as fractions of floor-to-ceiling height, and footprint settings. |
| `indoor_island`, `indoor_niche`, `indoor_upper_cabinets`, `indoor_room_names`, `indoor_kitchen_run` | Kitchen island, recessed niche, upper cabinets (layer `A-OBEN`), semantic room names, and the kitchen run along the walls (metres, scaled by the project's units per metre). |
| `outdoor_trunk`, `outdoor_hedges`, `outdoor_spots`, `outdoor_contour_labels`, `outdoor_roof` | Surveyor-style outdoor details: trunk diameters, hedges, spot heights, contour labels, and roof ridge/eave heights. |

## Capture settings stored per project

Every new project stores an `acquisition` section. It contains the selected
capture type, source locations, georeferencing state, and whether capture
settings are locked. `drone_gps` uses GPS alignment, while `camera` and
`panorama_360` start in `LOCAL` coordinates until control points or AprilTags
are applied.

For a `panorama_360` project, `sources.panorama_video_path` and
`sources.panorama_image_directory` are independent optional inputs. At least
one is required. The `panorama` section records the fixed video frame interval,
the generated frame/cubemap locations, and the source manifest used to map
each virtual camera image back to its original panorama.

Besides the sections copied from `defaults.yml`, the project stores
`settings.apriltags`, `settings.refinement` (`depth_model`: `large`, `small`
or `off`; `depth_fill`; `surface_fusion`), and `settings.storage` (which
intermediates to keep: control point inputs, classification inputs,
refinement inputs, and the optional processing cache). When tag coordinates are
set, the tag CRS is stored in `settings.control_points.crs` and, for non-GPS
captures, becomes the export CRS.
