# Configuration reference

SAMcloud separates global defaults from the settings saved in each project.
Global configuration files are stored in `config/`; project-specific choices
are written to `projects/<project-name>/project.samcloud.yml`.

## `defaults.yml`

| Setting | Meaning |
|---|---|
| `application.projects_directory` | Local folder holding reopenable SAMcloud projects. |
| `project.image_storage` | `external_reference` keeps source photos in their original location. |
| `coordinate_systems.input_crs` | CRS of image GPS data, normally `EPSG:4326` for drone EXIF. |
| `coordinate_systems.working_crs` | Internal coordinate frame. `LOCAL` is valid before georeferencing. |
| `coordinate_systems.export_crs` | Default LAS export preset key from `export_crs.yml`. |
| `workflow.use_gpu` | Enables GPU use for supported COLMAP and SAM3 stages. |
| `workflow.control_points_mode` | `none` runs through classification automatically; `after_alignment` pauses for manual control points. |
| `control_points.crs` | CRS for imported GCP coordinates; default `EPSG:2056+5728` is LV95 horizontal coordinates with LN02 height. |
| `control_points.min_points` | Minimum usable GCP count required to resume processing; default three. |
| `control_points.max_reprojection_px` | Maximum RMS reprojection error accepted while triangulating a marked GCP. |
| `control_points.min_triangulation_angle_deg` | Minimum view angle accepted for a triangulated GCP. |
| `control_points.max_residual_warning_m` | Residual threshold shown as a warning in the GCP report. |
| `acquisition.panorama.frame_interval_seconds` | Default fixed interval used to extract frames from a 360 degree MP4 video. |
| `acquisition.panorama.cubemap_face_size` | Resolution of the perspective cubemap views generated for COLMAP. |

## `colmap.yml`

This file defines camera assumptions, matching, RTK/GPS accuracy, control-point
requirements, and dense reconstruction settings. `control_point_minimum_observations`
is set to two because a point must be marked in at least two images.

The `features` section controls COLMAP SIFT extraction. `max_num_features` remains
the limit for every capture type. For `panorama_360` only,
`panorama_peak_threshold` (default `0.003`) and `panorama_edge_threshold` (default
`20`) are passed to COLMAP to retain sufficient cubemap-face features; they do not
affect `drone_gps` or `camera` projects.

`alignment.gravity_align_panorama` defaults to `true`. For horizon-levelled
`panorama_360` inputs, SAMcloud derives gravity from the calibrated cubemap rig,
rotates the final sparse model so the up face points to `+Z`, and uses that model
for dense reconstruction. The resulting report records the arbitrary scale,
rig-pose spread, and sparse horizontal-plane cross-check.

## `export_crs.yml`

This file defines the LAS export CRS choices shown in the settings page. Each
preset has a stored key, display label, PROJ-readable target, height-system
description, and optional `required_grids`. `LOCAL`, `EPSG:4326`, and
`EPSG:2056` are the standard presets. Add other CRSs here and install every
listed geoid or transformation grid yourself in the PyProj data directory;
SAMcloud rejects missing grids and never uses ballpark transformations. Restart
SAMcloud after changing this file.

## `sam.yml`

This file controls the SAM3 weight location, prompt confidence threshold, tile
size, overlap, and multi-view voting threshold. `inference.mode` selects the
default `segearth_ov3` head-fusion path; `legacy` remains available for comparison.
For SegEarth-OV3, `background_threshold` is the per-point background threshold,
`query_threshold` filters decoder queries, and `threshold_sweep` can list A/B
thresholds. The paper does not publish a fixed tau, so choose it per dataset.
The local SAM3 checkpoint is stored at `models/sam3.pt` and is excluded from Git.

## `flurseg.yml`

FlurSeg is the project's own EoMT-DINOv3-L semantic segmentation model (trained on
SAM3 pseudo-labels, folder `models/flurseg_eomt_dinov3l_*`, see its `INFO.md`). In
the project settings it can replace SAM3 for the **outdoor** class set
(`classification.engine: flurseg`); indoor projects always use SAM3. The file holds
the model folder, the sliding window (768 px, stride 384) and the mapping of the 11
FlurSeg classes to the SAMcloud outdoor classes (`other` maps to unknown). FlurSeg
uses the same weighted multi-view voting, confidence scores and step 6 as SAM3.
It needs `transformers` and `scipy` (in `requirements.txt`).

"Reclassify" on the dashboard (pipeline `--reclassify`) re-runs only the
classification and step 6 on the kept dense cloud, recreating the undistorted
images if compaction released them. This allows a SAM3/FlurSeg comparison on
exactly the same points.

The FlurSeg model folder is not part of the repository (`models/` is ignored); it will be
published on Hugging Face. Its `EVALUATION.md` holds the Grütistrasse comparison against SAM3
(about 15x faster per image, 93 % agreement in FlurSeg's 11-class taxonomy).

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

This file contains the Potree point budget, default display mode, and enabled
viewer features. Class colors and class visibility will be written to a
project-specific viewer manifest when viewer data is generated.

## Capture settings stored per project

Every new project stores an `acquisition` section. It contains the selected
capture type, source locations, georeferencing state, and whether capture
settings are locked. `drone_gps` uses GPS alignment, while `camera` and
`panorama_360` start in `LOCAL` coordinates until control points are applied.

For a `panorama_360` project, `sources.panorama_video_path` and
`sources.panorama_image_directory` are independent optional inputs. At least
one is required. The `panorama` section records the fixed video frame interval,
the generated frame/cubemap locations, and the source manifest used to map
each virtual camera image back to its original panorama.
