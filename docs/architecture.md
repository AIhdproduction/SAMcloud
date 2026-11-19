# Architecture

SAMcloud separates the interface, project persistence, photogrammetry,
semantic classification, and point cloud delivery so that each part can be
developed and tested independently.

```text
NiceGUI pages  (samcloud/app)
    -> core project manager  (project file, settings, storage)
        -> acquisition        360 degree frames and cubemap views
        -> engine.pipeline    COLMAP, AprilTags/GCPs, dense, SAM3/FlurSeg, step 6, geometry
        -> engine.export_las  LAS 1.4 export
        -> viewer             Potree octrees, previews
        -> dxf / ortho        optional CAD plans (background jobs)
```

## Packages

| Package | Responsibility |
|---|---|
| `samcloud/app/` | NiceGUI pages (`pages/`), layout and navigation, the process launcher (`processing.py`), the Potree and canvas viewers, DXF page controllers, the IFC viewer (`ifc_viewer.py`, `ifc_viewer.js`; three.js is vendored in `assets/vendor/three/`), the REST endpoints in `application.py`, and the UI translations (`i18n.py`, `locales/`) |
| `samcloud/core/` | paths, configuration loading, project creation and persistence (`project_manager.py`), workflow state, project deletion, storage compaction (`storage.py`) |
| `samcloud/acquisition/` | validation of exported Insta360 media, fixed-interval frame extraction, six perspective cubemap views per panorama, `inputs/panorama_manifest.json` |
| `samcloud/colmap/` | COLMAP command construction, alignment and camera groups, panorama gravity alignment, and the step 6 modules: `depth_prior`, `depth_fill`, `surface_fusion`, `mirror_filter`, `depth_edges`, `point_cleaning`, `plane_fill`, `superpoint_vote` |
| `samcloud/control_points/` | control point CSV and observations, automatic bow-tie target detection (`targets.py`, `auto_detect.py`), GCP adjustment, georeferencing, AprilTag detection/scale/print (`apriltags.py`, `apriltag_print.py`), cross target print (`cross_target_print.py`) |
| `samcloud/classification/` | class definitions, per-project custom classes, model profiles (Cyclone code mapping per model), service helpers |
| `samcloud/pointcloud/` | coordinate transformation and georeference services |
| `samcloud/viewer/` | point cloud previews, Potree conversion (`potree.py`), camera payloads, sparse display cleaning |
| `samcloud/dxf/` | DXF page backend: LAS/PLY source, scale, alignment, levels, point-cloud plan generation, writer, cancellable background jobs (`jobs.py`, `job_runner.py`) |
| `samcloud/ortho/` | true orthophoto, DSM, segmentation, clean plan generation and vector extraction (`vector/`) |
| `samcloud/bim/` | IFC4 export of the clean plan (indoor): `plan_model.py` (reads `plan_model.json`), `scene_builder.py` (walls, openings, rooms, slabs, ceilings), `furniture.py` (class mapping), `export.py` (IFC, IDS validation, `qa.json`), `viewer_export.py` (`model.glb`, `props.json`), `jobs.py` (freshness, CLI `python -m samcloud.bim.jobs`), `ui_data.py` (viewer and download data for the page), `core/` (IFC writer, geometry, QA) and `openings3d/` (3D window/door refinement from photos and cloud, `openings_3d.json`); see [ifc_export.md](ifc_export.md) |
| `samcloud/tools/` | command-line helpers (class import from a Nomivox preset, gravity alignment) |
| `samcloud/engine/` | the processing implementation; see below |

## Engine

`samcloud/engine/` holds the processing implementation. The UI starts it as a
separate Python process (`python -m samcloud.engine.pipeline`, and
`python -m samcloud.engine.export_las` for the manual export) so long runs do
not block the web server and can be cancelled or resumed. DXF page jobs use the
same model (`python -m samcloud.dxf.jobs <action>`). New code should reach the
engine through the named service packages instead of putting application logic
in `main.py`.

| Module | Role |
|---|---|
| `pipeline.py` | orchestrates all stages, writes progress and reports; supports `--reclassify` and `--refine-from-cache` re-runs |
| `sam_labeler.py` | SAM3 tiling, raw heads, SegEarth-OV3 fusion, instance masks |
| `flurseg_labeler.py` | FlurSeg sliding-window inference (outdoor) |
| `classify.py` | projection into cameras, weighted multi-view voting, depth-aware class check, suppress prompts, confidence and fine-term export |
| `instances.py` | instance association, refinement, and the `instance_id` product (see [instances.md](instances.md)) |
| `fine_terms.py` | per-point fine SAM terms (sofa, bed, ...) |
| `geometry/` | geometric refinement (terrain, support, features, neighbours, per-class rules) |
| `export_las.py` | LAS 1.4 export with class mapping, CRS transform, and `fine_term` / `instance_id` extra dimensions |
| `processing_cache.py` | optional cache of step 6 inputs |
| `eta.py` | remaining-time estimates learned per capture type |
| `run_summary.py` | run reports and stage durations |

Scripts such as `classify.py` run as scripts as well as modules, so modules they
import must not use relative imports.

## Data flow of a run

1. **Alignment.** COLMAP features, matching, and sparse mapping (pose-prior
   mapper for drone GPS/RTK), tie point filtering, GPS alignment.
2. **Scale or georeference.** For non-GPS projects with AprilTags enabled the
   tag step scales or georeferences the sparse model
   ([apriltag_scale.md](apriltag_scale.md)); control points can be applied
   manually or detected automatically. 360 projects are gravity-aligned.
3. **Dense reconstruction.** Undistortion, PatchMatch stereo, fusion to
   `products/dense/fused.ply` with a visibility file.
4. **Classification.** SAM3 (or FlurSeg for outdoor) per image, weighted 3D
   voting, and, when enabled, instance association and fine terms.
5. **Step 6 (improve and filter).** Depth prior, mirror detection, surface
   fusion or depth fill, floating point removal, superpoint vote; then the
   geometric refinement writes `classified.ply`. Fine terms and instance IDs are
   set aside before step 6 changes the point set and attached to the final cloud
   afterwards by nearest neighbour.
6. **Exports (manual).** LAS from the DXF page or the dashboard; DXF plans from
   the DXF page.

## Projects

Each user project is independent and can be reopened from its
`project.samcloud.yml` file. Source images remain in their original folder by
default; the project metadata stores the path and all generated results,
control point observations, reports, and viewer data are kept under the local
project directory. Which large intermediates survive a run is controlled by the
project's `storage` settings (`samcloud/core/storage.py`).

For 360 degree projects, `samcloud/acquisition/` validates exported Insta360
media, extracts fixed-interval video frames, generates six perspective views
per equirectangular panorama, and writes `inputs/panorama_manifest.json`. The
manifest preserves the relationship between a virtual COLMAP image and its
source panorama for later classification and inspection.

## Viewer and REST endpoints

The dense and sparse viewers use Potree when `assets/potree/` and
`tools/PotreeConverter/` are installed; otherwise a canvas viewer shows a
downsampled preview (`/api/pointcloud/{kind}`). Endpoints in
`samcloud/app/application.py`:

| Endpoint | Purpose |
|---|---|
| `/api/potree/{kind}/status`, `/api/potree/{kind}/{path}` | Potree build state and octree files |
| `/api/instances`, `/api/instances/{id}/points` | object instances and the points of one instance |
| `/api/cameras/sparse`, `/api/camera-thumbnail/{id}` | camera frusta and thumbnails for the sparse viewer |
| `/api/control-points/*` | markers, image crops, observation edits, optimize, continue (GCP review workspace) |
| `/api/apriltags/markers` | AprilTag squares for the viewers |
| `/api/dxf/payload/...`, `/api/dxf/background/...`, `/api/dxf/ortho/...` | DXF review data, backgrounds, and orthophoto previews |
| `/api/bim/{level}/model.glb`, `/api/bim/{level}/props.json`, `/api/bim/{level}/ifc` | IFC viewer model, element properties, and the IFC download |

## UI translations

All visible UI texts go through `samcloud/app/i18n.py`. Page code calls
`t("namespace.key", **params)`, or `translate_or(key, fallback)` for keys built
from data (for example class names via `classes.<name>`). The texts live in
`samcloud/app/locales/<language>/<namespace>.json` (`de` and `en`, identical
key sets, enforced by `tests/test_i18n.py`). The active language comes from
the user choice in `.samcloud/user_preferences.json`, then
`application.language` in `config/defaults.yml`, then `de`.

Translation is a display concern of the UI only: the engine, project files,
reports, `progress.json`, REST field names, and the LAS/DXF output are
language-neutral, and stored English engine texts (progress stages, AprilTag
warnings, viewer server messages) are translated when they are shown. Adding a
language means a new `locales/<code>/` folder and an entry in
`SUPPORTED_LANGUAGES`. Details: [i18n.md](i18n.md).
