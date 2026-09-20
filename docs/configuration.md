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
| `coordinate_systems.export_crs` | Default CRS for LAS export. Any CRS accepted by PROJ may be entered per project. |
| `workflow.use_gpu` | Enables GPU use for supported COLMAP and SAM3 stages. |
| `acquisition.panorama.frame_interval_seconds` | Default fixed interval used to extract frames from a 360 degree MP4 video. |
| `acquisition.panorama.cubemap_face_size` | Resolution of the perspective cubemap views generated for COLMAP. |

## `colmap.yml`

This file defines camera assumptions, matching, RTK/GPS accuracy, control-point
requirements, and dense reconstruction settings. `control_point_minimum_observations`
is set to two because a point must be marked in at least two images.

## `sam.yml`

This file controls the SAM3 weight location, prompt confidence threshold, tile
size, overlap, and multi-view voting threshold. The local SAM3 checkpoint is
stored at `models/sam3.pt` and is excluded from Git.

## `classes.json`

This file provides separate ordered prompt lists for `outdoor` and `indoor`
classification. The project setting `classification.class_set` chooses one of
them before a SAM3 run. The LAS exporter maps the selected IDs to
Cyclone-compatible LAS codes, including standard ASPRS codes where available
and custom codes for project-specific classes.

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
