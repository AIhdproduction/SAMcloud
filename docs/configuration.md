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
