# Example: drone photos

[Back to the README](../../README.md#examples)

> **Example results follow — no processed dataset yet.**
> This page describes how a drone capture is set up and processed. Images and
> measured results will be added once a complete example project has been
> prepared.

## What this capture type is for

Drone photos cover sites, buildings and terrain from above. The capture type
**Drone GPS/RTK** is the only one that SAMcloud georeferences automatically:
the GPS position stored in every photo aligns the reconstruction, so the
classified cloud and the LAS export are in real-world coordinates without
control points.

## Capturing and preparing the photos

- **Every photo needs GPS EXIF.** The run reads the GPS tags of all photos and
  stops before the metric alignment when any supported image has no readable
  GPS position.
- **Image formats:** `.jpg`, `.jpeg`, `.tif`, `.tiff` and `.png` in the image
  folder (subfolders included). Source images stay at their original location.
- **RTK or standard GPS.** The project dialog has an RTK switch (on by default
  for drone projects). When at least 90 % of the photos carry DJI RTK
  fixed metadata, RTK settings are used automatically. The GPS alignment
  accepts a maximum error of 0.1 m with RTK and 3.0 m with standard GPS
  (`georeferencing` in `config/colmap.yml`).
- **Several flights.** Cameras are grouped per flight by the image sequence
  number (`camera.grouping: sequence_number`).
- **Control points (optional).** For automatic control points, lay out the
  numbered 180 mm cross targets from the **Targets** page (two black triangles
  whose tips meet at the survey point) and survey them. The black printed
  variant is only reliably detected on light ground.

## Project settings

- **Class set:** **Outdoor**.
- **Classification engine:** SAM3 or **FlurSeg**. FlurSeg is about 15x faster
  but has 11 fixed classes; see
  [FlurSeg](../../README.md#flurseg-outdoor-alternative-to-sam3) for the
  comparison on a drone dataset and how to install the model. Large drone
  photos are split into SAM3 tiles (`tile_size` 1008 px, overlap at least
  0.2), which takes longer per photo than a cubemap face.
- **Control points mode** (`workflow.control_points_mode`):
  - `none` (default): GPS/RTK alone georeferences the model.
  - `after_alignment`: the run pauses after undistortion so three or more
    ground control points can be marked by hand; GCPs then take precedence
    over GPS.
  - `automatic` (needs a control point list): cross targets are detected, a
    GCP bundle adjustment runs, and the run pauses for review only when the
    quality gate fails.
  See [Targets, scale, and control points](../../README.md#targets-scale-and-control-points).

## Which pipeline steps apply

| Step | Drone GPS/RTK |
|---|---|
| Alignment | GPS pose priors and the pose-prior mapper; matching `auto` (exhaustive up to 300 photos, spatial GPS neighbours above) |
| GPS alignment | COLMAP `model_aligner` into a local ENU frame (metres, z up) |
| Dense stereo | Shared dense settings (`capture_profiles.drone_gps` in `config/colmap.yml` for overrides) |
| Classification | SAM3 or FlurSeg, then multi-view voting |
| Improve & filter (step 6) | Floating-point removal; the indoor-only stages are skipped |
| Geometric refinement | Runs automatically, because the model is georeferenced |
| LAS export | `LOCAL`, `EPSG:4326` (WGS84) or `EPSG:2056` (LV95 + LN02, needs the swisstopo CHGeo2004 grid) |
| DXF export | One site plan; the clean orthophoto variant adds contours and spot heights |

Details: [Processing pipeline](../../README.md#processing-pipeline),
[Geometric refinement](../../README.md#step-3-geometric-refinement),
[LAS export and coordinate systems](../../README.md#las-export-and-coordinate-systems),
[DXF export](../../README.md#dxf-export).

## Example results

Example results follow — no processed dataset yet. The worked
[360 degree indoor example](indoor_360.md) shows what a complete example looks
like.
