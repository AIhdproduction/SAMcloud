# Example: phone and handheld photos

[Back to the README](../../README.md#examples)

> **Example results follow — no processed dataset yet.**
> This page describes how photos from a phone or a handheld camera are set up
> and processed. Images and measured results will be added once a real
> project has been processed.

## What this capture type is for

Ordinary photos taken with a phone, a compact camera or a DSLR, indoors or
outdoors, for objects, rooms or small sites where no drone or 360 degree
camera is used. SAMcloud treats them as the capture type **Standard camera**
(`camera`).

## Capturing and preparing the photos

- **Image formats:** `.jpg`, `.jpeg`, `.tif`, `.tiff` and `.png` in the image
  folder (subfolders included). Source images stay at their original location.
- **Overlap.** Every photo pair is matched (exhaustive matching), and the run
  stops with "Check image overlap/quality" when COLMAP cannot build a
  reconstruction from the photos.
- **Image GPS is not used.** Only the drone capture type reads GPS EXIF, so a
  phone reconstruction starts in `LOCAL` with an arbitrary scale. Plan
  AprilTags or control points (below) if the result needs metres or a
  coordinate system.
- **AprilTags for scale (indoor).** Print 150 mm `tag36h11` tags on the
  **Targets** page at 100 %, check the 150 mm square, and stick them flat on
  the walls. Tags must be sharp and visible in at least two photos; use at
  least three tags spread through the building so the scale can be
  cross-checked. See [AprilTag scale and georeferencing](../apriltag_scale.md).

## Project settings

- **Class set:** **Indoor** or **Outdoor**. Indoor always uses SAM3; Outdoor
  can use SAM3 or **FlurSeg**
  ([FlurSeg](../../README.md#flurseg-outdoor-alternative-to-sam3)).
- **AprilTags:** on by default for Indoor and off for Outdoor; can be set
  either way. A single tag gives a scale; three or more non-collinear tags
  with surveyed centres give a full georeference.
- **Control points:** in `after_alignment` mode, three or more points marked in
  at least two undistorted images each georeference the model. Automatic
  control points need image GPS and are only available for drone projects.
  See [Targets, scale, and control points](../../README.md#targets-scale-and-control-points).

## Which pipeline steps apply

| Step | Indoor | Outdoor |
|---|---|---|
| Alignment | Exhaustive matching, incremental mapper, `LOCAL` frame | same |
| Scale / georeference | AprilTags or control points | control points (or AprilTags when switched on) |
| Classification | SAM3, multi-view voting | SAM3 or FlurSeg, multi-view voting |
| Improve & filter (step 6) | Depth-Anything prior, mirror filter, surface fusion, superpoint vote | floating-point removal only |
| Geometric refinement | not applied | only when georeferenced |
| LAS export | `LOCAL`, or the georeferenced CRS | `LOCAL`, or the georeferenced CRS |
| DXF export | one plan per storey | one site plan |

Details: [Processing pipeline](../../README.md#processing-pipeline),
[Improve & filter (step 6)](../../README.md#improve--filter-step-6),
[Capture types, processing modes, and coordinates](../../README.md#capture-types-processing-modes-and-coordinates),
[DXF export](../../README.md#dxf-export).

## Example results

Example results follow — no processed dataset yet. The worked
[360 degree indoor example](indoor_360.md) shows what a complete example looks
like.
