# Example: 360 degree outdoor

[Back to the README](../../README.md#examples)

> **Example results follow — no processed dataset yet.**
> This page describes how a 360 degree outdoor capture is set up and processed.
> Images and measured results will be added once a real project has been
> processed.

## What this capture type is for

A 360 degree camera (Insta360) carried through an outdoor area records the
surroundings in every direction at once, for example along a street, through a
courtyard or around a building. SAMcloud treats it as the capture type
**360 degree camera / Insta360** with the **Outdoor** class set.

## Capturing and preparing the footage

- **Export from Insta360 Studio first.** SAMcloud accepts stitched 2:1
  equirectangular `.mp4` videos and `.jpg`/`.jpeg` panoramas, or both together.
  Raw `.insv` files are not supported, and a panorama whose aspect ratio is not
  2:1 is rejected.
- **Export horizon-levelled panoramas.** For such inputs SAMcloud derives
  gravity from the cubemap rig and rotates the sparse model so that up is `+Z`
  (`alignment.gravity_align_panorama: true` in `config/colmap.yml`).
- **Frame interval.** Video frames are extracted once per second by default
  (`acquisition.panorama.frame_interval_seconds: 1.0` in
  `config/defaults.yml`); the project dialog accepts other intervals from
  0.1 s.
- **Cubemap views.** Each panorama is split into six perspective cubemap views
  for COLMAP (face size `cubemap_face_size: 1232` px by default, capped at the
  native resolution). Extracted frames, cubemap views and a source manifest
  are stored below `inputs/`, and the capture settings are locked once these
  derived inputs exist.
- **Image GPS is not used.** Only the drone capture type reads GPS EXIF, so a
  360 reconstruction starts in a local frame with an arbitrary scale. Plan
  control points or AprilTags (below) if the result needs metres or a
  coordinate system.

## Project settings

- **Class set:** choosing the 360 capture type preselects **Indoor**; switch
  it to **Outdoor** in the project dialog. This also switches the AprilTag
  option off by default (it can be switched on again).
- **Classification engine:** SAM3 or **FlurSeg**, the project's own outdoor
  model (about 15x faster on the drone test, fixed 11 classes). The published
  SAM3/FlurSeg comparison was made on drone photos, see
  [FlurSeg](../../README.md#flurseg-outdoor-alternative-to-sam3). With SAM3,
  the outdoor classes can be adjusted when the project is created.
- **Scale and georeference:**
  - *Manual control points* (`after_alignment` mode): three or more points,
    each marked in at least two undistorted images, georeference the model.
    The printable 180 mm cross targets from the **Targets** page give clearly
    defined points to mark.
  - *AprilTags* (when switched on, non-GPS projects): 150 mm `tag36h11` tags
    give the metric scale; three or more non-collinear tags with surveyed
    centres give a full georeference. See
    [AprilTag scale and georeferencing](../apriltag_scale.md).
  - *Automatic control points* need image GPS and are therefore only available
    for drone projects.
  See [Targets, scale, and control points](../../README.md#targets-scale-and-control-points).

## Which pipeline steps apply

| Step | 360 outdoor |
|---|---|
| Alignment | Rig-aware matching of the cubemap faces (exhaustive for small sets, sequential otherwise), fixed cubemap intrinsics, gravity alignment |
| Dense stereo | 360 profile with `window_step: 2` (about 2x faster PatchMatch) |
| Classification | SAM3 or FlurSeg, then multi-view voting |
| Improve & filter (step 6) | Floating-point removal only; Depth-Anything prior, mirror filter, surface fusion and superpoint vote are indoor-only |
| Geometric refinement | Only when the project is georeferenced (control points or surveyed AprilTags); otherwise skipped |
| LAS export | `LOCAL` always; after a control point georeference also WGS84 and LV95 (with surveyed AprilTags the tag CRS becomes the export CRS) |
| DXF export | One site plan; the clean orthophoto variant adds contours and spot heights |

Details: [Processing pipeline](../../README.md#processing-pipeline),
[Capture types, processing modes, and coordinates](../../README.md#capture-types-processing-modes-and-coordinates),
[DXF export](../../README.md#dxf-export).

## Example results

Example results follow — no processed dataset yet. The worked
[360 degree indoor example](indoor_360.md) shows what a complete example looks
like.
