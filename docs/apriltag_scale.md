# AprilTag scale and georeferencing

COLMAP reconstructions without GPS or control points have an arbitrary scale.
For indoor scans SAMcloud can recover a metric scale from printed AprilTags and,
when the tag centres have been surveyed, a full georeference. The code is in
`samcloud/control_points/apriltags.py` (detection, triangulation, scale),
`samcloud/control_points/apriltag_print.py` (printable sheets) and the
`_apriltag_step` function of `samcloud/engine/pipeline.py`.

## Tag family and size

- Family `tag36h11` (OpenCV `DICT_APRILTAG_36h11`), IDs 0 to 586.
- The tag is a black square of exactly **150 mm**, measured on the black edge
  (`APRILTAG_SIZE_M = 0.150`). The white quiet margin around it is not part of
  the size.
- Detection uses `cv2.aruco` with `CORNER_REFINE_APRILTAG`, which already
  reports corners in COLMAP's pixel convention (no half-pixel shift is added).

## Printing

Open the **Targets** page (works without a project) or press **AprilTag
drucken** on the Control points page. Choose a count (1 to 100) and a start ID
and press **PDF erstellen**. One A4 portrait page per tag contains:

- the tag, rendered at 222 px per module (about 300.7 dpi) so the black square
  is exactly 150 mm when printed at 100 %;
- a dashed cut line around the tag and its margin;
- tick marks at the middle of each cut edge: the tag centre is where the lines
  between opposite ticks cross, which is the point to survey;
- the ID, family, and size below the tag, and a 100 mm check bar;
- notes: print at 100 % ("actual size", not "fit to page"), check that the
  black square measures 150.0 mm, stick it flat on the wall.

When a project is open, the highest printed ID is stored in
`settings.apriltags.last_printed_id` and the dialog proposes the next ID.
IDs printed outside the app are not known to the project.

## Switching it on

The project dialog has the switch **AprilTags suchen (Maßstab)**. It is on by
default for the Indoor class set (including 360 projects, which default to
Indoor) and off for Outdoor, and can be set either way. The stored setting is
`settings.apriltags.enabled`. The pipeline runs the tag step (`--apriltags`) only
when the project is not georeferenced by GPS and no automatic control point
model is in use.

## What the pipeline does

The tag step runs after alignment and before undistortion.

1. **Detect.** Tags are searched in every COLMAP input image (for 360 projects
   the six cubemap faces per panorama). The result is cached in
   `control_points/apriltag_detections.json` and reused while it covers all
   images.
2. **Triangulate.** Every tag seen in at least two images gets its four corners
   and its centre triangulated in the sparse model.
3. **Scale.** For each tag, edge and diagonal lengths (six distances) are
   compared with the nominal 150 mm, and the median ratio is its scale factor.
   The model scale is the mean over the tags that agree.
   - A single tag is enough. With fewer than three tags the report carries the
     warning "keine Kontrolle möglich" because no cross-check exists.
   - With three or more tags, a tag whose factor deviates from the median by
     more than `apriltags.outlier_pct` (default 3 %) is marked an outlier,
     ignored, and warned about.
   - Without any tag the model stays unscaled and the report warns about it.
4. **Georeference (optional).** If `control_points/apriltags.csv` exists and at
   least three non-collinear tags with role `pass` have surveyed centres, the
   tags act as control points: they are fed into the same GCP adjustment as
   drone control points and the model is transformed into the project CRS (or
   into the local survey frame, see below). If that is not possible the step
   falls back to the scale-only result and adds the warning "Weniger als 3
   eingemessene AprilTags – nur Maßstab". If the adjustment's quality gate
   fails, the run pauses for review like the other control point modes.
5. **Report.** `reports/apriltag_report.json` records the mode (`scale`,
   `georef`, or `none`), the scale factor and applied factor, warnings, and per
   tag: ID, image count, size in mm before and after, deviation in percent,
   status (`ok` or `outlier`), centre and corner coordinates, reprojection
   error, and in `georef` mode the 3D error.

The scaled sparse model is written to `products/sparse/0_tagscale`. Dense
reconstruction continues from it. For 360 projects the gravity alignment still
runs after a scale-only result and is skipped after a tag georeference.

## Surveyed tag coordinates

Coordinates are optional. Enter or import them in either place:

- project dialog: **Tag-Koordinaten (optional)** upload and the CRS of the tags;
- Control points page: coordinate table (ID, X, Y, Z, role), import, and save.

File format (`control_points/apriltags.csv`): one row per tag with `id,x,y,z`
and an optional `role` column (`pass` or `control`, default `pass`). A header is
optional, the delimiter and decimal comma are detected, and an ID may be written
`TAG12`. The coordinate is the **centre** of the tag. The ID on the printed
sheet must match the ID in the file.

- `pass` tags constrain the adjustment; `control` tags are only checked.
- The tag CRS can be any CRS offered in the dialog. `LOCAL` means a private
  metric system (metres, z up): the model is placed exactly in that frame, no
  `georeference.json` is written, and the control point report carries
  `source_frame: local_tags`. `LOCAL` is rejected for drone GPS projects.
- For non-GPS captures the tag CRS also becomes the LAS/DXF export CRS.

## Review

- The Control points page shows the tag table and the report with size before
  and after, deviation, and status.
- The sparse viewer labels tags like control points, and the review workspace
  lists them with their 3D error.
- The dense viewer draws each tag as its square (endpoint
  `/api/apriltags/markers`). A click on a tag highlights it and shows its
  measured size and deviation from 150 mm; it takes priority over object
  instances. The report warnings are served with the markers.
- Points on a tag square (within 3 cm of the tag plane and 2 cm outside its
  corners) get `instance_id` 0 so that the stickers do not become objects.

## Limits

- Detection depends on image sharpness: tags must be flat, printed at 100 %,
  and visible with several pixels per module.
- Only a printed size of 150 mm is supported.
- A scale from one tag cannot be cross-checked; use at least three tags spread
  through the building.
