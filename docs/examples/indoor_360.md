# Example: 360 degree indoor

[Back to the README](../../README.md#examples)

This example follows one real indoor project, `test_home_v11_tag`, from input
to result. The input is a 360 degree video of an apartment, recorded with an
Insta360 camera and exported from Insta360 Studio as a stitched
equirectangular `.mp4`. The project uses the capture type **360 degree
camera / Insta360** with the **Indoor** class set and AprilTags for the
metric scale. The example shows every workflow step: alignment and tag scale,
dense cloud, "Improve & filter", semantic classification, the clean DXF plan,
the IFC model, and the project overview with the computing times.

All viewer images use the same camera, the same display settings and the same
crop; the ceiling is hidden with the viewer's height filter (upper limit
2.2 m) so the rooms are visible from above.

## 1. Input

<img src="../../assets/workflow_input_frame.jpg" alt="Crop of one equirectangular frame" width="800">

Crop of one equirectangular frame from the Insta360 video. 356 panorama frames
were extracted and split into 2,136 rig images. The AprilTag target TAG6 is
visible on the wardrobe.

## 2. Sparse cloud and camera alignment

<img src="../../assets/workflow_sparse_cloud.png" alt="Sparse cloud with aligned cameras" width="800">

COLMAP aligns the cameras (panorama rig in 17 min). The sparse cloud has
29,711 tie points (29,192 shown in the viewer). The eight AprilTags TAG0 to
TAG7 are detected and triangulated. Their known size of 150 mm gave a scale
factor of 1.236 (121.3 mm measured before scaling, all tags within 1.2 %).
See [AprilTag scale and georeferencing](../apriltag_scale.md).

## 3. Dense cloud before refinement

<img src="../../assets/dense_cloud_before_refinement.png" alt="Dense cloud before refinement" width="800">

COLMAP dense stereo yields about 4.09 million points (2 h 18 min). Gaps on
white walls and floor, noise, and see-through areas are visible.

## 4. Dense cloud after "Improve & filter" (step 6)

<img src="../../assets/dense_cloud_after_refinement.png" alt="Dense cloud after refinement" width="800">

The Depth-Anything prior and depth fill add about 13.9 million points
(18.0 million in total). Surface fusion snaps them onto one fused surface,
merges duplicates and removes mirror phantoms, see-through and hidden points,
leaving 3.36 million points (step 6 took 1 h 25 min, 1 h 02 min of it reading
depth maps). See [Improve & filter (step 6)](../../README.md#improve--filter-step-6).

## 5. Semantic classification

<img src="../../assets/dense_cloud_clas.png" alt="Classified dense cloud" width="800">

SAM3 (SegEarth-OV3) labels 1,440 images. Multi-view voting and the superpoint
vote (27,214 superpoints) assign a LAS class per point; colours are LAS
classes. 83.5 % of the points get a class, and 90 object instances are
found. Classification took 1 h 14 min. See
[Semantic classification](../../README.md#semantic-classification) and
[Object instances](../instances.md).

## 6. Clean DXF plan

The DXF and IFC export is created automatically after processing (the project
setting **Create DXF/IFC automatically after processing**, default on); the
DXF/IFC page shows the result and lets you recompute it.

<img src="../../assets/workflow_dxf_ifc_page.png" alt="DXF/IFC page with the clean plan of the apartment" width="800">

The clean plan (orthophoto variant) of the same apartment on the DXF/IFC page,
shown in the same orientation as the viewer images (entrance door at the top).
Walls are drawn as double lines with a grey fill. Five rooms go from door to
door and carry name and area: Gäste-WC 3.1 m², Bad 5.4 m², Zimmer 13.7 m²,
Schlafen 20.5 m² and Wohnen / Küche 48.6 m². Kitchen and living room are one
space, and the entrance hall is part of it because there is no door between
them. The small corridor joins the bedroom through the passage. The plan also
holds 5 doors with their widths (0.80 to 1.60 m), one passage (0.90 m, an
opening up to the ceiling without door leaf, drawn as jamb lines only) and 2
windows (1.85 m with two mullions, and 1.77 m with one mullion). Doors with a
measured open leaf (three here) are drawn with
their swing symbol on the measured hinge and swing side, sanitary and kitchen symbols,
the kitchen (counter drawn by its front lines, dashed upper cabinet), the tall
built-in units with the wall behind them, the dining table with its chairs
each in its own labelled box, and furniture outlines with their heights whose
shape is chosen against the measured footprint, all on Cyclone/CAD layers.
Door and window widths, sill and head heights come from the opening
refinement described in the next section.
See [DXF export](../../README.md#dxf-export) and
[Orthophoto clean plan](../ortho_plan.md).

## 7. IFC model

The same **Compute everything** button also writes the IFC. On indoor
projects the page is called **DXF/IFC**, and the run is: orthophoto, segment,
clean plan, **Refine openings**, clean plan again, DXF export, IFC export.
**Refine openings** measures every window and door on its wall plane from the
SAM3 masks of the cube-face photos (projected with the COLMAP poses, with
occlusion test) and from the point cloud: width, sill height (BRH), head
height (STH) and mullions; areas nobody saw stay unknown. It also tells a door
(door-class casing or lintel) from a passage (opening up to the ceiling, no
door) and from a candidate that is no opening at all, measures the head and the
headroom of every door and the hinge and swing of an open leaf, and finds
entrance doors in wall lines without opening. The DXF and the IFC
are both written from the same building model (`plan_model.json`), so walls,
rooms, doors and windows match. The DXF plan above and the IFC below come from this one model.

<img src="../../assets/workflow_ifc_viewer.png" alt="IFC viewer on the DXF/IFC page" width="800">

The IFC viewer below the DXF preview shows the model in 3D: 27 walls with
their openings, 5 doors, 1 passage (an opening without door), 2 windows, 5
rooms (6 room volumes, because a wall splits the bedroom into two pieces),
floor slab, 6 ceilings and 42 furniture items. The viewer splits the furniture by IFC class into the
element types *Möbel* (30, `IfcFurniture`), *Sanitär* (6,
`IfcSanitaryTerminal`), *Geräte* (1, `IfcElectricAppliance`) and *Sonstiges*
(5, `IfcBuildingElementProxy`), which add up to the 42 items. Element types
can be shown or hidden; ceilings, rooms and openings are hidden by default so
the view looks into the apartment. The export is checked
against an IDS file (0 failures) and `qa.json` confirms that the openings of
the DXF and the IFC are identical.

<img src="../../assets/workflow_ifc_window.png" alt="Picked window with its properties" width="800">

A click on an element highlights it and opens its properties. The picked
window is 1.85 m wide and 1.40 m high, with sill height (BRH) 0.65 m, head
height (STH) 2.05 m and 2 mullions. See [IFC export](../ifc_export.md).

## 8. Project overview in the UI

<img src="../../assets/Overview.png" alt="Project overview in the user interface" width="800">

The workflow panel tracks all steps and their computing times (5 h 28 min in
total for this project on one GPU: 5 h 23 min for the pipeline plus 4 min 32 s for the DXF/IFC export). The local source path is blanked out in
the screenshot. It shows the default German interface; English can be selected
in the settings (see [User interface](../../README.md#user-interface)).

## Other capture types

- [360 degree outdoor](outdoor_360.md)
- [Drone photos](drone.md)
- [Phone and handheld photos](phone.md)
