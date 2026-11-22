# IFC export

On indoor projects the DXF page is the **DXF/IFC** page. **Compute
everything** produces the clean DXF plan, a 3D DXF and an IFC4 building model
with one button. All come from the same building model, `plan_model.json`, so
walls, rooms, doors and windows are identical in the DXF and the IFC. The code
is in `samcloud/bim/` (export, viewer assets, 3D opening refinement); the jobs
are in `samcloud/dxf/jobs.py`.

The export described here is indoor only. For outdoor projects these jobs skip
the refinement and the IFC step and log "IFC export: indoor only". Outdoor
drone projects get their site IFC from a separate environment chain
(`samcloud/drone/run_environment.py`, published to `products/bim/site/`).

## Stages

**Compute everything** (`run_all`) runs, for an indoor level:

| # | Step | Result |
|---|---|---|
| 1 | Build orthophoto (`ortho`) | true orthophoto, height rasters |
| 2 | Segment orthophoto (`segment`) | per-pixel class raster |
| 3 | Generate clean plan (`clean_plan`) | `plan_model.json`, first plan pass |
| 4 | **Refine openings** (`refine_openings`) | `openings_3d.json`; the clean plan is generated again with the refined openings |
| 5 | Export DXF (`export`) | `<project>_<level>_clean.dxf` |
| 6 | **Export IFC** (`export_ifc`) | `products/bim/<level id>/`, `<project>_<level>_3d.dxf` in `products/dxf/export/`, and with several levels `products/bim/building/` |

Steps with a fresh cache are skipped unless **Force** is ticked. The page lists
the status of every step (orthophoto, segmentation, clean plan, refine
openings, DXF export, IFC export). The steps 1 to 3 and 5 are described in
[Orthophoto clean plan](ortho_plan.md).

## Refine openings

The 2D plan finds windows and doors in the orthophoto. **Refine openings**
measures them again on the wall planes, in the manner of a RoomPlan scan
(`samcloud/bim/openings3d/`, settings in `config/bim_openings.yaml`):

- Every wall plane gets a 5 cm grid. The cube-face photos (`front`, `back`,
  `left`, `right`) that look at the wall are chosen, SAM3 window and door masks
  are computed for them, and each grid cell collects votes. The masks are
  projected with the COLMAP poses; a depth buffer of the cloud removes cells
  that are hidden behind furniture or other walls.
- The point cloud adds evidence: gaps in the wall points, window and door
  points, and the height of the lintel.
- **Mullions** are detected inside windows, and a window band that mullions
  split into several parts is merged into one window. **Doors** are snapped to
  the gaps of the plan and get their head height from the points.
- A cell that no photo or point saw is *unknown*, not "wall" or "opening".
  Doors with a glass leaf are flagged (`glazed`).
- **Doors, passages and openings that are none** (`classify.py`, `frames.py`,
  `leaves.py`). The door-class points (Cyclone 103) of the classified cloud are
  the usable signal (the photo masks merge the frame into "door"). For every
  door candidate the refinement measures the casing: two jamb strips (door-class
  points from 0.10 m inside to 0.18 m outside the opening edge over the height
  of the opening; a wider door-class run outward of the jamb is a flat leaf, not
  a casing) and the head strip, and the head height: the lowest height above
  1.6 m where half of the columns inside the opening hold points. The headroom
  is the ceiling minus that head (`headroom_m`). The rules:
  - **door**: a jamb strip and the head strip, or a closed leaf filling the
    opening, or a lintel on an observed wall (headroom at least 0.15 m and a
    measured head of at least 1.85 m or 0.66 of the storey height). A lintel
    door that the 2D plan did not see also needs 60 % of its cells measured
    free and floor points under half of its width;
  - **passage**: no door evidence, headroom below 0.15 m and the opening cells
    measured free of points; one that the 2D plan did not see also needs the
    floor under it. It stays an `IfcOpeningElement` without `IfcDoor`, rooms
    on both sides are one room;
  - **no opening**: neither door evidence nor a measured free gap (unobserved,
    blocked by furniture in the wall band, or not on the observed wall): the
    candidate is dropped and the wall closes. Weak evidence (one jamb strip or
    only the head strip, door-class points in the gap) keeps a door the plan
    had.
  Every number is stored in the opening's `evidence` text and in the fields
  `headroom_m`, `frame_jambs` and `head_frame`; the measured head replaces the
  2.0 m default (`head_m`).
- **Open leaves**: a hinge-ray search around both jambs of a door finds the open
  leaf (0.5 to 1.3 m long, door high, at least 25 degrees open) and gives
  `hinge_side`, `swing_side`, `leaf_angle_deg` and `leaf_detected`. A leaf alone
  never makes a door, it is assigned only to a door that carries its hinge at its
  own jamb (a leaf two doors both find belongs to the closer jamb). Passages
  never carry a leaf. In the DXF such a door is the `DOOR` block on the
  measured hinge, swinging to the measured side, plus its jamb lines; in the IFC
  it sets the operation (below).
- **Exterior and entrance doors**: wall lines are scanned for door frames (two
  jamb strips or one, plus the head strip; a closed leaf in the wall plane counts
  as a casing) where no opening exists yet. A found frame becomes a door of the
  measured width hosted on its wall; a flat leaf lying against a wall, a single
  jamb or an empty gap never does (`scan:` in `config/bim_openings.yaml`).
- Without CUDA or usable photos the refinement falls back to points only. On
  any other error the plan keeps its 2D openings and the log shows a warning.
  When code or plan settings change during a run the refined openings cannot
  be applied; the log then says `REFINED OPENINGS NOT APPLIED` and the run
  has to be repeated.
  The refinement is idempotent: it always starts from the first plan pass and
  its freshness depends on the walls, so plan and refinement cannot loop.

The result is `openings_3d.json` next to `plan_model.json`. In the DXF, window
blocks carry the attributes `BRH` (sill height, Brüstungshöhe), `STH` (head
height, Sturzhöhe) and `MULLIONS` (count); door blocks carry `H` (head height)
and `GLAZED`. The mullions are drawn as lines on `A-WIND`.

## IFC content

`samcloud/bim/export.py` builds a scene from `plan_model.json` (defaults in
`config/bim_scene.yaml` and `config/bim_export.yaml`) and writes IFC4:

- project, site, building and one storey (`IfcBuildingStorey`);
- `IfcWall` with `IfcOpeningElement` voids, joined at corners and T-junctions,
  with the measured height where there is one. An opening takes both faces of
  its host wall, so lintel, sill and jambs are flush with it;
- `IfcDoor` and `IfcWindow` hosted in the walls with their lining and panel
  properties. Windows have a lining, mullions, panes and a `PartitioningType`.
  A door with a measured open leaf gets `OperationType` `SINGLE_SWING_LEFT` or
  `SINGLE_SWING_RIGHT` from its hinge and swing, named as IFC4 defines it: seen
  from the side the door opens towards, the hinge is on the left (a door hinged
  at the `-axis` end of its opening that swings to the `-normal` side, or one
  hinged at the `+axis` end that swings to the `+normal` side, is
  `SINGLE_SWING_LEFT`); the properties `HingeSide`, `SwingNormalSide` and
  `LeafOpeningAngleDeg` repeat the measurement and the property panel of the
  viewer shows them. Without a measured leaf the operation stays the default.
  A passage is only the `IfcOpeningElement` in its wall (no `IfcDoor`);
- `IfcSpace` per room with name and area (door reveals belong to the spaces)
  and the source of its floor and ceiling height, `IfcSlab` for the floor and
  `IfcCovering` for the ceilings, per room and flat, sloped or with two planes
  (ceiling and wall do not overlap);
- `IfcColumn` (rectangle or circle), `IfcBeam`, and `IfcStair` with
  `IfcStairFlight` (straight stairs), each only with evidence in the points
  (`structure` in `config/ortho_plan.yaml`);
- furniture and equipment from the plan, as `IfcFurniture`,
  `IfcSanitaryTerminal`, `IfcElectricAppliance`, `IfcAudioVisualAppliance`
  (screens), `IfcLightFixture`, `IfcSpaceHeater` or `IfcBuildingElementProxy`
  (mapping in `config/bim_furniture.yaml`, heights in `config/bim_export.yaml`),
  each with a type object and the property set `SAMcloud_Scan` (instance ID,
  label, semantic label, confidence, height source, layer, catalogue kind, size,
  shape source);
- objects on the ceiling on layer `A-CLNG-OBJ`, as `IfcLightFixture`,
  `IfcSensor` (smoke detector), `IfcFireSuppressionTerminal` (sprinkler),
  `IfcAirTerminal` (grille), `IfcAudioVisualAppliance` (projector) or a typed
  proxy when the kind is unknown.

### Shapes and objects from the scan

- **Furniture** (`samcloud/bim/furniture_recon/`,
  `config/furniture_recon.yaml`): each item gets a shape from the points in
  its footprint. `shape_source` is `parametric` (parts such as table top and
  legs, seat, back and armrests, boxes), `mesh` (compact closed mesh for
  fixtures, lamps, plants) or `footprint` (too little evidence, the footprint
  body stays). A part that is not measured is marked `symmetry` or `prior`.
  Chairs, tables and kitchen inserts that the plan lost can be added from
  their scan instance. The environment variable `SAMCLOUD_FURNITURE_RECON=0`
  switches the reconstruction off.
- **Room objects** (`samcloud/bim/room_objects/`, `config/room_objects.yaml`):
  wall-hung and resting objects keep their measured bottom, screens need both
  a screen label and a flat upright surface, and ceiling objects must reach
  the ceiling measured around them. Mirror images, curtain rails and glare are
  rejected. Sizes come only from measurements: the classified cloud, depth
  voxels that COLMAP supports and, for screens, photo outlines. A screen that
  has too few points can be measured from its mask outlines
  (`screen_masks.json.gz`, written by the classification): the same four
  corners must agree in at least three photos from different places and form
  an upright rectangle. Clouds classified before this file existed do not have
  it until they are reclassified.

### 3D DXF and building IFC

The 3D DXF is the written IFC as meshes (`samcloud/dxf/writer3d.py`): metres,
IFC coordinates, one layer `3D-<IFC class>` per class, GlobalId, class and
name as XDATA. Spaces and opening voids are not written. A failure is logged
and does not fail the IFC step.

With more than one level `products/bim/building/<project>.ifc` combines the
levels into one building with one storey each and cuts the slab where a stair
arrives. The levels must share plan frame and scale (within 1 %); otherwise
`building.json` gives the reason and no file is written. This is tested on
synthetic data only.

Validation runs on every export: an **IDS** check against
`config/bim/samcloud.ids` (mandatory attributes per element class),
EXPRESS/schema checks and a re-import of the file. `qa.json` and `qa.html` also
check wall and ceiling overlaps and the consistency with the plan: counts of
walls, rooms, doors, windows, furniture, ceiling objects, columns, beams and
stairs, the room area, and `openings_consistency`, which compares every door
and window of the DXF plan with the IFC. `opening_flush` checks that lintel
and jambs are flush with their wall (5 mm): a step prevents "IFC4 OK", a
measured reveal is listed as `proven`, and an opening that is wider than the
space between two crossing walls is a warning under `blocked_spans`.

### Outputs

`products/bim/<level id>/`:

| File | Content |
|---|---|
| `<project>_<level id>.ifc` | the IFC4 model |
| `qa.json`, `qa.html` | validation and consistency report |
| `<stem>.validation.*`, `<stem>.ids.html` | raw validation and IDS reports |
| `model.glb`, `props.json` | viewer model and element properties |
| `export.json` | manifest with the hash of `plan_model.json`, counts, IDS failures |

The IFC is *fresh* while the hash of `plan_model.json` in `export.json` matches
the plan; after a new plan the page shows a "stale" notice. **Compute
everything** also repeats the IFC step when the 3D DXF is missing or older
than the IFC.

Command line (same code as the job, without the page):

```bash
python -m samcloud.bim.jobs export --project projects/<project> --level level_1
```

Exit codes: 0 ok, 1 failed, 2 cancelled.

## IFC viewer

Below the DXF preview the page shows the IFC model in 3D (`/api/bim/<level>/model.glb`,
`props.json` and `ifc`):

- element classes can be shown or hidden (walls, ceilings, ceiling objects,
  floors, doors, windows, columns/beams/stairs, furniture, sanitary, appliances
  with lamps, screens and radiators, built-ins, rooms, openings, other). The furniture of the plan is split over *furniture* (`IfcFurniture`), *sanitary*, *appliances* and *other* (proxies), so each count is a part of the total; ceilings,
  rooms and openings are hidden by default so the view looks into the flat,
  while ceiling objects stay visible;
- a click selects an element and opens its property panel: name, class, type,
  GlobalId, for windows and doors the width, height, sill (BRH), head (STH),
  mullions and glazing, then the quantities and property sets;
- a horizontal section slider cuts the model at a chosen height;
- **Fit** and a full-screen mode; the IFC can be downloaded from the result
  card (**Download IFC**).

## Example results

The two projects below are processing runs of the same recording of one flat,
so they are no independent validation. The table and the totals after it are
from their earlier exports (2026-11-19), before the metric scale decision,
the wall corrections and the new objects; the dimensions still contain the
storey-height scale estimate. The export of the current code is shown in the
[indoor example](examples/indoor_360.md#current-dxf-and-ifc-test_home_v12_tag).

| | `test_home_v10_tag` | `test_home_v11_tag` |
|---|---|---|
| Rooms | 5 (Schlafen 19.4 m² including the hall behind the passage, Schlafen 13.0, Wohnen / Küche 46.8 including the entrance area, Bad 5.3, Bad 3.5) | 5 (Gäste-WC 3.1, Bad 5.4, Zimmer 13.7, Schlafen 20.5 including the small corridor behind the passage, Wohnen / Küche 48.6 including the entrance hall); 6 IfcSpace, a wall splits the bedroom into two pieces |
| Doors | 5 (entrance door 1.00 m, four interior doors 0.75 to 0.90 m, head 2.05 to 2.13 m), 4 of them with a measured open leaf, plus 1 passage (0.70 m, head 2.45 m, headroom 0.05 m) | 5 (0.80 to 1.60 m, head 2.00 to 2.08 m), 3 of them with a measured open leaf, plus 1 passage (0.90 m, head 2.45 m) |
| Windows | 2 in the living room: 1.74 m (BRH 1.09 m) and 1.80 m (BRH 0.80 m), each with 1 mullion | 2: 1.85 m (2 mullions, BRH 0.65 m) and 1.77 m (1 mullion, BRH 1.02 m) |
| Furniture | 46 items | 42 items |
| IDS failures | 0 | 0 |
| DXF vs IFC openings | identical | identical |

The QA report checks every room: its IfcSpace area must match the plan room within 3 % (plus 0.25 m² per room for door reveals); a room that the wall clipping split, shrank or dropped is listed and the summary then reads "with findings" (`room_area_check` in `qa.json`).
Rooms go from door to door and include half of each door reveal, so the room
areas add up to the measured interior floor, in those earlier exports 88.0 m² (v10) and 91.2 m² (v11) in
the plan; in the IFC the spaces, trimmed at the wall faces, add up to 87.4 m²
(v10) and 90.7 m² (v11). The kitchen run is part of the model (base cabinets
0.86 m and worktop to 0.90 m, sink and hob on it, upper and tall cabinets at
their measured heights).
