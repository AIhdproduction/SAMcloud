# IFC export

On indoor projects the DXF page is the **DXF/IFC** page. **Compute
everything** produces the clean DXF plan and an IFC4 building model with one
button. Both files come from the same building model, `plan_model.json`, so
walls, rooms, doors and windows are identical in the DXF and the IFC. The code
is in `samcloud/bim/` (export, viewer assets, 3D opening refinement); the jobs
are in `samcloud/dxf/jobs.py`.

The export is indoor only for now. For outdoor projects **Compute everything**
skips the refinement and the IFC step and logs "IFC export: indoor only".

## Stages

**Compute everything** (`run_all`) runs, for an indoor level:

| # | Step | Result |
|---|---|---|
| 1 | Build orthophoto (`ortho`) | true orthophoto, height rasters |
| 2 | Segment orthophoto (`segment`) | per-pixel class raster |
| 3 | Generate clean plan (`clean_plan`) | `plan_model.json`, first plan pass |
| 4 | **Refine openings** (`refine_openings`) | `openings_3d.json`; the clean plan is generated again with the refined openings |
| 5 | Export DXF (`export`) | `<project>_<level>_clean.dxf` |
| 6 | **Export IFC** (`export_ifc`) | `products/bim/<level id>/` |

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
    opening, or a lintel (headroom at least 0.30 m, measured head at least
    1.85 m) on an observed wall;
  - **passage**: no door evidence, headroom below 0.30 m and the opening cells
    measured free of points. It stays an `IfcOpeningElement` without `IfcDoor`,
    rooms on both sides are one room;
  - **no opening**: neither door evidence nor a measured free gap (unobserved,
    blocked by furniture in the wall band, or not on the observed wall): the
    candidate is dropped and the wall closes. Weak evidence (one jamb strip or
    only the head strip, door-class points in the gap) keeps the door.
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
- `IfcWall` with `IfcOpeningElement` voids, joined at corners and T-junctions;
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
- `IfcSpace` per room with name and area, `IfcSlab` for the floor and
  `IfcCovering` for the ceilings (ceiling and wall do not overlap);
- furniture and equipment from the plan, as `IfcFurniture`,
  `IfcSanitaryTerminal`, `IfcElectricAppliance` or `IfcBuildingElementProxy`
  (mapping in `config/bim_furniture.yaml`, heights in `config/bim_export.yaml`),
  each with a type object and the property set `SAMcloud_Scan` (instance ID,
  label, semantic label, confidence, height source, layer, catalogue kind, size).

Validation runs on every export: an **IDS** check against
`config/bim/samcloud.ids` (mandatory attributes per element class),
EXPRESS/schema checks and a re-import of the file. `qa.json` and `qa.html` also
check wall and ceiling overlaps and the consistency with the plan: counts of
walls, rooms, doors, windows and furniture, the room area, and
`openings_consistency`, which compares every door and window of the DXF plan
with the IFC.

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
the plan; after a new plan the page shows a "stale" notice.

Command line (same code as the job, without the page):

```bash
python -m samcloud.bim.jobs export --project projects/<project> --level level_1
```

Exit codes: 0 ok, 1 failed, 2 cancelled.

## IFC viewer

Below the DXF preview the page shows the IFC model in 3D (`/api/bim/<level>/model.glb`,
`props.json` and `ifc`):

- element classes can be shown or hidden (walls, ceilings, floors, doors,
  windows, furniture, sanitary, appliances, rooms, openings, other). The furniture of the plan is split over *furniture* (`IfcFurniture`), *sanitary*, *appliances* and *other* (proxies), so each count is a part of the total; ceilings,
  rooms and openings are hidden by default so the view looks into the flat;
- a click selects an element and opens its property panel: name, class, type,
  GlobalId, for windows and doors the width, height, sill (BRH), head (STH),
  mullions and glazing, then the quantities and property sets;
- a horizontal section slider cuts the model at a chosen height;
- **Fit** and a full-screen mode; the IFC can be downloaded from the result
  card (**Download IFC**).

## Example results

Both projects, from `qa.json` and `plan_model.json` of the current runs:

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
areas add up to the measured interior floor: 88.0 m² (v10) and 91.2 m² (v11) in
the plan; in the IFC the spaces, trimmed at the wall faces, add up to 87.4 m²
(v10) and 90.7 m² (v11). The kitchen run is part of the model (base cabinets
0.86 m and worktop to 0.90 m, sink and hob on it, upper and tall cabinets at
their measured heights).
