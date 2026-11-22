# Orthophoto clean plan

The DXF page can produce two plan variants from a classified project. The
**point cloud** variant cuts layers straight from the classified LAS (see the
README, *DXF export*). The **orthophoto (clean)** variant described here builds
a true orthophoto and a height raster from the cloud and the original photos,
segments the orthophoto, and vectorises the result into a cleaner plan with
straightened walls, catalogue blocks, room names and, outdoors, contours. The
code is in `samcloud/ortho/`; the jobs are in `samcloud/dxf/jobs.py`.

## Stages

Each stage is a button on the DXF page (indoor: per checked level; outdoor: the
site) and runs as a cancellable background job with its own cache. A stage is
enabled when the previous one is up to date; each shows a status chip
(*missing*, *up to date*, *stale*). **Compute everything** (`run_all`) runs
all of them, skipping stages whose cache is fresh unless **Force** is ticked.

| # | Job | Result | Cache |
|---|---|---|---|
| 1 | **Build orthophoto** (`ortho`) | true orthophoto, height rasters | `products/ortho/<level id>/` (`site/` outdoors) |
| 2 | **Segment** (`segment`) | per-pixel class raster of the orthophoto | same folder |
| 3 | **Generate clean plan** (`clean_plan`) | vector plan | `products/dxf/<level id>/ortho/` (`site/ortho/`) |
| 4 | **Export DXF** (`export`, plan variant *Orthophoto (clean)*) | DXF file | `products/dxf/export/<project>_<level>_clean.dxf` |

### 1. Orthophoto

`samcloud/ortho/build.py` reads the classified cloud (the same LAS/PLY the DXF
page uses, with the DXF frame: scale, gravity alignment, level) and builds:

- a **DSM** (surface height), a **DTM** (ground height) and a class raster from
  the cloud (`dsm.py`, `classes_pc.npy`), with indoor floor gaps closed from the
  scanned-room footprint (`dsm.surface_fill_indoor`);
- reduced depth buffers of the cloud per camera for visibility testing
  (`zbuffer.py`);
- the colour of every cell from the best views of the original photographs,
  feathered and blended (`colorize.py`, `blend.*`), with a limit on how far a
  distorted projection may leave the image so folded-back rays cannot smear
  colours onto the ground.

Indoors the cut is below the ceiling (`indoor.cut_below_ceiling`); outdoors the
whole surface is used. Default pixel size is 1 cm indoors and 2 cm outdoors.
Outputs: `ortho.png` (RGBA), `ortho_8000.jpg`, `ortho_4096.jpg`, `preview.jpg`,
`dsm.npy`, `dtm.npy`, `classes_pc.npy`, per-point fine labels (`fine_pc.npy`),
instance IDs (`instance_pc.npy`), `valid.png`, `view_count.png`, world files
(`ortho_local.pgw` and, when the project is georeferenced, `ortho.pgw`), and
`ortho.json` with the fingerprint. The page offers links for the 8000 px JPEG,
the 4096 px JPEG and the full PNG, and opens the folder.

### 2. Segmentation

`samcloud/ortho/segment.py` classifies the orthophoto. The **Segmentation model**
selector chooses the source:

| Source | Meaning |
|---|---|
| Point cloud | the classes already carried by the cloud |
| SAM3 | SAM3 run on orthophoto chunks (`chunk_px`, `chunk_overlap_px`) |
| FlurSeg | FlurSeg on the orthophoto (outdoor only) |
| Combined | the default. The image model fills pixels the cloud left unclassified. Where both disagree, the image model must reach `min_image_confidence` (0.5) and the height above ground decides: classes of the *high* code group (buildings, poles, ... at least `high_threshold` = 0.5 m above ground) or the *low* group (ground, road, water, ... at most `low_threshold` = 0.2 m) prefer the model that fits the height; otherwise the higher confidence wins (the cloud counts as `pointcloud_confidence` 0.6). The code groups are set in `ortho_segment.yaml`. |

Indoors, per-point fine terms (sofa, toilet, ...) are taken from the cloud when
available (`fine_pc.npy`) so the fine SAM pass on the orthophoto is skipped;
otherwise the indoor `fine_prompts` run on the orthophoto. Results:
`class_<source>.npy/png`, `conf_<source>.npy`, `fine_class.npy`, `fine_conf.npy`,
`fine_labels.json`, `segment_<source>.json`.

### 3. Clean plan

`samcloud/ortho/plan.py` (`generate_clean_plan`) combines the orthophoto, the
class raster, the height raster, the point cloud and the instance labels. Its
fingerprint includes the segmentation source, `instances.json`, and the
settings, so a changed input marks the plan stale.

**Indoor** (needs a level; thresholds are relative to the storey height so
unscaled 360 scans still work). The detectors always work in units normalised
by the storey height. The results are converted with `units_per_metre`: 1.0
when the cloud has metric evidence (`samcloud/core/metric_scale.py`,
`clean_plan.metric_scale`), else the storey height over 2.5 m. The reason is
stored as `source.scale_source` in `plan_model.json`.

- walls from measured wall faces in the upper height band, straightened and
  merged, with thickness estimated from the points; weak boundaries and stubs
  filtered (`wall_lines.py`, `regularize.py`);
- slanted walls at least 5 degrees off the two main directions
  (`wall_directions.py`); a piece that deviates by less than 0.40 m over its
  length is not used, so short walls at a small angle are missed;
- columns (`A-COLS`), beams (`S-BEAM`, dashed), straight stairs, and wall,
  floor and ceiling heights from the points (`structure/`); stairs and
  heights go to `plan_model.json` and the IFC;
- openings: doors and windows detected from wall gaps, door and window classes,
  and sills and lintels, drawn as door and window blocks;
- room boundaries from the ceiling (`A-AREA`) and **semantic room names** such
  as Bad, Küche, Schlafen, Wohnen from the fixtures inside (`room_names.py`);
- **catalogue blocks** for toilet (`WC_WALL`), washbasin, bathtub, shower, cooktop,
  kitchen sink, kitchen island, beds, dining tables, sofas and wardrobes in
  standard sizes, snapped to walls (`symbols.py`); sanitary blocks are placed
  per instance;
- a **kitchen run** along the walls (straight, L or U) with cooktop and sink
  inserts (`kitchen_run.py`), upper cabinets on `A-OBEN` (`upper_cabinets.py`),
  and recessed **niches** such as an open wardrobe or coat niche (`niche.py`);
- other furniture traced from the height map as an outline with height-step
  detail lines (`A-FURN-DETL`) and a text label such as "Sofa h=0.85";
  built-in fronts that are really a unit's face go to `A-EINB`
  (`builtin_fronts.py`, `vertical.py`);
- instance labels name furniture and fixtures (Regal, Tisch, Stuhl, Schrank).

### Double-line walls

With `walls.double_line: true` in `config/ortho_plan.yaml` (the default) every
indoor wall is drawn with both faces as a closed outline on `A-WALL`, filled on
`A-WALL-PATT` (`samcloud/ortho/vector/double_walls.py`):

Two parallel faces that the scan proves to be the two sides of one wall are
paired before these steps (`wall_sides.py`: rooms on both outer sides, no
ceiling running through, no third face between); such a wall stands on both
measured faces. The steps below decide the other walls.

1. **Solid side.** The side of a measured face with a room (ceiling room or free
   floor) in front of it is the room side; the other side is the wall body. Faces
   with no room on either side are oriented by a measured partner face, else the
   unmeasured (void) side is the outside.
2. **Measured thickness.** Two parallel faces whose bodies face each other at
   `min_thickness_m`..`max_thickness_m` are one wall; the distance is its thickness.
   A partner seen only beside a face (within `pair_gap_m`) lends its offset too.
3. **Interior or exterior.** Behind the wall body the next room is searched: a
   ceiling room within `probe_m` (or free floor right behind the wall) makes an
   interior wall, whose thickness may also come from that room's edge; nothing
   behind it makes an exterior wall. Door and window spans do not decide the kind.
4. **Learned thickness.** Measured pairs and room-edge distances are clustered
   into interior thickness classes; exterior walls use the measured exterior pairs
   or `exterior_thickness_m`. Collinear faces of one wall line share the measured
   thickness and are aligned on one line, so a face seen only in part still gets a
   straight, consistent second face.
   Above a door with a measured head two scanned planes replace such an assumed
   thickness of wall and lintel (`lintel_band`); a measured face never moves.
5. **Wall bodies.** Every face run becomes a band on its solid side; ends are
   extended into L corners, gaps up to `joint_reach_m` are closed, bands are united
   (clean L/T/X joints) and each door and window is cut through the full thickness
   of its host wall only (closed jambs). Steps and notches below `jog_m` are removed.
6. **Outline completion.** Where a room edge borders unmeasured outside and no
   wall was measured (for example behind kitchen units), an exterior wall is added
   at the end of the measured data, continuing the neighbouring wall line.

7. **Evidence checks.** A face whose measured data never forms a wall (wall-class
   points over `face_min_height_m` reaching `face_reach_m`, or a floor-connected
   surface such as a curtain) while furniture stands at it is a furniture front
   (counter, upper cabinet) and is dropped. A face that no camera of the level sees
   without looking through another face (a mirror phantom) is dropped.
8. **Doors.** A door detected as its open leaf (perpendicular to a wall, one end at
   it) becomes the opening in that wall next to the hinge, fitted to the stretch that
   is open at sill and mid height; a measured face made of door points standing at a
   wall is converted the same way. Doorways (no sill or mid-height points across the
   wall, floor seen through it) are added, a door span that is solid wall is dropped,
   a window overlapping a door keeps only its part beside the door, and wall stubs
   shorter than `min_stub_m` next to an opening are not built.
9. **Tall built-in units** (`tall_units: true`). A fridge, oven tower or tall kitchen
   cabinet hides the wall behind it, and its front can be taken as a wall face (measured,
   or completed from the room edge). A wall run with tall-unit evidence along at least
   `tall_unit_min_m` within `tall_unit_band_m` of its face (fine label kitchen cabinet at
   1.6 m or more, fridge, oven) is a unit front when the room in front of it is open floor,
   a wall stands behind it at a unit depth (`tall_unit_min_depth_m`..`tall_unit_max_depth_m`)
   and its footprint holds neither open floor nor observed low surfaces (counter tops,
   furniture; `tall_unit_max_free`, `tall_unit_max_low`). The wall behind is a wall body
   that rays from the front reach, or else a parallel wall on the same side ending next to
   the unit (`tall_unit_neighbour_gap_m`): that hidden wall is continued behind the unit to
   its far end. The unit reaches a perpendicular wall body within `tall_unit_side_gap_m`
   (blind corner). Wall runs whose body lies in a unit are removed; the unit is drawn on
   `A-EINB` as a rectangle with a cross and "KS" where fridge evidence lies at its front
   (a long unit is split into a fridge module of `tall_unit_fridge_m` at the end holding it)
   or "HS". The kitchen counter ends at such a unit like at a wall (a counter end within
   0.30 m reaches its side). Same idea as the bedroom wardrobe rule (`builtin_fronts.py`),
   which works on the single wall lines before this stage.

Short faces with no other wall nearby (furniture sides) and the reveals of
openings are not walls. Rooms (`A-AREA`) then follow the inner faces: a room grows
up to `room_gap_m` towards the faces and fills a wall-enclosed cell it is alone in;
two ceiling fragments with no wall between them are one room when either is below
`room_merge_max_area_m2`. Window blocks span the wall thickness and door jamb lines
run from face to face.

### Door classification in the second pass

On an indoor level the plan is generated twice (see [DXF and IFC](ifc_export.md#refine-openings)). The second pass
applies the refined openings, and the 3D refinement decides for every door candidate whether it is a **door**, a
**passage** or **no opening at all**:

- **door**: a casing of door-class points (Cyclone 103) around the opening (a jamb strip and the head strip), or a
  closed leaf that fills the opening, or a lintel on an observed wall (at least 0.15 m of wall above a measured head of
  at least 1.85 m or 0.66 of the storey height; where the first pass saw no opening, also a measured free gap with floor
  under it). Weak door evidence (a single jamb strip, door-class points in the gap) keeps the safe default, a door,
  where the first pass had one.
- **passage**: no door evidence, the opening reaches the ceiling (headroom below 0.15 m, measured) and its cells are
  measured free. A passage stays an opening without leaf or swing arc, rooms on both sides are one room, and its head
  height is the measured one (about 2.4 m instead of the 2.0 m default).
- **no opening**: neither door evidence nor a measured free gap, or no wall beside the opening that was actually
  observed (furniture in the wall band, a gap in unobserved wall). The candidate is dropped and the wall closes: the
  wall stage continues the host wall over the span when the measured faces leave a gap there (the candidate still trims the measured faces in the face
  tidy step like the opening of the first pass, so both faces of the wall pair up into one wall of the host thickness), and the 2D door
  heuristics (leaf reorientation, doorway detection, doors made of leaf faces) do not move, drop or add openings of
  the refinement (`walls.refined_openings`, set by the hook).

An **open leaf** never makes a door. It is assigned (hinge side, swing side, opening angle) to a door only when it
hinges at that door's own jamb; a leaf that two doors both find belongs to the one whose jamb is closer. In the DXF a
door with a measured leaf is drawn as the `DOOR` block on the measured hinge, swinging to the measured side, with its
two jamb lines; a door without a leaf, and every passage, is drawn as jamb lines only. The original host wall (the wall pieces of the first pass) stays
beside every refined door and passage up to `classify.host_reach_m`: where the measured faces leave a stretch of it open (an entrance door in a wall nobody
measured), the wall stage continues it (`door-fill` run of the host thickness), so a door never takes its wall away and the room reaches the door line. Wall lines are also scanned
for door frames where no opening exists yet (entrances, `scan:` in `config/bim_openings.yaml`); such a door is added
and hosted on its wall.

### Plan polish

After the double-line walls, `samcloud/ortho/vector/plan_polish.py` cleans the plan
(`plan_polish:` in `config/ortho_plan.yaml`, all lengths in metres × units per metre):

- **Rooms** (`A-AREA` and the room names) are the wall cells: the free cells of the wall
  bodies with door and window spans closed. Their outlines are the inner wall faces, so
  they are rectilinear (except along slanted walls) and never cross a room interior. An open-plan space with no wall
  between its parts is one room (kitchen run + sofa = "Wohnen / Küche"). A cell that holds
  several measured ceiling rooms is split only at a passage: a straight or one-step cut
  between two wall corners, at most `room_split_max_m` long, that leaves each ceiling room
  on its own side (for example the hall that opens to the living room). Every cell with
  floor gets a name and an area, also without fixtures ("Zimmer"; "Büro" with at least two
  desks or monitors); a corridor between doors is a "Flur" from `min_hall_area_m2` (0.5 m2; a cell below the room minimum that two doors open into, `small_hall_min_doors`, is kept as hall).
  Rooms go from door to door (`door_rooms.py`, `indoor_door_rooms:`): the wall cells are completed
  to the whole interior floor and split only by wall bodies, window spans and door lines (through
  the wall centre, across the full door width), so each side gets half of the door reveal. A
  passage (an opening without door leaf or lintel) does not split, furniture never carves or
  splits a room, and a hall or strip that no wall cell encloses belongs to the room it is open to
  (a floor reached only through doors is a room of its own). The shipped config sets
  `room_split_max_m: 0`: a ceiling-room step inside a wall cell no longer splits it. Rooms stay on
  measured ground (`support_*`: the top-view points; unobserved ground outside the wall line never
  becomes floor), wall gaps narrower than any door and missing stretches of an exterior wall (up to
  `exterior_gap_max_m`, bridged along the wall line) are closed for room building, a measured room
  that a door line runs through is cut at the line, and a floor piece closed in by doors is a room
  of its own. A door always separates two rooms, whatever else is open between them (rooms open to
  each other along at least `merge_contact_m` without a wall or door line are one room). A passage (no door)
  never separates. Room edges that follow the contour of the measured ground are
  closed straight along the wall axes (`support_step_m`). In the wall stage a stretch of a measured wall run that has no wall
  points, lies beside kitchen cabinets with room floor on its other side and makes up most of its run
  is a cabinet column (`cabinet_*` in `walls:`): the wall is dropped and a tall unit with its measured
  height (about 2.3 m) is drawn and exported instead.
- **Furniture** (`A-FURN`): fragments of one instance are merged; each outline becomes a
  rectangle along the wall axes when it covers the traced footprint (`furniture_rect_iou`),
  stays a rotated rectangle when the object really stands at an angle, and is otherwise a
  rectilinear outline (steps of at least `furniture_step_m`, at most
  `furniture_max_vertices` corners). Outlines are cut out of the walls and their room;
  traces inside a larger outline or a catalogue block are dropped, overlaps are cut back,
  a sliver narrower than `furniture_sliver_m` glued to a much larger outline is dropped,
  clutter below `furniture_min_area_m2` is dropped and ragged height-step lines are
  removed. A label that cannot be right for the size (a 4 m² "Kissen", a 1.8 m high
  "Bett") becomes "Möbel".
- **Dining tables and chairs** (`footprints.py`, before the furniture loop): table pieces,
  chairs and unlabelled furniture touching them form a dining group with one table top: the
  largest table piece shaped like a top (short side ≥ 0.5 m, ≥ 0.35 m², filling 45 % of its
  rectangle), else an unlabelled piece of that shape at table height (0.60–0.90 m) — a white
  or glass top returns few points, so the instance labels land on the chairs around it.
  Other table pieces with at least 0.06 m² outside the top are chairs. Chair pieces are
  clipped by the top (thin slivers along its edge removed) and merge only while the merged
  piece still fits one chair (0.70 m), so chairs side by side stay separate. A chair within
  0.60 m of a table becomes a seat box on the table axes (0.40–0.55 m, from a piece of at
  least 0.04 m²) pushed out from under the top; two boxes for one seat keep the larger.
  Chairs are labelled "Stuhl" (no height), the table "Tisch h=…"; a table with traced
  chairs keeps its own rectangle instead of the dining-table block with stock chairs.
- **Shape recognition** (`furniture_shapes: true`, `furniture_shapes.py`): every outline is
  checked against its traced footprint (the instance mask, holes filled — a person lying
  on a sofa is not a hole in it). Candidates: the IoU-optimal wall-axis rectangle, a rotated
  rectangle, an **L** (rectangle minus a corner notch: corner sofa, L desk), a **U**
  (rectangle minus a side notch), circle / ellipse for classes that can be round (tables,
  plants, lamps) and the rectilinear trace. The choice minimises `(1 − IoU) + penalty ×
  extra rectangles`; the penalty depends on the class (sofa 0.035, desk 0.06, default
  0.07, bed / table / chair 0.25), a notch must be at least 0.25 m on both sides and 6 %
  of the box, and an L / U / round outline must reach IoU 0.85 on its own (noisy traces
  stay rectangles). A notched or round outline replaces the fitted rectangle; a rectilinear
  trace becomes the best rectangle when that explains the footprint with IoU ≥ 0.70.
  The chosen kind, IoU and boundary distance per object are in the plan metrics
  (`furniture_shapes`).
- **Sofa detail**: the height profile across each edge of a sofa outline finds raised
  bands (backrest / armrest at least 0.12 m above the seat, depth where the profile falls
  half-way to the seat, 0.10–0.40 m). Their fronts are drawn as inner lines on
  `A-FURN-DETL`; on an L / U sofa the inner corner is continued across the seat (corner
  module). The label sits on the seat.
- **Kitchen run** (`A-EINB`): each leg is moved flush onto the final inner wall face behind
  it and all legs get one depth (0.60 m when the measurement is within the tolerance); a
  peninsula keeps its measured position, a diagonal inner-corner unit (chamfer) is kept at
  the moved corner until the scanned fronts decide about it (next point). Loose front lines that retrace the run or double a wall face are
  dropped; upper cabinets move onto the same wall.
- **CAD kitchen** (`kitchen_layout: true`, `kitchen_layout.py`): a wall-backed leg grows
  along its wall while counter heights (0.75–1.05 m) are measured in its front 0.30 m, up
  to a wall face, never into it. A free-standing leg (peninsula) takes a raised band
  beside it (≥ 0.08 m above the counter top, ≤ 1.30 m, at least half the leg length,
  0.15–0.60 m deep) as a bar: it becomes part of the counter and the step is an inner
  line; its free end continues to where counter-or-bar heights end. Cooktop and sink are
  mounted on the counter at their labels measured at counter height (densest label
  cluster), centred in the counter depth and fully inside it. A diagonal measured in the
  scan carries its insert parallel to it; a chamfer that only the label raster suggests
  does not turn it (`appliance_on_chamfer` is off). A full-depth tall unit (≥ 1.60 m) beyond a free leg end becomes a
  rectangle with a cross and "KS" (fridge label) or "HS". Upper cabinets (`A-OBEN`,
  dashed) get a cross. Traced furniture standing on the final counter or bar is absorbed.
  Finally the counter is drawn by its front and end lines only (`_counter_front`); the side
  on a wall face is the wall's line.
- **Kitchen fronts from the scan** (`kitchen_fronts.py`, `kitchen_legs.py`): a front across
  an inner corner becomes a diagonal unit at its measured angle (no evidence: square
  corner), a base steps back under its worktop (`A-FURN-DETL`), an upper cabinet ends
  where the scan sees the wall behind it, and a leg the label raster lost is added when
  its front, worktop, kitchen labels, join to the run and back are all found. A kitchen
  without any detected run gets none of this.
- **Labels**: an object label sits inside its object (horizontal, or along a narrow
  object), else beside it inside the room; labels of objects below
  `label_min_object_area_m2` are omitted. Inside an object a label keeps `label_margin_m`
  from the outline and from the object's own detail lines (sofa seat and backrest lines,
  height steps). Room names and furniture labels never leave
  their room (so they are never under a wall), and the DXF writes fills first and texts
  last, so no wall hatch can cover a text.

**Outdoor** (site plan):

- DTM contours from all ground-level points at 0.20 m resolution: 1 m major
  contours (`S-TOPO-MAJR`, brown-red), 0.25 m minor contours (`S-TOPO-MINR`,
  green), as 3D polylines, with labels on major contours;
- spot heights on a grid (`S-ELEV`, crosses with absolute height text);
- building footprints regularised to their main directions, with roof ridge and
  eave heights;
- trees from canopy-height watershed with neutral tree blocks and, when
  measurable, trunk diameter; hedges and shrubs as smooth outlines;
- curbs, road edges and fences as traced centre-lines, areas as a
  coverage-preserving partition;
- smart text labels inside large areas.

Layer names come from `config/dxf_layers.yaml` (for example `A-WALL`, `A-DOOR`,
`A-WIND`, `A-FURN`, `P-SANI`, `S-BLDG`, `S-VEG`) plus the plan layers of
`config/ortho_plan.yaml`.

### 4. Export

The export reads the cached clean plan, writes an R2018 DXF in metres (local or
original coordinates, as chosen on the page) and audits it with ezdxf before it
is moved into `products/dxf/export/`. The file is named
`<project>_<level>_clean.dxf` (`<project>_site_clean.dxf` outdoors). Export is
refused when the cached plan is stale; run **Generate clean plan** again.

Indoors the export adds what comes from `plan_model.json`: furniture outlines
as the projection of the [3D shapes](ifc_export.md#shapes-and-objects-from-the-scan)
and objects on the ceiling on the dashed layer `A-CLNG-OBJ`. The preview on
the page shows the cached plan without both.

## Review on the DXF page

- **Plan variant** switches the review between the point cloud and the clean
  plan; **Compare plans** overlays both, with separate opacity sliders.
- The background can be the orthophoto, the class colours, or none; the
  orthophoto opacity is adjustable. **Save snapshot PNG** writes the current
  view to `products/dxf/snapshots/`.
- The result viewer draws the real CAD block symbols, not rectangles.

## Configuration

`config/ortho.yaml` (orthophoto), `config/ortho_segment.yaml` (segmentation)
and `config/ortho_plan.yaml` (plan) hold all thresholds; see
[configuration.md](configuration.md). Project-level choices (plan variant,
segmentation source, view options) are stored in `products/dxf/state.json`.

## Limits

- The orthophoto needs the COLMAP model and the source photos; if the photos
  were moved, the stage fails with a message.
- Indoor plans work best on georeferenced or scaled projects; only scans
  without metric evidence rely on the ceiling height as scale.
- Wall thickness is measured only where both faces of a wall are seen; elsewhere
  it comes from the neighbouring room edge, the learned thickness classes or the
  configured exterior default (double-line walls). Outer walls scanned only
  from inside normally get 0.30 m.
- The indoor rules were tried on real data of one flat and one recording, plus
  synthetic scenes.
- Heavy stages use the GPU and can take minutes to hours on large projects.
