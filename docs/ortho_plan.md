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
unscaled 360 scans still work, `units_per_metre` is derived automatically):

- walls from measured wall faces in the upper height band, straightened and
  merged, with thickness estimated from the points; weak boundaries and stubs
  filtered (`wall_lines.py`, `regularize.py`);
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
- Indoor plans work best on georeferenced or scaled projects; unscaled scans
  rely on the ceiling height as scale.
- Wall thickness is estimated from points, not measured.
- Heavy stages use the GPU and can take minutes to hours on large projects.
