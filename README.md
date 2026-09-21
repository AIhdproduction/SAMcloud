# SAMcloud

SAMcloud is a local photogrammetry application that reconstructs 3D point
clouds from drone, standard-camera, and 360 degree Insta360 imagery with
COLMAP and assigns semantic LAS classes with SAM3. The project is designed for
the Digital Twin Programming module at HSLU.

## Project overview and objective

The objective is to provide a guided workflow comparable to common
photogrammetry applications: select drone images, align cameras, optionally
define ground control points, generate a dense point cloud, classify it, and
export a reusable LAS file. Every job is stored as a reopenable local SAMcloud
project.

The application supports three coordinate modes:

- `LOCAL` for unreferenced reconstructions.
- `EPSG:4326` for WGS84 GPS or RTK coordinates from drone images.
- `EPSG:2056` for Swiss LV95 coordinates.

The export CRS is configurable per project. Any coordinate reference system
accepted by PROJ can be used, not only LV95.

## Innovation and independent work

SAMcloud combines image-based photogrammetry, prompt-based semantic
segmentation, LAS classification, and project persistence in one local
workflow. The project differs from a generic point cloud viewer because it
keeps the relationship between drone images, reconstructed cameras, manually
defined control point observations, and the final classified point cloud.

The scope covers outdoor and indoor image datasets. It does not aim to replace
a commercial photogrammetry suite or provide a general-purpose BIM editor.

## Functionality

The target workflow is divided into independent stages:

1. Create or reopen a persistent project and select drone GPS/RTK, standard
   camera, or 360 degree camera capture. Choose either **No control points**
   for an uninterrupted automatic run, or **Use control points after
   alignment** to pause before dense reconstruction.
2. Select image data and configure camera, GPU, semantic classes, and coordinate settings.
   A 360 degree project accepts an exported 2:1 MP4 video, a directory of 2:1
   JPG panoramas, or both sources together.
3. Start processing. COLMAP aligns the images and creates the sparse model.
4. If **No control points** was selected, dense reconstruction and SAM3
   classification continue automatically. If control points were enabled, the
   run pauses after alignment so the points can be added before continuing.
5. Inspect the dense cloud in RGB or class mode and isolate individual classes.
6. Export the classified PLY to LAS manually after checking the result.

The UI is built with NiceGUI. Potree is used as the intended high-performance
viewer for large dense point clouds, including class visibility filters. The
existing command-line processing engine is retained in `samcloud/engine/`
while the UI workflow is developed around it.

## Architecture

```text
main.py
└── samcloud/
    ├── app/                 NiceGUI pages and application state
    ├── core/                project persistence, configuration, workflow state
    ├── colmap/              COLMAP command construction and integration
    ├── control_points/      CSV data and image observation persistence
    ├── classification/      SAM3 class configuration services
    ├── pointcloud/          LAS and coordinate transformation services
    ├── viewer/              Potree viewer manifests and class visibility
    └── engine/              existing COLMAP and SAM3 processing implementation
```

The `config/` directory contains editable defaults. The `projects/` directory
contains local, reopenable project folders and is intentionally excluded from
Git. More details are available in [the architecture documentation](docs/architecture.md).

## Development environment

- Language: Python 3.13.x (the pinned dependency set is verified with Python
  3.13; Python 3.14 is not supported by the pinned NumPy release)
- Environment: `venv` created by `setup.bat`
- Editor: Visual Studio Code
- Version control: Git and a public GitHub repository
- External modules: NiceGUI, Pandas, PyProj, PyYAML, COLMAP, Ultralytics SAM3,
  laspy, NumPy, OpenCV, and Matplotlib

### Verified Windows versions

The following versions are the reference stack for this repository:

| Component | Version / build |
|---|---|
| Python | 3.13.x (Windows x64) |
| COLMAP | 4.2.0, Windows CUDA, `colmap/COLMAP.bat` |
| PyTorch | 2.14.0 + CUDA 12.6 (`cu126`) |
| TorchVision | 0.29.0 + CUDA 12.6 (`cu126`) |
| NumPy | 2.2.6 |
| NiceGUI | 3.17.1 |
| Ultralytics | 8.4.138 |

The complete Python pin set is maintained in
[`requirements.txt`](requirements.txt). Do not replace the CUDA COLMAP build
with the `nocuda` archive: dense reconstruction requires the CUDA build.

Pandas is used to load, validate, and write ground control point CSV files.
PyProj validates and transforms configured coordinate reference systems.
NiceGUI provides the local user interface.

## Installation and start on Windows

1. Install Python 3.13.x (Windows x64). `setup.bat` selects it explicitly via
   `py -3.13`; no administrator password is required.
2. Download the Windows CUDA release of COLMAP 4.2.0 from
   <https://github.com/colmap/colmap/releases/tag/4.2.0>.
3. Extract it to `colmap/` so that `colmap/COLMAP.bat` exists. The installed
   build must print `COLMAP 4.2.0 ... with CUDA` when running
   `colmap\COLMAP.bat --help`.
4. Request SAM3 model access at <https://huggingface.co/facebook/sam3>.
5. Store the downloaded checkpoint as `models/sam3.pt`.
6. Run `setup.bat` once from the repository root. It creates the Python 3.13
   `venv`, checks COLMAP, and installs the pinned CUDA/Python dependencies.
   The script uses retry-friendly pip timeouts for large CUDA packages.
7. Run `start.bat`. Keep its console window open and open
   `http://127.0.0.1:8080` in a browser if it does not open automatically.

To verify the installation before starting the UI:

```bat
venv\Scripts\python --version
colmap\COLMAP.bat --help
venv\Scripts\python -c "import nicegui, torch, ultralytics; print(torch.__version__, torch.cuda.is_available())"
```

If `start.bat` closes immediately, run it from an existing Command Prompt in
the repository root. The error remains visible there; the most common cause
is an incomplete `setup.bat` run or an existing `venv` created with another
Python version.

The virtual environment, model weight, processing data, generated point
clouds, and local project folders are excluded by `.gitignore` and must never
be committed to the public repository.

## Configuration

| File | Purpose |
|---|---|
| `config/defaults.yml` | project defaults, coordinate systems, and workflow defaults |
| `config/colmap.yml` | camera, matching, GPS/RTK, control point, and dense settings |
| `config/sam.yml` | SAM3 model, confidence, tiles, and voting settings |
| `config/classes.json` | semantic prompts and class order |
| `config/viewer.yml` | point cloud viewer and class visibility defaults |

Each new project receives a copy of the defaults in
`projects/<project-name>/project.samcloud.yml`. Values saved there apply only
to that project. The complete setting reference is in
[docs/configuration.md](docs/configuration.md).

## Project files

```text
projects/<project-name>/
├── project.samcloud.yml       project state and project-specific settings
├── inputs/                    imported metadata and optional local copies
├── control_points/            CSV points and image observations
├── colmap/                    COLMAP database and reconstruction files
├── products/
│   ├── sparse/                sparse cloud viewer data
│   ├── dense/                 dense reconstruction results
│   ├── classified/            PLY and LAS exports
│   └── viewer/                generated Potree data and viewer manifest
└── reports/                   statistics, accuracy, and export reports
```

Source images are referenced at their original location by default. This
avoids duplicating large drone datasets while preserving the complete project
state. If images are moved, their path can be updated in the project settings.
For 360 degree projects, extracted video frames, cubemap views, and a source
manifest are stored below `inputs/`. Capture settings are locked once those
derived inputs are prepared.

## 360 degree Insta360 workflow

Export the footage from Insta360 Studio before creating the project. SAMcloud
supports stitched 2:1 equirectangular `.mp4` videos and `.jpg` panorama photos;
raw `.insv` files are not supported. Select `360 degree camera / Insta360`, then
choose `Indoor` or `Outdoor`. Add either source or both. Video frames are
extracted at a fixed interval of one second by default and converted into six
perspective cubemap views per panorama for the COLMAP 4.0.3 workflow.

Projects without drone GPS start in `LOCAL` coordinates. Add control points to
georeference them before exporting to a real coordinate reference system.

## Processing modes

Every project stores `workflow.control_points_mode` as either `none` or
`after_alignment`. The default `none` mode runs alignment, dense reconstruction,
and SAM3 classification without waiting for manual points. The
`after_alignment` mode deliberately pauses after COLMAP alignment. In both
modes the classified PLY is produced first; LAS export is a separate manual
action from the dashboard so the result can be reviewed before export.

## Tests

The `tests/` directory contains one test file for each core area: persistent
projects, control point CSV files, and coordinate-system validation. Run them
after setup with:

```bat
venv\Scripts\python -m pytest
```

## Author and group size

Single-developer project.

## License

SAMcloud is source-available under the [SAMcloud Personal and Educational Use
License](LICENSE). Private, educational, research, and other noncommercial use
are permitted. Companies require prior written agreement with the copyright
holder and a separate commercial license.
