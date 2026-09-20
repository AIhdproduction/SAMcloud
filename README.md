# SAMcloud

SAMcloud is a local photogrammetry application that reconstructs 3D point
clouds from drone images with COLMAP and assigns semantic LAS classes with
SAM3. The project is designed for the Digital Twin Programming module at HSLU.

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

The scope is outdoor drone imagery. It does not aim to replace a commercial
photogrammetry suite or provide a general-purpose BIM editor.

## Functionality

The target workflow is divided into independent stages:

1. Create or reopen a persistent project.
2. Select image data and configure camera, GPU, and coordinate settings.
3. Align images with COLMAP and inspect the sparse point cloud and cameras.
4. Import ground control points as `name,x,y,z` CSV and mark them in images.
5. Create a dense COLMAP point cloud.
6. Select the indoor or outdoor semantic class set and run SAM3 classification with multi-view voting.
7. Inspect the dense cloud in RGB or class mode and isolate individual classes.
8. Export classified PLY and LAS data in the chosen coordinate system.

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

- Language: Python 3.10 or later
- Environment: `venv`
- Editor: Visual Studio Code
- Version control: Git and a public GitHub repository
- External modules: NiceGUI, Pandas, PyProj, PyYAML, COLMAP, Ultralytics SAM3,
  laspy, NumPy, OpenCV, and Matplotlib

Pandas is used to load, validate, and write ground control point CSV files.
PyProj validates and transforms configured coordinate reference systems.
NiceGUI provides the local user interface.

## Installation and start on Windows

1. Install Python 3.10 or later and add it to `PATH`.
2. Download the Windows CUDA release of COLMAP from
   <https://github.com/colmap/colmap/releases>.
3. Extract it to `colmap/` so that `colmap/COLMAP.bat` exists, or install it
   elsewhere and add it to `PATH`.
4. Request SAM3 model access at <https://huggingface.co/facebook/sam3>.
5. Store the downloaded checkpoint as `models/sam3.pt`.
6. Run `setup.bat` once. It creates the local `venv` and installs all Python
   dependencies.
7. Run `start.bat` to open SAMcloud at `http://127.0.0.1:8080`.

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
