# Architecture

SAMcloud separates the interface, project persistence, photogrammetry,
semantic classification, and point cloud delivery so that each part can be
developed and tested independently.

```text
NiceGUI pages
    -> core project manager
        -> COLMAP alignment and dense reconstruction
        -> control point CSV and image observations
        -> SAM3 classification
        -> LAS export and Potree viewer data
```

The `samcloud/engine/` folder contains the existing command-line processing
implementation. It is preserved as the processing core while the UI services
are developed around it. New code should access it through the named service
packages instead of placing application logic in `main.py`.

Each user project is independent and can be reopened from its
`project.samcloud.yml` file. Source images remain in their original folder by
default; the project metadata stores the path and all generated results,
control point observations, reports, and viewer data are kept under the local
project directory.
