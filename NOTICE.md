# SAMcloud notice

SAMcloud
Copyright (C) 2026 AIhdproduction

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, version 3 of the License. The full license text is in
[LICENSE](LICENSE).

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

## Network use (AGPL section 13)

SAMcloud has a web interface. If you modify SAMcloud and let other people use
the modified version over a network, you must offer them the corresponding
source code of your version under the AGPL-3.0.

## Third-party software, models, and data

SAMcloud uses, and in some cases downloads, third-party software, model
weights, and data. These are not part of SAMcloud and are not licensed by the
copyright holder. Each component remains subject exclusively to its own
license, terms of use, and access conditions. Every user is solely responsible
for obtaining the required access and licenses and for complying with their
terms. This applies in particular to commercial use, which may require separate
licenses from the third parties, for example for the Depth-Anything-V2 Large and
Base weights (CC-BY-NC-4.0, non-commercial use only). The SAM 3 weights (SAM
License) and the DINOv3-based FlurSeg weights (DINOv3 License) allow commercial
use but carry use restrictions and the duty to accompany redistributions with a
copy of the license.

The licenses below were read from the installed package metadata
(`venv/Lib/site-packages/*.dist-info`), from the license files shipped in
`third_party/`, `assets/potree/` and `tools/PotreeConverter/`, or taken from the
upstream project. The pinned versions are those of `requirements.txt`. Entries
that were checked against upstream sources name the source URL; check the upstream
terms again before you redistribute or use a component commercially.

### External programs, models, and data

| Component | Used for | License |
|---|---|---|
| COLMAP 4.2.0 (CUDA build in `colmap/`) | camera alignment and dense reconstruction (called as an external program) | BSD-3-Clause ("new BSD", Copyright ETH Zurich and UNC Chapel Hill, and The COLMAP Contributors), per <https://github.com/colmap/colmap/blob/main/LICENSE>; the version is taken from the README, the folder has no license file of its own |
| SAM 3 model weights by Meta (`models/sam3.pt`, access via Hugging Face) | semantic segmentation and instance masks | Meta "SAM License" (<https://github.com/facebookresearch/sam3/blob/main/LICENSE>); Hugging Face lists it as "other" and gates the weights behind a login and acceptance of the conditions (<https://huggingface.co/facebook/sam3>). The license (last updated 2025-11-19; text verified 2026-11-15) grants a non-exclusive, worldwide, royalty-free right to use, reproduce, distribute and create derivative works (§1(a)). It contains no commercial-use restriction and no revenue or user threshold (it does not mention commercial use explicitly). Redistribution must include a copy of the license (§1(b)(i)). Excluded uses are trade-control and sanction targets, ITAR, military and warfare, nuclear, espionage and weapons uses (§1(b)(v)). Outputs (segmentations) are not restricted by the license. Meta may terminate on breach (§6) and modify the terms (§8). SAMcloud does not redistribute the weights |
| FlurSeg weights by AIhdproduction (EoMT with DINOv3-L backbone, trained on SAM 3 pseudo-labels, `models/flurseg_*`) | fast outdoor classification | own training result, fine-tuned from `tue-mps/eomt-dinov3-ade-semantic-large-512` (MIT, <https://huggingface.co/tue-mps/eomt-dinov3-ade-semantic-large-512>) with a DINOv3 backbone. The published weights are under the DINOv3 License (a custom Meta agreement, <https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md>; see <https://huggingface.co/AIhdproduction/Flurseg>). The weights fall under the DINOv3 License (last updated 2025-08-19) because EoMT's DINOv3 checkpoints are deltas to Meta's DINOv3 weights; the MIT label of the EoMT checkpoint covers the EoMT parts, not the DINOv3 weights. The terms match the SAM License: royalty-free use, distribution and derivative works; no commercial-use restriction or threshold; a copy of the license must accompany redistribution; the same excluded uses (§1(b)(v)); outputs unrestricted. SAM 3 was used only as a label teacher; its weights are not part of FlurSeg. The SAM License contains no clause restricting the training of other models on SAM 3 outputs |
| Depth-Anything-V2 code (`third_party/Depth-Anything-V2`, only `metric_depth/depth_anything_v2` is used) | indoor metric depth prior | Apache-2.0 (license file in the folder) |
| Depth-Anything-V2 Metric-Hypersim weights, Large (and Base) | indoor depth prior (default) | CC-BY-NC-4.0, non-commercial use only |
| Depth-Anything-V2 Metric-Hypersim weights, Small | indoor depth prior (small option) | Apache-2.0 |
| AprilTag `tag36h11` family (generated and detected through OpenCV's `cv2.aruco`, `DICT_APRILTAG_36h11`) | printable scale and survey targets, tag detection | tag family and detector design by the AprilRobotics / University of Michigan project, BSD-2-Clause upstream (Copyright The Regents of the University of Michigan, <https://github.com/AprilRobotics/apriltag/blob/master/LICENSE.md>). SAMcloud uses only the implementation in OpenCV (Apache-2.0) and ships no AprilTag source code |
| SegEarth-OV3 method (arXiv:2512.08730) | published method for SAM 3 head fusion, own implementation in `samcloud/engine/classify.py` | method and paper reference only; no code from the authors is included (only names and docstrings refer to it). Add their license if code is ever copied |
| swisstopo CHGeo2004 geoid grid (`ch_swisstopo_chgeo2004_ETRS89_LN02.tif`, fetched with `pyproj sync`) | LN02 heights for LV95 exports | swisstopo terms of use for free geodata: use, including commercial use, is free; a source reference ("©swisstopo" or "Federal Office of Topography swisstopo") is mandatory, <https://www.swisstopo.admin.ch/en/terms-of-use-free-geodata-and-geoservices>. Not verified separately for this single grid file |
| Windows system font Arial (read from `C:/Windows/Fonts` if present) | text on the printable target PDFs | not redistributed; Pillow's default font is used when Arial is missing |

### Viewer components (installed by hand into `assets/potree/` and `tools/`)

These folders are excluded from Git. Their own license files stay with them.

| Component | Used for | License |
|---|---|---|
| Potree 1.8.0 (`assets/potree/build`, version read from `potree.js`) | browser point cloud viewer | BSD-2-Clause (`assets/potree/LICENSE`) |
| PotreeConverter 2.x (`tools/PotreeConverter`, exact version not recorded in the files) | converts LAS to the Potree octree | BSD-2-Clause (`licenses/license_potree_converter.txt`); its `licenses/` folder also lists Brotli (MIT), nlohmann json (MIT), and LASzip (LGPL-2.1) |
| three.js, d3, jQuery UI, proj4js, OpenLayers 3, Cesium, MeshLine, plasio, sql.js, GeoPackage, json5, jsTree, spectrum, stats.js, tween.js, Brotli (in `assets/potree/libs`) | libraries shipped with the Potree distribution | the license file in each `libs/` subfolder: MIT for three.js, MeshLine, plasio, sql.js, GeoPackage, json5, jsTree, spectrum, stats.js, tween.js and Brotli; BSD-3-Clause for d3; BSD-2-Clause for OpenLayers 3; Apache-2.0 for Cesium; jQuery UI and proj4js carry their own MIT-style terms in their files |
| jQuery 3.1.1, i18next 1.8.0, shapefile 0.6.2, copc, laz-perf, zstd-codec, dat.gui, BinaryHeap (in `assets/potree/libs`, no separate license file) | libraries shipped with the Potree distribution | jQuery MIT and i18next MIT (file headers and upstream, <https://github.com/jquery/jquery/blob/main/LICENSE.txt>, <https://github.com/i18next/i18next/blob/master/LICENSE>); shapefile BSD-3-Clause (<https://github.com/mbostock/shapefile/blob/master/LICENSE.txt>); copc.js MIT (<https://github.com/connormanning/copc.js>); laz-perf Apache-2.0 (<https://github.com/hobuinc/laz-perf>); zstd-codec MIT (<https://github.com/yoshihitoh/zstd-codec/blob/master/LICENSE>); dat.gui Apache-2.0 (file header). `other/BinaryHeap.js` carries a header "Creative Commons attribution-noncommercial license; may also be considered MIT" (Marijn Haverbeke), so treat it as MIT only on that author's statement. Bundled third-party code inside the minified copc and zstd-codec files was not checked |

### Python packages (`requirements.txt`)

| Package | Version | Used for | License |
|---|---|---|---|
| Ultralytics | 8.4.138 | SAM3 inference | AGPL-3.0 |
| ultralytics-thop (dependency of Ultralytics) | 2.1.6 | model profiling | AGPL-3.0 |
| PyTorch | 2.14.0+cu126 | GPU inference | BSD-3-Clause; some bundled parts under Apache-2.0, MIT, BSL-1.0 and others (see its package metadata) |
| TorchVision | 0.29.0+cu126 | image models and transforms | BSD-3-Clause (metadata: "BSD") |
| Transformers | 5.22.1 | FlurSeg (EoMT-DINOv3) model code | Apache-2.0 |
| NiceGUI | 3.17.1 | user interface (bundles web front-end libraries, see its package) | MIT |
| pycolmap | 4.2.1 | control point and tag adjustment, model transforms | BSD-3-Clause |
| NumPy | 2.2.6 | arrays | BSD-3-Clause |
| SciPy | 1.18.1 | spatial and sparse algorithms | BSD-3-Clause |
| OpenCV (`opencv-python`) | 4.12.0.88 | image processing, ArUco/AprilTag detection | Apache-2.0 (bundled third-party libraries listed in its `LICENSE-3RD-PARTY.txt`) |
| laspy | 2.7.0 | LAS reading and writing | BSD-2-Clause |
| plyfile | 1.1.2 | PLY reading and writing | GPL-3.0-or-later (package metadata), see "License compatibility" |
| pyproj (PROJ) | 3.7.1 | coordinate transformations | MIT |
| ezdxf | 1.4.5 | DXF writing and checking | MIT |
| Shapely (GEOS) | 2.2.0 | plan geometry | BSD-3-Clause (GEOS: LGPL-2.1) |
| scikit-image | 0.26.0 | orthophoto and plan processing | BSD-3-Clause |
| scikit-learn | 1.9.1 | clustering | BSD-3-Clause |
| ExifRead | 3.5.0 | EXIF metadata of drone images | BSD-3-Clause |
| Pandas | 2.3.3 | control point tables | BSD-3-Clause |
| Matplotlib | 3.10.7 | previews and plots | Matplotlib license (PSF-based) |
| PyYAML | 6.0.3 | configuration files | MIT |
| tqdm | 4.67.1 | progress output | MPL-2.0 AND MIT |
| pytest | 8.4.2 | tests only | MIT |

Packages that pip installs as dependencies and that SAMcloud also uses
directly: Pillow 12.3.0 (printable PDFs and images; MIT-CMU),
huggingface_hub 1.32.0 (model download with the `hf` CLI; Apache-2.0), and
safetensors 0.8.0 (Apache-2.0). Ultralytics also pulls in timm 1.0.29 (Apache-2.0, package metadata) and the
Ultralytics fork of CLIP 1.0 (AGPL-3.0, the LICENSE file in its dist-info;
source <https://github.com/ultralytics/CLIP>).

### License compatibility

SAMcloud is licensed under AGPL-3.0. Ultralytics and ultralytics-thop are also
AGPL-3.0. plyfile is GPL-3.0-or-later; section 13 of both the GPL-3.0 and the
AGPL-3.0 allows such a combination. The MIT, BSD, and Apache-2.0 components are
compatible with the AGPL-3.0. Non-commercial or access-gated components (the
Depth-Anything-V2 Large and Base weights are non-commercial; SAM 3 and
FlurSeg/DINOv3 allow commercial use under their own licenses) are not code
that SAMcloud ships, and the AGPL does not relicense them.

Cyclone 3DR is a trademark of Leica Geosystems. SAMcloud only writes
classification codes and layer names that Cyclone 3DR can read and is not
affiliated with Leica Geosystems.

As an additional disclaimer permitted by AGPL-3.0 section 7, the copyright
holder is not responsible and not liable for third-party components, their
licenses, their availability, or any breach of their terms by a user.

## Results

Point clouds, classifications, coordinates, and heights produced with SAMcloud
are not certified. They must be checked by the user before any use, in
particular for surveying, engineering, legal, or safety-relevant purposes.
