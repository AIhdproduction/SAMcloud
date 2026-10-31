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
copyright holder. Examples:

| Component | Used for |
|---|---|
| COLMAP | camera alignment and dense reconstruction |
| Ultralytics (AGPL-3.0) | SAM3 inference |
| SAM 3 model weights by Meta | semantic segmentation (access via Hugging Face) |
| PyTorch, TorchVision | GPU inference |
| NiceGUI | user interface |
| PROJ / pyproj | coordinate transformations |
| swisstopo CHGeo2004 geoid grid | LN02 heights for LV95 exports |
| NumPy, OpenCV, laspy, plyfile, Pandas, Matplotlib, PyYAML, and others | see `requirements.txt` |

Each component remains subject exclusively to its own license, terms of use,
and access conditions. Every user is solely responsible for obtaining the
required access and licenses and for complying with their terms. This applies
in particular to commercial use, which may require separate licenses from the
third parties, for example for the SAM 3 model weights.

As an additional disclaimer permitted by AGPL-3.0 section 7, the copyright
holder is not responsible and not liable for third-party components, their
licenses, their availability, or any breach of their terms by a user.

## Results

Point clouds, classifications, coordinates, and heights produced with SAMcloud
are not certified. They must be checked by the user before any use, in
particular for surveying, engineering, legal, or safety-relevant purposes.
