"""
Wraps SAM3 (via ultralytics) to produce a per-pixel semantic class raster
for one image, given a list of class names as text prompts.

SAM3's encoder has a fixed internal input size (1008x1008 as of the current
ultralytics build). Feeding it a full drone photo means it gets downsampled
to that size before segmentation, which loses thin structures like cables
and signs. To avoid that, the image is split into tiles close to the
model's native input size and each tile is segmented separately, then
merged back by taking the highest-confidence class at every pixel.

Tiling is done so tile count is derived from image size, not a fixed
stride: the tiles are evenly spaced to exactly cover the image edge to
edge, so overlap grows a bit beyond the minimum instead of ever leaving a
small leftover tile at the border. See compute_tile_starts().

NOTE: SAM3 / SAM3SemanticPredictor is a very recent release (Nov 2025).
The exact call signature below matches the ultralytics docs at the time
this was written. If ultralytics changes the API, this is the only file
that needs updating - everything downstream just consumes the raster
this module returns.
"""

import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor


def compute_tile_starts(dim, tile, min_overlap_frac=0.2):
    """
    Returns (starts, tile_size) for one dimension (width or height).

    If the image is smaller than or equal to one tile, returns a single
    tile spanning the whole dimension.

    Otherwise, computes the minimum number of tiles needed so that overlap
    between neighbours is at least min_overlap_frac of the tile size, then
    spaces that many tiles EVENLY from 0 to (dim - tile). Because the tile
    count is rounded up, the actual overlap ends up slightly larger than
    the minimum instead of leaving a smaller, unevenly-sized last tile.
    """
    if dim <= 0 or tile <= 0:
        raise ValueError("dim and tile must be positive")
    if not 0 <= min_overlap_frac < 1:
        raise ValueError("min_overlap_frac must be in [0, 1)")

    if dim <= tile:
        return [0], dim

    stride_max = tile * (1.0 - min_overlap_frac)
    n_tiles = max(2, math.ceil((dim - tile) / stride_max) + 1)

    if n_tiles == 2:
        starts = [0, dim - tile]
    else:
        step = (dim - tile) / (n_tiles - 1)
        starts = [round(i * step) for i in range(n_tiles)]

    # de-duplicate in case rounding collapsed two starts onto each other
    starts = sorted(set(starts))
    starts[0] = 0
    starts[-1] = dim - tile
    return starts, tile


def iter_tiles(width, height, tile_size, min_overlap_frac=0.2):
    """Yields (x0, y0, x1, y1) boxes covering the full image, no leftovers."""
    xs, tile_w = compute_tile_starts(width, tile_size, min_overlap_frac)
    ys, tile_h = compute_tile_starts(height, tile_size, min_overlap_frac)
    for y0 in ys:
        for x0 in xs:
            yield x0, y0, x0 + tile_w, y0 + tile_h


class SemanticLabeler:
    def __init__(self, class_names, unknown_id=0,
                 model_path=Path(__file__).resolve().parents[2] / "models" / "sam3.pt", conf=0.4,
                 tile_size=1008, min_overlap_frac=0.2):
        """
        class_names: list of class names in class-id order, class-id 0 is
        reserved for "unknown/background" and must NOT be in this list.
        """
        self.class_names = class_names
        self.unknown_id = unknown_id
        self.conf = conf
        self.tile_size = tile_size
        self.min_overlap_frac = min_overlap_frac
        overrides = dict(
            conf=conf,
            # SAM3's current Ultralytics implementation uses a 644px inference
            # canvas on this GPU. Tiles remain 1008px and masks are returned
            # at tile resolution.
            imgsz=644,
            task="segment",
            mode="predict",
            model=model_path,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

    def _label_tile(self, tile_img):
        """Runs SAM3 with all class prompts on one tile and returns class/confidence rasters."""
        h, w = tile_img.shape[:2]
        raster = np.zeros((h, w), dtype=np.uint8)
        best_conf = np.zeros((h, w), dtype=np.float32)

        # SAM3SemanticPredictor accepts text, not prompt. A single inference
        # preserves the prompt-to-class mapping in result.boxes.cls and avoids
        # re-encoding the tile once per class.
        result = self.predictor(source=tile_img, text=self.class_names, verbose=False)[0]
        if result.masks is None or len(result.masks) == 0:
            return raster, best_conf

        if result.boxes is None:
            raise RuntimeError("SAM3 returned masks without per-mask class/confidence metadata")

        masks = result.masks.data.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(np.int64) + 1

        for mask, conf, class_idx in zip(masks, confs, class_ids):
            if not 1 <= class_idx <= len(self.class_names):
                continue
            mask_bool = mask > 0.5
            better = mask_bool & (conf > best_conf)
            raster[better] = class_idx
            best_conf[better] = conf

        return raster, best_conf

    def label_image(self, image_path):
        """
        Returns (H, W) uint8 array of class ids for the full-resolution
        image. 0 = unknown/no class matched. Internally tiles the image so
        SAM3 always sees crops close to its native input resolution, then
        merges tiles back by highest confidence per pixel.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Could not read image {image_path}")
        height, width = img.shape[:2]

        raster = np.zeros((height, width), dtype=np.uint8)
        best_conf = np.zeros((height, width), dtype=np.float32)
        for x0, y0, x1, y1 in iter_tiles(width, height, self.tile_size, self.min_overlap_frac):
            tile_img = img[y0:y1, x0:x1]
            tile_raster, tile_conf = self._label_tile(tile_img)

            region_conf = best_conf[y0:y1, x0:x1]
            region_raster = raster[y0:y1, x0:x1]
            better = tile_conf > region_conf
            region_raster[better] = tile_raster[better]
            region_conf[better] = tile_conf[better]
            raster[y0:y1, x0:x1] = region_raster
            best_conf[y0:y1, x0:x1] = region_conf

        # A valid image can contain none of the requested semantic classes.
        # Keep its all-zero raster so these points remain unknown.
        return raster
