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
import torch
import torch.nn.functional as F
import ultralytics
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


def aggregate_instances(inst_probs, conf_scores, query_threshold):
    """Aggregate retained instance masks with equation 1 of SegEarth-OV3."""
    if inst_probs.ndim != 4 or conf_scores.shape != inst_probs.shape[:2]:
        raise ValueError("inst_probs must be (nc, Q, H, W) and conf_scores must be (nc, Q)")
    if inst_probs.shape[1] == 0:
        return torch.zeros(
            (inst_probs.shape[0], inst_probs.shape[2], inst_probs.shape[3]),
            dtype=inst_probs.dtype,
            device=inst_probs.device,
        )
    retained = conf_scores >= query_threshold
    weighted = inst_probs * conf_scores[..., None, None]
    masked = weighted.masked_fill(~retained[..., None, None], float("-inf"))
    aggregated = masked.max(dim=1).values
    return torch.where(retained.any(dim=1)[:, None, None], aggregated, torch.zeros_like(aggregated))


def fuse_heads(sem_probs, inst_agg):
    """Fuse semantic and instance heads with equation 2 of SegEarth-OV3."""
    return torch.maximum(sem_probs, inst_agg)


def apply_presence_gate(fused, presence_scores):
    """Apply the SegEarth-OV3 per-prompt presence score."""
    return fused * presence_scores.reshape(-1, 1, 1)


def assign_classes(final_probs, background_threshold, unknown_id=0):
    """Assign 1-indexed classes to a fused 2D probability tensor."""
    confidence, class_indices = final_probs.max(dim=0)
    return torch.where(
        confidence >= background_threshold,
        class_indices + 1,
        torch.full_like(class_indices, unknown_id),
    )


class _RawHeadResult:
    """Small result object compatible with Ultralytics' predictor iteration."""

    def __init__(self, inst_agg, semantic_seg, presence_scores, raw_shapes, retained_query_counts):
        self.inst_agg = inst_agg
        self.semantic_seg = semantic_seg
        self.presence_scores = presence_scores
        self.raw_shapes = raw_shapes
        self.retained_query_counts = retained_query_counts
        self.speed = {}


class RawHeadSAM3SemanticPredictor(SAM3SemanticPredictor):
    """Return SAM3's raw semantic heads at the original tile resolution."""

    mask_query_chunk_size = 32

    def postprocess(self, preds, img, orig_imgs):
        required_keys = ("pred_logits", "pred_masks", "semantic_seg", "presence_logit_dec")
        for key in required_keys:
            if key not in preds:
                raise RuntimeError(
                    f"SAM3 raw output is missing '{key}' in ultralytics {ultralytics.__version__}"
                )

        pred_logits = preds["pred_logits"]
        pred_masks = preds["pred_masks"]
        semantic_seg = preds["semantic_seg"]
        presence_logit_dec = preds["presence_logit_dec"]
        raw_shapes = {
            "pred_logits": tuple(pred_logits.shape),
            "pred_masks": tuple(pred_masks.shape),
            "semantic_seg": tuple(semantic_seg.shape),
            "presence_logit_dec": tuple(presence_logit_dec.shape),
        }
        if pred_logits.ndim != 3 or pred_logits.shape[2] != 1 or pred_masks.ndim != 4 or semantic_seg.ndim != 4:
            raise RuntimeError(
                "Unexpected SAM3 raw head shapes in ultralytics "
                f"{ultralytics.__version__}: {raw_shapes}"
            )
        nc, query_count, mask_h, mask_w = pred_masks.shape
        if (
            pred_logits.shape[:2] != (nc, query_count)
            or semantic_seg.shape[:2] != (nc, 1)
            or presence_logit_dec.numel() != nc
        ):
            raise RuntimeError(
                "Incompatible SAM3 raw head shapes in ultralytics "
                f"{ultralytics.__version__}: {raw_shapes}"
            )

        if not isinstance(orig_imgs, list):
            raise RuntimeError("SAM3 raw-head predictor expected a list of original tile images")
        tile_h, tile_w = orig_imgs[0].shape[:2]
        query_threshold = self.query_threshold
        conf_scores = pred_logits.sigmoid().squeeze(-1)
        inst_agg = torch.zeros((nc, tile_h, tile_w), dtype=torch.float32, device=pred_masks.device)
        retained_query_counts = []
        for class_index in range(nc):
            retained = conf_scores[class_index] >= query_threshold
            retained_scores = conf_scores[class_index, retained]
            retained_query_counts.append(int(retained_scores.numel()))
            class_agg = inst_agg[class_index]
            for start in range(0, retained_scores.numel(), self.mask_query_chunk_size):
                stop = start + self.mask_query_chunk_size
                mask_logits = pred_masks[class_index, retained][start:stop]
                mask_probs_chunk = F.interpolate(
                    mask_logits[:, None].float(),
                    (tile_h, tile_w),
                    mode="bilinear",
                ).sigmoid()[:, 0]
                chunk_agg = aggregate_instances(
                    mask_probs_chunk[None],
                    retained_scores[start:stop][None],
                    query_threshold,
                )[0]
                class_agg = torch.maximum(class_agg, chunk_agg)
            inst_agg[class_index] = class_agg
        semantic = F.interpolate(semantic_seg.float(), (tile_h, tile_w), mode="bilinear").squeeze(1)
        return [_RawHeadResult(
            inst_agg,
            semantic,
            presence_logit_dec.sigmoid().reshape(nc),
            raw_shapes,
            retained_query_counts,
        )]


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


class HeadFusionLabeler:
    """Score projected points using SegEarth-OV3 SAM3 head fusion."""

    def __init__(self, class_names, unknown_id=0,
                 model_path=Path(__file__).resolve().parents[2] / "models" / "sam3.pt", conf=0.4,
                 tile_size=1008, min_overlap_frac=0.2, query_threshold=0.4):
        self.class_names = class_names
        self.unknown_id = unknown_id
        self.conf = conf
        self.tile_size = tile_size
        self.min_overlap_frac = min_overlap_frac
        self.query_threshold = query_threshold
        self.last_tile_diagnostics = None
        overrides = dict(
            conf=conf,
            imgsz=644,
            task="segment",
            mode="predict",
            model=model_path,
        )
        self.predictor = RawHeadSAM3SemanticPredictor(overrides=overrides)
        self.predictor.query_threshold = query_threshold
        # A raw-head result is intentionally not a rendered Ultralytics Results
        # object, so prediction output must remain disabled even if user settings enable it.
        self.predictor.args.save = False
        self.predictor.args.save_txt = False
        self.predictor.args.save_crop = False
        self.predictor.args.show = False
        self.predictor.args.verbose = False

    def _score_tile(self, tile_img):
        """Run SAM3 once and return final per-class probabilities for one tile."""
        result = self.predictor(source=tile_img, text=self.class_names, verbose=False)[0]
        sem_probs = result.semantic_seg.sigmoid()
        fused = fuse_heads(sem_probs, result.inst_agg)
        final_probs = apply_presence_gate(fused, result.presence_scores)
        self.last_tile_diagnostics = {
            "raw_shapes": result.raw_shapes,
            "presence_scores": result.presence_scores.detach().float().cpu().numpy(),
            "sem_min": float(sem_probs.min().item()),
            "sem_max": float(sem_probs.max().item()),
            "inst_agg_min": float(result.inst_agg.min().item()),
            "inst_agg_max": float(result.inst_agg.max().item()),
            "final_min": float(final_probs.min().item()),
            "final_max": float(final_probs.max().item()),
        }
        return final_probs

    @staticmethod
    def load_image(image_path):
        """Load an image through OpenCV and return its pixels and dimensions."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Could not read image {image_path}")
        height, width = img.shape[:2]
        return img, width, height

    def score_points(self, image_path, u, v, image=None):
        """Return final probabilities for in-bounds integer (u, v) image coordinates."""
        if image is None:
            image, width, height = self.load_image(image_path)
        else:
            height, width = image.shape[:2]
        u = np.asarray(u, dtype=np.int64)
        v = np.asarray(v, dtype=np.int64)
        if u.shape != v.shape:
            raise ValueError("u and v must have the same shape")
        if np.any((u < 0) | (u >= width) | (v < 0) | (v >= height)):
            raise ValueError("score_points requires in-bounds pixel coordinates")

        scores = np.zeros((u.size, len(self.class_names)), dtype=np.float32)
        for x0, y0, x1, y1 in iter_tiles(width, height, self.tile_size, self.min_overlap_frac):
            selected = (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
            if not np.any(selected):
                continue
            final_probs = self._score_tile(image[y0:y1, x0:x1])
            local_u = torch.from_numpy(u[selected] - x0).to(final_probs.device)
            local_v = torch.from_numpy(v[selected] - y0).to(final_probs.device)
            sampled = final_probs[:, local_v, local_u].transpose(0, 1)
            scores[selected] = np.maximum(scores[selected], sampled.detach().float().cpu().numpy())
        return scores
