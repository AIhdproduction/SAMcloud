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
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import ultralytics
from ultralytics.models.sam import SAM3SemanticPredictor

from samcloud.dxf.cancel import CancelToken, check_cancel


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

    def __init__(self, inst_agg, semantic_seg, presence_scores, raw_shapes, retained_query_counts,
                 instances=None):
        self.inst_agg = inst_agg
        self.semantic_seg = semantic_seg
        self.presence_scores = presence_scores
        self.raw_shapes = raw_shapes
        self.retained_query_counts = retained_query_counts
        # Optional decoder-query masks used exclusively by the 3D instance
        # side channel.  Keeping them separate is important: ``inst_agg`` and
        # the semantic path deliberately remain byte-for-byte unchanged.
        self.instances = instances
        self.speed = {}


class _PromptBatchingMixin:
    """Run long text prompt lists in small grounding passes after one backbone pass."""

    prompt_batch = 8

    def _inference_features(self, features, bboxes=None, labels=None, text: list[str] | None = None):
        if bboxes is not None or text is None or len(text) <= self.prompt_batch:
            return super()._inference_features(features, bboxes, labels, text=text)
        if self.model.names != text:
            self.model.set_classes(text=text)
        keys = ("pred_logits", "pred_masks", "semantic_seg", "presence_logit_dec", "pred_boxes")
        chunks = []
        for start in range(0, len(text), self.prompt_batch):
            stop = min(start + self.prompt_batch, len(text))
            outputs = self.model.forward_grounding(
                backbone_out=features,
                text_ids=torch.arange(start, stop, device=self.device, dtype=torch.long),
                geometric_prompt=self._get_dummy_prompt(stop - start),
            )
            chunks.append({key: outputs[key] for key in keys if key in outputs})
        merged = {}
        for key in chunks[0]:
            parts = [chunk[key] for chunk in chunks]
            # Standard SAM3 keeps presence logits as (classes, 1), while the
            # raw-head path accepts either shape via ``numel``. Preserve that
            # rank so the stock postprocessor can broadcast scores correctly.
            merged[key] = torch.cat(parts, dim=0)
        return merged


class BatchedSAM3SemanticPredictor(_PromptBatchingMixin, SAM3SemanticPredictor):
    """Standard SAM3 result predictor with WDDM-safe prompt batching."""


class RawHeadSAM3SemanticPredictor(_PromptBatchingMixin, SAM3SemanticPredictor):
    """Return SAM3's raw semantic heads at the original tile resolution."""

    mask_query_chunk_size = 32
    # False: return the heads at mask resolution; the caller fuses prompts into
    # classes there and upsamples only the class maps (much less GPU memory/time).
    full_resolution = True
    keep_instances = False
    instance_query_threshold = 0.5
    instance_prompt_mask = None
    # Text prompts per grounding pass. The image backbone runs once per tile;
    # many prompts (Nomivox presets: ~160) in one pass exhaust a 24 GB GPU.
    # Measured 2026-11-09 on 3200 px drone images (12 tiles, 169 outdoor prompts,
    # RTX 4090): 32 reserves 21.4 GB and spills into shared system memory under
    # WDDM (250 s/image); 16 -> 18.9 GB, 27 s; 8 -> 15.7 GB, 29 s. 8 leaves room
    # for other GPU applications at ~7 % cost.
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
        if not self.full_resolution:
            tile_h, tile_w = mask_h, mask_w
        query_threshold = self.query_threshold
        conf_scores = pred_logits.sigmoid().squeeze(-1)
        retained = conf_scores >= query_threshold
        retained_query_counts = [int(count) for count in retained.sum(dim=1).tolist()]
        inst_agg = torch.zeros((nc, tile_h, tile_w), dtype=torch.float32, device=pred_masks.device)
        # Equation 1 of SegEarth-OV3 (max over retained, score-weighted query masks)
        # is evaluated at mask resolution and upsampled once per prompt. Upsampling
        # every retained query first costs minutes per tile with ~160 prompts.
        for start in range(0, nc, self.mask_query_chunk_size):
            stop = min(start + self.mask_query_chunk_size, nc)
            keep = retained[start:stop]
            weighted = pred_masks[start:stop].float().sigmoid() * conf_scores[start:stop, :, None, None]
            weighted = weighted.masked_fill(~keep[..., None, None], 0.0)
            low_res = weighted.max(dim=1).values
            if (tile_h, tile_w) == (mask_h, mask_w):
                inst_agg[start:stop] = low_res
            else:
                inst_agg[start:stop] = F.interpolate(low_res[:, None], (tile_h, tile_w), mode="bilinear")[:, 0]
            del weighted, low_res
        semantic = F.interpolate(semantic_seg.float(), (tile_h, tile_w), mode="bilinear").squeeze(1)
        instances = None
        if bool(getattr(self, "keep_instances", False)):
            prompt_mask = getattr(self, "instance_prompt_mask", None)
            if prompt_mask is None:
                prompt_mask = torch.zeros(nc, dtype=torch.bool, device=conf_scores.device)
            else:
                prompt_mask = torch.as_tensor(prompt_mask, dtype=torch.bool, device=conf_scores.device)
                if prompt_mask.numel() != nc:
                    raise ValueError("instance_prompt_mask needs one value per flat prompt")
            instance_keep = (conf_scores >= float(getattr(self, "instance_query_threshold", query_threshold)))
            instance_keep &= prompt_mask[:, None]
            prompt_idx, query_idx = torch.nonzero(instance_keep, as_tuple=True)
            # Masks are intentionally retained at native mask resolution.  A
            # tile samples them with the same coordinate scale as the semantic
            # maps, avoiding a large full-resolution query tensor on the GPU.
            instances = {
                "prompt_idx": prompt_idx.detach().to(dtype=torch.int32, device="cpu"),
                "conf": conf_scores[prompt_idx, query_idx].detach().float().cpu(),
                "masks": pred_masks[prompt_idx, query_idx].detach().sigmoid().to(
                    dtype=torch.float16, device="cpu"
                ),
            }
        return [_RawHeadResult(
            inst_agg,
            semantic,
            presence_logit_dec.sigmoid().reshape(nc),
            raw_shapes,
            retained_query_counts,
            instances,
        )]


def effective_tile_size(width, height, tile_size):
    """Use one tile for images up to 1.5x a tile (1024 or 1440 px cubemap faces).

    SAM3 runs at a 644 px canvas anyway; splitting a 1024 px face into four
    1008 px tiles costs four passes for almost identical content.
    """
    longest = max(int(width), int(height))
    return longest if longest <= 1.5 * int(tile_size) else int(tile_size)


def resolve_prompts(class_names, prompts=None):
    """Return the SAM3 text prompts per class, in class-id order.

    Class names stay the identity of a class (summaries, LAS export); prompts
    are only what SAM3 reads. An entry is either one prompt string or a list
    of prompts (Nomivox style); each prompt is sent to SAM3 on its own and the
    class keeps the best of them. Never join synonyms into one long
    comma-separated prompt: that pushes SAM3's presence score towards zero.
    """
    if prompts is None:
        return list(class_names)
    prompts = list(prompts)
    if len(prompts) != len(class_names):
        raise ValueError(
            f"{len(prompts)} prompts for {len(class_names)} classes; both lists must have the same order and length"
        )
    for entry in prompts:
        if not isinstance(entry, str) and (not entry or not all(isinstance(item, str) for item in entry)):
            raise ValueError("every class needs a prompt string or a non-empty list of prompt strings")
    return prompts


def flatten_prompts(prompts):
    """Flatten per-class prompts; returns (flat prompt list, owning class index per prompt)."""
    flat, owner = [], []
    for class_index, entry in enumerate(prompts):
        items = [entry] if isinstance(entry, str) else list(entry)
        flat.extend(items)
        owner.extend([class_index] * len(items))
    return flat, np.asarray(owner, dtype=np.int64)


def reduce_prompt_scores(scores, owner, num_classes):
    """Max-pool per-prompt score maps (P, H, W) into per-class maps (C, H, W)."""
    if scores.shape[0] == num_classes and np.array_equal(owner, np.arange(num_classes)):
        return scores
    index = torch.as_tensor(owner, device=scores.device)
    return torch.stack([scores[index == class_index].amax(dim=0) for class_index in range(num_classes)])


class SemanticLabeler:
    def __init__(self, class_names, unknown_id=0,
                 model_path=Path(__file__).resolve().parents[2] / "models" / "sam3.pt", conf=0.4,
                 tile_size=1008, min_overlap_frac=0.2, prompts=None):
        """
        class_names: list of class names in class-id order, class-id 0 is
        reserved for "unknown/background" and must NOT be in this list.
        prompts: optional SAM3 text prompts, same order and length as class_names.
        """
        self.class_names = class_names
        self.prompts = resolve_prompts(class_names, prompts)
        self._flat_prompts, self._prompt_owner = flatten_prompts(self.prompts)
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
        # Resolve the base predictor at construction time so legacy callers
        # can monkeypatch the module-level SAM3SemanticPredictor symbol.
        predictor_type = type("BatchedSAM3SemanticPredictor", (_PromptBatchingMixin, SAM3SemanticPredictor), {})
        self.predictor = predictor_type(overrides=overrides)
        self.predictor.args.save = False
        self.predictor.args.save_txt = False
        self.predictor.args.save_crop = False
        self.predictor.args.show = False
        self.predictor.args.verbose = False

    def _label_tile(self, tile_img):
        """Runs SAM3 with all class prompts on one tile and returns class/confidence rasters."""
        h, w = tile_img.shape[:2]
        raster = np.zeros((h, w), dtype=np.uint8)
        best_conf = np.zeros((h, w), dtype=np.float32)

        # SAM3SemanticPredictor accepts text, not prompt. A single inference
        # preserves the prompt-to-class mapping in result.boxes.cls and avoids
        # re-encoding the tile once per class.
        result = self.predictor(source=tile_img, text=self._flat_prompts, verbose=False)[0]
        if result.masks is None or len(result.masks) == 0:
            return raster, best_conf

        if result.boxes is None:
            raise RuntimeError("SAM3 returned masks without per-mask class/confidence metadata")

        masks = result.masks.data.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        prompt_ids = result.boxes.cls.cpu().numpy().astype(np.int64)
        valid = (prompt_ids >= 0) & (prompt_ids < len(self._prompt_owner))
        class_ids = np.zeros_like(prompt_ids)
        class_ids[valid] = self._prompt_owner[prompt_ids[valid]] + 1

        for mask, conf, class_idx in zip(masks, confs, class_ids):
            if not 1 <= class_idx <= len(self.class_names):
                continue
            mask_bool = mask > 0.5
            better = mask_bool & (conf > best_conf)
            raster[better] = class_idx
            best_conf[better] = conf

        return raster, best_conf

    def _label_image_bgr(self, image: np.ndarray, *, progress: Callable[[float], None] | None = None,
                         cancel: CancelToken | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Label one BGR image with tiled SAM3 inference and return labels plus winning mask scores."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (height, width, 3)")
        img = np.ascontiguousarray(image, dtype=np.uint8)
        height, width = img.shape[:2]
        raster = np.zeros((height, width), dtype=np.uint8)
        best_conf = np.zeros((height, width), dtype=np.float32)
        tiles = list(iter_tiles(width, height, effective_tile_size(width, height, self.tile_size), self.min_overlap_frac))
        for index, (x0, y0, x1, y1) in enumerate(tiles, start=1):
            check_cancel(cancel)
            tile_raster, tile_conf = self._label_tile(img[y0:y1, x0:x1])
            region_conf = best_conf[y0:y1, x0:x1]
            region_raster = raster[y0:y1, x0:x1]
            better = tile_conf > region_conf
            region_raster[better] = tile_raster[better]
            region_conf[better] = tile_conf[better]
            raster[y0:y1, x0:x1] = region_raster
            best_conf[y0:y1, x0:x1] = region_conf
            if progress is not None:
                progress(index / len(tiles))
        check_cancel(cancel)
        return raster, best_conf.astype(np.float16)

    def label_image_scores(self, image: np.ndarray | Path | str, *, progress: Callable[[float], None] | None = None,
                           cancel: CancelToken | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return uint8 labels and float16 confidence for an image.

        Array inputs are RGB, matching orthophoto pixels, and are converted to
        BGR because the Ultralytics predictor receives OpenCV-style images.
        Path inputs are read by OpenCV and therefore already BGR.
        """
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("RGB image must have shape (height, width, 3)")
            bgr = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise RuntimeError(f"Could not read image {image}")
        return self._label_image_bgr(bgr, progress=progress, cancel=cancel)

    def label_image(self, image_path):
        """
        Returns (H, W) uint8 array of class ids for the full-resolution
        image. 0 = unknown/no class matched. Internally tiles the image so
        SAM3 always sees crops close to its native input resolution, then
        merges tiles back by highest confidence per pixel.
        """
        labels, _confidence = self.label_image_scores(image_path)
        return labels


class HeadFusionLabeler:
    """Score projected points using SegEarth-OV3 SAM3 head fusion."""

    def __init__(self, class_names, unknown_id=0,
                 model_path=Path(__file__).resolve().parents[2] / "models" / "sam3.pt", conf=0.4,
                 tile_size=1008, min_overlap_frac=0.2, query_threshold=0.4, prompts=None,
                 fuse_at_mask_resolution=True, fine_queries=None, instance_classes=None, instance_params=None):
        """``fine_queries`` optionally lists, per fine term, the flat prompt indices that
        answer it; their gated score maps are kept next to the class probabilities (see
        :meth:`score_points`, ``fine_scores_out``). The extra SAM work is zero: the term
        maps exist before the prompts are pooled into classes."""
        self.class_names = class_names
        self.prompts = resolve_prompts(class_names, prompts)
        self._flat_prompts, self._prompt_owner = flatten_prompts(self.prompts)
        # Fuse heads and pool prompts into classes at SAM3's mask resolution and
        # upsample only the class maps. The full-resolution path upsamples every
        # prompt first (~160 x tile pixels, several GB): on an RTX 4090 that took
        # 6.5 s instead of 2.0 s per 1440 px image and fragmented GPU memory so
        # later images got slower still; classes agree to 99.85 %.
        self.fuse_at_mask_resolution = bool(fuse_at_mask_resolution)
        self.unknown_id = unknown_id
        self.conf = conf
        self.tile_size = tile_size
        self.min_overlap_frac = min_overlap_frac
        self.query_threshold = query_threshold
        self.last_tile_diagnostics = None
        self.fine_queries = [list(map(int, group)) for group in fine_queries] if fine_queries else None
        self.last_fine_probs = None
        self.instance_classes = {int(value) for value in (instance_classes or [])}
        self.instance_params = dict(instance_params or {})
        # These prompts have a per-point competition of their own: an added mask never takes a point from the others.
        self.secondary_prompt_ids = {int(value) for value in self.instance_params.get("secondary_prompt_ids") or ()}
        # Masks of these prompts are also returned as outlines (screen_masks.py); the association never sees them.
        self.record_prompt_ids = {int(value) for value in self.instance_params.get("record_prompt_ids") or ()}
        self.last_instances = None
        overrides = dict(
            conf=conf,
            imgsz=644,
            task="segment",
            mode="predict",
            model=model_path,
        )
        self.predictor = RawHeadSAM3SemanticPredictor(overrides=overrides)
        self.predictor.query_threshold = query_threshold
        self.predictor.full_resolution = not self.fuse_at_mask_resolution
        self.predictor.keep_instances = bool(self.instance_classes and self.instance_params.get("enabled", True))
        self.predictor.instance_query_threshold = float(
            self.instance_params.get("query_threshold", query_threshold)
        )
        self.predictor.instance_prompt_mask = np.asarray(
            [(index + 1) in self.instance_classes for index in self._prompt_owner], dtype=bool
        )
        # A raw-head result is intentionally not a rendered Ultralytics Results
        # object, so prediction output must remain disabled even if user settings enable it.
        self.predictor.args.save = False
        self.predictor.args.save_txt = False
        self.predictor.args.save_crop = False
        self.predictor.args.show = False
        self.predictor.args.verbose = False

    def _score_tile(self, tile_img):
        """Run SAM3 once and return final per-class probabilities for one tile."""
        result = self.predictor(source=tile_img, text=self._flat_prompts, verbose=False)[0]
        sem_probs = result.semantic_seg.sigmoid()
        fused = fuse_heads(sem_probs, result.inst_agg)
        gated = apply_presence_gate(fused, result.presence_scores)
        final_probs = reduce_prompt_scores(gated, self._prompt_owner, len(self.class_names))
        tile_h, tile_w = tile_img.shape[:2]
        if tuple(final_probs.shape[1:]) != (tile_h, tile_w):
            final_probs = F.interpolate(final_probs[None].float(), (tile_h, tile_w), mode="bilinear")[0]
        self.last_fine_probs = None
        self.last_instances = getattr(result, "instances", None)
        if self.fine_queries:
            # Fine terms: max over the queries of each term, taken from the same gated maps the class
            # pooling used (no extra SAM call), then upsampled like the class maps.
            fine_probs = torch.stack([gated[group].amax(dim=0) for group in self.fine_queries])
            if tuple(fine_probs.shape[1:]) != (tile_h, tile_w):
                fine_probs = F.interpolate(fine_probs[None].float(), (tile_h, tile_w), mode="bilinear")[0]
            self.last_fine_probs = fine_probs
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

    def score_points(self, image_path, u, v, image=None, label_threshold=None,
                     label_exclude_ids=None, return_raw_labels=False, fine_scores_out=None,
                     instances_out=None):
        """Return final probabilities for in-bounds integer (u, v) image coordinates.

        With ``label_threshold`` a per-pixel label raster is returned as well.
        ``label_exclude_ids`` removes internal-only classes from that raster
        before selecting the winner. When ``return_raw_labels`` is true, the
        unfiltered raster is returned as a third value for side-channel masks.

        ``fine_scores_out`` (float32, shape ``(len(u), len(fine_queries))``, zero-filled by the
        caller) receives the per-fine-term scores of the same points, sampled exactly like the
        class probabilities; the returned values are unchanged.
        """
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
        point_local_instance = np.full(u.size, -1, dtype=np.int32)
        instance_best = np.zeros(u.size, dtype=np.float32)
        secondary_ids = getattr(self, "secondary_prompt_ids", None) or set()
        point_secondary_instance = np.full(u.size, -1, dtype=np.int32)
        secondary_best = np.zeros(u.size, dtype=np.float32)
        local_instances = []
        record_ids = getattr(self, "record_prompt_ids", None) or set()
        recorded_masks = []
        record_error = None
        if record_ids:
            try:
                if __package__:
                    from .screen_masks import mask_outline
                else:
                    from screen_masks import mask_outline
            except Exception as error:  # noqa: BLE001 - optional product
                record_error = error
                record_ids = self.record_prompt_ids = set()
        labels = best = raw_labels = raw_best = None
        if label_threshold is not None:
            labels = np.zeros((height, width), dtype=np.uint8)
            best = np.zeros((height, width), dtype=np.float32)
            if return_raw_labels:
                raw_labels = np.zeros((height, width), dtype=np.uint8)
                raw_best = np.zeros((height, width), dtype=np.float32)
        excluded = np.asarray(sorted(label_exclude_ids or ()), dtype=np.intp) - 1
        for x0, y0, x1, y1 in iter_tiles(width, height, effective_tile_size(width, height, self.tile_size), self.min_overlap_frac):
            selected = (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
            if not np.any(selected):
                continue
            final_probs = self._score_tile(image[y0:y1, x0:x1])
            local_u = torch.from_numpy(u[selected] - x0).to(final_probs.device)
            local_v = torch.from_numpy(v[selected] - y0).to(final_probs.device)
            sampled = final_probs[:, local_v, local_u].transpose(0, 1)
            scores[selected] = np.maximum(scores[selected], sampled.detach().float().cpu().numpy())
            fine_tile = getattr(self, "last_fine_probs", None) if fine_scores_out is not None else None
            if fine_tile is not None:
                fine_sampled = fine_tile[:, local_v, local_u].transpose(0, 1).detach().float().cpu().numpy()
                fine_scores_out[selected] = np.maximum(fine_scores_out[selected], fine_sampled)
            if instances_out is not None and self.last_instances is not None:
                raw_instances = self.last_instances
                masks = np.asarray(raw_instances.get("masks", ()), dtype=np.float16)
                prompt_indices = np.asarray(raw_instances.get("prompt_idx", ()), dtype=np.int32)
                confidences = np.asarray(raw_instances.get("conf", ()), dtype=np.float32)
                if masks.ndim == 3 and len(masks) == len(prompt_indices) == len(confidences):
                    threshold = float(self.instance_params.get("mask_threshold", 0.5))
                    min_pixels = int(self.instance_params.get("min_mask_pixels", 12))
                    local_positions = np.flatnonzero(selected)
                    lu = u[selected] - x0
                    lv = v[selected] - y0
                    tile_h, tile_w = y1 - y0, x1 - x0
                    for mask, prompt_idx, confidence in zip(masks, prompt_indices, confidences):
                        pixel_count = int(np.count_nonzero(mask >= threshold))
                        if pixel_count < min_pixels:
                            continue
                        if int(prompt_idx) in record_ids:  # also without a single visible point: a dark screen
                            try:
                                outline = mask_outline(mask, (x0, y0, x1, y1), threshold)
                                if outline is not None:
                                    recorded_masks.append({
                                        "prompt_idx": int(prompt_idx), "conf": float(confidence), "outline": outline[0],
                                        "cut": outline[1],
                                    })
                            except Exception as error:  # noqa: BLE001 - optional product: stop recording, keep classifying
                                record_error = error
                                record_ids = self.record_prompt_ids = set()
                        # Nearest sampling at the mask grid via pixel centres, matching the
                        # coordinate scale used when mask-resolution semantic
                        # maps are expanded to the tile.
                        mx = np.clip(((lu + 0.5) * mask.shape[1] / tile_w).astype(np.int64), 0, mask.shape[1] - 1)
                        my = np.clip(((lv + 0.5) * mask.shape[0] / tile_h).astype(np.int64), 0, mask.shape[0] - 1)
                        probabilities = mask[my, mx].astype(np.float32)
                        belongs = probabilities >= threshold
                        if not np.any(belongs):
                            continue
                        local_id = len(local_instances)
                        class_id = int(self._prompt_owner[int(prompt_idx)]) + 1
                        local_instances.append({
                            "class_id": class_id,
                            "prompt_idx": int(prompt_idx),
                            "conf": float(confidence),
                            "pixel_count": pixel_count,
                        })
                        positions = local_positions[belongs]
                        strength = float(confidence) * probabilities[belongs]
                        winner, winning = (
                            (point_secondary_instance, secondary_best) if int(prompt_idx) in secondary_ids
                            else (point_local_instance, instance_best)
                        )
                        better = strength > winning[positions]
                        if np.any(better):
                            chosen = positions[better]
                            winner[chosen] = local_id
                            winning[chosen] = strength[better]
            if labels is not None:
                raw_tile_best, raw_tile_class = final_probs.max(dim=0)
                raw_tile_best = raw_tile_best.detach().float().cpu().numpy()
                raw_tile_class = raw_tile_class.detach().cpu().numpy().astype(np.uint8) + 1
                if excluded.size:
                    allowed = final_probs.clone()
                    allowed[excluded] = 0
                    tile_best, tile_class = allowed.max(dim=0)
                else:
                    tile_best, tile_class = final_probs.max(dim=0)
                tile_best = tile_best.detach().float().cpu().numpy()
                tile_class = tile_class.detach().cpu().numpy().astype(np.uint8) + 1
                tile_class[tile_best < float(label_threshold)] = 0
                region_best = best[y0:y1, x0:x1]
                better = tile_best > region_best
                labels[y0:y1, x0:x1][better] = tile_class[better]
                region_best[better] = tile_best[better]
                if raw_labels is not None:
                    raw_tile_class[raw_tile_best < float(label_threshold)] = 0
                    raw_region_best = raw_best[y0:y1, x0:x1]
                    raw_better = raw_tile_best > raw_region_best
                    raw_labels[y0:y1, x0:x1][raw_better] = raw_tile_class[raw_better]
                    raw_region_best[raw_better] = raw_tile_best[raw_better]
        if instances_out is not None:
            instances_out.clear()
            instances_out["point_local_instance"] = point_local_instance
            if secondary_ids:
                instances_out["point_secondary_instance"] = point_secondary_instance
            instances_out["instances"] = local_instances
            if record_error is not None:
                instances_out["record_error"] = record_error
            elif record_ids:
                instances_out["recorded_masks"] = recorded_masks
        if labels is not None:
            if return_raw_labels:
                return scores, labels, raw_labels
            return scores, labels
        return scores
