"""Tests for SAM3 raw-head predictor instance retention and score_points instance sampling."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from samcloud.engine.sam_labeler import HeadFusionLabeler, RawHeadSAM3SemanticPredictor, _RawHeadResult


def test_postprocess_semantic_outputs_bit_identical_with_keep_instances() -> None:
    """Semantic outputs (inst_agg, semantic, presence) must be bit-identical whether keep_instances is True or False."""
    nc, query_count, mask_h, mask_w = 4, 6, 16, 16
    tile_h, tile_w = 32, 32
    rng = np.random.default_rng(20261113)

    pred_logits = torch.from_numpy(rng.standard_normal((nc, query_count, 1)).astype(np.float32))
    pred_masks = torch.from_numpy(rng.standard_normal((nc, query_count, mask_h, mask_w)).astype(np.float32))
    semantic_seg = torch.from_numpy(rng.standard_normal((nc, 1, mask_h, mask_w)).astype(np.float32))
    presence_logit_dec = torch.from_numpy(rng.standard_normal((nc,)).astype(np.float32))

    preds = {
        "pred_logits": pred_logits,
        "pred_masks": pred_masks,
        "semantic_seg": semantic_seg,
        "presence_logit_dec": presence_logit_dec,
    }
    orig_imgs = [np.zeros((tile_h, tile_w, 3), dtype=np.uint8)]

    for full_res in (True, False):
        predictor = RawHeadSAM3SemanticPredictor.__new__(RawHeadSAM3SemanticPredictor)
        predictor.full_resolution = full_res
        predictor.query_threshold = 0.4
        predictor.mask_query_chunk_size = 32
        predictor.instance_query_threshold = 0.5
        predictor.instance_prompt_mask = torch.tensor([True, False, True, False])

        predictor.keep_instances = False
        res_off = predictor.postprocess(preds, None, orig_imgs)[0]

        predictor.keep_instances = True
        res_on = predictor.postprocess(preds, None, orig_imgs)[0]

        assert torch.equal(res_off.inst_agg, res_on.inst_agg)
        assert torch.equal(res_off.semantic_seg, res_on.semantic_seg)
        assert torch.equal(res_off.presence_scores, res_on.presence_scores)
        assert res_off.instances is None
        assert res_on.instances is not None


def test_postprocess_filters_by_prompt_mask_and_query_threshold() -> None:
    """Only prompts allowed by instance_prompt_mask and queries >= instance_query_threshold are kept."""
    nc, query_count, mask_h, mask_w = 3, 4, 8, 8
    tile_h, tile_w = 8, 8

    # Target confidences:
    # prompt 0 (allowed): [0.72, 0.45, 0.81, 0.25] -> kept: query 0 (0.72) and query 2 (0.81)
    # prompt 1 (blocked): [0.95, 0.65, 0.10, 0.30] -> none kept even though >= 0.6
    # prompt 2 (allowed): [0.35, 0.68, 0.15, 0.52] -> kept: query 1 (0.68)
    target_confs = torch.tensor([
        [0.72, 0.45, 0.81, 0.25],
        [0.95, 0.65, 0.10, 0.30],
        [0.35, 0.68, 0.15, 0.52],
    ], dtype=torch.float32)

    pred_logits = torch.logit(target_confs)[:, :, None]
    pred_masks = torch.zeros((nc, query_count, mask_h, mask_w), dtype=torch.float32)
    for c in range(nc):
        for q in range(query_count):
            pred_masks[c, q] = float(c * 10 + q)
    semantic_seg = torch.zeros((nc, 1, mask_h, mask_w), dtype=torch.float32)
    presence_logit_dec = torch.zeros((nc,), dtype=torch.float32)

    preds = {
        "pred_logits": pred_logits,
        "pred_masks": pred_masks,
        "semantic_seg": semantic_seg,
        "presence_logit_dec": presence_logit_dec,
    }
    orig_imgs = [np.zeros((tile_h, tile_w, 3), dtype=np.uint8)]

    predictor = RawHeadSAM3SemanticPredictor.__new__(RawHeadSAM3SemanticPredictor)
    predictor.full_resolution = False
    predictor.query_threshold = 0.4
    predictor.mask_query_chunk_size = 32
    predictor.keep_instances = True
    predictor.instance_query_threshold = 0.60
    predictor.instance_prompt_mask = torch.tensor([True, False, True], dtype=torch.bool)

    res = predictor.postprocess(preds, None, orig_imgs)[0]
    instances = res.instances
    assert instances is not None

    prompt_idx = instances["prompt_idx"]
    conf = instances["conf"]
    masks = instances["masks"]

    assert prompt_idx.tolist() == [0, 0, 2]
    torch.testing.assert_close(conf, torch.tensor([0.72, 0.81, 0.68], dtype=torch.float32), atol=1e-5, rtol=1e-5)
    assert masks.shape == (3, mask_h, mask_w)
    assert masks.dtype == torch.float16
    expected_mask_0 = torch.sigmoid(torch.tensor(0.0)).to(torch.float16)
    expected_mask_1 = torch.sigmoid(torch.tensor(2.0)).to(torch.float16)
    expected_mask_2 = torch.sigmoid(torch.tensor(21.0)).to(torch.float16)
    assert torch.allclose(masks[0], expected_mask_0)
    assert torch.allclose(masks[1], expected_mask_1)
    assert torch.allclose(masks[2], expected_mask_2)


def test_score_points_two_instance_masks_and_competition() -> None:
    """Points inside masks get the correct local ids, -1 otherwise; overlapping points choose larger conf*prob."""
    labeler = HeadFusionLabeler.__new__(HeadFusionLabeler)
    labeler.class_names = ["clutter", "furniture"]
    labeler._flat_prompts = ["bag", "chair", "armchair"]
    labeler._prompt_owner = np.array([0, 1, 1], dtype=np.int64)
    labeler.tile_size = 100
    labeler.min_overlap_frac = 0.2
    labeler.fine_queries = None
    labeler.last_fine_probs = None
    labeler.last_tile_diagnostics = None
    labeler.instance_classes = {2}
    labeler.instance_params = {
        "mask_threshold": 0.5,
        "min_mask_pixels": 12,
    }
    labeler.last_instances = None

    # Mask resolution 20 x 20, tile size 100 x 100 -> each mask cell is 5 x 5 tile pixels.
    # Mask 0: chair (prompt 1), conf = 0.8. Region: y in [2, 8], x in [2, 8] (36 px >= 12). Prob = 0.80 -> strength = 0.64.
    # Mask 1: armchair (prompt 2), conf = 0.9. Region: y in [6, 12], x in [6, 12] (36 px >= 12). Prob = 0.90 -> strength = 0.81.
    mask0 = np.zeros((20, 20), dtype=np.float16)
    mask0[2:8, 2:8] = 0.80
    mask1 = np.zeros((20, 20), dtype=np.float16)
    mask1[6:12, 6:12] = 0.90

    fake_instances = {
        "prompt_idx": np.array([1, 2], dtype=np.int32),
        "conf": np.array([0.8, 0.9], dtype=np.float32),
        "masks": np.stack([mask0, mask1]),
    }

    def _stub_score_tile(tile_img):
        labeler.last_instances = fake_instances
        return torch.zeros((2, tile_img.shape[0], tile_img.shape[1]), dtype=torch.float32)

    labeler._score_tile = _stub_score_tile

    # Sample points:
    # Point 0: u=17, v=17 -> mask x=3, y=3 (inside Mask 0 only)
    # Point 1: u=47, v=47 -> mask x=9, y=9 (inside Mask 1 only)
    # Point 2: u=37, v=37 -> mask x=7, y=7 (in both: Mask 0 gives 0.64, Mask 1 gives 0.81 -> Mask 1 wins)
    # Point 3: u=87, v=87 -> mask x=17, y=17 (outside both masks)
    u = np.array([17, 47, 37, 87], dtype=np.int64)
    v = np.array([17, 47, 37, 87], dtype=np.int64)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    instances_out = {}

    labeler.score_points("dummy.png", u, v, image=image, instances_out=instances_out)

    assert "point_local_instance" in instances_out
    assert "instances" in instances_out
    local_ids = instances_out["point_local_instance"]
    assert local_ids[0] == 0
    assert local_ids[1] == 1
    assert local_ids[2] == 1
    assert local_ids[3] == -1

    meta = instances_out["instances"]
    assert len(meta) == 2
    assert meta[0]["class_id"] == 2
    assert meta[0]["prompt_idx"] == 1
    assert abs(meta[0]["conf"] - 0.8) < 1e-4
    assert meta[0]["pixel_count"] == 36
    assert meta[1]["class_id"] == 2
    assert meta[1]["prompt_idx"] == 2
    assert abs(meta[1]["conf"] - 0.9) < 1e-4
    assert meta[1]["pixel_count"] == 36


def test_score_points_handles_two_tiles_and_offset() -> None:
    """When an image requires two tiles, tile offsets must be correctly mapped to each tile's masks."""
    labeler = HeadFusionLabeler.__new__(HeadFusionLabeler)
    labeler.class_names = ["clutter", "furniture"]
    labeler._flat_prompts = ["bag", "chair"]
    labeler._prompt_owner = np.array([0, 1], dtype=np.int64)
    labeler.tile_size = 100
    labeler.min_overlap_frac = 0.2
    labeler.fine_queries = None
    labeler.last_fine_probs = None
    labeler.last_tile_diagnostics = None
    labeler.instance_classes = {2}
    labeler.instance_params = {
        "mask_threshold": 0.5,
        "min_mask_pixels": 12,
    }
    labeler.last_instances = None

    # Image 160 x 100 with tile size 100 produces 2 tiles: (0, 0, 100, 100) and (60, 0, 160, 100).
    # Tile 0 has instance 0 at local tile x in [10, 30], y in [40, 60] -> mask cell x in [2, 6], y in [8, 12].
    # Tile 1 has instance 1 at local tile x in [60, 80], y in [40, 60] -> mask cell x in [12, 16], y in [8, 12].
    mask_tile0 = np.zeros((20, 20), dtype=np.float16)
    mask_tile0[8:12, 2:6] = 0.85
    inst_tile0 = {
        "prompt_idx": np.array([1], dtype=np.int32),
        "conf": np.array([0.75], dtype=np.float32),
        "masks": mask_tile0[None, :, :],
    }

    mask_tile1 = np.zeros((20, 20), dtype=np.float16)
    mask_tile1[8:12, 12:16] = 0.90
    inst_tile1 = {
        "prompt_idx": np.array([1], dtype=np.int32),
        "conf": np.array([0.85], dtype=np.float32),
        "masks": mask_tile1[None, :, :],
    }

    tile_call_count = [0]

    def _stub_score_tile(tile_img):
        call_idx = tile_call_count[0]
        tile_call_count[0] += 1
        labeler.last_instances = inst_tile0 if call_idx == 0 else inst_tile1
        return torch.zeros((2, tile_img.shape[0], tile_img.shape[1]), dtype=torch.float32)

    labeler._score_tile = _stub_score_tile

    # Points:
    # Point 0: global u=22, v=52 (only in Tile 0; local u=22, local v=52 -> inside Tile 0's mask)
    # Point 1: global u=132, v=52 (only in Tile 1; local u=132-60=72, local v=52 -> inside Tile 1's mask)
    # Point 2: global u=132, v=12 (in Tile 1; local u=72, local v=12 -> outside mask)
    u = np.array([22, 132, 132], dtype=np.int64)
    v = np.array([52, 52, 12], dtype=np.int64)
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    instances_out = {}

    labeler.score_points("dummy.png", u, v, image=image, instances_out=instances_out)

    assert tile_call_count[0] == 2
    local_ids = instances_out["point_local_instance"]
    assert local_ids[0] == 0  # Tile 0's instance
    assert local_ids[1] == 1  # Tile 1's instance
    assert local_ids[2] == -1  # outside mask in Tile 1
    assert len(instances_out["instances"]) == 2
    assert instances_out["instances"][0]["conf"] == pytest.approx(0.75, abs=1e-4)
    assert instances_out["instances"][1]["conf"] == pytest.approx(0.85, abs=1e-4)


def test_score_points_non_integer_boundary_positions() -> None:
    """Points whose mapped mask coordinates are not integers or integers boundaries are sampled reliably."""
    labeler = HeadFusionLabeler.__new__(HeadFusionLabeler)
    labeler.class_names = ["clutter", "furniture"]
    labeler._flat_prompts = ["bag", "chair"]
    labeler._prompt_owner = np.array([0, 1], dtype=np.int64)
    labeler.tile_size = 100
    labeler.min_overlap_frac = 0.2
    labeler.fine_queries = None
    labeler.last_fine_probs = None
    labeler.last_tile_diagnostics = None
    labeler.instance_classes = {2}
    labeler.instance_params = {
        "mask_threshold": 0.5,
        "min_mask_pixels": 12,
    }
    labeler.last_instances = None

    # Mask resolution 17 x 19 on a 100 x 100 tile.
    # Non-integer cell scaling: (u * 17) // 100 and (v * 19) // 100.
    mask = np.zeros((19, 17), dtype=np.float16)
    # Fill mask region y in [4, 10], x in [3, 9] (36 px >= 12)
    mask[4:10, 3:9] = 0.88

    fake_instances = {
        "prompt_idx": np.array([1], dtype=np.int32),
        "conf": np.array([0.91], dtype=np.float32),
        "masks": mask[None, :, :],
    }

    def _stub_score_tile(tile_img):
        labeler.last_instances = fake_instances
        return torch.zeros((2, tile_img.shape[0], tile_img.shape[1]), dtype=torch.float32)

    labeler._score_tile = _stub_score_tile

    # Point 0: u=29, v=37 -> mx = (29*17)//100 = 4, my = (37*19)//100 = 7 (fractional: 4.93, 7.03; interior of [4:10, 3:9])
    # Point 1: u=71, v=83 -> mx = (71*17)//100 = 12, my = (83*19)//100 = 15 (exterior)
    # Also pass float coordinates to verify robust conversion to int64
    u = np.array([29.3, 71.7])
    v = np.array([37.8, 83.2])
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    instances_out = {}

    labeler.score_points("dummy.png", u, v, image=image, instances_out=instances_out)

    local_ids = instances_out["point_local_instance"]
    assert local_ids[0] == 0
    assert local_ids[1] == -1
