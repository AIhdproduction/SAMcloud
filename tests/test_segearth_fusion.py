"""Unit tests for the pure SegEarth-OV3 fusion and voting equations."""

import sys
from pathlib import Path

import numpy as np
import torch


ENGINE_DIR = Path(__file__).resolve().parents[1] / "samcloud" / "engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from classify import accumulate_weighted_votes, resolve_weighted_votes
from sam_labeler import aggregate_instances, apply_presence_gate, assign_classes, fuse_heads


def test_aggregate_instances_weights_queries_and_ignores_low_confidence_queries():
    inst_probs = torch.tensor([
        [
            [[0.8, 0.1], [0.2, 0.9]],
            [[0.9, 0.7], [0.5, 0.3]],
        ],
        [
            [[0.4, 0.6], [0.8, 0.2]],
            [[0.9, 0.9], [0.9, 0.9]],
        ],
    ])
    conf_scores = torch.tensor([[0.5, 0.3], [0.2, 0.1]])

    actual = aggregate_instances(inst_probs, conf_scores, query_threshold=0.4)

    torch.testing.assert_close(actual[0], torch.tensor([[0.4, 0.05], [0.1, 0.45]]))
    torch.testing.assert_close(actual[1], torch.zeros((2, 2)))
    torch.testing.assert_close(
        aggregate_instances(torch.empty((2, 0, 2, 2)), torch.empty((2, 0)), query_threshold=0.4),
        torch.zeros((2, 2, 2)),
    )


def test_fuse_heads_takes_the_elementwise_maximum():
    sem_probs = torch.tensor([[[0.8, 0.2]]])
    inst_agg = torch.tensor([[[0.4, 0.9]]])

    torch.testing.assert_close(fuse_heads(sem_probs, inst_agg), torch.tensor([[[0.8, 0.9]]]))


def test_presence_gate_multiplies_each_category_exactly():
    fused = torch.tensor([[[0.8, 0.4]], [[0.9, 0.7]]])
    presence = torch.tensor([1.0, 0.001])

    torch.testing.assert_close(
        apply_presence_gate(fused, presence),
        torch.tensor([[[0.8, 0.4]], [[0.0009, 0.0007]]]),
    )


def test_assign_classes_returns_one_indexed_ids_and_background():
    final_probs = torch.tensor([
        [[0.7, 0.2], [0.1, 0.3]],
        [[0.2, 0.8], [0.4, 0.2]],
    ])

    actual = assign_classes(final_probs, background_threshold=0.5, unknown_id=0)

    torch.testing.assert_close(actual, torch.tensor([[1, 2], [0, 0]]))


def test_weighted_voting_prefers_probability_mass_and_respects_thresholds_and_minimum_votes():
    scores = np.zeros((3, 2), dtype=np.float32)
    view_counts = np.zeros(3, dtype=np.uint16)

    accumulate_weighted_votes(
        scores, view_counts, np.array([0, 1]), np.array([[0.9, 0.1], [0.8, 0.1]], dtype=np.float32), 0.4
    )
    accumulate_weighted_votes(
        scores, view_counts, np.array([0]), np.array([[0.3, 0.4]], dtype=np.float32), 0.4
    )
    accumulate_weighted_votes(
        scores, view_counts, np.array([0]), np.array([[0.3, 0.4]], dtype=np.float32), 0.4
    )
    accumulate_weighted_votes(
        scores, view_counts, np.array([0]), np.array([[0.2, 0.3]], dtype=np.float32), 0.4
    )

    actual = resolve_weighted_votes(scores, view_counts, min_votes=2, unknown_id=0)

    np.testing.assert_array_equal(actual, np.array([1, 0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(view_counts, np.array([3, 1, 0], dtype=np.uint16))
