"""Prompts added to the instance query set never change how the other objects are associated."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from samcloud.classification.custom import global_classes_path
from samcloud.engine import classify
from samcloud.engine.instances import (
    associate_instances,
    attach_instances,
    secondary_prompt_indices,
    snapshot_instances,
    write_instance_product,
)
from samcloud.engine.sam_labeler import HeadFusionLabeler

GLOBAL = json.loads(global_classes_path().read_text(encoding="utf-8"))
PROMPTS = ["furniture", "bed", "cabinet", "television", "screen", "computer monitor", "ceiling mounted fixture",
           "ceiling lamp", "pendant light"]
BED, CABINET, TV, SCREEN, MONITOR, LAMP, PENDANT = 1, 2, 3, 4, 5, 7, 8
ADDED = (4, 5, 6, 7, 8)
FURNITURE, EQUIPMENT = 17, 16
PARAMS = {"min_points": 100, "split_radius_m": 0.05, "class_names": ["x"] * 19}


def _patch(origin, size, rng, step=0.023):
    axes = [np.arange(0.004, max(length, 0.005), step) for length in size]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3) + np.asarray(origin)
    return grid + rng.normal(0, 0.002, grid.shape)


class Scene:
    def __init__(self):
        self.rng = np.random.default_rng(20261122)
        self.parts, self.classes, self.masks, self.count = [], [], [], 0

    def patch(self, origin, size, class_id):
        points = _patch(origin, size, self.rng)
        rows = np.arange(self.count, self.count + len(points))
        self.parts.append(points)
        self.classes.append(np.full(len(points), class_id, np.uint8))
        self.count += len(points)
        return rows

    def seen(self, rows, prompt, class_id, views, conf=0.83):
        for view in views:
            pick = rows[self.rng.random(len(rows)) < 0.94]
            self.masks.append((pick, prompt, class_id, conf, view))

    def associate(self, masks=None, secondary=ADDED):
        masks = self.masks if masks is None else masks
        return associate_instances(
            np.concatenate([m[0] for m in masks]),
            np.concatenate([np.full(len(m[0]), number) for number, m in enumerate(masks)]),
            np.array([m[2] for m in masks]), np.array([m[3] for m in masks]), np.array([m[1] for m in masks]),
            self.count, np.concatenate(self.classes), np.concatenate(self.parts),
            {**PARAMS, "secondary_prompt_ids": list(secondary)}, PROMPTS, {0, 6},
            np.array([m[4] for m in masks]),
        )

    def primary_masks(self):
        return [mask for mask in self.masks if mask[1] not in ADDED]


def test_added_masks_do_not_change_the_other_objects():
    scene = Scene()
    bed = scene.patch((0.0, 0.0, 0.4), (1.23, 0.83, 0.005), FURNITURE)
    cabinet = scene.patch((2.31, 0.0, 0.0), (0.61, 0.005, 0.87), FURNITURE)
    monitor = scene.patch((4.13, 0.0, 0.9), (0.57, 0.005, 0.36), FURNITURE)
    lamp = scene.patch((1.07, 1.53, 2.43), (0.37, 0.39, 0.005), EQUIPMENT)
    scene.seen(bed, BED, FURNITURE, range(6))
    scene.seen(cabinet, CABINET, FURNITURE, range(6))
    # a stray "screen" that covers half of the bed and all of the cabinet, stronger than both
    scene.seen(np.concatenate((bed[: len(bed) // 2], cabinet)), SCREEN, FURNITURE, range(6), conf=0.97)
    scene.seen(monitor, MONITOR, FURNITURE, range(5))
    scene.seen(lamp, LAMP, EQUIPMENT, range(5))

    ids, metadata = scene.associate()
    base_ids, base_metadata = scene.associate(scene.primary_masks(), secondary=())

    owned = base_ids > 0
    np.testing.assert_array_equal(ids[owned], base_ids[owned])
    assert metadata[: len(base_metadata)] == base_metadata
    added = [item for item in metadata if item.get("secondary")]
    assert sorted(item["label"] for item in added) == ["ceiling lamp", "monitor"]
    assert set(ids[monitor][ids[monitor] > 0]).isdisjoint(set(base_ids[owned]))
    assert all(item["id"] > len(base_metadata) for item in added)
    # the rule is needed: with one common competition the same masks rob the bed and the cabinet
    mixed_ids, _ = scene.associate(secondary=())
    assert np.mean(mixed_ids[owned] == base_ids[owned]) < 0.8


def test_screen_synonym_of_a_television_is_one_object():
    scene = Scene()
    inner = scene.patch((0.0, 0.0, 0.8), (0.93, 0.005, 1.07), FURNITURE)
    rim = scene.patch((0.94, 0.0, 0.8), (0.31, 0.005, 1.07), FURNITURE)
    scene.seen(inner, TV, FURNITURE, range(6))
    scene.seen(np.concatenate((inner, rim)), SCREEN, FURNITURE, range(6), conf=0.95)
    ids, metadata = scene.associate()
    assert [item["label"] for item in metadata] == ["television"]
    assert metadata[0]["point_count"] == np.count_nonzero(ids) > 0.98 * scene.count


def test_two_names_of_one_lamp_are_one_object():
    scene = Scene()
    shade = scene.patch((1.0, 1.0, 2.21), (0.43, 0.41, 0.005), EQUIPMENT)
    inner = scene.patch((1.02, 1.03, 2.236), (0.39, 0.37, 0.005), EQUIPMENT)
    scene.seen(shade, LAMP, EQUIPMENT, range(6))
    scene.seen(inner, PENDANT, EQUIPMENT, range(6, 11))
    ids, metadata = scene.associate()
    assert len(metadata) == 1 and metadata[0]["point_count"] == np.count_nonzero(ids) > 0.98 * scene.count


def test_neighbouring_fixtures_stay_two_objects():
    scene = Scene()
    left = scene.patch((1.0, 1.0, 2.43), (0.31, 0.29, 0.005), EQUIPMENT)
    right = scene.patch((1.335, 1.0, 2.43), (0.31, 0.29, 0.005), EQUIPMENT)
    scene.seen(left, LAMP, EQUIPMENT, range(6))
    scene.seen(right, PENDANT, EQUIPMENT, range(6))
    assert len(scene.associate()[1]) == 2


def test_added_object_does_not_flood_free_points():
    scene = Scene()
    monitor = scene.patch((0.0, 0.0, 0.9), (0.57, 0.005, 0.36), FURNITURE)
    beside = scene.patch((0.58, 0.0, 0.9), (0.91, 0.005, 0.36), FURNITURE)
    scene.seen(monitor, MONITOR, FURNITURE, range(5))
    ids, _ = scene.associate()
    points = np.concatenate(scene.parts)
    assert points[ids > 0, 0].max() < 0.57 + 0.25
    assert np.count_nonzero(ids[beside]) < 0.4 * len(beside)


def test_attach_keeps_the_primary_objects(tmp_path):
    scene = Scene()
    bed = scene.patch((0.0, 0.0, 0.4), (1.23, 0.83, 0.005), FURNITURE)
    monitor = scene.patch((1.96, 0.0, 0.4), (0.57, 0.83, 0.005), FURNITURE)
    scene.patch((0.3, 0.84, 0.4), (1.43, 0.41, 0.005), FURNITURE)  # free points both objects could grow into
    scene.seen(bed, BED, FURNITURE, range(6))
    scene.seen(monitor, MONITOR, FURNITURE, range(6))
    points, classes = np.concatenate(scene.parts), np.concatenate(scene.classes)
    results = []
    for tag, masks, secondary in (("with", scene.masks, ADDED), ("without", scene.primary_masks(), ())):
        ids, metadata = scene.associate(masks, secondary)
        raw, final = tmp_path / f"{tag}_raw.ply", tmp_path / f"{tag}.ply"
        for path in (raw, final):
            classify.write_classified_ply(path, points, np.full((len(points), 3), 99, np.uint8), classes)
        write_instance_product(raw, ids, metadata)
        # the sidecar is per folder, so hand the metadata over directly
        assert snapshot_instances(raw, tmp_path / f"{tag}.npz", metadata)
        report = attach_instances(tmp_path / f"{tag}.npz", final, {"min_points": 100, "attach_radius_m": 0.05})
        with np.load(tmp_path / f"{tag}.npz") as archive:
            assert len(archive["instance_id"]) == len(points)
        from samcloud.engine.ply_io import read_vertex_array

        results.append((np.asarray(read_vertex_array(final)["instance_id"]), report["instances"]))
    (with_ids, with_meta), (base_ids, base_meta) = results
    np.testing.assert_array_equal(with_ids[base_ids > 0], base_ids[base_ids > 0])
    assert with_meta[: len(base_meta)] == base_meta
    assert [item["label"] for item in with_meta if item.get("secondary")] == ["monitor"]


def _selected(config):
    prompts = config["prompts"]["indoor"]
    flat = [text for group in prompts for text in group]
    indices = secondary_prompt_indices(config, "indoor", config["indoor"], prompts)
    return {flat[index] for index in indices}


def test_secondary_layer_is_exactly_what_was_added():
    selected = _selected(GLOBAL)
    ceiling = GLOBAL["prompts"]["indoor"][15]
    assert selected == {*ceiling, "flat screen tv", "flat screen", "computer monitor", "computer screen", "screen", "laptop"}
    assert "television" not in selected  # an old prompt: it keeps competing with bed and cabinet as before
    older_project_file = copy.deepcopy(GLOBAL)
    older_project_file.pop("instance_secondary")
    assert _selected(older_project_file) == selected


def test_suppress_queries_never_enter_the_layer():
    config = {"instance_secondary": {"indoor": {"classes": [], "prompts": ["screen"]}}}
    assert secondary_prompt_indices(config, "indoor", ["furniture"], [["furniture", "screen"], ["screen"]]) == {1}
    assert secondary_prompt_indices(GLOBAL, "outdoor", GLOBAL["outdoor"], GLOBAL["prompts"]["outdoor"]) == set()


def _labeler(secondary, instances):
    labeler = HeadFusionLabeler.__new__(HeadFusionLabeler)
    labeler.class_names = ["furniture"]
    labeler._flat_prompts = ["furniture", "bed", "screen"]
    labeler._prompt_owner = np.zeros(3, np.int64)
    labeler.tile_size, labeler.min_overlap_frac = 100, 0.2
    labeler.fine_queries = labeler.last_fine_probs = labeler.last_tile_diagnostics = None
    labeler.instance_params = {"mask_threshold": 0.5, "min_mask_pixels": 12}
    labeler.secondary_prompt_ids = set(secondary)

    def score_tile(tile):
        labeler.last_instances = instances
        return torch.zeros((1, *tile.shape[:2]), dtype=torch.float32)

    labeler._score_tile = score_tile
    return labeler


def test_added_mask_never_takes_a_point_in_the_view():
    bed, screen = np.zeros((20, 20), np.float16), np.zeros((20, 20), np.float16)
    bed[2:12, 2:12] = 0.71
    screen[6:16, 6:16] = 0.93
    both = {"prompt_idx": np.array([1, 2]), "conf": np.array([0.62, 0.97], np.float32), "masks": np.stack((bed, screen))}
    u = np.array([17, 41, 47, 73, 92])
    image, out = np.zeros((100, 100, 3), np.uint8), {}
    _labeler({2}, both).score_points("x.png", u, u, image=image, instances_out=out)
    assert out["point_local_instance"].tolist() == [0, 0, 0, -1, -1]
    assert out["point_secondary_instance"].tolist() == [-1, 1, 1, 1, -1]
    alone, plain = {key: value[:1] for key, value in both.items()}, {}
    _labeler(set(), alone).score_points("x.png", u, u, image=image, instances_out=plain)
    np.testing.assert_array_equal(plain["point_local_instance"], out["point_local_instance"])


@pytest.mark.parametrize("config", [None, {}])
def test_classify_passes_both_layers_to_the_association(tmp_path, monkeypatch, config):
    count = 140
    made, captured = {}, {}

    class Labeler:
        def load_image(self, _path):
            return np.zeros((16, 16, 3), np.uint8), 16, 16

        def score_points(self, _path, u, _v, instances_out=None, **_kwargs):
            instances_out.update(
                point_local_instance=np.where(np.arange(len(u)) < 90, 0, -1),
                point_secondary_instance=np.where(np.arange(len(u)) >= 50, 1, -1),
                instances=[{"class_id": 1, "prompt_idx": 1, "conf": 0.8}, {"class_id": 1, "prompt_idx": 2, "conf": 0.9}],
            )
            return np.full((len(u), 1), 0.9, np.float32)

    def make(*_args, **kwargs):
        made.update(kwargs["instance_params"])
        return Labeler()

    def associate(*args, **_kwargs):
        captured["pairs"] = set(zip(args[0].tolist(), args[1].tolist()))
        return np.zeros(count, np.uint32), []

    monkeypatch.setattr(classify, "make_point_labeler", make)
    monkeypatch.setattr(classify, "associate_instances", associate)
    monkeypatch.setattr(classify, "project_points", lambda *_a, **_k: (
        np.arange(count), np.zeros(count, int), np.zeros(count, int), np.full(count, 2.0)))
    args = SimpleNamespace(
        background_threshold=0.4, threshold_sweep="", dense_dir=str(tmp_path), labels_dir=None, images_dir=str(tmp_path),
        depth_check=0, out=str(tmp_path / "out.ply"), scores_out=None, min_votes=1, summary_file=None, class_set="indoor",
        engine="sam3", mode="segearth_ov3", query_threshold=0.4, no_instances=False,
    )
    fine = {"instance_classes": {"indoor": ["furniture"]}}
    if config is None:
        fine["instance_secondary"] = {"indoor": {"prompts": ["screen"]}}
    xyz = np.column_stack((np.linspace(0.003, 1.0, count), np.zeros(count), np.zeros(count)))
    classify.run_segearth_ov3(
        args, xyz, np.zeros((count, 3), np.uint8), {1: object()}, {1: SimpleNamespace(name="v.png", camera_id=1)},
        [(1, list(range(count)))], ["furniture"], 0, None, prompts=[["furniture", "bed", "screen"]],
        fine_config=fine, instance_config={"enabled": True},
    )
    assert made["secondary_prompt_ids"] == [2]
    assert captured["pairs"] == {(i, 0) for i in range(90)} | {(i, 1) for i in range(50, count)}
