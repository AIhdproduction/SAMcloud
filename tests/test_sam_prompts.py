"""SAM3 prompts are kept apart from class names and per class set."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from samcloud.core.paths import CONFIG_DIRECTORY
from samcloud.engine import sam_labeler


def _classes() -> dict:
    return json.loads((CONFIG_DIRECTORY / "classes.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("class_set", ["indoor", "outdoor"])
def test_prompts_are_separate_short_terms_aligned_with_classes(class_set: str) -> None:
    """Each prompt goes to SAM3 on its own: never comma-joined synonym strings."""
    data = _classes()
    prompts = data["prompts"][class_set]
    assert len(prompts) == len(data[class_set])
    for entry, name in zip(prompts, data[class_set]):
        assert isinstance(entry, list) and entry
        for prompt in entry:
            assert "," not in prompt and 1 <= len(prompt.split()) <= 4
        # the class name string carries the same prompts after its identity
        assert name.split(", ")[1:] == entry


@pytest.mark.parametrize("class_set", ["indoor", "outdoor"])
def test_las_codes_and_colours_cover_every_class(class_set: str) -> None:
    data = _classes()
    count = len(data[class_set]) + 1
    assert len(data["las_codes"][class_set]) == count and len(data["colors"][class_set]) == count
    assert all(0 <= code <= 255 for code in data["las_codes"][class_set])
    assert len(set(data["las_codes"][class_set])) == count


def test_class_sets_keep_the_names_other_steps_rely_on() -> None:
    data = _classes()
    indoor = {name.split(",")[0] for name in data["indoor"]}
    outdoor = {name.split(",")[0] for name in data["outdoor"]}
    assert {"floor", "ceiling", "wall"} <= indoor  # plane fill
    assert {"ground", "road", "sidewalk", "curb", "low vegetation", "tree", "building", "water"} <= outdoor  # geometry.yml
    assert not {"floor", "ceiling"} & outdoor and not {"ground", "road"} & indoor


def test_nomivox_import_splits_scenes_and_keeps_codes() -> None:
    from samcloud.tools.import_nomivox_classes import build_classes

    preset = {
        "unclassified_code": 1,
        "special": [{"id": "unclassified", "code": 1, "color": "#000000"}],
        "classes": [
            {"id": "ground", "name_de": "Boden", "code": 2, "color": "#8B6F47", "scene": "outdoor",
             "enabled": True, "prompts": ["ground", "terrain"]},
            {"id": "wall", "name_de": "Wand", "code": 100, "color": "#E0C9A6", "scene": "indoor",
             "enabled": True, "prompts": ["wall", "interior wall"]},
            {"id": "window", "name_de": "Fenster", "code": 104, "color": "#80DEEA", "scene": "both",
             "enabled": True, "prompts": ["window"]},
            {"id": "off", "name_de": "Aus", "code": 120, "color": "#FFFFFF", "scene": "both",
             "enabled": False, "prompts": ["off"]},
        ],
    }

    classes = build_classes(preset, "test")

    assert classes["indoor"] == ["wall, wall, interior wall", "window, window"]
    assert classes["outdoor"] == ["ground, ground, terrain", "window, window"]
    assert classes["las_codes"] == {"indoor": [1, 100, 104], "outdoor": [1, 2, 104]}
    assert classes["prompts"]["indoor"] == [["wall", "interior wall"], ["window"]]
    assert classes["labels_de"]["outdoor"] == ["Unklassifiziert", "Boden", "Fenster"]


def test_prompt_scores_are_max_pooled_per_class() -> None:
    import torch

    flat, owner = sam_labeler.flatten_prompts([["wall", "interior wall"], "floor", ["door", "doorway", "gate"]])
    assert flat == ["wall", "interior wall", "floor", "door", "doorway", "gate"]
    assert owner.tolist() == [0, 0, 1, 2, 2, 2]
    scores = torch.tensor([0.2, 0.7, 0.4, 0.1, 0.9, 0.3])[:, None, None].expand(6, 2, 3)

    reduced = sam_labeler.reduce_prompt_scores(scores, owner, 3)

    assert reduced.shape == (3, 2, 3)
    assert torch.allclose(reduced[:, 0, 0], torch.tensor([0.7, 0.4, 0.9]))


def test_legacy_labeler_maps_prompt_hits_back_to_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    class _OnePromptHit:
        def __init__(self, overrides=None):
            self.args = SimpleNamespace()

        class _Masks:
            data = torch.ones((1, 4, 4))

            def __len__(self) -> int:
                return 1

        def __call__(self, source, text, verbose=False):
            boxes = SimpleNamespace(conf=torch.tensor([0.9]), cls=torch.tensor([4.0]))  # 5th prompt = "doorway"
            return [SimpleNamespace(masks=self._Masks(), boxes=boxes)]

    monkeypatch.setattr(sam_labeler, "SAM3SemanticPredictor", _OnePromptHit)
    labeler = sam_labeler.SemanticLabeler(
        ["wall", "floor", "door"], prompts=[["wall", "interior wall"], ["floor"], ["door", "doorway"]],
    )

    raster, _conf = labeler._label_tile(np.zeros((4, 4, 3), dtype=np.uint8))

    assert np.all(raster == 3)  # prompt index 4 belongs to class 3 (door)


def test_resolve_prompts_defaults_and_validates() -> None:
    names = ["wall, interior wall", "ceiling, soffit"]
    assert sam_labeler.resolve_prompts(names) == names
    assert sam_labeler.resolve_prompts(names, ["wall", "ceiling"]) == ["wall", "ceiling"]
    with pytest.raises(ValueError):
        sam_labeler.resolve_prompts(names, ["wall"])


class _RecordingPredictor:
    calls: list = []

    def __init__(self, overrides=None):
        self.args = SimpleNamespace()

    def __call__(self, source, text, verbose=False):
        _RecordingPredictor.calls.append(list(text))
        return [SimpleNamespace(masks=None, boxes=None)]


def test_labelers_send_prompts_not_class_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sam_labeler, "SAM3SemanticPredictor", _RecordingPredictor)
    monkeypatch.setattr(sam_labeler, "RawHeadSAM3SemanticPredictor", _RecordingPredictor)
    _RecordingPredictor.calls = []
    names = ["wall, interior wall, partition wall", "ceiling, interior ceiling"]
    prompts = ["wall", "ceiling"]
    tile = np.zeros((32, 32, 3), dtype=np.uint8)

    legacy = sam_labeler.SemanticLabeler(names, prompts=prompts)
    raster, _conf = legacy._label_tile(tile)
    assert not raster.any()

    head = sam_labeler.HeadFusionLabeler(names, prompts=prompts)
    assert head.class_names == names
    with pytest.raises(AttributeError):  # fake result has no semantic heads; the call itself is what we check
        head._score_tile(tile)

    assert _RecordingPredictor.calls == [prompts, prompts]


def test_labeler_without_prompts_uses_class_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sam_labeler, "SAM3SemanticPredictor", _RecordingPredictor)
    _RecordingPredictor.calls = []
    names = ["ground, terrain", "road, street"]

    sam_labeler.SemanticLabeler(names)._label_tile(np.zeros((8, 8, 3), dtype=np.uint8))

    assert _RecordingPredictor.calls == [names]


def test_score_points_returns_thresholded_label_raster(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(sam_labeler, "RawHeadSAM3SemanticPredictor", _RecordingPredictor)
    head = sam_labeler.HeadFusionLabeler(["floor", "wall"], tile_size=64)
    probs = torch.zeros((2, 48, 64))
    probs[0, :, :20] = 0.9   # floor on the left
    probs[1, :, 40:] = 0.6   # wall on the right
    probs[1, :, 20:40] = 0.2  # below threshold in the middle
    monkeypatch.setattr(head, "_score_tile", lambda tile: probs)
    image = np.zeros((48, 64, 3), dtype=np.uint8)

    scores, labels = head.score_points(None, np.array([5, 30, 50]), np.array([10, 10, 10]), image=image,
                                       label_threshold=0.4)

    assert labels.shape == (48, 64)
    assert np.all(labels[:, :20] == 1) and np.all(labels[:, 20:40] == 0) and np.all(labels[:, 40:] == 2)
    assert scores.shape == (3, 2) and scores[0, 0] == pytest.approx(0.9)
    assert isinstance(head.score_points(None, np.array([5]), np.array([5]), image=image), np.ndarray)


def test_label_rasters_are_saved_per_view(tmp_path) -> None:
    import cv2

    from samcloud.engine.classify import label_raster_path, save_label_raster

    raster = np.random.default_rng(0).integers(0, 11, (24, 32), dtype=np.uint8)
    save_label_raster(tmp_path, "cubemap/front/panorama_000001.jpg", raster)

    path = label_raster_path(tmp_path, "cubemap/front/panorama_000001.jpg")
    assert path == tmp_path / "cubemap" / "front" / "panorama_000001.jpg.png"
    assert np.array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), raster)


def test_cubemap_faces_are_one_tile_but_large_photos_are_tiled() -> None:
    assert sam_labeler.effective_tile_size(1024, 1024, 1008) == 1024
    assert sam_labeler.effective_tile_size(1200, 900, 1008) == 1200
    assert sam_labeler.effective_tile_size(4000, 3000, 1008) == 1008
    assert len(list(sam_labeler.iter_tiles(1024, 1024, sam_labeler.effective_tile_size(1024, 1024, 1008), 0.2))) == 1


def test_native_1440_cubemap_faces_are_one_tile() -> None:
    assert sam_labeler.effective_tile_size(1440, 1440, 1008) == 1440
