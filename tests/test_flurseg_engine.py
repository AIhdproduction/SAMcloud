"""FlurSeg engine: class mapping, sliding-window inference, labeler interface and engine plumbing."""

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from samcloud.app.processing import ProcessingManager, classification_engine, reclassification_available
from samcloud.engine import pipeline
from samcloud.engine.flurseg_labeler import (
    CONFIG_PATH,
    FlurSegLabeler,
    build_mapping,
    load_config,
    sliding_window_probabilities,
)

FLURSEG_LABELS = ["other", "ground", "paved_surface", "low_vegetation", "tree", "building", "vehicle",
                  "construction", "curb", "road_marking", "utility_cover"]
CLASS_NAMES = ["ground, soil", "road, asphalt", "sidewalk", "curb, kerb", "low vegetation, grass", "tree",
               "building", "vehicle", "construction machinery, excavator", "manhole, manhole cover"]


class FakeEomt(torch.nn.Module):
    """Tiny stand-in: one query per FlurSeg class, mask logits follow the left/right image half."""

    def __init__(self, image_size=32, num_classes=11):
        super().__init__()
        self.config = SimpleNamespace(image_size=image_size)
        self.flurseg_labels = FLURSEG_LABELS[:num_classes]
        self.num_classes = num_classes
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, pixel_values):
        self.calls += 1
        batch, _, height, width = pixel_values.shape
        queries = self.num_classes
        class_logits = torch.full((batch, queries, self.num_classes + 1), -20.0)
        for query in range(queries):
            class_logits[:, query, query] = 20.0
        masks = torch.full((batch, queries, height // 4, width // 4), -20.0)
        # mean red channel > 0 -> "building" (5), else "paved_surface" (2)
        red = pixel_values[:, 0].mean(dim=(1, 2))
        for index in range(batch):
            masks[index, 5 if red[index] > 0 else 2] = 20.0
        return SimpleNamespace(class_queries_logits=class_logits, masks_queries_logits=masks)


def test_shipped_config_maps_every_model_class():
    config = load_config(CONFIG_PATH)
    mapping = config["class_mapping"]
    assert set(mapping) == set(FLURSEG_LABELS)
    assert mapping["other"] is None
    outdoor = [name.split(",", 1)[0].strip()
               for name in __import__("json").loads(Path("config/classes.json").read_text(encoding="utf-8"))["outdoor"]]
    assert all(target in outdoor for target in mapping.values() if target is not None)


def test_build_mapping_sums_into_samcloud_classes_and_drops_other():
    mapping = build_mapping(FLURSEG_LABELS, CLASS_NAMES, load_config(CONFIG_PATH)["class_mapping"])
    assert mapping.shape == (len(CLASS_NAMES), len(FLURSEG_LABELS))
    road = CLASS_NAMES.index("road, asphalt")
    assert mapping[road, FLURSEG_LABELS.index("paved_surface")] == 1
    assert mapping[road, FLURSEG_LABELS.index("road_marking")] == 1
    assert mapping[:, FLURSEG_LABELS.index("other")].sum() == 0
    assert mapping[CLASS_NAMES.index("sidewalk")].sum() == 0
    assert mapping.sum(axis=0)[1:].tolist() == [1.0] * 10


def test_build_mapping_rejects_unknown_targets_and_missing_classes():
    with pytest.raises(ValueError, match="unknown SAMcloud class"):
        build_mapping(["ground"], CLASS_NAMES, {"ground": "lava"})
    with pytest.raises(ValueError, match="no entry"):
        build_mapping(["ground", "tree"], CLASS_NAMES, {"ground": "ground"})


@pytest.mark.parametrize("height,width", [(20, 27), (32, 32), (71, 53)])
def test_sliding_window_probabilities_cover_any_image_size(height, width):
    model = FakeEomt(image_size=32)
    image = torch.full((3, height, width), 0.7)
    probs = sliding_window_probabilities(model, image, window=32, stride=16, batch_size=3)
    assert probs.shape == (11, height, width)
    assert torch.allclose(probs.sum(dim=0), torch.ones(height, width), atol=1e-4)
    assert int(probs.argmax(dim=0).unique().item()) == 5


def _small_config(tmp_path, window=32, stride=16):
    text = Path(CONFIG_PATH).read_text(encoding="utf-8")
    text = text.replace("window_size: 768", f"window_size: {window}").replace("stride: 384", f"stride: {stride}")
    path = tmp_path / "flurseg.yml"
    path.write_text(text, encoding="utf-8")
    return path


def test_labeler_scores_points_and_label_raster_in_samcloud_ids(tmp_path):
    model = FakeEomt(image_size=32)
    labeler = FlurSegLabeler(CLASS_NAMES, device="cpu", model=model, config_path=_small_config(tmp_path))
    assert model.calls == 0
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    image[..., 2] = 250  # BGR: strong red -> building
    u = np.array([0, 10, 49])
    v = np.array([0, 20, 39])
    scores, labels = labeler.score_points(None, u, v, image=image, label_threshold=0.4)
    building = CLASS_NAMES.index("building")
    assert scores.shape == (3, len(CLASS_NAMES))
    assert np.all(scores.argmax(axis=1) == building)
    assert np.all(scores[:, building] > 0.9)
    assert labels.shape == (40, 50) and labels.dtype == np.uint8
    assert set(np.unique(labels).tolist()) == {building + 1}

    dark = np.zeros((40, 50, 3), dtype=np.uint8)  # normalised red < 0 -> paved_surface -> road
    scores = labeler.score_points(None, u, v, image=dark)
    assert np.all(scores.argmax(axis=1) == CLASS_NAMES.index("road, asphalt"))
    with pytest.raises(ValueError, match="in-bounds"):
        labeler.score_points(None, np.array([50]), np.array([0]), image=dark)


def test_labeler_rejects_window_that_differs_from_model_size(tmp_path):
    config = _small_config(tmp_path)
    with pytest.raises(ValueError, match="window_size"):
        FlurSegLabeler(CLASS_NAMES, device="cpu", model=FakeEomt(image_size=64), config_path=config)


def test_classify_uses_flurseg_labeler_for_engine_flurseg(monkeypatch):
    from samcloud.engine import classify

    created = {}

    class Recorder:
        def __init__(self, class_names, unknown_id=0):
            created["names"] = class_names

    monkeypatch.setattr("samcloud.engine.flurseg_labeler.FlurSegLabeler", Recorder)
    labeler = classify.make_point_labeler(Namespace(engine="flurseg"), CLASS_NAMES, 0)
    assert isinstance(labeler, Recorder) and created["names"] == CLASS_NAMES


def test_pipeline_engine_is_sam3_for_indoor_and_scores_follow_engine():
    outdoor = Namespace(class_set="outdoor", classification_engine="flurseg", classification_mode="legacy")
    indoor = Namespace(class_set="indoor", classification_engine="flurseg", classification_mode="legacy")
    assert pipeline._classification_engine(outdoor) == "flurseg"
    assert pipeline._classification_engine(indoor) == "sam3"
    assert pipeline._writes_scores(outdoor) is True
    assert pipeline._writes_scores(indoor) is False
    assert pipeline._writes_scores(Namespace(class_set="indoor", classification_mode="segearth_ov3")) is True


def _command(classification):
    manager = object.__new__(ProcessingManager)
    return manager._pipeline_command(Path("D:/demo"), {
        "settings": {
            "workflow": {"use_gpu": True},
            "classification": classification,
            "coordinate_systems": {"working_crs": "LOCAL", "export_crs": "EPSG:2056"},
        },
        "acquisition": {"type": "camera", "sources": {"image_directory": "D:/images"}},
        "paths": {"image_directory": "D:/images"},
    }, "none")


@pytest.mark.parametrize("classification,expected", [
    ({"class_set": "outdoor", "engine": "flurseg"}, "flurseg"),
    ({"class_set": "outdoor"}, "sam3"),
    ({"class_set": "indoor", "engine": "flurseg"}, "sam3"),
    ({"class_set": "outdoor", "engine": "bogus"}, "sam3"),
])
def test_pipeline_command_forwards_engine(classification, expected):
    command = _command(classification)
    assert command[command.index("--classification-engine") + 1] == expected
    assert classification_engine(classification) == expected


def test_reclassification_available_accepts_compressed_inputs(tmp_path):
    dense = tmp_path / "products" / "dense"
    (dense / "sparse").mkdir(parents=True)
    (dense / "fused.ply.gz").write_bytes(b"x")
    assert not reclassification_available(tmp_path)
    (dense / "fused.ply.vis.gz").write_bytes(b"x")
    assert reclassification_available(tmp_path)


def test_pipeline_cli_accepts_engine_and_reclassify(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(pipeline, "_reclassify_project",
                        lambda args, *rest: seen.update(engine=args.classification_engine, reclassify=args.reclassify))
    monkeypatch.setattr("sys.argv", ["pipeline", "--outdir", str(tmp_path / "products"),
                                     "--classification-engine", "flurseg", "--reclassify"])
    pipeline.main()
    assert seen == {"engine": "flurseg", "reclassify": True}


def test_create_project_stores_engine_outdoor_only(tmp_path):
    from samcloud.core.project_manager import ProjectManager

    manager = ProjectManager(tmp_path / "projects")
    outdoor = manager.load(manager.create_project("Site", "D:/images", classification_engine="flurseg"))
    assert outdoor["settings"]["classification"]["engine"] == "flurseg"
    indoor = manager.load(manager.create_project("Room", "D:/images", capture_type="camera", class_set="indoor",
                                                 classification_engine="flurseg"))
    assert indoor["settings"]["classification"]["engine"] == "sam3"
    default = manager.load(manager.create_project("Default", "D:/images"))
    assert default["settings"]["classification"]["engine"] == "sam3"
    with pytest.raises(ValueError, match="classification model"):
        manager.create_project("Bad", "D:/images", classification_engine="yolo")


def test_accumulate_votes_fast_path_matches_add_at():
    from samcloud.engine.classify import accumulate_weighted_votes

    rng = np.random.default_rng(3)
    probs = rng.random((500, 4)).astype(np.float32)
    for rows in (np.sort(rng.choice(1000, 500, replace=False)), rng.integers(0, 50, 500)):
        fast_scores, fast_counts = np.zeros((1000, 4), np.float32), np.zeros(1000, np.uint16)
        ref_scores, ref_counts = np.zeros((1000, 4), np.float32), np.zeros(1000, np.uint16)
        accumulate_weighted_votes(fast_scores, fast_counts, rows, probs, 0.3)
        visible = probs.max(axis=1) >= 0.3
        np.add.at(ref_scores, rows[visible], probs[visible])
        np.add.at(ref_counts, rows[visible], 1)
        assert np.allclose(fast_scores, ref_scores) and np.array_equal(fast_counts, ref_counts)
