"""Behavioural tests for explicit orthophoto semantic segmentation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import cv2
import numpy as np
import pytest

from samcloud.classification.custom import class_name, load_project_classes
from samcloud.dxf.levels import Level
from samcloud.engine.flurseg_labeler import availability as flurseg_availability
from samcloud.ortho.segment import SegmentConfig, load_segment_result, segment_fingerprint, segment_ortho

SAM3_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "sam3.pt"


@pytest.fixture
def work_dir(request: pytest.FixtureRequest) -> Path:
    """Return an isolated directory for this test session."""
    configured_root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    if configured_root:
        path = Path(configured_root) / f"segment_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return request.getfixturevalue("tmp_path_factory").mktemp("segment")


def _level() -> Level:
    return Level("level_1", "1", 0.0, 3.0, 100, 1.0, False)


def _make_ortho(project: Path, rgba: np.ndarray, classes: np.ndarray, dsm: np.ndarray | None = None,
                dtm: np.ndarray | None = None, *, level: Level | None = None, fingerprint: str = "ortho-a") -> Path:
    root = project / "products" / "ortho" / ("site" if level is None else level.id)
    root.mkdir(parents=True, exist_ok=True)
    height, width = rgba.shape[:2]
    cv2.imwrite(str(root / "ortho.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    np.save(root / "dsm.npy", np.zeros((height, width), dtype=np.float32) if dsm is None else dsm.astype(np.float32))
    np.save(root / "dtm.npy", np.zeros((height, width), dtype=np.float32) if dtm is None else dtm.astype(np.float32))
    np.save(root / "classes_pc.npy", classes.astype(np.uint8))
    cv2.imwrite(str(root / "valid.png"), (rgba[..., 3] > 0).astype(np.uint8) * 255)
    cv2.imwrite(str(root / "view_count.png"), np.ones((height, width), dtype=np.uint8))
    (root / "ortho.json").write_text(json.dumps({"bounds": [0, 0, width, height], "pixel_size": 1.0, "width": width, "height": height}), encoding="utf-8")
    (root / "fingerprint.txt").write_text(fingerprint + "\n", encoding="utf-8")
    return root


def _project(work_dir: Path, mode: str = "outdoor") -> Path:
    project = work_dir.parent / f"{work_dir.name}_project"
    project.mkdir(parents=True, exist_ok=True)
    (project / "project.samcloud.yml").write_text(f"settings:\n  classification:\n    class_set: {mode}\n", encoding="utf-8")
    return project


class _FakeLabeler:
    def __init__(self, labels: np.ndarray | int, confidence: np.ndarray | float = 0.9, calls: list[np.ndarray] | None = None) -> None:
        self.labels = labels
        self.confidence = confidence
        self.calls = calls if calls is not None else []

    def label_image_scores(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(image.copy())
        shape = image.shape[:2]
        labels = np.full(shape, self.labels, dtype=np.uint8) if np.isscalar(self.labels) else self.labels.astype(np.uint8)
        confidence = np.full(shape, self.confidence, dtype=np.float16) if np.isscalar(self.confidence) else self.confidence.astype(np.float16)
        return labels, confidence


def _label_id(project: Path, mode: str, name: str) -> int:
    names = [class_name(entry) for entry in load_project_classes(project)[mode]]
    return names.index(name) + 1


def test_pointcloud_source_copies_classes(work_dir: Path) -> None:
    project = _project(work_dir)
    rgba = np.full((3, 4, 4), 255, dtype=np.uint8)
    classes = np.array([[2, 6, 0, 11], [73, 5, 84, 91], [9, 72, 3, 82]], dtype=np.uint8)
    _make_ortho(project, rgba, classes)

    result = segment_ortho(project, level=None, source="pointcloud")

    assert np.array_equal(np.load(result.classes_path), classes)
    confidence = np.load(result.confidence_path)
    assert np.all(confidence[classes > 0] == 1.0) and np.all(confidence[classes == 0] == 0.0)


def test_combined_prefers_high_classes_on_high_dsm_and_low_on_low(work_dir: Path) -> None:
    project = _project(work_dir)
    rgba = np.full((2, 2, 4), 255, dtype=np.uint8)
    point = np.array([[6, 5], [6, 0]], dtype=np.uint8)
    dsm = np.array([[0.1, 0.6], [0.3, 0.0]], dtype=np.float32)
    labels = np.array([
        [_label_id(project, "outdoor", "road"), _label_id(project, "outdoor", "low vegetation")],
        [_label_id(project, "outdoor", "road"), _label_id(project, "outdoor", "road")],
    ], dtype=np.uint8)
    confidence = np.array([[0.9, 0.9], [0.4, 0.8]], dtype=np.float16)
    _make_ortho(project, rgba, point, dsm)

    result = segment_ortho(project, level=None, source="combined", labeler_factory=lambda **_: _FakeLabeler(labels, confidence))

    classes = np.load(result.classes_path)
    assert classes[0, 0] == 11 and classes[0, 1] == 5 and classes[1, 0] == 6 and classes[1, 1] == 11


def test_chunk_stitching_has_no_seams(work_dir: Path) -> None:
    project = _project(work_dir)
    height, width = 700, 900
    rgba = np.full((height, width, 4), 255, dtype=np.uint8)
    xs, ys = np.meshgrid(np.arange(width, dtype=np.uint16), np.arange(height, dtype=np.uint16))
    rgba[..., 0] = (xs % 256).astype(np.uint8)
    rgba[..., 1] = (xs // 256).astype(np.uint8)
    rgba[..., 2] = (ys % 256).astype(np.uint8)
    _make_ortho(project, rgba, np.zeros((height, width), dtype=np.uint8))

    class PositionLabeler:
        def label_image_scores(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            absolute_x = image[..., 0].astype(np.uint16) + image[..., 1].astype(np.uint16) * 256
            labels = ((absolute_x + image[..., 2].astype(np.uint16)) % 3 + 1).astype(np.uint8)
            return labels, np.full(labels.shape, 0.9, dtype=np.float16)

    settings = SegmentConfig(chunk_px=256, chunk_overlap_px=32)
    result = segment_ortho(project, level=None, source="sam3", config=settings, labeler_factory=lambda **_: PositionLabeler())
    lookup = np.array([0, 2, 11, 73], dtype=np.uint8)
    expected = lookup[((xs + (ys % 256)) % 3 + 1).astype(np.uint8)]
    assert np.array_equal(np.load(result.classes_path), expected)

    # Second block: varying confidence rising per chunk and differing labels across chunk overlaps
    def independent_feather(y0: int, y1: int, x0: int, x1: int, h: int, w: int, ov: int) -> np.ndarray:
        def axis_weights(start: int, end: int, length: int) -> np.ndarray:
            weights = np.ones(end - start, dtype=np.float32)
            if ov <= 0:
                return weights
            ramp = np.linspace(1.0 / (ov + 1), 1.0, ov, dtype=np.float32)
            if start > 0:
                weights[:min(ov, len(weights))] = ramp[:min(ov, len(weights))]
            if end < length:
                weights[-min(ov, len(weights)):] = ramp[:min(ov, len(weights))][::-1]
            return weights
        return np.outer(axis_weights(y0, y1, h), axis_weights(x0, x1, w)).astype(np.float32)

    def independent_chunk_boxes(w: int, h: int, chunk_px: int, ov: int) -> list[tuple[int, int, int, int]]:
        stride = chunk_px - ov
        def starts(length: int) -> list[int]:
            if length <= chunk_px:
                return [0]
            values = list(range(0, length - chunk_px + 1, stride))
            final = length - chunk_px
            if values[-1] != final:
                values.append(final)
            return values
        return [(x0, y0, min(w, x0 + chunk_px), min(h, y0 + chunk_px)) for y0 in starts(h) for x0 in starts(w)]

    class VaryingConfidenceLabeler:
        def __init__(self) -> None:
            self.calls = 0

        def label_image_scores(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            idx = self.calls
            self.calls += 1
            label = (idx % 3) + 1
            conf = 0.51 + 0.027 * (idx + 1)
            h, w = image.shape[:2]
            return np.full((h, w), label, dtype=np.uint8), np.full((h, w), conf, dtype=np.float16)

    boxes = independent_chunk_boxes(width, height, 256, 32)
    expected_varying_labels = np.zeros((height, width), dtype=np.uint8)
    expected_priority = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.int32)
    single_owner = np.zeros((height, width), dtype=np.uint8)

    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        chunk_lbl = (idx % 3) + 1
        chunk_conf = np.float32(np.float16(0.51 + 0.027 * (idx + 1)))
        coverage[y0:y1, x0:x1] += 1
        single_owner[y0:y1, x0:x1] = chunk_lbl

        feather = independent_feather(y0, y1, x0, x1, height, width, 32)
        candidate_prio = chunk_conf * feather
        better = candidate_prio > expected_priority[y0:y1, x0:x1]

        lbl_region = expected_varying_labels[y0:y1, x0:x1]
        lbl_region[better] = chunk_lbl
        expected_varying_labels[y0:y1, x0:x1] = lbl_region

        prio_region = expected_priority[y0:y1, x0:x1]
        prio_region[better] = candidate_prio[better]
        expected_priority[y0:y1, x0:x1] = prio_region

    expected_varying_codes = lookup[expected_varying_labels]
    single_owner_codes = lookup[single_owner]

    varying_labeler = VaryingConfidenceLabeler()
    varying_result = segment_ortho(
        project, level=None, source="sam3", config=settings,
        labeler_factory=lambda **_: varying_labeler,
    )
    stitched_varying = np.load(varying_result.classes_path)

    overlaps = coverage > 1
    outside_overlaps = coverage == 1

    assert np.any(overlaps), "Image geometry must include chunk overlap regions"
    assert np.all(stitched_varying[outside_overlaps] == single_owner_codes[outside_overlaps])
    assert np.all(stitched_varying[overlaps] == expected_varying_codes[overlaps])
    assert np.array_equal(stitched_varying, expected_varying_codes)


def test_transparent_chunks_skipped(work_dir: Path) -> None:
    project = _project(work_dir)
    rgba = np.zeros((512, 512, 4), dtype=np.uint8)
    rgba[:256, :256] = (20, 30, 40, 255)
    _make_ortho(project, rgba, np.zeros((512, 512), dtype=np.uint8))
    calls: list[np.ndarray] = []

    segment_ortho(project, level=None, source="sam3", config=SegmentConfig(chunk_px=256, chunk_overlap_px=0),
                  labeler_factory=lambda **_: _FakeLabeler(1, calls=calls))

    assert len(calls) == 1


def test_flurseg_rejected_indoor(work_dir: Path) -> None:
    project = _project(work_dir, "indoor")
    level = _level()
    _make_ortho(project, np.full((2, 2, 4), 255, dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8), level=level)

    with pytest.raises(ValueError, match="only available for outdoor"):
        segment_ortho(project, level=level, source="flurseg")


def test_fine_labels_only_inside_furniture_classes(work_dir: Path) -> None:
    project = _project(work_dir, "indoor")
    level = _level()
    rgba = np.full((3, 3, 4), 255, dtype=np.uint8)
    point = np.full((3, 3), 100, dtype=np.uint8)
    point[1, 1] = 117
    point[0, 1] = 120
    point[1, 0] = 121
    point[2, 2] = 102
    _make_ortho(project, rgba, point, level=level)

    def factory(**kwargs: object) -> _FakeLabeler:
        names = kwargs["class_names"]
        if names[0] == "toilet":
            return _FakeLabeler(1, 0.9)
        return _FakeLabeler(_label_id(project, "indoor", "wall"), 0.4)

    result = segment_ortho(project, level=level, source="combined", labeler_factory=factory)

    fine = np.load(result.fine_classes_path)
    assert fine[1, 1] == 1 and fine[0, 1] == 1 and fine[1, 0] == 1
    assert fine[2, 2] == 0 and np.count_nonzero(fine) == 3


def test_load_segment_result_never_computes_and_detects_stale(work_dir: Path) -> None:
    project = _project(work_dir)
    root = _make_ortho(project, np.full((2, 2, 4), 255, dtype=np.uint8), np.ones((2, 2), dtype=np.uint8))
    segment_ortho(project, level=None, source="pointcloud")

    loaded = load_segment_result(project, None, "pointcloud")
    assert loaded is not None and not loaded.is_stale
    root.joinpath("fingerprint.txt").write_text("ortho-b\n", encoding="utf-8")
    stale = load_segment_result(project, None, "pointcloud")
    assert stale is not None and stale.is_stale


def test_fine_prompt_change_marks_segment_stale(work_dir: Path) -> None:
    project = _project(work_dir, "indoor")
    level = _level()
    _make_ortho(project, np.full((3, 3, 4), 255, dtype=np.uint8), np.full((3, 3), 117, dtype=np.uint8), level=level)
    baseline = SegmentConfig(fine_prompts_indoor=("sofa", "bed"), fine_allowed_codes_indoor=(117, 180))
    changed = SegmentConfig(fine_prompts_indoor=("sofa", "bed", "cat tree"), fine_allowed_codes_indoor=(117, 180))
    assert segment_fingerprint(project, level=level, source="combined", config=baseline) != segment_fingerprint(project, level=level, source="combined", config=changed)


@pytest.mark.skipif(not SAM3_MODEL_PATH.is_file(), reason="SAM3 model is missing")
def test_sam3_label_image_scores_matches_label_image(work_dir: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from samcloud.engine.sam_labeler import SemanticLabeler

    image = np.full((900, 1100, 3), (110, 110, 110), dtype=np.uint8)
    image[300:600] = (55, 130, 55)
    image[180:480, 420:760] = (180, 45, 35)
    path = work_dir.parent / f"{work_dir.name}_synthetic.png"
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    labeler = SemanticLabeler(["road", "lawn", "roof"], tile_size=1100)

    path_labels = labeler.label_image(path)
    labels, confidence = labeler.label_image_scores(image)

    assert np.array_equal(labels, path_labels)
    assert confidence.dtype == np.float16 and np.all((confidence[labels > 0] > 0) & (confidence[labels > 0] <= 1))


@pytest.mark.skipif(not SAM3_MODEL_PATH.is_file(), reason="SAM3 model is missing")
def test_sam3_label_image_scores_does_not_create_runs(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from samcloud.engine.sam_labeler import SemanticLabeler

    monkeypatch.chdir(work_dir)
    labeler = SemanticLabeler(["road", "lawn"], tile_size=1008)
    labels, confidence = labeler.label_image_scores(np.full((1008, 1008, 3), 120, dtype=np.uint8))

    assert labels.shape == confidence.shape == (1008, 1008)
    assert not (work_dir / "runs").exists()


@pytest.mark.skipif(not SAM3_MODEL_PATH.is_file(), reason="SAM3 model is missing")
def test_sam3_outdoor_prompt_batches_are_fast() -> None:
    import json
    import time
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from samcloud.engine.sam_labeler import SemanticLabeler

    prompts = json.loads((Path(__file__).resolve().parents[1] / "config" / "classes.json").read_text(encoding="utf-8"))["prompts"]["outdoor"]
    names = [f"class_{index}" for index in range(len(prompts))]
    labeler = SemanticLabeler(names, prompts=prompts, tile_size=1008)
    started = time.perf_counter()
    labels, confidence = labeler.label_image_scores(np.full((1008, 1008, 3), 120, dtype=np.uint8))

    assert labels.shape == confidence.shape == (1008, 1008)
    assert time.perf_counter() - started < 10.0


@pytest.mark.skipif(flurseg_availability() is not None, reason="FlurSeg model is unavailable")
def test_flurseg_label_image_scores_contract() -> None:
    from samcloud.classification.profiles import flurseg_profile
    from samcloud.engine.flurseg_labeler import FlurSegLabeler

    names = [entry.model_name for entry in flurseg_profile("outdoor").entries]
    labels, confidence = FlurSegLabeler(names).label_image_scores(np.full((64, 64, 3), 120, dtype=np.uint8))

    assert labels.shape == confidence.shape == (64, 64)
    assert labels.dtype == np.uint8 and confidence.dtype == np.float16
    assert labels.min() >= 0 and labels.max() <= len(names)


def test_import_has_no_model_side_effects() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import samcloud.ortho.segment, sys; print('ultralytics' in sys.modules)"],
        check=True, capture_output=True, text=True,
    )
    assert completed.stdout.strip() == "False"
