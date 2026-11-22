"""Per-point fine SAM terms: votes, aliases, hand-over through step 6, product PLY and vocabulary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import laspy
import numpy as np
import pytest
import torch

from samcloud.engine import classify
from samcloud.engine.export_las import _ply_chunks, export_las, read_ply_header
from samcloud.engine.fine_terms import (
    FineVocabulary,
    attach_fine_terms,
    build_fine_vocabulary,
    canonical_label,
    snapshot_fine_terms,
)
from samcloud.engine.ply_io import read_vertex_array, write_classified_ply
from samcloud.engine.sam_labeler import HeadFusionLabeler

CLASSES_JSON = Path(__file__).resolve().parents[1] / "config" / "classes.json"
CONFIG = json.loads(CLASSES_JSON.read_text(encoding="utf-8"))

SMALL_CONFIG = {
    "las_codes": {"indoor": [1, 180, 117]},
    "fine_terms": {"indoor": ["sofa", "couch", "bed"]},
    "fine_aliases": {"couch": "sofa"},
    "fine_term_classes": {"indoor": {"sofa": 117, "couch": 117, "bed": 117}},
}
SMALL_PROMPTS = [["bag"], ["furniture", "sofa", "couch", "bed"]]  # flat: bag, furniture, sofa, couch, bed


def _fake_predictor(counter: list[int]):
    """SAM stand-in: left half of the tile looks like a sofa, right half like a bed (mask resolution 4 x 4)."""

    def predictor(source=None, text=None, verbose=False):
        counter.append(1)
        logits = torch.full((5, 4, 4), -9.0)
        logits[1] = 9.0  # furniture everywhere
        logits[2, :, :2] = 9.0  # sofa left
        logits[3, :, :2] = 6.0  # couch left (weaker, same term)
        logits[4, :, 2:] = 9.0  # bed right
        result = SimpleNamespace(
            semantic_seg=logits, inst_agg=torch.zeros_like(logits), presence_scores=torch.ones(5),
            raw_shapes={}, retained_query_counts=None,
        )
        return [result]

    return predictor


def _head_labeler(counter: list[int], with_fine: bool) -> HeadFusionLabeler:
    labeler = HeadFusionLabeler.__new__(HeadFusionLabeler)
    labeler.class_names = ["clutter", "furniture"]
    labeler._flat_prompts = ["bag", "furniture", "sofa", "couch", "bed"]
    labeler._prompt_owner = np.array([0, 1, 1, 1, 1])
    labeler.fine_queries = [[2, 3], [4]] if with_fine else None
    labeler.last_fine_probs = None
    labeler.last_tile_diagnostics = None
    labeler.tile_size = 1008
    labeler.min_overlap_frac = 0.2
    labeler.predictor = _fake_predictor(counter)
    return labeler


class _VoteLabeler:
    """Point scorer with per-view noise: points 0-2 are a sofa, 3-5 a bed; class = furniture for all."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def load_image(_path):
        return np.zeros((8, 8, 3), dtype=np.uint8), 8, 8

    def score_points(self, _path, _u, _v, fine_scores_out=None, **_kwargs):
        noise = self.rng.normal(0.0, 0.05, (6, 2)).astype(np.float32)
        probs = np.zeros((6, 2), dtype=np.float32)
        probs[:, 1] = 0.9
        if fine_scores_out is not None:
            terms = np.zeros((6, 2), dtype=np.float32)
            terms[:3, 0], terms[:3, 1] = 0.9, 0.1  # sofa high on A
            terms[3:, 0], terms[3:, 1] = 0.15, 0.85  # bed high on B
            fine_scores_out[:] = np.clip(terms + noise, 0.0, 1.0)
        return probs


def _run_votes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, with_fine: bool) -> dict:
    labeler = _VoteLabeler(seed=7)
    monkeypatch.setattr(classify, "make_point_labeler", lambda *_a, **_k: labeler)
    monkeypatch.setattr(classify, "project_points", lambda *_a, **_k: (
        np.arange(6), np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64), np.full(6, 2.0),
    ))
    args = SimpleNamespace(
        background_threshold=0.4, threshold_sweep="", dense_dir=str(tmp_path), labels_dir=None, images_dir=str(tmp_path),
        depth_check=0, out=str(tmp_path / f"{name}.ply"), scores_out=str(tmp_path / f"{name}.npz"), min_votes=1,
        summary_file=None, class_set="indoor", engine="sam3", mode="segearth_ov3", query_threshold=0.4,
    )
    xyz = np.zeros((6, 3))
    items = [(view, [0, 1, 2, 3, 4, 5]) for view in (1, 2, 3)]
    images = {view: SimpleNamespace(name=f"view{view}.png", camera_id=1) for view in (1, 2, 3)}
    classify.run_segearth_ov3(
        args, xyz, np.zeros((6, 3), dtype=np.uint8), {1: object()}, images, items, ["clutter", "furniture"], 0, None,
        prompts=SMALL_PROMPTS, fine_config=SMALL_CONFIG if with_fine else None,
    )
    with np.load(args.scores_out) as archive:
        return {key: archive[key] for key in archive.files}


def test_fine_term_votes_pick_winner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. three noisy views through the real run_segearth_ov3 vote with a mocked point scorer
    with_fine = _run_votes(tmp_path, monkeypatch, "with", True)
    without = _run_votes(tmp_path, monkeypatch, "without", False)

    assert with_fine["fine_labels"].tolist() == ["sofa", "bed"]
    assert with_fine["fine_term"].tolist() == [1, 1, 1, 2, 2, 2]
    assert with_fine["fine_term"].dtype == np.uint8 and with_fine["fine_conf"].dtype == np.float16
    assert float(with_fine["fine_conf"].min()) >= 0.4
    assert "fine_term" not in without
    for key in without:  # class results do not change by a single bit
        np.testing.assert_array_equal(with_fine[key], without[key], err_msg=key)

    # 2. the real tile scorer: one SAM call per tile with or without terms, class probabilities identical
    u, v = np.array([1, 14]), np.array([8, 8])
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    plain_calls: list[int] = []
    fine_calls: list[int] = []
    plain = _head_labeler(plain_calls, with_fine=False).score_points("x.png", u, v, image=image)
    labeler = _head_labeler(fine_calls, with_fine=True)
    terms = np.zeros((2, 2), dtype=np.float32)
    fine = labeler.score_points("x.png", u, v, image=image, fine_scores_out=terms)

    assert len(plain_calls) == len(fine_calls) == 1
    np.testing.assert_array_equal(plain, fine)
    assert terms[0, 0] > 0.9 > 0.5 > terms[0, 1] and terms[1, 1] > 0.9 > 0.5 > terms[1, 0]


def test_fine_term_alias_canonical() -> None:
    aliases = CONFIG["fine_aliases"]

    assert canonical_label("couch", aliases) == "sofa"
    assert canonical_label("pillow", aliases) == "cushion"
    assert canonical_label("Sofa", aliases) == "sofa"
    assert canonical_label("armchair", aliases) == "armchair"

    vocabulary = build_fine_vocabulary(CONFIG, "indoor", CONFIG["prompts"]["indoor"])
    assert vocabulary is not None
    assert "couch" not in vocabulary.labels and "pillow" not in vocabulary.labels
    assert len(set(vocabulary.labels)) == len(vocabulary.labels) == len(CONFIG["fine_terms"]["indoor"]) - len(aliases)
    sofa = vocabulary.labels.index("sofa")
    assert len(vocabulary.queries[sofa]) == 2  # sofa and couch are both queries of the one label
    assert vocabulary.owner_class[sofa] == 17  # furniture
    assert build_fine_vocabulary(CONFIG, "outdoor", CONFIG["prompts"]["outdoor"]) is None


def _write_step6_inputs(tmp_path: Path):
    """SAM-stage cloud (6 points) with scores + fine terms, and the cloud step 6 leaves behind."""
    raw = tmp_path / "classified_raw.ply"
    scores = tmp_path / "classified_scores.npz"
    xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]], dtype=np.float32)
    classes = np.array([17, 17, 17, 2, 17, 17], dtype=np.uint8)
    # labels sofa=1, bed=2; point 3 is a wall carrying a sofa term that its class cannot own
    fine = np.array([1, 2, 1, 1, 1, 0], dtype=np.uint8)
    write_classified_ply(raw, xyz, np.full((6, 3), 120, dtype=np.uint8), classes)
    classify.save_score_summary(
        scores, np.ones((6, 3), dtype=np.float32), np.ones(6, dtype=np.uint16), classes,
        fine_term=fine, fine_conf=np.full(6, 0.8, dtype=np.float16), fine_labels=["sofa", "bed"],
    )
    return raw, scores, xyz, classes


def _step6_cloud(tmp_path: Path, xyz: np.ndarray) -> Path:
    """What step 6 might do: drop point 4, relabel point 2 (117 -> 100), add points near 0 and 1 and 5."""
    final_xyz = np.vstack([
        xyz[[0, 1, 2, 3, 5]],
        xyz[0] + [0.03, 0.0, 0.0],  # new, next to the sofa point, same class       -> sofa
        xyz[1] + [0.20, 0.0, 0.0],  # new, too far from every source point         -> none
        xyz[0] + [0.01, 0.0, 0.0],  # new, next to the sofa point, but now a wall  -> none
    ]).astype(np.float32)
    final_class = np.array([17, 17, 2, 2, 17, 17, 17, 2], dtype=np.uint8)
    final = tmp_path / "classified.ply"
    write_classified_ply(final, final_xyz, np.full((len(final_xyz), 3), 90, dtype=np.uint8), final_class)
    return final


def test_fine_term_survives_refinement(tmp_path: Path) -> None:
    raw, scores, xyz, _classes = _write_step6_inputs(tmp_path)

    assert snapshot_fine_terms(raw, scores)
    with np.load(scores) as archive:  # step 6 sees only per-point arrays of the cloud's length
        assert not {"fine_term", "fine_conf", "fine_labels"} & set(archive.files)
        assert all(len(archive[name]) == 6 for name in archive.files)
    source = scores.with_name("fine_terms_source.npz")
    assert source.is_file()

    final = _step6_cloud(tmp_path, xyz)
    report = attach_fine_terms(source, final, CLASSES_JSON, "indoor")

    term = np.asarray(read_vertex_array(final)["fine_term"])
    assert term.tolist() == [
        1,  # survivor keeps sofa
        2,  # survivor keeps bed
        0,  # class changed 117 -> 100: a wall cannot be a bed
        0,  # wall with a sofa term of the SAM stage: class does not own it
        0,  # source point had no term
        1,  # added point within 0.05 m of a sofa point, same class -> neighbour's term
        0,  # added point without a neighbour within 0.05 m
        0,  # added point next to the sofa point but of another class
    ]
    assert report["labelled_points"] == 3 and report["points"] == 8
    assert not source.exists()


def test_fine_term_written_to_product_ply(tmp_path: Path) -> None:
    raw, scores, xyz, _classes = _write_step6_inputs(tmp_path)
    assert snapshot_fine_terms(raw, scores)
    final = _step6_cloud(tmp_path, xyz)
    attach_fine_terms(scores.with_name("fine_terms_source.npz"), final, CLASSES_JSON, "indoor")

    header = final.read_bytes()[:400].decode("latin-1")
    assert "property uchar fine_term" in header and "property uchar scalar_class" in header
    vertex = read_vertex_array(final)
    assert vertex["fine_term"].dtype == np.uint8 and len(vertex) == 8

    vocabulary = json.loads((tmp_path / "fine_labels.json").read_text(encoding="utf-8"))
    assert vocabulary["labels"] == ["sofa", "bed"]
    assert vocabulary["ids"] == {"sofa": 1, "bed": 2}
    assert vocabulary["owner_las_code"] == {"sofa": 117, "bed": 117}
    assert vocabulary["property"] == "fine_term" and vocabulary["none"] == 0
    assert set(np.unique(vertex["fine_term"]).tolist()) <= set(vocabulary["ids"].values()) | {0}

    # the PLY readers of the LAS export / DXF stay valid with the extra property ...
    offset, count = read_ply_header(final)
    with final.open("rb") as handle:
        chunks = list(_ply_chunks(handle, offset, count, 3))
    assert sum(len(chunk) for chunk in chunks) == 8
    assert np.concatenate([chunk["fine_term"] for chunk in chunks]).tolist() == vertex["fine_term"].tolist()
    assert np.concatenate([chunk["scalar_class"] for chunk in chunks]).tolist() == vertex["scalar_class"].tolist()

    # ... and the LAS export carries it as an extra dimension
    export_las(final, tmp_path / "classified.las", CLASSES_JSON, 5, "LOCAL", "LOCAL", "indoor")
    las = laspy.read(tmp_path / "classified.las")
    assert np.asarray(las.fine_term).tolist() == vertex["fine_term"].tolist()


def test_classes_json_fine_terms_are_queries() -> None:
    indoor_terms = {
        str(term).strip().lower()
        for group in CONFIG["prompts"]["indoor"]
        for term in ([group] if isinstance(group, str) else group)
    }
    fine = CONFIG["fine_terms"]["indoor"]
    assert fine[:36] == [  # persisted fine_term values follow this order: new terms are only ever appended
        "sofa", "couch", "armchair", "bed", "nightstand", "table", "coffee table", "dining table", "desk", "chair",
        "wardrobe", "chest of drawers", "shelf", "bookcase", "sideboard", "tv stand", "television", "cat tree", "plant",
        "floor lamp", "cushion", "pillow", "blanket", "mattress", "clothes", "refrigerator", "washing machine",
        "kitchen counter", "kitchen cabinet", "stove", "kitchen sink", "toilet", "washbasin", "bathtub", "shower tray",
        "radiator",
    ]
    assert len(fine) == len(set(fine)) > 36
    for term in [*fine, *CONFIG["fine_aliases"]]:
        assert term in indoor_terms, f"fine term '{term}' is not a SAM query"
    for alias, label in CONFIG["fine_aliases"].items():
        assert label in indoor_terms and alias in fine

    classes = CONFIG["prompts"]["indoor"]
    codes = CONFIG["las_codes"]["indoor"]
    for term in fine:  # every term has an owner class that really queries it
        owner = CONFIG["fine_term_classes"]["indoor"][term]
        group = classes[codes.index(owner) - 1]
        assert term in [str(item).strip().lower() for item in group], f"{term} is not a query of class {owner}"
    assert CONFIG["fine_terms"]["outdoor"] == []
    assert FineVocabulary  # public type stays importable
