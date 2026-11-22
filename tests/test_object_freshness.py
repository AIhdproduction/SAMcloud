"""Object stages decide "up to date", and a cache older than a reclassification never brings old classes back."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from samcloud.engine import pipeline
from samcloud.engine.processing_cache import MANIFEST, SAM_CLOUD, mark_reclassified, stale_cache_reason
from samcloud.ortho import plan as plan_module


@pytest.mark.parametrize("stage", ["room_objects", "furniture_recon"])
@pytest.mark.parametrize("kind", ["code", "settings"])
def test_object_stage_code_and_settings_change_the_digest(tmp_path, stage, kind):
    package, config = tmp_path / "package", tmp_path / "config"
    (package / "bim" / stage).mkdir(parents=True)
    config.mkdir()
    code, settings = package / "bim" / stage / "detect.py", config / f"{stage}.yaml"
    code.write_text("LIMIT = 1\n", encoding="utf-8")
    settings.write_text("limit: 1\n", encoding="utf-8")
    before = plan_module._object_stage_fingerprint(package, config)
    assert before == plan_module._object_stage_fingerprint(package, config)
    changed = code if kind == "code" else settings
    changed.write_text(changed.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    assert plan_module._object_stage_fingerprint(package, config) != before


def test_object_stages_make_only_the_clean_plan_stale(tmp_path, monkeypatch):
    segment = SimpleNamespace(fingerprint="segment")
    monkeypatch.setattr(plan_module, "_available_clean_plan_settings", lambda *_args: plan_module.CleanPlanSettings())
    monkeypatch.setattr(plan_module, "load_segment_result", lambda *_args: segment)
    monkeypatch.setattr(plan_module, "_object_stage_fingerprint", lambda: "a")
    before = plan_module.clean_plan_fingerprint(tmp_path, None)
    wall_code = plan_module._wall_pipeline_code_fingerprint()
    monkeypatch.setattr(plan_module, "_object_stage_fingerprint", lambda: "b")
    assert plan_module.clean_plan_fingerprint(tmp_path, None) != before
    assert segment.fingerprint == "segment" and plan_module._wall_pipeline_code_fingerprint() == wall_code


@pytest.mark.parametrize("product", ["depth/depth_points.npz", "classified/screen_masks.json.gz", "dense/surface_fusion.json"])
def test_products_of_the_room_objects_count_even_when_depth_is_not_used_in_the_plan(tmp_path, monkeypatch, product):
    monkeypatch.setattr(plan_module, "_available_clean_plan_settings", lambda *_args: plan_module.CleanPlanSettings())
    monkeypatch.setattr(plan_module, "load_segment_result", lambda *_args: SimpleNamespace(fingerprint="segment"))
    monkeypatch.setattr(plan_module, "plan_depth_fingerprint", lambda _project: None)  # depth_evidence.use_in_plan: false
    before = plan_module.clean_plan_fingerprint(tmp_path, None)
    path = tmp_path / "products" / product
    path.parent.mkdir(parents=True)
    path.write_bytes(b"first")
    written = plan_module.clean_plan_fingerprint(tmp_path, None)
    assert written != before
    path.write_bytes(b"second version")
    assert plan_module.clean_plan_fingerprint(tmp_path, None) != written


def test_orthophoto_and_segmentation_do_not_read_the_object_stages():
    from samcloud.ortho import build, segment

    for module in (build, segment):
        assert not hasattr(module, "_object_stage_fingerprint")


def _cache(project):
    cache = project / "processing_cache"
    (cache / "dense" / "stereo" / "depth_maps").mkdir(parents=True)
    (cache / "dense" / "sparse").mkdir()
    (cache / SAM_CLOUD).write_bytes(b"old classes")
    (cache / MANIFEST).write_text(json.dumps({"class_set": "indoor", "created_at": "then"}), encoding="utf-8")
    return cache


def test_refine_from_a_cache_older_than_a_reclassify_refuses_and_changes_nothing(tmp_path):
    cache = _cache(tmp_path)
    assert stale_cache_reason(tmp_path) is None
    cached = {path: path.read_bytes() for path in cache.rglob("*") if path.is_file()}
    assert mark_reclassified(tmp_path)
    output = tmp_path / "products" / "classified" / "classified.ply"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"new classes")
    args = SimpleNamespace(class_set="indoor", geometry_refinement="1")
    with pytest.raises(SystemExit, match="reclassified.*old classes"):
        pipeline._refine_from_cache(args, tmp_path / "products", tmp_path / "reports", SimpleNamespace(), None, output)
    assert output.read_bytes() == b"new classes"
    assert all(path.read_bytes() == data for path, data in cached.items())


def test_without_a_cache_nothing_is_marked(tmp_path):
    assert not mark_reclassified(tmp_path)
    assert not (tmp_path / "processing_cache").exists()


@pytest.mark.parametrize("cache_after_sam", [False, True])
def test_only_a_reclassify_marks_the_cache(tmp_path, monkeypatch, cache_after_sam):
    _cache(tmp_path)
    for name in (
        "_plan_geometry_refinement", "_ensure_if_compressed", "run", "_discard_stale_instances", "_snapshot_instances_safely",
        "_cache_after_sam", "_refine_after_classification", "_finish_classification",
    ):
        monkeypatch.setattr(pipeline, name, lambda *_args, **_kwargs: None)
    for name in ("_labels_arguments", "_instance_arguments"):
        monkeypatch.setattr(pipeline, name, lambda *_args, **_kwargs: [])
    monkeypatch.setattr(pipeline, "_writes_scores", lambda _args: False)
    monkeypatch.setattr(pipeline, "_classification_engine", lambda _args: "sam3")
    args = SimpleNamespace(
        geometry_refinement="1", classes_json="classes.json", class_set="indoor", min_votes=1, sam_tile_size=1008,
        sam_tile_min_overlap=0.2, classification_mode="segearth_ov3", background_threshold=0.4, query_threshold=0.4,
        threshold_sweep="", progress_file=None,
    )
    products = tmp_path / "products"
    pipeline._classify_and_finish(
        args, products, products / "dense", products / "sparse_txt", tmp_path / "reports", SimpleNamespace(), None,
        products / "classified" / "classified.ply", False, cache_after_sam=cache_after_sam,
    )
    assert (stale_cache_reason(tmp_path) is not None) is (not cache_after_sam)


def test_a_cache_note_that_cannot_be_written_warns_and_the_reclassify_goes_on(tmp_path, monkeypatch, capsys):
    cache = _cache(tmp_path)

    def broken(*_args, **_kwargs):
        raise OSError("read-only cache")

    with monkeypatch.context() as patch:
        patch.setattr(type(cache), "write_text", broken)
        assert mark_reclassified(tmp_path) is False
    warning = [line for line in capsys.readouterr().out.splitlines() if "WARNING" in line]
    assert len(warning) == 1 and "read-only cache" in warning[0]
