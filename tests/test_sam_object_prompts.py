"""Screens and ceiling fixtures reach SAM3 on every start path; reclassify staging stays on the cache's volume."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from samcloud.classification.custom import (
    build_custom_classes,
    global_classes_path,
    load_project_classes,
    project_classes_path,
    upgrade_classes_config,
)
from samcloud.core.project_manager import ProjectManager
from samcloud.engine import classify, pipeline
from samcloud.engine.fine_terms import build_fine_vocabulary, label_owner_classes
from samcloud.engine.instances import _family_votes, _metadata
from samcloud.ortho.segment import _FINE_PROMPTS_V1, SegmentConfig, _fingerprint_settings, load_segment_config

GLOBAL = json.loads(global_classes_path().read_text(encoding="utf-8"))
SCREEN_PROMPTS = {"television", "flat screen tv", "flat screen", "computer monitor", "computer screen", "screen", "laptop"}
CEILING_PROMPTS = {"ceiling light", "pendant light", "spotlight", "smoke detector", "ceiling projector", "track light"}


def _prompts(config: dict, name: str) -> list[str]:
    names = [entry.split(",")[0].strip() for entry in config["indoor"]]
    return config["prompts"]["indoor"][names.index(name)]


def _revision_one() -> dict:
    """A classes file as it was written before screens and ceiling fixtures were prompted."""
    config = copy.deepcopy(GLOBAL)
    added = config.pop("class_upgrades")[0]["indoor"]
    config.pop("classes_revision")
    config.pop("instance_label_votes")
    for name in ("furniture", "ceiling equipment"):
        new = set(added["prompts"][name]["add"]) - {"television"}
        index = [entry.split(",")[0] for entry in config["indoor"]].index(name)
        values = [item for item in config["prompts"]["indoor"][index] if item not in new]
        if name == "ceiling equipment":  # "ceiling light" was the generic first prompt
            values.remove("ceiling light")
            values.insert(0, "ceiling light")
        config["prompts"]["indoor"][index] = values
        config["indoor"][index] = ", ".join([name, *values])
    config["instance_classes"]["indoor"].remove("ceiling equipment")
    legacy_terms = config["fine_terms"]["indoor"][:36]
    config["fine_terms"]["indoor"] = legacy_terms
    owners = config["fine_term_classes"]["indoor"]
    config["fine_term_classes"]["indoor"] = {term: owners[term] for term in legacy_terms}
    config["fine_aliases"] = {"couch": "sofa", "pillow": "cushion"}
    return config


def test_global_prompts_and_class_entries_agree() -> None:
    assert SCREEN_PROMPTS <= set(_prompts(GLOBAL, "furniture"))
    assert CEILING_PROMPTS <= set(_prompts(GLOBAL, "ceiling equipment"))
    assert _prompts(GLOBAL, "ceiling equipment")[0] == "ceiling mounted fixture"  # generic: never names an instance
    assert "projector" not in _prompts(GLOBAL, "ceiling equipment")  # the bare word fired on floor devices
    assert "ceiling equipment" in GLOBAL["instance_classes"]["indoor"]
    for entry, prompts in zip(GLOBAL["indoor"], GLOBAL["prompts"]["indoor"], strict=True):
        assert entry == ", ".join([entry.split(",")[0], *prompts])


def test_old_project_file_gets_the_same_prompts_and_vocabulary() -> None:
    old = _revision_one()
    before = copy.deepcopy(old)
    upgraded = upgrade_classes_config(old)
    assert old == before  # in memory only
    assert upgraded["classes_revision"] == GLOBAL["classes_revision"]
    for name in ("furniture", "ceiling equipment"):
        assert sorted(_prompts(upgraded, name)) == sorted(_prompts(GLOBAL, name))
        assert _prompts(upgraded, name)[0] == _prompts(GLOBAL, name)[0]
    assert upgraded["instance_classes"]["indoor"] == GLOBAL["instance_classes"]["indoor"]
    assert upgraded["las_codes"] == GLOBAL["las_codes"]
    assert upgraded["outdoor"] == GLOBAL["outdoor"] and upgraded["prompts"]["outdoor"] == GLOBAL["prompts"]["outdoor"]
    new = build_fine_vocabulary(upgraded, "indoor", upgraded["prompts"]["indoor"], warn=pytest.fail)
    reference = build_fine_vocabulary(GLOBAL, "indoor", GLOBAL["prompts"]["indoor"], warn=pytest.fail)
    assert new.labels == reference.labels


def test_persisted_fine_term_values_decode_unchanged() -> None:
    old = _revision_one()
    before = build_fine_vocabulary(old, "indoor", old["prompts"]["indoor"], warn=pytest.fail)
    now = build_fine_vocabulary(GLOBAL, "indoor", GLOBAL["prompts"]["indoor"], warn=pytest.fail)
    assert now.labels[:len(before.labels)] == before.labels
    assert label_owner_classes(GLOBAL, "indoor", before.labels).tolist() == before.owner_class.tolist()


def test_upgrade_keeps_edits_own_classes_and_ids() -> None:
    old = build_custom_classes(
        "indoor", {"prompts": {"furniture": ["table", "chair"]},
                   "added": [{"name": "display wall", "prompts": ["screen", "video wall"]}]}, _revision_one(),
    )
    # a file written back then has neither the revision nor the additions
    old.pop("classes_revision")
    index = [entry.split(",")[0] for entry in old["indoor"]].index("furniture")
    old["prompts"]["indoor"][index] = ["table", "chair"]
    upgraded = upgrade_classes_config(old)
    furniture = _prompts(upgraded, "furniture")
    assert furniture[:2] == ["table", "chair"]
    assert "television" in furniture and "computer monitor" in furniture
    assert "screen" not in furniture  # the own class already asks for it
    assert _prompts(upgraded, "display wall") == ["screen", "video wall"]
    assert [entry.split(",")[0] for entry in upgraded["indoor"]] == [entry.split(",")[0] for entry in old["indoor"]]
    assert upgraded["las_codes"] == old["las_codes"]


def test_current_revision_file_keeps_removed_prompts() -> None:
    edited = build_custom_classes("indoor", {"prompts": {"furniture": ["table", "chair"]}})
    assert _prompts(upgrade_classes_config(edited), "furniture") == ["table", "chair"]


def test_unrelated_class_list_is_left_alone() -> None:
    other = {"indoor": ["wall", "furniture"], "prompts": {"indoor": [["wall"], ["table"]]}}
    assert upgrade_classes_config(other) == other


def test_new_project_and_project_classes_path_get_screens(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")
    plain = manager.create_project("Plain", "D:/images", capture_type="camera", class_set="indoor")
    own = manager.create_project("Own", "D:/images", capture_type="camera", class_set="indoor",
                                 custom_classes={"added": [{"name": "piano"}]})
    for directory in (plain, own):
        assert SCREEN_PROMPTS <= set(_prompts(load_project_classes(directory), "furniture"))


def test_classify_cli_upgrades_a_customised_project_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "project"
    (directory / "config").mkdir(parents=True)
    (directory / "config" / "classes.json").write_text(json.dumps(_revision_one()), encoding="utf-8")
    seen: dict = {}
    monkeypatch.setattr(classify, "read_fused_points", lambda _path: (np.zeros((1, 3)), np.zeros((1, 3), np.uint8)))
    monkeypatch.setattr(classify, "read_vis_file", lambda _path: [[]])
    monkeypatch.setattr(classify, "load_model", lambda _path: ({}, {}))
    monkeypatch.setattr(classify, "mvs_image_order", lambda *_args: [])
    monkeypatch.setattr(classify, "build_image_to_points_index", lambda *_args: {})
    monkeypatch.setattr(classify, "run_segearth_ov3", lambda *args, **kwargs: seen.update(kwargs, names=args[6]))
    monkeypatch.setattr(sys, "argv", [
        "classify.py", "--dense-dir", str(tmp_path), "--images-dir", str(tmp_path), "--class-set", "indoor",
        "--classes-json", str(project_classes_path(directory)), "--out", str(tmp_path / "out.ply"),
    ])

    classify.main()

    config = seen["fine_config"]
    assert SCREEN_PROMPTS <= set(_prompts(config, "furniture"))
    assert seen["prompts"] == config["prompts"]["indoor"] and seen["names"] == config["indoor"]
    assert "ceiling equipment" in config["instance_classes"]["indoor"]


def test_synonyms_of_one_screen_outvote_the_carrier() -> None:
    labels = ["television", "computer monitor", "computer screen", "flat screen", "screen", "desk", "desk", "desk"]
    votes = _family_votes(labels, [1, 1, 1, 1, 1, 1, 2, 3])
    assert votes == ["monitor"] * 5 + ["desk"] * 3  # one view each: the more precise name wins the tie


def test_weak_names_only_win_alone_and_other_labels_stay() -> None:
    assert _family_votes(["screen", "screen", "television"], [1, 2, 1]) == ["television"] * 3
    assert _family_votes(["flat screen", "screen"], [1, 2]) == ["screen"] * 2
    assert _family_votes(["bedding", "duvet", "bed"], [1, 1, 1]) == ["bedding", "duvet", "bed"]


def test_lamp_with_a_single_detector_vote_stays_a_lamp() -> None:
    texts = ["ceiling mounted fixture", "ceiling light", "ceiling spotlight", "smoke detector"]
    prompt = [0, 1, 2, 0, 1, 2, 0, 1, 3]
    view = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    record = _metadata(np.ones(4, int), np.random.default_rng(3).normal(0, 0.03, (4, 3)), {1: list(range(9))},
                       [16] * 9, [0.9] * 9, prompt, view, texts, {0}, {})[0]
    assert record["label"] == "ceiling lamp"


def test_segment_fine_prompts_do_not_invalidate_outdoor() -> None:
    settings = load_segment_config()
    assert settings.fine_prompts_indoor == SegmentConfig().fine_prompts_indoor  # YAML and fallback agree
    assert settings.fine_prompts_indoor[:len(_FINE_PROMPTS_V1)] == _FINE_PROMPTS_V1
    assert {"flat screen", "computer monitor", "laptop"} <= set(settings.fine_prompts_indoor)
    assert _fingerprint_settings(settings, "outdoor")["fine_prompts_indoor"] == _FINE_PROMPTS_V1
    assert _fingerprint_settings(settings, "indoor")["fine_prompts_indoor"] == settings.fine_prompts_indoor


def _cache(tmp_path: Path) -> Path:
    dense = tmp_path / "project" / "processing_cache" / "dense"
    (dense / "stereo" / "depth_maps").mkdir(parents=True)
    (dense / "fused.ply").write_bytes(b"cloud")
    (dense / "stereo" / "depth_maps" / "a.geometric.bin").write_bytes(b"depth")
    (dense / "depth_prior.json").write_text("{}", encoding="utf-8")
    return dense


def test_staging_uses_the_cache_volume_and_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dense = _cache(tmp_path)
    monkeypatch.setattr(pipeline.shutil, "copy2", lambda *_args: pytest.fail("copied instead of linked"))

    staging = pipeline._stage_reclassify_cache(dense)

    assert staging.parent.parent == dense.parent.parent  # step 6 resolves the project from here
    assert (staging / "fused.ply").samefile(dense / "fused.ply")
    assert (staging / "stereo" / "depth_maps" / "a.geometric.bin").samefile(dense / "stereo" / "depth_maps" / "a.geometric.bin")
    assert not (staging / "depth_prior.json").exists()


def test_staging_refuses_a_copy_that_does_not_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dense = _cache(tmp_path)

    def no_link(*_args):
        raise OSError("different volume")

    monkeypatch.setattr(pipeline.os, "link", no_link)
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda _path: SimpleNamespace(free=pipeline.RECLASSIFY_RESERVE_BYTES))
    monkeypatch.setattr(pipeline.shutil, "copy2", lambda *_args: pytest.fail("copy must not start"))
    with pytest.raises(ValueError, match="only 5.0 GB are free"):
        pipeline._stage_reclassify_cache(dense)


def test_staging_copies_when_it_fits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dense = _cache(tmp_path)

    def no_link(*_args):
        raise OSError("different volume")

    monkeypatch.setattr(pipeline.os, "link", no_link)
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda _path: SimpleNamespace(free=2 * pipeline.RECLASSIFY_RESERVE_BYTES))
    staging = pipeline._stage_reclassify_cache(dense)
    assert (staging / "fused.ply").read_bytes() == b"cloud"


def test_reclassify_from_staging_never_releases_linked_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dense = _cache(tmp_path)
    project = tmp_path / "project"
    (dense / "fused.ply.vis").write_bytes(b"vis")
    for name in ("sparse", "images", "sparse_txt"):
        (dense / name).mkdir()
        (dense / name / "input.bin").write_bytes(b"x")
    seen: dict = {}
    monkeypatch.setattr(pipeline, "ensure_text_model", lambda _binary, text: text)
    monkeypatch.setattr(pipeline, "_is_georeferenced", lambda _outdir: True)
    monkeypatch.setattr(pipeline, "_classify_and_finish", lambda *args, **kwargs: seen.update(kwargs, dense=args[2]))
    monkeypatch.setitem(sys.modules, "pycolmap", SimpleNamespace())
    args = SimpleNamespace(images=None, panorama_workdir=None)
    pipeline._reclassify_project(args, project / "products", project / "reports", SimpleNamespace(settings={}), None,
                                 project / "classified.ply")
    assert seen["keep_caches"] is True and seen["cache_after_sam"] is False
    assert seen["dense"].parent.parent == project
