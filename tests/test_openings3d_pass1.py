"""Pass-1 snapshot handling of the clean plan (seed of the opening refinement)."""

from __future__ import annotations

import json
from pathlib import Path

from openings3d_synth import make_plan
from samcloud.bim.openings3d.model import OpeningsResult, save_openings_3d
from samcloud.bim.plan_model import pass1_plan_path, plan_model_path, save_plan_model
from samcloud.ortho import plan as plan_module


def _write_plan(directory: Path, length: float, fingerprint: str) -> None:
    plan = make_plan(length)
    plan.source = {"result_fingerprint": fingerprint}
    save_plan_model(plan, directory)


def _snapshot_length(directory: Path) -> float:
    return json.loads(pass1_plan_path(directory).read_text())["walls"][0]["centreline"][1][0]


def test_snapshot_is_written_when_no_refined_openings_were_applied(tmp_path: Path) -> None:
    _write_plan(tmp_path, 4.0, "fp1")
    plan_module._store_pass1_snapshot(tmp_path, {}, "fp1")
    assert _snapshot_length(tmp_path) == 4.0
    _write_plan(tmp_path, 4.5, "fp1")
    plan_module._store_pass1_snapshot(tmp_path, {"sha256": "x", "applied": False}, "fp1")
    assert _snapshot_length(tmp_path) == 4.5  # a pass without applied openings replaces the snapshot


def test_applied_pass_keeps_the_snapshot_unless_the_fingerprint_changed(tmp_path: Path) -> None:
    _write_plan(tmp_path, 4.0, "fp1")
    plan_module._store_pass1_snapshot(tmp_path, {}, "fp1")
    _write_plan(tmp_path, 4.2, "fp1")
    plan_module._store_pass1_snapshot(tmp_path, {"sha256": "x", "applied": True}, "fp1")
    assert _snapshot_length(tmp_path) == 4.0
    _write_plan(tmp_path, 4.2, "fp2")  # inputs/settings changed: the old snapshot is stale
    plan_module._store_pass1_snapshot(tmp_path, {"sha256": "x", "applied": True}, "fp2")
    assert _snapshot_length(tmp_path) == 4.2
    assert plan_model_path(tmp_path).is_file()


def test_applied_pass_without_snapshot_creates_one(tmp_path: Path) -> None:
    _write_plan(tmp_path, 4.0, "fp1")
    plan_module._store_pass1_snapshot(tmp_path, {"sha256": "x", "applied": True}, "fp1")
    assert pass1_plan_path(tmp_path).is_file()


def test_legacy_cache_without_snapshot_skips_the_hook_and_forces_a_replan(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from samcloud.bim.openings3d.config import config_hash, load_openings_config
    from samcloud.bim.openings3d.model import openings_3d_sha256
    from samcloud.bim.openings3d.walls import walls_fingerprint, wall_lines
    from samcloud.bim.openings3d.pipeline import plan_has_current_openings
    from samcloud.ortho.vector.indoor import Opening

    level = SimpleNamespace(id="level_1")
    directory = plan_module.clean_plan_cache_dir(tmp_path, level)
    config = load_openings_config()
    plan = make_plan(4.0)
    plan.source = {"result_fingerprint": "fp"}
    save_plan_model(plan, directory)
    save_openings_3d(OpeningsResult(openings=[], inputs={"walls": walls_fingerprint(wall_lines(plan, config)), "config_hash": config_hash(config)}), directory)
    assert openings_3d_sha256(directory) is not None and not pass1_plan_path(directory).is_file()
    two_d = [Opening("WINDOW", (2.0, 0.0), 1.0, 0.0, 0.25)]
    result, info = plan_module._apply_refined_openings(two_d, [], tmp_path, level, 1.0, "fp")
    assert result == two_d and info["applied"] is False and info["sha256"] == ""
    # the plan pass records the empty sha: run_all sees "plan lacks the current openings" and re-plans once the snapshot exists
    plan.source = {"result_fingerprint": "fp", "openings_3d": info}
    save_plan_model(plan, directory)
    assert plan_has_current_openings(tmp_path, "level_1") is False


def _prepare_refined_cache(tmp_path: Path, snapshot_source: dict | None, plan_fingerprint: str = "fp"):
    """Cache with a plan, a matching openings_3d.json (one door) and optionally a pass-1 snapshot with the given source."""
    from types import SimpleNamespace

    from samcloud.bim.openings3d.config import config_hash, load_openings_config
    from samcloud.bim.openings3d.model import RefinedOpening
    from samcloud.bim.openings3d.walls import walls_fingerprint, wall_lines

    level = SimpleNamespace(id="level_1")
    directory = plan_module.clean_plan_cache_dir(tmp_path, level)
    config = load_openings_config()
    plan = make_plan(4.0)
    plan.source = {"result_fingerprint": plan_fingerprint}
    save_plan_model(plan, directory)
    door = RefinedOpening(id="d1", kind="door", center=[2.0, 0.0], width_m=0.9, axis_angle=0.0, wall_thickness_m=0.25, host_wall_id=None, sill_m=0.0, head_m=2.1, height_source="photo")
    save_openings_3d(OpeningsResult(openings=[door], inputs={"walls": walls_fingerprint(wall_lines(plan, config)), "config_hash": config_hash(config)}), directory)
    if snapshot_source is not None:
        plan.source = snapshot_source
        save_plan_model(plan, directory)
        from samcloud.bim.plan_model import save_pass1_snapshot

        save_pass1_snapshot(directory)
    return level, directory


def test_hook_applies_refined_openings_only_with_a_pure_snapshot_of_the_same_fingerprint(tmp_path: Path) -> None:
    from samcloud.ortho.vector.indoor import Opening

    two_d = [Opening("DOOR", (1.0, 0.0), 0.8, 0.0, 0.25)]
    level, directory = _prepare_refined_cache(tmp_path, {"result_fingerprint": "fp"})
    assert plan_module.pass1_snapshot_is_current(directory, "fp") is True
    result, info = plan_module._apply_refined_openings(two_d, [], tmp_path, level, 1.0, "fp")
    # no face lines in this stub call -> refined door is dropped, but the hook DID take the apply path (applied True)
    assert info["applied"] is True
    # a snapshot of an older fingerprint (changed settings) must not be used: this pass is a pure pass 1
    result, info = plan_module._apply_refined_openings(two_d, [], tmp_path, level, 1.0, "fp-new")
    assert result == two_d and info["applied"] is False and info["sha256"] == ""


def test_polluted_snapshot_with_refined_openings_is_never_current_and_is_replaced(tmp_path: Path) -> None:
    from samcloud.ortho.vector.indoor import Opening

    polluted = {"result_fingerprint": "fp", "openings_3d": {"sha256": "x", "applied": True}}
    level, directory = _prepare_refined_cache(tmp_path, polluted)
    assert plan_module.pass1_snapshot_is_current(directory, "fp") is False
    two_d = [Opening("DOOR", (1.0, 0.0), 0.8, 0.0, 0.25)]
    result, info = plan_module._apply_refined_openings(two_d, [], tmp_path, level, 1.0, "fp")
    assert result == two_d and info["applied"] is False  # pure pass: becomes the new snapshot
    plan = make_plan(4.3)
    plan.source = {"result_fingerprint": "fp", "openings_3d": info}
    save_plan_model(plan, directory)
    plan_module._store_pass1_snapshot(directory, info, "fp")
    assert _snapshot_length(directory) == 4.3
    assert plan_module.pass1_snapshot_is_current(directory, "fp") is True


def test_snapshot_never_contains_refined_openings_over_repeated_passes(tmp_path: Path) -> None:
    """Pass sequence of run_all (pure pass, applied pass, applied pass again): the snapshot stays the pure pass-1 plan."""
    level, directory = _prepare_refined_cache(tmp_path, None)
    from samcloud.ortho.vector.indoor import Opening

    two_d = [Opening("DOOR", (1.0, 0.0), 0.8, 0.0, 0.25)]
    lengths = iter([4.0, 4.0, 4.0])
    for expected_applied in (False, True, True):
        _, info = plan_module._apply_refined_openings(two_d, [], tmp_path, level, 1.0, "fp")
        assert info["applied"] is expected_applied
        plan = make_plan(next(lengths))
        plan.source = {"result_fingerprint": "fp", "openings_3d": info}
        save_plan_model(plan, directory)
        plan_module._store_pass1_snapshot(directory, info, "fp")
        snapshot = json.loads(pass1_plan_path(directory).read_text())
        assert not (snapshot["source"].get("openings_3d") or {}).get("applied")
