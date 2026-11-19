"""Pipeline of the 3D opening refinement: full mode on rendered synthetic photographs, fallbacks, freshness, cache."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from openings3d_synth import Scene, build_evidence, make_plan
from samcloud.bim.openings3d import pipeline
from samcloud.bim.openings3d import scoring
from samcloud.bim.openings3d.config import load_openings_config
from samcloud.bim.openings3d.model import load_openings_3d, openings_3d_path
from samcloud.bim.openings3d.points import PointEvidence
from samcloud.bim.plan_model import save_plan_model
from samcloud.dxf.cancel import Cancelled
from samcloud.ortho.cameras import OrthoCamera, project_points

WINDOW = (1.0, 2.0, 0.9, 2.1)  # u0, u1, sill, head (m) of the synthetic window on the wall y = 0
LEVEL = "level_1"


def _camera(project: Path, index: int, x: float, size: int = 1232, focal: float = 700.0) -> OrthoCamera:
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # looks along +y, image down = -z
    centre = np.array([x, -3.0, 1.3])
    path = project / "inputs" / "cubemap" / "front" / f"img_{index}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    camera = OrthoCamera(
        f"front/img_{index}.jpg", path, rotation, -rotation @ centre, "PINHOLE", np.array([focal, focal, size / 2, size / 2]), size, size, centre,
    )
    corners = np.array([[WINDOW[0], 0.0, WINDOW[2]], [WINDOW[1], 0.0, WINDOW[2]], [WINDOW[1], 0.0, WINDOW[3]], [WINDOW[0], 0.0, WINDOW[3]]])
    u, v, _, _ = project_points(camera, torch.from_numpy(corners))
    image = np.full((size, size, 3), 120, np.uint8)  # grey wall
    cv2.fillPoly(image, [np.stack([u.numpy(), v.numpy()], 1).astype(np.int32)], (235, 235, 235))  # bright glass
    cv2.imwrite(str(path), image)
    return camera


def _fake_scorer(image_bgr: np.ndarray) -> np.ndarray:
    """Window class where the render is bright, wall class elsewhere (stand-in for SAM3)."""
    bright = (image_bgr.mean(axis=2) > 200).astype(np.float32)
    maps = np.stack([bright * 0.9, np.zeros_like(bright), (1 - bright) * 0.9, np.zeros_like(bright)])
    return np.stack([cv2.resize(channel, (616, 616), interpolation=cv2.INTER_AREA) for channel in maps])


def _wall_cloud() -> np.ndarray:
    """Dense points on the wall plane y = 0 with a window-shaped hole."""
    rng = np.random.default_rng(3)
    x, z = np.meshgrid(np.arange(0.0, 4.0, 0.02), np.arange(0.0, 2.6, 0.02))
    points = np.column_stack([x.ravel(), np.zeros(x.size), z.ravel()]) + rng.normal(0, 0.002, (x.size, 3))
    hole = (points[:, 0] > WINDOW[0]) & (points[:, 0] < WINDOW[1]) & (points[:, 2] > WINDOW[2]) & (points[:, 2] < WINDOW[3])
    return points[~hole]


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    save_plan_model(make_plan(4.0), root / "products" / "dxf" / LEVEL / "ortho")
    (root / "products" / "classified").mkdir(parents=True)
    (root / "products" / "classified" / "classified.ply").write_bytes(b"ply")
    (root / "products" / "dense" / "sparse").mkdir(parents=True)
    (root / "products" / "dense" / "sparse" / "images.bin").write_bytes(b"x")
    return root


@pytest.fixture()
def patched(monkeypatch: pytest.MonkeyPatch, project: Path):
    """Replace the PLY stream and the COLMAP loader; returns counters."""
    state = {"point_passes": 0, "scene": Scene(holes=[(WINDOW[0], WINDOW[1], WINDOW[2], WINDOW[3])], windows=[(WINDOW[0], WINDOW[1], WINDOW[2], WINDOW[3])]), "cameras": []}

    def fake_collect(source, lines, plan, frame, config, *, want_cloud, progress=None, cancel=None):
        state["point_passes"] += 1
        evidence, _, _ = build_evidence(state["scene"], config)
        return PointEvidence(grids={lines[0].id: evidence.grids}, cloud=_wall_cloud() if want_cloud else None, point_count=1)

    import samcloud.ortho.cameras as cameras_module

    monkeypatch.setattr(pipeline, "collect_point_evidence", fake_collect)
    monkeypatch.setattr(cameras_module, "load_cameras", lambda model_dir, project_dir: list(state["cameras"]))
    monkeypatch.setattr(pipeline, "cuda_available", lambda: True)
    (project / "inputs" / "cubemap" / "front").mkdir(parents=True, exist_ok=True)
    return state


def _refine(project: Path, **kwargs):
    return pipeline.refine_level_openings(project, LEVEL, device="cpu", **kwargs)


def test_full_mode_measures_a_window_from_rendered_photographs(project: Path, patched) -> None:
    patched["cameras"] = [_camera(project, i, x) for i, x in enumerate((1.0, 2.0, 3.0))]
    calls = []

    def scorer(image: np.ndarray) -> np.ndarray:
        calls.append(1)
        return _fake_scorer(image)

    outcome = _refine(project, scorer=scorer)
    assert outcome.status == "refined" and outcome.mode == "full" and outcome.changed
    assert len(calls) == 3
    result = load_openings_3d(project / "products" / "dxf" / LEVEL / "ortho")
    assert result.mode == "full" and result.stats["images"] == 3
    windows = [o for o in result.openings if o.kind == "window"]
    assert len(windows) == 1, [o.to_dict() for o in result.openings]
    window = windows[0]
    assert abs(window.width_m - 1.0) < 0.12
    assert abs(window.sill_m - WINDOW[2]) < 0.10 and abs(window.head_m - WINDOW[3]) < 0.10
    assert window.host_wall_id == "W001"
    # schema: PlanOpening-compatible keys plus bookkeeping
    payload = json.loads(openings_3d_path(project / "products" / "dxf" / LEVEL / "ortho").read_text())
    assert payload["schema_version"] == 1
    for key in ("id", "kind", "center", "width_m", "axis_angle", "wall_thickness_m", "host_wall_id", "sill_m", "head_m", "height_source", "mullion_count", "mullion_u", "glazed", "evidence", "confidence"):
        assert key in payload["openings"][0]
    for key in ("curtain_regions", "unconfirmed", "rejected", "inputs"):
        assert key in payload
    assert {"walls", "config_hash", "sparse_model", "point_source", "photo_path"} <= set(payload["inputs"])


def test_score_maps_are_cached_per_image_and_keyed_by_the_prompt_fingerprint(project: Path, patched) -> None:
    patched["cameras"] = [_camera(project, i, x) for i, x in enumerate((1.0, 2.0, 3.0))]
    calls = []

    def scorer(image: np.ndarray) -> np.ndarray:
        calls.append(1)
        return _fake_scorer(image)

    _refine(project, scorer=scorer, force=True)
    _refine(project, scorer=scorer, force=True)  # recomputed from cached maps: no new SAM calls
    assert len(calls) == 3
    config = load_openings_config()
    directory = scoring.scores_directory(project, config)
    assert directory.parent.name == "openings_cache" and len(list(directory.glob("front__img_*.npy"))) == 3
    changed = load_openings_config(overrides={"sam": {"map_resolution": 308}})
    assert scoring.scores_directory(project, changed) != directory
    # a re-written photograph invalidates only its own map
    _camera(project, 0, 1.0)
    import os
    os.utime(patched["cameras"][0].image_path, ns=(1_000_000_000, 1_000_000_000))
    _refine(project, scorer=scorer, force=True)
    assert len(calls) == 4


def test_fresh_result_is_not_recomputed_and_force_does(project: Path, patched) -> None:
    patched["cameras"] = [_camera(project, i, x) for i, x in enumerate((1.0, 2.0, 3.0))]
    assert _refine(project, scorer=_fake_scorer).status == "refined"
    passes = patched["point_passes"]
    outcome = _refine(project, scorer=_fake_scorer)
    assert outcome.status == "fresh" and not outcome.changed
    assert patched["point_passes"] == passes
    assert pipeline.refine_status(project, LEVEL) == "fresh"
    again = _refine(project, scorer=_fake_scorer, force=True)
    assert again.status == "refined" and patched["point_passes"] == passes + 1
    assert not again.changed  # deterministic: same file


def test_new_wall_set_makes_the_result_stale(project: Path, patched) -> None:
    patched["cameras"] = [_camera(project, i, x) for i, x in enumerate((1.0, 2.0, 3.0))]
    _refine(project, scorer=_fake_scorer)
    save_plan_model(make_plan(5.5), project / "products" / "dxf" / LEVEL / "ortho")
    assert pipeline.refine_status(project, LEVEL) == "stale"


def test_openings_do_not_change_the_fingerprint(project: Path, patched) -> None:
    """Re-planning with other openings must not make the refinement stale (no endless plan -> refine loop)."""
    patched["cameras"] = [_camera(project, i, x) for i, x in enumerate((1.0, 2.0, 3.0))]
    _refine(project, scorer=_fake_scorer)
    from openings3d_synth import seed_door

    save_plan_model(make_plan(4.0, openings=[seed_door("O001", 3.2, 0.9)]), project / "products" / "dxf" / LEVEL / "ortho")
    assert pipeline.refine_status(project, LEVEL) == "fresh"


def test_without_cuda_falls_back_to_points_only(project: Path, patched, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "cuda_available", lambda: False)
    patched["cameras"] = [_camera(project, 0, 2.0)]
    patched["scene"] = Scene(holes=[WINDOW], windows=[WINDOW], unobserved=[(3.2, 4.0, 0.0, 2.1)])  # gap at the wall end: not bounded by wall points
    outcome = pipeline.refine_level_openings(project, LEVEL, scorer=lambda image: pytest.fail("SAM must not run without CUDA"))
    assert outcome.status == "refined" and outcome.mode == "points"
    result = load_openings_3d(project / "products" / "dxf" / LEVEL / "ortho")
    assert result.mode == "points" and result.inputs["photo_path"] is False
    assert [o.kind for o in result.openings] == ["window"]  # the unbounded gap stays unknown
    assert result.openings[0].height_source == "points"
    assert abs(result.openings[0].sill_m - WINDOW[2]) < 0.08


def test_missing_images_fall_back_to_points_only(project: Path, patched) -> None:
    patched["cameras"] = []  # COLMAP model without usable photographs
    outcome = _refine(project, scorer=lambda image: pytest.fail("no images: SAM must not run"))
    assert outcome.status == "refined" and outcome.mode == "points"
    assert [o.height_source for o in load_openings_3d(project / "products" / "dxf" / LEVEL / "ortho").openings] == ["points"]


def test_cameras_with_missing_image_files_are_ignored(project: Path, patched) -> None:
    camera = _camera(project, 0, 2.0)
    camera.image_path.unlink()
    patched["cameras"] = [camera]
    assert _refine(project, scorer=lambda image: pytest.fail("unreadable photographs")).mode == "points"


def test_any_exception_keeps_the_2d_openings(project: Path, patched, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(pipeline, "collect_point_evidence", boom)
    patched["cameras"] = [_camera(project, 0, 2.0)]
    with caplog.at_level("WARNING"):
        outcome = _refine(project, scorer=_fake_scorer)
    assert outcome.status == "failed" and "disk on fire" in outcome.message
    assert not openings_3d_path(project / "products" / "dxf" / LEVEL / "ortho").exists()
    assert any("refinement failed" in record.message for record in caplog.records)


def test_cancellation_propagates(project: Path, patched) -> None:
    patched["cameras"] = [_camera(project, i, x) for i, x in enumerate((1.0, 2.0, 3.0))]

    def progress(fraction: float, stage: str) -> None:
        if "Scoring" in stage:
            raise Cancelled("stop")

    with pytest.raises(Cancelled):
        _refine(project, scorer=_fake_scorer, progress=progress)
    assert not openings_3d_path(project / "products" / "dxf" / LEVEL / "ortho").exists()


def _refine_json(project: Path, **kwargs) -> str:
    _refine(project, force=True, **kwargs)
    return openings_3d_path(project / "products" / "dxf" / LEVEL / "ortho").read_text(encoding="utf-8")


def test_refining_again_from_a_refined_plan_is_idempotent(project: Path, patched) -> None:
    """Second run starts from the plan that already holds the refined openings and slightly different walls (pass 2)."""
    from openings3d_synth import seed_door
    from samcloud.bim.plan_model import PlanOpening, PlanWall, load_plan_model, pass1_plan_path, save_pass1_snapshot

    patched["cameras"] = [_camera(project, i, x) for i, x in enumerate((1.0, 2.0, 3.0))]
    directory = project / "products" / "dxf" / LEVEL / "ortho"
    save_pass1_snapshot(directory)  # the plan of pass 1 (no refined openings applied)
    first = _refine_json(project, scorer=_fake_scorer)
    assert json.loads(first)["openings"], "synthetic scene must yield an opening"
    # pass 2: walls differ by a few cm (hook changed the wall builder input) and the plan now carries refined openings
    refined = load_openings_3d(directory)
    pass2 = make_plan(4.03, openings=[PlanOpening(o.id, o.kind, list(o.center), o.width_m, o.axis_angle, o.wall_thickness_m, o.host_wall_id) for o in refined.openings] + [seed_door("O009", 3.6, 0.5)])
    pass2.walls = [PlanWall(id="W001", centreline=[[0.0, 0.01], [4.03, 0.01]], thickness_m=0.25, kind="exterior")]
    save_plan_model(pass2, directory)
    assert pass1_plan_path(directory).is_file() and load_plan_model(directory).openings != load_plan_model(pass1_plan_path(directory)).openings
    assert pipeline.refine_status(project, LEVEL) == "fresh"  # same pass-1 walls: reused, no recompute
    passes = patched["point_passes"]
    assert _refine(project, scorer=_fake_scorer).status == "fresh" and patched["point_passes"] == passes
    second = _refine_json(project, scorer=_fake_scorer)  # forced re-run
    assert second == first


def test_without_a_snapshot_the_refined_plan_would_change_the_seeds(project: Path, patched) -> None:
    """Documents the legacy fallback: no pass-1 snapshot -> plan_model.json is the input (what the snapshot prevents)."""
    from samcloud.bim.plan_model import load_refine_plan, pass1_plan_path, save_pass1_snapshot

    directory = project / "products" / "dxf" / LEVEL / "ortho"
    assert not pass1_plan_path(directory).is_file()
    assert load_refine_plan(directory).walls[0].id == "W001"
    save_pass1_snapshot(directory)
    save_plan_model(make_plan(5.5), directory)
    assert load_refine_plan(directory).walls[0].centreline[1][0] == 4.0  # snapshot wins over the newer plan_model.json
