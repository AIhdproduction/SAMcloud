"""Screens from the mask outlines recorded during the SAM pass: dark screens without points, and what is no screen."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from shapely.geometry import LineString, box

from samcloud.bim.room_objects.evidence import build_evidence
from samcloud.bim.room_objects.screen_frames import MaskView, fit_screen_frames, load_mask_views
from samcloud.bim.room_objects.screens import detect_screens
from samcloud.bim.room_objects.settings import ReflectionSettings, ScreenFrameSettings, ScreenSettings
from samcloud.engine.instances import recorded_prompt_indices
from samcloud.engine.sam_labeler import HeadFusionLabeler
from samcloud.engine.screen_masks import FILE_NAME, mask_outline, read_screen_masks, write_screen_masks

ROOM = box(0.013, 0.017, 4.991, 4.013)
WALLS = [LineString((a, b)) for a, b in zip(list(ROOM.exterior.coords)[:-1], list(ROOM.exterior.coords)[1:])]
EYES = ((1.31, 2.37, 1.63), (1.93, 1.74, 1.57), (2.61, 2.13, 1.71), (3.37, 1.83, 1.52), (2.24, 2.91, 1.66))
FOCAL, SIZE = 611.0, 1000


def _corners(width=0.613, height=0.367, bottom=0.893):
    middle = np.asarray((2.283, 0.213, bottom + 0.5 * height))
    across, up = np.asarray((-0.5 * width, 0.0, 0.0)), np.asarray((0.0, 0.0, 0.5 * height))
    # seen from the front (the room is at +y), so the left of the picture is at +x
    return np.vstack((middle - across + up, middle + across + up, middle + across - up, middle - across - up))


def _views(corners, labels, eyes=EYES, conf=0.94, seed=7):
    rng = np.random.default_rng(seed)
    views = []
    for eye in np.asarray(eyes, dtype=float):
        forward = corners.mean(axis=0) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        axes = np.vstack((right, np.cross(forward, right), forward))
        camera = (corners - eye) @ axes.T
        pixels = FOCAL * camera[:, :2] / camera[:, 2:] + SIZE / 2 + rng.normal(0.0, 0.7, (4, 2))
        masks = [{"label": label, "conf": conf, "cut": False, "outline": pixels.copy()} for label in labels]
        views.append(MaskView(eye, axes, np.asarray((FOCAL, FOCAL, SIZE / 2, SIZE / 2)), (SIZE, SIZE), masks))
    return views


def _fit(views, rooms=(ROOM,)):
    return fit_screen_frames(views, list(rooms), ScreenFrameSettings())


@pytest.mark.parametrize("width,height", [(0.613, 0.367), (0.947, 0.351), (1.437, 0.823)])
def test_dark_screen_gets_its_measured_size_from_the_photos(width, height):
    frames, _ = _fit(_views(_corners(width, height), ("screen", "monitor")))
    assert len(frames) == 1
    frame = frames[0]
    assert frame.width == pytest.approx(width, abs=0.02) and frame.height == pytest.approx(height, abs=0.02)
    assert np.allclose(frame.centre, _corners(width, height).mean(axis=0), atol=0.02)
    assert frame.normal[1] > 0.98  # faces the cameras


def test_picture_that_only_answers_to_television_is_no_screen():
    frames, refused = _fit(_views(_corners(), ("television", "screen")))
    assert not frames and any("picture" in entry["reason"] for entry in refused)


def test_weak_monitor_name_on_a_mirror_is_no_screen():
    views = _views(_corners(), ("screen",), conf=0.92)
    for view in views:
        view.masks.append({**view.masks[0], "label": "monitor", "conf": 0.63})
    assert not _fit(views)[0]


def test_two_photos_are_not_enough():
    assert not _fit(_views(_corners(), ("screen", "monitor"), eyes=EYES[:2]))[0]


def test_photos_from_one_place_are_not_enough():
    eyes = [np.add(EYES[0], (0.03 * step, 0.01 * step, 0.0)) for step in range(5)]
    assert not _fit(_views(_corners(), ("screen", "monitor"), eyes=eyes))[0]


def test_panel_at_the_floor_is_no_screen():
    frames, refused = _fit(_views(_corners(bottom=0.061), ("screen", "monitor")))
    assert not frames and refused


def test_panel_seen_through_a_pane_from_another_room_is_no_screen():
    rooms = [box(0.013, 0.017, 4.991, 0.913), box(0.013, 1.013, 4.991, 4.013)]
    frames, refused = _fit(_views(_corners(), ("screen", "monitor")), rooms)
    assert not frames and any("another room" in entry["reason"] for entry in refused)


def test_panel_in_a_space_without_a_room_counts_as_inside():
    from samcloud.bim.room_objects.geometry import enclosed_spaces

    # the plan only knows the room behind a partition (faces at y 1.813 / 1.913); the cameras and the screen are in front
    known = box(0.013, 1.913, 4.991, 4.013)
    walls = [*WALLS, LineString(((0.013, 1.813), (4.991, 1.813))), LineString(((0.013, 1.913), (4.991, 1.913)))]
    eyes = [np.subtract(eye, (0.0, 1.2, 0.0)) for eye in EYES]
    views = _views(_corners(), ("screen", "monitor"), eyes=eyes)
    frames, refused = _fit(views, [known])
    assert not frames and any("outside the rooms" in entry["reason"] for entry in refused)
    spaces = enclosed_spaces([known], walls, None, face_gap_m=0.05, min_area_m2=1.0, min_width_m=0.60)
    assert len(spaces) == 1 and spaces[0].area == pytest.approx(4.978 * 1.796, rel=0.03)
    frames, _ = _fit(views, [known, *spaces])
    assert len(frames) == 1 and frames[0].width == pytest.approx(0.613, abs=0.02)


def test_cut_masks_never_give_a_size():
    views = _views(_corners(), ("screen", "monitor"))
    for view in views:
        for mask in view.masks:
            mask["cut"] = True
    assert not _fit(views)[0]


def _evidence(fragment=None):
    rng = np.random.default_rng(3)
    desk = np.column_stack((rng.uniform(1.71, 2.87, 900), rng.uniform(0.05, 0.63, 900), rng.normal(0.753, 0.003, 900)))
    parts, terms = [desk], [np.zeros(len(desk), np.int32)]
    if fragment is not None:
        width, height = fragment
        points = np.column_stack((
            rng.uniform(2.283 - width / 2, 2.283 + width / 2, 1500), rng.normal(0.213, 0.003, 1500),
            rng.uniform(1.0765 - height / 2, 1.0765 + height / 2, 1500),
        ))
        parts.append(points)
        terms.append(np.ones(len(points), np.int32))
    xyz = np.vstack(parts)
    return build_evidence(
        xyz, np.full(len(xyz), 17), units_per_metre=1.0, floor_z=0.0, ceiling_z=2.513, instance=None,
        fine=np.concatenate(terms), rgb=np.full((len(xyz), 3), 40.0), fine_names=["screen"], instance_labels={},
    )


def _detect(evidence):
    return detect_screens(
        evidence, rooms_m=[ROOM], wall_lines_m=WALLS, settings=ScreenSettings(), reflection=ReflectionSettings(),
        frames=ScreenFrameSettings(),
    )


def test_screen_without_a_single_point_is_added_on_its_desk():
    evidence = _evidence()
    assert not _detect(evidence)[0]
    evidence.screen_views = _views(_corners(), ("screen", "monitor"))
    found, report = _detect(evidence)
    assert len(found) == 1 and (found[0].semantic, found[0].mounting) == ("monitor", "resting")
    assert found[0].width == pytest.approx(0.613, abs=0.02) and report["photo_frames"][0]["used"] == "added"


def test_photos_replace_a_too_small_fragment():
    evidence = _evidence(fragment=(0.35, 0.30))
    evidence.screen_views = _views(_corners(), ("screen", "monitor"))
    found, report = _detect(evidence)
    assert len(found) == 1 and found[0].width == pytest.approx(0.613, abs=0.02)
    assert report["photo_frames"][0]["used"] == "replaces the cloud panel"


def test_outline_is_in_image_pixels_and_knows_when_it_is_cut():
    mask = np.zeros((72, 72), np.float32)
    mask[13:41, 17:59] = 0.93
    outline, cut = mask_outline(mask, (100, 50, 820, 770))
    ring = np.asarray(outline)
    assert not cut
    assert np.allclose(ring.min(axis=0), (270, 180), atol=6) and np.allclose(ring.max(axis=0), (690, 460), atol=6)
    mask[13:41, :59] = 0.93
    assert mask_outline(mask, (100, 50, 820, 770))[1]


def test_product_round_trip_and_a_moved_cloud_is_not_used(tmp_path):
    cloud = np.random.default_rng(5).uniform((0.0, 0.0, 0.0), (5.0, 4.0, 2.5), (20000, 3))
    view = {
        "name": "a.jpg", "size": [1000, 1000], "pinhole": [611.0, 611.0, 500.0, 500.0],
        "world_to_cam": np.column_stack((np.eye(3), (-1.0, -2.0, 0.0))).tolist(),
        "masks": [{"label": "computer screen", "conf": 0.91, "cut": False, "outline": [[1, 2], [30, 2], [30, 20]]}],
    }
    path = write_screen_masks(tmp_path / FILE_NAME, [view], cloud)
    assert len(read_screen_masks(path)["views"]) == 1
    loaded = load_mask_views(path, cloud, lambda points: np.asarray(points) * 2.0, units_per_metre=2.0, floor_z=0.4)
    assert np.allclose(loaded[0].eye, (1.0, 2.0, -0.2)) and loaded[0].masks[0]["label"] == "monitor"
    assert not load_mask_views(path, cloud + (30.0, 0.0, 0.0), lambda points: points, units_per_metre=1.0, floor_z=0.0)


def test_labeler_records_a_screen_without_points_and_changes_no_membership():
    mask = np.zeros((20, 20), np.float16)
    mask[4:14, 3:17] = 0.91
    raw = {"prompt_idx": np.array([2]), "conf": np.array([0.9], np.float32), "masks": mask[None]}

    def run(record):
        labeler = HeadFusionLabeler.__new__(HeadFusionLabeler)
        labeler.class_names, labeler._prompt_owner = ["furniture"], np.zeros(3, np.int64)
        labeler.tile_size, labeler.min_overlap_frac = 100, 0.2
        labeler.fine_queries = labeler.last_fine_probs = labeler.last_tile_diagnostics = None
        labeler.instance_params = {"mask_threshold": 0.5, "min_mask_pixels": 12}
        labeler.secondary_prompt_ids, labeler.record_prompt_ids = {2}, record

        def score(tile):
            labeler.last_instances = raw
            return torch.zeros((1, *tile.shape[:2]), dtype=torch.float32)

        labeler._score_tile = score
        out = {}
        # the only dense point lies outside the screen
        labeler.score_points("x", np.array([93]), np.array([93]), image=np.zeros((100, 100, 3), np.uint8), instances_out=out)
        return out

    recorded, plain = run({2}), run(set())
    assert len(recorded["recorded_masks"]) == 1 and "recorded_masks" not in plain
    for key in ("point_local_instance", "point_secondary_instance", "instances"):
        assert np.array_equal(recorded[key], plain[key])


def test_recorded_prompts_are_screens_and_the_secondary_layer():
    texts = ["furniture", "television", "computer monitor", "screen", "cabinet", "ceiling lamp"]
    assert recorded_prompt_indices("indoor", texts, {5}) == {1, 2, 3, 5}
    assert recorded_prompt_indices("outdoor", texts, {5}) == set()


def test_classify_writes_the_product_beside_the_cloud(tmp_path, monkeypatch):
    from samcloud.engine import classify

    class Labeler:
        def load_image(self, _path):
            return np.zeros((16, 16, 3), np.uint8), 16, 16

        def score_points(self, _path, u, _v, instances_out=None, **_kwargs):
            outline = [[2.0, 2.0], [9.0, 2.0], [9.0, 7.0], [2.0, 7.0]]
            instances_out.update(
                point_local_instance=np.full(len(u), -1), instances=[],
                recorded_masks=[{"prompt_idx": 1, "conf": 0.88, "cut": False, "outline": outline}],
            )
            return np.full((len(u), 1), 0.9, np.float32)

    monkeypatch.setattr(classify, "make_point_labeler", lambda *_a, **_k: Labeler())
    monkeypatch.setattr(classify, "project_points", lambda *_a, **_k: (
        np.arange(12), np.zeros(12, int), np.zeros(12, int), np.full(12, 2.0)))
    args = SimpleNamespace(
        background_threshold=0.4, threshold_sweep="", dense_dir=str(tmp_path), labels_dir=None, images_dir=str(tmp_path),
        depth_check=0, out=str(tmp_path / "out.ply"), scores_out=None, min_votes=1, summary_file=None, class_set="indoor",
        engine="sam3", mode="segearth_ov3", query_threshold=0.4, no_instances=False,
    )
    camera = SimpleNamespace(width=16, height=16, intrinsics=lambda: (9.0, 9.0, 8.0, 8.0))
    image = SimpleNamespace(name="v.png", camera_id=1, world_to_cam=lambda points: np.asarray(points) + (0.0, 0.0, 2.0))
    xyz = np.column_stack((np.linspace(0.003, 1.0, 12), np.zeros(12), np.zeros(12)))
    classify.run_segearth_ov3(
        args, xyz, np.zeros((12, 3), np.uint8), {1: camera}, {1: image}, [(1, list(range(12)))], ["furniture"], 0, None,
        prompts=[["furniture", "screen"]], fine_config={"instance_classes": {"indoor": ["furniture"]}},
        instance_config={"enabled": True},
    )
    stored = read_screen_masks(tmp_path / FILE_NAME)
    assert stored["views"][0]["masks"][0]["label"] == "screen" and stored["views"][0]["pinhole"] == [9.0, 9.0, 8.0, 8.0]


@pytest.mark.parametrize(
    "failure", ["write", "replace", "stat", "view_record", "intrinsics", "labeler", "nothing can be written"],
)
@pytest.mark.parametrize("older_product", [False, True])
def test_a_failing_mask_product_never_costs_the_classification(tmp_path, monkeypatch, capsys, failure, older_product):
    from samcloud.engine import classify, screen_masks

    product = tmp_path / FILE_NAME
    xyz = np.column_stack((np.linspace(0.003, 1.0, 12), np.zeros(12), np.zeros(12)))
    old = {"name": "old.png", "masks": []}
    if older_product:
        write_screen_masks(product, [old], xyz)

    class Labeler:
        def load_image(self, _path):
            return np.zeros((16, 16, 3), np.uint8), 16, 16

        def score_points(self, _path, u, _v, instances_out=None, **_kwargs):
            instances_out.update(point_local_instance=np.full(len(u), -1), instances=[])
            if failure == "labeler":
                instances_out["record_error"] = ValueError("outline")
            else:
                outline = [[2.0, 2.0], [9.0, 2.0], [9.0, 7.0]]
                instances_out["recorded_masks"] = [{"prompt_idx": 1, "conf": 0.88, "cut": False, "outline": outline}]
            return np.full((len(u), 1), 0.9, np.float32)

    def broken(*_args, **_kwargs):
        raise OSError("injected failure,\nsecond line")

    def intrinsics():
        if failure == "intrinsics":
            raise ValueError("not a pinhole camera")
        return 9.0, 9.0, 8.0, 8.0

    if failure in ("write", "view_record"):
        monkeypatch.setattr(classify, "write_screen_masks" if failure == "write" else "view_record", broken)
    elif failure == "replace":  # the half-written file stays under its temporary name
        real_replace = screen_masks.os.replace

        def replace(source, target):
            if str(target).endswith(FILE_NAME):
                raise OSError("injected failure")
            real_replace(source, target)

        monkeypatch.setattr(screen_masks.os, "replace", replace)
    elif failure == "stat":
        class Unreadable(type(product)):
            def stat(self, **kwargs):
                if self.name == FILE_NAME:
                    raise OSError("injected failure")
                return super().stat(**kwargs)

        save = classify.save_screen_masks
        monkeypatch.setattr(classify, "save_screen_masks", lambda path, *rest: save(Unreadable(path), *rest))
    elif failure == "nothing can be written":
        monkeypatch.setattr(classify, "write_screen_masks", broken)
        monkeypatch.setattr(classify, "invalidate_screen_masks", broken)
    monkeypatch.setattr(classify, "make_point_labeler", lambda *_a, **_k: Labeler())
    monkeypatch.setattr(classify, "project_points", lambda *_a, **_k: (
        np.arange(12), np.zeros(12, int), np.zeros(12, int), np.full(12, 2.0)))
    args = SimpleNamespace(
        background_threshold=0.4, threshold_sweep="", dense_dir=str(tmp_path), labels_dir=None, images_dir=str(tmp_path),
        depth_check=0, out=str(tmp_path / "out.ply"), scores_out=str(tmp_path / "scores.npz"), min_votes=1,
        summary_file=str(tmp_path / "summary.json"), class_set="indoor", engine="sam3", mode="segearth_ov3",
        query_threshold=0.4, no_instances=False,
    )
    camera = SimpleNamespace(width=16, height=16, intrinsics=intrinsics)
    images = {
        view: SimpleNamespace(name=f"v{view}.png", camera_id=1, world_to_cam=lambda points: np.asarray(points) + (0.0, 0.0, 2.0))
        for view in (1, 2)
    }
    classify.run_segearth_ov3(
        args, xyz, np.zeros((12, 3), np.uint8), {1: camera}, images, [(1, list(range(12))), (2, list(range(12)))],
        ["furniture"], 0, None, prompts=[["furniture", "screen"]], fine_config={"instance_classes": {"indoor": ["furniture"]}},
        instance_config={"enabled": True},
    )

    for name in ("out.ply", "scores.npz", "summary.json", "instances.json"):
        assert (tmp_path / name).stat().st_size > 0
    warnings = [line for line in capsys.readouterr().out.splitlines() if "WARNING" in line]
    assert len(warnings) == 1 and "continuing without them" in warnings[0]
    if failure == "nothing can be written":
        assert ("could not be invalidated" in warnings[0])
    elif older_product:
        stored = read_screen_masks(product)
        # with a failing stat the new file is complete, but nothing of the older run may be used in any case
        assert stored is None or old not in stored["views"]
        if failure != "stat":
            assert stored is None or stored["views"] == []
    elif failure != "stat":
        assert not product.exists()


def test_a_failing_outline_stops_the_recording_and_changes_no_membership(monkeypatch):
    from samcloud.engine import screen_masks

    def broken(*_args, **_kwargs):
        raise ValueError("outline")

    monkeypatch.setattr(screen_masks, "mask_outline", broken)
    mask = np.zeros((20, 20), np.float16)
    mask[4:14, 3:17] = 0.91
    raw = {"prompt_idx": np.array([2]), "conf": np.array([0.9], np.float32), "masks": mask[None]}

    def run(record):
        labeler = HeadFusionLabeler.__new__(HeadFusionLabeler)
        labeler.class_names, labeler._prompt_owner = ["furniture"], np.zeros(3, np.int64)
        labeler.tile_size, labeler.min_overlap_frac = 100, 0.2
        labeler.fine_queries = labeler.last_fine_probs = labeler.last_tile_diagnostics = None
        labeler.instance_params = {"mask_threshold": 0.5, "min_mask_pixels": 12}
        labeler.secondary_prompt_ids, labeler.record_prompt_ids = set(), record

        def score(tile):
            labeler.last_instances = raw
            return torch.zeros((1, *tile.shape[:2]), dtype=torch.float32)

        labeler._score_tile = score
        out = {}
        labeler.score_points("x", np.array([50]), np.array([45]), image=np.zeros((100, 100, 3), np.uint8), instances_out=out)
        return out, labeler

    (failed, labeler), (plain, _) = run({2}), run(set())
    assert isinstance(failed["record_error"], ValueError) and "recorded_masks" not in failed
    assert not labeler.record_prompt_ids
    for key in ("point_local_instance", "instances"):
        assert np.array_equal(failed[key], plain[key])


def test_frame_probe_converts_only_the_sample(tmp_path):
    plain = np.random.default_rng(11).uniform(0.0, 4.0, (5000, 3)).astype(np.float32)
    seen = []
    real = np.asarray

    def asarray(value, *args, **kwargs):
        if value is plain:
            seen.append(kwargs.get("dtype", args[0] if args else None))
        return real(value, *args, **kwargs)

    from samcloud.engine import screen_masks

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(screen_masks.np, "asarray", asarray)
        path = write_screen_masks(tmp_path / FILE_NAME, [], plain)
    assert seen == [None]  # the cloud itself keeps its dtype
    assert len(read_screen_masks(path)["frame_probe"]) == 256
