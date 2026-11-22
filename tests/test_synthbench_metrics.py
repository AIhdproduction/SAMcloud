"""Synthetic bench: metric correctness on hand-made ground truth / plan pairs (no pipeline)."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from samcloud.bim.synthbench.compare import compare_plans
from samcloud.bim.synthbench.metrics import (
    PlanFrame, adapt_plan, furniture_metrics, opening_metrics, room_metrics, wall_metrics,
)


def _truth() -> dict:
    t = 0.2
    h = t / 2
    walls = [
        {"id": "W1", "p0": [-h, -h], "p1": [4 + h, -h], "thickness": t, "exterior": True, "length": 4 + t},
        {"id": "W2", "p0": [4 + h, -h], "p1": [4 + h, 3 + h], "thickness": t, "exterior": True, "length": 3 + t},
        {"id": "W3", "p0": [4 + h, 3 + h], "p1": [-h, 3 + h], "thickness": t, "exterior": True, "length": 4 + t},
        {"id": "W4", "p0": [-h, 3 + h], "p1": [-h, -h], "thickness": t, "exterior": True, "length": 3 + t},
    ]
    room = {"name": "R", "polygon": {"exterior": [[0, 0], [4, 0], [4, 3], [0, 3]], "holes": []}, "area": 12.0, "ceiling_h": 2.5}
    openings = [
        {"id": "O1", "kind": "door", "host": "W1", "center": [1.0, -h], "width": 0.9, "sill": 0.0, "head": 2.0, "axis_angle": 0.0, "thickness": t},
        {"id": "O2", "kind": "window", "host": "W3", "center": [2.0, 3 + h], "width": 1.2, "sill": 0.9, "head": 2.1, "axis_angle": math.pi, "thickness": t},
    ]
    furniture = [
        {"id": "F1", "label": "sofa", "group": "sofa", "variant": "sofa", "footprint": {"exterior": [[0.5, 1], [2.5, 1], [2.5, 1.9], [0.5, 1.9]], "holes": []}, "z0": 0.0, "z1": 0.8},
        {"id": "F2", "label": "chair", "group": "seat", "variant": "chair", "footprint": {"exterior": [[3, 0.5], [3.45, 0.5], [3.45, 0.95], [3, 0.95]], "holes": []}, "z0": 0.0, "z1": 0.9},
        {"id": "F3", "label": "chair", "group": "seat", "variant": "chair", "footprint": {"exterior": [[3.46, 0.5], [3.91, 0.5], [3.91, 0.95], [3.46, 0.95]], "holes": []}, "z0": 0.0, "z1": 0.9},
    ]
    junctions = [{"point": [-h, -h], "walls": ["W1", "W4"]}, {"point": [4 + h, -h], "walls": ["W1", "W2"]},
                 {"point": [4 + h, 3 + h], "walls": ["W2", "W3"]}, {"point": [-h, 3 + h], "walls": ["W3", "W4"]}]
    return {"name": "hand", "family": "test", "seed": 0, "storey_height": 2.5, "max_height": 2.5, "world_yaw_deg": 0.0, "world_offset": [0, 0, 0],
            "walls": walls, "rooms": [room], "openings": openings, "furniture": furniture, "junctions": junctions, "columns": [], "beams": [], "stairs": []}


def _pred_walls(truth: dict) -> list[dict]:
    return [{"id": w["id"].replace("W", "P"), "p0": np.array(w["p0"]), "p1": np.array(w["p1"]), "thickness": w["thickness"], "length": w["length"]} for w in truth["walls"]]


def _rooms(truth: dict):
    from samcloud.bim.synthbench.geometry import polygon_from_list

    return [polygon_from_list(r["polygon"]) for r in truth["rooms"]]


def test_perfect_walls_score_perfectly() -> None:
    truth = _truth()
    m = wall_metrics(truth, _pred_walls(truth), _rooms(truth))
    assert m["recall"] == 1.0 and m["precision"] == 1.0
    assert m["face_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["thickness_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["length_coverage"] == pytest.approx(1.0)
    assert m["junction_gaps"] == 0 and m["parallel_overlaps"] == 0 and m["free_ends"] == 0


def test_shifted_thicker_and_missing_walls() -> None:
    truth = _truth()
    pred = _pred_walls(truth)
    pred[0]["p0"] = pred[0]["p0"] + np.array([0.0, -0.05])  # shift W1 outwards by 5 cm
    pred[0]["p1"] = pred[0]["p1"] + np.array([0.0, -0.05])
    pred[1]["thickness"] = 0.3  # W2 thicker; centre line unchanged -> inner face 5 cm into the room
    del pred[2]  # W3 missing
    pred.append({"id": "PX", "p0": np.array([2.0, 0.5]), "p1": np.array([2.0, 2.5]), "thickness": 0.1, "length": 2.0})  # false wall
    m = wall_metrics(truth, pred, _rooms(truth))
    per = {w["id"]: w for w in m["per_wall"]}
    assert per["W1"]["face_error"] == pytest.approx(0.05, abs=1e-6)
    assert per["W2"]["face_error"] == pytest.approx(0.05, abs=1e-6)
    assert per["W2"]["thickness_error"] == pytest.approx(0.1, abs=1e-6)
    assert m["missed"] == ["W3"]
    assert m["false_walls"] == ["PX"]
    assert m["recall"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.75)


def test_door_gap_is_excused_from_coverage() -> None:
    truth = _truth()
    pred = _pred_walls(truth)
    w1 = pred[0]
    left = dict(w1, id="P1a", p1=np.array([0.55, -0.1]))
    right = dict(w1, id="P1b", p0=np.array([1.45, -0.1]))
    left["length"] = float(np.linalg.norm(left["p1"] - left["p0"]))
    right["length"] = float(np.linalg.norm(right["p1"] - right["p0"]))
    pred = [left, right] + pred[1:]
    m = wall_metrics(truth, pred, _rooms(truth))
    per = {w["id"]: w for w in m["per_wall"]}
    assert per["W1"]["coverage"] == pytest.approx(1.0, abs=1e-6)
    assert per["W1"]["pieces"] == 2


def test_room_iou_and_area() -> None:
    truth = _truth()
    from shapely.geometry import box

    pred = [{"id": "R1", "name": "x", "polygon": box(0, 0, 4, 2.7), "area_m2": 10.8, "ceiling_h": 2.45}]
    m = room_metrics(truth["rooms"], pred)
    assert m["mean_iou"] == pytest.approx(0.9)
    assert m["area_error_rel"] == pytest.approx(0.1)
    assert m["ceiling_error_m"] == pytest.approx(0.05)
    assert m["count_ok"]


def test_openings_kind_width_and_host() -> None:
    truth = _truth()
    pred = [
        {"id": "Q1", "kind": "door", "center": np.array([1.05, -0.1]), "width": 0.95, "sill": 0.0, "head": 2.05, "host": "P1", "angle": 0.0},
        {"id": "Q2", "kind": "door", "center": np.array([2.0, 3.1]), "width": 1.2, "sill": 0.0, "head": 2.1, "host": "P3", "angle": 0.0},
        {"id": "Q3", "kind": "window", "center": np.array([4.1, 1.5]), "width": 1.0, "sill": 0.9, "head": 2.0, "host": "P2", "angle": math.pi / 2},
    ]
    assign = {"P1": "W1", "P2": "W2", "P3": "W3"}
    m = opening_metrics(truth["openings"], pred, assign)
    assert m["door"]["tp"] == 1 and m["door"]["pred"] == 2 and m["door"]["recall"] == 1.0
    assert m["window"]["recall"] == 0.0
    assert m["confusions"] == ["O2:window->door"]
    assert m["width_error_m"] == pytest.approx(0.025)
    assert m["host_correct"] == 1.0
    assert m["false"] == ["Q3:window w1.00"]


def test_furniture_merge_split_and_labels() -> None:
    truth = _truth()
    from shapely.geometry import box

    pred = [
        {"id": "G1", "label": "Sofa", "footprint": box(0.5, 1.0, 2.5, 1.9), "z0": 0.0, "z1": 0.8},
        {"id": "G2", "label": "table", "footprint": box(3.0, 0.5, 3.91, 0.95), "z0": 0.0, "z1": 0.9},  # merges both chairs
    ]
    m = furniture_metrics(truth["furniture"], pred)
    assert m["matched"] == 2
    assert m["merged"] == ["F2", "F3"]
    assert m["one_to_one_rate"] == pytest.approx(1 / 3)
    sofa = next(p for p in m["pairs"] if p["gt"] == "F1")
    assert sofa["label_ok"] and sofa["iou2d"] == pytest.approx(1.0) and sofa["iou3d"] == pytest.approx(1.0)
    assert m["close_count"] == 2 and m["close_pairs_separated"] == 0.0


def test_plan_frame_maps_rotated_scaled_plan_back_to_scene() -> None:
    yaw = math.radians(33.0)
    world_rot = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    truth = {"world_yaw_deg": 33.0, "world_offset": [1.5, -2.0, 0.1]}
    a = math.radians(-20.0)
    rotation = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    plan = {"frame": {"origin": [3.0, 1.0, 0.0], "scale_factor": 1.0, "rotation": rotation.tolist()}, "units_per_metre": 1.04, "floor_z": 0.1,
            "walls": [], "rooms": [], "openings": [], "furniture": []}
    frame = PlanFrame(plan, truth)
    scene_pt = np.array([2.0, 1.0, 0.0])
    world = scene_pt @ world_rot.T + np.array(truth["world_offset"])
    local = (world - np.array(plan["frame"]["origin"])) @ rotation.T
    assert np.allclose(frame.local_to_scene(local[None, :])[0], scene_pt)
    assert frame.metres(0.25) == pytest.approx(0.26)
    assert adapt_plan(plan, frame)["walls"] == []


def test_compare_plans_reports_shift() -> None:
    plan = {"frame": {"origin": [0, 0, 0], "scale_factor": 1.0, "rotation": np.eye(3).tolist()}, "units_per_metre": 1.0, "floor_z": 0.0,
            "walls": [{"id": "W1", "centreline": [[0, 0], [4, 0]], "thickness_m": 0.2}, {"id": "W2", "centreline": [[4, 0], [4, 3]], "thickness_m": 0.2}],
            "rooms": [{"id": "R1", "name": "A", "polygon": {"exterior": [[0, 0], [4, 0], [4, 3], [0, 3]], "holes": []}, "area_m2": 12.0}],
            "openings": [{"id": "O1", "kind": "door", "center": [1, 0], "width_m": 0.9}],
            "furniture": [{"id": "F1", "label": "Sofa", "semantic_label": "sofa", "footprint": [[1, 1], [3, 1], [3, 2], [1, 2]], "z_min": 0.0, "z_max": 0.8}]}
    same = compare_plans(plan, copy.deepcopy(plan))
    assert same["walls"]["matched"] == 2 and same["walls"]["shift"]["max"] == pytest.approx(0.0)
    assert same["rooms"]["pairs"][0]["iou"] == pytest.approx(1.0)
    moved = copy.deepcopy(plan)
    moved["walls"][1]["centreline"] = [[4.03, 0], [4.03, 3]]
    moved["openings"][0]["kind"] = "passage"
    diff = compare_plans(plan, moved)
    assert diff["walls"]["shift"]["max"] == pytest.approx(0.03, abs=1e-9)
    assert len(diff["walls"]["changed"]) == 1
    assert diff["openings"]["kind_changes"] == ["O1:door->passage"]
