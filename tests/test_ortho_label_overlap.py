"""resolve_label_overlaps: nearest free offset, shrink, room labels leave inserts, doors stay."""

from __future__ import annotations

import copy
import itertools
import json
import math

import pytest
from shapely import affinity
from shapely.geometry import Polygon, box

from samcloud.ortho.vector.labels import LabelSettings, resolve_label_overlaps

SETTINGS = LabelSettings()


def _text(text: str, x: float, y: float, upm: float, *, height: float = 0.12, rotation: float = 0.0, **extra: object) -> dict:
    return {"type": "text", "text": text, "point": [x * upm, y * upm], "height": height * upm, "rotation": rotation, **extra}


def _box(entity: dict, upm: float) -> Polygon:
    width = len(entity["text"]) * SETTINGS.char_width_factor * entity["height"]
    cx, cy = entity["point"]
    raw = box(cx - width / 2.0, cy - entity["height"] / 2.0, cx + width / 2.0, cy + entity["height"] / 2.0)
    return affinity.rotate(raw, entity["rotation"], origin=(cx, cy))


def _texts(layers: dict[str, list[dict]]) -> list[dict]:
    return [entity for entities in layers.values() for entity in entities if entity["type"] == "text"]


def _assert_no_overlap(layers: dict[str, list[dict]], upm: float) -> None:
    for first, second in itertools.combinations(_texts(layers), 2):
        assert _box(first, upm).intersection(_box(second, upm)).area <= 1e-6 * upm * upm, (first["text"], second["text"])


def _room(upm: float, width: float = 6.0, depth: float = 4.0) -> Polygon:
    return box(0.0, 0.0, width * upm, depth * upm)


def _stacked_scenario(upm: float) -> dict[str, list[dict]]:
    return {
        "A-ANNO": [
            _text("Wohnen / Essen 14.0 m2", 3.0, 2.0, upm, _room_name="Wohnen / Essen"),
            _text("Sofa h=0.80", 1.0, 1.0, upm),
            _text("Sofa h=0.80 x", 1.02, 1.0, upm),
        ],
        "A-DOOR": [_text("Tuer 0.89 m", 5.0, 3.0, upm, rotation=90.0)],
    }


@pytest.mark.parametrize("upm", [1.0, 0.8])
def test_stacked_furniture_labels_are_separated_and_room_label_stays(upm: float) -> None:
    layers = _stacked_scenario(upm)
    original = copy.deepcopy(layers)
    assert any(_box(a, upm).intersection(_box(b, upm)).area > 0 for a, b in itertools.combinations(_texts(layers), 2))
    metrics = resolve_label_overlaps(layers, [], [_room(upm)], upm, SETTINGS)
    _assert_no_overlap(layers, upm)
    assert layers["A-ANNO"][0]["point"] == original["A-ANNO"][0]["point"]
    assert layers["A-DOOR"] == original["A-DOOR"]
    moved = [(new, old) for new, old in zip(layers["A-ANNO"][1:], original["A-ANNO"][1:], strict=True) if new["point"] != old["point"]]
    assert moved and metrics["labels_moved"] == len(moved)
    for new, old in moved:
        shift = math.hypot(new["point"][0] - old["point"][0], new["point"][1] - old["point"][1])
        assert shift <= 0.6 * upm + 1e-9
    assert metrics["labels_unresolved"] == 0


@pytest.mark.parametrize("upm", [1.0, 0.8])
def test_room_label_over_insert_moves_inside_room_and_clears_it(upm: float) -> None:
    room = _room(upm, 3.0, 3.0)
    insert_footprint = box(1.0 * upm, 1.2 * upm, 2.0 * upm, 1.8 * upm)
    layers = {"A-ANNO": [_text("Bad 1.2 m2", 1.5, 1.5, upm, _room_name="Bad")]}
    assert _box(layers["A-ANNO"][0], upm).intersects(insert_footprint)
    metrics = resolve_label_overlaps(layers, [insert_footprint], [room], upm, SETTINGS)
    label = layers["A-ANNO"][0]
    assert metrics["labels_moved"] == 1 and metrics["labels_unresolved"] == 0
    assert label["point"] != [1.5 * upm, 1.5 * upm]
    assert room.covers(_box(label, upm))
    assert _box(label, upm).intersection(insert_footprint).area <= 1e-6 * upm * upm
    assert label["height"] == pytest.approx(0.12 * upm)


def test_room_label_without_free_spot_stays_and_is_counted() -> None:
    room = box(0.0, 0.0, 1.0, 1.0)
    obstacle = box(0.0, 0.0, 1.0, 1.0)
    layers = {"A-ANNO": [_text("Gaeste-WC 1.5 m2", 0.5, 0.5, 1.0, _room_name="Gaeste-WC")]}
    metrics = resolve_label_overlaps(layers, [obstacle], [room], 1.0, SETTINGS)
    assert layers["A-ANNO"][0]["point"] == [0.5, 0.5]
    assert metrics["labels_unresolved"] == 1 and metrics["labels_moved"] == 0


def test_door_label_is_never_moved() -> None:
    layers = {
        "A-DOOR": [_text("Tuer 0.89 m", 2.0, 2.0, 1.0)],
        "A-EINB": [_text("Regal h=1.50", 2.0, 2.0, 1.0)],
        "A-ANNO": [_text("Raum 4.0 m2", 2.0, 2.0, 1.0, _room_name="Raum")],
    }
    door_before = copy.deepcopy(layers["A-DOOR"])
    obstacle = box(1.5, 1.8, 2.5, 2.2)
    resolve_label_overlaps(layers, [obstacle], [_room(1.0)], 1.0, SETTINGS)
    assert layers["A-DOOR"] == door_before
    assert layers["A-EINB"][0]["point"] != [2.0, 2.0] or layers["A-EINB"][0]["height"] < 0.12


def test_label_shrinks_when_no_offset_is_free() -> None:
    # a corridor-sized room: the second label cannot be shifted anywhere inside, so it must shrink or stay
    room = box(0.0, 0.0, 1.4, 0.17)
    layers = {"A-ANNO": [
        _text("Regal", 0.7, 0.085, 1.0, height=0.12),
        _text("Regal", 0.7, 0.085, 1.0, height=0.12),
    ]}
    metrics = resolve_label_overlaps(layers, [], [room], 1.0, SETTINGS)
    assert metrics["labels_moved"] + metrics["labels_shrunk"] + metrics["labels_unresolved"] >= 1
    assert all(entity["height"] <= 0.12 for entity in layers["A-ANNO"])
    assert len(layers["A-ANNO"]) == 2  # never deleted


def test_non_overlapping_layout_is_byte_identical() -> None:
    layers = {
        "A-ANNO": [_text("Zimmer 9.0 m2", 1.0, 1.0, 1.0, _room_name="Zimmer"), _text("Sofa h=0.80", 4.0, 3.0, 1.0)],
        "A-EINB": [_text("Schrank", 5.0, 1.0, 1.0)],
        "A-DOOR": [_text("Tuer 0.89 m", 3.0, 0.2, 1.0, rotation=90.0)],
        "A-WALL": [{"type": "line", "start": [0.0, 0.0], "end": [6.0, 0.0]}],
    }
    before = json.dumps(layers, sort_keys=True)
    metrics = resolve_label_overlaps(layers, [box(5.5, 3.0, 5.9, 3.5)], [_room(1.0)], 1.0, SETTINGS)
    assert json.dumps(layers, sort_keys=True) == before
    assert metrics == {"labels_moved": 0, "labels_shrunk": 0, "labels_unresolved": 0}


def test_scaled_units_match_metric_result() -> None:
    metric = _stacked_scenario(1.0)
    scaled = _stacked_scenario(0.8)
    resolve_label_overlaps(metric, [], [_room(1.0)], 1.0, SETTINGS)
    resolve_label_overlaps(scaled, [], [_room(0.8)], 0.8, SETTINGS)
    for first, second in zip(_texts(metric), _texts(scaled), strict=True):
        assert second["point"] == pytest.approx([value * 0.8 for value in first["point"]])
        assert second["height"] == pytest.approx(first["height"] * 0.8)
