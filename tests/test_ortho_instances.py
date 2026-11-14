"""Behavioural coverage for optional orthographic instance rasters."""

from __future__ import annotations

import numpy as np

from samcloud.ortho.vector.footprints import object_instances
from samcloud.ortho.vector.raster_grid import RasterGrid


def _touching_tables(instance_raster: np.ndarray | None):
    grid = RasterGrid(0.0, 5.0, 0.05, 100, 100)
    dsm = np.zeros((100, 100), dtype=np.float32)
    classes = np.full((100, 100), 102, dtype=np.uint8)
    dsm[30:60, 20:80] = 0.75
    classes[30:60, 20:80] = 119
    fine = np.zeros_like(classes)
    fine[30:60, 20:80] = 1
    confidence = np.zeros_like(dsm)
    confidence[30:60, 20:80] = 1.0
    return object_instances(dsm, 0.0, classes, fine, confidence, ["table"], grid, 1.0, 2.6, instance_raster=instance_raster)


def test_touching_tables_split_by_instance_raster() -> None:
    raster = np.zeros((100, 100), dtype=np.uint32)
    raster[30:60, 20:50] = 11
    raster[30:60, 50:80] = 12
    instances, _metrics, _height = _touching_tables(raster)
    assert len(instances) == 2
    assert {item.instance_id for item in instances} == {11, 12}


def test_touching_tables_without_instance_raster_keep_legacy_result() -> None:
    instances, _metrics, _height = _touching_tables(None)
    assert len(instances) == 1
    assert instances[0].instance_id == 0


def test_instance_id_is_retained_on_footprint_object() -> None:
    raster = np.zeros((100, 100), dtype=np.uint32)
    raster[30:60, 20:50] = 11
    raster[30:60, 50:80] = 12
    instances, _metrics, _height = _touching_tables(raster)
    entity = {"type": "polyline", "_instance_id": instances[0].instance_id}
    assert entity["_instance_id"] in {11, 12}


def _unlabelled_pair(instance_labels):
    grid = RasterGrid(0.0, 5.0, 0.05, 100, 100)
    dsm = np.zeros((100, 100), dtype=np.float32)
    classes = np.full((100, 100), 102, dtype=np.uint8)
    dsm[30:60, 20:80] = 0.75
    classes[30:60, 20:80] = 119
    fine = np.zeros_like(classes)
    confidence = np.zeros_like(dsm)
    raster = np.zeros((100, 100), dtype=np.uint32)
    raster[30:60, 20:50] = 11
    raster[30:60, 50:80] = 12
    return object_instances(dsm, 0.0, classes, fine, confidence, [], grid, 1.0, 2.6, instance_raster=raster, instance_labels=instance_labels)


def test_instance_labels_name_split_objects() -> None:
    instances, metrics, _height = _unlabelled_pair({11: "table", 12: "chair"})
    assert {item.instance_id: item.label for item in instances} == {11: "table", 12: "chair"}
    assert all(item.has_fine_label for item in instances)
    assert metrics["instance_labels_applied"] == 2


def test_generic_and_opening_instance_labels_stay_unlabelled() -> None:
    instances, _metrics, _height = _unlabelled_pair({11: "furniture", 12: "window"})
    assert all(item.label is None and not item.has_fine_label for item in instances)


def test_no_instance_labels_keep_legacy_labels() -> None:
    instances, _metrics, _height = _unlabelled_pair(None)
    assert all(item.label is None for item in instances)


def test_fine_label_is_kept_over_instance_label() -> None:
    raster = np.zeros((100, 100), dtype=np.uint32)
    raster[30:60, 20:50] = 11
    raster[30:60, 50:80] = 12
    grid = RasterGrid(0.0, 5.0, 0.05, 100, 100)
    dsm = np.zeros((100, 100), dtype=np.float32)
    classes = np.full((100, 100), 102, dtype=np.uint8)
    dsm[30:60, 20:80] = 0.75
    classes[30:60, 20:80] = 119
    fine = np.zeros_like(classes)
    fine[30:60, 20:80] = 1
    confidence = np.zeros_like(dsm)
    confidence[30:60, 20:80] = 1.0
    instances, _metrics, _height = object_instances(dsm, 0.0, classes, fine, confidence, ["table"], grid, 1.0, 2.6, instance_raster=raster, instance_labels={11: "desk", 12: "desk"})
    assert {item.label for item in instances} == {"table"}


def test_sanitary_insert_gets_specific_german_label() -> None:
    from samcloud.ortho.plan import _fixture_display_name

    assert _fixture_display_name("Möbel", "BATHTUB", "bathtub") == "Badewanne"
    assert _fixture_display_name("Möbel", "BATHTUB", None) == "Badewanne"
    assert _fixture_display_name("Möbel", "WASHBASIN", "window") == "Lavabo"
    assert _fixture_display_name("Möbel", "WC", "toilet") == "WC"
    assert _fixture_display_name("Sofa", "WC", "toilet") == "Sofa"
