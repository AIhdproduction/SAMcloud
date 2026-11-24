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


# -- label size plausibility --------------------------------------------------------------------------------

def _footprint(width_px: int, depth_px: int, label: str | None):
    """An ObjectInstance whose mask is ``width_px`` x ``depth_px`` pixels of 5 cm."""
    from samcloud.ortho.vector.footprints import ObjectInstance

    mask = np.zeros((100, 100), dtype=bool)
    mask[20:20 + depth_px, 20:20 + width_px] = True
    return ObjectInstance(mask, label, 117, label is not None, 0.8, 1.0)


def _votes(instance, names: list[str], shares: dict[str, float], confidence: float = 0.9):
    """Fine-label raster over the instance mask: ``shares`` of the mask pixels per label name (rest unlabelled)."""
    fine = np.zeros(instance.mask.shape, dtype=np.uint8)
    pixels = np.flatnonzero(instance.mask.ravel())
    cursor = 0
    for name, share in shares.items():
        count = int(round(share * len(pixels)))
        fine.ravel()[pixels[cursor:cursor + count]] = names.index(name) + 1
        cursor += count
    return fine, np.where(fine > 0, confidence, 0.0).astype(np.float32)


def test_double_bed_named_nightstand_takes_the_next_plausible_label() -> None:
    from samcloud.ortho.vector.footprints import reconcile_label_sizes

    names = ["bed", "nightstand", "blanket", "cushion", "mattress", "clothes"]
    grid = RasterGrid(0.0, 5.0, 0.05, 100, 100)
    bed = _footprint(40, 36, "nightstand")  # 2.0 x 1.8 m
    fine, confidence = _votes(bed, names, {"blanket": 0.40, "bed": 0.20, "cushion": 0.15, "mattress": 0.05})
    assert reconcile_label_sizes([bed], fine, confidence, names, grid, 1.0) == 1
    assert bed.label == "bed" and bed.has_fine_label  # soft bedding on a bed-sized footprint is the bed
    # the votes are scale-safe: the same object in a 0.8157-unit scan
    scaled = _footprint(33, 29, "nightstand")  # 33 px * 0.05 / 0.8157 = 2.0 m
    fine, confidence = _votes(scaled, names, {"bed": 0.5})
    reconcile_label_sizes([scaled], fine, confidence, names, grid, 0.8157)
    assert scaled.label == "bed"


def test_a_small_footprint_keeps_its_nightstand_label_and_other_labels_are_untouched() -> None:
    from samcloud.ortho.vector.footprints import reconcile_label_sizes

    names = ["bed", "nightstand", "chair", "cushion", "wardrobe"]
    grid = RasterGrid(0.0, 5.0, 0.05, 100, 100)
    table = _footprint(14, 12, "nightstand")  # 0.7 x 0.6 m: a real bedside table
    fine, confidence = _votes(table, names, {"nightstand": 0.8})
    assert reconcile_label_sizes([table], fine, confidence, names, grid, 1.0) == 0 and table.label == "nightstand"
    # a long footprint whose only votes are soft goods is no bed (too narrow): the nightstand label goes, nothing replaces it
    narrow = _footprint(40, 16, "nightstand")  # 2.0 x 0.8 m
    fine, confidence = _votes(narrow, names, {"nightstand": 0.5, "cushion": 0.2})
    assert reconcile_label_sizes([narrow], fine, confidence, names, grid, 1.0) == 1
    assert narrow.label is None and not narrow.has_fine_label
    # a replacement label that cannot be that large either is skipped (another nightstand-sized label)
    other = _footprint(40, 36, "wardrobe")  # labels without a size limit are never touched
    fine, confidence = _votes(other, names, {"wardrobe": 0.9})
    assert reconcile_label_sizes([other], fine, confidence, names, grid, 1.0) == 0 and other.label == "wardrobe"
    disabled = _footprint(40, 36, "nightstand")
    fine, confidence = _votes(disabled, names, {"bed": 0.9})
    from samcloud.ortho.vector.footprints import LabelSizeSettings

    assert reconcile_label_sizes([disabled], fine, confidence, names, grid, 1.0, LabelSizeSettings(enabled=False)) == 0 and disabled.label == "nightstand"


def test_instance_label_nightstand_on_a_bed_sized_footprint_is_dropped_end_to_end() -> None:
    grid = RasterGrid(0.0, 5.0, 0.05, 100, 100)
    dsm = np.zeros((100, 100), dtype=np.float32)
    classes = np.full((100, 100), 102, dtype=np.uint8)
    dsm[30:66, 20:60] = 0.55  # 2.0 x 1.8 m
    classes[30:66, 20:60] = 117
    raster = np.zeros((100, 100), dtype=np.uint32)
    raster[30:66, 20:60] = 5
    zero = np.zeros_like(classes)
    instances, metrics, _height = object_instances(dsm, 0.0, classes, zero, np.zeros_like(dsm), [], grid, 1.0, 2.6, instance_raster=raster, instance_labels={5: "nightstand"})
    assert len(instances) == 1 and instances[0].label is None and not instances[0].has_fine_label
    assert metrics["instance_labels_applied"] == 1 and metrics["labels_size_corrected"] == 1
    # the same instance label on a small footprint stays
    dsm[:] = 0.0
    classes[:] = 102
    raster[:] = 0
    dsm[40:52, 30:44] = 0.55
    classes[40:52, 30:44] = 117
    raster[40:52, 30:44] = 5
    instances, metrics, _height = object_instances(dsm, 0.0, classes, zero, np.zeros_like(dsm), [], grid, 1.0, 2.6, instance_raster=raster, instance_labels={5: "nightstand"})
    assert [item.label for item in instances] == ["nightstand"] and metrics["labels_size_corrected"] == 0
