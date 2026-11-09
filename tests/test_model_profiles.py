"""Behaviour tests for Cyclone model profiles and LAS provenance."""

import json
from pathlib import Path
import shutil

import laspy
import numpy as np
import pytest

from samcloud.classification.profiles import (
    UnsupportedProfileModeError,
    detect_mode,
    flurseg_profile,
    load_cyclone_codes,
    resolve_las_profile,
    sam3_profile,
    translate_legacy_codes,
    validate_profile,
    legacy_translation,
)


def _config() -> dict:
    return json.loads(Path("config/classes.json").read_text(encoding="utf-8"))


def test_profiles_use_unique_non_reserved_codes() -> None:
    registry = load_cyclone_codes()
    for profile in (sam3_profile("outdoor", _config()), sam3_profile("indoor", _config()), flurseg_profile("outdoor")):
        validate_profile(profile)
        codes = [entry.cyclone_code for entry in profile.entries]
        assert len(codes) == len(set(codes))
        assert all(code in registry and not registry[code].get("reserved") for code in codes)


def test_flurseg_profile_reads_class_file_and_rejects_indoor() -> None:
    profile = flurseg_profile("outdoor", Path("tests/fixtures/flurseg_classes.yaml"))
    assert [(entry.model_name, entry.cyclone_code) for entry in profile.entries][-2:] == [("road_marking", 82), ("utility_cover", 75)]
    with pytest.raises(UnsupportedProfileModeError):
        flurseg_profile("indoor")


def test_resolve_profile_prefers_sidecar_then_vlr(tmp_path: Path) -> None:
    las_path = tmp_path / "cloud.las"
    las = laspy.LasData(laspy.LasHeader(point_format=7, version="1.4"))
    las.x, las.y, las.z = [0], [0], [0]
    las.header.vlrs.append(laspy.VLR("colmap_sam", 1, "profile", json.dumps({"model_profile": "sam3_outdoor", "mode": "outdoor", "class_mapping": {"2": "ground"}}).encode()))
    las.write(las_path)
    assert resolve_las_profile(las_path).profile_name == "sam3_outdoor"
    (tmp_path / "cloud.las.profile.json").write_text(json.dumps({"profile_name": "flurseg_v1", "mode": "outdoor", "class_set": "outdoor", "code_to_name": {}}), encoding="utf-8")
    assert resolve_las_profile(las_path).profile_name == "flurseg_v1"
    bare_path = tmp_path / "bare.las"
    bare = laspy.LasData(laspy.LasHeader(point_format=7, version="1.4"))
    bare.x, bare.y, bare.z = [0], [0], [0]
    bare.write(bare_path)
    assert resolve_las_profile(bare_path) is None


def test_detect_mode_uses_configured_share() -> None:
    assert detect_mode(np.array([100, 101, 102, 2]), {"mode_detection": {"indoor_share": 0.15}}) == "indoor"
    assert detect_mode(np.array([100, 2, 3, 5]), {"mode_detection": {"indoor_share": 0.5}}) == "outdoor"


def test_flurseg_fixture_has_the_complete_profile_table() -> None:
    profile = flurseg_profile("outdoor", Path("tests/fixtures/flurseg_classes.yaml"))
    assert [(entry.model_id, entry.model_name, entry.cyclone_code) for entry in profile.entries] == [
        (0, "other", 64), (1, "ground", 2), (2, "paved_surface", 71), (3, "low_vegetation", 3),
        (4, "tree", 5), (5, "building", 6), (6, "vehicle", 91), (7, "construction", 151),
        (8, "curb", 74), (9, "road_marking", 82), (10, "utility_cover", 75),
    ]


@pytest.mark.parametrize(("mode", "old_codes", "expected"), [
    ("outdoor", [0, 2, 11, 66, 67, 3, 5, 6, 68, 69, 70, 71, 72, 14, 73, 74, 9], [1, 2, 11, 73, 74, 3, 5, 6, 91, 151, 84, 81, 77, 72, 14, 180, 9]),
    ("indoor", [0, 2, 75, 6, 76, 77, 78, 79, 80, 81, 82], [1, 180, 100, 101, 102, 103, 104, 105, 107, 108, 109]),
])
def test_legacy_translation_matches_every_old_code(mode: str, old_codes: list[int], expected: list[int]) -> None:
    target = sam3_profile(mode, _config())
    mapping = legacy_translation(mode, target)
    assert [mapping[code] for code in old_codes] == expected


def test_fixture_based_flurseg_profile_uses_unique_valid_cyclone_codes(tmp_path: Path) -> None:
    classes_path = tmp_path / "model" / "classes.yaml"
    classes_path.parent.mkdir()
    shutil.copyfile(Path("tests/fixtures/flurseg_classes.yaml"), classes_path)

    profile = flurseg_profile("outdoor", classes_path)
    registry = load_cyclone_codes()
    codes = [entry.cyclone_code for entry in profile.entries]

    validate_profile(profile)
    assert len(codes) == len(set(codes))
    assert all(code in registry and not registry[code].get("reserved") for code in codes)


def test_legacy_vlr_translation_uses_names_from_a_real_las(tmp_path: Path) -> None:
    las_path = tmp_path / "legacy.las"
    las = laspy.LasData(laspy.LasHeader(point_format=7, version="1.4"))
    las.x, las.y, las.z = [0, 1], [0, 0], [0, 0]
    las.classification = [2, 3]
    class_mapping = {"2": "tree", "3": "ground"}
    las.header.vlrs.append(laspy.VLR(
        "colmap_sam", 1, "legacy mapping", json.dumps({"class_mapping": class_mapping}).encode("utf-8"),
    ))
    las.write(las_path)

    stored = laspy.read(las_path)
    vlr = next(item for item in stored.header.vlrs if item.user_id == "colmap_sam" and item.record_id == 1)
    mapping = json.loads(bytes(vlr.record_data).decode("utf-8"))["class_mapping"]
    translated = translate_legacy_codes(stored.classification, "outdoor", sam3_profile("outdoor", _config()), mapping)

    assert translated.tolist() == [5, 2]
