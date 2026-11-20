"""Capture-profile override of the dense max image size (``dense_size`` key)."""
from __future__ import annotations

from collections import namedtuple

import pytest
from PIL import Image

from samcloud.engine import pipeline
from samcloud.engine.pipeline import _dense_size_from_inputs

_Usage = namedtuple("_Usage", "total used free")


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """Project dir with one 8000 x 6000 image and plenty of disk space."""
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda _p: _Usage(1 << 50, 0, 1 << 50))
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8000, 6000)).save(images / "a.jpg")
    (tmp_path / "image_list.txt").write_text("a.jpg\nb.jpg\n", encoding="utf-8")
    return tmp_path, images


def _config(drone=None, **extra):
    config = {
        "max_image_size": "auto",
        "auto": {"ratio": 0.55, "minimum": 2000, "maximum": 3200},
        "capture_profiles": {"drone_gps": drone or {}, "camera": {}},
    }
    config.update(extra)
    return config


def test_no_override_matches_legacy(project):
    outdir, images = project
    config = _config()
    legacy = _dense_size_from_inputs(config, outdir, images)
    assert legacy[0] == 3200
    assert _dense_size_from_inputs(config, outdir, images, "drone_gps") == legacy
    assert _dense_size_from_inputs(config, outdir, images, None) == legacy


def test_drone_override_raises_auto_cap(project):
    outdir, images = project
    config = _config({"dense_size": {"max_image_size": "auto", "maximum": 4000}})
    size, _reason = _dense_size_from_inputs(config, outdir, images, "drone_gps")
    assert size == 4000


def test_explicit_integer_override_wins(project):
    outdir, images = project
    config = _config({"dense_size": {"max_image_size": 4000}})
    assert _dense_size_from_inputs(config, outdir, images, "drone_gps") == (4000, None)


def test_other_capture_types_unaffected(project):
    outdir, images = project
    config = _config({"dense_size": {"max_image_size": 4000}})
    assert _dense_size_from_inputs(config, outdir, images, "camera")[0] == 3200
    assert _dense_size_from_inputs(config, outdir, images, "panorama_360")[0] == 3200
    assert _dense_size_from_inputs(config, outdir, images)[0] == 3200
    assert config["max_image_size"] == "auto" and config["auto"]["maximum"] == 3200


@pytest.mark.parametrize("override", [
    {"max_image_size": "huge"},
    {"max_image_size": -5},
    {"max_image_size": "auto", "maximum": 0},
    {"max_image_size": "auto", "ratio": 2.0},
    {"max_image_size": "auto", "bogus": 1},
    "4000",
])
def test_invalid_override_raises_value_error(project, override):
    outdir, images = project
    config = _config({"dense_size": override})
    with pytest.raises(ValueError, match="dense_size"):
        _dense_size_from_inputs(config, outdir, images, "drone_gps")


def test_planned_dense_size_uses_drone_override(project):
    outdir, _images = project
    config = _config({"dense_size": {"max_image_size": "auto", "maximum": 4000}})
    planned = pipeline._planned_dense_size(config, 3200, 8000, 2, outdir, "drone_gps")
    assert planned == 4000
    assert pipeline._planned_dense_size(config, 3200, 8000, 2, outdir, "camera") == 3200
    assert pipeline._planned_dense_size(config, 3200, 8000, 2, outdir) == 3200
