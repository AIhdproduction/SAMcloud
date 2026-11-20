"""Configurable height datum of drone image altitudes (ellipsoidal | ln02 | egm96)."""

from __future__ import annotations

import numpy as np
import pytest

from samcloud.engine import extract_gps
from samcloud.engine.extract_gps import input_height_description, to_ellipsoidal_height
from samcloud.pointcloud import georeference as georef

LAT, LON, H_LN02 = 47.134, 8.755, 953.33


def _grids_available(source: str, target: str) -> bool:
    try:
        georef.strict_transformer(source, target)
    except ValueError:
        return False
    return True


def test_default_keeps_altitude_unchanged():
    assert to_ellipsoidal_height(LAT, LON, 512.345) == 512.345
    assert to_ellipsoidal_height(LAT, LON, 512.345, "ellipsoidal") == 512.345
    values = np.array([1.5, 2.5])
    assert np.array_equal(to_ellipsoidal_height([LAT] * 2, [LON] * 2, values), values)


def test_ln02_converts_and_round_trips():
    if not _grids_available("EPSG:4326+5728", "EPSG:4979"):
        pytest.skip("CHGeo2004 grid (LN02) not installed")
    ellipsoidal = to_ellipsoidal_height(LAT, LON, H_LN02, "ln02")
    assert ellipsoidal == pytest.approx(1001.2, abs=0.5)
    assert 46.0 < ellipsoidal - H_LN02 < 50.0
    back = georef.strict_transformer("EPSG:4979", "EPSG:2056+5728")
    _, _, height = back.transform(LON, LAT, ellipsoidal, errcheck=True)
    assert height == pytest.approx(H_LN02, abs=0.01)


def test_ln02_array_input():
    if not _grids_available("EPSG:4326+5728", "EPSG:4979"):
        pytest.skip("CHGeo2004 grid (LN02) not installed")
    alts = np.array([900.0, 953.33, 1000.0])
    result = to_ellipsoidal_height([LAT, LAT + 0.001, LAT + 0.002], [LON] * 3, alts, "ln02")
    assert result.shape == alts.shape
    assert np.all(result - alts > 40.0)


def test_egm96_converts_or_reports_missing_grid():
    if not _grids_available("EPSG:4326+5773", "EPSG:4979"):
        with pytest.raises(ValueError, match="grid|operation"):
            to_ellipsoidal_height(LAT, LON, 900.0, "egm96")
        pytest.skip("EGM96 grid not installed (clear error raised)")
    ellipsoidal = to_ellipsoidal_height(LAT, LON, 900.0, "egm96")
    assert 40.0 < ellipsoidal - 900.0 < 60.0


def test_gps_ref_writes_converted_altitude(tmp_path, monkeypatch):
    image_dir = tmp_path / "img"
    image_dir.mkdir()
    (image_dir / "a.jpg").write_bytes(b"x")
    monkeypatch.setattr(extract_gps, "read_gps", lambda path: (LAT, LON, 100.0))
    out = tmp_path / "gps_ref.txt"

    monkeypatch.setattr("sys.argv", ["extract_gps", "--images-dir", str(image_dir), "--out", str(out)])
    extract_gps.main()
    assert out.read_text().split()[3] == "100.000"

    if not _grids_available("EPSG:4326+5728", "EPSG:4979"):
        pytest.skip("CHGeo2004 grid (LN02) not installed")
    monkeypatch.setattr("sys.argv", ["extract_gps", "--images-dir", str(image_dir), "--out", str(out),
                                     "--input-height", "ln02"])
    extract_gps.main()
    assert float(out.read_text().split()[3]) > 140.0


def test_georeference_json_stores_height_reference(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    names = [f"img_{i}.jpg" for i in range(6)]
    enu = rng.uniform(-30.0, 30.0, size=(6, 3))
    enu[:, 2] = rng.uniform(0.0, 5.0, size=6)
    monkeypatch.setattr(georef, "camera_centres", lambda model_dir: dict(zip(names, enu)))
    ref = tmp_path / "gps_ref.txt"
    ref.write_text("\n".join(
        f"{n} {47.134 + p[1] * 9e-6:.8f} {8.755 + p[0] * 1.3e-5:.8f} {1001.2 + p[2]:.3f}"
        for n, p in zip(names, enu)
    ) + "\n")

    default = georef.fit_enu_to_ecef(tmp_path, ref)
    assert default["height_reference"] == georef.HEIGHT_REFERENCE
    description = input_height_description("ln02")
    custom = georef.fit_enu_to_ecef(tmp_path, ref, height_reference=description)
    out = tmp_path / "georeference.json"
    georef.save_georeference(out, custom)
    assert georef.load_georeference(out)["height_reference"] == description


@pytest.mark.parametrize("bad", ["wgs84", "", "LN03"])
def test_invalid_input_height_raises(bad):
    with pytest.raises(ValueError, match="input_height"):
        to_ellipsoidal_height(LAT, LON, 1.0, bad)
    with pytest.raises(ValueError, match="input_height"):
        input_height_description(bad)


def test_pipeline_helper_stores_configured_description(tmp_path, monkeypatch):
    from samcloud.engine import pipeline

    monkeypatch.setattr(
        pipeline, "load_global_config",
        lambda name: {"georeferencing": {"input_height": "ln02"}},
    )
    description = pipeline._configured_height_reference()
    assert description == input_height_description("ln02")

    names = [f"i{k}.jpg" for k in range(4)]
    enu = np.array([[0, 0, 0], [20, 3, 1], [4, 25, 2], [-15, 9, 4.0]])
    monkeypatch.setattr(georef, "camera_centres", lambda model_dir: dict(zip(names, enu)))
    ref = tmp_path / "gps_ref.txt"
    ref.write_text("\n".join(
        f"{n} {47.134 + p[1] * 9e-6:.8f} {8.755 + p[0] * 1.3e-5:.8f} {1001.2 + p[2]:.3f}"
        for n, p in zip(names, enu)) + "\n")
    result = georef.fit_enu_to_ecef(tmp_path, ref, height_reference=description)
    assert result["height_reference"] == description
