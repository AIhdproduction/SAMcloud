"""Regression tests for DJI RTK metadata parsing and selection."""

from __future__ import annotations

from pathlib import Path

from samcloud.engine.extract_gps import dji_position_stddevs, is_dji_rtk_project, parse_dji_rtk_xmp


def test_dji_xmp_fixed_rtk_metadata_is_parsed() -> None:
    jpeg = (
        b"\xff\xd8prefix<x:xmpmeta xmlns:drone-dji='http://www.dji.com/drone-dji/1.0/'>"
        b"<rdf:Description drone-dji:RtkFlag=\"50\" drone-dji:RtkStdLon=\"0.004\" "
        b"drone-dji:RtkStdLat=\"0.008\" drone-dji:RtkStdHgt=\"0.025\"/>"
        b"</x:xmpmeta>suffix\xff\xd9"
    )

    metadata = parse_dji_rtk_xmp(jpeg)

    assert metadata is not None
    assert (metadata.flag, metadata.std_lon_m, metadata.std_lat_m, metadata.std_hgt_m) == (50, 0.004, 0.008, 0.025)
    assert dji_position_stddevs(metadata, 3.0, 5.0) == (0.01, 0.01, 0.025)


def test_dji_xmp_missing_metadata_returns_none() -> None:
    assert parse_dji_rtk_xmp(b"\xff\xd8ordinary jpeg bytes without an XMP packet\xff\xd9") is None


def test_dji_float_rtk_uses_standard_gps_precision() -> None:
    metadata = parse_dji_rtk_xmp(b'<rdf:Description drone-dji:RtkFlag="34" drone-dji:RtkStdLon="0.01"/>')
    assert dji_position_stddevs(metadata, 3.0, 5.0) == (3.0, 3.0, 5.0)


def test_dji_rtk_project_requires_at_least_ninety_percent_fixed_images(tmp_path: Path) -> None:
    fixed = b'<rdf:Description drone-dji:RtkFlag="50" drone-dji:RtkStdLon="0.01" drone-dji:RtkStdLat="0.01" drone-dji:RtkStdHgt="0.02"/>'
    floating = b'<rdf:Description drone-dji:RtkFlag="34"/>'
    images = []
    for index in range(10):
        image = tmp_path / f"image_{index}.jpg"
        image.write_bytes(fixed if index < 9 else floating)
        images.append(image)

    assert is_dji_rtk_project(images)
    images[-2].write_bytes(floating)
    assert not is_dji_rtk_project(images)
