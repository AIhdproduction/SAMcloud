"""Tests for samcloud.drone.dataset_report with small synthetic JPEGs."""

import json
import math
from pathlib import Path

import cv2
import numpy as np

from samcloud.drone.dataset_report import (
    build_dataset_report, ground_footprint, parse_xmp_fields,
)

LAT, LON = 47.134627, 8.755046
M_PER_DEG_LAT = 111_195.0


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (150, 200), dtype=np.uint8)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


def _xmp(lat=LAT, lon=LON, pitch=-90.0, yaw=-150.3, abs_alt=520.0, rel_alt=20.0) -> bytes:
    xml = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" xmlns:tiff="http://ns.adobe.com/tiff/1.0/"
 tiff:Model="M3E"
 drone-dji:GpsLatitude="{lat:+.9f}" drone-dji:GpsLongitude="{lon:+.9f}"
 drone-dji:AbsoluteAltitude="{abs_alt:+.3f}" drone-dji:RelativeAltitude="{rel_alt:+.3f}"
 drone-dji:GimbalYawDegree="{yaw:+.2f}" drone-dji:GimbalPitchDegree="{pitch:+.2f}"
 drone-dji:FlightYawDegree="{yaw - 0.6:+.2f}" drone-dji:AltitudeType="RtkAlt"
 drone-dji:RtkFlag="50" drone-dji:RtkStdLon="0.01010" drone-dji:RtkStdLat="0.01162"
 drone-dji:RtkStdHgt="0.03957" drone-dji:DewarpData="2022-06-08;1,2"
 drone-dji:CalibratedFocalLength="200.0" drone-dji:CalibratedOpticalCenterX="100.0"
 drone-dji:CalibratedOpticalCenterY="75.0"
 drone-dji:UTCAtExposure="2025-07-14T12:23:33.549751"/></rdf:RDF></x:xmpmeta>"""
    return xml.encode("utf-8")


def _write(path: Path, img: np.ndarray, xmp: bytes | None) -> None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    data = buf.tobytes()
    if xmp is not None:
        payload = b"http://ns.adobe.com/xap/1.0/\x00" + xmp
        seg = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
        data = data[:2] + seg + data[2:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _rec(rep: dict, name: str) -> dict:
    return next(r for r in rep["images"] if r["file"].endswith(name))


def test_xmp_fields_parsed(tmp_path):
    _write(tmp_path / "in" / "f1" / "a.jpg", _texture(1), _xmp(pitch=-59.9, yaw=-150.3))
    rep = build_dataset_report(tmp_path / "in", tmp_path / "out", workers=2)
    r = _rec(rep, "a.jpg")
    assert r["camera_model"] == "M3E"
    assert r["flight"] == "f1"
    assert (r["width"], r["height"]) == (200, 150)
    assert abs(r["gimbal_pitch"] + 59.9) < 1e-6 and abs(r["gimbal_yaw"] + 150.3) < 1e-6
    assert abs(r["flight_yaw"] + 150.9) < 1e-6
    assert abs(r["lat"] - LAT) < 1e-7 and abs(r["lon"] - LON) < 1e-7
    assert r["abs_alt"] == 520.0 and r["rel_alt"] == 20.0
    assert r["rtk_flag"] == 50 and abs(r["rtk_std_hgt"] - 0.03957) < 1e-9
    assert r["dewarp"] is True and r["timestamp"].startswith("2025-07-14")
    assert len(r["phash"]) == 16
    assert parse_xmp_fields(b'drone-dji:RtkFlag="50"')["drone-dji:RtkFlag"] == "50"


def test_blurred_image_flagged(tmp_path):
    for i in range(3):
        _write(tmp_path / "in" / f"s{i}.jpg", _texture(10 + i),
               _xmp(lat=LAT + i * 1e-4))
    blurred = cv2.GaussianBlur(_texture(10), (0, 0), 4)
    _write(tmp_path / "in" / "blur.jpg", blurred, _xmp(lat=LAT + 5e-4))
    rep = build_dataset_report(tmp_path / "in", tmp_path / "out", workers=2)
    assert _rec(rep, "blur.jpg")["sharpness"] < 0.5 * _rec(rep, "s0.jpg")["sharpness"]
    flagged = {b["file"] for b in rep["blur"]["flagged"]}
    assert flagged == {"blur.jpg"}
    assert rep["image_count"] == 4  # flagged, never removed


def test_duplicates_flagged_shifted_not(tmp_path):
    base = _texture(5)
    noisy = np.clip(base.astype(np.int16) + np.random.default_rng(9).integers(-2, 3, base.shape),
                    0, 255).astype(np.uint8)
    _write(tmp_path / "in" / "a.jpg", base, _xmp())
    _write(tmp_path / "in" / "b.jpg", noisy, _xmp(lat=LAT + 0.05 / M_PER_DEG_LAT))
    _write(tmp_path / "in" / "c.jpg", _texture(77), _xmp(lat=LAT + 5.0 / M_PER_DEG_LAT))
    rep = build_dataset_report(tmp_path / "in", tmp_path / "out", workers=2)
    pairs = {(p["a"], p["b"]) for p in rep["duplicates"]["pairs"]}
    assert pairs == {("a.jpg", "b.jpg")}


def test_footprint_and_overlap_nadir(tmp_path):
    # pinhole: width 200 px, f = 200 px, 20 m above ground -> 20 m x 15 m = 300 m2
    poly = ground_footprint(-90.0, 37.0, 20.0, 200.0, 200, 150)
    xs, ys = poly[:, 0], poly[:, 1]
    area = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
    assert abs(area - 300.0) / 300.0 < 0.02
    assert abs(np.linalg.norm(poly, axis=1).max() - 12.5 * 1.0 * math.hypot(1, 0.75) * 20 / 12.5 / 2) < 0.5  # half diagonal 12.5 m
    _write(tmp_path / "in" / "a.jpg", _texture(1), _xmp(pitch=-90.0, yaw=0.0))
    _write(tmp_path / "in" / "b.jpg", _texture(2), _xmp(pitch=-90.0, yaw=0.0))
    rep = build_dataset_report(tmp_path / "in", tmp_path / "out", workers=2)
    for name in ("a.jpg", "b.jpg"):
        r = _rec(rep, name)
        assert abs(r["footprint_area_m2"] - 300.0) / 300.0 < 0.02
        assert abs(r["gsd_cm"] - 10.0) < 0.2        # 20 m / 200 px
    assert rep["overlap"]["overlap_median"] == 2.0
    assert rep["overlap"]["area_ge_3_views_m2"] == 0.0
    assert rep["overlap"]["covered_area_m2"] > 250.0
    assert rep["height"]["height_reference"].startswith("unknown")
    flight = rep["height"]["per_flight"]["(root)"]
    assert abs(flight["abs_minus_rel_takeoff_height"]["median"] - 500.0) < 1e-6


def test_json_and_md_written(tmp_path):
    _write(tmp_path / "in" / "f1" / "a.jpg", _texture(1), _xmp(pitch=-59.9))
    _write(tmp_path / "in" / "f2" / "b.jpg", _texture(2), _xmp(pitch=-90.0))
    out = tmp_path / "out"
    build_dataset_report(tmp_path / "in", out, workers=2)
    data = json.loads((out / "dataset_report.json").read_text(encoding="utf-8"))
    assert set(data["flights"]) == {"f1", "f2"}
    f1, f2 = data["flights"]["f1"], data["flights"]["f2"]
    assert f1["count"] == 1 and f1["oblique_count"] == 1 and f1["nadir_count"] == 0
    assert f2["nadir_count"] == 1 and f1["rtk_fix_share"] == 1.0
    assert "hardware" in data and "cpu_count" in data["hardware"]
    md = (out / "dataset_report.md").read_text(encoding="utf-8")
    assert "Flüge" in md and "f1" in md and "Höhenreferenz" in md


def test_missing_xmp_handled(tmp_path):
    _write(tmp_path / "in" / "noxmp.jpg", _texture(3), None)
    _write(tmp_path / "in" / "ok.jpg", _texture(4), _xmp())
    (tmp_path / "in" / "broken.jpg").write_bytes(b"not a jpeg")
    rep = build_dataset_report(tmp_path / "in", tmp_path / "out", workers=2)
    assert rep["image_count"] == 3
    r = _rec(rep, "noxmp.jpg")
    assert r["gimbal_pitch"] is None and r["lat"] is None and r["rtk_flag"] is None
    assert r["sharpness"] is not None and r["phash"] is not None and r["footprint_area_m2"] is None
    b = _rec(rep, "broken.jpg")
    assert b["sharpness"] is None and "error" in b
    assert math.isfinite(_rec(rep, "ok.jpg")["sharpness"])
