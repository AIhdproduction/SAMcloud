"""Dataset report for drone image sets: metadata, sharpness, duplicates, overlap, GSD.

Documents the input images before reconstruction. Nothing is removed: blurry
images and duplicate pairs are only flagged. The image folder is read-only.

CLI: ``python -m samcloud.drone.dataset_report --images <dir> --out <dir>``
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from samcloud.engine.extract_gps import parse_dji_rtk_xmp

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg"}
SENSOR_W_MM = 17.3
SENSOR_H_MM = 13.0
EARTH_R_M = 6371008.8
NADIR_PITCH_DEG = -80.0
BLUR_ABS_FLOOR = 15.0
BLUR_REL_FACTOR = 0.5
DUP_MAX_HAMMING = 6
DUP_MAX_GPS_M = 0.3
DUP_MAX_YAW_DEG = 10.0
MAX_RANGE_M = 150.0
HEIGHT_REFERENCE_NOTE = "unknown — verify ellipsoidal vs EGM96 vs LN02"
_XMP_READ_BYTES = 300_000
_ATTR = re.compile(r"((?:drone-dji|xmp|tiff|exif):[A-Za-z0-9]+)\s*=\s*[\"']([^\"']*)[\"']")

_PER_IMAGE_KEYS = (
    "file", "flight", "width", "height", "camera_model", "focal_mm", "focal_35mm_eq",
    "focal_px", "dewarp", "gimbal_pitch", "gimbal_yaw", "flight_yaw", "lat", "lon",
    "abs_alt", "rel_alt", "rtk_flag", "rtk_std_lon", "rtk_std_lat", "rtk_std_hgt",
    "timestamp", "altitude_type", "ntrip_mount", "gps_status",
    "sharpness", "phash", "footprint_area_m2", "gsd_cm", "dist_to_ground_m",
)


# ---------------------------------------------------------------- helpers

def _f(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max()),
            "mean": float(a.mean())}


def parse_xmp_fields(data: bytes) -> dict[str, str]:
    """Return all ``drone-dji:/xmp:/tiff:/exif:`` XMP attributes as a flat dict."""
    text = data[:_XMP_READ_BYTES].decode("utf-8", errors="ignore")
    return {m.group(1): m.group(2) for m in _ATTR.finditer(text)}


def phash64(grey: np.ndarray) -> int:
    """64-bit DCT perceptual hash (32x32 DCT, top-left 8x8 block, median threshold)."""
    small = cv2.resize(grey, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    block = cv2.dct(small)[:8, :8].flatten()
    med = float(np.median(block[1:]))
    bits = block > med
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def sharpness_of(grey: np.ndarray) -> float:
    """Variance of the Laplacian on the grey image downscaled to 1/4."""
    h, w = grey.shape[:2]
    small = cv2.resize(grey, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def _exif_basic(path: Path) -> dict:
    out: dict = {}
    try:
        import exifread
        with path.open("rb") as fh:
            tags = exifread.process_file(fh, details=False)
    except Exception:  # noqa: BLE001 - metadata is best effort
        return out
    model = tags.get("Image Model")
    if model:
        out["model"] = str(model).strip()
    fl = tags.get("EXIF FocalLength")
    if fl:
        try:
            out["focal_mm"] = float(fl.values[0].num) / float(fl.values[0].den)
        except Exception:  # noqa: BLE001
            pass
    f35 = tags.get("EXIF FocalLengthIn35mmFilm")
    if f35:
        out["focal_35"] = _f(str(f35))
    dt = tags.get("EXIF DateTimeOriginal")
    if dt:
        out["time"] = str(dt)
    return out


def analyse_image(path: Path, root: Path) -> dict:
    """Extract metadata, sharpness and pHash for one image (never raises)."""
    rec: dict = {k: None for k in _PER_IMAGE_KEYS}
    rel = path.relative_to(root)
    rec["file"] = rel.as_posix()
    rec["flight"] = rel.parent.as_posix() if str(rel.parent) != "." else "(root)"
    try:
        with path.open("rb") as fh:
            head = fh.read(_XMP_READ_BYTES)
    except OSError as exc:
        rec["error"] = f"read failed: {exc}"
        return rec
    x = parse_xmp_fields(head)
    exif = _exif_basic(path)
    rec["camera_model"] = x.get("tiff:Model") or x.get("drone-dji:DroneModel") or exif.get("model")
    rec["focal_mm"] = exif.get("focal_mm")
    rec["focal_35mm_eq"] = exif.get("focal_35")
    rec["dewarp"] = ("drone-dji:DewarpData" in x) if x else None
    rec["gimbal_pitch"] = _f(x.get("drone-dji:GimbalPitchDegree"))
    rec["gimbal_yaw"] = _f(x.get("drone-dji:GimbalYawDegree"))
    rec["flight_yaw"] = _f(x.get("drone-dji:FlightYawDegree"))
    rec["lat"] = _f(x.get("drone-dji:GpsLatitude"))
    rec["lon"] = _f(x.get("drone-dji:GpsLongitude"))
    rec["abs_alt"] = _f(x.get("drone-dji:AbsoluteAltitude"))
    rec["rel_alt"] = _f(x.get("drone-dji:RelativeAltitude"))
    rec["altitude_type"] = x.get("drone-dji:AltitudeType")
    rec["ntrip_mount"] = x.get("drone-dji:NTRIPMountPoint")
    rec["gps_status"] = x.get("drone-dji:GpsStatus")
    rtk = parse_dji_rtk_xmp(head)
    if rtk is not None:
        rec["rtk_flag"] = rtk.flag
        rec["rtk_std_lon"], rec["rtk_std_lat"], rec["rtk_std_hgt"] = (
            rtk.std_lon_m, rtk.std_lat_m, rtk.std_hgt_m)
    rec["timestamp"] = (x.get("drone-dji:UTCAtExposure") or x.get("xmp:CreateDate")
                        or exif.get("time"))
    calibrated = _f(x.get("drone-dji:CalibratedFocalLength"))
    cx = _f(x.get("drone-dji:CalibratedOpticalCenterX"))

    try:
        grey = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:  # noqa: BLE001
        grey = None
    if grey is None:
        rec["error"] = "image decode failed"
        return rec
    h, w = grey.shape[:2]
    rec["width"], rec["height"] = int(w), int(h)
    rec["sharpness"] = sharpness_of(grey)
    rec["phash"] = f"{phash64(grey):016x}"

    if calibrated is not None:
        scale = (w / (2.0 * cx)) if cx else 1.0
        rec["focal_px"] = calibrated * scale
    elif rec["focal_mm"]:
        rec["focal_px"] = rec["focal_mm"] * w / SENSOR_W_MM
    return rec


# ---------------------------------------------------------------- geometry

def _to_enu(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    x = math.radians(lon - lon0) * math.cos(math.radians(lat0)) * EARTH_R_M
    y = math.radians(lat - lat0) * EARTH_R_M
    return x, y


def camera_axes(pitch_deg: float, yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward, right, down unit vectors in ENU (yaw = bearing clockwise from north, roll 0)."""
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    h = np.array([math.sin(y), math.cos(y), 0.0])           # horizontal heading (E, N)
    fwd = np.array([h[0] * math.cos(p), h[1] * math.cos(p), math.sin(p)])
    right = np.array([math.cos(y), -math.sin(y), 0.0])
    down = np.cross(fwd, right)
    return fwd, right, down


def ground_footprint(pitch_deg: float, yaw_deg: float, height_m: float, focal_px: float,
                     width: int, height: int, max_range_m: float = MAX_RANGE_M,
                     samples: int = 8) -> np.ndarray:
    """Footprint polygon (N x 2, metres east/north of the camera) on the ground plane.

    The plane lies ``height_m`` below the camera. Rays above the horizon or hitting
    the ground farther than ``max_range_m`` (horizontally) are clipped to that range.
    """
    fwd, right, down = camera_axes(pitch_deg, yaw_deg)
    cx, cy = width / 2.0, height / 2.0
    edge = np.linspace(0.0, 1.0, samples, endpoint=False)
    pts = []
    corners = [(0, 0), (width, 0), (width, height), (0, height)]
    for i in range(4):
        (u0, v0), (u1, v1) = corners[i], corners[(i + 1) % 4]
        for t in edge:
            pts.append((u0 + (u1 - u0) * t, v0 + (v1 - v0) * t))
    out = []
    for u, v in pts:
        d = fwd + ((u - cx) / focal_px) * right + ((v - cy) / focal_px) * down
        if d[2] < -1e-9:
            s = height_m / -d[2]
            g = d[:2] * s
        else:
            g = d[:2] * 1e9
        r = float(np.hypot(*g))
        if r > max_range_m:
            g = g * (max_range_m / r)
        out.append(g)
    return np.asarray(out)


def _poly_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _ground_height(rec: dict) -> float | None:
    if rec["abs_alt"] is None or rec["rel_alt"] is None:
        return None
    return rec["abs_alt"] - rec["rel_alt"]


def compute_geometry(records: list[dict]) -> dict:
    """Add footprint area / GSD to the records and return the dataset overlap summary."""
    geo = [r for r in records if None not in (r["lat"], r["lon"], r["gimbal_pitch"],
                                              r["rel_alt"], r["focal_px"], r["width"])
           and r["gimbal_yaw"] is not None and r["rel_alt"] > 0.5]
    if not geo:
        return {"images_used": 0}
    lat0 = float(np.mean([r["lat"] for r in geo]))
    lon0 = float(np.mean([r["lon"] for r in geo]))
    polys = []
    for r in geo:
        h = float(r["rel_alt"])
        poly = ground_footprint(r["gimbal_pitch"], r["gimbal_yaw"], h, r["focal_px"],
                                r["width"], r["height"])
        r["footprint_area_m2"] = _poly_area(poly)
        sin_p = math.sin(math.radians(-r["gimbal_pitch"]))
        if sin_p > 1e-3:
            dist = h / sin_p
            r["dist_to_ground_m"] = dist
            r["gsd_cm"] = dist / r["focal_px"] * 100.0
        ex, ey = _to_enu(r["lat"], r["lon"], lat0, lon0)
        polys.append(poly + np.array([ex, ey]))
    allp = np.vstack(polys)
    xmin, ymin = np.floor(allp.min(axis=0))
    xmax, ymax = np.ceil(allp.max(axis=0))
    cell = 1.0
    while ((xmax - xmin) / cell) * ((ymax - ymin) / cell) > 25e6:
        cell *= 2.0
    nx, ny = int((xmax - xmin) / cell) + 1, int((ymax - ymin) / cell) + 1
    count = np.zeros((ny, nx), dtype=np.int32)
    for poly in polys:
        pix = np.round((poly - np.array([xmin, ymin])) / cell).astype(np.int32)
        mask = np.zeros((ny, nx), dtype=np.uint8)
        cv2.fillPoly(mask, [pix.reshape(-1, 1, 2)], 1)
        count += mask
    covered = count[count > 0]
    area = cell * cell
    summary = {
        "images_used": len(geo), "cell_size_m": cell, "ground_plane": "abs_alt - rel_alt",
        "max_range_m": MAX_RANGE_M, "covered_area_m2": float(covered.size * area),
        "overlap_median": float(np.median(covered)) if covered.size else 0.0,
        "overlap_p10": float(np.percentile(covered, 10)) if covered.size else 0.0,
        "overlap_p90": float(np.percentile(covered, 90)) if covered.size else 0.0,
    }
    for n in (3, 6, 9):
        summary[f"area_ge_{n}_views_m2"] = float((count >= n).sum() * area)
    return summary


# ---------------------------------------------------------------- dataset level

def _flight_summaries(records: list[dict], blur_flags: set[str]) -> dict:
    flights: dict[str, list[dict]] = {}
    for r in records:
        flights.setdefault(r["flight"], []).append(r)
    out = {}
    for name, rs in sorted(flights.items()):
        pitches = [r["gimbal_pitch"] for r in rs if r["gimbal_pitch"] is not None]
        flags = [r["rtk_flag"] for r in rs if r["rtk_flag"] is not None]
        takeoff = [r["abs_alt"] - r["rel_alt"] for r in rs
                   if r["abs_alt"] is not None and r["rel_alt"] is not None]

        def med(key):
            v = [r[key] for r in rs if r[key] is not None]
            return float(np.median(v)) if v else None

        out[name] = {
            "count": len(rs),
            "pitch": _stats(pitches),
            "nadir_count": sum(p <= NADIR_PITCH_DEG for p in pitches),
            "oblique_count": sum(p > NADIR_PITCH_DEG for p in pitches),
            "abs_alt": _stats([r["abs_alt"] for r in rs if r["abs_alt"] is not None]),
            "rel_alt": _stats([r["rel_alt"] for r in rs if r["rel_alt"] is not None]),
            "takeoff_height": _stats(takeoff),
            "rtk_fix_share": (sum(f == 50 for f in flags) / len(flags)) if flags else None,
            "rtk_median_std_m": {"lon": med("rtk_std_lon"), "lat": med("rtk_std_lat"),
                                 "hgt": med("rtk_std_hgt")},
            "sharpness": _stats([r["sharpness"] for r in rs if r["sharpness"] is not None]),
            "blur_flagged": sum(r["file"] in blur_flags for r in rs),
            "gsd_cm": _stats([r["gsd_cm"] for r in rs if r["gsd_cm"] is not None]),
        }
    return out


def find_blurry(records: list[dict], abs_floor: float = BLUR_ABS_FLOOR,
                rel_factor: float = BLUR_REL_FACTOR) -> list[dict]:
    """Flag images with sharpness < max(abs_floor, rel_factor * flight median)."""
    flights: dict[str, list[float]] = {}
    for r in records:
        if r["sharpness"] is not None:
            flights.setdefault(r["flight"], []).append(r["sharpness"])
    medians = {k: float(np.median(v)) for k, v in flights.items()}
    flagged = []
    for r in records:
        if r["sharpness"] is None:
            continue
        thr = max(abs_floor, rel_factor * medians[r["flight"]])
        if r["sharpness"] < thr:
            flagged.append({"file": r["file"], "sharpness": r["sharpness"], "threshold": thr})
    return flagged


def find_duplicates(records: list[dict]) -> list[dict]:
    """Flag pairs with pHash Hamming <= 6, GPS distance < 0.3 m and |yaw diff| < 10 deg."""
    pts = [r for r in records if r["phash"] and r["lat"] is not None and r["lon"] is not None
           and r["gimbal_yaw"] is not None]
    if not pts:
        return []
    lat0 = float(np.mean([r["lat"] for r in pts]))
    lon0 = float(np.mean([r["lon"] for r in pts]))
    xyz = np.array([[*_to_enu(r["lat"], r["lon"], lat0, lon0), r["abs_alt"] or 0.0]
                    for r in pts])
    hashes = [int(r["phash"], 16) for r in pts]
    pairs = []
    for i in range(len(pts)):
        d = np.linalg.norm(xyz[i + 1:] - xyz[i], axis=1)
        for k in np.nonzero(d < DUP_MAX_GPS_M)[0]:
            j = i + 1 + int(k)
            ham = bin(hashes[i] ^ hashes[j]).count("1")
            dyaw = _angle_diff(pts[i]["gimbal_yaw"], pts[j]["gimbal_yaw"])
            if ham <= DUP_MAX_HAMMING and dyaw < DUP_MAX_YAW_DEG:
                pairs.append({"a": pts[i]["file"], "b": pts[j]["file"], "hamming": ham,
                              "gps_distance_m": float(d[k]), "yaw_diff_deg": float(dyaw)})
    return pairs


def hardware_info(out_dir: Path) -> dict:
    """GPU (nvidia-smi), CPU count, RAM (psutil if present) and free disk of out_dir."""
    info: dict = {"cpu_count": os.cpu_count(), "gpu": None, "ram_gb": None, "free_disk_gb": None}
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False)
        if res.returncode == 0 and res.stdout.strip():
            info["gpu"] = [ln.strip() for ln in res.stdout.strip().splitlines()]
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        info["free_disk_gb"] = round(shutil.disk_usage(out_dir).free / 1e9, 1)
    except OSError:
        pass
    return info


def _height_reference(records: list[dict]) -> dict:
    per_flight = {}
    for name in sorted({r["flight"] for r in records}):
        rs = [r for r in records if r["flight"] == name]
        takeoff = [r["abs_alt"] - r["rel_alt"] for r in rs
                   if r["abs_alt"] is not None and r["rel_alt"] is not None]
        per_flight[name] = {
            "abs_alt_raw": _stats([r["abs_alt"] for r in rs if r["abs_alt"] is not None]),
            "abs_minus_rel_takeoff_height": _stats(takeoff),
        }

    def uniq(key):
        return sorted({r[key] for r in records if r[key]})

    return {
        "height_reference": HEIGHT_REFERENCE_NOTE,
        "xmp_datum_hints": {"AltitudeType": uniq("altitude_type"),
                            "NTRIPMountPoint": uniq("ntrip_mount"),
                            "GpsStatus": uniq("gps_status")},
        "per_flight": per_flight,
    }


def build_dataset_report(images_dir: str | Path, out_dir: str | Path, workers: int = 8) -> dict:
    """Analyse all JPEGs below ``images_dir`` and write dataset_report.json/.md to ``out_dir``."""
    root, out = Path(images_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    t0 = datetime.now(timezone.utc)
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 8))) as pool:
        records = list(pool.map(lambda p: analyse_image(p, root), files))
    overlap = compute_geometry(records)
    blur = find_blurry(records)
    dups = find_duplicates(records)
    blur_files = {b["file"] for b in blur}
    report = {
        "generated_utc": t0.isoformat(timespec="seconds"),
        "images_dir": str(root),
        "image_count": len(records),
        "flights": _flight_summaries(records, blur_files),
        "blur": {"abs_floor": BLUR_ABS_FLOOR, "rel_factor": BLUR_REL_FACTOR, "flagged": blur},
        "duplicates": {"max_hamming": DUP_MAX_HAMMING, "max_gps_m": DUP_MAX_GPS_M,
                       "max_yaw_deg": DUP_MAX_YAW_DEG, "pairs": dups},
        "overlap": overlap,
        "gsd_cm": _stats([r["gsd_cm"] for r in records if r["gsd_cm"] is not None]),
        "height": _height_reference(records),
        "hardware": hardware_info(out),
        "images": records,
        "elapsed_s": (datetime.now(timezone.utc) - t0).total_seconds(),
    }
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "dataset_report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


# ---------------------------------------------------------------- markdown

def _fmt(v, nd: int = 2) -> str:
    if v is None:
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _srange(s: dict | None, nd: int = 2) -> str:
    if not s:
        return "–"
    return f"{_fmt(s['min'], nd)} / {_fmt(s['median'], nd)} / {_fmt(s['max'], nd)}"


def render_markdown(rep: dict) -> str:
    """Render the report dict as concise German Markdown."""
    L = ["# Datensatz-Bericht Drohnenbilder", "",
         f"- Bilder: **{rep['image_count']}**  ·  Ordner: `{rep['images_dir']}`",
         f"- Erstellt (UTC): {rep['generated_utc']}  ·  Dauer: {_fmt(rep['elapsed_s'], 1)} s", "",
         "## Flüge", "",
         "| Flug | Bilder | Pitch min/med/max | Nadir | Schräg | Höhe abs min/med/max | RTK-Fix | Std lon/lat/hgt (m) | Blur | GSD med (cm) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, f in rep["flights"].items():
        std = f["rtk_median_std_m"]
        share = None if f["rtk_fix_share"] is None else f"{f['rtk_fix_share'] * 100:.0f} %"
        L.append(f"| {name} | {f['count']} | {_srange(f['pitch'], 1)} | {f['nadir_count']} | "
                 f"{f['oblique_count']} | {_srange(f['abs_alt'], 1)} | {_fmt(share)} | "
                 f"{_fmt(std['lon'], 3)} / {_fmt(std['lat'], 3)} / {_fmt(std['hgt'], 3)} | "
                 f"{f['blur_flagged']} | {_fmt((f['gsd_cm'] or {}).get('median'))} |")
    L += ["", "## Unschärfe", "",
          f"Schwelle: sharpness < max({rep['blur']['abs_floor']}, "
          f"{rep['blur']['rel_factor']} × Flug-Median). Markiert: **{len(rep['blur']['flagged'])}** (nichts entfernt).", ""]
    if rep["blur"]["flagged"]:
        L += ["| Datei | Schärfe | Schwelle |", "|---|---|---|"]
        L += [f"| {b['file']} | {b['sharpness']:.1f} | {b['threshold']:.1f} |"
              for b in rep["blur"]["flagged"][:50]]
        L.append("")
    d = rep["duplicates"]
    L += ["## Duplikate", "",
          f"pHash ≤ {d['max_hamming']}, GPS < {d['max_gps_m']} m, Yaw-Differenz < {d['max_yaw_deg']}°. "
          f"Markierte Paare: **{len(d['pairs'])}**.", ""]
    if d["pairs"]:
        L += ["| A | B | Hamming | GPS (m) | Yaw (°) |", "|---|---|---|---|---|"]
        L += [f"| {p['a']} | {p['b']} | {p['hamming']} | {p['gps_distance_m']:.3f} | {p['yaw_diff_deg']:.1f} |"
              for p in d["pairs"][:50]]
        L.append("")
    o = rep["overlap"]
    L += ["## Überlappung (Bodenebene)", ""]
    if o.get("images_used"):
        L += [f"- Bilder: {o['images_used']}  ·  Zelle: {o['cell_size_m']} m  ·  Bodenebene: {o['ground_plane']}  ·  max. Reichweite {o['max_range_m']} m",
              f"- Überlappung Median / P10 / P90: **{_fmt(o['overlap_median'], 1)} / {_fmt(o['overlap_p10'], 1)} / {_fmt(o['overlap_p90'], 1)}**",
              f"- Abgedeckte Fläche: {_fmt(o['covered_area_m2'], 0)} m²  ·  ≥3 Ansichten: {_fmt(o['area_ge_3_views_m2'], 0)} m²  ·  ≥6: {_fmt(o['area_ge_6_views_m2'], 0)} m²  ·  ≥9: {_fmt(o['area_ge_9_views_m2'], 0)} m²"]
    else:
        L.append("Nicht berechenbar (fehlende Metadaten).")
    g = rep["gsd_cm"]
    L += ["", "## GSD", "",
          f"min / median / max: {_srange(g, 2)} cm/px" if g else "Nicht berechenbar.", ""]
    h = rep["height"]
    L += ["## Höhenreferenz", "", f"- height_reference: **{h['height_reference']}**",
          f"- XMP-Hinweise: {json.dumps(h['xmp_datum_hints'], ensure_ascii=False)}", "",
          "| Flug | AbsoluteAltitude min/med/max (m) | Abs − Rel (Starthöhe) min/med/max (m) |",
          "|---|---|---|"]
    for name, v in h["per_flight"].items():
        L.append(f"| {name} | {_srange(v['abs_alt_raw'], 3)} | {_srange(v['abs_minus_rel_takeoff_height'], 3)} |")
    hw = rep["hardware"]
    L += ["", "## Hardware", "",
          f"- GPU: {', '.join(hw['gpu']) if hw['gpu'] else '–'}",
          f"- CPU-Kerne: {_fmt(hw['cpu_count'])}  ·  RAM: {_fmt(hw['ram_gb'], 1)} GB  ·  freier Speicher (Ausgabe-Laufwerk): {_fmt(hw['free_disk_gb'], 1)} GB", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Drone dataset report")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    rep = build_dataset_report(args.images, args.out, args.workers)
    logger.info("Report written: %d images in %.1f s", rep["image_count"], rep["elapsed_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
