"""
Reads GPS EXIF (lat/lon/alt) from every image in a folder and writes a
ref_images.txt in the format colmap model_aligner expects for --ref_is_gps 1:
    IMAGE_NAME  LAT  LON  ALT
Images without GPS EXIF are skipped with a warning by default. Use
--require-all to stop before georeferencing when any selected image lacks GPS.
"""

import argparse
from dataclasses import dataclass
import re
import sys
from pathlib import Path

import exifread


@dataclass(frozen=True)
class DjiRtkMetadata:
    """RTK precision values embedded by DJI in an image XMP packet."""

    flag: int
    std_lon_m: float | None
    std_lat_m: float | None
    std_hgt_m: float | None


_DJI_ATTRIBUTE = re.compile(
    r"(?:drone-dji:)?(?P<name>RtkFlag|RtkStdLon|RtkStdLat|RtkStdHgt)\s*=\s*[\"'](?P<value>[^\"']+)[\"']",
    flags=re.IGNORECASE,
)


def parse_dji_rtk_xmp(data: bytes) -> DjiRtkMetadata | None:
    """Read DJI RTK attributes from the first XMP packet bytes of a JPEG."""
    values = {
        match.group("name").casefold(): match.group("value")
        for match in _DJI_ATTRIBUTE.finditer(data[:300_000].decode("utf-8", errors="ignore"))
    }
    if "rtkflag" not in values:
        return None
    try:
        flag = int(float(values["rtkflag"]))
    except ValueError:
        return None
    return DjiRtkMetadata(
        flag=flag,
        std_lon_m=_xmp_float(values.get("rtkstdlon")),
        std_lat_m=_xmp_float(values.get("rtkstdlat")),
        std_hgt_m=_xmp_float(values.get("rtkstdhgt")),
    )


def read_dji_rtk_xmp(image_path: Path) -> DjiRtkMetadata | None:
    """Read DJI RTK metadata without loading the complete image file."""
    with image_path.open("rb") as image_file:
        return parse_dji_rtk_xmp(image_file.read(300_000))


def dji_position_stddevs(
    metadata: DjiRtkMetadata | None, standard_xy_m: float, standard_z_m: float,
) -> tuple[float, float, float]:
    """Choose per-image fixed-RTK precision or configured standard GPS values."""
    if metadata is not None and metadata.flag == 50 and all(
        value is not None for value in (metadata.std_lon_m, metadata.std_lat_m, metadata.std_hgt_m)
    ):
        return tuple(max(0.01, float(value)) for value in (
            metadata.std_lon_m, metadata.std_lat_m, metadata.std_hgt_m,
        ))
    return float(standard_xy_m), float(standard_xy_m), float(standard_z_m)


def is_dji_rtk_project(image_paths: list[Path]) -> bool:
    """Return true when at least 90 percent of the selected images are RTK fixed."""
    if not image_paths:
        return False
    fixed = 0
    for image_path in image_paths:
        try:
            metadata = read_dji_rtk_xmp(image_path)
        except OSError:
            metadata = None
        fixed += metadata is not None and metadata.flag == 50
    return fixed / len(image_paths) >= 0.90


def _xmp_float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _dms_to_decimal(dms, ref):
    degrees = float(dms[0].num) / float(dms[0].den)
    minutes = float(dms[1].num) / float(dms[1].den)
    seconds = float(dms[2].num) / float(dms[2].den)
    value = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        value = -value
    return value


def read_gps(image_path):
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    lat_tag = tags.get("GPS GPSLatitude")
    lat_ref = tags.get("GPS GPSLatitudeRef")
    lon_tag = tags.get("GPS GPSLongitude")
    lon_ref = tags.get("GPS GPSLongitudeRef")
    alt_tag = tags.get("GPS GPSAltitude")

    if not (lat_tag and lat_ref and lon_tag and lon_ref):
        return None

    lat = _dms_to_decimal(lat_tag.values, str(lat_ref))
    lon = _dms_to_decimal(lon_tag.values, str(lon_ref))
    alt = 0.0
    if alt_tag:
        alt = float(alt_tag.values[0].num) / float(alt_tag.values[0].den)

    return lat, lon, alt


INPUT_HEIGHT_CHOICES = ("ellipsoidal", "ln02", "egm96")
_VERTICAL_SOURCE_CRS = {"ln02": "EPSG:4326+5728", "egm96": "EPSG:4326+5773"}


_TRANSFORMER_CACHE: dict = {}


def input_height_description(input_height: str) -> str:
    """Describe the datum of the EXIF altitudes and how gps_ref.txt heights relate to it."""
    mode = _validate_input_height(input_height)
    if mode == "ellipsoidal":
        return "GPS altitude as recorded in the image EXIF (treated as ellipsoidal height)"
    names = {"ln02": "LN02 orthometric height (EPSG:5728)", "egm96": "EGM96 geoid height (EPSG:5773)"}
    return f"GPS altitude in image EXIF is {names[mode]}; converted once to ellipsoidal height (EPSG:4979)"


def _validate_input_height(input_height: str) -> str:
    mode = str(input_height).strip().lower()
    if mode not in INPUT_HEIGHT_CHOICES:
        raise ValueError(
            f"Invalid input_height '{input_height}'. Use one of: {', '.join(INPUT_HEIGHT_CHOICES)}."
        )
    return mode


def to_ellipsoidal_height(lat, lon, alt, input_height: str = "ellipsoidal"):
    """Convert altitudes of the given datum to ellipsoidal height (EPSG:4979).

    ``lat``/``lon``/``alt`` are scalars or arrays; the result has the shape of ``alt``.
    ``ellipsoidal`` returns ``alt`` unchanged. ``ln02`` and ``egm96`` need the matching PROJ
    geoid grid and raise ValueError with install instructions when it is missing.
    """
    import numpy as np

    mode = _validate_input_height(input_height)
    if mode == "ellipsoidal":
        return alt
    try:
        from samcloud.pointcloud.georeference import strict_transformer
    except ImportError:  # running as a plain script: make the repository root importable
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from samcloud.pointcloud.georeference import strict_transformer
    if mode not in _TRANSFORMER_CACHE:
        _TRANSFORMER_CACHE[mode] = strict_transformer(_VERTICAL_SOURCE_CRS[mode], "EPSG:4979")
    transformer = _TRANSFORMER_CACHE[mode]
    lon_a = np.atleast_1d(np.asarray(lon, dtype=float))
    lat_a = np.atleast_1d(np.asarray(lat, dtype=float))
    alt_a = np.atleast_1d(np.asarray(alt, dtype=float))
    _, _, height = transformer.transform(lon_a, lat_a, alt_a, errcheck=True)
    height = np.asarray(height, dtype=float)
    if not np.all(np.isfinite(height)):
        raise ValueError(
            f"Height conversion from {mode} returned invalid values; a required PROJ grid may be unavailable."
        )
    return height.reshape(np.shape(alt)) if np.ndim(alt) else float(height[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True, help="output ref_images.txt path")
    ap.add_argument("--require-all", action="store_true",
                    help="fail if any supported image has no readable GPS EXIF")
    ap.add_argument("--input-height", default="ellipsoidal", choices=INPUT_HEIGHT_CHOICES,
                    help="datum of the EXIF GPSAltitude; ln02/egm96 are converted to ellipsoidal height")
    args = ap.parse_args()
    # Fail early (before reading images) when a required grid is missing.
    to_ellipsoidal_height(47.0, 8.0, 0.0, args.input_height)

    images_dir = Path(args.images_dir)
    exts = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
    # COLMAP accepts image trees and retains names relative to --image_path.
    # The reference names must use exactly the same relative POSIX paths.
    image_paths = sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )

    lines = []
    missing = []
    for p in image_paths:
        try:
            gps = read_gps(p)
        except (OSError, ValueError, ZeroDivisionError):
            gps = None
        if gps is None:
            missing.append(p.relative_to(images_dir).as_posix())
            continue
        lat, lon, alt = gps
        alt = to_ellipsoidal_height(lat, lon, alt, args.input_height)
        image_name = p.relative_to(images_dir).as_posix()
        lines.append(f"{image_name} {lat:.8f} {lon:.8f} {alt:.3f}")

    if not lines:
        sys.exit(
            f"No GPS EXIF found in any image in {images_dir}. "
            "Without GPS tags there is nothing to georeference against, "
            "the reconstruction will stay in an arbitrary scale/frame."
        )

    if missing and args.require_all:
        preview = "\n".join(f"  {name}" for name in missing[:20])
        more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        sys.exit(
            f"{len(missing)} supported images have no readable GPS EXIF. "
            "Every image must have GPS before metric alignment.\n"
            f"{preview}{more}"
        )

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote GPS priors for {len(lines)} images to {args.out}")
    if missing:
        print(f"WARNING: {len(missing)} images had no GPS EXIF and were skipped:")
        for name in missing[:20]:
            print(f"  {name}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


if __name__ == "__main__":
    main()
