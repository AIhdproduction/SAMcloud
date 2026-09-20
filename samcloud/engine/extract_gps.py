"""
Reads GPS EXIF (lat/lon/alt) from every image in a folder and writes a
ref_images.txt in the format colmap model_aligner expects for --ref_is_gps 1:
    IMAGE_NAME  LAT  LON  ALT
Images without GPS EXIF are skipped with a warning by default. Use
--require-all to stop before georeferencing when any selected image lacks GPS.
"""

import argparse
import sys
from pathlib import Path

import exifread


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True, help="output ref_images.txt path")
    ap.add_argument("--require-all", action="store_true",
                    help="fail if any supported image has no readable GPS EXIF")
    args = ap.parse_args()

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
