import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(cmd, **kwargs):
    print("\n>>>", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}: {' '.join(str(c) for c in cmd)}")


def find_colmap_exe(colmap_bin):
    if colmap_bin:
        return colmap_bin
    found = shutil.which("colmap")
    if found:
        return found
    local = PROJECT_ROOT / "colmap" / "COLMAP.bat"
    if local.exists():
        return str(local)
    sys.exit(
        "colmap.exe not found. Either add it to PATH, put a portable COLMAP "
        "release in .\\colmap\\, or pass --colmap-bin explicitly."
    )


def write_image_list(images_dir, out_path):
    """Write image names relative to images_dir for COLMAP's recursive input mode."""
    extensions = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
    image_paths = sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions
    )
    if not image_paths:
        sys.exit(f"No supported images found in {images_dir}")
    out_path.write_text(
        "\n".join(p.relative_to(images_dir).as_posix() for p in image_paths) + "\n",
        encoding="utf-8",
    )
    return len(image_paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with the raw input photos")
    ap.add_argument("--outdir", required=True, help="project/output folder")
    ap.add_argument("--min-votes", type=int, default=1,
                    help="minimum agreeing images per point; default 1 keeps single-view classifications")
    ap.add_argument("--colmap-bin", default=None)
    ap.add_argument("--gpu", default="1", help="1 = use GPU for colmap steps, 0 = CPU only")
    ap.add_argument("--rtk", action="store_true",
                     help="photos were flown with RTK, GPS EXIF is cm-accurate. "
                          "Without this flag, standard consumer GPS accuracy (~meters) is assumed.")
    ap.add_argument("--gps-max-error", type=float, default=None,
                    help="maximum GPS alignment error in meters. Defaults to 0.1 for --rtk "
                         "and 3.0 for standard GPS.")
    ap.add_argument("--source-crs", default="LOCAL",
                    help="CRS of the reconstructed point cloud before LAS export")
    ap.add_argument("--export-crs", default="LOCAL",
                    help="target CRS of the exported LAS file")
    args = ap.parse_args()

    if args.gps_max_error is not None and args.gps_max_error <= 0:
        ap.error("--gps-max-error must be greater than zero")
    gps_alignment_max_error = (
        args.gps_max_error if args.gps_max_error is not None
        else 0.1 if args.rtk else 3.0
    )

    images_dir = Path(args.images).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    db_path = outdir / "database.db"
    sparse_dir = outdir / "sparse"
    dense_dir = outdir / "dense"
    sparse_dir.mkdir(exist_ok=True)
    dense_dir.mkdir(exist_ok=True)
    image_list_path = outdir / "image_list.txt"
    num_images = write_image_list(images_dir, image_list_path)
    print(f"Using {num_images} supported images from {images_dir}")

    colmap = find_colmap_exe(args.colmap_bin)

    # 1) Feature extraction
    run([colmap, "feature_extractor",
         "--database_path", str(db_path),
         "--image_path", str(images_dir),
         "--image_list_path", str(image_list_path),
         "--ImageReader.single_camera", "1",
         "--FeatureExtraction.use_gpu", args.gpu])

    # 2) Matching. Exhaustive is fine up to a few hundred images.
    run([colmap, "exhaustive_matcher",
         "--database_path", str(db_path),
         "--FeatureMatching.use_gpu", args.gpu])

    # 3) Sparse reconstruction (SfM)
    run([colmap, "mapper",
         "--database_path", str(db_path),
         "--image_path", str(images_dir),
         "--output_path", str(sparse_dir)])

    model0 = sparse_dir / "0"
    if not model0.exists():
        sys.exit("colmap mapper did not produce a reconstruction (sparse/0 missing). Check image overlap/quality.")

    # 3b) Georeference using GPS EXIF, otherwise the reconstruction is in an
    #     arbitrary scale/frame and downstream metric tiling (2cm grid, 10x10m
    #     tiles) breaks.
    gps_script = Path(__file__).resolve().parent / "extract_gps.py"
    gps_ref_path = outdir / "gps_ref.txt"
    run([sys.executable, str(gps_script),
         "--images-dir", str(images_dir),
         "--out", str(gps_ref_path),
         "--require-all"])

    print(
        "Using GPS alignment maximum error: "
        f"{gps_alignment_max_error:.3f} m "
        f"({'RTK' if args.rtk else 'standard GPS'})"
    )

    model0_geo = sparse_dir / "0_geo"
    model0_geo.mkdir(exist_ok=True)
    run([colmap, "model_aligner",
         "--input_path", str(model0),
         "--output_path", str(model0_geo),
         "--ref_images_path", str(gps_ref_path),
         "--ref_is_gps", "1",
         "--alignment_type", "enu",
         "--alignment_max_error", str(gps_alignment_max_error)])

    if not model0_geo.exists() or not any(model0_geo.iterdir()):
        sys.exit(
            "model_aligner produced no output. Check the gps_ref.txt and the "
            "colmap model_aligner flags for your installed COLMAP version "
            "(see README.md, this step is unverified)."
        )
    model0 = model0_geo

    # 4) Undistort for dense stereo, this also writes a PINHOLE sparse model
    #    we can read as plain text for the camera poses.
    run([colmap, "image_undistorter",
         "--image_path", str(images_dir),
         "--image_list_path", str(image_list_path),
         "--input_path", str(model0),
         "--output_path", str(dense_dir),
         "--output_type", "COLMAP"])

    # 5) Dense stereo
    run([colmap, "patch_match_stereo",
         "--workspace_path", str(dense_dir),
         "--PatchMatchStereo.geom_consistency", "1"])

    # 6) Fusion -> fused.ply + fused.ply.vis
    run([colmap, "stereo_fusion",
         "--workspace_path", str(dense_dir),
         "--output_path", str(dense_dir / "fused.ply")])

    # 7) Export the dense/sparse model as TEXT so classify.py can read poses
    #    without any COLMAP python bindings.
    dense_sparse_txt = dense_dir / "sparse_txt"
    dense_sparse_txt.mkdir(exist_ok=True)
    run([colmap, "model_converter",
         "--input_path", str(dense_dir / "sparse"),
         "--output_path", str(dense_sparse_txt),
         "--output_type", "TXT"])

    # 8) SAM3 classification + multi-view voting
    classify_script = Path(__file__).resolve().parent / "classify.py"
    run([sys.executable, str(classify_script),
         "--dense-dir", str(dense_dir),
         "--images-dir", str(dense_dir / "images"),
         "--model-dir", str(dense_sparse_txt),
         "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"),
         "--min-votes", str(args.min_votes),
         "--out", str(outdir / "classified.ply")])

    # 9) Export a standards-compliant LAS file with RGB and class IDs.
    las_script = Path(__file__).resolve().parent / "export_las.py"
    run([sys.executable, str(las_script),
         "--input", str(outdir / "classified.ply"),
         "--output", str(outdir / "classified.las"),
         "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"),
         "--source-crs", args.source_crs,
         "--target-crs", args.export_crs])

    print(f"\nFinished. Classified point cloud: {outdir / 'classified.ply'}")
    print(f"LAS with class IDs: {outdir / 'classified.las'}")


if __name__ == "__main__":
    main()
