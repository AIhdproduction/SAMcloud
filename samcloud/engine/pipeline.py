import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from samcloud.acquisition import CaptureValidationError, prepare_panorama_inputs


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


def resolve_input_images(args, outdir: Path, parser: argparse.ArgumentParser) -> tuple[Path, bool]:
    """Resolve normal images or prepare virtual perspective images for 360 media."""
    if args.capture_type != "panorama_360":
        if not args.images:
            parser.error("--images is required for drone_gps and camera capture types")
        images_dir = Path(args.images).resolve()
        if not images_dir.is_dir():
            parser.error(f"Image directory was not found: {images_dir}")
        return images_dir, False

    try:
        prepared = prepare_panorama_inputs(
            args.panorama_images or "",
            args.panorama_video or "",
            Path(args.panorama_workdir).resolve() if args.panorama_workdir else outdir / "inputs",
            args.frame_interval,
            args.cubemap_face_size,
        )
    except CaptureValidationError as error:
        parser.error(str(error))
    print(
        "Prepared 360 degree inputs: "
        f"{prepared.panorama_count} panoramas, {prepared.video_frame_count} video frames, "
        f"{prepared.cubemap_image_count} virtual camera images."
    )
    print(f"Panorama source manifest: {prepared.manifest_path}")
    return prepared.cubemap_directory, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-type", choices=("drone_gps", "camera", "panorama_360"), default="drone_gps")
    ap.add_argument("--images", help="folder with normal drone or camera photos")
    ap.add_argument("--panorama-images", help="folder with exported 2:1 JPG panoramas")
    ap.add_argument("--panorama-video", help="exported 2:1 MP4 panorama video")
    ap.add_argument("--panorama-workdir", help="project-owned directory for frames, cubemaps, and manifest")
    ap.add_argument("--frame-interval", type=float, default=1.0,
                    help="fixed 360 degree video frame interval in seconds")
    ap.add_argument("--cubemap-face-size", type=int, default=1024,
                    help="side length in pixels for generated cubemap faces")
    ap.add_argument("--outdir", required=True, help="project/output folder")
    ap.add_argument("--min-votes", type=int, default=1,
                    help="minimum agreeing images per point; default 1 keeps single-view classifications")
    ap.add_argument("--class-set", choices=("outdoor", "indoor"), default="outdoor",
                    help="semantic class set used for SAM3 classification")
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
    ap.add_argument(
        "--control-points-mode",
        choices=("none", "after_alignment"),
        default="none",
        help="continue automatically without control points, or stop after alignment",
    )
    ap.add_argument(
        "--stop-after-alignment",
        action="store_true",
        help="finish after sparse alignment so control points can be added",
    )
    ap.add_argument(
        "--skip-export",
        action="store_true",
        help="leave classified.ply ready for the manual LAS export step",
    )
    ap.add_argument(
        "--classified-output",
        default=None,
        help="path for the classified PLY (defaults to <outdir>/classified.ply)",
    )
    args = ap.parse_args()

    if args.control_points_mode == "after_alignment":
        args.stop_after_alignment = True

    if args.gps_max_error is not None and args.gps_max_error <= 0:
        ap.error("--gps-max-error must be greater than zero")
    if args.frame_interval <= 0:
        ap.error("--frame-interval must be greater than zero")
    if args.cubemap_face_size <= 0:
        ap.error("--cubemap-face-size must be greater than zero")
    gps_alignment_max_error = (
        args.gps_max_error if args.gps_max_error is not None
        else 0.1 if args.rtk else 3.0
    )

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    images_dir, is_panorama = resolve_input_images(args, outdir, ap)

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

    # 2) Video panoramas use sequential matching. A photo directory additionally
    # uses exhaustive matching so it can connect to the video frames as well.
    if is_panorama and args.panorama_video:
        run([colmap, "sequential_matcher",
             "--database_path", str(db_path),
             "--FeatureMatching.use_gpu", args.gpu,
             "--SequentialMatching.overlap", "10"])
    if not is_panorama or args.panorama_images:
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

    # 3b) GPS alignment only applies to drone images. Other capture types retain
    # a local frame until control points or another georeferencing step is used.
    if args.capture_type == "drone_gps":
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
                "colmap model_aligner flags for your installed COLMAP version."
            )
        model0 = model0_geo
    else:
        print("GPS alignment skipped. The reconstruction remains in LOCAL coordinates.")

    # 4) Undistort for dense stereo, this also writes a PINHOLE sparse model
    #    we can read as plain text for the camera poses.
    if args.stop_after_alignment:
        print(
            "Alignment finished. Waiting for control points before dense reconstruction."
        )
        return

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
    classified_output = (
        Path(args.classified_output).resolve()
        if args.classified_output
        else outdir / "classified.ply"
    )
    classified_output.parent.mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(classify_script),
         "--dense-dir", str(dense_dir),
         "--images-dir", str(dense_dir / "images"),
         "--model-dir", str(dense_sparse_txt),
         "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"),
         "--class-set", args.class_set,
         "--min-votes", str(args.min_votes),
         "--out", str(classified_output)])

    if args.skip_export:
        print(f"Finished. Classified point cloud ready for manual export: {classified_output}")
        return

    # 9) Export a standards-compliant LAS file with RGB and class IDs.
    las_script = Path(__file__).resolve().parent / "export_las.py"
    run([sys.executable, str(las_script),
         "--input", str(classified_output),
         "--output", str(outdir / "classified.las"),
         "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"),
         "--class-set", args.class_set,
         "--source-crs", args.source_crs,
         "--target-crs", args.export_crs])

    print(f"\nFinished. Classified point cloud: {classified_output}")
    print(f"LAS with class IDs: {outdir / 'classified.las'}")


if __name__ == "__main__":
    main()
