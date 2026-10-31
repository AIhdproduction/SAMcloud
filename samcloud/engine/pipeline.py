import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from samcloud.acquisition import CaptureValidationError, prepare_panorama_inputs
from samcloud.core.configuration import load_global_config
from samcloud.engine.run_summary import RunSummary
from samcloud.control_points.georeference import georeference_from_control_points, save_control_point_report
from samcloud.control_points.model_transform import transform_text_model
from samcloud.pointcloud.georeference import fit_enu_to_ecef, format_residuals, save_georeference


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProgressTracker:
    """Persist coarse pipeline progress for the local dashboard."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.started_at = time.time()

    def update(self, percent: float, stage: str, detail: str = "") -> None:
        payload = {
            "percent": round(max(0.0, min(100.0, float(percent))), 1),
            "stage": stage,
            "detail": detail,
            "started_at": self.started_at,
            "updated_at": time.time(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload), encoding="utf-8")


def run(cmd, *, summary: RunSummary | None = None, stage: str | None = None, **kwargs):
    print("\n>>>", " ".join(str(c) for c in cmd))
    started_at = time.perf_counter()
    result = subprocess.run(cmd, **kwargs)
    if summary and stage:
        summary.record_stage(stage, time.perf_counter() - started_at)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}: {' '.join(str(c) for c in cmd)}")


def find_colmap_exe(colmap_bin):
    if colmap_bin:
        return colmap_bin
    local = PROJECT_ROOT / "colmap" / "COLMAP.bat"
    if local.exists():
        return str(local)
    found = shutil.which("colmap")
    if found:
        return found
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


def _resume_after_control_points(
    args, outdir: Path, report_dir: Path, summary: RunSummary, progress: ProgressTracker | None,
    colmap_config: dict, dense_max_image_size: int, dense_num_iterations: int,
    dense_gpu_index: int, classified_output: Path,
) -> None:
    """Run Phase B without repeating feature extraction, mapping, or undistortion."""
    dense_dir = outdir / "dense"
    dense_sparse = dense_dir / "sparse"
    sparse_txt = dense_dir / "sparse_txt"
    images_dir = dense_dir / "images"
    missing = [str(path) for path in (dense_sparse, sparse_txt, images_dir) if not path.exists()]
    if missing:
        raise ValueError(
            "Cannot resume after control points; Phase A output is missing: " + ", ".join(missing)
        )
    pristine_sparse = dense_dir / "sparse_pre_gcp"
    pristine_sparse_txt = dense_dir / "sparse_txt_pre_gcp"
    _create_pristine_gcp_backups(dense_sparse, sparse_txt, pristine_sparse, pristine_sparse_txt)
    control_points_dir = Path(args.control_points_dir).resolve() if args.control_points_dir else outdir.parent / "control_points"
    settings = dict(load_global_config("defaults.yml").get("control_points", {}))
    gps_residuals = None
    gps_georeference = outdir / "georeference.json"
    if gps_georeference.exists():
        try:
            gps_residuals = json.loads(gps_georeference.read_text(encoding="utf-8")).get("residual_m")
        except (OSError, json.JSONDecodeError):
            pass
    georeference, report, transform = georeference_from_control_points(
        pristine_sparse_txt, control_points_dir / "control_points.csv", control_points_dir / "observations.json",
        args.control_points_crs, settings, gps_residuals=gps_residuals,
    )
    save_georeference(outdir / "georeference.json", georeference)
    save_control_point_report(report_dir / "control_points_report.json", report)
    transformed_txt = dense_dir / "sparse_txt_gcp"
    transform_text_model(pristine_sparse_txt, transformed_txt, *transform)
    transformed_sparse = dense_dir / "sparse_gcp"
    transformed_sparse.mkdir(parents=True, exist_ok=True)
    colmap = find_colmap_exe(args.colmap_bin)
    run([colmap, "model_converter", "--input_path", str(transformed_txt), "--output_path", str(transformed_sparse), "--output_type", "BIN"], summary=summary, stage="Applying control points")
    for path in transformed_sparse.iterdir():
        if path.is_file():
            shutil.copy2(path, dense_sparse / path.name)
    shutil.rmtree(transformed_sparse)
    shutil.rmtree(transformed_txt)
    _clear_stereo_cache(dense_dir)
    if progress:
        progress.update(62, "Building dense cloud")
    run([colmap, "patch_match_stereo", "--workspace_path", str(dense_dir),
         "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.gpu_index", str(dense_gpu_index),
         "--PatchMatchStereo.num_iterations", str(dense_num_iterations),
         "--PatchMatchStereo.max_image_size", str(dense_max_image_size)], summary=summary, stage="Dense stereo")
    run([colmap, "stereo_fusion", "--workspace_path", str(dense_dir), "--output_path", str(dense_dir / "fused.ply")], summary=summary, stage="Dense fusion")
    shutil.rmtree(sparse_txt)
    sparse_txt.mkdir(parents=True, exist_ok=True)
    run([colmap, "model_converter", "--input_path", str(dense_sparse), "--output_path", str(sparse_txt), "--output_type", "TXT"], summary=summary, stage="Preparing classification")
    _classify_and_finish(args, outdir, dense_dir, sparse_txt, report_dir, summary, progress, classified_output, True)


def _create_pristine_gcp_backups(
    sparse: Path, sparse_txt: Path, pristine_sparse: Path, pristine_sparse_txt: Path
) -> None:
    """Capture Phase-A model inputs once so revised clicks never double-transform them."""
    if not pristine_sparse.exists():
        shutil.copytree(sparse, pristine_sparse)
    if not pristine_sparse_txt.exists():
        shutil.copytree(sparse_txt, pristine_sparse_txt)


def _clear_stereo_cache(dense_dir: Path) -> None:
    """Discard depths/normals because COLMAP may otherwise reuse old geometry."""
    stereo = dense_dir / "stereo"
    for name in ("depth_maps", "normal_maps"):
        cache = stereo / name
        if cache.exists():
            shutil.rmtree(cache)


def _classify_and_finish(
    args, outdir: Path, dense_dir: Path, dense_sparse_txt: Path, report_dir: Path,
    summary: RunSummary, progress: ProgressTracker | None, classified_output: Path, georeferenced: bool,
) -> None:
    """Run the unchanged classification, refinement, and optional export tail."""
    classify_script = Path(__file__).resolve().parent / "classify.py"
    geometry_enabled = args.geometry_refinement == "1"
    classified_raw = classified_output.with_name("classified_raw.ply") if geometry_enabled else classified_output
    classified_scores = classified_output.parent / "classified_scores.npz"
    classified_output.parent.mkdir(parents=True, exist_ok=True)
    classify_command = [sys.executable, str(classify_script), "--dense-dir", str(dense_dir),
        "--images-dir", str(dense_dir / "images"), "--model-dir", str(dense_sparse_txt),
        "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"), "--class-set", args.class_set,
        "--min-votes", str(args.min_votes), "--tile-size", str(args.sam_tile_size),
        "--tile-min-overlap", str(args.sam_tile_min_overlap), "--mode", args.classification_mode,
        "--background-threshold", str(args.background_threshold), "--query-threshold", str(args.query_threshold),
        "--summary-file", str(report_dir / "classification_summary.json"), "--out", str(classified_raw)]
    if args.threshold_sweep:
        classify_command.extend(["--threshold-sweep", args.threshold_sweep])
    if geometry_enabled and args.classification_mode == "segearth_ov3":
        classify_command.extend(["--scores-out", str(classified_scores)])
    if args.progress_file:
        classify_command.extend(["--progress-file", str(args.progress_file)])
    run(classify_command, summary=summary, stage="SAM3 classification")
    if geometry_enabled and georeferenced:
        refine_script = Path(__file__).resolve().parent / "geometry" / "refine.py"
        run([sys.executable, str(refine_script), "--input", str(classified_raw), "--output", str(classified_output),
            "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"), "--class-set", args.class_set,
            "--config", str(PROJECT_ROOT / "config" / "geometry.yml"), "--summary-file", str(report_dir / "geometry_summary.json"),
            *(["--scores", str(classified_scores)] if args.classification_mode == "segearth_ov3" else [])], summary=summary, stage="Geometric refinement")
    elif geometry_enabled:
        shutil.copy2(classified_raw, classified_output)
    if args.skip_export:
        if progress:
            progress.update(100, "Complete")
        summary.write("Completed")
        return
    run([sys.executable, str(Path(__file__).resolve().parent / "export_las.py"), "--input", str(classified_output),
         "--output", str(outdir / "classified.las"), "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"),
         "--class-set", args.class_set, "--source-crs", args.source_crs, "--target-crs", args.export_crs,
         "--georeference", str(outdir / "georeference.json")], summary=summary, stage="LAS export")
    summary.write("Completed")


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
    ap.add_argument("--sam-tile-size", type=int, default=1008,
                    help="SAM3 input tile size in pixels")
    ap.add_argument("--sam-tile-min-overlap", type=float, default=0.2,
                    help="SAM3 tile overlap fraction")
    ap.add_argument("--classification-mode", choices=("legacy", "segearth_ov3"), default="segearth_ov3",
                    help="SAM3 classification path (default: segearth_ov3; legacy remains available for comparison)")
    ap.add_argument("--background-threshold", type=float, default=0.4,
                    help="SegEarth-OV3 background threshold")
    ap.add_argument("--query-threshold", type=float, default=0.4,
                    help="SegEarth-OV3 decoder query threshold")
    ap.add_argument("--threshold-sweep", default="",
                    help="optional comma-separated SegEarth-OV3 background thresholds")
    ap.add_argument("--class-set", choices=("outdoor", "indoor"), default="outdoor",
                    help="semantic class set used for SAM3 classification")
    ap.add_argument("--geometry-refinement", choices=("0", "1"), default="1",
                    help="1 = apply post-classification height and flatness checks to georeferenced outdoor clouds (default on; skipped for non-georeferenced projects)")
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
    ap.add_argument(
        "--resume-after-control-points",
        action="store_true",
        help="reuse the undistorted dense workspace and continue from marked control points",
    )
    ap.add_argument("--control-points-dir", default=None, help="project control_points directory")
    ap.add_argument("--control-points-crs", default="EPSG:2056+5728")
    ap.add_argument(
        "--progress-file",
        default=None,
        help="path for the dashboard progress JSON file",
    )
    args = ap.parse_args()

    if args.control_points_mode == "after_alignment":
        args.stop_after_alignment = True
    if args.resume_after_control_points and args.stop_after_alignment:
        ap.error("--resume-after-control-points cannot be combined with --stop-after-alignment")

    progress = ProgressTracker(Path(args.progress_file)) if args.progress_file else None
    if progress:
        progress.update(0, "Preparing images")

    if args.gps_max_error is not None and args.gps_max_error <= 0:
        ap.error("--gps-max-error must be greater than zero")
    if args.frame_interval <= 0:
        ap.error("--frame-interval must be greater than zero")
    if args.cubemap_face_size <= 0:
        ap.error("--cubemap-face-size must be greater than zero")
    if args.sam_tile_size not in {768, 1008, 1280}:
        ap.error("--sam-tile-size must be 768, 1008, or 1280")
    if not 0.0 <= args.sam_tile_min_overlap <= 0.8:
        ap.error("--sam-tile-min-overlap must be between 0.00 and 0.80")
    if not 0.0 <= args.background_threshold <= 1.0:
        ap.error("--background-threshold must be between 0 and 1")
    if not 0.0 <= args.query_threshold <= 1.0:
        ap.error("--query-threshold must be between 0 and 1")
    try:
        threshold_sweep = [float(item.strip()) for item in args.threshold_sweep.split(",") if item.strip()]
    except ValueError:
        ap.error("--threshold-sweep must be comma-separated floats")
    if any(not 0.0 <= threshold <= 1.0 for threshold in threshold_sweep):
        ap.error("--threshold-sweep values must be between 0 and 1")
    gps_alignment_max_error = (
        args.gps_max_error if args.gps_max_error is not None
        else 0.1 if args.rtk else 3.0
    )
    colmap_config = load_global_config("colmap.yml")
    camera_config = colmap_config.get("camera", {})
    dense_config = colmap_config.get("dense_reconstruction", {})
    max_image_size = int(camera_config.get("max_image_size", 3200))
    dense_max_image_size = int(dense_config.get("max_image_size", max_image_size))
    dense_num_iterations = int(dense_config.get("num_iterations", 3))
    dense_gpu_index = int(dense_config.get("gpu_index", 0))

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    classified_output = (
        Path(args.classified_output).resolve()
        if args.classified_output
        else outdir / "classified.ply"
    )
    report_dir = outdir.parent / "reports"
    summary = RunSummary(outdir, {
        "image_count": None,
        "capture_type": args.capture_type,
        "gps_mode": "RTK" if args.rtk else "Standard GPS" if args.capture_type == "drone_gps" else "Not used",
        "gps_max_error_m": gps_alignment_max_error if args.capture_type == "drone_gps" else "Not used",
        "gpu": args.gpu == "1",
        "dense_max_image_size": dense_max_image_size,
        "dense_num_iterations": dense_num_iterations,
        "class_set": args.class_set,
        "min_votes": args.min_votes,
        "sam_tile_size": args.sam_tile_size,
        "sam_tile_min_overlap": args.sam_tile_min_overlap,
        "classification_mode": args.classification_mode,
        "background_threshold": args.background_threshold,
        "query_threshold": args.query_threshold,
        "geometry_refinement": args.geometry_refinement == "1",
        "source_crs": args.source_crs,
        "export_crs": args.export_crs,
        "classified_output": str(classified_output),
        "classification_summary": str(report_dir / "classification_summary.json"),
    })
    if args.resume_after_control_points:
        _resume_after_control_points(
            args, outdir, report_dir, summary, progress, colmap_config, dense_max_image_size,
            dense_num_iterations, dense_gpu_index, classified_output,
        )
        return
    images_dir, is_panorama = resolve_input_images(args, outdir, ap)

    db_path = outdir / "database.db"
    sparse_dir = outdir / "sparse"
    dense_dir = outdir / "dense"
    sparse_dir.mkdir(exist_ok=True)
    dense_dir.mkdir(exist_ok=True)
    image_list_path = outdir / "image_list.txt"
    num_images = write_image_list(images_dir, image_list_path)
    summary.settings["image_count"] = num_images
    print(f"Using {num_images} supported images from {images_dir}")
    if progress:
        progress.update(5, "Preparing images", f"Found {num_images} supported images")

    colmap = find_colmap_exe(args.colmap_bin)

    # 1) Feature extraction
    if progress:
        progress.update(5, "Extracting features")
    run([colmap, "feature_extractor",
         "--database_path", str(db_path),
         "--image_path", str(images_dir),
         "--image_list_path", str(image_list_path),
         "--ImageReader.single_camera", "1",
         "--ImageReader.max_image_size", str(max_image_size),
         "--FeatureExtraction.use_gpu", args.gpu], summary=summary, stage="Feature extraction")
    if progress:
        progress.update(18, "Extracting features")

    # 2) Video panoramas use sequential matching. A photo directory additionally
    # uses exhaustive matching so it can connect to the video frames as well.
    if is_panorama and args.panorama_video:
        if progress:
            progress.update(18, "Matching images")
        run([colmap, "sequential_matcher",
             "--database_path", str(db_path),
             "--FeatureMatching.use_gpu", args.gpu,
             "--SequentialMatching.overlap", "10"], summary=summary, stage="Image matching")
        if progress:
            progress.update(25, "Matching images")
    if not is_panorama or args.panorama_images:
        if progress:
            progress.update(18 if not is_panorama else 25, "Matching images")
        run([colmap, "exhaustive_matcher",
             "--database_path", str(db_path),
             "--FeatureMatching.use_gpu", args.gpu], summary=summary, stage="Image matching")
        if progress:
            progress.update(32, "Matching images")

    # 3) Sparse reconstruction (SfM)
    if progress:
        progress.update(32, "Aligning cameras")
    run([colmap, "mapper",
         "--database_path", str(db_path),
         "--image_path", str(images_dir),
         "--output_path", str(sparse_dir)], summary=summary, stage="Sparse reconstruction")
    if progress:
        progress.update(52, "Aligning cameras")

    model0 = sparse_dir / "0"
    if not model0.exists():
        sys.exit("colmap mapper did not produce a reconstruction (sparse/0 missing). Check image overlap/quality.")

    # 3b) GPS alignment only applies to drone images. Other capture types retain
    # a local frame until control points or another georeferencing step is used.
    georeferenced = False
    gps_ref_path = outdir / "gps_ref.txt"
    if args.capture_type == "drone_gps":
        gps_script = Path(__file__).resolve().parent / "extract_gps.py"
        run([sys.executable, str(gps_script),
             "--images-dir", str(images_dir),
             "--out", str(gps_ref_path),
             "--require-all"], summary=summary, stage="Preparing GPS references")
        if progress:
            progress.update(55, "Applying GPS alignment")

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
             "--alignment_max_error", str(gps_alignment_max_error)], summary=summary, stage="GPS alignment")
        if progress:
            progress.update(57, "Applying GPS alignment")

        if not model0_geo.exists() or not any(model0_geo.iterdir()):
            sys.exit(
                "model_aligner produced no output. Check the gps_ref.txt and the "
                "colmap model_aligner flags for your installed COLMAP version."
            )
        model0 = model0_geo
        georeferenced = True
    else:
        print("GPS alignment skipped. The reconstruction remains in LOCAL coordinates.")

    # 4) Undistort for dense stereo, this also writes a PINHOLE sparse model
    #    we can read as plain text for the camera poses.
    if progress:
        progress.update(57, "Preparing dense reconstruction")
    run([colmap, "image_undistorter",
         "--image_path", str(images_dir),
         "--image_list_path", str(image_list_path),
         "--input_path", str(model0),
         "--output_path", str(dense_dir),
         "--output_type", "COLMAP",
         "--max_image_size", str(dense_max_image_size)], summary=summary, stage="Preparing dense reconstruction")
    if progress:
        progress.update(62, "Preparing dense reconstruction")

    # Phase A makes the actual marker workspace available: undistorted images
    # and a PINHOLE TEXT model. It intentionally stops before dense stereo.
    if args.stop_after_alignment:
        dense_sparse_txt = dense_dir / "sparse_txt"
        dense_sparse_txt.mkdir(parents=True, exist_ok=True)
        run([colmap, "model_converter",
             "--input_path", str(dense_dir / "sparse"),
             "--output_path", str(dense_sparse_txt),
             "--output_type", "TXT"], summary=summary, stage="Preparing control points")
        if georeferenced:
            try:
                georeference = fit_enu_to_ecef(dense_sparse_txt, gps_ref_path)
                save_georeference(outdir / "georeference.json", georeference)
            except Exception as error:
                print(f"WARNING: Could not create GPS georeference: {error}")
        if progress:
            progress.update(100, "Waiting for control points")
        print("Undistortion finished. Waiting for control points before dense reconstruction.")
        summary.write("Waiting for control points")
        return

    # 5) Dense stereo
    if progress:
        progress.update(62, "Building dense cloud")
    run([colmap, "patch_match_stereo",
         "--workspace_path", str(dense_dir),
         "--PatchMatchStereo.geom_consistency", "1",
         "--PatchMatchStereo.gpu_index", str(dense_gpu_index),
         "--PatchMatchStereo.num_iterations", str(dense_num_iterations),
         "--PatchMatchStereo.max_image_size", str(dense_max_image_size)], summary=summary, stage="Dense stereo")
    if progress:
        progress.update(78, "Building dense cloud")

    # 6) Fusion -> fused.ply + fused.ply.vis
    if progress:
        progress.update(78, "Fusing dense cloud")
    run([colmap, "stereo_fusion",
         "--workspace_path", str(dense_dir),
         "--output_path", str(dense_dir / "fused.ply")], summary=summary, stage="Dense fusion")
    if progress:
        progress.update(84, "Fusing dense cloud")

    # 7) Export the dense/sparse model as TEXT so classify.py can read poses
    #    without any COLMAP python bindings.
    dense_sparse_txt = dense_dir / "sparse_txt"
    dense_sparse_txt.mkdir(parents=True, exist_ok=True)
    if progress:
        progress.update(84, "Preparing classification")
    run([colmap, "model_converter",
         "--input_path", str(dense_dir / "sparse"),
         "--output_path", str(dense_sparse_txt),
         "--output_type", "TXT"], summary=summary, stage="Preparing classification")
    georeference_path = outdir / "georeference.json"
    if georeferenced:
        started_at = time.perf_counter()
        try:
            georeference = fit_enu_to_ecef(dense_sparse_txt, gps_ref_path)
            save_georeference(georeference_path, georeference)
        except Exception as error:
            print(f"WARNING: Could not create GPS georeference: {error}")
        else:
            summary.record_stage("Georeference", time.perf_counter() - started_at)
            summary.settings["georeference_residual_m"] = georeference["residual_m"]
            print(f"Georeference residuals: {format_residuals(georeference)}")
    if progress:
        progress.update(87, "Preparing classification")

    # 8) SAM3 classification + multi-view voting
    classify_script = Path(__file__).resolve().parent / "classify.py"
    geometry_enabled = args.geometry_refinement == "1"
    classified_raw = classified_output.with_name("classified_raw.ply") if geometry_enabled else classified_output
    classified_scores = classified_output.parent / "classified_scores.npz"
    classified_output.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress.update(87, "Classifying point cloud")
    classify_command = [sys.executable, str(classify_script),
         "--dense-dir", str(dense_dir),
         "--images-dir", str(dense_dir / "images"),
         "--model-dir", str(dense_sparse_txt),
         "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"),
         "--class-set", args.class_set,
         "--min-votes", str(args.min_votes),
         "--tile-size", str(args.sam_tile_size),
         "--tile-min-overlap", str(args.sam_tile_min_overlap),
         "--mode", args.classification_mode,
         "--background-threshold", str(args.background_threshold),
         "--query-threshold", str(args.query_threshold),
         "--summary-file", str(report_dir / "classification_summary.json"),
         "--out", str(classified_raw)]
    if args.threshold_sweep:
        classify_command.extend(["--threshold-sweep", args.threshold_sweep])
    if geometry_enabled and args.classification_mode == "segearth_ov3":
        classify_command.extend(["--scores-out", str(classified_scores)])
    if args.progress_file:
        classify_command.extend(["--progress-file", str(args.progress_file)])
    run(classify_command, summary=summary, stage="SAM3 classification")
    if progress:
        progress.update(98, "Classifying point cloud")

    if geometry_enabled:
        geometry_summary = report_dir / "geometry_summary.json"
        if georeferenced:
            refine_script = Path(__file__).resolve().parent / "geometry" / "refine.py"
            if progress:
                progress.update(98, "Geometric refinement")
            run([
                sys.executable, str(refine_script),
                "--input", str(classified_raw),
                "--output", str(classified_output),
                "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"),
                "--class-set", args.class_set,
                "--config", str(PROJECT_ROOT / "config" / "geometry.yml"),
                "--summary-file", str(geometry_summary),
                *(["--scores", str(classified_scores)]
                  if args.classification_mode == "segearth_ov3" else []),
                *(["--progress-file", str(args.progress_file)] if args.progress_file else []),
            ], summary=summary, stage="Geometric refinement")
        else:
            print("Geometric refinement skipped: reconstruction is not georeferenced.")
            shutil.copy2(classified_raw, classified_output)
            geometry_summary.parent.mkdir(parents=True, exist_ok=True)
            geometry_summary.write_text(json.dumps({"skipped": "not georeferenced"}, indent=2), encoding="utf-8")

    if args.skip_export:
        if progress:
            progress.update(100, "Complete")
        summary.write("Completed")
        print(f"Finished. Classified point cloud ready for manual export: {classified_output}")
        return

    # 9) Export a standards-compliant LAS file with RGB and class IDs.
    las_script = Path(__file__).resolve().parent / "export_las.py"
    if progress:
        progress.update(98, "Exporting LAS")
    run([sys.executable, str(las_script),
         "--input", str(classified_output),
         "--output", str(outdir / "classified.las"),
         "--classes-json", str(PROJECT_ROOT / "config" / "classes.json"),
         "--class-set", args.class_set,
         "--source-crs", args.source_crs,
         "--target-crs", args.export_crs,
         *(["--georeference", str(georeference_path)] if georeference_path.is_file() else [])],
        summary=summary, stage="LAS export")
    if progress:
        progress.update(100, "Complete")

    summary.write("Completed")
    print(f"\nFinished. Classified point cloud: {classified_output}")
    print(f"LAS with class IDs: {outdir / 'classified.las'}")


if __name__ == "__main__":
    main()
