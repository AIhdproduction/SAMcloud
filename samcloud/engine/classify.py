import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

if __package__:
    from .colmap_model import load_model
    from .ply_io import read_fused_points, read_vis_file, write_classified_ply
    from .sam_labeler import HeadFusionLabeler, SemanticLabeler
else:
    from colmap_model import load_model
    from ply_io import read_fused_points, read_vis_file, write_classified_ply
    from sam_labeler import HeadFusionLabeler, SemanticLabeler


def build_image_to_points_index(vis_list):
    """Invert the per-point visibility list into image_id -> [point_idx, ...]."""
    index = defaultdict(list)
    for point_idx, image_ids in enumerate(vis_list):
        for img_id in image_ids:
            index[int(img_id)].append(point_idx)
    return index


def write_progress(path: Path | None, processed: int, total: int) -> None:
    """Persist per-photo classification progress for the dashboard."""
    if path is None:
        return
    photo_percent = 100.0 if total == 0 else processed / total * 100.0
    payload = {
        "percent": round(87.0 + photo_percent * 0.11, 1),
        "stage": "Classifying point cloud",
        "detail": f"{processed} of {total} photos",
        "photo_processed": processed,
        "photo_total": total,
        "photo_percent": round(photo_percent, 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def accumulate_weighted_votes(scores, view_counts, point_indices, point_probs, background_threshold):
    """Accumulate one SegEarth-OV3 view using presence-filtered probabilities."""
    if point_probs.ndim != 2 or point_probs.shape[0] != len(point_indices):
        raise ValueError("point_probs must have one row per point index")
    visible = point_probs.max(axis=1) >= background_threshold
    np.add.at(scores, point_indices[visible], point_probs[visible])
    np.add.at(view_counts, point_indices[visible], 1)


def resolve_weighted_votes(scores, view_counts, min_votes, unknown_id):
    """Resolve SegEarth-OV3 class-weighted multi-view votes to class IDs."""
    best_class = np.argmax(scores, axis=1)
    has_score = np.any(scores != 0, axis=1)
    return np.where(
        (view_counts >= min_votes) & has_score,
        best_class + 1,
        unknown_id,
    ).astype(np.uint8)


def extract_top2_confidences(scores, view_counts, final_class, chunk_size=1_000_000):
    """Extract mean top-one and top-two probabilities without sorting all classes.

    The accumulator can be several gigabytes, so each ``argpartition`` operates
    on a bounded row block. ``final_class`` is deliberately used for top one so
    the exported class remains exactly the label written to the PLY.
    """
    if scores.ndim != 2 or len(view_counts) != len(scores) or len(final_class) != len(scores):
        raise ValueError("scores, view_counts and final_class must have matching point counts")
    num_points, num_classes = scores.shape
    top1_class = final_class.astype(np.uint8, copy=True)
    top1_conf = np.zeros(num_points, dtype=np.float16)
    top2_class = np.zeros(num_points, dtype=np.uint8)
    top2_conf = np.zeros(num_points, dtype=np.float16)
    if num_classes == 0:
        return top1_class, top1_conf, top2_class, top2_conf

    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        block = scores[start:end]
        counts = view_counts[start:end]
        if num_classes == 1:
            first = np.zeros(len(block), dtype=np.intp)
            second = None
        else:
            candidates = np.argpartition(block, num_classes - 2, axis=1)[:, -2:]
            candidate_scores = np.take_along_axis(block, candidates, axis=1)
            best_position = np.argmax(candidate_scores, axis=1)
            first = candidates[np.arange(len(block)), best_position]
            second = candidates[np.arange(len(block)), 1 - best_position]
        denominator = counts.astype(np.float32, copy=False)
        valid = counts > 0
        first_conf = np.zeros(len(block), dtype=np.float32)
        first_conf[valid] = block[np.arange(len(block))[valid], first[valid]] / denominator[valid]
        written = top1_class[start:end] > 0
        top1_conf[start:end] = np.where(written, first_conf, 0.0).astype(np.float16)
        if second is not None:
            second_conf = np.zeros(len(block), dtype=np.float32)
            second_conf[valid] = block[np.arange(len(block))[valid], second[valid]] / denominator[valid]
            has_second = valid & (second_conf > 0.0)
            top2_class[start:end] = np.where(has_second, second + 1, 0).astype(np.uint8)
            top2_conf[start:end] = second_conf.astype(np.float16)
    return top1_class, top1_conf, top2_class, top2_conf


def save_score_summary(path, scores, view_counts, final_class):
    """Write compact SAM confidence arrays for the main threshold accumulator."""
    top1_class, top1_conf, top2_class, top2_conf = extract_top2_confidences(
        scores, view_counts, final_class
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        top1_class=top1_class,
        top1_conf=top1_conf,
        top2_class=top2_class,
        top2_conf=top2_conf,
        view_count=view_counts.astype(np.uint16, copy=False),
    )


def project_points(xyz, point_indices, image, camera, image_size):
    """Project COLMAP-visible points and retain those in front of the camera and image bounds."""
    fx, fy, cx, cy = camera.intrinsics()
    pts_world = xyz[point_indices]
    pts_cam = image.world_to_cam(pts_world)
    z = pts_cam[:, 2]
    valid = z > 1e-6
    u = np.zeros_like(z)
    v = np.zeros_like(z)
    u[valid] = fx * pts_cam[valid, 0] / z[valid] + cx
    v[valid] = fy * pts_cam[valid, 1] / z[valid] + cy
    width, height = image_size
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    in_bounds = valid & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    return point_indices[in_bounds], ui[in_bounds], vi[in_bounds]


def parse_threshold_sweep(value):
    """Parse a comma-separated background-threshold list."""
    if not value:
        return []
    try:
        thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("threshold sweep must be comma-separated floats") from error
    if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("threshold sweep values must be between 0 and 1")
    return thresholds


def _class_counts(final_class, class_names):
    return {
        name.split(",", 1)[0].strip(): int((final_class == index).sum())
        for index, name in enumerate(class_names, start=1)
    }


def run_segearth_ov3(args, xyz, rgb, cameras, images, image_items, class_names, unknown_id, progress_path):
    """Classify with SegEarth-OV3 fusion and class-weighted 3D voting."""
    thresholds = [args.background_threshold]
    for threshold in parse_threshold_sweep(args.threshold_sweep):
        if threshold not in thresholds:
            thresholds.append(threshold)
    num_classes = len(class_names)
    bytes_per_threshold = xyz.shape[0] * (num_classes * np.dtype(np.float32).itemsize + np.dtype(np.uint16).itemsize)
    print(
        "      SegEarth-OV3 accumulator memory: "
        f"{bytes_per_threshold * len(thresholds) / 1024 ** 2:.1f} MiB "
        f"for {len(thresholds)} threshold(s)"
    )
    vote_states = {
        threshold: (
            np.zeros((xyz.shape[0], num_classes), dtype=np.float32),
            np.zeros(xyz.shape[0], dtype=np.uint16),
        )
        for threshold in thresholds
    }
    labeler = HeadFusionLabeler(
        class_names,
        unknown_id=unknown_id,
        tile_size=args.tile_size,
        min_overlap_frac=args.tile_min_overlap,
        query_threshold=args.query_threshold,
    )

    write_progress(progress_path, 0, len(image_items))
    for processed, (img_id, listed_points) in enumerate(tqdm(image_items, desc="images"), start=1):
        image = images[img_id]
        image_path = Path(args.images_dir) / image.name
        image_pixels, width, height = labeler.load_image(image_path)
        point_indices = np.array(listed_points, dtype=np.int64)
        sel_points, ui, vi = project_points(
            xyz, point_indices, image, cameras[image.camera_id], (width, height)
        )
        if len(sel_points):
            point_probs = labeler.score_points(image_path, ui, vi, image=image_pixels)
            for threshold, (scores, view_counts) in vote_states.items():
                accumulate_weighted_votes(scores, view_counts, sel_points, point_probs, threshold)
        write_progress(progress_path, processed, len(image_items))

    output_path = Path(args.out)
    threshold_summary = {}
    main_final_class = None
    for threshold, (scores, view_counts) in vote_states.items():
        final_class = resolve_weighted_votes(scores, view_counts, args.min_votes, unknown_id)
        if threshold == args.background_threshold:
            destination = output_path
            main_final_class = final_class
        else:
            destination = output_path.with_name(f"{output_path.stem}_tau{threshold:.2f}{output_path.suffix}")
        write_classified_ply(destination, xyz, rgb, final_class)
        threshold_summary[f"{threshold:.2f}"] = {
            "class_counts": _class_counts(final_class, class_names),
            "unknown_points": int((final_class == unknown_id).sum()),
        }

    if args.scores_out:
        main_scores, main_view_counts = vote_states[args.background_threshold]
        save_score_summary(args.scores_out, main_scores, main_view_counts, main_final_class)
        print(f"SAM confidence scores written to {args.scores_out}")

    class_counts = _class_counts(main_final_class, class_names)
    unknown_points = int((main_final_class == unknown_id).sum())
    if args.summary_file:
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "point_count": int(main_final_class.shape[0]),
                    "unknown_points": unknown_points,
                    "images_processed": len(image_items),
                    "class_counts": class_counts,
                    "mode": args.mode,
                    "background_threshold": args.background_threshold,
                    "query_threshold": args.query_threshold,
                    "threshold_sweep": threshold_summary,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    print(f"Done. Written to {args.out}")
    for i, name in enumerate(class_names, start=1):
        print(f"  {i:2d} {name:30s} {class_counts[name.split(',', 1)[0].strip()]:>10,} points")
    print(f"   0 {unknown_id:1d} {'unknown':28s} {unknown_points:>10,} points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-dir", required=True, help="colmap dense workspace (contains fused.ply, fused.ply.vis, sparse/)")
    ap.add_argument("--model-dir", default=None,
                    help="COLMAP text model directory; defaults to <dense-dir>/sparse")
    ap.add_argument("--images-dir", required=True, help="dense/images (undistorted images)")
    ap.add_argument("--classes-json", required=True)
    ap.add_argument("--class-set", choices=("outdoor", "indoor"), default="outdoor",
                    help="semantic class set used for SAM3 prompts")
    ap.add_argument("--min-votes", type=int, default=1,
                    help="minimum agreeing images before a class is assigned; default 1 keeps single-view classifications")
    ap.add_argument("--tile-size", type=int, default=1008, help="tile size in pixels, should match SAM3's native input size")
    ap.add_argument("--tile-min-overlap", type=float, default=0.2, help="minimum overlap fraction between neighbouring tiles")
    ap.add_argument("--mode", choices=("legacy", "segearth_ov3"), default="segearth_ov3",
                    help="SAM3 classification path (default: segearth_ov3; legacy remains available for comparison)")
    ap.add_argument("--background-threshold", type=float, default=0.4,
                    help="SegEarth-OV3 background threshold applied to every sampled point")
    ap.add_argument("--query-threshold", type=float, default=0.4,
                    help="SegEarth-OV3 initial confidence threshold for decoder queries")
    ap.add_argument("--threshold-sweep", default="",
                    help="optional comma-separated SegEarth-OV3 background thresholds")
    ap.add_argument("--out", required=True, help="output PLY path")
    ap.add_argument("--progress-file", default=None, help="dashboard progress JSON path")
    ap.add_argument("--summary-file", default=None, help="JSON file for the processing summary")
    ap.add_argument("--scores-out", default=None,
                    help="optional NPZ output with SegEarth-OV3 top-two confidence scores")
    args = ap.parse_args()
    if not 0.0 <= args.background_threshold <= 1.0:
        ap.error("--background-threshold must be between 0 and 1")
    if not 0.0 <= args.query_threshold <= 1.0:
        ap.error("--query-threshold must be between 0 and 1")
    try:
        parse_threshold_sweep(args.threshold_sweep)
    except ValueError as error:
        ap.error(str(error))

    dense_dir = Path(args.dense_dir)
    model_dir = Path(args.model_dir) if args.model_dir else dense_dir / "sparse"
    with open(args.classes_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    class_names = cfg[args.class_set]
    unknown_id = cfg.get("unknown_class_id", 0)

    print(f"[1/5] Loading dense point cloud from {dense_dir / 'fused.ply'}")
    xyz, rgb = read_fused_points(dense_dir / "fused.ply")
    print(f"      {xyz.shape[0]:,} points")

    print("[2/5] Loading visibility file")
    vis_list = read_vis_file(dense_dir / "fused.ply.vis")
    if len(vis_list) != xyz.shape[0]:
        sys.exit(
            f"fused.ply has {xyz.shape[0]} points but fused.ply.vis has {len(vis_list)}. "
            "These must come from the same stereo_fusion run."
        )

    print(f"[3/5] Loading camera poses from {model_dir}")
    cameras, images = load_model(model_dir)

    print("[4/5] Building image -> point index and running SAM3 per image")
    img_to_points = build_image_to_points_index(vis_list)
    num_classes = len(class_names)
    progress_path = Path(args.progress_file) if args.progress_file else None
    image_items = [
        (img_id, point_indices)
        for img_id, point_indices in img_to_points.items()
        if img_id in images and (Path(args.images_dir) / images[img_id].name).exists()
    ]
    if args.mode == "segearth_ov3":
        run_segearth_ov3(
            args, xyz, rgb, cameras, images, image_items, class_names, unknown_id, progress_path
        )
        return

    if args.scores_out:
        print("Note: --scores-out is ignored in legacy mode; legacy voting has no confidence scores.")

    # vote_counts[point_idx, class_idx] , class_idx 0..num_classes-1 maps to class_names
    vote_counts = np.zeros((xyz.shape[0], num_classes), dtype=np.uint16)

    labeler = SemanticLabeler(class_names, unknown_id=unknown_id,
                               tile_size=args.tile_size, min_overlap_frac=args.tile_min_overlap)

    write_progress(progress_path, 0, len(image_items))

    for processed, (img_id, point_indices) in enumerate(tqdm(image_items, desc="images"), start=1):
        image = images[img_id]
        camera = cameras[image.camera_id]
        fx, fy, cx, cy = camera.intrinsics()

        image_path = Path(args.images_dir) / image.name
        raster = labeler.label_image(image_path)  # (H, W) uint8, 0 = unknown

        point_indices = np.array(point_indices, dtype=np.int64)
        pts_world = xyz[point_indices]
        pts_cam = image.world_to_cam(pts_world)

        z = pts_cam[:, 2]
        valid = z > 1e-6
        u = np.zeros_like(z)
        v = np.zeros_like(z)
        u[valid] = fx * pts_cam[valid, 0] / z[valid] + cx
        v[valid] = fy * pts_cam[valid, 1] / z[valid] + cy

        h, w = raster.shape
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        in_bounds = valid & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

        sel_points = point_indices[in_bounds]
        sel_classes = raster[vi[in_bounds], ui[in_bounds]]

        has_class = sel_classes > 0
        sel_points = sel_points[has_class]
        sel_classes = sel_classes[has_class]  # 1-indexed class ids

        np.add.at(vote_counts, (sel_points, sel_classes - 1), 1)
        write_progress(progress_path, processed, len(image_items))

    print("[5/5] Resolving majority vote and writing output")
    best_class_idx = np.argmax(vote_counts, axis=1)
    best_class_votes = np.take_along_axis(vote_counts, best_class_idx[:, None], axis=1).squeeze(1)

    final_class = np.where(best_class_votes >= args.min_votes, best_class_idx + 1, unknown_id).astype(np.uint8)

    write_classified_ply(args.out, xyz, rgb, final_class)

    class_counts = {
        name.split(",", 1)[0].strip(): int((final_class == index).sum())
        for index, name in enumerate(class_names, start=1)
    }
    unknown_points = int((final_class == unknown_id).sum())
    if args.summary_file:
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "point_count": int(final_class.shape[0]),
                    "unknown_points": unknown_points,
                    "images_processed": len(image_items),
                    "class_counts": class_counts,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(f"Done. Written to {args.out}")
    for i, name in enumerate(class_names, start=1):
        count = class_counts[name.split(",", 1)[0].strip()]
        print(f"  {i:2d} {name:30s} {count:>10,} points")
    print(f"   0 {cfg.get('unknown_class_name', 'unknown'):30s} {unknown_points:>10,} points")


if __name__ == "__main__":
    main()
