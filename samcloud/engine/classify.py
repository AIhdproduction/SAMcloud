import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from colmap_model import load_model
from ply_io import read_fused_points, read_vis_file, write_classified_ply
from sam_labeler import SemanticLabeler


def build_image_to_points_index(vis_list):
    """Invert the per-point visibility list into image_id -> [point_idx, ...]."""
    index = defaultdict(list)
    for point_idx, image_ids in enumerate(vis_list):
        for img_id in image_ids:
            index[int(img_id)].append(point_idx)
    return index


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
    ap.add_argument("--out", required=True, help="output PLY path")
    args = ap.parse_args()

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
    # vote_counts[point_idx, class_idx] , class_idx 0..num_classes-1 maps to class_names
    vote_counts = np.zeros((xyz.shape[0], num_classes), dtype=np.uint16)

    labeler = SemanticLabeler(class_names, unknown_id=unknown_id,
                               tile_size=args.tile_size, min_overlap_frac=args.tile_min_overlap)

    for img_id, point_indices in tqdm(img_to_points.items(), desc="images"):
        if img_id not in images:
            continue  # image was in dense workspace but not in sparse text model, skip
        image = images[img_id]
        camera = cameras[image.camera_id]
        fx, fy, cx, cy = camera.intrinsics()

        image_path = Path(args.images_dir) / image.name
        if not image_path.exists():
            continue

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

    print("[5/5] Resolving majority vote and writing output")
    best_class_idx = np.argmax(vote_counts, axis=1)
    best_class_votes = np.take_along_axis(vote_counts, best_class_idx[:, None], axis=1).squeeze(1)

    final_class = np.where(best_class_votes >= args.min_votes, best_class_idx + 1, unknown_id).astype(np.uint8)

    write_classified_ply(args.out, xyz, rgb, final_class)

    print(f"Done. Written to {args.out}")
    for i, name in enumerate(class_names, start=1):
        count = int((final_class == i).sum())
        print(f"  {i:2d} {name:30s} {count:>10,} points")
    print(f"   0 {cfg.get('unknown_class_name', 'unknown'):30s} {int((final_class == unknown_id).sum()):>10,} points")


if __name__ == "__main__":
    main()
