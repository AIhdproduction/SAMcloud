import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from samcloud.classification.profiles import NEVER_CLASSIFIED_ID

if __package__:
    from .colmap_model import load_model
    from .ply_io import read_fused_points, read_vis_file, write_classified_ply
    from .sam_labeler import HeadFusionLabeler, SemanticLabeler
else:
    from colmap_model import load_model
    from ply_io import read_fused_points, read_vis_file, write_classified_ply
    from sam_labeler import HeadFusionLabeler, SemanticLabeler


def mvs_image_order(sparse_dir, image_ids=None):
    """Image ids in the order COLMAP's MVS model indexes them in fused.ply.vis.

    COLMAP numbers dense views by ``Reconstruction::RegImageIds()``: frame by
    frame, and inside a frame sensor by sensor. For rigs (360 cubemaps: six
    faces per panorama) that is NOT ascending image id; for plain photo and
    drone projects (one image per frame) both orders coincide. Falls back to
    sorted ids when the binary model cannot be read.
    """
    try:
        import pycolmap

        order = [int(image_id) for image_id in pycolmap.Reconstruction(str(sparse_dir)).reg_image_ids()]
        if order:
            return order
    except Exception as error:  # noqa: BLE001 - fall back, but say so
        print(f"WARNING: could not read MVS image order from {sparse_dir}: {error}; using sorted image ids")
    return sorted(int(image_id) for image_id in (image_ids or []))


def build_image_to_points_index(vis_list, image_order=None):
    """Invert the per-point visibility list into image_id -> [point_idx, ...].

    COLMAP's fused.ply.vis stores image *indices* of its MVS model, i.e.
    positions in ``mvs_image_order`` - not image ids. Pass that order to
    translate them; without it the raw indices are returned unchanged.
    """
    lookup = [int(image_id) for image_id in image_order] if image_order is not None else None
    index = defaultdict(list)
    for point_idx, image_indices in enumerate(vis_list):
        for image_index in image_indices:
            image_index = int(image_index)
            if lookup is None:
                index[image_index].append(point_idx)
            elif image_index < len(lookup):
                index[lookup[image_index]].append(point_idx)
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


def accumulate_weighted_votes(scores, view_counts, point_indices, point_probs, background_threshold, seen_counts=None):
    """Accumulate one SegEarth-OV3 view using presence-filtered probabilities."""
    if point_probs.ndim != 2 or point_probs.shape[0] != len(point_indices):
        raise ValueError("point_probs must have one row per point index")
    visible = point_probs.max(axis=1) >= background_threshold
    if seen_counts is not None:
        if len(point_indices) < 2 or bool(np.all(point_indices[1:] > point_indices[:-1])):
            seen_counts[point_indices] += 1
        else:
            np.add.at(seen_counts, point_indices, 1)
    rows = point_indices[visible]
    if len(rows) < 2 or bool(np.all(rows[1:] > rows[:-1])):
        # Strictly increasing = unique (the visibility index lists each point once per view):
        # plain fancy-index addition is ~5x faster than np.add.at.
        scores[rows] += point_probs[visible]
        view_counts[rows] += 1
    else:
        np.add.at(scores, rows, point_probs[visible])
        np.add.at(view_counts, rows, 1)


def resolve_weighted_votes(scores, view_counts, min_votes, unknown_id, seen_counts=None, never_classified_id=NEVER_CLASSIFIED_ID):
    """Resolve SegEarth-OV3 class-weighted multi-view votes to class IDs."""
    best_class = np.argmax(scores, axis=1)
    has_score = np.any(scores != 0, axis=1)
    resolved = np.where(
        (view_counts >= min_votes) & has_score,
        best_class + 1,
        unknown_id,
    ).astype(np.uint8)
    if seen_counts is None:
        return resolved
    return np.where(seen_counts == 0, never_classified_id, resolved).astype(np.uint8)


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


DEPTH_CHECK_DEFAULTS = {"radius": 16, "step": 4, "abs_tol": 0.05, "rel_tol": 0.03, "min_neighbors": 3}


def depth_consistent_probs(point_probs, ui, vi, z, labels, depth, background_threshold, settings=None):
    """Remove class votes that the view's depth map contradicts (in place); returns stats.

    For every point the winning class ``c`` of this view is checked against the
    depth of the surrounding pixels that SAM labels ``c`` (a (2r+1)^2 window,
    every ``step`` pixels). When the point lies clearly in front of or behind
    that region (``abs_tol`` + ``rel_tol`` x depth), the mask belongs to another
    surface - e.g. a sofa mask that spills over the wall behind it - and ``c`` is
    removed from this view's vote, so the next class can win. A point the depth
    map shows hidden behind a nearer surface in this view gets no vote at all.
    Pixels without measured depth (textureless walls) are not judged.
    """
    settings = {**DEPTH_CHECK_DEFAULTS, **(settings or {})}
    stats = {"checked": 0, "class_removed": 0, "occluded": 0}
    if depth is None or labels is None or not len(ui):
        return stats
    height, width = labels.shape[:2]
    dh, dw = depth.shape[:2]
    sx, sy = dw / width, dh / height
    tol = float(settings["abs_tol"]) + float(settings["rel_tol"]) * z
    own = depth[np.clip((vi * sy).astype(np.int64), 0, dh - 1), np.clip((ui * sx).astype(np.int64), 0, dw - 1)]
    occluded = (own > 0) & (own < z - tol)
    point_probs[occluded] = 0.0
    stats["occluded"] = int(occluded.sum())
    best = point_probs.argmax(axis=1)
    active = np.flatnonzero(~occluded & (point_probs.max(axis=1) >= background_threshold))
    if not len(active):
        return stats
    r, step = int(settings["radius"]), max(1, int(settings["step"]))
    offsets = np.arange(-r, r + 1, step)
    dx, dy = np.meshgrid(offsets, offsets)
    dx, dy = dx.ravel(), dy.ravel()
    for start in range(0, len(active), 200_000):
        idx = active[start:start + 200_000]
        nu = np.clip(ui[idx, None] + dx[None, :], 0, width - 1)
        nv = np.clip(vi[idx, None] + dy[None, :], 0, height - 1)
        same = labels[nv, nu] == (best[idx, None] + 1)
        nd = depth[np.clip((nv * sy).astype(np.int64), 0, dh - 1), np.clip((nu * sx).astype(np.int64), 0, dw - 1)]
        same &= nd > 0
        count = same.sum(axis=1)
        judged = count >= int(settings["min_neighbors"])
        if not judged.any():
            continue
        values = np.where(same[judged], nd[judged], np.nan)
        reference = np.nanmedian(values, axis=1)
        wrong = np.abs(z[idx[judged]] - reference) > tol[idx[judged]]
        rows = idx[judged][wrong]
        point_probs[rows, best[rows]] = 0.0
        stats["checked"] += int(judged.sum())
        stats["class_removed"] += int(wrong.sum())
    return stats


def refine_labels_with_depth(labels, depth, settings=None, num_classes=None, device=None):
    """Move SAM class edges onto the depth edges of the view's depth map; returns (labels, changed pixels).

    A labelled pixel whose depth does not match the depth of the surrounding
    pixels of its own class (window and tolerances as in
    :func:`depth_consistent_probs`) belongs to another surface: it takes the
    majority class of the neighbours at its own depth (at least
    ``min_neighbors`` of them), otherwise it keeps its label. Pixels without
    depth are left as SAM labelled them.
    """
    import torch

    settings = {**DEPTH_CHECK_DEFAULTS, **(settings or {})}
    if depth is None or labels is None:
        return labels, 0
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lab = torch.as_tensor(np.asarray(labels).astype(np.int64), device=device)
    height, width = lab.shape
    d = torch.as_tensor(np.asarray(depth, dtype=np.float32), device=device)
    if d.shape != lab.shape:
        d = torch.nn.functional.interpolate(d[None, None], size=(height, width), mode="nearest")[0, 0]
    classes = int(num_classes or int(lab.max()) + 1)
    r, step = int(settings["radius"]), max(1, int(settings["step"]))
    offsets = [(dy, dx) for dy in range(-r, r + 1, step) for dx in range(-r, r + 1, step)]
    lab_p = torch.nn.functional.pad(lab[None, None].float(), (r, r, r, r), mode="replicate")[0, 0].long()
    d_p = torch.nn.functional.pad(d[None, None], (r, r, r, r), mode="replicate")[0, 0]
    out = lab.clone()
    changed = 0
    minimum = int(settings["min_neighbors"])
    band = max(16, int(2_000_000 // max(width * len(offsets) // 64, 1)))
    for y0 in range(0, height, band):
        y1 = min(height, y0 + band)
        own_l, own_d = lab[y0:y1], d[y0:y1]
        tol = float(settings["abs_tol"]) + float(settings["rel_tol"]) * own_d
        nl = torch.stack([lab_p[y0 + r + dy:y1 + r + dy, r + dx:r + dx + width] for dy, dx in offsets])
        nd = torch.stack([d_p[y0 + r + dy:y1 + r + dy, r + dx:r + dx + width] for dy, dx in offsets])
        same = (nl == own_l[None]) & (nd > 0)
        count = same.sum(dim=0)
        reference = torch.nanmedian(torch.where(same, nd, torch.full_like(nd, float("nan"))), dim=0).values
        wrong = (own_l > 0) & (own_d > 0) & (count >= minimum) & ((own_d - reference).abs() > tol)
        if not bool(wrong.any()):
            continue
        at_depth = (nd - own_d[None]).abs() <= tol[None]
        at_depth &= (nd > 0) & (nl > 0)
        votes = torch.zeros((classes, y1 - y0, width), dtype=torch.int16, device=device)
        votes.scatter_add_(0, nl.clamp(0, classes - 1), at_depth.to(torch.int16))
        best_count, best_class = votes.max(dim=0)
        take = wrong & (best_count >= minimum) & (best_class != own_l)
        out[y0:y1][take] = best_class[take]
        changed += int(take.sum())
    return out.cpu().numpy().astype(np.uint8), changed


def project_points(xyz, point_indices, image, camera, image_size, return_depth=False):
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
    if return_depth:
        return point_indices[in_bounds], ui[in_bounds], vi[in_bounds], z[in_bounds]
    return point_indices[in_bounds], ui[in_bounds], vi[in_bounds]


def read_depth_map(dense_dir, image_name):
    """COLMAP geometric depth map of one dense view (0 = no depth) or None."""
    if not dense_dir:
        return None
    path = Path(dense_dir) / "stereo" / "depth_maps" / f"{image_name}.geometric.bin"
    if not path.is_file():
        return None
    try:
        from samcloud.colmap.plane_fill import read_colmap_array
    except ImportError:  # run as a plain script
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from samcloud.colmap.plane_fill import read_colmap_array
    return read_colmap_array(path)


def label_raster_path(labels_dir, image_name):
    """Path of the stored SAM label raster for one dense view (PNG, 0 = no class)."""
    return Path(labels_dir) / f"{image_name}.png"


def save_label_raster(labels_dir, image_name, labels):
    """Store one view's per-pixel SAM class ids for the class-confirmed plane fill."""
    import cv2

    path = label_raster_path(labels_dir, image_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(labels, dtype=np.uint8)):
        raise RuntimeError(f"Could not write label raster {path}")


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


def make_point_labeler(args, class_names, unknown_id, prompts=None):
    """Point scorer of the selected engine: SAM3 head fusion or the FlurSeg model (outdoor only)."""
    if getattr(args, "engine", "sam3") == "flurseg":
        if __package__:
            from .flurseg_labeler import FlurSegLabeler
        else:
            from flurseg_labeler import FlurSegLabeler
        print("      Engine: FlurSeg (own EoMT-DINOv3 model, config/flurseg.yml)")
        return FlurSegLabeler(class_names, unknown_id=unknown_id)
    return HeadFusionLabeler(
        class_names,
        unknown_id=unknown_id,
        tile_size=args.tile_size,
        min_overlap_frac=args.tile_min_overlap,
        query_threshold=args.query_threshold,
        prompts=prompts,
    )


def run_segearth_ov3(args, xyz, rgb, cameras, images, image_items, class_names, unknown_id, progress_path,
                     prompts=None):
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
            np.zeros(xyz.shape[0], dtype=np.uint16),
        )
        for threshold in thresholds
    }
    labeler = make_point_labeler(args, class_names, unknown_id, prompts)

    depth_check = bool(getattr(args, "depth_check", 1))
    depth_stats = {"checked": 0, "class_removed": 0, "occluded": 0, "raster_pixels_moved": 0}
    depth_settings = {"abs_tol": getattr(args, "depth_check_abs", 0.05), "rel_tol": getattr(args, "depth_check_rel", 0.03)}
    write_progress(progress_path, 0, len(image_items))
    for processed, (img_id, listed_points) in enumerate(tqdm(image_items, desc="images"), start=1):
        image = images[img_id]
        image_path = Path(args.images_dir) / image.name
        image_pixels, width, height = labeler.load_image(image_path)
        point_indices = np.array(listed_points, dtype=np.int64)
        sel_points, ui, vi, z = project_points(
            xyz, point_indices, image, cameras[image.camera_id], (width, height), return_depth=True
        )
        if len(sel_points):
            depth = read_depth_map(args.dense_dir, image.name) if depth_check else None
            if args.labels_dir or depth is not None:
                point_probs, labels = labeler.score_points(
                    image_path, ui, vi, image=image_pixels, label_threshold=args.background_threshold
                )
                if depth is not None:  # class edges onto the depth edges (raster, then the points)
                    labels, changed = refine_labels_with_depth(labels, depth, depth_settings, len(class_names) + 1)
                    depth_stats["raster_pixels_moved"] += changed
                if args.labels_dir:
                    save_label_raster(args.labels_dir, image.name, labels)
                if depth is not None:
                    view_stats = depth_consistent_probs(point_probs, ui, vi, z, labels, depth,
                                                        args.background_threshold, depth_settings)
                    for key, value in view_stats.items():
                        depth_stats[key] += value
            else:
                point_probs = labeler.score_points(image_path, ui, vi, image=image_pixels)
            for threshold, (scores, view_counts, seen_counts) in vote_states.items():
                accumulate_weighted_votes(scores, view_counts, sel_points, point_probs, threshold, seen_counts)
        write_progress(progress_path, processed, len(image_items))

    if depth_check:
        print("      Depth check: "
              f"{depth_stats['checked']:,} point views judged, {depth_stats['class_removed']:,} class votes removed "
              f"(mask of another surface), {depth_stats['occluded']:,} hidden point views skipped, "
              f"{depth_stats['raster_pixels_moved']:,} label pixels moved onto depth edges")
    output_path = Path(args.out)
    threshold_summary = {}
    main_final_class = None
    for threshold, (scores, view_counts, seen_counts) in vote_states.items():
        final_class = resolve_weighted_votes(scores, view_counts, args.min_votes, unknown_id, seen_counts, NEVER_CLASSIFIED_ID)
        if threshold == args.background_threshold:
            destination = output_path
            main_final_class = final_class
        else:
            destination = output_path.with_name(f"{output_path.stem}_tau{threshold:.2f}{output_path.suffix}")
        write_classified_ply(destination, xyz, rgb, final_class)
        threshold_summary[f"{threshold:.2f}"] = {
            "class_counts": _class_counts(final_class, class_names),
            "unknown_points": int((final_class == unknown_id).sum()),
            "never_classified_points": int((final_class == NEVER_CLASSIFIED_ID).sum()),
        }

    if args.scores_out:
        main_scores, main_view_counts, _main_seen_counts = vote_states[args.background_threshold]
        save_score_summary(args.scores_out, main_scores, main_view_counts, main_final_class)
        print(f"SAM confidence scores written to {args.scores_out}")

    class_counts = _class_counts(main_final_class, class_names)
    unknown_points = int((main_final_class == unknown_id).sum())
    never_classified_points = int((main_final_class == NEVER_CLASSIFIED_ID).sum())
    if args.summary_file:
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "point_count": int(main_final_class.shape[0]),
                    "unknown_points": unknown_points,
                    "never_classified_points": never_classified_points,
                    "images_processed": len(image_items),
                    "class_counts": class_counts,
                    "mode": args.mode,
                    "engine": getattr(args, "engine", "sam3"),
                    "background_threshold": args.background_threshold,
                    "query_threshold": args.query_threshold,
                    "threshold_sweep": threshold_summary,
                    "depth_check": depth_stats if depth_check else None,
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
    ap.add_argument("--engine", choices=("sam3", "flurseg"), default="sam3",
                    help="2D model: sam3 (default) or flurseg (own EoMT model, outdoor class set only; "
                         "always uses the weighted SegEarth-OV3 voting)")
    ap.add_argument("--background-threshold", type=float, default=0.4,
                    help="SegEarth-OV3 background threshold applied to every sampled point")
    ap.add_argument("--query-threshold", type=float, default=0.4,
                    help="SegEarth-OV3 initial confidence threshold for decoder queries")
    ap.add_argument("--threshold-sweep", default="",
                    help="optional comma-separated SegEarth-OV3 background thresholds")
    ap.add_argument("--out", required=True, help="output PLY path")
    ap.add_argument("--labels-dir", default=None,
                    help="optional folder for per-view SAM label rasters (PNG), used by the indoor plane fill")
    ap.add_argument("--progress-file", default=None, help="dashboard progress JSON path")
    ap.add_argument("--summary-file", default=None, help="JSON file for the processing summary")
    ap.add_argument("--depth-check", type=int, choices=(0, 1), default=1,
                    help="drop class votes the view's COLMAP depth map contradicts (mask of another surface)")
    ap.add_argument("--depth-check-abs", type=float, default=0.05, help="depth check tolerance, absolute part")
    ap.add_argument("--depth-check-rel", type=float, default=0.03, help="depth check tolerance, relative part")
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
    if args.engine == "flurseg" and args.class_set != "outdoor":
        ap.error("--engine flurseg is only available for the outdoor class set")

    dense_dir = Path(args.dense_dir)
    model_dir = Path(args.model_dir) if args.model_dir else dense_dir / "sparse"
    with open(args.classes_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    class_names = cfg[args.class_set]
    if args.engine == "flurseg":
        from samcloud.classification.profiles import flurseg_profile
        class_names = [entry.model_name for entry in flurseg_profile("outdoor").entries]
    # Optional short SAM3 prompts per class set (same order as the class names).
    prompts = None if args.engine == "flurseg" else cfg.get("prompts", {}).get(args.class_set) or None
    if prompts is not None and len(prompts) != len(class_names):
        sys.exit(f"classes.json: prompts.{args.class_set} must list one prompt per class ({len(class_names)})")
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
    image_order = mvs_image_order(dense_dir / "sparse", images.keys())
    img_to_points = build_image_to_points_index(vis_list, image_order)
    num_classes = len(class_names)
    progress_path = Path(args.progress_file) if args.progress_file else None
    image_items = [
        (img_id, point_indices)
        for img_id, point_indices in img_to_points.items()
        if img_id in images and (Path(args.images_dir) / images[img_id].name).exists()
    ]
    if args.mode == "segearth_ov3" or args.engine == "flurseg":
        run_segearth_ov3(
            args, xyz, rgb, cameras, images, image_items, class_names, unknown_id, progress_path,
            prompts=prompts,
        )
        return

    if args.scores_out:
        print("Note: --scores-out is ignored in legacy mode; legacy voting has no confidence scores.")

    # vote_counts[point_idx, class_idx] , class_idx 0..num_classes-1 maps to class_names
    vote_counts = np.zeros((xyz.shape[0], num_classes), dtype=np.uint16)

    labeler = SemanticLabeler(class_names, unknown_id=unknown_id,
                               tile_size=args.tile_size, min_overlap_frac=args.tile_min_overlap,
                               prompts=prompts)

    write_progress(progress_path, 0, len(image_items))

    for processed, (img_id, point_indices) in enumerate(tqdm(image_items, desc="images"), start=1):
        image = images[img_id]
        camera = cameras[image.camera_id]
        fx, fy, cx, cy = camera.intrinsics()

        image_path = Path(args.images_dir) / image.name
        raster = labeler.label_image(image_path)  # (H, W) uint8, 0 = unknown
        if args.labels_dir:
            save_label_raster(args.labels_dir, image.name, raster)

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
