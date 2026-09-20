"""Build explicit COLMAP commands for each workflow stage."""

from __future__ import annotations

from pathlib import Path


def alignment_commands(
    colmap: str,
    images: Path,
    database: Path,
    sparse_output: Path,
    gpu: bool,
    capture_type: str = "drone_gps",
) -> list[list[str]]:
    """Return commands for normal images or prepared 360 degree cubemap views."""
    gpu_value = "1" if gpu else "0"
    commands = [
        [colmap, "feature_extractor", "--database_path", str(database), "--image_path", str(images),
         "--ImageReader.single_camera", "1", "--FeatureExtraction.use_gpu", gpu_value],
    ]
    if capture_type == "panorama_360":
        commands.append(
            [colmap, "sequential_matcher", "--database_path", str(database),
             "--FeatureMatching.use_gpu", gpu_value, "--SequentialMatching.overlap", "10"]
        )
    else:
        commands.append(
            [colmap, "exhaustive_matcher", "--database_path", str(database),
             "--FeatureMatching.use_gpu", gpu_value]
        )
    commands.append(
        [colmap, "mapper", "--database_path", str(database), "--image_path", str(images),
         "--output_path", str(sparse_output)]
    )
    return commands


def dense_commands(colmap: str, images: Path, sparse_model: Path, dense_output: Path, gpu: bool) -> list[list[str]]:
    """Return the commands for image undistortion, dense stereo, and fusion."""
    gpu_value = "1" if gpu else "0"
    return [
        [colmap, "image_undistorter", "--image_path", str(images), "--input_path", str(sparse_model),
         "--output_path", str(dense_output), "--output_type", "COLMAP"],
        [colmap, "patch_match_stereo", "--workspace_path", str(dense_output),
         "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.gpu_index", gpu_value],
        [colmap, "stereo_fusion", "--workspace_path", str(dense_output),
         "--output_path", str(dense_output / "fused.ply")],
    ]
