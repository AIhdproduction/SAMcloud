"""Small, testable command builders for COLMAP alignment stages."""

from __future__ import annotations

import re
import sqlite3
import struct
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from samcloud.engine.extract_gps import dji_position_stddevs, read_dji_rtk_xmp


VALID_CAMERA_MODELS = {"SIMPLE_RADIAL", "RADIAL", "PINHOLE", "OPENCV", "FULL_OPENCV"}
PANORAMA_EXHAUSTIVE_MAX_IMAGES = 2000
PANORAMA_FACES_PER_FRAME = 6
PANORAMA_NEIGHBORING_FRAMES = 10


def adaptive_dense_size(
    image_long_side: int,
    image_count: int,
    free_bytes: int,
    *,
    ratio: float = 0.55,
    minimum: int = 2000,
    maximum: int = 3200,
    bytes_per_pixel: float = 24.0,
    safety: float = 1.25,
) -> tuple[int, str]:
    """Choose a dense resolution from input dimensions and available disk space."""
    if image_long_side <= 0:
        raise ValueError("Image long side must be positive")
    base = round(image_long_side * ratio / 100) * 100
    base = min(maximum, max(minimum, base), image_long_side)
    size = base
    while size > 1600 and image_count * size ** 2 * bytes_per_pixel * safety > free_bytes:
        size -= 100
    size = min(size, image_long_side)
    reason = f"Dense size {size}px from {image_long_side}px input (base {base}px)."
    if size < base:
        reason += " Reduced because available disk space is limited."
    return size, reason


def resolve_camera_model(value: str | None, is_panorama: bool) -> str:
    """Resolve a configured camera model for the current capture type."""
    requested = (value or "AUTO").upper()
    if requested == "AUTO":
        return "PINHOLE" if is_panorama else "OPENCV"
    if requested not in VALID_CAMERA_MODELS:
        raise ValueError(f"Unsupported COLMAP camera model: {requested}")
    return requested


def feature_extractor_command(
    colmap: str,
    database: Path,
    images: Path,
    image_list: Path,
    *,
    gpu: bool,
    camera_model: str,
    max_image_size: int,
    max_num_features: int,
    estimate_affine_shape: bool,
    domain_size_pooling: bool,
    capture_type: str = "drone_gps",
    panorama_peak_threshold: float = 0.003,
    panorama_edge_threshold: int = 20,
    single_camera: bool = False,
    single_camera_per_folder: bool = False,
    camera_params: str | None = None,
) -> list[str]:
    """Build feature extraction with explicitly selected initial camera assignments."""
    command = [
        colmap, "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(images),
        "--image_list_path", str(image_list),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1" if single_camera else "0",
        "--ImageReader.single_camera_per_folder", "1" if single_camera_per_folder else "0",
        "--FeatureExtraction.max_image_size", str(max_image_size),
        "--FeatureExtraction.use_gpu", "1" if gpu else "0",
        "--SiftExtraction.max_num_features", str(max_num_features),
        "--SiftExtraction.estimate_affine_shape", "1" if estimate_affine_shape else "0",
        "--SiftExtraction.domain_size_pooling", "1" if domain_size_pooling else "0",
    ]
    if capture_type == "panorama_360":
        command.extend([
            "--SiftExtraction.peak_threshold", str(panorama_peak_threshold),
            "--SiftExtraction.edge_threshold", str(panorama_edge_threshold),
        ])
    if camera_params:
        command.extend(["--ImageReader.camera_params", camera_params])
    return command


def select_matching_strategy(requested: str, image_count: int, has_pose_priors: bool) -> str:
    """Choose exhaustive or GPS spatial matching without requiring COLMAP."""
    if requested not in {"auto", "exhaustive", "spatial"}:
        raise ValueError(f"Unsupported matching strategy: {requested}")
    if requested == "exhaustive":
        return "exhaustive"
    if requested == "spatial":
        return "spatial" if has_pose_priors else "exhaustive"
    return "exhaustive" if image_count <= 300 or not has_pose_priors else "spatial"


def select_panorama_matching_strategy(image_count: int) -> str:
    """Choose reliable matching for cubemap faces from a 360 degree capture."""
    if image_count < 1:
        raise ValueError("Panorama image count must be positive")
    return "exhaustive" if image_count <= PANORAMA_EXHAUSTIVE_MAX_IMAGES else "sequential"


def panorama_matching_command(
    colmap: str,
    database: Path,
    *,
    gpu: bool,
    image_count: int,
) -> list[str]:
    """Build rig-aware matching for cubemap images across neighbouring panoramas."""
    strategy = select_panorama_matching_strategy(image_count)
    if strategy == "exhaustive":
        command = matching_command(
            colmap, database, gpu=gpu, strategy="exhaustive",
            max_num_neighbors=0, max_distance=0,
        )
    else:
        command = [
            colmap, "sequential_matcher",
            "--database_path", str(database),
            "--FeatureMatching.use_gpu", "1" if gpu else "0",
            "--SequentialMatching.overlap",
            str(PANORAMA_FACES_PER_FRAME * PANORAMA_NEIGHBORING_FRAMES),
        ]
    # A 90 degree cubemap has no useful interior overlap among faces in one
    # frame. COLMAP can instead verify pairs with the calibrated rig geometry.
    command.extend([
        "--FeatureMatching.rig_verification", "1",
        "--FeatureMatching.skip_image_pairs_in_same_frame", "1",
    ])
    return command


def _axis_angle_wxyz(axis: tuple[float, float, float], degrees: float) -> list[float]:
    """Return a unit quaternion in COLMAP's JSON [w, x, y, z] order."""
    radians = math.radians(degrees) / 2.0
    sine = math.sin(radians)
    return [math.cos(radians), *(component * sine for component in axis)]


def cubemap_rig_from_camera_rotations() -> dict[str, list[float]]:
    """Return virtual-camera orientations from the front-face rig frame.

    The face renderer uses COLMAP's image convention (x right, y down). These
    are the camera-to-rig rotations implied by its sampling directions.
    """
    return {
        "front": _axis_angle_wxyz((0.0, 1.0, 0.0), 0.0),
        "right": _axis_angle_wxyz((0.0, 1.0, 0.0), 90.0),
        "back": _axis_angle_wxyz((0.0, 1.0, 0.0), 180.0),
        "left": _axis_angle_wxyz((0.0, 1.0, 0.0), -90.0),
        "up": _axis_angle_wxyz((1.0, 0.0, 0.0), 90.0),
        "down": _axis_angle_wxyz((1.0, 0.0, 0.0), -90.0),
    }


def _inverse_unit_quaternion_wxyz(quaternion: list[float]) -> list[float]:
    """Invert a unit quaternion while retaining COLMAP's [w, x, y, z] order."""
    return [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]


def panorama_rig_config(face_size: int) -> list[dict[str, object]]:
    """Build the fixed six-camera COLMAP rig configuration for cubemap faces."""
    if face_size <= 0:
        raise ValueError("Cubemap face size must be positive")
    focal_length = face_size / 2.0
    camera_params = [focal_length, focal_length, focal_length, focal_length]
    rotations = cubemap_rig_from_camera_rotations()
    cameras: list[dict[str, object]] = []
    for face_name in ("front", "right", "back", "left", "up", "down"):
        camera: dict[str, object] = {
            "image_prefix": f"{face_name}/",
            "camera_model_name": "PINHOLE",
            "camera_params": camera_params,
        }
        if face_name == "front":
            camera["ref_sensor"] = True
        else:
            # COLMAP expects camera-from-rig, while the renderer naturally
            # defines each face as rig-from-camera.
            camera["cam_from_rig_rotation"] = _inverse_unit_quaternion_wxyz(rotations[face_name])
            camera["cam_from_rig_translation"] = [0.0, 0.0, 0.0]
        cameras.append(camera)
    return [{"cameras": cameras}]


def write_panorama_rig_config(path: Path, face_size: int) -> Path:
    """Write the calibrated fixed cubemap rig JSON consumed by COLMAP."""
    path.write_text(json.dumps(panorama_rig_config(face_size), indent=2), encoding="utf-8")
    return path


def matching_command(
    colmap: str,
    database: Path,
    *,
    gpu: bool,
    strategy: str,
    max_num_neighbors: int,
    max_distance: float,
) -> list[str]:
    """Build either exhaustive matching or GPS spatial matching."""
    command = [
        colmap,
        "spatial_matcher" if strategy == "spatial" else "exhaustive_matcher",
        "--database_path", str(database),
        "--FeatureMatching.use_gpu", "1" if gpu else "0",
    ]
    if strategy == "spatial":
        command.extend([
            "--SpatialMatching.max_num_neighbors", str(max_num_neighbors),
            "--SpatialMatching.max_distance", str(max_distance),
            "--SpatialMatching.ignore_z", "1",
        ])
    return command


def mapper_command(
    colmap: str,
    database: Path,
    images: Path,
    output: Path,
    *,
    use_pose_priors: bool,
    rtk: bool,
    georeferencing: dict[str, Any],
    per_image_covariances: bool = False,
    fixed_intrinsics: bool = False,
    fixed_rig_sensors: bool = False,
    init_min_num_inliers: int | None = None,
) -> list[str]:
    """Build an incremental mapper command, optionally constrained by GPS priors."""
    command = [
        colmap,
        "pose_prior_mapper" if use_pose_priors else "mapper",
        "--database_path", str(database),
        "--image_path", str(images),
        "--output_path", str(output),
        "--Mapper.ba_refine_principal_point", "0" if fixed_intrinsics else "1",
    ]
    if fixed_intrinsics:
        command.extend([
            "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_extra_params", "0",
        ])
    if fixed_rig_sensors:
        command.extend(["--Mapper.ba_refine_sensor_from_rig", "0"])
    if init_min_num_inliers is not None:
        command.extend(["--Mapper.init_min_num_inliers", str(init_min_num_inliers)])
    if use_pose_priors:
        prefix = "rtk" if rtk else "standard_gps"
        command.extend([
            "--overwrite_priors_covariance", "0" if per_image_covariances else "1",
            "--prior_position_std_x", str(georeferencing[f"{prefix}_prior_std_xy_m"]),
            "--prior_position_std_y", str(georeferencing[f"{prefix}_prior_std_xy_m"]),
            "--prior_position_std_z", str(georeferencing[f"{prefix}_prior_std_z_m"]),
            "--use_robust_loss_on_prior_position", "1",
        ])
    return command


def tiepoint_pass_commands(
    colmap: str,
    input_model: Path,
    filtered_model: Path,
    adjusted_model: Path,
    settings: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Build filtering and full-intrinsics bundle-adjustment commands."""
    filtering = [
        colmap, "point_filtering",
        "--input_path", str(input_model),
        "--output_path", str(filtered_model),
        "--min_track_len", str(settings["min_track_len"]),
        "--max_reproj_error", str(settings["max_reproj_error"]),
        "--min_tri_angle", str(settings["min_tri_angle"]),
    ]
    adjustment = [
        colmap, "bundle_adjuster",
        "--input_path", str(filtered_model),
        "--output_path", str(adjusted_model),
        "--BundleAdjustment.refine_focal_length", "1",
        "--BundleAdjustment.refine_principal_point", "1",
        "--BundleAdjustment.refine_extra_params", "1",
    ]
    return filtering, adjustment


DENSE_PROFILE_SECTIONS = ("patch_match", "fusion", "point_cleaning", "depth_fill", "plane_fill", "surface_fusion")


def dense_profile(dense_config: dict[str, Any], capture_type: str | None) -> dict[str, Any]:
    """Merge the capture-type profile over the shared dense settings.

    Drone, camera and 360 captures each have their own entry under
    ``capture_profiles`` so tuning one never silently changes another.
    """
    profile = {section: dict(dense_config.get(section) or {}) for section in DENSE_PROFILE_SECTIONS}
    override = (dense_config.get("capture_profiles") or {}).get(capture_type or "", {}) or {}
    for section in DENSE_PROFILE_SECTIONS:
        profile[section].update(override.get(section) or {})
    return profile


PATCH_MATCH_OPTIONS = {
    "window_radius": int,
    "window_step": int,
    "num_samples": int,
    "filter_min_ncc": float,
    "filter_min_num_consistent": int,
    "filter_min_triangulation_angle": float,
    "filter_geom_consistency_max_cost": float,
    "geom_consistency_max_cost": float,
}


def patch_match_arguments(patch_match: dict[str, Any]) -> list[str]:
    """Translate supported PatchMatch settings into COLMAP CLI arguments."""
    arguments: list[str] = []
    for name, cast in PATCH_MATCH_OPTIONS.items():
        if patch_match.get(name) is not None:
            arguments.extend([f"--PatchMatchStereo.{name}", str(cast(patch_match[name]))])
    return arguments


def stereo_fusion_command(
    colmap: str, dense_dir: Path, output: Path, fusion: dict[str, Any]
) -> list[str]:
    """Build dense fusion with explicit quality filtering."""
    return [
        colmap, "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--output_path", str(output),
        "--StereoFusion.min_num_pixels", str(fusion.get("min_num_pixels", 5)),
        "--StereoFusion.max_reproj_error", str(fusion.get("max_reproj_error", 1.0)),
        "--StereoFusion.max_depth_error", str(fusion.get("max_depth_error", 0.005)),
        "--StereoFusion.max_normal_error", str(fusion.get("max_normal_error", 10)),
    ]


def parse_model_analyzer_stats(output: str) -> dict[str, float]:
    """Extract useful values from COLMAP output without depending on exact wording."""
    patterns = {
        "points": r"(?:num(?:ber)?\s+of\s+)?points(?:3d)?\s*[:=]\s*([0-9]+)",
        "mean_track_length": r"mean\s+track\s+length\s*[:=]\s*([0-9.]+)",
        "mean_reprojection_error": r"mean\s+reprojection\s+error\s*[:=]\s*([0-9.]+)",
    }
    stats: dict[str, float] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            stats[name] = float(match.group(1))
    return stats


def filter_pass_is_safe(before_points: int | None, after_points: int | None) -> bool:
    """Reject a filtering result that is too destructive for stable reconstruction."""
    if before_points is None or after_points is None:
        return True
    if after_points < 1000:
        return False
    return (before_points - after_points) / max(before_points, 1) <= 0.60


def pose_prior_count(database: Path) -> int:
    """Return the number of pose priors in a COLMAP database."""
    with sqlite3.connect(database) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pose_priors'"
        ).fetchone()
        return int(connection.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0]) if exists else 0


def pose_prior_covariance_count(database: Path) -> int:
    """Return how many database pose priors have a stored covariance matrix."""
    with sqlite3.connect(database) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pose_priors'"
        ).fetchone()
        if not exists:
            return 0
        return int(connection.execute(
            "SELECT COUNT(*) FROM pose_priors WHERE position_covariance IS NOT NULL"
        ).fetchone()[0])


def write_pose_priors_from_gps_ref(
    database: Path,
    gps_ref: Path,
    images_dir: Path | None = None,
    *,
    standard_gps_std_xy_m: float = 3.0,
    standard_gps_std_z_m: float = 5.0,
) -> int:
    """Populate or update COLMAP 4.2 pose_priors rows from model_aligner GPS references."""
    gps_positions: dict[str, tuple[float, float, float]] = {}
    for raw_line in gps_ref.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) == 4:
            gps_positions[parts[0]] = (float(parts[1]), float(parts[2]), float(parts[3]))

    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(pose_priors)").fetchall()
        }
        required = {
            "pose_prior_id", "corr_data_id", "corr_sensor_id", "corr_sensor_type",
            "position", "position_covariance", "gravity", "coordinate_system",
        }
        if not required.issubset(columns):
            raise ValueError("The database does not use the COLMAP 4.2 pose_priors schema")
        images = connection.execute("SELECT image_id, name, camera_id FROM images").fetchall()
        missing = [name for _, name, _ in images if name not in gps_positions]
        if missing:
            raise ValueError(f"GPS references are missing {len(missing)} COLMAP image names")
        written = 0
        for image_id, name, camera_id in images:
            metadata = None
            if images_dir is not None:
                try:
                    metadata = read_dji_rtk_xmp(images_dir / name)
                except OSError:
                    metadata = None
            std_lon, std_lat, std_hgt = dji_position_stddevs(
                metadata, standard_gps_std_xy_m, standard_gps_std_z_m,
            )
            covariance = np.diag([std_lon ** 2, std_lat ** 2, std_hgt ** 2])
            position_blob = struct.pack("<3d", *gps_positions[name])
            covariance_blob = struct.pack("<9d", *covariance.ravel())
            exists = connection.execute(
                "SELECT 1 FROM pose_priors WHERE corr_data_id=? AND corr_sensor_type=0",
                (image_id,),
            ).fetchone()
            if exists:
                connection.execute(
                    "UPDATE pose_priors SET position=?, position_covariance=?, coordinate_system=0 "
                    "WHERE corr_data_id=? AND corr_sensor_type=0",
                    (position_blob, covariance_blob, image_id),
                )
            else:
                connection.execute(
                    "INSERT INTO pose_priors "
                    "(corr_data_id, corr_sensor_id, corr_sensor_type, "
                    "position, position_covariance, gravity, coordinate_system) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (image_id, camera_id, 0, position_blob, covariance_blob, None, 0),
                )
            written += 1
        return written


def extract_features_by_group(
    run_fn,
    colmap: str,
    database: Path | str,
    images_dir: Path | str,
    image_names: list[str],
    groups: dict[str, int],
    list_directory: Path | str,
    **feature_kwargs,
) -> dict[int, int]:
    """Run feature_extractor once per camera group so COLMAP 4 creates consistent cameras, rigs and frames."""
    counts = {}
    database_path = Path(database)
    images_path = Path(images_dir)
    target_dir = Path(list_directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    defaults = {
        "gpu": False,
        "camera_model": "OPENCV",
        "max_image_size": 4000,
        "max_num_features": 16384,
        "estimate_affine_shape": False,
        "domain_size_pooling": False,
    }
    merged_kwargs = {**defaults, **feature_kwargs}
    for group in sorted(set(groups.values())):
        names = [name for name in image_names if groups.get(name) == group]
        list_path = target_dir / f"image_list_group_{group + 1:02d}.txt"
        list_path.write_text("\n".join(names) + "\n", encoding="utf-8")
        run_fn(
            feature_extractor_command(
                colmap,
                database_path,
                images_path,
                list_path,
                single_camera=True,
                **merged_kwargs,
            )
        )
        counts[group] = len(names)
    return counts
