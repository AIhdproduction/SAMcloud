"""Tests for COLMAP alignment command construction and filtering safeguards."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import struct

from samcloud.colmap.alignment import (
    filter_pass_is_safe,
    mapper_command,
    resolve_camera_model,
    select_matching_strategy,
    stereo_fusion_command,
    tiepoint_pass_commands,
    write_pose_priors_from_gps_ref,
)
from samcloud.engine import pipeline
from samcloud.engine.run_summary import RunSummary


def _georeferencing() -> dict[str, float]:
    return {
        "rtk_prior_std_xy_m": 0.03,
        "rtk_prior_std_z_m": 0.05,
        "standard_gps_prior_std_xy_m": 3.0,
        "standard_gps_prior_std_z_m": 5.0,
    }


def test_pose_prior_mapper_uses_rtk_or_standard_uncertainty(tmp_path: Path) -> None:
    standard = mapper_command("colmap", tmp_path / "db", tmp_path / "images", tmp_path / "sparse", use_pose_priors=True, rtk=False, georeferencing=_georeferencing())
    rtk = mapper_command("colmap", tmp_path / "db", tmp_path / "images", tmp_path / "sparse", use_pose_priors=True, rtk=True, georeferencing=_georeferencing())
    assert standard[1] == rtk[1] == "pose_prior_mapper"
    assert standard[standard.index("--prior_position_std_x") + 1] == "3.0"
    assert standard[standard.index("--prior_position_std_z") + 1] == "5.0"
    assert rtk[rtk.index("--prior_position_std_x") + 1] == "0.03"
    assert rtk[rtk.index("--prior_position_std_z") + 1] == "0.05"


def test_pose_priors_keep_dji_per_image_covariance_and_mapper_preserves_it(tmp_path: Path) -> None:
    database = tmp_path / "database.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE images (image_id INTEGER, name TEXT, camera_id INTEGER)")
        connection.execute("INSERT INTO images VALUES (1, 'fixed.jpg', 7)")
        connection.execute(
            "CREATE TABLE pose_priors (pose_prior_id INTEGER, corr_data_id INTEGER, corr_sensor_id INTEGER, "
            "corr_sensor_type INTEGER, position BLOB, position_covariance BLOB, gravity BLOB, coordinate_system INTEGER)"
        )
    images = tmp_path / "images"; images.mkdir()
    images.joinpath("fixed.jpg").write_bytes(
        b'<rdf:Description drone-dji:RtkFlag="50" drone-dji:RtkStdLon="0.02" '
        b'drone-dji:RtkStdLat="0.03" drone-dji:RtkStdHgt="0.04"/>'
    )
    gps_ref = tmp_path / "gps_ref.txt"; gps_ref.write_text("fixed.jpg 47.0 8.0 500.0\n", encoding="utf-8")

    assert write_pose_priors_from_gps_ref(database, gps_ref, images) == 1
    with sqlite3.connect(database) as connection:
        covariance_blob = connection.execute("SELECT position_covariance FROM pose_priors").fetchone()[0]
    covariance = struct.unpack("<9d", covariance_blob)
    assert covariance == (0.0004, 0.0, 0.0, 0.0, 0.0009, 0.0, 0.0, 0.0, 0.0016)

    command = mapper_command(
        "colmap", database, images, tmp_path / "sparse", use_pose_priors=True, rtk=False,
        georeferencing=_georeferencing(), per_image_covariances=True,
    )
    assert command[command.index("--overwrite_priors_covariance") + 1] == "0"


def test_mapper_falls_back_without_priors(tmp_path: Path) -> None:
    command = mapper_command("colmap", tmp_path / "db", tmp_path / "images", tmp_path / "sparse", use_pose_priors=False, rtk=False, georeferencing=_georeferencing())
    assert command[1] == "mapper"
    assert "--prior_position_std_x" not in command


def test_auto_matching_uses_exhaustive_then_spatial() -> None:
    assert select_matching_strategy("auto", 300, True) == "exhaustive"
    assert select_matching_strategy("auto", 301, True) == "spatial"


def test_auto_camera_model_uses_opencv_for_photos() -> None:
    assert resolve_camera_model("AUTO", False) == "OPENCV"


def test_filter_pass_commands_include_thresholds(tmp_path: Path) -> None:
    filtering, adjustment = tiepoint_pass_commands(
        "colmap", tmp_path / "input", tmp_path / "filtered", tmp_path / "adjusted",
        {"max_reproj_error": 1.0, "min_tri_angle": 2.0, "min_track_len": 3},
    )
    assert filtering[1] == "point_filtering"
    assert filtering[filtering.index("--max_reproj_error") + 1] == "1.0"
    assert filtering[filtering.index("--min_tri_angle") + 1] == "2.0"
    assert filtering[filtering.index("--min_track_len") + 1] == "3"
    assert adjustment[1] == "bundle_adjuster"
    assert adjustment[adjustment.index("--BundleAdjustment.refine_principal_point") + 1] == "1"


def test_filter_guard_skips_destructive_passes() -> None:
    assert not filter_pass_is_safe(10_000, 3_900)
    assert not filter_pass_is_safe(2_000, 999)
    assert filter_pass_is_safe(10_000, 4_000)


def test_stereo_fusion_command_contains_configured_filtering(tmp_path: Path) -> None:
    command = stereo_fusion_command(
        "colmap", tmp_path / "dense", tmp_path / "dense" / "fused.ply",
        {"min_num_pixels": 5, "max_reproj_error": 1.0, "max_depth_error": 0.005, "max_normal_error": 10},
    )
    assert command[1] == "stereo_fusion"
    assert command[command.index("--StereoFusion.min_num_pixels") + 1] == "5"
    assert command[command.index("--StereoFusion.max_reproj_error") + 1] == "1.0"
    assert command[command.index("--StereoFusion.max_depth_error") + 1] == "0.005"


def test_mapper_overwrites_only_non_per_image_covariances(tmp_path: Path) -> None:
    command = mapper_command(
        "colmap", tmp_path / "db", tmp_path / "images", tmp_path / "sparse", use_pose_priors=True,
        rtk=False, georeferencing=_georeferencing(), per_image_covariances=False,
    )
    assert command[command.index("--overwrite_priors_covariance") + 1] == "1"


def test_tiepoint_filtering_creates_output_directories(tmp_path: Path, monkeypatch) -> None:
    recorded_commands: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        recorded_commands.append(list(cmd))
        if "--output_path" in cmd:
            output_path = Path(cmd[cmd.index("--output_path") + 1])
            assert output_path.exists(), f"Output directory {output_path} does not exist before running {cmd[1]}"
            assert output_path.is_dir(), f"Output path {output_path} is not a directory before running {cmd[1]}"
        return None

    monkeypatch.setattr(pipeline, "run", fake_run)
    monkeypatch.setattr(pipeline, "_analyze_model", lambda *args, **kwargs: {"points": 5000.0})

    model_path = tmp_path / "sparse" / "0"
    model_path.mkdir(parents=True, exist_ok=True)
    settings = {
        "enabled": True,
        "passes": [
            {"max_reproj_error": 1.0, "min_tri_angle": 2.0, "min_track_len": 3}
        ],
    }
    summary = RunSummary(tmp_path / "out", {})

    final_model, statistics = pipeline._run_tiepoint_filtering(
        "colmap",
        model_path,
        settings,
        summary,
        use_pose_priors=False,
    )

    assert len(recorded_commands) == 2
    assert recorded_commands[0][1] == "point_filtering"
    assert recorded_commands[1][1] == "bundle_adjuster"
    assert final_model == tmp_path / "sparse" / "0_ba_1"
    assert final_model.is_dir()
    assert (tmp_path / "sparse" / "0_filter_1").is_dir()
    assert len(statistics) == 1
    assert statistics[0]["skipped"] is False


def test_pose_prior_upsert_updates_existing_rows(tmp_path: Path) -> None:
    import numpy as np

    database = tmp_path / "database.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE images (image_id INTEGER, name TEXT, camera_id INTEGER)")
        connection.execute("INSERT INTO images VALUES (1, 'image_1.jpg', 1)")
        connection.execute("INSERT INTO images VALUES (2, 'image_2.jpg', 1)")
        connection.execute(
            "CREATE TABLE pose_priors (pose_prior_id INTEGER, corr_data_id INTEGER, corr_sensor_id INTEGER, "
            "corr_sensor_type INTEGER, position BLOB, position_covariance BLOB, gravity BLOB, coordinate_system INTEGER)"
        )
        dummy_pos = struct.pack("<3d", 0.0, 0.0, 0.0)
        nan_cov = struct.pack("<9d", *(float("nan"),) * 9)
        connection.execute(
            "INSERT INTO pose_priors VALUES (1, 1, 1, 0, ?, ?, NULL, 0)",
            (dummy_pos, nan_cov),
        )

    gps_ref = tmp_path / "gps_ref.txt"
    gps_ref.write_text(
        "image_1.jpg 47.1 8.1 500.0\n"
        "image_2.jpg 47.2 8.2 550.0\n",
        encoding="utf-8",
    )

    written = write_pose_priors_from_gps_ref(database, gps_ref)
    assert written == 2

    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT corr_data_id, corr_sensor_type, position, position_covariance, coordinate_system "
            "FROM pose_priors ORDER BY corr_data_id"
        ).fetchall()

    assert len(rows) == 2

    assert rows[0][0] == 1
    assert rows[0][1] == 0
    pos1 = struct.unpack("<3d", rows[0][2])
    assert pos1 == (47.1, 8.1, 500.0)
    cov1 = struct.unpack("<9d", rows[0][3])
    assert all(np.isfinite(cov1))
    assert rows[0][4] == 0

    assert rows[1][0] == 2
    assert rows[1][1] == 0
    pos2 = struct.unpack("<3d", rows[1][2])
    assert pos2 == (47.2, 8.2, 550.0)
    cov2 = struct.unpack("<9d", rows[1][3])
    assert all(np.isfinite(cov2))
    assert rows[1][4] == 0

