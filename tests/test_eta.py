"""Unit tests for ETA calculation, progress parsing, and rate learning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from samcloud.engine.eta import (
    DEFAULT_RATES,
    ProgressEstimator,
    RateStore,
    stage_units,
)


def _create_clock(initial_time: float = 0.0):
    time_state = [initial_time]

    def fake_clock() -> float:
        return time_state[0]

    return time_state, fake_clock


def test_stage_units() -> None:
    assert stage_units("matching", images=291, matching="exhaustive") == 42195.0
    assert (
        stage_units("matching", images=291, matching="spatial", spatial_neighbors=50)
        == 7275.0
    )
    assert (
        stage_units(
            "dense_stereo",
            images=291,
            max_image_size=1600,
            num_iterations=3,
            geometric_consistency=True,
        )
        == 145.5
    )
    assert (
        stage_units(
            "dense_stereo",
            images=291,
            max_image_size=3200,
            num_iterations=3,
            geometric_consistency=False,
        )
        == 291.0
    )


def test_parse_colmap_lines() -> None:
    rates = RateStore(path=None)
    estimator = ProgressEstimator(
        plan=[
            ("feature_extraction", 10.0),
            ("matching", 36.0),
            ("mapping", 291.0),
            ("dense_fusion", 291.0),
        ],
        rates=rates,
    )

    estimator.start_stage("feature_extraction", units=10.0)
    assert estimator.parse_line("I20261103 Processed file [4/10]") is True
    assert estimator.snapshot()["stage_percent"] == 40.0

    estimator.start_stage("matching", units=36.0)
    assert estimator.parse_line("Processing block [2/6, 3/6]") is True
    assert abs(estimator.snapshot()["stage_percent"] - 25.0) < 0.1

    estimator.start_stage("mapping", units=291.0)
    assert estimator.parse_line("Registering image #17 (num_reg_frames=97)") is True
    assert abs(estimator.snapshot()["stage_percent"] - 33.3) < 0.1

    estimator.start_stage("dense_fusion", units=291.0)
    assert (
        estimator.parse_line("Fusing image [146/291] with index 3 in 0.2s")
        is True
    )
    assert abs(estimator.snapshot()["stage_percent"] - 50.2) < 0.1

    assert estimator.parse_line("Unrelated non-matching log output") is False


def test_dense_stereo_two_passes() -> None:
    rates = RateStore(path=None)
    estimator = ProgressEstimator(
        plan=[("dense_stereo", 582.0)],
        rates=rates,
    )
    estimator.start_stage("dense_stereo", units=2 * 291)

    for view_idx in range(1, 292):
        assert estimator.parse_line(f"Processing view {view_idx} / 291") is True

    for view_idx in range(1, 11):
        assert estimator.parse_line(f"Processing view {view_idx} / 291") is True

    expected_percent = 100.0 * (291 + 9) / 582
    snapshot = estimator.snapshot()
    assert abs(snapshot["stage_percent"] - expected_percent) <= 0.2


def test_eta_uses_live_rate_after_progress() -> None:
    time_state, fake_clock = _create_clock(0.0)
    rates = RateStore(path=None)
    estimator = ProgressEstimator(
        plan=[("dense_stereo", 100.0)],
        rates=rates,
        clock=fake_clock,
    )

    estimator.start_stage("dense_stereo")
    time_state[0] = 500.0
    estimator.update(0.5, 1.0)

    snapshot = estimator.snapshot()
    assert 450.0 <= snapshot["stage_eta_seconds"] <= 550.0
    assert snapshot["eta_seconds"] == snapshot["stage_eta_seconds"]


def test_pending_stages_scaled_by_speed_factor() -> None:
    time_state, fake_clock = _create_clock(0.0)
    rates = RateStore(path=None)
    estimator = ProgressEstimator(
        plan=[("mapping", 100.0), ("classification", 100.0)],
        rates=rates,
        clock=fake_clock,
    )

    estimator.start_stage("mapping")
    time_state[0] = 380.0
    estimator.finish_stage(learn=False)

    snapshot = estimator.snapshot()
    assert snapshot["stage_key"] is None
    assert abs(snapshot["speed_factor"] - 2.0) < 0.01
    assert abs(snapshot["eta_seconds"] - 920.0) <= 1.0


def test_rate_store_learns_and_persists(tmp_path: Path) -> None:
    store_file = tmp_path / "r.json"
    store = RateStore(store_file, alpha=0.5)

    base_mapping_rate = DEFAULT_RATES["mapping"]
    store.update("mapping", 300.0, 100.0)
    expected_rate = (base_mapping_rate + 3.0) / 2.0
    assert store.rate("mapping") == pytest.approx(expected_rate)

    store.save()
    assert store_file.is_file()

    reloaded_store = RateStore(store_file)
    assert reloaded_store.rate("mapping") == pytest.approx(expected_rate)

    # Unknown stage keys in persisted JSON must be ignored
    payload = {
        "rates": {
            "mapping": 2.45,
            "foreign_stage": 123.4,
        }
    }
    store_file.write_text(json.dumps(payload), encoding="utf-8")

    clean_store = RateStore(store_file)
    assert "foreign_stage" not in clean_store.rates
    assert clean_store.rate("foreign_stage") == 1.0
    assert clean_store.rate("mapping") == pytest.approx(2.45)


def test_rate_profiles_are_kept_apart_per_capture_type(tmp_path: Path) -> None:
    from samcloud.engine.eta import CAPTURE_DEFAULT_RATES

    store_file = tmp_path / "rates.json"
    store_file.write_text(json.dumps({"rates": {"dense_stereo": 28.8}}), encoding="utf-8")  # old mixed block
    drone = RateStore(store_file, alpha=0.5, profile="drone_gps")
    pano = RateStore(store_file, alpha=0.5, profile="panorama_360")
    assert drone.rate("dense_stereo") == CAPTURE_DEFAULT_RATES["drone_gps"]["dense_stereo"]
    assert pano.rate("dense_stereo") == CAPTURE_DEFAULT_RATES["panorama_360"]["dense_stereo"]

    pano.update("dense_stereo", 100.0, 1.0)
    pano.save()
    drone.save()

    data = json.loads(store_file.read_text(encoding="utf-8"))
    assert data["profiles"]["panorama_360"]["dense_stereo"] == pytest.approx((38.0 + 100.0) / 2)
    assert data["profiles"]["drone_gps"]["dense_stereo"] == CAPTURE_DEFAULT_RATES["drone_gps"]["dense_stereo"]
    assert RateStore(store_file, profile="drone_gps").rate("dense_stereo") == CAPTURE_DEFAULT_RATES["drone_gps"]["dense_stereo"]
    assert RateStore(store_file, profile="panorama_360").rate("dense_stereo") == pytest.approx(69.0)


def test_panorama_matching_units_skip_pairs_inside_one_panorama() -> None:
    assert stage_units("matching", images=1350, matching="exhaustive", faces_per_frame=6) == 1350 * 1349 / 2 - 225 * 15
    assert stage_units("matching", images=3000, matching="sequential", sequential_overlap=60) == 3000 * 60


def test_estimator_plan_follows_capture_type_and_class_set() -> None:
    from argparse import Namespace

    from samcloud.engine.pipeline import _build_estimator

    config = {"dense_reconstruction": {"max_image_size": 1024}, "alignment": {"matching_strategy": "auto"}}
    common = dict(control_points_mode="none", stop_after_alignment=False, geometry_refinement="1")
    indoor = _build_estimator(Namespace(capture_type="panorama_360", class_set="indoor", depth_model="large", **common),
                              1350, config, resume=False)
    drone = _build_estimator(Namespace(capture_type="drone_gps", class_set="outdoor", depth_model="large", **common),
                             291, config, resume=False)

    indoor_keys = [stage.name for stage in indoor.stages]
    drone_keys = [stage.name for stage in drone.stages]
    assert {"depth_prior", "surface_fusion"} <= set(indoor_keys) and "plane_fill" not in indoor_keys
    from samcloud.engine.pipeline import _fusion_rebuilds_cloud

    replaced = _fusion_rebuilds_cloud(Namespace(capture_type="panorama_360", class_set="indoor", depth_model="large"))
    # extract mode replaces fill and cleaning; snap mode (default) needs both before fusion
    assert bool({"depth_fill", "point_cleaning"} & set(indoor_keys)) is not replaced
    classic = _build_estimator(Namespace(capture_type="panorama_360", class_set="indoor", depth_model="large",
                                         surface_fusion="0", **common), 1350, config, resume=False)
    assert {"depth_prior", "point_cleaning", "depth_fill"} <= {stage.name for stage in classic.stages}
    assert "point_cleaning" in drone_keys and not {"depth_prior", "depth_fill", "plane_fill"} & set(drone_keys)
    assert indoor.rates.profile == "panorama_360" and drone.rates.profile == "drone_gps"
    matching_units = next(stage.units for stage in indoor.stages if stage.name == "matching")
    assert matching_units == 1350 * 1349 / 2 - 225 * 15
