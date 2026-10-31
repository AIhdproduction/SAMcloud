"""Tests for control point CSV persistence."""

from pathlib import Path

import pandas as pd
import pytest

from samcloud.control_points.repository import read_control_points, validate_observations, write_control_points


def test_write_and_read_control_points(tmp_path: Path) -> None:
    csv_path = tmp_path / "control_points.csv"
    points = pd.DataFrame([{"name": "GCP-01", "x": 2600000.0, "y": 1200000.0, "z": 500.0}])

    write_control_points(points, csv_path)

    assert read_control_points(csv_path).to_dict("records") == points.to_dict("records")


def test_observations_require_unique_finite_image_clicks() -> None:
    assert validate_observations({"GCP-01": [{"image": "one.jpg", "u": 10, "v": 20}]}) == {
        "GCP-01": [{"image": "one.jpg", "u": 10.0, "v": 20.0}]
    }
    with pytest.raises(ValueError, match="one observation per image"):
        validate_observations({"GCP-01": [{"image": "one.jpg", "u": 1, "v": 2}, {"image": "one.jpg", "u": 3, "v": 4}]})
