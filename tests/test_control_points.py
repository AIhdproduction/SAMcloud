"""Tests for control point CSV persistence."""

from pathlib import Path

import pandas as pd

from samcloud.control_points.repository import read_control_points, write_control_points


def test_write_and_read_control_points(tmp_path: Path) -> None:
    csv_path = tmp_path / "control_points.csv"
    points = pd.DataFrame([{"name": "GCP-01", "x": 2600000.0, "y": 1200000.0, "z": 500.0}])

    write_control_points(points, csv_path)

    assert read_control_points(csv_path).to_dict("records") == points.to_dict("records")
