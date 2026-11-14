"""Tests for control point CSV persistence."""

from pathlib import Path

import pandas as pd
import pytest

from samcloud.control_points.repository import (
    apply_roles,
    load_detections,
    read_control_points,
    save_detections,
    set_detection_enabled,
    validate_observations,
    write_control_points,
)


def test_write_and_read_control_points(tmp_path: Path) -> None:
    csv_path = tmp_path / "control_points.csv"
    points = pd.DataFrame([{"name": "GCP-01", "x": 2600000.0, "y": 1200000.0, "z": 500.0}])

    write_control_points(points, csv_path)

    assert read_control_points(csv_path).to_dict("records") == [{"name": "GCP-01", "x": 2600000.0, "y": 1200000.0, "z": 500.0, "role": "pass"}]


def test_roles_are_normalised_and_default_from_name(tmp_path: Path) -> None:
    csv_path = tmp_path / "roles.csv"
    csv_path.write_text(
        "name,x,y,z,role\nPP,1,2,3,PP\nCP-01,4,5,6,CP\nA,7,8,9,Passpunkt\nB,10,11,12,Kontrollpunkt\n",
        encoding="utf-8",
    )
    assert read_control_points(csv_path)["role"].tolist() == ["pass", "control", "pass", "control"]


def test_agisoft_style_point_files_are_accepted(tmp_path: Path) -> None:
    semicolon = tmp_path / "semicolon.txt"
    semicolon.write_text("name;x;y;z\nCP-1;2'600'000,5;1'200'000,5;500,25\n", encoding="utf-8")
    assert read_control_points(semicolon).to_dict("records") == [{"name": "CP-1", "x": 2600000.5, "y": 1200000.5, "z": 500.25, "role": "control"}]
    headerless = tmp_path / "headerless.txt"
    headerless.write_text("P1 1 2 3\nCP2 4 5 6\n", encoding="utf-8")
    assert read_control_points(headerless)["role"].tolist() == ["pass", "control"]


def test_headerless_comma_csv_and_extra_columns_are_accepted(tmp_path: Path) -> None:
    comma = tmp_path / "headerless.csv"
    comma.write_text("P1,2600000.12,1200000.34,450.1\n", encoding="utf-8")
    assert read_control_points(comma).to_dict("records") == [{"name": "P1", "x": 2600000.12, "y": 1200000.34, "z": 450.1, "role": "pass"}]
    extra = tmp_path / "extra_columns.txt"
    extra.write_text("P2 1 2 3 0.02 0.02 0.05\n", encoding="utf-8")
    assert read_control_points(extra).to_dict("records") == [{"name": "P2", "x": 1.0, "y": 2.0, "z": 3.0, "role": "pass"}]


def test_decimal_commas_in_whitespace_file_are_not_csv(tmp_path: Path) -> None:
    path = tmp_path / "decimal_comma.txt"
    path.write_text("P1 2600000,12 1200000,34 450,1\n", encoding="utf-8")
    assert read_control_points(path).to_dict("records") == [{"name": "P1", "x": 2600000.12, "y": 1200000.34, "z": 450.1, "role": "pass"}]


def test_write_read_round_trip_preserves_roles(tmp_path: Path) -> None:
    path = tmp_path / "points.csv"
    write_control_points(pd.DataFrame([{"name": "P1", "x": 1, "y": 2, "z": 3, "role": "pass"}, {"name": "C1", "x": 4, "y": 5, "z": 6, "role": "control"}]), path)
    assert read_control_points(path)["role"].tolist() == ["pass", "control"]


def test_agisoft_passpunkte_file_roles_and_round_trip(tmp_path: Path) -> None:
    point_file = tmp_path / "Passpunkte_Drohne.txt"
    point_file.write_text(
        "PP1,2699905.377,1221224.273,905.968,\n"
        "PP2,2699887.237,1221233.440,904.684,\n"
        "PP3,2699856.720,1221233.401,902.765,\n"
        "PP4,2699886.348,1221195.751,906.728,\n"
        "BH_Extern,2699889.852,1221227.223,904.81,\n",
        encoding="utf-8",
    )

    points = read_control_points(point_file)
    assert len(points) == 5
    assert points["z"].tolist() == [905.968, 904.684, 902.765, 906.728, 904.81]
    assert points.loc[points["name"] == "BH_Extern", "role"].item() == "pass"

    output = tmp_path / "control_points.csv"
    write_control_points(apply_roles(points, {"BH_Extern": "control"}), output)
    assert read_control_points(output).loc[lambda table: table["name"] == "BH_Extern", "role"].item() == "control"


def test_observations_require_unique_finite_image_clicks() -> None:
    assert validate_observations({"GCP-01": [{"image": "one.jpg", "u": 10, "v": 20}]}) == {
        "GCP-01": [{"image": "one.jpg", "u": 10.0, "v": 20.0}]
    }
    with pytest.raises(ValueError, match="one observation per image"):
        validate_observations({"GCP-01": [{"image": "one.jpg", "u": 1, "v": 2}, {"image": "one.jpg", "u": 3, "v": 4}]})


def test_reviewer_flag_round_trip(tmp_path: Path) -> None:
    detections_path = tmp_path / "detections.json"
    initial = {
        "P1": {
            "role": "pass",
            "enabled": True,
            "status": "accepted",
            "observations": [],
        }
    }
    save_detections(detections_path, initial)
    set_detection_enabled(detections_path, "P1", False)
    loaded = load_detections(detections_path)
    assert loaded["P1"]["disabled_by_reviewer"] is True

    set_detection_enabled(detections_path, "P1", True)
    loaded = load_detections(detections_path)
    assert loaded["P1"]["disabled_by_reviewer"] is False


def test_apriltag_csv_round_trip_with_decimal_commas_and_default_role(tmp_path: Path) -> None:
    from samcloud.control_points.repository import (
        APRILTAG_CSV_NAME,
        read_apriltag_coordinates,
        write_apriltag_coordinates,
    )

    assert APRILTAG_CSV_NAME == "apriltags.csv"
    source = tmp_path / "surveyed.csv"
    source.write_text("id;x;y;z;role\n7;2'600'000,125;1'200'000,5;500,25;\n12;2600003,75;1200001,25;501,5;CP\n", encoding="utf-8")

    table = read_apriltag_coordinates(source)

    assert table["id"].tolist() == [7, 12]
    assert table["role"].tolist() == ["pass", "control"]
    assert table.loc[0, "x"] == pytest.approx(2600000.125)
    assert table.loc[1, "z"] == pytest.approx(501.5)

    target = tmp_path / "out" / "apriltags.csv"
    write_apriltag_coordinates(table, target)
    again = read_apriltag_coordinates(target)
    assert again.to_dict("records") == table.to_dict("records")


def test_apriltag_csv_headerless_and_without_role(tmp_path: Path) -> None:
    from samcloud.control_points.repository import read_apriltag_coordinates

    path = tmp_path / "plain.txt"
    path.write_text("3 1,5 2,25 0,75\n4 10 20 30\n", encoding="utf-8")

    table = read_apriltag_coordinates(path)

    assert table["id"].tolist() == [3, 4]
    assert table.loc[0, ["x", "y", "z"]].tolist() == [1.5, 2.25, 0.75]
    assert table["role"].tolist() == ["pass", "pass"]


def test_apriltag_csv_duplicate_ids_raise(tmp_path: Path) -> None:
    from samcloud.control_points.repository import read_apriltag_coordinates, write_apriltag_coordinates

    path = tmp_path / "dup.csv"
    path.write_text("id,x,y,z\n5,1,2,3\n5,4,5,6\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_apriltag_coordinates(path)
    with pytest.raises(ValueError):
        write_apriltag_coordinates(pd.DataFrame({"id": [1, 1], "x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]}), tmp_path / "w.csv")


def test_apriltag_point_name() -> None:
    from samcloud.control_points.repository import apriltag_point_name

    assert apriltag_point_name(7) == "TAG7"
