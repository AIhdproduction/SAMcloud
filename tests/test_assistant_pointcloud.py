"""AI assistant point-cloud measuring tools on a synthetic classified cloud (plan metres, floor-relative z)."""

from __future__ import annotations

import json

import pytest

from assistant_synthetic import CHAIR_BOX, EMPTY_BOX, LEVEL_ID, TABLE_BOX, TABLE_TOP_M, make_plan, make_project
from samcloud.assistant.pointcloud import CloudUnavailable, ProjectCloud, parse_box


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory):
    return make_project(tmp_path_factory.mktemp("assistant_cloud"))


@pytest.fixture(scope="module")
def cloud(project):
    return ProjectCloud.load(project, LEVEL_ID)


def test_load_is_cached_and_needs_a_cloud(project, cloud, tmp_path) -> None:
    assert ProjectCloud.load(project, LEVEL_ID) is cloud
    assert len(cloud) > 10000
    bare = make_project(tmp_path, with_cloud=False)
    with pytest.raises(CloudUnavailable):
        ProjectCloud.load(bare, LEVEL_ID, plan=make_plan())


def test_points_in_box_reports_classes_heights_and_instances(cloud) -> None:
    x0, x1, y0, y1 = CHAIR_BOX
    report = cloud.points_in_box({"x": [x0 - 0.05, x1 + 0.05], "y": [y0 - 0.05, y1 + 0.05]})
    json.dumps(report)
    assert report["count"] > 500
    assert report["classes"][0]["name"] == "furniture"
    chair = next(item for item in report["instances"] if item["id"] == 9)
    assert chair["label"] == "chair"
    assert chair["bbox"]["z"][1] == pytest.approx(0.92, abs=0.03)
    assert any(abs(b["z0"] - 0.4) < 1e-6 and b["count"] > 100 for b in report["z_histogram"])  # the seat at 0.46 m
    empty = cloud.points_in_box({**EMPTY_BOX, "z": [0.05, 2.0]})
    assert empty["count"] == 0 and empty["instances"] == []


def test_section_and_png(cloud) -> None:
    section = cloud.section(1.0, {"x": [3.5, 4.5], "y": [0.5, 2.0]}, cell_m=0.05)
    json.dumps(section)
    assert section["points"] > 50 and section["outlines"]
    assert section["classes"][0]["name"] == "wall"
    png = cloud.section_png(1.0, {"x": [3.5, 4.5], "y": [0.5, 2.0]})
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and len(png) > 60


def test_fit_surface_table_and_wall(cloud) -> None:
    x0, x1, y0, y1 = TABLE_BOX
    table = cloud.fit_surface({"x": [x0 - 0.1, x1 + 0.1], "y": [y0 - 0.1, y1 + 0.1]}, "table")
    assert table["ok"] and table["height_m"] == pytest.approx(TABLE_TOP_M, abs=0.01)
    assert table["top_size_m"] == pytest.approx([1.2, 0.8], abs=0.03)
    wall = cloud.fit_surface({"x": [3.7, 4.3], "y": [0.5, 2.0]}, "wall")
    assert wall["ok"] and wall["thickness_m"] == pytest.approx(0.20, abs=0.01)
    assert wall["centreline"][0][0] == pytest.approx(4.0, abs=0.01)
    floor = cloud.fit_surface({"x": [0.5, 3.5], "y": [2.5, 4.5]}, "floor")
    assert floor["ok"] and floor["height_m"] == pytest.approx(0.0, abs=0.01)
    with pytest.raises(ValueError):
        cloud.fit_surface(EMPTY_BOX, "roof")


@pytest.mark.parametrize("box", [None, {"x": [1, 2]}, {"x": [1, 1], "y": [0, 1]}, {"x": ["a", 2], "y": [0, 1]}, {"x": [0, float("inf")], "y": [0, 1]}])
def test_parse_box_rejects_malformed_boxes(box) -> None:
    with pytest.raises(ValueError):
        parse_box(box)
