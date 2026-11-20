"""Tests for checkpoint-derived height correction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import samcloud.pointcloud.georeference as georeference_module
from samcloud.drone.height_correction import (
    apply_height_shift,
    estimate_height_shift,
    revert_height_shift,
)


def _checkpoint_file(path: Path, values: list[float], *, statuses: list[str] | None = None) -> Path:
    payload = {"points": [
        {"name": f"PP{index + 1}", "status": (statuses or ["ok"] * len(values))[index], "dH": value}
        for index, value in enumerate(values)
    ]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_enu_to_target_applies_optional_height_offset_only_to_target_height(monkeypatch) -> None:
    class IdentityTransformer:
        def transform(self, x, y, z, **_kwargs):
            return x, y, z

    monkeypatch.setattr(georeference_module, "strict_transformer", lambda *_args: IdentityTransformer())
    monkeypatch.setattr(georeference_module, "export_target_crs", lambda _key: "EPSG:3857")
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    base = {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]}

    unchanged = georeference_module.enu_to_target(points, base, "test")
    corrected = georeference_module.enu_to_target(points, {**base, "height_offset_m": -0.218}, "test")

    assert unchanged == pytest.approx(points)
    assert corrected[:, :2] == pytest.approx(unchanged[:, :2])
    assert corrected[:, 2] == pytest.approx(unchanged[:, 2] - 0.218)


def test_estimate_height_shift_matches_checkpoint_values(tmp_path: Path) -> None:
    estimate = estimate_height_shift(_checkpoint_file(tmp_path / "checkpoints.json", [0.212, 0.221, 0.220]))

    assert estimate["height_offset_m"] == pytest.approx(-0.2177, abs=0.0005)
    assert estimate["max_loo_residual_m"] < 0.01
    assert [row["name"] for row in estimate["points"]] == ["PP1", "PP2", "PP3"]


@pytest.mark.parametrize("values, kwargs", [([0.2, 0.21], {}), ([0.0, 0.2, 0.0], {})])
def test_estimate_height_shift_refuses_insufficient_or_inconsistent_points(
    tmp_path: Path, values: list[float], kwargs: dict,
) -> None:
    with pytest.raises(ValueError, match="GCP adjustment"):
        estimate_height_shift(_checkpoint_file(tmp_path / "checkpoints.json", values), **kwargs)


def test_apply_and_revert_height_shift_round_trip(tmp_path: Path) -> None:
    project = tmp_path / "project"
    products = project / "products"
    products.mkdir(parents=True)
    original = {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]}
    georeference = products / "georeference.json"
    georeference.write_text(json.dumps(original), encoding="utf-8")
    estimate = estimate_height_shift(_checkpoint_file(tmp_path / "checkpoints.json", [0.212, 0.221, 0.220]))

    assert apply_height_shift(project, estimate) == georeference
    applied = json.loads(georeference.read_text(encoding="utf-8"))
    assert applied["height_offset_m"] == pytest.approx(-0.2177, abs=0.0005)
    assert applied["height_offset_source"]["points_used"] == ["PP1", "PP2", "PP3"]
    assert list(products.glob("georeference.json.bak_*"))

    assert revert_height_shift(project) == georeference
    assert json.loads(georeference.read_text(encoding="utf-8")) == original
