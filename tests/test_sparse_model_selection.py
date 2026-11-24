from pathlib import Path

from samcloud.engine.pipeline import _promote_largest_sparse_model


def _make_model(sparse: Path, index: int, images: int, points: int) -> None:
    model = sparse / str(index)
    model.mkdir(parents=True)
    (model / "images.bin").write_bytes(images.to_bytes(8, "little") + b"\0" * 8)
    (model / "points3D.bin").write_bytes(points.to_bytes(8, "little") + b"\0" * 8)
    (model / "marker.txt").write_text(f"model-{index}", encoding="utf-8")


def _marker(sparse: Path, index: int) -> str:
    return (sparse / str(index) / "marker.txt").read_text(encoding="utf-8")


def test_large_model_at_one_is_promoted(tmp_path):
    sparse = tmp_path / "sparse"
    _make_model(sparse, 0, 4, 12882)
    _make_model(sparse, 1, 275, 58732)
    info = _promote_largest_sparse_model(sparse)
    assert _marker(sparse, 0) == "model-1"
    assert _marker(sparse, 1) == "model-0"
    assert info["promoted_from"] == 1
    assert [m["registered_images"] for m in info["models"]] == [4, 275]
    assert sorted(p.name for p in sparse.iterdir()) == ["0", "1"]


def test_largest_already_zero_unchanged(tmp_path):
    sparse = tmp_path / "sparse"
    _make_model(sparse, 0, 275, 58732)
    _make_model(sparse, 1, 4, 12882)
    info = _promote_largest_sparse_model(sparse)
    assert _marker(sparse, 0) == "model-0"
    assert _marker(sparse, 1) == "model-1"
    assert info["promoted_from"] is None


def test_single_model_unchanged(tmp_path):
    sparse = tmp_path / "sparse"
    _make_model(sparse, 0, 10, 2000)
    info = _promote_largest_sparse_model(sparse)
    assert _marker(sparse, 0) == "model-0"
    assert info["promoted_from"] is None
    assert len(info["models"]) == 1


def test_three_models_largest_at_two(tmp_path):
    sparse = tmp_path / "sparse"
    _make_model(sparse, 0, 4, 100)
    _make_model(sparse, 1, 50, 5000)
    _make_model(sparse, 2, 200, 20000)
    info = _promote_largest_sparse_model(sparse)
    assert _marker(sparse, 0) == "model-2"
    assert _marker(sparse, 1) == "model-1"
    assert _marker(sparse, 2) == "model-0"
    assert info["promoted_from"] == 2


def test_tie_prefers_more_points(tmp_path):
    sparse = tmp_path / "sparse"
    _make_model(sparse, 0, 20, 1000)
    _make_model(sparse, 1, 20, 3000)
    _promote_largest_sparse_model(sparse)
    assert _marker(sparse, 0) == "model-1"


def test_missing_sparse_dir_is_safe(tmp_path):
    info = _promote_largest_sparse_model(tmp_path / "nope")
    assert info["models"] == []
