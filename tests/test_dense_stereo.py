"""Dense stereo workspace safety tests."""

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from samcloud.engine import pipeline


class _Pose:
    def __init__(self, translation: list[float]) -> None:
        self.rotation = SimpleNamespace(matrix=lambda: np.eye(3))
        self.translation = np.asarray(translation)


class _Image:
    def __init__(self, image_id: int, name: str, translation: list[float]) -> None:
        self.image_id = image_id
        self.name = name
        self._pose = _Pose(translation)

    def cam_from_world(self) -> _Pose:
        return self._pose


def test_filter_stereo_workspace_excludes_images_without_depth_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense = tmp_path / "dense"
    (dense / "sparse").mkdir(parents=True)
    stereo = dense / "stereo"
    stereo.mkdir()
    (stereo / "patch-match.cfg").write_text(
        "valid_a.jpg\n__auto__, 20\nvalid_b.jpg\n__auto__, 20\nsky.jpg\n__auto__, 20\n",
        encoding="utf-8",
    )
    (stereo / "fusion.cfg").write_text("valid_a.jpg\nvalid_b.jpg\nsky.jpg\n", encoding="utf-8")

    images = {
        1: _Image(1, "valid_a.jpg", [0.0, 0.0, 0.0]),
        2: _Image(2, "valid_b.jpg", [0.0, 0.0, 0.0]),
        3: _Image(3, "sky.jpg", [0.0, 0.0, -1.0]),
    }
    track = SimpleNamespace(elements=[SimpleNamespace(image_id=1), SimpleNamespace(image_id=2), SimpleNamespace(image_id=3)])
    points = {1: SimpleNamespace(xyz=np.asarray([0.0, 0.0, 1.0]), track=track)}
    reconstruction = SimpleNamespace(images=images, points3D=points)
    monkeypatch.setitem(sys.modules, "pycolmap", SimpleNamespace(Reconstruction=lambda _: reconstruction))

    excluded = pipeline._filter_stereo_workspace(dense)

    assert excluded == ["sky.jpg"]
    assert (stereo / "patch-match.cfg").read_text(encoding="utf-8") == (
        "valid_a.jpg\nvalid_b.jpg\nvalid_b.jpg\nvalid_a.jpg\n"
    )
    assert (stereo / "fusion.cfg").read_text(encoding="utf-8") == "valid_a.jpg\nvalid_b.jpg\n"


def test_filter_stereo_workspace_keeps_828_of_831_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense = tmp_path / "dense"
    (dense / "sparse").mkdir(parents=True)
    stereo = dense / "stereo"
    stereo.mkdir()
    names = [f"image_{index:04d}.jpg" for index in range(831)]
    invalid_names = set(names[-3:])
    (stereo / "patch-match.cfg").write_text(
        "".join(f"{name}\n__auto__, 20\n" for name in names), encoding="utf-8",
    )
    (stereo / "fusion.cfg").write_text("".join(f"{name}\n" for name in names), encoding="utf-8")

    images = {
        index: _Image(index, name, [0.0, 0.0, -1.0 if name in invalid_names else 0.0])
        for index, name in enumerate(names, start=1)
    }
    track = SimpleNamespace(elements=[SimpleNamespace(image_id=index) for index in images])
    reconstruction = SimpleNamespace(
        images=images,
        points3D={1: SimpleNamespace(xyz=np.asarray([0.0, 0.0, 1.0]), track=track)},
    )
    monkeypatch.setitem(sys.modules, "pycolmap", SimpleNamespace(Reconstruction=lambda _: reconstruction))

    excluded = pipeline._filter_stereo_workspace(dense)

    patch_lines = (stereo / "patch-match.cfg").read_text(encoding="utf-8").splitlines()
    retained = patch_lines[::2]
    assert excluded == sorted(invalid_names)
    assert len(retained) == 828
    assert not invalid_names.intersection(patch_lines)


def test_filter_stereo_workspace_strips_excluded_explicit_all_and_fusion_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense = tmp_path / "dense"
    (dense / "sparse").mkdir(parents=True)
    stereo = dense / "stereo"
    stereo.mkdir()
    patch_config = stereo / "patch-match.cfg"
    fusion_config = stereo / "fusion.cfg"
    config = (
        "alpha.jpg\nbeta.jpg, invalid.jpg\n"
        "beta.jpg\n__all__\n"
        "gamma.jpg\n__auto__, 2\n"
        "invalid.jpg\nalpha.jpg\n"
    )
    patch_config.write_text(config, encoding="utf-8")
    fusion_config.write_text(config, encoding="utf-8")

    images = {
        1: _Image(1, "alpha.jpg", [0.0, 0.0, 0.0]),
        2: _Image(2, "beta.jpg", [0.0, 0.0, 0.0]),
        3: _Image(3, "gamma.jpg", [0.0, 0.0, 0.0]),
        4: _Image(4, "invalid.jpg", [0.0, 0.0, -1.0]),
    }
    track = SimpleNamespace(elements=[SimpleNamespace(image_id=index) for index in images])
    reconstruction = SimpleNamespace(
        images=images,
        points3D={1: SimpleNamespace(xyz=np.asarray([0.0, 0.0, 1.0]), track=track)},
    )
    monkeypatch.setitem(sys.modules, "pycolmap", SimpleNamespace(Reconstruction=lambda _: reconstruction))

    assert pipeline._filter_stereo_workspace(dense) == ["invalid.jpg"]
    expected = (
        "alpha.jpg\nbeta.jpg\n"
        "beta.jpg\nalpha.jpg, gamma.jpg\n"
        "gamma.jpg\nalpha.jpg, beta.jpg\n"
    )
    assert patch_config.read_text(encoding="utf-8") == expected
    assert fusion_config.read_text(encoding="utf-8") == expected

    assert pipeline._filter_stereo_workspace(dense) == ["invalid.jpg"]
    assert patch_config.read_text(encoding="utf-8") == expected
    assert fusion_config.read_text(encoding="utf-8") == expected


def test_filter_stereo_workspace_skips_missing_mock_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert pipeline._filter_stereo_workspace(tmp_path / "dense") == []
    assert "workspace filtering was skipped" in capsys.readouterr().out


def test_resume_dense_stereo_rejects_missing_workspace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workspace files are missing"):
        pipeline._resume_dense_stereo(
            SimpleNamespace(), tmp_path / "products", tmp_path / "reports",
            pipeline.RunSummary(tmp_path / "products", {}), None, {}, 10, 1, 0,
            tmp_path / "products" / "classified.ply",
        )
