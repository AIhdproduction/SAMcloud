"""Video import: camera video frames, 360 video folders, validation, project data, command line, native dialogs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tkinter import TclError

import cv2
import numpy as np
import pytest

from samcloud.acquisition import panorama
from samcloud.acquisition.panorama import (
    CaptureValidationError,
    prepare_panorama_inputs,
    validate_capture_configuration,
)
from samcloud.acquisition.video import (
    CAMERA_VIDEO_EXTENSIONS,
    PANORAMA_VIDEO_EXTENSIONS,
    effective_image_directory,
    expected_frame_count,
    extract_frames,
    find_videos,
    prepare_camera_video_frames,
)
from samcloud.app.components import native_dialogs
from samcloud.app.processing import ProcessingManager
from samcloud.core.project_manager import ProjectManager
from samcloud.engine import pipeline

FPS = 10.0


def _checkerboard(width: int, height: int, cell: int = 4) -> np.ndarray:
    rows = (np.arange(height) // cell)[:, None]
    columns = (np.arange(width) // cell)[None, :]
    board = (((rows + columns) % 2) * 255).astype(np.uint8)
    return np.dstack([board, board, board])


def write_video(
    path: Path, frames: int, *, width: int = 64, height: int = 48, sharp: tuple[int, ...] = (), fps: float = FPS,
) -> Path:
    """Tiny synthetic mp4: every frame is a blurred checkerboard, the ``sharp`` indices stay crisp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    board = _checkerboard(width, height)
    blurred = cv2.GaussianBlur(board, (21, 21), 0)
    for index in range(frames):
        writer.write(board if index in sharp else blurred)
    writer.release()
    return path


# ----------------------------------------------------------------------------------------- 1. find_videos


def test_find_videos_file_folder_suffix_and_missing(tmp_path: Path) -> None:
    clip = tmp_path / "a.MP4"
    clip.write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.mov").write_bytes(b"x")
    (tmp_path / "sub" / "notes.txt").write_bytes(b"x")
    (tmp_path / "Z.mkv").write_bytes(b"x")

    assert find_videos(clip, CAMERA_VIDEO_EXTENSIONS) == [clip]
    found = find_videos(tmp_path, CAMERA_VIDEO_EXTENSIONS)
    assert [item.relative_to(tmp_path).as_posix() for item in found] == ["a.MP4", "sub/b.mov", "Z.mkv"]
    assert find_videos(tmp_path, PANORAMA_VIDEO_EXTENSIONS) == [clip]  # 360 accepts mp4 only
    assert find_videos(tmp_path / "sub" / "notes.txt", CAMERA_VIDEO_EXTENSIONS) == []
    assert find_videos(tmp_path / "missing.mp4", CAMERA_VIDEO_EXTENSIONS) == []


# ------------------------------------------------------------------------------------ 2./3. extract_frames


def test_extract_first_pick_matches_the_old_rule_over_two_videos(tmp_path: Path) -> None:
    first = write_video(tmp_path / "v1.mp4", 25)
    second = write_video(tmp_path / "v2.mp4", 25)
    frames = extract_frames([first, second], tmp_path / "frames", 1.0, pick="first")

    assert len(frames) == 6 == expected_frame_count([first, second], 1.0)
    assert [frame.path.name for frame in frames] == [f"frame_{index:06d}.jpg" for index in range(6)]
    assert [frame.source for frame in frames] == [first] * 3 + [second] * 3
    assert [round(frame.time_seconds * FPS) for frame in frames] == [0, 10, 20, 0, 10, 20]
    # the single-video 360 wrapper keeps writing the same names
    wide = write_video(tmp_path / "wide.mp4", 25, width=128, height=64)
    paths = panorama.extract_video_frames(wide, tmp_path / "wide_frames", 1.0)
    assert [path.name for path in paths] == ["frame_000000.jpg", "frame_000001.jpg", "frame_000002.jpg"]


def test_extract_first_pick_applies_the_frame_check(tmp_path: Path) -> None:
    video = write_video(tmp_path / "v.mp4", 25)  # 64x48 is not 2:1
    with pytest.raises(CaptureValidationError, match="2:1"):
        panorama.extract_video_frames(video, tmp_path / "frames", 1.0)


def test_extract_sharpest_pick_takes_the_sharp_frame_of_every_window(tmp_path: Path) -> None:
    video = write_video(tmp_path / "v.mp4", 25, sharp=(3, 14, 22))
    frames = extract_frames([video], tmp_path / "frames", 1.0, pick="sharpest")

    assert len(frames) == 3 == expected_frame_count([video], 1.0)
    assert [round(frame.time_seconds * FPS) for frame in frames] == [3, 14, 22]
    assert [frame.path.name for frame in frames] == [f"frame_{index:06d}.jpg" for index in range(3)]


def test_extract_reports_progress_over_all_videos_and_survives_a_failing_callback(tmp_path: Path) -> None:
    first = write_video(tmp_path / "v1.mp4", 25)
    second = write_video(tmp_path / "v2.mp4", 25)
    calls: list[tuple[int, int, str]] = []
    extract_frames([first, second], tmp_path / "a", 1.0, pick="sharpest",
                   progress_callback=lambda done, total, detail: calls.append((done, total, detail)))
    assert [done for done, _, _ in calls] == [1, 2, 3, 4, 5, 6] and all(total == 6 for _, total, _ in calls)

    def broken(*_args) -> None:
        raise RuntimeError("callback failure")

    assert len(extract_frames([first], tmp_path / "b", 1.0, progress_callback=broken)) == 3


def test_extract_errors_name_the_problem(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(CaptureValidationError, match="broken.mp4"):
        extract_frames([broken], tmp_path / "frames", 1.0)
    with pytest.raises(CaptureValidationError):
        extract_frames([], tmp_path / "frames", 1.0)


# ---------------------------------------------------------------------------- 4. prepare_camera_video_frames


def test_prepare_camera_video_frames_replaces_stale_frames_only(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    write_video(clips / "one.mp4", 25, sharp=(3, 14, 22))
    write_video(clips / "two.mp4", 25)
    frames_directory = tmp_path / "project" / "inputs" / "video_frames"
    frames_directory.mkdir(parents=True)
    for name in ("frame_000000.jpg", "frame_000042.jpg"):
        (frames_directory / name).write_bytes(b"old")
    (frames_directory / "keep.jpg").write_bytes(b"keep")
    (frames_directory / "frame_1.jpg").write_bytes(b"keep")  # not the 6-digit pattern
    (frames_directory / "notes.txt").write_bytes(b"keep")
    (frames_directory / "nested").mkdir()
    (frames_directory / "nested" / "frame_000001.jpg").write_bytes(b"keep")

    prepared = prepare_camera_video_frames(str(clips), frames_directory, 1.0)

    assert (prepared.frame_count, prepared.video_count) == (6, 2)
    assert not (frames_directory / "frame_000042.jpg").exists()
    assert (frames_directory / "frame_000005.jpg").is_file()
    assert (frames_directory / "frame_000000.jpg").read_bytes() != b"old"
    for kept in ("keep.jpg", "frame_1.jpg", "notes.txt"):
        assert (frames_directory / kept).read_bytes() in (b"keep",)
    assert (frames_directory / "nested" / "frame_000001.jpg").read_bytes() == b"keep"
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert prepared.manifest_path == frames_directory.parent / "video_frames_manifest.json"
    assert manifest["pick"] == "sharpest" and manifest["frame_interval_seconds"] == 1.0
    assert [entry["frame"] for entry in manifest["frames"]] == [f"frame_{index:06d}.jpg" for index in range(6)]
    assert manifest["frames"][0]["source_path"].endswith("one.mp4")
    assert round(manifest["frames"][1]["time_seconds"] * FPS) == 14


def test_prepare_camera_video_frames_rejects_a_folder_without_videos(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "readme.txt").write_text("nothing")
    frames_directory = tmp_path / "frames"
    frames_directory.mkdir()
    (frames_directory / "frame_000000.jpg").write_bytes(b"old")
    with pytest.raises(CaptureValidationError):
        prepare_camera_video_frames(str(empty), frames_directory, 1.0)
    assert (frames_directory / "frame_000000.jpg").read_bytes() == b"old"  # nothing is deleted on a failed check
    with pytest.raises(CaptureValidationError):
        prepare_camera_video_frames(str(tmp_path / "missing.mp4"), frames_directory, 1.0)
    clip = write_video(tmp_path / "clip.mp4", 12)
    with pytest.raises(CaptureValidationError):
        prepare_camera_video_frames(str(clip), frames_directory, 0)


# ------------------------------------------------------------------------------------------ 5. validation


def test_validate_camera_photos_or_video_not_both(tmp_path: Path) -> None:
    photos = tmp_path / "photos"
    photos.mkdir()
    clip = write_video(tmp_path / "clip.mp4", 12)
    validate_capture_configuration("camera", image_directory=str(photos), check_paths=True)
    validate_capture_configuration("camera", video_path=str(clip), check_paths=True)
    validate_capture_configuration("camera", video_path=str(tmp_path), check_paths=True)  # folder with a video
    with pytest.raises(CaptureValidationError, match="not both"):
        validate_capture_configuration("camera", image_directory=str(photos), video_path=str(clip))
    with pytest.raises(CaptureValidationError):
        validate_capture_configuration("camera")
    with pytest.raises(CaptureValidationError):
        validate_capture_configuration("camera", video_path=str(tmp_path / "clip.wmv"))
    with pytest.raises(CaptureValidationError):
        validate_capture_configuration("camera", video_path=str(photos), check_paths=True)  # folder without videos
    with pytest.raises(CaptureValidationError):
        validate_capture_configuration("camera", video_path=str(tmp_path / "gone.mp4"), check_paths=True)


def test_validate_drone_rejects_a_video(tmp_path: Path) -> None:
    with pytest.raises(CaptureValidationError):
        validate_capture_configuration("drone_gps", image_directory="D:/images", video_path="D:/data/clip.mp4")
    validate_capture_configuration("drone_gps", image_directory="D:/images")


def test_validate_360_video_folder(tmp_path: Path) -> None:
    folder = tmp_path / "videos"
    write_video(folder / "a.mp4", 12, width=128, height=64)
    validate_capture_configuration("panorama_360", panorama_video_path=str(folder), check_paths=True)
    validate_capture_configuration("panorama_360", panorama_video_path=str(folder))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CaptureValidationError, match="no MP4"):
        validate_capture_configuration("panorama_360", panorama_video_path=str(empty), check_paths=True)
    with pytest.raises(CaptureValidationError, match="MP4"):
        validate_capture_configuration("panorama_360", panorama_video_path=str(tmp_path / "tour.mov"))
    with pytest.raises(CaptureValidationError):
        validate_capture_configuration("panorama_360")


# --------------------------------------------------------------------------- 6. 360 folder of several videos


def test_prepare_panorama_inputs_from_a_folder_of_two_videos(tmp_path: Path) -> None:
    folder = tmp_path / "videos"
    write_video(folder / "a.mp4", 25, width=128, height=64)
    write_video(folder / "b.mp4", 25, width=128, height=64)
    (folder / "skip.mov").write_bytes(b"not used")

    prepared = prepare_panorama_inputs("", str(folder), tmp_path / "inputs", 1.0, face_size=16)

    assert prepared.video_frame_count == 6 and prepared.panorama_count == 6
    assert prepared.cubemap_image_count == 36
    frames = sorted((tmp_path / "inputs" / "panorama_frames").glob("frame_*.jpg"))
    assert [frame.name for frame in frames] == [f"frame_{index:06d}.jpg" for index in range(6)]
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    entries = manifest["panoramas"]
    assert [entry["source_kind"] for entry in entries] == ["video_frame"] * 6
    assert [Path(entry["source_path"]).name for entry in entries] == [frame.name for frame in frames]
    assert manifest["frame_interval_seconds"] == 1.0


# ------------------------------------------------------------------------------------ 7. project manager


def test_create_camera_video_project(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")
    directory = manager.create_project(
        "Clip scan", "", capture_type="camera", class_set="indoor",
        video_path="D:/data/clip.mp4", frame_interval_seconds=2.0,
    )
    project = manager.load(directory)
    acquisition = project["acquisition"]
    assert acquisition["sources"]["video_path"] == "D:/data/clip.mp4"
    assert acquisition["sources"]["image_directory"] == "" and project["paths"]["image_directory"] == ""
    assert acquisition["video"] == {"frame_interval_seconds": 2.0, "frame_directory": "inputs/video_frames"}
    assert effective_image_directory(directory, project) == directory / "inputs" / "video_frames"


def test_create_photo_projects_keep_their_image_directory(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")
    directory = manager.create_project("Photos", "D:/images", capture_type="camera", class_set="indoor")
    project = manager.load(directory)
    assert project["acquisition"]["sources"]["video_path"] == ""
    assert project["acquisition"]["video"]["frame_interval_seconds"] == 1.0
    assert effective_image_directory(directory, project) == Path("D:/images")
    drone = manager.create_project("Drone", "D:/drone", capture_type="drone_gps")
    assert effective_image_directory(drone, manager.load(drone)) == Path("D:/drone")
    assert effective_image_directory(drone, {"acquisition": {"sources": {}}, "paths": {}}) is None


def test_create_project_rejects_photos_and_video_together(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")
    with pytest.raises(ValueError):
        manager.create_project("Both", "D:/images", capture_type="camera", video_path="D:/data/clip.mp4")
    with pytest.raises(ValueError):
        manager.create_project("Drone video", "", capture_type="drone_gps", video_path="D:/data/clip.mp4")


# --------------------------------------------------------------------------------------- 8. command build


def _command(project: dict) -> list[str]:
    return ProcessingManager._pipeline_command(object(), Path("D:/demo"), project, "none")


def test_command_for_a_camera_video_project() -> None:
    project = {
        "settings": {"workflow": {}, "classification": {}, "coordinate_systems": {}},
        "acquisition": {
            "type": "camera",
            "sources": {"image_directory": "", "video_path": "D:/data/clip.mp4"},
            "video": {"frame_interval_seconds": 2.5},
            "georeferencing": {},
        },
    }
    command = _command(project)
    assert command[command.index("--images") + 1] == str(Path("D:/demo") / "inputs" / "video_frames")
    assert command[command.index("--video") + 1] == "D:/data/clip.mp4"
    assert command[command.index("--frame-interval") + 1] == "2.5"


def test_command_for_photo_projects_has_no_video_arguments() -> None:
    for capture_type in ("camera", "drone_gps"):
        project = {
            "settings": {"workflow": {}, "classification": {}, "coordinate_systems": {}},
            "acquisition": {
                "type": capture_type, "sources": {"image_directory": "D:/images", "video_path": ""},
                "georeferencing": {},
            },
        }
        command = _command(project)
        assert command[command.index("--images") + 1] == "D:/images"
        assert "--video" not in command and "--frame-interval" not in command


# --------------------------------------------------------------------------------------------- 9. pipeline


def _args(**values) -> argparse.Namespace:
    base = dict(
        capture_type="camera", images=None, video=None, panorama_images=None, panorama_video=None,
        frame_interval=1.0, cubemap_face_size=0, panorama_workdir=None,
    )
    base.update(values)
    return argparse.Namespace(**base)


def test_pipeline_estimates_for_a_camera_video(tmp_path: Path) -> None:
    video = write_video(tmp_path / "clip.mp4", 25, width=96, height=48)
    frames = tmp_path / "frames"
    frames.mkdir()
    args = _args(images=str(frames), video=str(video))
    assert pipeline._estimate_prepare_work(args) == (3, 3)
    assert pipeline._expected_input_long_side(args) == 96  # the empty frames folder is not consulted
    folder = tmp_path / "folder"
    write_video(folder / "a.mp4", 25)
    write_video(folder / "b.mp4", 25)
    assert pipeline._estimate_prepare_work(_args(images=str(frames), video=str(folder))) == (6, 6)
    assert pipeline._estimate_prepare_work(_args(images=str(frames), video=str(tmp_path / "missing.mp4"))) == (1, 1)


def test_pipeline_estimates_for_a_360_video_folder(tmp_path: Path) -> None:
    folder = tmp_path / "videos"
    write_video(folder / "a.mp4", 25, width=128, height=64)
    write_video(folder / "b.mp4", 25, width=128, height=64)
    args = _args(capture_type="panorama_360", panorama_video=str(folder))
    assert pipeline._estimate_prepare_work(args) == (6, 36)
    assert pipeline._panorama_source_width(args) == 128
    assert pipeline._expected_input_long_side(args) == panorama.native_face_size(128)


def test_resolve_input_images_extracts_camera_video_frames_into_images(tmp_path: Path) -> None:
    video = write_video(tmp_path / "clip.mp4", 25, sharp=(3, 14, 22))
    frames = tmp_path / "project" / "inputs" / "video_frames"
    (frames).mkdir(parents=True)
    (frames / "frame_000099.jpg").write_bytes(b"stale")
    args = _args(images=str(frames), video=str(video))

    images_dir, is_panorama, face_size = pipeline.resolve_input_images(
        args, tmp_path / "products", argparse.ArgumentParser(),
    )

    assert images_dir == frames.resolve() and is_panorama is False and face_size is None
    assert sorted(path.name for path in frames.glob("frame_*.jpg")) == [
        "frame_000000.jpg", "frame_000001.jpg", "frame_000002.jpg",
    ]


def test_resolve_input_images_without_video_keeps_photo_behaviour(tmp_path: Path) -> None:
    photos = tmp_path / "photos"
    photos.mkdir()
    args = _args(images=str(photos))
    assert pipeline.resolve_input_images(args, tmp_path / "out", argparse.ArgumentParser()) == (
        photos.resolve(), False, None,
    )


def test_resolve_input_images_reports_an_unusable_video_as_parser_error(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        pipeline.resolve_input_images(_args(images=str(frames), video=str(tmp_path / "gone.mp4")),
                                      tmp_path / "out", parser)


def test_video_argument_is_camera_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for capture_type in ("drone_gps", "panorama_360"):
        monkeypatch.setattr(sys, "argv", [
            "pipeline", "--capture-type", capture_type, "--video", "D:/data/clip.mp4",
            "--images", str(tmp_path), "--outdir", str(tmp_path / "products"),
        ])
        with pytest.raises(SystemExit) as raised:
            pipeline.main()
        assert raised.value.code == 2


# ------------------------------------------------------------------------------------- 11. native dialogs


class _FakeRoot:
    instances: list["_FakeRoot"] = []

    def __init__(self) -> None:
        self.destroyed = False
        self.calls: list[tuple] = []
        _FakeRoot.instances.append(self)

    def withdraw(self) -> None:
        self.calls.append(("withdraw",))

    def attributes(self, *args) -> None:
        self.calls.append(("attributes", *args))

    def destroy(self) -> None:
        self.destroyed = True


@pytest.fixture
def fake_tk(monkeypatch: pytest.MonkeyPatch):
    _FakeRoot.instances = []
    notices: list[tuple[str, dict]] = []
    monkeypatch.setattr(native_dialogs, "Tk", _FakeRoot)
    monkeypatch.setattr(native_dialogs.ui, "notify", lambda message, **kwargs: notices.append((message, kwargs)))
    return notices


def test_choose_directory_returns_the_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_tk) -> None:
    seen: dict = {}

    def ask(**kwargs):
        seen.update(kwargs)
        return str(tmp_path / "picked")

    monkeypatch.setattr(native_dialogs.filedialog, "askdirectory", ask)
    assert native_dialogs.choose_directory(str(tmp_path), "Pick") == str(tmp_path / "picked")
    assert seen["initialdir"] == str(tmp_path) and seen["title"] == "Pick"
    assert _FakeRoot.instances[0].destroyed and ("attributes", "-topmost", True) in _FakeRoot.instances[0].calls
    native_dialogs.choose_directory(str(tmp_path / "does-not-exist"), "Pick")
    assert seen["initialdir"] is None


def test_choose_file_uses_the_parent_folder_and_the_filetypes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_tk,
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    seen: dict = {}

    def ask(**kwargs):
        seen.update(kwargs)
        return str(clip)

    monkeypatch.setattr(native_dialogs.filedialog, "askopenfilename", ask)
    types = [("MP4", "*.mp4"), ("All files", "*.*")]
    assert native_dialogs.choose_file(str(clip), "Pick", types) == str(clip)
    assert seen["initialdir"] == str(tmp_path) and seen["filetypes"] == types
    native_dialogs.choose_file("", "Pick", types)
    assert seen["initialdir"] is None


def test_empty_selection_is_none(monkeypatch: pytest.MonkeyPatch, fake_tk) -> None:
    monkeypatch.setattr(native_dialogs.filedialog, "askdirectory", lambda **_kwargs: "")
    monkeypatch.setattr(native_dialogs.filedialog, "askopenfilename", lambda **_kwargs: ())
    assert native_dialogs.choose_directory("", "Pick") is None
    assert native_dialogs.choose_file("", "Pick", []) is None
    assert all(root.destroyed for root in _FakeRoot.instances) and fake_tk == []


def test_dialog_failure_notifies_and_returns_none(monkeypatch: pytest.MonkeyPatch, fake_tk) -> None:
    def broken(**_kwargs):
        raise TclError("no display")

    monkeypatch.setattr(native_dialogs.filedialog, "askdirectory", broken)
    monkeypatch.setattr(native_dialogs.filedialog, "askopenfilename", broken)
    assert native_dialogs.choose_directory("", "Pick") is None
    assert native_dialogs.choose_file("", "Pick", []) is None
    assert len(fake_tk) == 2 and all(kwargs.get("type") == "negative" for _, kwargs in fake_tk)
    assert all("no display" in message for message, _ in fake_tk)
    assert all(root.destroyed for root in _FakeRoot.instances)


# ------------------------------------------------------------------------------------- 12. locale parity


def test_new_locale_keys_exist_in_both_languages() -> None:
    base = Path(__file__).resolve().parents[1] / "samcloud" / "app" / "locales"

    def load(language: str, name: str) -> dict:
        return json.loads((base / language / f"{name}.json").read_text(encoding="utf-8"))

    wanted = {
        "projects": [
            ("create", key) for key in (
                "dialog_failed", "camera_video", "camera_video_hint", "select_video_file", "select_video_folder",
                "select_photo_folder",
            )
        ] + [("preflight", "detail_videos")] + [
            ("preflight", "note", key) for key in (
                "video_folder_empty", "videos_unreadable", "camera_photos_and_video", "video_format",
            )
        ],
        "settings": [("capture", "camera_video")],
    }
    for name, paths in wanted.items():
        for path in paths:
            texts = []
            for language in ("de", "en"):
                node = load(language, name)
                for part in path:
                    node = node[part]
                assert isinstance(node, str) and node
                texts.append(node)
            import re
            assert set(re.findall(r"\{(\w+)\}", texts[0])) == set(re.findall(r"\{(\w+)\}", texts[1])), path


@pytest.mark.anyio
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_file_and_folder_buttons_share_one_append_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    from nicegui import ui
    from test_apriltag_gui import _page_user

    built: dict = {}

    def route() -> None:
        field = ui.input("Video")
        native_dialogs.attach_file_button(field, "Pick file", [("All files", "*.*")])
        native_dialogs.attach_folder_button(field, "Pick folder", "video_library")
        built["field"] = field

    route.__module__ = "tests.video_button_route"
    async with _page_user("/video-buttons", route, None, monkeypatch):
        children = list(built["field"].slots["append"].children)
        assert [type(child) for child in children] == [ui.button, ui.button]
        assert [child.props.get("aria-label") for child in children] == ["Pick file", "Pick folder"]
        assert [child.props.get("icon") for child in children] == ["movie", "video_library"]
