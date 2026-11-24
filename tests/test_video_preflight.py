"""Create-form preflight for camera videos and 360 video folders."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from samcloud.acquisition import preflight as pf
from samcloud.acquisition.video import expected_frame_count
from test_video_import import write_video

GIB = pf.GIB
DENSE = {"max_image_size": "auto", "auto": {"ratio": 0.55, "minimum": 2000, "maximum": 3200}}
STORAGE = {"keep_control_point_inputs": True, "keep_classification_inputs": True,
           "keep_refinement_inputs": True, "keep_processing_cache": False}
HUGE = 10 ** 15


def _est(probe: pf.SourceProbe, free: int | None = HUGE, interval: float = 1.0):
    return pf.estimate(probe, interval=interval, storage=STORAGE, free_bytes=free, free_drive="D:",
                       dense_config=DENSE, requested_face_size=1232)


def test_camera_video_file(tmp_path: Path) -> None:
    clip = write_video(tmp_path / "clip.mp4", 25, width=64, height=48)
    probe = pf.probe_sources("camera", video_path=str(clip))
    assert len(probe.videos) == 1 and not probe.video_folder
    result = _est(probe)

    assert result.state == "ready" and result.capture_type == "camera"
    assert result.images == 3 == result.video_panoramas == expected_frame_count([clip], 1.0)
    assert result.video_count == 1 and result.video_duration == pytest.approx(2.5, abs=0.2)
    assert result.source_bytes == clip.stat().st_size and result.notes == ()
    # the extracted frames stay in the project: the photo estimate for the same size plus the frames
    assert result.frames_bytes == int(3 * 64 * 48 * pf.VIDEO_FRAME_JPEG_BYTES_PER_PIXEL) > 0
    photos = _est(pf.SourceProbe("camera", None, pf.FolderInfo("f", "ok", 3, 0, 0, (64, 48))))
    assert result.peak_typical - photos.peak_typical == result.frames_bytes
    assert result.peak_upper - photos.peak_upper == result.frames_bytes
    assert result.after - photos.after == result.frames_bytes


def test_camera_video_folder_with_an_unreadable_video(tmp_path: Path) -> None:
    folder = tmp_path / "clips"
    write_video(folder / "a.mp4", 25)
    write_video(folder / "b.mp4", 25)
    (folder / "broken.mp4").write_bytes(b"not a video at all")
    result = _est(pf.probe_sources("camera", video_path=str(folder)))

    assert result.state == "ready" and result.images == 6
    assert result.video_count == 2 and result.videos_unreadable == 1
    assert [(n.key, dict(n.params)) for n in result.notes] == [("videos_unreadable", {"count": 1})]


def test_camera_video_folder_without_videos_and_wrong_format(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "readme.txt").write_text("x")
    result = _est(pf.probe_sources("camera", video_path=str(empty)))
    assert result.state == "unknown" and [n.key for n in result.notes] == ["video_folder_empty"]

    clip = write_video(tmp_path / "clip.mp4", 25)
    renamed = tmp_path / "clip.wmv"
    shutil.copy(clip, renamed)
    result = _est(pf.probe_sources("camera", video_path=str(renamed)))
    assert result.state == "unknown" and result.images == 0
    assert [n.key for n in result.notes] == ["video_format"] and ".mkv" in result.notes[0].params["extensions"]

    missing = _est(pf.probe_sources("camera", video_path=str(tmp_path / "gone.mp4")))
    assert missing.state == "unknown" and [n.key for n in missing.notes] == ["video_not_found"]


def test_camera_never_gets_the_360_notes(tmp_path: Path) -> None:
    clip = write_video(tmp_path / "clip.mp4", 25, width=64, height=48)  # not 2:1
    keys = {n.key for n in _est(pf.probe_sources("camera", video_path=str(clip))).notes}
    assert not keys & {"video_not_mp4", "video_not_2to1"}
    mov = tmp_path / "clip.mov"
    shutil.copy(clip, mov)
    assert "video_not_mp4" not in {n.key for n in _est(pf.probe_sources("camera", video_path=str(mov))).notes}


def test_camera_photos_and_video_is_unknown_without_numbers(tmp_path: Path) -> None:
    photos = tmp_path / "photos"
    photos.mkdir()
    clip = write_video(tmp_path / "clip.mp4", 25)
    result = _est(pf.probe_sources("camera", image_directory=str(photos), video_path=str(clip)))
    assert result.state == "unknown" and [n.key for n in result.notes] == ["camera_photos_and_video"]
    assert (result.images, result.peak_typical, result.after) == (0, 0, 0)


def test_360_video_folder_sums_the_frames(tmp_path: Path) -> None:
    folder = tmp_path / "videos"
    write_video(folder / "a.mp4", 25, width=128, height=64)
    write_video(folder / "b.mp4", 25, width=128, height=64)
    (folder / "ignored.mov").write_bytes(b"x")
    probe = pf.probe_sources("panorama_360", panorama_video_path=str(folder))
    result = _est(probe)

    assert probe.video_folder and len(probe.videos) == 2
    assert result.state == "ready" and result.panoramas == 6 == result.video_panoramas and result.faces == 36
    assert result.video_count == 2 and result.frames_bytes == 0
    single = _est(pf.probe_sources("panorama_360", panorama_video_path=str(folder / "a.mp4")))
    assert single.panoramas == 3 and single.video_count == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    assert [n.key for n in _est(pf.probe_sources("panorama_360", panorama_video_path=str(empty))).notes] == [
        "video_folder_empty",
    ]


def test_preflight_frame_count_matches_the_extraction_rule(tmp_path: Path) -> None:
    folder = tmp_path / "clips"
    write_video(folder / "a.mp4", 37, fps=25.0)
    write_video(folder / "b.mp4", 140, fps=29.97)
    videos = sorted(folder.glob("*.mp4"))
    probe = pf.probe_sources("camera", video_path=str(folder))
    for interval in (0.5, 1.0, 1.5, 2.0):
        assert _est(probe, interval=interval).images == expected_frame_count(videos, interval)


def test_advice_interval_for_a_camera_video_when_space_is_short() -> None:
    video = pf.VideoInfo("v.mp4", "ok", True, 3840, 2160, 30.0, 9000, 10 ** 9)
    probe = pf.SourceProbe("camera", video, None, (video,))
    base = _est(probe, interval=1.0)
    free = int(base.peak_typical * 0.5)
    short = _est(probe, free=free, interval=1.0)

    assert short.status == "short" and short.advice_interval is not None and short.advice_interval > 1.0
    assert _est(probe, free=free, interval=short.advice_interval).status in ("ok", "tight")
    assert _est(probe, free=free, interval=short.advice_interval - 0.5).status == "short"


def test_panel_inputs_carry_the_camera_video_path() -> None:
    from samcloud.app.components import preflight as panel

    inputs = panel.PreflightInputs(
        capture_type="camera", image_directory="", panorama_video_path="", panorama_image_directory="",
        interval=1.0, storage=tuple(STORAGE.items()), projects_directory=".", video_path="D:/data/clip.mp4",
    )
    cache: dict = {}
    result = panel.compute(inputs, cache)
    assert result.state == "unknown" and cache["key"][-1] == "D:/data/clip.mp4"
    assert [n.key for n in result.notes] == ["video_not_found"]


@pytest.mark.anyio
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_form_shows_the_camera_video_and_its_frame_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from nicegui import ui

    from samcloud.app.components import preflight as panel
    from test_apriltag_gui import _element, _page_user, _projects_route

    monkeypatch.setattr(panel, "DEBOUNCE_SECONDS", 0.05)
    clip = write_video(tmp_path / "clip.mp4", 25)
    async with _page_user("/projects", _projects_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        with user:
            capture = next(e for e in user.find(kind=ui.select).elements if e.props.get("label") == "Capture type")
            capture.value = "camera"
        await user.should_not_see(marker="frame-interval-input")  # hidden until a video is entered
        _element(user, "camera-video-input").value = str(clip)
        await asyncio.sleep(0.8)
        assert _element(user, "frame-interval-input").visible
        assert _element(user, "preflight-headline").text.startswith("3 images, about ")
        _element(user, "frame-interval-input").value = 0.5
        await asyncio.sleep(0.8)
        assert _element(user, "preflight-headline").text.startswith("5 images, about ")


@pytest.mark.anyio
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_camera_outdoor_hint_follows_capture_type_and_class_set(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from nicegui import ui

    from test_apriltag_gui import _page_user, _projects_route

    def select(user, label: str):
        return next(e for e in user.find(kind=ui.select).elements if e.props.get("label") == label)

    def hint_visible(user) -> bool:
        return any(
            "camera-outdoor-hint" in getattr(e, "_markers", []) and e.visible
            for e in user.find(kind=ui.label).elements
        )

    async with _page_user("/projects", _projects_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        with user:
            assert not hint_visible(user)  # drone + outdoor by default
            select(user, "Capture type").value = "camera"
            await asyncio.sleep(0.1)
            assert hint_visible(user)
            select(user, "Semantic class set").value = "indoor"
            await asyncio.sleep(0.1)
            assert not hint_visible(user)
            select(user, "Semantic class set").value = "outdoor"
            await asyncio.sleep(0.1)
            assert hint_visible(user)
            select(user, "Capture type").value = "drone_gps"
            await asyncio.sleep(0.1)
            assert not hint_visible(user)
