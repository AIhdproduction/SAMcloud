"""Image count and storage check when a project is created (logic, locale parity, form panel)."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from samcloud.acquisition import preflight as pf
from samcloud.acquisition.panorama import expected_video_frame_count

GIB = pf.GIB
DENSE = {"max_image_size": "auto", "auto": {"ratio": 0.55, "minimum": 2000, "maximum": 3200}}
STORAGE_OFF = {"keep_control_point_inputs": True, "keep_classification_inputs": True,
               "keep_refinement_inputs": True, "keep_processing_cache": False}
HUGE = 10 ** 15


def _video(frames: int = 10674, fps: float = 29.97002997002997, width: int = 5760, height: int = 2880,
           size: int = 4_000_000_000, mp4: bool = True) -> pf.VideoInfo:
    return pf.VideoInfo("v.mp4", "ok", mp4, width, height, fps, frames, size)


def _folder(count: int, size: tuple[int, int] = (5760, 2880), nbytes: int = 1000, ignored: int = 0) -> pf.FolderInfo:
    return pf.FolderInfo("f", "ok", count, nbytes, ignored, size)


def _est(probe: pf.SourceProbe, free: int | None = HUGE, interval: float = 1.0, storage: dict | None = None):
    return pf.estimate(probe, interval=interval, storage=STORAGE_OFF if storage is None else storage,
                       free_bytes=free, free_drive="D:", dense_config=DENSE, requested_face_size=1232)


def _write_video(path: Path, frames: int, fps: float) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 32))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((32, 64, 3), index % 255, dtype=np.uint8))
    writer.release()


def _write_image(path: Path, width: int = 64, height: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((height, width, 3), 90, dtype=np.uint8))


# ------------------------------------------------------------------------------------------ frame count


@pytest.mark.parametrize("fps", [25.0, 29.97])
@pytest.mark.parametrize("interval", [1.0, 1.5, 2.0])
def test_frame_count_matches_the_extraction_rule(tmp_path: Path, fps: float, interval: float) -> None:
    video = tmp_path / "clip.mp4"
    _write_video(video, 140, fps)
    info = pf.read_video_info(str(video))
    assert info.status == "ok" and info.width == 64 and info.height == 32 and info.is_2to1
    expected = expected_video_frame_count(video, interval)
    assert expected is not None
    assert pf.kept_frame_count(info.frame_count, info.fps, interval) == expected


def test_unreadable_video_never_raises(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video at all")
    assert pf.read_video_info(str(broken)).status == "unreadable"
    assert pf.read_video_info(str(tmp_path / "missing.mp4")).status == "not_found"
    result = _est(pf.probe_sources("panorama_360", panorama_video_path=str(broken)))
    assert result.state == "unknown" and result.peak_typical == 0
    assert [n.key for n in result.notes] == ["video_unreadable"]


def test_video_warnings_for_wrong_container_and_aspect() -> None:
    result = _est(pf.SourceProbe("panorama_360", _video(width=4000, height=3000, mp4=False)))
    keys = {n.key for n in result.notes}
    assert {"video_not_mp4", "video_not_2to1"} <= keys and result.state == "ready"


# ------------------------------------------------------------------------------------ calibration (a, b)


def test_calibration_360_video() -> None:
    result = _est(pf.SourceProbe("panorama_360", _video()), interval=1.0)
    assert result.panoramas == 356 and result.faces == 2136 and result.dense_size == 1232
    assert result.peak_typical == pytest.approx(81.1e9, rel=0.10)
    assert result.peak_upper > result.peak_typical
    assert result.after == pytest.approx(4.4e9, rel=0.10)


def test_calibration_drone_photos() -> None:
    probe = pf.SourceProbe("drone_gps", None, _folder(291, (5280, 3956), 4_500_525_056))
    result = _est(probe)
    assert result.images == 291 and result.dense_size == 2900
    assert result.peak_typical == pytest.approx(75.7e9, rel=0.10)
    assert result.peak_upper == result.peak_typical  # every photo gets maps
    assert result.after == pytest.approx(9.89 * GIB, rel=0.10)


def test_dense_size_follows_the_free_space_rule() -> None:
    probe = pf.SourceProbe("drone_gps", None, _folder(291, (5280, 3956)))
    roomy = _est(probe, free=HUGE)
    tight = _est(probe, free=40 * GIB)
    assert roomy.dense_size == 2900 and tight.dense_size < roomy.dense_size
    assert tight.peak_typical < roomy.peak_typical


# ------------------------------------------------------------------------------------------ folders


def test_folder_nested_and_upper_case_extensions(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.JPG", 80, 60)
    _write_image(tmp_path / "sub" / "b.jpeg", 80, 60)
    _write_image(tmp_path / "sub" / "deep" / "c.PNG", 80, 60)
    _write_image(tmp_path / "x.bmp")
    (tmp_path / "d.heic").write_bytes(b"x")
    (tmp_path / "e.CR2").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("hi")
    folder = pf.scan_folder(str(tmp_path), pf.PHOTO_FOLDER_EXTENSIONS)
    assert (folder.status, folder.count, folder.ignored) == ("ok", 3, 3)
    assert folder.first_size == (80, 60) and folder.size_bytes > 0
    panorama = pf.scan_folder(str(tmp_path), pf.PANORAMA_PHOTO_EXTENSIONS)
    assert panorama.count == 2 and panorama.ignored == 4  # png now counts as ignored


def test_empty_and_missing_folder(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("nothing")
    (tmp_path / "x.heic").write_bytes(b"x")
    empty = pf.probe_sources("camera", image_directory=str(tmp_path))
    assert empty.folder.status == "no_images" and empty.folder.ignored == 1
    result = _est(empty)
    assert result.state == "unknown" and result.notes[0].key == "folder_no_images"
    assert _est(pf.probe_sources("camera", image_directory=str(tmp_path / "nope"))).notes[0].key == "folder_not_found"
    nothing = _est(pf.probe_sources("camera", image_directory=""))
    assert nothing.state == "empty" and nothing.notes == ()


def test_both_panorama_sources_add_up() -> None:
    video_only = _est(pf.SourceProbe("panorama_360", _video()))
    both = _est(pf.SourceProbe("panorama_360", _video(), _folder(20)))
    assert both.panoramas == 376 and both.faces == 376 * 6 and both.photo_count == 20
    assert both.peak_typical > video_only.peak_typical
    photos_only = _est(pf.SourceProbe("panorama_360", None, _folder(20)))
    assert photos_only.panoramas == 20 and photos_only.state == "ready"


def test_panorama_photo_not_2to1_is_a_warning() -> None:
    result = _est(pf.SourceProbe("panorama_360", None, _folder(3, (4000, 3000))))
    assert "photos_not_2to1" in {n.key for n in result.notes}


# --------------------------------------------------------------------------------- switches and status


def test_storage_switches_change_the_after_figure() -> None:
    probe = pf.SourceProbe("panorama_360", _video())
    keep_cache = _est(probe, storage={**STORAGE_OFF, "keep_processing_cache": True})
    assert keep_cache.after == keep_cache.peak_typical
    default = _est(probe)
    nothing_kept = _est(probe, storage={k: False for k in STORAGE_OFF})
    assert nothing_kept.after < default.after < keep_cache.after
    assert nothing_kept.peak_typical == default.peak_typical  # the peak does not depend on the switches


def test_three_status_levels_and_boundaries() -> None:
    probe = pf.SourceProbe("panorama_360", _video())
    base = _est(probe)
    typical, upper = base.peak_typical, base.peak_upper
    assert _est(probe, free=int(upper * 1.10) + 1).status == "ok"
    assert _est(probe, free=int(upper * 1.10) - 1).status == "tight"
    assert _est(probe, free=typical).status == "tight"
    assert _est(probe, free=typical - 1).status == "short"
    assert _est(probe, free=None).status == "unknown"


def test_advice_interval_is_the_smallest_fitting_step() -> None:
    probe = pf.SourceProbe("panorama_360", _video())
    base = _est(probe, interval=1.0)
    free = int(base.peak_typical * 0.55)
    short = _est(probe, free=free, interval=1.0)
    assert short.status == "short" and short.advice_interval is not None
    advice = short.advice_interval
    assert advice > 1.0 and (advice * 2) == int(advice * 2)  # a multiple of 0.5
    assert _est(probe, free=free, interval=advice).status in ("ok", "tight")
    assert _est(probe, free=free, interval=advice - 0.5).status == "short"
    folder_only = _est(pf.SourceProbe("camera", None, _folder(291, (5280, 3956))), free=10 * GIB)
    assert folder_only.status == "short" and folder_only.advice_interval is None  # advice is for video only


def test_formatting_uses_swiss_separators_and_gib() -> None:
    assert pf.format_count(2136) == "2'136" and pf.format_count(2136, "en") == "2,136"
    assert pf.format_gb(81.1e9) == "76 GB" and pf.format_gb(4.4e9) == "4.1 GB"
    assert pf.format_duration(356.16) == "5:56" and pf.format_duration(3725) == "1:02:05"


# ----------------------------------------------------------------------------------------- locale keys


def _placeholders(text: str) -> set[str]:
    return set(re.findall(r"\{(\w+)\}", text))


def _flat(node: Any, prefix: str = "") -> dict[str, str]:
    if isinstance(node, dict):
        return {k: v for key, value in node.items() for k, v in _flat(value, f"{prefix}{key}.").items()}
    return {prefix[:-1]: node}


def test_locale_key_parity_and_placeholders_and_note_coverage() -> None:
    base = Path(__file__).resolve().parents[1] / "samcloud" / "app" / "locales"
    de = _flat(json.loads((base / "de" / "projects.json").read_text(encoding="utf-8"))["preflight"])
    en = _flat(json.loads((base / "en" / "projects.json").read_text(encoding="utf-8"))["preflight"])
    assert de.keys() == en.keys()
    for key in de:
        assert _placeholders(de[key]) == _placeholders(en[key]), key
    source = (Path(pf.__file__)).read_text(encoding="utf-8")
    for key in set(re.findall(r'Note\("(\w+)"', source)):
        assert f"note.{key}" in de, key


# ------------------------------------------------------------------------------------------- the panel

pytestmark_gui = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _gui():
    from test_apriltag_gui import _element, _page_user, _projects_route

    return _element, _page_user, _projects_route


@pytest.mark.anyio
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_panel_shows_folder_count_and_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.app.components import preflight as panel

    _element, _page_user, _projects_route = _gui()
    monkeypatch.setattr(panel, "DEBOUNCE_SECONDS", 0.05)
    for index in range(4):
        _write_image(tmp_path / "photos" / f"img_{index}.JPG", 400, 300)
    async with _page_user("/projects", _projects_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        assert "Enter a source" in _element(user, "preflight-headline").text
        _element(user, "image-directory-input").value = str(tmp_path / "photos")
        await asyncio.sleep(0.8)
        headline = _element(user, "preflight-headline").text
        assert headline.startswith("4 images, about ") and headline.endswith(" GB")
        assert _element(user, "preflight-status").text in {"Enough storage", "Storage is tight", "Not enough storage"}
        assert _element(user, "preflight-peak").text.startswith("about ")
        _element(user, "image-directory-input").value = str(tmp_path / "does_not_exist")
        await asyncio.sleep(0.8)
        assert "No figures" in _element(user, "preflight-headline").text
        assert "not found" in " ".join(
            c.text for c in user.find(kind=__import__("nicegui").ui.label).elements if getattr(c, "text", ""))


@pytest.mark.anyio
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_stale_result_never_overwrites_a_newer_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.app.components import preflight as panel

    _element, _page_user, _projects_route = _gui()
    monkeypatch.setattr(panel, "DEBOUNCE_SECONDS", 0.05)
    release = threading.Event()
    seen: list[str] = []

    def fake_compute(inputs: panel.PreflightInputs, cache: dict) -> pf.PreflightResult:
        seen.append(inputs.image_directory)
        if inputs.image_directory == "slow":
            release.wait(5)
        images = 111 if inputs.image_directory == "slow" else 222
        return pf.PreflightResult("ready", (), "drone_gps", images=images, peak_typical=GIB, peak_upper=GIB,
                                  after=GIB, free=100 * GIB, free_drive="D:", status="ok",
                                  folder=_folder(images))

    monkeypatch.setattr(panel, "compute", fake_compute)
    async with _page_user("/projects", _projects_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        _element(user, "image-directory-input").value = "slow"
        await asyncio.sleep(0.4)  # the slow run is probing now
        assert "wird" not in _element(user, "preflight-headline").text
        _element(user, "image-directory-input").value = "fast"
        await asyncio.sleep(0.5)
        assert _element(user, "preflight-headline").text.startswith("222 images")
        release.set()
        await asyncio.sleep(0.5)
        assert _element(user, "preflight-headline").text.startswith("222 images")
    assert seen == ["slow", "fast"]


@pytest.mark.anyio
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_panel_recomputes_on_interval_and_switch_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.app.components import preflight as panel

    _element, _page_user, _projects_route = _gui()
    monkeypatch.setattr(panel, "DEBOUNCE_SECONDS", 0.05)
    monkeypatch.setattr(pf, "read_video_info", lambda path: _video())
    monkeypatch.setattr(pf, "free_space", lambda path: (HUGE, "D:"))
    async with _page_user("/projects", _projects_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        with user:
            capture = next(e for e in user.find(kind=__import__("nicegui").ui.select).elements
                           if e.props.get("label") == "Capture type")
            capture.value = "panorama_360"
        _element(user, "panorama-video-input").value = "C:/x/tour.mp4"
        await asyncio.sleep(0.8)
        first = _element(user, "preflight-headline").text
        assert first.startswith("356 panoramas, 2,136 cube faces")
        _element(user, "frame-interval-input").value = 2.0
        await asyncio.sleep(0.8)
        assert _element(user, "preflight-headline").text.startswith(f"{pf.kept_frame_count(10674, 29.97002997002997, 2.0)} panoramas")


@pytest.mark.anyio
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_panel_is_german(monkeypatch: pytest.MonkeyPatch, german_ui: None) -> None:
    _element, _page_user, _projects_route = _gui()
    async with _page_user("/projects", _projects_route, None, monkeypatch) as user:
        await asyncio.sleep(0.3)
        assert "Quelle angeben" in _element(user, "preflight-headline").text


def test_the_page_import_stays_light() -> None:
    import subprocess
    import sys

    code = "import sys; import samcloud.app.components.preflight; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[1]), env={**__import__("os").environ,
                         "PYTHONPATH": str(Path(__file__).resolve().parents[1])}, timeout=120)
    assert out.stdout.strip().endswith("False"), out.stderr[-400:]
