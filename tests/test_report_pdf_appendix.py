"""Appendix of the protocol: no empty section without log warnings, camera-list hint, CLI flag."""

from __future__ import annotations

from pathlib import Path

from samcloud.report.__main__ import main as cli_main
from samcloud.report.collect import Section
from samcloud.report.generate import list_protocols
from samcloud.report.pdf import _Ctx, sec_appendix
from test_report_collect import make_project
from test_report_pdf_round2 import texts


def make_ctx(language: str, camera_list: bool) -> _Ctx:
    return _Ctx(type("D", (), {"sections": {}, "project": {}})(), language, {"camera_list": camera_list}, None)


def test_appendix_without_warnings_says_so_and_hints_at_camera_list() -> None:
    section = Section("appendix", "ok", "derived", {"cameras_per_image": [{"name": "a.jpg", "camera_id": 1}], "warnings": []}, [])
    de = texts(sec_appendix(make_ctx("de", False), section))
    assert "Keine Warnungen im Lauf-Log gefunden." in de and "Kamera-Liste pro Bild" in de and "--camera-list" in de
    en = texts(sec_appendix(make_ctx("en", False), section))
    assert "No warnings found in the run log." in en and "--camera-list" in en
    enabled = texts(sec_appendix(make_ctx("en", True), section))
    assert "No warnings found" in enabled and "switched off" not in enabled


def test_appendix_with_warnings_lists_them_without_the_empty_line() -> None:
    section = Section("appendix", "ok", "derived", {"cameras_per_image": [], "warnings": ["WARNING: something"]}, [])
    text = texts(sec_appendix(make_ctx("en", False), section))
    assert "WARNING: something" in text and "No warnings found" not in text


def test_cli_camera_list_flag_and_no_camera_list(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    assert cli_main([str(project), "--lang", "en", "--camera-list"]) == 0
    assert cli_main([str(project), "--lang", "en", "--no-camera-list"]) == 0
    assert len(list_protocols(project)) == 2
