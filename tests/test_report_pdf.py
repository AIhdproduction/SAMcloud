"""Processing protocol PDF: generation, robustness, naming, translations, charts and CLI on synthetic data."""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from samcloud.report import charts
from samcloud.report.__main__ import main as cli_main
from samcloud.report.collect import SECTION_KEYS, ProtocolData, Section, collect_protocol_data
from samcloud.report.generate import generate_protocol, list_protocols, next_protocol_path
from samcloud.report.i18n_text import LOCALES, Translator, fmt_duration, fmt_int, fmt_num
from samcloud.report.pdf import build_protocol_pdf
from test_report_collect import make_project


@pytest.fixture(autouse=True)
def chart_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Chart folders of generate_protocol go below the test directory."""
    folder = tmp_path / "chart_tmp"
    folder.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    return folder


def read(path: Path) -> bytes:
    return path.read_bytes()


def test_generate_german_pdf_with_json_and_charts(tmp_path: Path, chart_temp: Path) -> None:
    project = make_project(tmp_path)
    phases: list[tuple[float, str]] = []
    path = generate_protocol(project, language="de", options={"uncompressed": True},
                             progress=lambda fraction, phase: phases.append((fraction, phase)),
                             now=datetime(2030, 1, 2, 3, 4))
    assert path == project / "reports" / "protocol_20300102_0304.pdf"
    content = read(path)
    assert content.startswith(b"%PDF") and len(content) > 5000
    assert b"Verarbeitungsprotokoll" in content
    assert (project / "reports" / "protocol_data.json").is_file()
    assert phases[0][1] == "collect" and phases[-1] == (1.0, "pdf")
    assert {phase for _f, phase in phases} == {"collect", "charts", "pdf"}
    fractions = [fraction for fraction, _p in phases]
    assert fractions == sorted(fractions)
    assert list(chart_temp.glob("samcloud_protocol_charts_*"))  # left in place on purpose


def test_generate_english_pdf(tmp_path: Path) -> None:
    path = generate_protocol(make_project(tmp_path), language="en", options={"uncompressed": True})
    content = read(path)
    assert content.startswith(b"%PDF") and b"Processing Report" in content
    assert b"Verarbeitungsprotokoll" not in content


def test_options_switch_off_camera_list_and_class_details(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    full = generate_protocol(project, language="de", now=datetime(2030, 1, 1, 1, 1))
    small = generate_protocol(project, language="de", options={"camera_list": False, "class_details": False},
                              now=datetime(2030, 1, 1, 1, 2))
    assert small.stat().st_size <= full.stat().st_size


def test_all_sections_error_or_no_data_still_gives_a_pdf(tmp_path: Path) -> None:
    sections = {key: Section(key, "error" if index % 2 else "no_data", None, {}, ["synthetic warning"])
                for index, key in enumerate(SECTION_KEYS)}
    data = ProtocolData(project={"name": "empty", "path": str(tmp_path)}, sections=sections, language="de")
    out = build_protocol_pdf(data, tmp_path / "e.pdf", language="de", options={}, chart_dir=tmp_path / "none")
    assert read(out).startswith(b"%PDF")
    data.sections.clear()  # even a data object without sections
    out = build_protocol_pdf(data, tmp_path / "e2.pdf", language="en", options={}, chart_dir=None)
    assert read(out).startswith(b"%PDF")


@pytest.mark.parametrize("garbage", ["x", 5, ["a", 1], {"a": "b"}, None])
def test_wrong_types_in_every_section_do_not_stop_the_pdf(tmp_path: Path, garbage: object) -> None:
    data = collect_protocol_data(make_project(tmp_path / "p"), language="de")
    for section in data.sections.values():
        for key in list(section.data):
            if key != "not_recorded":
                section.data[key] = copy.deepcopy(garbage)
    out = build_protocol_pdf(data, tmp_path / "garbage.pdf", language="de", options={}, chart_dir=tmp_path / "c")
    assert read(out).startswith(b"%PDF")


def test_nested_wrong_types_do_not_stop_the_pdf(tmp_path: Path) -> None:
    data = collect_protocol_data(make_project(tmp_path / "p"), language="en")
    data.sections["control_points"].data["points"] = "x"
    data.sections["control_points"].data["rms_pass"] = ["not", "a", "dict"]
    data.sections["alignment"].data["images"] = [{"center_xyz": "bad"}, 7, None]
    data.sections["alignment"].data["overlap"] = {"counts": "bad", "origin_xy": 3, "cell_size": "x"}
    data.sections["classification"].data["classes"] = [{"name": "a", "points": "x"}, "b"]
    data.sections["refinement"].data["stages"] = [None, {"stage": 3, "removed": "x"}]
    data.sections["runtimes"].data["steps"] = [{"name": None, "seconds": "x"}]
    out = build_protocol_pdf(data, tmp_path / "nested.pdf", language="en", options={}, chart_dir=tmp_path / "c")
    assert read(out).startswith(b"%PDF")


def test_next_protocol_path_never_overwrites(tmp_path: Path) -> None:
    moment = datetime(2031, 5, 6, 7, 8)
    first = next_protocol_path(tmp_path, moment)
    assert first == tmp_path / "reports" / "protocol_20310506_0708.pdf"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"%PDF")
    second = next_protocol_path(tmp_path, moment)
    assert second.name == "protocol_20310506_0708_2.pdf"
    second.write_bytes(b"%PDF")
    assert next_protocol_path(tmp_path, moment).name == "protocol_20310506_0708_3.pdf"


def test_list_protocols_newest_first_and_ignores_other_files(tmp_path: Path) -> None:
    assert list_protocols(tmp_path) == []
    folder = tmp_path / "reports"
    folder.mkdir()
    files = []
    for index, name in enumerate(("protocol_a.pdf", "protocol_b.pdf", "protocol_c.pdf")):
        path = folder / name
        path.write_bytes(b"%PDF")
        os.utime(path, (1_000_000 + index * 100, 1_000_000 + index * 100))
        files.append(path)
    (folder / "protocol_data.json").write_text("{}", encoding="utf-8")
    (folder / "other.pdf").write_bytes(b"%PDF")
    os.utime(files[0], (2_000_000, 2_000_000))
    assert list_protocols(tmp_path) == [files[0], files[2], files[1]]


def test_cancel_check_stops_between_phases(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    calls: list[int] = []

    def cancel() -> None:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        generate_protocol(project, language="de", cancel_check=cancel)
    assert not list_protocols(project)


def test_cli_returns_zero_and_prints_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = make_project(tmp_path)
    assert cli_main([str(project), "--lang", "en", "--no-camera-list", "--no-class-details"]) == 0
    output = capsys.readouterr().out
    assert "charts" in output and "protocol_" in output
    assert len(list_protocols(project)) == 1
    assert cli_main([str(project), "--json-only"]) == 0
    assert json.loads((project / "reports" / "protocol_data.json").read_text(encoding="utf-8"))["version"] == 1
    assert cli_main([str(tmp_path / "does_not_exist")]) == 1


def test_translations_are_complete_and_used_keys_exist() -> None:
    def flat(prefix: str, value: object, out: dict[str, str]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                flat(f"{prefix}.{key}" if prefix else key, item, out)
        else:
            out[prefix] = str(value)

    catalogues: dict[str, dict[str, str]] = {}
    for language in ("de", "en"):
        catalogues[language] = {}
        flat("", json.loads((LOCALES / language / "protocol.json").read_text(encoding="utf-8")), catalogues[language])
    assert set(catalogues["de"]) == set(catalogues["en"])
    assert all(text.strip() for text in catalogues["de"].values())
    source = "".join((Path(charts.__file__).parent / name).read_text(encoding="utf-8") for name in ("pdf.py", "charts.py"))
    used = set(re.findall(r'\btr\("([a-z_0-9]+\.[a-z_0-9]+)"', source))
    used |= {"summary.rms_pass", "summary.rms_control"}
    assert used, "no translation keys found"
    assert not sorted(key for key in used if key not in catalogues["de"])
    assert "ß" not in "".join(catalogues["de"].values())  # Swiss spelling: no sharp s


def test_translator_is_explicit_per_language_and_falls_back() -> None:
    assert Translator("de")("section.alignment") == "Bildausrichtung"
    assert Translator("en")("section.alignment") == "Image alignment"
    assert Translator("en-US").language == "en"
    assert Translator("de")("does.not.exist") == "does.not.exist"
    assert Translator("de").cls("floor") == "Boden"
    assert Translator("en").cls("nonexistent_class") == "nonexistent_class"


def test_number_formatting() -> None:
    de, en = Translator("de"), Translator("en")
    assert fmt_int(de, 1234567) == "1’234’567" and fmt_int(en, 1234567) == "1,234,567"
    assert fmt_num(de, 0.4349, 2) == "0.43" and fmt_num(de, None) == "–" and fmt_num(de, "x") == "–"
    assert fmt_num(de, True) == "–"
    assert fmt_duration(12.34) == "12.3 s" and fmt_duration(3725) == "1:02:05" and fmt_duration(None) == "–"


def test_charts_return_none_without_data_and_never_raise(tmp_path: Path) -> None:
    tr = Translator("de")
    assert charts.camera_map_chart({}, tmp_path, tr) is None
    assert charts.overlap_chart({"overlap": {"counts": [[0, 0]], "origin_xy": [0, 0], "cell_size": 1}}, tmp_path, tr) is None
    assert charts.error_vector_chart({}, tmp_path, tr) is None
    assert charts.class_chart({"classes": []}, tmp_path, tr) is None
    assert charts.confidence_chart({"confidence_histogram": {"edges": [0, 1], "counts": [1, 2]}}, tmp_path, tr) is None
    assert charts.refinement_chart({"stages": [{"stage": "a"}]}, tmp_path, tr) is None
    assert charts.runtime_chart({"steps": []}, tmp_path, tr) is None
    assert charts.overview_chart({"overview_image": "missing.tif"}, {}, tmp_path, tmp_path, tr) is None
    assert charts.camera_map_chart({"images": "garbage"}, tmp_path, tr) is None
    assert charts.class_chart("garbage", tmp_path, tr) is None  # type: ignore[arg-type]


def test_charts_with_synthetic_data_write_pngs(tmp_path: Path) -> None:
    tr = Translator("en")
    alignment = {
        "images": [{"center_xyz": [i, i * 0.5, 1.0], "reprojection_rms_px": 0.3 + i * 0.01} for i in range(10)],
        "overlap": {"counts": [[0, 2, 3], [1, 5, 0]], "origin_xy": [0.0, 0.0], "cell_size": 1.0},
    }
    control = {"local_points_xy": [
        {"name": "P1", "role": "pass", "x_m": -5.0, "y_m": 2.0, "err_x_m": 0.01, "err_y_m": -0.02},
        {"name": "C1", "role": "control", "x_m": 5.0, "y_m": -2.0, "err_x_m": 0.03, "err_y_m": 0.0}]}
    classification = {"total_points": 100, "classes": [{"name": "floor", "points": 60, "share_pct": 60.0},
                                                       {"name": "wall", "points": 40}],
                      "confidence_histogram": {"edges": [0.0, 0.5, 1.0], "counts": [10, 90]}}
    results = [
        charts.camera_map_chart(alignment, tmp_path, tr), charts.overlap_chart(alignment, tmp_path, tr),
        charts.overview_chart({}, alignment, None, tmp_path, tr), charts.error_vector_chart(control, tmp_path, tr),
        charts.class_chart(classification, tmp_path, tr), charts.confidence_chart(classification, tmp_path, tr),
        charts.refinement_chart({"stages": [{"stage": "a", "removed": 5, "added": 9}]}, tmp_path, tr),
        charts.runtime_chart({"steps": [{"name": "x", "seconds": 61.0}, {"name": "y", "seconds": 5.0}]}, tmp_path, tr),
    ]
    assert all(path is not None and path.is_file() and path.read_bytes().startswith(b"\x89PNG") for path in results)
