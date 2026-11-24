"""Protocol PDF round 2: translated names, not-applicable markers, no coordinate dumps, image names, SAM scores."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from reportlab.platypus import Paragraph, Table

from samcloud.report import metrics
from samcloud.report.collect import Section, collect_protocol_data
from samcloud.report.i18n_text import Translator
from samcloud.report.pdf import _Ctx, sec_alignment, sec_products, sec_refinement, sec_runtimes
from samcloud.report.textutil import drop_coordinates, scrub, short_name, summarize_value
from test_report_collect import make_project, put


def texts(flowables: list[Any]) -> str:
    out: list[str] = []
    for item in flowables:
        if isinstance(item, Paragraph):
            out.append(item.getPlainText())
        elif isinstance(item, Table):
            for row in item._cellvalues:
                out += [cell.getPlainText() for cell in row if isinstance(cell, Paragraph)]
    return "\n".join(out)


def ctx_for(language: str = "de") -> _Ctx:
    return _Ctx(type("D", (), {"sections": {}, "project": {}})(), language, {}, None)


def test_not_applicable_is_not_reported_as_not_recorded(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    (project / "reports" / "apriltag_report.json").unlink()
    report = json.loads((project / "reports" / "control_points_report.json").read_text(encoding="utf-8"))
    put(project, "reports/control_points_report.json", report)
    data = collect_protocol_data(project, language="de")
    control = data.sections["control_points"].data
    assert control["not_applicable"]["apriltag"] == "no_apriltags"
    assert not any(name.startswith("apriltag") for name in control["not_recorded"])
    summary = data.sections["summary"].data
    assert summary["not_applicable"]["apriltag_scale_factor"] == "no_apriltags"
    ctx = ctx_for("de")
    assert ctx.missing(summary, "apriltag_scale_factor") == "keine AprilTags"
    assert ctx_for("en").missing(summary, "apriltag_scale_factor") == "no AprilTags"


def test_no_control_role_points_show_no_control_points(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    report = json.loads((project / "reports" / "control_points_report.json").read_text(encoding="utf-8"))
    for point in report["points"]:
        point["role"] = "pass"
    put(project, "reports/control_points_report.json", report)
    data = collect_protocol_data(project, language="de")
    assert data.sections["control_points"].data["not_applicable"]["rms_control"] == "no_control_points"
    summary = data.sections["summary"].data
    assert ctx_for().missing(summary, "rms_control_x_m") == "keine Kontrollpunkte"
    assert "rms_control_x_m" not in summary["not_recorded"]
    # recorded-by-new-run fields stay "not recorded"
    assert "samcloud_version" in data.sections["title"].data["not_recorded"]


def test_dxf_qa_never_dumps_dicts_or_coordinates(tmp_path: Path) -> None:
    value = {"bbox": [2600123.456, 1200456.789, 0.0], "origin": [2600000.1, 1200000.2], "entities": 21982,
             "schema": "IFC4", "nested": {"center_xyz": [2600000.5, 1.0, 2.0], "count": 3}, "items": [1, 2, 3]}
    cleaned = drop_coordinates(value)
    assert "bbox" not in cleaned and "origin" not in cleaned and "center_xyz" not in cleaned["nested"]
    summary = summarize_value(value, lambda n: str(n), lambda n: f"{n} entries")
    assert len(summary) <= 80 and "{" not in summary and "[" not in summary and not re.search(r"\d{6,}", summary.replace("21982", ""))
    assert summarize_value(2600123.456, str, str) == "\u2013"
    section = Section("products", "ok", "derived", {"files": [], "dxf_qa": {"passed": 0, "failed": 1, "rules": [
        {"rule": "extents", "ok": False, "value": value, "threshold": {"bounds": [2600000.0, 1200000.0], "max": 5}}]}}, [])
    text = texts(sec_products(ctx_for("en"), section))
    assert "bbox" not in text and "2600" not in text and "1200456" not in text and "{" not in text and "[" not in text


def test_scrub_removes_paths_and_coordinate_numbers() -> None:
    text = scrub("failed at -0.59332255953081747 -1.7467627898459306 in D:\\Customer Street 5\\proj\\x.ply, east 2600123.5")
    assert "0.5933" not in text and "Customer" not in text and "2600123" not in text and "x.ply" in text
    assert scrub("panorama_000189.jpg, 1.1571, 5'206'811") == "panorama_000189.jpg, 1.1571, 5'206'811"


def test_image_names_show_only_basename_or_face() -> None:
    assert short_name("C:\\Users\\someone\\Street 5\\photos\\IMG_0001.JPG") == "IMG_0001.JPG"
    assert short_name("/data/Some Street/frames/pano_000001.jpg") == "pano_000001.jpg"
    assert short_name("D:/x/Some Street/front/panorama_000001.jpg") == "front/panorama_000001.jpg"
    assert short_name("back\\panorama_2.jpg") == "back/panorama_2.jpg"
    section = Section("alignment", "ok", "derived", {"unregistered_images": ["D:/Private Street 9/pics/a.jpg", "q/up/b.jpg"]}, [])
    text = texts(sec_alignment(ctx_for("de"), section))
    assert "Private" not in text and "a.jpg" in text and "up/b.jpg" in text


def test_names_are_translated_with_humanised_fallback() -> None:
    de, en = Translator("de"), Translator("en")
    assert de.step("Tie point filtering pass 2") == "Filterung der Verkn\u00fcpfungspunkte, Durchgang 2"
    assert en.step("Bundle adjustment pass 1") == "Bundle adjustment, pass 1"
    assert de.step("environment: load") == "Umgebung: Laden"
    assert de.step("Dense stereo") == "Dichte Stereo-Berechnung"
    assert de.stage("surface_fusion.removed_hidden") == "Verdeckte Punkte entfernt"
    assert de.reason("not run") == "nicht ausgef\u00fchrt" and en.reason("not run") == "not run"
    assert de.name("gps", "Not used") == "nicht verwendet"
    for raw in ("some_new.stage_key", "Some unknown step"):
        label = de.stage(raw) if "." in raw else de.step(raw)
        assert "_" not in label and "." not in label
    assert de.stage("surface_fusion.brand_new_removal") == "Brand new removal"


def test_refinement_counts_mirrors_once_and_evidence_is_no_added_points() -> None:
    stages = [
        {"stage": "mirror_detection", "removed": 197070, "detected": 7, "seconds": 5.0},
        {"stage": "depth_fill", "before": 100, "after": 600, "added": 500, "seconds": 9.0},
        {"stage": "surface_fusion", "before": 600, "after": 300, "removed": 300, "seconds": 3.0},
        {"stage": "surface_fusion.mirror_points_removed", "removed": 197070},
        {"stage": "surface_fusion.removed_hidden", "removed": 67872},
        {"stage": "depth_evidence", "added": 15391527, "seconds": 4.0},
        {"stage": "plane_fill", "skipped": True, "reason": "not run"},
    ]
    section = Section("refinement", "ok", "run", {"stages": stages, "final_points": 300}, [])
    text = texts(sec_refinement(ctx_for("de"), section))
    assert text.count("197\u2019070") == 1
    assert "Spiegel erkannt: 7" in text
    assert "\u21b3 Spiegelpunkte entfernt" in text and "\u21b3 Verdeckte Punkte entfernt" in text
    assert "15.4 Mio. Punkte (nicht Teil der Wolke)" in text and "15\u2019391\u2019527" not in text
    assert "not run" not in text and "nicht ausgef\u00fchrt" in text
    assert "surface_fusion" not in text


def test_runtime_step_names_are_translated() -> None:
    section = Section("runtimes", "ok", "derived", {"steps": [{"name": "Tie point filtering pass 1", "seconds": 3.0},
                                                               {"name": "environment: tin", "seconds": 2.0}]}, [])
    text = texts(sec_runtimes(ctx_for("de"), section))
    assert "Durchgang 1" in text and "Umgebung: Dreiecksnetz (TIN)" in text and "pass" not in text


def test_confidence_histogram_excludes_placeholder_scores(tmp_path: Path) -> None:
    classes = np.array([1, 1, 1, 1, 2, 2, 2, 0], dtype=np.uint8)
    conf = np.array([0.3, 0.9, 0.5, 0.0, 0.5, 0.7, 1.0, 0.0], dtype=np.float16)
    path = tmp_path / "classified_scores.npz"
    np.savez(path, top1_class=classes, top1_conf=conf)
    plain = metrics.confidence_histogram(path, bins=10, class_names=["wall", "floor"])
    assert plain["without_own_score"] == 1 and plain["points"] == 6  # conf 0 with a class is never a real score
    filled = metrics.confidence_histogram(path, bins=10, class_names=["wall", "floor"], placeholder_values=(0.5, 1.0))
    assert filled["points"] == 3 and filled["without_own_score"] == 4 and filled["unknown_points"] == 1
    assert filled["per_class"]["wall"]["points"] == 2 and filled["per_class"]["floor"]["points"] == 1
    assert filled["per_class"]["floor"]["p50"] == 0.7 or abs(filled["per_class"]["floor"]["p50"] - 0.7) < 0.01
