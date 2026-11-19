"""Synthetic evaluation against the reference scene (WP5.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from samcloud.bim.core.model import ElementKind, OpeningElement, Scene, WallElement
from samcloud.bim.core.qa import (
    evaluate_against_reference,
    evaluation_section,
    load_tolerances,
    render_evaluation_html,
)
from samcloud.bim.core.qa.evaluation import _to_frame, frame_shift
from samcloud.bim.core.synth.room import RoomSpec, build_room_scene


@pytest.fixture(scope="module")
def reference() -> Scene:
    return build_room_scene(RoomSpec())


def test_reference_against_itself_is_perfect(reference: Scene) -> None:
    report = evaluate_against_reference(reference, reference)
    for kind, detection in report.detection.items():
        assert detection.precision == 1.0 and detection.recall == 1.0, kind
    assert report.detection["wall"].reference == 4
    assert report.detection["window"].matched == 2 and report.detection["door"].matched == 1
    assert report.classification_accuracy == 1.0
    for key, stats in report.errors.items():
        assert stats.max == pytest.approx(0.0, abs=1e-9), key
    assert report.errors["opening.width"].count == 3


def test_missing_and_extra_elements_lower_precision_and_recall(reference: Scene) -> None:
    broken = reference.model_copy(deep=True)
    window = broken.elements_of(ElementKind.WINDOW)[0]
    opening = broken.elements[window.params.opening_id]  # type: ignore[attr-defined]
    del broken.elements[window.id]
    del broken.elements[opening.id]
    assert isinstance(opening, OpeningElement)
    first_wall = broken.elements_of(ElementKind.WALL)[0]
    assert isinstance(first_wall, WallElement)
    report = evaluate_against_reference(broken, reference)
    assert report.detection["opening"].recall == pytest.approx(2 / 3)
    assert report.detection["opening"].precision == 1.0
    assert report.detection["window"].recall == pytest.approx(0.5)
    assert report.unmatched_reference["opening"] == [opening.id]
    swapped = evaluate_against_reference(reference, broken)
    assert swapped.detection["opening"].precision == pytest.approx(2 / 3)
    assert swapped.detection["opening"].recall == 1.0
    assert swapped.unmatched_extracted["opening"] == [opening.id]


def test_dimension_error_is_measured(reference: Scene) -> None:
    changed = reference.model_copy(deep=True)
    for element in changed.elements_of(ElementKind.OPENING):
        assert isinstance(element, OpeningElement)
        element.params.width += 0.05
    report = evaluate_against_reference(changed, reference)
    assert report.errors["opening.width"].mean == pytest.approx(0.05)
    assert report.errors["opening.width"].max == pytest.approx(0.05)
    assert report.errors["opening.height"].max == pytest.approx(0.0, abs=1e-9)


def test_section_needs_a_reference_scene(tmp_path: Path, reference: Scene) -> None:
    (tmp_path / "scene").mkdir()
    reference.save(tmp_path / "scene" / "scene.json")
    assert evaluation_section(tmp_path) is None
    reference.save(tmp_path / "scene" / "reference_scene.json")
    section = evaluation_section(tmp_path)
    assert section is not None
    assert section["reference"] == "scene/reference_scene.json"
    assert section["detection"]["wall"]["recall"] == 1.0
    html_text = render_evaluation_html(section)
    assert "opening.width" in html_text and "<table>" in html_text
    json.dumps(section)


def test_different_origin_offsets_evaluate_with_zero_error(reference: Scene) -> None:
    """Identical absolute geometry in two local frames: no error, full recall."""
    shift = (12.5, -7.25, 0.4)
    ref_offset = reference.settings.origin_offset
    shifted = _to_frame(reference, (-shift[0], -shift[1], -shift[2]))
    shifted.settings.origin_offset = (
        ref_offset[0] + shift[0],
        ref_offset[1] + shift[1],
        ref_offset[2] + shift[2],
    )
    report = evaluate_against_reference(shifted, reference)
    for kind, detection in report.detection.items():
        assert detection.precision == 1.0 and detection.recall == 1.0, kind
    assert report.detection["wall"].matched == 4
    assert report.classification_accuracy == 1.0
    for key, stats in report.errors.items():
        assert stats.max == pytest.approx(0.0, abs=1e-9), key
    assert frame_shift(shifted, reference) == pytest.approx((-shift[0], -shift[1], -shift[2]))
