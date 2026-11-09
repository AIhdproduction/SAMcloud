"""Tests for the Potree viewer integration and fallback behaviour."""

from __future__ import annotations

import json


def test_script_embeds_config_and_markers() -> None:
    from samcloud.app import potree_viewer

    config = {
        "element_id": "sam-potree-dense",
        "kind": "dense",
        "title": "Dense Cloud",
        "metadata_url": "/api/potree/dense/metadata.json",
        "class_colors": [[1, 2, 3]],
        "class_labels": ["Boden"],
        "marker_endpoint": "/api/control-points/markers",
    }

    script_content = potree_viewer._script(config)

    prefix = "const config = "
    start_pos = script_content.index(prefix) + len(prefix)
    end_pos = script_content.index(";\n", start_pos)
    json_chunk = script_content[start_pos:end_pos]
    parsed_config = json.loads(json_chunk)

    assert parsed_config == config
    assert parsed_config["metadata_url"] == "/api/potree/dense/metadata.json"
    assert script_content.count("/api/potree/dense/metadata.json") == 1


def test_render_falls_back_to_canvas_viewer_without_potree(monkeypatch) -> None:
    import samcloud.app.pointcloud_viewer as pointcloud_viewer
    import samcloud.app.potree_viewer as potree_viewer
    import samcloud.viewer.potree as potree_module

    monkeypatch.setattr(potree_module, "potree_available", lambda: False, raising=False)

    def _unexpected_potree_render(*args, **kwargs):
        raise AssertionError("render_potree_viewer was called while potree is unavailable")

    monkeypatch.setattr(potree_viewer, "render_potree_viewer", _unexpected_potree_render)

    head_calls: list[str] = []
    html_calls: list[tuple[str, bool]] = []
    body_calls: list[str] = []

    monkeypatch.setattr(pointcloud_viewer.ui, "add_head_html", head_calls.append)
    monkeypatch.setattr(
        pointcloud_viewer.ui,
        "html",
        lambda content, sanitize=False: html_calls.append((content, sanitize)),
    )
    monkeypatch.setattr(pointcloud_viewer.ui, "add_body_html", body_calls.append)

    pointcloud_viewer.render_pointcloud_viewer("sparse", title="x")

    assert len(body_calls) == 1
    assert "/api/pointcloud/sparse" in body_calls[0]


def test_render_uses_potree_when_available(monkeypatch) -> None:
    import samcloud.app.pointcloud_viewer as pointcloud_viewer
    import samcloud.app.potree_viewer as potree_viewer
    import samcloud.viewer.potree as potree_module

    monkeypatch.setattr(potree_module, "potree_available", lambda: True, raising=False)

    recorded_calls: list[tuple[str, dict]] = []

    def _record_potree(kind: str, **kwargs):
        recorded_calls.append((kind, kwargs))

    monkeypatch.setattr(potree_viewer, "render_potree_viewer", _record_potree)

    pointcloud_viewer.render_pointcloud_viewer(
        "sparse",
        title="Sparse Cloud",
        marker_endpoint="/api/control-points/markers",
    )

    assert len(recorded_calls) == 1
    kind, kwargs = recorded_calls[0]
    assert kind == "sparse"
    assert kwargs.get("marker_endpoint") == "/api/control-points/markers"


def test_wide_page_layout_class(monkeypatch) -> None:
    from samcloud.app import layout

    recorded: list[str] = []

    class _FakeElement:
        def __init__(self, tag: str = "") -> None:
            self.tag = tag

        def classes(self, classes_str: str) -> _FakeElement:
            recorded.append(classes_str)
            return self

        def props(self, *args, **kwargs) -> _FakeElement:
            return self

        def mark(self, *markers: str) -> _FakeElement:
            return self

        def __enter__(self) -> _FakeElement:
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            return None

    class _FakeUI:
        def add_head_html(self, *args, **kwargs) -> None:
            pass

        def colors(self, *args, **kwargs) -> None:
            pass

        def dark_mode(self, *args, **kwargs) -> None:
            pass

        def element(self, tag: str) -> _FakeElement:
            return _FakeElement(tag)

        def html(self, *args, **kwargs) -> _FakeElement:
            return _FakeElement("html")

        def button(self, *args, **kwargs) -> _FakeElement:
            return _FakeElement("button")

        def label(self, *args, **kwargs) -> _FakeElement:
            return _FakeElement("label")

    monkeypatch.setattr(layout, "ui", _FakeUI())

    recorded.clear()
    layout.render_page("x", lambda: None, current_path="/dense", wide=True)
    assert any("sam-page-wide" in item for item in recorded)
    assert "sam-page sam-page-wide" in recorded

    recorded.clear()
    layout.render_page("x", lambda: None, current_path="/dense", wide=False)
    assert not any("sam-page-wide" in item for item in recorded)
    assert "sam-page" in recorded


def test_sparse_toolbar_has_no_class_buttons_but_marker_toggle() -> None:
    from samcloud.app.potree_viewer import _html

    content = _html("x", show_classes=False, show_markers=True)
    assert 'data-action="markers"' in content
    assert 'data-action="classification"' not in content
    assert 'data-action="classes"' not in content


def test_dense_toolbar_keeps_class_buttons() -> None:
    from samcloud.app.potree_viewer import _html

    content = _html("x", show_classes=True, show_markers=False)
    assert 'data-action="classification"' in content
    assert 'data-action="classes"' in content
    assert 'data-action="markers"' not in content


def test_render_sets_show_flags(monkeypatch) -> None:
    from samcloud.app import potree_viewer

    head_calls: list[str] = []
    html_calls: list[str] = []
    body_calls: list[str] = []

    class _FakeElement:
        def classes(self, *args, **kwargs) -> _FakeElement:
            return self

    class _FakeUI:
        def add_head_html(self, content: str) -> None:
            head_calls.append(content)

        def html(self, content: str, sanitize: bool = False) -> _FakeElement:
            html_calls.append(content)
            return _FakeElement()

        def add_body_html(self, content: str) -> None:
            body_calls.append(content)

    fake_ui = _FakeUI()
    monkeypatch.setattr(potree_viewer, "ui", fake_ui)

    potree_viewer.render_potree_viewer(
        "sparse",
        title="t",
        class_colors=[[1, 2, 3]],
        marker_endpoint="/m",
    )
    assert len(body_calls) == 1
    prefix = "const config = "
    start_pos = body_calls[0].index(prefix) + len(prefix)
    end_pos = body_calls[0].index(";\n", start_pos)
    sparse_config = json.loads(body_calls[0][start_pos:end_pos])
    assert sparse_config["show_classes"] is False
    assert sparse_config["show_markers"] is True

    body_calls.clear()
    potree_viewer.render_potree_viewer(
        "dense",
        title="t",
        class_colors=[[1, 2, 3]],
    )
    assert len(body_calls) == 1
    start_pos = body_calls[0].index(prefix) + len(prefix)
    end_pos = body_calls[0].index(";\n", start_pos)
    dense_config = json.loads(body_calls[0][start_pos:end_pos])
    assert dense_config["show_classes"] is True
    assert dense_config["show_markers"] is False


def test_marker_double_click_navigates_to_control_points() -> None:
    from samcloud.app import potree_viewer

    config = {
        "element_id": "sam-potree-sparse",
        "kind": "sparse",
        "title": "Sparse Cloud",
        "metadata_url": "/api/potree/sparse/metadata.json",
        "class_colors": [],
        "class_labels": [],
        "marker_endpoint": "/api/control-points/markers",
    }
    script = potree_viewer._script(config)
    assert "/control-points?marker=${encodeURIComponent(marker.name)}" in script
    assert "dblclick" in script


def test_review_script_selects_requested_marker() -> None:
    from samcloud.app.pages import gcp_review

    script = gcp_review._workspace_script()
    assert "get('marker')" in script


