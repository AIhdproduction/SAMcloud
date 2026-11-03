"""Tests for conditional navigation visibility in the layout sidebar."""

from samcloud.app import layout


def test_navigation_hides_project_pages_without_project() -> None:
    visible_paths = [path for _, path, _ in layout._visible_navigation(False)]
    assert visible_paths == ["/", "/projects", "/settings"]


def test_navigation_shows_all_pages_with_project() -> None:
    visible_paths = [path for _, path, _ in layout._visible_navigation(True)]
    assert visible_paths == [
        "/",
        "/projects",
        "/sparse",
        "/control-points",
        "/dense",
        "/settings",
    ]
