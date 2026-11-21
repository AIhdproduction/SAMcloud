from datetime import datetime, timezone

from samcloud.app.pages.projects import format_created
from samcloud.core.project_manager import PROJECT_FILENAME

ISO = "2026-11-20T17:48:51.696143+00:00"


def _local(pattern):
    return datetime.fromisoformat(ISO).astimezone().strftime(pattern)


def test_iso_utc_german(tmp_path):
    result = format_created({"project": {"created_at": ISO}}, tmp_path, "de")
    assert result.endswith(_local("%d.%m.%Y %H:%M"))
    assert result.startswith("Erstellt")


def test_iso_utc_english(tmp_path):
    result = format_created({"project": {"created_at": ISO}}, tmp_path, "en")
    assert result.endswith(_local("%Y-%m-%d %H:%M"))
    assert result.startswith("Created")


def test_missing_created_at_uses_mtime(tmp_path):
    (tmp_path / PROJECT_FILENAME).write_text("x", encoding="utf-8")
    expected = datetime.fromtimestamp((tmp_path / PROJECT_FILENAME).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    assert format_created({"project": {}}, tmp_path, "en").endswith(expected)


def test_garbage_without_file_returns_none(tmp_path):
    assert format_created({"project": {"created_at": "garbage"}}, tmp_path, "en") is None
    assert format_created({}, tmp_path, "de") is None


class _FakeElement:
    def __init__(self, ui, text=None):
        self.ui, self.text = ui, text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def classes(self, *a, **k):
        return self

    def style(self, *a, **k):
        return self

    def props(self, *a, **k):
        return self


class _FakeUi:
    def __init__(self):
        self.labels = []
        self.buttons = []

    def element(self, *a, **k):
        return _FakeElement(self)

    row = column = element

    def icon(self, *a, **k):
        return _FakeElement(self)

    def label(self, text=""):
        self.labels.append(text)
        return _FakeElement(self)

    def button(self, text="", **k):
        self.buttons.append(text)
        return _FakeElement(self)


def test_home_recent_projects_shows_created_date(tmp_path, monkeypatch):
    from samcloud.app.pages import home

    project = {"project": {"name": "Demo", "created_at": ISO}}

    class _Projects:
        def discover_projects(self):
            return [tmp_path]

        def load(self, directory):
            return project

    class _State:
        projects = _Projects()

    fake = _FakeUi()
    monkeypatch.setattr(home, "ui", fake)
    monkeypatch.setattr(home, "state", _State())
    monkeypatch.setattr(home, "current_language", lambda: "en")
    home._render_recent_projects()
    assert any(label.startswith("Created") and label.endswith(_local("%Y-%m-%d %H:%M")) for label in fake.labels)
