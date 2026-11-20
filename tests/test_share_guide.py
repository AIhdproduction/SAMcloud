"""Cloudflare guide text (de/en), best-effort lifecycle rule and the connection test warning."""

from __future__ import annotations

import pytest
from botocore.client import Config
from botocore.stub import Stubber
import boto3

from samcloud.app.i18n import t
from samcloud.share import ui as share_ui
from samcloud.share.providers.cloudflare_r2 import CloudflareR2Provider

CREDS = {"account_id": "acc123", "access_key_id": "AKIAEXAMPLEKEY1234", "secret_access_key": "s3cr3t",
         "bucket": "my-bucket", "public_base_url": "https://pub-abc.r2.dev"}


def _client() -> object:
    return boto3.client("s3", endpoint_url="https://acc123.r2.cloudflarestorage.com", region_name="auto",
                        aws_access_key_id="x", aws_secret_access_key="y", config=Config(signature_version="s3v4"))


def _rule(days: int, prefix: str | None = "share/", status: str = "Enabled") -> dict:
    return {"ID": "samcloud-auto-delete", "Status": status, "Filter": {"Prefix": prefix or ""},
            "Expiration": {"Days": days}}


def test_guide_is_english_by_default(_english_ui: None) -> None:
    html = share_ui.guide_html()
    assert "Guide: Set up Cloudflare R2 for SAMcloud" in html and "Create a bucket" in html
    assert '<a href="https://dash.cloudflare.com/"' in html, "URLs are clickable"
    assert "<code>samcloud-auto-delete</code>" in html and "Never share your API keys" in html
    assert html.count("<ol>") == 6


def test_guide_is_german_verbatim_when_language_is_german(german_ui: None) -> None:
    html = share_ui.guide_html()
    assert "Anleitung: Cloudflare R2 für SAMcloud einrichten" in html
    for text in ("Cloudflare-Konto erstellen", "Öffentlichen Zugriff aktivieren", "Apply to specific buckets only",
                 "Automatische Löschung einrichten", "Delete objects nach 15 Tagen",
                 "Gib deine API-Schlüssel niemals weiter"):
        assert text in html
    assert "Create a bucket" not in html
    assert t("share.setup.lifecycle_missing") == "Löschregel nicht gefunden – bitte Schritt 5 der Anleitung ausführen"


@pytest.mark.parametrize(("rules", "expected"), [
    ([_rule(15)], True), ([_rule(10, prefix=None)], True), ([_rule(30)], False), ([_rule(15, prefix="tmp/")], False),
    ([_rule(15, status="Disabled")], False), ([], False)])
def test_rule_covers(rules: list, expected: bool) -> None:
    provider = CloudflareR2Provider(CREDS, client=_client())
    assert any(provider._rule_covers(rule, 15) for rule in rules) is expected


def test_lifecycle_existing_rule_means_no_put() -> None:
    client = _client()
    with Stubber(client) as stub:
        stub.add_response("get_bucket_lifecycle_configuration", {"Rules": [_rule(15)]}, {"Bucket": "my-bucket"})
        assert CloudflareR2Provider(CREDS, client=client).ensure_lifecycle(15) == "present"
        stub.assert_no_pending_responses()


def test_lifecycle_access_denied_is_reported_not_raised() -> None:
    client = _client()
    with Stubber(client) as stub:
        stub.add_client_error("get_bucket_lifecycle_configuration", "AccessDenied")
        assert CloudflareR2Provider(CREDS, client=client).ensure_lifecycle(15) == "unreadable"
    with Stubber(client) as stub:  # readable but not writable
        stub.add_response("get_bucket_lifecycle_configuration", {"Rules": []}, {"Bucket": "my-bucket"})
        stub.add_client_error("put_bucket_lifecycle_configuration", "AccessDenied")
        assert CloudflareR2Provider(CREDS, client=client).ensure_lifecycle(15) == "missing"


def _patch(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    monkeypatch.setattr(CloudflareR2Provider, "test_connection", lambda self: "ok")
    monkeypatch.setattr(CloudflareR2Provider, "ensure_lifecycle", lambda self, days=None: status)
    monkeypatch.setattr(CloudflareR2Provider, "client", property(lambda self: None))


def test_connection_test_passes_with_warning_when_the_rule_cannot_be_checked(
        monkeypatch: pytest.MonkeyPatch, german_ui: None) -> None:
    _patch(monkeypatch, "missing")
    message, warning = share_ui._test_connection(dict(CREDS))
    assert message == "Verbindung erfolgreich."
    assert warning == "Löschregel nicht gefunden – bitte Schritt 5 der Anleitung ausführen"


def test_unreadable_rules_give_only_an_informational_note(monkeypatch: pytest.MonkeyPatch, german_ui: None) -> None:
    _patch(monkeypatch, "unreadable")
    message, warning = share_ui._test_connection(dict(CREDS))
    assert warning is None
    assert message == ("Verbindung erfolgreich. Die Löschregel kann mit diesem Token nicht geprüft werden – "
                       "wenn du Schritt 5 erledigt hast, ist alles in Ordnung.")


def test_unreadable_note_and_account_id_hint_in_english(monkeypatch: pytest.MonkeyPatch, _english_ui: None) -> None:
    _patch(monkeypatch, "unreadable")
    message, warning = share_ui._test_connection(dict(CREDS))
    assert warning is None and message.startswith("Connection successful. The deletion rule cannot be checked")
    html = share_ui.guide_html()
    assert "Account Details" in html and "<code>.r2.cloudflarestorage.com</code>" in html and "32 characters" in html


def test_german_guide_explains_where_to_find_the_account_id(german_ui: None) -> None:
    html = share_ui.guide_html()
    assert "Account Details" in html and "32 Zeichen" in html and "ohne https://" in html
    assert "<code>.r2.cloudflarestorage.com</code>" in html


@pytest.mark.parametrize("status", ["present", "set"])
def test_connection_test_has_no_warning_with_a_15_day_rule(monkeypatch: pytest.MonkeyPatch, status: str,
                                                           _english_ui: None) -> None:
    _patch(monkeypatch, status)
    message, warning = share_ui._test_connection(dict(CREDS))
    assert warning is None and "15 days" in message


def test_package_ships_every_potree_resource_the_viewer_requests(tmp_path) -> None:
    """URLs written as resourcePath + 'dir/file' in potree.js (and the navigation cube faces) exist in the package."""
    import re
    from pathlib import Path

    import numpy as np

    from samcloud.drone.contracts import PointSet
    from samcloud.share import export

    from test_share_export import make_project

    project = make_project(tmp_path, points=200)
    package = tmp_path / "package"
    points = PointSet(xyz=np.random.default_rng(1).uniform(0, 5, (50, 3)), classes=np.full(50, 2, dtype=np.uint16))
    export.assemble_package(project, package, points=points, original_points=50, metadata={})
    source = (export.POTREE_ASSETS / "build" / "potree" / "potree.js").read_text(encoding="utf-8")
    pattern = r"""resourcePath\}?\s*\+?\s*['"`/]*((?:icons|images|textures)/[A-Za-z0-9_./-]+\.(?:svg|png|jpg))"""
    wanted = set(re.findall(pattern, source))
    wanted |= {"icons/close.svg", "icons/eye_2.png", "images/compas.svg", "textures/matcap/matcap.jpg"}
    wanted |= {f"textures/navigation/{face}.png" for face in "FBLRDU"}
    assert len(wanted) >= 10
    missing = sorted(name for name in wanted if not (package / "lib" / "potree" / "resources" / name).is_file())
    assert not missing, missing
    assert export.scan_package(package) == []
    assert any(p.suffix == ".png" for p in (package / "lib" / "potree" / "resources").rglob("*"))


def test_overlay_panels_stack_above_the_potree_marker_labels() -> None:
    import re

    from samcloud.share import export

    css = (export.TEMPLATE_DIRECTORY / "css" / "share.css").read_text(encoding="utf-8")

    def rule(selector: str) -> str:
        return re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css).group(1)

    def z(selector: str) -> int:
        return int(re.search(r"z-index:\s*(-?\d+)", rule(selector)).group(1))

    potree_max = max(int(v) for v in re.findall(r"z-index:\s*(\d+)", (export.POTREE_ASSETS / "build/potree/potree.css")
                                                  .read_text(encoding="utf-8")))
    assert "isolation: isolate" in rule(".share-area") and z(".share-area") == 0, "labels cannot leave the area"
    for selector in (".share-toolbar", ".share-legend", ".share-status"):
        assert z(selector) > z(".share-area")
    assert z(".share-legend") > potree_max or "isolation: isolate" in rule(".share-area")


def test_potree_viewer_container_is_an_isolated_stacking_context() -> None:
    """Potree labels (z-index up to 10000) must stay below NiceGUI dialogs (q-dialog, z-index ~6000)."""
    import re

    from samcloud.app import potree_viewer

    rule = re.search(r"\.sam-potree \{\{?([^}]*)\}", potree_viewer._HEAD).group(1)
    assert "position:relative" in rule and "z-index:0" in rule and "isolation:isolate" in rule
