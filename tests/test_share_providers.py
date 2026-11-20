"""Share providers (local copy / preview server) and the QR code."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from samcloud.share.providers import LocalProvider, ProviderError, get_provider
from samcloud.share.providers.local import stop_preview_servers
from samcloud.share.qr import make_qr

SHARE_ID = "AbCdEfGhIjKlMnOpQrStUv"


def _package(root: Path) -> Path:
    package = root / "package"
    (package / "pointcloud").mkdir(parents=True)
    (package / "index.html").write_text("<html>share</html>", encoding="utf-8")
    (package / "pointcloud" / "octree.bin").write_bytes(b"x" * 5000)
    return package


def test_local_upload_copies_reports_bytes_and_returns_file_url(tmp_path: Path) -> None:
    package = _package(tmp_path)
    provider = LocalProvider(tmp_path / "site")
    calls: list[tuple[int, int]] = []
    url = provider.upload(package, SHARE_ID, lambda done, total: calls.append((done, total)))
    target = tmp_path / "site" / SHARE_ID
    assert (target / "index.html").read_text(encoding="utf-8") == "<html>share</html>"
    assert (target / "pointcloud" / "octree.bin").stat().st_size == 5000
    assert url.startswith("file:///") and url.endswith(f"{SHARE_ID}/index.html")
    assert calls[0][0] == 0 and calls[-1] == (calls[-1][1], calls[-1][1]) and calls[-1][1] > 5000
    assert [c[0] for c in calls] == sorted(c[0] for c in calls)


def test_local_delete_removes_package_and_ignores_unknown_ids(tmp_path: Path) -> None:
    provider = LocalProvider(tmp_path / "site")
    provider.upload(_package(tmp_path), SHARE_ID)
    provider.delete(SHARE_ID)
    assert not (tmp_path / "site" / SHARE_ID).exists()
    provider.delete(SHARE_ID)  # unknown id: no error
    assert (tmp_path / "site").is_dir()


@pytest.mark.parametrize("bad", ["../escape", "a/b", "short", "", "x" * 80, "..\\evil_evil_evil"])
def test_provider_rejects_unsafe_share_ids(tmp_path: Path, bad: str) -> None:
    provider = LocalProvider(tmp_path / "site")
    with pytest.raises(ProviderError):
        provider.upload(_package(tmp_path), bad)
    with pytest.raises(ProviderError):
        provider.delete(bad)


def test_upload_requires_index_html(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ProviderError):
        LocalProvider(tmp_path / "site").upload(tmp_path / "empty", SHARE_ID)


def test_preview_server_serves_the_package_over_http(tmp_path: Path) -> None:
    provider = LocalProvider(tmp_path / "site", serve=True)
    try:
        url = provider.upload(_package(tmp_path), SHARE_ID)
        assert url.startswith("http://127.0.0.1:") and url.endswith(f"/{SHARE_ID}/index.html")
        with urllib.request.urlopen(url, timeout=5) as response:
            assert response.read() == b"<html>share</html>"
    finally:
        stop_preview_servers()


def test_get_provider_from_config(tmp_path: Path) -> None:
    config = {"provider": "local", "providers": {"local": {"directory": None, "serve": False}}}
    provider = get_provider(config, tmp_path / "proj")
    assert isinstance(provider, LocalProvider)
    assert provider.directory == tmp_path / "proj" / "products" / "share" / "site"
    custom = get_provider({"provider": "local", "providers": {"local": {"directory": str(tmp_path / "x")}}})
    assert custom.directory == tmp_path / "x"
    with pytest.raises(ProviderError):
        get_provider({"provider": "unknown-cloud"}, tmp_path)


def test_qr_png_decodes_to_the_url_and_svg_is_written(tmp_path: Path) -> None:
    url = "https://share.example.org/p/AbCdEfGhIjKlMnOpQrStUv/index.html"
    png = make_qr(url, tmp_path / "qr.png", tmp_path / "qr.svg")
    image = cv2.imread(str(png))
    assert image is not None
    decoded, points, _ = cv2.QRCodeDetector().detectAndDecode(image)
    assert decoded == url and points is not None
    svg = (tmp_path / "qr.svg").read_text(encoding="utf-8")
    assert "<svg" in svg


def test_qr_decodes_a_long_file_url(tmp_path: Path) -> None:
    url = (tmp_path / "site" / SHARE_ID / "index.html").as_uri()
    png = make_qr(url, tmp_path / "long.png")
    decoded, _points, _ = cv2.QRCodeDetector().detectAndDecode(cv2.imread(str(png)))
    assert decoded == url
    assert np.asarray(cv2.imread(str(png))).shape[0] > 100


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    import urllib.error

    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def test_preview_server_supports_byte_ranges(tmp_path: Path) -> None:
    package = _package(tmp_path)
    payload = bytes(range(256)) * 20
    (package / "pointcloud" / "octree.bin").write_bytes(payload)
    provider = LocalProvider(tmp_path / "site", serve=True)
    try:
        base = provider.upload(package, SHARE_ID).rsplit("/index.html", 1)[0] + "/pointcloud/octree.bin"
        status, headers, body = _get(base, {"Range": "bytes=10-19"})
        assert status == 206 and body == payload[10:20]
        assert headers["Content-Range"] == f"bytes 10-19/{len(payload)}" and headers["Content-Length"] == "10"
        status, _h, body = _get(base, {"Range": "bytes=5000-"})
        assert status == 206 and body == payload[5000:]
        status, _h, body = _get(base, {"Range": "bytes=-4"})
        assert status == 206 and body == payload[-4:]
        status, headers, _b = _get(base, {"Range": f"bytes={len(payload)}-"})
        assert status == 416 and headers["Content-Range"] == f"bytes */{len(payload)}"
        status, headers, body = _get(base)
        assert status == 200 and body == payload and headers["Accept-Ranges"] == "bytes"
    finally:
        stop_preview_servers()
