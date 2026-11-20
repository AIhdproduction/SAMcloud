"""Local provider: copies the package into a directory and optionally serves it over HTTP for previews."""

from __future__ import annotations

import functools
import io
import logging
import re
import shutil
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from samcloud.share.providers.base import (
    ProgressCallback,
    ProviderError,
    ShareProvider,
    directory_size,
    validate_share_id,
)

LOGGER = logging.getLogger(__name__)
_SERVERS: dict[Path, tuple[ThreadingHTTPServer, threading.Thread]] = {}
_SERVERS_LOCK = threading.Lock()


class _QuietHandler(SimpleHTTPRequestHandler):
    """Static file handler with single-range support (Potree 2 reads octree.bin by byte ranges)."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - signature of the base class
        LOGGER.debug("share preview: " + format, *args)

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):  # type: ignore[override]
        header = self.headers.get("Range")
        if not header:
            return super().send_head()
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
        if not match or not (match.group(1) or match.group(2)):
            return super().send_head()  # unsupported syntax: serve the whole file
        if match.group(1):
            start = int(match.group(1))
            end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
        else:  # suffix range: last N bytes
            start = max(size - int(match.group(2)), 0)
            end = size - 1
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(end - start + 1)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        return io.BytesIO(data)


def stop_preview_servers() -> None:
    """Shut down every preview server started by a ``LocalProvider``."""
    with _SERVERS_LOCK:
        servers = list(_SERVERS.values())
        _SERVERS.clear()
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class LocalProvider(ShareProvider):
    """Copy to ``<directory>/<share_id>``; URL is ``file://`` or, with ``serve``, ``http://host:port/<id>/index.html``.

    Note: browsers block ``fetch`` on ``file://`` pages, so the Potree data only loads over HTTP; use ``serve``
    (or any web server) to view the result.
    """

    name = "local"

    def __init__(self, directory: Path | str, *, serve: bool = False, host: str = "127.0.0.1", port: int = 0) -> None:
        self.directory = Path(directory)
        self.serve = bool(serve)
        self.host = host
        self.port = int(port)

    def target(self, share_id: str) -> Path:
        """Directory that holds the published package."""
        return self.directory / validate_share_id(share_id)

    def upload(self, local_dir: Path, share_id: str, progress_cb: ProgressCallback | None = None) -> str:
        source = Path(local_dir)
        if not (source / "index.html").is_file():
            raise ProviderError("package has no index.html")
        target = self.target(share_id)
        total = directory_size(source)
        done = 0
        if progress_cb:
            progress_cb(0, total)
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            done += path.stat().st_size
            if progress_cb:
                progress_cb(done, total)
        if self.serve:
            return self._ensure_server().url_for(share_id)
        return (target / "index.html").resolve().as_uri()

    def delete(self, share_id: str) -> None:
        target = self.target(share_id)
        if target.is_dir():
            shutil.rmtree(target)

    def _ensure_server(self) -> "_Preview":
        root = self.directory.resolve()
        root.mkdir(parents=True, exist_ok=True)
        with _SERVERS_LOCK:
            entry = _SERVERS.get(root)
            if entry is None:
                handler = functools.partial(_QuietHandler, directory=str(root))
                server = ThreadingHTTPServer((self.host, self.port), handler)
                thread = threading.Thread(target=server.serve_forever, name="share-preview", daemon=True)
                thread.start()
                entry = _SERVERS[root] = (server, thread)
            server = entry[0]
        return _Preview(self.host, server.server_address[1])


class _Preview:
    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port

    def url_for(self, share_id: str) -> str:
        return f"http://{self.host}:{self.port}/{share_id}/index.html"
