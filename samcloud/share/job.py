"""Background share job: export, upload and QR code with progress in ``reports/share_progress.json``.

The job runs in a worker thread (like ``ProcessingManager`` jobs) so the NiceGUI event loop never blocks. The
UI polls ``read_progress``; every outcome ends in a final state (``done``, ``error`` or ``cancelled``).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from samcloud.share import export
from samcloud.share.export import ShareCancelled
from samcloud.share import project as project_share
from samcloud.share.providers import ProviderError, ShareProvider, get_provider
from samcloud.share.qr import make_qr

LOGGER = logging.getLogger(__name__)

_POINTCLOUD_PHASES: list[tuple[str, str, float]] = [
    ("prepare", "Vorbereiten", 0.15),
    ("octree", "Octree bauen", 0.45),
    ("assemble", "Viewer zusammenstellen", 0.05),
    ("upload", "Hochladen", 0.30),
    ("qr", "QR erzeugen", 0.05),
]
_PROJECT_PHASES: list[tuple[str, str, float]] = [
    ("prepare", "Vorbereiten", 0.12), ("octree", "Octree bauen", 0.40), ("dxf", "DXF aufbereiten", 0.10),
    ("ifc", "IFC aufbereiten", 0.05), ("assemble", "Viewer zusammenstellen", 0.03),
    ("upload", "Hochladen", 0.25), ("qr", "QR erzeugen", 0.05),
]
_ACTIVE_STATES = {"running"}


@dataclass(frozen=True)
class ShareKind:
    """The storage and package rules of one independently shareable product."""

    name: str
    key: str
    progress_relative: Path
    output_relative: Path
    phases: list[tuple[str, str, float]]
    available: Callable[[Path, dict[str, Any] | None], bool]
    build: Callable[..., dict[str, Any]]


# ------------------------------------------------------------------------------------------------
# availability / stored result
# ------------------------------------------------------------------------------------------------

def share_available(project_dir: Path | str, project: dict[str, Any] | None = None) -> bool:
    """True once step 6 (refinement) is completed or the final classified cloud exists."""
    directory = Path(project_dir)
    if project is None:
        project = export._load_project(directory)
    if str(project.get("workflow", {}).get("refinement", "")) == "completed":
        return True
    return (directory / "products" / "classified" / "classified.ply").is_file()


def _build_pointcloud_package(project_dir: Path, package_dir: Path, *, config: dict[str, Any],
                              progress_for_phase: Callable[[int], Callable[[float, str], None]], check: Callable[[], None],
                              max_points: int, converter: Path | None = None) -> dict[str, Any]:
    """The original point-cloud pipeline, retained as the default share kind."""
    points, original = export.prepare_points(project_dir, max_points, progress_for_phase(0), check)
    check()
    metadata = export.build_octree(
        points, package_dir.parent / "work", package_dir / export.OCTREE_DIRECTORY, progress_for_phase(1), check,
        encoding=str(config.get("octree_encoding", "BROTLI")), converter=converter)
    legend = export.assemble_package(project_dir, package_dir, points=points, original_points=original,
                                     metadata=metadata, progress=progress_for_phase(2))
    problems = export.scan_package(package_dir)
    if problems:
        raise RuntimeError("Datenschutzprüfung fehlgeschlagen: " + "; ".join(problems[:5]))
    return {"points": int(legend["points"])}


POINTCLOUD = ShareKind("pointcloud", "share", Path("reports") / "share_progress.json", Path("products") / "share",
                       _POINTCLOUD_PHASES, share_available, _build_pointcloud_package)
PROJECT = ShareKind("project", "project_share", Path("reports") / "project_share_progress.json", Path("products") / "share" / "project",
                    _PROJECT_PHASES, project_share.project_share_available, project_share.build_project_package)

# Backward-compatible aliases for existing imports and point-cloud tests.
PROGRESS_RELATIVE = POINTCLOUD.progress_relative
SHARE_RELATIVE = POINTCLOUD.output_relative
PHASES = POINTCLOUD.phases


def get_share(project_dir: Path | str, project: dict[str, Any] | None = None, *, kind: ShareKind = POINTCLOUD) -> dict[str, Any] | None:
    """The stored ``share:`` block of the project (url, share_id, ...) or ``None``."""
    if project is None:
        project = export._load_project(Path(project_dir))
    block = project.get(kind.key)
    return dict(block) if isinstance(block, dict) and block.get("url") else None


def read_progress(project_dir: Path | str, *, kind: ShareKind = POINTCLOUD) -> dict[str, Any]:
    """Current progress document (``{"state": "idle"}`` when there is none)."""
    try:
        data = json.loads((Path(project_dir) / kind.progress_relative).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"state": "idle", "percent": 0.0}
    return data if isinstance(data, dict) else {"state": "idle", "percent": 0.0}


class _ProgressWriter:
    """Writes the progress document atomically; thread safe."""

    def __init__(self, project_dir: Path, kind: ShareKind = POINTCLOUD) -> None:
        self.kind = kind
        self.path = project_dir / kind.progress_relative
        self.lock = threading.Lock()
        self.doc: dict[str, Any] = {
            "state": "running", "phase": "", "phase_index": -1, "phases": [label for _k, label, _w in kind.phases],
            "phases_done": [], "percent": 0.0, "message": "", "error": None, "result": None,
            "started_at": time.time(), "finished_at": None,
        }

    def update(self, **changes: Any) -> None:
        with self.lock:
            self.doc.update(changes)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.doc, ensure_ascii=False), encoding="utf-8")
            for attempt in range(40):  # Windows refuses the replace while a poller has the file open
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError:
                    if attempt == 39:
                        raise
                    time.sleep(0.025)

    def phase(self, index: int) -> Any:
        key, label, _weight = self.kind.phases[index]
        base = sum(weight for _k, _l, weight in self.kind.phases[:index])
        weight = self.kind.phases[index][2]
        with self.lock:
            done = list(self.doc["phases_done"])
        if key not in done:
            done.append(key)
        self.update(phase=label, phase_index=index, phases_done=done, percent=round(100 * base, 1), message=label)

        def report(fraction: float, message: str = "") -> None:
            fraction = min(max(float(fraction), 0.0), 1.0)
            self.update(percent=round(100 * (base + weight * fraction), 1), message=message or label)

        return report


# ------------------------------------------------------------------------------------------------
# job
# ------------------------------------------------------------------------------------------------

def _remove_below(project_dir: Path, path: Path, kind: ShareKind = POINTCLOUD) -> None:
    """Remove ``path`` when it lies below ``<project>/products/share`` (own staging data only)."""
    root = (project_dir / kind.output_relative).resolve()
    resolved = path.resolve()
    if resolved != root and root in resolved.parents and resolved.exists():
        shutil.rmtree(resolved, ignore_errors=True)


class ShareJob:
    """One export/upload run for one project."""

    def __init__(self, project_dir: Path | str, *, config: dict[str, Any] | None = None,
                 provider: ShareProvider | None = None, max_points: int | None = None,
                 project_manager: Any | None = None, converter: Path | None = None, kind: ShareKind = POINTCLOUD) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.config = config if config is not None else export.load_share_config()
        self.provider = provider
        self.max_points = int(max_points or self.config.get("max_points") or export.DEFAULT_MAX_POINTS)
        self.converter = converter
        self.kind = kind
        self._project_manager = project_manager
        self.cancel_event = threading.Event()
        self.progress = _ProgressWriter(self.project_dir, kind)
        self.thread: threading.Thread | None = None

    # control ------------------------------------------------------------------------------------
    def cancel(self) -> None:
        """Ask the job to stop; it ends in state ``cancelled``."""
        self.cancel_event.set()

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise ShareCancelled()

    def start(self) -> "ShareJob":
        """Run in a daemon thread."""
        self.progress.update(state="running")
        self.thread = threading.Thread(target=self.run, name="share-job", daemon=True)
        self.thread.start()
        return self

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    # execution ----------------------------------------------------------------------------------
    def run(self) -> dict[str, Any] | None:
        """Execute all phases; never raises. Returns the stored share block on success."""
        share_root = self.project_dir / self.kind.output_relative
        package_dir = share_root / "package"
        work_dir = share_root / "work"
        share_id: str | None = None
        provider = self.provider
        try:
            self.progress.update(state="running", started_at=time.time(), error=None, result=None, phases_done=[])
            if not self.kind.available(self.project_dir, None):
                raise RuntimeError(
                    "Verfügbar nach dem DXF/IFC-Export: Punktwolke, DXF und IFC-Modell müssen vorhanden sein."
                    if self.kind.name == PROJECT.name else "Verfügbar nach Schritt 6: die klassifizierte Punktwolke fehlt.")
            provider = provider or get_provider(self.config, self.project_dir)
            self._drop_previous_share()
            for stale in (package_dir, work_dir):
                _remove_below(self.project_dir, stale, self.kind)

            metadata = self.kind.build(self.project_dir, package_dir, config=self.config, progress_for_phase=self.progress.phase,
                                       check=self.check, max_points=self.max_points, converter=self.converter)

            problems = export.scan_package(package_dir)
            if problems:
                raise RuntimeError("Datenschutzprüfung fehlgeschlagen: " + "; ".join(problems[:5]))
            size_mb = round(export.package_size_mb(package_dir), 2)
            self.check()

            report = self.progress.phase(len(self.kind.phases) - 2)
            share_id = secrets.token_urlsafe(16)
            if provider.expiry_days:
                try:
                    provider.ensure_lifecycle(provider.expiry_days)
                except ProviderError as error:  # the rule may already exist; the upload decides success
                    LOGGER.warning("share: lifecycle rule not set: %s", error)

            def upload_progress(done: int, total: int) -> None:
                self.check()
                report(done / total if total else 1.0, f"{done / 1e6:.1f} / {total / 1e6:.1f} MB")

            url = provider.upload(package_dir, share_id, upload_progress)
            self.check()

            report = self.progress.phase(len(self.kind.phases) - 1)
            share_root.mkdir(parents=True, exist_ok=True)
            make_qr(url, share_root / "qr.png", share_root / "qr.svg")
            report(1.0, "QR-Code erzeugt")
            created = datetime.now(timezone.utc)
            block = {
                "url": url, "share_id": share_id, "created_at": created.isoformat(timespec="seconds"),
                "provider": provider.name, "points": int(metadata["points"]), "size_mb": size_mb,
            }
            if self.kind.name == PROJECT.name:
                block["views"] = list(metadata["views"])
                block["levels"] = list(metadata["levels"])
            if provider.expiry_days:
                block["expires_at"] = (created + timedelta(days=provider.expiry_days)).isoformat(timespec="seconds")
            self._store_share(block)
            self.progress.update(state="done", percent=100.0, message="Fertig", result=block, finished_at=time.time())
            return block
        except ShareCancelled:
            self._rollback(provider, share_id)
            self.progress.update(state="cancelled", message="Abgebrochen", finished_at=time.time())
        except Exception as error:  # the dialog must always reach a final state
            LOGGER.exception("share job failed")
            self._rollback(provider, share_id)
            self.progress.update(state="error", error=str(error) or error.__class__.__name__,
                                 message="Fehler", finished_at=time.time())
        finally:
            for staging in (package_dir, work_dir):
                _remove_below(self.project_dir, staging, self.kind)
        return None

    # helpers ------------------------------------------------------------------------------------
    def _manager(self) -> Any:
        if self._project_manager is None:
            from samcloud.core.project_manager import ProjectManager

            self._project_manager = ProjectManager()
        return self._project_manager

    def _store_share(self, block: dict[str, Any]) -> None:
        manager = self._manager()
        project = manager.load(self.project_dir)
        project[self.kind.key] = block
        manager.save(self.project_dir, project)

    def _drop_previous_share(self) -> None:
        previous = get_share(self.project_dir, kind=self.kind)
        if previous is None:
            return
        try:
            get_provider(self.config, self.project_dir, name=previous.get("provider")).delete(str(previous["share_id"]))
        except Exception as error:  # an old link that cannot be removed must not block the new one
            LOGGER.warning("share: previous package could not be removed: %s", error)

    def _rollback(self, provider: ShareProvider | None, share_id: str | None) -> None:
        if provider is not None and share_id:
            try:
                provider.delete(share_id)
            except Exception as error:
                LOGGER.warning("share: rollback delete failed: %s", error)


def delete_share(project_dir: Path | str, *, config: dict[str, Any] | None = None, provider: ShareProvider | None = None,
                 project_manager: Any | None = None, kind: ShareKind = POINTCLOUD) -> bool:
    """Remove the published package and the stored ``share:`` block ("Link löschen"); False when none exists."""
    directory = Path(project_dir).resolve()
    block = get_share(directory, kind=kind)
    if block is None:
        return False
    config = config if config is not None else export.load_share_config()
    provider = provider or get_provider(config, directory, name=block.get("provider"))
    provider.delete(str(block["share_id"]))  # a failure keeps the block so the user can retry
    if project_manager is None:
        from samcloud.core.project_manager import ProjectManager

        project_manager = ProjectManager()
    project = project_manager.load(directory)
    project.pop(kind.key, None)
    project_manager.save(directory, project)
    for name in ("qr.png", "qr.svg"):
        (directory / kind.output_relative / name).unlink(missing_ok=True)
    (directory / kind.progress_relative).unlink(missing_ok=True)
    return True


def share_state(block: dict[str, Any] | None, now: datetime | None = None) -> str:
    """``none`` | ``active`` | ``expired`` for a stored share block (no ``expires_at`` = active)."""
    if not block:
        return "none"
    expires = block.get("expires_at")
    if not expires:
        return "active"
    try:
        moment = datetime.fromisoformat(str(expires))
    except ValueError:
        return "active"
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return "expired" if moment <= (now or datetime.now(timezone.utc)) else "active"


def extend_share(project_dir: Path | str, *, config: dict[str, Any] | None = None, provider: ShareProvider | None = None,
                 project_manager: Any | None = None, kind: ShareKind = POINTCLOUD) -> dict[str, Any]:
    """"Verlängern": restart the lifetime of the published package (same link) and store the new ``expires_at``."""
    directory = Path(project_dir).resolve()
    block = get_share(directory, kind=kind)
    if block is None:
        raise ProviderError("Es gibt keinen geteilten Link.")
    config = config if config is not None else export.load_share_config()
    provider = provider or get_provider(config, directory, name=block.get("provider"))
    expires = provider.extend(str(block["share_id"]))
    block["expires_at"] = expires.isoformat(timespec="seconds")
    if project_manager is None:
        from samcloud.core.project_manager import ProjectManager

        project_manager = ProjectManager()
    project = project_manager.load(directory)
    project[kind.key] = block
    project_manager.save(directory, project)
    return block


class ShareManager:
    """Keeps the running job per project; the UI starts, polls and cancels through it."""

    def __init__(self) -> None:
        self._jobs: dict[tuple[Path, str], ShareJob] = {}
        self._lock = threading.Lock()

    def is_running(self, project_dir: Path | str, *, kind: ShareKind = POINTCLOUD) -> bool:
        with self._lock:
            job = self._jobs.get((Path(project_dir).resolve(), kind.name))
        return job is not None and job.is_alive()

    def start(self, project_dir: Path | str, **options: Any) -> ShareJob:
        """Start a job unless one already runs for the project (then that job is returned)."""
        kind = options.get("kind", POINTCLOUD)
        directory = Path(project_dir).resolve()
        key = (directory, kind.name)
        with self._lock:
            current = self._jobs.get(key)
            if current is not None and current.is_alive():
                return current
            job = ShareJob(directory, **options)
            self._jobs[key] = job
        return job.start()

    def cancel(self, project_dir: Path | str, *, kind: ShareKind = POINTCLOUD) -> bool:
        with self._lock:
            job = self._jobs.get((Path(project_dir).resolve(), kind.name))
        if job is None or not job.is_alive():
            return False
        job.cancel()
        return True

    def status(self, project_dir: Path | str, *, kind: ShareKind = POINTCLOUD) -> dict[str, Any]:
        """Progress document; a ``running`` file without a live job (app restart) is reported as an error."""
        progress = read_progress(project_dir, kind=kind)
        if progress.get("state") in _ACTIVE_STATES and not self.is_running(project_dir, kind=kind):
            progress = {**progress, "state": "error", "error": "Der Vorgang wurde unterbrochen."}
        return progress


share_manager = ShareManager()
