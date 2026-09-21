"""Background processing jobs started from the NiceGUI application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from threading import Lock, Thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from samcloud.app.state import ApplicationState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WINDOWS_NO_CONSOLE = 0x08000000


class ProcessingManager:
    """Start the engine without blocking the NiceGUI event loop."""

    def __init__(self, app_state: ApplicationState) -> None:
        self._state = app_state
        self._jobs: dict[Path, subprocess.Popen[str]] = {}
        self._lock = Lock()

    def is_running(self, project_directory: Path) -> bool:
        """Return whether a pipeline or export job is still running."""
        project_directory = project_directory.resolve()
        with self._lock:
            process = self._jobs.get(project_directory)
            if process is None:
                return False
            if process.poll() is None:
                return True
            self._jobs.pop(project_directory, None)
            return False

    def start_pipeline(self, project_directory: Path) -> None:
        """Run alignment, dense reconstruction, and SAM3, leaving LAS manual."""
        project_directory = project_directory.resolve()
        if self.is_running(project_directory):
            raise ValueError("Processing is already running for this project")

        project = self._state.projects.load(project_directory)
        workflow = project.setdefault("workflow", {})
        mode = workflow.get(
            "control_points_mode",
            "after_alignment"
            if project.get("settings", {}).get("workflow", {}).get("control_points_enabled", False)
            else "none",
        )
        if mode not in {"none", "after_alignment"}:
            raise ValueError("Invalid control-point mode in project settings")

        workflow["alignment"] = "running"
        workflow["control_points"] = "not_started" if mode == "after_alignment" else "skipped"
        workflow["dense_cloud"] = "queued"
        workflow["classification"] = "queued"
        workflow["export"] = "not_started"
        project.setdefault("acquisition", {})["locked"] = True
        project.setdefault("processing", {})["last_error"] = ""
        self._state.projects.save(project_directory, project)

        command = self._pipeline_command(project_directory, project, mode)
        log_path = project_directory / "reports" / "pipeline.log"
        try:
            process = self._launch(command, log_path)
        except OSError as error:
            workflow["alignment"] = "failed"
            project.setdefault("processing", {})["last_error"] = str(error)
            self._state.projects.save(project_directory, project)
            raise
        with self._lock:
            self._jobs[project_directory] = process
        Thread(
            target=self._watch_pipeline,
            args=(project_directory, mode, process),
            daemon=True,
        ).start()

    def export_las(self, project_directory: Path) -> None:
        """Export the classified PLY to LAS as a separate manual action."""
        project_directory = project_directory.resolve()
        if self.is_running(project_directory):
            raise ValueError("A processing job is still running for this project")

        project = self._state.projects.load(project_directory)
        classified = project_directory / "products" / "classified" / "classified.ply"
        if not classified.exists():
            raise ValueError("Classified PLY is not available yet")

        settings = project.get("settings", {})
        classification = settings.get("classification", {})
        coordinates = settings.get("coordinate_systems", {})
        class_set = classification.get("class_set", "outdoor")
        output = classified.with_suffix(".las")
        command = [
            sys.executable,
            "-m",
            "samcloud.engine.export_las",
            "--input",
            str(classified),
            "--output",
            str(output),
            "--classes-json",
            str(PROJECT_ROOT / "config" / "classes.json"),
            "--class-set",
            class_set,
            "--source-crs",
            coordinates.get("working_crs", "LOCAL"),
            "--target-crs",
            coordinates.get("export_crs", "LOCAL"),
        ]

        project.setdefault("workflow", {})["export"] = "running"
        project.setdefault("processing", {})["last_error"] = ""
        self._state.projects.save(project_directory, project)
        try:
            process = self._launch(command, project_directory / "reports" / "export.log")
        except OSError as error:
            project.setdefault("workflow", {})["export"] = "failed"
            project.setdefault("processing", {})["last_error"] = str(error)
            self._state.projects.save(project_directory, project)
            raise
        with self._lock:
            self._jobs[project_directory] = process
        Thread(target=self._watch_export, args=(project_directory, process), daemon=True).start()

    def _pipeline_command(self, project_directory: Path, project: dict, mode: str) -> list[str]:
        settings = project.get("settings", {})
        workflow_settings = settings.get("workflow", {})
        acquisition = project.get("acquisition", {})
        sources = acquisition.get("sources", {})
        capture_type = acquisition.get("type", "drone_gps")
        products = project_directory / "products"
        classified_output = products / "classified" / "classified.ply"
        command = [
            sys.executable,
            "-m",
            "samcloud.engine.pipeline",
            "--capture-type",
            capture_type,
            "--outdir",
            str(products),
            "--gpu",
            "1" if workflow_settings.get("use_gpu", True) else "0",
            "--class-set",
            settings.get("classification", {}).get("class_set", "outdoor"),
            "--control-points-mode",
            mode,
            "--skip-export",
            "--classified-output",
            str(classified_output),
            "--source-crs",
            settings.get("coordinate_systems", {}).get("working_crs", "LOCAL"),
            "--export-crs",
            settings.get("coordinate_systems", {}).get("export_crs", "LOCAL"),
        ]
        if capture_type == "panorama_360":
            if sources.get("panorama_image_directory"):
                command.extend(["--panorama-images", sources["panorama_image_directory"]])
            if sources.get("panorama_video_path"):
                command.extend(["--panorama-video", sources["panorama_video_path"]])
            command.extend(["--panorama-workdir", str(project_directory / "inputs")])
            command.extend([
                "--frame-interval",
                str(acquisition.get("panorama", {}).get("frame_interval_seconds", 1.0)),
                "--cubemap-face-size",
                str(acquisition.get("panorama", {}).get("cubemap_face_size", 1024)),
            ])
        else:
            image_directory = sources.get("image_directory") or project.get("paths", {}).get("image_directory", "")
            command.extend(["--images", image_directory])
            if acquisition.get("georeferencing", {}).get("rtk_enabled", False):
                command.append("--rtk")
        return command

    @staticmethod
    def _launch(command: list[str], log_path: Path) -> subprocess.Popen[str]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        try:
            return subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=_WINDOWS_NO_CONSOLE if sys.platform == "win32" else 0,
            )
        except Exception:
            log_file.close()
            raise

    def _watch_pipeline(self, project_directory: Path, mode: str, process: subprocess.Popen[str]) -> None:
        return_code = process.wait()
        project = self._state.projects.load(project_directory)
        workflow = project.setdefault("workflow", {})
        if return_code == 0:
            workflow["alignment"] = "completed"
            if mode == "after_alignment":
                workflow["control_points"] = "waiting"
            else:
                workflow["control_points"] = "skipped"
                workflow["dense_cloud"] = "completed"
                workflow["classification"] = "completed"
        else:
            current = "alignment" if workflow.get("dense_cloud") == "queued" else "dense_cloud"
            workflow[current] = "failed"
            project.setdefault("processing", {})["last_error"] = (
                f"Pipeline ended with exit code {return_code}. See reports/pipeline.log."
            )
        self._state.projects.save(project_directory, project)
        with self._lock:
            self._jobs.pop(project_directory, None)

    def _watch_export(self, project_directory: Path, process: subprocess.Popen[str]) -> None:
        return_code = process.wait()
        project = self._state.projects.load(project_directory)
        workflow = project.setdefault("workflow", {})
        if return_code == 0:
            workflow["export"] = "completed"
        else:
            workflow["export"] = "failed"
            project.setdefault("processing", {})["last_error"] = (
                f"LAS export ended with exit code {return_code}. See reports/export.log."
            )
        self._state.projects.save(project_directory, project)
        with self._lock:
            self._jobs.pop(project_directory, None)
