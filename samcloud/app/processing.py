"""Background processing jobs started from the NiceGUI application."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any

from samcloud.core.configuration import load_global_config
from samcloud.core.storage import ensure_text_model, storage_settings
from samcloud.pointcloud.georeference import fit_enu_to_ecef, save_georeference

if TYPE_CHECKING:
    from samcloud.app.state import ApplicationState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WINDOWS_NO_CONSOLE = 0x08000000


def optimize_control_points(project_directory: Path, project: dict[str, Any]) -> bool:
    """Re-run the automatic GCP adjustment from the preserved pristine model only."""
    from samcloud.engine.pipeline import _automatic_control_points

    project_directory = project_directory.resolve()
    products = project_directory / "products"
    sparse = products / "sparse"
    pristine_model = sparse / "pre_gcp"
    pristine_text = sparse / "pre_gcp_txt"
    if not pristine_model.is_dir():
        raise ValueError("The pristine sparse model required for control-point optimization is unavailable")
    ensure_text_model(pristine_model, pristine_text)
    control_settings = dict(load_global_config("defaults.yml").get("control_points", {}))
    # Only values the user edits in the UI override the global defaults; the rest of the project
    # snapshot may be outdated and must not shadow newer detection/gate defaults.
    project_control = project.get("settings", {}).get("control_points", {})
    for key in ("crs", "min_points", "min_observations_per_point"):
        if key in project_control:
            control_settings[key] = project_control[key]
    configured_images = project.get("acquisition", {}).get("sources", {}).get("image_directory") or project.get("paths", {}).get("image_directory")
    image_directory = Path(configured_images) if configured_images else project_directory / "inputs" / "images"
    if not image_directory.is_dir():
        raise ValueError("The original project image directory is unavailable")
    result = _automatic_control_points(
        pristine_model,
        pristine_text,
        image_directory,
        project_directory / "control_points",
        str(control_settings.get("crs", "EPSG:2056+5728")),
        products / "gps_ref.txt",
        control_settings,
        project_directory / "reports" / "control_points_report.json",
        rtk=bool(project.get("acquisition", {}).get("georeferencing", {}).get("rtk_enabled", False)),
    )
    return result is not None


class ProcessingManager:
    """Start the engine without blocking the NiceGUI event loop."""

    def __init__(self, app_state: ApplicationState) -> None:
        self._state = app_state
        self._jobs: dict[Path, subprocess.Popen[str]] = {}
        self._review_jobs: set[Path] = set()
        self._lock = Lock()

    def is_running(self, project_directory: Path) -> bool:
        """Return whether a pipeline or export job is still running."""
        project_directory = project_directory.resolve()
        with self._lock:
            if project_directory in self._review_jobs:
                return True
            process = self._jobs.get(project_directory)
            if process is None:
                return False
            if process.poll() is None:
                return True
            self._jobs.pop(project_directory, None)
            return False

    def optimize_control_points(self, project_directory: Path) -> None:
        """Start only the reviewed GCP adjustment, never a new dense pipeline."""
        project_directory = project_directory.resolve()
        if self.is_running(project_directory):
            raise ValueError("A processing job is already running for this project")
        project = self._state.projects.load(project_directory)
        if not (project_directory / "control_points" / "detections.json").is_file():
            raise ValueError("Automatic control-point detections are not available")
        project.setdefault("workflow", {})["control_points"] = "running"
        project.setdefault("processing", {}).update({
            "last_error": "",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "progress": {"percent": 56, "stage": "Optimizing control points", "detail": ""},
        })
        self._state.projects.save(project_directory, project)
        with self._lock:
            self._review_jobs.add(project_directory)
        Thread(target=self._watch_control_point_optimization, args=(project_directory, project), daemon=True).start()

    def _watch_control_point_optimization(self, project_directory: Path, project: dict[str, Any]) -> None:
        try:
            optimize_control_points(project_directory, project)
            refreshed = self._state.projects.load(project_directory)
            refreshed.setdefault("workflow", {})["control_points"] = "waiting"
            refreshed.setdefault("processing", {}).update({
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "progress": {"percent": 57, "stage": "Control points optimized", "detail": ""},
            })
            self._state.projects.save(project_directory, refreshed)
        except (OSError, ValueError, RuntimeError) as error:
            refreshed = self._state.projects.load(project_directory)
            refreshed.setdefault("workflow", {})["control_points"] = "waiting"
            refreshed.setdefault("processing", {}).update({
                "last_error": str(error),
                "progress": {"percent": 57, "stage": "Control-point optimization failed", "detail": str(error)},
            })
            self._state.projects.save(project_directory, refreshed)
        finally:
            with self._lock:
                self._review_jobs.discard(project_directory)

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
        if mode not in {"none", "after_alignment", "automatic"}:
            raise ValueError("Invalid control-point mode in project settings")

        workflow["alignment"] = "running"
        workflow["control_points"] = "not_started" if mode in {"after_alignment", "automatic"} else "skipped"
        workflow["dense_cloud"] = "queued"
        workflow["classification"] = "queued"
        workflow["refinement"] = "queued"
        workflow["export"] = "not_started"
        project.setdefault("acquisition", {})["locked"] = True
        project.setdefault("processing", {}).update({
            "last_error": "",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": "",
            "progress": {"percent": 0, "stage": "Preparing images", "detail": ""},
        })
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
        georeference = self._ensure_georeference(project_directory)
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
        if georeference is not None:
            command.extend(["--georeference", str(georeference)])

        project.setdefault("workflow", {})["export"] = "running"
        project.setdefault("processing", {}).update({
            "last_error": "",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": "",
            "progress": {"percent": 98, "stage": "Exporting LAS", "detail": ""},
        })
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

    def refine_from_cache(self, project_directory: Path) -> None:
        """Re-run only workflow step 6 from the project's processing cache."""
        from samcloud.engine.processing_cache import load_manifest

        project_directory = project_directory.resolve()
        if self.is_running(project_directory):
            raise ValueError("Processing is already running for this project")
        if load_manifest(project_directory) is None:
            raise ValueError("No processing cache: enable it in the settings and process the project once")
        project = self._state.projects.load(project_directory)
        workflow = project.setdefault("workflow", {})
        workflow["refinement"] = "running"
        project.setdefault("processing", {}).update({
            "last_error": "", "started_at": datetime.now(timezone.utc).isoformat(), "completed_at": "",
            "progress": {"percent": 93, "stage": "Improving and filtering", "detail": ""},
        })
        self._state.projects.save(project_directory, project)
        command = [*self._pipeline_command(project_directory, project, "none"), "--refine-from-cache"]
        try:
            process = self._launch(command, project_directory / "reports" / "pipeline.log")
        except OSError as error:
            workflow["refinement"] = "failed"
            project["processing"]["last_error"] = str(error)
            self._state.projects.save(project_directory, project)
            raise
        with self._lock:
            self._jobs[project_directory] = process
        Thread(target=self._watch_pipeline, args=(project_directory, "refine", process), daemon=True).start()

    def continue_after_control_points(self, project_directory: Path) -> None:
        """Start Phase B after validating that enough GCP clicks are available."""
        project_directory = project_directory.resolve()
        if self.is_running(project_directory):
            raise ValueError("Processing is already running for this project")
        project = self._state.projects.load(project_directory)
        if project.get("workflow", {}).get("control_points") != "waiting":
            raise ValueError("Control points are not waiting for continuation")
        from samcloud.control_points.repository import load_detections, load_observations, read_control_points

        settings = project.get("settings", {}).get("control_points", {})
        minimum_points = int(settings.get("min_points", 3))
        minimum_observations = int(settings.get("min_observations_per_point", 2))
        control_points = project_directory / "control_points"
        points = read_control_points(control_points / "control_points.csv")
        detections = load_detections(control_points / "detections.json")
        observations = load_observations(control_points / "observations.json")
        usable = sum(
            len([item for item in detections.get(str(name), {}).get("observations", []) if item.get("enabled", True)]) >= minimum_observations
            if str(name) in detections else len(observations.get(str(name), [])) >= minimum_observations
            for name in points["name"]
        )
        if usable < minimum_points:
            raise ValueError(f"At least {minimum_points} control points with {minimum_observations} observations are required")
        workflow = project.setdefault("workflow", {})
        workflow.update({"control_points": "running", "dense_cloud": "running", "classification": "queued", "refinement": "queued"})
        project.setdefault("processing", {}).update({"last_error": "", "started_at": datetime.now(timezone.utc).isoformat(), "completed_at": "", "progress": {"percent": 57, "stage": "Applying control points", "detail": ""}})
        self._state.projects.save(project_directory, project)
        command = self._pipeline_command(project_directory, project, "none", resume_after_control_points=True)
        try:
            process = self._launch(command, project_directory / "reports" / "pipeline.log")
        except OSError as error:
            workflow["control_points"] = "failed"
            project["processing"]["last_error"] = str(error)
            self._state.projects.save(project_directory, project)
            raise
        with self._lock:
            self._jobs[project_directory] = process
        Thread(target=self._watch_pipeline, args=(project_directory, "resume", process), daemon=True).start()

    @staticmethod
    def _ensure_georeference(project_directory: Path) -> Path | None:
        """Create a reusable GPS georeference for projects processed before this feature."""
        products = project_directory / "products"
        georeference = products / "georeference.json"
        if georeference.is_file():
            return georeference
        gps_ref = products / "gps_ref.txt"
        model_dir = products / "dense" / "sparse_txt"
        binary_model = products / "dense" / "sparse"
        if not gps_ref.is_file() or not binary_model.is_dir():
            return None
        ensure_text_model(binary_model, model_dir)
        try:
            save_georeference(georeference, fit_enu_to_ecef(model_dir, gps_ref))
        except (OSError, ValueError) as error:
            raise ValueError(f"Could not create GPS georeference for LAS export: {error}") from error
        return georeference

    def _pipeline_command(self, project_directory: Path, project: dict, mode: str, *, resume_after_control_points: bool = False) -> list[str]:
        settings = project.get("settings", {})
        workflow_settings = settings.get("workflow", {})
        acquisition = project.get("acquisition", {})
        sources = acquisition.get("sources", {})
        capture_type = acquisition.get("type", "drone_gps")
        products = project_directory / "products"
        classified_output = products / "classified" / "classified.ply"
        classification = settings.get("classification", {})
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
            classification.get("class_set", "outdoor"),
            "--min-votes",
            str(classification.get("min_votes", 1)),
            "--sam-tile-size",
            str(classification.get("tile_size", 1008)),
            "--sam-tile-min-overlap",
            str(classification.get("tile_min_overlap", 0.2)),
            "--classification-mode",
            classification.get("mode", "segearth_ov3"),
            "--background-threshold",
            str(classification.get("background_threshold", 0.4)),
            "--query-threshold",
            str(classification.get("query_threshold", 0.4)),
            "--geometry-refinement",
            "1" if classification.get("geometry_refinement", True) else "0",
            "--depth-model",
            str(settings.get("refinement", {}).get("depth_model", "large")),
            "--depth-fill",
            "1" if settings.get("refinement", {}).get("depth_fill", True) else "0",
            "--surface-fusion",
            "1" if settings.get("refinement", {}).get("surface_fusion", True) else "0",
            "--keep-processing-cache",
            "1" if storage_settings(settings).get("keep_processing_cache") else "0",
            "--control-points-mode",
            mode,
            "--skip-export",
            "--classified-output",
            str(classified_output),
            "--source-crs",
            settings.get("coordinate_systems", {}).get("working_crs", "LOCAL"),
            "--export-crs",
            settings.get("coordinate_systems", {}).get("export_crs", "LOCAL"),
            "--progress-file",
            str(project_directory / "reports" / "progress.json"),
        ]
        threshold_sweep = classification.get("threshold_sweep", [])
        if threshold_sweep:
            threshold_sweep_value = (
                threshold_sweep
                if isinstance(threshold_sweep, str)
                else ",".join(str(value) for value in threshold_sweep)
            )
            command.extend(["--threshold-sweep", threshold_sweep_value])
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
                str(acquisition.get("panorama", {}).get("cubemap_face_size", 0) or 0),
            ])
        else:
            image_directory = sources.get("image_directory") or project.get("paths", {}).get("image_directory", "")
            command.extend(["--images", image_directory])
            if acquisition.get("georeferencing", {}).get("rtk_enabled", False):
                command.append("--rtk")
        if resume_after_control_points:
            control_points = settings.get("control_points", {})
            command.extend(["--resume-after-control-points", "--control-points-dir", str(project_directory / "control_points"),
                            "--control-points-crs", str(control_points.get("crs", "EPSG:2056+5728"))])
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
        if return_code == 0 and mode == "refine":
            workflow["refinement"] = "completed"
            project.setdefault("processing", {}).update({
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "progress": {"percent": 100, "stage": "Complete", "detail": ""},
            })
        elif return_code == 0:
            workflow["alignment"] = "completed"
            if mode == "after_alignment":
                workflow["control_points"] = "waiting"
            elif mode == "automatic":
                report_path = project_directory / "reports" / "control_points_report.json"
                try:
                    import json
                    passed = bool(json.loads(report_path.read_text(encoding="utf-8")).get("gate", {}).get("passed"))
                except (OSError, ValueError, json.JSONDecodeError):
                    passed = True
                if passed:
                    workflow["control_points"] = "completed"
                    workflow["dense_cloud"] = "completed"
                    workflow["classification"] = "completed"
                    workflow["refinement"] = "completed"
                else:
                    workflow["control_points"] = "waiting"
            elif mode == "resume":
                workflow["control_points"] = "completed"
                workflow["dense_cloud"] = "completed"
                workflow["classification"] = "completed"
                workflow["refinement"] = "completed"
            else:
                workflow["control_points"] = "skipped"
                workflow["dense_cloud"] = "completed"
                workflow["classification"] = "completed"
                workflow["refinement"] = "completed"
            project.setdefault("processing", {}).update({
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "progress": {"percent": 100, "stage": "Complete", "detail": ""},
            })
        else:
            current = "refinement" if mode == "refine" else "alignment" if workflow.get("dense_cloud") == "queued" else "dense_cloud"
            workflow[current] = "failed"
            project.setdefault("processing", {})["last_error"] = (
                f"Pipeline ended with exit code {return_code}. See reports/pipeline.log."
            )
            project.setdefault("processing", {}).setdefault("progress", {}).update({
                "stage": "Failed",
                "detail": project["processing"]["last_error"],
            })
        self._state.projects.save(project_directory, project)
        with self._lock:
            self._jobs.pop(project_directory, None)

    def _watch_export(self, project_directory: Path, process: subprocess.Popen[str]) -> None:
        return_code = process.wait()
        project = self._state.projects.load(project_directory)
        workflow = project.setdefault("workflow", {})
        if return_code == 0:
            workflow["export"] = "completed"
            project.setdefault("processing", {}).update({
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "progress": {"percent": 100, "stage": "Complete", "detail": ""},
            })
        else:
            workflow["export"] = "failed"
            project.setdefault("processing", {})["last_error"] = (
                f"LAS export ended with exit code {return_code}. See reports/export.log."
            )
            project.setdefault("processing", {}).setdefault("progress", {}).update({
                "stage": "Failed",
                "detail": project["processing"]["last_error"],
            })
        self._state.projects.save(project_directory, project)
        with self._lock:
            self._jobs.pop(project_directory, None)
