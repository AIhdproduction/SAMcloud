"""Background processing jobs started from the NiceGUI application."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any

from samcloud.acquisition.video import effective_image_directory
from samcloud.app.i18n import catalogue, t
from samcloud.classification.custom import project_classes_path
from samcloud.core.configuration import load_global_config
from samcloud.core.project_manager import apriltags_enabled
from samcloud.core.storage import ensure_text_model, storage_settings
from samcloud.engine.auto_dxf import auto_dxf_enabled
from samcloud.pointcloud.georeference import ensure_project_georeference

if TYPE_CHECKING:
    from samcloud.app.state import ApplicationState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WINDOWS_NO_CONSOLE = 0x08000000


def _project_control_points_mode(project: dict[str, Any]) -> str:
    """Return the configured mode, defaulting legacy project files to coordinate-aware auto."""
    mode = project.get("workflow", {}).get("control_points_mode", "auto")
    if mode not in {"none", "after_alignment", "automatic", "auto"}:
        raise ValueError("invalid control-point mode")
    return mode


def optimize_control_points(project_directory: Path, project: dict[str, Any]) -> bool:
    """Re-run the automatic GCP adjustment from the preserved pristine model only."""
    from samcloud.engine.pipeline import _rerun_control_points

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
    configured_images = effective_image_directory(project_directory, project)
    cubemap_directory = project_directory / "inputs" / "cubemap"
    capture_type = project.get("acquisition", {}).get("capture_type") or project.get("acquisition", {}).get("type")
    if capture_type == "panorama_360" and cubemap_directory.is_dir():
        image_directory = cubemap_directory
    else:
        image_directory = configured_images if configured_images else project_directory / "inputs" / "images"
    if not image_directory.is_dir():
        raise ValueError("The original project image directory is unavailable")
    result = _rerun_control_points(
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
            raise ValueError(t("processing.error.job_running"))
        project = self._state.projects.load(project_directory)
        if not (project_directory / "control_points" / "detections.json").is_file():
            raise ValueError(t("processing.error.detections_missing"))
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
            raise ValueError(t("processing.error.already_running"))

        project = self._state.projects.load(project_directory)
        workflow = project.setdefault("workflow", {})
        try:
            mode = _project_control_points_mode(project)
        except ValueError:
            raise ValueError(t("processing.error.invalid_control_point_mode"))

        workflow["alignment"] = "running"
        workflow["control_points"] = "not_started" if mode in {"after_alignment", "automatic", "auto"} else "skipped"
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
            raise ValueError(t("processing.error.job_still_running"))

        project = self._state.projects.load(project_directory)
        classified = project_directory / "products" / "classified" / "classified.ply"
        if not classified.exists():
            raise ValueError(t("processing.error.classified_missing"))

        settings = project.get("settings", {})
        classification = settings.get("classification", {})
        instance_settings = classification.get("instances", {})
        coordinates = settings.get("coordinate_systems", {})
        class_set = classification.get("class_set", "outdoor")
        engine = classification.get("engine", "sam3")
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
            str(project_classes_path(project_directory)),
            "--class-set",
            class_set,
            "--engine",
            engine,
            "--source-crs",
            coordinates.get("working_crs", "LOCAL"),
            "--target-crs",
            coordinates.get("export_crs", "LOCAL"),
        ]
        if georeference is not None:
            command.extend(["--georeference", str(georeference)])
        # outdoor/drone projects with a processing boundary: the LAS product is clipped to it (classified.ply stays raw)
        from samcloud.drone.perimeter import BOUNDARY_GEOJSON, clip_polygon_lv95

        if clip_polygon_lv95(project_directory) is not None:
            command.extend(["--clip-geojson", str(project_directory / BOUNDARY_GEOJSON)])

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
            raise ValueError(t("processing.error.already_running"))
        if load_manifest(project_directory) is None:
            raise ValueError(t("processing.error.no_cache"))
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

    def reclassify(self, project_directory: Path) -> None:
        """Re-run only the classification (and step 6) on the finished dense cloud with the current settings."""
        project_directory = project_directory.resolve()
        if self.is_running(project_directory):
            raise ValueError(t("processing.error.already_running"))
        if not reclassification_available(project_directory):
            raise ValueError(t("processing.error.dense_not_kept"))
        project = self._state.projects.load(project_directory)
        workflow = project.setdefault("workflow", {})
        workflow["classification"] = "running"
        project.setdefault("processing", {}).update({
            "last_error": "", "started_at": datetime.now(timezone.utc).isoformat(), "completed_at": "",
            "progress": {"percent": 87, "stage": "Reclassifying", "detail": ""},
        })
        self._state.projects.save(project_directory, project)
        command = [*self._pipeline_command(project_directory, project, "none"), "--reclassify"]
        try:
            process = self._launch(command, project_directory / "reports" / "pipeline.log")
        except OSError as error:
            workflow["classification"] = "failed"
            project["processing"]["last_error"] = str(error)
            self._state.projects.save(project_directory, project)
            raise
        with self._lock:
            self._jobs[project_directory] = process
        Thread(target=self._watch_pipeline, args=(project_directory, "reclassify", process), daemon=True).start()

    def continue_after_control_points(self, project_directory: Path) -> None:
        """Start Phase B after validating that enough GCP clicks are available."""
        project_directory = project_directory.resolve()
        if self.is_running(project_directory):
            raise ValueError(t("processing.error.already_running"))
        project = self._state.projects.load(project_directory)
        if project.get("workflow", {}).get("control_points") != "waiting":
            raise ValueError(t("processing.error.control_points_not_waiting"))
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
            raise ValueError(t("processing.error.control_points_insufficient", points=minimum_points, observations=minimum_observations))
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
        return ensure_project_georeference(project_directory)

    def _pipeline_command(self, project_directory: Path, project: dict, mode: str, *, resume_after_control_points: bool = False) -> list[str]:
        settings = project.get("settings", {})
        workflow_settings = settings.get("workflow", {})
        acquisition = project.get("acquisition", {})
        sources = acquisition.get("sources", {})
        capture_type = acquisition.get("type", "drone_gps")
        products = project_directory / "products"
        classified_output = products / "classified" / "classified.ply"
        classification = settings.get("classification", {})
        instance_settings = classification.get("instances", {})
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
            "--classes-json",
            str(project_classes_path(project_directory)),
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
            "--classification-engine",
            classification_engine(classification),
            "--background-threshold",
            str(classification.get("background_threshold", 0.4)),
            "--query-threshold",
            str(classification.get("query_threshold", 0.4)),
            "--instances-enabled",
            "1" if instance_settings.get("enabled", True) else "0",
            "--instance-query-threshold",
            str(instance_settings.get("query_threshold", 0.5)),
            "--instance-mask-threshold",
            str(instance_settings.get("mask_threshold", 0.5)),
            "--instance-min-mask-pixels",
            str(instance_settings.get("min_mask_pixels", 12)),
            "--instance-merge-overlap",
            str(instance_settings.get("merge_overlap", 0.5)),
            "--instance-min-shared-points",
            str(instance_settings.get("min_shared_points", 10)),
            "--instance-min-views",
            str(instance_settings.get("min_views", 2)),
            "--instance-min-points",
            str(instance_settings.get("min_points", 100)),
            "--instance-split-radius-m",
            str(instance_settings.get("split_radius_m", 0.05)),
            "--instance-debug-ply",
            "1" if instance_settings.get("debug_ply", False) else "0",
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
            "--auto-dxf",
            "1" if auto_dxf_requested(settings) else "0",
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
            video_source = sources.get("video_path") if capture_type == "camera" else ""
            if video_source:
                image_directory = str(effective_image_directory(project_directory, project))
            else:
                image_directory = sources.get("image_directory") or project.get("paths", {}).get("image_directory", "")
            command.extend(["--images", image_directory])
            if video_source:
                command.extend([
                    "--video", video_source,
                    "--frame-interval",
                    str(acquisition.get("video", {}).get("frame_interval_seconds", 1.0)),
                ])
            if acquisition.get("georeferencing", {}).get("rtk_enabled", False):
                command.append("--rtk")
        if apriltags_enabled(project):
            command.append("--apriltags")
        control_points = settings.get("control_points", {})
        command.extend(["--control-points-crs", str(control_points.get("crs", "EPSG:2056+5728"))])
        if resume_after_control_points:
            command.extend(["--resume-after-control-points", "--control-points-dir", str(project_directory / "control_points")])
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
        launched_at = time.time()
        return_code = process.wait()
        try:  # per-run copy of pipeline.log for the processing protocol (never overwrites, never raises)
            from samcloud.report.run_metrics_writer import archive_pipeline_log

            archive_pipeline_log(project_directory, not_before=launched_at - 5)
        except Exception:  # noqa: BLE001
            pass
        project = self._state.projects.load(project_directory)
        workflow = project.setdefault("workflow", {})
        if return_code == 0 and mode == "refine":
            workflow["refinement"] = "completed"
            project.setdefault("processing", {}).update({
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "progress": {"percent": 100, "stage": "Complete", "detail": ""},
            })
        elif return_code == 0 and mode == "reclassify":
            workflow["classification"] = "completed"
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
            current = ("refinement" if mode == "refine" else "classification" if mode == "reclassify"
                       else "alignment" if workflow.get("dense_cloud") == "queued" else "dense_cloud")
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
            classification = project.setdefault("settings", {}).setdefault("classification", {})
            classification["model_profile"] = (
                "flurseg_v1" if classification.get("engine") == "flurseg" else f"sam3_{classification.get('class_set', 'outdoor')}"
            )
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


CLASSIFICATION_ENGINES = {"sam3", "flurseg"}


def classification_engine(classification: dict) -> str:
    """Engine stored in the project settings; FlurSeg only applies to the outdoor class set."""
    engine = str(classification.get("engine", "sam3") or "sam3")
    if engine not in CLASSIFICATION_ENGINES or classification.get("class_set", "outdoor") != "outdoor":
        return "sam3"
    return engine


def settings_use_flurseg(settings: dict | None) -> bool:
    """True when the project settings select the FlurSeg engine (point clouds only, no DXF/IFC)."""
    classification = (settings or {}).get("classification", {}) if isinstance(settings, dict) else {}
    return isinstance(classification, dict) and classification_engine(classification) == "flurseg"


def dxf_ifc_allowed(settings: dict | None) -> bool:
    """DXF/IFC exist only for SAM3 projects; FlurSeg projects deliver point clouds only."""
    return not settings_use_flurseg(settings)


def auto_dxf_requested(settings: dict | None) -> bool:
    """Whether ``--auto-dxf 1`` is passed: the project setting AND a SAM3 engine."""
    return dxf_ifc_allowed(settings) and auto_dxf_enabled(settings)


def project_uses_flurseg(project_directory: Path | str | None) -> bool:
    """Read the project file of a directory and tell whether it classifies with FlurSeg (False when unreadable)."""
    if project_directory is None:
        return False
    from samcloud.core.configuration import load_yaml
    from samcloud.core.project_manager import PROJECT_FILENAME

    try:
        project = load_yaml(Path(project_directory) / PROJECT_FILENAME)
    except Exception:  # noqa: BLE001 - missing/corrupt project file: do not hide pages because of it
        return False
    return settings_use_flurseg(project.get("settings") if isinstance(project, dict) else None)


def reclassification_available(project_directory: Path) -> bool:
    """Whether the finished dense cloud (fused.ply, visibility, dense cameras) was kept for a reclassification."""
    dense = Path(project_directory) / "products" / "dense"

    def kept(name: str) -> bool:
        return (dense / name).is_file() or (dense / f"{name}.gz").is_file()

    return kept("fused.ply") and kept("fused.ply.vis") and (dense / "sparse").is_dir()


# Progress texts persisted by this module (project JSON / progress.json stay English); translated only for display.
_PROGRESS_TEXT_KEYS = {
    "Optimizing control points": "processing.stage.optimizing_control_points",
    "Control points optimized": "processing.stage.control_points_optimized",
    "Control-point optimization failed": "processing.stage.control_point_optimization_failed",
    "Exporting LAS": "processing.stage.exporting_las",
    "Improving and filtering": "processing.stage.improving_and_filtering",
    "Reclassifying": "processing.stage.reclassifying",
    "Applying control points": "processing.stage.applying_control_points",
    "Complete": "processing.stage.complete",
    "Failed": "processing.stage.failed",
}
_EXIT_MESSAGES = (
    (re.compile(r"Pipeline ended with exit code (-?\d+)\. See reports/pipeline\.log\."), "processing.error.pipeline_exit"),
    (re.compile(r"LAS export ended with exit code (-?\d+)\. See reports/export\.log\."), "processing.error.export_exit"),
)
_KNOWN_TEXT_PREFIXES = ("workflow.stage.", "workflow.step.")


def display_progress_text(text: object) -> object:
    """Translate a persisted progress stage or error message for the UI; unknown free text is returned unchanged.

    Known texts are the stages/errors written above and the English labels of the ``workflow.stage``/
    ``workflow.step`` keys (estimator stage names written by the engine). The stored value is never changed.
    """
    if not isinstance(text, str) or not text:
        return text
    key = _PROGRESS_TEXT_KEYS.get(text)
    if key is not None:
        return t(key)
    for pattern, message_key in _EXIT_MESSAGES:
        match = pattern.fullmatch(text)
        if match is not None:
            return t(message_key, code=match.group(1))
    english = catalogue("en")
    for catalogue_key, value in english.items():
        if value == text and catalogue_key.startswith(_KNOWN_TEXT_PREFIXES):
            return t(catalogue_key)
    return text
