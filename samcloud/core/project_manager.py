"""Persistent SAMcloud project folders and project metadata."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from samcloud.core.configuration import load_global_config, load_yaml
from samcloud.core.paths import PROJECTS_DIRECTORY, ensure_workspace_directories


PROJECT_FILENAME = "project.samcloud.yml"
PROJECT_SUBDIRECTORIES = (
    "inputs",
    "control_points",
    "colmap",
    "products/sparse",
    "products/dense",
    "products/classified",
    "products/viewer",
    "reports",
)


class ProjectManager:
    """Create, find, load, and save reusable local projects."""

    def __init__(self, projects_directory: Path | None = None) -> None:
        self.projects_directory = projects_directory or PROJECTS_DIRECTORY

    def create_project(self, name: str, image_directory: str = "") -> Path:
        """Create a self-contained project folder without copying source images."""
        normalized_name = self._normalize_name(name)
        project_directory = self.projects_directory / normalized_name
        if project_directory.exists():
            raise FileExistsError(f"A project named '{normalized_name}' already exists")

        ensure_workspace_directories()
        project_directory.mkdir(parents=True)
        for relative_directory in PROJECT_SUBDIRECTORIES:
            (project_directory / relative_directory).mkdir(parents=True, exist_ok=True)

        project = self._new_project_data(normalized_name, image_directory)
        self.save(project_directory, project)
        return project_directory

    def discover_projects(self) -> list[Path]:
        """Return all valid project folders ordered by name."""
        if not self.projects_directory.exists():
            return []
        return sorted(
            path.parent for path in self.projects_directory.glob(f"*/{PROJECT_FILENAME}")
        )

    def load(self, project_directory: Path) -> dict[str, Any]:
        """Load an existing project metadata file."""
        project_file = project_directory / PROJECT_FILENAME
        if not project_file.exists():
            raise FileNotFoundError(f"No SAMcloud project file found in {project_directory}")
        return load_yaml(project_file)

    def save(self, project_directory: Path, project: dict[str, Any]) -> None:
        """Persist project metadata and update its modification timestamp."""
        project["project"]["updated_at"] = datetime.now(timezone.utc).isoformat()
        project_file = project_directory / PROJECT_FILENAME
        with project_file.open("w", encoding="utf-8") as file:
            yaml.safe_dump(project, file, sort_keys=False, allow_unicode=True)

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized_name = "-".join(name.strip().split()).lower()
        if not normalized_name:
            raise ValueError("Project name must not be empty")
        if any(character in normalized_name for character in '<>:"/\\|?*'):
            raise ValueError("Project name contains unsupported filename characters")
        return normalized_name

    @staticmethod
    def _new_project_data(name: str, image_directory: str) -> dict[str, Any]:
        defaults = deepcopy(load_global_config("defaults.yml"))
        now = datetime.now(timezone.utc).isoformat()
        return {
            "project": {
                "schema_version": 1,
                "name": name,
                "created_at": now,
                "updated_at": now,
            },
            "paths": {
                "image_directory": image_directory,
                "image_storage": "external_reference",
            },
            "settings": defaults,
            "workflow": {
                "alignment": "not_started",
                "control_points": "not_started",
                "dense_cloud": "not_started",
                "classification": "not_started",
                "export": "not_started",
            },
        }
