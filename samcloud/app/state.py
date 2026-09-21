"""Application state for the local single-user desktop workflow."""

from pathlib import Path

from samcloud.core.project_manager import ProjectManager
from samcloud.app.processing import ProcessingManager


class ApplicationState:
    """Keep the currently open project while the local app is running."""

    def __init__(self) -> None:
        self.projects = ProjectManager()
        self.active_project_directory: Path | None = None
        self.processing = ProcessingManager(self)

    def open_project(self, project_directory: Path) -> None:
        """Set the project used by all workflow pages."""
        self.projects.load(project_directory)
        self.active_project_directory = project_directory

    def active_project(self) -> dict | None:
        """Return the open project metadata, if a project has been selected."""
        if self.active_project_directory is None:
            return None
        return self.projects.load(self.active_project_directory)


state = ApplicationState()
