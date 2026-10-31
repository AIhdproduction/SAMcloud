"""Safe deletion of complete local SAMcloud projects."""

from __future__ import annotations

import shutil
from pathlib import Path

from samcloud.core.project_manager import PROJECT_FILENAME


def delete_project(projects_directory: Path, project_directory: Path) -> None:
    """Delete one project folder after validating its exact workspace scope."""
    root = projects_directory.resolve()
    target = project_directory.resolve()
    if target.parent != root:
        raise ValueError("Only direct project folders can be deleted")
    if target == root or not (target / PROJECT_FILENAME).is_file():
        raise ValueError("The selected project is not valid")
    shutil.rmtree(target)
