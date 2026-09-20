"""Central paths used by the application."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = PROJECT_ROOT / "config"
PROJECTS_DIRECTORY = PROJECT_ROOT / "projects"
MODELS_DIRECTORY = PROJECT_ROOT / "models"


def ensure_workspace_directories() -> None:
    """Create local runtime directories when they do not exist yet."""
    PROJECTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    MODELS_DIRECTORY.mkdir(parents=True, exist_ok=True)
