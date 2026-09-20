"""Load and expose semantic classes configured for SAM3."""

import json
from pathlib import Path

from samcloud.core.paths import CONFIG_DIRECTORY


def load_class_definitions(path: Path | None = None) -> list[str]:
    """Return the ordered English prompt strings used by the SAM3 engine."""
    configuration_path = path or CONFIG_DIRECTORY / "classes.json"
    data = json.loads(configuration_path.read_text(encoding="utf-8"))
    return list(data["aussen"])
