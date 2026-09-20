"""Create the class scheme consumed by the Potree viewer integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_viewer_manifest(path: Path, classes: list[dict[str, Any]], point_cloud_url: str) -> None:
    """Write a viewer manifest containing labels, colors, and class visibility."""
    manifest = {
        "point_cloud_url": point_cloud_url,
        "display_mode": "rgb",
        "classes": classes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
