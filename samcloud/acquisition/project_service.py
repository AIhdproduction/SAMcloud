"""Project-aware acquisition preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from samcloud.acquisition.panorama import PreparedPanoramaInputs, prepare_panorama_inputs


def prepare_project_panorama(project_directory: Path, project: dict[str, Any]) -> PreparedPanoramaInputs:
    """Prepare a saved 360 degree project and record its derived input assets."""
    acquisition = project.get("acquisition", {})
    if acquisition.get("type") != "panorama_360":
        raise ValueError("Only 360 degree projects can prepare panorama inputs")
    if acquisition.get("locked"):
        raise ValueError("Capture settings are locked because processing has started")

    sources = acquisition.get("sources", {})
    panorama = acquisition.get("panorama", {})
    prepared = prepare_panorama_inputs(
        sources.get("panorama_image_directory", ""),
        sources.get("panorama_video_path", ""),
        project_directory / "inputs",
        float(panorama.get("frame_interval_seconds", 1.0)),
        int(panorama.get("cubemap_face_size", 1024)),
    )
    panorama["prepared"] = {
        "panorama_count": prepared.panorama_count,
        "video_frame_count": prepared.video_frame_count,
        "cubemap_image_count": prepared.cubemap_image_count,
        "manifest_path": str(prepared.manifest_path.relative_to(project_directory)),
    }
    acquisition["locked"] = True
    return prepared
