"""Load and expose semantic classes configured for SAM3."""

import json
from pathlib import Path

from samcloud.core.paths import CONFIG_DIRECTORY


def load_class_definitions(class_set: str = "outdoor", path: Path | None = None) -> list[str]:
    """Return the ordered English prompt strings used by the SAM3 engine."""
    configuration_path = path or CONFIG_DIRECTORY / "classes.json"
    data = json.loads(configuration_path.read_text(encoding="utf-8"))
    if class_set not in {"outdoor", "indoor"}:
        raise ValueError(f"Unsupported class set: {class_set}")
    return list(data[class_set])


def _class_configuration(path: Path | None = None) -> dict:
    return json.loads((path or CONFIG_DIRECTORY / "classes.json").read_text(encoding="utf-8"))


def class_las_codes(class_set: str, config: dict | None = None) -> list[int]:
    """LAS code per class id (index 0 = unknown), from classes.json or the legacy Cyclone table."""
    config = config if config is not None else _class_configuration()
    codes = config.get("las_codes", {}).get(class_set)
    if codes is None:
        from samcloud.engine.export_las import LIVE_FALLBACK_CODE_SETS

        codes = [int(code) for code in LIVE_FALLBACK_CODE_SETS[class_set]]
    if len(codes) != len(config[class_set]) + 1:
        raise ValueError(f"classes.json: las_codes.{class_set} needs one code per class plus unknown")
    return [int(code) for code in codes]


def class_default_colors(class_set: str, config: dict | None = None) -> list[str] | None:
    """Default hex colour per class id (index 0 = unknown) when classes.json defines them."""
    config = config if config is not None else _class_configuration()
    colors = config.get("colors", {}).get(class_set)
    if colors is None or len(colors) != len(config[class_set]) + 1:
        return None
    return [str(color).upper() for color in colors]
