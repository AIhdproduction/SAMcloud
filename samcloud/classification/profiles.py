"""Cyclone class-code model profiles and LAS provenance helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Literal, Mapping

import numpy as np

from samcloud.core.configuration import load_dxf_export_config, load_global_config, load_yaml
from samcloud.core.paths import CONFIG_DIRECTORY, PROJECT_ROOT

NEVER_CLASSIFIED_ID = np.uint8(255)


class UnsupportedProfileModeError(ValueError):
    """Raised when a model profile is used with an unsupported project mode."""


@dataclass(frozen=True, slots=True)
class ModelProfileEntry:
    model_id: int
    model_name: str
    cyclone_code: int
    cyclone_name: str


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    supported_modes: tuple[Literal["indoor", "outdoor"], ...]
    entries: tuple[ModelProfileEntry, ...]
    point_states: Mapping[str, int]

    def codes_for_scalar_classes(self) -> np.ndarray:
        """Return LAS codes indexed by the one-based scalar PLY class id."""
        return np.asarray([self.point_states["unclassified"], *(entry.cyclone_code for entry in self.entries)], dtype=np.uint8)

    def code_to_name(self) -> dict[str, str]:
        result = {str(self.point_states["never_classified"]): "Created, never classified", str(self.point_states["unclassified"]): "Unclassified"}
        result.update({str(entry.cyclone_code): entry.model_name for entry in self.entries})
        return result


@dataclass(frozen=True, slots=True)
class ProfileInfo:
    profile_name: str
    mode: Literal["indoor", "outdoor"]
    class_set: str
    code_to_name: Mapping[str, str]
    created_at: str | None = None
    samcloud_version: str | None = None


def load_cyclone_codes() -> dict[int, dict[str, object]]:
    """Load the complete Cyclone code registry."""
    data = load_global_config("cyclone_codes.yaml")
    codes = data.get("codes", {})
    result = {int(code): dict(value) for code, value in codes.items()}
    if set(result) != set(range(256)):
        raise ValueError("cyclone_codes.yaml must list every code from 0 through 255")
    return result


def validate_profile(profile: ModelProfile) -> None:
    """Ensure semantic profile entries are unique, known and not reserved."""
    registry = load_cyclone_codes()
    codes = [entry.cyclone_code for entry in profile.entries]
    if len(codes) != len(set(codes)):
        raise ValueError(f"Profile '{profile.name}' contains duplicate Cyclone codes")
    for code in codes:
        if code not in registry:
            raise ValueError(f"Profile '{profile.name}' uses unknown Cyclone code {code}")
        if registry[code].get("reserved"):
            raise ValueError(f"Profile '{profile.name}' uses reserved Cyclone code {code}")


def sam3_profile(mode: Literal["indoor", "outdoor"], classes_config: dict) -> ModelProfile:
    """Build a SAM3 profile from the effective global or project classes.json."""
    codes = classes_config["las_codes"][mode]
    names = [entry.split(",", 1)[0].strip() for entry in classes_config[mode]]
    if len(codes) != len(names) + 1:
        raise ValueError(f"classes.json: las_codes.{mode} needs one code per class plus unknown")
    registry = load_cyclone_codes()
    entries = tuple(
        ModelProfileEntry(index, name, int(codes[index]), str(registry[int(codes[index])]["name"]))
        for index, name in enumerate(names, start=1)
    )
    profile = ModelProfile(f"sam3_{mode}", (mode,), entries, {"never_classified": 0, "unclassified": 1})
    validate_profile(profile)
    return profile


def _profile_from_yaml(path: Path) -> ModelProfile:
    data = load_yaml(path)
    entries = tuple(ModelProfileEntry(**entry) for entry in data["entries"])
    profile = ModelProfile(data["name"], tuple(data["supported_modes"]), entries, data["point_states"])
    validate_profile(profile)
    return profile


def flurseg_profile(mode: Literal["indoor", "outdoor"], classes_path: Path | None = None) -> ModelProfile:
    """Load FlurSeg's profile from its class file, falling back to the bundled profile."""
    if mode != "outdoor":
        raise UnsupportedProfileModeError("The FlurSeg v1 profile supports outdoor projects only")
    if classes_path is None:
        configured = load_global_config("flurseg.yml")["model"]["directory"]
        candidate = PROJECT_ROOT / configured
        classes_path = candidate.parent / "classes.yaml"
    if classes_path.is_file():
        data = load_yaml(classes_path)
        registry = load_cyclone_codes()
        entries = tuple(
            ModelProfileEntry(int(item["id"]), str(item["name"]), int(item["cyclone_code"]), str(item.get("cyclone_name") or registry[int(item["cyclone_code"])]["name"]))
            for item in data["classes"] if isinstance(item.get("cyclone_code"), int)
        )
        profile = ModelProfile("flurseg_v1", ("outdoor",), entries, data.get("point_states", {"never_classified": 0, "unclassified": 1}))
        validate_profile(profile)
        return profile
    return _profile_from_yaml(CONFIG_DIRECTORY / "model_profiles" / "flurseg_v1.yaml")


def profile_for_engine(engine: str, mode: Literal["indoor", "outdoor"], classes_config: dict, flurseg_classes_path: Path | None = None) -> ModelProfile:
    if engine == "flurseg":
        return flurseg_profile(mode, flurseg_classes_path)
    return sam3_profile(mode, classes_config)


def legacy_translation(mode: Literal["indoor", "outdoor"], target: ModelProfile, vlr_mapping: Mapping[str, str] | None = None) -> dict[int, int]:
    """Return old LAS-code to current Cyclone-code translation by semantic name."""
    old_codes = load_yaml(CONFIG_DIRECTORY / "model_profiles" / "legacy_v1.yaml")["old_codes"][mode]
    names = {entry.model_name: entry.cyclone_code for entry in target.entries}
    if vlr_mapping is not None:
        return {int(code): (1 if name.lower().startswith("unknown") else names.get(name, 1)) for code, name in vlr_mapping.items()}
    source_names = {
        "outdoor": ["Unknown/noise", "ground", "road", "sidewalk", "curb", "low vegetation", "tree", "building", "vehicle", "construction machinery", "utility pole", "traffic sign", "fence", "outdoor wall", "power line", "construction material", "water"],
        "indoor": ["Unknown/noise", "construction material", "wall", "ceiling", "floor", "door", "window", "stairs", "railing", "column", "beam"],
    }[mode]
    return {int(code): (1 if index == 0 else names.get(source_names[index], 1)) for index, code in enumerate(old_codes)}


def translate_legacy_codes(values: np.ndarray, mode: Literal["indoor", "outdoor"], target: ModelProfile, vlr_mapping: Mapping[str, str] | None = None) -> np.ndarray:
    mapping = legacy_translation(mode, target, vlr_mapping)
    return np.asarray([mapping.get(int(value), 1) for value in values], dtype=np.uint8)


def sidecar_path(las_path: Path) -> Path:
    return Path(las_path).with_suffix(Path(las_path).suffix + ".profile.json")


def write_profile_sidecar(las_path: Path, profile: ModelProfile, mode: Literal["indoor", "outdoor"], class_set: str) -> None:
    """Write provenance atomically beside a LAS file."""
    payload = {"profile_name": profile.name, "mode": mode, "class_set": class_set, "code_to_name": profile.code_to_name(), "created_at": datetime.now(UTC).isoformat()}
    try:
        version = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        version = ""
    if version:
        payload["samcloud_version"] = version
    destination = sidecar_path(las_path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)


def resolve_las_profile(las_path: Path) -> ProfileInfo | None:
    """Read LAS profile provenance from sidecar first, then the colmap_sam VLR."""
    payload: dict | None = None
    companion = sidecar_path(las_path)
    if companion.is_file():
        payload = json.loads(companion.read_text(encoding="utf-8"))
    else:
        import laspy
        las = laspy.read(las_path)
        for vlr in las.header.vlrs:
            if vlr.user_id == "colmap_sam" and vlr.record_id == 1:
                candidate = json.loads(bytes(vlr.record_data).decode("utf-8"))
                if "model_profile" in candidate:
                    payload = {"profile_name": candidate["model_profile"], "mode": candidate["mode"], "class_set": candidate.get("semantic_class_set", candidate["mode"]), "code_to_name": candidate.get("class_mapping", {})}
                break
    if payload is None:
        return None
    return ProfileInfo(str(payload["profile_name"]), payload["mode"], str(payload.get("class_set", payload["mode"])), payload.get("code_to_name", payload.get("class_mapping", {})), payload.get("created_at"), payload.get("samcloud_version"))


def detect_mode(classification: np.ndarray, config: dict | None = None) -> Literal["indoor", "outdoor"]:
    """Detect the project mode from Cyclone indoor structural class-code share."""
    settings = config if config is not None else load_dxf_export_config()
    threshold = float(settings["mode_detection"]["indoor_share"])
    values = np.asarray(classification)
    share = 0.0 if values.size == 0 else float(np.isin(values, [100, 101, 102]).sum()) / values.size
    return "indoor" if share > threshold else "outdoor"
