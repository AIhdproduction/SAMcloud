"""Model access of the outdoor assistant domain: the environment-chain Scene, its files and the support data.

The outdoor model is the :class:`samcloud.bim.core.model.Scene` the drone chain builds in ``_step_scene``
(``products/environment/cache/scene.pkl``: ``{"scene", "dxf", "qa"}``).  The assistant keeps one JSON container per
project in ``products/assistant/site/`` (:func:`model_path`):

* ``scene.json``: the current outdoor model WITH the accepted edits (``Scene.save``; the "model file" of the history);
* ``scene.base.json``: the automatic Scene the edits are applied on (written by the history / the pipeline hook);
* ``edits.json`` / ``state.json``: the generic history files (:mod:`samcloud.assistant.history`).

Without ``edits.json`` ``scene.json`` is only a copy of the pipeline Scene (refreshed whenever ``scene.pkl`` is newer);
with ``edits.json`` the pipeline hook (:mod:`samcloud.assistant.domains.outdoor_pipeline`) keeps it current.

Pipeline Scenes are not always valid when re-read (float noise such as a coverage of ``1.0000000000000002`` above its
``le=1`` bound): :func:`load_scene_json` clamps such bound violations, and every Scene the assistant handles went once
through that JSON round trip (:func:`normalize_scene`), so hashes of pickled and re-read Scenes agree.

The support data (TIN, pavement outline, drone config) is loaded lazily and cached per file modification time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

LOGGER = logging.getLogger(__name__)

SITE_LEVEL = "site"
MODEL_FILENAME = "scene.json"
BASE_FILENAME = "scene.base.json"
EDITS_FILENAME = "edits.json"
_BOUND_ERRORS = {"less_than_equal": "le", "less_than": "lt", "greater_than_equal": "ge", "greater_than": "gt"}
_LOCK = threading.RLock()
_SCENE_CACHE: dict[str, tuple[tuple[int, int], Any]] = {}
_PICKLE_CACHE: dict[str, tuple[tuple[int, int], Any]] = {}


# ------------------------------------------------------------------------------------------------------------ paths

def environment_dir(project: Path | str) -> Path:
    return Path(project) / "products" / "environment"


def cache_dir(project: Path | str) -> Path:
    return environment_dir(project) / "cache"


def scene_pickle_path(project: Path | str) -> Path:
    return cache_dir(project) / "scene.pkl"


def assistant_site_dir(project: Path | str, level_id: str = SITE_LEVEL) -> Path:
    from samcloud.assistant.paths import assistant_dir

    return assistant_dir(project, level_id)


def edits_file(project: Path | str, level_id: str = SITE_LEVEL) -> Path:
    return assistant_site_dir(project, level_id) / EDITS_FILENAME


def raw_model_path(project: Path | str, level_id: str = SITE_LEVEL) -> Path:
    """``scene.json`` of the level (no synchronisation)."""
    return assistant_site_dir(project, level_id) / MODEL_FILENAME


def base_path(project: Path | str, level_id: str = SITE_LEVEL) -> Path:
    return assistant_site_dir(project, level_id) / BASE_FILENAME


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return int(stat.st_mtime_ns), int(stat.st_size)


# --------------------------------------------------------------------------------------------------- scene loading

def _walk_set(data: Any, loc: tuple[Any, ...], value: float) -> bool:
    """Set ``value`` at a pydantic error location (union tags such as ``"tree"`` in the path are skipped)."""
    node = data
    keys = list(loc)
    for index, key in enumerate(keys):
        last = index == len(keys) - 1
        if isinstance(node, dict):
            if key not in node:
                if isinstance(node.get("kind"), str) and node.get("kind") == key:
                    if last:
                        return False
                    continue  # discriminator tag of the element union
                return False
            if last:
                node[key] = value
                return True
            node = node[key]
        elif isinstance(node, list) and isinstance(key, int) and 0 <= key < len(node):
            if last:
                node[key] = value
                return True
            node = node[key]
        else:
            return False
    return False


def _repair(data: Any, errors: list[dict[str, Any]]) -> bool:
    """Clamp numeric bound violations (float noise of the pipeline) in place; True when anything was repaired."""
    repaired = False
    for error in errors:
        bound = _BOUND_ERRORS.get(str(error.get("type")))
        value = error.get("input")
        if bound is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        limit = (error.get("ctx") or {}).get(bound)
        if not isinstance(limit, (int, float)):
            continue
        limit = float(limit)
        if bound == "lt":
            limit = math.nextafter(limit, -math.inf)
        elif bound == "gt":
            limit = math.nextafter(limit, math.inf)
        if abs(float(value) - limit) > 1e-6 * max(1.0, abs(limit)):
            continue  # a real violation, not float noise: let the validation fail
        repaired = _walk_set(data, tuple(error.get("loc") or ()), limit) or repaired
    return repaired


def load_scene_json(data: str | bytes) -> Any:
    """A Scene from its JSON (``Scene.save`` format); bound violations from float noise are clamped."""
    from pydantic import ValidationError

    from samcloud.bim.core.model import Scene

    try:
        return Scene.model_validate_json(data)
    except ValidationError as first:
        raw = json.loads(data)
        error: ValidationError = first
        for _ in range(4):
            if not _repair(raw, error.errors()):
                raise error
            try:
                return Scene.model_validate(raw)
            except ValidationError as again:
                error = again
        raise error


def load_scene_file(path: Path | str) -> Any:
    return load_scene_json(Path(path).read_bytes())


def save_scene(scene: Any, path: Path | str) -> None:
    """``Scene.save``: deterministic JSON, the exact bytes :func:`scene_file_hash` hashes."""
    scene.save(path)


def scene_file_hash(scene: Any) -> str:
    return hashlib.sha256(scene.to_json().encode("utf-8")).hexdigest()


def scene_semantic_hash(scene: Any) -> str:
    return scene_file_hash(scene)


def normalize_scene(scene: Any) -> Any:
    """``scene`` after one JSON round trip (the form every assistant Scene has)."""
    return load_scene_json(scene.to_json())


def _cached(cache: dict, path: Path, loader: Callable[[Path], Any]) -> Any:
    stamp = _stamp(path)
    if stamp is None:
        raise FileNotFoundError(str(path))
    key = str(path.resolve())
    with _LOCK:
        hit = cache.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
    value = loader(path)
    with _LOCK:
        cache[key] = (stamp, value)
        while len(cache) > 6:
            cache.pop(next(iter(cache)))
    return value


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_pickle_cached(path: Path | str, default: Any = None) -> Any:
    """A cache pickle (cached per modification time); ``default`` when it does not exist."""
    path = Path(path)
    if not path.is_file():
        return default
    return _cached(_PICKLE_CACHE, path, _load_pickle)


def load_pipeline_scene(project: Path | str) -> Any:
    """The normalised Scene of ``scene.pkl`` (cached per modification time; never mutate it)."""
    def loader(path: Path) -> Any:
        data = _load_pickle(path)
        scene = data.get("scene") if isinstance(data, dict) else data
        if scene is None:
            raise FileNotFoundError(f"{path.name} holds no scene")
        return normalize_scene(scene)

    return _cached(_SCENE_CACHE, scene_pickle_path(project), loader)


def load_model_cached(path: Path | str) -> Any:
    """A ``scene.json`` / ``scene.base.json`` (cached per modification time; never mutate it)."""
    return _cached(_SCENE_CACHE, Path(path), load_scene_file)


def ensure_model_file(project: Path | str, level_id: str = SITE_LEVEL) -> Path:
    """``scene.json``, refreshed from ``scene.pkl`` while the level has no ``edits.json``.

    With ``edits.json`` the pipeline hook keeps it current; when it is missing there, the base snapshot is copied (the
    history then re-applies the edits on it), never the edited ``scene.pkl`` (that would apply them twice).
    """
    path = raw_model_path(project, level_id)
    pickle_path = scene_pickle_path(project)
    try:
        if edits_file(project, level_id).is_file():
            if not path.is_file():
                base = base_path(project, level_id)
                if base.is_file():
                    save_scene(load_scene_file(base), path)
                elif pickle_path.is_file():
                    save_scene(load_pipeline_scene(project), path)
            return path
        if not pickle_path.is_file():
            return path
        model_stamp, pickle_stamp = _stamp(path), _stamp(pickle_path)
        if model_stamp is None or pickle_stamp is None or model_stamp[0] < pickle_stamp[0]:
            save_scene(load_pipeline_scene(project), path)
    except Exception as error:  # noqa: BLE001 - a broken model file is reported by the caller
        LOGGER.warning("assistant: outdoor scene.json not refreshed (%s)", type(error).__name__)
    return path


def model_path(project: Path | str, level_id: str = SITE_LEVEL) -> Path:
    """History model file (``scene.json``), synchronised by :func:`ensure_model_file`."""
    return ensure_model_file(project, level_id)


# ------------------------------------------------------------------------------------------------------ geometry

def origin_offset(scene: Any) -> np.ndarray:
    """LV95 / LN02 of the local origin (``settings.origin_offset``)."""
    return np.asarray(tuple(scene.settings.origin_offset or (0.0, 0.0, 0.0)), dtype=np.float64)


def to_local_xy(scene: Any, e: float, n: float) -> tuple[float, float]:
    offset = origin_offset(scene)
    return float(e) - float(offset[0]), float(n) - float(offset[1])


def to_lv95_xy(scene: Any, x: float, y: float) -> tuple[float, float]:
    offset = origin_offset(scene)
    return float(x) + float(offset[0]), float(y) + float(offset[1])


def to_local_z(scene: Any, h: float) -> float:
    return float(h) - float(origin_offset(scene)[2])


def to_ln02(scene: Any, z: float) -> float:
    return float(z) + float(origin_offset(scene)[2])


def mesh_sampler(vertices: Any, triangles: Any) -> Callable[[np.ndarray], np.ndarray] | None:
    """Height sampler ``xy (N,2) -> z (N,)`` of a triangle mesh (upper side wins, NaN outside); None when empty."""
    if vertices is None or triangles is None or not len(triangles):
        return None
    from samcloud.terrain.tin import tin_sampler

    return tin_sampler(SimpleNamespace(vertices=np.asarray(vertices, float), triangles=np.asarray(triangles, np.int64)))


# ----------------------------------------------------------------------------------------------- support data

@dataclass
class EnvSupport:
    """Pipeline data the edits need: the final TIN, the paved area / outline (curb snapping) and the drone config."""

    tin: Any = None
    paved: Any = None
    outline: Any = None
    config: dict[str, Any] = field(default_factory=dict)
    _tin_sampler: Any = None

    @classmethod
    def for_project(cls, project: Path | str | None) -> "EnvSupport":
        config: dict[str, Any] = {}
        try:
            from samcloud.drone.contracts import load_drone_config

            config = load_drone_config()
        except Exception as error:  # noqa: BLE001 - defaults are used without the config file
            LOGGER.info("assistant: drone config not loaded (%s)", type(error).__name__)
        if project is None:
            return cls(config=config)
        cache = cache_dir(project)
        tin = None
        paved = outline = None
        try:
            tin = load_pickle_cached(cache / "tin.pkl")
        except Exception as error:  # noqa: BLE001 - without a TIN heights fall back to the terrain element
            LOGGER.info("assistant: tin.pkl not loaded (%s)", type(error).__name__)
        try:
            pave = load_pickle_cached(cache / "pavement.pkl")
            if isinstance(pave, dict) and pave.get("polygons"):
                from shapely.ops import unary_union

                polygons = list(pave.get("polygons") or ())
                paved_parts = [p.polygon for p in polygons if getattr(p, "kind", "") in ("road", "sidewalk")]
                paved = unary_union(paved_parts) if paved_parts else None
                outline = unary_union([p.polygon.boundary for p in polygons])
        except Exception as error:  # noqa: BLE001 - curbs are then snapped without the outline
            LOGGER.info("assistant: pavement.pkl not loaded (%s)", type(error).__name__)
        return cls(tin=tin, paved=paved, outline=outline, config=config)

    def tin_sample(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if self.tin is None or not len(getattr(self.tin, "triangles", ())):
            return np.full(len(xy), np.nan)
        if self._tin_sampler is None:
            self._tin_sampler = mesh_sampler(self.tin.vertices, self.tin.triangles)
        return np.asarray(self._tin_sampler(xy), dtype=np.float64)


__all__ = ["BASE_FILENAME", "EnvSupport", "MODEL_FILENAME", "SITE_LEVEL", "base_path", "cache_dir", "edits_file",
           "ensure_model_file", "environment_dir", "load_model_cached", "load_pickle_cached", "load_pipeline_scene",
           "load_scene_file", "load_scene_json", "mesh_sampler", "model_path", "normalize_scene", "origin_offset",
           "raw_model_path", "save_scene", "scene_file_hash", "scene_pickle_path", "scene_semantic_hash", "to_ln02",
           "to_local_xy", "to_local_z", "to_lv95_xy"]
