"""Where the AI assistant keeps and finds the files of one project level.

The assistant only ever acts on ONE project's result files:

* ``<project>/products/assistant/<level_id>/`` (:func:`assistant_dir`) holds its own state: ``edits.json``,
  ``plan_model.base.json`` (the automatic plan the edits are applied on) and ``state.json``.
* ``plan_model.json`` and the clean-plan ``result.json`` live in the ortho cache directory of the level
  (``samcloud.ortho.plan.clean_plan_cache_dir``); the path rules are reused, never duplicated here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

ASSISTANT_PRODUCT_DIRECTORY = ("products", "assistant")
CLASSIFIED_PLY = ("products", "classified", "classified.ply")


def assistant_dir(project: Path | str, level_id: str) -> Path:
    """``<project>/products/assistant/<level_id>/`` (not created here)."""
    return Path(project).joinpath(*ASSISTANT_PRODUCT_DIRECTORY, str(level_id))


def result_cache_dir(project: Path | str, level_id: str) -> Path:
    """The clean-plan (ortho) cache directory of a level: ``result.json`` and ``plan_model.json`` live there."""
    from samcloud.ortho.plan import clean_plan_cache_dir

    level = None if str(level_id) == "site" else SimpleNamespace(id=str(level_id))
    return clean_plan_cache_dir(project, level)  # type: ignore[arg-type]


def plan_model_path(project: Path | str, level_id: str) -> Path:
    """``plan_model.json`` of a level (the file the IFC export reads)."""
    from samcloud.bim.plan_model import plan_model_path as _plan_model_path

    return _plan_model_path(result_cache_dir(project, level_id))


def classified_ply_path(project: Path | str) -> Path:
    """The classified project point cloud (``scalar_class``, ``instance_id`` vertex fields)."""
    return Path(project).joinpath(*CLASSIFIED_PLY)
