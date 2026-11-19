"""3D opening refinement: measured windows and doors (width, sill/head, mullions, glazing) from photographs + points.

Entry points: :func:`refine_level_openings` (compute ``openings_3d.json``), :func:`refine_status` (freshness) and
:func:`apply_refined_openings` (hook of the clean plan).  Thresholds: ``config/bim_openings.yaml``.
"""

from .apply import apply_refined_openings, load_matching_openings
from .config import Settings, config_hash, load_openings_config
from .model import OPENINGS_3D_FILENAME, OpeningsResult, RefinedOpening, load_openings_3d, openings_3d_sha256, save_openings_3d
from .pipeline import RefineOutcome, level_cache_dir, plan_has_current_openings, refine_level_openings, refine_status

__all__ = [
    "OPENINGS_3D_FILENAME", "OpeningsResult", "RefineOutcome", "RefinedOpening", "Settings", "apply_refined_openings", "config_hash", "level_cache_dir",
    "load_matching_openings", "plan_has_current_openings", "load_openings_3d", "load_openings_config", "openings_3d_sha256", "refine_level_openings", "refine_status", "save_openings_3d",
]
