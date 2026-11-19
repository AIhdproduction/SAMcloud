"""Quality assurance of exports: the IDS gate and the validation chain."""

from __future__ import annotations

from samcloud.bim.core.qa.deviation import (
    DeviationResult,
    DeviationSettings,
    Face,
    apply_metrics,
    compute_deviation,
    element_faces,
    face_coverage,
    signed_distances,
    status_from_metrics,
    tolerance_colors,
)
from samcloud.bim.core.qa.evaluation import (
    EvaluationReport,
    evaluate_against_reference,
    evaluation_section,
    render_evaluation_html,
)
from samcloud.bim.core.qa.ids_spec import IDS_PATH, build_ids, write_ids
from samcloud.bim.core.qa.tolerances import (
    TOLERANCES_PATH,
    CeilingSagReference,
    ToleranceColors,
    ToleranceProfile,
    load_tolerances,
)
from samcloud.bim.core.qa.validate import ValidationReport, reimport_check, validate_ifc

__all__ = [
    "EvaluationReport",
    "evaluate_against_reference",
    "evaluation_section",
    "render_evaluation_html",
    "IDS_PATH",
    "TOLERANCES_PATH",
    "CeilingSagReference",
    "DeviationResult",
    "DeviationSettings",
    "Face",
    "ToleranceColors",
    "ToleranceProfile",
    "ValidationReport",
    "apply_metrics",
    "build_ids",
    "compute_deviation",
    "element_faces",
    "face_coverage",
    "load_tolerances",
    "reimport_check",
    "signed_distances",
    "status_from_metrics",
    "tolerance_colors",
    "validate_ifc",
    "write_ids",
]
