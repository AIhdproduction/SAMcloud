"""Workflow stage names shared by the UI and processing services."""

from enum import Enum


class WorkflowStage(str, Enum):
    """Persistent project stages shown in the user interface."""

    ALIGNMENT = "alignment"
    CONTROL_POINTS = "control_points"
    DENSE_CLOUD = "dense_cloud"
    CLASSIFICATION = "classification"
    EXPORT = "export"


STAGE_LABELS = {
    WorkflowStage.ALIGNMENT: "Image alignment",
    WorkflowStage.CONTROL_POINTS: "Control points",
    WorkflowStage.DENSE_CLOUD: "Dense point cloud",
    WorkflowStage.CLASSIFICATION: "Semantic classification",
    WorkflowStage.EXPORT: "Export",
}
