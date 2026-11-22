"""Enumerations of the scene model (docs/MASTERPROMPT.md sections 3.0, 5, 5.1 and 7)."""

from __future__ import annotations

from enum import StrEnum


class DataSource(StrEnum):
    """Kind of capture that produced the point cloud."""

    TLS_STATIONARY = "tls_stationary"
    SLAM = "slam"
    PHOTOGRAMMETRY = "photogrammetry"


class VisibilityMode(StrEnum):
    """How image visibility is derived."""

    STATION = "station"
    FREE_CAMERA = "free_camera"


class Scope(StrEnum):
    """Which parts of the site are modeled."""

    INDOOR_OUTDOOR = "indoor_outdoor"
    INDOOR_ONLY = "indoor_only"
    OUTDOOR_ONLY = "outdoor_only"


class IndoorDetail(StrEnum):
    """Indoor level of detail."""

    STRUCTURE_ONLY = "structure_only"
    STRUCTURE_AND_FURNITURE = "structure_and_furniture"


class ImageMode(StrEnum):
    """Whether images take part in the run."""

    WITH_IMAGES = "with_images"
    NO_IMAGES = "no_images"


class ParamOrigin(StrEnum):
    """Where a parameter value comes from."""

    MEASURED = "measured"
    FITTED = "fitted"
    SNAPPED = "snapped"
    TEMPLATE = "template"
    ASSUMED = "assumed"
    MANUAL = "manual"


class Provenance(StrEnum):
    """Who created or last changed an element."""

    SCRIPT = "script"
    AI = "ai"
    MANUAL = "manual"


class Status(StrEnum):
    """Quality status derived from metrics and tolerances."""

    OK = "ok"
    WARNING = "warning"
    FAIL = "fail"
    UNSURE = "unsure"


class ElementKind(StrEnum):
    """Element kinds represented by the scene model."""

    WALL = "wall"
    WALL_JOIN = "wall_join"
    SLAB = "slab"
    CEILING = "ceiling"
    OPENING = "opening"
    WINDOW = "window"
    DOOR = "door"
    SPACE = "space"
    STAIR = "stair"
    FURNITURE = "furniture"
    TERRAIN = "terrain"
    BUILDING = "building"
    TREE = "tree"
    SHRUB = "shrub"
    SITE_WALL = "site_wall"
    FENCE = "fence"
    PAVEMENT = "pavement"
    CURB = "curb"
    MARKING = "marking"
    MANHOLE = "manhole"
    VEGETATION_AREA = "vegetation_area"
    COLUMN = "column"
    BEAM = "beam"


class WallJustification(StrEnum):
    """Position of the wall axis relative to the wall body."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class WallFunction(StrEnum):
    """Interior or exterior wall."""

    INTERIOR = "interior"
    EXTERIOR = "exterior"
    UNKNOWN = "unknown"


class Tristate(StrEnum):
    """True, false or not known (never reported as measured)."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class JoinType(StrEnum):
    """Wall join type. X crossings are split into two T joins."""

    L = "L"
    T = "T"


class JoinEnd(StrEnum):
    """Which part of a wall takes part in a join (IFC ATSTART, ATEND, ATPATH)."""

    START = "start"
    END = "end"
    PATH = "path"


class Side(StrEnum):
    """Left or right, seen along the wall axis or in the door swing direction."""

    LEFT = "left"
    RIGHT = "right"


class DoorOperation(StrEnum):
    """Door operation type."""

    SINGLE_SWING_LEFT = "single_swing_left"
    SINGLE_SWING_RIGHT = "single_swing_right"
    DOUBLE_SWING = "double_swing"
    SLIDING = "sliding"
    OTHER = "other"


class WindowOperation(StrEnum):
    """Window operation type."""

    FIXED = "fixed"
    SIDE_HUNG_LEFT = "side_hung_left"
    SIDE_HUNG_RIGHT = "side_hung_right"
    TOP_HUNG = "top_hung"
    BOTTOM_HUNG = "bottom_hung"
    TILT_AND_TURN = "tilt_and_turn"
    SLIDING = "sliding"
    OTHER = "other"


class CeilingIfcClass(StrEnum):
    """IFC class used when exporting ceilings."""

    SLAB = "slab"
    COVERING = "covering"
