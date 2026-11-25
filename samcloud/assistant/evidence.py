"""The "no evidence = nothing" rule of the AI assistant, enforced in code.

An object may only be added where the project point cloud shows it: enough points of a SAM3 class that is
plausible for the kind, with a real height spread above the floor.  Without that evidence the edit engine
(``samcloud.assistant.edits``) refuses the edit, unless the user explicitly asks for a manual item (it is then
flagged ``fit["evidence"] = "manual"``, drawn on ``A-FURN-MANUAL`` and marked in the IFC).

The furniture kind table (:data:`FURNITURE_KINDS`) lives here too: one row per kind gives the display label the
automatic pipeline uses (German, ``samcloud.ortho.plan.FINE_LABEL_GERMAN_NAMES``), the canonical English semantic
label (the fine term), the catalogue kind, the DXF layer, default dimensions for manual items and the plausible
SAM3 classes.  IFC class / type then follow ``config/bim_furniture.yaml`` exactly as for automatic items.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------------------------
# configuration (metres / point counts; nothing project specific)
# --------------------------------------------------------------------------------------------------------------------

# Minimum number of plausible-class points inside the box (the reconstruction needs about as many, see
# config/furniture_recon.yaml ``min_points``).
EVIDENCE_MIN_POINTS = 60
# The plausible points must spread at least this much in height (p95 - p5): a flat smear on the floor is no object.
EVIDENCE_MIN_HEIGHT_SPREAD_M = 0.12
# Points closer than this to the floor are floor / noise, never object evidence.
EVIDENCE_FLOOR_CLEARANCE_M = 0.03
# Share of plausible-class points among all object points in the box (a wall or a door alone is no chair).
EVIDENCE_MIN_PLAUSIBLE_SHARE = 0.25

# SAM3 indoor class indices (config/classes.json order, see samcloud.bim.synthbench.spec).
CLASS_LOOSE_ITEMS = 1
CLASS_HVAC = 15
CLASS_FURNITURE = 17
CLASS_APPLIANCES = 18
CLASS_SANITARY = 19

MANUAL_LAYER = "A-FURN-MANUAL"


@dataclass(frozen=True, slots=True)
class FurnitureKind:
    """One furniture kind the assistant can add or assign."""

    key: str
    label: str  # display label as the automatic pipeline writes it (German), used for IFC names "Stuhl 01"
    label_en: str
    semantic_label: str  # canonical English fine term
    catalogue_kind: str | None
    layer: str
    size_m: tuple[float, float, float]  # default width, depth, height of a manual item
    classes: tuple[int, ...]  # plausible SAM3 classes
    default_z_m: tuple[float, float] | None = None  # wall-mounted kinds: metric [low, high] above the floor
    aliases: tuple[str, ...] = field(default_factory=tuple)


_F = (CLASS_FURNITURE,)
_FA = (CLASS_FURNITURE, CLASS_APPLIANCES)
_S = (CLASS_SANITARY,)

FURNITURE_KINDS: dict[str, FurnitureKind] = {kind.key: kind for kind in (
    FurnitureKind("chair", "Stuhl", "Chair", "chair", None, "A-FURN", (0.45, 0.50, 0.90), _F, aliases=("stuhl",)),
    FurnitureKind("armchair", "Sessel", "Armchair", "armchair", None, "A-FURN", (0.80, 0.85, 0.85), _F, aliases=("sessel",)),
    FurnitureKind("stool", "Hocker", "Stool", "stool", None, "A-FURN", (0.40, 0.40, 0.65), _F, aliases=("hocker",)),
    FurnitureKind("sofa", "Sofa", "Sofa", "sofa", None, "A-FURN", (2.00, 0.90, 0.85), _F, aliases=("couch",)),
    FurnitureKind("table", "Tisch", "Table", "table", None, "A-FURN", (1.20, 0.80, 0.75), _F, aliases=("tisch",)),
    FurnitureKind("dining_table", "Esstisch", "Dining table", "dining table", None, "A-FURN", (1.60, 0.90, 0.75), _F, aliases=("esstisch", "dining table")),
    FurnitureKind("coffee_table", "Couchtisch", "Coffee table", "coffee table", None, "A-FURN", (1.00, 0.60, 0.45), _F, aliases=("couchtisch", "coffee table")),
    FurnitureKind("desk", "Schreibtisch", "Desk", "desk", None, "A-FURN", (1.40, 0.70, 0.75), _F, aliases=("schreibtisch",)),
    FurnitureKind("bed", "Bett", "Bed", "bed", None, "A-FURN", (1.60, 2.00, 0.50), _F, aliases=("bett",)),
    FurnitureKind("nightstand", "Nachttisch", "Nightstand", "nightstand", None, "A-FURN", (0.45, 0.40, 0.55), _F, aliases=("nachttisch",)),
    FurnitureKind("wardrobe", "Schrank", "Wardrobe", "wardrobe", None, "A-FURN", (1.20, 0.60, 2.00), _F, aliases=("kleiderschrank",)),
    FurnitureKind("cabinet", "Schrank", "Cabinet", "cabinet", None, "A-FURN", (0.80, 0.45, 1.00), _F, aliases=("schrank",)),
    FurnitureKind("chest_of_drawers", "Kommode", "Chest of drawers", "chest of drawers", None, "A-FURN", (1.00, 0.50, 0.90), _F, aliases=("kommode", "chest of drawers")),
    FurnitureKind("shelf", "Regal", "Shelf", "shelf", None, "A-FURN", (0.80, 0.35, 1.80), _F, aliases=("regal",)),
    FurnitureKind("bookcase", "Bücherregal", "Bookcase", "bookcase", None, "A-FURN", (0.80, 0.35, 2.00), _F, aliases=("bücherregal",)),
    FurnitureKind("sideboard", "Sideboard", "Sideboard", "sideboard", None, "A-FURN", (1.60, 0.45, 0.75), _F, aliases=("tv stand",)),
    FurnitureKind("refrigerator", "Kühlschrank", "Refrigerator", "refrigerator", None, "A-FURN", (0.60, 0.65, 1.80), _FA, aliases=("kühlschrank", "fridge")),
    FurnitureKind("washing_machine", "Waschmaschine", "Washing machine", "washing machine", None, "A-FURN", (0.60, 0.60, 0.85), _FA, aliases=("waschmaschine", "washing machine")),
    FurnitureKind("toilet", "WC", "Toilet", "toilet", None, "P-SANI", (0.40, 0.60, 0.40), _S, aliases=("wc",)),
    FurnitureKind("washbasin", "Lavabo", "Washbasin", "washbasin", None, "P-SANI", (0.60, 0.48, 0.20), _S, default_z_m=(0.65, 0.85), aliases=("lavabo", "waschbecken")),
    FurnitureKind("bathtub", "Badewanne", "Bathtub", "bathtub", None, "P-SANI", (1.70, 0.75, 0.60), _S, aliases=("badewanne",)),
    FurnitureKind("shower", "Dusche", "Shower", "shower tray", None, "P-SANI", (0.90, 0.90, 0.10), _S, aliases=("dusche", "shower tray")),
    FurnitureKind("plant", "Pflanze", "Plant", "plant", None, "A-FURN", (0.40, 0.40, 1.00), (CLASS_FURNITURE, CLASS_LOOSE_ITEMS), aliases=("pflanze",)),
    FurnitureKind("cat_tree", "Kratzbaum", "Cat tree", "cat tree", None, "A-FURN", (0.50, 0.50, 1.50), _F, aliases=("kratzbaum", "cat tree")),
    FurnitureKind("floor_lamp", "Stehlampe", "Floor lamp", "floor lamp", None, "A-FURN", (0.40, 0.40, 1.60), _F, aliases=("stehlampe", "floor lamp")),
    FurnitureKind("radiator", "Heizkörper", "Radiator", "radiator", None, "A-FURN", (1.00, 0.10, 0.60), (CLASS_HVAC, CLASS_FURNITURE, CLASS_APPLIANCES), default_z_m=(0.15, 0.75), aliases=("heizkörper",)),
)}


def _normal(text: str) -> str:
    return " ".join(str(text or "").strip().lower().replace("_", " ").replace("-", " ").split())


_KIND_LOOKUP: dict[str, str] = {}
for _kind in FURNITURE_KINDS.values():
    for _name in (_kind.key, _kind.label, _kind.label_en, _kind.semantic_label, *_kind.aliases):
        _KIND_LOOKUP.setdefault(_normal(_name), _kind.key)


def resolve_kind(text: str | None) -> FurnitureKind | None:
    """The kind for a key, display label (German or English), fine term or alias; None when unknown."""
    key = _KIND_LOOKUP.get(_normal(text or ""))
    return None if key is None else FURNITURE_KINDS[key]


def kind_of_item(label: str | None, semantic_label: str | None) -> FurnitureKind | None:
    """Best kind for an existing plan item (semantic label first, then the display label)."""
    return resolve_kind(semantic_label) or resolve_kind(label)


# --------------------------------------------------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------------------------------------------------

@dataclass(slots=True)
class Evidence:
    """Result of :func:`check_evidence`: ``ok`` only with enough plausible points and height spread."""

    ok: bool
    points: int  # plausible-class object points in the box
    classes: dict[str, int]  # all object points in the box per SAM3 class name
    reason: str
    height_spread_m: float = 0.0
    total_points: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "points": self.points, "classes": dict(self.classes), "reason": self.reason,
                "height_spread_m": round(self.height_spread_m, 3), "total_points": self.total_points}


def check_evidence(cloud: Any, box: dict[str, Any], kind: str) -> Evidence:
    """Is there a ``kind`` object in ``box`` (plan metres, see ``ProjectCloud``)?  Never raises for an empty box."""
    spec = resolve_kind(kind)
    if spec is None:
        return Evidence(False, 0, {}, f"unknown furniture kind {kind!r}")
    if cloud is None:
        return Evidence(False, 0, {}, "no project point cloud")
    selection = cloud.select(box, min_height_m=EVIDENCE_FLOOR_CLEARANCE_M)
    classes = cloud.class_counts(selection)
    total = int(len(selection))
    if not total:
        return Evidence(False, 0, classes, "no points in the box", 0.0, 0)
    plausible = selection[np.isin(cloud.cls[selection], np.asarray(spec.classes, dtype=cloud.cls.dtype))]
    count = int(len(plausible))
    spread = 0.0
    if count >= 2:
        heights = cloud.height_m(plausible)
        spread = float(np.percentile(heights, 95) - np.percentile(heights, 5))
    names = ", ".join(cloud.class_name(code) for code in spec.classes)
    if count < EVIDENCE_MIN_POINTS:
        return Evidence(False, count, classes, f"only {count} points of class {names} (need {EVIDENCE_MIN_POINTS})", spread, total)
    if spread < EVIDENCE_MIN_HEIGHT_SPREAD_M:
        return Evidence(False, count, classes, f"points spread only {spread:.2f} m in height (need {EVIDENCE_MIN_HEIGHT_SPREAD_M:.2f} m)", spread, total)
    if count < EVIDENCE_MIN_PLAUSIBLE_SHARE * total:
        return Evidence(False, count, classes, f"only {count} of {total} points are of class {names}", spread, total)
    return Evidence(True, count, classes, f"{count} points of class {names}, {spread:.2f} m high", spread, total)
