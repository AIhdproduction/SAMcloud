"""Parametric furniture made of boxes, cylinders and spheres (item frame: origin on the floor, +y = back)."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .spec import CLASS_APPLIANCES, CLASS_CEILING_EQUIPMENT, CLASS_FURNITURE, CLASS_HVAC, CLASS_SANITARY, FurnitureSpec, Part


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(round(rng.uniform(lo, hi), 3))


def _legs(w: float, d: float, h: float, s: float = 0.05, inset: float = 0.04) -> list[Part]:
    x, y = w / 2 - inset - s / 2, d / 2 - inset - s / 2
    return [Part("box", (sx * x, sy * y, h / 2), (s, s, h)) for sx in (-1, 1) for sy in (-1, 1)]


def table_legs(rng, w=None, d=None, h=None) -> tuple[list[Part], dict]:
    w = w or _u(rng, 1.2, 1.8)
    d = d or _u(rng, 0.75, 0.95)
    h = h or _u(rng, 0.72, 0.77)
    top = 0.04
    return [Part("box", (0, 0, h - top / 2), (w, d, top))] + _legs(w, d, h - top), {"w": w, "d": d, "h": h}


def table_pedestal(rng, w=None, d=None, h=None) -> tuple[list[Part], dict]:
    w = w or _u(rng, 1.0, 1.4)
    d = d or _u(rng, 0.7, 0.9)
    h = h or _u(rng, 0.72, 0.76)
    return [
        Part("box", (0, 0, h - 0.02), (w, d, 0.04)),
        Part("cyl", (0, 0, (h - 0.04) / 2), (0.06, 0.06, h - 0.04)),
        Part("box", (0, 0, 0.02), (min(w, d) * 0.6, min(w, d) * 0.6, 0.04)),
    ], {"w": w, "d": d, "h": h}


def table_trestle(rng, w=None, d=None, h=None) -> tuple[list[Part], dict]:
    w = w or _u(rng, 1.6, 2.2)
    d = d or _u(rng, 0.8, 1.0)
    h = h or _u(rng, 0.73, 0.77)
    x = w / 2 - 0.2
    return [
        Part("box", (0, 0, h - 0.025), (w, d, 0.05)),
        Part("box", (-x, 0, (h - 0.05) / 2), (0.07, d * 0.8, h - 0.05)),
        Part("box", (x, 0, (h - 0.05) / 2), (0.07, d * 0.8, h - 0.05)),
        Part("box", (0, 0, 0.35), (2 * x, 0.06, 0.08)),
    ], {"w": w, "d": d, "h": h}


def table_round(rng, r=None, h=None) -> tuple[list[Part], dict]:
    r = r or _u(rng, 0.45, 0.6)
    h = h or _u(rng, 0.72, 0.76)
    return [
        Part("cyl", (0, 0, h - 0.02), (r, r, 0.04)),
        Part("cyl", (0, 0, (h - 0.04) / 2), (0.06, 0.06, h - 0.04)),
        Part("cyl", (0, 0, 0.02), (r * 0.55, r * 0.55, 0.04)),
    ], {"w": 2 * r, "d": 2 * r, "h": h}


def coffee_table(rng) -> tuple[list[Part], dict]:
    return table_legs(rng, w=_u(rng, 0.9, 1.2), d=_u(rng, 0.5, 0.65), h=_u(rng, 0.38, 0.45))


def desk(rng) -> tuple[list[Part], dict]:
    return table_legs(rng, w=_u(rng, 1.2, 1.6), d=_u(rng, 0.65, 0.8), h=0.74)


def chair(rng, arms: bool = False) -> tuple[list[Part], dict]:
    w = _u(rng, 0.44, 0.5) + (0.12 if arms else 0.0)
    d = _u(rng, 0.46, 0.52)
    seat = _u(rng, 0.44, 0.47)
    back = _u(rng, 0.82, 0.92)
    parts = [Part("box", (0, 0, seat - 0.02), (w, d, 0.04))] + _legs(w, d, seat - 0.04, s=0.035, inset=0.02)
    parts.append(Part("box", (0, d / 2 - 0.015, (seat + back) / 2), (w, 0.03, back - seat)))
    if arms:
        for sx in (-1, 1):
            parts.append(Part("box", (sx * (w / 2 - 0.03), 0, seat + 0.2), (0.05, d * 0.85, 0.04)))
            parts.append(Part("box", (sx * (w / 2 - 0.03), -d / 2 + 0.06, seat + 0.1), (0.04, 0.04, 0.2)))
    return parts, {"w": w, "d": d, "h": back}


def sofa(rng, arms: bool = True, w=None) -> tuple[list[Part], dict]:
    w = w or _u(rng, 1.8, 2.4)
    d = _u(rng, 0.85, 0.98)
    seat = 0.42
    back_t = 0.2
    arm_w = 0.18 if arms else 0.0
    parts = [
        Part("box", (0, -back_t / 2, seat / 2), (w - 2 * arm_w, d - back_t, seat)),
        Part("box", (0, d / 2 - back_t / 2, 0.41), (w, back_t, 0.82)),
    ]
    if arms:
        for sx in (-1, 1):
            parts.append(Part("box", (sx * (w / 2 - arm_w / 2), -back_t / 2, 0.31), (arm_w, d - back_t, 0.62)))
    return parts, {"w": w, "d": d, "h": 0.82}


def sofa_l(rng) -> tuple[list[Part], dict]:
    w = _u(rng, 2.2, 2.7)
    l2 = _u(rng, 1.5, 1.8)
    d = 0.9
    parts = [
        Part("box", (0, 0, 0.21), (w, d, 0.42)),
        Part("box", (0, d / 2 - 0.1, 0.41), (w, 0.2, 0.82)),
        Part("box", (-w / 2 + d / 2, -d / 2 - (l2 - d) / 2, 0.21), (d, l2 - d, 0.42)),
        Part("box", (-w / 2 + 0.1, -d / 2 - (l2 - d) / 2 + 0.1, 0.41), (0.2, l2 - d + 0.2, 0.82)),
    ]
    return parts, {"w": w, "d": l2, "h": 0.82}


def armchair_round(rng) -> tuple[list[Part], dict]:
    r = _u(rng, 0.38, 0.45)
    parts = [Part("cyl", (0, 0, 0.21), (r, r, 0.42))]
    for k in range(7):  # curved backrest of short boxes on an arc behind the seat
        a = math.radians(30 + k * 20)
        parts.append(Part("box", (math.cos(a) * (r - 0.06), math.sin(a) * (r - 0.06), 0.6), (0.12, 0.08, 0.36), yaw_deg=math.degrees(a) + 90))
    return parts, {"w": 2 * r, "d": 2 * r, "h": 0.78}


def bed(rng, w=None) -> tuple[list[Part], dict]:
    w = w or float(rng.choice([0.9, 1.4, 1.6, 1.8]))
    length = _u(rng, 2.0, 2.1)
    head_h = _u(rng, 0.9, 1.1)
    return [
        Part("box", (0, -0.04, 0.25), (w, length - 0.08, 0.5)),
        Part("box", (0, length / 2 - 0.04, head_h / 2), (w + 0.04, 0.08, head_h)),
    ], {"w": w, "d": length, "h": head_h}


def wardrobe(rng, w=None) -> tuple[list[Part], dict]:
    w = w or _u(rng, 1.0, 2.0)
    h = _u(rng, 2.0, 2.2)
    return [Part("box", (0, 0, h / 2), (w, 0.6, h))], {"w": w, "d": 0.6, "h": h}


def shelf(rng, w=None) -> tuple[list[Part], dict]:
    w = w or _u(rng, 0.8, 1.2)
    d, h = 0.35, _u(rng, 1.6, 2.0)
    parts = [Part("box", (sx * (w / 2 - 0.01), 0, h / 2), (0.02, d, h)) for sx in (-1, 1)]
    parts.append(Part("box", (0, d / 2 - 0.005, h / 2), (w, 0.01, h)))
    for k in range(5):
        z = 0.02 + k * (h - 0.04) / 4
        parts.append(Part("box", (0, 0, z), (w - 0.04, d, 0.02)))
    return parts, {"w": w, "d": d, "h": h}


def sideboard(rng, w=None) -> tuple[list[Part], dict]:
    w = w or _u(rng, 1.4, 1.9)
    d, h = 0.45, _u(rng, 0.75, 0.85)
    return [Part("box", (0, 0, (h + 0.12) / 2), (w, d, h - 0.12))] + _legs(w, d, 0.12, s=0.04, inset=0.03), {"w": w, "d": d, "h": h}


def nightstand(rng) -> tuple[list[Part], dict]:
    w, d, h = _u(rng, 0.4, 0.5), 0.4, _u(rng, 0.45, 0.55)
    return [Part("box", (0, 0, h / 2), (w, d, h))], {"w": w, "d": d, "h": h}


def tv_stand(rng) -> tuple[list[Part], dict]:
    w, d, h = _u(rng, 1.4, 1.8), 0.4, _u(rng, 0.45, 0.55)
    return [Part("box", (0, 0, h / 2), (w, d, h))], {"w": w, "d": d, "h": h}


def plant(rng) -> tuple[list[Part], dict]:
    r = _u(rng, 0.14, 0.2)
    crown = _u(rng, 0.3, 0.42)
    pot_h = _u(rng, 0.3, 0.4)
    return [
        Part("cyl", (0, 0, pot_h / 2), (r, r, pot_h)),
        Part("cyl", (0, 0, pot_h + 0.15), (0.02, 0.02, 0.3)),
        Part("sphere", (0, 0, pot_h + 0.25 + crown), (crown, crown, crown)),
    ], {"w": 2 * crown, "d": 2 * crown, "h": pot_h + 0.25 + 2 * crown}


def toilet(rng) -> tuple[list[Part], dict]:
    return [
        Part("cyl", (0, -0.1, 0.2), (0.19, 0.19, 0.4)),
        Part("box", (0, 0.19, 0.6), (0.38, 0.18, 0.4)),
        Part("box", (0, 0.12, 0.2), (0.22, 0.2, 0.4)),
    ], {"w": 0.38, "d": 0.68, "h": 0.8}


def washbasin(rng) -> tuple[list[Part], dict]:
    w = _u(rng, 0.5, 0.65)
    return [Part("box", (0, 0, 0.8), (w, 0.45, 0.16)), Part("cyl", (0, 0.1, 0.36), (0.08, 0.08, 0.72))], {"w": w, "d": 0.45, "h": 0.88}


def bathtub(rng) -> tuple[list[Part], dict]:
    w = _u(rng, 1.6, 1.8)
    return [Part("box", (0, 0, 0.29), (w, 0.75, 0.58))], {"w": w, "d": 0.75, "h": 0.58}


def shower_tray(rng) -> tuple[list[Part], dict]:
    s = _u(rng, 0.8, 1.0)
    return [Part("box", (0, 0, 0.03), (s, s, 0.06))], {"w": s, "d": s, "h": 0.06}


def kitchen_counter(rng, w=None) -> tuple[list[Part], dict]:
    w = w or _u(rng, 1.8, 3.0)
    return [Part("box", (0, 0.01, 0.43), (w, 0.58, 0.86)), Part("box", (0, 0, 0.88), (w, 0.62, 0.04))], {"w": w, "d": 0.62, "h": 0.9}


def kitchen_sink(rng) -> tuple[list[Part], dict]:
    return [Part("box", (0, 0, 0.905), (0.6, 0.48, 0.01))], {"w": 0.6, "d": 0.48, "h": 0.91}


def stove(rng) -> tuple[list[Part], dict]:
    return [Part("box", (0, 0, 0.905), (0.58, 0.5, 0.01))], {"w": 0.58, "d": 0.5, "h": 0.91}


def refrigerator(rng) -> tuple[list[Part], dict]:
    h = _u(rng, 1.8, 2.1)
    return [Part("box", (0, 0, h / 2), (0.6, 0.62, h))], {"w": 0.6, "d": 0.62, "h": h}


def upper_cabinets(rng, w=None) -> tuple[list[Part], dict]:
    w = w or _u(rng, 1.2, 2.4)
    return [Part("box", (0, 0, 1.8), (w, 0.35, 0.7))], {"w": w, "d": 0.35, "h": 2.15}


def wall_shelf(rng) -> tuple[list[Part], dict]:
    w, z = _u(rng, 0.7, 1.1), _u(rng, 1.45, 1.8)
    return [Part("box", (0, 0, z + 0.02), (w, 0.26, 0.04))], {"w": w, "d": 0.26, "h": z + 0.04}


def tv_wall(rng) -> tuple[list[Part], dict]:
    w, z = _u(rng, 1.0, 1.6), _u(rng, 0.85, 1.05)
    return [Part("box", (0, 0, z + 0.28 * w), (w, 0.04, 0.56 * w))], {"w": w, "d": 0.04, "h": z + 0.56 * w}


def monitor(rng, top=0.74) -> tuple[list[Part], dict]:
    w = _u(rng, 0.52, 0.72)
    h = 0.57 * w
    bottom = top + 0.12
    return [
        Part("box", (0, -0.03, bottom + h / 2), (w, 0.03, h)),
        Part("box", (0, 0.02, top + 0.08), (0.06, 0.04, 0.15)),
        Part("box", (0, 0.02, top + 0.007), (0.2, 0.16, 0.014)),
    ], {"w": w, "d": 0.2, "h": bottom + h}


def spot(rng, ceiling=2.5) -> tuple[list[Part], dict]:
    r = _u(rng, 0.06, 0.08)
    return [Part("cyl", (0, 0, ceiling - 0.018), (r, r, 0.03))], {"w": 2 * r, "d": 2 * r, "h": ceiling}


def ceiling_lamp(rng, ceiling=2.5) -> tuple[list[Part], dict]:
    r, drop = _u(rng, 0.2, 0.32), _u(rng, 0.12, 0.18)
    return [
        Part("cyl", (0, 0, ceiling - drop + 0.03), (r, r, 0.06)),
        Part("cyl", (0, 0, ceiling - (drop - 0.06) / 2 - 0.002), (0.07, 0.07, drop - 0.06)),
    ], {"w": 2 * r, "d": 2 * r, "h": ceiling}


def smoke_detector(rng, ceiling=2.5) -> tuple[list[Part], dict]:
    return [Part("cyl", (0, 0, ceiling - 0.022), (0.065, 0.065, 0.04))], {"w": 0.13, "d": 0.13, "h": ceiling}


def vent(rng, ceiling=2.5) -> tuple[list[Part], dict]:
    w = _u(rng, 0.26, 0.36)
    return [Part("box", (0, 0, ceiling - 0.014), (w, 0.2, 0.024))], {"w": w, "d": 0.2, "h": ceiling}


def curtain_rail(rng, ceiling=2.5) -> tuple[list[Part], dict]:
    w = _u(rng, 1.2, 1.8)
    return [Part("box", (0, 0, ceiling - 0.02), (w, 0.05, 0.036))], {"w": w, "d": 0.05, "h": ceiling}


def wall_clock(rng) -> tuple[list[Part], dict]:
    return [Part("box", (0, 0, 1.9), (0.32, 0.03, 0.32))], {"w": 0.32, "d": 0.03, "h": 2.06}


CATALOGUE: dict[str, tuple[str, int, Callable[..., tuple[list[Part], dict]]]] = {
    # kind: (SAM3 instance label, class index, generator)
    "table_legs": ("dining table", CLASS_FURNITURE, table_legs),
    "table_pedestal": ("table", CLASS_FURNITURE, table_pedestal),
    "table_trestle": ("dining table", CLASS_FURNITURE, table_trestle),
    "table_round": ("table", CLASS_FURNITURE, table_round),
    "coffee_table": ("coffee table", CLASS_FURNITURE, coffee_table),
    "desk": ("desk", CLASS_FURNITURE, desk),
    "chair": ("chair", CLASS_FURNITURE, lambda rng, **k: chair(rng, arms=False)),
    "chair_arms": ("chair", CLASS_FURNITURE, lambda rng, **k: chair(rng, arms=True)),
    "sofa": ("sofa", CLASS_FURNITURE, lambda rng, **k: sofa(rng, arms=True, **k)),
    "sofa_noarms": ("sofa", CLASS_FURNITURE, lambda rng, **k: sofa(rng, arms=False, **k)),
    "sofa_l": ("sofa", CLASS_FURNITURE, sofa_l),
    "armchair_round": ("armchair", CLASS_FURNITURE, armchair_round),
    "bed": ("bed", CLASS_FURNITURE, bed),
    "wardrobe": ("wardrobe", CLASS_FURNITURE, wardrobe),
    "shelf": ("shelf", CLASS_FURNITURE, shelf),
    "sideboard": ("sideboard", CLASS_FURNITURE, sideboard),
    "nightstand": ("nightstand", CLASS_FURNITURE, nightstand),
    "tv_stand": ("tv stand", CLASS_FURNITURE, tv_stand),
    "plant": ("plant", CLASS_FURNITURE, plant),
    "toilet": ("toilet", CLASS_SANITARY, toilet),
    "washbasin": ("washbasin", CLASS_SANITARY, washbasin),
    "bathtub": ("bathtub", CLASS_SANITARY, bathtub),
    "shower_tray": ("shower tray", CLASS_SANITARY, shower_tray),
    "kitchen_counter": ("kitchen counter", CLASS_APPLIANCES, kitchen_counter),
    "kitchen_sink": ("kitchen sink", CLASS_APPLIANCES, kitchen_sink),
    "stove": ("stove", CLASS_APPLIANCES, stove),
    "refrigerator": ("refrigerator", CLASS_FURNITURE, refrigerator),
    "upper_cabinets": ("kitchen cabinet", CLASS_APPLIANCES, upper_cabinets),
    "wall_shelf": ("shelf", CLASS_FURNITURE, wall_shelf),
    "tv_wall": ("television", CLASS_FURNITURE, tv_wall),
    "monitor": ("monitor", CLASS_FURNITURE, monitor),
    "spot": ("spotlight", CLASS_CEILING_EQUIPMENT, spot),
    "ceiling_lamp": ("ceiling lamp", CLASS_CEILING_EQUIPMENT, ceiling_lamp),
    "smoke_detector": ("smoke detector", CLASS_CEILING_EQUIPMENT, smoke_detector),
    "vent": ("ventilation grille", CLASS_HVAC, vent),
    # negatives: classified as ceiling equipment, never a ceiling object
    "neg_curtain_rail": ("ceiling fixture", CLASS_CEILING_EQUIPMENT, curtain_rail),
    "neg_wall_clock": ("ceiling fixture", CLASS_CEILING_EQUIPMENT, wall_clock),
}

# Coarse groups used for "group accuracy" (a dining table reported as "table" is still a table).
LABEL_GROUPS = {
    "table": {"table", "dining table", "coffee table", "desk"},
    "seat": {"chair", "armchair", "stool"},
    "sofa": {"sofa", "couch"},
    "bed": {"bed", "mattress"},
    "storage": {"wardrobe", "shelf", "bookcase", "sideboard", "cabinet", "chest of drawers", "nightstand", "tv stand"},
    "kitchen": {"kitchen counter", "kitchen cabinet", "kitchen sink", "stove", "refrigerator", "dishwasher", "oven", "wall cabinet"},
    "sanitary": {"toilet", "washbasin", "bathtub", "shower tray", "shower", "shower enclosure"},
    "plant": {"plant"},
    "display": {"television", "monitor", "screen", "laptop"},
}


def label_group(label: str | None) -> str:
    text = (label or "").strip().lower()
    for group, members in LABEL_GROUPS.items():
        if text in members:
            return group
    return text or "unknown"


def make_item(kind: str, rng: np.random.Generator, fid: str, center: tuple[float, float], yaw_deg: float, instance_id: int, **kwargs) -> FurnitureSpec:
    label, class_idx, generator = CATALOGUE[kind]
    parts, _dims = generator(rng, **kwargs)
    return FurnitureSpec(fid, label, class_idx, (float(center[0]), float(center[1])), float(yaw_deg), parts, instance_id, kind)


def item_dims(kind: str, rng: np.random.Generator, **kwargs) -> dict:
    """Dimensions a generator would produce (consumes the same random numbers as :func:`make_item`)."""
    return CATALOGUE[kind][2](rng, **kwargs)[1]
