"""Which reconstruction model a plan furniture item gets (by catalogue kind, semantic label, label)."""

from __future__ import annotations

_FAMILY_NAMES: dict[str, tuple[str, ...]] = {
    "table": ("table", "coffee table", "dining table", "desk", "tisch", "couchtisch", "esstisch", "schreibtisch", "table_dining", "side table", "beistelltisch"),
    "chair": ("chair", "stool", "stuhl", "hocker", "office chair", "bürostuhl", "buerostuhl", "bench", "bank"),
    "sofa": ("sofa", "couch", "armchair", "sessel"),
    "bed": ("bed", "bett", "bed_double", "bed_single", "mattress", "matratze"),
    "kitchen_base": ("kitchen_base", "unterschrank"),
    "worktop": ("worktop", "arbeitsplatte"),
    "kitchen_insert": ("kitchen_sink", "cooktop", "spüle", "spuele", "herd", "hob"),
    "toilet": ("toilet", "wc", "wc_wall", "wc_floor"),
    "mesh": (
        "washbasin", "lavabo", "waschbecken", "shower", "shower enclosure",
        "floor lamp", "stehlampe", "lamp", "lampe", "plant", "pflanze", "cat tree", "kratzbaum",
    ),
    # kitchen counters / islands without a separate worktop item: one box up to the measured worktop (never the uppers above)
    "counter": ("fine_kitchen", "kitchen counter", "counter", "kitchen_island", "island", "segment", "küche", "kueche"),
    "box": (
        # a bathtub / shower tray is an accurate oriented box (its basin is not modelled); fixtures with a shape get a mesh
        "bathtub", "badewanne", "shower tray", "dusche",
        "cabinet", "schrank", "wardrobe", "kleiderschrank", "shelf", "regal", "bookcase", "bücherregal", "bookshelf",
        "chest of drawers", "kommode", "dresser", "sideboard", "tv stand", "nightstand", "nachttisch", "refrigerator", "fridge",
        "kühlschrank", "kuehlschrank", "washing machine", "waschmaschine", "dishwasher", "geschirrspüler", "dryer", "trockner",
        "oven", "backofen", "microwave", "microwave oven", "mikrowelle", "tall_unit", "tall unit", "hochschrank", "oven_tower",
        "oven tower", "builtin_front", "upper_cabinet", "wall cabinet", "kitchen cabinet", "television", "tv",
        "fernseher", "radiator", "heizkörper", "wardrobe_builtin", "niche",
    ),
}
_LOOKUP: dict[str, str] = {name: family for family, names in _FAMILY_NAMES.items() for name in names}

# families that stand on the floor (bottom snapped to the floor when evidence reaches close to it)
STANDING_FAMILIES = frozenset({"table", "chair", "sofa", "bed", "kitchen_base", "toilet", "counter"})
# reconstruction windows of the settings keyed by family
WINDOW_KEY = {
    "table": "table", "chair": "chair", "sofa": "sofa", "bed": "bed", "kitchen_base": "kitchen", "worktop": "kitchen",
    "kitchen_insert": "kitchen", "counter": "kitchen", "toilet": "toilet", "mesh": "mesh", "box": "box", "auto": "box",
}
# storage, lamps and plants may contain open bands (shelf boards, lamp pole); sanitary fixtures do not
TALL_GAP_FAMILIES = frozenset({"box", "auto"})
TALL_GAP_NAMES = frozenset({"floor lamp", "stehlampe", "lamp", "lampe", "plant", "pflanze", "cat tree", "kratzbaum"})
# kinds that may legitimately hang on a wall (not snapped to the floor)
MOUNTED_NAMES = frozenset({
    "upper_cabinet", "wall cabinet", "washbasin", "lavabo", "waschbecken", "wall shelf", "hood", "television", "tv", "wall tv",
    "mirror cabinet", "shelf", "regal", "wc_wall",
})


def _normal(text: object) -> str:
    value = str(text or "").strip().lower()
    return value.split(" h=")[0].strip()


def family_of(label: str | None, semantic_label: str | None, catalogue_kind: str | None) -> str:
    """Reconstruction family: table | chair | sofa | bed | kitchen_base | worktop | kitchen_insert | toilet | mesh | box | auto."""
    for text in (catalogue_kind, semantic_label, label):
        hit = _LOOKUP.get(_normal(text))
        if hit is not None:
            return hit
    return "auto"


def may_hang(label: str | None, semantic_label: str | None, catalogue_kind: str | None, layer: str | None) -> bool:
    """Whether the item may float above the floor (wall-mounted kinds and A-OBEN)."""
    if str(layer or "") == "A-OBEN":
        return True
    return any(_normal(text) in MOUNTED_NAMES for text in (catalogue_kind, semantic_label, label))


def allows_tall_gaps(family: str, label: str | None, semantic_label: str | None, catalogue_kind: str | None) -> bool:
    """Whether the object may have open height bands inside it (shelves, lamp poles, plants, cat trees)."""
    return family in TALL_GAP_FAMILIES or any(_normal(text) in TALL_GAP_NAMES for text in (catalogue_kind, semantic_label, label))


# low storage pieces whose top carries other objects (a TV, a lamp): those objects are not part of the piece
LOW_PIECE_NAMES = frozenset({
    "sideboard", "tv stand", "nightstand", "nachttisch", "chest of drawers", "kommode", "dresser", "lowboard", "tv-möbel",
})


def is_low_piece(label: str | None, semantic_label: str | None, catalogue_kind: str | None) -> bool:
    return any(_normal(text) in LOW_PIECE_NAMES for text in (catalogue_kind, semantic_label, label))
