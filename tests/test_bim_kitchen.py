"""Kitchen in the building model and the IFC: base cabinets, worktop, sink, hob, measured upper / tall cabinets."""

from __future__ import annotations

import ifcopenshell
import ifcopenshell.geom
import numpy as np
import pytest
from shapely.geometry import Polygon

import test_bim_scene_builder as scene_tests
from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.plan_model import _furniture, load_bim_export_config
from samcloud.bim.scene_builder import build_scene

UPMS = (1.0, 0.8124)
FLOOR = 0.5


def _ring(x0: float, y0: float, x1: float, y1: float, upm: float) -> list[list[float]]:
    return [[x * upm, y * upm] for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]


def _layers(upm: float) -> dict:
    outline = _ring(0.5, 0.2, 3.5, 0.8, upm)
    meta = {"_einb_kind": "fine_kitchen", "_kitchen_run": "straight", "_run_legs": [outline], "_counter_front": True, "_counter_outline": outline}
    return {
        "A-EINB": [
            # the run is drawn as open front / end lines (the back is the wall face): two parts, one run
            {"type": "polyline", "points": [[0.5 * upm, 0.8 * upm], [3.5 * upm, 0.8 * upm]], "closed": False, **meta},
            {"type": "polyline", "points": [[3.5 * upm, 0.8 * upm], [3.5 * upm, 0.2 * upm]], "closed": False, **meta},
            {"type": "insert", "name": "KITCHEN_SINK_860X500", "point": [1.2 * upm, 0.2 * upm], "rotation": 0.0, "xscale": upm, "yscale": upm,
             "attributes": {"LABEL": "Spüle"}, "_catalogue": {"kind": "KITCHEN_SINK", "layer": "A-EINB", "wall_snapped": True, "on_kitchen_run": True}},
            {"type": "insert", "name": "COOKTOP_580X510", "point": [2.4 * upm, 0.2 * upm], "rotation": 0.0, "xscale": upm, "yscale": upm,
             "attributes": {"LABEL": "Herd"}, "_catalogue": {"kind": "COOKTOP", "layer": "A-EINB", "wall_snapped": True, "on_kitchen_run": True}},
            {"type": "polyline", "points": _ring(3.55, 0.2, 4.0, 0.8, upm), "closed": True, "holes": [], "_einb_kind": "tall_unit", "_tall_unit": "HS"},
        ],
        "A-OBEN": [
            {"type": "polyline", "points": _ring(0.5, 0.2, 2.0, 0.55, upm), "closed": True, "holes": [], "_upper_cabinet": True},
            {"type": "polyline", "points": _ring(2.2, 0.2, 3.4, 0.55, upm), "closed": True, "holes": [], "_upper_cabinet": True},
        ],
    }


def _instances(upm: float) -> list[dict]:
    def hull(x0: float, y0: float, x1: float, y1: float) -> Polygon:
        return Polygon(_ring(x0, y0, x1, y1, upm))

    def z(metres: float) -> float:
        return (FLOOR + metres) * upm

    return [
        {"id": 11, "label": "kitchen cabinet", "hull": hull(0.4, 0.1, 2.1, 0.7), "z_min": z(1.40), "z_max": z(2.20), "confidence": 0.8},
        # a low cabinet instance over the second upper cabinet: bottom at 0.3 m, no upper cabinet -> not matched
        {"id": 12, "label": "kitchen cabinet", "hull": hull(2.1, 0.1, 3.5, 0.7), "z_min": z(0.30), "z_max": z(0.95), "confidence": 0.8},
        {"id": 13, "label": "refrigerator", "hull": hull(3.5, 0.1, 4.1, 0.9), "z_min": z(0.0), "z_max": z(1.85), "confidence": 0.9},
    ]


@pytest.mark.parametrize("upm", UPMS)
def test_counter_run_becomes_base_cabinets_and_worktop_once(upm: float) -> None:
    furniture = _furniture(_layers(upm), {}, load_bim_export_config(), _instances(upm), FLOOR * upm, upm)
    names = [item.label for item in furniture]
    assert names.count("Unterschrank") == 1 and names.count("Arbeitsplatte") == 1  # two open lines, one run
    base = next(item for item in furniture if item.label == "Unterschrank")
    top = next(item for item in furniture if item.label == "Arbeitsplatte")
    assert base.default_height_m == pytest.approx(0.86) and top.default_z_m == pytest.approx([0.86, 0.90])
    assert Polygon(base.footprint).area == pytest.approx(3.0 * 0.6 * upm * upm)


@pytest.mark.parametrize("upm", UPMS)
def test_sink_and_hob_sit_in_the_worktop(upm: float) -> None:
    furniture = _furniture(_layers(upm), {}, load_bim_export_config(), _instances(upm), FLOOR * upm, upm)
    sink = next(item for item in furniture if item.catalogue_kind == "KITCHEN_SINK")
    hob = next(item for item in furniture if item.catalogue_kind == "COOKTOP")
    assert sink.default_z_m == pytest.approx([0.80, 0.90]) and hob.default_z_m == pytest.approx([0.90, 0.93])
    # a sink that is not on a run keeps the plain default height
    loose = _layers(upm)
    next(e for e in loose["A-EINB"] if e.get("name") == "KITCHEN_SINK_860X500")["_catalogue"]["on_kitchen_run"] = False
    again = next(item for item in _furniture(loose, {}, load_bim_export_config(), [], 0.0, upm) if item.catalogue_kind == "KITCHEN_SINK")
    assert again.default_z_m is None and again.default_height_m == pytest.approx(0.9)


@pytest.mark.parametrize("upm", UPMS)
def test_upper_and_tall_cabinets_take_their_measured_height(upm: float) -> None:
    furniture = _furniture(_layers(upm), {}, load_bim_export_config(), _instances(upm), FLOOR * upm, upm)
    uppers = sorted((item for item in furniture if item.layer == "A-OBEN"), key=lambda item: item.footprint[0][0])
    first, second = uppers
    assert first.instance_id == 11 and first.z_source == "instance_bbox"
    assert (first.z_min, first.z_max) == pytest.approx(((FLOOR + 1.40) * upm, (FLOOR + 2.20) * upm))
    assert second.instance_id is None and second.default_z_m == [1.45, 2.15]  # a low instance is no upper cabinet
    tall = next(item for item in furniture if item.catalogue_kind == "tall_unit")
    assert tall.instance_id == 13 and tall.z_max == pytest.approx((FLOOR + 1.85) * upm) and tall.z_min <= FLOOR * upm + 1e-9


def _bbox(entity) -> np.ndarray:
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    verts = np.array(ifcopenshell.geom.create_shape(settings, entity).geometry.verts).reshape(-1, 3)
    return np.array([verts.min(axis=0), verts.max(axis=0)])


@pytest.mark.parametrize("upm", UPMS)
def test_ifc_carries_the_kitchen_with_types_and_heights(upm: float, tmp_path) -> None:
    plan = scene_tests.make_plan(upm)
    plan.furniture = _furniture(_layers(upm), {}, load_bim_export_config(), _instances(upm), FLOOR * upm, upm)
    path = tmp_path / "kitchen.ifc"
    write_ifc(build_scene(plan), path)
    file = ifcopenshell.open(str(path))
    by_name = {e.Name: e for e in file.by_type("IfcProduct") if e.Name}
    base = by_name["Unterschrank 01"]
    assert (base.is_a(), base.PredefinedType, base.ObjectType) == ("IfcFurniture", "USERDEFINED", "KitchenUnit")
    worktop = by_name["Arbeitsplatte 01"]
    assert (worktop.is_a(), worktop.PredefinedType, worktop.ObjectType) == ("IfcFurniture", "USERDEFINED", "Worktop")
    sink = by_name["Spüle 01"]
    assert (sink.is_a(), sink.PredefinedType) == ("IfcSanitaryTerminal", "SINK")
    hob = by_name["Herd 01"]
    assert (hob.is_a(), hob.PredefinedType) == ("IfcElectricAppliance", "ELECTRICCOOKER")
    tall = by_name["Hochschrank 01"]  # readable names, the kind stays in the object type
    assert (tall.is_a(), tall.ObjectType) == ("IfcFurniture", "TallUnit")
    assert by_name["Oberschrank 01"].ObjectType == "WallCabinet"
    assert not any(name.split(" ")[0] in ("tall_unit", "upper_cabinet", "builtin_front") for name in by_name)
    names = ("Unterschrank 01", "Arbeitsplatte 01", "Spüle 01", "Herd 01", "Hochschrank 01", "Oberschrank 01", "Oberschrank 02")
    heights = {name: _bbox(by_name[name])[:, 2] for name in names}
    assert heights["Unterschrank 01"] == pytest.approx((FLOOR, FLOOR + 0.86), abs=1e-6)
    assert heights["Arbeitsplatte 01"] == pytest.approx((FLOOR + 0.86, FLOOR + 0.90), abs=1e-6)  # base + worktop = 0.90 m
    assert heights["Spüle 01"] == pytest.approx((FLOOR + 0.80, FLOOR + 0.90), abs=1e-6)
    assert heights["Herd 01"] == pytest.approx((FLOOR + 0.90, FLOOR + 0.93), abs=1e-6)
    assert heights["Hochschrank 01"] == pytest.approx((FLOOR, FLOOR + 1.85), abs=1e-6)
    assert heights["Oberschrank 01"] == pytest.approx((FLOOR + 1.40, FLOOR + 2.20), abs=1e-6)  # measured from the instance
    assert heights["Oberschrank 02"] == pytest.approx((FLOOR + 1.45, FLOOR + 2.15), abs=1e-6)  # default
