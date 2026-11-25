"""Synthetic drone / environment-chain project for the outdoor AI-assistant tests.

A 40 x 30 m site (local metres) on a gently sloping terrain with a road strip along the south edge (y 0-6), a curb on the
road edge, a manhole in the road, two modelled trees, a fence along the east edge, an L-shaped garden wall and a
gable-roof building.  The checkpoints of the chain (``products/environment/cache/*.pkl``) are written like the real
steps do, then the real ``_step_scene`` builds ``scene.pkl``.  The classified working cloud (``load.pkl``) shows the
ground, the road, both trees, the manhole and ALSO a third tree and a second manhole that the model does not contain (for
measured adds); around :data:`EMPTY_POINT` there are only ground points (no evidence).

The local frame sits at a fake origin (the LV95 projection origin, no real site).  Nothing here comes from a real project.
"""

from __future__ import annotations

import json
import math
import pickle
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon

FAKE_ORIGIN = (2_600_000.0, 1_200_000.0, 500.0)
SITE = (0.0, 0.0, 40.0, 30.0)
ROAD = Polygon([(0.0, 0.0), (40.0, 0.0), (40.0, 6.0), (0.0, 6.0)])
TREE_1 = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, "synthetic-tree-1")), "xy": (12.13, 18.27), "height": 8.0, "crown": 5.0, "trunk": 0.30}
TREE_2 = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, "synthetic-tree-2")), "xy": (28.41, 22.36), "height": 6.0, "crown": 4.0, "trunk": 0.25}
EXTRA_TREE = {"xy": (16.37, 9.41), "height": 7.0, "crown": 4.4, "trunk": 0.32}  # in the cloud only
MANHOLE = {"id": "manhole-0", "xy": (20.17, 3.08), "diameter": 0.62}
EXTRA_MANHOLE = {"xy": (8.23, 2.91), "diameter": 0.6}  # in the cloud only
EMPTY_POINT = (34.07, 14.93)  # only ground points around it
FENCE_AXIS = [(37.93, 9.07), (38.11, 18.42), (37.87, 27.61)]
FENCE_HEIGHT = 1.2
WALL_AXIS = [(4.07, 10.13), (4.11, 20.07), (10.03, 26.11)]
WALL_THICKNESS, WALL_HEIGHT = 0.3, 1.0
CURB_AXIS = [(1.03, 6.07), (19.91, 6.04), (38.97, 6.09)]
BUILDING = (22.1, 10.2, 30.3, 16.1)  # x0, y0, x1, y1
BUILDING_EAVE, BUILDING_RIDGE = 5.0, 8.0
TERRAIN_STEP = 0.5


def terrain_z(x: Any, y: Any) -> Any:
    """Terrain height (local metres): a gentle slope with a slight wave (road: the same surface)."""
    return 0.02 * np.asarray(x, float) + 0.01 * np.asarray(y, float) + 0.03 * np.sin(np.asarray(x, float) * 0.37)


def lv95(x: float, y: float) -> tuple[float, float]:
    return x + FAKE_ORIGIN[0], y + FAKE_ORIGIN[1]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ------------------------------------------------------------------------------------------------------------- cloud

def _tree_points(rng: np.random.Generator, xy: tuple[float, float], height: float, crown: float, trunk: float) -> tuple[np.ndarray, np.ndarray]:
    base = float(terrain_z(*xy))
    crown_base = base + height * 0.4
    crown_height = height - height * 0.4
    n = 1400
    u = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * math.pi, n)
    r = np.sqrt(1.0 - u ** 2) * rng.uniform(0.8, 1.0, n)
    canopy = np.column_stack([xy[0] + r * np.cos(phi) * crown / 2.0, xy[1] + r * np.sin(phi) * crown / 2.0,
                              crown_base + (u + 1.0) / 2.0 * crown_height])
    m = 260
    angle = rng.uniform(0.0, 2.0 * math.pi, m)
    radius = trunk / 2.0 + rng.normal(0.0, 0.004, m)
    stem = np.column_stack([xy[0] + radius * np.cos(angle), xy[1] + radius * np.sin(angle), base + rng.uniform(0.3, height * 0.4, m)])
    return canopy, stem


def _disc(rng: np.random.Generator, xy: tuple[float, float], diameter: float, n: int = 90) -> np.ndarray:
    radius = diameter / 2.0 * np.sqrt(rng.uniform(0.0, 1.0, n)) * 0.97
    angle = rng.uniform(0.0, 2.0 * math.pi, n)
    x, y = xy[0] + radius * np.cos(angle), xy[1] + radius * np.sin(angle)
    return np.column_stack([x, y, terrain_z(x, y) + 0.004 + rng.normal(0.0, 0.002, n)])


def make_pointset() -> Any:
    from samcloud.drone.contracts import PointSet

    rng = _rng(7)
    gx, gy = np.meshgrid(np.arange(0.13, 40.0, 0.35), np.arange(0.11, 30.0, 0.35))
    gx = gx.ravel() + rng.normal(0.0, 0.03, gx.size)
    gy = gy.ravel() + rng.normal(0.0, 0.03, gy.size)
    ground = np.column_stack([gx, gy, terrain_z(gx, gy) + rng.normal(0.0, 0.004, gx.size)])
    classes = np.where(gy < 6.0, 11, 2).astype(np.uint16)
    parts, codes = [ground], [classes]
    for tree in (TREE_1, TREE_2, EXTRA_TREE):
        canopy, stem = _tree_points(rng, tree["xy"], tree["height"], tree["crown"], tree["trunk"])
        parts += [canopy, stem]
        codes += [np.full(len(canopy), 5, np.uint16), np.full(len(stem), 95, np.uint16)]
    for manhole in (MANHOLE, EXTRA_MANHOLE):
        disc = _disc(rng, manhole["xy"], manhole["diameter"])
        parts.append(disc)
        codes.append(np.full(len(disc), 75, np.uint16))
    fence = []
    for (x0, y0), (x1, y1) in zip(FENCE_AXIS[:-1], FENCE_AXIS[1:]):
        t = rng.uniform(0.0, 1.0, 400)
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        fence.append(np.column_stack([x, y, terrain_z(x, y) + rng.uniform(0.05, FENCE_HEIGHT, 400)]))
    fence_pts = np.vstack(fence)
    parts.append(fence_pts)
    codes.append(np.full(len(fence_pts), 77, np.uint16))
    xyz = np.vstack(parts)
    return PointSet(xyz=xyz, classes=np.concatenate(codes))


# ----------------------------------------------------------------------------------------------------------- elements

def make_trees() -> list[Any]:
    from samcloud.bim.core.model import ParamOrigin, Status, TreeElement, TreeParams

    trees = []
    for number, tree in enumerate((TREE_1, TREE_2), start=1):
        base = float(terrain_z(*tree["xy"]))
        trees.append(TreeElement(
            id=tree["id"], name=f"Baum {number}",
            params=TreeParams(trunk_xy=tree["xy"], base_z=base, height=tree["height"], crown_diameter=tree["crown"],
                              trunk_diameter=tree["trunk"], crown_shape="ellipsoid", crown_base_z=base + tree["height"] * 0.4,
                              crown_offset=(0.0, 0.0), trunk_measured=False),
            param_origin={**{name: ParamOrigin.MEASURED for name in ("trunk_xy", "base_z", "height", "crown_diameter", "crown_shape")},
                          "trunk_diameter": ParamOrigin.ASSUMED},
            status=Status.WARNING, source_instance_ids=[f"tree-{number}"], evidence={"tree_id": number, "views": 4, "points": 1600},
            notes=["assumed: trunk_diameter"]))
    return trees


def make_site_elements(*, renumber: int = 0, drop_fence: bool = False) -> list[Any]:
    """Site wall, fence, curb and manhole as the site step builds them (``renumber`` shifts the order-dependent ids)."""
    from samcloud.bim.core.model import (CurbElement, CurbParams, FenceElement, FenceParams, ManholeElement, ManholeParams,
                                         ParamOrigin, SiteWallElement, SiteWallParams, Status)
    from samcloud.outdoor_model.site_walls import wall_polygon

    wall_axis = np.asarray(WALL_AXIS)
    base = terrain_z(wall_axis[:, 0], wall_axis[:, 1])
    ring = wall_polygon(wall_axis, WALL_THICKNESS)
    wall = SiteWallElement(
        id=f"site-wall-{renumber}", tag=str(renumber), name="Mauer",
        params=SiteWallParams(axis=[tuple(map(float, p)) for p in wall_axis], footprint=[tuple(map(float, p)) for p in ring],
                              thickness=WALL_THICKNESS, base_z=float(base.min()), top_z=float(base.min()) + WALL_HEIGHT,
                              base_z_axis=[float(v) for v in base], top_z_axis=[float(v) + WALL_HEIGHT for v in base]),
        param_origin={"axis": ParamOrigin.FITTED, "footprint": ParamOrigin.FITTED, "thickness": ParamOrigin.FITTED,
                      "base_z": ParamOrigin.MEASURED, "top_z": ParamOrigin.FITTED},
        evidence={"points": 900, "thickness_m": WALL_THICKNESS, "vertices": 3, "top_varies": True}, status=Status.OK)
    fence_axis = np.asarray(FENCE_AXIS)
    bottom = terrain_z(fence_axis[:, 0], fence_axis[:, 1])
    fence = FenceElement(
        id=f"fence-{renumber}", tag=str(renumber), name="Zaun",
        params=FenceParams(axis=[(float(x), float(y), float(b) + FENCE_HEIGHT) for (x, y), b in zip(fence_axis, bottom)],
                           height=FENCE_HEIGHT, bottom_z=[float(v) for v in bottom], thickness=0.04),
        param_origin={"axis": ParamOrigin.FITTED, "height": ParamOrigin.FITTED, "bottom_z": ParamOrigin.MEASURED},
        evidence={"points": 800, "height_m": FENCE_HEIGHT, "vertices": 3, "dxf_bottom_line": False}, status=Status.OK)
    curb_axis = np.asarray(CURB_AXIS)
    foot = terrain_z(curb_axis[:, 0], curb_axis[:, 1])
    curb = CurbElement(
        id=f"curb-{renumber}", tag=str(renumber),
        params=CurbParams(axis=[(float(x), float(y), float(f) + 0.12) for (x, y), f in zip(curb_axis, foot)], width=0.12, height=0.25,
                          foot_z=[float(v) for v in foot], road_side=-1, gutter_width=0.0),
        param_origin={"axis": ParamOrigin.MEASURED, "width": ParamOrigin.ASSUMED, "height": ParamOrigin.ASSUMED,
                      "curb_sweep": ParamOrigin.ASSUMED, "foot_z": ParamOrigin.MEASURED},
        source_instance_ids=[str(renumber)], evidence={"length_m": 37.94, "vertices": 3, "road_side": -1, "flush": False,
                                                       "reveal_m": 0.12, "gutter_width_m": 0.0, "gutter": "none"},
        status=Status.OK)
    mx, my = MANHOLE["xy"]
    manhole = ManholeElement(
        id=f"manhole-{renumber}", tag=str(renumber),
        params=ManholeParams(center=(mx, my, float(terrain_z(mx, my)) + 0.004), diameter=MANHOLE["diameter"], type="Schacht"),
        param_origin={"center": ParamOrigin.MEASURED, "diameter": ParamOrigin.FITTED, "type": ParamOrigin.ASSUMED},
        source_instance_ids=[str(renumber)], evidence={"z_deckel_local": round(float(terrain_z(mx, my)), 3), "roundness": 0.95,
                                                         "diameter_measured_m": MANHOLE["diameter"], "snapped": False, "standard": ""},
        status=Status.OK)
    elements = [wall, fence, curb, manhole]
    if drop_fence:
        elements.remove(fence)
    return elements


def make_building() -> Any:
    from samcloud.bim.core.model import BuildingElement, BuildingParams, ParamOrigin, Status

    x0, y0, x1, y1 = BUILDING
    g = float(terrain_z(x0, y0)) - 0.05
    ym = (y0 + y1) / 2.0
    v = [(x0, y0, g), (x1, y0, g), (x1, y1, g), (x0, y1, g),
         (x0, y0, g + BUILDING_EAVE), (x1, y0, g + BUILDING_EAVE), (x1, y1, g + BUILDING_EAVE), (x0, y1, g + BUILDING_EAVE),
         (x0, ym, g + BUILDING_RIDGE), (x1, ym, g + BUILDING_RIDGE)]
    ground = [(0, 2, 1), (0, 3, 2)]
    wall_faces = [(0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (5, 6, 9), (3, 0, 4), (3, 4, 7), (7, 4, 8)]
    roof = [(4, 5, 9), (4, 9, 8), (6, 7, 8), (6, 8, 9)]
    triangles = ground + wall_faces + roof
    roles = [0] * len(ground) + [1] * len(wall_faces) + [2] * len(roof)
    return BuildingElement(id="building-1", name="Building 1",
                           params=BuildingParams(vertices=[tuple(map(float, p)) for p in v], triangles=triangles, face_roles=roles),
                           param_origin={"vertices": ParamOrigin.FITTED}, status=Status.OK, evidence={"lod": "2.2"})


# --------------------------------------------------------------------------------------------------------- checkpoints

def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def make_tin() -> Any:
    from samcloud.terrain.tin import build_labeled_tin

    gx, gy = np.meshgrid(np.arange(0.0, 40.0 + 1e-9, 1.0), np.arange(0.0, 30.0 + 1e-9, 1.0))
    ground = np.column_stack([gx.ravel(), gy.ravel(), terrain_z(gx.ravel(), gy.ravel())])
    return build_labeled_tin(ground, regions={"road": [ROAD], "sidewalk": [], "path": [], "lawn": []}, initial_spacing=2.0,
                             tolerance=0.05, max_vertices=4000)


def write_checkpoints(project: Path, *, renumber: int = 0, drop_fence: bool = False, new_tree_ids: bool = False) -> None:
    from samcloud.drone.contracts import Evidence, LocalFrame, PavementPolygon
    from samcloud.outdoor_model.vegetation import VegetationResult

    out = project / "products" / "environment"
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    frame = LocalFrame(FAKE_ORIGIN)
    if not (cache / "load.pkl").is_file():
        _dump(cache / "load.pkl", (make_pointset(), frame))
    tin = make_tin()
    _dump(cache / "tin.pkl", tin)
    xs = np.arange(SITE[0] + TERRAIN_STEP / 2, SITE[2], TERRAIN_STEP)
    ys = np.arange(SITE[3] - TERRAIN_STEP / 2, SITE[1], -TERRAIN_STEP)
    gx, gy = np.meshgrid(xs, ys)
    _dump(cache / "terrain.pkl", {"dtm": terrain_z(gx, gy), "validity": np.ones(gx.shape, np.uint8), "bounds": SITE,
                                  "resolution": TERRAIN_STEP})
    trees = make_trees()
    if new_tree_ids:
        trees = [tree.model_copy(update={"id": str(uuid.uuid4())}) for tree in trees]
    _dump(cache / "vegetation.pkl", VegetationResult(trees, [], []))
    _dump(cache / "pavement.pkl", {"polygons": [PavementPolygon(0, "road", ROAD, Evidence(points=5000))], "curbs": [], "lawns": [],
                                   "topology_qa": []})
    _dump(cache / "buildings.pkl", {"elements": [make_building()], "qa": []})
    _dump(cache / "site.pkl", {"elements": make_site_elements(renumber=renumber, drop_fence=drop_fence), "qa": []})
    state = out / "state.json"
    if not state.is_file():
        state.write_text(json.dumps({"steps": {}}), encoding="utf-8")


def run_scene_step(project: Path) -> Any:
    """The real ``_step_scene`` on the synthetic checkpoints; returns the stored Scene."""
    from samcloud.drone.contracts import load_drone_config
    from samcloud.drone.run_environment import _step_scene

    out = project / "products" / "environment"
    _step_scene(project, out, load_drone_config(), out / "cache")
    with (out / "cache" / "scene.pkl").open("rb") as handle:
        return pickle.load(handle)["scene"]


def make_outdoor_project(root: Path, name: str = "outdoor_synthetic") -> Path:
    """A drone project with checkpoints and ``scene.pkl`` (no exports)."""
    project = Path(root) / name
    project.mkdir(parents=True, exist_ok=True)
    (project / "project.samcloud.yml").write_text("name: outdoor_synthetic\nmode: outdoor\nacquisition:\n  type: drone_gps\n", encoding="utf-8")
    write_checkpoints(project)
    run_scene_step(project)
    return project


def renumbered_recompute(project: Path, *, drop_fence: bool = False) -> Any:
    """A recompute that renumbers the order-dependent ids and gives the trees new uuids (optionally drops the fence)."""
    write_checkpoints(project, renumber=3, drop_fence=drop_fence, new_tree_ids=True)
    return run_scene_step(project)


__all__ = ["BUILDING", "BUILDING_EAVE", "BUILDING_RIDGE", "CURB_AXIS", "EMPTY_POINT", "EXTRA_MANHOLE", "EXTRA_TREE", "FAKE_ORIGIN",
           "FENCE_AXIS", "FENCE_HEIGHT", "MANHOLE", "TREE_1", "TREE_2", "WALL_AXIS", "lv95", "make_outdoor_project", "make_pointset",
           "renumbered_recompute", "run_scene_step", "terrain_z", "write_checkpoints"]


_SHARED: dict[str, Path] = {}


def shared_project() -> Path:
    """One synthetic project per test process (read-only use; copy it with :func:`copy_project` before writing)."""
    import os

    if "base" not in _SHARED or not (_SHARED["base"] / "products" / "environment" / "cache" / "scene.pkl").is_file():
        root = Path(os.path.realpath(os.environ["SAMCLOUD_TEST_WORKDIR"])) / f"outdoor_shared_{os.getpid()}_{uuid.uuid4().hex[:6]}"
        _SHARED["base"] = make_outdoor_project(root)
    return _SHARED["base"]


def copy_project(target: Path) -> Path:
    """A writable copy of :func:`shared_project` below ``target``."""
    import shutil

    destination = Path(target) / "outdoor_synthetic"
    shutil.copytree(shared_project(), destination)
    return destination


# ------------------------------------------------------------------------------------------------------ golden digest

def _rounded(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits) + 0.0
    if isinstance(value, dict):
        return {str(k): _rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v, digits) for v in value]
    if isinstance(value, np.ndarray):
        return _rounded(value.tolist(), digits)
    if isinstance(value, np.generic):
        return _rounded(value.item(), digits)
    if type(value).__name__ in ("Vec2", "Vec3"):  # ezdxf vectors
        return _rounded([float(c) for c in value], digits)
    return value


def _sha(value: Any) -> str:
    import hashlib

    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def golden_digest(scene: Any, dxf_result: Any, work: Path) -> dict[str, Any]:
    """Digest of a scene-step output: Scene JSON, neutral DXF entities, the DXF document and the IFC (no export language,
    fixed IFC time stamp; floats rounded to 1e-6 so platform noise below that does not count)."""
    import re
    from datetime import datetime, timezone

    import ezdxf

    from samcloud.bim.core.ifc.writer import write_ifc
    from samcloud.drone.contracts import LocalFrame
    from samcloud.dxf.writer import build_dxf_document

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    layers = {name: _rounded(entities) for name, entities in dxf_result.layers.items()}
    styling = _rounded({"blocks": sorted(dxf_result.blocks), "colors": dxf_result.layer_colors,
                        "linetypes": dxf_result.layer_linetypes, "styles": dxf_result.layer_styles})
    frame = LocalFrame(tuple(float(v) for v in scene.settings.origin_offset))
    doc = build_dxf_document(dxf_result, coordinates="local", origin_offset=frame.origin_lv95)
    dxf_path = work / "golden.dxf"
    doc.saveas(dxf_path)
    rows = []
    for entity in ezdxf.readfile(dxf_path).modelspace():
        if entity.dxftype() == "TEXT" and entity.dxf.layer == "X-NOTE":
            continue  # the export date
        attribs = {key: value for key, value in entity.dxf.all_existing_dxf_attribs().items() if key not in ("handle", "owner")}
        extra: Any = None
        if entity.dxftype() == "POLYLINE":
            extra = [list(v.dxf.location) for v in entity.vertices]
        elif entity.dxftype() == "LWPOLYLINE":
            extra = [list(p) for p in entity.get_points()]
        elif entity.dxftype() == "INSERT":
            extra = sorted((a.dxf.tag, a.dxf.text) for a in entity.attribs)
        rows.append(json.dumps(_rounded([entity.dxftype(), attribs, extra]), sort_keys=True, default=str))
    ifc_path = work / "golden.ifc"
    write_ifc(scene, ifc_path, timestamp=datetime(2000, 1, 1, tzinfo=timezone.utc))
    ifc_text = re.sub(r"FILE_NAME\(.*?\);", "FILE_NAME();", ifc_path.read_text(encoding="utf-8"), flags=re.S)
    scene_data = _rounded(json.loads(scene.to_json()), 9)
    return {"scene_sha256": _sha(scene_data), "dxf_entities_sha256": _sha(layers), "dxf_styling_sha256": _sha(styling),
            "dxf_document_sha256": _sha(sorted(rows)), "ifc_sha256": _sha(ifc_text),
            "counts": {"elements": len(scene.elements), "layers": {name: len(items) for name, items in sorted(dxf_result.layers.items())},
                       "dxf_entities": len(rows), "ifc_lines": ifc_text.count("\n")}}
