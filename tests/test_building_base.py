"""Building base: wall bottoms follow the terrain; one wall and one roof per house in the IFC."""
from collections import Counter

import ifcopenshell
import ifcopenshell.geom
import numpy as np
import trimesh
from shapely.geometry import Polygon

from samcloud.bim.core.ifc.writer import write_ifc
from samcloud.buildings.base import make_ground_function
from samcloud.buildings.extrude import extrude_lod22
from samcloud.buildings.partition import Partition
from samcloud.buildings.validate import validate_model
from samcloud.drone.contracts import BuildingModel, LocalFrame, RoofPlane
from samcloud.outdoor_model.buildings import build_building_elements
from samcloud.drone.contracts import Opening
from samcloud.outdoor_model.scene import assemble_scene

EMBED = 0.02


def _terrain(x0: float = 0.0, y0: float = 0.0):
    """Sloping, slightly curved terrain: rises 1.5 m over 10 m along x, 4 cm waves, 1 cm noise-like ripple."""
    def terrain(x: float, y: float) -> float:
        u = x - x0
        return 0.15 * u + 0.04 * np.sin(0.9 * u + 0.3) + 0.01 * np.sin(7.3 * u + 2.1 * (y - y0)) + 0.3

    return terrain


def _plane(height: float, outline) -> RoofPlane:
    return RoofPlane(np.array([0.0, 0.0, 1.0]), -height, np.array([], dtype=int), 0.0, outline=Polygon(outline))


def _house(terrain, x0: float = 0.0, y0: float = 0.0, settings=None):
    footprint = Polygon([(x0, y0), (x0 + 10.0, y0), (x0 + 10.0, y0 + 8.0), (x0, y0 + 8.0)])
    cells = [Polygon([(x0, y0), (x0 + 5.0, y0), (x0 + 5.0, y0 + 8.0), (x0, y0 + 8.0)]), Polygon([(x0 + 5.0, y0), (x0 + 10.0, y0), (x0 + 10.0, y0 + 8.0), (x0 + 5.0, y0 + 8.0)])]
    planes = [_plane(7.0, cells[0].exterior.coords), _plane(8.3, cells[1].exterior.coords)]
    ground = make_ground_function(terrain, footprint, settings)
    vertices, faces, roles = extrude_lod22(Partition(cells, [(0, 1, 8), (1, 0, 8)]), np.array([1, 2]), planes, footprint, ground, 2.0)
    return footprint, ground, vertices, faces, roles


def _floor_boundary_edges(faces, roles):
    count: Counter = Counter()
    for face in faces[roles == 0]:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            count[tuple(sorted((int(a), int(b))))] += 1
    return [edge for edge, n in count.items() if n == 1]


def test_wall_bottom_follows_sloping_terrain_without_gap_and_is_watertight():
    terrain = _terrain()
    footprint, _ground, vertices, faces, roles = _house(terrain)
    floor_vertices = np.unique(faces[roles == 0].ravel())
    for index in floor_vertices:
        x, y, z = vertices[index]
        assert abs(z - (terrain(x, y) - EMBED)) <= 0.01
    # the bottom edge between two vertices stays within a few cm of the terrain, sampled every 0.5 m
    worst = 0.0
    edges = _floor_boundary_edges(faces, roles)
    assert len(edges) > 4  # the curved terrain forced extra vertices on the long edges
    for a, b in edges:
        start, end = vertices[a], vertices[b]
        length = float(np.linalg.norm(end[:2] - start[:2]))
        for t in np.linspace(0.0, 1.0, max(2, int(length / 0.5) + 1)):
            point = start + (end - start) * t
            worst = max(worst, abs(point[2] - (terrain(point[0], point[1]) - EMBED)))
    assert worst <= 0.04
    report = validate_model(vertices, faces, np.array([[1.0, 1.0, 7.0], [8.0, 1.0, 8.3]]), footprint)
    assert report["status"] == "ok", report["reason"]
    assert trimesh.Trimesh(vertices=vertices, faces=faces, process=False).is_watertight


def test_flat_min_mode_puts_every_bottom_at_the_lowest_terrain():
    terrain = _terrain()
    footprint, ground, vertices, faces, roles = _house(terrain, settings={"base": {"mode": "flat_min"}})
    xs, ys = np.meshgrid(np.linspace(0.0, 10.0, 201), np.linspace(0.0, 8.0, 161))
    lowest = min(terrain(x, y) for x, y in zip(xs.ravel(), ys.ravel(), strict=True))
    bottoms = vertices[np.unique(faces[roles == 0].ravel()), 2]
    assert np.allclose(bottoms, lowest - EMBED, atol=0.01)
    assert len(np.unique(np.round(bottoms, 6))) == 1
    assert trimesh.Trimesh(vertices=vertices, faces=faces, process=False).is_watertight


def test_walls_never_exceed_the_roof():
    _fp, _ground, vertices, faces, roles = _house(_terrain())
    wall_z = vertices[np.unique(faces[roles == 1].ravel()), 2]
    roof_z = vertices[np.unique(faces[roles == 2].ravel()), 2]
    assert wall_z.max() <= roof_z.max() + 0.01


def _model(building_id: int, x0: float, terrain) -> BuildingModel:
    _fp, _ground, vertices, faces, roles = _house(terrain, x0=x0)
    return BuildingModel(building_id, "2.2", vertices, faces, roles, "ok")


def _world(product) -> np.ndarray:
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    return np.asarray(ifcopenshell.geom.create_shape(settings, product).geometry.verts, float).reshape(-1, 3)


def test_two_houses_give_two_walls_and_two_roofs_with_all_faces(tmp_path):
    models = [_model(1, 0.0, _terrain()), _model(2, 30.0, _terrain(30.0))]
    openings = {1: [Opening(1, 0, "window", 0.5, 1.5, 2.0, 3.0)], 2: []}
    elements, qa = build_building_elements(models, openings, {})
    assert not qa
    scene = assemble_scene(elements, frame=LocalFrame((2600000, 1200000, 500)))
    path = tmp_path / "two.ifc"
    write_ifc(scene, path)
    ifc = ifcopenshell.open(str(path))

    walls, roofs = ifc.by_type("IfcWall"), ifc.by_type("IfcRoof")
    assert len(walls) == 2 and len(roofs) == 2
    assert sorted(w.Name for w in walls) == ["Fassade 1", "Fassade 2"]
    assert not [s for s in ifc.by_type("IfcSlab") if s.PredefinedType == "ROOF"]
    for model, wall, roof in zip(models, sorted(walls, key=lambda w: w.Name), sorted(roofs, key=lambda r: r.Name), strict=True):
        wall_faces = int((model.face_roles == 1).sum())
        roof_faces = int((model.face_roles == 2).sum())
        for product, expected in ((wall, wall_faces), (roof, roof_faces)):
            item = product.Representation.Representations[0].Items[0]
            assert len(item.CoordIndex) == expected
        assert _world(wall)[:, 2].max() <= _world(roof)[:, 2].max() + 0.01
    # S-BLDG layer on the walls, a presentation style on the body
    for wall in walls:
        shape = wall.Representation.Representations[0]
        assert any(layer.Name == "S-BLDG" for layer in shape.LayerAssignments)
        assert shape.Items[0].StyledByItem
    # the window voids the merged wall of house 1 and sits on the facade
    openings_ifc = ifc.by_type("IfcOpeningElement")
    assert len(openings_ifc) == 1
    host = openings_ifc[0].VoidsElements[0].RelatingBuildingElement
    assert host.Name == "Fassade 1"
    box = _world(host)
    centre = _world(openings_ifc[0]).mean(axis=0)
    assert box[:, 0].min() - 0.3 <= centre[0] <= box[:, 0].max() + 0.3
    assert box[:, 1].min() - 0.3 <= centre[1] <= box[:, 1].max() + 0.3


def _fine_hip_roof(levels: int = 2, seed: int = 4):
    """8 x 6 m hip roof (ridge z=5 from x=2..6, eaves z=3), each plane split into many noisy triangles."""
    rng = np.random.default_rng(seed)
    corners = {"a": (0, 0, 3), "b": (8, 0, 3), "c": (8, 6, 3), "d": (0, 6, 3), "r1": (2, 3, 5), "r2": (6, 3, 5)}
    planes = [("a", "b", "r2"), ("a", "r2", "r1"), ("c", "d", "r1"), ("c", "r1", "r2"), ("d", "a", "r1"), ("b", "c", "r2")]
    vertices: list[np.ndarray] = []
    lookup: dict[tuple, int] = {}

    def vid(point) -> int:
        key = tuple(np.round(point, 6))
        if key not in lookup:
            lookup[key] = len(vertices)
            vertices.append(np.asarray(point, float) + np.array([0.0, 0.0, rng.normal(0.0, 0.0004)]))
        return lookup[key]

    def split(tri, level):
        if level == 0:
            return [tri]
        p, q, r = (np.asarray(corners[n], float) if isinstance(n, str) else n for n in tri)
        m = [(p + q) / 2, (q + r) / 2, (r + p) / 2]
        children = [(p, m[0], m[2]), (m[0], q, m[1]), (m[2], m[1], r), (m[0], m[1], m[2])]
        return [t for c in children for t in split(c, level - 1)]

    faces = []
    for tri in planes:
        for t in split(tri, levels):
            pts = [np.asarray(corners[n], float) if isinstance(n, str) else n for n in t]
            faces.append([vid(p) for p in pts])
    return np.asarray(vertices), np.asarray(faces)


def test_hip_roof_dxf_has_only_outline_ridge_and_hips_no_diagonals():
    from samcloud.outdoor_model.buildings import roof_lines

    vertices, faces = _fine_hip_roof()
    assert len(faces) > 90
    chains = roof_lines(vertices, faces, coplanar_deg=2.0)
    assert len(chains) == 6  # one closed eave loop (4 corners) + 1 ridge + 4 hips
    loops = [c for c in chains if len(c) > 2]
    assert len(loops) == 1 and len(loops[0]) == 5 and np.allclose(loops[0][0], loops[0][-1])
    straight = [c for c in chains if len(c) == 2]  # merged collinear chains, DP 1 cm
    assert len(straight) == 5
    ridge = [c for c in straight if abs(c[0][2] - 5.0) < 0.01 and abs(c[1][2] - 5.0) < 0.01]
    assert len(ridge) == 1 and abs(np.linalg.norm(ridge[0][0] - ridge[0][1]) - 4.0) < 0.02


def test_roof_edge_is_its_own_closed_light_blue_layer_and_not_drawn_twice():
    from samcloud.bim.core.model import BuildingElement, BuildingParams
    from samcloud.outdoor_model.buildings import to_dxf_entities
    from samcloud.outdoor_model.style import load_outdoor_style

    vertices, faces = _fine_hip_roof()
    floor = [(0, 0, 0), (8, 0, 0), (8, 6, 0), (0, 6, 0)]
    start = len(vertices)
    all_vertices = np.vstack([vertices, np.asarray(floor, float)])
    all_faces = np.vstack([faces, [[start, start + 2, start + 1], [start, start + 3, start + 2]]])
    roles = [2] * len(faces) + [0, 0]
    element = BuildingElement(id="building-1", tag="1", name="Building 1", params=BuildingParams(
        vertices=[tuple(map(float, v)) for v in all_vertices], triangles=[tuple(map(int, f)) for f in all_faces], face_roles=roles))
    dxf = to_dxf_entities([element], LocalFrame((2600000, 1200000, 500)))
    edge = dxf["S-BLDG-ROOF-EDGE"]
    assert len(edge) == 1 and edge[0]["closed"] and len(edge[0]["points"]) == 4  # one closed polyline, 4 corners
    inner = dxf["S-BLDG-ROOF"]
    assert len(inner) == 5 and all(len(e["points"]) == 2 for e in inner)  # ridge + 4 hips, no eave duplicated
    style = load_outdoor_style()["roof_edge"]
    assert style["layer"] == "S-BLDG-ROOF-EDGE" and not style["ifc"]
    r, g, b = style["rgb"]
    assert b >= 240 and g >= 170 and r <= 140 and style["aci"] == 151
