"""Tests for samcloud.drone.contracts and config/drone_outdoor.yaml."""
import numpy as np
import yaml

from samcloud.core.paths import CONFIG_DIRECTORY
from samcloud.drone import contracts as c


def test_local_frame_round_trip():
    frame = c.LocalFrame((2600000.123, 1200000.456, 430.789))
    rng = np.random.default_rng(1)
    xyz = rng.normal(size=(20, 3)) * 50
    enh = frame.to_lv95(xyz)
    assert np.allclose(enh[0], xyz[0] + np.array(frame.origin_lv95))
    assert np.allclose(frame.from_lv95(enh), xyz)


def test_pointset_subset_mask_and_index():
    rng = np.random.default_rng(2)
    n = 30
    ps = c.PointSet(
        xyz=rng.normal(size=(n, 3)),
        classes=np.arange(n, dtype=np.uint16),
        confidence=rng.random(n).astype(np.float32),
        margin=rng.random(n).astype(np.float32),
        view_count=np.arange(n, dtype=np.uint16) + 7,
        normals=rng.normal(size=(n, 3)),
        rgb=rng.integers(0, 255, (n, 3), dtype=np.uint8),
        adjusted=rng.random(n) > 0.5,
    )
    mask = ps.classes % 3 == 0
    idx = np.array([5, 2, 9])
    for sel, expect in ((mask, np.flatnonzero(mask)), (idx, idx)):
        sub = ps.subset(sel)
        assert len(sub) == len(expect)
        for name in ("xyz", "classes", "confidence", "margin", "view_count", "normals", "rgb", "adjusted"):
            assert np.array_equal(getattr(sub, name), getattr(ps, name)[expect])
    sparse = c.PointSet(xyz=ps.xyz, classes=ps.classes).subset(mask)
    assert sparse.confidence is None and sparse.rgb is None and sparse.adjusted is None
    assert len(sparse) == int(mask.sum())


def test_plane_distance_sign():
    plane = c.Plane(normal=np.array([0.0, 0.0, 1.0]), d=-2.0, inliers=np.array([0]), rmse=0.0)
    dist = plane.distance(np.array([[1.0, 2.0, 3.5], [0.3, 0.1, 0.7]]))
    assert np.allclose(dist, [1.5, -1.3])


def _tin():
    verts = np.array([[0, 0, 0], [1, 0, 0.1], [1, 1, 0.2], [0, 1, 0.3], [2, 0, 0.4], [2, 1, 0.5]], float)
    tris = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]])
    labels = np.array([c.TIN_LABELS["terrain"], c.TIN_LABELS["terrain"],
                       c.TIN_LABELS["road"], c.TIN_LABELS["road"]], np.uint8)
    return c.LabeledTin(verts, tris, labels)


def test_labeled_tin_select_and_without():
    tin = _tin()
    road = tin.select("road")
    assert len(road.triangles) == 2 and len(road.vertices) == 4
    assert (road.labels == c.TIN_LABELS["road"]).all()
    orig = tin.vertices[tin.triangles[2:]]
    assert np.allclose(road.vertices[road.triangles], orig)
    rest = tin.without("road")
    assert len(rest.triangles) == 2 and len(rest.vertices) == 4
    assert rest.triangles.max() < len(rest.vertices)
    assert (rest.labels == c.TIN_LABELS["terrain"]).all()
    assert len(tin.select("terrain", "road").triangles) == 4
    assert len(tin.select("lawn").triangles) == 0


def test_load_drone_config_yaml_merge_and_defaults(tmp_path):
    yaml_path = CONFIG_DIRECTORY / "drone_outdoor.yaml"
    with open(yaml_path, encoding="utf-8") as fh:
        file_cfg = yaml.safe_load(fh)
    assert c.load_drone_config() == file_cfg
    assert c._DEFAULT_CONFIG == file_cfg
    assert file_cfg["terrain"]["csf"]["cloth_resolution_m"] == 0.5
    part = tmp_path / "partial.yaml"
    part.write_text("terrain:\n  csf: {rigidness: 3}\ndxf: {hatch: true}\n", encoding="utf-8")
    merged = c.load_drone_config(part)
    assert merged["terrain"]["csf"]["rigidness"] == 3
    assert merged["terrain"]["csf"]["iterations"] == 500
    assert merged["dxf"]["hatch"] is True and merged["dxf"]["contour_major_m"] == 1.0
    assert merged["buildings"] == file_cfg["buildings"]


def test_class_codes_match_cyclone_codes():
    with open(CONFIG_DIRECTORY / "cyclone_codes.yaml", encoding="utf-8") as fh:
        codes = yaml.safe_load(fh)["codes"]
    expected = {
        c.GROUND: "Ground", c.LOW_VEGETATION: "Low Vegetation", c.TREE: "High Vegetation",
        c.BUILDING: "Building", c.ROAD: "Road Surface", c.SITE_WALL: "Retaining wall",
        c.SIDEWALK: "Side walk", c.CURB: "Curbstone", c.MANHOLE: "Manhole",
        c.FENCE: "Guard rails", c.SHRUB: "Shrub", c.TREE_TRUNK: "Trunk", c.DOOR: "Door",
        c.WINDOW: "Window", c.ROOF_INCLINED: "Inclined roof", c.ROOF_FLAT: "Flat roof",
    }
    assert sorted(expected) == [2, 3, 5, 6, 11, 72, 73, 74, 75, 77, 94, 95, 103, 104, 144, 145]
    for code, name in expected.items():
        assert codes[code]["name"] == name, code
