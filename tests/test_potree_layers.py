"""Potree viewer layer panel: environment_layers helper, generated HTML/JS and the viewer_export grouping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from samcloud.app import potree_viewer

ORIGIN = [2600100.0, 1200200.0, 400.0]
NODE_NAMES = ["terrain", "building_001", "building_002", "vegetation_tree_01", "pavement_street", "bench_misc"]


def _write_environment_glb(project: Path, names=NODE_NAMES) -> Path:
    scene = trimesh.Scene()
    for index, name in enumerate(names):
        box = trimesh.creation.box(extents=(2.0 + index * 0.3, 3.1, 1.7))
        box.apply_translation((index * 5.3, 0.7, 0.4))
        scene.add_geometry(box, node_name=name, geom_name=name)
    scene.metadata["origin_lv95"] = ORIGIN
    path = project / "products" / "environment" / "environment.glb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scene.export(file_type="glb"))
    return path


def test_environment_layers_lists_groups_and_origin(tmp_path: Path) -> None:
    _write_environment_glb(tmp_path)
    layers = potree_viewer.environment_layers(tmp_path)
    assert layers["available"] is True
    assert layers["glb"]["origin_lv95"] == ORIGIN
    assert layers["glb"]["url"] == potree_viewer.ENVIRONMENT_GLB_URL
    groups = layers["groups"]
    assert groups["terrain"] == ["terrain"]
    assert groups["building"] == ["building_001", "building_002"]
    assert groups["vegetation"] == ["vegetation_tree_01"]
    assert groups["pavement"] == ["pavement_street"]
    assert groups["other"] == ["bench_misc"]
    for key in ("model", "dgm", "buildings", "vegetation", "streets", "rgb", "classes"):
        assert layers["layers"][key]["available"] is True, key
    # no DSM node in the GLB -> DOM stays unavailable
    assert layers["layers"]["dom"]["available"] is False
    assert layers["layers"]["dom"]["reason"] == "nicht berechnet"


def test_missing_glb_marks_layers_unavailable(tmp_path: Path) -> None:
    layers = potree_viewer.environment_layers(tmp_path)
    assert layers["available"] is False
    assert layers["glb"]["available"] is False and layers["glb"]["nodes"] == []
    for key in ("model", "dgm", "dom", "buildings", "vegetation", "streets"):
        assert layers["layers"][key]["available"] is False, key
        assert layers["layers"][key]["reason"] == "nicht berechnet"
    # point cloud toggles never depend on the meshes
    assert layers["layers"]["rgb"]["available"] and layers["layers"]["classes"]["available"]


def test_corrupt_glb_is_unavailable_not_an_error(tmp_path: Path) -> None:
    path = tmp_path / "products" / "environment" / "environment.glb"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a glb at all, just some text")
    layers = potree_viewer.environment_layers(tmp_path)
    assert layers["available"] is False
    assert layers["layers"]["buildings"]["available"] is False


def test_node_group_prefers_extras_then_name() -> None:
    assert potree_viewer.node_group("abc", {"ifc_class": "IfcWall"}) == "building"
    assert potree_viewer.node_group("abc", {"ifc_class": "IfcGeographicElement"}) == "vegetation"
    assert potree_viewer.node_group("abc", {"category": "pavement"}) == "pavement"
    assert potree_viewer.node_group("DSM_surface") == "dsm"
    assert potree_viewer.node_group("xyz") == "other"


def _dense_page(tmp_path: Path) -> tuple[str, str, dict]:
    config = potree_viewer._viewer_config("dense", "x", [[1, 2, 3]], ["Boden"])
    html_text = potree_viewer._html("sam-potree-dense")
    return html_text, potree_viewer._script(config), config


def test_dense_viewer_shows_points_only_even_with_environment_meshes(tmp_path: Path, monkeypatch) -> None:
    """environment.glb / model.glb next to the cloud must never be loaded into the dense point-cloud viewer."""
    _write_environment_glb(tmp_path)
    site = tmp_path / "products" / "bim" / "site"
    site.mkdir(parents=True)
    (site / "model.glb").write_bytes((tmp_path / "products" / "environment" / "environment.glb").read_bytes())
    from samcloud.app.state import state

    monkeypatch.setattr(state, "active_project_directory", tmp_path, raising=False)
    html_text, script, config = _dense_page(tmp_path)
    assert "environment" not in config
    assert "data-layers-panel" not in html_text and 'data-action="layers"' not in html_text
    for forbidden in ("GLTFLoader", "THREE.MeshLambertMaterial", ".glb", "environment.glb", "model.glb", "ENVIRONMENT"):
        assert forbidden not in script, forbidden
    assert "__LAYERS_JS__" not in script
    # points are rendered as points (no surface/mesh reconstruction)
    assert "Potree.PointShape.CIRCLE" in script


def test_page_without_layers_is_unchanged() -> None:
    html_text = potree_viewer._html("sam-potree-sparse")
    assert "data-layers-panel" not in html_text and 'data-action="layers"' not in html_text
    script = potree_viewer._script({"element_id": "e", "class_colors": [], "class_labels": []})
    assert "GLTFLoader" not in script and "__LAYERS_JS__" not in script


def test_local_to_viewer_matrix_without_georeference_is_none(tmp_path: Path) -> None:
    assert potree_viewer.local_to_viewer_matrix(tmp_path, ORIGIN) is None


def test_local_to_viewer_matrix_inverts_georeference(tmp_path: Path, monkeypatch) -> None:
    """Fake ENU->LV95 map (rotation 0.96 deg, scale 1.0003, shift): the matrix must map local back to ENU."""
    from samcloud.drone import frames

    angle = np.radians(0.96)
    rot = 1.0003 * np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    shift = np.array([2600137.3, 1200211.9, 431.2])

    def enu_to_lv95(xyz, georef, **kwargs):
        return np.asarray(xyz, dtype=np.float64) @ rot.T + shift

    monkeypatch.setattr(frames, "project_georeference", lambda project: {})
    monkeypatch.setattr(frames, "enu_to_lv95", enu_to_lv95)
    matrix = np.array(potree_viewer.local_to_viewer_matrix(tmp_path, ORIGIN)).reshape(4, 4)
    local = np.array([[0.0, 0.0, 0.0], [31.7, -12.3, 4.1], [-55.2, 40.9, 9.3]])
    enu = local @ matrix[:3, :3].T + matrix[:3, 3]
    assert np.allclose(enu_to_lv95(enu, {}) - np.asarray(ORIGIN), local, atol=1e-6)


# ---------------------------------------------------------------------------------------- viewer_export


@pytest.fixture(scope="module")
def room_ifc(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from samcloud.bim.core.ifc import write_ifc
    from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene

    root = tmp_path_factory.mktemp("layers_ifc")
    ifc = root / "room.ifc"
    write_ifc(build_room_scene(RoomSpec(layout=RoomLayout.RECTANGLE)), ifc, run_id="layers-test")
    return ifc


def test_viewer_export_grouping_option(room_ifc: Path, tmp_path: Path) -> None:
    from samcloud.bim.viewer_export import export_viewer_assets

    default = export_viewer_assets(room_ifc, tmp_path / "default")
    grouped = export_viewer_assets(room_ifc, tmp_path / "grouped", categorize=True)

    assert default.categories == {}
    assert "categories" not in json.loads(default.props_path.read_text(encoding="utf-8"))
    default_nodes = potree_viewer.read_glb_document(default.glb_path)["nodes"]
    assert all("category" not in node["extras"] for node in default_nodes)

    assert grouped.categories and set(grouped.categories) == {node["name"] for node in default_nodes}
    grouped_nodes = potree_viewer.read_glb_document(grouped.glb_path)["nodes"]
    for node in grouped_nodes:
        assert node["extras"]["category"] == grouped.categories[node["name"]]
    assert "building" in grouped.categories.values()
    assert json.loads(grouped.props_path.read_text(encoding="utf-8"))["categories"] == grouped.categories
    # the geometry of both exports is identical
    assert default.counts == grouped.counts
    assert {n["name"] for n in default_nodes} == {n["name"] for n in grouped_nodes}


def test_category_helpers() -> None:
    from samcloud.bim.viewer_export import categorize_nodes, category_for_ifc_class

    assert category_for_ifc_class("IfcWall") == "building"
    assert category_for_ifc_class("IfcPlant") == "vegetation"
    assert category_for_ifc_class("IfcSomethingNew") == "other"
    assert categorize_nodes({"a": {"ifc_class": "IfcSlab"}, "b": {}}) == {"a": "building", "b": "other"}
