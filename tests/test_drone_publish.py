"""Publishing outdoor (drone) exports into the app layout for level key "site"."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct

import pytest

from samcloud.app.dxf_controller import DxfPageController
from samcloud.bim import jobs as bim_jobs
from samcloud.bim import ui_data
from samcloud.drone.publish import _strict_loads_glb, environment_exports, publish_environment_exports, sanitize_glb


def _workdir(name: str) -> Path:
    path = Path(os.environ["SAMCLOUD_TEST_WORKDIR"]) / "drone_publish" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _write_ifc(path: Path) -> None:
    import ifcopenshell
    import ifcopenshell.api

    model = ifcopenshell.api.run("project.create_file", version="IFC4")
    project = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcProject", name="Site")
    ifcopenshell.api.run("unit.assign_unit", model)
    context = ifcopenshell.api.run("context.add_context", model, context_type="Model")
    body = ifcopenshell.api.run(
        "context.add_context", model, context_type="Model", context_identifier="Body",
        target_view="MODEL_VIEW", parent=context,
    )
    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite", name="Site")
    ifcopenshell.api.run("aggregate.assign_object", model, products=[site], relating_object=project)
    wall = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcWall", name="Wall 1")
    ifcopenshell.api.run("spatial.assign_container", model, products=[wall], relating_structure=site)
    representation = ifcopenshell.api.run(
        "geometry.add_wall_representation", model, context=body, length=5.0, height=3.0, thickness=0.3
    )
    ifcopenshell.api.run("geometry.assign_representation", model, product=wall, representation=representation)
    ifcopenshell.api.run("pset.add_pset", model, product=wall, name="Pset_WallCommon")
    model.write(str(path))


def _project(name: str, with_glb: bool = True) -> tuple[Path, dict[str, Path]]:
    project = _workdir(name)
    env = project / "products" / "environment"
    env.mkdir(parents=True)
    _write_ifc(env / "outdoor.ifc")
    (env / "outdoor.dxf").write_text("0\nEOF\n", encoding="utf-8")
    exports = {"ifc": env / "outdoor.ifc", "dxf": env / "outdoor.dxf"}
    if with_glb:
        (env / "environment.glb").write_bytes(b"glTFfake")
        exports["glb"] = env / "environment.glb"
    return project, exports


def test_publish_creates_app_layout() -> None:
    project, exports = _project("layout")
    published = publish_environment_exports(project, exports)
    export_dir = project / "products" / "dxf" / "export"
    assert list(export_dir.glob("*Site*clean.dxf")) == [published["dxf"]]
    assert (project / "products" / "bim" / "site" / "model.glb").is_file()
    ifc = ui_data.find_ifc(project, "site")
    assert ifc is not None and ifc.name == f"{project.name}_site.ifc"
    data = ui_data.props(project, "site")
    assert data["ifc_file"] == ifc.name and data["elements"]
    assert any(info["ifc_class"] == "IfcWall" for info in data["elements"].values())
    last = json.loads((export_dir / "last_export.json").read_text(encoding="utf-8"))
    assert last["coordinates"] == "LV95" and last["files"] and last["audit"] == "passed"
    manifest = json.loads((project / "products" / "bim" / "site" / "export.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "environment" and manifest["ifc_sha256"]


def test_environment_freshness_follows_ifc_hash() -> None:
    project, exports = _project("fresh")
    assert bim_jobs.ifc_export_status(project, "site") == "missing"
    published = publish_environment_exports(project, exports)
    assert bim_jobs.ifc_export_status(project, "site") == "fresh"
    assert ui_data.ifc_status(project, "site") == "fresh"
    published["ifc"].write_bytes(published["ifc"].read_bytes() + b"\n/* changed */")
    assert bim_jobs.ifc_export_status(project, "site") == "stale"


def test_missing_sources_skipped_and_sources_copied() -> None:
    project, exports = _project("skip", with_glb=False)
    published = publish_environment_exports(
        project, {**exports, "glb": project / "nope.glb", "laz": project / "nope.laz"}
    )
    assert "laz" not in published
    assert all(path.is_file() for path in exports.values())
    assert set(environment_exports(project)) == {"ifc", "dxf"}
    assert publish_environment_exports(project, {}) == {}


def test_publish_is_idempotent() -> None:
    project, exports = _project("idem")
    first = publish_environment_exports(project, exports)
    snapshot = sorted(str(p.relative_to(project)) for p in (project / "products").rglob("*") if p.is_file())
    second = publish_environment_exports(project, exports)
    assert first == second
    assert snapshot == sorted(str(p.relative_to(project)) for p in (project / "products").rglob("*") if p.is_file())
    assert bim_jobs.ifc_export_status(project, "site") == "fresh"


def test_indoor_freshness_unchanged() -> None:
    project = _workdir("indoor")
    level = "level_1"
    directory = bim_jobs.bim_output_dir(project, level)
    directory.mkdir(parents=True)
    for name in ("a.ifc", "model.glb", "props.json"):
        (directory / name).write_text("{}", encoding="utf-8")
    plan = bim_jobs.plan_model_file(project, level)
    plan.parent.mkdir(parents=True)
    plan.write_text('{"v": 1}', encoding="utf-8")
    manifest = {"ifc": "a.ifc", "glb": "model.glb", "props": "props.json", "plan_model_sha256": bim_jobs.plan_model_sha256(project, level)}
    (directory / "export.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert bim_jobs.ifc_export_status(project, level) == "fresh"
    plan.write_text('{"v": 2}', encoding="utf-8")
    assert bim_jobs.ifc_export_status(project, level) == "stale"
    plan.unlink()
    assert bim_jobs.ifc_export_status(project, level) == "stale"


def test_controller_outdoor_download_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    project, exports = _project("controller")
    las = project / "c.las"
    las.write_bytes(b"x")
    monkeypatch.setattr(DxfPageController, "blocked_message", property(lambda _self: None))
    monkeypatch.setattr(DxfPageController, "mode", property(lambda _self: "outdoor"))
    controller = DxfPageController(project, las_path=las)
    assert controller.outdoor_export_files() == {}
    assert controller.ifc_export_status(None) == "missing"
    published = publish_environment_exports(project, exports)
    files = controller.outdoor_export_files()
    assert files["dxf"].resolve() == published["dxf"].resolve() and files["ifc"].resolve() == published["ifc"].resolve()
    assert controller.ifc_export_status(None) == "fresh"
    monkeypatch.setattr(DxfPageController, "mode", property(lambda _self: "indoor"))
    assert controller.outdoor_export_files() == {}


def _nan_glb() -> bytes:
    """GLB with one good and one NaN mesh ('dsm-surface'), NaN also in the accessor bounds like environment.glb."""
    import numpy as np

    good = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    bad = np.array([[0, 0, 0], [float("nan")] * 3, [0, 1, 0]], dtype="<f4")
    binary = good.tobytes() + bad.tobytes()
    document = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 1]}],
        "nodes": [{"name": "good", "mesh": 0}, {"name": "dsm-surface", "mesh": 1}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}, {"primitives": [{"attributes": {"POSITION": 1}}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3", "min": [float("nan")] * 3, "max": [float("nan")] * 3},
        ],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}, {"buffer": 0, "byteOffset": 36, "byteLength": 36}],
        "buffers": [{"byteLength": len(binary)}],
    }
    raw = json.dumps(document).encode("utf-8")  # contains the literal NaN
    raw += b" " * ((-len(raw)) % 4)
    chunks = struct.pack("<II", len(raw), 0x4E4F534A) + raw + struct.pack("<II", len(binary), 0x004E4942) + binary
    return struct.pack("<III", 0x46546C67, 2, 12 + len(chunks)) + chunks


def test_model_glb_built_from_ifc_matches_props() -> None:
    project, exports = _project("from_ifc")
    exports["glb"].write_bytes(_nan_glb())  # environment.glb is NOT what gets published
    published = publish_environment_exports(project, exports)
    document, binary = _strict_loads_glb(published["glb"].read_bytes())
    assert binary and all(node["name" ] != "dsm-surface" for node in document["nodes"])
    data = ui_data.props(project, "site")
    guids = set(data["elements"])
    assert guids and {node["name"] for node in document["nodes"]} == guids
    for node in document["nodes"]:
        assert node["extras"]["ifc_class"] == data["elements"][node["name"]]["ifc_class"]


def test_nan_glb_is_never_published(monkeypatch: pytest.MonkeyPatch) -> None:
    from samcloud.drone import publish as publish_module

    project, exports = _project("nan")
    exports["glb"].write_bytes(_nan_glb())
    with pytest.raises(ValueError):
        _strict_loads_glb(exports["glb"].read_bytes())
    # force the fallback path: viewer assets from the IFC fail -> environment.glb is sanitised
    monkeypatch.setattr(publish_module, "_viewer_assets_from_ifc", lambda *_a, **_k: False)
    published = publish_environment_exports(project, exports)
    document, _binary = _strict_loads_glb(published["glb"].read_bytes())  # strict: raises on any NaN
    assert [node["name"] for node in document["nodes"]] == ["good"]
    assert document["scenes"][0]["nodes"] == [0]
    assert exports["glb"].is_file()
    assert _strict_loads_glb(sanitize_glb(_nan_glb()))[0]["nodes"]
