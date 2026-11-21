"""Potree-based point cloud viewer with the same controls as the canvas preview viewer."""

from __future__ import annotations

import html
import json
import logging
import re
import struct
from pathlib import Path

import numpy as np
from nicegui import ui

from samcloud.app.i18n import current_language, t, translate_or

_LIB = "/assets-static/potree"
LOGGER = logging.getLogger(__name__)

_HEAD = f"""
<link rel="stylesheet" type="text/css" href="{_LIB}/build/potree/potree.css">
<link rel="stylesheet" type="text/css" href="{_LIB}/libs/jquery-ui/jquery-ui.min.css">
<link rel="stylesheet" type="text/css" href="{_LIB}/libs/spectrum/spectrum.css">
<script src="{_LIB}/libs/jquery/jquery-3.1.1.min.js"></script>
<script src="{_LIB}/libs/spectrum/spectrum.js"></script>
<script src="{_LIB}/libs/jquery-ui/jquery-ui.min.js"></script>
<script src="{_LIB}/libs/other/BinaryHeap.js"></script>
<script src="{_LIB}/libs/tween/tween.min.js"></script>
<script src="{_LIB}/libs/d3/d3.js"></script>
<script src="{_LIB}/libs/proj4/proj4.js"></script>
<script src="{_LIB}/libs/i18next/i18next.js"></script>
<script src="{_LIB}/build/potree/potree.js"></script>
<style>
.sam-potree {{ position:relative; z-index:0; isolation:isolate; width:100%; height:calc(100vh - 150px); min-height:480px; overflow:hidden; border:1px solid #28475b; border-radius:9px; background:#050b12; }}
.sam-potree:fullscreen {{ width:100vw; height:100vh; border:0; border-radius:0; }}
.sam-potree-area {{ position:absolute; inset:0; }}
.sam-potree-toolbar {{ position:absolute; z-index:20; top:12px; left:12px; right:12px; display:flex; flex-wrap:wrap; gap:8px; align-items:center; pointer-events:none; }}
.sam-potree-toolbar > * {{ pointer-events:auto; }}
.sam-potree-toolbar button {{ border:1px solid #47718a; border-radius:5px; padding:6px 10px; background:#102131; color:#dcecf6; cursor:pointer; }}
.sam-potree-toolbar button.is-selected {{ border-color:#26b7ef; background:#16516c; }}
.sam-potree-toolbar label {{ display:flex; align-items:center; gap:6px; padding:5px 9px; border:1px solid #28475b; border-radius:5px; background:rgba(9,19,29,.86); color:#b9ccda; font-size:12px; }}
.sam-potree-clip-value {{ min-width:46px; text-align:right; font-variant-numeric:tabular-nums; color:#dcecf6; }}
.sam-potree-clip-value[data-clip-label="min"] {{ text-align:left; }}
.sam-potree-height {{ display:flex; align-items:center; gap:6px; padding:5px 9px; border:1px solid #28475b; border-radius:5px; background:rgba(9,19,29,.86); color:#b9ccda; font-size:12px; }}
.sam-potree-range {{ position:relative; width:180px; height:18px; }}
.sam-potree-range::before {{ content:""; position:absolute; left:0; right:0; top:7px; height:4px; border-radius:2px; background:#28475b; }}
.sam-potree-range-fill {{ position:absolute; top:7px; height:4px; border-radius:2px; background:#26b7ef; }}
.sam-potree-range input {{ position:absolute; left:0; top:0; width:100%; height:18px; margin:0; padding:0; background:none; pointer-events:none; -webkit-appearance:none; appearance:none; }}
.sam-potree-range input:focus {{ outline:none; }}
.sam-potree-range input::-webkit-slider-runnable-track {{ height:18px; background:transparent; border:0; }}
.sam-potree-range input::-moz-range-track {{ background:transparent; border:0; }}
.sam-potree-range input::-webkit-slider-thumb {{ -webkit-appearance:none; appearance:none; pointer-events:auto; width:14px; height:14px; margin-top:2px; border-radius:50%; background:#dcecf6; border:2px solid #26b7ef; cursor:ew-resize; }}
.sam-potree-range input::-moz-range-thumb {{ pointer-events:auto; width:10px; height:10px; border-radius:50%; background:#dcecf6; border:2px solid #26b7ef; cursor:ew-resize; }}
.sam-potree-status {{ position:absolute; z-index:20; left:12px; bottom:12px; padding:6px 9px; border-radius:5px; background:rgba(9,19,29,.86); color:#b9ccda; font-size:12px; }}
.sam-potree-classes {{ position:absolute; z-index:20; top:64px; right:12px; max-height:calc(100% - 90px); overflow:auto; width:min(300px,32vw); padding:10px; border:1px solid #365c74; border-radius:8px; background:rgba(9,19,29,.94); color:#eaf2f8; font-size:12px; display:none; }}
.sam-potree-classes.is-open {{ display:block; }}
.sam-potree-classes label {{ display:flex; align-items:center; gap:7px; padding:3px 0; cursor:pointer; }}
.sam-potree-classes i {{ width:12px; height:12px; border-radius:2px; display:inline-block; }}
.sam-potree .annotation-titlebar {{ font-size:14px; font-weight:700; padding:3px 9px; border:2px solid #ffffff; border-radius:6px; color:#06202f; box-shadow:0 0 0 2px rgba(0,0,0,.55); }}
.sam-potree .annotation {{ opacity:1 !important; }}
.sam-potree .annotation-titlebar {{ color:#06202f !important; }}
.sam-potree .annotation.is-hidden {{ display:none !important; }}
.sam-potree-camera-card {{ position:absolute; z-index:25; right:12px; bottom:12px; width:min(320px,42vw); padding:10px; border:1px solid #47718a; border-radius:7px; background:rgba(9,19,29,.94); color:#eaf2f8; font-size:12px; display:none; }}
.sam-potree-camera-card.is-open {{ display:block; }}
.sam-potree-camera-card img {{ display:block; width:100%; max-height:210px; margin-top:8px; object-fit:contain; background:#050b12; cursor:zoom-in; }}
.sam-potree-camera-card a {{ color:#59c8f3; display:inline-block; margin-top:7px; }}
.sam-potree-point-card {{ position:absolute; z-index:26; min-width:150px; max-width:260px; padding:9px 10px; border:1px solid #47718a; border-radius:7px; background:rgba(9,19,29,.96); color:#eaf2f8; font-size:12px; display:none; pointer-events:none; }}
.sam-potree-point-card.is-open {{ display:block; }}
.sam-potree-point-card strong {{ display:block; margin-bottom:3px; font-size:14px; }}
</style>
"""


# ----------------------------------------------------------------------------------------
# Outdoor product layers (environment.glb meshes next to the point cloud)
# ----------------------------------------------------------------------------------------

ENVIRONMENT_GLB_URL = "/api/environment/environment.glb"
_ENVIRONMENT_GLB_REL = Path("products") / "environment" / "environment.glb"
_IFC_VIEWER_REL = Path("products") / "environment" / "viewer"

# Layer key -> (German label, node group it controls; None = not mesh based).
LAYER_DEFINITIONS: dict[str, tuple[str, str | None]] = {
    "rgb": ("RGB", None),
    "classes": ("Klassen", None),
    "dgm": ("DGM", "terrain"),
    "dom": ("DOM", "dsm"),
    "buildings": ("Gebäude", "building"),
    "vegetation": ("Vegetation", "vegetation"),
    "streets": ("Strassen/Beläge", "pavement"),
    "model": ("Gesamtmodell", None),
}
GROUP_LABELS = {
    "terrain": "DGM",
    "dsm": "DOM",
    "building": "Gebäude",
    "vegetation": "Vegetation",
    "pavement": "Strassen/Beläge",
    "other": "Sonstiges",
}
_NOT_COMPUTED = "nicht berechnet"
# Name tokens (prefix match) that put a GLB node into a group; checked in this order.
_GROUP_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dsm", ("dsm", "dom", "surface", "oberflaeche")),
    ("terrain", ("terrain", "dgm", "ground", "boden", "gelaende", "dtm")),
    ("building", ("building", "gebaeude", "haus", "house", "roof", "dach", "facade", "fassade")),
    ("vegetation", ("veg", "tree", "baum", "bush", "strauch", "hedge", "hecke", "grass", "wiese", "forest", "wald")),
    ("pavement", ("street", "road", "strasse", "pave", "belag", "sidewalk", "trottoir", "curb", "kerb", "parking", "path", "weg", "asphalt")),
)
_IFC_TO_GROUP = {"building": "building", "vegetation": "vegetation", "terrain": "terrain", "pavement": "pavement"}
_ENVIRONMENT_ROUTES_REGISTERED = False


def read_glb_document(path: Path | str) -> dict:
    """Parse the JSON chunk of a GLB file (raises ValueError when it is not a valid GLB)."""
    data = Path(path).read_bytes()
    if len(data) < 20 or struct.unpack("<I", data[:4])[0] != 0x46546C67:
        raise ValueError("not a GLB file")
    length, kind = struct.unpack("<II", data[12:20])
    if kind != 0x4E4F534A or 20 + length > len(data):
        raise ValueError("GLB has no JSON chunk")
    return json.loads(data[20:20 + length].decode("utf-8"))


def node_group(name: str, extras: dict | None = None) -> str:
    """Layer group of one GLB node: ``extras.category``/``extras.ifc_class`` first, then the node name."""
    extras = extras or {}
    category = str(extras.get("category") or "")
    if category in GROUP_LABELS:
        return category
    if extras.get("ifc_class"):
        from samcloud.bim.viewer_export import category_for_ifc_class

        mapped = _IFC_TO_GROUP.get(category or category_for_ifc_class(str(extras["ifc_class"])))
        if mapped:
            return mapped
    tokens = [token for token in re.split(r"[^a-z0-9]+", str(name).lower()) if token]
    for group, prefixes in _GROUP_TOKENS:
        if any(token.startswith(prefix) for token in tokens for prefix in prefixes):
            return group
    return "other"


def _glb_nodes(document: dict) -> list[dict]:
    """Named nodes of a GLB document that carry a mesh, with their group."""
    result: list[dict] = []
    for index, node in enumerate(document.get("nodes", [])):
        if "mesh" not in node:
            continue
        name = str(node.get("name") or f"node_{index}")
        result.append({"name": name, "group": node_group(name, node.get("extras"))})
    return result


def _scene_origin(document: dict) -> list[float] | None:
    """``origin_lv95`` from the glTF scene extras (trimesh) or asset extras."""
    candidates = [scene.get("extras") for scene in document.get("scenes", [])]
    candidates += [document.get("extras"), (document.get("asset") or {}).get("extras")]
    for extras in candidates:
        origin = (extras or {}).get("origin_lv95")
        if isinstance(origin, (list, tuple)) and len(origin) == 3:
            return [float(value) for value in origin]
    return None


def environment_layers(project_dir: Path | str) -> dict:
    """List the outdoor products of a project for the viewer layer panel (file system only, no browser).

    Returns ``{"available", "glb": {available, path, url, origin_lv95, nodes, reason}, "groups": {group: [node names]},
    "rasters": {dgm, dom, hillshade}, "ifc_viewer": {available, glb_path, props_path}, "layers": {key: {label,
    group, available, reason}}}``. A missing or unreadable ``environment.glb`` marks every mesh layer unavailable.
    """
    project = Path(project_dir)
    env_dir = project / "products" / "environment"
    glb_path = project / _ENVIRONMENT_GLB_REL
    glb: dict = {"available": False, "path": None, "url": None, "origin_lv95": None, "nodes": [], "reason": _NOT_COMPUTED}
    groups: dict[str, list[str]] = {}
    if glb_path.is_file():
        try:
            document = read_glb_document(glb_path)
            nodes = _glb_nodes(document)
            if nodes:
                glb = {"available": True, "path": str(glb_path), "url": ENVIRONMENT_GLB_URL,
                       "origin_lv95": _scene_origin(document), "nodes": nodes, "reason": None}
                for node in nodes:
                    groups.setdefault(node["group"], []).append(node["name"])
            else:
                glb["reason"] = "environment.glb enthält keine Objekte"
        except (OSError, ValueError, json.JSONDecodeError) as error:
            LOGGER.warning("environment.glb unreadable: %s", error)
            glb["reason"] = "environment.glb nicht lesbar"

    def first(*patterns: str) -> str | None:
        for pattern in patterns:
            matches = sorted(env_dir.glob(pattern)) if env_dir.is_dir() else []
            if matches:
                return str(matches[0])
        return None

    rasters = {
        "dgm": first("*dgm*.tif", "*dtm*.tif", "*terrain*.tif"),
        "dom": first("*dom*.tif", "*dsm*.tif"),
        "hillshade": first("*hillshade*.png", "*hillshade*.tif"),
    }
    viewer_dir = project / _IFC_VIEWER_REL
    ifc_viewer = {
        "available": (viewer_dir / "model.glb").is_file() and (viewer_dir / "props.json").is_file(),
        "glb_path": str(viewer_dir / "model.glb"),
        "props_path": str(viewer_dir / "props.json"),
    }

    layers: dict[str, dict] = {}
    for key, (label, group) in LAYER_DEFINITIONS.items():
        if key in ("rgb", "classes"):
            available, reason = True, None
        elif key == "model":
            available, reason = bool(glb["available"]), (None if glb["available"] else _NOT_COMPUTED)
        else:
            available = bool(groups.get(group or ""))
            reason = None if available else _NOT_COMPUTED
        layers[key] = {"label": label, "group": group, "available": available, "reason": reason}
    return {"available": bool(glb["available"]), "glb": glb, "groups": groups, "rasters": rasters,
            "ifc_viewer": ifc_viewer, "layers": layers}


def local_to_viewer_matrix(project_dir: Path | str, origin_lv95: list[float] | tuple[float, ...]) -> list[float] | None:
    """Row-major 4x4 that maps the GLB's local frame to the Potree scene frame, or ``None`` without georeference.

    The Potree cloud is built from ``classified.ply`` and therefore lives in the raw COLMAP ENU frame, while the GLB
    holds ``local = LV95(E, N, H_LN02) - origin_lv95`` (rotated ~1 degree against ENU). The georeference maps ENU to
    LV95, so the matrix is its affine linearisation at the point whose LV95 position is ``origin_lv95`` (Newton
    iteration from ENU 0, then ``enu = x + J^-1 (lv95 - f(x))`` with ``lv95 = local + origin_lv95``).
    """
    try:
        from samcloud.drone import frames

        georef = frames.project_georeference(project_dir)
        target = np.asarray(origin_lv95, dtype=np.float64).reshape(3)
        x = np.zeros(3)
        for _ in range(3):
            jac = frames.local_jacobian(x, georef)
            x = x + np.linalg.solve(jac, target - frames.enu_to_lv95(x[None], georef)[0])
        jac = frames.local_jacobian(x, georef)
        inverse = np.linalg.inv(jac)
        matrix = np.eye(4)
        matrix[:3, :3] = inverse
        matrix[:3, 3] = x + inverse @ (target - frames.enu_to_lv95(x[None], georef)[0])
        return [float(value) for value in matrix.reshape(-1)]
    except Exception as error:  # no georeference/grids: the viewer then assumes cloud and meshes share a frame
        LOGGER.info("no local->viewer transform for the environment meshes: %s", error)
        return None


def register_environment_routes() -> None:
    """Serve ``environment.glb`` of the active project at ``ENVIRONMENT_GLB_URL`` (idempotent)."""
    global _ENVIRONMENT_ROUTES_REGISTERED
    if _ENVIRONMENT_ROUTES_REGISTERED:
        return
    from nicegui import app
    from starlette.responses import FileResponse, JSONResponse

    from samcloud.app.state import state

    @app.get(ENVIRONMENT_GLB_URL)
    def environment_glb():
        project = state.active_project_directory
        path = Path(project) / _ENVIRONMENT_GLB_REL if project is not None else None
        if path is None or not path.is_file():
            return JSONResponse({"error": _NOT_COMPUTED}, status_code=404)
        return FileResponse(path, media_type="model/gltf-binary", headers={"Cache-Control": "no-cache"})

    _ENVIRONMENT_ROUTES_REGISTERED = True


# ----------------------------------------------------------------------------------------
# Control point markers (AprilTags + surveyed Passpunkte/Kontrollpunkte) of the dense viewer
# ----------------------------------------------------------------------------------------

CONTROL_POINT_MARKERS_URL = "/api/viewer/control-point-markers"
_CONTROL_POINT_ROUTES_REGISTERED = False
_ROLE_KIND = {"pass": "passpunkt", "control": "kontrollpunkt"}


def _survey_files(project: Path) -> list[Path]:
    """control_points/control_points.csv, else every inputs/survey/*.txt|*.csv (sorted)."""
    primary = project / "control_points" / "control_points.csv"
    if primary.is_file():
        return [primary]
    survey = project / "inputs" / "survey"
    if not survey.is_dir():
        return []
    return sorted(path for path in survey.iterdir() if path.is_file() and path.suffix.lower() in {".txt", ".csv"})


def _read_survey_points(project: Path) -> list[tuple[str, np.ndarray, str]]:
    """Surveyed ``(name, [x, y, z] in the project CRS, role)``; unreadable files are skipped with a log entry."""
    from samcloud.control_points.repository import read_control_points

    points: list[tuple[str, np.ndarray, str]] = []
    seen: set[str] = set()
    for path in _survey_files(project):
        try:
            table = read_control_points(path)
        except Exception as error:  # malformed survey file: skip, never break the viewer
            LOGGER.warning("control points: cannot read %s: %s", path, error)
            continue
        for row in table.itertuples(index=False):
            name = str(row.name)
            xyz = np.array([row.x, row.y, row.z], dtype=np.float64)
            if name in seen or not np.all(np.isfinite(xyz)):
                continue
            seen.add(name)
            points.append((name, xyz, str(row.role)))
    return points


def _lv95_to_viewer(lv95_xyz: np.ndarray, georef: dict) -> np.ndarray:
    """Inverse of ``frames.enu_to_lv95`` (LV95/LN02 -> raw COLMAP ENU = Potree frame) by Newton iteration."""
    from samcloud.drone import frames

    target = np.asarray(lv95_xyz, dtype=np.float64).reshape(-1, 3)
    x = np.zeros_like(target)
    jac = frames.local_jacobian(np.zeros(3), georef)
    for step in range(8):
        residual = target - frames.enu_to_lv95(x, georef)
        x = x + np.linalg.solve(jac, residual.T).T
        if step == 1:
            jac = frames.local_jacobian(x.mean(axis=0), georef)
        if float(np.abs(residual).max()) < 1e-6:
            break
    return x


def _latest_checkpoints(project: Path) -> dict[str, dict]:
    """Per-point results of the newest ``reports/checkpoints_*.json`` (empty when absent or unreadable)."""
    reports = sorted((project / "reports").glob("checkpoints_*.json"), key=lambda path: path.stat().st_mtime) if (project / "reports").is_dir() else []
    if not reports:
        return {}
    try:
        data = json.loads(reports[-1].read_text(encoding="utf-8"))
        return {str(item["name"]): item for item in data.get("points", []) if isinstance(item, dict) and "name" in item}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        LOGGER.warning("control points: cannot read %s: %s", reports[-1], error)
        return {}


def _checkpoint_tooltip(name: str, result: dict | None) -> tuple[str, bool]:
    """Tooltip text ("PP1 · dH +0.212 m") and whether the point counts as not found/unusable."""
    if not result:
        return name, False
    status = str(result.get("status", ""))
    if status in {"ok", "implausible"} and isinstance(result.get("dH"), (int, float)):
        text = f"{name} · dH {float(result['dH']):+.3f} m"
        if status == "implausible":
            text += " · unplausibel"
        return text, status == "implausible"
    if status == "few_views":
        return f"{name} · zu wenige Ansichten", True
    return f"{name} · nicht gefunden", True


def _model_frame_points(project: Path) -> dict[str, list[float]]:
    """Triangulated model coordinates per name from detections.json (the viewer frame, no georeference needed)."""
    try:
        from samcloud.app.gcp_review import marker_payload

        return {str(item["name"]): [float(v) for v in item["coordinates"]]
                for item in marker_payload(project).get("markers", [])
                if isinstance(item.get("coordinates"), (list, tuple)) and len(item["coordinates"]) == 3}
    except Exception as error:
        LOGGER.info("control points: no model-frame detections: %s", error)
        return {}


def control_point_markers(project_dir: Path | str | None) -> list[dict]:
    """AprilTag and surveyed control point markers of a project, positioned in the dense viewer frame.

    AprilTags come unchanged from ``apriltag_marker_payload``. Surveyed points (``control_points.csv`` or
    ``inputs/survey``) are converted LV95/LN02 -> ENU with the project georeference; without one the triangulated
    model coordinates of ``detections.json`` are used. Each marker has ``name``, ``kind`` (apriltag | passpunkt |
    kontrollpunkt), ``coordinates``, ``state_color`` and ``tooltip`` (checkpoint dH/status when a report exists).
    Missing files or georeference give an empty list for that source and a log entry, never an exception.
    """
    if project_dir is None:
        return []
    project = Path(project_dir)
    markers: list[dict] = []
    try:
        from samcloud.app.gcp_review import apriltag_marker_payload

        for tag in apriltag_marker_payload(project):
            tag = dict(tag)
            tag.setdefault("tooltip", tag["name"])
            markers.append(tag)
    except Exception as error:
        LOGGER.warning("control points: AprilTag markers unavailable: %s", error)
    taken = {str(item["name"]) for item in markers}
    try:
        survey = [item for item in _read_survey_points(project) if item[0] not in taken]
    except Exception as error:
        LOGGER.warning("control points: survey unavailable: %s", error)
        survey = []
    if not survey:
        return markers
    positions: dict[str, list[float]] = {}
    try:
        from samcloud.drone import frames

        enu = _lv95_to_viewer(np.vstack([item[1] for item in survey]), frames.project_georeference(project))
        positions = {item[0]: [float(v) for v in row] for item, row in zip(survey, enu)}
    except Exception as error:
        LOGGER.info("control points: no georeference (%s); using triangulated model coordinates", error)
        positions = _model_frame_points(project)
    checkpoints = _latest_checkpoints(project)
    for name, _xyz, role in survey:
        coordinates = positions.get(name)
        if coordinates is None or not all(np.isfinite(coordinates)):
            continue
        kind = _ROLE_KIND.get(role, "passpunkt")
        tooltip, unusable = _checkpoint_tooltip(name, checkpoints.get(name))
        markers.append({"name": name, "kind": kind, "role": role, "enabled": True, "coordinates": coordinates,
                        "state_color": "disabled" if unusable else kind, "tooltip": tooltip})
    return markers


def register_control_point_routes() -> None:
    """Serve ``control_point_markers`` of the active project at ``CONTROL_POINT_MARKERS_URL`` (idempotent)."""
    global _CONTROL_POINT_ROUTES_REGISTERED
    if _CONTROL_POINT_ROUTES_REGISTERED:
        return
    from nicegui import app
    from starlette.responses import JSONResponse

    from samcloud.app.state import state

    @app.get(CONTROL_POINT_MARKERS_URL)
    def control_point_marker_endpoint():
        project = state.active_project_directory
        warnings: list[str] = []
        if project is not None:
            try:
                from samcloud.app.gcp_review import apriltag_warnings

                warnings = apriltag_warnings(Path(project))
            except Exception as error:
                LOGGER.info("control points: no AprilTag warnings: %s", error)
        return JSONResponse({"markers": control_point_markers(project), "warnings": warnings})

    _CONTROL_POINT_ROUTES_REGISTERED = True


def render_potree_viewer(
    kind: str,
    *,
    title: str,
    class_colors: list[list[int]],
    class_labels: list[str] | None = None,
    class_names_de: list[str] | None = None,
    instance_labels_de: dict[str, str] | None = None,
    fine_term_names_de: list[str] | None = None,
    marker_endpoint: str | None = None,
    camera_endpoint: str | None = None,
) -> None:
    """Render a Potree viewer for the active project's sparse or dense cloud."""
    markers_disabled = False
    if kind == "dense" and marker_endpoint:
        marker_endpoint, markers_disabled = _dense_marker_setup()
    config = _viewer_config(kind, title, class_colors, class_labels, class_names_de, instance_labels_de, marker_endpoint, camera_endpoint, fine_term_names_de)
    element_id = config["element_id"]
    ui.add_head_html(_HEAD)
    ui.html(
        _html(
            element_id,
            show_classes=config["show_classes"],
            show_markers=config["show_markers"],
            show_cameras=config["show_cameras"],
            markers_label=t("nav.control_points"),
            markers_disabled=markers_disabled,
        ),
        sanitize=False,
    ).classes("w-full")
    ui.add_body_html(_script(config))


def _dense_marker_setup() -> tuple[str, bool]:
    """Marker endpoint of the dense viewer and whether the button is disabled (no control points at all)."""
    try:
        from samcloud.app.state import state

        register_control_point_routes()
        return CONTROL_POINT_MARKERS_URL, not control_point_markers(state.active_project_directory)
    except Exception as error:  # the markers are optional and must never break the viewer
        LOGGER.warning("control point markers unavailable: %s", error)
        return CONTROL_POINT_MARKERS_URL, True


def _viewer_config(
    kind: str,
    title: str,
    class_colors: list[list[int]],
    class_labels: list[str] | None = None,
    class_names_de: list[str] | None = None,
    instance_labels_de: dict[str, str] | None = None,
    marker_endpoint: str | None = None,
    camera_endpoint: str | None = None,
    fine_term_names_de: list[str] | None = None,
) -> dict:
    """Build the browser configuration for one Potree cloud."""
    language = current_language()
    labels = class_labels or []
    names_de = class_names_de or []
    terms_de = fine_term_names_de or []
    instance_names = instance_labels_de or {}
    return {
        "element_id": f"sam-potree-{kind}",
        "kind": kind,
        "title": title,
        "metadata_url": f"/api/potree/{kind}/metadata.json",
        "status_url": f"/api/potree/{kind}/status",
        "class_colors": class_colors,
        "class_labels": class_labels or [],
        "class_names_de": class_names_de or [],
        "instance_labels_de": instance_labels_de or {},
        "fine_term_names_de": fine_term_names_de or [],
        "language": language,
        "class_display_labels": [_class_display_label(label) for label in labels],
        "class_display_names": _class_display_names(labels, names_de, language),
        "fine_term_display_names": _fine_term_display_names(terms_de, instance_names, language),
        "enable_point_info": kind == "dense",
        "marker_endpoint": marker_endpoint,
        "camera_endpoint": camera_endpoint,
        "show_classes": kind == "dense",
        "show_markers": bool(marker_endpoint),
        "show_cameras": bool(camera_endpoint),
    }


def _split_class_label(label: str) -> tuple[str, str]:
    """Split an optional "<LAS code>: " prefix from a class label (e.g. "2: Ground" -> ("2: ", "Ground"))."""
    prefix, separator, name = label.partition(": ")
    if separator and prefix.strip().isdigit():
        return prefix + separator, name
    return "", label


def _class_display_name(name: str) -> str:
    """Translated display name of one class; unknown names are shown unchanged."""
    text = str(name)
    translated = translate_or(f"classes.{text}", "")
    if not translated:
        translated = translate_or(f"classes.{text.strip().lower()}", "")
    return translated or text


def _class_display_label(label: str) -> str:
    """Display label for the class legend; keeps the LAS code prefix and translates the class name."""
    prefix, name = _split_class_label(str(label))
    return prefix + _class_display_name(name)


def _class_display_names(labels: list[str], names_de: list[str], language: str) -> list[str]:
    """Class names for the point info card in the UI language (German keeps the configured German names)."""
    names: list[str] = []
    for index in range(max(len(labels), len(names_de))):
        label_name = _split_class_label(str(labels[index]))[1] if index < len(labels) else ""
        german = str(names_de[index]) if index < len(names_de) and names_de[index] else ""
        if language == "de":
            names.append(german or (_class_display_name(label_name) if label_name else ""))
        else:
            names.append(_class_display_name(label_name) if label_name else german)
    return names


def _fine_term_display_names(names_de: list[str], instance_labels_de: dict[str, str], language: str) -> list[str]:
    """Fine-term names for the point info card; non-German UIs map the German names back to the English terms."""
    if language == "de":
        return [str(name) for name in names_de]
    english_by_german: dict[str, str] = {}
    for english, german in instance_labels_de.items():
        english_by_german.setdefault(str(german), str(english))
    return [english_by_german.get(str(name), str(name)) if name else "" for name in names_de]


# Status/error texts returned by samcloud.viewer through the API. They stay English on the wire (the
# scripts compare against them); the viewers translate them only for display.
_SERVER_MESSAGE_KEYS = {
    "Potree or PotreeConverter is not installed": "viewer.server.not_installed",
    "Point cloud has not been prepared yet": "viewer.server.not_prepared",
    "Preparing point cloud": "viewer.server.preparing",
    "No sparse point cloud is available yet": "viewer.server.no_sparse",
    "No dense point cloud is available yet": "viewer.server.no_dense",
    "Unsupported point cloud kind": "viewer.server.unsupported_kind",
}
_SERVER_MESSAGE_PREFIX_KEYS = {
    "PotreeConverter failed: ": "viewer.server.converter_failed",
}


def server_message_strings() -> dict[str, dict[str, str]]:
    """Display texts of known server messages: exact raw text, and raw prefixes whose rest fills {details}."""
    return {
        "server_messages": {raw: t(key) for raw, key in _SERVER_MESSAGE_KEYS.items()},
        "server_prefixes": {raw: t(key) for raw, key in _SERVER_MESSAGE_PREFIX_KEYS.items()},
    }


def _viewer_strings() -> dict[str, object]:
    """Translated texts used by the viewer script (placeholders are filled in the browser)."""
    return {
        **server_message_strings(),
        "class_fallback": t("viewer.classes.fallback"),
        "class_other": t("viewer.classes.other"),
        "unavailable": t("viewer.status.unavailable"),
        "prepare_failed": t("viewer.status.prepare_failed"),
        "preparing_elapsed": t("viewer.status.preparing_elapsed"),
        "ready_count": t("viewer.status.ready_count"),
        "ready": t("viewer.status.ready"),
        "retry": t("common.retry"),
        "points": t("common.points"),
        "number_locale": t("viewer.number_locale"),
        "marker_open_review": t("viewer.markers.open_review"),
        "camera_fallback_title": t("viewer.camera.fallback_title"),
        "camera_details": t("viewer.camera.details"),
        "camera_photo_alt": t("viewer.camera.photo_alt"),
        "camera_photo_open": t("viewer.camera.photo_open"),
        "camera_show_photo": t("viewer.camera.show_photo"),
        "tag_size": t("viewer.tag.size"),
        "tag_seen_in": t("viewer.tag.seen_in"),
        "tag_error": t("viewer.tag.error"),
        "tag_outlier": t("viewer.tag.outlier"),
        "tag_ok": t("viewer.tag.ok"),
        "tag_no_check": t("viewer.tag.no_check"),
    }


def _html(
    element_id: str,
    *,
    show_classes: bool = True,
    show_markers: bool = False,
    show_cameras: bool = False,
    markers_label: str | None = None,
    markers_disabled: bool = False,
) -> str:
    def esc(value: str) -> str:
        return html.escape(value, quote=True)

    if markers_label is None:
        markers_label = t("nav.control_points")
    if markers_disabled:
        markers_html = f'<button data-action="markers" disabled title="{esc(t("viewer.markers.none"))}" style="opacity:.5;cursor:not-allowed">{esc(markers_label)}</button>'
    else:
        markers_html = f'<button data-action="markers" class="is-selected">{esc(markers_label)}</button>'
    markers_btn = f"    {markers_html}\n" if show_markers else ""
    classes_btns = (
        '    <button data-action="rgb" class="is-selected">RGB</button>\n'
        f'    <button data-action="classification">{esc(t("viewer.toolbar.classification"))}</button>\n'
        f'    <button data-action="classes">{esc(t("viewer.toolbar.class_toggle"))}</button>\n'
        if show_classes
        else ""
    )
    cameras_btns = (
        f'    <button data-action="cameras" class="is-selected">{esc(t("viewer.toolbar.cameras"))}</button>\n'
        f'    <button data-action="camera-colour">{esc(t("viewer.toolbar.camera_colour"))}</button>\n'
        f'    <button data-action="camera-sphere-texture" style="display:none">{esc(t("viewer.toolbar.camera_sphere_texture"))}</button>\n'
        if show_cameras
        else ""
    )
    camera_sliders = (
        f'    <label>{esc(t("viewer.toolbar.camera_size"))} <input data-action="camera-size" type="range" min="0.25" max="0.75" step="0.25" value="0.5"></label>\n'
        if show_cameras
        else ""
    )
    return f"""
<div id="{element_id}" class="sam-potree">
  <div class="sam-potree-area"></div>
  <div class="sam-potree-toolbar">
    <button data-action="fit">{esc(t("viewer.toolbar.fit"))}</button>
{markers_btn}{cameras_btns}{classes_btns}    <button data-action="earth" class="is-selected">{esc(t("viewer.toolbar.terrain"))}</button>
    <button data-action="orbit">{esc(t("viewer.toolbar.orbit"))}</button>
    <button data-action="edl" class="is-selected">{esc(t("viewer.toolbar.shading"))}</button>
    <button data-action="fullscreen">{esc(t("viewer.toolbar.fullscreen"))}</button>
    <label>{esc(t("viewer.toolbar.point_size"))} <input data-action="size" type="range" min="0.5" max="6" step="0.25" value="1.5"></label>
    <div class="sam-potree-height" title="{esc(t("viewer.toolbar.height_tooltip"))}">{esc(t("viewer.toolbar.height"))} <span class="sam-potree-clip-value" data-clip-label="min">–</span><span class="sam-potree-range"><span class="sam-potree-range-fill"></span><input data-action="clip-min" type="range" min="0" max="1000" step="1" value="0" aria-label="{esc(t("viewer.toolbar.height_lower"))}"><input data-action="clip-max" type="range" min="0" max="1000" step="1" value="1000" aria-label="{esc(t("viewer.toolbar.height_upper"))}"></span><span class="sam-potree-clip-value" data-clip-label="max">–</span></div>
{camera_sliders}
  </div>
  <div class="sam-potree-classes"></div>
  <aside class="sam-potree-camera-card"></aside>
  <aside class="sam-potree-point-card"></aside>
  <div class="sam-potree-status">{esc(t("viewer.status.preparing"))}</div>
</div>
"""


def _script(config: dict) -> str:
    return """
<script type="module">
import * as THREE from "/assets-static/potree/libs/three.js/build/three.module.js";
const config = __CONFIG__;
const viewerText = __STRINGS__;
const tr = (key, params) => String(viewerText[key] ?? key).replace(/\\{(\\w+)\\}/g, (match, name) => (params && params[name] !== undefined && params[name] !== null ? String(params[name]) : match));
const numberLocale = viewerText.number_locale || "en-US";
const serverText = message => {
  const raw = String(message ?? "");
  const exact = (viewerText.server_messages || {})[raw];
  if (exact) return exact;
  for (const [prefix, template] of Object.entries(viewerText.server_prefixes || {})) {
    if (raw.startsWith(prefix)) return String(template).split("{details}").join(raw.slice(prefix.length));
  }
  return raw;
};
const start = () => {
  const root = document.getElementById(config.element_id);
  if (!root || root.dataset.ready === "1" || typeof Potree === "undefined") { setTimeout(start, 200); return; }
  root.dataset.ready = "1";
  const status = root.querySelector(".sam-potree-status");
  const viewer = new Potree.Viewer(root.querySelector(".sam-potree-area"));
  viewer.setEDLEnabled(true);
  viewer.setFOV(60);
  viewer.setPointBudget(3000000);
  viewer.setBackground("black");
  // Left drag rotates around the point under the cursor, right drag pans (same as the previous SAMcloud viewer).
  const earth = viewer.earthControls;
  const swapButtons = handler => event => {
    const drag = event.drag;
    if (!drag) return handler(event);
    const original = drag.mouse;
    if (original === Potree.MOUSE.LEFT) drag.mouse = Potree.MOUSE.RIGHT;
    else if (original === Potree.MOUSE.RIGHT) drag.mouse = Potree.MOUSE.LEFT;
    try { return handler(event); } finally { drag.mouse = original; }
  };
  for (const type of ["drag", "drop"]) {
    earth._listeners[type] = (earth._listeners[type] || []).map(swapButtons);
  }
  viewer.setControls(earth);
  const scheme = {};
  config.class_colors.forEach((colour, index) => {
    scheme[index] = { visible: true, name: (config.class_display_labels || [])[index] || config.class_labels[index] || tr("class_fallback", { index }), color: [colour[0] / 255, colour[1] / 255, colour[2] / 255, 1.0] };
  });
  scheme.DEFAULT = { visible: true, name: tr("class_other"), color: [0.7, 0.7, 0.7, 1.0] };
  viewer.setClassifications(scheme);
  const panel = root.querySelector(".sam-potree-classes");
  panel.innerHTML = Object.keys(scheme).filter(key => key !== "DEFAULT").map(key => {
    const c = scheme[key].color.map(v => Math.round(v * 255));
    return `<label><input type="checkbox" data-class="${key}" checked><i style="background:rgb(${c[0]},${c[1]},${c[2]})"></i>${scheme[key].name}</label>`;
  }).join("");
  panel.addEventListener("change", event => {
    const key = event.target.dataset.class;
    if (key === undefined) return;
    scheme[key].visible = event.target.checked;
    viewer.setClassifications(scheme);
  });
   let cloud = null;
   const cameraSettingsKey = "samcloud:sparse-camera-settings:v1";
   const savedCameraSettings = (() => { try { return JSON.parse(localStorage.getItem(cameraSettingsKey) || "{}"); } catch (_error) { return {}; } })();
  const select = (actions, active) => actions.forEach(name => { const button = root.querySelector(`[data-action="${name}"]`); if (button) button.classList.toggle("is-selected", name === active); });
  root.querySelector(".sam-potree-toolbar").addEventListener("click", event => {
    const action = event.target.dataset && event.target.dataset.action;
    if (!action) return;
    if (action === "fit") {
      try {
        const box = viewer.getBoundingBox(viewer.scene.pointclouds).clone();
        if (markersVisible) for (const item of markerObjects) box.expandByPoint(item.sphere.position);
        const node = new THREE.Object3D();
        node.boundingBox = box;
        viewer.zoomTo(node, 0.55, 0);
      } catch (_error) { viewer.fitToScreen(0.55); }
    }
    if (action === "markers") {
      markersVisible = !markersVisible;
      for (const item of markerObjects) {
        item.sphere.visible = markersVisible;
        if (item.square) item.square.visible = markersVisible;
        const element = item.annotation.domElement && item.annotation.domElement[0];
        if (element) element.classList.toggle("is-hidden", !markersVisible);
      }
      event.target.classList.toggle("is-selected", markersVisible);
    }
    if (action === "cameras") {
      camerasVisible = !camerasVisible;
      syncCameraSettings();
      saveCameraSettings();
    }
    if (action === "camera-sphere-texture") {
      cameraSphereTextures = !cameraSphereTextures;
      event.target.classList.toggle("is-selected", cameraSphereTextures);
      saveCameraSettings();
      syncSphereTextures();
      return;
    }
    if (action === "camera-colour") {
      cameraErrorColours = !cameraErrorColours;
      updateCameraColours();
      event.target.classList.toggle("is-selected", cameraErrorColours);
    }
    if ((action === "rgb" || action === "classification") && cloud) {
      cloud.material.activeAttributeName = action === "rgb" ? "rgba" : "classification";
      select(["rgb", "classification"], action);
    }
    if (action === "classes") panel.classList.toggle("is-open");
    if (action === "earth") { viewer.setControls(viewer.earthControls); select(["earth", "orbit"], "earth"); }
    if (action === "orbit") { viewer.setControls(viewer.orbitControls); select(["earth", "orbit"], "orbit"); }
    if (action === "edl") { viewer.setEDLEnabled(!viewer.getEDLEnabled()); event.target.classList.toggle("is-selected", viewer.getEDLEnabled()); }
    if (action === "fullscreen") { document.fullscreenElement ? document.exitFullscreen() : root.requestFullscreen(); }
   });
   root.querySelector('[data-action="size"]').addEventListener("input", event => { if (cloud) cloud.material.size = Number(event.target.value); });
   /* Höhenfilter: ein Regler mit zwei Knöpfen (unten/oben) von tiefstem bis höchstem Punkt; alles ausserhalb wird über eine
      unsichtbare Potree-Clipbox ausgeblendet. Die Box wird nur in scene.volumes geführt (nicht in
      der three.js-Szene), damit sie weder sichtbar noch per Maus verschiebbar ist. */
   const clipMin = root.querySelector('[data-action="clip-min"]');
   const clipMax = root.querySelector('[data-action="clip-max"]');
   const clipFill = root.querySelector('.sam-potree-range-fill');
   let heightBounds = null;
   let clipVolume = null;
   const sliderHeight = value => heightBounds.min[2] + (heightBounds.max[2] - heightBounds.min[2]) * Number(value) / 1000;
   const applyHeightClip = changed => {
     if (!heightBounds) return;
     let low = Number(clipMin.value);
     let high = Number(clipMax.value);
     /* Die Knöpfe dürfen sich nicht überholen: der bewegte Knopf bleibt am anderen stehen. */
     if (low > high) {
       if (changed === clipMax) { clipMax.value = low; high = low; } else { clipMin.value = high; low = high; }
     }
     clipFill.style.left = `calc(7px + (100% - 14px) * ${low / 1000})`;
     clipFill.style.width = `calc((100% - 14px) * ${(high - low) / 1000})`;
     /* Liegen beide Knöpfe ganz rechts, muss der untere greifbar bleiben. */
     clipMin.style.zIndex = low > 990 ? "2" : "1";
     clipMax.style.zIndex = low > 990 ? "1" : "2";
     const zLow = sliderHeight(low);
     const zHigh = sliderHeight(high);
     root.querySelector('[data-clip-label="min"]').textContent = zLow.toFixed(2);
     root.querySelector('[data-clip-label="max"]').textContent = zHigh.toFixed(2);
     const active = low > 0 || high < 1000;
     if (!clipVolume) {
       clipVolume = new Potree.BoxVolume();
       clipVolume.visible = false;
       viewer.scene.volumes.push(clipVolume);
     }
     const [minX, minY] = heightBounds.min;
     const [maxX, maxY] = heightBounds.max;
     const margin = Math.max(maxX - minX, maxY - minY, 1);
     clipVolume.position.set((minX + maxX) / 2, (minY + maxY) / 2, (zLow + zHigh) / 2);
     clipVolume.scale.set(maxX - minX + margin, maxY - minY + margin, Math.max(zHigh - zLow, 1e-4));
     clipVolume.updateMatrixWorld();
     clipVolume.clip = active;
     viewer.setClipTask(active ? Potree.ClipTask.SHOW_INSIDE : Potree.ClipTask.NONE);
   };
   const initHeightClip = metadata => {
     const position = (metadata.attributes || []).find(attribute => attribute.name === "position");
     const bounds = position && Array.isArray(position.min) && Array.isArray(position.max) ? position : metadata.boundingBox;
     if (!bounds || !Array.isArray(bounds.min) || !Array.isArray(bounds.max)) return;
     heightBounds = { min: bounds.min.map(Number), max: bounds.max.map(Number) };
     applyHeightClip(null);
   };
   [clipMin, clipMax].forEach(control => control.addEventListener("input", event => applyHeightClip(event.target)));
   [["camera-size", "cameraSize"]].forEach(([action, key]) => {
     const control = root.querySelector(`[data-action="${action}"]`);
     if (control) control.addEventListener("input", event => {
       if (key === "cameraSize") cameraSize = Number(event.target.value);
       if (key === "cameraPathWidth") cameraPathWidth = Number(event.target.value);
       if (key === "cameraPhotoSize") cameraPhotoSize = Number(event.target.value);
       if (cameraPath) cameraPath.material.linewidth = cameraPathWidth;
       saveCameraSettings();
       syncCameraBillboards();
     });
   });
  const pageStarted = Date.now();
  let loadAttempts = 0;
  const formatElapsed = seconds => {
    const s = Math.max(0, Math.floor(seconds));
    return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  };
  const showError = message => {
    status.innerHTML = "";
    status.append(`${tr("unavailable", { message })} `);
    const retry = document.createElement("button");
    retry.type = "button";
    retry.textContent = tr("retry");
    retry.className = "sam-potree-retry";
    retry.style.cssText = "margin-left:8px;padding:2px 8px;border-radius:4px;border:1px solid #3a5a74;background:#16324a;color:#e6f1f8;cursor:pointer;";
    retry.addEventListener("click", () => { loadAttempts = 0; pollStatus(); });
    status.append(retry);
  };
  const loadCloud = () => {
    Potree.loadPointCloud(config.metadata_url, config.kind).then(event => {
      cloud = event.pointcloud;
      viewer.scene.addPointCloud(cloud);
      cloud.material.size = 1.5;
      cloud.material.pointSizeType = Potree.PointSizeType.ADAPTIVE;
      cloud.material.shape = Potree.PointShape.CIRCLE;
      cloud.material.activeAttributeName = "rgba";
      viewer.fitToScreen(0.55);
      setTimeout(() => viewer.fitToScreen(0.55), 1500);
      fetch(config.metadata_url, { cache: "no-store" }).then(response => response.json()).then(metadata => {
        initHeightClip(metadata);
        const count = Number(metadata.points || 0).toLocaleString(numberLocale);
        status.textContent = tr("ready_count", { title: config.title, count });
      }).catch(() => { status.textContent = tr("ready", { title: config.title }); });
      loadMarkers();
      loadCameras();
    }).catch(error => {
      loadAttempts += 1;
      if (loadAttempts < 3) { setTimeout(pollStatus, 2000); return; }
      showError(error && error.message ? error.message : String(error));
    });
  };
  const pollStatus = async () => {
    let state;
    try {
      const response = await fetch(config.status_url, { cache: "no-store" });
      state = await response.json();
    } catch (error) {
      setTimeout(pollStatus, 2000);
      return;
    }
    if (state.state === "ready") { loadCloud(); return; }
    if (state.state === "error") { showError(serverText(state.message) || tr("prepare_failed")); return; }
    if (state.state === "missing" && state.message && !String(state.message).includes("not been prepared")) {
      status.textContent = tr("unavailable", { message: serverText(state.message) });
      return;
    }
    const startedMs = state.started_at ? state.started_at * 1000 : pageStarted;
    status.textContent = tr("preparing_elapsed", { elapsed: formatElapsed((Date.now() - startedMs) / 1000) });
    setTimeout(pollStatus, 2000);
  };
  pollStatus();
  const colours = { pass: "#26b7ef", control: "#ffb020", disabled: "#6b7c8a", needs_review: "#ff5a5a", apriltag: "#b06cff", passpunkt: "#ff9f1a", kontrollpunkt: "#18d6e6" };
  let markerWarnings = [];
  let highlightedTag = null;
  const markerObjects = [];
  let markersVisible = true;
   const cameraCard = root.querySelector(".sam-potree-camera-card");
   const cameraLayer = new THREE.Group();
   const cameraMarkerLayer = new THREE.Group();
   const cameraPhotoLayer = new THREE.Group();
   const cameraObjects = [];
   const cameraPhotos = new Map();
   const cameraPhotoLoads = new Set();
   const maxCameraPhotos = 72;
   let cameraPath = null;
   let camerasVisible = savedCameraSettings.positions !== false;
   let cameraPathVisible = false; /* Kamerapfad entfernt */
   let cameraPhotosVisible = true; /* Foto-Ansicht (Info-Karte) immer verfügbar */
   let cameraSize = Math.min(.75, Math.max(.25, Math.round((Number(savedCameraSettings.size) || .5) * 4) / 4));
   let cameraPathWidth = Number(savedCameraSettings.pathWidth) || 1.5;
   let cameraPhotoSize = Number(savedCameraSettings.photoSize) || 1;
   let cameraErrorColours = false;
   let cameraPanorama = false;
   let cameraSphereTextures = savedCameraSettings.sphereTextures === true;
   const sphereTextures = new Map();
   const sphereTextureLoads = new Set();
   const maxSphereTextures = 48;
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
   cameraLayer.add(cameraMarkerLayer);
   cameraLayer.add(cameraPhotoLayer);
   viewer.scene.scene.add(cameraLayer);
   const saveCameraSettings = () => {
     try { localStorage.setItem(cameraSettingsKey, JSON.stringify({ positions: camerasVisible, path: cameraPathVisible, photos: cameraPhotosVisible, size: cameraSize, pathWidth: cameraPathWidth, photoSize: cameraPhotoSize, sphereTextures: cameraSphereTextures })); } catch (_error) {}
   };
   const syncCameraSettings = () => {
     cameraMarkerLayer.visible = camerasVisible;
     cameraPhotoLayer.visible = cameraPhotosVisible;
     if (cameraPath) cameraPath.visible = cameraPathVisible;
     [["cameras", camerasVisible], ["camera-path", cameraPathVisible], ["camera-photos", cameraPhotosVisible]].forEach(([action, selected]) => { const control = root.querySelector(`[data-action="${action}"]`); if (control) control.classList.toggle("is-selected", selected); });
     [["camera-size", cameraSize], ["camera-path-width", cameraPathWidth], ["camera-photo-size", cameraPhotoSize]].forEach(([action, value]) => { const control = root.querySelector(`[data-action="${action}"]`); if (control) control.value = value; });
   };
   const textureLoader = new THREE.TextureLoader();
   const syncCameraBillboards = () => {};
   const syncSphereTextures = () => {
     if (!cameraPanorama || !cameraObjects.length) return;
     const activeCamera = viewer.scene.getActiveCamera();
     if (!cameraSphereTextures) {
       for (const item of cameraObjects) {
         if (item.marker.material.map) { item.marker.material.map = null; item.marker.material.needsUpdate = true; }
       }
       return;
     }
     const nearest = cameraObjects.map(item => ({ item, distance: activeCamera.position.distanceTo(item.marker.position) }))
       .sort((a, b) => a.distance - b.distance).slice(0, maxSphereTextures);
     const wanted = new Set(nearest.map(entry => entry.item.camera.id));
     for (const item of cameraObjects) {
       if (!wanted.has(item.camera.id) && item.marker.material.map) {
         const texture = sphereTextures.get(item.camera.id);
         item.marker.material.map = null; item.marker.material.needsUpdate = true;
         if (texture) { texture.dispose(); sphereTextures.delete(item.camera.id); }
       }
     }
     for (const { item } of nearest) {
       const id = item.camera.id;
       if (sphereTextures.has(id)) { if (item.marker.material.map !== sphereTextures.get(id)) { item.marker.material.map = sphereTextures.get(id); item.marker.material.color.set(0xffffff); item.marker.material.needsUpdate = true; } continue; }
       if (sphereTextureLoads.has(id)) continue;
       sphereTextureLoads.add(id);
       textureLoader.load(`/api/camera-thumbnail/${encodeURIComponent(id)}?size=640`, texture => {
         sphereTextureLoads.delete(id);
         if (!cameraSphereTextures) { texture.dispose(); return; }
         sphereTextures.set(id, texture);
         item.marker.material.map = texture; item.marker.material.color.set(0xffffff); item.marker.material.needsUpdate = true;
       }, undefined, () => sphereTextureLoads.delete(id));
     }
   };
   syncCameraSettings();
   viewer.addEventListener("update", () => {
     const camera = viewer.scene.getActiveCamera();
    for (const item of markerObjects) {
      const distance = camera.position.distanceTo(item.sphere.position);
      item.sphere.scale.setScalar(Math.max(0.5, distance * 0.012 / (item.radius || 0.18)));
    }
     for (const item of cameraObjects) {
       const distance = camera.position.distanceTo(item.marker.position);
       item.marker.scale.setScalar(Math.max(0.35 * cameraSize, distance * 0.012 * cameraSize));
     }
     syncCameraBillboards();
     syncSphereTextures();
   });
  const loadMarkers = async () => {
    if (!config.marker_endpoint) return;
    try {
      const response = await fetch(config.marker_endpoint, { cache: "no-store" });
      if (!response.ok) return;
      const payload = await response.json();
      markerWarnings = Array.isArray(payload.warnings) ? payload.warnings : [];
      for (const marker of payload.markers || []) {
        if (!Array.isArray(marker.coordinates) || marker.coordinates.length !== 3) continue;
        const annotation = new Potree.Annotation({ position: new THREE.Vector3(...marker.coordinates), title: marker.name });
        viewer.scene.annotations.add(annotation);
        const colour = colours[marker.state_color] || colours.needs_review;
        const isTag = marker.kind === "apriltag";
        const radius = isTag ? 0.05 : 0.18;
        const sphere = new THREE.Mesh(
          new THREE.SphereGeometry(radius, 20, 14),
          new THREE.MeshBasicMaterial({ color: colour, depthTest: false, transparent: true, opacity: 0.95 })
        );
        sphere.position.set(...marker.coordinates);
        sphere.renderOrder = 999;
        viewer.scene.scene.add(sphere);
        let square = null;
        if (Array.isArray(marker.corners) && marker.corners.length === 4) {
          const squareGeometry = new THREE.BufferGeometry();
          squareGeometry.setAttribute("position", new THREE.Float32BufferAttribute(marker.corners.flat(), 3));
          square = new THREE.LineLoop(squareGeometry, new THREE.LineBasicMaterial({ color: colour, depthTest: false, transparent: true, opacity: 0.95 }));
          square.renderOrder = 998;
          square.visible = markersVisible;
          viewer.scene.scene.add(square);
        }
        markerObjects.push({ annotation, sphere, square, marker, radius, colour });
        const element = annotation.domElement && annotation.domElement[0];
        if (element) {
          element.style.setProperty("--sam-marker", colour);
          element.title = marker.tooltip || tr("marker_open_review");
          const titlebar = element.querySelector(".annotation-titlebar");
          if (titlebar) { titlebar.style.background = colour; }
          element.addEventListener("click", () => window.dispatchEvent(new CustomEvent("samcloud-marker-selected", { detail: { name: marker.name } })));
          element.addEventListener("dblclick", event => {
            event.preventDefault();
            event.stopPropagation();
            window.location.href = isTag && marker.tag_id !== undefined
              ? `/control-points?tag=${encodeURIComponent(marker.tag_id)}`
              : `/control-points?marker=${encodeURIComponent(marker.name)}`;
          });
        }
      }
    } catch (error) { console.warn("Marker konnten nicht geladen werden", error); }
  };
  const cameraColour = camera => {
    if (cameraErrorColours && Number.isFinite(camera.mean_reprojection_error)) {
      const error = Math.min(4, Math.max(0, Number(camera.mean_reprojection_error)));
      return new THREE.Color().setHSL((1 - error / 4) * .34, .86, .55);
    }
    const count = Math.max(1, cameraObjects.length - 1);
    return new THREE.Color().setHSL((Number(camera.index || 0) / count) * .72, .86, .55);
  };
  const updateCameraColours = () => {
    const pathColours = [];
    for (const item of cameraObjects) {
      const colour = cameraColour(item.camera);
      item.marker.material.color.copy(colour);
      pathColours.push(colour.r, colour.g, colour.b);
    }
    if (cameraPath && cameraPath.geometry.getAttribute("color")) {
      cameraPath.geometry.setAttribute("color", new THREE.Float32BufferAttribute(pathColours, 3));
    }
  };
   let cameraCardTimer = null;
   let cameraCardPinned = false;
   let cameraCardHovered = false;
   const cameraCardHoldMs = 5000;
   const armCameraCardTimer = () => {
     clearTimeout(cameraCardTimer);
     cameraCardTimer = setTimeout(() => {
       if (cameraCardHovered) return;
       cameraCardPinned = false;
       if (cameraCard) cameraCard.classList.remove("is-open");
     }, cameraCardHoldMs);
   };
   if (cameraCard) {
     cameraCard.addEventListener("mouseenter", () => { cameraCardHovered = true; clearTimeout(cameraCardTimer); });
     cameraCard.addEventListener("mouseleave", () => { cameraCardHovered = false; armCameraCardTimer(); });
   }
   const showCamera = (camera, includeImage) => {
     if (!cameraCard || !cameraPhotosVisible) return;
     /* Ein angeklicktes Foto bleibt 5 s unten rechts stehen; Hover auf andere Marker ersetzt es nicht. */
     if (!includeImage && cameraCardPinned) return;
     if (includeImage) cameraCardPinned = true;
    cameraCard.innerHTML = "";
    const title = document.createElement("strong");
    title.textContent = camera.source_name || camera.image_name || tr("camera_fallback_title");
    cameraCard.append(title);
    const details = document.createElement("div");
    const error = Number.isFinite(camera.mean_reprojection_error) ? `${Number(camera.mean_reprojection_error).toFixed(2)} px` : "–";
    details.textContent = tr("camera_details", { index: Number(camera.index) + 1, points: camera.points3d || 0, error });
    cameraCard.append(details);
    if (includeImage && camera.id) {
      const imageUrl = `/api/camera-thumbnail/${encodeURIComponent(camera.id)}?size=320`;
      const image = document.createElement("img");
      image.alt = camera.source_name || camera.image_name || tr("camera_photo_alt");
      image.src = imageUrl;
      image.title = tr("camera_photo_open");
      image.addEventListener("click", () => window.open(imageUrl.replace("size=320", "size=1024"), "_blank", "noopener"));
      cameraCard.append(image);
      const link = document.createElement("a");
      link.href = imageUrl;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = tr("camera_show_photo");
      cameraCard.append(link);
    }
    cameraCard.classList.add("is-open");
    armCameraCardTimer();
  };
  const pickCamera = (event, includeImage) => {
     if (!cameraObjects.length || !camerasVisible || !cameraPhotosVisible) return null;
    const rect = root.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, viewer.scene.getActiveCamera());
    const hit = raycaster.intersectObjects(cameraObjects.map(item => item.marker), false)[0];
    if (hit && hit.object.userData.camera) { showCamera(hit.object.userData.camera, includeImage); return hit.object.userData.camera; }
    return null;
  };
  const cameraAt = event => {
    if (!cameraObjects.length || !camerasVisible) return null;
    const rect = root.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, viewer.scene.getActiveCamera());
    const hit = raycaster.intersectObjects(cameraObjects.map(item => item.marker), false)[0];
    return hit && hit.object.userData.camera ? hit.object.userData.camera : null;
  };
  /* Doppelklick auf eine Kameraposition öffnet direkt das zugehörige Foto. */
  root.querySelector(".sam-potree-area").addEventListener("dblclick", event => {
    const camera = cameraAt(event);
    if (!camera || !camera.id) return;
    event.preventDefault();
    event.stopPropagation();
    window.open(`/api/camera-thumbnail/${encodeURIComponent(camera.id)}?size=1024`, "_blank", "noopener");
  }, true);
  const area = root.querySelector(".sam-potree-area");
  let pointerDown = null;
  let pointOverlay = null;
  let pointHelper = null;
  const pointCard = root.querySelector(".sam-potree-point-card");
  const clearPointInfo = () => {
    if (pointOverlay) {
      viewer.scene.scene.remove(pointOverlay);
      pointOverlay.geometry.dispose();
      pointOverlay.material.map && pointOverlay.material.map.dispose();
      pointOverlay.material.dispose();
      pointOverlay = null;
    }
    if (pointHelper) { viewer.scene.scene.remove(pointHelper); pointHelper.geometry.dispose(); pointHelper.material.dispose(); pointHelper = null; }
    if (highlightedTag) { if (highlightedTag.square) highlightedTag.square.material.color.set(highlightedTag.colour); highlightedTag = null; }
    if (pointCard) pointCard.classList.remove("is-open");
  };
  const readPickedValue = (point, names) => {
    for (const name of names) {
      const value = point && point[name];
      if (Array.isArray(value) || ArrayBuffer.isView(value)) return Number(value[0]);
      if (value !== undefined && value !== null) return Number(value);
    }
    return 0;
  };
  const showPointCard = (event, title, details) => {
    if (!pointCard) return;
    pointCard.innerHTML = "";
    const heading = document.createElement("strong"); heading.textContent = title; pointCard.append(heading);
    for (const detail of details) { const line = document.createElement("div"); line.textContent = detail; pointCard.append(line); }
    const bounds = root.getBoundingClientRect();
    pointCard.style.left = `${Math.min(bounds.width - 275, Math.max(10, event.clientX - bounds.left + 12))}px`;
    pointCard.style.top = `${Math.min(bounds.height - 100, Math.max(10, event.clientY - bounds.top + 12))}px`;
    pointCard.classList.add("is-open");
  };
  const showInstanceOverlay = async instance => {
    clearPointInfo();
    const response = await fetch(`/api/instances/${encodeURIComponent(instance.id)}/points`, { cache: "no-store" });
    if (!response.ok) return;
    const buffer = await response.arrayBuffer();
    const view = new DataView(buffer);
    const count = view.getUint32(8, true);
    if (buffer.byteLength < 16 + count * 15) return;
    const positions = new Float32Array(count * 3);
    const colours = new Float32Array(count * 3);
    for (let index = 0, offset = 16; index < count; index += 1, offset += 15) {
      positions[index * 3] = view.getFloat32(offset, true); positions[index * 3 + 1] = view.getFloat32(offset + 4, true); positions[index * 3 + 2] = view.getFloat32(offset + 8, true);
      colours[index * 3] = .5 + view.getUint8(offset + 12) / 510; colours[index * 3 + 1] = .5 + view.getUint8(offset + 13) / 510; colours[index * 3 + 2] = .5 + view.getUint8(offset + 14) / 510;
    }
    const textureCanvas = document.createElement("canvas"); textureCanvas.width = textureCanvas.height = 16;
    const textureContext = textureCanvas.getContext("2d"); textureContext.beginPath(); textureContext.arc(8, 8, 7, 0, Math.PI * 2); textureContext.fillStyle = "white"; textureContext.fill();
    const geometry = new THREE.BufferGeometry(); geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3)); geometry.setAttribute("color", new THREE.BufferAttribute(colours, 3));
    const material = new THREE.PointsMaterial({ size: 2.4, vertexColors: true, map: new THREE.CanvasTexture(textureCanvas), alphaTest: .5, transparent: true, depthWrite: false, sizeAttenuation: true });
    pointOverlay = new THREE.Points(geometry, material); pointOverlay.renderOrder = 10; viewer.scene.scene.add(pointOverlay);
    if (Array.isArray(instance.bbox_min) && Array.isArray(instance.bbox_max)) {
      pointHelper = new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(...instance.bbox_min), new THREE.Vector3(...instance.bbox_max)), 0xffffff);
      viewer.scene.scene.add(pointHelper);
    }
  };
  /* AprilTag-Marker: Kugel per Raycast, Quadrat/Zentrum mit Pixeltoleranz. */
  const tagAt = event => {
    const tags = markerObjects.filter(item => item.marker && item.marker.kind === "apriltag" && markersVisible);
    if (!tags.length) return null;
    const rect = root.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    const activeCamera = viewer.scene.getActiveCamera();
    raycaster.setFromCamera(pointer, activeCamera);
    const hit = raycaster.intersectObjects(tags.map(item => item.sphere), false)[0];
    if (hit) return tags.find(item => item.sphere === hit.object) || null;
    const toScreen = position => {
      const projected = new THREE.Vector3(...position).project(activeCamera);
      return projected.z > 1 ? null : { x: (projected.x + 1) / 2 * rect.width + rect.left, y: (1 - projected.y) / 2 * rect.height + rect.top };
    };
    const tolerance = 10;
    let best = null; let bestDistance = Infinity;
    for (const item of tags) {
      const centre = toScreen(item.marker.coordinates);
      if (!centre) continue;
      let distance = Math.hypot(event.clientX - centre.x, event.clientY - centre.y);
      const corners = Array.isArray(item.marker.corners) ? item.marker.corners.map(toScreen) : [];
      if (corners.length === 4 && corners.every(Boolean)) {
        let inside = false;
        for (let a = 0, b = 3; a < 4; b = a++) {
          if ((corners[a].y > event.clientY) !== (corners[b].y > event.clientY)
            && event.clientX < (corners[b].x - corners[a].x) * (event.clientY - corners[a].y) / (corners[b].y - corners[a].y) + corners[a].x) inside = !inside;
        }
        if (inside) distance = 0;
        else {
          for (let a = 0; a < 4; a += 1) {
            const p = corners[a], q = corners[(a + 1) % 4];
            const lengthSq = (q.x - p.x) ** 2 + (q.y - p.y) ** 2 || 1;
            const t = Math.max(0, Math.min(1, ((event.clientX - p.x) * (q.x - p.x) + (event.clientY - p.y) * (q.y - p.y)) / lengthSq));
            distance = Math.min(distance, Math.hypot(event.clientX - (p.x + t * (q.x - p.x)), event.clientY - (p.y + t * (q.y - p.y))));
          }
        }
      }
      if (distance <= tolerance && distance < bestDistance) { best = item; bestDistance = distance; }
    }
    return best;
  };
  const showTag = (event, item) => {
    clearPointInfo();
    const marker = item.marker;
    const lines = [];
    if (Number.isFinite(Number(marker.tag_size_mm))) {
      const deviation = Number(marker.tag_deviation_pct);
      lines.push(tr("tag_size", { size: Number(marker.tag_size_mm).toFixed(1) }) + (Number.isFinite(deviation) ? ` (${deviation >= 0 ? "+" : ""}${deviation.toFixed(1)} %)` : ""));
    }
    const imageCount = Array.isArray(marker.images) ? marker.images.length : Number(marker.images);
    if (Number.isFinite(imageCount)) lines.push(tr("tag_seen_in", { count: imageCount }));
    const totalError = marker.errors ? marker.errors.total_m : null;
    if (totalError !== null && totalError !== undefined && Number.isFinite(Number(totalError))) lines.push(tr("tag_error", { error: (Number(totalError) * 1000).toFixed(0) }));
    let statusText = marker.status === "outlier" ? tr("tag_outlier") : tr("tag_ok");
    /* The server warning text is matched literally (data, not UI text); only the displayed suffix is translated. */
    if (markerWarnings.some(text => String(text).includes("keine Kontrolle möglich"))) statusText += `, ${tr("tag_no_check")}`;
    lines.push(statusText);
    if (item.square) { item.square.material.color.set(0xffffff); highlightedTag = item; }
    if (Array.isArray(marker.corners) && marker.corners.length === 4) {
      const box = new THREE.Box3();
      for (const corner of marker.corners) box.expandByPoint(new THREE.Vector3(...corner));
      pointHelper = new THREE.Box3Helper(box, 0xffffff);
      viewer.scene.scene.add(pointHelper);
    }
    showPointCard(event, `AprilTag ${marker.tag_id}`, lines);
  };
  const handlePointClick = async event => {
    if (!config.enable_point_info || !cloud) return;
    const canvasRect = viewer.renderer.domElement.getBoundingClientRect();
    const mouse = { x: event.clientX - canvasRect.left, y: event.clientY - canvasRect.top };
    const hit = Potree.Utils.getMousePointCloudIntersection(mouse, viewer.scene.getActiveCamera(), viewer, viewer.scene.pointclouds, { pickClipped: true });
    const point = hit && hit.point;
    if (!point) { clearPointInfo(); return; }
    const classId = readPickedValue(point, ["classification", "class", "scalar_class"]);
    const instanceId = readPickedValue(point, ["pointSourceID", "pointSourceId", "point_source_id", "source id", "point source id"]);
    const className = (config.class_display_names || [])[classId] || config.class_names_de[classId] || config.class_labels[classId] || tr("class_fallback", { index: classId });
    const fineTerm = readPickedValue(point, ["user data", "userData", "user_data", "userdata", "user_data_8"]);
    const fineName = fineTerm > 0 ? ((config.fine_term_display_names || config.fine_term_names_de || [])[fineTerm] || "") : "";
    const showFallback = () => {
      if (fineName) showPointCard(event, fineName, [className]); else showPointCard(event, className, []);
    };
    if (!instanceId) { clearPointInfo(); showFallback(); return; }
    try {
      const response = await fetch("/api/instances", { cache: "no-store" });
      const payload = response.ok ? await response.json() : { instances: [] };
      const instance = (payload.instances || []).find(item => Number(item.id) === instanceId);
      if (!instance) { clearPointInfo(); showFallback(); return; }
      /* German UI: configured German object names; other languages: the detector's (English) object label. */
      const instanceLabel = item => (config.language === "de" || !item.label
        ? config.instance_labels_de[item.label] || item.label_de || item.label
        : item.label) || className;
      const label = instanceLabel(instance);
      const sameLabel = (payload.instances || []).filter(item => instanceLabel(item) === label).length;
      const title = sameLabel > 1 && Number(instance.label_index) > 0 ? `${label} ${instance.label_index}` : label;
      const pointCount = tr("points", { count: Number(instance.point_count || 0).toLocaleString(numberLocale) });
      showPointCard(event, title, [className, pointCount]);
      await showInstanceOverlay(instance);
      showPointCard(event, title, [className, pointCount]);
    } catch (_error) { clearPointInfo(); showFallback(); }
  };
  area.addEventListener("pointerdown", event => { pointerDown = { x: event.clientX, y: event.clientY }; });
  area.addEventListener("pointerup", event => {
    if (!pointerDown || Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y) > 4) pointerDown = null;
  });
  area.addEventListener("pointermove", event => pickCamera(event, false));
  area.addEventListener("click", event => {
    const isClick = pointerDown && Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y) <= 4;
    pointerDown = null;
    if (isClick && config.enable_point_info) { const tag = tagAt(event); if (tag) { showTag(event, tag); return; } }
    if (pickCamera(event, true) || !isClick) return;
    handlePointClick(event);
  });
  window.addEventListener("keydown", event => { if (event.key === "Escape") clearPointInfo(); });
  const loadCameras = async () => {
    if (!config.camera_endpoint || cameraObjects.length) return;
    try {
      const response = await fetch(config.camera_endpoint, { cache: "no-store" });
      if (!response.ok) return;
      const payload = await response.json();
      const cameras = (payload.cameras || []).filter(camera => Array.isArray(camera.position) && camera.position.length === 3);
      cameraPanorama = payload.is_panorama === true;
      const cone = new THREE.ConeGeometry(.65, 1.5, 4);
      const ball = new THREE.SphereGeometry(.7, 24, 16);
      const sphereButton = root.querySelector('[data-action="camera-sphere-texture"]');
      if (sphereButton) sphereButton.style.display = cameraPanorama ? "" : "none";
      for (const camera of cameras) {
        const marker = new THREE.Mesh(cameraPanorama ? ball : cone, new THREE.MeshBasicMaterial({ depthTest: true, depthWrite: true }));
        marker.position.set(...camera.position);
        const direction = new THREE.Vector3(...(camera.direction || [0, 0, 1]));
        if (!cameraPanorama && direction.lengthSq() > 0) marker.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
        marker.userData.camera = camera;
         cameraMarkerLayer.add(marker);
        cameraObjects.push({ camera, marker });
      }
       updateCameraColours();
       const sphereToggle = root.querySelector('[data-action="camera-sphere-texture"]');
       if (sphereToggle) sphereToggle.classList.toggle("is-selected", cameraSphereTextures);
       syncCameraSettings();
       syncCameraBillboards();
    } catch (error) { console.warn("Kamerapositionen konnten nicht geladen werden", error); }
  };
};
start();
</script>
""".replace("__STRINGS__", json.dumps(_viewer_strings()).replace("</", "<\\/")).replace("__CONFIG__", json.dumps(config))
