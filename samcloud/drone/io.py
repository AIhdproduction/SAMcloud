"""Load a project's classified cloud as a ``PointSet`` with LAS class codes.

``classified.ply`` stores ``scalar_class`` as an internal one-based index into the class set
(0 = unclassified).  ``classes.json`` maps it to LAS codes through ``las_codes[class_set]``
(``las_codes[i]`` is the code of internal index ``i``); ``prompts[class_set][i - 1]`` lists the
names of internal index ``i`` (the first entry is the canonical name).
"""

from __future__ import annotations

import gzip
import io
import json
import logging
from pathlib import Path

import numpy as np

from samcloud.classification.custom import project_classes_path
from samcloud.drone.contracts import LocalFrame, PointSet

LOGGER = logging.getLogger(__name__)

NEVER_CLASSIFIED_INTERNAL = 255
LAS_NEVER_CLASSIFIED = 0
LAS_UNCLASSIFIED = 1


def _load_class_config(classes_json: Path | str | None, products_dir: Path | str | None = None) -> dict:
    """Read the explicit classes.json, else the project's (``products_dir.parent``), else the global one."""
    if classes_json is not None:
        path = Path(classes_json)
    else:
        path = project_classes_path(Path(products_dir).parent if products_dir is not None else None)
    return json.loads(path.read_text(encoding="utf-8"))


def internal_to_las(internal: np.ndarray, *, classes_json: Path | str | None = None,
                    class_set: str = "outdoor", products_dir: Path | str | None = None) -> np.ndarray:
    """Map internal ``scalar_class`` indices to LAS codes (uint16).

    255 maps to 0 (never classified); any other index outside the table maps to 1 (unclassified).
    """
    table = np.asarray(_load_class_config(classes_json, products_dir)["las_codes"][class_set], dtype=np.uint16)
    internal = np.asarray(internal).astype(np.int64).reshape(-1)
    out = np.full(internal.shape, LAS_UNCLASSIFIED, dtype=np.uint16)
    valid = (internal >= 0) & (internal < len(table))
    out[valid] = table[internal[valid]]
    out[internal == NEVER_CLASSIFIED_INTERNAL] = LAS_NEVER_CLASSIFIED
    return out


def class_code(name: str, *, classes_json: Path | str | None = None, class_set: str = "outdoor",
               products_dir: Path | str | None = None) -> int:
    """LAS code of a class given any of its prompt names (case-insensitive); ``KeyError`` if unknown."""
    config = _load_class_config(classes_json, products_dir)
    wanted = name.strip().lower()
    for index, terms in enumerate(config["prompts"][class_set], start=1):
        if wanted in (t.lower() for t in terms):
            return int(config["las_codes"][class_set][index])
    raise KeyError(f"class {name!r} not found in class set {class_set!r}")


def _read_scores(classified_dir: Path) -> dict[str, np.ndarray] | None:
    plain = classified_dir / "classified_scores.npz"
    zipped = classified_dir / "classified_scores.npz.gz"
    try:
        if plain.is_file():
            data = np.load(plain)
        elif zipped.is_file():
            data = np.load(io.BytesIO(gzip.open(zipped, "rb").read()))
        else:
            return None
        return {key: data[key] for key in data.files}
    except Exception as exc:
        LOGGER.warning("classified_scores.npz unreadable (%s); continuing without scores", exc)
        return None


def load_classified_pointset(products_dir: Path | str, *, classes_json: Path | str | None = None,
                             class_set: str = "outdoor", max_points: int | None = None,
                             ply_path: Path | str | None = None, load_scores: bool = True,
                             frame: str = "enu", project_dir: Path | str | None = None) -> PointSet:
    """Read ``products/classified/classified.ply`` into a ``PointSet`` with LAS ``classes``.

    ``classes_json`` defaults to the project's classes.json (``products_dir.parent``) or the global
    one.  Scores from ``classified_scores.npz[.gz]`` fill confidence (top1_conf), margin
    (top1_conf - top2_conf) and view_count when their length equals the cloud length; otherwise
    they stay ``None`` and a warning is logged.  ``max_points`` applies a deterministic stride.
    ``ply_path`` overrides the PLY location.  ``frame="lv95_local"`` returns xyz/normals in the
    LV95-aligned local frame (see ``load_classified_pointset_lv95`` for the matching ``LocalFrame``;
    ``project_dir`` defaults to ``products_dir.parent``).  Default ``"enu"`` keeps the COLMAP frame.
    """
    result = _load_enu(products_dir, classes_json=classes_json, class_set=class_set, max_points=max_points,
                       ply_path=ply_path, load_scores=load_scores)
    if frame == "enu":
        return result
    if frame != "lv95_local":
        raise ValueError(f"unknown frame {frame!r}; expected 'enu' or 'lv95_local'")
    return _to_lv95_local(result, Path(project_dir) if project_dir else Path(products_dir).parent)[0]


def load_classified_pointset_lv95(products_dir: Path | str, *, project_dir: Path | str | None = None,
                                  **kwargs) -> tuple[PointSet, LocalFrame]:
    """Like ``load_classified_pointset`` but in the LV95-aligned local frame, plus its ``LocalFrame``."""
    result = _load_enu(products_dir, **kwargs)
    return _to_lv95_local(result, Path(project_dir) if project_dir else Path(products_dir).parent)


def _to_lv95_local(points: PointSet, project_dir: Path,
                   georef: dict | None = None) -> tuple[PointSet, LocalFrame]:
    from samcloud.drone import frames

    georef = georef if georef is not None else frames.project_georeference(project_dir)
    enu = points.xyz
    lv95 = frames.enu_to_lv95(enu, georef)
    frame = frames.make_local_frame(lv95)
    jac = frames.local_jacobian(enu.mean(axis=0), georef)
    points.xyz = frames.to_local(lv95, frame)
    if points.normals is not None:
        points.normals = frames.transform_normals(points.normals, jac)
    return points, frame


def _load_enu(products_dir: Path | str, *, classes_json: Path | str | None = None,
              class_set: str = "outdoor", max_points: int | None = None,
              ply_path: Path | str | None = None, load_scores: bool = True) -> PointSet:
    """Implementation of the ENU-frame loader."""
    import open3d as o3d

    products = Path(products_dir)
    ply = Path(ply_path) if ply_path else products / "classified" / "classified.ply"
    if not ply.is_file():
        raise FileNotFoundError(ply)
    cloud = o3d.t.io.read_point_cloud(str(ply))
    point = cloud.point
    n = len(point.positions)
    if "scalar_class" not in point:
        raise ValueError(f"{ply.name} has no scalar_class property")
    stride = 1 if not max_points or n <= max_points else -(-n // int(max_points))
    sel = slice(None, None, stride)

    xyz = point.positions.numpy()[sel].astype(np.float64)
    internal = point["scalar_class"].numpy().reshape(-1)[sel]
    result = PointSet(
        xyz=xyz,
        classes=internal_to_las(internal, classes_json=classes_json, class_set=class_set, products_dir=products),
    )
    if "colors" in point:
        colors = point.colors.numpy()[sel]
        result.rgb = colors.astype(np.uint8) if colors.dtype == np.uint8 else \
            np.clip(colors.astype(np.float64) * 255.0, 0, 255).astype(np.uint8)
    if "normals" in point:
        result.normals = point.normals.numpy()[sel].astype(np.float64)

    scores = _read_scores(ply.parent) if load_scores else None
    if scores is not None:
        if "top1_conf" in scores and len(scores["top1_conf"]) == n:
            top1 = scores["top1_conf"].astype(np.float32)
            result.confidence = top1[sel]
            if "top2_conf" in scores and len(scores["top2_conf"]) == n:
                result.margin = (top1 - scores["top2_conf"].astype(np.float32))[sel]
            if "view_count" in scores and len(scores["view_count"]) == n:
                result.view_count = scores["view_count"].astype(np.uint16)[sel]
        else:
            size = len(scores["top1_conf"]) if "top1_conf" in scores else "missing"
            LOGGER.warning("classified_scores size %s does not match %d points; scores ignored", size, n)
    return result
