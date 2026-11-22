"""Write a synthetic scan as a real SAMcloud indoor project folder (same input path as a processed 360 scan).

Written files (all that ``samcloud.engine.auto_dxf`` / ``samcloud.dxf.jobs.run_all`` read for an indoor level):

* ``project.samcloud.yml``                      class_set indoor, engine sam3 (-> model profile sam3_indoor),
                                                ``settings.scale.cloud_units: metric`` (the synthetic cloud is in metres)
* ``products/classified/classified.ply``        x y z, rgb, scalar_class (class index 0..19), fine_term, instance_id
* ``products/classified/fine_labels.json``      fine-term vocabulary (same builder as the engine)
* ``products/classified/instances.json``        SAM3 instance records (schema_version 1)
* ``products/classified/instances_source.npz``  per-point instance source (format parity; not read by the DXF chain)
* ``products/sparse/0``                         COLMAP model (pinhole cameras of the rendered photos + 3D points)
* ``inputs/images/*.jpg``                       rendered photos (orthophoto colour source)
* ``products/dxf/state.json``                   only ``segment_source: pointcloud`` (no levels: storeys are detected)

Not written (documented skips): ``inputs/cubemap`` (-> opening refinement runs points-only), SAM3 weights / ortho SAM
segmentation (segment source ``pointcloud``), depth products (``products/depth``), ``reports/run_metrics.json``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .scan import Photo, ScanResult
from .spec import CLASS_NAMES, SceneSpec

INSTANCE_CLASSES = {5, 6, 9, 15, 16, 17, 18, 19}


def _fine_vocabulary() -> tuple[list[str], np.ndarray, dict[str, Any]]:
    from samcloud.core.paths import CONFIG_DIRECTORY
    from samcloud.engine.fine_terms import label_owner_classes

    config = json.loads((CONFIG_DIRECTORY / "classes.json").read_text(encoding="utf-8"))
    aliases = config.get("fine_aliases") or {}
    labels: list[str] = []
    for term in (config.get("fine_terms") or {}).get("indoor", []):
        canonical = aliases.get(term, term)
        if canonical not in labels:
            labels.append(canonical)
    owners = label_owner_classes(config, "indoor", labels)
    return labels, owners, config


def write_classified_cloud(directory: Path, spec: SceneSpec, scan: ScanResult, labels_by_instance: dict[int, str]) -> dict[str, Any]:
    from plyfile import PlyData, PlyElement
    from samcloud.engine.fine_terms import write_fine_labels_json

    directory.mkdir(parents=True, exist_ok=True)
    xyz = spec.to_world(scan.xyz)
    labels, owners, config = _fine_vocabulary()
    ids = {label: index for index, label in enumerate(labels, start=1)}
    from samcloud.bim.room_objects.evidence import canonical

    for label, index in list(ids.items()):
        ids.setdefault(canonical(label), index)  # an instance label (monitor) and its fine term (computer monitor)
    fine = np.zeros(len(xyz), dtype=np.uint8)
    for instance_id, label in labels_by_instance.items():
        index = ids.get(label)
        if index is None:
            continue
        sel = (scan.instance == instance_id) & (scan.cls == owners[index - 1])
        fine[sel] = index
    data = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                     ("scalar_class", "u1"), ("fine_term", "u1"), ("instance_id", "u4")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = scan.rgb[:, 0], scan.rgb[:, 1], scan.rgb[:, 2]
    data["scalar_class"] = scan.cls
    data["fine_term"] = fine
    data["instance_id"] = scan.instance
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(directory / "classified.ply"))
    write_fine_labels_json(directory / "fine_labels.json", "indoor", labels, owners, config)
    records = instance_records(xyz, scan, labels_by_instance)
    (directory / "instances.json").write_text(json.dumps({"schema_version": 1, "instances": records}, indent=2), encoding="utf-8")
    np.savez_compressed(directory / "instances_source.npz", xyz=xyz.astype(np.float32), instance_id=scan.instance.astype(np.uint32),
                        scalar_class=scan.cls.astype(np.uint8), metadata_json=np.array(json.dumps(records)))
    return {"points": int(len(xyz)), "fine_labelled": int(np.count_nonzero(fine)), "instances": len(records)}


def instance_records(xyz_world: np.ndarray, scan: ScanResult, labels_by_instance: dict[int, str], min_points: int = 30) -> list[dict[str, Any]]:
    rng = np.random.default_rng(3)
    records = []
    for instance_id in sorted(int(i) for i in np.unique(scan.instance) if i > 0):
        sel = scan.instance == instance_id
        if int(sel.sum()) < min_points:
            continue
        classes = Counter(scan.cls[sel].tolist())
        class_id = int(classes.most_common(1)[0][0])
        if class_id not in INSTANCE_CLASSES:
            continue
        pts = xyz_world[sel]
        records.append({
            "id": instance_id, "class_id": class_id, "class_name": CLASS_NAMES.get(class_id, str(class_id)),
            "label": labels_by_instance.get(instance_id, CLASS_NAMES.get(class_id, "object")), "label_index": 0,
            "point_count": int(sel.sum()), "centroid": pts.mean(axis=0).tolist(), "bbox_min": pts.min(axis=0).tolist(),
            "bbox_max": pts.max(axis=0).tolist(), "view_count": int(len(np.unique(scan.station[sel]))),
            "confidence": float(round(rng.uniform(0.75, 0.97), 4)),
        })
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)
    for group in by_label.values():
        group.sort(key=lambda item: tuple(item["centroid"]))
        for index, record in enumerate(group, start=1):
            record["label_index"] = index
    return records


def write_sparse_model(project: Path, spec: SceneSpec, photos: list[Photo], cloud_world: np.ndarray) -> Path:
    import pycolmap

    images_dir = project / "inputs" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    model_dir = project / "products" / "sparse" / "0"
    model_dir.mkdir(parents=True, exist_ok=True)
    reconstruction = pycolmap.Reconstruction()
    width, height, focal = photos[0].width, photos[0].height, photos[0].focal
    camera = pycolmap.Camera(camera_id=1, model="PINHOLE", width=width, height=height, params=np.array([focal, focal, width / 2.0, height / 2.0]))
    reconstruction.add_camera_with_trivial_rig(camera)
    rw = spec.world_rotation()
    for image_id, photo in enumerate(photos, start=1):
        rotation = photo.rotation @ rw.T  # cam_from_world
        centre = spec.to_world(photo.centre[None, :])[0]
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), -rotation @ centre)
        reconstruction.add_image_with_trivial_frame(pycolmap.Image(name=photo.name, camera_id=1, image_id=image_id), pose)
        ok = cv2.imwrite(str(images_dir / photo.name), cv2.cvtColor(photo.image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise OSError(f"could not write {images_dir / photo.name}")
    rng = np.random.default_rng(1)
    pick = rng.choice(len(cloud_world), size=min(3000, len(cloud_world)), replace=False)
    for point in cloud_world[pick]:
        reconstruction.add_point3D(np.asarray(point, dtype=np.float64), pycolmap.Track())
    reconstruction.write(model_dir)
    return model_dir


def write_project(project: Path, spec: SceneSpec, scan: ScanResult, photos: list[Photo]) -> dict[str, Any]:
    from samcloud.dxf.state import DxfProjectState, save_state

    project.mkdir(parents=True, exist_ok=True)
    labels = {item.instance_id: item.label for item in spec.furniture}
    labels.update({o.instance_id: o.kind for o in spec.openings if o.instance_id})
    labels.update({c.instance_id: "column" for c in spec.columns})
    info = write_classified_cloud(project / "products" / "classified", spec, scan, labels)
    write_sparse_model(project, spec, photos, spec.to_world(scan.xyz))
    document = {
        "project": {"schema_version": 2, "name": project.name},
        "paths": {"image_directory": str((project / "inputs" / "images").resolve()), "image_storage": "internal"},
        "acquisition": {"type": "synthetic_bench"},
        "settings": {"classification": {"class_set": "indoor", "engine": "sam3"}, "paths": {"image_directory": str((project / "inputs" / "images").resolve())},
                     "workflow": {"auto_dxf": True},
                     # The synthetic cloud is in true metres: mark it metric like a real metric project
                     # (samcloud.core.metric_scale), so the bench measures the delivered metres, not the storey heuristic.
                     "scale": {"cloud_units": "metric"}},
        "workflow": {},
    }
    (project / "project.samcloud.yml").write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    save_state(project, DxfProjectState(segment_source="pointcloud", plan_variant="ortho"))
    info["photos"] = len(photos)
    return info
