"""Generate one bench case: scene -> mesh -> scan -> project folder + ground truth files."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .groundtruth import class_samples, ground_truth
from .mesh import build_scene_mesh
from .project import write_project
from .scan import render_photos, simulate_scan
from .scenes import check_scene, make_scene
from .spec import SceneSpec


def generate_case(family: str, seed: int, case_dir: Path, spec: SceneSpec | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    spec = spec or make_scene(family, seed)
    problems = check_scene(spec)
    case_dir = Path(case_dir)
    gt_dir = case_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    mesh = build_scene_mesh(spec)
    scan = simulate_scan(spec, mesh)
    photos = render_photos(spec, mesh, scan.stations)
    project = case_dir / spec.name
    info = write_project(project, spec, scan, photos)
    (gt_dir / "scene.json").write_text(spec.to_json(), encoding="utf-8")
    truth = ground_truth(spec)
    truth["scene_problems"] = problems
    truth["scan"] = scan.stats
    (gt_dir / "ground_truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    samples = class_samples(spec, mesh)
    np.savez_compressed(gt_dir / "class_samples.npz", **samples)
    elements = [{"id": e.id, "kind": e.kind, "class_idx": e.class_idx, "instance_id": e.instance_id, "label": e.label} for e in mesh.elements]
    np.savez_compressed(gt_dir / "mesh.npz", vertices=mesh.vertices.astype(np.float32), faces=mesh.faces, face_element=mesh.face_element,
                        elements=np.array(json.dumps(elements)))
    info.update({"project": str(project), "scene_problems": problems, "scan": scan.stats, "generate_seconds": round(time.perf_counter() - started, 2)})
    (case_dir / "generate.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info
