"""Curved curbs swept along an absolute directrix must land on the axis, not at twice its offset."""

from __future__ import annotations

import math
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import numpy as np

from samcloud.bim.core.ifc import write_ifc
from samcloud.bim.core.model import CurbElement, CurbParams, Scene


def test_swept_curb_vertices_follow_axis(tmp_path: Path) -> None:
    cx, cy, radius, z0, height = 50.0, 140.0, 10.0, 0.5, 0.25
    axis = [(cx + radius * math.cos(0.15 * i), cy + radius * math.sin(0.15 * i), z0) for i in range(6)]
    scene = Scene()
    scene.add(CurbElement(params=CurbParams(axis=axis, height=height)))
    path = tmp_path / "curb_sweep.ifc"
    write_ifc(scene, path)
    file = ifcopenshell.open(str(path))
    assert len(file.by_type("IfcFixedReferenceSweptAreaSolid")) == 1
    (curb,) = [c for c in file.by_type("IfcCivilElement") if c.ObjectType == "Randstein"]
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    verts = np.asarray(ifcopenshell.geom.create_shape(settings, curb).geometry.verts).reshape(-1, 3)
    assert len(verts) > 0
    xy_axis = np.asarray([(p[0], p[1]) for p in axis])
    for x, y, z in verts:
        # distance to the axis polyline (segment-wise)
        d = min(
            np.linalg.norm(np.array([x, y]) - (a + np.clip(np.dot(np.array([x, y]) - a, b - a) / np.dot(b - a, b - a), 0, 1) * (b - a)))
            for a, b in zip(xy_axis, xy_axis[1:])
        )
        assert d < 0.2, (x, y, d)
        assert z0 - height - 0.01 <= z <= z0 + 0.01, z  # the axis is the curb top: the body hangs below it
