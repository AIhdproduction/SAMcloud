"""Former lawn is plain terrain: grass noise removed from the ground, no lawn label / element / DXF entity."""

from __future__ import annotations

import ezdxf
import numpy as np
from shapely.geometry import box

from samcloud.drone import contracts as c
from samcloud.drone import qa_cad
from samcloud.outdoor_model import terrain as tr
from samcloud.terrain.clean import lawn_ground_dtm
from samcloud.terrain.rasters import build_dtm_raster
from samcloud.terrain.tin import build_labeled_tin

BOUNDS = (0.0, 0.0, 30.0, 30.0)
RES = 0.25
LAWN = box(8, 8, 26, 26)


def _truth(x, y):
    return 0.04 * x + 0.02 * y + 0.01 * np.sin(x / 3.0)


def _points(seed=5):
    """Ground points: bare outside the lawn, grass tops (+4..16 cm) on 65 % of the lawn points."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 30.0, (180_000, 2))
    z = _truth(xy[:, 0], xy[:, 1]) + rng.normal(0, 0.008, len(xy))
    lawn = (xy[:, 0] > 8) & (xy[:, 0] < 26) & (xy[:, 1] > 8) & (xy[:, 1] < 26)
    grass = lawn & (rng.random(len(xy)) < 0.65)
    z[grass] += rng.uniform(0.04, 0.16, grass.sum())
    return np.column_stack((xy, z)), lawn


def _rms(dtm):
    rows, cols = np.indices(dtm.shape)
    x, y = (cols + 0.5) * RES, 30.0 - (rows + 0.5) * RES
    inside = (x > 9) & (x < 25) & (y > 9) & (y < 25)
    return float(np.sqrt(np.mean((dtm[inside] - _truth(x, y)[inside]) ** 2)))


def test_lawn_ground_rms_below_three_cm_and_no_lawn_outputs() -> None:
    pts, lawn = _points()
    dtm, validity, _ = build_dtm_raster(pts, resolution=RES, bounds=BOUNDS)
    before = _rms(dtm)
    new, mask, report = lawn_ground_dtm(dtm, BOUNDS, RES, pts[lawn], pts, config=c.load_drone_config())
    after = _rms(new)
    assert before > 0.05 and after < 0.03 and report["cells"] > 3000
    # final TIN over the new DTM: RMS to the true ground in the former lawn < 3 cm, one terrain, no lawn label
    rows, cols = np.nonzero(np.isfinite(new))
    support = np.column_stack(((cols + 0.5) * RES, 30.0 - (rows + 0.5) * RES, new[rows, cols]))
    regions = {"road": [box(0, 0, 4, 30)]}
    tin = build_labeled_tin(support, regions=regions, tolerance=0.02, initial_spacing=2.0, max_vertices=30000)
    sampler = tr.TinSampler(tin.vertices, tin.triangles)
    probe = np.random.default_rng(1).uniform(9, 25, (2000, 2))
    error = sampler.sample(probe) - _truth(probe[:, 0], probe[:, 1])
    assert np.sqrt(np.nanmean(error**2)) < 0.03
    assert not (np.asarray(tin.labels) == c.TIN_LABELS["lawn"]).any()
    elements, _qa = tr.build_terrain_elements(tin, config=c.load_drone_config())
    assert len(elements) == 1 and elements[0].evidence["lawn_triangles"] == 0 and elements[0].evidence["lawn_triangle_indices"] == []
    frame = c.LocalFrame(origin_lv95=(2_600_000.0, 1_200_000.0, 430.0))
    entities = tr.to_dxf_entities(elements, frame, tin=tin, config=c.load_drone_config())
    assert "S-VEG" not in entities and "S-ELEV" not in entities


def test_lawn_cells_never_rise_and_holes_are_interpolated() -> None:
    dtm = np.full((40, 40), 5.0)
    xs = np.array([[2.0, 2.0, 5.2]] * 12 + [[2.76, 2.0, 5.4]] * 12)   # two cells with points, two cells between them without
    new, mask, report = lawn_ground_dtm(dtm, (0, 0, 10, 10), 0.25, xs, xs, config=c.load_drone_config())
    assert np.all(new[mask] <= 5.0 + 1e-9)                                              # never higher than before
    assert report["interpolated_cells"] > 0 and np.all(np.isfinite(new))


def test_qa_passes_without_lawn_layers(tmp_path) -> None:
    doc = ezdxf.new("R2018")
    for name in ("S-TREE", "S-PAVE", "S-TOPO-TIN", "S-TOPO-MAJR", "S-TOPO-TEXT"):
        doc.layers.add(name)
    doc.layers.get("S-TOPO-TIN").freeze()
    msp = doc.modelspace()
    msp.add_3dface([(2_600_000, 1_200_000, 430), (2_600_001, 1_200_000, 430), (2_600_000, 1_200_001, 430)], dxfattribs={"layer": "S-TOPO-TIN"})
    msp.add_lwpolyline([(2_600_000 + x, 1_200_000 + y) for x, y in ((0, 0), (5, 0), (5, 4), (0, 4))], close=True, dxfattribs={"layer": "S-PAVE"})
    path = tmp_path / "nolawn.dxf"
    doc.saveas(path)
    results = qa_cad.check_dxf(path)
    assert [r for r in results if not r["ok"]] == []
    assert not any(r["rule"] == "15_count_vegetation_area" for r in results)   # no lawn category is required
