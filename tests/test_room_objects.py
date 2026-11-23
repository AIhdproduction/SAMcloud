"""Wall-hung extents, screens and ceiling objects measured from synthetic points (samcloud.bim.room_objects)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from samcloud.bim.plan_model import BuildingPlan, PlanFurniture, PlanRoom, PlanWall
from samcloud.bim.room_objects import (
    Mirror, RoomObjectSettings, add_room_objects, apply_measured_extents, build_evidence, load_room_evidence,
    load_room_object_settings,
)
from samcloud.bim.room_objects.geometry import box_mesh, prism_mesh

ROOM = box(0.013, 0.013, 4.987, 4.013)
WALLS = [
    LineString(((0.013, 0.013), (4.987, 0.013))), LineString(((4.987, 0.013), (4.987, 4.013))),
    LineString(((4.987, 4.013), (0.013, 4.013))), LineString(((0.013, 4.013), (0.013, 0.013))),
]
CEILING_Z = 2.507
FINE = ["sofa", "television", "computer monitor", "screen", "ceiling lamp", "spotlight", "smoke detector"]
WALL, CEILING, FLOOR, HVAC, EQUIPMENT, FURNITURE = 2, 3, 4, 15, 16, 17


class Cloud:
    """Points of one synthetic room; surfaces are sampled with noise and off any cell border."""

    def __init__(self, seed: int = 11) -> None:
        self.rng = np.random.default_rng(seed)
        self.parts: list[tuple] = []
        self.labels: dict[int, str] = {}

    def add(self, xyz, cls, instance=0, term="", grey=120.0, noise=0.003):
        xyz = np.asarray(xyz, dtype=float) + self.rng.normal(0.0, noise, (len(xyz), 3))
        self.parts.append((xyz, cls, instance, FINE.index(term) + 1 if term else 0, grey))

    def rect(self, origin, a, b, cls, step=0.021, **kwargs):
        origin, a, b = (np.asarray(value, dtype=float) for value in (origin, a, b))
        na, nb = max(2, int(np.linalg.norm(a) / step)), max(2, int(np.linalg.norm(b) / step))
        u, v = np.meshgrid((np.arange(na) + 0.37) / na, (np.arange(nb) + 0.43) / nb)
        self.add(origin + u.reshape(-1, 1) * a + v.reshape(-1, 1) * b, cls, **kwargs)

    def block(self, low, high, cls, skip=(), **kwargs):
        x0, y0, z0 = low
        dx, dy, dz = np.subtract(high, low)
        faces = {
            "bottom": ((x0, y0, z0), (dx, 0, 0), (0, dy, 0)), "top": ((x0, y0, z0 + dz), (dx, 0, 0), (0, dy, 0)),
            "south": ((x0, y0, z0), (dx, 0, 0), (0, 0, dz)), "north": ((x0, y0 + dy, z0), (dx, 0, 0), (0, 0, dz)),
            "west": ((x0, y0, z0), (0, dy, 0), (0, 0, dz)), "east": ((x0 + dx, y0, z0), (0, dy, 0), (0, 0, dz)),
        }
        for name, (origin, a, b) in faces.items():
            if name not in skip:
                self.rect(origin, a, b, cls, **kwargs)

    def disc(self, x, y, z, radius, cls, height=0.0, **kwargs):
        points = np.stack(np.meshgrid(*[np.arange(-radius, radius, 0.013) + 0.0041] * 2), axis=-1).reshape(-1, 2)
        points = points[np.linalg.norm(points, axis=1) <= radius]
        self.add(np.column_stack((x + points[:, 0], y + points[:, 1], np.full(len(points), z))), cls, **kwargs)
        for level in np.arange(0.0, height, 0.017):
            angle = np.arange(0, 2 * math.pi, 0.016 / radius) + 0.01
            rim = np.column_stack((x + radius * np.cos(angle), y + radius * np.sin(angle), np.full(len(angle), z + level)))
            self.add(rim, cls, **kwargs)

    def ceiling(self, slope=0.0):
        self.rect((0.02, 0.02, CEILING_Z), (4.96, 0, 4.96 * slope), (0, 3.99, 0), CEILING, step=0.041)

    def evidence(self, upm=1.0, floor_z=0.0, mirrors=None, depth=None, fine_names=FINE):
        xyz = np.vstack([part[0] for part in self.parts])
        cls, instance, term, grey = (
            np.concatenate([np.full(len(part[0]), part[index]) for part in self.parts]) for index in range(1, 5)
        )
        local = xyz * upm + np.asarray((0.0, 0.0, floor_z))
        return build_evidence(
            local, cls, units_per_metre=upm, floor_z=floor_z, ceiling_z=floor_z + CEILING_Z * upm, instance=instance, fine=term,
            rgb=np.repeat(grey[:, None], 3, axis=1), fine_names=fine_names, instance_labels=self.labels,
            depth_xyz_local=None if depth is None else depth[0], depth_cls=None if depth is None else depth[1], mirrors=mirrors,
        )


def _item(footprint, *, label="shelf", instance=7, upm=1.0, floor_z=0.0, top=1.67, **kwargs):
    return PlanFurniture(
        id=kwargs.pop("id", "F001"), instance_id=instance, label=label, semantic_label=label, catalogue_kind=None,
        layer="A-FURN", footprint=[[x * upm, y * upm] for x, y in footprint], z_min=floor_z, z_max=floor_z + top * upm,
        z_source="instance_bbox", **kwargs,
    )


def _detect(cloud, items=None, *, windows=(), wall_area=None, rooms=(ROOM,), walls=WALLS, **evidence):
    items = [] if items is None else items
    objects, report = add_room_objects(
        items, cloud.evidence(**evidence), rooms=list(rooms), wall_lines=list(walls), wall_area=wall_area,
        settings=RoomObjectSettings(), item_factory=PlanFurniture, window_lines=list(windows),
    )
    return items, objects, report


def _extent(cloud, *items, walls=WALLS, **evidence):
    ev = cloud.evidence(**evidence)
    return apply_measured_extents(list(items), ev, list(walls), RoomObjectSettings().extent)


def _closed(mesh) -> bool:
    directed: dict[tuple[int, int], int] = {}
    for a, b, c in mesh["faces"]:
        for edge in ((a, b), (b, c), (c, a)):
            directed[edge] = directed.get(edge, 0) + 1
    return all(count == 1 and directed.get((b, a)) == 1 for (a, b), count in directed.items())


def _volume(mesh) -> float:
    vertices, faces = np.asarray(mesh["vertices"]), np.asarray(mesh["faces"])
    a, b, c = (vertices[faces[:, index]] for index in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


# --------------------------------------------------------------------------------------- vertical extent


def _shelf_scene(face_y=0.013):
    cloud = Cloud()
    cloud.rect((0.5, face_y, 0.03), (2.2, 0, 0), (0, 0, 2.1), WALL, step=0.027)
    cloud.block((1.113, face_y + 0.004, 1.617), (1.987, 0.263, 1.661), FURNITURE, skip=("south",), instance=7)
    return cloud, box(1.1, 0.0, 2.0, 0.30).exterior.coords[:-1]


@pytest.mark.parametrize("upm,floor_z", [(1.0, 0.0), (0.83, -0.37)])
def test_wall_shelf_keeps_its_measured_bottom(upm, floor_z):
    cloud, footprint = _shelf_scene()
    item = _item(footprint, upm=upm, floor_z=floor_z)
    report = _extent(cloud, item, upm=upm, floor_z=floor_z)
    assert item.mounting == "wall" and report["wall"]
    assert (item.z_min - floor_z) / upm == pytest.approx(1.617, abs=0.015)
    assert (item.z_max - floor_z) / upm == pytest.approx(1.661, abs=0.015)
    assert item.solids[0]["z_min"] == item.z_min and item.fit["plan_footprint"]
    ys = np.asarray(item.footprint)[:, 1] / upm
    assert ys.min() == pytest.approx(0.013, abs=0.03) and ys.max() == pytest.approx(0.263, abs=0.03)


def test_the_rule_needs_no_shelf_name():
    cloud, footprint = _shelf_scene()
    item = _item(footprint, label="furniture")
    _extent(cloud, item)
    assert item.mounting == "wall"


def test_wall_face_is_measured_when_the_plan_face_is_off():
    cloud, footprint = _shelf_scene(face_y=0.187)  # the plan draws the wall at 0.013
    item = _item(footprint)
    _extent(cloud, item)
    assert item.mounting == "wall"
    assert np.asarray(item.footprint)[:, 1].min() == pytest.approx(0.187, abs=0.03)


def test_wardrobe_with_occluded_lower_part_still_stands():
    cloud = Cloud()
    cloud.rect((0.2, 0.013, 0.03), (0.8, 0, 0), (0, 0, 2.2), WALL)  # wall seen beside the wardrobe only
    cloud.block((1.117, 0.03, 0.71), (2.083, 0.60, 2.103), FURNITURE, skip=("south", "bottom"), instance=7)
    item = _item(box(1.1, 0.01, 2.1, 0.61).exterior.coords[:-1], label="wardrobe", top=2.1)
    report = _extent(cloud, item)
    assert item.mounting is None and item.z_min == 0.0
    assert "F001" in report["standing"]


def test_support_to_the_floor_beats_a_visible_wall():
    cloud, footprint = _shelf_scene()
    cloud.block((1.117, 0.03, 0.04), (1.163, 0.09, 0.60), FURNITURE)  # a post of the open shelf unit
    item = _item(footprint)
    _extent(cloud, item)
    assert item.mounting is None


def test_points_of_the_piece_below_are_no_support():
    cloud, footprint = _shelf_scene()
    cloud.block((1.05, 0.12, 0.03), (2.05, 0.9, 0.48), FURNITURE, instance=9)
    bed = _item(box(1.05, 0.12, 2.05, 0.9).exterior.coords[:-1], label="bed", instance=9, id="F002", top=0.48)
    shelf = _item(footprint)
    _extent(cloud, shelf, bed)
    assert shelf.mounting == "wall" and bed.mounting is None and bed.z_min == 0.0


@pytest.mark.parametrize("label", ["sofa", "desk", "chair", "bed", "cat tree", "floor lamp"])
def test_thin_supports_never_float_for_a_visible_wall(label):
    cloud, footprint = _shelf_scene()
    item = _item(footprint, label=label)
    _extent(cloud, item)
    assert item.mounting is None and item.z_min == 0.0


def test_open_wardrobe_with_its_back_board_seen_still_stands():
    cloud = Cloud()
    cloud.rect((1.05, 0.013, 0.03), (1.0, 0, 0), (0, 0, 2.0), WALL)  # the back board, classified as wall
    cloud.block((1.1, 0.06, 0.71), (2.0, 0.55, 2.0), FURNITURE, skip=("south", "bottom"), instance=7)
    cloud.add(np.random.default_rng(3).uniform((1.15, 0.2, 0.15), (1.95, 0.5, 0.62), (600, 3)), 1)  # coats hanging inside
    item = _item(box(1.1, 0.02, 2.0, 0.56).exterior.coords[:-1], label="shelf", top=2.0)
    report = _extent(cloud, item)
    assert item.mounting is None and report["standing"]["F001"]["reason"] == "space below is not empty"


def test_front_in_the_plane_of_the_wall_below_still_stands():
    cloud = Cloud()
    cloud.rect((1.0, 0.013, 0.03), (1.0, 0, 0), (0, 0, 1.7), WALL)  # the lower part of the front, classified as wall
    cloud.rect((1.1, 0.031, 0.74), (0.8, 0, 0), (0, 0, 0.5), FURNITURE, instance=7)
    item = _item(box(1.1, 0.01, 1.9, 0.12).exterior.coords[:-1], label="cabinet", top=1.24)
    _extent(cloud, item)
    assert item.mounting is None


def test_fragment_does_not_lift_a_larger_item():
    cloud, _footprint = _shelf_scene()
    item = _item(box(0.5, 0.0, 2.9, 0.8).exterior.coords[:-1])
    report = _extent(cloud, item)
    assert item.mounting is None and "fragment" in report["standing"]["F001"]["reason"]


def test_wall_shelf_the_top_view_lost_is_added():
    cloud, _footprint = _shelf_scene()
    cloud.labels[7] = "shelf"
    cloud.block((1.0, 0.12, 0.03), (2.1, 0.9, 0.62), FURNITURE, instance=9)
    sofa = _item(box(1.0, 0.12, 2.1, 0.9).exterior.coords[:-1], label="sofa", instance=9, top=0.62)
    items, _, report = _detect(cloud, [sofa])
    assert len(items) == 2 and report["wall_shelves_added"]
    shelf = items[1]
    assert shelf.mounting == "wall" and shelf.fit["added"] == "wall shelf"
    assert shelf.z_min == pytest.approx(1.617, abs=0.015)


def test_shelf_instance_already_in_the_plan_is_not_doubled():
    cloud, footprint = _shelf_scene()
    cloud.labels[7] = "shelf"
    cloud.labels[12] = "shelf"
    cloud.block((1.113, 0.017, 1.65), (1.987, 0.2, 1.70), FURNITURE, instance=12)
    shelf = _item(footprint)
    _extent(cloud, shelf)
    assert len(_detect(cloud, [shelf])[0]) == 1


def test_shelf_instance_without_a_visible_wall_below_is_not_added():
    cloud = Cloud()
    cloud.block((1.113, 0.017, 1.617), (1.987, 0.263, 1.661), FURNITURE, instance=7)
    cloud.labels[7] = "shelf"
    assert not _detect(cloud)[0]


def test_object_resting_on_another_piece():
    cloud = Cloud()
    cloud.block((2.113, 1.517, 0.763), (2.387, 1.783, 1.09), FURNITURE, instance=7)
    table = _item(box(1.7, 1.2, 2.8, 2.1).exterior.coords[:-1], label="table", instance=None, id="F002", top=0.75)
    plant = _item(box(2.1, 1.5, 2.4, 1.8).exterior.coords[:-1], label="plant", top=1.09)
    _extent(cloud, plant, table)
    assert plant.mounting == "resting" and plant.z_min == pytest.approx(0.763, abs=0.02)


# --------------------------------------------------------------------------------------- screens


def _tv(cloud, x=1.02, y=0.072, width=1.22, height=0.69, bottom=0.91, term="television", instance=5):
    cloud.rect((x, y, bottom), (width, 0, 0), (0, 0, height), FURNITURE, term=term, instance=instance, grey=35.0)
    if instance:
        cloud.labels[instance] = "television" if term == "television" else "monitor"


@pytest.mark.parametrize("upm,floor_z", [(1.0, 0.0), (1.19, 0.42)])
def test_tv_on_the_wall_is_one_thin_panel(upm, floor_z):
    cloud = Cloud()
    _tv(cloud)
    rooms = [Polygon(np.asarray(ROOM.exterior.coords) * upm)]
    walls = [LineString(np.asarray(line.coords) * upm) for line in WALLS]
    items, objects, _ = _detect(cloud, upm=upm, floor_z=floor_z, rooms=rooms, walls=walls)
    assert len(items) == 1 and not objects
    screen = items[0]
    assert (screen.semantic_label, screen.mounting, screen.shape_source) == ("television", "wall", "mesh")
    assert screen.fit["width_m"] == pytest.approx(1.22, abs=0.04) and screen.fit["height_m"] == pytest.approx(0.69, abs=0.04)
    assert (screen.z_min - floor_z) / upm == pytest.approx(0.91, abs=0.03)
    assert screen.facing_deg == pytest.approx(90.0, abs=2.0)
    ys = np.asarray(screen.footprint)[:, 1] / upm
    assert ys.min() >= 0.012 and ys.max() - ys.min() <= 0.06  # thin, not inside the wall
    assert _closed(screen.mesh) and _volume(screen.mesh) > 0


@pytest.mark.parametrize("terms,expected", [(("television",), "screen"), (("television", "computer monitor"), "monitor")])
def test_small_television_on_a_desk_is_named_by_its_point_terms(terms, expected):
    cloud = Cloud()
    for term in terms:
        cloud.rect((2.213, 1.687, 0.871), (0.61, 0, 0), (0, 0, 0.36), FURNITURE, term=term, instance=8, grey=40.0)
    cloud.labels[8] = "television"  # what the instance was called before monitors were prompted
    cloud.rect((2.0, 1.4, 0.753), (1.05, 0, 0), (0, 0.65, 0), FURNITURE, instance=3)
    desk = _item(box(2.0, 1.4, 3.05, 2.05).exterior.coords[:-1], label="desk", instance=3, top=0.753)
    items, _, _ = _detect(cloud, [desk])
    screens = [item for item in items if item is not desk]
    assert len(screens) == 1 and screens[0].semantic_label == expected


def test_monitor_with_stand_yaw_and_tilt_on_a_desk():
    cloud = Cloud()
    yaw, tilt = math.radians(27.0), math.radians(8.0)
    u = np.asarray((math.cos(yaw), math.sin(yaw), 0.0))
    back = np.asarray((-math.sin(yaw), math.cos(yaw), 0.0))
    up = math.cos(tilt) * np.asarray((0.0, 0.0, 1.0)) + math.sin(tilt) * back
    middle = np.asarray((2.513, 1.687, 0.871))  # bottom edge
    cloud.rect(middle - 0.30 * u, 0.60 * u, 0.34 * up, FURNITURE, term="computer monitor", instance=8, grey=40.0)
    cloud.labels[8] = "monitor"
    cloud.rect((2.0, 1.4, 0.753), (1.05, 0, 0), (0, 0.65, 0), FURNITURE, instance=3)
    cloud.rect(middle - 0.025 * u + 0.03 * back - (0, 0, 0.11), 0.05 * u, (0, 0, 0.10), FURNITURE, step=0.009, noise=0.001)
    desk = _item(box(2.0, 1.4, 3.05, 2.05).exterior.coords[:-1], label="desk", instance=3, top=0.753)
    items, _, _ = _detect(cloud, [desk])
    screens = [item for item in items if item is not desk]
    assert len(screens) == 1 and desk in items  # not merged into the carrier
    screen = screens[0]
    assert (screen.semantic_label, screen.mounting) == ("monitor", "resting")
    assert screen.fit["width_m"] == pytest.approx(0.60, abs=0.04) and screen.fit["height_m"] == pytest.approx(0.34, abs=0.04)
    assert abs(screen.fit["tilt_deg"]) == pytest.approx(8.0, abs=2.0)
    assert abs(math.sin(math.radians(screen.facing_deg - (27.0 + 90.0)))) < math.sin(math.radians(3.0))
    assert screen.fit["stand"]


def test_foot_in_the_panel_plane_does_not_make_the_panel_taller():
    cloud = Cloud()
    cloud.rect((2.013, 0.30, 0.883), (0.56, 0, 0), (0, 0, 0.32), FURNITURE, term="computer monitor", instance=8)
    cloud.rect((2.263, 0.30, 0.753), (0.06, 0, 0), (0, 0, 0.13), FURNITURE, step=0.011, term="computer monitor", instance=8)
    cloud.rect((1.9, 0.1, 0.75), (0.8, 0, 0), (0, 0.5, 0), FURNITURE)
    cloud.labels[8] = "monitor"
    items, _, _ = _detect(cloud)
    assert items[0].fit["height_m"] == pytest.approx(0.32, abs=0.04)


def test_bezel_of_a_dark_screen_completes_the_panel():
    cloud = Cloud()
    cloud.rect((2.06, 0.10, 0.96), (0.30, 0, 0), (0, 0, 0.21), FURNITURE, step=0.028, term="computer monitor", instance=8)
    cloud.labels[8] = "monitor"
    rng = np.random.default_rng(4)
    along, up = np.arange(1.953, 2.553, 0.008), np.arange(0.913, 1.263, 0.008)
    edges = [np.column_stack((along, np.full(len(along), 0.10), np.full(len(along), z))) for z in (0.917, 1.261)]
    edges += [np.column_stack((np.full(len(up), x), np.full(len(up), 0.10), up)) for x in (1.957, 2.549)]
    depth = np.vstack(edges) + rng.normal(0, 0.001, (2 * len(along) + 2 * len(up), 3))
    items, _, _ = _detect(cloud, depth=(depth, np.full(len(depth), FURNITURE)))
    assert len(items) == 1
    assert items[0].fit["width_m"] == pytest.approx(0.60, abs=0.07) and items[0].fit["height_m"] == pytest.approx(0.35, abs=0.07)


def test_screen_label_without_screen_term_is_rejected():
    cloud = Cloud()
    cloud.rect((1.02, 2.0, 0.11), (1.6, 0, 0), (0, 0, 0.39), FURNITURE, instance=5)  # the foot end of a bed
    cloud.labels[5] = "television"
    items, _, report = _detect(cloud)
    assert not items and "no screen term" in report["screens_rejected"][0]["reason"]


def test_non_planar_hit_is_no_screen():
    cloud = Cloud()
    cloud.add(np.random.default_rng(2).uniform((1.0, 1.0, 0.8), (1.6, 1.6, 1.4), (4000, 3)), FURNITURE, term="television")
    assert not _detect(cloud)[0]


def test_sliver_is_no_screen():
    cloud = Cloud()
    cloud.rect((2.0, 0.07, 0.85), (0.14, 0, 0), (0, 0, 0.41), FURNITURE, term="television")
    assert not _detect(cloud)[0]


def test_screen_seen_in_a_window_pane_is_outside_the_rooms():
    cloud = Cloud()
    _tv(cloud, y=-0.80, instance=0)
    items, _, report = _detect(cloud)
    assert not items and "outside" in report["screens_rejected"][0]["reason"]


def test_mirror_image_of_a_screen_is_dropped():
    cloud = Cloud()
    cloud.rect((4.91, 1.2, 0.93), (0, 1.0, 0), (0, 0, 0.58), FURNITURE, step=0.018, term="television")
    cloud.rect((1.09, 1.2, 0.93), (0, 1.0, 0), (0, 0, 0.58), FURNITURE, step=0.033, term="television")
    cloud.rect((0.80, 1.2, 0.40), (0.35, 0, 0), (0, 1.0, 0), FURNITURE)  # something that could carry it
    cloud.rect((1.09, 1.2, 0.80), (0.1, 0, 0), (0, 1.0, 0), FURNITURE)
    mirror = Mirror(np.asarray((3.0, 1.7, 1.2)), np.asarray((1.0, 0.0, 0.0)))
    assert len(_detect(cloud)[0]) == 2  # two screens without the mirror
    items, _, report = _detect(cloud, mirrors=[mirror])
    assert len(items) == 1 and np.asarray(items[0].footprint)[:, 0].mean() > 4.5
    assert any(entry["reason"] == "mirror reflection" for entry in report["screens_rejected"])


def test_plan_item_of_the_screen_is_replaced_not_doubled():
    cloud = Cloud()
    _tv(cloud)
    old = _item(box(1.0, 0.02, 2.3, 0.2).exterior.coords[:-1], label="television", instance=5)
    items, _, report = _detect(cloud, [old])
    assert len(items) == 1 and items[0] is not old and report["screen_items_replaced"] == ["F001"]


def test_carrier_named_after_its_screen_is_kept():
    cloud = Cloud()
    _tv(cloud)
    desk = _item(box(0.9, 0.02, 2.4, 0.62).exterior.coords[:-1], label="television", instance=None, top=0.75)
    items, _, report = _detect(cloud, [desk])
    assert desk in items and desk.semantic_label == "furniture" and len(items) == 2
    assert report["screen_carriers_renamed"] == ["F001"]


def test_halves_of_one_large_tv_are_one_screen():
    cloud = Cloud()
    _tv(cloud, x=1.013, y=0.07, width=1.12, height=1.1, bottom=0.71, instance=5)
    _tv(cloud, x=2.19, y=0.07, width=0.73, height=1.1, bottom=0.71, instance=6)
    items, _, _ = _detect(cloud)
    assert len(items) == 1 and items[0].fit["width_m"] == pytest.approx(1.907, abs=0.06)


def test_glossy_splashback_with_a_few_screen_terms_is_no_screen():
    cloud = Cloud()
    cloud.rect((2.0, 0.06, 0.93), (0.7, 0, 0), (0, 0, 0.5), 18, step=0.006)
    cloud.rect((2.2, 0.06, 1.0), (0.3, 0, 0), (0, 0, 0.35), FURNITURE, step=0.031, term="television")
    items, _, report = _detect(cloud)
    assert not items and "too little" in report["screens_rejected"][0]["reason"]


def test_two_monitors_side_by_side_stay_two():
    cloud = Cloud()
    _tv(cloud, x=1.013, y=0.10, width=0.55, height=0.32, bottom=0.92, term="computer monitor", instance=5)
    _tv(cloud, x=1.641, y=0.10, width=0.55, height=0.32, bottom=0.92, term="computer monitor", instance=6)
    assert len(_detect(cloud)[0]) == 2


# a flat of two rooms side by side; the plan knows only the left one (x < 3.987), the right one is closed in by walls
_KNOWN = box(0.013, 0.013, 3.987, 4.013)
_RIGHT_WALLS = [
    LineString(((4.113, 0.013), (6.987, 0.013))), LineString(((6.987, 0.013), (6.987, 4.013))),
    LineString(((6.987, 4.013), (4.113, 4.013))),
    # the partition: its faces stop at a doorway (y 1.507 .. 2.393) that leads into the known room
    LineString(((3.987, 0.013), (3.987, 1.507))), LineString(((3.987, 2.393), (3.987, 4.013))),
    LineString(((4.113, 0.013), (4.113, 1.507))), LineString(((4.113, 2.393), (4.113, 4.013))),
]
_LEFT_WALLS = [
    LineString(((0.013, 0.013), (3.987, 0.013))), LineString(((3.987, 4.013), (0.013, 4.013))),
    LineString(((0.013, 4.013), (0.013, 0.013))),
]
_EAST = (6.987, 0.013)  # start of the outer wall on the right
# the wall bodies between the faces, as the plan model passes them
_BODIES = box(-0.187, -0.187, 7.187, 4.213).difference(box(0.013, 0.013, 6.987, 4.013)).union(
    box(3.987, 0.013, 4.113, 1.507)).union(box(3.987, 2.393, 4.113, 4.013))
_OPEN_EAST = _BODIES.difference(box(6.9, 0.4, 7.3, 3.6))  # the outer wall on the right is missing
_PARTITION = box(3.987, -0.187, 4.113, 1.507).union(box(3.987, 2.393, 4.113, 4.213))  # partition reaching the outer faces


def _monitor_on_a_desk(cloud, x=5.213, y=3.607, width=0.613, bottom=0.883, height=0.357):
    cloud.rect((x, y, bottom), (width, 0, 0), (0, 0, height), FURNITURE, term="computer monitor", instance=8, grey=40.0)
    cloud.labels[8] = "monitor"
    cloud.rect((x - 0.31, y - 0.53, 0.753), (width + 0.62, 0, 0), (0, 0.75, 0), FURNITURE, instance=3)


def _screens(items):
    return [item for item in items if (item.fit or {}).get("added") == "screen"]


def test_monitor_in_a_room_the_plan_did_not_recognise_is_kept():
    cloud = Cloud()
    _monitor_on_a_desk(cloud)
    items, _, report = _detect(cloud, rooms=[_KNOWN], walls=_LEFT_WALLS + _RIGHT_WALLS, wall_area=_BODIES)
    screens = _screens(items)
    assert len(screens) == 1 and screens[0].semantic_label == "monitor"
    assert screens[0].fit["width_m"] == pytest.approx(0.613, abs=0.04)
    centre = np.asarray(screens[0].fit["centre"])
    assert centre[0] == pytest.approx(5.52, abs=0.03) and centre[1] == pytest.approx(3.607, abs=0.03)
    # the space is only reported, no room is invented
    (space,) = report["screen_spaces_without_room"]
    assert space["area_m2"] == pytest.approx(2.874 * 4.0, rel=0.03) and space["centre"][0] == pytest.approx(5.55, abs=0.05)


def test_monitor_beyond_an_open_side_stays_outside_the_rooms():
    cloud = Cloud()
    _monitor_on_a_desk(cloud)
    walls = _LEFT_WALLS + [line for line in _RIGHT_WALLS if line.coords[0] != _EAST]  # no outer wall on the right
    items, _, report = _detect(cloud, rooms=[_KNOWN], walls=walls, wall_area=_OPEN_EAST)
    assert not _screens(items) and not report["screen_spaces_without_room"]
    assert any("outside" in entry["reason"] for entry in report["screens_rejected"])


def test_enclosed_spaces_ignore_slivers_pockets_and_gaps():
    from samcloud.bim.room_objects.geometry import enclosed_spaces

    def spaces(walls, rooms=(_KNOWN,), bodies=None):
        return enclosed_spaces(list(rooms), walls, bodies, face_gap_m=0.05, min_area_m2=1.0, min_width_m=0.60)

    assert len(spaces(_LEFT_WALLS + _RIGHT_WALLS, bodies=_BODIES)) == 1
    no_east = [line for line in _RIGHT_WALLS if line.coords[0] != _EAST]
    assert not spaces(_LEFT_WALLS + no_east, bodies=_OPEN_EAST)
    # a 0.40 m strip (1.2 m2) between a room outline and a wall face is a sliver, not a room
    strip = [
        LineString(((0.013, 4.413), (3.987, 4.413))), LineString(((0.013, 4.013), (0.013, 4.413))),
        LineString(((3.987, 4.013), (3.987, 4.413))),
    ]
    assert not spaces(_LEFT_WALLS + strip)
    # a closed 0.7 x 0.9 m pocket is a shaft or a built-in cupboard
    pocket = [
        LineString(((3.987, 0.013), (4.687, 0.013))), LineString(((4.687, 0.013), (4.687, 0.913))),
        LineString(((4.687, 0.913), (3.987, 0.913))),
    ]
    assert not spaces(_LEFT_WALLS + pocket)

    def east_from(y):
        return [line if line.coords[0] != _EAST else LineString(((6.987, y), (6.987, 4.013))) for line in _RIGHT_WALLS]

    # faces that stop 0.08 m short of each other at a corner still close the space; a 0.40 m gap leaves it open
    assert len(spaces(_LEFT_WALLS + east_from(0.093), bodies=_PARTITION)) == 1
    assert not spaces(_LEFT_WALLS + east_from(0.413), bodies=_PARTITION)


@pytest.mark.parametrize("end_x,kept", [(4.957, True), (5.263, False)])
def test_panel_reaching_through_a_wall_across_it_is_refused(end_x, kept):
    cloud = Cloud()
    start = end_x - 0.553
    cloud.rect((start, 2.013, 0.893), (0.553, 0, 0), (0, 0, 0.37), FURNITURE, term="computer monitor", instance=8, grey=40.0)
    cloud.labels[8] = "monitor"
    cloud.rect((start - 0.1, 1.713, 0.753), (4.97 - start + 0.1, 0, 0), (0, 0.55, 0), FURNITURE, instance=3)
    items, _, report = _detect(cloud)
    assert len(_screens(items)) == (1 if kept else 0)
    if not kept:
        assert any(entry["reason"] == "reaches through a wall" for entry in report["screens_rejected"])


def test_wall_tv_with_a_small_yaw_error_is_still_kept():
    cloud = Cloud()
    yaw = math.radians(6.0)  # one end 0.13 m further from the wall: a fit error, not a panel through the wall
    u = np.asarray((math.cos(yaw), math.sin(yaw), 0.0))
    cloud.rect(np.asarray((1.013, 0.021, 0.91)), 1.22 * u, (0, 0, 0.69), FURNITURE, term="television", instance=5, grey=35.0)
    cloud.labels[5] = "television"
    items, _, _ = _detect(cloud)
    assert len(items) == 1 and items[0].semantic_label == "television"


def test_panel_inside_a_wall_body_is_refused_a_flush_tv_is_kept():
    body = box(-0.187, -0.187, 5.187, 0.013)
    inside = Cloud()
    inside.rect((2.013, -0.061, 1.003), (0.47, 0, 0), (0, 0, 0.33), FURNITURE, term="screen", grey=40.0)
    items, _, report = _detect(inside, wall_area=body)
    assert not items and report["screens_rejected"][0]["reason"] == "inside a wall"
    flush = Cloud()
    _tv(flush)
    assert len(_detect(flush, wall_area=body)[0]) == 1


def test_small_panel_at_the_floor_is_refused_a_standing_tv_is_kept():
    small = Cloud()
    small.rect((2.013, 1.513, 0.131), (0.45, 0, 0), (0, 0, 0.39), FURNITURE, term="computer monitor", grey=40.0)
    small.rect((1.9, 1.3, 0.05), (0.7, 0, 0), (0, 0.4, 0), FURNITURE)
    items, _, report = _detect(small)
    assert not items and any(entry["reason"] == "at the floor" for entry in report["screens_rejected"])
    large = Cloud()
    large.rect((1.513, 1.513, 0.153), (1.43, 0, 0), (0, 0, 0.81), FURNITURE, term="television", instance=5, grey=35.0)
    large.labels[5] = "television"
    found = _detect(large)[0]
    assert len(found) == 1 and found[0].semantic_label == "television"  # a large television may stand low on its foot


# --------------------------------------------------------------------------------------- ceiling


def _kinds(objects):
    return sorted(item["semantic_label"] for item in objects)


def test_lamp_spot_detector_and_grille():
    cloud = Cloud()
    cloud.ceiling()
    cloud.disc(1.213, 1.117, CEILING_Z - 0.163, 0.247, EQUIPMENT, height=0.16, grey=252.0)
    cloud.disc(3.117, 1.083, CEILING_Z - 0.021, 0.061, EQUIPMENT, height=0.02, grey=251.0)
    cloud.disc(1.109, 2.917, CEILING_Z - 0.036, 0.066, EQUIPMENT, height=0.035, grey=170.0, term="smoke detector", instance=4)
    cloud.labels[4] = "smoke detector"
    cloud.rect((3.013, 2.817, CEILING_Z - 0.024), (0.31, 0, 0), (0, 0.205, 0), HVAC, step=0.015)
    _, objects, _ = _detect(cloud)
    assert _kinds(objects) == ["ceiling lamp", "smoke detector", "spotlight", "ventilation grille"]
    lamp = next(item for item in objects if item["semantic_label"] == "ceiling lamp")
    assert lamp["z_min"] == pytest.approx(CEILING_Z - 0.163, abs=0.03)
    assert lamp["z_max"] == pytest.approx(CEILING_Z, abs=0.02)  # attached
    width = np.ptp(np.asarray(lamp["footprint"])[:, 0])
    assert width == pytest.approx(0.494, abs=0.06)
    for item in objects:
        assert _closed(item["mesh"]) and _volume(item["mesh"]) > 0 and item["layer"] == "A-CLNG-OBJ"


def test_labelled_lamp_on_an_unseen_neck():
    cloud = Cloud()
    cloud.ceiling()
    cloud.disc(2.513, 2.017, CEILING_Z - 0.23, 0.22, EQUIPMENT, height=0.06, grey=250.0, term="ceiling lamp")
    _, objects, _ = _detect(cloud)
    assert _kinds(objects) == ["ceiling lamp"]
    assert objects[0]["z_max"] < CEILING_Z - 0.12  # the neck is not invented


def test_unlabelled_lamp_that_is_on_is_closed_up_to_the_ceiling():
    cloud = Cloud()
    cloud.ceiling()
    cloud.disc(2.513, 2.017, CEILING_Z - 0.18, 0.24, EQUIPMENT, height=0.075, grey=250.0)
    _, objects, _ = _detect(cloud)
    assert _kinds(objects) == ["ceiling lamp"]
    assert objects[0]["z_max"] == pytest.approx(CEILING_Z, abs=0.02)


def test_works_on_a_cloud_without_instances_and_terms():
    cloud = Cloud()
    cloud.ceiling()
    cloud.disc(2.513, 2.017, CEILING_Z - 0.14, 0.23, EQUIPMENT, height=0.14, grey=250.0)
    _, objects, _ = _detect(cloud, fine_names=[])
    assert _kinds(objects) == ["ceiling lamp"]


def test_object_follows_a_sloped_ceiling():
    cloud = Cloud()
    slope = 0.21
    cloud.ceiling(slope)
    for drop in (0.11, 0.07, 0.03):  # shade and neck of a lamp, all parallel to the slope
        origin = (2.013, 1.517, CEILING_Z + 2.013 * slope - drop)
        cloud.rect(origin, (0.43, 0, 0.43 * slope), (0, 0.37, 0), EQUIPMENT, grey=250.0)
    _, objects, _ = _detect(cloud)
    assert len(objects) == 1
    a, b, c = objects[0]["ceiling_plane"]
    assert a == pytest.approx(slope, abs=0.01) and abs(b) < 0.01
    vertices = np.asarray(objects[0]["mesh"]["vertices"])
    top = vertices[len(vertices) // 2:]
    assert np.allclose(top[:, 2], a * top[:, 0] + b * top[:, 1] + c, atol=0.01)
    assert np.ptp(top[:, 2]) > 0.05


def test_unknown_fixture_with_evidence_is_kept_as_unknown():
    cloud = Cloud()
    cloud.ceiling()
    cloud.rect((2.013, 1.517, CEILING_Z - 0.022), (0.19, 0, 0), (0, 0.18, 0), EQUIPMENT, step=0.014, grey=165.0)
    assert _kinds(_detect(cloud)[1]) == ["ceiling fixture"]


def test_lamp_prompt_on_a_grille_stays_a_grille():
    cloud = Cloud()
    cloud.ceiling()
    cloud.rect((2.013, 1.517, CEILING_Z - 0.022), (0.30, 0, 0), (0, 0.20, 0), HVAC, step=0.015, instance=6, term="ceiling lamp")
    cloud.labels[6] = "smoke detector"
    assert _kinds(_detect(cloud)[1]) == ["ventilation grille"]


def _curtain_rail(cloud):
    cloud.block((1.613, 3.853, CEILING_Z - 0.041), (3.387, 3.897, CEILING_Z - 0.003), EQUIPMENT, skip=("top",))
    return {"windows": [LineString(((1.7, 4.013), (3.3, 4.013)))]}


def _lintel(cloud):
    cloud.block((2.1, 1.9, CEILING_Z - 0.2), (2.9, 2.06, CEILING_Z - 0.01), EQUIPMENT)
    return {"wall_area": box(1.5, 1.88, 3.5, 2.08)}


def _wall_top(cloud):
    cloud.rect((0.113, 0.5, CEILING_Z - 0.02), (0.07, 0, 0), (0, 1.9, 0), EQUIPMENT)
    return {}


def _cabinet_top(cloud):
    cloud.block((2.013, 2.011, 1.95), (2.61, 2.52, 2.30), FURNITURE)
    cloud.rect((2.06, 2.06, 2.305), (0.4, 0, 0), (0, 0.35, 0), EQUIPMENT)
    return {}


def _wall_clock(cloud):
    cloud.rect((1.2, 0.04, 1.75), (0.33, 0, 0), (0, 0, 0.33), EQUIPMENT)
    return {}


def _noise(cloud):
    cloud.add(np.random.default_rng(8).uniform((1, 1, 2.2), (4, 3, 2.49), (120, 3)), EQUIPMENT)
    return {}


def _outside(cloud):
    cloud.disc(5.6, 2.0, CEILING_Z - 0.03, 0.08, EQUIPMENT, grey=250.0)
    return {}


def test_lamp_beside_the_line_of_a_window_is_no_curtain_rail():
    cloud = Cloud()
    cloud.ceiling()
    cloud.block((2.1, 2.0, CEILING_Z - 0.04), (3.4, 2.09, CEILING_Z - 0.003), EQUIPMENT, skip=("top",), grey=250.0)
    assert len(_detect(cloud, windows=[LineString(((1.7, 4.013), (3.3, 4.013)))])[1]) == 1


@pytest.mark.parametrize("build", [_curtain_rail, _lintel, _wall_top, _cabinet_top, _wall_clock, _noise, _outside])
def test_no_ceiling_objects(build):
    cloud = Cloud()
    cloud.ceiling()
    kwargs = build(cloud)
    assert not _detect(cloud, **kwargs)[1]


def test_mirror_image_of_a_lamp_is_dropped():
    cloud = Cloud()
    cloud.ceiling()
    cloud.disc(3.913, 2.017, CEILING_Z - 0.15, 0.25, EQUIPMENT, height=0.15, grey=250.0)
    cloud.disc(2.091, 2.013, CEILING_Z - 0.15, 0.25, EQUIPMENT, height=0.12, grey=250.0)
    cloud.rect((2.05, 1.97, CEILING_Z - 0.04), (0.08, 0, 0), (0, 0.08, 0), EQUIPMENT)
    assert len(_detect(cloud)[1]) == 2
    _, objects, report = _detect(cloud, mirrors=[Mirror(np.asarray((3.0, 2.0, 1.4)), np.asarray((1.0, 0.0, 0.0)))])
    assert len(objects) == 1 and np.asarray(objects[0]["footprint"])[:, 0].mean() > 3.5
    assert any(entry["reason"] == "mirror reflection" for entry in report["ceiling_rejected"])


def test_weak_twin_across_a_wall_is_a_reflection_without_a_mirror_report():
    cloud = Cloud()
    cloud.ceiling()
    cloud.disc(3.913, 2.017, CEILING_Z - 0.15, 0.25, EQUIPMENT, height=0.15, grey=250.0)
    angle = np.arange(0.01, 2 * math.pi, 0.22)
    for drop, radius in ((0.15, 0.25), (0.10, 0.22), (0.04, 0.25)):
        rim = np.column_stack((2.091 + radius * np.cos(angle), 2.013 + radius * np.sin(angle), np.full(len(angle), CEILING_Z - drop)))
        cloud.add(rim, EQUIPMENT, grey=250.0)
        cloud.add(rim + (0.004, 0.004, 0.012), EQUIPMENT, grey=250.0)
    rooms = [box(0.013, 0.013, 2.95, 4.013), box(3.05, 0.013, 4.987, 4.013)]
    walls = [*WALLS, LineString(((2.95, 0.013), (2.95, 4.013))), LineString(((3.05, 0.013), (3.05, 4.013)))]
    _, objects, report = _detect(cloud, rooms=rooms, walls=walls)
    assert len(objects) == 1 and np.asarray(objects[0]["footprint"])[:, 0].mean() > 3.5
    assert any("weak twin" in entry["reason"] for entry in report["ceiling_rejected"])


def test_weak_twin_behind_the_mirror_front_of_a_deep_wardrobe_is_a_reflection():
    cloud = Cloud()
    cloud.ceiling()
    cloud.disc(4.153, 2.017, CEILING_Z - 0.15, 0.25, EQUIPMENT, height=0.15, grey=250.0)
    angle = np.arange(0.01, 2 * math.pi, 0.22)
    for drop, radius in ((0.15, 0.25), (0.10, 0.22), (0.04, 0.25)):
        rim = np.column_stack((1.351 + radius * np.cos(angle), 2.013 + radius * np.sin(angle), np.full(len(angle), CEILING_Z - drop)))
        cloud.add(rim, EQUIPMENT, grey=250.0)
        cloud.add(rim + (0.004, 0.004, 0.012), EQUIPMENT, grey=250.0)
    # the mirror is half way (2.752), the wall 0.87 m behind it
    rooms = [box(0.013, 0.013, 1.783, 4.013), box(1.883, 0.013, 4.987, 4.013)]
    walls = [*WALLS, LineString(((1.783, 0.013), (1.783, 4.013))), LineString(((1.883, 0.013), (1.883, 4.013)))]
    _, objects, report = _detect(cloud, rooms=rooms, walls=walls)
    assert len(objects) == 1 and np.asarray(objects[0]["footprint"])[:, 0].mean() > 3.5
    assert any("weak twin" in entry["reason"] for entry in report["ceiling_rejected"])


def test_real_lamps_in_twin_rooms_both_stay():
    cloud = Cloud()
    cloud.ceiling()
    for x in (2.087, 3.913):
        cloud.disc(x, 2.017, CEILING_Z - 0.15, 0.25, EQUIPMENT, height=0.15, grey=250.0)
    rooms = [box(0.013, 0.013, 2.95, 4.013), box(3.05, 0.013, 4.987, 4.013)]
    assert len(_detect(cloud, rooms=rooms, walls=[*WALLS, LineString(((3.0, 0.013), (3.0, 4.013)))])[1]) == 2


def _lamp_in_the_unrecognised_room():
    cloud = Cloud()
    cloud.ceiling()
    cloud.rect((4.127, 0.031, CEILING_Z), (2.84, 0, 0), (0, 3.96, 0), CEILING, step=0.041)  # the right room's ceiling
    cloud.disc(5.537, 2.019, CEILING_Z - 0.163, 0.247, EQUIPMENT, height=0.16, grey=252.0)
    cloud.disc(1.213, 1.117, CEILING_Z - 0.163, 0.247, EQUIPMENT, height=0.16, grey=252.0)  # a lamp in the known room
    return cloud


def test_ceiling_lamp_in_a_room_the_plan_did_not_recognise_is_kept():
    items, objects, report = _detect(
        _lamp_in_the_unrecognised_room(), rooms=[_KNOWN], walls=_LEFT_WALLS + _RIGHT_WALLS, wall_area=_BODIES,
    )
    assert not items and _kinds(objects) == ["ceiling lamp", "ceiling lamp"]
    right = max(objects, key=lambda item: np.asarray(item["footprint"])[:, 0].mean())
    assert np.asarray(right["footprint"])[:, 0].mean() == pytest.approx(5.537, abs=0.03)
    assert np.ptp(np.asarray(right["footprint"])[:, 0]) == pytest.approx(0.494, abs=0.06)
    assert right["z_min"] == pytest.approx(CEILING_Z - 0.163, abs=0.03)
    assert len(report["screen_spaces_without_room"]) == 1  # reported, no room written


def test_ceiling_lamp_beyond_an_open_side_stays_outside_the_rooms():
    walls = _LEFT_WALLS + [line for line in _RIGHT_WALLS if line.coords[0] != _EAST]
    _, objects, report = _detect(_lamp_in_the_unrecognised_room(), rooms=[_KNOWN], walls=walls, wall_area=_OPEN_EAST)
    assert _kinds(objects) == ["ceiling lamp"] and np.asarray(objects[0]["footprint"])[:, 0].mean() < 4.0
    assert any(entry["reason"] == "outside the rooms" for entry in report["ceiling_rejected"])


def test_depth_only_voxels_never_measure(tmp_path: Path):
    from plyfile import PlyData, PlyElement

    classified = tmp_path / "products" / "classified"
    classified.mkdir(parents=True)
    vertex = np.zeros(1, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("scalar_class", "u1")])
    vertex["scalar_class"] = FLOOR
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(classified / "classified.ply"))
    (classified / "instances.json").write_text(json.dumps({"instances": [{"id": 3, "label": "flat screen tv"}]}), encoding="utf-8")
    (tmp_path / "products" / "depth").mkdir()
    np.savez(
        tmp_path / "products" / "depth" / "depth_points.npz",
        xyz=np.asarray([(1.01, 1.02, 2.03), (1.11, 1.02, 2.03), (1.21, 1.02, 2.03)], dtype=np.float32),
        class_id=np.asarray([16, 16, 16], np.uint8), source=np.asarray([1, 2, 3], np.uint8), n_views=np.asarray([1, 9, 4], np.uint16),
    )
    evidence = load_room_evidence(
        classified / "instances.json", SimpleNamespace(to_local=np.asarray), units_per_metre=1.0, floor_z=0.0, ceiling_z=2.5,
    )
    assert int(evidence.from_depth.sum()) == 1  # COLMAP supported AND seen by two views
    assert evidence.xyz[evidence.from_depth][0, 0] == pytest.approx(1.21)
    assert evidence.instance_labels[3] == "television"


# --------------------------------------------------------------------------------------- settings, plan, IFC, DXF, viewer


def test_config_matches_the_built_in_defaults():
    assert load_room_object_settings() == RoomObjectSettings()


def test_meshes_are_closed_with_outward_normals():
    vertices, faces = box_mesh(np.asarray((1.1, 2.2, 0.7)), [np.eye(3)[1], np.eye(3)[0], np.eye(3)[2]], (0.2, 0.1, 0.3))
    assert _closed({"faces": faces.tolist()}) and _volume({"vertices": vertices, "faces": faces}) == pytest.approx(0.048)
    outline = Polygon([(0.01, 0.02), (1.03, 0.04), (0.98, 0.41), (0.52, 0.39), (0.47, 0.93), (0.03, 0.97)])
    mesh = prism_mesh(outline, lambda xy: np.full(len(xy), 2.0), lambda xy: np.full(len(xy), 2.2))
    assert _closed({"faces": mesh[1].tolist()})
    assert _volume({"vertices": mesh[0], "faces": mesh[1]}) == pytest.approx(0.2 * outline.area, rel=1e-6)


def _plan() -> BuildingPlan:
    cloud, footprint = _shelf_scene()
    cloud.ceiling()
    _tv(cloud, x=3.0)
    cloud.disc(2.513, 2.017, CEILING_Z - 0.15, 0.23, EQUIPMENT, height=0.15, grey=250.0)
    cloud.disc(1.109, 2.917, CEILING_Z - 0.036, 0.066, EQUIPMENT, height=0.035, grey=170.0, term="smoke detector", instance=4)
    cloud.labels[4] = "smoke detector"
    cloud.rect((3.013, 2.817, CEILING_Z - 0.024), (0.31, 0, 0), (0, 0.205, 0), HVAC, step=0.015)
    cloud.rect((4.013, 1.517, CEILING_Z - 0.022), (0.19, 0, 0), (0, 0.18, 0), EQUIPMENT, step=0.014, grey=165.0)
    shelf = _item(footprint)
    _extent(cloud, shelf)
    items, objects, _ = _detect(cloud, [shelf])
    ring = [(-0.087, -0.087), (5.087, -0.087), (5.087, 4.113), (-0.087, 4.113)]
    walls = [
        PlanWall(id=f"W{i + 1:03d}", centreline=[list(ring[i]), list(ring[(i + 1) % 4])], thickness_m=0.20, kind="exterior",
                 face_a=[list(coords) for coords in WALLS[i].coords])
        for i in range(4)
    ]
    room = PlanRoom(
        id="R001", name="Zimmer", polygon={"exterior": [list(p) for p in ROOM.exterior.coords[:-1]], "holes": []},
        area_m2=ROOM.area, label_point=[2.5, 2.0],
    )
    return BuildingPlan(
        project_name="objects", level_id="level_1", level_name="1", units_per_metre=1.0,
        frame={"origin": [0.0, 0.0, 0.0], "scale_factor": 1.0, "rotation": np.eye(3).tolist()}, floor_z=0.0, ceiling_z=CEILING_Z,
        walls=walls, rooms=[room], furniture=items, ceiling_objects=objects,
    )


def test_plan_json_round_trip_and_old_files_still_load():
    plan = _plan()
    data = json.loads(json.dumps(plan.to_dict()))
    again = BuildingPlan.from_dict(data)
    assert again.ceiling_objects == data["ceiling_objects"]
    assert [item.mounting for item in again.furniture] == [item.mounting for item in plan.furniture]
    data.pop("ceiling_objects")
    for item in data["furniture"]:
        item.pop("mounting")
    old = BuildingPlan.from_dict(data)
    assert old.ceiling_objects == [] and all(item.mounting is None for item in old.furniture)


def test_ifc_classes_heights_qa_and_dxf3d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import ifcopenshell
    import ifcopenshell.util.element

    from samcloud.bim import export
    from samcloud.bim.plan_model import save_plan_model
    from samcloud.dxf.writer3d import build_dxf3d_document

    plan = _plan()
    save_plan_model(plan, tmp_path)
    monkeypatch.setattr(export, "plan_model_file", lambda _root, _key: tmp_path / "plan_model.json")

    result = export.export_level(tmp_path, "level_1")

    qa = json.loads(result.qa_json_path.read_text(encoding="utf-8"))
    assert result.ids_failures == 0 and result.express_findings == 0 and qa["summary"].startswith("IFC4 OK")
    assert qa["consistency"]["furniture"] == {"plan": 2, "ifc": 2}
    assert qa["consistency"]["ceiling_objects"] == {"plan": 4, "ifc": 4}
    model = ifcopenshell.open(str(result.ifc_path))
    assert [e.PredefinedType for e in model.by_type("IfcAudioVisualAppliance")] == ["DISPLAY"]
    assert [e.PredefinedType for e in model.by_type("IfcSensor")] == ["SMOKESENSOR"]
    assert [e.PredefinedType for e in model.by_type("IfcAirTerminal")] == ["GRILLE"]
    assert len(model.by_type("IfcLightFixture")) == 1 and len(model.by_type("IfcBuildingElementProxy")) == 1
    shelf = model.by_type("IfcFurniture")[0]
    assert ifcopenshell.util.element.get_psets(shelf)["SAMcloud_Scan"]["ZMin"] == pytest.approx(1.617, abs=0.015)
    layers = build_dxf3d_document(result.ifc_path)[1].layers
    assert {"3D-IfcAudioVisualAppliance", "3D-IfcLightFixture", "3D-IfcSensor", "3D-IfcAirTerminal"} <= set(layers)


def test_2d_dxf_has_a_dashed_overhead_layer(tmp_path: Path):
    from samcloud.bim.furniture_recon.dxf_outline import apply_reconstructed_outlines
    from samcloud.dxf.las_source import Frame
    from samcloud.dxf.result import BackgroundImage, DxfPlanResult
    from samcloud.dxf.writer import build_dxf_document

    plan = _plan()
    (tmp_path / "plan_model.json").write_text(json.dumps(plan.to_dict()), encoding="utf-8")
    result = DxfPlanResult(
        project_name="objects", level_name="1", mode="indoor", profile_name=None, scale_factor=1.0,
        frame=Frame(origin=np.zeros(3), scale_factor=1.0, rotation=np.eye(3)), settings_fingerprint="x",
        layers={"A-FURN": []}, statuses={}, layer_colors={"A-FURN": 34},
        background=BackgroundImage(None, None, (0.0, 0.0, 5.0, 4.0)), fingerprint_inputs={},
    )

    updated = apply_reconstructed_outlines(result, tmp_path)

    assert len(updated.layers["A-CLNG-OBJ"]) == len(plan.ceiling_objects) == 4
    assert len(updated.layers["A-FURN"]) == 1  # the TV, added as an outline
    document = build_dxf_document(updated)
    assert document.layers.get("A-CLNG-OBJ").dxf.linetype == "DASHED"
    assert len(document.modelspace().query('LWPOLYLINE[layer=="A-CLNG-OBJ"]')) == 4
    assert not document.audit().has_errors


def test_viewer_shows_ceiling_objects_while_ceilings_are_hidden():
    from samcloud.bim.ui_data import viewer_config

    def element(ifc_class, layer=None, predefined=None):
        psets = {"SAMcloud_Scan": {"Layer": layer}} if layer else {}
        return {"ifc_class": ifc_class, "predefined_type": predefined, "psets": psets}

    config = viewer_config("level_1", {"elements": {
        "c": element("IfcCovering", predefined="CEILING"), "l": element("IfcLightFixture", "A-CLNG-OBJ"),
        "p": element("IfcBuildingElementProxy", "A-CLNG-OBJ"), "s": element("IfcSensor", "A-CLNG-OBJ"),
        "f": element("IfcLightFixture", "A-FURN"),
    }})
    assert "ceilings" in config["hidden"] and "ceiling_objects" not in config["hidden"]
    assert [config["by_node"][key] for key in "clpsf"] == ["ceilings", "ceiling_objects", "ceiling_objects", "ceiling_objects", "appliances"]


@pytest.mark.parametrize("language", ["de", "en"])
def test_viewer_category_has_a_label(language):
    path = Path(__file__).parents[1] / "samcloud" / "app" / "locales" / language / "ifc.json"
    assert json.loads(path.read_text(encoding="utf-8"))["category"]["ceiling_objects"]
