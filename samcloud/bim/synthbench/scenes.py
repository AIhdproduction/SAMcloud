"""Seeded scene families and the fixed DEV / VALIDATION scene lists of the synthetic indoor bench.

Every family draws all dimensions, positions and options from ``numpy.random.default_rng(seed)``; the same
(family, seed) always gives the identical scene.  The VALIDATION set uses seeds that are never used during
development (rule: do not tune pipeline thresholds on it).

Furniture placement is collision aware: an item is only added when its footprint lies inside a room and does not
intersect other furniture, columns, stairs or door sweeps (items whose z ranges do not overlap may stack, e.g. upper
cabinets over the worktop).  An item that does not fit anywhere is skipped (deterministically).
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from .furniture import CATALOGUE
from .geometry import blockers, collides, item_footprint, item_z, room_polygons, validate_scene
from .spec import (
    BeamSpec, ColumnSpec, CutSpec, FurnitureSpec, OpeningSpec, ProjectionSpec, RoomSeed, ScanSpec, SceneSpec, StairSpec,
    WallSpec,
)


class Builder:
    """Small helper to write scenes in terms of walls, wall-relative openings and wall-relative furniture."""

    def __init__(self, name: str, family: str, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.spec = SceneSpec(name=name, family=family, seed=seed, storey_height=2.5, walls=[])
        self._instance = 0
        self._rooms_cache: list | None = None
        self.skipped: list[str] = []

    def u(self, lo: float, hi: float, digits: int = 3) -> float:
        return float(round(self.rng.uniform(lo, hi), digits))

    def next_instance(self) -> int:
        self._instance += 1
        return self._instance

    # -- walls -----------------------------------------------------------------------------------------------------
    def wall(self, p0, p1, thickness: float, exterior: bool = False) -> str:
        wid = f"W{len(self.spec.walls) + 1:02d}"
        self.spec.walls.append(WallSpec(wid, (float(round(p0[0], 4)), float(round(p0[1], 4))), (float(round(p1[0], 4)), float(round(p1[1], 4))), float(round(thickness, 3)), exterior))
        self._rooms_cache = None
        return wid

    def ring(self, points, thickness, exterior: bool = True) -> list[str]:
        ids = []
        for i, p in enumerate(points):
            q = points[(i + 1) % len(points)]
            t = thickness[i] if isinstance(thickness, (list, tuple)) else thickness
            ids.append(self.wall(p, q, t, exterior))
        return ids

    def rect(self, x0, y0, x1, y1, thickness, exterior: bool = True) -> list[str]:
        """Walls around the interior rectangle [x0,x1]x[y0,y1]; returns [south, east, north, west]."""
        h = thickness / 2.0
        return self.ring([(x0 - h, y0 - h), (x1 + h, y0 - h), (x1 + h, y1 + h), (x0 - h, y1 + h)], thickness, exterior)

    def room(self, name: str, point, ceiling: float) -> None:
        self.spec.rooms.append(RoomSeed(name, (float(point[0]), float(point[1])), float(round(ceiling, 3))))

    def rooms(self) -> list:
        if self._rooms_cache is None:
            self._rooms_cache = room_polygons(self.spec)
        return self._rooms_cache

    def along(self, wall_id: str, point) -> float:
        """Distance from ``wall.p0`` of the projection of ``point`` onto the wall line."""
        wall = self.spec.wall(wall_id)
        return float(np.dot(np.subtract(point, wall.p0), wall.direction()))

    # -- openings --------------------------------------------------------------------------------------------------
    def opening(self, kind: str, wall_id: str, offset: float, width: float, sill: float, head: float, *, swing: int = 1,
                hinge: str | None = None, leaf: float | None = None) -> str:
        oid = f"O{len(self.spec.openings) + 1:02d}"
        inst = self.next_instance() if kind in ("door", "window") else 0
        hinge = hinge or str(self.rng.choice(["left", "right"]))
        leaf_angle = float(leaf if leaf is not None else self.u(70, 100, 1))
        self.spec.openings.append(OpeningSpec(oid, kind, wall_id, float(round(offset, 3)), float(round(width, 3)), float(round(sill, 3)),
                                              float(round(head, 3)), int(swing), hinge, leaf_angle, inst))
        return oid

    def _offset(self, wall_id: str, width: float, frac: float | None, at=None, margin: float = 0.15) -> float:
        wall = self.spec.wall(wall_id)
        lo, hi = margin + width / 2 + wall.thickness / 2, wall.length - margin - width / 2 - wall.thickness / 2
        if at is not None:
            return float(min(max(self.along(wall_id, at), lo), hi))
        return lo + (hi - lo) * (frac if frac is not None else self.u(0.1, 0.9))

    def door(self, wall_id: str, frac: float | None = None, width: float | None = None, swing: int = 1, at=None) -> str:
        width = width or self.u(0.78, 0.95)
        return self.opening("door", wall_id, self._offset(wall_id, width, frac, at), width, 0.0, self.u(2.0, 2.12), swing=swing)

    def window(self, wall_id: str, frac: float | None = None, width: float | None = None, sill: float | None = None,
               head: float | None = None, at=None) -> str:
        width = width or self.u(0.9, 2.0)
        wall = self.spec.wall(wall_id)
        width = min(width, wall.length - wall.thickness - 0.5)
        sill = sill if sill is not None else self.u(0.45, 1.0)
        head = head if head is not None else min(self.spec.storey_height - 0.2, self.u(2.05, 2.25))
        return self.opening("window", wall_id, self._offset(wall_id, width, frac, at, 0.2), width, sill, head)

    def passage(self, wall_id: str, frac: float | None = None, width: float | None = None, at=None) -> str:
        width = width or self.u(0.9, 1.4)
        self.u(2.0, 2.2)  # keep the rng stream identical to the former lintel height draw
        # A passage has no lintel: it is free up to the ceiling (rule: opening with a lintel below the ceiling = door).
        head = max([self.spec.storey_height] + [r.ceiling_h for r in self.spec.rooms])
        return self.opening("passage", wall_id, self._offset(wall_id, width, frac, at), width, 0.0, head)

    # -- furniture -------------------------------------------------------------------------------------------------
    def draw(self, kind: str, **kwargs) -> tuple[list, dict]:
        return CATALOGUE[kind][2](self.rng, **kwargs)

    def try_add(self, kind: str, parts: list, center, yaw_deg: float, clearance: float = 0.0) -> FurnitureSpec | None:
        label, class_idx, _generator = CATALOGUE[kind]
        fid = f"F{len(self.spec.furniture) + 1:02d}"
        item = FurnitureSpec(fid, label, class_idx, (float(round(center[0], 4)), float(round(center[1], 4))), float(round(yaw_deg, 3) % 360.0), parts, 0, kind)
        footprint = item_footprint(item)
        free = unary_union([room["polygon"] for room in self.rooms()])
        if not free.buffer(0.005).contains(footprint):
            return None
        if collides(footprint, item_z(item), blockers(self.spec), clearance):
            return None
        item.instance_id = self.next_instance()
        self.spec.furniture.append(item)
        return item

    def pose_against(self, wall_id: str, along: float, toward, depth: float, gap: float = 0.02) -> tuple[tuple[float, float], float]:
        wall = self.spec.wall(wall_id)
        d, n = wall.direction(), wall.normal()
        side = 1.0 if float(np.dot(np.subtract(toward, wall.p0), n)) > 0 else -1.0
        centre = np.asarray(wall.p0, dtype=float) + d * along + n * side * (wall.thickness / 2 + gap + depth / 2)
        back = -n * side
        return (float(centre[0]), float(centre[1])), math.degrees(math.atan2(-back[0], back[1]))

    def against(self, kind: str, wall_id: str, at: Iterable, toward, gap: float = 0.02, parts=None, dims=None, **kwargs) -> FurnitureSpec | None:
        """Back of the item to the face of ``wall_id`` (side of ``toward``); try the positions ``at`` in order."""
        if parts is None:
            parts, dims = self.draw(kind, **kwargs)
        for point in at:
            along = self.along(wall_id, point) if isinstance(point, (tuple, list)) else float(point)
            centre, yaw = self.pose_against(wall_id, along, toward, dims["d"], gap)
            item = self.try_add(kind, parts, centre, yaw)
            if item is not None:
                return item
        self.skipped.append(kind)
        return None

    def free(self, kind: str, room_name: str, yaw: float | None = None, tries: int = 80, clearance: float = 0.12, **kwargs) -> FurnitureSpec | None:
        parts, _dims = self.draw(kind, **kwargs)
        room = next(r for r in self.rooms() if r["name"] == room_name)
        minx, miny, maxx, maxy = room["polygon"].bounds
        for _ in range(tries):
            c = (self.u(minx, maxx), self.u(miny, maxy))
            if not room["polygon"].contains(Point(c)):
                continue
            item = self.try_add(kind, parts, c, yaw if yaw is not None else self.u(0, 360, 1), clearance)
            if item is not None:
                return item
        self.skipped.append(kind)
        return None

    def group_table(self, kind: str, centre, yaw: float, chairs: int = 4, chair_kind: str = "chair", gap: float = 0.04) -> None:
        tparts, td = self.draw(kind)
        table = self.try_add(kind, tparts, centre, yaw)
        if table is None:
            self.skipped.append(kind)
            return
        a = math.radians(yaw)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        if kind == "table_round":
            for k in range(chairs):
                cparts, cd = self.draw(chair_kind)
                ang = a + 2 * math.pi * k / chairs
                r = td["w"] / 2 + cd["d"] / 2 + gap
                c = (centre[0] + r * math.cos(ang), centre[1] + r * math.sin(ang))
                self.try_add(chair_kind, cparts, c, math.degrees(ang) - 90.0)
            return
        slots = [(-0.25, 1), (0.25, 1), (-0.25, -1), (0.25, -1)][:chairs]
        for sx, sy in slots:
            cparts, cd = self.draw(chair_kind)
            local = np.array([sx * td["w"], sy * (td["d"] / 2 + cd["d"] / 2 + gap)])
            self.try_add(chair_kind, cparts, np.asarray(centre) + rot @ local, yaw + (0.0 if sy > 0 else 180.0))

    # -- structure -------------------------------------------------------------------------------------------------
    def column(self, center, shape: str, size) -> None:
        cid = f"C{len(self.spec.columns) + 1:02d}"
        self.spec.columns.append(ColumnSpec(cid, (float(center[0]), float(center[1])), shape, (float(size[0]), float(size[1])), 0.0, self.next_instance()))

    def beam(self, p0, p1, width: float, depth: float) -> None:
        self.spec.beams.append(BeamSpec(f"B{len(self.spec.beams) + 1:02d}", (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])), float(width), float(depth)))

    def stair(self, start, yaw_deg: float, width: float, steps: int, rise: float, run: float) -> None:
        self.spec.stairs.append(StairSpec(f"S{len(self.spec.stairs) + 1:02d}", (float(start[0]), float(start[1])), float(yaw_deg), float(width), int(steps), float(rise), float(run)))

    def finish(self, scan: ScanSpec, features: list[str]) -> SceneSpec:
        self.spec.scan = scan
        self.spec.features = features + ([f"skipped (no space): {', '.join(self.skipped)}"] if self.skipped else [])
        self.spec.world_yaw_deg = self.u(0.0, 360.0, 2)
        self.spec.world_offset = (self.u(-6.0, 6.0), self.u(-6.0, 6.0), self.u(-0.3, 0.3))
        return self.spec


def _noise(b: Builder, choices=(0.004, 0.006, 0.010, 0.014)) -> float:
    return float(b.rng.choice(choices))


# ---------------------------------------------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------------------------------------------

def rect_basic(seed: int) -> SceneSpec:
    b = Builder(f"rect_basic_{seed}", "rect_basic", seed)
    b.spec.storey_height = b.u(2.4, 2.7, 2)
    W, D, t = b.u(4.6, 6.2, 2), b.u(3.6, 4.8, 2), b.u(0.2, 0.35, 2)
    s, e, n, w = b.rect(0, 0, W, D, t)
    b.room("Wohnen", (W / 2, D / 2), b.spec.storey_height)
    b.door(s, at=(W * b.u(0.7, 0.85), 0), swing=1)
    b.window(n, at=(W * b.u(0.35, 0.65), D))
    if b.rng.random() < 0.6:
        b.window(e, width=b.u(0.8, 1.2), sill=b.u(0.4, 0.6))
    mid = (W / 2, D / 2)
    b.against("sofa", w, at=[(0, D * f) for f in (0.6, 0.55, 0.5, 0.65)], toward=mid)
    b.against("shelf", n, at=[(x, D) for x in (0.7, 1.0, 1.4)], toward=mid)
    yaw = float(b.rng.choice([0.0, 90.0, b.u(10, 35)]))
    b.group_table(str(b.rng.choice(["table_legs", "table_pedestal"])), (W * b.u(0.58, 0.64), D * b.u(0.45, 0.52)), yaw)
    b.free("plant", "Wohnen", yaw=0.0)
    b.free("armchair_round", "Wohnen")
    return b.finish(ScanSpec(noise_m=_noise(b)), ["rectangular room", "door", "windows", "table+chairs", "sofa with arms", "shelf", "plant", "round armchair", "rotated table"])


def l_kitchen(seed: int) -> SceneSpec:
    b = Builder(f"l_kitchen_{seed}", "l_kitchen", seed)
    b.spec.storey_height = b.u(2.45, 2.65, 2)
    W, D = b.u(6.0, 7.0, 2), b.u(5.0, 6.0, 2)
    cw, cd = b.u(2.2, 2.8, 2), b.u(1.8, 2.4, 2)  # cut-out corner (north-east)
    t = b.u(0.25, 0.35, 2)
    h = t / 2
    ring = [(-h, -h), (W + h, -h), (W + h, D - cd + h), (W - cw + h, D - cd + h), (W - cw + h, D + h), (-h, D + h)]
    walls = b.ring(ring, t)  # 0 south, 1 east, 2 step (faces south), 3 step (faces east), 4 north, 5 west
    south, east, step_s, step_e, north, west = walls
    b.room("Wohnen / Küche", (1.5, 1.5), b.spec.storey_height)
    b.door(south, at=(W * b.u(0.25, 0.45), 0))
    b.window(north, at=((W - cw) * 0.6, D), width=min(1.4, (W - cw) * 0.35))
    b.window(east, width=b.u(1.0, 1.6))
    mid = (W * 0.4, D * 0.4)
    run = b.u(2.4, 3.0, 2)
    y_top = D - b.u(0.3, 0.6)  # north end of the counter
    cparts, cdims = b.draw("kitchen_counter", w=run)
    counter = b.against("kitchen_counter", west, at=[(0, y_top - run / 2)], toward=mid, parts=cparts, dims=cdims)
    if counter is not None:
        for kind, pos in (("kitchen_sink", 0.3), ("stove", 0.72)):
            p, dd = b.draw(kind)
            b.against(kind, west, at=[(0, y_top - run * pos)], toward=mid, gap=0.02 + (cdims["d"] - dd["d"]) / 2, parts=p, dims=dd)
        b.against("refrigerator", west, at=[(0, y_top - run - 0.31)], toward=mid)
        p, dd = b.draw("upper_cabinets", w=round(run * 0.8, 3))
        b.against("upper_cabinets", west, at=[(0, y_top - run / 2)], toward=mid, parts=p, dims=dd)
    b.group_table("table_round", (W * 0.5, D * 0.35), b.u(0, 120), chairs=3, chair_kind=str(b.rng.choice(["chair", "chair_arms"])))
    b.against("sideboard", south, at=[(W * f, 0) for f in (0.8, 0.7, 0.6)], toward=mid)
    return b.finish(ScanSpec(noise_m=_noise(b)), ["L-shaped room", "kitchen run (counter+sink+hob+fridge+upper cabinets)", "round table", "rotated chairs"])


def bedroom_slanted(seed: int) -> SceneSpec:
    b = Builder(f"bedroom_slanted_{seed}", "bedroom_slanted", seed)
    b.spec.storey_height = b.u(2.4, 2.6, 2)
    Wa, Wb, D = b.u(3.8, 4.6, 2), b.u(3.0, 3.8, 2), b.u(4.0, 4.8, 2)
    te, ti = b.u(0.3, 0.4, 2), b.u(0.10, 0.15, 2)
    angle = b.u(30, 60, 1)
    cut = b.u(1.4, 2.0, 2)
    h = te / 2
    x_split = Wa + ti / 2
    total = Wa + ti + Wb
    dy = min(cut * math.tan(math.radians(angle)), D * 0.6)
    ring = [(-h, -h), (total + h, -h), (total + h, D + h - dy), (total + h - cut, D + h), (-h, D + h)]
    south, east, slanted, north, west = b.ring(ring, te)
    mid_wall = b.wall((x_split, -h), (x_split, D + h), ti)
    b.room("Schlafen", (Wa / 2, D / 2), b.spec.storey_height)
    b.room("Arbeiten", (x_split + Wb / 2, D / 3), b.spec.storey_height)
    b.door(mid_wall, at=(x_split, D * b.u(0.15, 0.3)), width=b.u(0.78, 0.86), swing=-1)
    b.door(south, at=(x_split + Wb * b.u(0.2, 0.4), 0))
    b.window(slanted, frac=0.5, width=round(min(1.0, b.spec.wall(slanted).length * 0.4), 3))
    b.window(north, at=(Wa * b.u(0.3, 0.6), D), width=b.u(0.9, 1.3))
    ca, cb = (Wa / 2, D / 2), (x_split + Wb / 2, D / 3)
    bw = float(b.rng.choice([1.4, 1.6, 1.8]))
    bparts, bd = b.draw("bed", w=bw)
    bed = b.against("bed", west, at=[(0, D * f) for f in (0.5, 0.55, 0.45, 0.6)], toward=ca, gap=0.01, parts=bparts, dims=bd)
    if bed is not None:
        for side in (-1, 1):
            p, nd = b.draw("nightstand")
            yc = bed.center[1] + side * (bw / 2 + 0.04 + nd["w"] / 2 + 0.02)
            b.against("nightstand", west, at=[(0, yc)], toward=ca, gap=0.01, parts=p, dims=nd)
    b.against("wardrobe", south, at=[(Wa * f, 0) for f in (0.6, 0.5, 0.7, 0.4)], toward=ca, w=b.u(1.2, 1.8))
    desk = b.against("desk", mid_wall, at=[(x_split, D * f) for f in (0.7, 0.75, 0.6)], toward=cb)
    if desk is not None:
        cparts, cdm = b.draw("chair_arms")
        a = math.radians(desk.yaw_deg)
        front = np.array([math.sin(a), -math.cos(a)])
        b.try_add("chair_arms", cparts, np.asarray(desk.center) + front * (0.38 + cdm["d"] / 2 + 0.05), desk.yaw_deg)
    b.against("shelf", east, at=[(total, y) for y in (0.9, 1.3, 1.7)], toward=cb)
    b.against("sofa_noarms", south, at=[(x_split + Wb * f, 0) for f in (0.75, 0.7, 0.65)], toward=cb, w=b.u(1.5, 1.8))
    return b.finish(ScanSpec(noise_m=_noise(b)), [f"slanted wall {angle:.0f} deg", "window in slanted wall", "thin interior wall", "bed with headboard", "nightstands", "wardrobe", "desk + armchair", "sofa without arms"])


def apartment(seed: int) -> SceneSpec:
    b = Builder(f"apartment_{seed}", "apartment", seed)
    hs = sorted([b.u(2.35, 2.45, 2), b.u(2.5, 2.6, 2), b.u(2.65, 2.85, 2)])
    b.spec.storey_height = hs[1]
    te = b.u(0.35, 0.4, 2)
    t_bath, t_bed, t_corr = 0.10, b.u(0.15, 0.2, 2), b.u(0.2, 0.25, 2)
    Wb, Wr, Wl = b.u(2.0, 2.4, 2), b.u(3.2, 3.8, 2), b.u(4.5, 5.5, 2)
    Dn, Dc = b.u(3.0, 3.6, 2), b.u(1.3, 1.6, 2)
    Dl = Dn + Dc + t_corr
    h = te / 2
    x1 = Wb + t_bath / 2
    x2 = Wb + t_bath + Wr + t_bed / 2
    xe = Wb + t_bath + Wr + t_bed + Wl
    y1 = Dc + t_corr / 2
    s, e, n, w = b.ring([(-h, -h), (xe + h, -h), (xe + h, Dl + h), (-h, Dl + h)], te)
    w_bath = b.wall((x1, y1), (x1, Dl + h), t_bath)
    w_bed = b.wall((x2, -h), (x2, Dl + h), t_bed)
    w_corr = b.wall((-h, y1), (x2, y1), t_corr)
    b.room("Bad", (Wb / 2, y1 + Dn / 2), hs[0])
    b.room("Schlafen", (x1 + Wr / 2, y1 + Dn / 2), hs[1])
    b.room("Gang", (Wb, Dc / 2), hs[0])
    b.room("Wohnen", (x2 + Wl / 2, Dl / 2), hs[2])
    b.door(w_corr, at=(Wb * 0.62, y1), width=0.7, swing=-1)
    b.door(w_corr, at=(x1 + Wr * 0.3, y1), width=b.u(0.8, 0.9), swing=1)
    b.passage(w_bed, at=(x2, Dc / 2), width=round(min(Dc - 0.25, b.u(0.9, 1.2)), 3))
    b.door(s, at=(Wb * 0.5, 0), width=1.0, swing=1)
    b.window(n, at=(Wb * 0.4, Dl), width=0.6, sill=1.4, head=2.0)
    b.window(n, at=(x1 + Wr * 0.5, Dl), width=b.u(1.2, 1.6), sill=b.u(0.8, 1.0), head=2.1)
    b.window(e, at=(xe, Dl * 0.3), width=b.u(1.0, 1.4), sill=b.u(0.6, 0.9), head=2.2)
    b.window(e, at=(xe, Dl * 0.75), width=b.u(1.8, 2.2), sill=0.05, head=min(hs[2] - 0.15, 2.3))
    cbath = (Wb / 2, y1 + Dn / 2)
    b.against("bathtub", n, at=[(Wb / 2, Dl)], toward=cbath, gap=0.01)
    b.against("toilet", w_bath, at=[(x1, Dl - 1.2), (x1, Dl - 1.0)], toward=cbath, gap=0.01)
    b.against("washbasin", w_bath, at=[(x1, Dl - 2.0), (x1, Dl - 1.8), (x1, y1 + 0.5)], toward=cbath, gap=0.01)
    b.against("shower_tray", w, at=[(0, y1 + 0.6), (0, y1 + 0.75)], toward=cbath, gap=0.01)
    cbed = (x1 + Wr / 2, y1 + Dn / 2)
    b.against("bed", n, at=[(x1 + Wr * f, Dl) for f in (0.6, 0.5, 0.7)], toward=cbed, gap=0.01, w=float(b.rng.choice([1.4, 1.6])))
    b.against("wardrobe", w_bed, at=[(x2, y1 + 0.8), (x2, y1 + 1.0)], toward=cbed, w=b.u(1.0, 1.4))
    cl = (x2 + Wl / 2, Dl / 2)
    b.against("sofa_l", w_bed, at=[(x2, Dl * f) for f in (0.6, 0.65, 0.55)], toward=cl, gap=0.02)
    b.against("tv_stand", e, at=[(xe, Dl * f) for f in (0.5, 0.55)], toward=cl)
    b.free("coffee_table", "Wohnen")
    b.free("plant", "Wohnen", yaw=0.0)
    b.free("armchair_round", "Wohnen")
    return b.finish(ScanSpec(noise_m=_noise(b)), ["4 rooms incl. corridor", "wall thickness 0.10/0.15-0.20/0.20-0.25/0.35-0.40", "3 ceiling heights", "passage without door", "high bathroom window", "french window (sill 0.05)", "bathroom set", "L sofa", "bed"])


def structure(seed: int) -> SceneSpec:
    b = Builder(f"structure_{seed}", "structure", seed)
    b.spec.storey_height = b.u(2.7, 2.9, 2)
    W, D, t = b.u(7.0, 8.0, 2), b.u(5.6, 6.4, 2), b.u(0.3, 0.38, 2)
    s, e, n, w = b.rect(0, 0, W, D, t)
    b.room("Halle", (W / 2, D / 2), b.spec.storey_height)
    b.door(s, at=(W * b.u(0.15, 0.3), 0))
    b.window(n, at=(W * b.u(0.65, 0.85), D))
    b.window(e, at=(W, D * b.u(0.3, 0.6)))
    cx, cy = W * b.u(0.47, 0.53), D * b.u(0.45, 0.55)
    if b.rng.random() < 0.5:
        b.column((cx, cy), "box", (b.u(0.25, 0.4), b.u(0.25, 0.4)))
    else:
        r = b.u(0.15, 0.22)
        b.column((cx, cy), "cyl", (r, r))
    b.beam((cx, 0.0), (cx, D), b.u(0.2, 0.3), b.u(0.25, 0.4))
    steps = int(b.rng.integers(10, 14))
    rise = round(b.spec.storey_height / (steps + 1), 3)
    run, sw = b.u(0.25, 0.29), b.u(0.85, 1.0)
    b.stair((0.25, D - sw / 2 - 0.01), 0.0, sw, steps, rise, run)
    nz0, nz1 = b.u(0.8, 1.0), b.u(1.8, 2.0)
    ny = D * b.u(0.3, 0.45)
    nd = min(t - 0.1, 0.18)
    b.spec.cuts.append(CutSpec("N1", [(-nd, ny - 0.4), (0.02, ny - 0.4), (0.02, ny + 0.4), (-nd, ny + 0.4)], nz0, nz1))
    px = W * b.u(0.6, 0.72)
    b.spec.projections.append(ProjectionSpec("P1", [(px - 0.2, -0.05), (px + 0.2, -0.05), (px + 0.2, 0.2), (px - 0.2, 0.2)]))
    b._rooms_cache = None
    b.group_table("table_trestle", (W * 0.25 + 0.4, D * 0.38), 0.0)
    b.free("table_pedestal", "Halle")
    b.against("sofa_l", e, at=[(W, D * f) for f in (0.62, 0.7, 0.55)], toward=(W / 2, D / 2), gap=0.02)
    return b.finish(ScanSpec(noise_m=_noise(b)), ["free-standing column", "ceiling beam (soffit drop)", "straight stair", "wall niche", "wall projection", "trestle table", "pedestal table", "L sofa", "thick walls"])


def irregular(seed: int) -> SceneSpec:
    b = Builder(f"irregular_{seed}", "irregular", seed)
    b.spec.storey_height = b.u(2.4, 2.6, 2)
    t = b.u(0.22, 0.32, 2)
    k = int(b.rng.choice([5, 6]))
    base = np.linspace(0, 2 * math.pi, k, endpoint=False) + b.u(0, 0.5)
    base = base + np.array([b.u(-0.15, 0.15) for _ in range(k)])
    radii = [b.u(3.0, 4.0) for _ in range(k)]
    ring = [(r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, base)]
    walls = b.ring(ring, t)
    b.room("Atelier", (0.0, 0.0), b.spec.storey_height)
    lengths = [b.spec.wall(wid).length for wid in walls]
    order = [int(i) for i in np.argsort(lengths)[::-1]]
    b.door(walls[order[2]], frac=0.5)
    b.window(walls[order[0]], frac=0.5, width=min(1.6, lengths[order[0]] * 0.4))
    b.window(walls[order[1]], frac=0.5, width=min(1.2, lengths[order[1]] * 0.35))
    b.against("shelf", walls[order[1]], at=[lengths[order[1]] * f for f in (0.22, 0.78, 0.3)], toward=(0, 0))
    b.against("sideboard", walls[order[0]], at=[lengths[order[0]] * f for f in (0.78, 0.22, 0.7)], toward=(0, 0), w=1.4)
    b.free("desk", "Atelier", yaw=b.u(10, 40))
    b.free("plant", "Atelier", yaw=0.0)
    b.free("armchair_round", "Atelier")
    vx, vy = ring[order[3] % k]
    cx, cy = vx * 0.75, vy * 0.75
    unscanned = [[(round(cx - 0.8, 3), round(cy - 0.8, 3)), (round(cx + 0.8, 3), round(cy - 0.8, 3)), (round(cx + 0.8, 3), round(cy + 0.8, 3)), (round(cx - 0.8, 3), round(cy + 0.8, 3))]]
    scan = ScanSpec(noise_m=_noise(b, (0.006, 0.010)), label_noise=0.03, unscanned=unscanned)
    return b.finish(scan, [f"irregular {k}-gon room", "non-orthogonal walls", "unscanned area", "label noise 3%", "rotated desk", "round armchair", "plant"])


def dense(seed: int) -> SceneSpec:
    b = Builder(f"dense_{seed}", "dense", seed)
    b.spec.storey_height = b.u(2.45, 2.6, 2)
    W, D, te, ti = b.u(5.0, 6.0, 2), b.u(4.0, 4.6, 2), b.u(0.25, 0.3, 2), b.u(0.12, 0.16, 2)
    Ws = b.u(1.6, 2.0, 2)
    h = te / 2
    xs = W + ti / 2
    s, e, n, w = b.ring([(-h, -h), (W + ti + Ws + h, -h), (W + ti + Ws + h, D + h), (-h, D + h)], te)
    mid_wall = b.wall((xs, -h), (xs, D + h), ti)
    b.room("Wohnen", (W / 2, D / 2), b.spec.storey_height)
    b.room("Abstellraum", (xs + Ws / 2, D / 2), b.spec.storey_height)
    b.door(s, at=(W * b.u(0.08, 0.2), 0))
    b.door(mid_wall, at=(xs, D * b.u(0.15, 0.3)), width=0.8, swing=1)
    b.window(n, at=(W * b.u(0.65, 0.85), D))
    c = (W / 2, D / 2)
    x = 0.02
    for kind, kw in (("wardrobe", {"w": b.u(1.0, 1.2)}), ("shelf", {"w": b.u(0.8, 1.0)}), ("sideboard", {"w": b.u(1.2, 1.5)})):
        parts, dd = b.draw(kind, **kw)
        b.against(kind, n, at=[(x + dd["w"] / 2, D)], toward=c, gap=0.01, parts=parts, dims=dd)
        x += dd["w"] + 0.01
    bparts, bd = b.draw("bed", w=0.9)
    ns_w = 0.45
    yb = D - 0.3 - ns_w - 0.01 - bd["w"] / 2
    bed = b.against("bed", w, at=[(0, yb), (0, yb - 0.2)], toward=c, gap=0.01, parts=bparts, dims=bd)
    if bed is not None:
        p, nd = b.draw("nightstand")
        b.against("nightstand", w, at=[(0, bed.center[1] + bd["w"] / 2 + 0.01 + nd["w"] / 2)], toward=c, gap=0.01, parts=p, dims=nd)
    tparts, td = b.draw("table_legs", w=1.4, d=0.8)
    tc = (W * 0.62, D * 0.38)
    if b.try_add("table_legs", tparts, tc, 0.0) is not None:
        for sx in (-0.25, 0.25):
            cparts, cd = b.draw("chair")
            b.try_add("chair", cparts, (tc[0] + sx * 1.4, tc[1] + 0.4 + cd["d"] / 2 + 0.02), 0.0)
        cparts, cd = b.draw("chair")
        b.try_add("chair", cparts, (tc[0] + 0.7 + cd["d"] / 2 + 0.02, tc[1]), 90.0)
    b.against("shelf", e, at=[(W + ti + Ws, D * f) for f in (0.6, 0.5, 0.75)], toward=(xs + Ws / 2, D / 2), w=0.8)
    scan = ScanSpec(noise_m=0.015, dropout_patches=6, unvisited_rooms=["Abstellraum"])
    return b.finish(scan, ["touching furniture rows (1 cm gaps)", "single bed + nightstand touching", "chairs close to table", "noise 15 mm", "dropout patches", "unvisited storage room"])


def objects(seed: int) -> SceneSpec:
    """Wall shelves, screens and ceiling objects, with standing pieces and negatives that must not become one."""
    b = Builder(f"objects_{seed}", "objects", seed)
    b.spec.storey_height = b.u(2.4, 2.65, 2)
    H = b.spec.storey_height
    W, D, t = b.u(5.2, 6.2, 2), b.u(4.2, 5.0, 2), b.u(0.22, 0.32, 2)
    s, e, n, w = b.rect(0, 0, W, D, t)
    b.room("Wohnen", (W / 2, D / 2), H)
    b.door(s, at=(W * 0.82, 0), swing=1)
    window = b.window(n, at=(W * 0.5, D), width=b.u(1.2, 1.6))
    mid = (W / 2, D / 2)
    # standing pieces whose lower part is hard to see (they must still reach the floor)
    b.against("wardrobe", n, at=[(0.7, D), (0.8, D)], toward=mid, w=1.0)
    b.against("sofa", w, at=[(0, D * 0.36), (0, D * 0.4)], toward=mid, w=1.6)
    stand = b.against("tv_stand", e, at=[(W, D * 0.62), (W, D * 0.55)], toward=mid)
    b.against("tv_wall", e, at=[(W, D * 0.62) if stand is None else tuple(stand.center)], toward=mid, gap=0.01)
    desk = b.against("desk", s, at=[(W * 0.22, 0), (W * 0.3, 0)], toward=mid)
    if desk is not None:
        parts, _dims = b.draw("monitor")
        a = math.radians(desk.yaw_deg)
        back = np.array([-math.sin(a), math.cos(a)])
        b.try_add("monitor", parts, np.asarray(desk.center) + 0.12 * back, desk.yaw_deg)
    b.against("wall_shelf", s, at=[(W * f, 0) for f in (0.55, 0.5, 0.6)], toward=mid, gap=0.0)
    b.against("wall_shelf", w, at=[(0, D * 0.38)], toward=mid, gap=0.0)  # above the sofa
    for kind, at in (
        ("ceiling_lamp", (W * 0.52, D * 0.5)), ("spot", (W * 0.2, D * 0.7)), ("spot", (W * 0.2, D * 0.3)),
        ("smoke_detector", (W * 0.75, D * 0.28)), ("vent", (W * 0.8, D * 0.78)),
    ):
        parts, _dims = b.draw(kind, ceiling=H)
        b.try_add(kind, parts, (at[0] + b.u(-0.1, 0.1), at[1] + b.u(-0.1, 0.1)), 0.0)
    opening = next(o for o in b.spec.openings if o.id == window)
    parts, dims = b.draw("neg_curtain_rail", ceiling=H)
    b.against("neg_curtain_rail", n, at=[opening.offset], toward=mid, gap=0.1, parts=parts, dims=dims)
    b.against("neg_wall_clock", w, at=[(0, D * 0.8)], toward=mid, gap=0.005)
    scan = ScanSpec(noise_m=float(b.rng.choice([0.004, 0.006, 0.010])), voxel_m=0.015)
    return b.finish(scan, ["wall shelves", "TV on the wall", "monitor on a desk", "ceiling lamp, spots, detector, vent", "negatives: curtain rail, wall clock, wardrobe, sofa"])


FAMILIES: dict[str, Callable[[int], SceneSpec]] = {
    "rect_basic": rect_basic,
    "l_kitchen": l_kitchen,
    "bedroom_slanted": bedroom_slanted,
    "apartment": apartment,
    "structure": structure,
    "irregular": irregular,
    "dense": dense,
    "objects": objects,
}

DEV_SET: list[tuple[str, int]] = [
    ("rect_basic", 101), ("l_kitchen", 102), ("bedroom_slanted", 103), ("apartment", 104), ("structure", 105),
    ("irregular", 106), ("dense", 107),
]
VAL_SET: list[tuple[str, int]] = [
    ("rect_basic", 211), ("l_kitchen", 212), ("bedroom_slanted", 213), ("apartment", 214), ("structure", 215),
    ("irregular", 216), ("dense", 217),
]
# scenes of the room objects (samcloud.bim.room_objects); their own sets, so dev / val stay comparable
OBJECTS_SET: list[tuple[str, int]] = [("objects", 301), ("objects", 302), ("objects", 303), ("objects", 304)]
OBJECTS_VAL_SET: list[tuple[str, int]] = [("objects", 411), ("objects", 412), ("objects", 413), ("objects", 414)]
SETS = {"dev": DEV_SET, "val": VAL_SET, "objects": OBJECTS_SET, "objects_val": OBJECTS_VAL_SET}


def make_scene(family: str, seed: int) -> SceneSpec:
    return FAMILIES[family](int(seed))


def scene_list(name: str) -> list[tuple[str, int]]:
    """``dev`` / ``val`` or an explicit list ``family:seed,family:seed``."""
    if name in SETS:
        return list(SETS[name])
    result = []
    for token in name.split(","):
        family, _, seed = token.strip().partition(":")
        result.append((family, int(seed)))
    return result


def check_scene(spec: SceneSpec) -> list[str]:
    return validate_scene(spec)
