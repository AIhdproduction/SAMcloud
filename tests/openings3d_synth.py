"""Synthetic wall-plane evidence (noisy point counts and photograph votes) for the 3D opening tests.

One straight exterior wall of ``length_m`` metres along +x (plan units == metres), grid cell 5 cm.
Everything carries random noise (seeded), nothing sits exactly on a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from samcloud.bim.openings3d.config import Settings, load_openings_config
from samcloud.bim.openings3d.evidence import LineEvidence, build_line_evidence
from samcloud.bim.openings3d.voting import LineVotes
from samcloud.bim.openings3d.walls import WallLine, wall_lines
from samcloud.bim.plan_model import BuildingPlan, PlanOpening, PlanWall

CELL = 0.05
HEIGHT_M = 2.6

Rect = tuple[float, float, float, float]  # u0, u1, v0, v1 in metres


def make_plan(length_m: float = 4.0, kind: str = "exterior", openings: list[PlanOpening] | None = None) -> BuildingPlan:
    walls = [PlanWall(id="W001", centreline=[[0.0, 0.0], [length_m, 0.0]], thickness_m=0.25, kind=kind)]
    frame = {"origin": [0.0, 0.0, 0.0], "scale_factor": 1.0, "rotation": np.eye(3).tolist()}
    return BuildingPlan("synthetic", "level_1", "Level 1", 1.0, frame, 0.0, HEIGHT_M, walls=walls, openings=list(openings or []))


def seed_door(opening_id: str, centre_u: float, width_m: float) -> PlanOpening:
    return PlanOpening(opening_id, "door", [centre_u, 0.0], width_m, 0.0, 0.25, "W001")


def _cells(rect: Rect) -> tuple[slice, slice]:
    u0, u1, v0, v1 = rect
    return slice(int(round(v0 / CELL)), int(round(v1 / CELL))), slice(max(int(round(u0 / CELL)), 0), max(int(round(u1 / CELL)), 0))  # a rect left of the line is clipped, not wrapped


@dataclass
class Scene:
    """What the synthetic wall shows; each list holds rectangles in metres."""

    length_m: float = 4.0
    kind: str = "exterior"
    holes: list[Rect] = field(default_factory=list)  # no wall points (real openings or unobserved gaps)
    unobserved: list[Rect] = field(default_factory=list)  # no photograph saw these cells (and no points)
    windows: list[Rect] = field(default_factory=list)  # window class votes
    doors: list[Rect] = field(default_factory=list)  # door class votes
    furniture: list[Rect] = field(default_factory=list)  # furniture (curtain) votes
    mullions: list[float] = field(default_factory=list)  # bar centres (m), 6 cm wide, wall points + dark photo stripe + vote dip
    see_through: list[Rect] = field(default_factory=list)  # rays pass through the plane here
    seed: int = 11


def make_gap_plan(pieces: list[tuple[float, float]], kind: str = "exterior") -> BuildingPlan:
    """Collinear wall pieces along +x (``(x0, x1)`` each, plan units == metres) separated by gaps."""
    walls = [PlanWall(id=f"W{index + 1:03d}", centreline=[[x0, 0.0], [x1, 0.0]], thickness_m=0.25, kind=kind) for index, (x0, x1) in enumerate(pieces)]
    frame = {"origin": [0.0, 0.0, 0.0], "scale_factor": 1.0, "rotation": np.eye(3).tolist()}
    return BuildingPlan("synthetic", "level_1", "Level 1", 1.0, frame, 0.0, HEIGHT_M, walls=walls)


def _shifted(scene: Scene, dx: float, length_m: float) -> Scene:
    """The same scene expressed in the coordinates of a wall line that starts at world x = ``-dx``."""
    move = lambda rects: [(r[0] + dx, r[1] + dx, r[2], r[3]) for r in rects]  # noqa: E731
    return replace(
        scene, length_m=length_m, holes=move(scene.holes), unobserved=move(scene.unobserved), windows=move(scene.windows), doors=move(scene.doors),
        furniture=move(scene.furniture), see_through=move(scene.see_through), mullions=[m + dx for m in scene.mullions],
    )


def build_evidence(scene: Scene, config: Settings | None = None, plan: BuildingPlan | None = None, line_index: int = 0) -> tuple[LineEvidence, BuildingPlan, Settings]:
    """Noisy grids + votes for ``scene`` and the derived LineEvidence.

    With ``plan`` the scene is given in world x (what the cameras saw) and drawn onto wall line ``line_index`` of that plan,
    whatever its start and length are, like a projection of the photographs onto a slightly different wall set.
    """
    config = config or load_openings_config()
    if plan is None:
        plan = make_plan(scene.length_m, scene.kind)
    line: WallLine = wall_lines(plan, config)[line_index]
    if scene.length_m != line.length_m or float(line.a[0]) != 0.0:
        scene = _shifted(scene, -float(line.a[0]), float(line.length_m))
    rng = np.random.default_rng(scene.seed)
    nv, nu = int(round(HEIGHT_M / CELL)), int(round(scene.length_m / CELL))

    wall = rng.poisson(9.0, (nv, nu)).astype(np.int32)
    win_points = np.zeros((nv, nu), np.int32)
    door_points = np.zeros((nv, nu), np.int32)
    other = rng.poisson(0.05, (nv, nu)).astype(np.int32)
    for rect in [*scene.holes, *scene.unobserved]:
        rows, cols = _cells(rect)
        wall[rows, cols] = rng.poisson(0.08, wall[rows, cols].shape)
    for position in scene.mullions:  # the bar itself is wall-class points inside the hole
        v0, v1 = _window_rows(scene, position)
        bar = (slice(int(round(v0 / CELL)), int(round(v1 / CELL))), slice(int(round((position - 0.03) / CELL)), int(round((position + 0.03) / CELL))))
        wall[bar] = rng.poisson(9.0, wall[bar].shape)

    sw = rng.uniform(0.8, 1.3, (nv, nu))
    for rect in scene.unobserved:
        rows, cols = _cells(rect)
        sw[rows, cols] = 0.0
    win_p = np.clip(rng.normal(0.04, 0.02, (nv, nu)), 0, 1)
    door_p = np.clip(rng.normal(0.04, 0.02, (nv, nu)), 0, 1)
    fur_p = np.clip(rng.normal(0.03, 0.02, (nv, nu)), 0, 1)
    for rect in scene.windows:
        rows, cols = _cells(rect)
        win_p[rows, cols] = np.clip(rng.normal(0.85, 0.05, win_p[rows, cols].shape), 0, 1)
    for rect in scene.doors:
        rows, cols = _cells(rect)
        door_p[rows, cols] = np.clip(rng.normal(0.8, 0.05, door_p[rows, cols].shape), 0, 1)
    for rect in scene.furniture:
        rows, cols = _cells(rect)
        fur_p[rows, cols] = np.clip(rng.normal(0.8, 0.05, fur_p[rows, cols].shape), 0, 1)
    gray = np.clip(rng.normal(0.55, 0.02, (nv, nu)), 0, 1)
    for rect in scene.windows:
        rows, cols = _cells(rect)
        gray[rows, cols] = np.clip(rng.normal(0.85, 0.02, gray[rows, cols].shape), 0, 1)
    for position in scene.mullions:
        v0, v1 = _window_rows(scene, position)
        rows = slice(int(round(v0 / CELL)), int(round(v1 / CELL)))
        cols = slice(int(round((position - 0.03) / CELL)), int(round((position + 0.03) / CELL)))
        win_p[rows, cols] = np.clip(rng.normal(0.1, 0.03, win_p[rows, cols].shape), 0, 1)
        gray[rows, cols] = np.clip(rng.normal(0.2, 0.02, gray[rows, cols].shape), 0, 1)

    n_images = np.where(sw > 0, rng.integers(4, 8, (nv, nu)), 0).astype(float)
    behind = np.zeros((nv, nu))
    for rect in scene.see_through:
        rows, cols = _cells(rect)
        behind[rows, cols] = n_images[rows, cols] * 0.8
    votes = LineVotes(
        sw=sw, win=sw * win_p, door=sw * door_p, wall=sw * 0.3, n=n_images, occ=rng.uniform(0.0, 1.0, (nv, nu)) * (sw > 0), fur=n_images * fur_p, behind=behind,
        rays=n_images.copy(), vis=n_images.copy(), rgb=np.repeat(gray[None], 3, axis=0),
    )
    grids = {"wall": wall, "win": win_points, "door": door_points, "other": other}
    return build_line_evidence(line, grids, votes, config), plan, config


def _window_rows(scene: Scene, position: float) -> tuple[float, float]:
    """Rows spanned by the window that contains a mullion position."""
    for rect in scene.windows:
        if rect[0] <= position <= rect[1]:
            return rect[2], rect[3]
    raise ValueError("mullion outside every window")
