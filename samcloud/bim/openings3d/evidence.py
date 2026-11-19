"""Per wall line evidence grids (photograph votes + point counts) in the form the detection rules read."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndi

from .config import Settings
from .voting import LineVotes
from .walls import WallLine


@dataclass
class LineEvidence:
    """Everything the rules need for one wall line; all arrays are (nv, nu) at ``grid.cell_m``.

    ``solid``  wall points are dense here (real wall), ``obs`` some photograph saw the cell,
    ``pw``/``pd`` smoothed window/door probability, ``behind`` fraction of rays that pass through the
    plane (seen through), ``furf`` fraction of furniture votes, ``gray`` mean observed brightness.
    ``votes`` is None in the points-only mode (then ``obs`` is all False).
    """

    line: WallLine
    grids: dict[str, np.ndarray]
    solid: np.ndarray
    obs: np.ndarray
    pw: np.ndarray
    pd: np.ndarray
    behind: np.ndarray
    furf: np.ndarray
    gray: np.ndarray
    votes: LineVotes | None
    items: list[dict] = field(default_factory=list)
    curtains: list[dict] = field(default_factory=list)
    bands: list[dict] = field(default_factory=list)

    @property
    def nv(self) -> int:
        return int(self.solid.shape[0])

    @property
    def nu(self) -> int:
        return int(self.solid.shape[1])


def solid_mask(grids: dict[str, np.ndarray], config: Settings) -> np.ndarray:
    """Dense wall-class points, denser than window/door points in the same cell."""
    wall = grids["wall"].astype(float)
    density = ndi.uniform_filter(wall, 3) * 9
    return (density >= config.points.solid_density) & (wall > np.maximum(grids["win"], grids["door"]))


def build_line_evidence(line: WallLine, grids: dict[str, np.ndarray], votes: LineVotes | None, config: Settings) -> LineEvidence:
    """Derive the rule inputs from the raw point grids and (optionally) the merged photograph votes."""
    solid = solid_mask(grids, config)
    shape = solid.shape
    if votes is None:
        zeros = np.zeros(shape)
        return LineEvidence(line, grids, solid, np.zeros(shape, bool), zeros, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), None)
    sw = votes.sw
    obs = sw > config.grid.min_observed_weight
    sigma = config.windows.smooth_sigma_cells
    with np.errstate(all="ignore"):
        pw = np.where(obs, votes.win / sw, 0.0)
        pd = np.where(obs, votes.door / sw, 0.0)
        furf = np.where(votes.vis > 0.5, votes.fur / np.maximum(votes.vis, 1e-9), 0.0)
        behind = np.where(votes.rays > 0.5, votes.behind / np.maximum(votes.rays, 1e-9), 0.0)
    pw = np.nan_to_num(ndi.gaussian_filter(pw, sigma))
    pd = np.nan_to_num(ndi.gaussian_filter(pd, sigma))
    gray = votes.rgb.mean(0) * 255.0
    return LineEvidence(line, grids, solid, obs, pw, pd, behind, furf, gray, votes)
