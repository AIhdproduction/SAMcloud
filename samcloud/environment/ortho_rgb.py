"""Top-down RGB (and height) raster of the coloured dense cloud, used to refine surface edges.

The raster is north-up in the local metric frame; row 0 is the northern edge at ``ymax``.  For every
cell the colour of the upper points is taken (mean over the points above the ``z_percentile`` of
the cell), so what a drone camera sees from above is what the raster shows.  Small holes are filled
from the nearest valid cell.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

LOGGER = logging.getLogger(__name__)

_MAX_CELLS = 80_000_000


@dataclass
class OrthoRGB:
    """North-up colour raster: ``rgb`` (H, W, 3) uint8, ``valid`` (H, W), optional ``height`` (H, W)."""

    rgb: np.ndarray
    valid: np.ndarray
    xmin: float
    ymax: float
    cell: float
    height: np.ndarray | None = None
    _lab: np.ndarray | None = field(default=None, repr=False)
    _lab_sigma: float = field(default=-1.0, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.rgb.shape[0]), int(self.rgb.shape[1])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        h, w = self.shape
        return (self.xmin, self.ymax - h * self.cell, self.xmin + w * self.cell, self.ymax)

    def lab(self, sigma_m: float = 0.0) -> np.ndarray:
        """CIE Lab image (float32, invalid cells 0), optionally Gaussian smoothed with normalised convolution."""
        if self._lab is not None and self._lab_sigma == sigma_m:
            return self._lab
        from skimage.color import rgb2lab

        lab = np.empty(self.rgb.shape, dtype=np.float32)
        for r0 in range(0, self.rgb.shape[0], 512):  # blocks keep the float64 temporaries small
            lab[r0:r0 + 512] = rgb2lab(self.rgb[r0:r0 + 512])
        weight = self.valid.astype(np.float32)
        lab *= weight[..., None]
        sigma = sigma_m / self.cell
        if sigma > 0.3:
            den = ndimage.gaussian_filter(weight, sigma)
            for k in range(3):
                lab[..., k] = ndimage.gaussian_filter(lab[..., k], sigma) / np.maximum(den, 1e-6)
            lab *= (den > 1e-3)[..., None]
        self._lab, self._lab_sigma = lab, sigma_m
        return lab

    def _rc(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return np.vstack(((self.ymax - xy[:, 1]) / self.cell - 0.5, (xy[:, 0] - self.xmin) / self.cell - 0.5))

    def _bilinear(self, image: np.ndarray, weight: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rc = self._rc(xy)
        den = ndimage.map_coordinates(weight, rc, order=1, mode="constant", cval=0.0)
        if image.ndim == 3:
            num = np.stack([ndimage.map_coordinates(image[..., k] * weight, rc, order=1, mode="constant", cval=0.0)
                            for k in range(image.shape[2])], axis=1)
            return num / np.maximum(den, 1e-6)[:, None], den
        return ndimage.map_coordinates(image * weight, rc, order=1, mode="constant", cval=0.0) / np.maximum(den, 1e-6), den

    def sample_lab(self, xy, sigma_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Bilinear Lab at ``xy`` (n, 2) and a ``covered`` flag (all four neighbour cells valid)."""
        value, den = self._bilinear(self.lab(sigma_m), self.valid.astype(np.float32), xy)
        return value, den > 0.99

    def sample_height(self, xy) -> tuple[np.ndarray, np.ndarray]:
        """Bilinear top height at ``xy`` and a ``covered`` flag; NaN everywhere when there is no height layer."""
        xy = np.asarray(xy, dtype=float)
        if self.height is None:
            return np.full(len(xy), np.nan), np.zeros(len(xy), bool)
        weight = (self.valid & np.isfinite(self.height)).astype(np.float32)
        value, den = self._bilinear(np.nan_to_num(self.height), weight, xy)
        return value, den > 0.99

    def to_geotiff(self, path: str | Path, frame=None) -> Path:
        """Write RGBA (alpha 255 = valid) as GeoTIFF; with a ``LocalFrame`` georeferenced like ``dtm.tif``."""
        import rasterio
        from rasterio.transform import from_origin

        if frame is not None:
            east, north, _ = frame.to_lv95(np.array([self.xmin, self.ymax, 0.0]))
        else:
            east, north = self.xmin, self.ymax
        h, w = self.shape
        data = np.concatenate([np.moveaxis(self.rgb, 2, 0), (self.valid.astype(np.uint8) * 255)[None]], axis=0)
        profile = dict(driver="GTiff", height=h, width=w, count=4, dtype="uint8", compress="lzw",
                       transform=from_origin(float(east), float(north), self.cell, self.cell))
        if frame is not None:
            try:
                profile["crs"] = f"EPSG:{int(frame.epsg)}"
            except Exception:  # noqa: BLE001 - georeferencing is optional
                LOGGER.warning("could not set CRS of the ortho raster")
        if h >= 16 and w >= 16:
            profile.update(tiled=True, blockxsize=16, blockysize=16)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data)
            dst.update_tags(LOCAL_XMIN=repr(float(self.xmin)), LOCAL_YMAX=repr(float(self.ymax)), CELL_M=repr(float(self.cell)))
        return out

    @classmethod
    def from_geotiff(cls, path: str | Path) -> "OrthoRGB":
        """Read a file written by :meth:`to_geotiff` (local origin is taken from its tags; no height layer)."""
        import rasterio

        with rasterio.open(path) as src:
            data = src.read()
            tags = src.tags()
            cell = float(tags.get("CELL_M", src.transform.a))
            xmin = float(tags["LOCAL_XMIN"]) if "LOCAL_XMIN" in tags else float(src.transform.c)
            ymax = float(tags["LOCAL_YMAX"]) if "LOCAL_YMAX" in tags else float(src.transform.f)
        return cls(np.ascontiguousarray(np.moveaxis(data[:3], 0, 2)), data[3] > 0, xmin, ymax, cell)

    def preview_png(self, path: str | Path) -> Path:
        """Write the colour image as PNG (invalid cells black)."""
        from PIL import Image

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.where(self.valid[..., None], self.rgb, 0).astype(np.uint8)).save(out)
        return out


def build_ortho_rgb(
    xyz: np.ndarray,
    rgb: np.ndarray | None,
    *,
    cell_m: float = 0.03,
    z_percentile: float = 0.7,
    hole_fill_cells: int = 3,
    bounds: tuple[float, float, float, float] | None = None,
) -> OrthoRGB | None:
    """Rasterise coloured points top-down; ``None`` when there are no colours.

    Per cell the mean colour (and height) of the points at or above the ``z_percentile`` of the
    cell's heights is used.  Cells without points that are within ``hole_fill_cells`` of a valid
    cell take its colour.  The cell size grows automatically so that the raster stays below
    80 M cells.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if rgb is None or len(xyz) == 0 or len(rgb) != len(xyz):
        return None
    if bounds is None:
        lo, hi = xyz[:, :2].min(axis=0), xyz[:, :2].max(axis=0)
        bounds = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
    xmin, ymin, xmax, ymax = bounds
    cell = float(cell_m)
    while (np.ceil((xmax - xmin) / cell) + 1) * (np.ceil((ymax - ymin) / cell) + 1) > _MAX_CELLS:
        cell *= 1.25
    w = int(np.ceil((xmax - xmin) / cell)) + 1
    h = int(np.ceil((ymax - ymin) / cell)) + 1
    top = ymax
    col = np.floor((xyz[:, 0] - xmin) / cell).astype(np.int64)
    row = np.floor((top - xyz[:, 1]) / cell).astype(np.int64)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    cell_id = (row * w + col)[inside]
    z = xyz[inside, 2]
    colours = np.asarray(rgb, dtype=np.float64)[inside]
    order = np.lexsort((z, cell_id))
    cell_id, z, colours = cell_id[order], z[order], colours[order]
    starts = np.flatnonzero(np.r_[True, cell_id[1:] != cell_id[:-1]])
    counts = np.diff(np.r_[starts, len(cell_id)])
    first = starts + np.floor(float(np.clip(z_percentile, 0.0, 1.0)) * (counts - 1)).astype(np.int64)
    last = starts + counts  # exclusive
    csum = np.vstack((np.zeros((1, 3)), np.cumsum(colours, axis=0)))
    zsum = np.r_[0.0, np.cumsum(z)]
    n_top = (last - first).astype(np.float64)
    mean_rgb = (csum[last] - csum[first]) / n_top[:, None]
    mean_z = (zsum[last] - zsum[first]) / n_top
    image = np.zeros((h * w, 3), dtype=np.float32)
    height = np.full(h * w, np.nan, dtype=np.float32)
    valid = np.zeros(h * w, dtype=bool)
    ids = cell_id[starts]
    image[ids], height[ids], valid[ids] = mean_rgb, mean_z, True
    image = image.reshape(h, w, 3)
    height = height.reshape(h, w)
    valid = valid.reshape(h, w)
    if hole_fill_cells > 0 and not valid.all():
        dist, (ri, ci) = ndimage.distance_transform_edt(~valid, return_indices=True)
        fill = (~valid) & (dist <= hole_fill_cells)
        image[fill] = image[ri[fill], ci[fill]]
        height[fill] = height[ri[fill], ci[fill]]
        valid = valid | fill
    return OrthoRGB(np.clip(np.rint(image), 0, 255).astype(np.uint8), valid, float(xmin), float(top), cell, height)
