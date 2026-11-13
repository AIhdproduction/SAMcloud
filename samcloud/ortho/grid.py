"""North-up raster grid helpers and world-file affine coefficients."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from samcloud.dxf.las_source import Frame


@dataclass(frozen=True, slots=True)
class OrthoGrid:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    pixel_size: float
    width: int
    height: int

    def local_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """Map local coordinates to fractional pixel indices, with integer indices at pixel centres."""
        value = np.asarray(points, dtype=np.float64)
        output = np.empty(value.shape[:-1] + (2,), dtype=np.float64)
        output[..., 0] = (value[..., 0] - self.min_x) / self.pixel_size - 0.5
        output[..., 1] = (self.max_y - value[..., 1]) / self.pixel_size - 0.5
        return output

    def pixel_to_local(self, pixels: np.ndarray) -> np.ndarray:
        """Map fractional pixel indices to local coordinates using the pixel-centre convention."""
        value = np.asarray(pixels, dtype=np.float64)
        output = np.empty(value.shape[:-1] + (2,), dtype=np.float64)
        output[..., 0] = self.min_x + (value[..., 0] + 0.5) * self.pixel_size
        output[..., 1] = self.max_y - (value[..., 1] + 0.5) * self.pixel_size
        return output

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.min_x, self.min_y, self.max_x, self.max_y

    def local_world_file(self) -> tuple[float, float, float, float, float, float]:
        return self.pixel_size, 0.0, 0.0, -self.pixel_size, self.min_x + self.pixel_size / 2.0, self.max_y - self.pixel_size / 2.0

    def original_world_file(self, frame: Frame) -> tuple[float, float, float, float, float, float]:
        """Return a rotated PGW affine transform for original-frame pixel centres."""
        x0, y0 = self.pixel_to_local(np.array([0.0, 0.0]))
        origin = frame.to_original(np.array([[x0, y0, 0.0]]))[0]
        dx = frame.to_original(np.array([[x0 + self.pixel_size, y0, 0.0]]))[0] - origin
        dy = frame.to_original(np.array([[x0, y0 - self.pixel_size, 0.0]]))[0] - origin
        return float(dx[0]), float(dx[1]), float(dy[0]), float(dy[1]), float(origin[0]), float(origin[1])


def grid_for_bounds(bounds: tuple[float, float, float, float], pixel_size: float, max_side: int) -> OrthoGrid:
    """Build an outward-rounded grid, increasing pixel size where a side exceeds ``max_side``."""
    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    if pixel_size <= 0 or max_side < 1 or max_x < min_x or max_y < min_y:
        raise ValueError("Invalid orthophoto grid bounds")
    requested = max(pixel_size, (max_x - min_x) / max_side, (max_y - min_y) / max_side)
    width = max(1, math.ceil((max_x - min_x) / requested))
    height = max(1, math.ceil((max_y - min_y) / requested))
    return OrthoGrid(min_x, min_y, min_x + width * requested, min_y + height * requested, requested, width, height)
