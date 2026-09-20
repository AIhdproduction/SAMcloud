"""Coordinate reference system validation and transformation helpers."""

from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer


LOCAL_CRS = "LOCAL"


def validate_crs(crs_name: str) -> str:
    """Accept LOCAL or any CRS that PROJ can resolve, such as EPSG:2056."""
    normalized = crs_name.strip().upper()
    if normalized == LOCAL_CRS:
        return LOCAL_CRS
    CRS.from_user_input(normalized)
    return normalized


def transform_xyz(xyz: np.ndarray, source_crs: str, target_crs: str) -> np.ndarray:
    """Transform an Nx3 coordinate array between configured CRS values."""
    source = validate_crs(source_crs)
    target = validate_crs(target_crs)
    if source == target:
        return xyz.copy()
    if LOCAL_CRS in {source, target}:
        raise ValueError("Local coordinates require a defined transformation before export")
    transformer = Transformer.from_crs(source, target, always_xy=True)
    x, y, z = transformer.transform(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    return np.column_stack((x, y, z))
