"""DSM, DTM and per-cell source-coordinate production."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

from samcloud.dxf.levels import Level
from samcloud.dxf.las_source import Frame, classified_ply_source, is_ply_source, iter_source_chunks, read_las_info
from samcloud.ortho.config import OrthoConfig
from samcloud.ortho.grid import OrthoGrid


@dataclass(slots=True)
class DsmResult:
    dsm: np.ndarray
    dtm: np.ndarray
    classes: np.ndarray
    validity: np.ndarray
    footprint: np.ndarray
    colmap_xyz: np.ndarray
    top_rgb: np.ndarray
    normals: np.ndarray
    affine_matrix: np.ndarray | None
    affine_residual: float
    xyz_method: str
    full_colmap_xyz: np.ndarray
    fine_pc: np.ndarray
    fine_pc_confidence: np.ndarray


def read_cloud(
    project_dir: Path,
    point_source: Path,
    frame: Frame,
    *,
    voxel_size: float | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, ...]:
    """Read classified points preserving raw PLY COLMAP coordinates for reprojection."""
    if is_ply_source(point_source):
        source = classified_ply_source(point_source)
        from samcloud.engine.export_las import _ply_chunks, read_ply_header

        offset, count = read_ply_header(point_source)
        total_points = max(count, 1)
        points_read = 0
        transform = source.transformer()
        raw_parts: list[np.ndarray] = []
        local_parts: list[np.ndarray] = []
        classes: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        fine_terms: list[np.ndarray] = []
        voxel_key_parts: list[np.ndarray] = []
        voxel_point_parts: list[np.ndarray] = []
        with point_source.open("rb") as handle:
            for chunk in _ply_chunks(handle, offset, count, 500_000):
                raw = np.column_stack((chunk["x"], chunk["y"], chunk["z"])).astype(np.float64)
                raw_parts.append(raw)
                if voxel_size is not None:
                    chunk_keys, chunk_pts = _chunk_voxel_representatives(raw, voxel_size)
                    if len(chunk_keys) > 0:
                        voxel_key_parts.append(chunk_keys)
                        voxel_point_parts.append(chunk_pts)
                local_parts.append(frame.to_local(transform(raw)))
                classes.append(source.classify(np.asarray(chunk["scalar_class"], dtype=np.uint8)))
                colors.append(np.column_stack((chunk["red"], chunk["green"], chunk["blue"])).astype(np.uint8))
                names = chunk.dtype.names or ()
                fine_terms.append(np.asarray(chunk["fine_term"], dtype=np.uint8) if "fine_term" in names else np.zeros(len(chunk), dtype=np.uint8))
                points_read += len(raw)
                if progress is not None:
                    progress(min(points_read / total_points, 1.0), "Reading point cloud")
        voxel_points = _deduplicate_voxel_keys(voxel_key_parts, voxel_point_parts) if voxel_size is not None else None
        values = np.vstack(raw_parts), np.vstack(local_parts), np.concatenate(classes), np.vstack(colors), np.concatenate(fine_terms)
        return (*values, voxel_points) if voxel_size is not None else values
    info = read_las_info(point_source)
    total_points = max(info.point_count, 1)
    points_read = 0
    raw_parts: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    fine_terms: list[np.ndarray] = []
    voxel_key_parts: list[np.ndarray] = []
    voxel_point_parts: list[np.ndarray] = []
    for chunk in iter_source_chunks(point_source, 500_000):
        raw_chunk = np.column_stack((chunk.x, chunk.y, chunk.z)).astype(np.float64)
        raw_parts.append(raw_chunk)
        if voxel_size is not None:
            chunk_keys, chunk_pts = _chunk_voxel_representatives(raw_chunk, voxel_size)
            if len(chunk_keys) > 0:
                voxel_key_parts.append(chunk_keys)
                voxel_point_parts.append(chunk_pts)
        classes.append(np.asarray(chunk.classification, dtype=np.uint8))
        colors.append((np.column_stack((chunk.red, chunk.green, chunk.blue)) / 257).astype(np.uint8))
        dimensions = {str(name).lower() for name in getattr(chunk.point_format, "dimension_names", ())}
        fine_terms.append(np.asarray(chunk.fine_term, dtype=np.uint8) if "fine_term" in dimensions else np.zeros(len(chunk), dtype=np.uint8))
        points_read += len(raw_chunk)
        if progress is not None:
            progress(min(points_read / total_points, 1.0), "Reading point cloud")
    voxel_points = _deduplicate_voxel_keys(voxel_key_parts, voxel_point_parts) if voxel_size is not None else None
    raw = np.vstack(raw_parts)
    values = raw, frame.to_local(raw), np.concatenate(classes), np.vstack(colors), np.concatenate(fine_terms)
    return (*values, voxel_points) if voxel_size is not None else values


def _voxel_keys(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Compute packed 21-bit int64 voxel keys for XYZ points."""
    if voxel_size <= 0:
        raise ValueError("Voxel size must be positive")
    cells = np.floor(points / voxel_size).astype(np.int64)
    mask = np.int64((1 << 21) - 1)
    return ((cells[:, 0] & mask) << np.int64(42)) | ((cells[:, 1] & mask) << np.int64(21)) | (cells[:, 2] & mask)


def _chunk_voxel_representatives(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute chunk-unique packed keys and their first-occurrence representative points."""
    if len(points) == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float64)
    keys = _voxel_keys(points, voxel_size)
    unique_keys, first = np.unique(keys, return_index=True)
    return unique_keys, points[first]


def _deduplicate_voxel_keys(key_parts: list[np.ndarray], point_parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate chunk keys and points, keeping the first occurrence across chunks."""
    if not key_parts:
        return np.empty((0, 3), dtype=np.float64)
    all_keys = np.concatenate(key_parts)
    all_points = np.vstack(point_parts)
    _unique_keys, first = np.unique(all_keys, return_index=True)
    return all_points[first]


def _new_voxel_representatives(points: np.ndarray, voxel_size: float, seen: set[int] | None = None) -> np.ndarray:
    """Keep one raw point per voxel key; backwards compatible with existing signature."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float64)
    keys = _voxel_keys(pts, voxel_size)
    unique_keys, first = np.unique(keys, return_index=True)
    if seen is not None:
        keep = np.fromiter((int(index) for key, index in zip(unique_keys, first, strict=True) if int(key) not in seen), dtype=np.int64)
        seen.update(int(key) for key in unique_keys)
        return pts[keep]
    return pts[first]


def build_dsm(
    raw_xyz: np.ndarray,
    local_xyz: np.ndarray,
    classes: np.ndarray,
    grid: OrthoGrid,
    level: Level | None,
    config: OrthoConfig,
    work_dir: Path,
    *,
    rgb: np.ndarray | None = None,
    fine_terms: np.ndarray | None = None,
    voxel_colmap_xyz: np.ndarray | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> DsmResult:
    """Rasterise the highest selected point per north-up grid cell and fill small holes."""
    if progress is not None:
        progress(0.0, "Building surface model")
    raw = np.asarray(raw_xyz, dtype=np.float64)
    local = np.asarray(local_xyz, dtype=np.float64)
    code = np.asarray(classes, dtype=np.uint8)
    fine = np.zeros(len(raw), dtype=np.uint8) if fine_terms is None else np.asarray(fine_terms, dtype=np.uint8)
    if fine.shape != (len(raw),):
        raise ValueError("Fine terms must have shape (N,)")
    colours = np.zeros((len(raw), 3), dtype=np.uint8) if rgb is None else np.asarray(rgb, dtype=np.uint8)
    if colours.shape != (len(raw), 3):
        raise ValueError("Point colours must have shape (N, 3)")
    selected = np.isfinite(local).all(axis=1)
    if level is not None:
        cut = (level.ceiling_z - config.indoor.cut_below_ceiling) if level.ceiling_z is not None else (level.floor_z + config.indoor.cut_fallback_above_floor)
        selected &= local[:, 2] >= level.floor_z - config.indoor.floor_margin
        selected &= local[:, 2] <= cut
        selected &= ~np.isin(code, config.indoor.ceiling_codes)

    cell = max(grid.pixel_size, config.dsm.cell_indoor if level is not None else config.dsm.cell_outdoor)
    if cell <= grid.pixel_size * 1.0001:
        yy = np.floor((grid.max_y - local[:, 1]) / grid.pixel_size).astype(np.int64)
        xx = np.floor((local[:, 0] - grid.min_x) / grid.pixel_size).astype(np.int64)
        selected &= (xx >= 0) & (xx < grid.width) & (yy >= 0) & (yy < grid.height)
        dsm = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
        top_raw = np.full((grid.height, grid.width, 3), np.nan, dtype=np.float32)
        top_class = np.zeros((grid.height, grid.width), dtype=np.uint8)
        top_rgb = np.zeros((grid.height, grid.width, 3), dtype=np.uint8)

        if np.any(selected):
            source_indices = np.flatnonzero(selected)
            z_sel = local[source_indices, 2]
            order = np.argsort(z_sel, kind="stable")
            flat = yy[source_indices] * grid.width + xx[source_indices]
            flat_sorted = flat[order]
            _uniq, last_rev = np.unique(flat_sorted[::-1], return_index=True)
            winners = source_indices[order[len(order) - 1 - last_rev]]
            win_rows = yy[winners]
            win_cols = xx[winners]
            dsm[win_rows, win_cols] = local[winners, 2].astype(np.float32)
            top_class[win_rows, win_cols] = code[winners]
            top_raw[win_rows, win_cols] = raw[winners].astype(np.float32)
            top_rgb[win_rows, win_cols] = colours[winners]

        if progress is not None:
            progress(0.4, "Building surface model")

        measured = np.isfinite(dsm)
        validity = np.zeros(dsm.shape, dtype=np.uint8)
        validity[measured] = 1
        labels, count = ndimage.label(~measured)
        if measured.any() and config.dsm.hole_fill_max_cells > 0 and count > 0:
            sizes = np.bincount(labels.ravel())
            small = sizes <= config.dsm.hole_fill_max_cells
            small[0] = False
            fill = small[labels] & ~measured
            if np.any(fill):
                _distances, nearest = ndimage.distance_transform_edt(~measured, return_indices=True)
                rows, cols = nearest[:, fill]
                dsm[fill] = dsm[rows, cols]
                top_raw[fill] = top_raw[rows, cols]
                top_class[fill] = top_class[rows, cols]
                top_rgb[fill] = top_rgb[rows, cols]
                validity[fill] = 2

        footprint = np.zeros(dsm.shape, dtype=bool)
        if level is not None and config.dsm.surface_fill_indoor:
            footprint = _fill_indoor_surface(dsm, validity, top_raw, top_class, top_rgb, grid.min_x, grid.max_y, grid.pixel_size, level, config)

        if progress is not None:
            progress(0.7, "Building surface model")

        affine, residual = _fit_affine(local, raw)
        if residual <= config.affine_max_residual:
            rows, cols = np.indices(dsm.shape)
            pixel = np.column_stack((cols.ravel(), rows.ravel()))
            xy = grid.pixel_to_local(pixel)
            local_grid = np.column_stack((xy, dsm.ravel()))
            valid = np.isfinite(local_grid[:, 2])
            derived = np.full((len(local_grid), 3), np.nan, dtype=np.float32)
            derived[valid] = _apply_affine(local_grid[valid], affine).astype(np.float32)
            top_raw = derived.reshape(top_raw.shape)
            method = "affine"
        else:
            work_dir.mkdir(parents=True, exist_ok=True)
            memmap_path = work_dir / "top_colmap_xyz.npy"
            mapped = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=np.float32, shape=top_raw.shape)
            mapped[:] = top_raw
            mapped.flush()
            top_raw = mapped
            method = "memmap"
    else:
        coarse_width = max(1, int(math.ceil((grid.width * grid.pixel_size) / cell)))
        coarse_height = max(1, int(math.ceil((grid.height * grid.pixel_size) / cell)))

        yy_c = np.floor((grid.max_y - local[:, 1]) / cell).astype(np.int64)
        xx_c = np.floor((local[:, 0] - grid.min_x) / cell).astype(np.int64)
        selected_c = selected & (xx_c >= 0) & (xx_c < coarse_width) & (yy_c >= 0) & (yy_c < coarse_height)

        coarse_dsm = np.full((coarse_height, coarse_width), np.nan, dtype=np.float32)
        coarse_top_raw = np.full((coarse_height, coarse_width, 3), np.nan, dtype=np.float32)
        coarse_top_class = np.zeros((coarse_height, coarse_width), dtype=np.uint8)
        coarse_top_rgb = np.zeros((coarse_height, coarse_width, 3), dtype=np.uint8)

        if np.any(selected_c):
            source_indices = np.flatnonzero(selected_c)
            z_sel = local[source_indices, 2]
            order = np.argsort(z_sel, kind="stable")
            flat = yy_c[source_indices] * coarse_width + xx_c[source_indices]
            flat_sorted = flat[order]
            _uniq, last_rev = np.unique(flat_sorted[::-1], return_index=True)
            winners = source_indices[order[len(order) - 1 - last_rev]]
            win_rows = yy_c[winners]
            win_cols = xx_c[winners]
            coarse_dsm[win_rows, win_cols] = local[winners, 2].astype(np.float32)
            coarse_top_class[win_rows, win_cols] = code[winners]
            coarse_top_raw[win_rows, win_cols] = raw[winners].astype(np.float32)
            coarse_top_rgb[win_rows, win_cols] = colours[winners]

        if progress is not None:
            progress(0.4, "Building surface model")

        measured_c = np.isfinite(coarse_dsm)
        coarse_validity = np.zeros(coarse_dsm.shape, dtype=np.uint8)
        coarse_validity[measured_c] = 1
        labels, count = ndimage.label(~measured_c)
        area_limit = config.dsm.hole_fill_max_area_m2_indoor if level is not None else config.dsm.hole_fill_max_area_m2
        if area_limit is not None:
            max_cells = int(round(area_limit / (cell * cell)))
        else:
            max_cells = config.dsm.hole_fill_max_cells

        if measured_c.any() and max_cells > 0 and count > 0:
            sizes = np.bincount(labels.ravel())
            small = sizes <= max_cells
            small[0] = False
            fill = small[labels] & ~measured_c
            if np.any(fill):
                _distances, nearest = ndimage.distance_transform_edt(~measured_c, return_indices=True)
                rows, cols = nearest[:, fill]
                coarse_dsm[fill] = coarse_dsm[rows, cols]
                coarse_top_raw[fill] = coarse_top_raw[rows, cols]
                coarse_top_class[fill] = coarse_top_class[rows, cols]
                coarse_top_rgb[fill] = coarse_top_rgb[rows, cols]
                coarse_validity[fill] = 2

        coarse_footprint = np.zeros(coarse_dsm.shape, dtype=bool)
        if level is not None and config.dsm.surface_fill_indoor:
            coarse_footprint = _fill_indoor_surface(coarse_dsm, coarse_validity, coarse_top_raw, coarse_top_class, coarse_top_rgb, grid.min_x, grid.max_y, cell, level, config)

        if progress is not None:
            progress(0.7, "Building surface model")

        valid_coarse = coarse_validity > 0
        z_filled_nan_replaced = coarse_dsm.copy()
        nearest_valid: np.ndarray | None = None
        if np.any(valid_coarse):
            if not np.all(valid_coarse):
                _distances, nearest_valid = ndimage.distance_transform_edt(~valid_coarse, return_indices=True)
                z_filled_nan_replaced[~valid_coarse] = coarse_dsm[nearest_valid[0, ~valid_coarse], nearest_valid[1, ~valid_coarse]]
        else:
            z_filled_nan_replaced[:] = 0.0

        affine, residual = _fit_affine(local, raw)
        method = "affine" if residual <= config.affine_max_residual else "memmap"

        dsm = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
        validity = np.zeros((grid.height, grid.width), dtype=np.uint8)
        footprint = np.zeros((grid.height, grid.width), dtype=bool)
        top_class = np.zeros((grid.height, grid.width), dtype=np.uint8)
        top_rgb = np.zeros((grid.height, grid.width, 3), dtype=np.uint8)

        if method == "affine":
            top_raw = np.full((grid.height, grid.width, 3), np.nan, dtype=np.float32)
        else:
            work_dir.mkdir(parents=True, exist_ok=True)
            memmap_path = work_dir / "top_colmap_xyz.npy"
            mapped = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=np.float32, shape=(grid.height, grid.width, 3))
            raw_filled_nan_replaced = coarse_top_raw.copy()
            if np.any(valid_coarse):
                if not np.all(valid_coarse) and nearest_valid is not None:
                    for ch in range(3):
                        raw_filled_nan_replaced[~valid_coarse, ch] = coarse_top_raw[nearest_valid[0, ~valid_coarse], nearest_valid[1, ~valid_coarse], ch]
            else:
                raw_filled_nan_replaced[:] = 0.0

        block_size = 1024
        col_indices = np.arange(grid.width)
        col_f_1d = (col_indices + 0.5) * (grid.pixel_size / cell) - 0.5
        col_c_1d = np.rint(col_f_1d).astype(np.int64)
        col_in_bounds = (col_c_1d >= 0) & (col_c_1d < coarse_width)
        col_c_clamped = np.clip(col_c_1d, 0, coarse_width - 1)
        if method == "affine":
            x_1d = grid.min_x + (col_indices + 0.5) * grid.pixel_size

        for r_start in range(0, grid.height, block_size):
            r_end = min(grid.height, r_start + block_size)
            row_indices = np.arange(r_start, r_end)
            row_f_1d = (row_indices + 0.5) * (grid.pixel_size / cell) - 0.5
            row_c_1d = np.rint(row_f_1d).astype(np.int64)
            row_in_bounds = (row_c_1d >= 0) & (row_c_1d < coarse_height)
            row_c_clamped = np.clip(row_c_1d, 0, coarse_height - 1)

            col_f_2d, row_f_2d = np.meshgrid(col_f_1d, row_f_1d)

            nearest_validity = coarse_validity[row_c_clamped[:, None], col_c_clamped[None, :]]
            in_bounds = row_in_bounds[:, None] & col_in_bounds[None, :]
            cell_valid = in_bounds & (nearest_validity > 0)

            z_block = ndimage.map_coordinates(z_filled_nan_replaced, [row_f_2d, col_f_2d], order=1, mode="nearest").astype(np.float32)
            z_block[~cell_valid] = np.nan
            dsm[r_start:r_end, :] = z_block

            val_block = np.where(cell_valid, nearest_validity, 0).astype(np.uint8)
            validity[r_start:r_end, :] = val_block
            footprint[r_start:r_end, :] = in_bounds & coarse_footprint[row_c_clamped[:, None], col_c_clamped[None, :]]

            cls_block = coarse_top_class[row_c_clamped[:, None], col_c_clamped[None, :]]
            top_class[r_start:r_end, :] = np.where(cell_valid, cls_block, 0).astype(np.uint8)
            rgb_block = coarse_top_rgb[row_c_clamped[:, None], col_c_clamped[None, :]]
            top_rgb[r_start:r_end, :] = np.where(cell_valid[:, :, None], rgb_block, 0).astype(np.uint8)

            if method == "affine":
                y_1d = grid.max_y - (row_indices + 0.5) * grid.pixel_size
                x_2d, y_2d = np.meshgrid(x_1d, y_1d)
                local_block = np.column_stack((x_2d.ravel(), y_2d.ravel(), z_block.ravel()))
                valid_local = np.isfinite(local_block[:, 2])
                derived = np.full((len(local_block), 3), np.nan, dtype=np.float32)
                if np.any(valid_local):
                    derived[valid_local] = _apply_affine(local_block[valid_local], affine).astype(np.float32)
                top_raw[r_start:r_end, :] = derived.reshape((r_end - r_start, grid.width, 3))
            else:
                for ch in range(3):
                    raw_ch_block = ndimage.map_coordinates(raw_filled_nan_replaced[:, :, ch], [row_f_2d, col_f_2d], order=1, mode="nearest").astype(np.float32)
                    raw_ch_block[~cell_valid] = np.nan
                    mapped[r_start:r_end, :, ch] = raw_ch_block

        if method == "memmap":
            mapped.flush()
            top_raw = mapped

    if progress is not None:
        progress(0.85, "Building surface model")

    dtm = _build_dtm(dsm, top_class, level, config)
    normals = _normals(dsm, grid.pixel_size)
    voxel = np.asarray(voxel_colmap_xyz, dtype=np.float64) if voxel_colmap_xyz is not None else _one_dimensional_voxel_cloud(raw, config.zbuffer.voxel_indoor if level is not None else config.zbuffer.voxel_outdoor)

    if progress is not None:
        progress(1.0, "Building surface model")

    fine_pc, fine_pc_confidence = _rasterize_fine_terms(local, code, fine, grid, level, config)
    return DsmResult(dsm, dtm, top_class, validity, footprint, top_raw, top_rgb, normals, affine if method == "affine" else None, residual, method, voxel, fine_pc, fine_pc_confidence)


def _rasterize_fine_terms(
    local_xyz: np.ndarray,
    classes: np.ndarray,
    fine_terms: np.ndarray,
    grid: OrthoGrid,
    level: Level | None,
    config: OrthoConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise point-cloud fine terms by per-column majority above the floor band."""
    shape = (grid.height, grid.width)
    labels = np.zeros(shape, dtype=np.uint8)
    confidence = np.zeros(shape, dtype=np.float16)
    if level is None or not np.any(fine_terms):
        return labels, confidence

    local = np.asarray(local_xyz, dtype=np.float64)
    code = np.asarray(classes, dtype=np.uint8)
    terms = np.asarray(fine_terms, dtype=np.uint8)
    ceiling = float(level.ceiling_z) if level.ceiling_z is not None else float(level.floor_z + config.indoor.cut_fallback_above_floor)
    rows = np.floor((grid.max_y - local[:, 1]) / grid.pixel_size).astype(np.int64)
    cols = np.floor((local[:, 0] - grid.min_x) / grid.pixel_size).astype(np.int64)
    height = local[:, 2]
    column = (
        np.isfinite(local).all(axis=1)
        & (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
        & (height >= level.floor_z + config.indoor.floor_margin)
        & (height <= ceiling)
        & ~np.isin(code, (100, 101, 102, *config.indoor.ceiling_codes))
    )
    if not np.any(column):
        return labels, confidence

    flat = rows[column] * grid.width + cols[column]
    total = np.zeros(grid.height * grid.width, dtype=np.uint32)
    np.add.at(total, flat, 1)
    candidates = terms[column] > 0
    if not np.any(candidates):
        return labels, confidence
    candidate_flat = flat[candidates]
    candidate_terms = terms[column][candidates]
    candidate_heights = height[column][candidates]
    keys = candidate_flat.astype(np.int64) * 256 + candidate_terms.astype(np.int64)
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    cells = unique // 256
    terms_by_key = (unique % 256).astype(np.uint8)
    top_by_key = np.full(len(unique), -np.inf, dtype=np.float64)
    np.maximum.at(top_by_key, inverse, candidate_heights)
    max_counts = np.zeros_like(total)
    np.maximum.at(max_counts, cells, counts)
    tied = counts == max_counts[cells]
    top_tied = np.full_like(total, -np.inf, dtype=np.float64)
    np.maximum.at(top_tied, cells[tied], top_by_key[tied])
    winners = tied & (top_by_key == top_tied[cells])
    winning_terms = np.zeros_like(total, dtype=np.uint8)
    np.maximum.at(winning_terms, cells[winners], terms_by_key[winners])
    winning_counts = np.zeros_like(total)
    np.maximum.at(winning_counts, cells[winners], counts[winners])
    labels.flat[:] = winning_terms
    supported = winning_terms > 0
    confidence.flat[supported] = (winning_counts[supported] / total[supported]).astype(np.float16)
    return labels, confidence


def _fill_indoor_surface(dsm: np.ndarray, validity: np.ndarray, top_raw: np.ndarray, top_class: np.ndarray, top_rgb: np.ndarray, min_x: float, max_y: float, cell: float, level: Level, config: OrthoConfig) -> np.ndarray:
    """Fill limited indoor footprint gaps by harmonic interpolation or floor-plane fallback."""
    valid = validity > 0
    radius = max(1, int(math.ceil(config.dsm.surface_fill_close_m / cell)))
    axis = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(axis, axis)
    disk = xx * xx + yy * yy <= radius * radius
    footprint = ndimage.binary_fill_holes(ndimage.binary_closing(valid, structure=disk))
    labels, count = ndimage.label(~valid)
    if not valid.any() or count == 0:
        return footprint
    sizes = np.bincount(labels.ravel())
    small = sizes.astype(np.float64) * cell * cell <= config.dsm.surface_fill_max_hole_m2
    small[0] = False
    fill = footprint & ~valid & small[labels]
    if not np.any(fill):
        return footprint

    fallback_plane: np.ndarray | None = None
    fill_labels, fill_count = ndimage.label(fill)
    for component_id in range(1, fill_count + 1):
        component = fill_labels == component_id
        ring = ndimage.binary_dilation(component, structure=_CARDINAL_NEIGHBOURS) & ~component & valid
        if config.dsm.surface_fill_method == "laplace" and np.count_nonzero(ring) >= 8:
            dsm[component] = _laplace_interpolate_component(dsm, component, valid)
        else:
            if fallback_plane is None:
                fallback_plane = _floor_plane(dsm, validity, min_x, max_y, cell, level)
            rows, cols = np.nonzero(component)
            if fallback_plane is None:
                dsm[component] = np.float32(level.floor_z)
            else:
                x = min_x + (cols + 0.5) * cell
                y = max_y - (rows + 0.5) * cell
                dsm[component] = (fallback_plane[0] * x + fallback_plane[1] * y + fallback_plane[2]).astype(np.float32)

    _distances, nearest = ndimage.distance_transform_edt(~valid, return_indices=True)
    near_rows, near_cols = nearest[:, fill]
    top_raw[fill] = top_raw[near_rows, near_cols]
    top_class[fill] = 0
    top_rgb[fill] = 0
    validity[fill] = 3
    return footprint


_CARDINAL_NEIGHBOURS = ndimage.generate_binary_structure(2, 1)


def _laplace_interpolate_component(dsm: np.ndarray, component: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Solve the discrete Laplace equation with valid neighbouring cells as Dirichlet data."""
    rows, cols = np.nonzero(component)
    count = len(rows)
    indices = np.full(component.shape, -1, dtype=np.int32)
    indices[rows, cols] = np.arange(count, dtype=np.int32)
    matrix = sparse.lil_matrix((count, count), dtype=np.float64)
    rhs = np.zeros(count, dtype=np.float64)
    height, width = component.shape
    for index, (row, col) in enumerate(zip(rows, cols, strict=True)):
        diagonal = 0
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            other_row, other_col = row + drow, col + dcol
            if not (0 <= other_row < height and 0 <= other_col < width):
                continue
            diagonal += 1
            other_index = indices[other_row, other_col]
            if other_index >= 0:
                matrix[index, other_index] = -1.0
            elif valid[other_row, other_col]:
                rhs[index] += float(dsm[other_row, other_col])
        matrix[index, index] = float(diagonal)
    values = spsolve(matrix.tocsr(), rhs)
    if not np.isfinite(values).all():
        raise ValueError("Surface-fill Laplace interpolation did not produce finite heights")
    return np.asarray(values, dtype=np.float32)


def _floor_plane(dsm: np.ndarray, validity: np.ndarray, min_x: float, max_y: float, cell: float, level: Level) -> np.ndarray | None:
    """Return the robust floor plane when sufficient measured floor cells exist."""
    rows, cols = np.indices(dsm.shape)
    height = (level.ceiling_z - level.floor_z) if level.ceiling_z is not None else 2.5
    measured = (validity == 1) & (dsm >= level.floor_z - 0.1) & (dsm <= level.floor_z + 0.25 * height)
    if np.count_nonzero(measured) < 50:
        return None
    x = min_x + (cols[measured] + 0.5) * cell
    y = max_y - (rows[measured] + 0.5) * cell
    return _robust_floor_plane(x, y, dsm[measured])


def _robust_floor_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Fit z=ax+by+c after one MAD-based residual rejection pass."""
    design = np.column_stack((x, y, np.ones(len(x))))
    plane, *_ = np.linalg.lstsq(design, z, rcond=None)
    residual = z - design @ plane
    median = np.median(residual)
    mad = np.median(np.abs(residual - median))
    if mad <= np.finfo(float).eps:
        keep = np.abs(residual - median) <= 1e-6
    else:
        keep = np.abs(residual - median) <= 3.0 * 1.4826 * mad
    if np.count_nonzero(keep) >= 3:
        plane, *_ = np.linalg.lstsq(design[keep], z[keep], rcond=None)
    return plane


def _one_dimensional_voxel_cloud(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Fallback for direct callers that do not supply the streaming voxel cloud."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float64)
    _keys, reps = _chunk_voxel_representatives(pts, voxel_size)
    return reps


def _fit_affine(local: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, float]:
    count = min(len(local), 20_000)
    if count < 4:
        return np.zeros((4, 3), dtype=np.float64), float("inf")
    indices = np.linspace(0, len(local) - 1, count, dtype=np.int64)
    design = np.column_stack((local[indices], np.ones(count)))
    matrix, *_ = np.linalg.lstsq(design, raw[indices], rcond=None)
    verify = np.linspace(0, len(local) - 1, min(len(local), 5_000), dtype=np.int64)
    residual = np.linalg.norm(np.column_stack((local[verify], np.ones(len(verify)))) @ matrix - raw[verify], axis=1)
    return matrix, float(np.max(residual, initial=0.0))


def _apply_affine(local: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.column_stack((local, np.ones(len(local)))) @ matrix


def _build_dtm(dsm: np.ndarray, codes: np.ndarray, level: Level | None, config: OrthoConfig) -> np.ndarray:
    if level is not None:
        return np.where(np.isfinite(dsm), np.float32(level.floor_z), np.nan).astype(np.float32)
    ground = np.isin(codes, config.dsm.ground_codes) & np.isfinite(dsm)
    result = np.full(dsm.shape, np.nan, dtype=np.float32)
    result[ground] = dsm[ground]
    if ground.any():
        nearest = ndimage.distance_transform_edt(~ground, return_indices=True)[1]
        result[~ground] = result[nearest[0, ~ground], nearest[1, ~ground]]
    return result


def _normals(dsm: np.ndarray, pixel_size: float) -> np.ndarray:
    filled = np.where(np.isfinite(dsm), dsm, np.nanmedian(dsm) if np.isfinite(dsm).any() else 0.0)
    gy, gx = np.gradient(filled.astype(np.float64), pixel_size)
    normals = np.stack((-gx, gy, np.ones_like(gx)), axis=-1)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    return normals.astype(np.float32)
