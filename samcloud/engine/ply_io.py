"""
Reads a COLMAP fused.ply (dense point cloud) and its companion fused.ply.vis
visibility file, and writes the classified result back out as PLY with an
extra 'class' scalar field (uint8).
"""

import numpy as np
import struct
from plyfile import PlyData, PlyElement


def read_fused_points(ply_path):
    """Returns Nx3 float32 xyz array and Nx3 uint8 rgb array (rgb may be None)."""
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rgb = None
    if all(name in v.data.dtype.names for name in ("red", "green", "blue")):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


def read_vis_file(vis_path):
    """
    COLMAP fused.ply.vis format, per point in the same order as fused.ply:
      uint32 num_visible_images
      uint32[num_visible_images] image_ids
    Returns a list of numpy arrays (one per point), each holding the image
    ids that observe that point.
    """
    vis = []
    with open(vis_path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            n = struct.unpack("<I", f.read(4))[0]
            if n > 0:
                ids = np.frombuffer(f.read(4 * n), dtype=np.uint32)
            else:
                ids = np.empty(0, dtype=np.uint32)
            vis.append(ids)
    return vis


def write_classified_ply(out_path, xyz, rgb, class_ids):
    n = xyz.shape[0]
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("scalar_class", "u1"),
    ]
    data = np.empty(n, dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    else:
        data["red"] = data["green"] = data["blue"] = 200
    data["scalar_class"] = class_ids.astype(np.uint8)
    el = PlyElement.describe(data, "vertex")
    PlyData([el], text=False).write(str(out_path))
