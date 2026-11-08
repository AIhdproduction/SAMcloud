"""
Reads a COLMAP fused.ply (dense point cloud) and its companion fused.ply.vis
visibility file, and writes the classified result back out as PLY with an
extra 'class' scalar field (uint8).
"""

import numpy as np
import struct
from plyfile import PlyData, PlyElement


_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_vertex_array(ply_path):
    """The vertex element of a PLY as an independent numpy structured array.

    plyfile without memory mapping parses large files very slowly (about 4 s
    per million points); a binary file whose first element is ``vertex`` with
    plain scalar properties is read here in one ``np.fromfile`` call instead.
    The result does not keep the file open, so it can be rewritten in place.
    Anything else (ASCII, list properties, other element order) goes through
    plyfile.
    """
    with open(ply_path, "rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {ply_path}")
        fmt, elements, current = None, [], None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header without end_header: {ply_path}")
            words = line.decode("ascii", errors="replace").split()
            if not words or words[0] in ("comment", "obj_info"):
                continue
            if words[0] == "end_header":
                break
            if words[0] == "format":
                fmt = words[1]
            elif words[0] == "element":
                current = [words[1], int(words[2]), []]
                elements.append(current)
            elif words[0] == "property" and current is not None:
                current[2].append(None if words[1] == "list" else (words[2], words[1]))
        simple = (
            fmt in ("binary_little_endian", "binary_big_endian") and elements and elements[0][0] == "vertex"
            and all(prop is not None and prop[1] in _PLY_TYPES for prop in elements[0][2])
        )
        if simple:
            order = "<" if fmt == "binary_little_endian" else ">"
            dtype = np.dtype([(name, order + _PLY_TYPES[kind]) for name, kind in elements[0][2]])
            data = np.fromfile(handle, dtype=dtype, count=elements[0][1])
            if len(data) != elements[0][1]:
                raise ValueError(f"Truncated PLY vertex data: {ply_path}")
            return data
    return PlyData.read(str(ply_path), mmap=False)["vertex"].data


def read_fused_points(ply_path):
    """Returns Nx3 float32 xyz array and Nx3 uint8 rgb array (rgb may be None)."""
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rgb = None
    if all(name in v.data.dtype.names for name in ("red", "green", "blue")):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


def read_classified_ply(ply_path):
    """Return xyz, RGB and scalar class IDs from a classified PLY.

    Read without memory-mapping and return copies: a live map keeps the file
    locked on Windows and a later in-place rewrite fails with Errno 22.
    """
    vertex = read_vertex_array(ply_path)
    names = vertex.dtype.names
    if "scalar_class" not in names:
        raise ValueError(f"Classified PLY has no scalar_class property: {ply_path}")
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb, np.array(vertex["scalar_class"], dtype=np.uint8, copy=True)


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
