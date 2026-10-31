import struct
from pathlib import Path

from samcloud.viewer.preview import ensure_preview


def test_dense_preview_samples_binary_ply(tmp_path: Path):
    project = tmp_path / "project"
    source = project / "products" / "classified" / "classified.ply"
    source.parent.mkdir(parents=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar scalar_class\nend_header\n"
    ).encode("ascii")
    records = b"".join(struct.pack("<fffBBBB", float(i), i + 1, i + 2, 10, 20, 30, i) for i in range(3))
    source.write_bytes(header + records)

    preview, count = ensure_preview(project, "dense", max_points=2)

    assert count == 2
    payload = preview.read_bytes()
    assert payload[:4] == b"SAMV"
    assert struct.unpack("<I", payload[8:12])[0] == 2
    assert len(payload) == 16 + 2 * 16


def test_sparse_preview_reads_colmap_binary_points(tmp_path: Path):
    project = tmp_path / "project"
    source = project / "products" / "sparse" / "0" / "points3D.bin"
    source.parent.mkdir(parents=True)
    records = []
    for point_id in range(3):
        records.append(struct.pack("<QdddBBBdQ", point_id, point_id, 0.0, 0.0, 100, 110, 120, 0.1, 0))
    source.write_bytes(struct.pack("<Q", 3) + b"".join(records))

    preview, count = ensure_preview(project, "sparse", max_points=2)

    assert count == 2
    assert preview.stat().st_size == 16 + 2 * 16
