"""Stream a classified binary PLY to a LAS 1.4 file."""

import argparse
import json
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS, Transformer


PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("scalar_class", "u1"),
    ]
)

# LAS class codes used by Cyclone 3DR. Standard ASPRS classes are used whenever
# one describes the semantic class. The remaining classes use the Cyclone
# recommended user-defined range starting at 66.
CLASSIFICATION_CODES = np.array(
    [
        0,   # Unknown/noise -> Unclassified
        2,   # Ground
        11,  # Road surface
        66,  # Sidewalk
        67,  # Curb
        3,   # Low vegetation
        5,   # High vegetation / tree
        6,   # Building
        68,  # Vehicle
        69,  # Construction machinery
        70,  # Utility pole / streetlight
        71,  # Traffic sign
        72,  # Fence / barrier
        14,  # Wire - conductor
        73,  # Scaffolding
        74,  # Construction container / material
        9,   # Water
    ],
    dtype=np.uint8,
)


def read_ply_header(path: Path) -> tuple[int, int]:
    """Return the binary payload offset and vertex count of a supported PLY."""
    with path.open("rb") as source:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            line = source.readline()
            if not line:
                raise ValueError("PLY header is incomplete")
            header.extend(line)
            if len(header) > 65536:
                raise ValueError("PLY header is unexpectedly large")

        header_text = header.decode("ascii")
        if "format binary_little_endian 1.0" not in header_text:
            raise ValueError("Only binary_little_endian PLY files are supported")
        if "property uchar scalar_class" not in header_text:
            raise ValueError("PLY does not contain the scalar_class property")

        vertex_count = None
        for line in header_text.splitlines():
            fields = line.split()
            if len(fields) == 3 and fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                break
        if vertex_count is None:
            raise ValueError("PLY header does not contain a vertex count")

        return source.tell(), vertex_count


def load_class_mapping(path: Path, target_crs: str) -> dict[str, object]:
    """Build human-readable metadata from the pipeline class configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    class_names = config["aussen"]
    source_names = [config.get("unknown_class_name", "Unknown/noise"), *class_names]
    mapping = {
        str(las_code): source_names[source_id]
        for source_id, las_code in enumerate(CLASSIFICATION_CODES)
    }
    return {
        "classification_field": "classification",
        "coordinate_reference": target_crs,
        "classification_scheme": "Cyclone 3DR / ASPRS LAS 1.4",
        "class_mapping": mapping,
    }


def export_las(
    input_path: Path,
    output_path: Path,
    classes_path: Path,
    chunk_size: int,
    source_crs: str,
    target_crs: str,
) -> int:
    payload_offset, vertex_count = read_ply_header(input_path)
    transformer = _create_transformer(source_crs, target_crs)
    metadata = json.dumps(load_class_mapping(classes_path, target_crs), separators=(",", ":")).encode("utf-8")

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0.0, 0.0, 0.0])
    if target_crs.upper() != "LOCAL":
        header.add_crs(CRS.from_user_input(target_crs))
    header.vlrs.append(
        laspy.VLR(
            user_id="colmap_sam",
            record_id=1,
            description="Semantic class mapping",
            record_data=metadata,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("rb") as source, laspy.open(output_path, mode="w", header=header) as destination:
        source.seek(payload_offset)
        remaining = vertex_count
        while remaining:
            count = min(chunk_size, remaining)
            data = source.read(count * PLY_DTYPE.itemsize)
            if len(data) != count * PLY_DTYPE.itemsize:
                raise ValueError("PLY data ended before all declared vertices were read")
            points = np.frombuffer(data, dtype=PLY_DTYPE)
            xyz = np.column_stack((points["x"], points["y"], points["z"]))
            if transformer is not None:
                x, y, z = transformer.transform(xyz[:, 0], xyz[:, 1], xyz[:, 2])
                xyz = np.column_stack((x, y, z))

            las_points = laspy.ScaleAwarePointRecord.zeros(count, header=header)
            las_points.x = xyz[:, 0]
            las_points.y = xyz[:, 1]
            las_points.z = xyz[:, 2]
            las_points.red = points["red"].astype(np.uint16) * 257
            las_points.green = points["green"].astype(np.uint16) * 257
            las_points.blue = points["blue"].astype(np.uint16) * 257
            if points["scalar_class"].max(initial=0) >= len(CLASSIFICATION_CODES):
                raise ValueError("PLY contains a class ID not defined by this export")
            las_points.classification = CLASSIFICATION_CODES[points["scalar_class"]]
            las_points.return_number = np.ones(count, dtype=np.uint8)
            las_points.number_of_returns = np.ones(count, dtype=np.uint8)
            destination.write_points(las_points)
            remaining -= count

    return vertex_count


def _create_transformer(source_crs: str, target_crs: str) -> Transformer | None:
    """Return a transformer or validate that no transformation is required."""
    source = source_crs.upper()
    target = target_crs.upper()
    if source == target:
        return None
    if "LOCAL" in {source, target}:
        raise ValueError("A local point cloud needs a defined CRS before transformation")
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a classified PLY as LAS 1.4")
    parser.add_argument("--input", required=True, type=Path, help="classified PLY with scalar_class")
    parser.add_argument("--output", required=True, type=Path, help="destination LAS path")
    parser.add_argument("--classes-json", required=True, type=Path, help="pipeline class configuration")
    parser.add_argument("--source-crs", default="LOCAL", help="CRS of the input point cloud, for example EPSG:2056")
    parser.add_argument("--target-crs", default="LOCAL", help="CRS written to LAS, for example EPSG:2056")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="points written per batch")
    args = parser.parse_args()

    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    count = export_las(
        args.input,
        args.output,
        args.classes_json,
        args.chunk_size,
        args.source_crs,
        args.target_crs,
    )
    print(f"Wrote {count:,} classified points to {args.output}")


if __name__ == "__main__":
    main()
