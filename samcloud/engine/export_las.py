"""Stream a classified binary PLY to a LAS 1.4 file."""

import argparse
import json
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS, Transformer

from samcloud.pointcloud.georeference import (
    enu_to_target,
    export_height_system,
    export_target_crs,
    format_residuals,
    load_georeference,
    require_export_grids,
    strict_transformer,
)


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
LEGACY_CLASSIFICATION_CODE_SETS = {
    "outdoor": np.array(
        [
            0, 2, 11, 66, 67, 3, 5, 6, 68, 69, 70, 71, 72, 14, 73, 74, 9,
        ],
        dtype=np.uint8,
    ),
    "indoor": np.array(
        [
            0, 2, 75, 6, 76, 77, 78, 79, 80, 81, 82,
        ],
        dtype=np.uint8,
    ),
}


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


def classification_codes(config: dict, class_set: str) -> np.ndarray:
    """LAS code per class id: ``las_codes`` of classes.json (Nomivox) or the legacy Cyclone table."""
    if class_set not in config or class_set not in LEGACY_CLASSIFICATION_CODE_SETS:
        raise ValueError(f"Unsupported class set: {class_set}")
    codes = config.get("las_codes", {}).get(class_set)
    if codes is None:
        return LEGACY_CLASSIFICATION_CODE_SETS[class_set]
    if len(codes) != len(config[class_set]) + 1:
        raise ValueError(f"classes.json: las_codes.{class_set} needs one code per class plus unknown")
    return np.asarray(codes, dtype=np.uint8)


def load_class_mapping(path: Path, target_crs: str, class_set: str) -> dict[str, object]:
    """Build human-readable metadata from the pipeline class configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    class_names = [name.split(",", 1)[0].strip() for name in config[class_set]]
    source_names = [config.get("unknown_class_name", "Unknown/noise"), *class_names]
    mapping = {
        str(int(las_code)): source_names[source_id]
        for source_id, las_code in enumerate(classification_codes(config, class_set))
    }
    return {
        "classification_field": "classification",
        "coordinate_reference": export_target_crs(target_crs),
        "height_system": export_height_system(target_crs),
        "classification_scheme": "Cyclone 3DR / ASPRS LAS 1.4",
        "semantic_class_set": class_set,
        "class_mapping": mapping,
    }


def export_las(
    input_path: Path,
    output_path: Path,
    classes_path: Path,
    chunk_size: int,
    source_crs: str,
    target_crs: str,
    class_set: str,
    georeference_path: Path | None = None,
) -> int:
    payload_offset, vertex_count = read_ply_header(input_path)
    output_crs = export_target_crs(target_crs)
    require_export_grids(target_crs)
    georeference = _load_georeference_for_export(source_crs, target_crs, georeference_path)
    transformer = _create_transformer(source_crs, output_crs, georeference is not None)
    codes = classification_codes(json.loads(Path(classes_path).read_text(encoding="utf-8")), class_set)
    metadata = json.dumps(
        load_class_mapping(classes_path, target_crs, class_set), separators=(",", ":")
    ).encode("utf-8")

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = _las_scales(output_crs)
    header.offsets = _offsets_from_transformed_data(
        input_path, payload_offset, vertex_count, chunk_size, transformer, georeference, target_crs
    )
    if target_crs.upper() != "LOCAL":
        header.add_crs(CRS.from_user_input(output_crs))
    header.vlrs.append(
        laspy.VLR(
            user_id="colmap_sam",
            record_id=1,
            description="Semantic class mapping and height system",
            record_data=metadata,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("rb") as source, laspy.open(output_path, mode="w", header=header) as destination:
        for points in _ply_chunks(source, payload_offset, vertex_count, chunk_size):
            count = len(points)
            xyz = np.column_stack((points["x"], points["y"], points["z"]))
            xyz = _transform_xyz(xyz, transformer, georeference, target_crs)

            las_points = laspy.ScaleAwarePointRecord.zeros(count, header=header)
            las_points.x = xyz[:, 0]
            las_points.y = xyz[:, 1]
            las_points.z = xyz[:, 2]
            las_points.red = points["red"].astype(np.uint16) * 257
            las_points.green = points["green"].astype(np.uint16) * 257
            las_points.blue = points["blue"].astype(np.uint16) * 257
            if points["scalar_class"].max(initial=0) >= len(codes):
                raise ValueError("PLY contains a class ID not defined by this export")
            las_points.classification = codes[points["scalar_class"]]
            las_points.return_number = np.ones(count, dtype=np.uint8)
            las_points.number_of_returns = np.ones(count, dtype=np.uint8)
            destination.write_points(las_points)

    if georeference is not None:
        print(f"Georeference residuals: {format_residuals(georeference)}")
    print(f"Export height system: {export_height_system(target_crs)}")

    return vertex_count


def _create_transformer(
    source_crs: str,
    target_crs: str,
    has_georeference: bool = False,
) -> Transformer | None:
    """Return a transformer or validate that no transformation is required."""
    source = source_crs.upper()
    target = target_crs.upper()
    if source == target:
        return None
    if source == "LOCAL":
        if has_georeference:
            return None
        raise ValueError(
            "This project is in LOCAL coordinates and has no GPS georeference. "
            "It must be exported as LOCAL; georeferencing via control points is not available yet."
        )
    if target == "LOCAL":
        raise ValueError("A defined CRS cannot be transformed to LOCAL coordinates")
    return strict_transformer(source_crs, target_crs)


def _load_georeference_for_export(
    source_crs: str,
    target_crs: str,
    georeference_path: Path | None,
) -> dict[str, object] | None:
    if source_crs.upper() != "LOCAL" or target_crs.upper() == "LOCAL":
        return None
    if georeference_path is None:
        return None
    path = Path(georeference_path)
    if not path.is_file():
        raise ValueError(f"Georeference file was not found: {path}")
    return load_georeference(path)


def _ply_chunks(source, payload_offset: int, vertex_count: int, chunk_size: int):
    source.seek(payload_offset)
    remaining = vertex_count
    while remaining:
        count = min(chunk_size, remaining)
        data = source.read(count * PLY_DTYPE.itemsize)
        if len(data) != count * PLY_DTYPE.itemsize:
            raise ValueError("PLY data ended before all declared vertices were read")
        yield np.frombuffer(data, dtype=PLY_DTYPE)
        remaining -= count


def _transform_xyz(
    xyz: np.ndarray,
    transformer: Transformer | None,
    georeference: dict[str, object] | None,
    target_crs: str,
) -> np.ndarray:
    if georeference is not None:
        return enu_to_target(xyz, georeference, target_crs)
    if transformer is None:
        return xyz
    x, y, z = transformer.transform(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    return np.column_stack((x, y, z))


def _offsets_from_transformed_data(
    input_path: Path,
    payload_offset: int,
    vertex_count: int,
    chunk_size: int,
    transformer: Transformer | None,
    georeference: dict[str, object] | None,
    target_crs: str,
) -> np.ndarray:
    minimum: np.ndarray | None = None
    with input_path.open("rb") as source:
        for points in _ply_chunks(source, payload_offset, vertex_count, chunk_size):
            xyz = np.column_stack((points["x"], points["y"], points["z"]))
            transformed = _transform_xyz(xyz, transformer, georeference, target_crs)
            if not np.all(np.isfinite(transformed)):
                raise ValueError("Coordinate transformation produced non-finite values")
            chunk_minimum = transformed.min(axis=0)
            minimum = chunk_minimum if minimum is None else np.minimum(minimum, chunk_minimum)
    if minimum is None:
        raise ValueError("PLY contains no vertices")
    return np.floor(minimum)


def _las_scales(target_crs: str) -> np.ndarray:
    """Return LAS scales appropriate for geographic or metre-based coordinates."""
    if target_crs.upper() == "LOCAL":
        return np.array([0.001, 0.001, 0.001])
    if CRS.from_user_input(target_crs).is_geographic:
        return np.array([1e-7, 1e-7, 0.001])
    return np.array([0.001, 0.001, 0.001])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a classified PLY as LAS 1.4")
    parser.add_argument("--input", required=True, type=Path, help="classified PLY with scalar_class")
    parser.add_argument("--output", required=True, type=Path, help="destination LAS path")
    parser.add_argument("--classes-json", required=True, type=Path, help="pipeline class configuration")
    parser.add_argument("--class-set", choices=("outdoor", "indoor"), default="outdoor",
                        help="semantic class set used in the PLY")
    parser.add_argument("--source-crs", default="LOCAL", help="CRS of the input point cloud, for example EPSG:2056")
    parser.add_argument("--target-crs", default="LOCAL", help="CRS written to LAS, for example EPSG:2056")
    parser.add_argument("--georeference", type=Path, default=None,
                        help="ENU-to-ECEF georeference JSON for LOCAL GPS-aligned projects")
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
        args.class_set,
        args.georeference,
    )
    print(f"Wrote {count:,} classified points to {args.output}")


if __name__ == "__main__":
    main()
