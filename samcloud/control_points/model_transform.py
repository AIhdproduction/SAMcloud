"""Apply a world-coordinate similarity to a COLMAP TEXT model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from samcloud.engine.colmap_model import qvec2rotmat


def transform_text_model(
    source: Path, destination: Path, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> None:
    """Transform camera centres and points while retaining cameras and tracks verbatim.

    COLMAP 4.2 exposes ``model_transformer`` but its transform-file grammar is
    undocumented in the shipped help. This explicit TEXT transformation avoids
    relying on that private format. The world change is ``X' = s R X + t``.
    """
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.txt",):
        _copy_if_present(source / name, destination / name)
    _transform_rigs(source / "rigs.txt", destination / "rigs.txt", scale)
    _transform_frames(source / "frames.txt", destination / "frames.txt", scale, rotation, translation)
    _transform_images(source / "images.txt", destination / "images.txt", scale, rotation, translation)
    points = source / "points3D.txt"
    if points.exists():
        _transform_points(points, destination / "points3D.txt", scale, rotation, translation)


def _transform_images(source: Path, destination: Path, scale: float, rotation: np.ndarray, translation: np.ndarray) -> None:
    output: list[str] = []
    lines = source.read_text(encoding="utf-8").splitlines()
    image_record = True
    for line in lines:
        if not line or line.startswith("#"):
            output.append(line)
            continue
        if not image_record:
            output.append(line)
            image_record = True
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"Invalid COLMAP image record: {line}")
        # Rcw' = Rcw Rs^T, tcw' = s tcw - Rcw' ts.
        transformed_rotation, transformed_translation = _transform_pose(
            fields[1:5], fields[5:8], scale, rotation, translation
        )
        fields[1:5] = [_format(value) for value in rotmat2qvec(transformed_rotation)]
        fields[5:8] = [_format(value) for value in transformed_translation]
        output.append(" ".join(fields))
        image_record = False
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


def _transform_frames(source: Path, destination: Path, scale: float, rotation: np.ndarray, translation: np.ndarray) -> None:
    """Transform every RIG_FROM_WORLD pose while retaining frame data IDs."""
    if not source.exists():
        return
    output = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            output.append(line)
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"Invalid COLMAP frame record: {line}")
        transformed_rotation, transformed_translation = _transform_pose(fields[2:6], fields[6:9], scale, rotation, translation)
        fields[2:6] = [_format(value) for value in rotmat2qvec(transformed_rotation)]
        fields[6:9] = [_format(value) for value in transformed_translation]
        output.append(" ".join(fields))
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


def _transform_rigs(source: Path, destination: Path, scale: float) -> None:
    """Copy rigs and scale internal sensor translations when a sensor has a pose.

    Sensor rotations and translations are expressed in rig coordinates. A world
    rotation never changes them, but the similarity scale changes rig units.
    """
    if not source.exists():
        return
    output = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            output.append(line)
            continue
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid COLMAP rig record: {line}")
        try:
            non_reference_sensors = int(fields[1]) - 1
        except ValueError as error:
            raise ValueError(f"Invalid COLMAP rig sensor count: {line}") from error
        if non_reference_sensors < 0:
            raise ValueError(f"Invalid COLMAP rig sensor count: {line}")
        index = 4
        for _ in range(non_reference_sensors):
            if len(fields) < index + 3:
                raise ValueError(f"Malformed COLMAP rig sensor entry: {line}")
            try:
                has_pose = int(fields[index + 2])
            except ValueError as error:
                raise ValueError(f"Malformed COLMAP rig sensor pose flag: {line}") from error
            if has_pose not in {0, 1}:
                raise ValueError(f"Malformed COLMAP rig sensor pose flag: {line}")
            index += 3
            if has_pose:
                if len(fields) < index + 7:
                    raise ValueError(f"Malformed COLMAP rig sensor pose: {line}")
                fields[index + 4:index + 7] = [_format(scale * float(value)) for value in fields[index + 4:index + 7]]
                index += 7
        if len(fields) != index:
            raise ValueError(f"Malformed COLMAP rig record: {line}")
        output.append(" ".join(fields))
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


def _transform_pose(qvec, tvec, scale: float, rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_rotation = qvec2rotmat(np.asarray(qvec, dtype=float))
    camera_translation = np.asarray(tvec, dtype=float)
    transformed_rotation = camera_rotation @ rotation.T
    return transformed_rotation, scale * camera_translation - transformed_rotation @ translation


def _copy_if_present(source: Path, destination: Path) -> None:
    if source.exists():
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _transform_points(source: Path, destination: Path, scale: float, rotation: np.ndarray, translation: np.ndarray) -> None:
    output: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            output.append(line)
            continue
        fields = line.split()
        if len(fields) < 8:
            output.append(line)
            continue
        transformed = scale * rotation @ np.asarray(fields[1:4], dtype=float) + translation
        fields[1:4] = [_format(value) for value in transformed]
        output.append(" ".join(fields))
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


def rotmat2qvec(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation to COLMAP's scalar-first quaternion."""
    trace = float(np.trace(rotation))
    if trace > 0:
        factor = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array([0.25 * factor, (rotation[2, 1] - rotation[1, 2]) / factor, (rotation[0, 2] - rotation[2, 0]) / factor, (rotation[1, 0] - rotation[0, 1]) / factor])
    else:
        index = int(np.argmax(np.diag(rotation)))
        next_index, last_index = (index + 1) % 3, (index + 2) % 3
        factor = 2.0 * np.sqrt(1.0 + rotation[index, index] - rotation[next_index, next_index] - rotation[last_index, last_index])
        quaternion = np.zeros(4)
        quaternion[index + 1] = 0.25 * factor
        quaternion[0] = (rotation[last_index, next_index] - rotation[next_index, last_index]) / factor
        quaternion[next_index + 1] = (rotation[next_index, index] + rotation[index, next_index]) / factor
        quaternion[last_index + 1] = (rotation[last_index, index] + rotation[index, last_index]) / factor
    return quaternion if quaternion[0] >= 0 else -quaternion


def _format(value: float) -> str:
    return format(float(value), ".17g")
