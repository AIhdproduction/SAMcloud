"""Per-point fine terms: which object a classified point is (sofa, bed, cat tree ...).

SAM3 scores every prompt term as its own query, but the terms are pooled into
classes. The labeler keeps the term scores of the configured fine vocabulary
(``fine_terms`` in ``config/classes.json``), the classifier votes them per point
like the class scores and stores the winner next to the class:

* ``classified_scores.npz``: ``fine_term`` (uint8, 0 = none, 1..F = index into
  ``fine_labels`` + 1), ``fine_conf`` (float16), ``fine_labels`` (string array).
* ``products/classified/classified.ply``: vertex property ``fine_term`` plus the
  vocabulary file ``fine_labels.json`` next to it.

Step 6 of the pipeline adds, removes, moves and relabels points. The terms are
not carried through every stage: they are snapshotted before step 6 and moved
onto the final cloud afterwards by nearest neighbour (:func:`attach_fine_terms`).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FINE_NONE = 0
FINE_RADIUS_M = 0.05
SOURCE_NAME = "fine_terms_source.npz"
LABELS_NAME = "fine_labels.json"
FINE_KEYS = ("fine_term", "fine_conf", "fine_labels")


@dataclass
class FineVocabulary:
    """Canonical fine labels, the class each one belongs to and the SAM queries that answer it."""

    labels: list[str]
    owner_class: np.ndarray  # 1-based class index per label (the scalar_class value of its class)
    queries: list[list[int]] = field(default_factory=list)  # flat prompt indices per label

    @property
    def size(self) -> int:
        return len(self.labels)


def _flatten(prompts) -> tuple[list[str], np.ndarray]:
    flat, owner = [], []
    for class_index, entry in enumerate(prompts):
        items = [entry] if isinstance(entry, str) else list(entry)
        flat.extend(items)
        owner.extend([class_index] * len(items))
    return flat, np.asarray(owner, dtype=np.int64)


def canonical_label(term: str, aliases: dict[str, str] | None = None) -> str:
    """Canonical stored label of a term: ``couch`` -> ``sofa``, ``pillow`` -> ``cushion``."""
    name = str(term).strip().lower()
    mapping = {str(key).strip().lower(): str(value).strip().lower() for key, value in (aliases or {}).items()}
    seen = set()
    while name in mapping and name not in seen:  # tolerate chained aliases
        seen.add(name)
        name = mapping[name]
    return name


def _owner_class_index(config: dict, class_set: str, term: str, label: str) -> int | None:
    """1-based class index of the class that owns a term (via ``fine_term_classes`` LAS codes)."""
    owners = (config.get("fine_term_classes") or {}).get(class_set) or {}
    code = owners.get(term, owners.get(label))
    codes = list((config.get("las_codes") or {}).get(class_set) or [])
    if code is None or int(code) not in codes[1:]:
        return None
    return codes.index(int(code))


def fine_term_list(config: dict, class_set: str) -> list[str]:
    return [str(term).strip().lower() for term in ((config.get("fine_terms") or {}).get(class_set) or [])]


def build_fine_vocabulary(config: dict, class_set: str, prompts, warn=print) -> FineVocabulary | None:
    """Resolve the fine vocabulary against the SAM prompt list that is actually queried.

    ``prompts`` is the flat-able per-class prompt list (classes first, internal queries after).
    Terms that are not a query of their owner class are dropped with a warning, so the
    vocabulary never claims a term SAM did not score. Returns None when there is nothing to do.
    """
    terms = fine_term_list(config, class_set)
    if not terms or prompts is None:
        return None
    aliases = config.get("fine_aliases") or {}
    flat, flat_owner = _flatten(prompts)
    lowered = [str(item).strip().lower() for item in flat]
    labels: list[str] = []
    owners: list[int] = []
    queries: list[list[int]] = []
    for term in terms:
        label = canonical_label(term, aliases)
        class_index = _owner_class_index(config, class_set, term, label)
        if class_index is None:
            warn(f"WARNING: fine term '{term}' has no owner class and is skipped")
            continue
        found = [i for i, name in enumerate(lowered) if name == term and flat_owner[i] == class_index - 1]
        if not found:
            warn(f"WARNING: fine term '{term}' is not a SAM query of its class and is skipped")
            continue
        if label in labels:
            position = labels.index(label)
            if owners[position] != class_index:
                warn(f"WARNING: alias '{term}' belongs to another class than '{label}' and is skipped")
                continue
            queries[position].extend(i for i in found if i not in queries[position])
        else:
            labels.append(label)
            owners.append(class_index)
            queries.append(list(found))
    if not labels:
        return None
    return FineVocabulary(labels, np.asarray(owners, dtype=np.uint8), queries)


def resolve_fine_terms(scores, counts, background_threshold: float, chunk_size: int = 1_000_000):
    """Winning fine term per point from the accumulated weighted votes.

    ``scores`` is (N, F) summed term scores over the views a point was visible in and ``counts``
    the number of those views. The winner is the term with the highest sum; it is kept when its
    mean score reaches the background threshold. Returns ``fine_term`` (uint8, 0 = none, 1..F)
    and ``fine_conf`` (float16, the mean score of the winner).
    """
    num_points = scores.shape[0]
    term = np.zeros(num_points, dtype=np.uint8)
    conf = np.zeros(num_points, dtype=np.float16)
    if scores.shape[1] == 0:
        return term, conf
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        block = scores[start:end]
        best = np.argmax(block, axis=1)
        views = counts[start:end].astype(np.float32)
        mean = np.zeros(end - start, dtype=np.float32)
        valid = views > 0
        mean[valid] = block[np.arange(end - start)[valid], best[valid]] / views[valid]
        keep = valid & (mean >= background_threshold)
        term[start:end] = np.where(keep, best + 1, FINE_NONE).astype(np.uint8)
        conf[start:end] = np.where(keep, mean, 0.0).astype(np.float16)
    return term, conf


def mask_removed_class_votes(term_probs, point_probs, owner_class) -> None:
    """Zero the term scores whose owner class lost its vote in this view (occluded / depth-checked)."""
    if term_probs.size == 0:
        return
    alive = point_probs[:, np.asarray(owner_class, dtype=np.intp) - 1] > 0
    term_probs *= alive


# --------------------------------------------------------------------------- step 6 hand-over


def _read_vertex(path: Path):
    from samcloud.engine.ply_io import read_vertex_array

    return read_vertex_array(path)


def snapshot_fine_terms(classified_ply: Path, scores_path: Path, source_path: Path | None = None) -> bool:
    """Move the fine arrays out of the scores file before step 6 and remember where the points were.

    Step 6 stages keep ``classified_scores.npz`` row-aligned with the cloud and treat every array in
    it as per-point. The vocabulary (and the terms, which later stages cannot carry) would break that,
    so they leave the file and go to ``fine_terms_source.npz`` together with the SAM-stage positions
    and classes. Returns True when a snapshot was written.
    """
    scores_path = Path(scores_path)
    if not scores_path.is_file():
        return False
    with np.load(scores_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if "fine_term" not in arrays or "fine_labels" not in arrays:
        return False
    vertex = _read_vertex(Path(classified_ply))
    if len(vertex) != len(arrays["fine_term"]):
        return False
    source_path = Path(source_path) if source_path else scores_path.with_name(SOURCE_NAME)
    np.savez(
        source_path,
        xyz=np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float32),
        scalar_class=np.asarray(vertex["scalar_class"], dtype=np.uint8),
        fine_term=np.asarray(arrays["fine_term"], dtype=np.uint8),
        fine_conf=np.asarray(arrays.get("fine_conf", np.zeros(len(vertex), np.float16)), dtype=np.float16),
        fine_labels=np.asarray(arrays["fine_labels"]),
    )
    del vertex
    stripped = {name: values for name, values in arrays.items() if name not in FINE_KEYS}
    np.savez(scores_path, **stripped)
    return True


def label_owner_classes(config: dict, class_set: str, labels) -> np.ndarray:
    """1-based owner class per canonical label (0 when the label is unknown to the configuration)."""
    aliases = config.get("fine_aliases") or {}
    owners = np.zeros(len(labels), dtype=np.uint8)
    reverse: dict[str, list[str]] = {}
    for term in fine_term_list(config, class_set):
        reverse.setdefault(canonical_label(term, aliases), []).append(term)
    for position, label in enumerate(labels):
        for term in [str(label), *reverse.get(str(label), [])]:
            index = _owner_class_index(config, class_set, term, str(label))
            if index is not None:
                owners[position] = index
                break
    return owners


def inherit_fine_terms(
    source_xyz, source_term, source_conf, target_xyz, target_class, owner_class, radius: float = FINE_RADIUS_M,
):
    """Fine term of every target point: nearest source point within ``radius``, kept only while the
    target's class is the class that owns the term. Returns (fine_term uint8, fine_conf float16)."""
    from scipy.spatial import cKDTree

    count = len(target_xyz)
    term = np.zeros(count, dtype=np.uint8)
    conf = np.zeros(count, dtype=np.float16)
    if count == 0 or len(source_xyz) == 0 or not np.any(source_term):
        return term, conf
    tree = cKDTree(np.asarray(source_xyz, dtype=np.float64))
    owners = np.concatenate(([0], np.asarray(owner_class, dtype=np.uint8)))
    for start in range(0, count, 500_000):
        end = min(start + 500_000, count)
        distance, nearest = tree.query(np.asarray(target_xyz[start:end], dtype=np.float64), k=1,
                                       distance_upper_bound=float(radius) * (1.0 + 1e-9), workers=-1)
        found = np.isfinite(distance)
        index = np.where(found, nearest, 0)
        candidate = np.where(found, source_term[index], 0).astype(np.uint8)
        allowed = (candidate > 0) & (owners[candidate] == np.asarray(target_class[start:end], dtype=np.uint8))
        term[start:end] = np.where(allowed, candidate, FINE_NONE)
        conf[start:end] = np.where(allowed, source_conf[index], 0.0).astype(np.float16)
    return term, conf


def write_fine_labels_json(path: Path, class_set: str, labels, owner_class, config: dict) -> None:
    codes = list((config.get("las_codes") or {}).get(class_set) or [])
    payload = {
        "schema": 1,
        "class_set": class_set,
        "property": "fine_term",
        "none": FINE_NONE,
        "labels": [str(label) for label in labels],  # property value = list position + 1
        "ids": {str(label): index for index, label in enumerate(labels, start=1)},
        "owner_class_id": {str(label): int(owner) for label, owner in zip(labels, owner_class)},
        "owner_las_code": {
            str(label): int(codes[int(owner)]) for label, owner in zip(labels, owner_class) if 0 < int(owner) < len(codes)
        },
        "aliases": dict(config.get("fine_aliases") or {}),
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_ply_with_property(path: Path, vertex, fine_term) -> None:
    from plyfile import PlyData, PlyElement

    names = [name for name in vertex.dtype.names if name != "fine_term"]
    dtype = [(name, vertex.dtype[name]) for name in names] + [("fine_term", "u1")]
    data = np.empty(len(vertex), dtype=dtype)
    for name in names:
        data[name] = vertex[name]
    data["fine_term"] = fine_term
    handle, temporary = tempfile.mkstemp(prefix=f".{path.stem}.fine_", suffix=path.suffix, dir=path.parent)
    os.close(handle)
    try:
        PlyData([PlyElement.describe(data, "vertex")], text=False).write(temporary)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def attach_fine_terms(
    source_path: Path, final_ply: Path, classes_json: Path, class_set: str, radius: float = FINE_RADIUS_M,
    keep_source: bool = False,
) -> dict:
    """Write ``fine_term`` into the final product PLY and the vocabulary next to it.

    Survivors keep their term (nearest source point at distance 0), added or moved points take the
    term of the nearest source point within ``radius``, and every term is dropped when the point's
    final class is not the class that owns it (the vote of step 6 changed it). Returns a report.
    """
    source_path, final_ply = Path(source_path), Path(final_ply)
    if not source_path.is_file():
        return {"enabled": False, "reason": "no fine term snapshot"}
    from samcloud.classification.custom import upgrade_classes_config

    config = upgrade_classes_config(json.loads(Path(classes_json).read_text(encoding="utf-8")))
    with np.load(source_path) as archive:
        source = {name: archive[name] for name in archive.files}
    labels = [str(label) for label in source["fine_labels"]]
    owners = label_owner_classes(config, class_set, labels)
    vertex = _read_vertex(final_ply)
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"]))
    term, _conf = inherit_fine_terms(
        source["xyz"], source["fine_term"], source["fine_conf"], xyz,
        np.asarray(vertex["scalar_class"], dtype=np.uint8), owners, radius,
    )
    _write_ply_with_property(final_ply, vertex, term)
    write_fine_labels_json(final_ply.with_name(LABELS_NAME), class_set, labels, owners, config)
    report = {
        "enabled": True, "points": int(len(term)), "labelled_points": int(np.count_nonzero(term)),
        "source_labelled_points": int(np.count_nonzero(source["fine_term"])), "radius_m": float(radius),
        "labels": labels,
    }
    if not keep_source:
        source_path.unlink(missing_ok=True)
    return report
