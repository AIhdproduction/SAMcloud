"""Association and product helpers for SAM3 object instances.

The module deliberately has no model dependency.  It receives the compact
per-view memberships produced by the classifier and can consequently be
tested with small numpy arrays.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

SOURCE_NAME = "instances_source.npz"
JSON_NAME = "instances.json"
TAG_REPORT_NAME = "apriltag_report.json"
TAG_INPLANE_MARGIN_M = 0.02
TAG_PLANE_TOLERANCE_M = 0.03


def tag_point_mask(xyz, report_path) -> np.ndarray | None:
    """Boolean mask of points lying on a detected AprilTag square, or None (no report / no tag points).

    A point belongs to a tag when it is within TAG_PLANE_TOLERANCE_M of the tag plane and inside the
    corner quad expanded by TAG_INPLANE_MARGIN_M. The report corners are in the frame of the cloud.
    """
    if report_path is None:
        return None
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    tags = report.get("tags") if isinstance(report, dict) else None
    if not isinstance(tags, list):
        return None
    xyz = np.asarray(xyz, dtype=np.float64)
    mask = np.zeros(len(xyz), dtype=bool)
    reach = TAG_INPLANE_MARGIN_M + TAG_PLANE_TOLERANCE_M
    for tag in tags:
        corners = tag.get("corners_xyz") if isinstance(tag, dict) else None
        try:
            quad = np.asarray(corners, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if quad.shape != (4, 3) or not np.all(np.isfinite(quad)):
            continue
        centre = quad.mean(axis=0)
        normal = np.cross(quad[1] - quad[0], quad[3] - quad[0])
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        lower, upper = quad.min(axis=0) - reach, quad.max(axis=0) + reach
        near = np.flatnonzero(np.all((xyz >= lower) & (xyz <= upper), axis=1))
        if not len(near):
            continue
        delta = xyz[near] - centre
        inside = np.abs(delta @ normal) <= TAG_PLANE_TOLERANCE_M
        for index in range(4):
            edge = quad[(index + 1) % 4] - quad[index]
            inward = np.cross(normal, edge)
            norm = np.linalg.norm(inward)
            if norm < 1e-12:
                inside[:] = False
                break
            inward /= norm
            if (centre - quad[index]) @ inward < 0:
                inward = -inward
            inside &= (xyz[near] - quad[index]) @ inward >= -TAG_INPLANE_MARGIN_M
        mask[near[inside]] = True
    return mask if mask.any() else None


def _read_vertex_array_func():
    """ply_io.read_vertex_array; works as package module and as script-mode import."""
    if __package__:
        from .ply_io import read_vertex_array
    else:  # classify.py runs as a script, sibling modules import top-level.
        from ply_io import read_vertex_array
    return read_vertex_array


def _atomic_path(path: Path) -> Path:
    """Reserve a temporary sibling file so a crash never leaves a torn product."""
    handle, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(handle)
    return Path(name)


def _group_by_id(ids: np.ndarray):
    """Sort points by positive id once: (order, unique ids, group starts, counts)."""
    positive = np.flatnonzero(ids > 0)
    if not len(positive):
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty
    order = positive[np.argsort(ids[positive], kind="stable")]
    sorted_ids = ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    counts = np.diff(np.r_[starts, len(order)])
    return order, sorted_ids[starts].astype(np.int64), starts, counts


def _group_stats(order, starts, counts, xyz):
    """Per-group mean / min / max of ``xyz`` rows without per-id passes."""
    coordinates = xyz[order]
    mean = np.add.reduceat(coordinates, starts, axis=0) / counts[:, None]
    return mean, np.minimum.reduceat(coordinates, starts, axis=0), np.maximum.reduceat(coordinates, starts, axis=0)


def _class_name(class_id: int, params: dict) -> str:
    names = params.get("class_names") or []
    if 1 <= class_id <= len(names):
        return str(names[class_id - 1]).split(",", 1)[0].strip()
    return str(class_id)


def _metadata(instance_id, xyz, component_sources, vinst_class, vinst_conf, vinst_prompt,
              vinst_view, prompt_texts, generic_prompt_ids, params):
    """Build the stable public ``instances.json`` records from final components."""
    records = []
    order, identifiers, starts, counts = _group_by_id(np.asarray(instance_id))
    if len(identifiers):
        mean, minimum, maximum = _group_stats(order, starts, counts, np.asarray(xyz, dtype=np.float64))
    vinst_class = np.asarray(vinst_class)
    vinst_conf = np.asarray(vinst_conf, dtype=np.float64)
    vinst_prompt = np.asarray(vinst_prompt)
    vinst_view = np.asarray(vinst_view)
    for row, identifier in enumerate(identifiers.tolist()):
        sources = np.asarray(component_sources[int(identifier)], dtype=np.int32)
        classes = vinst_class[sources]
        confidences = vinst_conf[sources]
        class_id = int(Counter(classes.tolist()).most_common(1)[0][0])
        prompts = vinst_prompt[sources]
        specific = [int(prompt) for prompt in prompts if int(prompt) not in generic_prompt_ids]
        selected = specific or [int(prompt) for prompt in prompts]
        labels = [str(prompt_texts[prompt]) for prompt in selected if 0 <= prompt < len(prompt_texts)]
        label = Counter(labels).most_common(1)[0][0] if labels else _class_name(class_id, params)
        records.append({
            "id": int(identifier),
            "class_id": class_id,
            "class_name": _class_name(class_id, params),
            "label": label,
            "label_index": 0,
            "point_count": int(counts[row]),
            "centroid": mean[row].tolist(),
            "bbox_min": minimum[row].tolist(),
            "bbox_max": maximum[row].tolist(),
            "view_count": int(len(np.unique(vinst_view[sources]))),
            "confidence": float(np.average(confidences) if len(confidences) else 0.0),
        })
    by_label = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)
    for group in by_label.values():
        group.sort(key=lambda item: (item["centroid"][0], item["centroid"][1], item["centroid"][2]))
        for index, record in enumerate(group, start=1):
            record["label_index"] = index
    return records


def _spatial_components(xyz: np.ndarray, root: np.ndarray, radius: float) -> np.ndarray:
    """Label points by (root, spatially connected) component, labels 0..k-1.

    Occupied voxels of edge ``radius`` are linked through the 26-neighbourhood
    inside one root; points within ``radius`` are always in the same or an
    adjacent voxel, so nothing connected at that distance is ever split.
    """
    if not len(xyz):
        return np.zeros(0, dtype=np.int64)
    if radius <= 0:
        return np.unique(root, return_inverse=True)[1].astype(np.int64)
    cells = np.floor((xyz - xyz.min(axis=0)) / radius).astype(np.int64) + 1
    dims = cells.max(axis=0) + 2
    n_roots = int(root.max()) + 1
    if float(n_roots) * float(dims[0]) * float(dims[1]) * float(dims[2]) >= 2.0 ** 62:
        return _spatial_components_exact(xyz, root, radius)
    code = ((root * dims[0] + cells[:, 0]) * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    unique_code, inverse = np.unique(code, return_inverse=True)
    edges_a, edges_b = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                delta = (dx * dims[1] + dy) * dims[2] + dz
                if delta <= 0:  # forward half of the neighbourhood only
                    continue
                target = unique_code + delta
                position = np.minimum(np.searchsorted(unique_code, target), len(unique_code) - 1)
                hit = unique_code[position] == target
                edges_a.append(np.flatnonzero(hit))
                edges_b.append(position[hit])
    a, b = np.concatenate(edges_a), np.concatenate(edges_b)
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(len(unique_code),) * 2)
    _, cell_label = connected_components(graph, directed=False)
    return cell_label[inverse].astype(np.int64)


def _spatial_components_exact(xyz: np.ndarray, root: np.ndarray, radius: float) -> np.ndarray:
    """Fallback for pathological extents: per-root exact radius graph."""
    labels = np.zeros(len(xyz), dtype=np.int64)
    offset = 0
    for value in np.unique(root):
        members = np.flatnonzero(root == value)
        pairs = cKDTree(xyz[members]).query_pairs(radius, output_type="ndarray")
        graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
                           shape=(len(members),) * 2)
        n_parts, part = connected_components(graph, directed=False)
        labels[members] = part + offset
        offset += n_parts
    return labels


# --------------------------------------------------------------------------- refinement
_SOFT_LABELS = frozenset({"cushion", "pillow", "blanket", "mattress"})
_HOST_LABELS = frozenset({"bed", "sofa"})
_ALIAS_CACHE: dict | None = None
_NEIGHBOURS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) != (0, 0, 0)]


def _default_label_aliases() -> dict:
    """``instance_label_aliases`` of the global config/classes.json (cached, {} when unreadable)."""
    global _ALIAS_CACHE
    if _ALIAS_CACHE is None:
        try:
            path = Path(__file__).resolve().parents[2] / "config" / "classes.json"
            raw = json.loads(path.read_text(encoding="utf-8")).get("instance_label_aliases", {})
            _ALIAS_CACHE = {str(key).strip().lower(): str(value) for key, value in raw.items()}
        except (OSError, ValueError):
            _ALIAS_CACHE = {}
    return _ALIAS_CACHE


def _canonical_label(label, aliases: dict) -> str:
    text = str(label or "")
    return aliases.get(text.strip().lower(), text)


def _voxel_codes(xyz: np.ndarray, radius: float):
    cells = np.floor((xyz - xyz.min(axis=0)) / radius).astype(np.int64) + 1
    dims = cells.max(axis=0) + 2
    if float(dims[0]) * float(dims[1]) * float(dims[2]) >= 2.0 ** 62:
        return None, dims
    return (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2], dims


def _neighbour_cells(unique_code: np.ndarray, delta: int):
    target = unique_code + delta
    position = np.minimum(np.searchsorted(unique_code, target), len(unique_code) - 1)
    return position, unique_code[position] == target


def _instance_contacts(xyz: np.ndarray, member: np.ndarray, count: int, radius: float):
    """Directed contact table: sorted pair keys ``a * count + b`` and, per pair, the number of
    distinct points of ``a`` that have a point of ``b`` within ``radius``.

    Voxels of edge ``radius`` only preselect boundary candidates (voxels adjacent to a voxel of
    another instance); the exact radius test then runs on that small subset with a cKDTree.
    """
    empty = np.zeros(0, dtype=np.int64)
    code, dims = _voxel_codes(xyz, radius)
    if code is None or count < 2:
        return empty, empty
    unique_code, cell = np.unique(code, return_inverse=True)
    entry_key, entry_n = np.unique(cell.astype(np.int64) * count + member, return_counts=True)
    entry_cell, entry_id = entry_key // count, entry_key % count
    cell_start = np.searchsorted(entry_cell, np.arange(len(unique_code) + 1))
    entry_code = unique_code[entry_cell]
    flagged = np.zeros(len(entry_key), dtype=bool)
    for dx, dy, dz in [(0, 0, 0)] + _NEIGHBOURS:
        delta = (dx * dims[1] + dy) * dims[2] + dz
        position = np.minimum(np.searchsorted(unique_code, entry_code + delta), len(unique_code) - 1)
        hit = np.flatnonzero(unique_code[position] == entry_code + delta)
        if not len(hit):
            continue
        start = cell_start[position[hit]]
        size = cell_start[position[hit] + 1] - start
        total = int(size.sum())
        first = np.repeat(hit, size)
        second = np.repeat(start, size) + (np.arange(total) - np.repeat(np.cumsum(size) - size, size))
        flagged[first[entry_id[first] != entry_id[second]]] = True
    if not flagged.any():
        return empty, empty
    candidate = np.flatnonzero(flagged[np.searchsorted(entry_key, cell.astype(np.int64) * count + member)])
    cand_xyz, cand_member = xyz[candidate], member[candidate]
    pairs = cKDTree(cand_xyz).query_pairs(radius, output_type="ndarray")
    if not len(pairs):
        return empty, empty
    pairs = pairs[cand_member[pairs[:, 0]] != cand_member[pairs[:, 1]]]
    if not len(pairs):
        return empty, empty
    point = np.concatenate([pairs[:, 0], pairs[:, 1]])
    other = np.concatenate([cand_member[pairs[:, 1]], cand_member[pairs[:, 0]]])
    unique_pair = np.unique(point.astype(np.int64) * count + other)       # distinct (point, other instance)
    directed = cand_member[unique_pair // count] * count + unique_pair % count
    keys, weight = np.unique(directed, return_counts=True)
    return keys, weight.astype(np.int64)


def _grow_ids(xyz, point_class, ids, instance_class, radius, max_distance) -> int:
    """Multi-source region growing of ``ids`` over unassigned points of the instance class.

    Works on occupied voxels (edge ``radius``) of one class at a time; each voxel takes the id
    of the nearest seed voxel (Dijkstra with centre distances, capped at ``max_distance``).
    Modifies ``ids`` in place and returns the number of newly labelled points.
    """
    grown = 0
    modulus = len(instance_class)
    for class_id in np.unique(instance_class[instance_class > 0]):
        selected = np.flatnonzero(point_class == class_id)
        if not len(selected):
            continue
        sub_ids = ids[selected]
        candidate = sub_ids == 0
        seed = (sub_ids > 0) & (instance_class[sub_ids] == class_id)
        if not candidate.any() or not seed.any():
            continue
        code, dims = _voxel_codes(xyz[selected], radius)
        if code is None:
            continue
        unique_code, cell = np.unique(code, return_inverse=True)
        key, n = np.unique(cell[seed].astype(np.int64) * modulus + sub_ids[seed], return_counts=True)
        seed_cell, seed_id = key // modulus, key % modulus
        order = np.lexsort((-n, seed_cell))
        first = order[np.r_[True, seed_cell[order][1:] != seed_cell[order][:-1]]]
        seed_node, seed_winner = seed_cell[first], seed_id[first]
        rows, cols, lengths = [], [], []
        for dx, dy, dz in _NEIGHBOURS:
            delta = (dx * dims[1] + dy) * dims[2] + dz
            if delta <= 0:
                continue
            position, hit = _neighbour_cells(unique_code, delta)
            rows.append(np.flatnonzero(hit))
            cols.append(position[hit])
            lengths.append(np.full(int(hit.sum()), radius * float(np.sqrt(dx * dx + dy * dy + dz * dz))))
        size = (len(unique_code),) * 2
        graph = coo_matrix((np.concatenate(lengths), (np.concatenate(rows), np.concatenate(cols))), shape=size).tocsr()
        _, _, source = dijkstra(graph, directed=False, indices=seed_node, min_only=True,
                                return_predecessors=True, limit=max_distance)
        winner = np.zeros(len(unique_code), dtype=np.int64)
        winner[seed_node] = seed_winner
        node_id = np.zeros(len(unique_code), dtype=np.int64)
        reached = source >= 0
        node_id[reached] = winner[source[reached]]
        fill = np.flatnonzero(candidate)
        assigned = node_id[cell[fill]]
        ids[selected[fill]] = assigned
        grown += int(np.count_nonzero(assigned))
    return grown


def _merge_groups(xyz, member, count, labels, klass, sizes, radius, min_contacts, share, soft_contacts):
    """Connected components of the instance contact graph: soft furnishings join their bed /
    sofa, same class + same label instances with enough contact merge. Returns group per instance."""
    group = np.arange(count)
    keys, weight = _instance_contacts(xyz, member, count, radius)
    if not len(keys):
        return group
    first, second = keys // count, keys % count
    lookup = np.minimum(np.searchsorted(keys, second * count + first), len(keys) - 1)
    reverse = np.where(keys[lookup] == second * count + first, weight[lookup], 0)
    contact = np.minimum(weight, reverse)
    soft = np.array([label in _SOFT_LABELS for label in labels])
    host = np.array([label in _HOST_LABELS for label in labels])
    edge_a, edge_b = [np.zeros(0, dtype=np.int64)], [np.zeros(0, dtype=np.int64)]
    pick = soft[first] & host[second] & (weight >= soft_contacts)
    if pick.any():
        f, s, w = first[pick], second[pick], weight[pick]
        best = np.lexsort((-w, f))
        best = best[np.r_[True, f[best][1:] != f[best][:-1]]]
        edge_a.append(f[best])
        edge_b.append(s[best])
    same = np.array([(klass[a] == klass[b]) and labels[a] == labels[b]
                     for a, b in zip(first.tolist(), second.tolist())], dtype=bool)
    need = np.maximum(min_contacts, share * np.minimum(sizes[first], sizes[second]))
    pick = same & (first < second) & (contact >= need)
    edge_a.append(first[pick])
    edge_b.append(second[pick])
    a, b = np.concatenate(edge_a), np.concatenate(edge_b)
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(count, count))
    return connected_components(graph, directed=False)[1]


def refine_instance_ids(xyz, point_class, instance_id, metadata, params=None, exclude_mask=None):
    """Canonicalise, merge, grow and renumber instance IDs; returns ``(ids, metadata)``.

    1. labels go through ``instance_label_aliases`` ("casement window" -> "window");
    2. cushion / pillow / blanket / mattress touching a bed or sofa join that bed / sofa;
    3. same class + same label instances with enough voxel contact are merged;
    4. every instance grows over connected unassigned points of its class, at most
       ``instance_grow_max_m`` from its original points;
    5. IDs are renumbered 1..K by centroid and all metadata is recomputed.

    ``exclude_mask`` (AprilTag squares) points never belong to an instance: they are cleared up front
    and are not claimed by growth.
    """
    params = dict(params or {})
    ids_in = np.asarray(instance_id, dtype=np.uint32)
    if exclude_mask is not None:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)
        ids_in = np.where(exclude_mask, 0, ids_in).astype(np.uint32)
    xyz = np.asarray(xyz, dtype=np.float64)
    point_class = np.asarray(point_class).astype(np.int64)
    order, identifiers, starts, counts = _group_by_id(ids_in)
    if not len(identifiers):
        return np.zeros(len(ids_in), dtype=np.uint32), []
    aliases = dict(_default_label_aliases())
    aliases.update({str(k).strip().lower(): str(v) for k, v in (params.get("label_aliases") or {}).items()})
    radius = float(params.get("split_radius_m", 0.05))
    min_contacts = int(params.get("instance_merge_min_contacts", 30))
    share = float(params.get("instance_merge_contact_share", 0.02))
    soft_contacts = int(params.get("instance_soft_min_contacts", 10))
    grow_max = float(params.get("instance_grow_max_m", 0.6))
    by_old = {int(item["id"]): item for item in (metadata or []) if "id" in item}
    count = len(identifiers)
    member = np.full(len(ids_in), -1, dtype=np.int64)
    member[order] = np.repeat(np.arange(count), counts)
    positive = np.flatnonzero(member >= 0)
    votes = np.bincount(member[positive] * 256 + np.clip(point_class[positive], 0, 255), minlength=count * 256)
    majority = votes.reshape(count, 256).argmax(axis=1)
    records = [by_old.get(int(identifier), {}) for identifier in identifiers]
    klass = np.array([int(rec["class_id"]) if "class_id" in rec else int(majority[row])
                      for row, rec in enumerate(records)], dtype=np.int64)
    labels = [_canonical_label(rec.get("label") or _class_name(int(klass[row]), params), aliases)
              for row, rec in enumerate(records)]

    def merge(member_all, n, names, klasses, sizes):
        if radius <= 0 or n < 2:
            return np.arange(n)
        inside = np.flatnonzero(member_all >= 0)
        return _merge_groups(xyz[inside], member_all[inside], n, names, klasses, sizes,
                             radius, min_contacts, share, soft_contacts)

    def describe(group):
        n_groups = int(group.max()) + 1
        base = np.zeros(n_groups, dtype=np.int64)        # largest member = metadata donor
        for row in np.argsort(counts, kind="stable"):
            base[group[row]] = row
        group_label = [labels[int(base[g])] for g in range(n_groups)]
        host_size = np.zeros(n_groups, dtype=np.int64)
        for row in range(count):                          # the largest bed / sofa member names the group
            g = int(group[row])
            if labels[row] in _HOST_LABELS and counts[row] > host_size[g]:
                host_size[g] = counts[row]
                group_label[g] = labels[row]
        return n_groups, base, group_label, klass[base]

    group = merge(member, count, labels, klass, counts)
    n_groups, base, group_label, group_class = describe(group)
    ids = np.zeros(len(ids_in), dtype=np.int64)
    ids[positive] = group[member[positive]] + 1
    if grow_max > 0 and radius > 0:
        _grow_ids(xyz, point_class, ids, np.r_[0, group_class], radius, grow_max)
        if exclude_mask is not None:
            ids[exclude_mask] = 0
        # growth enlarges the contact surfaces: fragments that only now touch broadly merge
        grown_counts = np.bincount(ids, minlength=n_groups + 1)[1:]
        second = merge(ids - 1, n_groups, group_label, group_class, grown_counts)
        if int(second.max()) + 1 < n_groups:
            group = second[group]
            ids = np.where(ids > 0, second[np.maximum(ids - 1, 0)] + 1, 0)
            n_groups, base, group_label, group_class = describe(group)

    order, identifiers, starts, counts_out = _group_by_id(ids)
    mean, minimum, maximum = _group_stats(order, starts, counts_out, xyz)
    rank = np.lexsort((mean[:, 2], mean[:, 1], mean[:, 0]))
    renumber = np.zeros(int(identifiers.max()) + 1, dtype=np.uint32)
    renumber[identifiers[rank]] = np.arange(1, len(rank) + 1, dtype=np.uint32)
    final = np.zeros(len(ids_in), dtype=np.uint32)
    active = ids > 0
    final[active] = renumber[ids[active]]
    members_of = defaultdict(list)
    for row in range(count):
        members_of[int(group[row])].append(row)
    out = []
    for new, row in enumerate(rank.tolist(), start=1):
        g = int(identifiers[row]) - 1
        rows = members_of[g]
        record = dict(records[int(base[g])])
        views = [int(records[r].get("view_count", 0)) for r in rows]
        weights = np.array([counts[r] for r in rows], dtype=np.float64)
        confs = np.array([float(records[r].get("confidence", 0.0)) for r in rows])
        class_id = int(group_class[g])
        record.update({
            "id": new, "class_id": class_id,
            "class_name": record["class_name"] if record.get("class_name") and record.get("class_id") == class_id
            else _class_name(class_id, params),
            "label": group_label[g], "label_index": 0, "point_count": int(counts_out[row]),
            "centroid": mean[row].tolist(), "bbox_min": minimum[row].tolist(), "bbox_max": maximum[row].tolist(),
        })
        if "view_count" in record or any(views):
            record["view_count"] = int(sum(views))
        if "confidence" in record:
            record["confidence"] = float(np.average(confs, weights=weights)) if weights.sum() else 0.0
        out.append(record)
    by_label = defaultdict(list)
    for record in out:
        by_label[record["label"]].append(record)
    for items in by_label.values():
        for number, record in enumerate(items, start=1):  # already centroid ordered
            record["label_index"] = number
    return final, out


def associate_instances(point_idx, vinst_idx, vinst_class, vinst_conf, vinst_prompt, n_points,
                        resolved_class, xyz, params, prompt_texts, generic_prompt_ids=(),
                        vinst_view=None, exclude_mask=None):
    """Associate per-image SAM masks into class-consistent 3D object IDs.

    ``point_idx`` and ``vinst_idx`` are one compact membership row per visible
    point.  All large temporary indices are int32; scipy owns the sparse matrix
    itself, which avoids a dense view-instance by point allocation.
    """
    params = dict(params or {})
    point_idx = np.asarray(point_idx, dtype=np.int32)
    vinst_idx = np.asarray(vinst_idx, dtype=np.int32)
    xyz = np.asarray(xyz, dtype=np.float64)
    resolved_class = np.asarray(resolved_class)
    vinst_class = np.asarray(vinst_class, dtype=np.int32)
    vinst_conf = np.asarray(vinst_conf, dtype=np.float32)
    vinst_prompt = np.asarray(vinst_prompt, dtype=np.int32)
    count = len(vinst_class)
    if not (len(vinst_conf) == len(vinst_prompt) == count):
        raise ValueError("view-instance metadata arrays must have equal lengths")
    if vinst_view is None:
        vinst_view = np.arange(count, dtype=np.int32)
    else:
        vinst_view = np.asarray(vinst_view, dtype=np.int32)
    empty = np.zeros(n_points, dtype=np.uint32)
    if not len(point_idx) or not count:
        return empty, []
    if point_idx.min(initial=0) < 0 or point_idx.max(initial=-1) >= n_points:
        raise ValueError("point_idx contains an out-of-range point")
    if vinst_idx.min(initial=0) < 0 or vinst_idx.max(initial=-1) >= count:
        raise ValueError("vinst_idx contains an out-of-range instance")

    # int32 data: the product accumulates shared-point counts and must not overflow.
    membership = coo_matrix((np.ones(len(point_idx), dtype=np.int32), (vinst_idx, point_idx)),
                            shape=(count, n_points)).tocsr()
    membership.sum_duplicates()
    membership.data[:] = 1  # duplicate (vinst, point) pairs were summed: binarise so shared <= size
    sizes = np.diff(membership.indptr).astype(np.int64, copy=False)
    overlap = (membership @ membership.T).tocoo()
    threshold = float(params.get("merge_overlap", 0.5))
    min_shared = int(params.get("min_shared_points", 10))
    first, second, shared = overlap.row, overlap.col, overlap.data.astype(np.int64, copy=False)
    keep = (first < second) & (shared >= min_shared)
    first, second, shared = first[keep], second[keep], shared[keep]
    keep = vinst_class[first] == vinst_class[second]
    first, second, shared = first[keep], second[keep], shared[keep]
    keep = shared / np.minimum(sizes[first], sizes[second]) >= threshold
    first, second = first[keep], second[keep]
    del overlap, membership
    graph = coo_matrix((np.ones(len(first), dtype=np.int8), (first, second)), shape=(count, count))
    n_roots, roots = connected_components(graph, directed=False)
    roots = roots.astype(np.int64, copy=False)

    # Conf-weighted point votes, fully vectorised: a point must have multiple
    # distinct source views and a clear (>= 50 %) winner of the right class.
    min_views = int(params.get("min_views", 2))
    pair_root = roots[vinst_idx]
    pair_key = point_idx.astype(np.int64) * n_roots + pair_root
    unique_key, inverse = np.unique(pair_key, return_inverse=True)
    weight = np.bincount(inverse, weights=vinst_conf[vinst_idx].astype(np.float64), minlength=len(unique_key))
    n_views = int(vinst_view.max(initial=0)) + 1
    view_key = np.unique(pair_key * n_views + vinst_view[vinst_idx].astype(np.int64))
    distinct_views = np.bincount(np.searchsorted(unique_key, view_key // n_views), minlength=len(unique_key))
    key_point = unique_key // n_roots
    key_root = unique_key % n_roots
    total = np.bincount(key_point, weights=weight, minlength=n_points)
    # Winner per point: highest weight (lowest root on ties) -> first row of its point group.
    order = np.lexsort((key_root, -weight, key_point))
    is_first = np.r_[True, key_point[order][1:] != key_point[order][:-1]]
    win = order[is_first]
    win_point, win_root = key_point[win], key_root[win]
    root_class = np.zeros(n_roots, dtype=np.int64)
    root_class[roots] = vinst_class
    good = (distinct_views[win] >= min_views) & (weight[win] >= 0.5 * total[win_point])
    good &= np.asarray(resolved_class)[win_point].astype(np.int64) == root_class[win_root]
    assigned_root = np.full(n_points, -1, dtype=np.int64)
    assigned_root[win_point[good]] = win_root[good]

    # Split spatially disconnected components before numbering.
    radius = float(params.get("split_radius_m", 0.05))
    min_points = int(params.get("min_points", 100))
    assigned = np.flatnonzero(assigned_root >= 0)
    assigned_xyz = xyz[assigned]
    component = _spatial_components(assigned_xyz, assigned_root[assigned], radius)
    final_count = int(component.max()) + 1 if len(component) else 0
    sizes_final = np.bincount(component, minlength=final_count)
    kept = np.flatnonzero(sizes_final >= min_points)
    instance_id = np.zeros(n_points, dtype=np.uint32)
    component_sources = {}
    if len(kept):
        mean = np.stack([np.bincount(component, weights=assigned_xyz[:, axis], minlength=final_count)[kept]
                         / sizes_final[kept] for axis in range(3)], axis=1)
        # Stable ordering gives consecutive IDs independent of hashing/matrix order.
        rank_order = np.lexsort((mean[:, 2], mean[:, 1], mean[:, 0]))
        new_id = np.zeros(final_count, dtype=np.uint32)
        new_id[kept[rank_order]] = np.arange(1, len(kept) + 1, dtype=np.uint32)
        instance_id[assigned] = new_id[component]
        component_root = np.zeros(final_count, dtype=np.int64)
        component_root[component] = assigned_root[assigned]
        by_root = np.argsort(roots, kind="stable")
        bounds = np.searchsorted(roots[by_root], np.arange(n_roots + 1))
        for old in kept[rank_order]:
            root = int(component_root[old])
            component_sources[int(new_id[old])] = by_root[bounds[root]:bounds[root + 1]].astype(np.int32)
    metadata = _metadata(instance_id, xyz, component_sources, vinst_class, vinst_conf, vinst_prompt,
                         vinst_view, list(prompt_texts), set(generic_prompt_ids), params)
    return refine_instance_ids(xyz, resolved_class, instance_id, metadata, params, exclude_mask)


def _write_ply_with_instance_id(path: Path, vertex, instance_id, colors=None) -> None:
    from plyfile import PlyData, PlyElement
    names = [name for name in vertex.dtype.names if name != "instance_id"]
    dtype = [(name, vertex.dtype[name]) for name in names] + [("instance_id", "<u4")]
    data = np.empty(len(vertex), dtype=dtype)
    for name in names:
        data[name] = vertex[name]
    if colors is not None:
        data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    data["instance_id"] = np.asarray(instance_id, dtype=np.uint32)
    path = Path(path)
    temporary = _atomic_path(path)
    try:
        PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(temporary))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_instances_json(path: Path, metadata) -> None:
    path = Path(path)
    temporary = _atomic_path(path)
    try:
        temporary.write_text(json.dumps({"schema_version": 1, "instances": metadata}, indent=2,
                                        ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_instance_product(classified_ply: Path, instance_id, metadata, debug_ply: bool = False) -> None:
    """Add IDs to an already-written classified PLY and emit its sidecar."""
    read_vertex_array = _read_vertex_array_func()

    classified_ply = Path(classified_ply)
    _write_ply_with_instance_id(classified_ply, read_vertex_array(classified_ply), instance_id)
    write_instances_json(classified_ply.with_name(JSON_NAME), metadata)
    if debug_ply:
        write_debug_ply(classified_ply, classified_ply.with_name("classified_instances.ply"))


def snapshot_instances(classified_ply: Path, source_path: Path | None = None, metadata=None) -> bool:
    """Set aside IDs before step 6 replaces the point set; absent IDs are a no-op."""
    classified_ply = Path(classified_ply)
    source_path = Path(source_path) if source_path is not None else classified_ply.with_name(SOURCE_NAME)
    if not classified_ply.is_file():
        return False
    read_vertex_array = _read_vertex_array_func()

    try:
        vertex = read_vertex_array(classified_ply)
    except ValueError:  # not a readable PLY: nothing to set aside
        return False
    if "instance_id" not in (vertex.dtype.names or ()):
        return False
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float32)
    instance_id = np.asarray(vertex["instance_id"], dtype=np.uint32)
    classes = np.asarray(vertex["scalar_class"], dtype=np.uint8)
    if metadata is None:
        json_path = classified_ply.with_name(JSON_NAME)
        metadata = json.loads(json_path.read_text(encoding="utf-8")).get("instances", []) if json_path.is_file() else []
    temporary = _atomic_path(source_path)
    try:
        with open(temporary, "wb") as handle:
            np.savez(handle, xyz=xyz, instance_id=instance_id, scalar_class=classes,
                     metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
        os.replace(temporary, source_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return True


def attach_instances(source_path: Path, final_ply: Path, params=None, *, keep_source: bool = True,
                     tag_report: Path | None = None) -> dict:
    """Transfer saved instance IDs to a re-extracted cloud with a class guard."""
    source_path, final_ply = Path(source_path), Path(final_ply)
    if not source_path.is_file() or not final_ply.is_file():
        return {"enabled": False, "reason": "no instance snapshot"}
    params = dict(params or {})
    with np.load(source_path) as archive:
        source = {name: archive[name] for name in archive.files}
    read_vertex_array = _read_vertex_array_func()

    vertex = read_vertex_array(final_ply)
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    instance_id = np.zeros(len(vertex), dtype=np.uint32)
    original = np.asarray(source["instance_id"], dtype=np.uint32)
    if tag_report is None:  # <project>/products/classified/classified.ply -> <project>/reports/
        for candidate in (final_ply.parent.parent.parent, final_ply.parent.parent):
            if (candidate / "reports" / TAG_REPORT_NAME).is_file():
                tag_report = candidate / "reports" / TAG_REPORT_NAME
                break
    tag_mask = tag_point_mask(xyz, tag_report)
    if tag_mask is not None:  # snapshots made before the tag report existed may still carry ids on tags
        source_mask = tag_point_mask(source["xyz"], tag_report)
        if source_mask is not None:
            original = np.where(source_mask, 0, original).astype(np.uint32)
    valid = original > 0
    if np.any(valid) and len(vertex):
        distance, nearest = cKDTree(np.asarray(source["xyz"])[valid]).query(
            xyz, distance_upper_bound=float(params.get("attach_radius_m", 0.05))
        )
        kept_ids = original[valid]
        kept_classes = np.asarray(source["scalar_class"])[valid]
        found = np.isfinite(distance) & (nearest < len(kept_ids))
        instance_id[found] = kept_ids[nearest[found]]
        same_class = np.zeros(len(vertex), dtype=bool)
        same_class[found] = np.asarray(vertex["scalar_class"])[found] == kept_classes[nearest[found]]
        instance_id[~same_class] = 0
    min_points = int(params.get("min_points", 100))
    order, identifiers, starts, counts = _group_by_id(instance_id)
    small = identifiers[counts < min_points]
    if len(small):
        instance_id[np.isin(instance_id, small)] = 0
        order, identifiers, starts, counts = _group_by_id(instance_id)
    old_metadata = json.loads(str(source.get("metadata_json", "[]").item()))
    remapped, metadata = refine_instance_ids(xyz, np.asarray(vertex["scalar_class"]), instance_id,
                                             old_metadata, params, tag_mask)
    _write_ply_with_instance_id(final_ply, vertex, remapped)
    write_instances_json(final_ply.with_name(JSON_NAME), metadata)
    return {"enabled": True, "count": len(metadata), "points": int(np.count_nonzero(remapped)), "instances": metadata}


def write_debug_ply(classified_ply: Path, destination: Path) -> None:
    """Write a stable per-instance colour diagnostic PLY without changing the product."""
    read_vertex_array = _read_vertex_array_func()

    vertex = read_vertex_array(classified_ply)
    ids = np.asarray(vertex["instance_id"], dtype=np.uint32)
    colors = np.full((len(ids), 3), 128, dtype=np.uint8)
    active = ids > 0
    values = ids[active].astype(np.uint64)
    colors[active, 0] = ((values * 97 + 31) % 256).astype(np.uint8)
    colors[active, 1] = ((values * 57 + 101) % 256).astype(np.uint8)
    colors[active, 2] = ((values * 17 + 211) % 256).astype(np.uint8)
    _write_ply_with_instance_id(Path(destination), vertex, ids, colors)
