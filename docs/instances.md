# Object instances

Semantic classes say *what* a point is (furniture); instances say *which
object* it belongs to (chair 3). SAMcloud derives them from SAM3's instance
masks and stores them as a per-point `instance_id`. The code is in
`samcloud/engine/instances.py`; the masks come from `samcloud/engine/sam_labeler.py`
and `samcloud/engine/classify.py`.

## What gets an instance

- Only classes listed in `instance_classes` of `config/classes.json`:
  - indoor: furniture, appliances, sanitary fixtures, door, window, hvac,
    electrical, column;
  - outdoor: vehicle, construction machinery, tree, tree trunk, utility pole,
    traffic sign, manhole, column.
- Walls, floors and ceilings never get an instance. Points without one have
  `instance_id` 0.
- Points on AprilTag squares are excluded (`instance_id` 0), see
  [apriltag_scale.md](apriltag_scale.md).
- The feature is on by default (`classification.instances.enabled`) and is
  switched off with `--no-instances` on the pipeline or `enabled: false`. When it
  is off, no instance data is written.

## How instances are built

1. **Per view.** SAM3's instance head yields masks with a confidence. Masks
   below `query_threshold` or smaller than `min_mask_pixels` are dropped.
2. **Association in 3D.** Each mask is projected onto the dense points visible
   in that view. Two masks of the same class are linked when they share at
   least `min_shared_points` points and at least `merge_overlap` of the smaller
   mask. A point joins the object that wins by confidence-weighted votes, with at
   least `min_views` distinct views, at least half of the weight, and the same
   class as the point's final class. Each object is split into spatially
   connected pieces (`split_radius_m`) and pieces below `min_points` are dropped.
3. **Refinement** (`refine_instance_ids`):
   - labels are made canonical through `instance_label_aliases` ("casement
     window" becomes window);
   - cushion, pillow, blanket and mattress touching a bed or sofa join that bed
     or sofa;
   - objects of the same class and label with enough voxel contact are merged;
   - every object grows over connected unassigned points of its class by at
     most 0.6 m;
   - IDs are renumbered 1 to K by centroid and all metadata is recomputed.
4. **Step 6 survival.** Step 6 changes the point set, so the IDs are set aside
   before it (`instances_source.npz`) and moved onto the final cloud by nearest
   neighbour within 5 cm, guarded by the class (`attach_instances`). Objects that
   end up below `min_points` are removed. The same snapshot pattern is used for
   fine terms (`samcloud/engine/fine_terms.py`).

Settings: `instances` in `config/sam.yml` and `classification.instances` in a
project (see [configuration.md](configuration.md)).

## Products

| Product | Content |
|---|---|
| `products/classified/classified.ply` | vertex property `instance_id` (uint32) next to `fine_term` |
| `products/classified/instances.json` | `schema_version` and a list of instances: `id`, `class_id`, `class_name`, `label` (English, canonical), `label_index` (1, 2, ... per label, ordered by position), `point_count`, `centroid`, `bbox_min`, `bbox_max`, `view_count`, `confidence` |
| LAS export | extra dimension `instance_id` (uint32); `fine_term` (uint8) is written the same way |
| Potree viewer data | the viewer LAS stores the instance ID in `point_source_id` (16 bit; IDs above 65 535 are set to 0) and the fine term in `user_data` |
| `products/classified/instances_source.npz` (and a copy in `processing_cache/`) | snapshot set aside before step 6 and used by step 6 re-runs |

`classified_instances.ply` (one colour per instance) is written only when
`debug_ply` is on.

## Viewer

- Base colours stay as they are (RGB or class colours).
- Clicking a point of an instance opens a card with the German object name
  (`instance_labels_de`, for example *Sofa*, *Tisch*, *Stuhl*; with a number when
  several objects share the name), its class, and its point count. The whole
  object is shown in its original colours with a bounding box.
- If a point has no instance, the card falls back to the point's fine term or
  class.
- Endpoints: `GET /api/instances` (metadata with `label_de`) and
  `GET /api/instances/{id}/points` (the points of one instance, binary, at most
  200 000 points). Picking reads the instance ID from the Potree
  `point_source_id` attribute. The height filter hides points in the viewer
  through an invisible Potree clip box.

## DXF and orthophoto plans

- The point-cloud DXF reads `instance_id` from the classified LAS or PLY and
  `instances.json` next to it. Instance labels become text and block attributes
  (for example the `instance` attribute of the `FURNITURE` block); a change of
  `instances.json` or of the IDs marks cached previews as stale.
- The orthophoto clean plan rasterises the instance IDs
  (`instance_pc.npy`), names furniture and fixtures after their instance label
  (Regal, Tisch, Stuhl, Schrank), and builds sanitary blocks from instances.
  See [ortho_plan.md](ortho_plan.md).

## Limits

- Instance quality depends on SAM3: overlapping objects of the same label can
  merge, and a large object can fragment before refinement merges it.
- Fine terms come from a fixed vocabulary (`fine_terms` in `config/classes.json`);
  unknown objects fall back to their class.
- Viewer data only carries 16-bit IDs; in a project with more than 65 535
  instances, those above 65 535 are set to 0 (shown as no instance) in the
  viewer, and a warning is logged (the PLY and LAS keep 32 bit).
