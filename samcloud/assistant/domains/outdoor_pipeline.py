"""The outdoor assistant inside the environment chain: history backend, the ``_step_scene`` hook and the manual marking.

* :func:`outdoor_backend` is the :class:`~samcloud.assistant.domains.base.HistoryBackend` of a drone project: model file
  ``products/assistant/site/scene.json`` (``Scene.save``), base snapshot ``scene.base.json``.
* :func:`scene_plan` is called by ``run_environment.build_scene_outputs`` right after ``assemble_scene`` and before any
  DXF builder: with ``edits.json`` it writes the pipeline Scene as ``scene.base.json``, re-applies the active and stale
  edits by signature (:meth:`History._rebuild`), commits ``scene.json`` / ``edits.json`` / ``state.json`` and returns a
  :class:`ScenePlan` whose element groups replace the builder inputs, so DXF and IFC are generated from the edited
  elements (no DXF patch).  Stale edits are reported in ``products/assistant/site/pipeline_report.json`` like the indoor
  hooks.  Without ``edits.json`` nothing is imported or written and the chain output stays byte-identical.
* Elements added without point-cloud evidence (``evidence["evidence"] == "manual"``) are drawn on ``<layer>-MANUAL``
  (style from ``assistant_layers`` in ``config/dxf_layers.yaml``); :func:`adjust_manual_qa` lets the CAD QA count them
  with their category so the marking itself is no QA failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from samcloud.assistant.domains.outdoor_edits import (
    OutdoorProposal, diff_scenes, is_manual, kind_of, resolve_signature,
)
from samcloud.assistant.domains.outdoor_scene import (
    BASE_FILENAME, SITE_LEVEL, EnvSupport, load_scene_file, model_path, normalize_scene, save_scene, scene_file_hash,
    scene_semantic_hash,
)

LOGGER = logging.getLogger(__name__)

MANUAL_SUFFIX = "-MANUAL"
STEP_NAME = "environment_scene"
_GROUP_OF_KIND = {"tree": 0, "shrub": 0, "vegetation_area": 0, "pavement": 1, "terrain": 2, "building": 3, "wall": 3, "window": 3,
                  "door": 3, "opening": 3}
_FALLBACK_MANUAL_STYLE = {"color": 6, "linetype": "DASHED", "lineweight": 18}


# --------------------------------------------------------------------------------------------------- history backend

def outdoor_backend(project: Path | str) -> Any:
    """History model access of a drone project (``scene.json`` / ``scene.base.json`` in ``products/assistant/site``)."""
    from samcloud.assistant.domains.base import HistoryBackend
    from samcloud.assistant.edits import EditOp

    holder: dict[str, EnvSupport] = {}

    def support() -> EnvSupport:
        if "support" not in holder:
            holder["support"] = EnvSupport.for_project(project)
        return holder["support"]

    def new_proposal(scene: Any) -> OutdoorProposal:
        return OutdoorProposal(scene, support=support(), project=project)

    return HistoryBackend(kind="outdoor", base_filename=BASE_FILENAME, model_path=model_path, load=load_scene_file, save=save_scene,
                          semantic_hash=scene_semantic_hash, file_hash=scene_file_hash, new_proposal=new_proposal,
                          op_from_dict=EditOp.from_dict, resolve_signature=resolve_signature, diff=diff_scenes,
                          extra={"project": str(project)})


# -------------------------------------------------------------------------------------------------- pipeline hook

def apply_pipeline_edits(project: Path | str, scene: Any, logger: logging.Logger | None = None) -> Any | None:
    """The pipeline ``scene`` with the active edits re-applied (None without ``edits.json``, without an applied edit or on
    an error: the chain then goes on with its own Scene).  Never raises."""
    from samcloud.dxf.assistant_hooks import _info, _record, has_edits, stale_edits

    if not has_edits(project, SITE_LEVEL):
        return None
    log = logger or LOGGER
    info = _info(SITE_LEVEL, STEP_NAME)
    edited = None
    try:
        from samcloud.assistant.history import History, _locked

        history = History(project, SITE_LEVEL, outdoor_backend(project))
        with _locked(history.directory):
            history._read()
            base = normalize_scene(scene)
            save_scene(base, history.base_path)  # scene.base.json = the pre-edit Scene of this run
            rebuilt, applied, _stale = history._rebuild(base)
            history._commit(rebuilt)
        info.update(applied=[str(item) for item in applied], changed=bool(applied))
        edited = rebuilt if applied else None
    except Exception as error:  # noqa: BLE001 - the assistant never breaks the pipeline
        info["error"] = f"{type(error).__name__}: {error}"
        log.warning(f"Assistant edits of the outdoor site could not be applied ({STEP_NAME}): {info['error']}")
        edited = None
    info["stale"] = stale_edits(project, SITE_LEVEL)
    if info["stale"]:
        log.warning(f"Assistant edits of the outdoor site no longer apply: {[item['id'] for item in info['stale']]}")
    _record(project, SITE_LEVEL, info)
    return edited


@dataclass
class ScenePlan:
    """The edited Scene and the builder inputs derived from it (see :func:`scene_plan`)."""

    scene: Any
    groups: list[list[Any]]
    dxf_groups: list[list[Any]]
    removed_ids: set[str] = field(default_factory=set)
    manual: list[Any] = field(default_factory=list)

    def fitted_curbs(self, fitted: Any) -> Any:
        """Fitted curbs without those whose curb element was deleted (they would still be drawn from the fit)."""
        if not fitted:
            return fitted
        removed = {element_id for element_id in self.removed_ids if element_id.startswith("curb-")}
        return [curb for curb in fitted if f"curb-{int(curb.curb_id)}" not in removed]

    def add_manual_entities(self, entities: dict[str, Any], frame: Any, config: dict[str, Any] | None) -> dict[str, Any]:
        """``entities`` plus the manual trees / manholes on ``<layer>-MANUAL``."""
        if not self.manual:
            return entities
        from samcloud.outdoor_model import civil, vegetation

        trees = [element for element in self.manual if kind_of(element) == "tree"]
        manholes = [element for element in self.manual if kind_of(element) == "manhole"]
        built: dict[str, list[dict]] = {}
        if trees:
            for layer, items in vegetation.to_dxf_entities(trees, frame, config=config).items():
                built.setdefault(layer, []).extend(items)
        if manholes:
            for layer, items in civil.to_dxf_entities(manholes, frame, config=config).items():
                built.setdefault(layer, []).extend(items)
        renamed: dict[str, list[dict]] = {}
        for layer, items in built.items():
            target = f"{layer}{MANUAL_SUFFIX}"
            renamed[target] = [{**item, "layer": target} if "layer" in item else dict(item) for item in items]
        return {**entities, "assistant_manual": renamed}

    def style(self, result: Any) -> Any:
        """Colour / linetype of the ``-MANUAL`` layers in an assembled DXF result (in place)."""
        styles = manual_layer_styles()
        for layer in list(getattr(result, "layers", {}) or {}):
            if not layer.endswith(MANUAL_SUFFIX):
                continue
            style = styles.get(layer) or styles.get("*") or dict(_FALLBACK_MANUAL_STYLE)
            result.layer_colors[layer] = int(style.get("color", 6))
            result.layer_linetypes[layer] = str(style.get("linetype", "DASHED"))
            result.layer_styles[layer] = {key: style[key] for key in ("lineweight", "true_color") if key in style}
        return result


def manual_layer_styles() -> dict[str, dict[str, Any]]:
    """``assistant_layers`` of ``config/dxf_layers.yaml`` ({name: style}); ``"*"`` = the fallback."""
    styles: dict[str, dict[str, Any]] = {"*": dict(_FALLBACK_MANUAL_STYLE)}
    try:
        from samcloud.assistant.dxf_patch import assistant_layer_styles

        styles.update({name: dict(style) for name, style in assistant_layer_styles().items()})
    except Exception as error:  # noqa: BLE001 - the fallback style is enough
        LOGGER.debug("assistant: manual layer styles fallback (%s)", error)
    return styles


def regroup(groups: list[list[Any]], scene: Any) -> list[list[Any]]:
    """The elements of ``scene`` sorted into the builder groups of ``groups`` (by id; new elements by kind)."""
    owner: dict[str, int] = {}
    for index, group in enumerate(groups):
        for element in group or ():
            owner[str(element.id)] = index
    out: list[list[Any]] = [[] for _ in groups]
    last = len(groups) - 1
    for element in scene.elements.values():
        index = owner.get(element.id)
        if index is None:
            index = min(_GROUP_OF_KIND.get(kind_of(element), last), last)
        out[index].append(element)
    return out


def scene_plan(project: Path | str, scene: Any, groups: list[list[Any]], edited_scene: Any = None,
               logger: logging.Logger | None = None) -> ScenePlan | None:
    """The plan of the scene step: ``edited_scene`` (assistant QA) or the pipeline ``scene`` with the stored edits
    (None when nothing is edited)."""
    if edited_scene is None:
        edited_scene = apply_pipeline_edits(project, scene, logger)
        if edited_scene is None:
            return None
    new_groups = regroup(groups, edited_scene)
    original = {str(element.id) for group in groups for element in group or ()}
    manual = [element for element in edited_scene.elements.values() if is_manual(element)]
    dxf_groups = [[element for element in group if not is_manual(element)] for group in new_groups]
    return ScenePlan(edited_scene, new_groups, dxf_groups, original - set(edited_scene.elements), manual)


# ------------------------------------------------------------------------------------------------------------ CAD QA

def adjust_manual_qa(results: list[dict[str, Any]], dxf_path: Path | str, ifc_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Rules 1 (block inserts), 15 (counts) and 21 (IFC attributes vs DXF blocks) as if the ``-MANUAL`` entities sat on
    their category layer (``ifc_path`` default: ``outdoor.ifc`` next to the DXF).

    The QA itself stays untouched; without ``-MANUAL`` layers the results are returned unchanged.  Never raises.
    """
    try:
        import ezdxf

        from samcloud.drone import qa_cad

        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
        base_layers = set(qa_cad.BLOCK_RULES) | set(qa_cad.DXF_CATEGORY_LAYERS.values())
        moved = 0
        for entity in msp:
            layer = str(entity.dxf.layer)
            if layer.endswith(MANUAL_SUFFIX) and layer[: -len(MANUAL_SUFFIX)] in base_layers:
                entity.dxf.layer = layer[: -len(MANUAL_SUFFIX)]  # in memory only, the file is not saved
                moved += 1
        if not moved:
            return results
        ifc = None
        ifc_file = Path(ifc_path) if ifc_path is not None else Path(dxf_path).with_name("outdoor.ifc")
        adjusted = []
        for result in results:
            rule = str(result.get("rule") or "")
            if rule == "1_block_inserts":
                result = qa_cad._rule_block_inserts(msp)
            elif rule.startswith("15_count_") and isinstance(result.get("value"), dict):
                category = rule[len("15_count_"):]
                value = dict(result["value"])
                value["dxf"] = qa_cad._count_dxf(msp, category)
                expected = value.get("scene")
                wrong = [f"{src}={n} (scene={expected})" for src, n in value.items() if src != "scene" and n is not None and n != expected]
                result = qa_cad._result(rule, not wrong, value, result.get("threshold", "scene == dxf == ifc"), wrong)
            elif rule == "21_ifc_attributes_vs_dxf" and ifc_file.is_file():
                if ifc is None:
                    import ifcopenshell

                    ifc = ifcopenshell.open(str(ifc_file))
                result = qa_cad._rule_ifc_attributes_vs_dxf(msp, ifc) or result
            adjusted.append(result)
        return adjusted
    except Exception as error:  # noqa: BLE001 - the original results stay valid
        LOGGER.warning("assistant: manual QA adjustment skipped (%s)", type(error).__name__)
        return results


def adjust_manual_qa_safely(project: Path | str, results: list[dict[str, Any]], dxf_path: Path | str,
                            ifc_path: Path | str | None = None) -> list[dict[str, Any]]:
    """:func:`adjust_manual_qa` only for a project with ``edits.json`` (pipeline hook of ``_step_qa``)."""
    from samcloud.dxf.assistant_hooks import has_edits

    if not has_edits(project, SITE_LEVEL):
        return results
    return adjust_manual_qa(results, dxf_path, ifc_path)


__all__ = ["MANUAL_SUFFIX", "ScenePlan", "adjust_manual_qa", "adjust_manual_qa_safely", "apply_pipeline_edits", "manual_layer_styles",
           "outdoor_backend", "regroup", "scene_plan"]
