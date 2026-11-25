"""The domain interface of the AI assistant: what a project type (indoor plan, outdoor site, ...) plugs in.

The agent, the tools, the service and the panel never talk to a project model directly: they ask the domain of the
project level (:func:`samcloud.assistant.domains.registry.domain_for`) for its model, its tools, its QA, its selection
helpers, its prompt vocabulary, its history backend and the export jobs to run after an accept / undo.

Tool tiers: ``"elements"`` (default: single-element tools, proposal + accept) and ``"full"`` (full control: project-wide
tools, applied directly after a passed QA).  A domain lists the tiers it really supports in :attr:`supported_tiers`;
:func:`samcloud.assistant.domains.registry.resolve_tier` falls back to ``"elements"`` for any other tier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

TIER_ELEMENTS = "elements"
TIER_FULL = "full"
TIERS: tuple[str, ...] = (TIER_ELEMENTS, TIER_FULL)


def normalize_tier(tier: Any) -> str:
    """``"full"`` or ``"elements"`` (anything else, including None, is ``"elements"``)."""
    return TIER_FULL if str(tier or "").strip().lower() == TIER_FULL else TIER_ELEMENTS


@dataclass(frozen=True)
class ToolDef:
    """One tool the model may call.

    ``handler(ctx, call, arguments) -> ToolResult`` runs it (``ctx`` is the turn's
    :class:`~samcloud.assistant.tools.ToolContext`, ``arguments`` a copy of the call's JSON object).  ``kind`` is
    ``"read"`` (measures / lists, never changes the proposal) or ``"edit"`` (changes the proposal).  ``image_note`` is
    appended to the description when images may be sent to the model.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    kind: str = "read"
    image_note: str = ""

    @property
    def is_edit(self) -> bool:
        return self.kind == "edit"


@dataclass
class HistoryBackend:
    """Model access of :class:`samcloud.assistant.history.History` for one domain (load / save / hash / replay).

    ``model_path(project, level_id)`` is the model file the pipeline and the exports read; ``base_filename`` the name of
    the automatic-model snapshot in the assistant directory; ``load(path)`` / ``save(model, path)`` read and write a model
    file (``save`` writes exactly the bytes ``file_hash`` hashes); ``semantic_hash(model)`` compares models;
    ``new_proposal(model)`` starts a proposal (``.apply(op, cloud=, replay=)``, ``.changes``, ``.signatures``, ``.notes``,
    ``.ops``, ``.plan``); ``op_from_dict`` rebuilds a stored op; ``resolve_signature(model, sig, preferred_id, strict)``
    finds an element again after a recompute; ``diff(base, edited)`` lists the element changes.
    """

    kind: str
    base_filename: str
    model_path: Callable[[Path, str], Path]
    load: Callable[[Path], Any]
    save: Callable[[Any, Path], None]
    semantic_hash: Callable[[Any], str]
    file_hash: Callable[[Any], str]
    new_proposal: Callable[[Any], Any]
    op_from_dict: Callable[[dict[str, Any]], Any]
    resolve_signature: Callable[..., str | None]
    diff: Callable[[Any, Any], list[Any]]
    extra: dict[str, Any] = field(default_factory=dict)


class AssistantDomain(ABC):
    """A project type the assistant can work on (one instance per project level).

    Coordinates: ``x`` / ``y`` in the domain's 2D frame (indoor: plan metres), heights in metres.  Selections are plain
    dicts: ``{"type": "element", "kind", "id", "label", ...}``, ``{"type": "point", "x", "y"}`` or
    ``{"type": "unknown", ...}``.  Methods that take ``model`` use the current model on disk when it is None.
    """

    kind: str = ""
    supported_tiers: tuple[str, ...] = (TIER_ELEMENTS,)

    def __init__(self, project: Path | str, level_id: str, *, mode: str | None = None) -> None:
        self.project = Path(project)
        self.level_id = str(level_id)
        self.mode = mode

    # ---------------------------------------------------------------------------------------------------- status
    @abstractmethod
    def availability(self) -> tuple[bool, str | None]:
        """(True, None) when the level can be edited, else (False, i18n reason key)."""

    # ----------------------------------------------------------------------------------------------------- model
    @abstractmethod
    def load_base(self) -> Any:
        """The model a turn starts from (the current model file, with the accepted edits)."""

    def current_model(self) -> Any:
        """The model on disk (default: :meth:`load_base`)."""
        return self.load_base()

    @abstractmethod
    def new_proposal(self, base: Any) -> Any:
        """A proposal on ``base``: ``.apply(op)``, ``.changes``, ``.summary()``, ``.needs_confirmation()``, ``.plan``,
        ``.base``, ``.notes`` (``base`` is never mutated)."""

    def view_frame(self) -> dict[str, Any]:
        """``{"units_per_metre": canvas units per metre, "floor_z": floor height in metres or None}`` for the viewers."""
        return {"units_per_metre": 1.0, "floor_z": None}

    # ------------------------------------------------------------------------------------------------- elements
    @abstractmethod
    def element_list(self, filter: Any = None, model: Any = None) -> list[dict[str, Any]]:  # noqa: A002 - tool name
        """Compact rows (``id``, ``type``, ``label``, main dimensions, ``position``) of the model's elements."""

    @abstractmethod
    def element_details(self, element_id: str, model: Any = None) -> dict[str, Any]:
        """Everything about one element; ``{"found": False, ...}`` when it does not exist."""

    # ------------------------------------------------------------------------------------------------ selection
    @abstractmethod
    def hit_test(self, x: float, y: float, model: Any = None) -> dict[str, Any]:
        """Element under a 2D click, else ``{"type": "point", "x", "y"}``."""

    @abstractmethod
    def selection_from_global_id(self, global_id: str, point: list[float] | None = None,
                                 model: Any = None) -> dict[str, Any]:
        """Selection of an IFC pick: the element of ``global_id``, else a point, else ``{"type": "unknown"}``."""

    @abstractmethod
    def chip_label(self, sel: dict[str, Any], lang: str | None = None) -> str:
        """Short label of a selection for the chips and the chat."""

    @abstractmethod
    def outline(self, sel: dict[str, Any], model: Any = None) -> list[list[list[float]]]:
        """Closed 2D rings (domain frame, metres) of a selected element for the overlays; [] when unknown."""

    @abstractmethod
    def z_range(self, sel: dict[str, Any], model: Any = None) -> tuple[float, float] | None:
        """Absolute height range (metres, IFC frame) of a selected element; None when unknown."""

    @abstractmethod
    def global_ids_for(self, selections: list[dict[str, Any]], model: Any = None) -> list[str]:
        """IFC GlobalIds of the selected elements (points / unknown ids skipped)."""

    def refresh_selection(self, sel: dict[str, Any], model: Any = None) -> dict[str, Any] | None:
        """A selection re-read from ``model`` (new labels after an accept / undo); None when the element is gone."""
        return dict(sel) if isinstance(sel, dict) else None

    # ----------------------------------------------------------------------------------------------------- tools
    @abstractmethod
    def read_tools(self, tier: str = TIER_ELEMENTS) -> list[ToolDef]:
        """Read / measure tools of a tier."""

    @abstractmethod
    def edit_tools(self, tier: str = TIER_ELEMENTS) -> list[ToolDef]:
        """Edit tools of a tier."""

    def tools(self, tier: str = TIER_ELEMENTS) -> list[ToolDef]:
        """All tools of a tier: read tools first, then edit tools."""
        return [*self.read_tools(tier), *self.edit_tools(tier)]

    def cloud_loader(self, base: Any) -> Callable[[], Any] | None:
        """Lazy loader of the domain's point cloud for the tools (None = no cloud)."""
        return None

    def tool_context(self, base: Any, *, settings: Any = None, allow_manual: bool = False, images_allowed: bool = False,
                     tier: str = TIER_ELEMENTS) -> Any:
        """The :class:`~samcloud.assistant.tools.ToolContext` of one turn on ``base``."""
        from samcloud.assistant.tools import ToolContext

        return ToolContext(plan=base, proposal=self.new_proposal(base), settings=settings, allow_manual=allow_manual,
                           images_allowed=images_allowed, cloud_loader=self.cloud_loader(base), project=self.project,
                           level_id=self.level_id, domain=self, tier=normalize_tier(tier))

    # -------------------------------------------------------------------------------------------------------- QA
    @abstractmethod
    def run_qa(self, proposal: Any, baseline: Any, *, work_dir: Path) -> Any:
        """QA of a proposal against ``baseline``: an object with ``ok``, ``new_findings`` and ``summary``."""

    # --------------------------------------------------------------------------------------------------- prompts
    @abstractmethod
    def vocabulary(self, lang: str) -> dict[str, Any]:
        """Prompt vocabulary: ``project_type``, ``model_phrase`` (what is edited), ``element_types`` {kind: name},
        ``rules`` (list of rule texts; the entry ``"{tier}"`` marks where the tier rule goes) and optional
        ``closing`` {tier: text} for the end of the system prompt."""

    def context_message(self, model: Any, selection: list[dict[str, Any]] | None, lang: str | None = None, *,
                        max_elements: int = 80) -> str:
        """The per-turn context message (selection + bounded element list); generic JSON by default."""
        import json

        rows = self.element_list(model=model)
        cap = max(0, int(max_elements))
        payload = {"project_type": self.vocabulary(str(lang or "en")).get("project_type"),
                   "selection": [self.chip_label(sel, lang) for sel in list(selection or []) if isinstance(sel, dict)],
                   "elements": rows[:cap]}
        if len(rows) > cap:
            payload["more"] = len(rows) - cap
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)

    # ------------------------------------------------------------------------------------------- history / export
    @abstractmethod
    def history_backend(self) -> HistoryBackend:
        """Model access of the edit history (see :class:`HistoryBackend`)."""

    @abstractmethod
    def export_actions(self, after: str = "accept") -> list[str]:
        """Ordered page job actions that bring DXF / IFC up to date after ``"accept"`` or ``"undo"`` (the page starts
        them with ``assistant=True``)."""

    @abstractmethod
    def preview(self, proposal: Any) -> dict[str, list[dict[str, Any]]]:
        """Viewer colours of a proposal: ``{"added": [...], "removed": [...], "moved": [...]}`` (see the service)."""


class UnavailableDomain(AssistantDomain):
    """A recognised project type the assistant cannot edit yet: every query is empty, ``availability`` gives the reason."""

    reason_key: str = "assistant.agent.no_plan"

    def availability(self) -> tuple[bool, str | None]:
        return False, self.reason_key

    def load_base(self) -> Any:
        raise FileNotFoundError(self.reason_key)

    def new_proposal(self, base: Any) -> Any:
        raise NotImplementedError(self.reason_key)

    def element_list(self, filter: Any = None, model: Any = None) -> list[dict[str, Any]]:  # noqa: A002
        return []

    def element_details(self, element_id: str, model: Any = None) -> dict[str, Any]:
        return {"found": False, "id": str(element_id)}

    def hit_test(self, x: float, y: float, model: Any = None) -> dict[str, Any]:
        return {"type": "point", "x": float(round(float(x), 3)), "y": float(round(float(y), 3))}

    def selection_from_global_id(self, global_id: str, point: list[float] | None = None,
                                 model: Any = None) -> dict[str, Any]:
        return {"type": "unknown", "global_id": str(global_id or "")}

    def chip_label(self, sel: dict[str, Any], lang: str | None = None) -> str:
        return str((sel or {}).get("label") or (sel or {}).get("id") or "")

    def outline(self, sel: dict[str, Any], model: Any = None) -> list[list[list[float]]]:
        return []

    def z_range(self, sel: dict[str, Any], model: Any = None) -> tuple[float, float] | None:
        return None

    def global_ids_for(self, selections: list[dict[str, Any]], model: Any = None) -> list[str]:
        return []

    def read_tools(self, tier: str = TIER_ELEMENTS) -> list[ToolDef]:
        return []

    def edit_tools(self, tier: str = TIER_ELEMENTS) -> list[ToolDef]:
        return []

    def run_qa(self, proposal: Any, baseline: Any, *, work_dir: Path) -> Any:
        raise NotImplementedError(self.reason_key)

    def vocabulary(self, lang: str) -> dict[str, Any]:
        return {"project_type": self.kind, "model_phrase": "", "element_types": {}, "rules": ["{tier}"]}

    def history_backend(self) -> HistoryBackend:
        raise NotImplementedError(self.reason_key)

    def export_actions(self, after: str = "accept") -> list[str]:
        return []

    def preview(self, proposal: Any) -> dict[str, list[dict[str, Any]]]:
        return {"added": [], "removed": [], "moved": []}


__all__ = ["AssistantDomain", "HistoryBackend", "TIERS", "TIER_ELEMENTS", "TIER_FULL", "ToolDef", "UnavailableDomain",
           "normalize_tier"]
