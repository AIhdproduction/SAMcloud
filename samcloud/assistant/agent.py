"""Agent loop of the AI assistant: one user sentence + selection -> a checked edit proposal.

:func:`run_turn` talks to the configured provider with the tools of the project's domain
(:mod:`samcloud.assistant.domains`, default indoor) for the active tool tier, executes the tool calls on a proposal of the
domain (never on the model file), runs the domain's QA once the model is done and returns a :class:`TurnResult`.  The
system prompt is built from the domain's vocabulary and the tier (proposal or full-control wording).  It never raises for provider, tool or QA problems: they end up in
``TurnResult.error`` / ``error_kind`` (translated and redacted).  ``plan_model.json`` is only read, never written.

Guards before every request: provider configured and a model chosen, tool support (Ollama), cost limits
(:func:`samcloud.assistant.costs.check_limits` with an upper estimate; Ollama is never limited) and at most
``settings.max_tool_calls`` tool calls per turn.  Images (section slices) are only sent when ``settings.text_only`` is
off and the model reports image support; photos and panoramas are never sent.

Context budget: every request is kept inside :data:`CONTEXT_BUDGET_RATIO` of the model's effective context window
(:func:`context_window`; Ollama: the ``num_ctx`` the adapter sends).  When a prompt would be larger, older tool results
are compressed, older chat turns dropped and the element list shortened (selected / nearby elements stay) -- a request
over the budget is never sent.  When the provider still reports a full context or a repeating model (Ollama "token repeat
limit reached") or the answer comes back empty / truncated, the same question with the same selection is asked ONCE more
in a fresh context (no chat history, short element list, no tool history); ``TurnResult.retried_fresh`` marks that.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from samcloud.app.i18n import current_language, t
from samcloud.assistant import prompts
from samcloud.assistant.costs import CostLedger, check_limits, cost_of, estimate_request_cost, price_for
from samcloud.assistant.credentials import KEY_FIELDS, redact
from samcloud.assistant.domains.base import normalize_tier
from samcloud.assistant.edits import Proposal
from samcloud.assistant.paths import assistant_dir
from samcloud.assistant.providers.base import (
    CHARS_PER_TOKEN,
    ChatMessage,
    ModelInfo,
    ProviderError,
    ToolSpec,
    Usage,
    get_provider,
    is_context_error,
)
from samcloud.assistant.qa import QaResult, run_qa
from samcloud.assistant.tools import (
    MANUAL_NOT_ALLOWED_KEY,
    NO_EVIDENCE_KEY,
    compact,
    dispatch,
    short_args,
    tool_specs,
)

LOGGER = logging.getLogger(__name__)

HISTORY_TURNS = 3  # previous question/answer pairs sent as context (default of settings.context_history_turns)
HISTORY_CHARS = 800  # characters per previous message
IMAGE_CHARS_ESTIMATE = 6000  # an attached image counted as this many characters in the cost / size estimate
QA_DIRNAME = "qa_tmp"
MAX_QA_CORRECTIONS = 1

DEFAULT_CONTEXT_TOKENS = 32768  # context window assumed when the provider does not report one
CONTEXT_BUDGET_RATIO = 0.70  # share of the context window a prompt may use
OUTPUT_MARGIN_TOKENS = 256
MIN_OUTPUT_TOKENS = 256
MIN_CHARS_PER_TOKEN = 2.0  # calibration never goes below this (protects against odd usage numbers)
ELEMENT_CAPS: tuple[int, ...] = (prompts.MAX_CONTEXT_ELEMENTS, 40, 20, 10, 0)  # element-list sizes, largest first
HALF_ELEMENT_CAP = 40  # the element list is first cut to this size, then older chat turns are dropped
FRESH_ELEMENT_CAP = 10  # element list of the fresh-context retry (selection and nearest elements)
KEEP_RECENT_TOOL_RESULTS = 2  # newest tool results compressed last
TOOL_RESULT_CHARS: tuple[int, ...] = (1500, 400)  # compression steps of tool results (characters)


@dataclass
class TurnResult:
    """Outcome of one assistant turn (also what the UI shows).

    ``context_window`` is the effective context window (tokens) of the model, ``prompt_tokens`` the estimated size of the
    largest request of this turn, ``retried_fresh`` True when the question was asked again in a fresh chat context after a
    context / repetition error (a note says so).
    """

    reply: str = ""
    proposal: Proposal | None = None
    summary: list[dict[str, Any]] = field(default_factory=list)
    qa: QaResult | None = None
    needs_confirmation: bool = False
    asks_manual: bool = False
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    error_kind: str | None = None
    provider: str = ""
    model: str = ""
    notes: list[str] = field(default_factory=list)
    retried_fresh: bool = False
    context_window: int | None = None
    prompt_tokens: int = 0
    # full control: the service applied the proposal directly after the passed QA (history entry ``edit_id``)
    applied: bool = False
    edit_id: str | None = None
    tier: str = "elements"

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view for the UI (no proposal object, no plan data beyond the summary)."""
        return {
            "reply": self.reply, "summary": self.summary, "needs_confirmation": self.needs_confirmation,
            "asks_manual": self.asks_manual, "has_proposal": self.proposal is not None,
            "qa": None if self.qa is None else {"ok": self.qa.ok, "new_findings": self.qa.new_findings, "summary": self.qa.summary},
            "usage": {"input_tokens": self.usage.input_tokens, "output_tokens": self.usage.output_tokens},
            "cost_usd": self.cost_usd, "tool_log": self.tool_log, "error": self.error, "error_kind": self.error_kind,
            "provider": self.provider, "model": self.model, "notes": list(self.notes),
            "retried_fresh": self.retried_fresh, "context_window": self.context_window,
            "prompt_tokens": self.prompt_tokens, "applied": self.applied, "edit_id": self.edit_id, "tier": self.tier,
        }


class _Stop(Exception):
    """Ends the loop with a translated message (no proposal)."""

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind


class _ContextFull(Exception):
    """The prompt does not fit the budget even after every reduction (nothing was sent)."""

    def __init__(self, tokens: int, budget: int) -> None:
        super().__init__(f"{tokens} > {budget}")
        self.tokens = tokens
        self.budget = budget


class _EmptyAnswer(Exception):
    """The model answered nothing (no text, no tool call) or was cut off -- typical for an overflowing context."""


_CONTEXT_PROBLEMS = (ProviderError, _ContextFull, _EmptyAnswer)


# ------------------------------------------------------------------------------------------------------ context budget

@dataclass
class ContextBudget:
    """Token budget of the requests of one turn."""

    window: int
    prompt_tokens: int
    max_output: int
    chars_per_token: float = CHARS_PER_TOKEN

    def tokens(self, chars: int) -> int:
        return int(max(0, chars) / self.chars_per_token) + 1

    def calibrate(self, chars: int, input_tokens: int) -> None:
        """Use the real token count of a request when it shows more tokens per character (never fewer)."""
        if input_tokens > 0 and chars > 0:
            ratio = chars / float(input_tokens)
            if ratio < self.chars_per_token:
                self.chars_per_token = max(MIN_CHARS_PER_TOKEN, ratio)


def context_budget(window: int, max_tokens: int) -> ContextBudget:
    """Budget for a context window: prompts up to :data:`CONTEXT_BUDGET_RATIO` of it, the answer always fits."""
    window = max(1024, int(window))
    max_output = max(MIN_OUTPUT_TOKENS, min(int(max_tokens), window // 4))
    prompt = min(int(window * CONTEXT_BUDGET_RATIO), window - max_output - OUTPUT_MARGIN_TOKENS)
    return ContextBudget(window=window, prompt_tokens=max(512, prompt), max_output=max_output)


def context_window(provider: Any, info: ModelInfo) -> int:
    """Effective context window (tokens): what the adapter uses (Ollama ``num_ctx``), else the model's reported context
    length, else :data:`DEFAULT_CONTEXT_TOKENS`."""
    method = getattr(provider, "context_window_for", None)
    if callable(method):
        try:
            value = int(method(info))
        except Exception:  # noqa: BLE001 - a broken adapter hint must not stop the turn
            value = 0
        if value > 0:
            return value
    if info.context_length:
        return int(info.context_length)
    return DEFAULT_CONTEXT_TOKENS


class _Conversation:
    """The messages of one attempt: system rules, context (element list), chat history, the question, the tool loop."""

    def __init__(self, system: str, context_for: Callable[[int], str], element_cap: int, history: list[ChatMessage],
                 prompt: str) -> None:
        self.system = ChatMessage("system", system)
        self._context_for = context_for
        self.element_cap = element_cap
        self.context = ChatMessage("user", context_for(element_cap))
        self.history = list(history)
        self.prompt = ChatMessage("user", prompt)
        self.loop: list[ChatMessage] = []

    def messages(self) -> list[ChatMessage]:
        return [self.system, self.context, *self.history, self.prompt, *self.loop]

    def shrink_elements(self, floor: int = 0) -> bool:
        smaller = [cap for cap in ELEMENT_CAPS if floor <= cap < self.element_cap]
        if not smaller:
            return False
        self.element_cap = smaller[0]
        self.context.content = self._context_for(self.element_cap)
        return True


def compress_tool_result(text: str, limit: int) -> str:
    """A tool result of at most about ``limit`` characters: JSON re-compacted (fewer list items, 2 decimals), else cut."""
    if len(text) <= limit:
        return text
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if value is not None:
        for max_list in (10, 4, 2):
            candidate = json.dumps(compact(value, digits=2, max_list=max_list), ensure_ascii=False, separators=(",", ":"))
            if len(candidate) <= limit:
                return candidate
    return json.dumps({"truncated": True, "preview": text[: max(0, limit - 40)]}, ensure_ascii=False)


def _compress(messages: list[ChatMessage], limit: int) -> bool:
    changed = False
    for message in messages:
        if len(message.content or "") > limit:
            message.content = compress_tool_result(message.content, limit)
            changed = True
    return changed


def _shrink_steps(conv: _Conversation) -> Iterator[None]:
    """Reductions in the order of least loss; yields after each one."""
    tools = [message for message in conv.loop if message.role == "tool"]
    if _compress(tools[:-KEEP_RECENT_TOOL_RESULTS], TOOL_RESULT_CHARS[0]):
        yield
    while conv.shrink_elements(floor=HALF_ELEMENT_CAP):
        yield
    while conv.history:
        del conv.history[0]
        yield
    while conv.shrink_elements(floor=FRESH_ELEMENT_CAP):
        yield
    for limit in TOOL_RESULT_CHARS:
        if _compress(tools, limit):
            yield
    if any(message.images for message in conv.loop):
        _strip_images(conv.loop)
        yield
    while conv.shrink_elements():
        yield


def _fit(conv: _Conversation, tools: list[ToolSpec], budget: ContextBudget) -> int:
    """Shrink ``conv`` until its estimated size fits ``budget.prompt_tokens``; returns the estimated prompt tokens."""
    tokens = budget.tokens(_prompt_chars(conv.messages(), tools))
    if tokens <= budget.prompt_tokens:
        return tokens
    for _ in _shrink_steps(conv):
        tokens = budget.tokens(_prompt_chars(conv.messages(), tools))
        if tokens <= budget.prompt_tokens:
            return tokens
    raise _ContextFull(tokens, budget.prompt_tokens)


# ----------------------------------------------------------------------------------------------------------- helpers

def _secrets(creds: dict[str, Any] | None) -> tuple[str, ...]:
    """The non-empty key values of ``creds`` (the Ollama address is not a secret) for :func:`redact`."""
    return tuple(str(creds.get(name) or "").strip() for name in KEY_FIELDS.values()
                 if creds and str(creds.get(name) or "").strip())


def _history(history_messages: list[Any] | None, secrets: tuple[str, ...] = (), turns: int = HISTORY_TURNS) -> list[ChatMessage]:
    """The last ``turns`` previous user / assistant message pairs (short, text only) as chat messages."""
    count = max(0, int(turns)) * 2
    if count == 0:
        return []
    messages: list[ChatMessage] = []
    for item in list(history_messages or [])[-count:]:
        if isinstance(item, ChatMessage):
            role, text = item.role, item.content
        elif isinstance(item, dict):
            role, text = str(item.get("role") or ""), str(item.get("text") or item.get("content") or "")
        else:
            continue
        if role not in ("user", "assistant") or not text.strip():
            continue
        if len(text) > HISTORY_CHARS:
            text = text[: HISTORY_CHARS - 3] + "..."
        messages.append(ChatMessage(role, redact(text, secrets)))
    return messages


def _prompt_chars(messages: list[ChatMessage], tools: list[ToolSpec]) -> int:
    total = 0
    for message in messages:
        total += len(message.content or "")
        for call in message.tool_calls:
            total += len(call.name) + len(json.dumps(call.arguments, default=str))
        total += IMAGE_CHARS_ESTIMATE * len(message.images)
    total += sum(len(spec.name) + len(spec.description) + len(json.dumps(spec.parameters)) for spec in tools)
    return total


def _strip_images(messages: list[ChatMessage]) -> None:
    for message in messages:
        if message.images:
            message.images = []


def _error_result(result: TurnResult, message: str, kind: str, secrets: tuple[str, ...] = ()) -> TurnResult:
    result.error = redact(message, secrets)
    result.error_kind = kind
    result.proposal = None
    result.summary = []
    result.needs_confirmation = False
    if not result.reply:
        result.reply = result.error
    return result


def _add_cost(result: TurnResult, usage: Usage, price: Any, ledger: CostLedger | None) -> None:
    result.usage.input_tokens += int(usage.input_tokens or 0)
    result.usage.output_tokens += int(usage.output_tokens or 0)
    cost = cost_of(usage, price)
    if cost is None:
        return
    result.cost_usd = (result.cost_usd or 0.0) + cost
    if ledger is not None and cost > 0:
        try:
            ledger.add(cost)
        except OSError as error:
            LOGGER.warning("assistant: cost ledger not written (%s)", type(error).__name__)


def _is_context_problem(error: BaseException) -> bool:
    if isinstance(error, ProviderError):
        return is_context_error(error)
    return isinstance(error, (_ContextFull, _EmptyAnswer))


def _describe_context_problem(error: BaseException, budget: ContextBudget) -> str:
    if isinstance(error, _ContextFull):
        return t("assistant.agent.context_full", tokens=error.tokens, window=budget.window)
    if isinstance(error, _EmptyAnswer):
        return t("assistant.agent.empty_answer")
    return str(error)


def _resolve_tier(domain: Any, settings: Any) -> str:
    from samcloud.assistant.domains.registry import resolve_tier

    return resolve_tier(domain, settings)


def _domain_qa(domain: Any, proposal: Any, base: Any, work_dir: Path) -> QaResult:
    """QA of a proposal through its domain.  The indoor plan QA goes through this module's ``run_qa`` (the name tests
    replace); other domains run their own :meth:`~samcloud.assistant.domains.base.AssistantDomain.run_qa`."""
    if getattr(domain, "kind", "") == "indoor":
        return run_qa(proposal.plan, baseline=base, work_dir=work_dir)
    return domain.run_qa(proposal, base, work_dir=work_dir)


# ---------------------------------------------------------------------------------------------------------- run_turn

def run_turn(*, project: Path | str, level_id: str, prompt: str, selection: list[dict[str, Any]] | None, settings: Any,
             creds: dict[str, Any], provider: Any = None, allow_manual: bool = False,
             history_messages: list[Any] | None = None, plan: Any = None, ledger: CostLedger | None = None,
             cloud: Any = None, domain: Any = None, tier: str | None = None) -> TurnResult:
    """One assistant turn; never raises for provider / tool / QA problems and never writes ``plan_model.json``.

    It only reads the plan; re-applying accepted edits after a pipeline recompute (``ensure_applied``) is done by
    :meth:`samcloud.assistant.service.AssistantService.ask`, never here.  Every text it returns or sends is redacted with
    the stored keys and the non-empty key values of ``creds``.

    ``plan`` (optional) is the base plan to work on (default: ``plan_model.json`` of the level); ``ledger`` the
    project's :class:`CostLedger` (default: the one of ``project``); ``cloud`` an already loaded point cloud.
    ``domain`` is the project's :class:`~samcloud.assistant.domains.base.AssistantDomain` (default: the indoor plan of
    the level); ``tier`` the tool tier (default: :func:`~samcloud.assistant.domains.registry.resolve_tier` of the
    domain and the settings).
    """
    project = Path(project)
    language = current_language()
    model = str(getattr(settings, "model", "") or "").strip()
    provider_name = str(getattr(settings, "provider", "") or "")
    result = TurnResult(provider=provider_name, model=model)
    secrets = _secrets(creds)
    try:
        return _run(result, project=project, level_id=str(level_id), prompt=redact(str(prompt or ""), secrets), selection=list(selection or []),
                    settings=settings, creds=creds or {}, provider=provider, allow_manual=allow_manual,
                    history_messages=history_messages, plan=plan, ledger=ledger, cloud=cloud, language=language,
                    domain=domain, tier=tier)
    except ProviderError as error:
        return _error_result(result, str(error), error.kind, secrets)
    except _Stop as stop:
        return _error_result(result, stop.message, stop.kind, secrets)
    except Exception as error:  # noqa: BLE001 - a turn must never crash the page
        LOGGER.exception("assistant: turn failed (%s)", type(error).__name__)
        return _error_result(result, t("assistant.agent.unexpected", detail=redact(type(error).__name__, secrets)), "error", secrets)


def _run(result: TurnResult, *, project: Path, level_id: str, prompt: str, selection: list[dict[str, Any]], settings: Any,
         creds: dict[str, Any], provider: Any, allow_manual: bool, history_messages: list[Any] | None, plan: Any,
         ledger: CostLedger | None, cloud: Any, language: str, domain: Any = None, tier: str | None = None) -> TurnResult:
    model = result.model
    secrets = _secrets(creds)
    if not prompt.strip():
        raise _Stop(t("assistant.agent.empty_prompt"), "empty_prompt")
    if provider is None:
        if not result.provider:
            raise _Stop(t("assistant.agent.not_configured"), "not_configured")
        try:
            provider = get_provider(result.provider, settings, creds)
        except ValueError:
            raise _Stop(t("assistant.agent.not_configured"), "not_configured") from None
    if not model:
        raise _Stop(t("assistant.agent.no_model"), "no_model")

    # model capabilities (one small request; for Ollama /api/show) -- no chat is sent before these checks pass
    info = provider.model_info(model)
    if not isinstance(info, ModelInfo):
        info = ModelInfo(id=model)
    if info.supports_tools is False:
        raise _Stop(t("assistant.agent.no_tools", model=model), "no_tools")
    images_allowed = (not bool(getattr(settings, "text_only", True))) and info.supports_images is True
    price = price_for(info, settings)

    # the domain and the model this turn works on (read only)
    if domain is None:
        from samcloud.assistant.domains.indoor import IndoorDomain

        domain = IndoorDomain(project, level_id)
    tier = normalize_tier(tier) if tier is not None else _resolve_tier(domain, settings)
    result.tier = tier
    if plan is None:
        available, reason = domain.availability()
        if not available:
            key = reason or "assistant.agent.no_plan"
            raise _Stop(t(key), key.rsplit(".", 1)[-1])
        plan = domain.load_base()
    if ledger is None:
        ledger = CostLedger(project)

    budget = context_budget(context_window(provider, info), int(getattr(settings, "max_output_tokens", 4096) or 4096))
    result.context_window = budget.window
    try:
        turns = max(0, int(getattr(settings, "context_history_turns", HISTORY_TURNS)))
    except (TypeError, ValueError):
        turns = HISTORY_TURNS
    attempt = dict(project=project, level_id=level_id, prompt=prompt, selection=selection, settings=settings,
                   provider=provider, allow_manual=allow_manual, plan=plan, ledger=ledger, cloud=cloud, language=language,
                   images_allowed=images_allowed, price=price, budget=budget, secrets=secrets, domain=domain, tier=tier)
    try:
        return _attempt(result, history=_history(history_messages, secrets, turns), element_cap=ELEMENT_CAPS[0], **attempt)
    except _CONTEXT_PROBLEMS as error:
        if not _is_context_problem(error):
            raise
        LOGGER.info("assistant: context / repetition problem (%s); asking again in a fresh context", type(error).__name__)

    # the same question with the same selection once more: no chat history, short element list, no tool history
    result.retried_fresh = True
    result.tool_log = []
    try:
        _attempt(result, history=[], element_cap=FRESH_ELEMENT_CAP, **attempt)
    except _CONTEXT_PROBLEMS as error:
        if not _is_context_problem(error):
            raise
        raise _Stop(t("assistant.agent.context_failed", detail=_describe_context_problem(error, budget)), "context") from None
    result.notes = [t("assistant.agent.retried_fresh"), *result.notes]
    return result


def _attempt(result: TurnResult, *, project: Path, level_id: str, prompt: str, selection: list[dict[str, Any]], settings: Any,
             provider: Any, allow_manual: bool, plan: Any, ledger: CostLedger, cloud: Any, language: str,
             images_allowed: bool, price: Any, budget: ContextBudget, secrets: tuple[str, ...],
             history: list[ChatMessage], element_cap: int, domain: Any, tier: str) -> TurnResult:
    """One run of the tool loop on a fresh proposal; raises the context problems for :func:`_run` to retry."""
    model = result.model
    ctx = domain.tool_context(plan, settings=settings, allow_manual=allow_manual, images_allowed=images_allowed, tier=tier)
    if cloud is not None:
        ctx.cloud = cloud
    base = ctx.proposal.base
    tools = tool_specs(domain, tier, images_allowed=images_allowed)
    conv = _Conversation(prompts.system_prompt(language, vocabulary=domain.vocabulary(language), tier=tier),
                         lambda cap: domain.context_message(plan, selection, language, max_elements=cap),
                         element_cap, history, prompt)
    max_calls = max(0, int(getattr(settings, "max_tool_calls", 20) or 0))
    timeout = float(getattr(settings, "request_timeout_s", 120.0) or 120.0)
    calls_used = 0
    qa_corrections = 0
    qa: QaResult | None = None
    reply = ""

    while True:
        if not images_allowed:
            _strip_images(conv.messages())
        tokens = _fit(conv, tools, budget)
        result.prompt_tokens = max(result.prompt_tokens, tokens)
        messages = conv.messages()
        chars = _prompt_chars(messages, tools)
        estimate = estimate_request_cost(chars, budget.max_output, price)
        reason = check_limits(settings, estimate, ledger, provider=result.provider, model=model)
        if reason:
            raise _Stop(reason, "cost_limit")
        response = provider.chat(messages, tools, model=model, max_tokens=budget.max_output, timeout=timeout)
        _add_cost(result, response.usage, price, ledger)
        budget.calibrate(chars, int(response.usage.input_tokens or 0))
        answer = response.message
        answer.images = []
        conv.loop.append(ChatMessage("assistant", answer.content or "", list(answer.tool_calls)))

        if answer.tool_calls:
            images: list[bytes] = []
            for call in answer.tool_calls:
                if calls_used >= max_calls:
                    raise _Stop(t("assistant.agent.too_many_tools", count=max_calls), "too_many_tools")
                calls_used += 1
                outcome = dispatch(ctx, call)
                result.tool_log.append({"tool": call.name, "args": redact(short_args(call.arguments), secrets), "ok": not outcome.error,
                                        "error_key": outcome.error_key})
                conv.loop.append(ChatMessage("tool", outcome.content, tool_call_id=call.id))
                if images_allowed:
                    images.extend(outcome.images)
            if images and images_allowed:
                conv.loop.append(ChatMessage("user", prompts.images_note(language), images=images))
            continue

        reply = (answer.content or "").strip()
        if not reply and (response.stop_reason == "max_tokens" or not ctx.proposal.summary()):
            raise _EmptyAnswer()
        if not ctx.proposal.summary():
            qa = None
            break
        qa = _domain_qa(domain, ctx.proposal, base, assistant_dir(project, level_id) / QA_DIRNAME)
        if qa.ok:
            break
        if qa_corrections < MAX_QA_CORRECTIONS:
            qa_corrections += 1
            conv.loop.append(ChatMessage("user", prompts.qa_feedback(qa.new_findings, language)))
            continue
        # still failing after the correction round: the proposal is discarded, the findings are reported
        result.qa = qa
        result.reply = t("assistant.agent.qa_failed", findings=prompts.format_findings(qa.new_findings))
        if reply:
            result.reply = f"{result.reply}\n\n{reply}"
        result.proposal = None
        result.summary = []
        result.needs_confirmation = False
        result.notes = list(ctx.proposal.notes)
        result.reply = redact(result.reply, secrets)
        return result

    summary = ctx.proposal.summary()
    result.qa = qa
    result.notes = list(ctx.proposal.notes)
    result.asks_manual = ctx.last_edit_key in (NO_EVIDENCE_KEY, MANUAL_NOT_ALLOWED_KEY)
    if summary:
        result.proposal = ctx.proposal
        result.summary = summary
        result.needs_confirmation = ctx.proposal.needs_confirmation()
        result.reply = reply or t("assistant.agent.proposal_ready", count=len(summary))
    else:
        result.reply = reply or t("assistant.agent.no_change")
    result.reply = redact(result.reply, secrets)
    return result


__all__ = ["ContextBudget", "TurnResult", "compress_tool_result", "context_budget", "context_window", "run_turn"]
