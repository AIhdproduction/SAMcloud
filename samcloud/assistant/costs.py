"""Cost estimation and per-project daily cost limits of the AI assistant.

Prices are USD per million tokens: live from OpenRouter (:class:`ModelInfo` fields) or from the editable table
``prices_usd_per_mtok`` in ``config/ai_assistant.yaml`` (longest model-id prefix wins). Ollama is free and never limited.
A limit of 0 means "no limit". The daily spend of a project is kept in ``<project>/products/assistant/cost_ledger.json``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
from datetime import date
from pathlib import Path

from samcloud.app.i18n import t
from samcloud.assistant.providers.base import ModelInfo, Usage
from samcloud.assistant.settings import AssistantSettings

LOGGER = logging.getLogger(__name__)
LEDGER_RELATIVE = Path("products") / "assistant" / "cost_ledger.json"
CHARS_PER_TOKEN = 4
Price = tuple[float, float]  # (input, output) USD per million tokens
_LOCK = threading.Lock()


def price_for(model_info: ModelInfo | str | None, settings: AssistantSettings) -> Price | None:
    """``(input, output)`` USD per million tokens: live model price, else the settings table, else ``None``."""
    if isinstance(model_info, ModelInfo):
        if model_info.input_usd_per_mtok is not None and model_info.output_usd_per_mtok is not None:
            return float(model_info.input_usd_per_mtok), float(model_info.output_usd_per_mtok)
        model_id = model_info.id
    else:
        model_id = str(model_info or settings.model or "")
    if settings.provider == "ollama":
        return 0.0, 0.0
    best: str | None = None
    for prefix in settings.prices_usd_per_mtok:
        if prefix and model_id.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    if best is None:
        return None
    entry = settings.prices_usd_per_mtok[best]
    return float(entry["input"]), float(entry["output"])


def estimate_request_cost(prompt_chars: int, max_output_tokens: int, price: Price | None) -> float | None:
    """Upper estimate of one request in USD (≈ 4 characters per input token, full output budget); ``None`` without price."""
    if price is None:
        return None
    input_tokens = math.ceil(max(0, int(prompt_chars)) / CHARS_PER_TOKEN)
    return (input_tokens * price[0] + max(0, int(max_output_tokens)) * price[1]) / 1_000_000


def cost_of(usage: Usage, price: Price | None) -> float | None:
    """Actual cost of a request in USD from its token usage; ``None`` without price."""
    if price is None:
        return None
    return (usage.input_tokens * price[0] + usage.output_tokens * price[1]) / 1_000_000


class CostLedger:
    """Daily assistant spend (USD) of one project, keyed by ISO date."""

    def __init__(self, project_dir: Path | str) -> None:
        self.project_dir = Path(project_dir)
        self.path = self.project_dir / LEDGER_RELATIVE

    def _read(self) -> dict[str, float]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        result: dict[str, float] = {}
        for key, value in data.items():
            try:
                result[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _today() -> str:
        return date.today().isoformat()

    def spent_today(self) -> float:
        """USD spent today in this project."""
        return self._read().get(self._today(), 0.0)

    def add(self, usd: float) -> float:
        """Add ``usd`` to today's spend (ignored when not positive); returns the new total of today."""
        with _LOCK:
            data = self._read()
            today = self._today()
            if usd and usd > 0:
                data[today] = round(data.get(today, 0.0) + float(usd), 6)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                handle, temporary = tempfile.mkstemp(prefix=".cost_ledger_", suffix=".json", dir=self.path.parent)
                try:
                    with os.fdopen(handle, "w", encoding="utf-8") as stream:
                        json.dump(data, stream, indent=2, sort_keys=True)
                    os.replace(temporary, self.path)
                except BaseException:
                    try:
                        os.unlink(temporary)
                    except OSError:
                        pass
                    raise
            return data.get(today, 0.0)


def _money(value: float) -> str:
    return f"{value:.2f}"


def check_limits(settings: AssistantSettings, estimated_usd: float | None, ledger: CostLedger | None = None, *,
                 provider: str | None = None, model: str | None = None) -> str | None:
    """Translated reason why the next request must not be sent, or ``None`` when it is allowed.

    Call before every provider request with the :func:`estimate_request_cost` result. Ollama is never limited; a limit
    of 0 means no limit; an unknown price (``None``) is blocked while any limit is set.
    """
    provider = provider or settings.provider
    if provider == "ollama":
        return None
    request_limit = max(0.0, float(settings.cost_limit_request_usd or 0))
    day_limit = max(0.0, float(settings.cost_limit_project_day_usd or 0))
    if request_limit <= 0 and day_limit <= 0:
        return None
    if estimated_usd is None:
        return t("assistant.settings.cost.unknown_price", model=model or settings.model or "?")
    if request_limit > 0 and estimated_usd > request_limit:
        return t("assistant.settings.cost.request_limit", cost=_money(estimated_usd), limit=_money(request_limit))
    if day_limit > 0 and ledger is not None:
        spent = ledger.spent_today()
        if spent + estimated_usd > day_limit:
            return t("assistant.settings.cost.day_limit", spent=_money(spent), cost=_money(estimated_usd),
                     limit=_money(day_limit))
    return None


def describe_price(price: Price | None) -> str:
    """Short display text of a price (for the UI)."""
    if price is None:
        return t("assistant.settings.cost.price_unknown")
    return t("assistant.settings.cost.price", input=_money(price[0]), output=_money(price[1]))


__all__ = ["Price", "price_for", "estimate_request_cost", "cost_of", "CostLedger", "check_limits",
           "describe_price", "LEDGER_RELATIVE"]
