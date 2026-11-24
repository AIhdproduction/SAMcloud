"""Report translations for an explicit language (never touches the app's global language state).

The app's ``samcloud.app.i18n`` translates only for the *current* language, so the report reads
``samcloud/app/locales/<lang>/protocol.json`` (and ``classes.json`` for class names) directly. Lookup falls back
from the requested language to English and finally to the key itself; it never raises. Thread-safe (read-only
caches).
"""

from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
LOCALES = Path(__file__).resolve().parents[1] / "app" / "locales"
SUPPORTED = ("de", "en")
FALLBACK = "en"
_LOCK = threading.Lock()
_WARNED: set[str] = set()


def _flatten(prefix: str, value: Any, target: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, target)
    elif isinstance(value, str):
        target[prefix] = value


@lru_cache(maxsize=8)
def _catalogue(language: str, name: str) -> dict[str, str]:
    path = LOCALES / language / f"{name}.json"
    flat: dict[str, str] = {}
    try:
        _flatten("", json.loads(path.read_text(encoding="utf-8")), flat)
    except (OSError, ValueError) as error:
        LOGGER.warning("report translations %s not readable: %s", path, error)
    return flat


def normalize_language(language: str | None) -> str:
    """"de", "de-CH", "EN" -> supported code; None or unknown -> the app's current language (default "de")."""
    if isinstance(language, str):
        code = language.strip().lower().replace("_", "-").split("-", 1)[0]
        if code in SUPPORTED:
            return code
    try:
        from samcloud.app.i18n import current_language

        code = current_language()
        return code if code in SUPPORTED else "de"
    except Exception:  # noqa: BLE001 - report must work without the app layer
        return "de"


class Translator:
    """``tr("section.alignment")``, ``tr("doc.page", page=1, total=3)``, ``tr.cls("floor")``."""

    def __init__(self, language: str | None = None) -> None:
        self.language = normalize_language(language)

    def __call__(self, key: str, **params: Any) -> str:
        text = _catalogue(self.language, "protocol").get(key)
        if text is None:
            text = _catalogue(FALLBACK, "protocol").get(key)
        if text is None:
            with _LOCK:
                if key not in _WARNED:
                    _WARNED.add(key)
                    LOGGER.warning("report translation missing: %s", key)
            return key
        if params:
            try:
                return text.format(**params)
            except (KeyError, IndexError, ValueError):
                return text
        return text

    def has(self, key: str) -> bool:
        return key in _catalogue(self.language, "protocol") or key in _catalogue(FALLBACK, "protocol")

    def lookup(self, key: str, fallback: str) -> str:
        """Translation of ``key`` or ``fallback`` (for data-driven keys such as capture types)."""
        return self(key) if self.has(key) else fallback

    def name(self, prefix: str, raw: Any) -> str:
        """Label of a data-driven key (``prefix.<normalised raw>``); unknown keys are humanised, never printed raw."""
        import re

        from samcloud.report.textutil import humanise, norm

        text = str(raw)
        key = norm(text)
        if self.has(f"{prefix}.{key}"):
            return self(f"{prefix}.{key}")
        match = re.match(r"^(.*?)[\s_]+pass[\s_]+(\d+)$", text, re.IGNORECASE)
        if match and self.has(f"{prefix}.{norm(match.group(1))}_pass"):
            return self(f"{prefix}.{norm(match.group(1))}_pass", n=match.group(2))
        return humanise(text)

    def step(self, raw: Any) -> str:
        """Pipeline / environment / DXF-IFC step name ("environment: load" -> "Umgebung: Laden")."""
        from samcloud.report.textutil import humanise, norm

        text = str(raw)
        if text.lower().startswith("environment:"):
            rest = text.split(":", 1)[1].strip()
            key = f"step.env_{norm(rest)}"
            return f"{self('step.environment')}: {self(key) if self.has(key) else humanise(rest)}"
        return self.name("step", text)

    def stage(self, raw: Any) -> str:
        """Step-6 stage key; "surface_fusion.removed_hidden" -> its sub label."""
        text = str(raw)
        if "." in text:
            return self.name("sub", text.split(".", 1)[1])
        return self.name("step", text)

    def reason(self, raw: Any) -> str:
        """Skip / not-run reason: known texts translated, others scrubbed of coordinates."""
        import re

        from samcloud.report.textutil import norm, scrub

        text = str(raw)
        if text.startswith("TopologyException"):
            return self("reason.topology")
        match = re.match(r"^Dense size (\d+)px from (\d+)px input \(base (\d+)px\)\.?$", text)
        if match:
            return self("reason.dense_size", size=match.group(1), input=match.group(2), base=match.group(3))
        key = f"reason.{norm(text)}"
        return self(key) if self.has(key) else scrub(text)

    def cls(self, name: Any) -> str:
        """Translated class name (from classes.json) or the raw name."""
        text = str(name)
        return _catalogue(self.language, "classes").get(text) or _catalogue(FALLBACK, "classes").get(text) or text

    # number formatting -------------------------------------------------------------------------
    @property
    def thousands(self) -> str:
        return "’" if self.language == "de" else ","


# ------------------------------------------------------------------------------------ number formatting


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if value == value and value not in (float("inf"), float("-inf")) else None


def fmt_num(tr: Translator, value: Any, decimals: int = 2, dash: str = "–") -> str:
    """Number with thousands separator (de: right single quote, en: comma) and a decimal point; non-numbers -> dash."""
    number = _number(value)
    if number is None:
        return dash
    return f"{number:,.{max(0, int(decimals))}f}".replace(",", tr.thousands)


def fmt_int(tr: Translator, value: Any, dash: str = "–") -> str:
    return fmt_num(tr, value, 0, dash)


def fmt_duration(seconds: Any, dash: str = "–") -> str:
    """"12.3 s", "4:05" (m:ss) or "1:05:03" (h:mm:ss)."""
    number = _number(seconds)
    if number is None or number < 0:
        return dash
    if number < 60:
        return f"{number:.1f} s"
    total = int(round(number))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
