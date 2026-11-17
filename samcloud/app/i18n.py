"""Central UI translations (German default, English) for the SAMcloud web interface.

Only text shown in the browser is translated. Internal values (IDs, status codes,
class names, file contents) never pass through this module.

Usage::

    from samcloud.app.i18n import t, translate_or, current_language

    t("nav.overview")                          # text in the current language
    t("common.points", count=3)                # str.format placeholders
    translate_or(f"classes.{name}", name)      # dynamic keys with a data fallback

Catalogues live in ``samcloud/app/locales/<lang>/<namespace>.json``; a key is
addressed as ``<namespace>.<path>``. Lookup falls back from the current language
to English and finally to the key itself; it never raises.

The language is resolved from (highest priority first) the stored user choice in
``<project root>/.samcloud/user_preferences.json``, ``application.language`` in
``config/defaults.yml`` and :data:`DEFAULT_LANGUAGE`.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from samcloud.core.paths import PROJECT_ROOT


SUPPORTED_LANGUAGES: dict[str, str] = {"de": "Deutsch", "en": "English"}
DEFAULT_LANGUAGE = "de"
FALLBACK_LANGUAGE = "en"
LOCALES_DIRECTORY = Path(__file__).resolve().parent / "locales"
PREFERENCES_PATH = PROJECT_ROOT / ".samcloud" / "user_preferences.json"
PREFERENCE_KEY = "ui_language"

_LOGGER = logging.getLogger(__name__)
_LOCK = threading.RLock()
_catalogues: dict[str, dict[str, str]] | None = None
_language_cache: str | None = None
_warned: set[tuple[str, str]] = set()


# --------------------------------------------------------------------------- catalogues


def _flatten(prefix: str, value: Any, target: dict[str, str], source: Path) -> None:
    """Flatten nested JSON objects into dotted keys with string leaves."""
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}", item, target, source)
    elif isinstance(value, str):
        target[prefix] = value
    else:
        _warn_once("catalogue-value", f"{source}:{prefix}", "Ignoring non-text translation %s in %s", prefix, source)


def _load_language(language: str) -> dict[str, str]:
    """Read every namespace file of one language into a flat key -> text mapping."""
    catalogue: dict[str, str] = {}
    directory = LOCALES_DIRECTORY / language
    if not directory.is_dir():
        return catalogue
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            _warn_once("catalogue-file", str(path), "Cannot read translation file %s: %s", path, error)
            continue
        if not isinstance(data, dict):
            _warn_once("catalogue-file", str(path), "Translation file %s must contain a JSON object", path)
            continue
        _flatten(path.stem, data, catalogue, path)
    return catalogue


def _catalogue_set() -> dict[str, dict[str, str]]:
    """Return all catalogues, loading them lazily once."""
    global _catalogues
    catalogues = _catalogues
    if catalogues is not None:
        return catalogues
    with _LOCK:
        if _catalogues is None:
            _catalogues = {language: _load_language(language) for language in SUPPORTED_LANGUAGES}
        return _catalogues


def reload_translations() -> None:
    """Drop cached catalogues and the cached language (used by tests and after editing locale files)."""
    global _catalogues, _language_cache
    with _LOCK:
        _catalogues = None
        _language_cache = None


def catalogue(language: str) -> dict[str, str]:
    """Return a copy of the flat key -> text catalogue of one language."""
    return dict(_catalogue_set().get(language, {}))


# --------------------------------------------------------------------------- lookup


def _lookup(key: str) -> str | None:
    catalogues = _catalogue_set()
    text = catalogues.get(current_language(), {}).get(key)
    if text is None:
        text = catalogues.get(FALLBACK_LANGUAGE, {}).get(key)
    return text


def _format(text: str, params: dict[str, Any]) -> str:
    if not params:
        return text
    try:
        return text.format(**params)
    except (KeyError, IndexError, ValueError, AttributeError, TypeError):
        return text


def t(key: str, **params: Any) -> str:
    """Translate ``key`` into the current language; falls back to English, then to the key itself."""
    text = _lookup(key)
    if text is None:
        return key
    return _format(text, params)


def translate_or(key: str, fallback: str, **params: Any) -> str:
    """Translate ``key`` when the current or English catalogue knows it, otherwise return ``fallback``."""
    text = _lookup(key)
    if text is None:
        return fallback
    return _format(text, params)


def has_translation(key: str) -> bool:
    """Whether the current or English catalogue contains ``key``."""
    return _lookup(key) is not None


# --------------------------------------------------------------------------- language resolution


def _warn_once(kind: str, value: str, message: str, *args: Any) -> None:
    marker = (kind, value)
    if marker in _warned:
        return
    _warned.add(marker)
    _LOGGER.warning(message, *args)


def normalize_language(value: Any) -> str | None:
    """Map values such as "DE", "de-CH" or "en_US" to a supported code, or None when unsupported."""
    if not isinstance(value, str):
        return None
    code = value.strip().lower().replace("_", "-").split("-", 1)[0]
    return code if code in SUPPORTED_LANGUAGES else None


def _read_preferences() -> dict[str, Any]:
    """Read the preferences file; a missing or broken file counts as empty."""
    path = PREFERENCES_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        _warn_once("preferences-file", str(path), "Cannot read UI preferences %s: %s", path, error)
        return {}
    if not isinstance(data, dict):
        _warn_once("preferences-file", str(path), "UI preferences %s must contain a JSON object", path)
        return {}
    return data


def _write_preferences(data: dict[str, Any]) -> None:
    """Write the preferences file atomically (temp file in the same directory + os.replace)."""
    path = PREFERENCES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".user_preferences.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _read_config_language() -> Any:
    """Return the raw ``application.language`` value from config/defaults.yml, or None."""
    try:
        from samcloud.core.configuration import load_global_config

        configuration = load_global_config("defaults.yml")
    except Exception as error:  # missing file, invalid YAML, non-mapping content
        _warn_once("config-file", "defaults.yml", "Cannot read UI language from config/defaults.yml: %s", error)
        return None
    application = configuration.get("application") if isinstance(configuration, dict) else None
    if not isinstance(application, dict):
        return None
    return application.get("language")


def user_language() -> str | None:
    """Return the language stored by the user in the settings, or None."""
    raw = _read_preferences().get(PREFERENCE_KEY)
    if raw is None:
        return None
    language = normalize_language(raw)
    if language is None:
        _warn_once("user", str(raw), "Ignoring unsupported stored UI language %r", raw)
    return language


def config_language() -> str | None:
    """Return the language configured in config/defaults.yml, or None when missing or unsupported."""
    raw = _read_config_language()
    if raw is None:
        return None
    language = normalize_language(raw)
    if language is None:
        _warn_once("config", str(raw), "Ignoring unsupported UI language %r in config/defaults.yml", raw)
    return language


def _resolve_language() -> str:
    return user_language() or config_language() or DEFAULT_LANGUAGE


def current_language() -> str:
    """Return the active UI language (cached; refreshed by set_language / clear_user_language)."""
    global _language_cache
    language = _language_cache
    if language is not None:
        return language
    with _LOCK:
        if _language_cache is None:
            _language_cache = _resolve_language()
        return _language_cache


def set_language(language: str) -> str:
    """Store the user's language choice and make it active. Returns the normalised code."""
    global _language_cache
    normalized = normalize_language(language)
    if normalized is None:
        raise ValueError(f"Unsupported UI language: {language!r}")
    with _LOCK:
        data = _read_preferences()
        data[PREFERENCE_KEY] = normalized
        _write_preferences(data)
        _language_cache = normalized
    return normalized


def clear_user_language() -> str:
    """Remove the stored choice so the config default applies again. Returns the new active language."""
    global _language_cache
    with _LOCK:
        if PREFERENCES_PATH.exists():
            data = _read_preferences()
            if PREFERENCE_KEY in data:
                data.pop(PREFERENCE_KEY)
                _write_preferences(data)
        _language_cache = None
        return current_language()
