"""Text helpers of the protocol: name normalisation, humanised fallbacks, image names, coordinate-safe values."""

from __future__ import annotations

import re
from typing import Any

FACES = {"back", "down", "front", "left", "right", "up"}
FORBIDDEN_KEYS = {"bbox", "bounds", "origin", "coordinates", "coords", "extent", "extents", "center", "centre"}
FORBIDDEN_SUFFIXES = ("_xyz", "_xy", "center", "centre", "_bbox", "_bounds")
_NUMBER_TEXT = re.compile(r"(?<![\w.])-?\d{6,}(?:\.\d+)?(?![\w])|-?\d+\.\d{6,}")
_WIN_PATH = re.compile(r"[A-Za-z]:[\\/](?:[^\\/:*?\"<>|\r\n]+[\\/])*[^\\/:*?\"<>|\s,;]+")


def norm(text: Any) -> str:
    """Lower-case key: every run of non-alphanumeric characters becomes one underscore."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def humanise(text: Any) -> str:
    """Readable fallback for an unknown key: "removed_hidden" or "a.b_c" -> "Removed hidden", "a b c"."""
    words = re.sub(r"[_.]+", " ", str(text)).strip()
    return words[:1].upper() + words[1:] if words else ""


def short_name(name: Any) -> str:
    """Image name without its source folders; cubemap faces keep their face folder ("front/pano_000001.jpg")."""
    parts = [part for part in re.split(r"[\\/]+", str(name)) if part]
    if not parts:
        return ""
    if len(parts) >= 2 and parts[-2].lower() in FACES:
        return f"{parts[-2].lower()}/{parts[-1]}"
    return parts[-1]


def scrub(text: Any) -> str:
    """Free text without absolute paths (kept: file name), long coordinate-like numbers and many-decimal numbers."""
    value = _WIN_PATH.sub(lambda match: re.split(r"[\\/]", match.group(0))[-1], str(text))
    return _NUMBER_TEXT.sub("…", value)


def _forbidden(key: Any) -> bool:
    name = str(key).lower()
    return name in FORBIDDEN_KEYS or name.endswith(FORBIDDEN_SUFFIXES)


def drop_coordinates(value: Any) -> Any:
    """Recursively remove keys that carry positions (bbox, bounds, origin, coordinates, *_xyz, *_xy, *center)."""
    if isinstance(value, dict):
        return {key: drop_coordinates(item) for key, item in value.items() if not _forbidden(key)}
    if isinstance(value, (list, tuple)):
        return [drop_coordinates(item) for item in value]
    return value


def summarize_value(value: Any, fmt_number: Any, entries_text: Any, limit: int = 80, depth: int = 0) -> str:
    """Scalar as text, dict/list as a short summary (never a raw dump), at most ``limit`` characters.

    ``fmt_number(x)`` formats numbers, ``entries_text(n)`` renders "n entries". Floats of 100000 or more (could be
    projected coordinates) are shown as a dash.
    """
    value = drop_coordinates(value)
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    if isinstance(value, int):
        return fmt_number(value)
    if isinstance(value, float):
        return "–" if abs(value) >= 100000 else fmt_number(value)
    if isinstance(value, (list, tuple)):
        return entries_text(len(value))
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if isinstance(item, (dict, list, tuple)):
                text = entries_text(len(item)) if isinstance(item, (list, tuple)) or depth else summarize_value(
                    item, fmt_number, entries_text, limit, depth + 1)
            else:
                text = summarize_value(item, fmt_number, entries_text, limit, depth + 1)
            parts.append(f"{key}: {text}")
        return _cut(", ".join(parts) or "–", limit)
    return _cut(scrub(value), limit)


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
