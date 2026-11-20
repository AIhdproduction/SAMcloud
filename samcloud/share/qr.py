"""QR code of the share link (PNG and SVG via segno)."""

from __future__ import annotations

from pathlib import Path

import segno


def make_qr(url: str, png_path: Path | str, svg_path: Path | str | None = None, *, scale: int = 12, border: int = 4) -> Path:
    """Write the QR code of ``url`` as PNG (and SVG when ``svg_path`` is given); returns the PNG path."""
    code = segno.make(url, error="l", micro=False)
    png = Path(png_path)
    png.parent.mkdir(parents=True, exist_ok=True)
    code.save(str(png), kind="png", scale=scale, border=border, dark="#000000", light="#ffffff")
    if svg_path is not None:
        svg = Path(svg_path)
        svg.parent.mkdir(parents=True, exist_ok=True)
        code.save(str(svg), kind="svg", scale=scale, border=border, dark="#000000", light="#ffffff")
    return png
