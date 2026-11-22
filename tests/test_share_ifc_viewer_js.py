from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "samcloud" / "share" / "project_template"


def test_ifc_html_register_and_module_contract() -> None:
    html = (TEMPLATE / "ifc.html").read_text(encoding="utf-8")

    assert 'href="css/register.css"' in html
    assert re.search(r'<script\s+src="js/register\.js"\s+data-root=""\s+data-view="ifc"></script>', html)
    assert re.search(r'<script\s+type="module"\s+src="js/ifc_viewer\.js"></script>', html)


def test_ifc_viewer_uses_vendored_relative_three_modules() -> None:
    source = (TEMPLATE / "js" / "ifc_viewer.js").read_text(encoding="utf-8")
    specifiers = re.findall(r'^\s*import(?:\s+[^"\']+?\s+from)?\s*["\']([^"\']+)["\'];?\s*$', source, re.MULTILINE)

    assert specifiers == [
        "../lib/three/three.module.js",
        "../lib/three/OrbitControls.js",
        "../lib/three/GLTFLoader.js",
    ]
    assert all(specifier.startswith("../lib/three/") for specifier in specifiers)
    assert not any(specifier == "three" or "://" in specifier for specifier in specifiers)


def test_ifc_viewer_static_data_and_resilience_contract() -> None:
    source = (TEMPLATE / "js" / "ifc_viewer.js").read_text(encoding="utf-8")

    for required in ("view.json", "model.glb", "by_node", "hidden", "colors", "webglcontextlost", "setPixelRatio"):
        assert required in source


def test_ifc_viewer_has_valid_module_syntax(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    source = TEMPLATE / "js" / "ifc_viewer.js"
    target = tmp_path / "ifc_viewer.mjs"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    subprocess.run([node, "--check", str(target)], check=True, capture_output=True, text=True)
