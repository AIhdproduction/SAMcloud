"""Synthetic bench: one tiny end-to-end run through the real auto_dxf chain (opt-in, ~40-90 s).

Enable with ``set SAMCLOUD_SYNTHBENCH_E2E=1``; skipped otherwise so the default suite stays fast.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from samcloud.bim.synthbench.runner import run_case
from samcloud.bim.synthbench.scenes import make_scene

pytestmark = pytest.mark.skipif(os.environ.get("SAMCLOUD_SYNTHBENCH_E2E") != "1", reason="set SAMCLOUD_SYNTHBENCH_E2E=1 to run the pipeline end to end")


def test_rect_scene_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import samcloud.bim.synthbench.runner as runner

    original = runner.make_scene

    def coarse(family: str, seed: int):
        spec = original(family, seed)
        spec.scan = replace(spec.scan, angular_step_deg=0.6, voxel_m=0.025)
        return spec

    monkeypatch.setattr(runner, "make_scene", coarse)
    record = run_case("rect_basic", 101, tmp_path, timeout_s=600)
    assert "error" not in record, record.get("traceback")
    summary = record["metrics"]["summary"]
    assert summary["status"] == "completed"
    assert summary["rooms_pred"] == summary["rooms_gt"] == 1
    assert summary["wall_R"] >= 0.75
    assert summary["ifc4_ok"] is True
    assert make_scene("rect_basic", 101).name in record["name"]
