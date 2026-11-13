"""Job-runner integration coverage for orthophoto clean-plan exports."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import uuid

import pytest

from samcloud.dxf.cancel import CancelToken
from samcloud.dxf.jobs import run_clean_plan, run_export, run_generate, run_ortho, run_segment
from samcloud.ortho.plan import clean_plan_cache_dir
from ortho_synthetic import make_outdoor_project


@pytest.fixture()
def work_dir(request: pytest.FixtureRequest) -> Path:
    configured_root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    if configured_root:
        path = Path(configured_root) / f"ortho_jobs_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return request.getfixturevalue("tmp_path")


def test_jobs_clean_plan_and_export_variant(work_dir: Path) -> None:
    project = make_outdoor_project(work_dir, seed=20261110)
    point_path = project.project_dir / "products" / "classified" / "classified.ply"
    cancel = CancelToken()
    logger = logging.getLogger("samcloud.ortho.jobs.test")
    progress = lambda _fraction, _stage: None
    run_ortho(project.project_dir, point_path, None, progress, cancel, logger, mode="outdoor")
    run_segment(project.project_dir, point_path, None, progress, cancel, logger, mode="outdoor", source="pointcloud")
    run_clean_plan(project.project_dir, point_path, None, progress, cancel, logger, mode="outdoor")
    cache = clean_plan_cache_dir(project.project_dir, None)
    assert (cache / "result.json").is_file()
    run_export(project.project_dir, point_path, None, progress, cancel, logger, mode="outdoor", variant="ortho")
    clean_exports = list((project.project_dir / "products" / "dxf" / "export").glob("*_clean.dxf"))
    assert len(clean_exports) == 1
    assert "S-PAVE" in clean_exports[0].read_bytes().decode("latin1", errors="ignore")
    run_generate(project.project_dir, point_path, None, progress, cancel, logger, mode="outdoor")
    run_export(project.project_dir, point_path, None, progress, cancel, logger, mode="outdoor", variant="pointcloud")
    assert any(path.name != clean_exports[0].name for path in (project.project_dir / "products" / "dxf" / "export").glob("*.dxf"))
