"""Test workspace fixture compatible with the project's no-tmpdir test command."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    """Give each test a durable directory below SAMCLOUD_TEST_WORKDIR."""
    root = Path(os.environ["SAMCLOUD_TEST_WORKDIR"])
    root.mkdir(parents=True, exist_ok=True)
    root = Path(os.path.realpath(root))  # expand Windows 8.3 short names (HDPROD~1)
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)[:64] + "_"
    path = root / f"{prefix}{uuid.uuid4().hex}"
    path.mkdir()
    return path
