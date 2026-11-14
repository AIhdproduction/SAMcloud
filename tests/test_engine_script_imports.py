"""Engine modules run as scripts (classify.py) must import their siblings top-level."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / "samcloud" / "engine"


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    return env


def test_instances_work_with_top_level_imports(tmp_path):
    script = tmp_path / "run_instances.py"
    script.write_text(textwrap.dedent(f'''
        import sys
        sys.path.insert(0, r"{ENGINE}")
        import numpy as np
        import instances
        import ply_io

        out = r"{tmp_path / 'classified.ply'}"
        xyz = np.array([[0, 0, 0], [1, 0, 0], [5, 5, 5], [6, 5, 5]], dtype=np.float32)
        rgb = np.full((4, 3), 128, dtype=np.uint8)
        ply_io.write_classified_ply(out, xyz, rgb, np.array([1, 1, 2, 2], dtype=np.uint8))
        ids = np.array([1, 1, 2, 2], dtype=np.int32)
        meta = [
            {{"id": 1, "class_id": 1, "point_count": 2}},
            {{"id": 2, "class_id": 2, "point_count": 2}},
        ]
        instances.write_instance_product(out, ids, meta)
        arr = ply_io.read_vertex_array(out)
        assert "instance_id" in arr.dtype.names, arr.dtype.names
        assert arr["instance_id"].tolist() == [1, 1, 2, 2]
    '''), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=_env(),
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path / "instances.json").read_text(encoding="utf-8"))
    assert data


def test_classify_script_help_runs(tmp_path):
    result = subprocess.run([sys.executable, str(ENGINE / "classify.py"), "--help"], cwd=tmp_path,
                            env=_env(), capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
