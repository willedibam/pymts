from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_cli_help() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)

    command: list[str]
    if shutil.which("pymts"):
        command = ["pymts", "--help"]
    else:
        command = [sys.executable, "-m", "pymts", "--help"]

    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Synthetic multivariate time series generation toolkit." in result.stdout
