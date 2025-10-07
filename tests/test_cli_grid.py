from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_grid_expansion_and_generation(tmp_path: Path) -> None:
    config_path = tmp_path / "grid.yaml"
    config_path.write_text(
        """
configs:
  - model: kuramoto
    M: [3]
    T: [32]
    K: [0.5, 1.0]
    n_realizations: 1
    seed: 21
  - model: gbm
    M: 2
    T: 32
    mu: [0.0, 0.05]
    sigma: 0.2
    dt: 0.01
    n_realizations: 1
    seed: 9
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)

    dry_run_cmd = [
        sys.executable,
        "-m",
        "pymts",
        "generate",
        "--config",
        str(config_path),
        "--dry-run",
    ]
    dry_run = subprocess.run(dry_run_cmd, capture_output=True, text=True, env=env)
    assert dry_run.returncode == 0, dry_run.stderr
    assert dry_run.stdout.count("config_id=") == 4

    outdir = tmp_path / "outputs"
    run_cmd = [
        sys.executable,
        "-m",
        "pymts",
        "generate",
        "--config",
        str(config_path),
        "--outdir",
        str(outdir),
        "--save",
        "--csv",
    ]
    run = subprocess.run(run_cmd, capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr

    parquet_files = sorted(outdir.rglob("*.parquet"))
    metadata_files = sorted(outdir.rglob("*.metadata.json"))
    assert len(parquet_files) == 4
    assert len(metadata_files) == 4

    # Ensure files are placed under model/config_id directories.
    for path in parquet_files:
        assert path.parent.parent.parent == outdir
