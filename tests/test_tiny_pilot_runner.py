from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_tiny_pilot(*args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/run_tiny_pilot.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=timeout,
    )


def test_tiny_pilot_dry_run_uses_run_registry_and_local_files(tmp_path: Path) -> None:
    result = _run_tiny_pilot(
        "--variant",
        "csdm_topk",
        "--base-output-dir",
        str(tmp_path),
        "--dry-run",
        "--max-steps",
        "1",
    )

    assert "run_registered_experiment.py" in result.stdout
    assert "tiny_real_csdm_topk.yaml" in result.stdout
    assert "train.py" in result.stdout
    assert "--local-files-only" in result.stdout
    assert "--max_steps 1" in result.stdout
    run_dir = Path(result.stdout.strip().splitlines()[-1])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "9A"
    assert manifest["metadata"]["status"] == "planned"


def test_tiny_pilot_allow_downloads_removes_local_files_only(tmp_path: Path) -> None:
    result = _run_tiny_pilot(
        "--variant",
        "ce",
        "--base-output-dir",
        str(tmp_path),
        "--dry-run",
        "--allow-downloads",
    )

    train_line = [line for line in result.stdout.splitlines() if "train.py" in line][-1]
    assert "--local-files-only" not in train_line


def test_tiny_pilot_all_dry_run_runs_four_variants(tmp_path: Path) -> None:
    result = _run_tiny_pilot(
        "--variant",
        "all",
        "--base-output-dir",
        str(tmp_path),
        "--dry-run",
        "--override",
        "max_steps=1",
    )

    assert result.stdout.count("run_registered_experiment.py") == 4
    assert result.stdout.count("train.py") == 4
    assert "tiny_real_ce.yaml" in result.stdout
    assert "tiny_real_kd.yaml" in result.stdout
    assert "tiny_real_csdm.yaml" in result.stdout
    assert "tiny_real_csdm_topk.yaml" in result.stdout
    assert "--max_steps 1" in result.stdout
