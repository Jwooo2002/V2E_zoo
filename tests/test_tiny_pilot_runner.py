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


def test_tiny_pilot_forwards_timeout_flags(tmp_path: Path) -> None:
    timeout_result = _run_tiny_pilot(
        "--variant",
        "ce",
        "--base-output-dir",
        str(tmp_path / "timeout"),
        "--dry-run",
        "--timeout-seconds",
        "123",
    )
    assert "--timeout-seconds 123" in timeout_result.stdout

    no_timeout_result = _run_tiny_pilot(
        "--variant",
        "ce",
        "--base-output-dir",
        str(tmp_path / "no_timeout"),
        "--dry-run",
        "--no-timeout",
    )
    assert "--no-timeout" in no_timeout_result.stdout
    run_dir = Path(no_timeout_result.stdout.strip().splitlines()[-1])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["timeout_seconds"] is None
    assert manifest["metadata"]["timeout_disabled"] is True


def test_tiny_pilot_forwards_storage_preflight_override(tmp_path: Path) -> None:
    result = _run_tiny_pilot(
        "--variant",
        "csdm_topk",
        "--base-output-dir",
        str(tmp_path),
        "--dry-run",
        "--storage-min-free-gb",
        "20",
        "--artifact-health-check",
        "--artifact-health-max-files",
        "10",
        "--artifact-health-cache-sample-size",
        "5",
    )

    assert "--storage-min-free-gb 20" in result.stdout
    assert "--artifact-health-check" in result.stdout
    assert "--artifact-health-max-files 10" in result.stdout
    assert "--artifact-health-cache-sample-size 5" in result.stdout
