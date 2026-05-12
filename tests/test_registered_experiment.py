from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_create_run_manifest_cli_creates_manifest_and_copies_config(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_run_manifest.py",
            "--output-dir",
            str(tmp_path),
            "--stage",
            "8E",
            "--config",
            "configs/experiments/smoke_mock.yaml",
            "--metadata",
            "purpose=test",
            "--print-path",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )

    run_dir = Path(result.stdout.strip())
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.is_file()
    assert (run_dir / "configs" / "smoke_mock.yaml").is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stage"] == "8E"
    assert manifest["metadata"]["purpose"] == "test"
    assert "git" in manifest
    assert "env" in manifest


def test_run_registered_experiment_dry_run_creates_manifest_and_plan(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_registered_experiment.py",
            "--experiment",
            "configs/experiments/smoke_mock.yaml",
            "--base-output-dir",
            str(tmp_path),
            "--dry-run",
            "--override",
            "max_steps=1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )

    run_dir = Path(result.stdout.splitlines()[-1])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    planned = (run_dir / "logs" / "planned_command.txt").read_text(encoding="utf-8")
    assert manifest["metadata"]["status"] == "planned"
    assert "--max_steps 1" in planned or "--max_steps" in planned
    assert not (run_dir / "logs" / "train.stdout").exists()


def test_run_registered_experiment_executes_mock_smoke_under_run_dir(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_registered_experiment.py",
            "--experiment",
            "configs/experiments/smoke_mock.yaml",
            "--base-output-dir",
            str(tmp_path),
            "--override",
            "max_steps=1",
            "--with-eval",
            "--with-perturbation",
            "--with-needle",
            "--with-report",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )

    run_dir = Path(result.stdout.strip().splitlines()[-1])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["status"] == "success"
    assert manifest["metadata"]["returncodes"]["train"] == 0
    assert manifest["metadata"]["returncodes"]["report"] == 0
    assert (run_dir / "logs" / "train.stdout").is_file()
    assert (run_dir / "logs" / "train.stderr").is_file()
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "cache" / "teacher_logits").is_dir()
    assert (run_dir / "evals" / "eval.json").is_file()
    assert (run_dir / "evals" / "perturbation.json").is_file()
    assert (run_dir / "evals" / "needle.json").is_file()
    assert (run_dir / "reports" / "report.json").is_file()
    assert (run_dir / "reports" / "report.csv").is_file()
    assert (run_dir / "reports" / "report.md").is_file()
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_step_1_opt_1.pt"
    assert checkpoint_path.is_file()
    assert manifest["metadata"]["eval_checkpoint"] == str(checkpoint_path)
    eval_metrics = json.loads((run_dir / "evals" / "eval.json").read_text(encoding="utf-8"))
    perturbation_metrics = json.loads((run_dir / "evals" / "perturbation.json").read_text(encoding="utf-8"))
    assert eval_metrics["metadata"]["checkpoint_loaded"] is True
    assert perturbation_metrics["metadata"]["checkpoint_loaded"] is True
    assert eval_metrics["metadata"]["student_checkpoint"] == str(checkpoint_path)
    assert eval_metrics["metadata"]["checkpoint_step"] == 1
    assert perturbation_metrics["metadata"]["student_checkpoint"] == str(checkpoint_path)
    assert perturbation_metrics["metadata"]["checkpoint_step"] == 1
    assert perturbation_metrics["metadata"]["checkpoint_loaded"] is True
    assert "full_vocab" in perturbation_metrics
    assert "topk" in perturbation_metrics
    assert "full_vocab" in perturbation_metrics["by_mode"]["delta_projection"]
    assert "topk" in perturbation_metrics["by_mode"]["delta_projection"]
    assert perturbation_metrics["metadata"]["student_checkpoint"] == str(checkpoint_path)
    assert perturbation_metrics["summary"]["checkpoint_loaded"] is True
    assert str(run_dir / "checkpoints") in " ".join(manifest["command"])
    assert str(run_dir / "cache" / "teacher_logits") in " ".join(manifest["command"])


def test_registered_experiment_overrides_do_not_write_repo_root_outputs(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/run_registered_experiment.py",
            "--experiment",
            "configs/experiments/smoke_mock.yaml",
            "--base-output-dir",
            str(tmp_path),
            "--override",
            "max_steps=1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )

    assert any(tmp_path.iterdir())
