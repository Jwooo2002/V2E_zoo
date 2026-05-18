from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "cdm_mamba_kd_small_experiment_runner",
    ROOT / "scripts" / "run_small_experiment.py",
)
assert RUNNER_SPEC is not None
assert RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)


def test_load_experiment_rejects_unknown_keys(tmp_path: Path) -> None:
    experiment = tmp_path / "bad.yaml"
    experiment.write_text("config: configs/train_config.yaml\nsurprise: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported experiment key"):
        runner.load_experiment(experiment)


def test_parse_override_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unsupported override key"):
        runner.parse_override("surprise=1")


def test_overrides_are_typed_and_mapped_to_train_flags(tmp_path: Path) -> None:
    config = runner.load_experiment(
        ROOT / "configs" / "experiments" / "smoke_mock.yaml",
        [
            "max_steps=3",
            "topk_enabled=false",
            "teacher_cache_enabled=true",
            "csdm_weight=0.05",
            "storage_min_free_gb=12.5",
            f"checkpoint_output_dir={tmp_path / 'ckpt'}",
        ],
    )

    command = runner.build_command(config)

    assert command[0] == sys.executable
    assert command[1].endswith("train.py")
    assert command[command.index("--config") + 1] == "configs/train_config.yaml"
    assert command[command.index("--max_steps") + 1] == "3"
    assert command[command.index("--csdm-weight") + 1] == "0.05"
    assert command[command.index("--storage-min-free-gb") + 1] == "12.5"
    assert "--no-topk-enabled" in command
    assert "--teacher-cache-enabled" in command
    assert str(tmp_path / "ckpt") in command


def test_false_store_true_flags_are_omitted() -> None:
    config = runner.load_experiment(ROOT / "configs" / "experiments" / "smoke_real_mamba.yaml", [])

    command = runner.build_command(config)

    assert "--mock" not in command
    assert command[command.index("--teacher-type") + 1] == "hf"
    assert command[command.index("--student-type") + 1] == "mamba"


def test_runner_dry_run_prints_command_without_training() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_small_experiment.py",
            "--experiment",
            "configs/experiments/smoke_mock.yaml",
            "--dry-run",
            "--override",
            "max_steps=1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )

    assert result.stderr == ""
    assert "train.py" in result.stdout
    assert "--max_steps 1" in result.stdout


def test_runner_mock_experiment_subprocess_runs_one_step(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    checkpoint_dir = tmp_path / "ckpt"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_small_experiment.py",
            "--experiment",
            "configs/experiments/smoke_mock.yaml",
            "--override",
            "max_steps=1",
            "--override",
            "seq_len=8",
            "--override",
            "top_k=4",
            "--override",
            f"teacher_cache_dir={cache_dir}",
            "--override",
            f"checkpoint_output_dir={checkpoint_dir}",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )

    assert "train.py" in result.stdout
    assert '"step": 1' in result.stdout
    assert list(cache_dir.glob("*.pt"))
    assert checkpoint_dir.joinpath("checkpoint_step_1_opt_1.pt").is_file()
