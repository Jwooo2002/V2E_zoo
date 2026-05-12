#!/usr/bin/env python
"""Run a small experiment inside a reproducible run registry directory."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_small_experiment import (  # noqa: E402
    build_command,
    load_experiment,
    shell_join,
)
from utils.manifest import (  # noqa: E402
    RunManifest,
    copy_config_files,
    create_run_dir,
    generate_run_id,
    get_env_info,
    get_git_info,
    write_manifest,
)
from utils.checkpointing import latest_checkpoint  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--base-output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage", default="8E")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty-git", action="store_true")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--with-eval", action="store_true")
    parser.add_argument("--with-perturbation", action="store_true")
    parser.add_argument("--with-needle", action="store_true")
    parser.add_argument("--with-report", action="store_true")
    parser.add_argument(
        "--eval-trained-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass the latest run checkpoint to evaluation commands when one exists.",
    )
    return parser.parse_args(argv)


def _run_command(command: list[str], *, stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            timeout=300,
        )
    return int(result.returncode)


def _experiment_config(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    overrides = list(args.override)
    explicit_keys = {item.split("=", 1)[0].strip() for item in overrides if "=" in item}
    if "checkpoint_output_dir" not in explicit_keys:
        overrides.append(f"checkpoint_output_dir={run_dir / 'checkpoints'}")
    if "teacher_cache_dir" not in explicit_keys:
        overrides.append(f"teacher_cache_dir={run_dir / 'cache' / 'teacher_logits'}")
    return load_experiment(args.experiment, overrides)


def _latest_eval_checkpoint(
    args: argparse.Namespace,
    run_dir: Path,
    experiment_config: dict[str, Any],
) -> Path | None:
    if not args.eval_trained_checkpoint:
        return None
    checkpoint_dir = Path(str(experiment_config.get("checkpoint_output_dir", run_dir / "checkpoints")))
    return latest_checkpoint(checkpoint_dir)


def _is_mock_experiment(experiment_config: dict[str, Any]) -> bool:
    return (
        bool(experiment_config.get("mock"))
        or (
            experiment_config.get("teacher_type", "mock") == "mock"
            and experiment_config.get("student_type", "mock") == "mock"
            and experiment_config.get("dataset_type", "mock") == "mock"
        )
    )


def _append_common_eval_flags(command: list[str], experiment_config: dict[str, Any]) -> None:
    key_to_flag = {
        "teacher_type": "--teacher-type",
        "student_type": "--student-type",
        "teacher_model_name_or_path": "--teacher-model-name-or-path",
        "student_model_name_or_path": "--student-model-name-or-path",
        "tokenizer_name_or_path": "--tokenizer-name-or-path",
        "dataset_type": "--dataset-type",
        "data_path": "--data-path",
        "student_vocab_size": "--student-vocab-size",
        "seq_len": "--seq-len",
        "batch_size": "--batch-size",
        "student_hidden_size": "--student-hidden-size",
        "student_num_layers": "--student-num-layers",
        "mixed_precision": "--mixed-precision",
        "top_k": "--top-k",
    }
    for key, flag in key_to_flag.items():
        value = experiment_config.get(key)
        if value is not None:
            command.extend([flag, str(value)])
    if experiment_config.get("local_files_only"):
        command.append("--local-files-only")
    if "topk_enabled" in experiment_config:
        command.append("--topk-enabled" if experiment_config["topk_enabled"] else "--no-topk-enabled")


def _eval_command(
    run_dir: Path,
    experiment_config: dict[str, Any],
    checkpoint_path: Path | None,
) -> list[str] | None:
    if not _is_mock_experiment(experiment_config) and checkpoint_path is None:
        return None
    config_path = str(experiment_config.get("config", "configs/train_config.yaml"))
    command = [
        sys.executable,
        str(ROOT / "evaluate.py"),
        "--config",
        config_path,
        "--mode",
        "all",
        "--max_batches",
        "2",
    ]
    if _is_mock_experiment(experiment_config):
        command.append("--mock")
    _append_common_eval_flags(command, experiment_config)
    if checkpoint_path is not None:
        command.extend(["--student-checkpoint", str(checkpoint_path)])
    return command


def _perturbation_command(
    run_dir: Path,
    experiment_config: dict[str, Any],
    checkpoint_path: Path | None,
) -> list[str] | None:
    if not _is_mock_experiment(experiment_config) and checkpoint_path is None:
        return None
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_perturbation_benchmark.py"),
        "--config",
        str(experiment_config.get("config", "configs/train_config.yaml")),
        "--max-batches",
        "2",
        "--output-json",
        str(run_dir / "evals" / "perturbation.json"),
        "--output-csv",
        str(run_dir / "evals" / "perturbation.csv"),
    ]
    if _is_mock_experiment(experiment_config):
        command.append("--mock")
    _append_common_eval_flags(command, experiment_config)
    if checkpoint_path is not None:
        command.extend(["--student-checkpoint", str(checkpoint_path)])
    return command


def _needle_command(run_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_needle_benchmark.py"),
        "--mock",
        "--predictor",
        "oracle",
        "--num-examples",
        "4",
        "--context-lengths",
        "128",
        "--needle-positions",
        "0.1,0.5,0.9",
        "--output-json",
        str(run_dir / "evals" / "needle.json"),
        "--output-csv",
        str(run_dir / "evals" / "needle.csv"),
    ]


def _report_command(run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "summarize_results.py"),
        "--output-dir",
        str(run_dir / "reports"),
    ]
    for name, eval_json in (
        ("eval", run_dir / "evals" / "eval.json"),
        ("perturbation", run_dir / "evals" / "perturbation.json"),
        ("needle", run_dir / "evals" / "needle.json"),
    ):
        if eval_json.is_file():
            command.extend(["--eval-json", f"{name}={eval_json}"])
    return command


def _manifest(
    *,
    run_id: str,
    run_dir: Path,
    args: argparse.Namespace,
    commands: list[list[str]],
    status: str,
    returncodes: dict[str, int | None],
    eval_checkpoint: Path | None = None,
) -> RunManifest:
    git_info = get_git_info(ROOT)
    metadata: dict[str, Any] = {
        "experiment": str(args.experiment),
        "overrides": list(args.override),
        "status": status,
        "returncodes": returncodes,
        "dry_run": bool(args.dry_run),
        "eval_trained_checkpoint": bool(args.eval_trained_checkpoint),
        "eval_checkpoint": None if eval_checkpoint is None else str(eval_checkpoint),
    }
    if git_info.is_dirty and not args.allow_dirty_git:
        metadata["warning"] = "git working tree is dirty"
    return RunManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        project="cdm-mamba-kd",
        stage=str(args.stage),
        command=commands[0] if commands else [],
        config_paths=[str(args.experiment)],
        output_dir=str(run_dir),
        git=asdict(git_info),
        env=asdict(get_env_info()),
        metadata=metadata,
    )


def run_registered_experiment(args: argparse.Namespace) -> Path:
    run_id = args.run_id or generate_run_id(
        prefix="run",
        extra={"experiment": str(args.experiment), "overrides": sorted(args.override)},
    )
    run_dir = create_run_dir(args.base_output_dir, run_id)
    copy_config_files([args.experiment], run_dir / "configs")

    experiment_config = _experiment_config(args, run_dir)
    train_command = build_command(experiment_config)
    commands = [train_command]
    returncodes: dict[str, int | None] = {"train": None}
    status = "planned" if args.dry_run else "success"

    if args.dry_run:
        (run_dir / "logs" / "planned_command.txt").write_text(shell_join(train_command) + "\n", encoding="utf-8")
        print(shell_join(train_command), flush=True)
    else:
        train_code = _run_command(
            train_command,
            stdout_path=run_dir / "logs" / "train.stdout",
            stderr_path=run_dir / "logs" / "train.stderr",
        )
        returncodes["train"] = train_code
        if train_code != 0:
            status = "failed"

    eval_checkpoint = (
        None if args.dry_run or status != "success" else _latest_eval_checkpoint(args, run_dir, experiment_config)
    )

    if not args.dry_run and status == "success" and args.with_eval:
        eval_command = _eval_command(run_dir, experiment_config, eval_checkpoint)
        if eval_command is not None:
            commands.append(eval_command)
            eval_code = _run_command(
                eval_command,
                stdout_path=run_dir / "evals" / "eval.json",
                stderr_path=run_dir / "logs" / "eval.stderr",
            )
            returncodes["eval"] = eval_code
            if eval_code != 0:
                status = "failed"
        else:
            returncodes["eval"] = None

    if not args.dry_run and status == "success" and args.with_perturbation:
        command = _perturbation_command(run_dir, experiment_config, eval_checkpoint)
        if command is not None:
            commands.append(command)
            code = _run_command(
                command,
                stdout_path=run_dir / "logs" / "perturbation.stdout",
                stderr_path=run_dir / "logs" / "perturbation.stderr",
            )
            returncodes["perturbation"] = code
            if code != 0:
                status = "failed"
        else:
            returncodes["perturbation"] = None

    if not args.dry_run and status == "success" and args.with_needle:
        command = _needle_command(run_dir)
        commands.append(command)
        code = _run_command(
            command,
            stdout_path=run_dir / "logs" / "needle.stdout",
            stderr_path=run_dir / "logs" / "needle.stderr",
        )
        returncodes["needle"] = code
        if code != 0:
            status = "failed"

    if not args.dry_run and status == "success" and args.with_report:
        command = _report_command(run_dir)
        commands.append(command)
        code = _run_command(
            command,
            stdout_path=run_dir / "logs" / "report.stdout",
            stderr_path=run_dir / "logs" / "report.stderr",
        )
        returncodes["report"] = code
        if code != 0:
            status = "failed"

    manifest = _manifest(
        run_id=run_id,
        run_dir=run_dir,
        args=args,
        commands=commands,
        status=status,
        returncodes=returncodes,
        eval_checkpoint=eval_checkpoint,
    )
    write_manifest(manifest, run_dir / "manifest.json")
    print(run_dir, flush=True)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_registered_experiment(args)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
