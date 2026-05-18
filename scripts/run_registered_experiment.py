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
    validate_execution_paths,
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
from utils.artifact_health import check_artifacts  # noqa: E402


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
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Subprocess timeout for train/eval/report commands. Defaults to 300 seconds for smoke runs.",
    )
    parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="Disable subprocess timeouts for longer real training runs.",
    )
    parser.add_argument(
        "--eval-trained-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass the latest run checkpoint to evaluation commands when one exists.",
    )
    parser.add_argument(
        "--artifact-health-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After training, scan run cache/checkpoint .pt files before eval/report steps.",
    )
    parser.add_argument(
        "--artifact-health-max-files",
        type=int,
        default=None,
        help="Legacy quick-triage cap across all artifacts during --artifact-health-check.",
    )
    parser.add_argument(
        "--artifact-health-cache-sample-size",
        type=int,
        default=64,
        help="Number of cache files to sample during --artifact-health-check; checkpoints are always scanned.",
    )
    parser.add_argument(
        "--artifact-health-full-cache",
        action="store_true",
        help="Scan every cache file during --artifact-health-check instead of sampling.",
    )
    return parser.parse_args(argv)


def _effective_timeout(args: argparse.Namespace) -> float | None:
    if args.no_timeout:
        return None
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive, or use --no-timeout.")
    return float(args.timeout_seconds)


def _run_command(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float | None,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_label = "disabled" if timeout_seconds is None else f"{timeout_seconds:g} seconds"
            stderr_handle.write(f"Command timed out after {timeout_label}.\n")
            if exc.stderr:
                stderr_text = (
                    exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr)
                )
                stderr_handle.write(stderr_text)
            if exc.stdout:
                stdout_text = (
                    exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout)
                )
                stdout_handle.write(stdout_text)
            return 124
    return int(result.returncode)


def _first_nonempty_line(path: Path, *, max_chars: int = 500) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped[:max_chars]
    return None


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
        "--dual-report",
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
    failure_stage: str | None = None,
    failure_reason: str | None = None,
    failure_log_path: Path | None = None,
) -> RunManifest:
    git_info = get_git_info(ROOT)
    timeout_seconds = _effective_timeout(args)
    metadata: dict[str, Any] = {
        "experiment": str(args.experiment),
        "overrides": list(args.override),
        "status": status,
        "returncodes": returncodes,
        "dry_run": bool(args.dry_run),
        "timeout_seconds": timeout_seconds,
        "timeout_disabled": timeout_seconds is None,
        "eval_trained_checkpoint": bool(args.eval_trained_checkpoint),
        "eval_checkpoint": None if eval_checkpoint is None else str(eval_checkpoint),
        "artifact_health_check": bool(args.artifact_health_check),
        "artifact_health_max_files": args.artifact_health_max_files,
        "artifact_health_cache_sample_size": (
            None if args.artifact_health_full_cache else args.artifact_health_cache_sample_size
        ),
        "artifact_health_full_cache": bool(args.artifact_health_full_cache),
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "failure_log_path": None if failure_log_path is None else str(failure_log_path),
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


def run_registered_experiment(args: argparse.Namespace) -> tuple[Path, str]:
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
    timeout_seconds = _effective_timeout(args)
    failure_stage: str | None = None
    failure_reason: str | None = None
    failure_log_path: Path | None = None

    if args.dry_run:
        (run_dir / "logs" / "planned_command.txt").write_text(shell_join(train_command) + "\n", encoding="utf-8")
        print(shell_join(train_command), flush=True)
    else:
        try:
            validate_execution_paths(experiment_config)
        except (OSError, ValueError) as exc:
            status = "failed"
            returncodes["preflight"] = 1
            failure_stage = "preflight"
            failure_reason = str(exc)
            failure_log_path = run_dir / "logs" / "preflight.stderr"
            failure_log_path.parent.mkdir(parents=True, exist_ok=True)
            failure_log_path.write_text(f"{failure_reason}\n", encoding="utf-8")

        if status == "success":
            train_code = _run_command(
                train_command,
                stdout_path=run_dir / "logs" / "train.stdout",
                stderr_path=run_dir / "logs" / "train.stderr",
                timeout_seconds=timeout_seconds,
            )
            returncodes["train"] = train_code
            if train_code != 0:
                status = "failed"
                failure_stage = "train"
                failure_log_path = run_dir / "logs" / "train.stderr"
                failure_reason = _first_nonempty_line(failure_log_path) or f"train exited with code {train_code}"

    if not args.dry_run and args.artifact_health_check:
        health_report = check_artifacts(
            run_dir,
            cache_sample_size=None if args.artifact_health_full_cache else args.artifact_health_cache_sample_size,
            max_files=args.artifact_health_max_files,
        )
        health_path = run_dir / "artifacts" / "artifact_health.json"
        health_path.parent.mkdir(parents=True, exist_ok=True)
        health_path.write_text(json.dumps(health_report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        health_code = 0 if health_report.ok else 1
        returncodes["artifact_health"] = health_code
        if health_code != 0:
            status = "failed"
            if failure_stage is None:
                failure_stage = "artifact_health"
                failure_log_path = health_path
                failure_reason = (
                    "artifact health failed: "
                    f"corrupt_count={health_report.corrupt_count}, "
                    f"missing_count={health_report.missing_count}, "
                    f"checked_count={health_report.checked_count}"
                )

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
                timeout_seconds=timeout_seconds,
            )
            returncodes["eval"] = eval_code
            if eval_code != 0:
                status = "failed"
                failure_stage = "eval"
                failure_log_path = run_dir / "logs" / "eval.stderr"
                failure_reason = _first_nonempty_line(failure_log_path) or f"eval exited with code {eval_code}"
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
                timeout_seconds=timeout_seconds,
            )
            returncodes["perturbation"] = code
            if code != 0:
                status = "failed"
                failure_stage = "perturbation"
                failure_log_path = run_dir / "logs" / "perturbation.stderr"
                failure_reason = _first_nonempty_line(failure_log_path) or f"perturbation exited with code {code}"
        else:
            returncodes["perturbation"] = None

    if not args.dry_run and status == "success" and args.with_needle:
        command = _needle_command(run_dir)
        commands.append(command)
        code = _run_command(
            command,
            stdout_path=run_dir / "logs" / "needle.stdout",
            stderr_path=run_dir / "logs" / "needle.stderr",
            timeout_seconds=timeout_seconds,
        )
        returncodes["needle"] = code
        if code != 0:
            status = "failed"
            failure_stage = "needle"
            failure_log_path = run_dir / "logs" / "needle.stderr"
            failure_reason = _first_nonempty_line(failure_log_path) or f"needle exited with code {code}"

    if not args.dry_run and status == "success" and args.with_report:
        command = _report_command(run_dir)
        commands.append(command)
        code = _run_command(
            command,
            stdout_path=run_dir / "logs" / "report.stdout",
            stderr_path=run_dir / "logs" / "report.stderr",
            timeout_seconds=timeout_seconds,
        )
        returncodes["report"] = code
        if code != 0:
            status = "failed"
            failure_stage = "report"
            failure_log_path = run_dir / "logs" / "report.stderr"
            failure_reason = _first_nonempty_line(failure_log_path) or f"report exited with code {code}"

    manifest = _manifest(
        run_id=run_id,
        run_dir=run_dir,
        args=args,
        commands=commands,
        status=status,
        returncodes=returncodes,
        eval_checkpoint=eval_checkpoint,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        failure_log_path=failure_log_path,
    )
    write_manifest(manifest, run_dir / "manifest.json")
    print(run_dir, flush=True)
    return run_dir, status


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _run_dir, status = run_registered_experiment(args)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(str(exc)) from exc
    return 0 if status in {"success", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
