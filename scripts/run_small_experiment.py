"""Run a small CSDM-Mamba-KD experiment from a flat YAML file."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


KEY_TO_FLAG: dict[str, str | None] = {
    "config": "--config",
    "mock": "--mock",
    "max_steps": "--max_steps",
    "teacher_type": "--teacher-type",
    "student_type": "--student-type",
    "teacher_model_name_or_path": "--teacher-model-name-or-path",
    "dataset_type": "--dataset-type",
    "data_path": "--data-path",
    "tokenizer_name_or_path": "--tokenizer-name-or-path",
    "max_examples": "--max-examples",
    "text_field": "--text-field",
    "allow_student_vocab_resize": "--allow-student-vocab-resize",
    "student_model_name_or_path": "--student-model-name-or-path",
    "student_vocab_size": "--student-vocab-size",
    "student_hidden_size": "--student-hidden-size",
    "student_num_layers": "--student-num-layers",
    "student_state_extraction": "--student-state-extraction",
    "off_state_mode": "--off-state-mode",
    "delta_alt_mode": "--delta-alt-mode",
    "off_logits_mode": "--off-logits-mode",
    "off_state_detach_direction": "--off-state-detach-direction",
    "seq_len": "--seq-len",
    "batch_size": "--batch-size",
    "gradient_accumulation_steps": "--gradient-accumulation-steps",
    "mixed_precision": "--mixed-precision",
    "csdm_weight": "--csdm-weight",
    "kd_weight": "--kd-weight",
    "ce_weight": "--ce-weight",
    "local_files_only": "--local-files-only",
    "topk_enabled": "--topk-enabled",
    "top_k": "--top-k",
    "topk_include_labels": "--topk-include-labels",
    "topk_renormalize": "--topk-renormalize",
    "teacher_cache_enabled": "--teacher-cache-enabled",
    "teacher_cache_dir": "--teacher-cache-dir",
    "teacher_cache_overwrite": "--teacher-cache-overwrite",
    "teacher_cache_use_top_k": "--teacher-cache-use-top-k",
    "teacher_cache_top_k": "--teacher-cache-top-k",
    "checkpoint_output_dir": "--checkpoint-output-dir",
    "save_every_steps": "--save-every-steps",
    "save_at_end": "--save-at-end",
    "resume_from": "--resume-from",
    "auto_resume": "--auto-resume",
    "strict_resume": "--strict-resume",
    "load_optimizer": "--load-optimizer",
    "load_rng_state": "--load-rng-state",
}

BOOLEAN_OPTIONAL_KEYS = {
    "allow_student_vocab_resize",
    "off_state_detach_direction",
    "topk_enabled",
    "topk_include_labels",
    "topk_renormalize",
    "teacher_cache_enabled",
    "teacher_cache_overwrite",
    "teacher_cache_use_top_k",
    "save_at_end",
    "auto_resume",
    "strict_resume",
    "load_optimizer",
    "load_rng_state",
}

STORE_TRUE_KEYS = {"mock", "local_files_only"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_path", type=Path, nargs="?", help="Flat YAML experiment file.")
    parser.add_argument("--experiment", type=Path, default=None, help="Flat YAML experiment file.")
    parser.add_argument("--dry-run", action="store_true", help="Print the train.py command without running it.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a flat YAML key. May be provided more than once.",
    )
    return parser.parse_args()


def parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Override must be KEY=VALUE, got {raw!r}.")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override key must not be empty: {raw!r}.")
    if key not in KEY_TO_FLAG:
        raise ValueError(f"Unsupported override key {key!r}.")
    return key, yaml.safe_load(value)


def _parse_override(raw: str) -> tuple[str, Any]:
    return parse_override(raw)


def _validate_flat_config(config: dict[str, Any]) -> None:
    unknown = sorted(set(config) - set(KEY_TO_FLAG))
    if unknown:
        raise ValueError(f"Unsupported experiment key(s): {', '.join(unknown)}.")
    for key, value in config.items():
        if isinstance(value, (dict, list, tuple)):
            raise ValueError(f"Experiment key {key!r} must be a scalar value in the flat YAML file.")


def load_experiment(path: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Experiment YAML must contain a flat mapping.")

    config = dict(loaded)
    config = apply_overrides(config, overrides or [])

    _validate_flat_config(config)
    if "config" not in config:
        raise ValueError("Experiment YAML must include config: path/to/train_config.yaml.")
    return config


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    merged = dict(config)
    for raw_override in overrides:
        key, value = parse_override(raw_override)
        merged[key] = value
    return merged


def _bool_flag(flag: str, value: Any) -> list[str]:
    if not isinstance(value, bool):
        raise ValueError(f"{flag} expects a boolean value.")
    if value:
        return [flag]
    return [f"--no-{flag[2:]}"]


def build_command(config: dict[str, Any]) -> list[str]:
    command = [sys.executable, str(ROOT / "train.py")]
    for key, flag in KEY_TO_FLAG.items():
        if key not in config or config[key] is None:
            continue
        value = config[key]
        if flag is None:
            continue
        if key in STORE_TRUE_KEYS:
            if not isinstance(value, bool):
                raise ValueError(f"{key} expects a boolean value.")
            if value:
                command.append(flag)
            continue
        if key in BOOLEAN_OPTIONAL_KEYS:
            command.extend(_bool_flag(flag, value))
            continue
        if isinstance(value, bool):
            raise ValueError(f"{key} expects a non-boolean scalar value.")
        command.extend([flag, str(value)])
    return command


def build_train_command(config: dict[str, Any]) -> list[str]:
    return build_command(config)


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main() -> None:
    args = parse_args()
    experiment = args.experiment if args.experiment is not None else args.experiment_path
    if experiment is None:
        raise SystemExit("--experiment is required.")
    try:
        config = load_experiment(experiment, args.override)
        command = build_command(config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(str(exc)) from exc

    print(shell_join(command), flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
