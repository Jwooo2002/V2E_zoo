#!/usr/bin/env python
"""Run selected Stage 11A 7B sanity top-k variants through the run registry."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ENV_VARS = ("CSDM_TEACHER_PATH", "CSDM_TOKENIZER_PATH", "CSDM_DATA_PATH")

VARIANT_CONFIGS = {
    "kd_topk": ROOT / "configs" / "experiments" / "train_7b_sanity_kd_topk.yaml",
    "csdm_topk": ROOT / "configs" / "experiments" / "train_7b_sanity_csdm_topk.yaml",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[*VARIANT_CONFIGS, "all"], default="csdm_topk")
    parser.add_argument("--base-output-dir", type=Path, default=Path("/tmp/csdm_7b_sanity"))
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--student-hidden-size", type=int, default=None)
    parser.add_argument("--student-num-layers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--with-perturbation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow HuggingFace downloads by overriding local_files_only=false.",
    )
    parser.add_argument(
        "--local-files-only",
        dest="local_files_only",
        action="store_true",
        default=True,
        help="Keep HuggingFace loading restricted to local files. This is the default.",
    )
    parser.add_argument(
        "--no-local-files-only",
        dest="local_files_only",
        action="store_false",
        help="Equivalent to --allow-downloads for the generated train command.",
    )
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument(
        "--storage-min-free-gb",
        type=float,
        default=None,
        help="Forward a train.py storage preflight threshold for checkpoint/cache destinations.",
    )
    parser.add_argument(
        "--artifact-health-check",
        action="store_true",
        help="Ask the registered runner to scan cache/checkpoint artifacts after training.",
    )
    parser.add_argument(
        "--artifact-health-max-files",
        type=int,
        default=None,
        help="Legacy quick-triage cap across all artifacts when --artifact-health-check is set.",
    )
    parser.add_argument(
        "--artifact-health-cache-sample-size",
        type=int,
        default=None,
        help="Forward a cache sample size for --artifact-health-check; checkpoints are still all scanned.",
    )
    parser.add_argument(
        "--artifact-health-full-cache",
        action="store_true",
        help="Ask the registered runner to scan every cache file during --artifact-health-check.",
    )
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--no-timeout", action="store_true", help="Disable registry subprocess timeouts.")
    return parser.parse_args(argv)


def _selected_variants(name: str) -> list[str]:
    if name == "all":
        return ["kd_topk", "csdm_topk"]
    return [name]


def _required_env_lines() -> list[str]:
    lines = ["Required environment variables:"]
    for name in REQUIRED_ENV_VARS:
        value = os.environ.get(name)
        lines.append(f"  {name}={value if value else '<unset>'}")
    return lines


def _missing_required_env() -> list[str]:
    return [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]


def _registry_command(args: argparse.Namespace, variant: str) -> list[str]:
    run_id = f"stage11a_{variant}_{time.time_ns()}"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_registered_experiment.py"),
        "--experiment",
        str(VARIANT_CONFIGS[variant].relative_to(ROOT)),
        "--base-output-dir",
        str(args.base_output_dir),
        "--run-id",
        run_id,
        "--stage",
        "11A",
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.with_perturbation:
        command.append("--with-perturbation")
    if args.with_report:
        command.append("--with-report")
    if args.no_timeout:
        command.append("--no-timeout")
    if args.artifact_health_check:
        command.append("--artifact-health-check")
    if args.artifact_health_max_files is not None:
        command.extend(["--artifact-health-max-files", str(args.artifact_health_max_files)])
    if args.artifact_health_cache_sample_size is not None:
        command.extend(["--artifact-health-cache-sample-size", str(args.artifact_health_cache_sample_size)])
    if args.artifact_health_full_cache:
        command.append("--artifact-health-full-cache")

    local_files_only = bool(args.local_files_only) and not args.allow_downloads
    command.extend(["--override", f"local_files_only={str(local_files_only).lower()}"])
    command.extend(["--override", f"max_steps={args.max_steps}"])
    if args.seq_len is not None:
        command.extend(["--override", f"seq_len={args.seq_len}"])
    if args.top_k is not None:
        command.extend(["--override", f"top_k={args.top_k}"])
    if args.student_hidden_size is not None:
        command.extend(["--override", f"student_hidden_size={args.student_hidden_size}"])
    if args.student_num_layers is not None:
        command.extend(["--override", f"student_num_layers={args.student_num_layers}"])
    if args.storage_min_free_gb is not None:
        command.extend(["--override", f"storage_min_free_gb={args.storage_min_free_gb:g}"])
    for override in args.override:
        command.extend(["--override", override])
    return command


def run(args: argparse.Namespace) -> int:
    if args.dry_run:
        print("\n".join(_required_env_lines()), flush=True)
    else:
        missing = _missing_required_env()
        if missing:
            print(
                "Missing required environment variable(s) for non-dry-run 7B sanity launch: "
                + ", ".join(missing),
                file=sys.stderr,
                flush=True,
            )
            return 2

    returncode = 0
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}", flush=True)

    for variant in _selected_variants(args.variant):
        command = _registry_command(args, variant)
        print(f"[{variant}] {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if result.returncode != 0:
            returncode = result.returncode
            break
    return returncode


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
