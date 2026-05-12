#!/usr/bin/env python
"""Run Stage 9A tiny real pilot variants through the run registry."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

VARIANT_CONFIGS = {
    "ce": ROOT / "configs" / "pilots" / "tiny_real_ce.yaml",
    "kd": ROOT / "configs" / "pilots" / "tiny_real_kd.yaml",
    "csdm": ROOT / "configs" / "pilots" / "tiny_real_csdm.yaml",
    "csdm_topk": ROOT / "configs" / "pilots" / "tiny_real_csdm_topk.yaml",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[*VARIANT_CONFIGS, "all"], default="csdm_topk")
    parser.add_argument("--base-output-dir", type=Path, default=Path("/tmp/csdm_tiny_pilot"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--with-eval", action="store_true")
    parser.add_argument("--with-perturbation", action="store_true")
    parser.add_argument("--with-needle", action="store_true")
    parser.add_argument("--with-report", action="store_true")
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
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def _selected_variants(name: str) -> list[str]:
    if name == "all":
        return ["ce", "kd", "csdm", "csdm_topk"]
    return [name]


def _registry_command(args: argparse.Namespace, variant: str) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_registered_experiment.py"),
        "--experiment",
        str(VARIANT_CONFIGS[variant].relative_to(ROOT)),
        "--base-output-dir",
        str(args.base_output_dir),
        "--stage",
        "9A",
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.with_eval:
        command.append("--with-eval")
    if args.with_perturbation:
        command.append("--with-perturbation")
    if args.with_needle:
        command.append("--with-needle")
    if args.with_report:
        command.append("--with-report")

    local_files_only = bool(args.local_files_only) and not args.allow_downloads
    command.extend(["--override", f"local_files_only={str(local_files_only).lower()}"])
    if args.max_steps is not None:
        command.extend(["--override", f"max_steps={args.max_steps}"])
    for override in args.override:
        command.extend(["--override", override])
    return command


def run(args: argparse.Namespace) -> int:
    returncode = 0
    for variant in _selected_variants(args.variant):
        command = _registry_command(args, variant)
        print(f"[{variant}] {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            returncode = result.returncode
            break
    return returncode


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
