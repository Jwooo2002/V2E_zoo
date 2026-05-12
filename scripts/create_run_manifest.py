#!/usr/bin/env python
"""Create a reproducible run manifest directory."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.manifest import (  # noqa: E402
    copy_config_files,
    create_run_dir,
    generate_run_id,
    get_git_info,
    get_env_info,
    write_manifest,
    RunManifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--prefix", default="run")
    parser.add_argument("--stage", default=None)
    parser.add_argument("--config", action="append", default=[], type=Path)
    parser.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--command", default=None, help="Command string to record in the manifest.")
    parser.add_argument("--allow-dirty-git", action="store_true")
    parser.add_argument("--print-path", action="store_true")
    parser.add_argument("command_args", nargs=argparse.REMAINDER, help="Alternative command after --.")
    return parser.parse_args(argv)


def parse_metadata(items: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"metadata must be KEY=VALUE, got {item!r}.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("metadata key must not be empty.")
        try:
            metadata[key] = json.loads(value)
        except json.JSONDecodeError:
            metadata[key] = value
    return metadata


def command_from_args(args: argparse.Namespace) -> list[str]:
    if args.command is not None:
        return shlex.split(args.command)
    command_args = list(args.command_args)
    if command_args and command_args[0] == "--":
        command_args = command_args[1:]
    return command_args


def create_manifest(args: argparse.Namespace) -> Path:
    metadata = parse_metadata(args.metadata)
    git_info = get_git_info(ROOT)
    if git_info.is_dirty and not args.allow_dirty_git:
        metadata["warning"] = "git working tree is dirty"
    run_id = args.run_id or generate_run_id(
        prefix=args.prefix,
        extra={"stage": args.stage, "config": [str(path) for path in args.config], "command": command_from_args(args)},
    )
    run_dir = create_run_dir(args.output_dir, run_id)
    copied_configs = copy_config_files(args.config, run_dir / "configs") if args.config else []
    manifest = RunManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        project="cdm-mamba-kd",
        stage=args.stage,
        command=command_from_args(args),
        config_paths=[str(path) for path in args.config],
        output_dir=str(run_dir),
        git=asdict(git_info),
        env=asdict(get_env_info()),
        metadata={**metadata, "copied_configs": [str(path) for path in copied_configs]},
    )
    write_manifest(manifest, run_dir / "manifest.json")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = create_manifest(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.print_path:
        print(run_dir)
    else:
        print(f"manifest={run_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
