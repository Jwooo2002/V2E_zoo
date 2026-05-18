#!/usr/bin/env python
"""Read-only health check for run cache/checkpoint artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.artifact_health import check_artifacts  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Run directory or parent directory to scan.")
    parser.add_argument("--skip-cache", action="store_true", help="Do not scan teacher cache .pt files.")
    parser.add_argument("--skip-checkpoints", action="store_true", help="Do not scan checkpoint .pt files.")
    parser.add_argument(
        "--cache-sample-size",
        type=int,
        default=None,
        help="Scan at most this many cache files, sampled deterministically. All checkpoints are still scanned.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Limit scanned artifacts for quick triage.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--fail-on-corrupt",
        action="store_true",
        help="Exit nonzero if any corrupt or missing artifact is found.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = check_artifacts(
            args.root,
            include_cache=not args.skip_cache,
            include_checkpoints=not args.skip_checkpoints,
            cache_sample_size=args.cache_sample_size,
            max_files=args.max_files,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    payload = report.to_dict()
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded, flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    if args.fail_on_corrupt and not report.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
