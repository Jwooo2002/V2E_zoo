"""Run and summarize small CSDM-Mamba ablation matrices."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_small_experiment import (  # noqa: E402
    KEY_TO_FLAG,
    apply_overrides,
    build_command,
    parse_override,
    shell_join,
)


RUNNER_KEYS = {
    "name",
    "description",
    "requires_mamba",
    "skip_reason",
    "tags",
}
METRIC_KEYS = ("total", "ce", "kd", "csdm", "grad_norm", "optimizer_step", "cuda_memory_mb")
SUMMARY_COLUMNS = ("name", "status", *METRIC_KEYS, "returncode", "reason")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "matrix_path",
        type=Path,
        nargs="?",
        help="Ablation matrix YAML file. Defaults to configs/ablations/csdm_mamba_smoke.yaml.",
    )
    parser.add_argument("--matrix", type=Path, default=None, help="Ablation matrix YAML file.")
    parser.add_argument("--dry-run", action="store_true", help="Print variant commands without running them.")
    parser.add_argument("--only", action="append", default=[], help="Run only this variant name. Repeatable.")
    parser.add_argument("--skip", action="append", default=[], help="Skip this variant name. Repeatable.")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/csdm_ablations"))
    parser.add_argument("--max-workers", type=int, default=1, help="Reserved; Stage 8A executes serially.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Apply a flat train.py override to every variant. Repeatable.",
    )
    return parser.parse_args()


def _safe_variant_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not sanitized:
        raise ValueError(f"variant name {name!r} is not usable as a log filename.")
    return sanitized


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Ablation matrix YAML must contain a mapping.")
    return loaded


def load_matrix(path: Path) -> dict[str, Any]:
    matrix = _load_yaml_mapping(path)
    base = matrix.get("base", {})
    variants = matrix.get("variants", [])
    if not isinstance(base, dict):
        raise ValueError("Ablation matrix 'base' must be a mapping.")
    if not isinstance(variants, list):
        raise ValueError("Ablation matrix 'variants' must be a list.")
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"Variant at index {index} must be a mapping.")
        if "name" not in variant:
            raise ValueError(f"Variant at index {index} is missing required key 'name'.")
        _validate_variant_keys(variant, context=f"variant {variant['name']!r}")
    _validate_variant_keys(base, context="base")
    if "config" not in base:
        raise ValueError("Ablation matrix base must include config: path/to/train_config.yaml.")
    return matrix


def _validate_variant_keys(config: dict[str, Any], *, context: str) -> None:
    unknown = sorted(set(config) - set(KEY_TO_FLAG) - RUNNER_KEYS)
    if unknown:
        raise ValueError(f"Unsupported {context} key(s): {', '.join(unknown)}.")
    for key, value in config.items():
        if key in RUNNER_KEYS:
            continue
        if isinstance(value, (dict, list, tuple)):
            raise ValueError(f"{context} key {key!r} must be a scalar value.")


def _override_keys(overrides: list[str]) -> set[str]:
    keys: set[str] = set()
    for raw_override in overrides:
        key, _ = parse_override(raw_override)
        keys.add(key)
    return keys


def _selector_set(values: list[str] | None) -> set[str]:
    selected: set[str] = set()
    for value in values or []:
        selected.update(item.strip() for item in value.split(",") if item.strip())
    return selected


def _variant_labels(variant: dict[str, Any]) -> set[str]:
    labels = {str(variant["name"])}
    tags = variant.get("tags", [])
    if isinstance(tags, str):
        labels.add(tags)
    elif isinstance(tags, (list, tuple)):
        labels.update(str(tag) for tag in tags)
    elif tags:
        raise ValueError(f"variant {variant['name']!r} tags must be a string or list of strings.")
    return labels


def _normalize_mock_flag(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    if (
        normalized.get("teacher_type", "mock") != "mock"
        or normalized.get("student_type", "mock") != "mock"
        or normalized.get("dataset_type", "mock") != "mock"
    ):
        normalized["mock"] = False
    return normalized


def _with_variant_paths(
    config: dict[str, Any],
    *,
    variant: dict[str, Any],
    output_dir: Path,
    override_keys: set[str],
) -> dict[str, Any]:
    merged = dict(config)
    name = str(variant["name"])
    variant_dir = output_dir / _safe_variant_name(name)
    if "checkpoint_output_dir" not in variant and "checkpoint_output_dir" not in override_keys:
        merged["checkpoint_output_dir"] = str(variant_dir / "checkpoints")
    if "teacher_cache_dir" not in variant and "teacher_cache_dir" not in override_keys:
        merged["teacher_cache_dir"] = str(variant_dir / "teacher_cache")
    return merged


def build_variant_configs(
    matrix: dict[str, Any],
    *,
    output_dir: Path,
    overrides: list[str] | None = None,
    only: list[str] | None = None,
    skip: list[str] | None = None,
) -> list[dict[str, Any]]:
    base = dict(matrix["base"])
    variants = list(matrix["variants"])
    only_set = _selector_set(only)
    skip_set = _selector_set(skip)
    override_list = overrides or []
    parsed_override_keys = _override_keys(override_list)
    selected: list[dict[str, Any]] = []

    for variant in variants:
        labels = _variant_labels(variant)
        if only_set and not labels.intersection(only_set):
            continue
        if labels.intersection(skip_set):
            continue
        merged = {**base, **variant}
        merged = apply_overrides(merged, override_list)
        merged = _with_variant_paths(
            merged,
            variant=variant,
            output_dir=output_dir,
            override_keys=parsed_override_keys,
        )
        merged = _normalize_mock_flag(merged)
        selected.append(merged)

    known: set[str] = set()
    for variant in variants:
        known.update(_variant_labels(variant))
    if only_set:
        missing = sorted(only_set - known)
        if missing:
            raise ValueError(f"Unknown --only variant/tag(s): {', '.join(missing)}.")
    missing_skip = sorted(skip_set - known)
    if missing_skip:
        raise ValueError(f"Unknown --skip variant/tag(s): {', '.join(missing_skip)}.")
    return selected


def mamba_ssm_available() -> bool:
    return importlib.util.find_spec("mamba_ssm") is not None


def _requires_mamba(config: dict[str, Any]) -> bool:
    return bool(config.get("requires_mamba")) or config.get("student_type") == "mamba"


def _train_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in RUNNER_KEYS}


def command_for_variant(config: dict[str, Any]) -> list[str]:
    return build_command(_train_config(config))


def parse_json_lines(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def extract_final_metrics(stdout: str) -> dict[str, Any]:
    final: dict[str, Any] = {}
    for record in parse_json_lines(stdout):
        if any(key in record for key in METRIC_KEYS):
            final = record
    return {key: final[key] for key in METRIC_KEYS if key in final}


def _record(
    *,
    name: str,
    status: str,
    command: list[str],
    returncode: int | None,
    metrics: dict[str, Any] | None,
    stdout_path: Path,
    stderr_path: Path,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "returncode": returncode,
        "metrics": metrics or {},
        "command": command,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "reason": reason,
    }


def run_variant(config: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    name = str(config["name"])
    safe_name = _safe_variant_name(name)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{safe_name}.stdout"
    stderr_path = logs_dir / f"{safe_name}.stderr"
    command = command_for_variant(config)

    if config.get("skip_reason"):
        reason = str(config["skip_reason"])
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(reason + "\n", encoding="utf-8")
        return _record(
            name=name,
            status="skipped",
            command=command,
            returncode=None,
            metrics={},
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            reason=reason,
        )

    if _requires_mamba(config) and not mamba_ssm_available():
        reason = "mamba_ssm is unavailable; optional real-Mamba variant skipped."
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(reason + "\n", encoding="utf-8")
        return _record(
            name=name,
            status="skipped",
            command=command,
            returncode=None,
            metrics={},
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            reason=reason,
        )

    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return _record(
        name=name,
        status="success" if result.returncode == 0 else "failed",
        command=command,
        returncode=result.returncode,
        metrics=extract_final_metrics(result.stdout),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        reason=None if result.returncode == 0 else _first_error_line(result.stderr),
    )


def _first_error_line(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:500]
    return None


def write_summary(records: list[dict[str, Any]], *, output_dir: Path, summary_json: Path | None, summary_csv: Path | None) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    default_json = output_dir / "ablation_summary.json"
    default_csv = output_dir / "ablation_summary.csv"
    json_paths = [default_json]
    csv_paths = [default_csv]
    if summary_json is not None and summary_json != default_json:
        json_paths.append(summary_json)
    if summary_csv is not None and summary_csv != default_csv:
        csv_paths.append(summary_csv)

    for path in json_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = []
    for record in records:
        metrics = record.get("metrics") or {}
        row = {
            "name": record.get("name"),
            "status": record.get("status"),
            "returncode": record.get("returncode"),
            "reason": record.get("reason"),
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key, "")
        rows.append(row)

    for path in csv_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)

    return default_json, default_csv


def dry_run(configs: list[dict[str, Any]]) -> None:
    for config in configs:
        command = command_for_variant(config)
        print(f"{config['name']}: {shell_join(command)}", flush=True)


def run_matrix(
    matrix: dict[str, Any],
    *,
    output_dir: Path,
    overrides: list[str] | None = None,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    configs = build_variant_configs(
        matrix,
        output_dir=output_dir,
        overrides=overrides,
        only=only,
        skip=skip,
    )
    records: list[dict[str, Any]] = []
    for config in configs:
        record = run_variant(config, output_dir=output_dir)
        records.append(record)
        if record["status"] == "failed" and not continue_on_error:
            break
    return records


def main() -> None:
    args = parse_args()
    matrix_path = args.matrix or args.matrix_path or ROOT / "configs" / "ablations" / "csdm_mamba_smoke.yaml"
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive.")
    try:
        matrix = load_matrix(matrix_path)
        configs = build_variant_configs(
            matrix,
            output_dir=args.output_dir,
            overrides=args.override,
            only=args.only,
            skip=args.skip,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.max_workers != 1:
        print("Stage 8A executes variants serially; ignoring --max-workers > 1.", file=sys.stderr, flush=True)

    if args.dry_run:
        dry_run(configs)
        return

    records: list[dict[str, Any]] = []
    exit_code = 0
    for config in configs:
        record = run_variant(config, output_dir=args.output_dir)
        records.append(record)
        if record["status"] == "failed":
            exit_code = int(record["returncode"] or 1)
            if not args.continue_on_error:
                break
    default_json, default_csv = write_summary(
        records,
        output_dir=args.output_dir,
        summary_json=args.summary_json,
        summary_csv=args.summary_csv,
    )
    print(f"summary_json={default_json}", flush=True)
    print(f"summary_csv={default_csv}", flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
