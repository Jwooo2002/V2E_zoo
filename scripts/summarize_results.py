"""Aggregate ablation and evaluation JSON results into tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.results import (  # noqa: E402
    RunRecord,
    flatten_metrics,
    load_ablation_summary,
    load_evaluation_json,
    merge_records,
    ordered_columns,
    to_markdown_table,
    write_csv,
    write_json,
    write_markdown,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-summary", type=Path, default=None)
    parser.add_argument("--eval-json", action="append", default=[], metavar="[NAME=]PATH")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/csdm_report"))
    parser.add_argument("--columns", action="append", default=[])
    parser.add_argument("--float-precision", type=int, default=6)
    parser.add_argument("--print-markdown", action="store_true")
    parser.add_argument("--sort", default=None)
    parser.add_argument("--sort-by", default=None)
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--filter-status", default=None)
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args(_normalize_sort_arg(sys.argv[1:] if argv is None else argv))


def _normalize_sort_arg(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        current = argv[index]
        if current == "--sort" and index + 1 < len(argv) and argv[index + 1].startswith("-"):
            normalized.append(f"--sort={argv[index + 1]}")
            index += 2
            continue
        normalized.append(current)
        index += 1
    return normalized


def parse_eval_json_arg(raw: str) -> tuple[str | None, Path]:
    if "=" not in raw:
        return None, Path(raw)
    name, path = raw.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"eval-json must be NAME=PATH or PATH, got {raw!r}.")
    return name.strip(), Path(path)


def parse_columns(values: list[str]) -> list[str] | None:
    columns: list[str] = []
    for value in values:
        columns.extend(item.strip() for item in value.split(",") if item.strip())
    return columns or None


def parse_filters(values: list[str], *, status: str | None = None) -> list[tuple[str, str]]:
    filters: list[tuple[str, str]] = []
    if status is not None:
        filters.append(("status", status))
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Filter must be KEY=VALUE, got {raw!r}.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Filter key must not be empty: {raw!r}.")
        filters.append((key, value))
    return filters


def apply_filters(rows: list[dict[str, Any]], filters: list[tuple[str, str]]) -> list[dict[str, Any]]:
    if not filters:
        return rows
    return [row for row in rows if all(str(row.get(key, "")) == value for key, value in filters)]


def sort_rows(rows: list[dict[str, Any]], sort_key: str | None, *, fail_on_missing: bool = False) -> list[dict[str, Any]]:
    if not sort_key:
        return rows
    descending = sort_key.startswith("-")
    key = sort_key[1:] if descending else sort_key
    if not any(key in row for row in rows):
        message = f"Sort column {key!r} is missing"
        if fail_on_missing:
            raise KeyError(message)
        print(f"warning: {message}", file=sys.stderr, flush=True)
        return rows
    return sorted(rows, key=lambda row: _sort_value(row.get(key)), reverse=descending)


def _sort_value(value: Any) -> tuple[int, Any]:
    if value is None:
        return (1, "")
    if isinstance(value, (int, float)):
        return (0, float(value))
    try:
        return (0, float(str(value)))
    except ValueError:
        return (0, str(value))


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[RunRecord] = []
    if args.ablation_summary is not None:
        records.extend(load_ablation_summary(args.ablation_summary, fail_on_missing=args.fail_on_missing))
    eval_by_name: dict[str, Any] = {}
    unnamed_eval: Any | None = None
    for raw_eval in args.eval_json:
        name, path = parse_eval_json_arg(raw_eval)
        payload = load_evaluation_json(path, fail_on_missing=args.fail_on_missing)
        if payload is None:
            continue
        if name is None:
            if len(records) == 1:
                unnamed_eval = payload
            else:
                raise ValueError(
                    "Unnamed --eval-json can be used only when the ablation summary has exactly one run."
                )
        if name is not None:
            eval_by_name[name] = payload

    if unnamed_eval is not None:
        rows = merge_records(records, unnamed_eval)
    else:
        rows = merge_records(records, eval_by_name)
    if not records and not eval_by_name and unnamed_eval is None:
        raise ValueError("Provide --ablation-summary and/or at least one --eval-json.")
    return rows


def main() -> None:
    args = parse_args()
    if args.float_precision < 0:
        raise SystemExit("--float-precision must be non-negative.")
    try:
        columns = parse_columns(args.columns)
        filters = parse_filters(args.filter, status=args.filter_status)
        sort_key = args.sort_by if args.sort_by is not None else args.sort
        if args.descending and sort_key is not None and not sort_key.startswith("-"):
            sort_key = f"-{sort_key}"
        rows = sort_rows(apply_filters(load_rows(args), filters), sort_key, fail_on_missing=args.fail_on_missing)
        if columns is None:
            columns = ordered_columns(rows)
        output_dir = args.output_dir
        json_path = output_dir / "report.json"
        csv_path = output_dir / "report.csv"
        markdown_path = output_dir / "report.md"
        write_json(rows, json_path)
        write_csv(
            rows,
            csv_path,
            columns=columns,
            float_precision=args.float_precision,
            fail_on_missing=args.fail_on_missing,
        )
        markdown = to_markdown_table(
            rows,
            columns=columns,
            float_precision=args.float_precision,
            fail_on_missing=args.fail_on_missing,
        )
        write_markdown(
            rows,
            markdown_path,
            columns=columns,
            float_precision=args.float_precision,
            fail_on_missing=args.fail_on_missing,
        )
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.print_markdown:
        print(markdown)
    print(f"report_json={json_path}", flush=True)
    print(f"report_csv={csv_path}", flush=True)
    print(f"report_md={markdown_path}", flush=True)


if __name__ == "__main__":
    main()
