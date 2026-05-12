"""Result aggregation helpers for ablation and evaluation summaries."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


Scalar = str | int | float | bool | None
Row = dict[str, Any]


@dataclass
class RunRecord:
    """Normalized ablation run record before final table flattening."""

    name: str
    metrics: dict[str, Scalar] = field(default_factory=dict)
    status: str | None = None
    command: list[str] | None = None
    returncode: int | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    reason: str | None = None
    metadata: dict[str, Scalar] | None = None

    def to_row(self) -> Row:
        row: Row = {
            "name": self.name,
            "status": self.status,
            "returncode": self.returncode,
        }
        row.update(self.metrics)
        if self.reason is not None:
            row["reason"] = self.reason
        if self.command is not None:
            row["command"] = " ".join(self.command)
        if self.stdout_path is not None:
            row["stdout_path"] = self.stdout_path
        if self.stderr_path is not None:
            row["stderr_path"] = self.stderr_path
        if self.metadata:
            for key in sorted(self.metadata):
                row[f"metadata.{key}"] = self.metadata[key]
        return row


def load_json(path: str | Path, *, fail_on_missing: bool = True) -> Any:
    """Load JSON from ``path`` with optional missing-file tolerance."""

    json_path = Path(path)
    if not json_path.is_file():
        if fail_on_missing:
            raise FileNotFoundError(json_path)
        return None
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {json_path}: {exc}") from exc


def load_ablation_summary(path: str | Path, *, fail_on_missing: bool = True) -> list[RunRecord]:
    """Load a Stage 8A ablation summary JSON into ``RunRecord`` objects."""

    payload = load_json(path, fail_on_missing=fail_on_missing)
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError("Ablation summary JSON must contain a list of records.")

    records: list[RunRecord] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Ablation summary record #{index} must be a mapping.")
        records.append(
            RunRecord(
                name=str(item.get("name") or f"ablation_{index}"),
                status=None if item.get("status") is None else str(item["status"]),
                metrics=flatten_metrics(item.get("metrics") or item.get("last_metrics") or {}),
                command=list(item["command"]) if isinstance(item.get("command"), list) else None,
                returncode=None if item.get("returncode") is None else int(item["returncode"]),
                stdout_path=None if item.get("stdout_path") is None else str(item["stdout_path"]),
                stderr_path=None if item.get("stderr_path") is None else str(item["stderr_path"]),
                reason=None if item.get("reason") is None else str(item["reason"]),
                metadata=_metadata_from_item(item.get("metadata")),
            )
        )
    return records


def _metadata_from_item(value: Any) -> dict[str, Scalar] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Ablation record metadata must be a mapping when present.")
    return {str(key): _to_scalar(item) for key, item in value.items()}


def load_evaluation_json(path: str | Path, *, fail_on_missing: bool = True) -> Any:
    """Load an evaluation JSON file as emitted by ``evaluate.py``."""

    payload = load_json(path, fail_on_missing=fail_on_missing)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("Evaluation JSON must contain an object.")
    return payload


def flatten_metrics(metrics: Any, *, prefix: str = "") -> dict[str, Scalar]:
    """Flatten nested metric mappings using dot-separated keys."""

    flattened: dict[str, Scalar] = {}
    if metrics is None:
        return flattened
    if not isinstance(metrics, dict):
        if prefix:
            flattened[prefix] = _to_scalar(metrics)
        return flattened
    for key, value in metrics.items():
        full_key = str(key) if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, prefix=full_key))
        else:
            flattened[full_key] = _to_scalar(value)
    return flattened


def merge_records(records: Iterable[RunRecord | Row], eval_records: dict[str, Any] | None = None) -> list[Row]:
    """Merge ablation rows with optional evaluation metrics by run name."""

    rows = [_as_row(record) for record in records]
    if not eval_records:
        return rows

    row_names = {row["name"] for row in rows}
    eval_section_keys = {"perplexity", "perturbation", "needle"}
    looks_like_single_eval_payload = bool(eval_section_keys.intersection(eval_records))
    if len(rows) == 1 and looks_like_single_eval_payload and not any(name in row_names for name in eval_records):
        rows[0].update(flatten_metrics(eval_records))
        return rows

    by_name = {str(row["name"]): row for row in rows}
    for name, metrics in eval_records.items():
        metric_row = flatten_metrics(metrics)
        if name in by_name:
            by_name[name].update(metric_row)
        else:
            row: Row = {"name": str(name), "status": "ok", "returncode": None}
            row.update(metric_row)
            rows.append(row)
            by_name[str(name)] = row
    return rows


def ordered_columns(rows: list[Row]) -> list[str]:
    """Return deterministic table columns with common identifiers first."""

    preferred = ["name", "status", "returncode", "total", "ce", "kd", "csdm", "grad_norm", "optimizer_step"]
    available = set().union(*(row.keys() for row in rows)) if rows else set()
    columns = [column for column in preferred if column in available]
    columns.extend(sorted(key for key in available if key not in columns))
    return columns


def write_csv(
    rows: Iterable[RunRecord | Row],
    path: str | Path,
    *,
    columns: list[str] | None = None,
    float_precision: int | None = None,
    fail_on_missing: bool = False,
) -> Path:
    """Write rows to CSV with an atomic replace in the target directory."""

    flat_rows = [_as_row(row) for row in rows]
    selected_columns = columns or ordered_columns(flat_rows)
    _validate_columns(flat_rows, selected_columns, fail_on_missing=fail_on_missing)
    output_path = Path(path)

    def writer(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=selected_columns)
            csv_writer.writeheader()
            for row in flat_rows:
                csv_writer.writerow(
                    {column: _format_value(row.get(column), None) for column in selected_columns}
                )

    _atomic_write(output_path, writer)
    return output_path


def write_json(rows: Iterable[RunRecord | Row], path: str | Path) -> Path:
    """Write flat rows as JSON with an atomic replace."""

    flat_rows = [_as_row(row) for row in rows]
    output_path = Path(path)

    def writer(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(flat_rows, handle, indent=2, sort_keys=True)
            handle.write("\n")

    _atomic_write(output_path, writer)
    return output_path


def to_markdown_table(
    rows: Iterable[RunRecord | Row],
    *,
    columns: list[str] | None = None,
    float_precision: int = 6,
    fail_on_missing: bool = False,
) -> str:
    """Render rows as a compact GitHub-flavored Markdown table."""

    flat_rows = [_as_row(row) for row in rows]
    selected_columns = columns or ordered_columns(flat_rows)
    _validate_columns(flat_rows, selected_columns, fail_on_missing=fail_on_missing)
    if not selected_columns:
        return ""
    lines = [
        "| " + " | ".join(selected_columns) + " |",
        "| " + " | ".join("---" for _ in selected_columns) + " |",
    ]
    for row in flat_rows:
        values = [_escape_markdown(_format_value(row.get(column), float_precision)) for column in selected_columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(
    rows: Iterable[RunRecord | Row],
    path: str | Path,
    *,
    columns: list[str] | None = None,
    float_precision: int = 6,
    fail_on_missing: bool = False,
) -> Path:
    """Write a Markdown table with an atomic replace."""

    table = to_markdown_table(
        rows,
        columns=columns,
        float_precision=float_precision,
        fail_on_missing=fail_on_missing,
    )
    output_path = Path(path)

    def writer(tmp_path: Path) -> None:
        tmp_path.write_text(table + ("\n" if table else ""), encoding="utf-8")

    _atomic_write(output_path, writer)
    return output_path


def _as_row(record: RunRecord | Row) -> Row:
    if isinstance(record, RunRecord):
        return record.to_row()
    return dict(record)


def _to_scalar(value: Any) -> Scalar:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True) if isinstance(value, list) else str(value)


def _format_value(value: Any, float_precision: int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and float_precision is not None:
        return f"{value:.{float_precision}f}"
    return str(value)


def _validate_columns(rows: list[Row], columns: list[str], *, fail_on_missing: bool) -> None:
    if not fail_on_missing:
        return
    available = set().union(*(row.keys() for row in rows)) if rows else set()
    missing = [column for column in columns if column not in available]
    if missing:
        raise KeyError(f"Requested column(s) missing from all records: {', '.join(missing)}")


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _atomic_write(path: Path, write_fn: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_fn(tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
