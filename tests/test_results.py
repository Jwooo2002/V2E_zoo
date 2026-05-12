from __future__ import annotations

import csv
import json
from pathlib import Path

from utils.results import (
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


def test_flatten_metrics_flattens_nested_dicts() -> None:
    metrics = {"perplexity": {"loss": 1.25, "perplexity": 3.5}, "total": 2.0}

    assert flatten_metrics(metrics) == {
        "perplexity.loss": 1.25,
        "perplexity.perplexity": 3.5,
        "total": 2.0,
    }


def test_flatten_metrics_preserves_dual_perturbation_names() -> None:
    metrics = {
        "full_vocab": {"delta_kl": 0.12},
        "topk": {"delta_kl": 0.03},
        "by_mode": {
            "delta_projection": {
                "full_vocab": {"kl_on": 0.1},
                "topk": {"kl_on": 0.04},
            }
        },
    }

    flattened = flatten_metrics(metrics)

    assert flattened["full_vocab.delta_kl"] == 0.12
    assert flattened["topk.delta_kl"] == 0.03
    assert flattened["by_mode.delta_projection.full_vocab.kl_on"] == 0.1
    assert flattened["by_mode.delta_projection.topk.kl_on"] == 0.04


def test_load_ablation_summary_parses_stage8a_records(tmp_path: Path) -> None:
    path = tmp_path / "ablation_summary.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "ce_kd",
                    "status": "success",
                    "metrics": {"total": 1.2, "kd": 0.3},
                    "command": ["python", "train.py"],
                    "returncode": 0,
                    "stdout_path": "/tmp/stdout",
                    "stderr_path": "/tmp/stderr",
                }
            ]
        ),
        encoding="utf-8",
    )

    records = load_ablation_summary(path)

    assert records == [
        RunRecord(
            name="ce_kd",
            status="success",
            metrics={"total": 1.2, "kd": 0.3},
            command=["python", "train.py"],
            returncode=0,
            stdout_path="/tmp/stdout",
            stderr_path="/tmp/stderr",
        )
    ]


def test_load_evaluation_json_parses_evaluate_output(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    payload = {
        "perplexity": {"loss": 6.0, "perplexity": 403.0},
        "perturbation": {"kl_on": 0.1, "kl_off": 0.2, "delta_kl": 0.1},
        "needle": {"accuracy": 1.0},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_evaluation_json(path) == payload


def test_merge_records_combines_ablation_and_eval_metrics_by_name() -> None:
    records = [RunRecord(name="ce_kd", status="success", metrics={"total": 1.2})]
    eval_records = {"ce_kd": {"perturbation": {"delta_kl": 0.01}, "needle": {"accuracy": 1.0}}}

    rows = merge_records(records, eval_records)

    assert rows == [
        {
            "name": "ce_kd",
            "status": "success",
            "returncode": None,
            "total": 1.2,
            "perturbation.delta_kl": 0.01,
            "needle.accuracy": 1.0,
        }
    ]


def test_write_csv_writes_expected_headers_and_blank_missing_cells(tmp_path: Path) -> None:
    rows = [
        {"name": "a", "status": "success", "total": 1.0},
        {"name": "b", "status": "failed"},
    ]
    path = tmp_path / "report.csv"

    write_csv(rows, path)

    with path.open("r", encoding="utf-8", newline="") as handle:
        parsed = list(csv.DictReader(handle))
    assert parsed[0]["name"] == "a"
    assert parsed[0]["total"] == "1.0"
    assert parsed[1]["total"] == ""


def test_json_and_markdown_writers(tmp_path: Path) -> None:
    rows = [{"name": "ce_kd", "status": "success", "total": 1.23456}]
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    write_json(rows, json_path)
    write_markdown(rows, markdown_path, columns=["name", "total"], float_precision=2)

    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["total"] == 1.23456
    assert markdown_path.read_text(encoding="utf-8") == (
        "| name | total |\n"
        "| --- | --- |\n"
        "| ce_kd | 1.23 |\n"
    )


def test_to_markdown_table_formats_floats_and_missing_metrics() -> None:
    rows = [
        {"name": "ce_kd", "status": "success", "total": 1.23456},
        {"name": "ce_only", "status": "success"},
    ]

    table = to_markdown_table(rows, columns=["name", "total"], float_precision=3)

    assert "| ce_kd | 1.235 |" in table
    assert "| ce_only |  |" in table


def test_ordered_columns_are_deterministic() -> None:
    rows = [{"status": "success", "name": "a", "z_extra": 1, "a_extra": 2, "total": 3}]

    assert ordered_columns(rows) == ["name", "status", "total", "a_extra", "z_extra"]
