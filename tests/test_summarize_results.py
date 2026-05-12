from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_ablation_summary(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "name": "ce_only",
                    "status": "success",
                    "metrics": {"total": 3.0, "ce": 3.0, "kd": 0.2, "csdm": 0.0, "optimizer_step": 1},
                    "command": ["python", "train.py"],
                    "returncode": 0,
                },
                {
                    "name": "ce_kd",
                    "status": "success",
                    "metrics": {"total": 1.5, "ce": 3.0, "kd": 0.1, "csdm": 0.0, "optimizer_step": 1},
                    "command": ["python", "train.py"],
                    "returncode": 0,
                },
                {
                    "name": "failed_run",
                    "status": "failed",
                    "metrics": {},
                    "command": ["python", "train.py"],
                    "returncode": 1,
                },
            ]
        ),
        encoding="utf-8",
    )


def _write_single_ablation_summary(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "name": "ce_kd",
                    "status": "success",
                    "metrics": {"total": 1.5},
                    "returncode": 0,
                }
            ]
        ),
        encoding="utf-8",
    )


def _write_eval(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "perplexity": {"loss": 6.5, "perplexity": 665.0, "num_tokens": 32},
                "perturbation": {"kl_on": 0.1, "kl_off": 0.12, "delta_kl": 0.02, "num_tokens": 32},
                "needle": {"accuracy": 1.0, "num_examples": 4},
            }
        ),
        encoding="utf-8",
    )


def test_cli_writes_json_csv_markdown_and_prints_table(tmp_path: Path) -> None:
    ablation_path = tmp_path / "ablation_summary.json"
    eval_path = tmp_path / "eval.json"
    output_dir = tmp_path / "report"
    _write_ablation_summary(ablation_path)
    _write_eval(eval_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_results.py",
            "--ablation-summary",
            str(ablation_path),
            "--eval-json",
            f"ce_kd={eval_path}",
            "--output-dir",
            str(output_dir),
            "--columns",
            "name,status,total,perturbation.delta_kl",
            "--float-precision",
            "2",
            "--print-markdown",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )

    assert "| name | status | total | perturbation.delta_kl |" in result.stdout
    assert "| ce_kd | success | 1.50 | 0.02 |" in result.stdout
    assert (output_dir / "report.json").is_file()
    assert (output_dir / "report.csv").is_file()
    assert (output_dir / "report.md").is_file()
    rows = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert rows[1]["perplexity.loss"] == 6.5


def test_single_unnamed_eval_json_merges_when_one_run_exists(tmp_path: Path) -> None:
    ablation_path = tmp_path / "ablation_summary.json"
    eval_path = tmp_path / "eval.json"
    output_dir = tmp_path / "report"
    _write_single_ablation_summary(ablation_path)
    _write_eval(eval_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_results.py",
            "--ablation-summary",
            str(ablation_path),
            "--eval-json",
            str(eval_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )

    rows = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert rows == [
        {
            "name": "ce_kd",
            "status": "success",
            "returncode": 0,
            "total": 1.5,
            "perplexity.loss": 6.5,
            "perplexity.num_tokens": 32,
            "perplexity.perplexity": 665.0,
            "perturbation.delta_kl": 0.02,
            "perturbation.kl_off": 0.12,
            "perturbation.kl_on": 0.1,
            "perturbation.num_tokens": 32,
            "needle.accuracy": 1.0,
            "needle.num_examples": 4,
        }
    ]


def test_filter_status_and_sort_by_total(tmp_path: Path) -> None:
    ablation_path = tmp_path / "ablation_summary.json"
    output_dir = tmp_path / "report"
    _write_ablation_summary(ablation_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_results.py",
            "--ablation-summary",
            str(ablation_path),
            "--output-dir",
            str(output_dir),
            "--filter-status",
            "success",
            "--sort-by",
            "total",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )

    rows = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in rows] == ["ce_kd", "ce_only"]
    assert all(row["status"] == "success" for row in rows)


def test_sort_by_missing_column_warns_without_failure(tmp_path: Path) -> None:
    ablation_path = tmp_path / "ablation_summary.json"
    output_dir = tmp_path / "report"
    _write_ablation_summary(ablation_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_results.py",
            "--ablation-summary",
            str(ablation_path),
            "--output-dir",
            str(output_dir),
            "--sort-by",
            "missing.metric",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )

    assert "warning: Sort column 'missing.metric' is missing" in result.stderr


def test_malformed_json_exits_nonzero(tmp_path: Path) -> None:
    ablation_path = tmp_path / "bad.json"
    ablation_path.write_text("{not json", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_results.py",
            "--ablation-summary",
            str(ablation_path),
            "--output-dir",
            str(tmp_path / "report"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "Malformed JSON" in result.stderr or "Malformed JSON" in result.stdout


def test_csv_contains_blank_for_missing_eval_metric(tmp_path: Path) -> None:
    ablation_path = tmp_path / "ablation_summary.json"
    output_dir = tmp_path / "report"
    _write_ablation_summary(ablation_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_results.py",
            "--ablation-summary",
            str(ablation_path),
            "--output-dir",
            str(output_dir),
            "--columns",
            "name,perplexity.loss",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )

    with (output_dir / "report.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["perplexity.loss"] == ""
