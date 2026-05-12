from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from evals.needle import (
    NeedleConfig,
    contains_score,
    evaluate_predictions,
    exact_match_score,
    generate_needle_examples,
    oracle_predict,
    wrong_predict,
)


ROOT = Path(__file__).resolve().parents[1]


def test_generate_needle_examples_is_deterministic() -> None:
    config = NeedleConfig(num_examples=2, context_lengths=[16, 32], needle_positions=[0.1, 0.9], seed=7)

    first = generate_needle_examples(config)
    second = generate_needle_examples(config)

    assert first == second
    assert len(first) == 2 * 2 * 2


def test_generate_needle_examples_changes_with_seed() -> None:
    config_a = NeedleConfig(num_examples=1, context_lengths=[16], needle_positions=[0.5], seed=1)
    config_b = NeedleConfig(num_examples=1, context_lengths=[16], needle_positions=[0.5], seed=2)

    [example_a] = generate_needle_examples(config_a)
    [example_b] = generate_needle_examples(config_b)

    assert example_a.context != example_b.context
    assert example_a.answer != example_b.answer


def test_needle_context_contains_needle_and_respects_position() -> None:
    config = NeedleConfig(num_examples=1, context_lengths=[11], needle_positions=[0.8], seed=3)

    [example] = generate_needle_examples(config)

    assert example.needle in example.context
    assert example.answer in example.needle
    assert example.context.count(example.needle) == 1
    assert example.position == round(0.8 * (example.context_length - 1))
    assert 0 <= example.position < example.context_length


def test_scoring_functions() -> None:
    assert exact_match_score("  Value_123  ", "value_123") == 1.0
    assert exact_match_score("Value_123 is here", "value_123") == 0.0
    assert contains_score("The answer is VALUE_123.", "value_123") == 1.0
    assert contains_score("", "value_123") == 0.0


def test_oracle_and_wrong_predictors_score_as_expected() -> None:
    examples = generate_needle_examples(NeedleConfig(num_examples=3, context_lengths=[16], needle_positions=[0.5]))

    oracle_metrics = evaluate_predictions(examples, oracle_predict(examples))
    wrong_metrics = evaluate_predictions(examples, wrong_predict(examples))

    assert oracle_metrics["accuracy_exact"] == 1.0
    assert oracle_metrics["accuracy_contains"] == 1.0
    assert wrong_metrics["accuracy_exact"] == 0.0
    assert wrong_metrics["accuracy_contains"] == 0.0


def test_evaluate_predictions_aggregates_by_context_and_position() -> None:
    examples = generate_needle_examples(
        NeedleConfig(num_examples=2, context_lengths=[16, 32], needle_positions=[0.0, 1.0])
    )

    metrics = evaluate_predictions(examples, oracle_predict(examples))

    assert metrics["num_examples"] == 8
    assert set(metrics["by_context_length"]) == {"16", "32"}
    assert set(metrics["by_position"]) == {"0.000", "1.000"}
    assert metrics["by_context_length"]["16"]["accuracy_exact"] == 1.0
    assert metrics["by_position"]["1.000"]["num_examples"] == 4


def test_evaluate_predictions_rejects_length_mismatch() -> None:
    examples = generate_needle_examples(NeedleConfig(num_examples=1, context_lengths=[16], needle_positions=[0.5]))

    with pytest.raises(ValueError):
        evaluate_predictions(examples, [])


def _run_needle_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/run_needle_benchmark.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )


def test_needle_benchmark_cli_mock_oracle_json() -> None:
    result = _run_needle_cli(
        "--mock",
        "--predictor",
        "oracle",
        "--num-examples",
        "2",
        "--context-lengths",
        "16,32",
        "--needle-positions",
        "0.1,0.9",
    )
    payload = json.loads(result.stdout)

    assert payload["accuracy_exact"] == 1.0
    assert payload["accuracy_contains"] == 1.0
    assert payload["num_examples"] == 8
    assert payload["metadata"]["predictor"] == "oracle"


def test_needle_benchmark_cli_writes_json_and_csv(tmp_path: Path) -> None:
    output_json = tmp_path / "needle.json"
    output_csv = tmp_path / "needle.csv"

    result = _run_needle_cli(
        "--mock",
        "--predictor",
        "oracle",
        "--num-examples",
        "1",
        "--context-lengths",
        "16,32",
        "--needle-positions",
        "0.1,0.5,0.9",
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
        "--print-summary",
    )

    payload = json.loads(result.stdout)
    assert "needle_summary" in result.stderr
    assert json.loads(output_json.read_text(encoding="utf-8")) == payload
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == payload["num_examples"]
    assert {"example_id", "context_length", "position", "answer", "prediction", "exact", "contains"}.issubset(
        rows[0]
    )


def test_evaluate_py_needle_and_all_still_work() -> None:
    needle = subprocess.run(
        [
            sys.executable,
            "evaluate.py",
            "--config",
            "configs/train_config.yaml",
            "--mock",
            "--mode",
            "needle",
            "--max_batches",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    all_modes = subprocess.run(
        [
            sys.executable,
            "evaluate.py",
            "--config",
            "configs/train_config.yaml",
            "--mock",
            "--mode",
            "all",
            "--max_batches",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )

    needle_metrics = json.loads(needle.stdout)
    all_metrics = json.loads(all_modes.stdout)
    assert set(needle_metrics) == {"accuracy", "num_examples", "seq_len", "needle_position", "mode"}
    assert math.isfinite(float(needle_metrics["accuracy"]))
    assert all_metrics["needle"] == needle_metrics


def test_needle_benchmark_mock_does_not_import_real_optional_modules() -> None:
    code = """
import builtins
import importlib

forbidden = ('transformers', 'mamba_ssm')
real_import = builtins.__import__
real_import_module = importlib.import_module

def blocked(name):
    return any(name == item or name.startswith(item + '.') for item in forbidden)

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if blocked(name):
        raise AssertionError(f'{name} imported by mock needle benchmark')
    return real_import(name, globals, locals, fromlist, level)

def guarded_import_module(name, package=None):
    if blocked(name):
        raise AssertionError(f'{name} imported through importlib by mock needle benchmark')
    return real_import_module(name, package)

builtins.__import__ = guarded_import
importlib.import_module = guarded_import_module
module = importlib.import_module('scripts.run_needle_benchmark')
module.main(['--mock', '--predictor', 'oracle', '--num-examples', '1', '--context-lengths', '16', '--needle-positions', '0.5'])
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
