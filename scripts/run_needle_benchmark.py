#!/usr/bin/env python
"""Run the synthetic Needle-in-a-Haystack benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.needle import (  # noqa: E402
    NeedleConfig,
    evaluate_predictions,
    generate_needle_examples,
    oracle_predict,
    predict_with_model,
    wrong_predict,
)


CSV_COLUMNS = (
    "example_id",
    "context_length",
    "position",
    "position_index",
    "answer",
    "prediction",
    "exact",
    "contains",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock predictors.")
    parser.add_argument("--predictor", choices=("oracle", "wrong"), default="oracle")
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--context-lengths", default="128")
    parser.add_argument("--needle-positions", default="0.1,0.5,0.9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument("--config", type=Path, default=None, help="Reserved for future model-backed generation.")
    parser.add_argument("--teacher-type", choices=("mock", "hf"), default=None)
    parser.add_argument("--student-type", choices=("mock", "mamba"), default=None)
    parser.add_argument("--tokenizer-name-or-path", default=None)
    parser.add_argument("--model-generation", action="store_true")
    return parser.parse_args(argv)


def _parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer value.")
    parsed = [int(value) for value in values]
    if any(value <= 0 for value in parsed):
        raise ValueError("context lengths must be positive.")
    return parsed


def _parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one float value.")
    parsed = [float(value) for value in values]
    if any(value < 0.0 or value > 1.0 for value in parsed):
        raise ValueError("needle positions must be in [0, 1].")
    return parsed


def _build_config(args: argparse.Namespace) -> NeedleConfig:
    return NeedleConfig(
        num_examples=args.num_examples,
        context_lengths=_parse_int_list(args.context_lengths),
        needle_positions=_parse_float_list(args.needle_positions),
        seed=args.seed,
    )


def _predict(args: argparse.Namespace, examples: list[Any]) -> list[str]:
    if args.model_generation:
        return predict_with_model(None, None, examples)
    if not args.mock:
        raise SystemExit("Stage 8D real model generation is not implemented; use --mock.")
    if args.predictor == "oracle":
        return oracle_predict(examples)
    if args.predictor == "wrong":
        return wrong_predict(examples)
    raise ValueError(f"Unsupported predictor {args.predictor!r}.")


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    config = _build_config(args)
    examples = generate_needle_examples(config)
    predictions = _predict(args, examples)
    metrics = evaluate_predictions(examples, predictions)
    return {
        "accuracy_exact": metrics["accuracy_exact"],
        "accuracy_contains": metrics["accuracy_contains"],
        "num_examples": metrics["num_examples"],
        "by_context_length": metrics["by_context_length"],
        "by_position": metrics["by_position"],
        "metadata": {
            "mode": "synthetic_needle",
            "predictor": args.predictor,
            "mock": args.mock,
            "context_lengths": config.context_lengths,
            "needle_positions": config.needle_positions,
            "seed": config.seed,
            "model_generation": args.model_generation,
        },
        "examples": metrics["examples"],
    }


def _write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> None:
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in CSV_COLUMNS} for row in payload["examples"])


def _print_summary(payload: dict[str, Any]) -> None:
    print(
        "needle_summary "
        f"exact={payload['accuracy_exact']:.4f} "
        f"contains={payload['accuracy_contains']:.4f} "
        f"num_examples={payload['num_examples']}",
        file=sys.stderr,
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = _build_payload(args)
    _write_outputs(payload, args)
    if args.print_summary:
        _print_summary(payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
