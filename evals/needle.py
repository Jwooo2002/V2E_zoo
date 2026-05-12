"""Synthetic Needle-in-a-Haystack benchmark utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
import random
from typing import Any

from train import TrainConfig


@dataclass(frozen=True)
class NeedleExample:
    context: str
    question: str
    answer: str
    needle: str
    position: int
    context_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NeedleConfig:
    num_examples: int = 8
    context_lengths: list[int] = field(default_factory=lambda: [128])
    needle_positions: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    seed: int = 42
    key_prefix: str = "key"
    value_prefix: str = "value"
    filler_token: str = "hay"
    template: str = "The secret {key} is {value}."
    question_template: str = "What is the value for {key}?"

    def __post_init__(self) -> None:
        if self.num_examples <= 0:
            raise ValueError("num_examples must be positive.")
        if not self.context_lengths:
            raise ValueError("context_lengths must not be empty.")
        if not self.needle_positions:
            raise ValueError("needle_positions must not be empty.")
        for context_length in self.context_lengths:
            if int(context_length) <= 0:
                raise ValueError(f"context lengths must be positive, got {context_length}.")
        for position in self.needle_positions:
            if float(position) < 0.0 or float(position) > 1.0:
                raise ValueError(f"needle positions must be in [0, 1], got {position}.")


def generate_needle_examples(config: NeedleConfig) -> list[NeedleExample]:
    """Generate deterministic synthetic key-value retrieval examples.

    ``num_examples`` is interpreted per context-length/position pair. A config
    with two context lengths, three positions, and four examples yields
    ``2 * 3 * 4`` examples.
    """

    rng = random.Random(config.seed)
    examples: list[NeedleExample] = []
    example_id = 0
    for context_length in config.context_lengths:
        slot_count = int(context_length)
        for requested_position in config.needle_positions:
            insert_index = _position_to_index(float(requested_position), slot_count)
            for local_index in range(config.num_examples):
                key = _unique_token(config.key_prefix, rng, example_id)
                value = _unique_token(config.value_prefix, rng, example_id)
                needle = config.template.format(key=key, value=value)
                question = config.question_template.format(key=key, value=value)
                filler = _filler_tokens(config.filler_token, rng, example_id, slot_count)
                filler[insert_index] = needle
                context = " ".join(filler)
                examples.append(
                    NeedleExample(
                        context=context,
                        question=question,
                        answer=value,
                        needle=needle,
                        position=insert_index,
                        context_length=slot_count,
                        metadata={
                            "example_id": example_id,
                            "local_index": local_index,
                            "requested_position": float(requested_position),
                            "position_fraction": insert_index / float(max(1, slot_count - 1)),
                            "key": key,
                            "value": value,
                        },
                    )
                )
                example_id += 1
    return examples


def exact_match_score(prediction: str, answer: str) -> float:
    """Case-insensitive exact match after trimming whitespace."""

    return 1.0 if prediction.strip().lower() == answer.strip().lower() else 0.0


def contains_score(prediction: str, answer: str) -> float:
    """Case-insensitive substring score."""

    return 1.0 if answer.strip().lower() in prediction.strip().lower() else 0.0


def oracle_predict(examples: Sequence[NeedleExample]) -> list[str]:
    """Return the ground-truth answer for pipeline validation."""

    return [example.answer for example in examples]


def wrong_predict(examples: Sequence[NeedleExample]) -> list[str]:
    """Return deterministic incorrect predictions for negative tests."""

    return [f"wrong_answer_{int(example.metadata.get('example_id', index)):04d}" for index, example in enumerate(examples)]


def predict_with_model(
    model: Any,
    tokenizer: Any,
    examples: Sequence[NeedleExample],
    **_: Any,
) -> list[str]:
    del model, tokenizer, examples
    raise NotImplementedError(
        "Real model generation for the Needle benchmark is not implemented in Stage 8D. "
        "Use --mock --predictor oracle/wrong for pipeline validation."
    )


def evaluate_predictions(
    examples: Sequence[NeedleExample],
    predictions: Sequence[str],
) -> dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError(
            "examples and predictions must have the same length, "
            f"got {len(examples)} and {len(predictions)}."
        )
    if not examples:
        raise ValueError("at least one Needle example is required.")

    rows: list[dict[str, Any]] = []
    by_context: dict[str, dict[str, float | int]] = {}
    by_position: dict[str, dict[str, float | int]] = {}
    exact_sum = 0.0
    contains_sum = 0.0
    for example, prediction in zip(examples, predictions, strict=True):
        exact = exact_match_score(prediction, example.answer)
        contains = contains_score(prediction, example.answer)
        exact_sum += exact
        contains_sum += contains
        context_key = str(example.context_length)
        position_key = _format_position_key(float(example.metadata.get("requested_position", 0.0)))
        _accumulate_group(by_context, context_key, exact, contains)
        _accumulate_group(by_position, position_key, exact, contains)
        rows.append(
            {
                "example_id": int(example.metadata.get("example_id", len(rows))),
                "context_length": example.context_length,
                "position": float(example.metadata.get("requested_position", 0.0)),
                "position_index": example.position,
                "answer": example.answer,
                "prediction": prediction,
                "exact": exact,
                "contains": contains,
            }
        )

    num_examples = len(examples)
    return {
        "accuracy_exact": _finite_score(exact_sum / float(num_examples), "accuracy_exact"),
        "accuracy_contains": _finite_score(contains_sum / float(num_examples), "accuracy_contains"),
        "num_examples": num_examples,
        "by_context_length": _finalize_groups(by_context),
        "by_position": _finalize_groups(by_position),
        "examples": rows,
    }


def evaluate_needle_scaffold(
    config: TrainConfig,
    max_batches: int | None = None,
) -> dict[str, float | int | str]:
    """Compatibility wrapper for ``evaluate.py --mock --mode needle``.

    This uses the deterministic oracle predictor to validate benchmark
    plumbing. It is not evidence of real long-context retrieval capability.
    """

    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")

    seq_len = config.mock.seq_len
    if seq_len <= 2:
        raise ValueError("needle scaffold requires seq_len greater than 2.")

    batches = max_batches if max_batches is not None else 1
    num_examples = int(config.mock.batch_size * batches)
    examples = generate_needle_examples(
        NeedleConfig(
            num_examples=num_examples,
            context_lengths=[seq_len],
            needle_positions=[0.5],
            seed=config.seed,
        )
    )
    metrics = evaluate_predictions(examples, oracle_predict(examples))
    needle_position = examples[0].position
    accuracy = float(metrics["accuracy_exact"])
    if not math.isfinite(accuracy):
        raise FloatingPointError("needle scaffold produced non-finite accuracy.")

    return {
        "accuracy": accuracy,
        "num_examples": num_examples,
        "seq_len": seq_len,
        "needle_position": needle_position,
        "mode": "synthetic_mock",
    }


def _position_to_index(position_fraction: float, slot_count: int) -> int:
    if slot_count <= 1:
        return 0
    return max(0, min(slot_count - 1, int(round(position_fraction * (slot_count - 1)))))


def _unique_token(prefix: str, rng: random.Random, example_id: int) -> str:
    return f"{prefix}_{example_id:04d}_{rng.randrange(1_000_000):06d}"


def _filler_tokens(prefix: str, rng: random.Random, example_id: int, slot_count: int) -> list[str]:
    return [f"{prefix}_{example_id:04d}_{index:04d}_{rng.randrange(10_000):04d}" for index in range(slot_count)]


def _format_position_key(position: float) -> str:
    return f"{position:.3f}"


def _accumulate_group(
    groups: dict[str, dict[str, float | int]],
    key: str,
    exact: float,
    contains: float,
) -> None:
    entry = groups.setdefault(key, {"exact_sum": 0.0, "contains_sum": 0.0, "num_examples": 0})
    entry["exact_sum"] = float(entry["exact_sum"]) + exact
    entry["contains_sum"] = float(entry["contains_sum"]) + contains
    entry["num_examples"] = int(entry["num_examples"]) + 1


def _finalize_groups(groups: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
    finalized: dict[str, dict[str, float | int]] = {}
    for key in sorted(groups):
        entry = groups[key]
        count = int(entry["num_examples"])
        finalized[key] = {
            "accuracy_exact": _finite_score(float(entry["exact_sum"]) / float(count), f"{key}.accuracy_exact"),
            "accuracy_contains": _finite_score(
                float(entry["contains_sum"]) / float(count),
                f"{key}.accuracy_contains",
            ),
            "num_examples": count,
        }
    return finalized


def _finite_score(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise FloatingPointError(f"needle benchmark produced non-finite {name}.")
    return value
