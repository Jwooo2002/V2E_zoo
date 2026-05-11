"""Synthetic Needle-in-a-Haystack scaffold for mock evaluation only."""

from __future__ import annotations

import math

import torch

from train import TrainConfig


def evaluate_needle_scaffold(
    config: TrainConfig,
    max_batches: int | None = None,
) -> dict[str, float | int | str]:
    """Run a deterministic synthetic retrieval metric scaffold.

    This is intentionally a modest mock-data check. It does not evaluate real
    long-context reasoning or any real teacher/student model.
    """

    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")

    seq_len = config.mock.seq_len
    if seq_len <= 2:
        raise ValueError("needle scaffold requires seq_len greater than 2.")

    batches = max_batches if max_batches is not None else 1
    num_examples = int(config.mock.batch_size * batches)
    needle_position = max(1, min(seq_len - 2, seq_len // 2))
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    correct = 0
    for _ in range(num_examples):
        tokens = torch.randint(
            low=0,
            high=config.mock.vocab_size,
            size=(seq_len,),
            generator=generator,
            dtype=torch.long,
        )
        needle_value = int(tokens[needle_position].item())
        retrieved_value = int(tokens[needle_position].item())
        correct += int(retrieved_value == needle_value)

    accuracy = correct / float(num_examples)
    if not math.isfinite(accuracy):
        raise FloatingPointError("needle scaffold produced non-finite accuracy.")

    return {
        "accuracy": accuracy,
        "num_examples": num_examples,
        "seq_len": seq_len,
        "needle_position": needle_position,
        "mode": "synthetic_mock",
    }
