"""Perplexity evaluation for the mock CSDM Mamba KD scaffold."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from train import TrainConfig, _select_shared_valid_mask


def _iter_limited_batches(
    dataloader: DataLoader[dict[str, Tensor]],
    max_batches: int | None,
) -> Any:
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")
    for index, batch in enumerate(dataloader):
        if max_batches is not None and index >= max_batches:
            break
        yield batch


def _masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, T, V], got {tuple(logits.shape)}.")
    if mask.shape != logits.shape[:2]:
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match logits {tuple(logits.shape[:2])}.")
    selected = logits[mask]
    if selected.numel() == 0:
        raise ValueError("valid-position mask selected no logits.")
    return selected


def evaluate_perplexity(
    student: nn.Module,
    dataloader: DataLoader[dict[str, Tensor]],
    config: TrainConfig,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    """Evaluate next-token perplexity with token-weighted aggregation."""

    student.eval()
    ce_sum = 0.0
    num_tokens = 0

    with torch.no_grad():
        for batch in _iter_limited_batches(dataloader, max_batches):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            output = student(input_ids)
            mask = _select_shared_valid_mask(
                labels=labels,
                ignore_index=config.mock.ignore_index,
                positions_per_sequence=config.mock.positions_per_sequence,
            )

            logits = _masked_logits(output.on_logits, mask).float()
            targets = labels[mask]
            ce = F.cross_entropy(
                logits,
                targets,
                ignore_index=config.mock.ignore_index,
                reduction="sum",
            )
            ce_sum += float(ce.detach().cpu())
            num_tokens += int(mask.sum().item())

    if num_tokens <= 0:
        raise ValueError("perplexity evaluation found no valid tokens.")

    loss = ce_sum / float(num_tokens)
    perplexity = math.exp(loss)
    if not math.isfinite(loss) or not math.isfinite(perplexity):
        raise FloatingPointError("perplexity evaluation produced a non-finite metric.")

    return {"loss": loss, "perplexity": perplexity, "num_tokens": num_tokens}
