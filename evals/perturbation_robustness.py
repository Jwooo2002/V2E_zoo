"""Perturbation robustness KL evaluation for mock CSDM Mamba KD."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from losses.kd_loss import kd_kl_loss
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
    return selected.reshape(1, selected.shape[0], selected.shape[1])


def evaluate_perturbation_robustness(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader[dict[str, Tensor]],
    config: TrainConfig,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    """Compare teacher||student KL on clean and off-trajectory student states."""

    student.eval()
    teacher.eval()
    kl_on_sum = 0.0
    kl_off_sum = 0.0
    num_tokens = 0

    with torch.no_grad():
        for batch in _iter_limited_batches(dataloader, max_batches):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            teacher_logits = teacher(input_ids)
            output = student(input_ids)
            mask = _select_shared_valid_mask(
                labels=labels,
                ignore_index=config.mock.ignore_index,
                positions_per_sequence=config.mock.positions_per_sequence,
            )

            teacher_masked = _masked_logits(teacher_logits, mask).float()
            on_masked = _masked_logits(output.on_logits, mask).float()
            off_masked = _masked_logits(output.off_logits, mask).float()
            kl_on = kd_kl_loss(
                on_masked,
                teacher_masked,
                tau=config.loss.tau,
                reduction="none",
            )
            kl_off = kd_kl_loss(
                off_masked,
                teacher_masked,
                tau=config.loss.tau,
                reduction="none",
            )
            kl_on_sum += float(kl_on.sum().detach().cpu())
            kl_off_sum += float(kl_off.sum().detach().cpu())
            num_tokens += int(mask.sum().item())

    if num_tokens <= 0:
        raise ValueError("perturbation robustness evaluation found no valid tokens.")

    kl_on_value = kl_on_sum / float(num_tokens)
    kl_off_value = kl_off_sum / float(num_tokens)
    delta_kl = kl_off_value - kl_on_value
    for name, value in {
        "kl_on": kl_on_value,
        "kl_off": kl_off_value,
        "delta_kl": delta_kl,
    }.items():
        if not math.isfinite(value):
            raise FloatingPointError(f"perturbation robustness produced non-finite {name}.")

    return {
        "kl_on": kl_on_value,
        "kl_off": kl_off_value,
        "delta_kl": delta_kl,
        "num_tokens": num_tokens,
    }
