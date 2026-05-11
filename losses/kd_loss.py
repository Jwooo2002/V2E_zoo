"""Knowledge-distillation losses in logit space."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _validate_logits(student_logits: Tensor, teacher_logits: Tensor, tau: float) -> None:
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student_logits and teacher_logits must have the same shape: "
            f"{tuple(student_logits.shape)} != {tuple(teacher_logits.shape)}"
        )
    if student_logits.ndim != 3:
        raise ValueError(
            "KD logits must be rank-3 [B, T, V] or [B, N, V]; "
            f"got shape {tuple(student_logits.shape)}"
        )


def kd_kl_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    tau: float = 2.0,
    reduction: str = "mean",
) -> Tensor:
    """Compute tau-scaled KL(teacher || student) over the vocab dimension.

    Args:
        student_logits: Student logits with shape [B, T, V] or [B, N, V].
        teacher_logits: Teacher logits with the same shape. Detached internally.
        tau: Distillation temperature. Must be positive.
        reduction:
            - "none": return per-position KL with shape [B, T] or [B, N].
            - "mean": sum over vocab, then mean over non-vocab positions.
            - "sum": sum over all non-vocab positions.
            - "batchmean": sum over all positions divided by batch size.
    """

    _validate_logits(student_logits, teacher_logits, tau)
    if reduction not in {"none", "mean", "sum", "batchmean"}:
        raise ValueError(f"unsupported reduction: {reduction}")

    teacher_logits = teacher_logits.detach()
    student_log_probs = F.log_softmax(student_logits / tau, dim=-1)
    teacher_probs = F.softmax(teacher_logits / tau, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / tau, dim=-1)

    per_vocab_kl = teacher_probs * (teacher_log_probs - student_log_probs)
    per_position_kl = per_vocab_kl.sum(dim=-1) * (tau**2)

    if reduction == "none":
        return per_position_kl
    if reduction == "mean":
        return per_position_kl.mean()
    if reduction == "sum":
        return per_position_kl.sum()
    return per_position_kl.sum() / student_logits.shape[0]
