"""Knowledge-distillation losses in logit space."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def gather_logits_by_indices(logits: Tensor, indices: Tensor) -> Tensor:
    """Gather rank-3 logits by per-position vocab indices.

    Args:
        logits: Tensor shaped [B, T, V] or [B, N, V].
        indices: Long tensor shaped [B, T, K] or [B, N, K].

    Returns:
        Gathered logits shaped like ``indices``. Gradients flow to ``logits``.
    """

    if logits.ndim != 3:
        raise ValueError(f"logits must be rank-3 [B, T, V] or [B, N, V], got {tuple(logits.shape)}.")
    if indices.ndim != 3:
        raise ValueError(f"indices must be rank-3 [B, T, K] or [B, N, K], got {tuple(indices.shape)}.")
    if indices.dtype != torch.long:
        raise TypeError(f"indices must have dtype torch.long, got {indices.dtype}.")
    if logits.shape[:2] != indices.shape[:2]:
        raise ValueError(
            "logits and indices must share [B, T/N] prefix shape: "
            f"{tuple(logits.shape[:2])} != {tuple(indices.shape[:2])}."
        )
    if indices.shape[-1] <= 0:
        raise ValueError("indices must select at least one vocab entry.")
    vocab_size = logits.shape[-1]
    if vocab_size <= 0:
        raise ValueError("logits vocab dimension must be non-empty.")
    if bool(indices.numel()):
        min_index = int(indices.min().item())
        max_index = int(indices.max().item())
        if min_index < 0 or max_index >= vocab_size:
            raise IndexError(f"indices must be in [0, {vocab_size}), got min={min_index}, max={max_index}.")
    return logits.gather(dim=-1, index=indices)


def build_topk_indices(
    teacher_logits: Tensor,
    labels: Tensor | None = None,
    top_k: int = 256,
    include_labels: bool = True,
) -> Tensor:
    """Build selected-vocab indices from detached raw teacher logits.

    The base selection is ``top_k`` over the teacher vocab dimension. If
    ``include_labels`` is true and labels are provided, one extra label index is
    appended per position. Labels below zero, such as ``-100``, are ignored by
    duplicating the first teacher top-k index for that position. Duplicates are
    intentionally allowed to keep the utility simple and shape-stable.
    """

    if teacher_logits.ndim != 3:
        raise ValueError(
            "teacher_logits must be rank-3 [B, T, V] or [B, N, V], "
            f"got {tuple(teacher_logits.shape)}."
        )
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    vocab_size = teacher_logits.shape[-1]
    if top_k > vocab_size:
        raise ValueError(f"top_k must be <= vocab size {vocab_size}, got {top_k}.")

    indices = teacher_logits.detach().topk(top_k, dim=-1).indices
    if not include_labels or labels is None:
        return indices
    if labels.shape != teacher_logits.shape[:2]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match teacher prefix shape {tuple(teacher_logits.shape[:2])}."
        )
    if labels.dtype != torch.long:
        raise TypeError(f"labels must have dtype torch.long, got {labels.dtype}.")

    valid_labels = labels.ge(0)
    if bool(valid_labels.any()):
        valid_values = labels[valid_labels]
        min_label = int(valid_values.min().item())
        max_label = int(valid_values.max().item())
        if min_label < 0 or max_label >= vocab_size:
            raise IndexError(f"valid labels must be in [0, {vocab_size}), got min={min_label}, max={max_label}.")
    label_indices = torch.where(valid_labels, labels, indices[..., 0]).unsqueeze(-1)
    return torch.cat([indices, label_indices], dim=-1)


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
    topk_indices: Tensor | None = None,
    renormalize_topk: bool = True,
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
        topk_indices: Optional selected-vocab indices from raw teacher logits.
            When provided, the same indices are gathered for teacher and
            student at each position.
        renormalize_topk: If true, compute the selected-vocab approximation by
            renormalizing over selected K. If false, gather full-vocab
            probabilities/log-probabilities and sum only the selected terms.
    """

    _validate_logits(student_logits, teacher_logits, tau)
    if reduction not in {"none", "mean", "sum", "batchmean"}:
        raise ValueError(f"unsupported reduction: {reduction}")

    teacher_logits = teacher_logits.detach()
    if topk_indices is not None:
        if renormalize_topk:
            student_logits = gather_logits_by_indices(student_logits, topk_indices)
            teacher_logits = gather_logits_by_indices(teacher_logits, topk_indices)
            student_log_probs = F.log_softmax(student_logits / tau, dim=-1)
            teacher_probs = F.softmax(teacher_logits / tau, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits / tau, dim=-1)
        else:
            student_log_probs = gather_logits_by_indices(F.log_softmax(student_logits / tau, dim=-1), topk_indices)
            teacher_log_probs_full = F.log_softmax(teacher_logits / tau, dim=-1)
            teacher_log_probs = gather_logits_by_indices(teacher_log_probs_full, topk_indices)
            teacher_probs = gather_logits_by_indices(teacher_log_probs_full.exp(), topk_indices)
    else:
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
