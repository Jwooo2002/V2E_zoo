"""Perturbation robustness KL metrics for CSDM-Mamba evaluation."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from losses.kd_loss import build_topk_indices, gather_logits_by_indices
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


def _validate_logits_pair(teacher_logits: Tensor, student_logits: Tensor) -> None:
    if teacher_logits.ndim != 3 or student_logits.ndim != 3:
        raise ValueError(
            "teacher_logits and student_logits must be rank-3 [B, T, V], "
            f"got {tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}."
        )
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "teacher_logits and student_logits must have the same shape, "
            f"got {tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}."
        )


def _validate_mask(mask: Tensor | None, shape: torch.Size) -> Tensor | None:
    if mask is None:
        return None
    if mask.shape != shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} must match [B, T] shape {tuple(shape)}.")
    return mask.to(dtype=torch.bool)


def _finite_metric(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise FloatingPointError(f"perturbation robustness produced non-finite {name}.")
    return value


def kl_teacher_student(
    teacher_logits: Tensor,
    student_logits: Tensor,
    mask: Tensor | None = None,
    tau: float = 1.0,
    reduction: str = "mean",
    topk_indices: Tensor | None = None,
    renormalize_topk: bool = True,
) -> Tensor:
    """Compute ``KL(p_teacher || p_student)`` for evaluation.

    This helper is intentionally separate from the training KD loss because
    reporting uses the literal KL value, not the distillation loss scaled by
    ``tau**2``. Teacher logits are detached internally. Optional top-k indices
    must be built from clean teacher logits and are applied identically to the
    teacher and student tensors.
    """

    if tau <= 0:
        raise ValueError("tau must be positive.")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be one of: mean, sum, none.")
    _validate_logits_pair(teacher_logits, student_logits)
    valid_mask = _validate_mask(mask, teacher_logits.shape[:2])

    teacher = teacher_logits.detach().float() / tau
    student = student_logits.float() / tau
    if topk_indices is not None:
        if topk_indices.shape[:2] != teacher.shape[:2]:
            raise ValueError(
                "topk_indices must share [B, T] shape with logits, "
                f"got {tuple(topk_indices.shape[:2])} and {tuple(teacher.shape[:2])}."
            )
        if renormalize_topk:
            teacher_selected = gather_logits_by_indices(teacher, topk_indices)
            student_selected = gather_logits_by_indices(student, topk_indices)
            teacher_log_probs = F.log_softmax(teacher_selected, dim=-1)
            teacher_probs = teacher_log_probs.exp()
            student_log_probs = F.log_softmax(student_selected, dim=-1)
        else:
            teacher_log_probs_full = F.log_softmax(teacher, dim=-1)
            student_log_probs_full = F.log_softmax(student, dim=-1)
            teacher_log_probs = gather_logits_by_indices(teacher_log_probs_full, topk_indices)
            teacher_probs = teacher_log_probs.exp()
            student_log_probs = gather_logits_by_indices(student_log_probs_full, topk_indices)
    else:
        teacher_log_probs = F.log_softmax(teacher, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_log_probs = F.log_softmax(student, dim=-1)

    per_position = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    if valid_mask is not None:
        per_position = torch.where(valid_mask, per_position, torch.zeros_like(per_position))

    if reduction == "none":
        return per_position
    if reduction == "sum":
        return per_position.sum()

    if valid_mask is None:
        return per_position.mean()
    count = valid_mask.sum()
    if int(count.item()) <= 0:
        raise ValueError("mask selected no valid tokens.")
    return per_position.sum() / count.to(per_position.dtype)


def aggregate_position_metrics(
    kl_on: Tensor,
    kl_off: Tensor,
    mask: Tensor | None = None,
) -> dict[str, list[float] | list[int]]:
    """Average on/off/delta KL by sequence position."""

    if kl_on.shape != kl_off.shape or kl_on.ndim != 2:
        raise ValueError(
            "kl_on and kl_off must have matching shape [B, T], "
            f"got {tuple(kl_on.shape)} and {tuple(kl_off.shape)}."
        )
    valid_mask = _validate_mask(mask, kl_on.shape)
    if valid_mask is None:
        valid_mask = torch.ones_like(kl_on, dtype=torch.bool)

    position_kl_on: list[float] = []
    position_kl_off: list[float] = []
    position_delta: list[float] = []
    position_counts: list[int] = []
    for position in range(kl_on.shape[1]):
        position_mask = valid_mask[:, position]
        count = int(position_mask.sum().item())
        if not bool(position_mask.any()):
            on_value = 0.0
            off_value = 0.0
        else:
            on_value = float(kl_on[:, position][position_mask].detach().float().mean().cpu())
            off_value = float(kl_off[:, position][position_mask].detach().float().mean().cpu())
        position_kl_on.append(_finite_metric("position_kl_on", on_value))
        position_kl_off.append(_finite_metric("position_kl_off", off_value))
        position_delta.append(_finite_metric("position_delta_kl", off_value - on_value))
        position_counts.append(count)
    return {
        "position_kl_on": position_kl_on,
        "position_kl_off": position_kl_off,
        "position_delta_kl": position_delta,
        "position_num_tokens": position_counts,
    }


def compute_perturbation_metrics(
    teacher_logits: Tensor,
    on_logits: Tensor,
    off_logits: Tensor,
    mask: Tensor | None = None,
    tau: float = 1.0,
    topk_indices: Tensor | None = None,
    renormalize_topk: bool = True,
    position_wise: bool = False,
    labels: Tensor | None = None,
    top_k: int | None = None,
    include_labels: bool = True,
    include_position_metrics: bool | None = None,
) -> dict[str, float | int | list[float] | list[int] | list[dict[str, float | int]]]:
    """Compute token-weighted perturbation robustness metrics.

    ``labels``/``top_k`` are kept for compatibility with the Stage 4 evaluator;
    new callers should pass ``topk_indices`` explicitly after building them
    from clean teacher logits.
    """

    _validate_logits_pair(teacher_logits, on_logits)
    _validate_logits_pair(teacher_logits, off_logits)
    valid_mask = _validate_mask(mask, teacher_logits.shape[:2])
    if include_position_metrics is not None:
        position_wise = include_position_metrics

    if topk_indices is None and top_k is not None:
        topk_indices = build_topk_indices(
            teacher_logits.detach().float(),
            labels=labels,
            top_k=top_k,
            include_labels=include_labels,
        )

    kl_on_positions = kl_teacher_student(
        teacher_logits,
        on_logits,
        mask=valid_mask,
        tau=tau,
        reduction="none",
        topk_indices=topk_indices,
        renormalize_topk=renormalize_topk,
    )
    kl_off_positions = kl_teacher_student(
        teacher_logits,
        off_logits,
        mask=valid_mask,
        tau=tau,
        reduction="none",
        topk_indices=topk_indices,
        renormalize_topk=renormalize_topk,
    )

    if valid_mask is None:
        num_tokens = int(teacher_logits.shape[0] * teacher_logits.shape[1])
        kl_on_value = float(kl_on_positions.detach().float().mean().cpu())
        kl_off_value = float(kl_off_positions.detach().float().mean().cpu())
    else:
        num_tokens = int(valid_mask.sum().item())
        if num_tokens <= 0:
            raise ValueError("perturbation robustness evaluation found no valid tokens.")
        kl_on_value = float(kl_on_positions[valid_mask].detach().float().mean().cpu())
        kl_off_value = float(kl_off_positions[valid_mask].detach().float().mean().cpu())

    metrics: dict[str, float | int | list[float] | list[int] | list[dict[str, float | int]]] = {
        "kl_on": _finite_metric("kl_on", kl_on_value),
        "kl_off": _finite_metric("kl_off", kl_off_value),
        "delta_kl": _finite_metric("delta_kl", kl_off_value - kl_on_value),
        "num_tokens": num_tokens,
    }
    if position_wise:
        position_metrics = aggregate_position_metrics(kl_on_positions, kl_off_positions, valid_mask)
        metrics.update(
            {
                "position_kl_on": position_metrics["position_kl_on"],
                "position_kl_off": position_metrics["position_kl_off"],
                "position_delta_kl": position_metrics["position_delta_kl"],
                "position_num_tokens": position_metrics["position_num_tokens"],
                "positions": [
                    {
                        "position": index,
                        "kl_on": position_metrics["position_kl_on"][index],
                        "kl_off": position_metrics["position_kl_off"][index],
                        "delta_kl": position_metrics["position_delta_kl"][index],
                        "num_tokens": position_metrics["position_num_tokens"][index],
                    }
                    for index in range(len(position_metrics["position_delta_kl"]))
                ],
            }
        )
    return metrics


def compute_dual_perturbation_metrics(
    teacher_logits: Tensor,
    on_logits: Tensor,
    off_logits: Tensor,
    labels: Tensor | None = None,
    mask: Tensor | None = None,
    tau: float = 1.0,
    top_k: int = 128,
    include_labels: bool = True,
    renormalize_topk: bool = True,
    position_wise: bool = False,
    include_position_metrics: bool | None = None,
) -> dict[str, dict[str, float | int | bool | list[float] | list[int] | list[dict[str, float | int]]]]:
    """Report perturbation KL on both full and selected teacher vocabularies.

    ``full_vocab`` is the literal full-distribution ``KL(teacher || student)``.
    ``topk`` is a selected-vocabulary diagnostic using indices built from the
    detached clean-prefix teacher logits, optionally plus valid gold labels.
    Both sections use the same mask and token weighting.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    _validate_logits_pair(teacher_logits, on_logits)
    _validate_logits_pair(teacher_logits, off_logits)
    valid_mask = _validate_mask(mask, teacher_logits.shape[:2])
    if include_position_metrics is not None:
        position_wise = include_position_metrics

    full_vocab = compute_perturbation_metrics(
        teacher_logits=teacher_logits,
        on_logits=on_logits,
        off_logits=off_logits,
        mask=valid_mask,
        tau=tau,
        topk_indices=None,
        renormalize_topk=renormalize_topk,
        position_wise=position_wise,
    )
    topk_indices = build_topk_indices(
        teacher_logits.detach().float(),
        labels=labels,
        top_k=top_k,
        include_labels=include_labels,
    )
    topk = compute_perturbation_metrics(
        teacher_logits=teacher_logits,
        on_logits=on_logits,
        off_logits=off_logits,
        mask=valid_mask,
        tau=tau,
        topk_indices=topk_indices,
        renormalize_topk=renormalize_topk,
        position_wise=position_wise,
    )
    topk.update(
        {
            "top_k": top_k,
            "include_labels": include_labels,
            "renormalize_topk": renormalize_topk,
            "selected_vocab_size": int(topk_indices.shape[-1]),
        }
    )
    if int(full_vocab["num_tokens"]) != int(topk["num_tokens"]):
        raise ValueError(
            "full-vocab and top-k perturbation metrics must use the same token count, "
            f"got {full_vocab['num_tokens']} and {topk['num_tokens']}."
        )
    return {"full_vocab": full_vocab, "topk": topk}


def _merge_metrics(
    total: dict[str, float | int],
    batch_metrics: dict[str, float | int | list[float] | list[int] | list[dict[str, float | int]]],
) -> None:
    num_tokens = int(batch_metrics["num_tokens"])
    total["kl_on_sum"] = float(total.get("kl_on_sum", 0.0)) + float(batch_metrics["kl_on"]) * num_tokens
    total["kl_off_sum"] = float(total.get("kl_off_sum", 0.0)) + float(batch_metrics["kl_off"]) * num_tokens
    total["num_tokens"] = int(total.get("num_tokens", 0)) + num_tokens


def _merge_position_lists(
    totals: dict[str, list[float]],
    counts: list[int],
    batch_metrics: dict[str, float | int | list[float] | list[int] | list[dict[str, float | int]]],
) -> tuple[dict[str, list[float]], list[int]]:
    position_kl_on = batch_metrics.get("position_kl_on")
    position_kl_off = batch_metrics.get("position_kl_off")
    position_num_tokens = batch_metrics.get("position_num_tokens")
    if (
        not isinstance(position_kl_on, list)
        or not isinstance(position_kl_off, list)
        or not isinstance(position_num_tokens, list)
    ):
        return totals, counts
    if not totals:
        totals = {
            "kl_on_sum": [0.0 for _ in position_kl_on],
            "kl_off_sum": [0.0 for _ in position_kl_off],
        }
        counts = [0 for _ in position_kl_on]
    if len(position_kl_on) != len(totals["kl_on_sum"]):
        raise ValueError("position-wise metric length changed across batches.")
    for index, (on_value, off_value, count) in enumerate(
        zip(position_kl_on, position_kl_off, position_num_tokens, strict=True)
    ):
        count_int = int(count)
        totals["kl_on_sum"][index] += float(on_value) * count_int
        totals["kl_off_sum"][index] += float(off_value) * count_int
        counts[index] += count_int
    return totals, counts


def _finalize_position_lists(
    totals: dict[str, list[float]],
    counts: list[int],
) -> dict[str, list[float] | list[int]]:
    if not totals:
        return {}
    position_kl_on: list[float] = []
    position_kl_off: list[float] = []
    position_delta: list[float] = []
    for index, count in enumerate(counts):
        if count <= 0:
            on_value = 0.0
            off_value = 0.0
        else:
            on_value = totals["kl_on_sum"][index] / float(count)
            off_value = totals["kl_off_sum"][index] / float(count)
        position_kl_on.append(_finite_metric("position_kl_on", on_value))
        position_kl_off.append(_finite_metric("position_kl_off", off_value))
        position_delta.append(_finite_metric("position_delta_kl", off_value - on_value))
    return {
        "position_kl_on": position_kl_on,
        "position_kl_off": position_kl_off,
        "position_delta_kl": position_delta,
        "position_num_tokens": counts,
    }


def evaluate_perturbation_robustness(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader[dict[str, Tensor]],
    config: TrainConfig,
    device: torch.device,
    max_batches: int | None = None,
    *,
    top_k: int | None = None,
    include_labels: bool | None = None,
    renormalize_topk: bool | None = None,
    include_position_metrics: bool = False,
) -> dict[str, float | int | list[float] | list[int] | list[dict[str, float | int]]]:
    """Compare teacher||student KL on clean and off-trajectory student states."""

    student.eval()
    teacher.eval()
    effective_top_k = top_k if top_k is not None else (config.topk.top_k if config.topk.enabled else None)
    effective_include_labels = config.topk.include_labels if include_labels is None else include_labels
    effective_renormalize = config.topk.renormalize_topk if renormalize_topk is None else renormalize_topk
    totals: dict[str, float | int] = {}
    position_totals: dict[str, list[float]] = {}
    position_counts: list[int] = []

    with torch.no_grad():
        for batch in _iter_limited_batches(dataloader, max_batches):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                teacher_logits = teacher(input_ids, attention_mask=attention_mask)
                output = student(input_ids, attention_mask=attention_mask)
            else:
                teacher_logits = teacher(input_ids)
                output = student(input_ids)
            teacher_logits = teacher_logits.to(output.on_logits.device)
            labels = labels.to(output.on_logits.device)
            mask = _select_shared_valid_mask(
                labels=labels,
                ignore_index=config.vocab.ignored_label_id,
                positions_per_sequence=config.mock.positions_per_sequence,
            )

            topk_indices = None
            if effective_top_k is not None:
                topk_indices = build_topk_indices(
                    teacher_logits.detach().float(),
                    labels=labels,
                    top_k=effective_top_k,
                    include_labels=effective_include_labels,
                )

            batch_metrics = compute_perturbation_metrics(
                teacher_logits=teacher_logits,
                on_logits=output.on_logits,
                off_logits=output.off_logits,
                mask=mask,
                tau=config.loss.tau,
                topk_indices=topk_indices,
                renormalize_topk=effective_renormalize,
                position_wise=include_position_metrics,
            )
            _merge_metrics(totals, batch_metrics)
            if include_position_metrics:
                position_totals, position_counts = _merge_position_lists(
                    position_totals,
                    position_counts,
                    batch_metrics,
                )

    num_tokens = int(totals.get("num_tokens", 0))
    if num_tokens <= 0:
        raise ValueError("perturbation robustness evaluation found no valid tokens.")

    kl_on_value = float(totals["kl_on_sum"]) / float(num_tokens)
    kl_off_value = float(totals["kl_off_sum"]) / float(num_tokens)
    metrics: dict[str, float | int | list[float] | list[int] | list[dict[str, float | int]]] = {
        "kl_on": _finite_metric("kl_on", kl_on_value),
        "kl_off": _finite_metric("kl_off", kl_off_value),
        "delta_kl": _finite_metric("delta_kl", kl_off_value - kl_on_value),
        "num_tokens": num_tokens,
    }
    if include_position_metrics:
        position_metrics = _finalize_position_lists(position_totals, position_counts)
        metrics.update(position_metrics)
        metrics["positions"] = [
            {
                "position": index,
                "kl_on": position_metrics["position_kl_on"][index],
                "kl_off": position_metrics["position_kl_off"][index],
                "delta_kl": position_metrics["position_delta_kl"][index],
                "num_tokens": position_metrics["position_num_tokens"][index],
            }
            for index in range(len(position_metrics.get("position_delta_kl", [])))
        ]
    return metrics
