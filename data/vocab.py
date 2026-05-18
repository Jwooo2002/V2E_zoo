"""Vocabulary alignment and token id validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class VocabAlignmentReport:
    tokenizer_vocab_size: int | None
    teacher_vocab_size: int | None
    student_vocab_size: int | None
    pad_token_id: int | None
    eos_token_id: int | None
    max_input_id: int | None = None
    max_label_id: int | None = None
    ignored_label_id: int = -100
    padded_model_vocab: bool = False
    vocab_padding: int = 0
    valid: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def get_tokenizer_vocab_size(tokenizer: Any) -> int:
    """Return tokenizer size, preferring ``len(tokenizer)`` over ``vocab_size``."""

    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        raise ValueError("Tokenizer must expose __len__ or vocab_size for vocab alignment.")
    vocab_size = int(vocab_size)
    if vocab_size <= 0:
        raise ValueError(f"tokenizer vocab size must be positive, got {vocab_size}.")
    return vocab_size


def validate_token_id_ranges(
    input_ids: Tensor,
    labels: Tensor,
    vocab_size: int | None = None,
    ignored_label_id: int = -100,
    *,
    input_vocab_size: int | None = None,
    label_vocab_size: int | None = None,
) -> None:
    """Validate token ids against tokenizer and model vocabulary supports.

    ``labels`` may contain ``ignored_label_id`` for masked causal-LM positions;
    all other labels must be in ``[0, label_vocab_size)`` and all
    ``input_ids`` must be in ``[0, input_vocab_size)``. For legacy callers,
    ``vocab_size`` sets both ranges.
    """

    if input_vocab_size is None:
        input_vocab_size = vocab_size
    if label_vocab_size is None:
        label_vocab_size = vocab_size
    if input_vocab_size is None or label_vocab_size is None:
        raise ValueError("input_vocab_size and label_vocab_size must be provided when vocab_size is omitted.")
    input_vocab_size = int(input_vocab_size)
    label_vocab_size = int(label_vocab_size)
    if input_vocab_size <= 0:
        raise ValueError(f"input_vocab_size must be positive, got {input_vocab_size}.")
    if label_vocab_size <= 0:
        raise ValueError(f"label_vocab_size must be positive, got {label_vocab_size}.")
    if input_ids.shape != labels.shape:
        raise ValueError(
            "input_ids and labels must have the same shape, "
            f"got {tuple(input_ids.shape)} and {tuple(labels.shape)}."
        )
    _require_integer_tensor("input_ids", input_ids)
    _require_integer_tensor("labels", labels)
    if input_ids.numel() == 0:
        raise ValueError("input_ids must not be empty.")
    if labels.numel() == 0:
        raise ValueError("labels must not be empty.")

    input_min = int(input_ids.min().detach().cpu())
    input_max = int(input_ids.max().detach().cpu())
    if input_min < 0:
        raise ValueError(
            f"input_ids contain negative token id {input_min}; expected ids in [0, {input_vocab_size})."
        )
    if input_max >= input_vocab_size:
        raise ValueError(
            f"input_ids contain token id {input_max} outside vocabulary size {input_vocab_size}; "
            f"expected ids in [0, {input_vocab_size})."
        )

    valid_labels = labels.ne(ignored_label_id)
    if not bool(valid_labels.any()):
        return
    selected = labels[valid_labels]
    label_min = int(selected.min().detach().cpu())
    label_max = int(selected.max().detach().cpu())
    if label_min < 0:
        raise ValueError(
            f"labels contain negative token id {label_min}; only {ignored_label_id} may be negative."
        )
    if label_max >= label_vocab_size:
        raise ValueError(
            f"labels contain token id {label_max} outside vocabulary size {label_vocab_size}; "
            f"expected valid labels in [0, {label_vocab_size}) or {ignored_label_id}."
        )


def validate_vocab_alignment(
    tokenizer_vocab_size: int | None,
    teacher_vocab_size: int | None,
    student_vocab_size: int | None,
    *,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
    strict: bool = True,
    ignored_label_id: int = -100,
    allow_padded_model_vocab: bool = True,
) -> VocabAlignmentReport:
    """Check tokenizer token support is compatible with teacher/student logits.

    Teacher and student vocab sizes must match because KD/CSDM compare their
    logits directly. Tokenizer vocab may be smaller than the model vocab for
    padded embedding/logit tables such as Qwen-style models.
    """

    warnings: list[str] = []
    errors: list[str] = []

    sizes = {
        "tokenizer": tokenizer_vocab_size,
        "teacher": teacher_vocab_size,
        "student": student_vocab_size,
    }
    for name, size in sizes.items():
        if size is not None and int(size) <= 0:
            errors.append(f"{name}_vocab_size must be positive, got {size}.")

    tokenizer_size = None if tokenizer_vocab_size is None else int(tokenizer_vocab_size)
    teacher_size = None if teacher_vocab_size is None else int(teacher_vocab_size)
    student_size = None if student_vocab_size is None else int(student_vocab_size)
    model_vocab_size = _first_not_none(teacher_size, student_size)
    padded_model_vocab = False
    vocab_padding = 0

    if teacher_size is not None and student_size is not None and teacher_size != student_size:
        errors.append(
            "teacher and student vocab sizes must match: "
            f"teacher={teacher_vocab_size}, student={student_vocab_size}."
        )
    if tokenizer_size is not None and model_vocab_size is not None:
        if tokenizer_size > model_vocab_size:
            model_vocab_name = "teacher" if teacher_size is not None else "student"
            errors.append(
                f"tokenizer vocab size must not exceed {model_vocab_name} vocab size: "
                f"tokenizer={tokenizer_size}, {model_vocab_name}={model_vocab_size}."
            )
        elif tokenizer_size < model_vocab_size:
            vocab_padding = model_vocab_size - tokenizer_size
            if allow_padded_model_vocab:
                padded_model_vocab = True
                warnings.append(
                    "tokenizer vocab is smaller than model vocab; treating model vocab as padded. "
                    f"tokenizer={tokenizer_size}, model={model_vocab_size}, padding={vocab_padding}."
                )
            else:
                errors.append(
                    "tokenizer and model vocab sizes must match when padded model vocab is disabled: "
                    f"tokenizer={tokenizer_size}, model={model_vocab_size}."
                )

    reference_vocab_size = _first_not_none(tokenizer_size, model_vocab_size)
    _validate_special_token_id("pad_token_id", pad_token_id, reference_vocab_size, errors, warnings)
    _validate_special_token_id("eos_token_id", eos_token_id, reference_vocab_size, errors, warnings)

    report = VocabAlignmentReport(
        tokenizer_vocab_size=tokenizer_size,
        teacher_vocab_size=teacher_size,
        student_vocab_size=student_size,
        pad_token_id=None if pad_token_id is None else int(pad_token_id),
        eos_token_id=None if eos_token_id is None else int(eos_token_id),
        ignored_label_id=ignored_label_id,
        padded_model_vocab=padded_model_vocab,
        vocab_padding=vocab_padding,
        valid=not errors,
        warnings=warnings,
        errors=errors,
    )
    if strict and errors:
        raise ValueError("Vocabulary alignment failed: " + " ".join(errors))
    return report


def _first_not_none(*values: int | None) -> int | None:
    for value in values:
        if value is not None:
            return int(value)
    return None


def _require_integer_tensor(name: str, tensor: Tensor) -> None:
    if tensor.dtype == torch.bool or tensor.dtype.is_floating_point or tensor.dtype.is_complex:
        raise TypeError(f"{name} must contain integer token ids, got dtype {tensor.dtype}.")


def _validate_special_token_id(
    name: str,
    token_id: int | None,
    vocab_size: int | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    if token_id is None:
        return
    token_id = int(token_id)
    if token_id < 0:
        errors.append(f"{name} must be non-negative when set, got {token_id}.")
    if vocab_size is None:
        warnings.append(f"{name}={token_id} could not be range-checked because no vocab size was provided.")
    elif token_id >= vocab_size:
        errors.append(f"{name}={token_id} is outside vocabulary size {vocab_size}.")
