from __future__ import annotations

import types

import pytest
import torch

from data.vocab import (
    VocabAlignmentReport,
    get_tokenizer_vocab_size,
    validate_token_id_ranges,
    validate_vocab_alignment,
)
from models.student_mamba import MockStudentMamba
from models.teacher_wrapper import MockTeacherWrapper


class LenTokenizer:
    vocab_size = 7
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self) -> int:
        return 11


class AttrTokenizer:
    vocab_size = 13


def test_get_tokenizer_vocab_size_prefers_len_over_vocab_size() -> None:
    assert get_tokenizer_vocab_size(LenTokenizer()) == 11


def test_get_tokenizer_vocab_size_falls_back_to_vocab_size() -> None:
    assert get_tokenizer_vocab_size(AttrTokenizer()) == 13


def test_get_tokenizer_vocab_size_rejects_missing_or_invalid_size() -> None:
    with pytest.raises(ValueError, match="must expose"):
        get_tokenizer_vocab_size(object())

    with pytest.raises(ValueError, match="positive"):
        get_tokenizer_vocab_size(types.SimpleNamespace(vocab_size=0))


def test_validate_token_id_ranges_accepts_valid_input_ids_and_labels() -> None:
    validate_token_id_ranges(
        input_ids=torch.tensor([[0, 4], [8, 2]]),
        labels=torch.tensor([[1, 5], [7, -100]]),
        vocab_size=9,
        ignored_label_id=-100,
    )


def test_validate_token_id_ranges_accepts_ignored_labels() -> None:
    validate_token_id_ranges(
        input_ids=torch.tensor([[0, 1]]),
        labels=torch.tensor([[-100, -100]]),
        vocab_size=2,
        ignored_label_id=-100,
    )


def test_validate_token_id_ranges_rejects_negative_input_ids() -> None:
    with pytest.raises(ValueError, match="negative token id"):
        validate_token_id_ranges(
            input_ids=torch.tensor([[-1, 0]]),
            labels=torch.tensor([[0, -100]]),
            vocab_size=4,
        )


def test_validate_token_id_ranges_rejects_input_ids_outside_vocab() -> None:
    with pytest.raises(ValueError, match="outside vocabulary size"):
        validate_token_id_ranges(
            input_ids=torch.tensor([[0, 4]]),
            labels=torch.tensor([[1, -100]]),
            vocab_size=4,
        )


def test_validate_token_id_ranges_rejects_labels_outside_vocab() -> None:
    with pytest.raises(ValueError, match="labels contain token id"):
        validate_token_id_ranges(
            input_ids=torch.tensor([[0, 1]]),
            labels=torch.tensor([[4, -100]]),
            vocab_size=4,
        )


def test_validate_token_id_ranges_rejects_bool_tensors() -> None:
    with pytest.raises(TypeError, match="integer token ids"):
        validate_token_id_ranges(
            input_ids=torch.tensor([[True, False]]),
            labels=torch.tensor([[0, -100]]),
            vocab_size=2,
        )


def test_validate_vocab_alignment_passes_matching_sizes() -> None:
    report = validate_vocab_alignment(
        tokenizer_vocab_size=11,
        teacher_vocab_size=11,
        student_vocab_size=11,
        pad_token_id=0,
        eos_token_id=1,
    )

    assert isinstance(report, VocabAlignmentReport)
    assert report.valid
    assert report.tokenizer_vocab_size == 11
    assert report.teacher_vocab_size == 11
    assert report.student_vocab_size == 11
    assert report.pad_token_id == 0
    assert report.eos_token_id == 1


def test_validate_vocab_alignment_rejects_mismatch_in_strict_mode() -> None:
    with pytest.raises(ValueError, match="tokenizer and teacher vocab sizes must match"):
        validate_vocab_alignment(
            tokenizer_vocab_size=12,
            teacher_vocab_size=11,
            student_vocab_size=11,
            strict=True,
        )


def test_validate_vocab_alignment_non_strict_returns_errors_without_raising() -> None:
    report = validate_vocab_alignment(
        tokenizer_vocab_size=12,
        teacher_vocab_size=11,
        student_vocab_size=10,
        strict=False,
    )

    assert not report.valid
    assert report.errors


def test_validate_vocab_alignment_rejects_added_pad_token_out_of_range() -> None:
    with pytest.raises(ValueError, match="pad_token_id=11"):
        validate_vocab_alignment(
            tokenizer_vocab_size=11,
            teacher_vocab_size=11,
            student_vocab_size=11,
            pad_token_id=11,
        )


def test_mock_teacher_and_student_expose_vocab_size() -> None:
    teacher = MockTeacherWrapper(vocab_size=17, hidden_size=8)
    student = MockStudentMamba(vocab_size=17, hidden_size=8)

    assert teacher.vocab_size == 17
    assert student.vocab_size == 17
