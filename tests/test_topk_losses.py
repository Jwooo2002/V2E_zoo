from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
import importlib.util

import pytest
import torch
import torch.nn.functional as F

from losses.cdm_loss import csdm_loss
from losses.kd_loss import build_topk_indices, gather_logits_by_indices, kd_kl_loss


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SPEC = importlib.util.spec_from_file_location("cdm_mamba_kd_train_topk", ROOT / "train.py")
assert TRAIN_SPEC is not None
assert TRAIN_SPEC.loader is not None
train_module = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train_module
TRAIN_SPEC.loader.exec_module(train_module)


def test_gather_logits_by_indices_validates_and_preserves_gradients() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)
    indices = torch.tensor(
        [
            [[0, 2], [1, 3], [4, 0]],
            [[3, 1], [2, 2], [0, 4]],
        ],
        dtype=torch.long,
    )

    gathered = gather_logits_by_indices(logits, indices)
    gathered.sum().backward()

    assert gathered.shape == (2, 3, 2)
    assert torch.equal(gathered.detach(), logits.detach().gather(dim=-1, index=indices))
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_gather_logits_by_indices_rejects_bad_indices() -> None:
    logits = torch.randn(1, 2, 4)

    with pytest.raises(TypeError):
        gather_logits_by_indices(logits, torch.zeros(1, 2, 2, dtype=torch.int32))
    with pytest.raises(IndexError):
        gather_logits_by_indices(logits, torch.tensor([[[0, 4], [1, 2]]], dtype=torch.long))
    with pytest.raises(ValueError):
        gather_logits_by_indices(logits, torch.zeros(1, 3, 2, dtype=torch.long))


def test_build_topk_indices_uses_detached_teacher_logits_and_optional_labels() -> None:
    teacher = torch.tensor(
        [
            [
                [0.0, 6.0, 1.0, 5.0, 2.0],
                [7.0, 1.0, 6.0, 0.0, 5.0],
            ]
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[4, -100]], dtype=torch.long)

    indices = build_topk_indices(teacher, labels=labels, top_k=2, include_labels=True)

    assert indices.shape == (1, 2, 3)
    assert torch.equal(indices[..., :2], torch.tensor([[[1, 3], [0, 2]]]))
    assert int(indices[0, 0, 2]) == 4
    assert int(indices[0, 1, 2]) == int(indices[0, 1, 0])
    assert not indices.requires_grad


def test_build_topk_indices_is_not_influenced_by_student_logits() -> None:
    teacher = torch.zeros(1, 1, 6)
    teacher[..., 1] = 10.0
    teacher[..., 3] = 9.0
    student = torch.zeros(1, 1, 6)
    student[..., 5] = 100.0

    indices = build_topk_indices(teacher, top_k=2, include_labels=False)

    assert torch.equal(indices, torch.tensor([[[1, 3]]]))
    assert int(student.argmax(dim=-1).item()) == 5


def test_topk_kd_matches_manual_selected_vocab_renormalized_kl() -> None:
    student = torch.randn(2, 3, 7, requires_grad=True)
    teacher = torch.randn(2, 3, 7, requires_grad=True)
    indices = build_topk_indices(teacher, top_k=3, include_labels=False)
    tau = 1.7

    loss = kd_kl_loss(student, teacher, tau=tau, topk_indices=indices, renormalize_topk=True)

    student_selected = student.gather(dim=-1, index=indices)
    teacher_selected = teacher.detach().gather(dim=-1, index=indices)
    expected = (
        F.softmax(teacher_selected / tau, dim=-1)
        * (
            F.log_softmax(teacher_selected / tau, dim=-1)
            - F.log_softmax(student_selected / tau, dim=-1)
        )
    ).sum(dim=-1).mean() * (tau**2)
    assert torch.allclose(loss, expected)

    loss.backward()
    assert student.grad is not None
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_topk_csdm_gradients_flow_only_to_off_logits() -> None:
    off = torch.randn(2, 3, 8, requires_grad=True)
    teacher = torch.randn(2, 3, 8, requires_grad=True)
    fake = torch.randn(2, 3, 8, requires_grad=True)
    labels = torch.tensor([[1, 2, -100], [3, -100, 4]], dtype=torch.long)
    indices = build_topk_indices(teacher, labels=labels, top_k=3, include_labels=True)

    loss = csdm_loss(off, teacher, fake, topk_indices=indices, renormalize_topk=True)
    loss.backward()

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert off.grad is not None
    assert torch.isfinite(off.grad).all()
    assert off.grad.abs().sum() > 0
    assert teacher.grad is None
    assert fake.grad is None


def test_topk_kd_without_renormalization_is_finite() -> None:
    student = torch.randn(1, 2, 6, requires_grad=True)
    teacher = torch.randn(1, 2, 6)
    indices = build_topk_indices(teacher, top_k=2, include_labels=False)

    loss = kd_kl_loss(student, teacher, topk_indices=indices, renormalize_topk=False)
    loss.backward()

    assert torch.isfinite(loss)
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_train_mock_subprocess_runs_two_steps_with_topk_enabled() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--config",
            "configs/train_config.yaml",
            "--mock",
            "--max_steps",
            "2",
            "--topk-enabled",
            "--top-k",
            "16",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )

    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]

    assert [record["step"] for record in records] == [1, 2]
    for record in records:
        for key in ("total", "ce", "kd", "csdm", "grad_norm"):
            assert key in record
            assert math.isfinite(float(record[key]))
