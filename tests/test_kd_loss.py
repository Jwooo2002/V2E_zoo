import pytest
import torch

from losses.kd_loss import kd_kl_loss


def test_kd_kl_loss_finite_scalar() -> None:
    student = torch.randn(2, 3, 7, requires_grad=True)
    teacher = torch.randn(2, 3, 7)

    loss = kd_kl_loss(student, teacher)

    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_kd_kl_loss_gradients_flow_to_student_only() -> None:
    student = torch.randn(2, 3, 7, requires_grad=True)
    teacher = torch.randn(2, 3, 7, requires_grad=True)

    loss = kd_kl_loss(student, teacher)
    loss.backward()

    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_kd_kl_loss_none_reduction_shape() -> None:
    student = torch.randn(2, 5, 11)
    teacher = torch.randn(2, 5, 11)

    loss = kd_kl_loss(student, teacher, reduction="none")

    assert loss.shape == (2, 5)


def test_kd_kl_loss_shape_mismatch_raises() -> None:
    student = torch.randn(2, 3, 7)
    teacher = torch.randn(2, 4, 7)

    with pytest.raises((AssertionError, ValueError)):
        kd_kl_loss(student, teacher)


def test_kd_kl_loss_invalid_rank_raises() -> None:
    student = torch.randn(2, 7)
    teacher = torch.randn(2, 7)

    with pytest.raises((AssertionError, ValueError)):
        kd_kl_loss(student, teacher)


def test_kd_kl_loss_invalid_tau_raises() -> None:
    student = torch.randn(2, 3, 7)
    teacher = torch.randn(2, 3, 7)

    with pytest.raises(ValueError):
        kd_kl_loss(student, teacher, tau=0.0)
