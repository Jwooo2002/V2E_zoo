import pytest
import torch

from losses.cdm_loss import center_logits, csdm_loss, rms


def test_center_logits_preserves_shape_and_centers_vocab_mean() -> None:
    x = torch.randn(2, 3, 7)

    centered = center_logits(x)

    assert centered.shape == x.shape
    assert torch.allclose(centered.mean(dim=-1), torch.zeros(2, 3), atol=1e-6)


def test_rms_returns_singleton_vocab_dimension() -> None:
    x = torch.randn(2, 3, 7)

    value = rms(x, eps=1e-8)

    assert value.shape == (2, 3, 1)
    assert torch.isfinite(value).all()


def test_csdm_loss_finite_scalar_for_sequence_logits() -> None:
    off = torch.randn(2, 3, 7, requires_grad=True)
    teacher = torch.randn(2, 3, 7)
    fake = torch.randn(2, 3, 7)

    loss = csdm_loss(off, teacher, fake)

    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_csdm_loss_gradients_flow_to_off_logits_only() -> None:
    off = torch.zeros(2, 3, 7, requires_grad=True)
    teacher = torch.zeros(2, 3, 7)
    fake = torch.zeros(2, 3, 7)
    teacher[..., 0] = 4.0
    fake[..., -1] = 4.0
    teacher.requires_grad_()
    fake.requires_grad_()

    loss = csdm_loss(off, teacher, fake)
    loss.backward()

    assert off.grad is not None
    assert torch.isfinite(off.grad).all()
    assert off.grad.abs().sum() > 0
    assert teacher.grad is None
    assert fake.grad is None


def test_csdm_loss_shape_mismatch_raises() -> None:
    off = torch.randn(2, 3, 7)
    teacher = torch.randn(2, 4, 7)
    fake = torch.randn(2, 3, 7)

    with pytest.raises((AssertionError, ValueError)):
        csdm_loss(off, teacher, fake)


def test_csdm_loss_invalid_rank_raises() -> None:
    off = torch.randn(2, 7)
    teacher = torch.randn(2, 7)
    fake = torch.randn(2, 7)

    with pytest.raises((AssertionError, ValueError)):
        csdm_loss(off, teacher, fake)


def test_csdm_loss_invalid_tau_raises() -> None:
    off = torch.randn(2, 3, 7)
    teacher = torch.randn(2, 3, 7)
    fake = torch.randn(2, 3, 7)

    with pytest.raises(ValueError):
        csdm_loss(off, teacher, fake, tau=0.0)


def test_csdm_loss_invalid_scale_bounds_raise() -> None:
    off = torch.randn(2, 3, 7)
    teacher = torch.randn(2, 3, 7)
    fake = torch.randn(2, 3, 7)

    with pytest.raises(ValueError):
        csdm_loss(off, teacher, fake, scale_min=2.0, scale_max=1.0)
