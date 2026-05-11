"""Continuous-state distribution matching losses for mock logit tensors."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def center_logits(x: Tensor) -> Tensor:
    """Subtract the vocab-dimension mean while preserving shape."""

    return x - x.mean(dim=-1, keepdim=True)


def rms(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Stable root-mean-square over the vocab dimension with shape [..., 1]."""

    if eps <= 0:
        raise ValueError("eps must be positive.")
    value = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return value.to(dtype=x.dtype)


def _validate_csdm_logits(
    off_logits: Tensor,
    teacher_logits: Tensor,
    fake_logits: Tensor,
    tau: float,
) -> None:
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if off_logits.shape != teacher_logits.shape or off_logits.shape != fake_logits.shape:
        raise ValueError(
            "off_logits, teacher_logits, and fake_logits must have the same shape: "
            f"{tuple(off_logits.shape)}, {tuple(teacher_logits.shape)}, "
            f"{tuple(fake_logits.shape)}"
        )
    if off_logits.ndim != 3:
        raise ValueError(
            "CSDM logits must be rank-3 [B, T, V] or [B, N, V]; "
            f"got shape {tuple(off_logits.shape)}"
        )


def csdm_loss(
    off_logits: Tensor,
    teacher_logits: Tensor,
    fake_logits: Tensor,
    tau: float = 2.0,
    lambda_score: float = 0.1,
    residual_clip: float = 3.0,
    scale_min: float = 0.05,
    scale_max: float = 5.0,
    eps: float = 1e-8,
) -> Tensor:
    """Compute the Stage 1 CSDM off-trajectory logit loss.

    The teacher and fake-student logits are detached internally. Only
    ``off_logits`` receives gradients.
    """

    _validate_csdm_logits(off_logits, teacher_logits, fake_logits, tau)
    if residual_clip <= 0:
        raise ValueError("residual_clip must be positive.")
    if scale_min <= 0 or scale_max <= 0 or scale_min > scale_max:
        raise ValueError("scale bounds must be positive with scale_min <= scale_max.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    u = center_logits(off_logits / tau)

    teacher_log_probs = F.log_softmax(teacher_logits.detach() / tau, dim=-1)
    fake_log_probs = F.log_softmax(fake_logits.detach() / tau, dim=-1)
    residual = center_logits(teacher_log_probs - fake_log_probs)
    residual = torch.clamp(residual, -residual_clip, residual_clip)

    scale = rms(u.detach(), eps=eps) / (rms(residual, eps=eps) + eps)
    scale = torch.clamp(scale, scale_min, scale_max)

    residual = residual.detach()
    scale = scale.detach()
    target = (u + lambda_score * scale * residual).detach()

    return 0.5 * (u - target).pow(2).mean() * (tau**2)
