"""Stage 2 off-trajectory state construction for mock Mamba states."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class StateBatch:
    """Container for student recurrent states and optional transition metadata."""

    h: Tensor
    delta: Optional[Tensor] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class OffTrajectoryConfig:
    """Configuration for student-state off-trajectory perturbations.

    ``delta_perturb_eps`` is reserved for
    :meth:`MambaStateAdapter.forward_with_delta_scale`; ``make_off_state`` does
    not consume it because this mock Stage 2 engine does not access private
    Mamba internals.
    """

    delta_perturb_eps: float = 0.10
    noise_sigma: float = 0.01
    rho_min: float = 0.0
    rho_max: float = 1.0
    detach_direction: bool = True
    eps: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("delta_perturb_eps", "noise_sigma", "rho_min", "rho_max", "eps"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.rho_min > self.rho_max:
            raise ValueError("rho_min must be <= rho_max.")
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma must be non-negative.")
        if self.eps <= 0:
            raise ValueError("eps must be positive.")
        if self.delta_perturb_eps < 0:
            raise ValueError("delta_perturb_eps must be non-negative.")


class DeltaPerturbationEngine:
    """Construct off-trajectory Mamba student states from mock tensors.

    The preferred direction uses an alternate student recurrent state
    ``h_delta_alt`` produced by a delta-perturbed student transition. If
    ``h_delta_alt`` is omitted, this engine falls back to a Gaussian-noise-only
    baseline and does not represent the preferred CSDM direction.
    """

    def __init__(self, config: OffTrajectoryConfig | None = None) -> None:
        self.config = config or OffTrajectoryConfig()

    def make_off_state(
        self,
        h: Tensor,
        h_delta_alt: Optional[Tensor] = None,
        delta: Optional[Tensor] = None,
    ) -> Tensor:
        """Return an off-trajectory student state with the same shape as ``h``.

        Args:
            h: Student recurrent state shaped ``[B, D]`` or ``[B, T, D]``.
            h_delta_alt: Optional alternate student state from a delta-scaled
                transition. It must exactly match ``h`` in shape, device, and
                dtype. It is never a teacher state.
            delta: Optional student transition delta tensor accepted for future
                adapter integrations. It is intentionally unused here.
        """

        del delta
        self._validate_state(h, name="h")

        h_off = h
        if h_delta_alt is not None:
            self._validate_alt_state(h, h_delta_alt)
            direction = h_delta_alt - h
            if self.config.detach_direction:
                direction = direction.detach()
            rho = self._sample_rho(h)
            h_off = h + rho * direction

        if self.config.noise_sigma > 0:
            rms_h = torch.sqrt(h.float().pow(2).mean(dim=-1, keepdim=True) + self.config.eps)
            rms_h = rms_h.to(dtype=h.dtype)
            h_off = h_off + self.config.noise_sigma * rms_h * torch.randn_like(h)

        if h_off.shape != h.shape:
            raise RuntimeError(
                f"off-state shape changed unexpectedly: {tuple(h_off.shape)} != {tuple(h.shape)}"
            )
        return h_off

    def _sample_rho(self, h: Tensor) -> Tensor:
        rho_shape = h.shape[:-1] + (1,)
        return torch.empty(rho_shape, device=h.device, dtype=h.dtype).uniform_(
            self.config.rho_min,
            self.config.rho_max,
        )

    @staticmethod
    def _validate_state(h: Tensor, name: str) -> None:
        if not torch.is_tensor(h):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if not h.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor.")
        if h.ndim not in {2, 3}:
            raise ValueError(
                f"{name} must have rank 2 or 3, e.g. [B, D] or [B, T, D]; "
                f"got shape {tuple(h.shape)}."
            )

    @classmethod
    def _validate_alt_state(cls, h: Tensor, h_delta_alt: Tensor) -> None:
        cls._validate_state(h_delta_alt, name="h_delta_alt")
        if h_delta_alt.shape != h.shape:
            raise ValueError(
                "h_delta_alt must have exactly the same shape as h to avoid "
                f"silent broadcasting: {tuple(h_delta_alt.shape)} != {tuple(h.shape)}."
            )
        if h_delta_alt.device != h.device:
            raise ValueError(
                f"h_delta_alt must be on the same device as h: {h_delta_alt.device} != {h.device}."
            )
        if h_delta_alt.dtype != h.dtype:
            raise TypeError(
                f"h_delta_alt must have the same dtype as h: {h_delta_alt.dtype} != {h.dtype}."
            )


class MambaStateAdapter:
    """Placeholder interface for future student Mamba state extraction.

    Implementations should expose only student-side operations. No method here
    should call a teacher model or pass Mamba states to a teacher.
    """

    def forward_clean(self, input_ids: Tensor, *args: Any, **kwargs: Any) -> StateBatch:
        """Return clean student states.

        Expected mock-compatible shapes:
        - ``input_ids``: ``[B, T]``
        - optional ``attention_mask`` keyword: ``[B, T]`` when the concrete
          student adapter supports masks
        - returned ``StateBatch.h``: ``[B, T, D]`` or selected states ``[B, D]``
        - returned ``StateBatch.delta`` when available: adapter-defined student
          transition deltas broadcastable inside the adapter only
        """

        raise NotImplementedError

    def forward_with_delta_scale(
        self,
        input_ids: Tensor,
        delta_scale: float,
        *args: Any,
        **kwargs: Any,
    ) -> StateBatch:
        """Return alternate student states from a delta-scaled transition.

        ``delta_scale`` is where ``OffTrajectoryConfig.delta_perturb_eps`` is
        intended to be applied by concrete adapters. Expected returned
        ``StateBatch.h`` shape must match ``forward_clean(input_ids,
        attention_mask=...).h`` exactly, such as ``[B, T, D]`` or ``[B, D]``.
        ``input_ids`` and optional ``attention_mask`` remain clean token inputs;
        ``delta_scale`` is applied only inside student-side transition logic.
        """

        raise NotImplementedError

    def logits_from_state(self, h: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Project student recurrent state to student logits.

        Expected mock-compatible shapes:
        - ``h``: ``[B, T, D]`` or ``[B, D]``
        - returned logits: ``[B, T, V]`` or ``[B, V]`` from the student head
        This method projects student states only. It must not call a teacher or
        pass student recurrent states to a teacher model.
        """

        raise NotImplementedError
