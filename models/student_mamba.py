"""Mock Mamba student for Stage 3 training scaffolding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.cdm_engine import DeltaPerturbationEngine, OffTrajectoryConfig


@dataclass(frozen=True)
class StudentOutput:
    on_logits: Tensor
    off_logits: Tensor
    fake_logits: Tensor
    h: Tensor
    h_off: Tensor
    h_delta_alt: Tensor


class StudentMamba(nn.Module, ABC):
    """Base student interface for future real Mamba integrations."""

    @abstractmethod
    def forward(self, input_ids: Tensor) -> StudentOutput:
        """Return on/off-trajectory student logits."""


class MockStudentMamba(StudentMamba):
    """Small recurrent student with a student-only off-state surrogate.

    ``h_delta_alt`` is produced by a lightweight projection of the student
    hidden state. It is only a mock delta-transition surrogate and does not
    represent real Mamba internals.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        hidden_size: int = 256,
        delta_scale: float = 0.1,
        off_config: OffTrajectoryConfig | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.sequence = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.delta_perturb_proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.delta_scale = delta_scale
        self.off_engine = DeltaPerturbationEngine(off_config)

    def forward(self, input_ids: Tensor) -> StudentOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}.")
        embeddings = self.embedding(input_ids)
        h, _ = self.sequence(embeddings)
        on_logits = self.lm_head(h)

        h_delta_alt = h + self.delta_scale * torch.tanh(self.delta_perturb_proj(h))
        h_off = self.off_engine.make_off_state(h, h_delta_alt=h_delta_alt)
        off_logits = self.lm_head(h_off)

        with torch.no_grad():
            fake_logits = self.lm_head(h_off.detach()).detach()

        return StudentOutput(
            on_logits=on_logits,
            off_logits=off_logits,
            fake_logits=fake_logits,
            h=h,
            h_off=h_off,
            h_delta_alt=h_delta_alt,
        )
