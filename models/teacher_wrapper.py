"""Teacher wrappers for Stage 3 mock training."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class TeacherWrapper(nn.Module, ABC):
    """Base teacher interface.

    Teachers consume only clean token prefixes. They never receive Mamba
    recurrent states such as ``h_t``, ``h'_t``, or ``h_delta_alt``.
    """

    @abstractmethod
    def forward(self, input_ids: Tensor) -> Tensor:
        """Return teacher logits aligned as ``p_phi(y | x_{<=t})``."""


class MockTeacherWrapper(TeacherWrapper):
    """Frozen deterministic teacher over token-prefix summaries."""

    def __init__(self, vocab_size: int = 1024, hidden_size: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.prefix_proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}.")
        with torch.no_grad():
            token_embeddings = self.embedding(input_ids)
            steps = torch.arange(
                1,
                input_ids.shape[1] + 1,
                device=input_ids.device,
                dtype=token_embeddings.dtype,
            ).view(1, -1, 1)
            prefix_state = token_embeddings.cumsum(dim=1) / steps
            hidden = torch.tanh(self.prefix_proj(prefix_state))
            logits = self.lm_head(hidden)
        return logits.detach()
