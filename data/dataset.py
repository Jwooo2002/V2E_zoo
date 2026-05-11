"""Mock text data for Stage 3 training smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MockTextDatasetConfig:
    vocab_size: int = 1024
    seq_len: int = 128
    num_samples: int = 1024
    seed: int = 42
    ignore_index: int = -100


class MockTextDataset(Dataset[dict[str, Tensor]]):
    """Deterministic random-token dataset with next-token labels.

    Each sample is generated from ``seed + index``. ``labels[t]`` is the next
    token ``input_ids[t + 1]`` and ``labels[-1]`` is ``ignore_index`` because no
    next-token target exists for the final placeholder position.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        seq_len: int = 128,
        num_samples: int = 1024,
        seed: int = 42,
        ignore_index: int = -100,
    ) -> None:
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1.")
        if seq_len <= 1:
            raise ValueError("seq_len must be greater than 1.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        self.config = MockTextDatasetConfig(
            vocab_size=vocab_size,
            seq_len=seq_len,
            num_samples=num_samples,
            seed=seed,
            ignore_index=ignore_index,
        )

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0 or index >= self.config.num_samples:
            raise IndexError(index)
        generator = torch.Generator()
        generator.manual_seed(self.config.seed + index)
        input_ids = torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=(self.config.seq_len,),
            generator=generator,
            dtype=torch.long,
        )
        labels = torch.empty_like(input_ids)
        labels[:-1] = input_ids[1:]
        labels[-1] = self.config.ignore_index
        return {"input_ids": input_ids, "labels": labels}
