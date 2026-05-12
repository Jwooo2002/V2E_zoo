"""Text data utilities for mock and local Stage 7A smoke tests."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


PathInput = str | Path
_TEXT_FORMATS = frozenset({"auto", "text", "jsonl"})


@dataclass(frozen=True)
class TextDatasetConfig:
    """Configuration for tiny local text or JSONL datasets."""

    path: PathInput | None = None
    paths: PathInput | Sequence[PathInput] | None = None
    seq_len: int = 128
    stride: int | None = None
    max_examples: int | None = None
    seed: int = 42
    add_special_tokens: bool = True
    add_eos: bool = False
    shuffle: bool = False
    file_format: str = "auto"
    text_field: str = "text"
    text_key: str | None = None
    ignore_index: int = -100
    pad_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.path is not None and self.paths is not None:
            raise ValueError("Use either path or paths, not both.")
        raw_paths: PathInput | Sequence[PathInput] | None = self.paths if self.paths is not None else self.path
        if raw_paths is None:
            raise ValueError("path or paths must be provided.")
        if isinstance(raw_paths, (str, Path)):
            paths = (Path(raw_paths),)
        else:
            paths = tuple(Path(path) for path in raw_paths)
        if not paths:
            raise ValueError("paths must contain at least one file.")
        if self.seq_len <= 1:
            raise ValueError("seq_len must be greater than 1.")
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be positive when provided.")
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError("max_examples must be positive when provided.")
        if self.file_format not in _TEXT_FORMATS:
            allowed = ", ".join(sorted(_TEXT_FORMATS))
            raise ValueError(f"Unsupported file_format {self.file_format!r}; expected one of: {allowed}.")
        text_field = self.text_key if self.text_key is not None else self.text_field
        if not text_field:
            raise ValueError("text_field must be non-empty.")
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "text_field", text_field)


class TokenizedTextDataset(Dataset[dict[str, Tensor]]):
    """Fixed-length causal-LM examples from tiny local text or JSONL files.

    Examples are padded using ``tokenizer.padding_side`` when needed. Labels
    are next-token shifted, with the final real-token position and all padding
    positions set to ``ignore_index``.
    """

    def __init__(self, config: TextDatasetConfig | Any, tokenizer: Any | None = None) -> None:
        if isinstance(config, TextDatasetConfig):
            dataset_config = config
            dataset_tokenizer = tokenizer
        elif isinstance(tokenizer, TextDatasetConfig):
            dataset_config = tokenizer
            dataset_tokenizer = config
        else:
            raise TypeError("TokenizedTextDataset expects TextDatasetConfig and tokenizer.")
        if dataset_tokenizer is None:
            raise TypeError("tokenizer must be provided.")

        self.tokenizer = dataset_tokenizer
        self.config = dataset_config
        texts = _read_text_records(dataset_config)
        examples = _build_tokenized_examples(dataset_tokenizer, dataset_config, texts)
        if dataset_config.shuffle:
            rng = random.Random(dataset_config.seed)
            rng.shuffle(examples)
        if dataset_config.max_examples is not None:
            examples = examples[: dataset_config.max_examples]
        if not examples:
            raise ValueError("No tokenized examples were produced.")
        self._examples = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0 or index >= len(self._examples):
            raise IndexError(index)
        return {key: value.clone() for key, value in self._examples[index].items()}


def _read_text_records(config: TextDatasetConfig) -> list[str]:
    records: list[str] = []
    for path in config.paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        file_format = _resolve_file_format(path, config.file_format)
        if file_format == "text":
            records.extend(_read_txt_records(path))
        elif file_format == "jsonl":
            records.extend(_read_jsonl_records(path, config.text_field))
        else:
            raise AssertionError(f"Unhandled file_format: {file_format!r}")
    return records


def _resolve_file_format(path: Path, file_format: str) -> str:
    if file_format != "auto":
        return file_format
    return "jsonl" if path.suffix.lower() == ".jsonl" else "text"


def _read_txt_records(path: Path) -> list[str]:
    records: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                records.append(text)
    return records


def _read_jsonl_records(path: Path, text_field: str) -> list[str]:
    records: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row {line_number} in {path} must be an object.")
            value = row.get(text_field)
            if not isinstance(value, str):
                raise ValueError(f"JSONL row {line_number} in {path} is missing string field {text_field!r}.")
            if value.strip():
                records.append(value)
    return records


def _build_tokenized_examples(
    tokenizer: Any,
    config: TextDatasetConfig,
    texts: Sequence[str],
) -> list[dict[str, Tensor]]:
    examples: list[dict[str, Tensor]] = []
    for text in texts:
        token_ids = _encode_text(
            tokenizer,
            text,
            add_special_tokens=config.add_special_tokens,
            add_eos=config.add_eos,
        )
        if not token_ids:
            continue
        chunks = _token_chunks(token_ids, config.seq_len, config.stride)
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            input_ids, attention_mask = _pad_token_ids(chunk, tokenizer, config)
            labels = _next_token_labels(input_ids, attention_mask, config.ignore_index)
            examples.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
    return examples


def _encode_text(tokenizer: Any, text: str, *, add_special_tokens: bool, add_eos: bool) -> list[int]:
    if hasattr(tokenizer, "encode"):
        try:
            token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        except TypeError:
            token_ids = tokenizer.encode(text)
    elif callable(tokenizer):
        try:
            output = tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
        except TypeError:
            output = tokenizer(text)
        token_ids = _extract_input_ids(output)
    else:
        raise TypeError("tokenizer must provide encode(text) or be callable.")

    ids = _normalize_token_ids(token_ids)
    if add_eos:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            ids.append(int(eos_token_id))
    return ids


def _extract_input_ids(output: Any) -> Any:
    if isinstance(output, Mapping):
        return output["input_ids"]
    if hasattr(output, "input_ids"):
        return output.input_ids
    raise TypeError("tokenizer output must expose input_ids.")


def _normalize_token_ids(token_ids: Any) -> list[int]:
    if isinstance(token_ids, Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        if len(token_ids) != 1:
            raise ValueError("tokenizer returned a batch for a single text example.")
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _token_chunks(token_ids: list[int], seq_len: int, stride: int | None) -> list[list[int]]:
    if stride is None:
        return [_truncate_token_ids(token_ids, seq_len)]

    chunks: list[list[int]] = []
    start = 0
    while start < len(token_ids):
        chunks.append(token_ids[start : start + seq_len])
        if start + seq_len >= len(token_ids):
            break
        start += stride
    return chunks


def _truncate_token_ids(token_ids: list[int], seq_len: int) -> list[int]:
    return token_ids[:seq_len]


def _pad_token_ids(
    token_ids: list[int],
    tokenizer: Any,
    config: TextDatasetConfig,
) -> tuple[Tensor, Tensor]:
    padding_side = getattr(tokenizer, "padding_side", "right")
    if padding_side not in {"left", "right"}:
        raise ValueError("tokenizer.padding_side must be 'left' or 'right'.")

    pad_count = config.seq_len - len(token_ids)
    if pad_count < 0:
        raise ValueError("token_ids must be truncated before padding.")
    if pad_count:
        pad_token_id = _resolve_pad_token_id(tokenizer, config)
        padding = [pad_token_id] * pad_count
        if padding_side == "left":
            input_ids = padding + token_ids
            attention_mask = [0] * pad_count + [1] * len(token_ids)
        else:
            input_ids = token_ids + padding
            attention_mask = [1] * len(token_ids) + [0] * pad_count
    else:
        input_ids = token_ids
        attention_mask = [1] * len(token_ids)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)


def _resolve_pad_token_id(tokenizer: Any, config: TextDatasetConfig) -> int:
    if config.pad_token_id is not None:
        return int(config.pad_token_id)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    raise ValueError("A pad_token_id is required when examples are shorter than seq_len.")


def _next_token_labels(input_ids: Tensor, attention_mask: Tensor, ignore_index: int) -> Tensor:
    labels = torch.full_like(input_ids, fill_value=ignore_index)
    valid_positions = attention_mask.nonzero(as_tuple=False).flatten()
    if valid_positions.numel() >= 2:
        labels[valid_positions[:-1]] = input_ids[valid_positions[1:]]
    return labels
