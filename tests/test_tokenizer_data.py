from __future__ import annotations

import json
import random
import sys
import types
from pathlib import Path

import pytest
import torch

from data.dataset import TextDatasetConfig, TokenizedTextDataset
from data.tokenizer import TokenizerConfig, load_tokenizer


class SimpleTokenizer:
    pad_token_id = 0
    eos_token_id = 27
    padding_side = "right"
    truncation_side = "right"
    vocab_size = 32

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [ord(char) - 96 for char in text.lower() if "a" <= char.lower() <= "z"]
        if add_special_tokens:
            return [28] + ids + [29]
        return ids


def test_load_tokenizer_uses_lazy_transformers_import_and_eos_pad_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTokenizer:
        pad_token = None
        pad_token_id = None
        eos_token = "</s>"
        eos_token_id = 2
        padding_side = "right"

    class FakeAutoTokenizer:
        calls: list[tuple[str, dict[str, object]]] = []

        @classmethod
        def from_pretrained(cls, name: str, **kwargs: object) -> FakeTokenizer:
            cls.calls.append((name, kwargs))
            return FakeTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))

    tokenizer = load_tokenizer(
        TokenizerConfig(
            tokenizer_name_or_path="tiny-tokenizer",
            local_files_only=True,
            padding_side="left",
            pad_token_strategy="eos",
        )
    )

    assert "AutoTokenizer" not in vars(sys.modules[load_tokenizer.__module__])
    assert FakeAutoTokenizer.calls[0][0] == "tiny-tokenizer"
    assert FakeAutoTokenizer.calls[0][1]["local_files_only"] is True
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "</s>"
    assert tokenizer.pad_token_id == 2


def test_load_tokenizer_new_pad_strategy_adds_special_token(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTokenizer:
        pad_token = None
        pad_token_id = None
        padding_side = "right"

        def __init__(self) -> None:
            self.added: dict[str, str] | None = None

        def add_special_tokens(self, tokens: dict[str, str]) -> None:
            self.added = tokens
            self.pad_token = tokens["pad_token"]
            self.pad_token_id = 99

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_name: str, **_kwargs: object) -> FakeTokenizer:
            return FakeTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))

    tokenizer = load_tokenizer(
        TokenizerConfig(name_or_path="tiny-tokenizer", pad_token_strategy="new", new_pad_token="<pad>")
    )

    assert tokenizer.added == {"pad_token": "<pad>"}
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.pad_token_id == 99


def test_pad_token_strategy_none_does_not_fall_back_to_eos_for_padding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeTokenizer:
        pad_token = None
        pad_token_id = None
        eos_token = "</s>"
        eos_token_id = 2
        padding_side = "right"
        vocab_size = 8

        def __len__(self) -> int:
            return self.vocab_size

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del text, add_special_tokens
            return [3, 4]

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_name: str, **_kwargs: object) -> FakeTokenizer:
            return FakeTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))
    tokenizer = load_tokenizer(TokenizerConfig(name_or_path="tiny-tokenizer", pad_token_strategy="none"))
    path = tmp_path / "tiny.txt"
    path.write_text("ab\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pad_token_id is required"):
        TokenizedTextDataset(
            TextDatasetConfig(path=path, seq_len=4, add_special_tokens=False),
            tokenizer,
        )


def test_tokenized_text_dataset_builds_next_token_labels_and_attention_mask(tmp_path: Path) -> None:
    path = tmp_path / "tiny.txt"
    path.write_text("ab\nabc\n\n", encoding="utf-8")

    dataset = TokenizedTextDataset(
        TextDatasetConfig(path=path, seq_len=5, add_special_tokens=False),
        SimpleTokenizer(),
    )

    first = dataset[0]
    second = dataset[1]

    assert len(dataset) == 2
    assert torch.equal(first["input_ids"], torch.tensor([1, 2, 0, 0, 0]))
    assert torch.equal(first["attention_mask"], torch.tensor([1, 1, 0, 0, 0]))
    assert torch.equal(first["labels"], torch.tensor([2, -100, -100, -100, -100]))
    assert torch.equal(second["labels"], torch.tensor([2, 3, -100, -100, -100]))


def test_tokenized_text_dataset_left_padding_masks_padding_labels(tmp_path: Path) -> None:
    path = tmp_path / "tiny.txt"
    path.write_text("abc\n", encoding="utf-8")
    tokenizer = SimpleTokenizer()
    tokenizer.padding_side = "left"

    dataset = TokenizedTextDataset(
        config=TextDatasetConfig(path=path, seq_len=5, add_special_tokens=False),
        tokenizer=tokenizer,
    )
    sample = dataset[0]

    assert torch.equal(sample["input_ids"], torch.tensor([0, 0, 1, 2, 3]))
    assert torch.equal(sample["attention_mask"], torch.tensor([0, 0, 1, 1, 1]))
    assert torch.equal(sample["labels"], torch.tensor([-100, -100, 2, 3, -100]))


def test_tokenized_jsonl_dataset_shuffle_and_max_examples_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "tiny.jsonl"
    texts = ["ab", "bc", "cd", "de"]
    path.write_text("\n".join(json.dumps({"text": text}) for text in texts), encoding="utf-8")
    config = TextDatasetConfig(
        path=path,
        seq_len=4,
        add_special_tokens=False,
        shuffle=True,
        seed=11,
        max_examples=2,
    )

    first = TokenizedTextDataset(config, SimpleTokenizer())
    again = TokenizedTextDataset(SimpleTokenizer(), config)
    expected = texts[:]
    random.Random(11).shuffle(expected)
    expected_ids = [ord(text[0]) - 96 for text in expected[:2]]

    assert len(first) == 2
    assert [int(first[index]["input_ids"][0]) for index in range(len(first))] == expected_ids
    assert [
        first[index]["input_ids"].tolist() for index in range(len(first))
    ] == [again[index]["input_ids"].tolist() for index in range(len(again))]


def test_tokenized_jsonl_dataset_requires_string_text_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"not_text": "abc"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing string field"):
        TokenizedTextDataset(TextDatasetConfig(path=path, seq_len=4), SimpleTokenizer())


def test_tokenized_text_dataset_filters_single_token_chunks(tmp_path: Path) -> None:
    path = tmp_path / "single.txt"
    path.write_text("a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No tokenized examples"):
        TokenizedTextDataset(
            TextDatasetConfig(path=path, seq_len=4, add_special_tokens=False),
            SimpleTokenizer(),
        )
