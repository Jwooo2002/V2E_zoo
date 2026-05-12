from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_fake_transformers(tmp_path: Path, *, vocab_size: int = 64, record_path: Path | None = None) -> None:
    record_line = (
        f"        with open({str(record_path)!r}, 'w', encoding='utf-8') as handle:\n"
        "            handle.write(json.dumps({'attention_mask': None if attention_mask is None else attention_mask.cpu().tolist(), 'input_ids': input_ids.cpu().tolist()}))\n"
        if record_path is not None
        else "        pass\n"
    )
    source = f"""
import json
import types
import torch
from torch import nn

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return FakeTokenizer()

class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2
    vocab_size = {vocab_size}

    def __len__(self):
        return self.vocab_size

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        for token in text.split():
            value = sum(ord(ch) for ch in token) % (self.vocab_size - 3)
            ids.append(value + 3)
        return ids

class AutoModelForCausalLM(nn.Module):
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.input_embeddings = nn.Embedding({vocab_size}, 4)
        self.config = types.SimpleNamespace(vocab_size={vocab_size})

    def get_input_embeddings(self):
        return self.input_embeddings

    def forward(self, *, input_ids, attention_mask=None):
{record_line}
        base = torch.arange({vocab_size}, dtype=torch.float32, device=input_ids.device)
        logits = input_ids.float().unsqueeze(-1) * 0.01 + base.view(1, 1, -1)
        return types.SimpleNamespace(logits=logits + self.weight.float().view(1, 1, 1))
"""
    (tmp_path / "transformers.py").write_text(
        textwrap.dedent(source),
        encoding="utf-8",
    )


def _run_text_training(
    tmp_path: Path,
    *,
    topk: bool = False,
    teacher_type: str = "mock",
    batch_size: int = 1,
    max_examples: int | None = None,
    teacher_cache_dir: Path | None = None,
    fake_vocab_size: int = 64,
    record_path: Path | None = None,
    text: str = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
) -> subprocess.CompletedProcess[str]:
    _write_fake_transformers(tmp_path, vocab_size=fake_vocab_size, record_path=record_path)
    data_path = tmp_path / "tiny.txt"
    data_path.write_text(text, encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    command = [
        sys.executable,
        "train.py",
        "--config",
        "configs/train_config.yaml",
        "--dataset-type",
        "text",
        "--data-path",
        str(data_path),
        "--tokenizer-name-or-path",
        "fake-tokenizer",
        "--teacher-type",
        teacher_type,
        "--student-type",
        "mock",
        "--max_steps",
        "1",
        "--seq-len",
        "8",
        "--batch-size",
        str(batch_size),
        "--gradient-accumulation-steps",
        "1",
        "--mixed-precision",
        "no",
    ]
    if teacher_type == "hf":
        command.extend(["--teacher-model-name-or-path", "fake-hf-teacher", "--local-files-only"])
    if max_examples is not None:
        command.extend(["--max-examples", str(max_examples)])
    if topk:
        command.extend(["--topk-enabled", "--top-k", "8"])
    if teacher_cache_dir is not None:
        command.extend(["--teacher-cache-enabled", "--teacher-cache-dir", str(teacher_cache_dir)])
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )


def _records(result: subprocess.CompletedProcess[str]) -> list[dict[str, object]]:
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]


def test_train_text_dataset_mock_teacher_student_subprocess(tmp_path: Path) -> None:
    result = _run_text_training(tmp_path, topk=False)
    records = _records(result)

    assert [record["step"] for record in records] == [1]
    for key in ("total", "ce", "kd", "csdm", "grad_norm"):
        assert key in records[0]
        assert math.isfinite(float(records[0][key]))


def test_train_text_dataset_topk_mock_teacher_student_subprocess(tmp_path: Path) -> None:
    result = _run_text_training(tmp_path, topk=True)
    records = _records(result)

    assert [record["step"] for record in records] == [1]
    for key in ("total", "ce", "kd", "csdm", "grad_norm"):
        assert key in records[0]
        assert math.isfinite(float(records[0][key]))


def test_train_text_dataset_batch_larger_than_examples_does_not_hang(tmp_path: Path) -> None:
    result = _run_text_training(tmp_path, batch_size=2, max_examples=1)
    records = _records(result)

    assert [record["step"] for record in records] == [1]


def test_train_text_dataset_topk_with_teacher_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    result = _run_text_training(tmp_path, topk=True, teacher_cache_dir=cache_dir)
    records = _records(result)

    assert [record["step"] for record in records] == [1]
    assert list(cache_dir.glob("*.pt"))


def test_train_text_dataset_hf_teacher_receives_attention_mask(tmp_path: Path) -> None:
    record_path = tmp_path / "teacher_record.json"
    result = _run_text_training(
        tmp_path,
        teacher_type="hf",
        batch_size=1,
        record_path=record_path,
        text="alpha beta gamma",
    )
    records = _records(result)
    teacher_record = json.loads(record_path.read_text(encoding="utf-8"))

    assert [record["step"] for record in records] == [1]
    assert teacher_record["attention_mask"] is not None
    assert 0 in teacher_record["attention_mask"][0]


def test_train_text_dataset_tokenizer_teacher_vocab_mismatch_fails_clearly(tmp_path: Path) -> None:
    _write_fake_transformers(tmp_path, vocab_size=65)
    data_path = tmp_path / "tiny.txt"
    data_path.write_text("alpha beta gamma", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--config",
            "configs/train_config.yaml",
            "--dataset-type",
            "text",
            "--data-path",
            str(data_path),
            "--tokenizer-name-or-path",
            "fake-tokenizer",
            "--teacher-type",
            "mock",
            "--student-type",
            "mamba",
            "--student-vocab-size",
            "64",
            "--max_steps",
            "1",
            "--seq-len",
            "8",
            "--batch-size",
            "1",
            "--gradient-accumulation-steps",
            "1",
            "--mixed-precision",
            "no",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode != 0
    assert "vocab sizes must match" in result.stderr
