from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
import importlib.util

import torch

from data.dataset import MockTextDataset
from models.student_mamba import MockStudentMamba
from models.teacher_wrapper import MockTeacherWrapper


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SPEC = importlib.util.spec_from_file_location("cdm_mamba_kd_train", ROOT / "train.py")
assert TRAIN_SPEC is not None
assert TRAIN_SPEC.loader is not None
train_module = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train_module
TRAIN_SPEC.loader.exec_module(train_module)
compute_losses = train_module.compute_losses
load_train_config = train_module.load_train_config


def test_mock_dataset_next_token_shift_and_ignore_index() -> None:
    dataset = MockTextDataset(vocab_size=32, seq_len=8, num_samples=4, seed=123, ignore_index=-100)

    first = dataset[0]
    again = dataset[0]

    assert torch.equal(first["input_ids"], again["input_ids"])
    assert torch.equal(first["labels"][:-1], first["input_ids"][1:])
    assert int(first["labels"][-1]) == -100


def test_teacher_is_frozen_and_uses_only_tokens() -> None:
    teacher = MockTeacherWrapper(vocab_size=32, hidden_size=16)
    input_ids = torch.randint(0, 32, (2, 8))

    logits = teacher(input_ids)

    assert logits.shape == (2, 8, 32)
    assert not logits.requires_grad
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_student_fake_logits_detached_and_loss_gradients() -> None:
    config = load_train_config(ROOT / "configs" / "train_config.yaml")
    teacher = MockTeacherWrapper(config.mock.vocab_size, config.mock.hidden_size)
    student = MockStudentMamba(config.mock.vocab_size, config.mock.hidden_size)
    batch = next(
        iter(
            torch.utils.data.DataLoader(
                MockTextDataset(
                    vocab_size=config.mock.vocab_size,
                    seq_len=config.mock.seq_len,
                    num_samples=2,
                    seed=config.seed,
                    ignore_index=config.mock.ignore_index,
                ),
                batch_size=2,
            )
        )
    )

    teacher_logits = teacher(batch["input_ids"])
    output = student(batch["input_ids"])
    losses = compute_losses(output, teacher_logits, batch["labels"], config)
    losses["total"].backward()

    assert output.fake_logits.shape == output.off_logits.shape
    assert not output.fake_logits.requires_grad
    assert teacher_logits.requires_grad is False
    assert student.embedding.weight.grad is not None
    assert student.embedding.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_train_mock_subprocess_runs_two_optimizer_steps() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--config",
            "configs/train_config.yaml",
            "--mock",
            "--max_steps",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )

    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]

    assert [record["step"] for record in records] == [1, 2]
    for record in records:
        for key in ("total", "ce", "kd", "csdm", "grad_norm"):
            assert key in record
            assert math.isfinite(float(record[key]))


def test_mock_training_imports_no_real_llama_or_mamba_modules() -> None:
    forbidden = ("transformers", "mamba_ssm")
    assert all(name not in sys.modules for name in forbidden)
