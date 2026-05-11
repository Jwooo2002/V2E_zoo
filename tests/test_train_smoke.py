from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
import importlib.util

import pytest
import torch

from data.dataset import MockTextDataset
from models.cdm_engine import OffTrajectoryConfig
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
run_training = train_module.run_training
LogitCacheConfig = train_module.LogitCacheConfig


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


def test_mock_student_casts_delta_surrogate_to_hidden_dtype_and_keeps_gradients() -> None:
    class Float64DeltaProjection(torch.nn.Module):
        def __init__(self, projection: torch.nn.Module) -> None:
            super().__init__()
            self.projection = projection

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            return self.projection(h).double()

    student = MockStudentMamba(
        vocab_size=32,
        hidden_size=16,
        off_config=OffTrajectoryConfig(
            noise_sigma=0.0,
            rho_min=1.0,
            rho_max=1.0,
            detach_direction=False,
        ),
    )
    student.delta_perturb_proj = Float64DeltaProjection(student.delta_perturb_proj)
    input_ids = torch.randint(0, 32, (2, 8))

    output = student(input_ids)
    loss = output.on_logits.float().mean() + output.off_logits.float().mean()
    loss.backward()

    assert output.h_delta_alt.dtype == output.h.dtype
    assert output.h_delta_alt.device == output.h.device
    assert output.h_delta_alt.requires_grad
    assert output.on_logits.shape == (2, 8, 32)
    assert output.off_logits.shape == output.on_logits.shape
    assert output.fake_logits.shape == output.off_logits.shape
    assert not output.fake_logits.requires_grad
    assert student.embedding.weight.grad is not None
    assert student.embedding.weight.grad.abs().sum() > 0
    projection = student.delta_perturb_proj.projection
    assert projection.weight.grad is not None
    assert projection.weight.grad.abs().sum() > 0


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
            "--gradient-accumulation-steps",
            "1",
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


def test_train_mock_subprocess_with_teacher_cache_writes_tmp_only(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--config",
            "configs/train_config.yaml",
            "--mock",
            "--max_steps",
            "2",
            "--gradient-accumulation-steps",
            "1",
            "--teacher-cache-enabled",
            "--teacher-cache-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )

    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    cache_files = list(tmp_path.glob("*.pt"))

    assert [record["step"] for record in records] == [1, 2]
    assert cache_files
    assert all(path.parent == tmp_path for path in cache_files)


def test_train_mock_subprocess_topk_uses_full_cached_teacher_logits(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--config",
            "configs/train_config.yaml",
            "--mock",
            "--max_steps",
            "2",
            "--gradient-accumulation-steps",
            "1",
            "--topk-enabled",
            "--top-k",
            "8",
            "--teacher-cache-enabled",
            "--teacher-cache-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )

    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]

    assert [record["step"] for record in records] == [1, 2]
    assert list(tmp_path.glob("*.pt"))


def _small_cache_config(tmp_path: Path, *, overwrite: bool = False, use_top_k: bool = False) -> train_module.TrainConfig:
    config = load_train_config(ROOT / "configs" / "train_config.yaml")
    return replace(
        config,
        gradient_accumulation_steps=1,
        mixed_precision="no",
        teacher_cache=LogitCacheConfig(
            enabled=True,
            cache_dir=str(tmp_path),
            dtype="float32",
            overwrite=overwrite,
            use_top_k=use_top_k,
            top_k=4,
        ),
        mock=replace(
            config.mock,
            batch_size=2,
            seq_len=8,
            vocab_size=32,
            hidden_size=16,
            num_samples=2,
            positions_per_sequence=4,
        ),
    )


def test_train_teacher_cache_computes_once_for_repeated_batch_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_cache_config(tmp_path, overwrite=False)
    teacher = MockTeacherWrapper(config.mock.vocab_size, config.mock.hidden_size)
    calls = {"count": 0}
    original_forward = teacher.forward

    def counting_forward(input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        calls["count"] += 1
        return original_forward(input_ids, attention_mask=attention_mask)

    teacher.forward = counting_forward  # type: ignore[method-assign]
    monkeypatch.setattr(train_module, "_build_teacher", lambda config, device: teacher.to(device))

    run_training(config, max_steps=2)

    assert calls["count"] == 1
    assert len(list(tmp_path.glob("*.pt"))) == 1


def test_train_teacher_cache_overwrite_recomputes_repeated_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_cache_config(tmp_path, overwrite=True)
    teacher = MockTeacherWrapper(config.mock.vocab_size, config.mock.hidden_size)
    calls = {"count": 0}
    original_forward = teacher.forward

    def counting_forward(input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        calls["count"] += 1
        return original_forward(input_ids, attention_mask=attention_mask)

    teacher.forward = counting_forward  # type: ignore[method-assign]
    monkeypatch.setattr(train_module, "_build_teacher", lambda config, device: teacher.to(device))

    run_training(config, max_steps=2)

    assert calls["count"] == 2
    assert len(list(tmp_path.glob("*.pt"))) == 1


def test_train_topk_only_teacher_cache_path_raises_not_implemented(tmp_path: Path) -> None:
    config = _small_cache_config(tmp_path, use_top_k=True)

    with pytest.raises(NotImplementedError, match="Top-k-only teacher logit cache entries"):
        run_training(config, max_steps=1)


def test_mock_training_imports_no_real_llama_or_mamba_modules() -> None:
    forbidden = ("transformers", "mamba_ssm")
    assert all(name not in sys.modules for name in forbidden)
