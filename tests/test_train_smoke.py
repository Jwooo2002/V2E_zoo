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


def test_train_mock_checkpoint_save_resume_and_auto_resume(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    first = subprocess.run(
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
            "--checkpoint-output-dir",
            str(checkpoint_dir),
            "--save-at-end",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    first_records = [json.loads(line) for line in first.stdout.splitlines() if line.startswith("{")]
    checkpoint_path = checkpoint_dir / "checkpoint_step_2_opt_2.pt"

    assert checkpoint_path.is_file()
    assert first_records[-1]["checkpoint_path"] == str(checkpoint_path)

    resumed = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--config",
            "configs/train_config.yaml",
            "--mock",
            "--max_steps",
            "4",
            "--gradient-accumulation-steps",
            "1",
            "--resume-from",
            str(checkpoint_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    resumed_records = [json.loads(line) for line in resumed.stdout.splitlines() if line.startswith("{")]
    train_records = [record for record in resumed_records if "total" in record]

    assert resumed_records[0]["event"] == "resume"
    assert resumed_records[0]["optimizer_step"] == 2
    assert [record["optimizer_step"] for record in train_records] == [3, 4]

    auto = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--config",
            "configs/train_config.yaml",
            "--mock",
            "--max_steps",
            "3",
            "--gradient-accumulation-steps",
            "1",
            "--checkpoint-output-dir",
            str(checkpoint_dir),
            "--auto-resume",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    auto_records = [json.loads(line) for line in auto.stdout.splitlines() if line.startswith("{")]
    auto_train_records = [record for record in auto_records if "total" in record]

    assert auto_records[0]["event"] == "resume"
    assert auto_records[0]["optimizer_step"] == 2
    assert [record["optimizer_step"] for record in auto_train_records] == [3]


def test_train_mock_strict_resume_rejects_config_metadata_mismatch(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    subprocess.run(
        [
            sys.executable,
            "train.py",
            "--config",
            "configs/train_config.yaml",
            "--mock",
            "--max_steps",
            "1",
            "--seq-len",
            "8",
            "--batch-size",
            "1",
            "--gradient-accumulation-steps",
            "1",
            "--checkpoint-output-dir",
            str(checkpoint_dir),
            "--save-at-end",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    checkpoint_path = checkpoint_dir / "checkpoint_step_1_opt_1.pt"
    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--config",
            "configs/train_config.yaml",
            "--mock",
            "--max_steps",
            "2",
            "--seq-len",
            "8",
            "--batch-size",
            "2",
            "--gradient-accumulation-steps",
            "1",
            "--resume-from",
            str(checkpoint_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode != 0
    assert "Checkpoint config snapshot is incompatible" in result.stderr


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


def test_mock_training_does_not_instantiate_real_teacher_or_student() -> None:
    script = r"""
import builtins
import importlib
import importlib.util
import sys
from pathlib import Path

forbidden = ("transformers", "mamba_ssm")
real_import = builtins.__import__
real_import_module = importlib.import_module

def is_forbidden(name):
    return any(name == item or name.startswith(item + ".") for item in forbidden)

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if is_forbidden(name):
        raise AssertionError(f"{name} imported during mock training isolation test")
    return real_import(name, globals, locals, fromlist, level)

def guarded_import_module(name, package=None):
    if is_forbidden(name):
        raise AssertionError(f"{name} imported through importlib during mock training isolation test")
    return real_import_module(name, package)

builtins.__import__ = guarded_import
importlib.import_module = guarded_import_module

root = Path.cwd()
spec = importlib.util.spec_from_file_location("cdm_mamba_kd_train_isolation", root / "train.py")
assert spec is not None
assert spec.loader is not None
train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train
spec.loader.exec_module(train)

calls = {"hf_teacher": 0, "real_mamba": 0}

def forbidden_hf_teacher(*_args, **_kwargs):
    calls["hf_teacher"] += 1
    raise AssertionError("HuggingFaceTeacherWrapper should not be instantiated for mock training")

def forbidden_real_mamba(*_args, **_kwargs):
    calls["real_mamba"] += 1
    raise AssertionError("RealMambaStudent should not be instantiated for mock training")

train.HuggingFaceTeacherWrapper = forbidden_hf_teacher
train.RealMambaStudent = forbidden_real_mamba

config = train.derive_runtime_config(
    train.argparse.Namespace(
        config=root / "configs" / "train_config.yaml",
        mock=True,
        max_steps=1,
        teacher_type="hf",
        student_type="mamba",
        teacher_model_name_or_path="forbidden-teacher",
        dataset_type="text",
        data_path="forbidden.txt",
        tokenizer_name_or_path="forbidden-tokenizer",
        max_examples=1,
        text_field="text",
        student_model_name_or_path="forbidden-student",
        student_vocab_size=None,
        student_hidden_size=None,
        seq_len=8,
        batch_size=2,
        gradient_accumulation_steps=1,
        mixed_precision="no",
        csdm_weight=None,
        kd_weight=None,
        ce_weight=None,
        local_files_only=True,
        topk_enabled=None,
        top_k=None,
        topk_include_labels=None,
        topk_renormalize=None,
        teacher_cache_enabled=False,
        teacher_cache_dir=None,
        teacher_cache_dtype=None,
        teacher_cache_overwrite=False,
        teacher_cache_use_top_k=False,
        teacher_cache_top_k=None,
        hf_torch_dtype=None,
        hf_device_map=None,
        trust_remote_code=False,
        hf_attn_implementation=None,
        use_safetensors=None,
        load_in_8bit=False,
        load_in_4bit=False,
    )
)
assert config.data.dataset_type == "mock"
train.run_training(config, max_steps=1)
assert calls == {"hf_teacher": 0, "real_mamba": 0}
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
