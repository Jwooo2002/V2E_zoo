from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from models.teacher_wrapper import HuggingFaceTeacherConfig, HuggingFaceTeacherWrapper


FAKE_VOCAB_SIZE = 23
ROOT = Path(__file__).resolve().parents[1]
TRAIN_SPEC = importlib.util.spec_from_file_location("cdm_mamba_kd_train_hf_tests", ROOT / "train.py")
assert TRAIN_SPEC is not None
assert TRAIN_SPEC.loader is not None
train = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train
TRAIN_SPEC.loader.exec_module(train)


class RecordingLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def log(self, step: int, metrics: dict[str, object]) -> None:
        self.records.append({"step": step, **metrics})


def install_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> type[nn.Module]:
    fake_transformers = types.ModuleType("transformers")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> "FakeTokenizer":
            return cls()

    class FakeModel(nn.Module):
        instances: list["FakeModel"] = []
        max_input_id: int | None = None
        forward_calls: int = 0

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.config = types.SimpleNamespace(vocab_size=FAKE_VOCAB_SIZE)
            self.forward_grad_enabled: bool | None = None
            self.forward_attention_mask: torch.Tensor | None = None
            FakeModel.instances.append(self)

        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> "FakeModel":
            return cls()

        def forward(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> object:
            self.forward_grad_enabled = torch.is_grad_enabled()
            self.forward_attention_mask = attention_mask
            FakeModel.max_input_id = int(input_ids.max().item())
            FakeModel.forward_calls += 1
            base = torch.arange(FAKE_VOCAB_SIZE, dtype=torch.float32, device=input_ids.device)
            logits = input_ids.float().unsqueeze(-1) * 0.01 + base.view(1, 1, -1)
            logits = logits + self.weight.float().view(1, 1, 1)
            return types.SimpleNamespace(logits=logits)

    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_transformers.AutoModelForCausalLM = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return FakeModel


def tiny_hf_config(gradient_accumulation_steps: int = 1) -> train.TrainConfig:
    return train.TrainConfig(
        seed=7,
        teacher_type="hf",
        student_type="mock",
        mixed_precision="bf16",
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        loss=replace(train.LossConfig(), ce_weight=0.2, kd_weight=1.0, csdm_weight=0.0),
        mock=train.MockConfig(
            batch_size=1,
            seq_len=6,
            vocab_size=5,
            hidden_size=12,
            num_samples=8,
            positions_per_sequence=4,
            ignore_index=-100,
        ),
        hf_teacher=train.HuggingFaceRuntimeConfig(
            model_name_or_path="fake-local-model",
            torch_dtype="float32",
            device_map=None,
            local_files_only=True,
        ),
    )


def test_hf_teacher_mock_student_training_smoke_no_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeModel = install_fake_transformers(monkeypatch)
    logger = RecordingLogger()

    train.run_training(tiny_hf_config(), max_steps=1, logger=logger)  # type: ignore[arg-type]

    fake_model = FakeModel.instances[-1]  # type: ignore[attr-defined]
    assert logger.records[0]["step"] == 1
    assert logger.records[0]["optimizer_step"] == 1
    assert logger.records[0]["accumulation_steps"] == 1
    assert logger.records[0]["csdm"] == 0.0
    assert fake_model.forward_grad_enabled is False
    assert fake_model.forward_attention_mask is not None
    assert torch.equal(fake_model.forward_attention_mask, torch.ones_like(fake_model.forward_attention_mask))
    assert FakeModel.max_input_id is not None and FakeModel.max_input_id < FAKE_VOCAB_SIZE  # type: ignore[attr-defined]
    assert all(not parameter.requires_grad for parameter in fake_model.parameters())


def test_hf_teacher_wrapper_returns_detached_logits_and_frozen_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeModel = install_fake_transformers(monkeypatch)
    wrapper = HuggingFaceTeacherWrapper(
        HuggingFaceTeacherConfig(
            model_name_or_path="fake-local-model",
            torch_dtype="float32",
            device_map=None,
            local_files_only=True,
        )
    )
    fake_model = FakeModel.instances[-1]  # type: ignore[attr-defined]

    input_ids = torch.randint(0, FAKE_VOCAB_SIZE, (2, 4))
    attention_mask = torch.ones_like(input_ids)
    logits = wrapper(input_ids, attention_mask=attention_mask)

    assert logits.shape == (2, 4, FAKE_VOCAB_SIZE)
    assert not logits.requires_grad
    assert fake_model.forward_grad_enabled is False
    assert fake_model.forward_attention_mask is attention_mask
    assert all(not parameter.requires_grad for parameter in wrapper.model.parameters())


def test_compute_losses_shape_and_vocab_mismatch_raise_clear_errors() -> None:
    config = tiny_hf_config()
    output = train.StudentOutput(
        on_logits=torch.randn(2, 4, 5, requires_grad=True),
        off_logits=torch.randn(2, 4, 5, requires_grad=True),
        fake_logits=torch.randn(2, 4, 5),
        h=torch.randn(2, 4, 3),
        h_off=torch.randn(2, 4, 3),
        h_delta_alt=torch.randn(2, 4, 3),
    )
    labels = torch.randint(0, 5, (2, 4))

    with pytest.raises(ValueError, match=r"share \[B, T\] shape before masking"):
        train.compute_losses(output, torch.randn(2, 5, 5), labels, config)

    with pytest.raises(ValueError, match="share vocab size before masking"):
        train.compute_losses(output, torch.randn(2, 4, 6), labels, config)


def test_gradient_accumulation_two_microbatches_one_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeModel = install_fake_transformers(monkeypatch)
    logger = RecordingLogger()

    train.run_training(tiny_hf_config(gradient_accumulation_steps=2), max_steps=1, logger=logger)  # type: ignore[arg-type]

    assert len(logger.records) == 1
    record = logger.records[0]
    assert record["step"] == 1
    assert record["optimizer_step"] == 1
    assert record["micro_step"] == 2
    assert record["accumulation_steps"] == 2
    assert record["accumulation_progress"] == "2/2"
    assert FakeModel.forward_calls == 2  # type: ignore[attr-defined]


def test_hf_cpu_device_map_keeps_training_device_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    config = tiny_hf_config()
    config = replace(config, hf_teacher=replace(config.hf_teacher, device_map="cpu"))

    assert train._training_device(config) == torch.device("cpu")


def test_nonfinite_loss_raises_before_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_transformers(monkeypatch)
    logger = RecordingLogger()

    def nonfinite_losses(_output, _teacher_logits, _labels, _config):
        value = torch.tensor(float("nan"), requires_grad=True)
        zero = torch.tensor(0.0)
        return {"total": value, "ce": zero, "kd": zero, "csdm": zero}

    monkeypatch.setattr(train, "compute_losses", nonfinite_losses)

    with pytest.raises(FloatingPointError, match="non-finite total loss"):
        train.run_training(tiny_hf_config(), max_steps=1, logger=logger)  # type: ignore[arg-type]

    assert logger.records == []


def test_mock_flag_forces_mock_teacher_student_despite_type_overrides() -> None:
    args = types.SimpleNamespace(
        config=ROOT / "configs" / "train_config.yaml",
        mock=True,
        teacher_type="hf",
        student_type="mock",
        teacher_model_name_or_path="should-not-load",
        seq_len=None,
        batch_size=None,
        gradient_accumulation_steps=None,
        mixed_precision=None,
        csdm_weight=None,
        kd_weight=None,
        ce_weight=None,
        local_files_only=False,
    )

    config = train.derive_runtime_config(args)

    assert config.teacher_type == "mock"
    assert config.student_type == "mock"
    assert config.loss.csdm_weight == 0.1
