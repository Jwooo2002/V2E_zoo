from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import textwrap
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

import models.student_mamba as student_mamba_module


ROOT = Path(__file__).resolve().parents[1]
FAKE_TEACHER_VOCAB_SIZE = 23

TRAIN_SPEC = importlib.util.spec_from_file_location("cdm_mamba_kd_train_real_mamba_tests", ROOT / "train.py")
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


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> type[nn.Module]:
    fake_transformers = types.ModuleType("transformers")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> "FakeTokenizer":
            return cls()

    class FakeModel(nn.Module):
        instances: list["FakeModel"] = []
        forward_calls = 0

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.input_embeddings = nn.Embedding(FAKE_TEACHER_VOCAB_SIZE, 4)
            self.config = types.SimpleNamespace(vocab_size=FAKE_TEACHER_VOCAB_SIZE)
            self.forward_grad_enabled: bool | None = None
            self.forward_attention_mask: torch.Tensor | None = None
            self.max_input_id: int | None = None
            FakeModel.instances.append(self)

        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> "FakeModel":
            return cls()

        def get_input_embeddings(self) -> nn.Embedding:
            return self.input_embeddings

        def forward(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> object:
            self.forward_grad_enabled = torch.is_grad_enabled()
            self.forward_attention_mask = attention_mask
            self.max_input_id = int(input_ids.max().item())
            FakeModel.forward_calls += 1
            base = torch.arange(FAKE_TEACHER_VOCAB_SIZE, dtype=torch.float32, device=input_ids.device)
            logits = input_ids.float().unsqueeze(-1) * 0.01 + base.view(1, 1, -1)
            logits = logits + self.weight.float().view(1, 1, 1)
            return types.SimpleNamespace(logits=logits)

    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_transformers.AutoModelForCausalLM = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return FakeModel


def _install_fake_public_mamba(monkeypatch: pytest.MonkeyPatch) -> type[nn.Module]:
    real_import_module = importlib.import_module

    class FakeMambaConfig:
        def __init__(self, **kwargs: object) -> None:
            self.d_model = int(kwargs["d_model"])
            self.vocab_size = int(kwargs["vocab_size"])
            self.n_layer = int(kwargs["n_layer"])

    class FakeBackbone(nn.Module):
        def __init__(self, config: FakeMambaConfig) -> None:
            super().__init__()
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embedding(input_ids)

    class FakeMambaLMHeadModel(nn.Module):
        instances: list["FakeMambaLMHeadModel"] = []

        def __init__(
            self,
            config: FakeMambaConfig,
            device: str | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__()
            self.config = config
            self.backbone = FakeBackbone(config)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            to_kwargs: dict[str, object] = {}
            if device is not None:
                to_kwargs["device"] = torch.device(device)
            if dtype is not None:
                to_kwargs["dtype"] = dtype
            if to_kwargs:
                self.to(**to_kwargs)
            FakeMambaLMHeadModel.instances.append(self)

    fake_mamba = types.ModuleType("mamba_ssm")
    fake_mamba.__version__ = "fake"
    fake_config = types.ModuleType("mamba_ssm.models.config_mamba")
    fake_config.MambaConfig = FakeMambaConfig
    fake_mixer = types.ModuleType("mamba_ssm.models.mixer_seq_simple")
    fake_mixer.MambaConfig = FakeMambaConfig
    fake_mixer.MambaLMHeadModel = FakeMambaLMHeadModel
    fake_modules = {
        "mamba_ssm": fake_mamba,
        "mamba_ssm.models.config_mamba": fake_config,
        "mamba_ssm.models.mixer_seq_simple": fake_mixer,
    }

    def fake_import_module(name: str, package: str | None = None) -> types.ModuleType:
        if name in fake_modules:
            return fake_modules[name]
        if name.startswith("mamba_ssm."):
            raise ImportError(f"fake mamba module does not provide {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(student_mamba_module.importlib, "import_module", fake_import_module)
    return FakeMambaLMHeadModel


def _tiny_hf_real_mamba_config(*, csdm_weight: float = 0.1) -> train.TrainConfig:
    return train.TrainConfig(
        seed=13,
        teacher_type="hf",
        student_type="mamba",
        mixed_precision="no",
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        loss=replace(train.LossConfig(), ce_weight=0.2, kd_weight=1.0, csdm_weight=csdm_weight),
        mock=train.MockConfig(
            batch_size=1,
            seq_len=5,
            vocab_size=7,
            hidden_size=8,
            num_samples=4,
            positions_per_sequence=4,
            ignore_index=-100,
        ),
        mamba_student=train.MambaStudentConfig(
            vocab_size=7,
            hidden_size=10,
            num_layers=1,
            torch_dtype="float32",
            device="cpu",
            state_extraction="last_hidden",
            off_state_mode="projection",
            delta_alt_mode="delta_projection",
            off_logits_mode="lm_head",
            off_state_detach_direction=True,
        ),
        hf_teacher=train.HuggingFaceRuntimeConfig(
            model_name_or_path="fake-local-hf-teacher",
            torch_dtype="float32",
            device_map="cpu",
            local_files_only=True,
        ),
    )


def test_hf_teacher_real_mamba_training_smoke_uses_fake_modules_no_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeTeacher = _install_fake_transformers(monkeypatch)
    FakeMambaLMHeadModel = _install_fake_public_mamba(monkeypatch)
    logger = RecordingLogger()
    captured_student: dict[str, nn.Module] = {}
    real_build_student = train._build_student

    def capture_student(config: train.TrainConfig, teacher_vocab_size: int, device: torch.device) -> nn.Module:
        student = real_build_student(config, teacher_vocab_size, device)
        captured_student["student"] = student
        return student

    monkeypatch.setattr(train, "_build_student", capture_student)

    train.run_training(_tiny_hf_real_mamba_config(), max_steps=1, logger=logger)  # type: ignore[arg-type]

    record = logger.records[0]
    assert record["step"] == 1
    assert record["optimizer_step"] == 1
    for key in ("total", "ce", "kd", "csdm", "grad_norm"):
        assert key in record
        assert math.isfinite(float(record[key]))

    fake_teacher = FakeTeacher.instances[-1]  # type: ignore[attr-defined]
    assert fake_teacher.forward_grad_enabled is False
    assert fake_teacher.forward_attention_mask is not None
    assert torch.equal(fake_teacher.forward_attention_mask, torch.ones_like(fake_teacher.forward_attention_mask))
    assert fake_teacher.max_input_id is not None and fake_teacher.max_input_id < FAKE_TEACHER_VOCAB_SIZE
    assert all(parameter.grad is None for parameter in fake_teacher.parameters())
    assert all(not parameter.requires_grad for parameter in fake_teacher.parameters())

    fake_mamba = FakeMambaLMHeadModel.instances[-1]  # type: ignore[attr-defined]
    assert fake_mamba.config.vocab_size == FAKE_TEACHER_VOCAB_SIZE
    assert fake_mamba.config.d_model == 10
    assert fake_mamba.backbone.embedding.weight.grad is not None
    assert fake_mamba.backbone.embedding.weight.grad.abs().sum() > 0
    assert fake_mamba.lm_head.weight.grad is not None
    assert fake_mamba.lm_head.weight.grad.abs().sum() > 0
    assert "student" in captured_student


def test_hf_teacher_real_mamba_explicit_vocab_mismatch_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_transformers(monkeypatch)
    _install_fake_public_mamba(monkeypatch)
    config = replace(
        _tiny_hf_real_mamba_config(csdm_weight=0.0),
        student_vocab_size_explicit=True,
        mamba_student=replace(_tiny_hf_real_mamba_config().mamba_student, vocab_size=17),
    )

    with pytest.raises(ValueError, match="vocab sizes must match"):
        train.run_training(config, max_steps=1, logger=RecordingLogger())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mamba_overrides",
    [
        {"state_extraction": "none"},
        {"off_state_mode": "none"},
        {"off_state_mode": "placeholder"},
        {"off_logits_mode": "placeholder"},
        {"delta_alt_mode": "identity"},
        {"delta_alt_mode": "noise"},
    ],
)
def test_real_mamba_csdm_requires_non_placeholder_off_state_before_loading_optional_deps(
    mamba_overrides: dict[str, str],
) -> None:
    config = replace(
        _tiny_hf_real_mamba_config(csdm_weight=0.1),
        mamba_student=replace(
            _tiny_hf_real_mamba_config().mamba_student,
            **mamba_overrides,
        ),
    )

    with pytest.raises(ValueError, match="requires an exposed approximate off-state path"):
        train.run_training(config, max_steps=1, logger=RecordingLogger())  # type: ignore[arg-type]


def test_hf_mamba_cli_csdm_override_survives_hf_smoke_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(ROOT / "configs" / "train_config.yaml"),
            "--teacher-type",
            "hf",
            "--student-type",
            "mamba",
            "--teacher-model-name-or-path",
            "fake-local-hf-teacher",
            "--student-hidden-size",
            "64",
            "--student-num-layers",
            "2",
            "--mixed-precision",
            "no",
            "--csdm-weight",
            "0.03",
            "--max_steps",
            "1",
        ],
    )

    args = train.parse_args()
    config = train.derive_runtime_config(args)

    assert config.teacher_type == "hf"
    assert config.student_type == "mamba"
    assert config.loss.csdm_weight == pytest.approx(0.03)
    assert config.mamba_student.hidden_size == 64
    assert config.mamba_student.num_layers == 2


def _mamba_available() -> bool:
    try:
        importlib.import_module("mamba_ssm")
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _mamba_available(), reason="mamba_ssm is not installed")
def test_hf_teacher_real_mamba_subprocess_smoke_skips_without_real_mamba(tmp_path: Path) -> None:
    fake_transformers = tmp_path / "transformers.py"
    fake_transformers.write_text(
        textwrap.dedent(
            f"""
            import types
            import torch
            from torch import nn

            FAKE_VOCAB_SIZE = {FAKE_TEACHER_VOCAB_SIZE}

            class AutoTokenizer:
                @classmethod
                def from_pretrained(cls, *_args, **_kwargs):
                    return cls()

            class AutoModelForCausalLM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = nn.Parameter(torch.ones(()))
                    self.input_embeddings = nn.Embedding(FAKE_VOCAB_SIZE, 4)
                    self.config = types.SimpleNamespace(vocab_size=FAKE_VOCAB_SIZE)

                @classmethod
                def from_pretrained(cls, *_args, **_kwargs):
                    return cls()

                def get_input_embeddings(self):
                    return self.input_embeddings

                def forward(self, *, input_ids, attention_mask=None):
                    del attention_mask
                    base = torch.arange(FAKE_VOCAB_SIZE, dtype=torch.float32, device=input_ids.device)
                    logits = input_ids.float().unsqueeze(-1) * 0.01 + base.view(1, 1, -1)
                    return types.SimpleNamespace(logits=logits + self.weight.float().view(1, 1, 1))
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--config",
            "configs/train_config.yaml",
            "--teacher-type",
            "hf",
            "--student-type",
            "mamba",
            "--teacher-model-name-or-path",
            "fake-local-hf-teacher",
            "--local-files-only",
            "--max_steps",
            "1",
            "--batch-size",
            "1",
            "--seq-len",
            "4",
            "--gradient-accumulation-steps",
            "1",
            "--mixed-precision",
            "no",
            "--csdm-weight",
            "0.0",
            "--student-vocab-size",
            str(FAKE_TEACHER_VOCAB_SIZE),
            "--student-hidden-size",
            "16",
            "--student-num-layers",
            "1",
            "--student-state-extraction",
            "last_hidden",
            "--off-state-mode",
            "placeholder",
            "--off-logits-mode",
            "placeholder",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    if result.returncode != 0:
        combined = f"{result.stdout}\n{result.stderr}"
        optional_kernel_markers = (
            "mamba-ssm is required",
            "No module named 'mamba_ssm'",
            "causal_conv1d",
            "selective_scan",
            "mamba_inner_fn",
            "not implemented for 'CPU'",
            "not compiled with CUDA",
        )
        if any(marker in combined for marker in optional_kernel_markers):
            pytest.skip(f"real mamba_ssm forward is unsupported in this environment: {combined[-500:]}")
        pytest.fail(combined)

    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert [record["step"] for record in records] == [1]
    for key in ("total", "ce", "kd", "csdm", "grad_norm"):
        assert key in records[0]
        assert math.isfinite(float(records[0][key]))
