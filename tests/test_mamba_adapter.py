from __future__ import annotations

import argparse
import importlib
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from models.student_mamba import MambaStudentConfig, MockStudentMamba, RealMambaStudent


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SPEC = importlib.util.spec_from_file_location("cdm_mamba_kd_train_mamba_tests", ROOT / "train.py")
assert TRAIN_SPEC is not None
assert TRAIN_SPEC.loader is not None
train = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train
TRAIN_SPEC.loader.exec_module(train)


def test_student_mamba_module_imports_without_mamba_ssm() -> None:
    assert "models.student_mamba" in sys.modules


def test_mock_student_mamba_shapes_and_optional_attention_mask() -> None:
    student = MockStudentMamba(vocab_size=31, hidden_size=12)
    input_ids = torch.randint(0, 31, (2, 7))
    attention_mask = torch.ones_like(input_ids)

    output = student(input_ids, attention_mask=attention_mask)

    assert output.on_logits.shape == (2, 7, 31)
    assert output.off_logits.shape == output.on_logits.shape
    assert output.fake_logits.shape == output.off_logits.shape
    assert output.h.shape == (2, 7, 12)
    assert output.h_off.shape == output.h.shape
    assert output.h_delta_alt.shape == output.h.shape
    assert not output.fake_logits.requires_grad


def test_mamba_student_config_defaults() -> None:
    config = MambaStudentConfig()

    assert config.model_name_or_path is None
    assert config.vocab_size == 50257
    assert config.hidden_size == 768
    assert config.num_layers is None
    assert config.state_size is None
    assert config.torch_dtype == "bfloat16"
    assert config.device is None
    assert config.trust_remote_code is False
    assert config.use_pretrained is False
    assert config.local_files_only is False
    assert config.delta_perturb_eps == pytest.approx(0.10)
    assert config.noise_sigma == pytest.approx(0.01)


def test_real_mamba_student_missing_dependency_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> types.ModuleType:
        if name == "mamba_ssm":
            raise ImportError("missing mamba")
        return real_import_module(name, package)

    monkeypatch.setattr("models.student_mamba.importlib.import_module", fake_import_module)

    with pytest.raises(ImportError, match="mamba-ssm is required for RealMambaStudent"):
        RealMambaStudent()


def test_real_mamba_student_fake_dependency_without_public_model_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module
    fake_mamba = types.ModuleType("mamba_ssm")

    def fake_import_module(name: str, package: str | None = None) -> types.ModuleType:
        if name == "mamba_ssm":
            return fake_mamba
        if name.startswith("mamba_ssm."):
            raise ImportError(f"missing public mamba module {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(
        "models.student_mamba.importlib.import_module",
        fake_import_module,
    )
    with pytest.raises(NotImplementedError, match="MambaLMHeadModel"):
        RealMambaStudent(MambaStudentConfig(vocab_size=17, hidden_size=8))


def test_train_mock_path_does_not_instantiate_real_mamba(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_real_mamba(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("RealMambaStudent should not be instantiated for mock training")

    monkeypatch.setattr(train, "RealMambaStudent", fail_real_mamba)
    config = replace(
        train.load_train_config(ROOT / "configs" / "train_config.yaml"),
        gradient_accumulation_steps=1,
        mixed_precision="no",
        mock=train.MockConfig(
            batch_size=1,
            seq_len=5,
            vocab_size=19,
            hidden_size=8,
            num_samples=1,
            positions_per_sequence=3,
            ignore_index=-100,
        ),
    )

    train.run_training(config, max_steps=1)


def test_cli_student_type_mamba_parses_without_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(ROOT / "configs" / "train_config.yaml"),
            "--student-type",
            "mamba",
            "--student-model-name-or-path",
            "local-mamba",
            "--student-vocab-size",
            "101",
            "--student-hidden-size",
            "64",
            "--local-files-only",
            "--max_steps",
            "1",
        ],
    )

    args = train.parse_args()
    config = train.derive_runtime_config(args)

    assert args.student_type == "mamba"
    assert config.student_type == "mamba"
    assert config.mamba_student.model_name_or_path == "local-mamba"
    assert config.mamba_student.vocab_size == 101
    assert config.mamba_student.hidden_size == 64
    assert config.mamba_student.local_files_only is True


def test_train_mamba_path_with_fake_dependency_raises_clear_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module
    fake_mamba = types.ModuleType("mamba_ssm")

    def fake_import_module(name: str, package: str | None = None) -> types.ModuleType:
        if name == "mamba_ssm":
            return fake_mamba
        if name.startswith("mamba_ssm."):
            raise ImportError(f"missing public mamba module {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(
        "models.student_mamba.importlib.import_module",
        fake_import_module,
    )
    config = replace(
        train.load_train_config(ROOT / "configs" / "train_config.yaml"),
        student_type="mamba",
        gradient_accumulation_steps=1,
        mixed_precision="no",
        mock=train.MockConfig(
            batch_size=1,
            seq_len=5,
            vocab_size=19,
            hidden_size=8,
            num_samples=1,
            positions_per_sequence=3,
            ignore_index=-100,
        ),
    )

    with pytest.raises((ImportError, NotImplementedError), match="RealMambaStudent|MambaLMHeadModel|mamba"):
        train.run_training(config, max_steps=1)


def test_derive_runtime_config_default_mock_does_not_instantiate_real_mamba(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_real_mamba(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("RealMambaStudent should not be instantiated during config parsing")

    monkeypatch.setattr(train, "RealMambaStudent", fail_real_mamba)
    args = argparse.Namespace(
        config=ROOT / "configs" / "train_config.yaml",
        mock=False,
        teacher_type=None,
        student_type=None,
        teacher_model_name_or_path=None,
        student_model_name_or_path=None,
        student_vocab_size=None,
        student_hidden_size=None,
        seq_len=None,
        batch_size=None,
        gradient_accumulation_steps=None,
        mixed_precision=None,
        csdm_weight=None,
        kd_weight=None,
        ce_weight=None,
        local_files_only=False,
        topk_enabled=None,
        top_k=None,
        topk_include_labels=None,
        topk_renormalize=None,
        teacher_cache_enabled=None,
        teacher_cache_dir=None,
        teacher_cache_overwrite=None,
        teacher_cache_use_top_k=None,
        teacher_cache_top_k=None,
    )

    config = train.derive_runtime_config(args)

    assert config.student_type == "mock"
