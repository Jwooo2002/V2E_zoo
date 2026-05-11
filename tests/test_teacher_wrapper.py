from __future__ import annotations

import builtins
import importlib
import inspect
import sys
import types
from dataclasses import replace

import pytest
import torch
from torch import nn

import models.teacher_wrapper as teacher_wrapper_module
from models.teacher_wrapper import (
    HuggingFaceTeacherConfig,
    HuggingFaceTeacherWrapper,
    MockTeacherWrapper,
    TeacherWrapper,
    parse_torch_dtype,
)


def _install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], type[object], type[nn.Module]]:
    fake_transformers = types.ModuleType("transformers")
    calls: dict[str, object] = {}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name_or_path: str, **kwargs: object) -> "FakeTokenizer":
            calls["tokenizer"] = (model_name_or_path, kwargs)
            return cls()

    class FakeModel(nn.Module):
        instances: list["FakeModel"] = []

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.eval_called = False
            self.forward_grad_enabled: bool | None = None
            self.forward_attention_mask: torch.Tensor | None = None
            FakeModel.instances.append(self)

        @classmethod
        def from_pretrained(cls, model_name_or_path: str, **kwargs: object) -> "FakeModel":
            calls["model"] = (model_name_or_path, kwargs)
            return cls()

        def eval(self) -> "FakeModel":
            self.eval_called = True
            return super().eval()

        def forward(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> object:
            self.forward_grad_enabled = torch.is_grad_enabled()
            self.forward_attention_mask = attention_mask
            logits = torch.ones(input_ids.shape[0], input_ids.shape[1], 19, requires_grad=True)
            return types.SimpleNamespace(logits=logits)

    fake_transformers.AutoModelForCausalLM = FakeModel
    fake_transformers.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return calls, FakeTokenizer, FakeModel


def test_mock_teacher_returns_frozen_no_grad_logits_with_attention_mask() -> None:
    teacher = MockTeacherWrapper(vocab_size=17, hidden_size=8)
    input_ids = torch.randint(0, 17, (2, 5))
    attention_mask = torch.ones_like(input_ids)

    logits = teacher(input_ids, attention_mask=attention_mask)

    assert logits.shape == (2, 5, 17)
    assert not logits.requires_grad
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_dtype_parser_maps_supported_strings_and_cpu_fallback() -> None:
    assert parse_torch_dtype("float32") is torch.float32
    assert parse_torch_dtype("float16", device_map="cuda") is torch.float16
    assert parse_torch_dtype("bfloat16", device_map="cuda") is torch.bfloat16
    assert parse_torch_dtype("float16", device_map="cpu") is torch.float32
    assert parse_torch_dtype("bfloat16", device_map=None) is torch.float32

    with pytest.raises(ValueError, match="Unsupported torch_dtype"):
        parse_torch_dtype("float64")


def test_importing_teacher_wrapper_does_not_import_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("transformers", None)

    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "transformers":
            raise AssertionError("transformers should not be imported at module import time")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.reload(teacher_wrapper_module)

    assert "transformers" not in sys.modules


def test_importing_models_package_does_not_import_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("transformers", None)
    sys.modules.pop("models", None)

    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "transformers":
            raise AssertionError("transformers should not be imported by models package import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    models_module = importlib.import_module("models")

    assert hasattr(models_module, "HuggingFaceTeacherWrapper")
    assert "transformers" not in sys.modules


def test_forward_signatures_are_token_only_and_reject_mamba_state_names() -> None:
    expected = ["self", "input_ids", "attention_mask"]
    assert list(inspect.signature(TeacherWrapper.forward).parameters) == expected
    assert list(inspect.signature(MockTeacherWrapper.forward).parameters) == expected
    assert list(inspect.signature(HuggingFaceTeacherWrapper.forward).parameters) == expected

    teacher = MockTeacherWrapper(vocab_size=11, hidden_size=4)
    with pytest.raises(TypeError):
        teacher(torch.randint(0, 11, (1, 3)), h_t=torch.zeros(1, 3, 4))  # type: ignore[call-arg]


def test_hf_wrapper_import_error_is_informative(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def missing_transformers(name: str, *args: object, **kwargs: object) -> object:
        if name == "transformers":
            raise ImportError("no transformers here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_transformers)
    sys.modules.pop("transformers", None)

    with pytest.raises(ImportError, match="requires the optional 'transformers' package"):
        HuggingFaceTeacherWrapper(HuggingFaceTeacherConfig(model_name_or_path="local-test-model"))


def test_hf_wrapper_loads_fake_transformers_and_returns_detached_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, FakeTokenizer, FakeModel = _install_fake_transformers(monkeypatch)

    config = HuggingFaceTeacherConfig(
        model_name_or_path="tiny-local-model",
        torch_dtype="float16",
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        local_files_only=True,
        load_in_8bit=True,
    )
    wrapper = HuggingFaceTeacherWrapper(config)
    fake_model = FakeModel.instances[-1]

    tokenizer_name, tokenizer_kwargs = calls["tokenizer"]  # type: ignore[misc]
    model_name, model_kwargs = calls["model"]  # type: ignore[misc]
    assert tokenizer_name == "tiny-local-model"
    assert tokenizer_kwargs == {"trust_remote_code": True, "local_files_only": True}
    assert model_name == "tiny-local-model"
    assert model_kwargs == {
        "torch_dtype": torch.float16,
        "device_map": "cuda",
        "trust_remote_code": True,
        "local_files_only": True,
        "load_in_8bit": True,
        "attn_implementation": "flash_attention_2",
    }
    assert wrapper.tokenizer.__class__ is FakeTokenizer
    assert wrapper.model is fake_model
    assert fake_model.eval_called
    assert all(not parameter.requires_grad for parameter in fake_model.parameters())

    input_ids = torch.randint(0, 13, (2, 4))
    attention_mask = torch.ones_like(input_ids)
    logits = wrapper(input_ids, attention_mask=attention_mask)

    assert fake_model.forward_grad_enabled is False
    assert fake_model.forward_attention_mask is attention_mask
    assert logits.shape == (2, 4, 19)
    assert not logits.requires_grad


def test_hf_wrapper_omits_non_quantized_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, _, _ = _install_fake_transformers(monkeypatch)

    config = HuggingFaceTeacherConfig(
        model_name_or_path="tiny-local-model",
        torch_dtype="float32",
        device_map="cpu",
    )
    HuggingFaceTeacherWrapper(config)

    _, model_kwargs = calls["model"]  # type: ignore[misc]
    assert "load_in_8bit" not in model_kwargs
    assert "load_in_4bit" not in model_kwargs


def test_hf_wrapper_passes_8bit_quantization_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _, _ = _install_fake_transformers(monkeypatch)

    config = HuggingFaceTeacherConfig(
        model_name_or_path="tiny-local-model",
        torch_dtype="float32",
        device_map="cpu",
        load_in_8bit=True,
    )
    HuggingFaceTeacherWrapper(config)

    _, model_kwargs = calls["model"]  # type: ignore[misc]
    assert model_kwargs["load_in_8bit"] is True
    assert "load_in_4bit" not in model_kwargs


def test_hf_wrapper_passes_4bit_quantization_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _, _ = _install_fake_transformers(monkeypatch)

    config = HuggingFaceTeacherConfig(
        model_name_or_path="tiny-local-model",
        torch_dtype="float32",
        device_map="cpu",
        load_in_4bit=True,
    )
    HuggingFaceTeacherWrapper(config)

    _, model_kwargs = calls["model"]  # type: ignore[misc]
    assert "load_in_8bit" not in model_kwargs
    assert model_kwargs["load_in_4bit"] is True


def test_hf_wrapper_rejects_both_quantization_modes_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _, _ = _install_fake_transformers(monkeypatch)

    config = HuggingFaceTeacherConfig(
        model_name_or_path="tiny-local-model",
        load_in_8bit=True,
        load_in_4bit=True,
    )

    with pytest.raises(ValueError, match="load_in_8bit and load_in_4bit cannot both be True"):
        HuggingFaceTeacherWrapper(config)
    assert calls == {}


def test_hf_wrapper_wraps_loading_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_transformers = types.ModuleType("transformers")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> "FakeTokenizer":
            return cls()

    class FailingModel:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
            raise OSError("auth or local files failure")

    fake_transformers.AutoModelForCausalLM = FailingModel
    fake_transformers.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    config = replace(HuggingFaceTeacherConfig(model_name_or_path="gated-model"), local_files_only=True)
    with pytest.raises(RuntimeError, match="Failed to load HuggingFace teacher"):
        HuggingFaceTeacherWrapper(config)
