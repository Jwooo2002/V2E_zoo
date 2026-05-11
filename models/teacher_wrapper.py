"""Teacher wrappers for mock and HuggingFace causal-LM teachers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


class TeacherWrapper(nn.Module, ABC):
    """Base teacher interface.

    Teachers consume only clean token prefixes. They never receive Mamba
    recurrent states such as ``h_t``, ``h'_t``, or ``h_delta_alt``.
    """

    @abstractmethod
    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Return logits where ``logits[:, t]`` predicts after prefix ``x_{<=t}``."""


class MockTeacherWrapper(TeacherWrapper):
    """Frozen deterministic teacher over token-prefix summaries."""

    def __init__(self, vocab_size: int = 1024, hidden_size: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.prefix_proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}.")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids, "
                f"got {tuple(attention_mask.shape)} and {tuple(input_ids.shape)}."
            )
        with torch.no_grad():
            token_embeddings = self.embedding(input_ids)
            steps = torch.arange(
                1,
                input_ids.shape[1] + 1,
                device=input_ids.device,
                dtype=token_embeddings.dtype,
            ).view(1, -1, 1)
            prefix_state = token_embeddings.cumsum(dim=1) / steps
            hidden = torch.tanh(self.prefix_proj(prefix_state))
            logits = self.lm_head(hidden)
        return logits.detach()


@dataclass(frozen=True)
class HuggingFaceTeacherConfig:
    """Configuration for a frozen HuggingFace causal-LM teacher."""

    model_name_or_path: str
    torch_dtype: str = "bfloat16"
    device_map: str | None = "auto"
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    local_files_only: bool = False
    use_safetensors: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False


def parse_torch_dtype(torch_dtype: str, device_map: str | None = "auto") -> torch.dtype:
    """Parse config dtype strings, falling back to fp32 for CPU execution."""

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        dtype = dtype_map[torch_dtype]
    except KeyError as exc:
        allowed = ", ".join(sorted(dtype_map))
        raise ValueError(f"Unsupported torch_dtype {torch_dtype!r}; expected one of: {allowed}.") from exc

    cpu_device_map = device_map is None or device_map == "cpu"
    likely_cpu = cpu_device_map or (device_map == "auto" and not torch.cuda.is_available())
    if likely_cpu and dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _load_transformers_classes() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "HuggingFaceTeacherWrapper requires the optional 'transformers' package. "
            "Install transformers or keep using the mock teacher config."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def _looks_like_missing_safetensors(exc: Exception) -> bool:
    message = str(exc).lower()
    if "safetensor" not in message:
        return False
    missing_markers = (
        "no file",
        "not found",
        "does not appear",
        "doesn't appear",
        "missing",
        "cannot find",
        "could not find",
    )
    return any(marker in message for marker in missing_markers)


class HuggingFaceTeacherWrapper(TeacherWrapper):
    """Frozen HuggingFace causal-LM teacher over clean token prefixes only."""

    def __init__(self, config: HuggingFaceTeacherConfig) -> None:
        super().__init__()
        if config.load_in_8bit and config.load_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit cannot both be True.")

        self.config = config
        AutoModelForCausalLM, AutoTokenizer = _load_transformers_classes()

        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": config.trust_remote_code,
            "local_files_only": config.local_files_only,
        }
        model_kwargs: dict[str, Any] = {
            "torch_dtype": parse_torch_dtype(config.torch_dtype, config.device_map),
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
            "local_files_only": config.local_files_only,
            "use_safetensors": config.use_safetensors,
        }
        if config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if config.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        if config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = config.attn_implementation

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, **tokenizer_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
        except Exception as exc:
            if config.use_safetensors and _looks_like_missing_safetensors(exc):
                raise RuntimeError(
                    "Failed to load HuggingFace teacher with use_safetensors=True because "
                    f"{config.model_name_or_path!r} does not appear to provide safetensors weights. "
                    "Try a model with safetensors; set use_safetensors=false only if torch>=2.6 "
                    "and the model source is trusted, or upgrade to torch>=2.6 before loading "
                    "legacy PyTorch checkpoints."
                ) from exc
            raise RuntimeError(
                "Failed to load HuggingFace teacher model/tokenizer from "
                f"{config.model_name_or_path!r}. Check authentication, local_files_only, "
                "model path, and optional quantization dependencies."
            ) from exc

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}.")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids, "
                f"got {tuple(attention_mask.shape)} and {tuple(input_ids.shape)}."
            )
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits.detach()
