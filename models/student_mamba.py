"""Mock and real-Mamba student adapter scaffolds."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from contextlib import contextmanager
import importlib
from types import ModuleType
from typing import Any, Iterator

import torch
from torch import Tensor, nn

from models.cdm_engine import DeltaPerturbationEngine, OffTrajectoryConfig


@dataclass(frozen=True)
class StudentOutput:
    on_logits: Tensor
    off_logits: Tensor
    fake_logits: Tensor
    h: Tensor
    h_off: Tensor
    h_delta_alt: Tensor


@dataclass(frozen=True)
class MambaStudentConfig:
    """Configuration for the optional real Mamba student scaffold."""

    model_name_or_path: str | None = None
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int | None = None
    state_size: int | None = None
    torch_dtype: str = "bfloat16"
    device: str | None = None
    trust_remote_code: bool = False
    use_pretrained: bool = False
    local_files_only: bool = False
    delta_perturb_eps: float = 0.10
    noise_sigma: float = 0.01
    use_reference_forward: bool = False


class StudentMamba(nn.Module, ABC):
    """Base student interface for future real Mamba integrations."""

    @abstractmethod
    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> StudentOutput:
        """Return on/off-trajectory student logits."""


class MockStudentMamba(StudentMamba):
    """Small recurrent student with a student-only off-state surrogate.

    ``h_delta_alt`` is produced by a lightweight projection of the student
    hidden state. It is only a mock delta-transition surrogate and does not
    represent real Mamba internals.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        hidden_size: int = 256,
        delta_scale: float = 0.1,
        off_config: OffTrajectoryConfig | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.sequence = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.delta_perturb_proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.delta_scale = delta_scale
        self.off_engine = DeltaPerturbationEngine(off_config)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> StudentOutput:
        del attention_mask
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}.")
        embeddings = self.embedding(input_ids)
        h, _ = self.sequence(embeddings)
        on_logits = self.lm_head(h)

        h_delta_alt = h + self.delta_scale * torch.tanh(self.delta_perturb_proj(h))
        h_delta_alt = h_delta_alt.to(device=h.device, dtype=h.dtype)
        h_off = self.off_engine.make_off_state(h, h_delta_alt=h_delta_alt)
        off_logits = self.lm_head(h_off)

        with torch.no_grad():
            fake_logits = self.lm_head(h_off.detach()).detach()

        return StudentOutput(
            on_logits=on_logits,
            off_logits=off_logits,
            fake_logits=fake_logits,
            h=h,
            h_off=h_off,
            h_delta_alt=h_delta_alt,
        )


class RealMambaStudent(StudentMamba):
    """Opt-in real Mamba student adapter for Stage 6C forward smoke.

    This class imports ``mamba_ssm`` lazily and uses public Mamba classes only.
    Stage 6C verifies tiny model instantiation and logit-shape-compatible
    forward output. It does not implement real ``h_delta_alt`` extraction,
    off-trajectory state construction, or CSDM training with real Mamba.
    ``off_logits`` mirrors ``on_logits`` as a smoke-only placeholder.
    """

    def __init__(
        self,
        config: MambaStudentConfig | None = None,
        off_config: OffTrajectoryConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MambaStudentConfig()
        self.off_config = off_config or OffTrajectoryConfig(
            delta_perturb_eps=self.config.delta_perturb_eps,
            noise_sigma=self.config.noise_sigma,
        )
        self._device = torch.device(self.config.device) if self.config.device is not None else torch.device("cpu")
        self._dtype = _resolve_torch_dtype(self.config.torch_dtype, self._device)
        try:
            self.mamba_ssm: ModuleType = importlib.import_module("mamba_ssm")
        except ImportError as exc:
            raise ImportError(
                "mamba-ssm is required for RealMambaStudent. "
                "Install it only when running real Mamba experiments."
            ) from exc

        self.model_kind = "MambaLMHeadModel"
        self.model = self._build_lm_head_model()
        self.to(device=self._device, dtype=self._dtype)

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def mamba_ssm_version(self) -> str | None:
        version = getattr(self.mamba_ssm, "__version__", None)
        return str(version) if version is not None else None

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> StudentOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}.")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids, "
                f"got {tuple(attention_mask.shape)} and {tuple(input_ids.shape)}."
            )
        del attention_mask

        input_ids = input_ids.to(self._device)
        try:
            hidden, on_logits = self._forward_backbone_logits(
                input_ids,
                reference_causal_conv=self._device.type == "cpu" or self.config.use_reference_forward,
                reference_selective_scan=self._device.type == "cpu" or self.config.use_reference_forward,
            )
        except (RuntimeError, TypeError) as exc:
            if self._device.type != "cuda" or self.config.use_reference_forward or not _is_causal_conv1d_api_error(exc):
                raise
            hidden, on_logits = self._forward_backbone_logits(
                input_ids,
                reference_causal_conv=True,
                reference_selective_scan=False,
            )
        _validate_logits(on_logits, input_ids=input_ids, vocab_size=self.config.vocab_size)

        # Stage 6C smoke placeholder only. Real h'_t / h_delta_alt extraction is
        # intentionally deferred to Stage 6D/6E.
        h = hidden
        h_off = hidden
        h_delta_alt = hidden
        off_logits = on_logits
        fake_logits = on_logits.detach()
        return StudentOutput(
            on_logits=on_logits,
            off_logits=off_logits,
            fake_logits=fake_logits,
            h=h,
            h_off=h_off,
            h_delta_alt=h_delta_alt,
        )

    def _forward_backbone_logits(
        self,
        input_ids: Tensor,
        *,
        reference_causal_conv: bool,
        reference_selective_scan: bool,
    ) -> tuple[Tensor, Tensor]:
        with _mamba_reference_kernel_patch(
            reference_causal_conv=reference_causal_conv,
            reference_selective_scan=reference_selective_scan,
        ):
            hidden = self.model.backbone(input_ids)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            on_logits = self.model.lm_head(hidden)
        return hidden, on_logits

    def _build_lm_head_model(self) -> nn.Module:
        lm_cls = _required_mamba_lm_head_model_class()
        if self.config.use_pretrained:
            if self.config.model_name_or_path is None:
                raise ValueError("model_name_or_path is required when use_pretrained=True.")
            if not hasattr(lm_cls, "from_pretrained"):
                raise NotImplementedError(
                    "RealMambaStudent pretrained loading requires public "
                    "MambaLMHeadModel.from_pretrained."
                )
            kwargs: dict[str, Any] = {"device": str(self._device), "dtype": self._dtype}
            if self.config.local_files_only:
                kwargs["local_files_only"] = True
            try:
                return lm_cls.from_pretrained(self.config.model_name_or_path, **kwargs)
            except TypeError as exc:
                if self.config.local_files_only:
                    raise RuntimeError(
                        "RealMambaStudent local_files_only=True could not be passed to "
                        "MambaLMHeadModel.from_pretrained. Refusing to retry without it."
                    ) from exc
                kwargs.pop("local_files_only", None)
                return lm_cls.from_pretrained(self.config.model_name_or_path, **kwargs)

        config_cls = _required_mamba_config_class()
        mamba_config = _instantiate_mamba_config(config_cls, self.config, self._device)
        return lm_cls(mamba_config, device=str(self._device), dtype=self._dtype)


def _resolve_torch_dtype(torch_dtype: str, device: torch.device) -> torch.dtype:
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
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _required_mamba_lm_head_model_class() -> type[nn.Module]:
    candidates = (
        ("mamba_ssm.models.mixer_seq_simple", "MambaLMHeadModel"),
        ("mamba_ssm", "MambaLMHeadModel"),
    )
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        cls = getattr(module, class_name, None)
        if cls is not None:
            return cls
    raise NotImplementedError(
        "RealMambaStudent requires public MambaLMHeadModel from mamba_ssm for Stage 6C smoke."
    )


def _required_mamba_config_class() -> type[Any]:
    candidates = (
        ("mamba_ssm.models.config_mamba", "MambaConfig"),
        ("mamba_ssm.models.mixer_seq_simple", "MambaConfig"),
        ("mamba_ssm", "MambaConfig"),
    )
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        cls = getattr(module, class_name, None)
        if cls is not None:
            return cls
    raise NotImplementedError("RealMambaStudent requires public MambaConfig from mamba_ssm.")


def _instantiate_mamba_config(config_cls: type[Any], config: MambaStudentConfig, device: torch.device) -> Any:
    n_layer = config.num_layers if config.num_layers is not None else 1
    if n_layer <= 0:
        raise ValueError(f"num_layers must be positive, got {n_layer}.")
    kwargs: dict[str, Any] = {
        "d_model": config.hidden_size,
        "n_layer": n_layer,
        "vocab_size": config.vocab_size,
        "fused_add_norm": device.type == "cuda" and not config.use_reference_forward,
        "rms_norm": device.type == "cuda" and not config.use_reference_forward,
        "pad_vocab_size_multiple": 1,
    }
    if config.state_size is not None:
        kwargs["ssm_cfg"] = {"d_state": config.state_size}
    return config_cls(**kwargs)


_MISSING_ATTR = object()


def _is_causal_conv1d_api_error(exc: BaseException) -> bool:
    message = str(exc)
    return "causal_conv1d" in message and (
        "incompatible function arguments" in message
        or "invalid combination of arguments" in message
        or "takes" in message
        or "got an unexpected keyword argument" in message
    )


@contextmanager
def _mamba_reference_kernel_patch(*, reference_causal_conv: bool, reference_selective_scan: bool) -> Iterator[None]:
    """Temporarily use public reference kernels for smoke forward.

    Some mamba-ssm builds expose CUDA-only causal-conv1d/selective-scan fast
    paths even when tensors are on CPU, and some mamba-ssm / causal-conv1d
    version pairs have incompatible fused CUDA signatures. Stage 6C only needs
    an import/forward smoke, so this temporarily switches public module globals
    to reference paths when mamba-ssm provides them.
    """

    if not reference_causal_conv and not reference_selective_scan:
        yield
        return
    try:
        mamba_simple = importlib.import_module("mamba_ssm.modules.mamba_simple")
    except ImportError:
        yield
        return
    selective_scan_interface = None
    if reference_selective_scan:
        try:
            selective_scan_interface = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
        except ImportError:
            selective_scan_interface = None
    original_selective_scan_fn = (
        getattr(mamba_simple, "selective_scan_fn") if hasattr(mamba_simple, "selective_scan_fn") else _MISSING_ATTR
    )
    original_causal_conv1d_fn = (
        getattr(mamba_simple, "causal_conv1d_fn") if hasattr(mamba_simple, "causal_conv1d_fn") else _MISSING_ATTR
    )
    if selective_scan_interface is not None and hasattr(selective_scan_interface, "selective_scan_ref"):
        setattr(mamba_simple, "selective_scan_fn", selective_scan_interface.selective_scan_ref)
    if reference_causal_conv:
        setattr(mamba_simple, "causal_conv1d_fn", None)
    try:
        yield
    finally:
        _restore_module_attr(mamba_simple, "selective_scan_fn", original_selective_scan_fn)
        _restore_module_attr(mamba_simple, "causal_conv1d_fn", original_causal_conv1d_fn)


def _restore_module_attr(module: ModuleType, name: str, value: object) -> None:
    if value is _MISSING_ATTR:
        if hasattr(module, name):
            delattr(module, name)
        return
    setattr(module, name, value)


def _validate_logits(logits: Tensor, input_ids: Tensor, vocab_size: int) -> None:
    expected = (*input_ids.shape, vocab_size)
    if logits.shape != expected:
        raise RuntimeError(f"RealMambaStudent logits must have shape {expected}, got {tuple(logits.shape)}.")
