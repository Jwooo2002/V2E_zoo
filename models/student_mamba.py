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
    h: Tensor | None
    h_off: Tensor | None
    h_delta_alt: Tensor | None
    metadata: dict[str, Any] | None = None


_STATE_EXTRACTION_MODES = frozenset({"last_hidden", "embedding", "none"})
_OFF_STATE_MODES = frozenset({"projection", "placeholder", "none"})
_DELTA_ALT_MODES = frozenset({"delta_projection", "noise", "identity"})
_OFF_LOGITS_MODES = frozenset({"lm_head", "projection_head", "placeholder"})


def _validate_mode(name: str, value: str, allowed_values: frozenset[str]) -> None:
    if value not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(f"Unsupported {name} {value!r}; expected one of: {allowed}.")


def _shape_or_none(tensor: Tensor | None) -> list[int] | None:
    return None if tensor is None else list(tensor.shape)


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
    state_extraction: str = "last_hidden"
    expose_states: bool = True
    off_state_mode: str = "projection"
    delta_alt_mode: str = "delta_projection"
    off_logits_mode: str = "lm_head"
    off_state_detach_direction: bool = True

    def __post_init__(self) -> None:
        _validate_mode("state_extraction", self.state_extraction, _STATE_EXTRACTION_MODES)
        _validate_mode("off_state_mode", self.off_state_mode, _OFF_STATE_MODES)
        _validate_mode("delta_alt_mode", self.delta_alt_mode, _DELTA_ALT_MODES)
        _validate_mode("off_logits_mode", self.off_logits_mode, _OFF_LOGITS_MODES)


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
            metadata={
                "adapter": "mock",
                "off_state_mode": "projection",
                "delta_alt_mode": "delta_projection",
                "off_logits_mode": "lm_head",
                "off_state_available": True,
                "delta_alt_available": True,
                "smoke_placeholder_off_logits": False,
                "off_logits_placeholder": False,
                "fake_logits_detached": not fake_logits.requires_grad,
            },
        )


class RealMambaStudent(StudentMamba):
    """Opt-in real Mamba student adapter for Stage 6C-6E smoke checks.

    This class imports ``mamba_ssm`` lazily and uses public Mamba classes only.
    Stage 6C verifies tiny model instantiation and logit-shape-compatible
    forward output. Stage 6D exposes a student-side representation ``h`` using
    either the public backbone output or a documented token-embedding fallback
    for shape plumbing. Stage 6E adds an approximate student-side off-state
    path. It does not implement true Mamba delta-kernel perturbation or
    recurrent-state injection; those remain future work.
    """

    def __init__(
        self,
        config: MambaStudentConfig | None = None,
        off_config: OffTrajectoryConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MambaStudentConfig()
        base_off_config = off_config or OffTrajectoryConfig(
            delta_perturb_eps=self.config.delta_perturb_eps,
            noise_sigma=self.config.noise_sigma,
        )
        self.off_config = OffTrajectoryConfig(
            delta_perturb_eps=base_off_config.delta_perturb_eps,
            noise_sigma=base_off_config.noise_sigma,
            rho_min=base_off_config.rho_min,
            rho_max=base_off_config.rho_max,
            detach_direction=self.config.off_state_detach_direction,
            eps=base_off_config.eps,
        )
        self.off_engine = DeltaPerturbationEngine(self.off_config)
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
        self.delta_projection = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.delta_perturb_proj = self.delta_projection
        self.off_projection_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
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

        h = self._extract_state(input_ids=input_ids, last_hidden=hidden)
        h_delta_alt, h_off, off_logits, metadata = self._build_off_trajectory_outputs(
            h=h,
            on_logits=on_logits,
            input_ids=input_ids,
        )
        fake_logits = off_logits.detach()
        exposed_h = h if self.config.expose_states else None
        exposed_h_off = h_off if self.config.expose_states else None
        exposed_h_delta_alt = h_delta_alt if self.config.expose_states else None
        metadata.update(
            {
                "state_extraction": self.config.state_extraction,
                "expose_states": self.config.expose_states,
                "off_state_detach_direction": self.config.off_state_detach_direction,
                "fake_logits_detached": not fake_logits.requires_grad,
                "h_shape": _shape_or_none(exposed_h),
                "h_off_shape": _shape_or_none(exposed_h_off),
                "h_delta_alt_shape": _shape_or_none(exposed_h_delta_alt),
            }
        )
        return StudentOutput(
            on_logits=on_logits,
            off_logits=off_logits,
            fake_logits=fake_logits,
            h=exposed_h,
            h_off=exposed_h_off,
            h_delta_alt=exposed_h_delta_alt,
            metadata=metadata,
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
            lm_head = getattr(self.model, "lm_head", None)
            on_logits = self.off_projection_head(hidden) if lm_head is None else lm_head(hidden)
        return hidden, on_logits

    def _extract_state(self, *, input_ids: Tensor, last_hidden: Tensor) -> Tensor | None:
        """Return the Stage 6D/6E student-side state scaffold.

        ``last_hidden`` is the preferred smoke representation because it is the
        public Mamba backbone output used by the LM head. ``embedding`` is a
        provisional fallback for shape plumbing only and is not claimed to be a
        final recurrent Mamba state.
        """

        if self.config.state_extraction == "none":
            return None
        if self.config.state_extraction == "last_hidden":
            return last_hidden
        if self.config.state_extraction == "embedding":
            return self._embedding_state(input_ids)
        raise AssertionError(f"Unhandled state_extraction mode: {self.config.state_extraction!r}")

    def _embedding_state(self, input_ids: Tensor) -> Tensor:
        backbone = getattr(self.model, "backbone", None)
        embedding = getattr(backbone, "embedding", None)
        if embedding is None and hasattr(self.model, "get_input_embeddings"):
            embedding = self.model.get_input_embeddings()
        if embedding is None:
            raise NotImplementedError(
                "state_extraction='embedding' requires a public embedding module "
                "on the Mamba model. It is a provisional Stage 6D fallback only."
            )
        state = embedding(input_ids)
        if not isinstance(state, Tensor):
            raise RuntimeError("Mamba embedding state extraction did not return a tensor.")
        return state

    def _build_off_trajectory_outputs(
        self,
        *,
        h: Tensor | None,
        on_logits: Tensor,
        input_ids: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, Tensor, dict[str, Any]]:
        """Build the Stage 6E approximate student-side off-state path."""

        metadata: dict[str, Any] = {
            "adapter": "real_mamba",
            "off_state_mode": self.config.off_state_mode,
            "delta_alt_mode": self.config.delta_alt_mode,
            "off_logits_mode": self.config.off_logits_mode,
            "smoke_placeholder_off_logits": False,
            "off_logits_placeholder": False,
            "off_logits_source": None,
            "off_state_source": None,
            "delta_alt_source": None,
            "off_state_available": False,
            "delta_alt_available": False,
        }

        if h is None:
            metadata.update(
                {
                    "smoke_placeholder_off_logits": True,
                    "off_logits_placeholder": True,
                    "off_logits_source": "placeholder_no_state",
                    "off_state_source": "no_exposed_state",
                }
            )
            return None, None, on_logits, metadata

        if self.config.off_state_mode == "none":
            metadata.update(
                {
                    "smoke_placeholder_off_logits": True,
                    "off_logits_placeholder": True,
                    "off_logits_source": "placeholder_no_off_state",
                    "off_state_source": "none",
                }
            )
            return None, None, on_logits, metadata

        if self.config.off_state_mode == "placeholder":
            h_delta_alt = h
            h_off = h
            metadata.update(
                {
                    "smoke_placeholder_off_logits": True,
                    "off_logits_placeholder": True,
                    "off_logits_source": "placeholder",
                    "off_state_source": "placeholder",
                    "delta_alt_source": "identity",
                    "off_state_available": True,
                    "delta_alt_available": True,
                }
            )
            return h_delta_alt, h_off, on_logits, metadata
        else:
            self._validate_state_for_off_path(h, input_ids=input_ids)
            h_delta_alt = self._build_delta_alt(h)
            h_off = self.off_engine.make_off_state(h, h_delta_alt=h_delta_alt)
            metadata.update(
                {
                    "off_state_source": "delta_perturbation_engine",
                    "delta_alt_source": self.config.delta_alt_mode,
                    "off_state_available": True,
                    "delta_alt_available": True,
                }
            )

        if self.config.off_logits_mode == "placeholder":
            metadata.update(
                {
                    "smoke_placeholder_off_logits": True,
                    "off_logits_placeholder": True,
                    "off_logits_source": "placeholder",
                }
            )
            return h_delta_alt, h_off, on_logits, metadata

        self._validate_state_for_off_path(h_off, input_ids=input_ids)
        off_logits, source = self._logits_from_off_state(h_off)
        _validate_logits(off_logits, input_ids=input_ids, vocab_size=self.config.vocab_size)
        metadata["off_logits_source"] = source
        return h_delta_alt, h_off, off_logits, metadata

    def _build_delta_alt(self, h: Tensor) -> Tensor:
        if self.config.delta_alt_mode == "delta_projection":
            return h + self.off_config.delta_perturb_eps * torch.tanh(self.delta_projection(h))
        if self.config.delta_alt_mode == "noise":
            rms_h = torch.sqrt(h.float().pow(2).mean(dim=-1, keepdim=True) + self.off_config.eps).to(dtype=h.dtype)
            return h + self.off_config.noise_sigma * rms_h * torch.randn_like(h)
        if self.config.delta_alt_mode == "identity":
            return h
        raise AssertionError(f"Unhandled delta_alt_mode: {self.config.delta_alt_mode!r}")

    def _logits_from_off_state(self, h_off: Tensor) -> tuple[Tensor, str]:
        if self.config.off_logits_mode == "projection_head":
            return self.off_projection_head(h_off), "projection_head"
        if self.config.off_logits_mode == "lm_head":
            lm_head = getattr(self.model, "lm_head", None)
            if lm_head is None:
                return self.off_projection_head(h_off), "projection_head"
            return lm_head(h_off), "lm_head"
        raise AssertionError(f"Unhandled off_logits_mode: {self.config.off_logits_mode!r}")

    def _validate_state_for_off_path(self, h: Tensor, *, input_ids: Tensor) -> None:
        if h.ndim != 3:
            raise ValueError(f"RealMambaStudent h must have shape [B, T, D], got {tuple(h.shape)}.")
        if h.shape[:2] != input_ids.shape:
            raise ValueError(
                "RealMambaStudent h must match input_ids batch/time dimensions: "
                f"{tuple(h.shape[:2])} != {tuple(input_ids.shape)}."
            )
        if h.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "RealMambaStudent h hidden dimension must match config.hidden_size for "
                f"Stage 6E projection heads: {h.shape[-1]} != {self.config.hidden_size}."
            )

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
