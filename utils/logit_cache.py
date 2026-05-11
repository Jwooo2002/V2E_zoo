from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch


_SUPPORTED_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class LogitCacheConfig:
    enabled: bool = False
    cache_dir: str = "cache/teacher_logits"
    format: str = "pt"
    dtype: str = "float16"
    device: str = "cpu"
    use_top_k: bool = False
    top_k: int = 256
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.format != "pt":
            raise ValueError("LogitCacheConfig.format must be 'pt'")
        if self.dtype not in _SUPPORTED_DTYPES:
            supported = ", ".join(sorted(_SUPPORTED_DTYPES))
            raise ValueError(f"Unsupported cache dtype {self.dtype!r}; expected one of {supported}")
        if self.top_k <= 0:
            raise ValueError("LogitCacheConfig.top_k must be positive")
        try:
            torch.device(self.device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid cache device {self.device!r}") from exc

    @property
    def torch_dtype(self) -> torch.dtype:
        return _SUPPORTED_DTYPES[self.dtype]

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


@dataclass
class LogitCacheEntry:
    logits: torch.Tensor | None = None
    topk_values: torch.Tensor | None = None
    topk_indices: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


ComputeFn = Callable[..., torch.Tensor | LogitCacheEntry]


class TeacherLogitCache:
    """Disk cache for frozen teacher LM logits on clean token prefixes.

    Cache keys are derived only from teacher-output-affecting inputs:
    ``input_ids``, optional ``attention_mask``, and canonical JSON ``extra``.
    The ``extra`` payload should contain teacher, tokenizer, or prompt-format
    metadata only. It must not include student states, rho/sigma, off-state
    tensors, adapters, or other student-side CSDM construction details.

    Top-k mode stores only the largest teacher logits per position. Loss code
    must explicitly handle top-k KD, and omitting full logits makes KD an
    approximation over the retained teacher support.
    """

    def __init__(self, config: LogitCacheConfig) -> None:
        self.config = config
        self.cache_dir = Path(config.cache_dir)

    def make_key(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        hasher = hashlib.sha256()
        self._hash_tensor(hasher, "input_ids", input_ids)
        if attention_mask is not None:
            self._hash_tensor(hasher, "attention_mask", attention_mask)
        else:
            hasher.update(b"attention_mask:none")
        if extra is not None:
            encoded = json.dumps(extra, sort_keys=True, separators=(",", ":")).encode("utf-8")
            hasher.update(b"extra:")
            hasher.update(encoded)
        else:
            hasher.update(b"extra:null")
        return hasher.hexdigest()

    def exists(self, key: str) -> bool:
        return self._path_for_key(key).exists()

    def save(
        self,
        key: str,
        logits: torch.Tensor | None = None,
        topk_values: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        if logits is not None:
            self._validate_logits(logits)

        original_vocab_size = logits.shape[-1] if logits is not None else None

        if self.config.use_top_k and logits is not None:
            if self.config.top_k > original_vocab_size:
                raise ValueError(
                    f"top_k={self.config.top_k} exceeds logits vocab size {original_vocab_size}"
                )
            topk_values, topk_indices = torch.topk(logits.detach(), k=self.config.top_k, dim=-1)
            logits = None

        self._validate_topk(topk_values, topk_indices)

        cache_metadata = self._build_metadata(
            logits=logits,
            topk_values=topk_values,
            topk_indices=topk_indices,
            vocab_size=original_vocab_size,
            user_metadata=metadata,
        )

        entry = {
            "metadata": cache_metadata,
            "logits": self._prepare_float_tensor(logits),
            "topk_values": self._prepare_float_tensor(topk_values),
            "topk_indices": self._prepare_index_tensor(topk_indices),
        }

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for_key(key)
        torch.save(entry, path)
        return path

    def load(self, key: str, map_location: str | torch.device | None = None) -> LogitCacheEntry:
        location = map_location if map_location is not None else self.config.torch_device
        path = self._path_for_key(key)
        try:
            raw = torch.load(path, map_location=location, weights_only=True)
        except TypeError:
            raw = torch.load(path, map_location=location)

        entry = LogitCacheEntry(
            logits=self._detach_optional(raw.get("logits")),
            topk_values=self._detach_optional(raw.get("topk_values")),
            topk_indices=self._detach_optional(raw.get("topk_indices")),
            metadata=dict(raw.get("metadata") or {}),
        )
        self._validate_entry(entry)
        return entry

    def get_or_compute(
        self,
        input_ids: torch.Tensor,
        compute_fn: ComputeFn,
        attention_mask: torch.Tensor | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LogitCacheEntry:
        key = self.make_key(input_ids, attention_mask=attention_mask, extra=extra)
        if self.config.enabled and self.exists(key) and not self.config.overwrite:
            return self.load(key)

        with torch.no_grad():
            result = compute_fn(input_ids, attention_mask=attention_mask)

        if isinstance(result, LogitCacheEntry):
            entry = result
        elif isinstance(result, torch.Tensor):
            entry = LogitCacheEntry(logits=result)
        else:
            raise TypeError("compute_fn must return a Tensor or LogitCacheEntry")
        self._validate_prefix_shape(entry, input_ids)

        metadata = dict(entry.metadata)
        metadata.setdefault("input_shape", list(input_ids.shape))
        if attention_mask is not None:
            metadata.setdefault("attention_mask_shape", list(attention_mask.shape))

        if not self.config.enabled:
            return self._entry_without_grad(
                LogitCacheEntry(
                    logits=entry.logits,
                    topk_values=entry.topk_values,
                    topk_indices=entry.topk_indices,
                    metadata=metadata,
                )
            )

        self.save(
            key,
            logits=entry.logits,
            topk_values=entry.topk_values,
            topk_indices=entry.topk_indices,
            metadata=metadata,
        )
        return self.load(key)

    def _path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{key}.{self.config.format}"

    @staticmethod
    def _hash_tensor(hasher: Any, name: str, tensor: torch.Tensor) -> None:
        detached = tensor.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tuple(detached.shape)).encode("utf-8"))
        hasher.update(str(detached.dtype).encode("utf-8"))
        hasher.update(detached.view(torch.uint8).numpy().tobytes())

    @staticmethod
    def _validate_logits(logits: torch.Tensor) -> None:
        if logits.ndim != 3:
            raise ValueError(f"Teacher LM logits must have rank 3 [B, T, V], got {logits.shape}")

    @staticmethod
    def _validate_topk(
        topk_values: torch.Tensor | None,
        topk_indices: torch.Tensor | None,
    ) -> None:
        if (topk_values is None) != (topk_indices is None):
            raise ValueError("topk_values and topk_indices must be provided together")
        if topk_values is None or topk_indices is None:
            return
        if topk_values.ndim != 3:
            raise ValueError(
                f"Teacher top-k logits must have rank 3 [B, T, K], got {topk_values.shape}"
            )
        if topk_values.shape != topk_indices.shape:
            raise ValueError(
                "topk_values and topk_indices must have the same shape, "
                f"got {topk_values.shape} and {topk_indices.shape}"
            )

    @staticmethod
    def _validate_prefix_shape(entry: LogitCacheEntry, input_ids: torch.Tensor) -> None:
        expected = tuple(input_ids.shape)
        for name, tensor in (
            ("logits", entry.logits),
            ("topk_values", entry.topk_values),
            ("topk_indices", entry.topk_indices),
        ):
            if tensor is not None and tuple(tensor.shape[:2]) != expected:
                raise ValueError(
                    f"{name} prefix shape {tuple(tensor.shape[:2])} does not match "
                    f"input_ids shape {expected}."
                )

    def _build_metadata(
        self,
        logits: torch.Tensor | None,
        topk_values: torch.Tensor | None,
        topk_indices: torch.Tensor | None,
        vocab_size: int | None,
        user_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = dict(user_metadata or {})
        metadata["dtype"] = self.config.dtype
        metadata["top_k"] = self.config.top_k if self.config.use_top_k else None

        source = logits if logits is not None else topk_values
        if source is not None:
            metadata.setdefault("input_shape", list(source.shape[:2]))
        if logits is not None:
            metadata.setdefault("logit_shape", list(logits.shape))
            metadata["vocab_size"] = logits.shape[-1]
        elif vocab_size is not None:
            metadata["vocab_size"] = vocab_size
        if topk_values is not None:
            metadata.setdefault("topk_shape", list(topk_values.shape))
        if topk_indices is not None:
            metadata.setdefault("topk_indices_dtype", "int64")
        return metadata

    def _prepare_float_tensor(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.detach().to(device=self.config.torch_device, dtype=self.config.torch_dtype)

    def _prepare_index_tensor(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.detach().to(device=self.config.torch_device, dtype=torch.long)

    @staticmethod
    def _detach_optional(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.detach()

    def _entry_without_grad(self, entry: LogitCacheEntry) -> LogitCacheEntry:
        return LogitCacheEntry(
            logits=self._detach_optional(entry.logits),
            topk_values=self._detach_optional(entry.topk_values),
            topk_indices=self._detach_optional(entry.topk_indices),
            metadata=dict(entry.metadata),
        )

    def _validate_entry(self, entry: LogitCacheEntry) -> None:
        if entry.logits is not None:
            self._validate_logits(entry.logits)
        self._validate_topk(entry.topk_values, entry.topk_indices)
