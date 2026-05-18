"""Training scaffold for CSDM Mamba KD smoke paths."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.dataset import MockTextDataset, TextDatasetConfig, TokenizedTextDataset
from data.tokenizer import TokenizerConfig, load_tokenizer
from data.vocab import get_tokenizer_vocab_size, validate_token_id_ranges, validate_vocab_alignment
from losses.cdm_loss import csdm_loss
from losses.kd_loss import build_topk_indices, kd_kl_loss
from models.cdm_engine import OffTrajectoryConfig
from models.student_mamba import (
    MambaStudentConfig,
    MockStudentMamba,
    RealMambaStudent,
    StudentMamba,
    StudentOutput,
)
from models.teacher_wrapper import HuggingFaceTeacherConfig, HuggingFaceTeacherWrapper, MockTeacherWrapper
from utils.checkpointing import latest_checkpoint, load_checkpoint, load_training_checkpoint, save_training_checkpoint
from utils.distributed import (
    DistributedContext,
    RankZeroLogger,
    average_float_dict,
    barrier,
    cleanup_distributed,
    effective_batch_size,
    init_distributed,
    rank_local_dir,
)
from utils.logit_cache import LogitCacheConfig, LogitCacheEntry, TeacherLogitCache
from utils.logger import ConsoleLogger
from utils.storage import validate_storage_paths


@dataclass(frozen=True)
class LossConfig:
    ce_weight: float = 0.2
    kd_weight: float = 1.0
    csdm_weight: float = 0.1
    tau: float = 2.0
    lambda_score: float = 0.1
    residual_clip: float = 3.0
    scale_min: float = 0.05
    scale_max: float = 5.0


@dataclass(frozen=True)
class MockConfig:
    batch_size: int = 2
    seq_len: int = 128
    vocab_size: int = 1024
    hidden_size: int = 256
    num_samples: int = 1024
    positions_per_sequence: int = 64
    ignore_index: int = -100


@dataclass(frozen=True)
class DataConfig:
    dataset_type: str = "mock"
    path: str | None = None
    tokenizer_name_or_path: str | None = None
    seq_len: int = 128
    stride: int | None = None
    max_examples: int | None = None
    text_field: str = "text"
    add_eos: bool = True
    shuffle: bool = False
    seed: int = 42
    use_fast: bool = True
    trust_remote_code: bool = False
    local_files_only: bool = False
    pad_token_strategy: str = "eos"


@dataclass(frozen=True)
class VocabConfig:
    strict_alignment: bool = True
    allow_student_vocab_resize: bool = False
    ignored_label_id: int = -100


@dataclass(frozen=True)
class HuggingFaceRuntimeConfig:
    model_name_or_path: str | None = None
    torch_dtype: str = "bfloat16"
    device_map: str | None = "auto"
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    local_files_only: bool = False
    use_safetensors: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False


@dataclass(frozen=True)
class TopKConfig:
    enabled: bool = False
    top_k: int = 256
    include_labels: bool = True
    renormalize_topk: bool = True


@dataclass(frozen=True)
class CheckpointConfig:
    output_dir: str = "checkpoints"
    save_every_steps: int = 0
    save_at_end: bool = False
    resume_from: str | None = None
    auto_resume: bool = False
    strict_resume: bool = True
    load_optimizer: bool = True
    load_rng_state: bool = True


@dataclass(frozen=True)
class DistributedConfig:
    mode: str = "none"
    backend: str | None = None
    find_unused_parameters: bool = False


@dataclass(frozen=True)
class StorageConfig:
    min_free_gb: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    teacher_type: str = "mock"
    student_type: str = "mock"
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    topk: TopKConfig = field(default_factory=TopKConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    teacher_cache: LogitCacheConfig = field(default_factory=LogitCacheConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vocab: VocabConfig = field(default_factory=VocabConfig)
    mock: MockConfig = field(default_factory=MockConfig)
    mamba_student: MambaStudentConfig = field(default_factory=MambaStudentConfig)
    student_vocab_size_explicit: bool = False
    hf_teacher: HuggingFaceRuntimeConfig = field(default_factory=HuggingFaceRuntimeConfig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mock", action="store_true", help="Run the mock-only Stage 3 scaffold.")
    parser.add_argument("--max_steps", type=int, default=2, help="Number of optimizer steps.")
    parser.add_argument("--teacher-type", choices=("mock", "hf"), default=None)
    parser.add_argument("--student-type", choices=("mock", "mamba"), default=None)
    parser.add_argument("--teacher-model-name-or-path", default=None)
    parser.add_argument("--hf-torch-dtype", choices=("float32", "float16", "bfloat16"), default=None)
    parser.add_argument("--hf-device-map", default=None, help="HF teacher device_map; use 'none' for rank-local .to(device).")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow or disallow remote code for HF teacher/tokenizer loading.",
    )
    parser.add_argument(
        "--use-safetensors",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require safetensors for HF teacher loading when available.",
    )
    parser.add_argument("--dataset-type", choices=("mock", "text", "jsonl"), default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--tokenizer-name-or-path", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--text-field", default=None)
    parser.add_argument(
        "--allow-student-vocab-resize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Opt into explicit student vocab resizing paths when implemented. Disabled by default.",
    )
    parser.add_argument("--student-model-name-or-path", default=None)
    parser.add_argument("--student-vocab-size", type=int, default=None)
    parser.add_argument("--student-hidden-size", type=int, default=None)
    parser.add_argument("--student-num-layers", type=int, default=None)
    parser.add_argument("--student-state-extraction", choices=("last_hidden", "embedding", "none"), default=None)
    parser.add_argument("--off-state-mode", choices=("projection", "placeholder", "none"), default=None)
    parser.add_argument("--delta-alt-mode", choices=("delta_projection", "noise", "identity"), default=None)
    parser.add_argument("--off-logits-mode", choices=("lm_head", "projection_head", "placeholder"), default=None)
    parser.add_argument(
        "--off-state-detach-direction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Detach h_delta_alt - h before building the real-Mamba smoke off-state.",
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default=None)
    parser.add_argument("--csdm-weight", type=float, default=None)
    parser.add_argument("--kd-weight", type=float, default=None)
    parser.add_argument("--ce-weight", type=float, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--topk-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable selected-vocab KD/CSDM approximation.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Number of teacher top-k logits to select.")
    parser.add_argument(
        "--topk-include-labels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append valid target labels to teacher top-k indices.",
    )
    parser.add_argument(
        "--topk-renormalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Renormalize KD/CSDM distributions over selected vocab entries.",
    )
    parser.add_argument(
        "--teacher-cache-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable disk caching for frozen teacher logits on clean token prefixes.",
    )
    parser.add_argument("--teacher-cache-dir", type=str, default=None)
    parser.add_argument(
        "--teacher-cache-overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recompute and overwrite teacher cache entries even when a matching key exists.",
    )
    parser.add_argument(
        "--teacher-cache-use-top-k",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store top-k-only teacher cache entries. Training raises until this loss path is implemented.",
    )
    parser.add_argument("--teacher-cache-top-k", type=int, default=None)
    parser.add_argument(
        "--teacher-cache-distributed-policy",
        choices=("rank_local", "shared_readonly", "rank_zero_write"),
        default=None,
    )
    parser.add_argument("--checkpoint-output-dir", type=str, default=None)
    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument("--save-at-end", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--strict-resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--load-optimizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--load-rng-state", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--distributed-mode",
        choices=("none", "env", "ddp"),
        default=None,
        help="Stage 7E distributed scaffold mode. 'env' is rank-aware without DDP; 'ddp' is reserved.",
    )
    parser.add_argument("--distributed", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"), default=None)
    parser.add_argument("--ddp-find-unused-parameters", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--storage-min-free-gb",
        type=float,
        default=None,
        help="Optional preflight free-space threshold for checkpoint/cache destinations. Disabled at 0.",
    )
    return parser.parse_args()


def _nested_get(data: dict[str, Any], key: str, default: Any) -> Any:
    return data.get(key, default)


def _normalize_student_type(value: str) -> str:
    if value in {"mock", "mock_mamba"}:
        return "mock"
    if value in {"mamba", "real_mamba"}:
        return "mamba"
    raise ValueError(f"Unsupported student_type {value!r}; expected 'mock' or 'mamba'.")


def _normalize_teacher_type(value: str) -> str:
    if value in {"mock", "hf"}:
        return value
    raise ValueError(f"Unsupported teacher_type {value!r}; expected 'mock' or 'hf'.")


def _optional_str(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _optional_device_map(value: Any) -> str | None:
    if value in (None, "", "none", "None", "null", "Null"):
        return None
    return str(value)


def _parse_logit_cache_config(raw: dict[str, Any]) -> LogitCacheConfig:
    return LogitCacheConfig(
        enabled=bool(_nested_get(raw, "enabled", LogitCacheConfig.enabled)),
        cache_dir=str(_nested_get(raw, "cache_dir", LogitCacheConfig.cache_dir)),
        format=str(_nested_get(raw, "format", LogitCacheConfig.format)),
        dtype=str(_nested_get(raw, "dtype", LogitCacheConfig.dtype)),
        device=str(_nested_get(raw, "device", LogitCacheConfig.device)),
        use_top_k=bool(_nested_get(raw, "use_top_k", LogitCacheConfig.use_top_k)),
        top_k=int(_nested_get(raw, "top_k", LogitCacheConfig.top_k)),
        overwrite=bool(_nested_get(raw, "overwrite", LogitCacheConfig.overwrite)),
        distributed_policy=str(_nested_get(raw, "distributed_policy", LogitCacheConfig.distributed_policy)),
    )


def _normalize_dataset_type(value: str) -> str:
    if value in {"mock", "text", "jsonl"}:
        return value
    raise ValueError(f"Unsupported dataset_type {value!r}; expected 'mock', 'text', or 'jsonl'.")


def _parse_data_config(raw: dict[str, Any]) -> DataConfig:
    return DataConfig(
        dataset_type=_normalize_dataset_type(str(_nested_get(raw, "dataset_type", DataConfig.dataset_type))),
        path=_optional_str(raw.get("path")),
        tokenizer_name_or_path=_optional_str(raw.get("tokenizer_name_or_path")),
        seq_len=int(_nested_get(raw, "seq_len", DataConfig.seq_len)),
        stride=None if raw.get("stride") in (None, "") else int(raw["stride"]),
        max_examples=None if raw.get("max_examples") in (None, "") else int(raw["max_examples"]),
        text_field=str(_nested_get(raw, "text_field", DataConfig.text_field)),
        add_eos=bool(_nested_get(raw, "add_eos", DataConfig.add_eos)),
        shuffle=bool(_nested_get(raw, "shuffle", DataConfig.shuffle)),
        seed=int(_nested_get(raw, "seed", DataConfig.seed)),
        use_fast=bool(_nested_get(raw, "use_fast", DataConfig.use_fast)),
        trust_remote_code=bool(_nested_get(raw, "trust_remote_code", DataConfig.trust_remote_code)),
        local_files_only=bool(_nested_get(raw, "local_files_only", DataConfig.local_files_only)),
        pad_token_strategy=str(_nested_get(raw, "pad_token_strategy", DataConfig.pad_token_strategy)),
    )


def _parse_vocab_config(raw: dict[str, Any]) -> VocabConfig:
    return VocabConfig(
        strict_alignment=bool(_nested_get(raw, "strict_alignment", VocabConfig.strict_alignment)),
        allow_student_vocab_resize=bool(
            _nested_get(raw, "allow_student_vocab_resize", VocabConfig.allow_student_vocab_resize)
        ),
        ignored_label_id=int(_nested_get(raw, "ignored_label_id", VocabConfig.ignored_label_id)),
    )


def _parse_checkpoint_config(raw: dict[str, Any]) -> CheckpointConfig:
    return CheckpointConfig(
        output_dir=str(_nested_get(raw, "output_dir", CheckpointConfig.output_dir)),
        save_every_steps=int(_nested_get(raw, "save_every_steps", CheckpointConfig.save_every_steps)),
        save_at_end=bool(_nested_get(raw, "save_at_end", CheckpointConfig.save_at_end)),
        resume_from=_optional_str(raw.get("resume_from")),
        auto_resume=bool(_nested_get(raw, "auto_resume", CheckpointConfig.auto_resume)),
        strict_resume=bool(_nested_get(raw, "strict_resume", CheckpointConfig.strict_resume)),
        load_optimizer=bool(_nested_get(raw, "load_optimizer", CheckpointConfig.load_optimizer)),
        load_rng_state=bool(_nested_get(raw, "load_rng_state", CheckpointConfig.load_rng_state)),
    )


def _parse_distributed_config(raw: dict[str, Any]) -> DistributedConfig:
    legacy_enabled = raw.get("enabled")
    mode = str(_nested_get(raw, "mode", "env" if legacy_enabled else "none"))
    if mode not in {"none", "env", "ddp"}:
        raise ValueError("distributed.mode must be one of: none, env, ddp.")
    return DistributedConfig(
        mode=mode,
        backend=_optional_str(raw.get("backend")),
        find_unused_parameters=bool(
            _nested_get(raw, "find_unused_parameters", DistributedConfig.find_unused_parameters)
        ),
    )


def _parse_storage_config(raw: dict[str, Any]) -> StorageConfig:
    return StorageConfig(min_free_gb=float(_nested_get(raw, "min_free_gb", StorageConfig.min_free_gb)))


def _parse_mamba_student_config(raw: dict[str, Any]) -> MambaStudentConfig:
    return MambaStudentConfig(
        model_name_or_path=_optional_str(raw.get("model_name_or_path")),
        vocab_size=int(_nested_get(raw, "vocab_size", MambaStudentConfig.vocab_size)),
        hidden_size=int(_nested_get(raw, "hidden_size", MambaStudentConfig.hidden_size)),
        num_layers=(
            None
            if raw.get("num_layers") in (None, "")
            else int(raw.get("num_layers", MambaStudentConfig.num_layers))
        ),
        state_size=(
            None
            if raw.get("state_size") in (None, "")
            else int(raw.get("state_size", MambaStudentConfig.state_size))
        ),
        torch_dtype=str(_nested_get(raw, "torch_dtype", MambaStudentConfig.torch_dtype)),
        device=_optional_str(raw.get("device", MambaStudentConfig.device)),
        trust_remote_code=bool(_nested_get(raw, "trust_remote_code", MambaStudentConfig.trust_remote_code)),
        use_pretrained=bool(_nested_get(raw, "use_pretrained", MambaStudentConfig.use_pretrained)),
        local_files_only=bool(_nested_get(raw, "local_files_only", MambaStudentConfig.local_files_only)),
        delta_perturb_eps=float(_nested_get(raw, "delta_perturb_eps", MambaStudentConfig.delta_perturb_eps)),
        noise_sigma=float(_nested_get(raw, "noise_sigma", MambaStudentConfig.noise_sigma)),
        use_reference_forward=bool(_nested_get(raw, "use_reference_forward", MambaStudentConfig.use_reference_forward)),
        state_extraction=str(_nested_get(raw, "state_extraction", MambaStudentConfig.state_extraction)),
        expose_states=bool(_nested_get(raw, "expose_states", MambaStudentConfig.expose_states)),
        off_state_mode=str(_nested_get(raw, "off_state_mode", MambaStudentConfig.off_state_mode)),
        delta_alt_mode=str(_nested_get(raw, "delta_alt_mode", MambaStudentConfig.delta_alt_mode)),
        off_logits_mode=str(_nested_get(raw, "off_logits_mode", MambaStudentConfig.off_logits_mode)),
        off_state_detach_direction=bool(
            _nested_get(raw, "off_state_detach_direction", MambaStudentConfig.off_state_detach_direction)
        ),
        allow_student_vocab_resize=bool(
            _nested_get(raw, "allow_student_vocab_resize", MambaStudentConfig.allow_student_vocab_resize)
        ),
    )


def load_train_config(path: Path) -> TrainConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    loss_raw = raw.get("loss", {})
    mock_raw = raw.get("mock", {})
    data_raw = raw.get("data", {})
    vocab_raw = raw.get("vocab", {}) or {}
    teacher_raw = raw.get("teacher", {})
    student_raw = raw.get("student", {})
    hf_raw = raw.get("hf_teacher", raw.get("hf_teacher_example", {}))
    teacher_cache_raw = raw.get("teacher_cache", {}) or {}
    checkpoint_raw = raw.get("checkpoint", {}) or {}
    distributed_raw = raw.get("distributed", {}) or {}
    storage_raw = raw.get("storage", {}) or {}
    loss = LossConfig(
        ce_weight=float(_nested_get(loss_raw, "ce_weight", LossConfig.ce_weight)),
        kd_weight=float(_nested_get(loss_raw, "kd_weight", LossConfig.kd_weight)),
        csdm_weight=float(_nested_get(loss_raw, "csdm_weight", LossConfig.csdm_weight)),
        tau=float(_nested_get(loss_raw, "tau", LossConfig.tau)),
        lambda_score=float(_nested_get(loss_raw, "lambda_score", LossConfig.lambda_score)),
        residual_clip=float(_nested_get(loss_raw, "residual_clip", LossConfig.residual_clip)),
        scale_min=float(_nested_get(loss_raw, "scale_min", LossConfig.scale_min)),
        scale_max=float(_nested_get(loss_raw, "scale_max", LossConfig.scale_max)),
    )
    mock = MockConfig(
        batch_size=int(_nested_get(mock_raw, "batch_size", MockConfig.batch_size)),
        seq_len=int(_nested_get(mock_raw, "seq_len", mock_raw.get("sequence_length", MockConfig.seq_len))),
        vocab_size=int(_nested_get(mock_raw, "vocab_size", MockConfig.vocab_size)),
        hidden_size=int(_nested_get(mock_raw, "hidden_size", MockConfig.hidden_size)),
        num_samples=int(_nested_get(mock_raw, "num_samples", MockConfig.num_samples)),
        positions_per_sequence=int(
            _nested_get(mock_raw, "positions_per_sequence", MockConfig.positions_per_sequence)
        ),
        ignore_index=int(_nested_get(mock_raw, "ignore_index", MockConfig.ignore_index)),
    )
    topk_raw = raw.get("topk", {}) or {}
    top_k_legacy = raw.get("top_k_kd")
    topk_enabled_default = top_k_legacy not in (None, "") if "enabled" not in topk_raw else TopKConfig.enabled
    top_k_default = int(top_k_legacy) if top_k_legacy not in (None, "") else TopKConfig.top_k
    topk = TopKConfig(
        enabled=bool(_nested_get(topk_raw, "enabled", topk_enabled_default)),
        top_k=int(_nested_get(topk_raw, "top_k", top_k_default)),
        include_labels=bool(_nested_get(topk_raw, "include_labels", TopKConfig.include_labels)),
        renormalize_topk=bool(_nested_get(topk_raw, "renormalize_topk", TopKConfig.renormalize_topk)),
    )
    checkpoint = _parse_checkpoint_config(checkpoint_raw)
    distributed = _parse_distributed_config(distributed_raw)
    hf_teacher = HuggingFaceRuntimeConfig(
        model_name_or_path=_optional_str(hf_raw.get("model_name_or_path")),
        torch_dtype=str(hf_raw.get("torch_dtype", HuggingFaceRuntimeConfig.torch_dtype)),
        device_map=_optional_str(hf_raw.get("device_map", HuggingFaceRuntimeConfig.device_map)),
        trust_remote_code=bool(hf_raw.get("trust_remote_code", HuggingFaceRuntimeConfig.trust_remote_code)),
        attn_implementation=_optional_str(
            hf_raw.get("attn_implementation", HuggingFaceRuntimeConfig.attn_implementation)
        ),
        local_files_only=bool(hf_raw.get("local_files_only", HuggingFaceRuntimeConfig.local_files_only)),
        use_safetensors=bool(hf_raw.get("use_safetensors", HuggingFaceRuntimeConfig.use_safetensors)),
        load_in_8bit=bool(hf_raw.get("load_in_8bit", HuggingFaceRuntimeConfig.load_in_8bit)),
        load_in_4bit=bool(hf_raw.get("load_in_4bit", HuggingFaceRuntimeConfig.load_in_4bit)),
    )
    mamba_student = _parse_mamba_student_config(student_raw)
    data = _parse_data_config(data_raw)
    vocab = _parse_vocab_config(vocab_raw)
    return TrainConfig(
        seed=int(_nested_get(raw, "seed", TrainConfig.seed)),
        teacher_type=_normalize_teacher_type(str(teacher_raw.get("type", raw.get("teacher_type", "mock")))),
        student_type=_normalize_student_type(str(student_raw.get("type", raw.get("student_type", "mock")))),
        mixed_precision=str(raw.get("mixed_precision", raw.get("precision", TrainConfig.mixed_precision))),
        gradient_accumulation_steps=int(
            _nested_get(raw, "gradient_accumulation_steps", TrainConfig.gradient_accumulation_steps)
        ),
        learning_rate=float(_nested_get(raw, "learning_rate", TrainConfig.learning_rate)),
        max_grad_norm=float(_nested_get(raw, "max_grad_norm", TrainConfig.max_grad_norm)),
        topk=topk,
        distributed=distributed,
        storage=_parse_storage_config(storage_raw),
        teacher_cache=_parse_logit_cache_config(teacher_cache_raw),
        checkpoint=checkpoint,
        loss=loss,
        data=data,
        vocab=vocab,
        mock=mock,
        mamba_student=mamba_student,
        hf_teacher=hf_teacher,
    )


def _load_model_config_hf_defaults(config_path: Path) -> HuggingFaceRuntimeConfig:
    model_config_path = config_path.parent / "model_config.yaml"
    if not model_config_path.exists():
        return HuggingFaceRuntimeConfig()
    with model_config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    hf_raw = raw.get("hf_teacher", raw.get("hf_teacher_example", {}))
    if hf_raw.get("enabled") is False:
        return HuggingFaceRuntimeConfig()
    return HuggingFaceRuntimeConfig(
        model_name_or_path=_optional_str(hf_raw.get("model_name_or_path")),
        torch_dtype=str(hf_raw.get("torch_dtype", HuggingFaceRuntimeConfig.torch_dtype)),
        device_map=_optional_str(hf_raw.get("device_map", HuggingFaceRuntimeConfig.device_map)),
        trust_remote_code=bool(hf_raw.get("trust_remote_code", HuggingFaceRuntimeConfig.trust_remote_code)),
        attn_implementation=_optional_str(
            hf_raw.get("attn_implementation", HuggingFaceRuntimeConfig.attn_implementation)
        ),
        local_files_only=bool(hf_raw.get("local_files_only", HuggingFaceRuntimeConfig.local_files_only)),
        use_safetensors=bool(hf_raw.get("use_safetensors", HuggingFaceRuntimeConfig.use_safetensors)),
        load_in_8bit=bool(hf_raw.get("load_in_8bit", HuggingFaceRuntimeConfig.load_in_8bit)),
        load_in_4bit=bool(hf_raw.get("load_in_4bit", HuggingFaceRuntimeConfig.load_in_4bit)),
    )


def _load_model_config_student_defaults(config_path: Path) -> MambaStudentConfig:
    model_config_path = config_path.parent / "model_config.yaml"
    if not model_config_path.exists():
        return MambaStudentConfig()
    with model_config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    student_raw = raw.get("student", {}) or {}
    return _parse_mamba_student_config(student_raw)


def derive_runtime_config(args: argparse.Namespace) -> TrainConfig:
    """Apply CLI precedence: YAML, HF smoke defaults, then explicit CLI flags."""

    config = load_train_config(args.config)

    if args.mock:
        config = replace(config, teacher_type="mock", student_type="mock", data=replace(config.data, dataset_type="mock"))
    else:
        teacher_type = args.teacher_type if args.teacher_type is not None else config.teacher_type
        student_type = args.student_type if args.student_type is not None else config.student_type
        config = replace(
            config,
            teacher_type=_normalize_teacher_type(teacher_type),
            student_type=_normalize_student_type(student_type),
        )

    if config.teacher_type == "hf":
        model_hf = _load_model_config_hf_defaults(args.config)
        hf_teacher = replace(
            config.hf_teacher,
            model_name_or_path=config.hf_teacher.model_name_or_path or model_hf.model_name_or_path,
            torch_dtype=config.hf_teacher.torch_dtype or model_hf.torch_dtype,
            device_map=config.hf_teacher.device_map if config.hf_teacher.device_map is not None else model_hf.device_map,
            trust_remote_code=config.hf_teacher.trust_remote_code or model_hf.trust_remote_code,
            attn_implementation=config.hf_teacher.attn_implementation or model_hf.attn_implementation,
            local_files_only=config.hf_teacher.local_files_only or model_hf.local_files_only,
            use_safetensors=config.hf_teacher.use_safetensors,
            load_in_8bit=config.hf_teacher.load_in_8bit or model_hf.load_in_8bit,
            load_in_4bit=config.hf_teacher.load_in_4bit or model_hf.load_in_4bit,
        )
        config = replace(
            config,
            hf_teacher=hf_teacher,
            mock=replace(config.mock, batch_size=1, seq_len=128),
            gradient_accumulation_steps=1,
            loss=replace(config.loss, ce_weight=0.2, kd_weight=1.0, csdm_weight=0.0),
        )

    model_student = _load_model_config_student_defaults(args.config)
    config = replace(
        config,
        mamba_student=replace(
            config.mamba_student,
            model_name_or_path=config.mamba_student.model_name_or_path or model_student.model_name_or_path,
            vocab_size=config.mamba_student.vocab_size or model_student.vocab_size,
            hidden_size=config.mamba_student.hidden_size or model_student.hidden_size,
            num_layers=config.mamba_student.num_layers if config.mamba_student.num_layers is not None else model_student.num_layers,
            state_size=config.mamba_student.state_size if config.mamba_student.state_size is not None else model_student.state_size,
            torch_dtype=config.mamba_student.torch_dtype or model_student.torch_dtype,
            device=config.mamba_student.device if config.mamba_student.device is not None else model_student.device,
            trust_remote_code=config.mamba_student.trust_remote_code or model_student.trust_remote_code,
            use_pretrained=config.mamba_student.use_pretrained or model_student.use_pretrained,
            local_files_only=config.mamba_student.local_files_only or model_student.local_files_only,
            delta_perturb_eps=config.mamba_student.delta_perturb_eps,
            noise_sigma=config.mamba_student.noise_sigma,
            use_reference_forward=config.mamba_student.use_reference_forward or model_student.use_reference_forward,
            state_extraction=config.mamba_student.state_extraction or model_student.state_extraction,
            expose_states=config.mamba_student.expose_states,
            off_state_mode=config.mamba_student.off_state_mode or model_student.off_state_mode,
            delta_alt_mode=config.mamba_student.delta_alt_mode or model_student.delta_alt_mode,
            off_logits_mode=config.mamba_student.off_logits_mode or model_student.off_logits_mode,
            off_state_detach_direction=config.mamba_student.off_state_detach_direction,
            allow_student_vocab_resize=(
                config.mamba_student.allow_student_vocab_resize or model_student.allow_student_vocab_resize
            ),
        ),
    )

    if not args.mock and args.teacher_type is not None:
        config = replace(config, teacher_type=args.teacher_type)
    if not args.mock and args.student_type is not None:
        config = replace(config, student_type=_normalize_student_type(args.student_type))
    if args.teacher_model_name_or_path is not None:
        config = replace(
            config,
            hf_teacher=replace(config.hf_teacher, model_name_or_path=args.teacher_model_name_or_path),
        )
    if getattr(args, "hf_torch_dtype", None) is not None:
        config = replace(config, hf_teacher=replace(config.hf_teacher, torch_dtype=args.hf_torch_dtype))
    if getattr(args, "hf_device_map", None) is not None:
        config = replace(config, hf_teacher=replace(config.hf_teacher, device_map=_optional_device_map(args.hf_device_map)))
    if getattr(args, "trust_remote_code", None) is not None:
        config = replace(
            config,
            hf_teacher=replace(config.hf_teacher, trust_remote_code=args.trust_remote_code),
            data=replace(config.data, trust_remote_code=args.trust_remote_code),
        )
    if getattr(args, "use_safetensors", None) is not None:
        config = replace(config, hf_teacher=replace(config.hf_teacher, use_safetensors=args.use_safetensors))
    if not args.mock and getattr(args, "dataset_type", None) is not None:
        config = replace(config, data=replace(config.data, dataset_type=_normalize_dataset_type(args.dataset_type)))
    if not args.mock and getattr(args, "data_path", None) is not None:
        config = replace(config, data=replace(config.data, path=args.data_path))
    if not args.mock and getattr(args, "tokenizer_name_or_path", None) is not None:
        config = replace(config, data=replace(config.data, tokenizer_name_or_path=args.tokenizer_name_or_path))
    if not args.mock and getattr(args, "max_examples", None) is not None:
        config = replace(config, data=replace(config.data, max_examples=args.max_examples))
    if not args.mock and getattr(args, "text_field", None) is not None:
        config = replace(config, data=replace(config.data, text_field=args.text_field))
    if getattr(args, "allow_student_vocab_resize", None) is not None:
        config = replace(
            config,
            vocab=replace(config.vocab, allow_student_vocab_resize=args.allow_student_vocab_resize),
            mamba_student=replace(config.mamba_student, allow_student_vocab_resize=args.allow_student_vocab_resize),
        )
    if getattr(args, "student_model_name_or_path", None) is not None:
        config = replace(
            config,
            mamba_student=replace(config.mamba_student, model_name_or_path=args.student_model_name_or_path),
        )
    if getattr(args, "student_vocab_size", None) is not None:
        config = replace(
            config,
            mamba_student=replace(config.mamba_student, vocab_size=args.student_vocab_size),
            student_vocab_size_explicit=True,
        )
    if getattr(args, "student_hidden_size", None) is not None:
        config = replace(config, mamba_student=replace(config.mamba_student, hidden_size=args.student_hidden_size))
    if getattr(args, "student_num_layers", None) is not None:
        config = replace(config, mamba_student=replace(config.mamba_student, num_layers=args.student_num_layers))
    if getattr(args, "student_state_extraction", None) is not None:
        config = replace(
            config,
            mamba_student=replace(config.mamba_student, state_extraction=args.student_state_extraction),
        )
    if getattr(args, "off_state_mode", None) is not None:
        config = replace(config, mamba_student=replace(config.mamba_student, off_state_mode=args.off_state_mode))
    if getattr(args, "delta_alt_mode", None) is not None:
        config = replace(config, mamba_student=replace(config.mamba_student, delta_alt_mode=args.delta_alt_mode))
    if getattr(args, "off_logits_mode", None) is not None:
        config = replace(config, mamba_student=replace(config.mamba_student, off_logits_mode=args.off_logits_mode))
    if getattr(args, "off_state_detach_direction", None) is not None:
        config = replace(
            config,
            mamba_student=replace(
                config.mamba_student,
                off_state_detach_direction=args.off_state_detach_direction,
            ),
        )
    if args.local_files_only:
        config = replace(
            config,
            hf_teacher=replace(config.hf_teacher, local_files_only=True),
            mamba_student=replace(config.mamba_student, local_files_only=True),
            data=replace(config.data, local_files_only=True),
        )
    if args.seq_len is not None:
        config = replace(config, mock=replace(config.mock, seq_len=args.seq_len), data=replace(config.data, seq_len=args.seq_len))
    if args.batch_size is not None:
        config = replace(config, mock=replace(config.mock, batch_size=args.batch_size))
    if args.gradient_accumulation_steps is not None:
        config = replace(config, gradient_accumulation_steps=args.gradient_accumulation_steps)
    if args.mixed_precision is not None:
        config = replace(config, mixed_precision=args.mixed_precision)
    if args.ce_weight is not None:
        config = replace(config, loss=replace(config.loss, ce_weight=args.ce_weight))
    if args.kd_weight is not None:
        config = replace(config, loss=replace(config.loss, kd_weight=args.kd_weight))
    if args.csdm_weight is not None:
        config = replace(config, loss=replace(config.loss, csdm_weight=args.csdm_weight))
    if getattr(args, "topk_enabled", None) is not None:
        config = replace(config, topk=replace(config.topk, enabled=args.topk_enabled))
    if getattr(args, "top_k", None) is not None:
        config = replace(config, topk=replace(config.topk, top_k=args.top_k))
    if getattr(args, "topk_include_labels", None) is not None:
        config = replace(config, topk=replace(config.topk, include_labels=args.topk_include_labels))
    if getattr(args, "topk_renormalize", None) is not None:
        config = replace(config, topk=replace(config.topk, renormalize_topk=args.topk_renormalize))
    if getattr(args, "teacher_cache_enabled", None) is not None:
        config = replace(config, teacher_cache=replace(config.teacher_cache, enabled=args.teacher_cache_enabled))
    if getattr(args, "teacher_cache_dir", None) is not None:
        config = replace(config, teacher_cache=replace(config.teacher_cache, cache_dir=args.teacher_cache_dir))
    if getattr(args, "teacher_cache_overwrite", None) is not None:
        config = replace(config, teacher_cache=replace(config.teacher_cache, overwrite=args.teacher_cache_overwrite))
    if getattr(args, "teacher_cache_use_top_k", None) is not None:
        config = replace(config, teacher_cache=replace(config.teacher_cache, use_top_k=args.teacher_cache_use_top_k))
    if getattr(args, "teacher_cache_top_k", None) is not None:
        config = replace(config, teacher_cache=replace(config.teacher_cache, top_k=args.teacher_cache_top_k))
    if getattr(args, "teacher_cache_distributed_policy", None) is not None:
        config = replace(
            config,
            teacher_cache=replace(
                config.teacher_cache,
                distributed_policy=args.teacher_cache_distributed_policy,
            ),
        )
    if getattr(args, "checkpoint_output_dir", None) is not None:
        config = replace(config, checkpoint=replace(config.checkpoint, output_dir=args.checkpoint_output_dir))
    if getattr(args, "save_every_steps", None) is not None:
        config = replace(config, checkpoint=replace(config.checkpoint, save_every_steps=args.save_every_steps))
    if getattr(args, "save_at_end", None) is not None:
        config = replace(config, checkpoint=replace(config.checkpoint, save_at_end=args.save_at_end))
    if getattr(args, "resume_from", None) is not None:
        config = replace(config, checkpoint=replace(config.checkpoint, resume_from=args.resume_from))
    if getattr(args, "auto_resume", None) is not None:
        config = replace(config, checkpoint=replace(config.checkpoint, auto_resume=args.auto_resume))
    if getattr(args, "strict_resume", None) is not None:
        config = replace(config, checkpoint=replace(config.checkpoint, strict_resume=args.strict_resume))
    if getattr(args, "load_optimizer", None) is not None:
        config = replace(config, checkpoint=replace(config.checkpoint, load_optimizer=args.load_optimizer))
    if getattr(args, "load_rng_state", None) is not None:
        config = replace(config, checkpoint=replace(config.checkpoint, load_rng_state=args.load_rng_state))
    if getattr(args, "distributed_mode", None) is not None:
        config = replace(config, distributed=replace(config.distributed, mode=args.distributed_mode))
    if getattr(args, "distributed", None) is not None:
        config = replace(config, distributed=replace(config.distributed, mode="env" if args.distributed else "none"))
    if getattr(args, "distributed_backend", None) is not None:
        config = replace(config, distributed=replace(config.distributed, backend=args.distributed_backend))
    if getattr(args, "ddp_find_unused_parameters", None) is not None:
        config = replace(
            config,
            distributed=replace(
                config.distributed,
                find_unused_parameters=args.ddp_find_unused_parameters,
            ),
        )
    if getattr(args, "storage_min_free_gb", None) is not None:
        config = replace(config, storage=replace(config.storage, min_free_gb=args.storage_min_free_gb))
    if config.student_type == "mamba" and config.teacher_type == "mock":
        config = replace(config, mock=replace(config.mock, vocab_size=config.mamba_student.vocab_size))
    if config.teacher_type == "hf" and config.data.dataset_type != "mock" and config.data.tokenizer_name_or_path is None:
        config = replace(config, data=replace(config.data, tokenizer_name_or_path=config.hf_teacher.model_name_or_path))
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infinite_loader(
    loader: DataLoader[dict[str, Tensor]],
    sampler: Any | None = None,
) -> Iterator[dict[str, Tensor]]:
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
            epoch += 1
        for batch in loader:
            yield batch


def _masked_sequence_logits(logits: Tensor, mask: Tensor) -> Tensor:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, T, V], got {tuple(logits.shape)}.")
    if mask.shape != logits.shape[:2]:
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match logits {tuple(logits.shape[:2])}.")
    selected = logits[mask]
    if selected.numel() == 0:
        raise ValueError("valid-position mask selected no logits.")
    return selected.reshape(1, selected.shape[0], selected.shape[1])


def _select_shared_valid_mask(
    labels: Tensor,
    ignore_index: int,
    positions_per_sequence: int,
) -> Tensor:
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [B, T], got {tuple(labels.shape)}.")
    valid_mask = labels.ne(ignore_index)
    if positions_per_sequence > 0:
        limited_mask = torch.zeros_like(valid_mask)
        for row in range(labels.shape[0]):
            valid_positions = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
            count = min(positions_per_sequence, int(valid_positions.numel()))
            if count > 0:
                limited_mask[row, valid_positions[:count]] = True
        valid_mask = limited_mask
    if not bool(valid_mask.any()):
        raise ValueError("batch has no valid next-token positions.")
    return valid_mask


def _validate_teacher_student_logits(output: StudentOutput, teacher_logits: Tensor) -> None:
    expected_prefix = output.on_logits.shape[:2]
    if teacher_logits.shape[:2] != expected_prefix or teacher_logits.shape[:2] != output.off_logits.shape[:2]:
        raise ValueError(
            "teacher_logits, on_logits, and off_logits must share [B, T] shape before masking: "
            f"teacher={tuple(teacher_logits.shape)}, on={tuple(output.on_logits.shape)}, "
            f"off={tuple(output.off_logits.shape)}."
        )
    expected_vocab = output.on_logits.shape[-1]
    if teacher_logits.shape[-1] != expected_vocab or teacher_logits.shape[-1] != output.off_logits.shape[-1]:
        raise ValueError(
            "teacher_logits, on_logits, and off_logits must share vocab size before masking: "
            f"teacher={tuple(teacher_logits.shape)}, on={tuple(output.on_logits.shape)}, "
            f"off={tuple(output.off_logits.shape)}."
        )
    if output.fake_logits.shape != output.off_logits.shape:
        raise ValueError(
            "fake_logits must be student-derived and match off_logits shape before masking: "
            f"fake={tuple(output.fake_logits.shape)}, off={tuple(output.off_logits.shape)}."
        )


def compute_losses(
    output: StudentOutput,
    teacher_logits: Tensor,
    labels: Tensor,
    config: TrainConfig,
) -> dict[str, Tensor]:
    """Compute CE, on-trajectory KD, and off-trajectory CSDM on one shared mask.

    Teacher logits are token-prefix aligned: ``teacher_logits[:, t]`` represents
    ``p_phi(y | x_{<=t})`` and is compared to ``on_logits[:, t]`` and
    ``off_logits[:, t]``. The final label position is ignored, so CE/KD/CSDM do
    not train on the last-token placeholder.
    """

    _validate_teacher_student_logits(output, teacher_logits)
    mask = _select_shared_valid_mask(
        labels=labels,
        ignore_index=config.vocab.ignored_label_id,
        positions_per_sequence=config.mock.positions_per_sequence,
    )
    ce_logits = output.on_logits[mask]
    ce_labels = labels[mask]
    ce = F.cross_entropy(ce_logits.float(), ce_labels, ignore_index=config.vocab.ignored_label_id)

    on_masked = _masked_sequence_logits(output.on_logits, mask)
    off_masked = _masked_sequence_logits(output.off_logits, mask)
    teacher_masked = _masked_sequence_logits(teacher_logits, mask)
    fake_masked = _masked_sequence_logits(output.fake_logits, mask)
    labels_masked = labels[mask].reshape(1, -1)

    topk_indices = None
    if config.topk.enabled:
        topk_indices = build_topk_indices(
            teacher_masked.float(),
            labels=labels_masked,
            top_k=config.topk.top_k,
            include_labels=config.topk.include_labels,
        )

    kd = kd_kl_loss(
        on_masked.float(),
        teacher_masked.float(),
        tau=config.loss.tau,
        topk_indices=topk_indices,
        renormalize_topk=config.topk.renormalize_topk,
    )
    if config.loss.csdm_weight == 0:
        csdm = off_masked.new_zeros(())
    else:
        csdm = csdm_loss(
            off_masked.float(),
            teacher_masked.float(),
            fake_masked.float(),
            tau=config.loss.tau,
            lambda_score=config.loss.lambda_score,
            residual_clip=config.loss.residual_clip,
            scale_min=config.loss.scale_min,
            scale_max=config.loss.scale_max,
            topk_indices=topk_indices,
            renormalize_topk=config.topk.renormalize_topk,
        )
    total = (
        config.loss.ce_weight * ce
        + config.loss.kd_weight * kd
        + config.loss.csdm_weight * csdm
    )
    return {"total": total, "ce": ce.detach(), "kd": kd.detach(), "csdm": csdm.detach()}


def grad_norm(parameters: Iterator[nn.Parameter] | list[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = parameter.grad.detach().float().norm(2).item()
            squared += value * value
    return math.sqrt(squared)


def _autocast_context(mixed_precision: str, device: torch.device) -> Any:
    if mixed_precision == "no" or device.type != "cuda":
        return nullcontext()
    if mixed_precision == "bf16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if mixed_precision == "fp16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError("mixed_precision must be one of: no, fp16, bf16.")


def _training_device(config: TrainConfig, distributed: DistributedContext | None = None) -> torch.device:
    if distributed is not None and distributed.enabled:
        return distributed.device
    if config.teacher_type == "hf" and config.hf_teacher.device_map == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_training_tokenizer(config: TrainConfig) -> Any | None:
    if config.data.dataset_type == "mock":
        return None
    tokenizer_name_or_path = config.data.tokenizer_name_or_path
    if tokenizer_name_or_path is None and config.teacher_type == "hf":
        tokenizer_name_or_path = config.hf_teacher.model_name_or_path
    tokenizer = load_tokenizer(
        TokenizerConfig(
            tokenizer_name_or_path=tokenizer_name_or_path,
            use_fast=config.data.use_fast,
            trust_remote_code=config.data.trust_remote_code,
            local_files_only=config.data.local_files_only,
            pad_token_strategy=config.data.pad_token_strategy,
        )
    )
    return tokenizer


def _build_training_dataset(
    config: TrainConfig,
    *,
    tokenizer: Any | None,
    vocab_size: int,
) -> torch.utils.data.Dataset[dict[str, Tensor]]:
    if config.data.dataset_type == "mock":
        return MockTextDataset(
            vocab_size=vocab_size,
            seq_len=config.mock.seq_len,
            num_samples=config.mock.num_samples,
            seed=config.seed,
            ignore_index=config.vocab.ignored_label_id,
        )
    if tokenizer is None:
        raise ValueError("Tokenizer is required for text/jsonl datasets.")
    if config.data.path is None:
        raise ValueError("--data-path or data.path is required for text/jsonl datasets.")
    return TokenizedTextDataset(
        tokenizer=tokenizer,
        config=TextDatasetConfig(
            path=config.data.path,
            seq_len=config.data.seq_len,
            stride=config.data.stride,
            max_examples=config.data.max_examples,
            seed=config.data.seed,
            add_eos=config.data.add_eos,
            shuffle=config.data.shuffle,
            file_format=config.data.dataset_type,
            text_field=config.data.text_field,
            ignore_index=config.vocab.ignored_label_id,
        ),
    )


def _teacher_vocab_size(teacher: MockTeacherWrapper | HuggingFaceTeacherWrapper) -> int:
    return int(teacher.vocab_size)


def _build_teacher(config: TrainConfig, device: torch.device) -> MockTeacherWrapper | HuggingFaceTeacherWrapper:
    if config.teacher_type == "mock":
        return MockTeacherWrapper(config.mock.vocab_size, config.mock.hidden_size).to(device)
    if config.teacher_type == "hf":
        if not config.hf_teacher.model_name_or_path:
            raise ValueError(
                "HF teacher training requires --teacher-model-name-or-path or a model_name_or_path in config."
            )
        teacher = HuggingFaceTeacherWrapper(
            HuggingFaceTeacherConfig(
                model_name_or_path=config.hf_teacher.model_name_or_path,
                torch_dtype=config.hf_teacher.torch_dtype,
                device_map=config.hf_teacher.device_map,
                trust_remote_code=config.hf_teacher.trust_remote_code,
                attn_implementation=config.hf_teacher.attn_implementation,
                local_files_only=config.hf_teacher.local_files_only,
                use_safetensors=config.hf_teacher.use_safetensors,
                load_in_8bit=config.hf_teacher.load_in_8bit,
                load_in_4bit=config.hf_teacher.load_in_4bit,
            )
        )
        if config.hf_teacher.device_map is None:
            teacher = teacher.to(device)
        return teacher
    raise ValueError(f"Unsupported teacher_type {config.teacher_type!r}.")


def _build_student(config: TrainConfig, teacher_vocab_size: int, device: torch.device) -> StudentMamba:
    if config.student_type == "mock":
        return MockStudentMamba(
            vocab_size=teacher_vocab_size,
            hidden_size=config.mock.hidden_size,
            off_config=OffTrajectoryConfig(),
        ).to(device)
    if config.student_type == "mamba":
        student_vocab_size = config.mamba_student.vocab_size
        if not config.student_vocab_size_explicit:
            student_vocab_size = teacher_vocab_size
        student_config = replace(
            config.mamba_student,
            vocab_size=student_vocab_size,
            device=config.mamba_student.device or str(device),
        )
        student = RealMambaStudent(
            student_config,
            off_config=OffTrajectoryConfig(
                delta_perturb_eps=student_config.delta_perturb_eps,
                noise_sigma=student_config.noise_sigma,
                detach_direction=student_config.off_state_detach_direction,
            ),
        )
        return student.to(torch.device(student_config.device or str(device)))
    raise ValueError(f"Unsupported student_type {config.student_type!r}.")


def _effective_student_vocab_size(config: TrainConfig, teacher_vocab_size: int) -> int:
    if config.student_type == "mock":
        return teacher_vocab_size
    if config.student_type == "mamba":
        if not config.student_vocab_size_explicit:
            return teacher_vocab_size
        return config.mamba_student.vocab_size
    raise ValueError(f"Unsupported student_type {config.student_type!r}.")


def _tokenizer_special_id(tokenizer: Any | None, name: str) -> int | None:
    if tokenizer is None:
        return None
    value = getattr(tokenizer, name, None)
    return None if value is None else int(value)


def _validate_vocab_for_training(
    config: TrainConfig,
    *,
    tokenizer_vocab_size: int | None,
    teacher_vocab_size: int,
    tokenizer: Any | None,
) -> Any:
    student_vocab_size = _effective_student_vocab_size(config, teacher_vocab_size)
    try:
        return validate_vocab_alignment(
            tokenizer_vocab_size=tokenizer_vocab_size,
            teacher_vocab_size=teacher_vocab_size,
            student_vocab_size=student_vocab_size,
            pad_token_id=_tokenizer_special_id(tokenizer, "pad_token_id"),
            eos_token_id=_tokenizer_special_id(tokenizer, "eos_token_id"),
            strict=config.vocab.strict_alignment,
            ignored_label_id=config.vocab.ignored_label_id,
        )
    except ValueError as exc:
        if config.vocab.allow_student_vocab_resize or config.mamba_student.allow_student_vocab_resize:
            raise NotImplementedError(
                "Student vocab resizing was requested but Stage 7B does not silently resize "
                "student embeddings/projection heads. Rebuild the student with a matching vocab "
                "or add an explicit resize implementation for the selected student type. "
                f"tokenizer={tokenizer_vocab_size}, teacher={teacher_vocab_size}, student={student_vocab_size}."
            ) from exc
        raise ValueError(
            "vocab sizes must match for teacher/student training alignment: "
            f"tokenizer={tokenizer_vocab_size}, teacher={teacher_vocab_size}, "
            f"student={student_vocab_size}. "
            "Teacher and student logits must share a vocab size. Tokenizer vocab may be "
            "smaller than the model vocab for padded embedding/logit tables, but it must "
            "not exceed the model vocab. Omit --student-vocab-size to align the student "
            "automatically when supported, or enable an explicit resize path later. "
            f"{exc}"
        ) from exc


def _teacher_cache_extra(config: TrainConfig, teacher_vocab_size: int) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "teacher_type": config.teacher_type,
        "vocab_size": teacher_vocab_size,
        "teacher_compute_precision": config.mixed_precision,
        "cache_dtype": config.teacher_cache.dtype,
    }
    if config.teacher_type == "mock":
        extra.update(
            {
                "mock_hidden_size": config.mock.hidden_size,
                "teacher_seed": config.seed,
                "teacher_impl": "MockTeacherWrapper",
            }
        )
    elif config.teacher_type == "hf":
        extra.update(
            {
                "model_name_or_path": config.hf_teacher.model_name_or_path,
                "torch_dtype": config.hf_teacher.torch_dtype,
                "trust_remote_code": config.hf_teacher.trust_remote_code,
                "attn_implementation": config.hf_teacher.attn_implementation,
                "use_safetensors": config.hf_teacher.use_safetensors,
                "load_in_8bit": config.hf_teacher.load_in_8bit,
                "load_in_4bit": config.hf_teacher.load_in_4bit,
                "teacher_impl": "HuggingFaceTeacherWrapper",
            }
        )
    return extra


def _teacher_logits_from_cache_entry(entry: LogitCacheEntry) -> Tensor:
    if entry.logits is not None:
        return entry.logits
    if entry.topk_values is not None and entry.topk_indices is not None:
        raise NotImplementedError(
            "Top-k-only teacher logit cache entries cannot be used by train.py yet. "
            "Use full-logit cache entries for now, or disable --teacher-cache-use-top-k; "
            "top-k-only cache-to-loss support is future work."
        )
    raise ValueError("Teacher logit cache entry contained neither full logits nor top-k tensors.")


def _checkpoint_metadata(
    config: TrainConfig,
    *,
    tokenizer_vocab_size: int | None,
    teacher_vocab_size: int,
    vocab_report: Any,
    distributed: DistributedContext | None = None,
) -> dict[str, Any]:
    student_vocab_size = _effective_student_vocab_size(config, teacher_vocab_size)
    return {
        "project_stage": "7E",
        "teacher_type": config.teacher_type,
        "teacher_model_name_or_path": config.hf_teacher.model_name_or_path,
        "teacher_vocab_size": teacher_vocab_size,
        "student_type": config.student_type,
        "student_vocab_size": student_vocab_size,
        "student_hidden_size": config.mock.hidden_size if config.student_type == "mock" else config.mamba_student.hidden_size,
        "student_num_layers": None if config.student_type == "mock" else config.mamba_student.num_layers,
        "student_state_extraction": None if config.student_type == "mock" else config.mamba_student.state_extraction,
        "off_state_mode": "projection" if config.student_type == "mock" else config.mamba_student.off_state_mode,
        "delta_alt_mode": "delta_projection" if config.student_type == "mock" else config.mamba_student.delta_alt_mode,
        "off_logits_mode": "lm_head" if config.student_type == "mock" else config.mamba_student.off_logits_mode,
        "off_state_detach_direction": None
        if config.student_type == "mock"
        else config.mamba_student.off_state_detach_direction,
        "tokenizer_name_or_path": config.data.tokenizer_name_or_path,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "dataset_type": config.data.dataset_type,
        "data_path": config.data.path,
        "seq_len": config.mock.seq_len if config.data.dataset_type == "mock" else config.data.seq_len,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "mixed_precision": config.mixed_precision,
        "learning_rate": config.learning_rate,
        "max_grad_norm": config.max_grad_norm,
        "topk_enabled": config.topk.enabled,
        "top_k": config.topk.top_k,
        "topk_renormalize": config.topk.renormalize_topk,
        "teacher_cache_enabled": config.teacher_cache.enabled,
        "csdm_weight": config.loss.csdm_weight,
        "kd_weight": config.loss.kd_weight,
        "ce_weight": config.loss.ce_weight,
        "tau": config.loss.tau,
        "distributed_enabled": bool(distributed.enabled) if distributed is not None else False,
        "distributed_world_size": distributed.world_size if distributed is not None else 1,
        "distributed_rank": distributed.rank if distributed is not None else 0,
        "distributed_local_rank": distributed.local_rank if distributed is not None else 0,
        "effective_batch_size": effective_batch_size(
            config.mock.batch_size,
            config.gradient_accumulation_steps,
            distributed.world_size if distributed is not None else 1,
        ),
        "vocab_alignment": asdict(vocab_report) if hasattr(vocab_report, "__dataclass_fields__") else vocab_report,
    }


def _resolve_resume_checkpoint(config: TrainConfig) -> Path | None:
    if config.checkpoint.resume_from:
        return Path(config.checkpoint.resume_from)
    if config.checkpoint.auto_resume:
        return latest_checkpoint(config.checkpoint.output_dir)
    return None


def _validate_resume_metadata(
    checkpoint_metadata: dict[str, Any],
    expected_metadata: dict[str, Any],
    *,
    strict: bool,
) -> None:
    if not strict:
        return
    keys = (
        "teacher_type",
        "teacher_model_name_or_path",
        "teacher_vocab_size",
        "student_type",
        "student_vocab_size",
        "student_hidden_size",
        "student_num_layers",
        "student_state_extraction",
        "off_state_mode",
        "delta_alt_mode",
        "off_logits_mode",
        "off_state_detach_direction",
        "tokenizer_name_or_path",
        "tokenizer_vocab_size",
        "dataset_type",
        "data_path",
        "seq_len",
        "gradient_accumulation_steps",
        "mixed_precision",
        "learning_rate",
        "max_grad_norm",
        "topk_enabled",
        "top_k",
        "topk_renormalize",
        "teacher_cache_enabled",
        "csdm_weight",
        "kd_weight",
        "ce_weight",
        "tau",
    )
    mismatches: list[str] = []
    for key in keys:
        if checkpoint_metadata.get(key) != expected_metadata.get(key):
            mismatches.append(
                f"{key}: checkpoint={checkpoint_metadata.get(key)!r}, current={expected_metadata.get(key)!r}"
            )
    if mismatches:
        raise ValueError("Checkpoint metadata is incompatible with the current run: " + "; ".join(mismatches))


def _normalized_resume_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    normalized = dict(config)
    normalized.pop("checkpoint", None)
    normalized.pop("storage", None)
    teacher_cache = normalized.get("teacher_cache")
    if isinstance(teacher_cache, dict):
        teacher_cache = dict(teacher_cache)
        teacher_cache.pop("cache_dir", None)
        teacher_cache.pop("overwrite", None)
        normalized["teacher_cache"] = teacher_cache
    return normalized


def _validate_resume_config_snapshot(
    checkpoint_config: dict[str, Any] | None,
    current_config: dict[str, Any],
    *,
    strict: bool,
) -> None:
    if not strict:
        return
    saved = _normalized_resume_config(checkpoint_config)
    current = _normalized_resume_config(current_config)
    if saved is None:
        raise ValueError("Checkpoint config snapshot is missing; cannot strict-resume.")
    if saved != current:
        mismatches = _resume_config_mismatches(saved, current)
        details = "; ".join(mismatches[:20])
        suffix = f": {details}" if details else "."
        raise ValueError("Checkpoint config snapshot is incompatible with the current run" + suffix)


def _resume_config_mismatches(
    checkpoint_config: dict[str, Any],
    current_config: dict[str, Any],
    *,
    prefix: str = "",
) -> list[str]:
    sentinel = object()
    mismatches: list[str] = []
    for key in sorted(set(checkpoint_config) | set(current_config)):
        path = f"{prefix}.{key}" if prefix else str(key)
        checkpoint_value = checkpoint_config.get(key, sentinel)
        current_value = current_config.get(key, sentinel)
        if isinstance(checkpoint_value, dict) and isinstance(current_value, dict):
            mismatches.extend(_resume_config_mismatches(checkpoint_value, current_value, prefix=path))
        elif checkpoint_value != current_value:
            checkpoint_display = "<missing>" if checkpoint_value is sentinel else repr(checkpoint_value)
            current_display = "<missing>" if current_value is sentinel else repr(current_value)
            mismatches.append(f"{path}: checkpoint={checkpoint_display}, current={current_display}")
    return mismatches


def _load_resume_state_if_needed(
    config: TrainConfig,
    *,
    student: StudentMamba,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    expected_metadata: dict[str, Any],
    logger: ConsoleLogger,
) -> int:
    resume_path = _resolve_resume_checkpoint(config)
    if resume_path is None:
        return 0
    checkpoint_payload = load_checkpoint(resume_path, map_location=device)
    checkpoint_metadata = checkpoint_payload.get("metadata") or {}
    if not isinstance(checkpoint_metadata, dict):
        raise TypeError("training checkpoint metadata must be a dict.")
    _validate_resume_config_snapshot(
        checkpoint_payload.get("config"),
        asdict(config),
        strict=config.checkpoint.strict_resume,
    )
    _validate_resume_metadata(
        checkpoint_metadata,
        expected_metadata,
        strict=config.checkpoint.strict_resume,
    )
    state = load_training_checkpoint(
        resume_path,
        student,
        optimizer=optimizer,
        map_location=device,
        strict=config.checkpoint.strict_resume,
        load_optimizer=config.checkpoint.load_optimizer,
        load_rng_state=config.checkpoint.load_rng_state,
    )
    logger.log(
        state.optimizer_step,
        {
            "event": "resume",
            "resume_from": str(state.path),
            "optimizer_step": state.optimizer_step,
        },
    )
    return state.optimizer_step


def _save_training_state(
    config: TrainConfig,
    *,
    student: StudentMamba,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: dict[str, Any],
) -> Path:
    return save_training_checkpoint(
        config.checkpoint.output_dir,
        student,
        optimizer,
        step=step,
        optimizer_step=step,
        config=asdict(config),
        metadata=metadata,
        rng_state=True,
    )


def _advance_batches(batches: Iterator[dict[str, Tensor]], count: int) -> None:
    for _ in range(count):
        next(batches)


def _storage_preflight_paths(config: TrainConfig) -> list[str]:
    paths: list[str] = []
    if config.teacher_cache.enabled:
        paths.append(config.teacher_cache.cache_dir)
    if config.checkpoint.save_every_steps > 0 or config.checkpoint.save_at_end:
        paths.append(config.checkpoint.output_dir)
    return list(dict.fromkeys(paths))


def _run_storage_preflight(config: TrainConfig) -> None:
    if config.storage.min_free_gb <= 0:
        return
    paths = _storage_preflight_paths(config)
    if not paths:
        return
    validate_storage_paths(paths, min_free_gb=config.storage.min_free_gb)


def run_training(config: TrainConfig, max_steps: int, logger: ConsoleLogger | None = None) -> None:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if config.checkpoint.save_every_steps < 0:
        raise ValueError("checkpoint.save_every_steps must be non-negative.")
    if config.storage.min_free_gb < 0:
        raise ValueError("storage.min_free_gb must be non-negative.")
    if config.teacher_cache.enabled and config.teacher_cache.use_top_k:
        raise NotImplementedError(
            "Top-k-only teacher logit cache entries cannot be used by train.py yet. "
            "Use full-logit cache entries for now, or disable --teacher-cache-use-top-k; "
            "top-k-only cache-to-loss support is future work."
        )
    if (
        config.student_type == "mamba"
        and config.loss.csdm_weight > 0
        and (
            config.mamba_student.off_state_mode == "placeholder"
            or config.mamba_student.off_state_mode == "none"
            or config.mamba_student.off_logits_mode == "placeholder"
            or config.mamba_student.state_extraction == "none"
            or config.mamba_student.delta_alt_mode != "delta_projection"
        )
    ):
        raise ValueError(
            "RealMambaStudent CSDM smoke training requires an exposed approximate off-state path. "
            "Use --student-state-extraction last_hidden, --off-state-mode projection, "
            "--delta-alt-mode delta_projection, and a non-placeholder --off-logits-mode, "
            "or set --csdm-weight 0.0."
        )

    distributed = init_distributed(mode=config.distributed.mode, backend=config.distributed.backend)
    try:
        set_seed(config.seed)
        device = _training_device(config, distributed)
        config = _apply_distributed_teacher_cache_policy(config, distributed)
        _run_storage_preflight(config)
        _run_training_inner(config, max_steps=max_steps, logger=logger, distributed=distributed, device=device)
    finally:
        cleanup_distributed(distributed)


def _apply_distributed_teacher_cache_policy(config: TrainConfig, distributed: DistributedContext) -> TrainConfig:
    if not distributed.enabled or not config.teacher_cache.enabled:
        return config
    policy = config.teacher_cache.distributed_policy
    if policy == "rank_local":
        return replace(
            config,
            teacher_cache=replace(
                config.teacher_cache,
                cache_dir=rank_local_dir(config.teacher_cache.cache_dir, distributed),
            ),
        )
    if policy == "rank_zero_write":
        return config if distributed.is_rank_zero else replace(
            config,
            teacher_cache=replace(config.teacher_cache, enabled=False),
        )
    if policy == "shared_readonly":
        raise NotImplementedError(
            "teacher_cache.distributed_policy=shared_readonly is reserved for prefilled caches. "
            "Use rank_local for Stage 7E smoke runs."
        )
    raise ValueError(
        "teacher_cache.distributed_policy must be one of: rank_local, shared_readonly, rank_zero_write."
    )


def _run_training_inner(
    config: TrainConfig,
    *,
    max_steps: int,
    logger: ConsoleLogger | None,
    distributed: DistributedContext,
    device: torch.device,
) -> None:
    tokenizer = _load_training_tokenizer(config)
    tokenizer_vocab_size = None
    if tokenizer is not None:
        tokenizer_vocab_size = get_tokenizer_vocab_size(tokenizer)
        if tokenizer_vocab_size <= 1:
            raise ValueError(f"tokenizer vocab size must be greater than 1, got {tokenizer_vocab_size}.")
        if config.data.dataset_type != "mock":
            config = replace(config, mock=replace(config.mock, vocab_size=tokenizer_vocab_size))
    teacher = _build_teacher(config, device)
    teacher_vocab_size = _teacher_vocab_size(teacher)
    if teacher_vocab_size <= 1:
        raise ValueError(f"teacher vocab_size must be greater than 1, got {teacher_vocab_size}.")
    vocab_report = _validate_vocab_for_training(
        config,
        tokenizer_vocab_size=tokenizer_vocab_size,
        teacher_vocab_size=teacher_vocab_size,
        tokenizer=tokenizer,
    )

    dataset = _build_training_dataset(
        config,
        tokenizer=tokenizer,
        vocab_size=teacher_vocab_size,
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=False,
            drop_last=False,
        )
        if distributed.enabled
        else None
    )
    loader = DataLoader(dataset, batch_size=config.mock.batch_size, shuffle=False, sampler=sampler, drop_last=False)
    batches = infinite_loader(loader, sampler=sampler)

    student = _build_student(config, teacher_vocab_size, device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)
    base_logger = logger if logger is not None else ConsoleLogger()
    logger = RankZeroLogger(base_logger, distributed)
    autocast_context = _autocast_context(config.mixed_precision, device)
    teacher_cache = TeacherLogitCache(config.teacher_cache) if config.teacher_cache.enabled else None
    teacher_cache_extra = _teacher_cache_extra(config, teacher_vocab_size)
    checkpoint_metadata = _checkpoint_metadata(
        config,
        tokenizer_vocab_size=tokenizer_vocab_size,
        teacher_vocab_size=teacher_vocab_size,
        vocab_report=vocab_report,
        distributed=distributed,
    )
    start_optimizer_step = _load_resume_state_if_needed(
        config,
        student=student,
        optimizer=optimizer,
        device=device,
        expected_metadata=checkpoint_metadata,
        logger=logger,
    )
    if start_optimizer_step > 0:
        _advance_batches(batches, start_optimizer_step * config.gradient_accumulation_steps)
    if start_optimizer_step >= max_steps:
        return
    for step in range(start_optimizer_step + 1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_metrics = {"total": 0.0, "ce": 0.0, "kd": 0.0, "csdm": 0.0}

        for micro_step in range(1, config.gradient_accumulation_steps + 1):
            batch = next(batches)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            validate_token_id_ranges(
                input_ids,
                labels,
                input_vocab_size=tokenizer_vocab_size or teacher_vocab_size,
                label_vocab_size=teacher_vocab_size,
                ignored_label_id=config.vocab.ignored_label_id,
            )
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            elif config.teacher_type == "hf":
                attention_mask = torch.ones_like(input_ids)
            with autocast_context:
                if teacher_cache is None:
                    teacher_logits = teacher(input_ids, attention_mask=attention_mask)
                else:
                    entry = teacher_cache.get_or_compute(
                        input_ids,
                        compute_fn=lambda clean_input_ids, attention_mask=None: teacher(
                            clean_input_ids,
                            attention_mask=attention_mask,
                        ),
                        attention_mask=attention_mask,
                        extra=teacher_cache_extra,
                    )
                    teacher_logits = _teacher_logits_from_cache_entry(entry)
                output = student(input_ids)
                loss_device = output.on_logits.device
                teacher_logits = teacher_logits.to(loss_device)
                labels = labels.to(loss_device)
                losses = compute_losses(output, teacher_logits, labels, config)
                for name, value in losses.items():
                    if not torch.isfinite(value):
                        raise FloatingPointError(f"non-finite {name} loss encountered before backward")
                scaled_loss = losses["total"] / config.gradient_accumulation_steps
            scaled_loss.backward()
            for name in accum_metrics:
                accum_metrics[name] += float(losses[name].detach().cpu())

        grad_before_clip = grad_norm(list(student.parameters()))
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm)
        optimizer.step()

        divisor = float(config.gradient_accumulation_steps)
        metrics = {name: value / divisor for name, value in accum_metrics.items()}
        metrics["grad_norm"] = grad_before_clip
        metrics = average_float_dict(metrics, distributed)
        metrics["optimizer_step"] = step
        metrics["micro_step"] = config.gradient_accumulation_steps
        metrics["accumulation_steps"] = config.gradient_accumulation_steps
        metrics["accumulation_progress"] = f"{config.gradient_accumulation_steps}/{config.gradient_accumulation_steps}"
        metrics["rank"] = distributed.rank
        metrics["local_rank"] = distributed.local_rank
        metrics["world_size"] = distributed.world_size
        metrics["effective_batch_size"] = effective_batch_size(
            config.mock.batch_size,
            config.gradient_accumulation_steps,
            distributed.world_size,
        )
        if torch.cuda.is_available():
            metrics["cuda_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        should_save = config.checkpoint.save_every_steps > 0 and step % config.checkpoint.save_every_steps == 0
        should_save = should_save or (config.checkpoint.save_at_end and step == max_steps)
        if should_save and distributed.is_rank_zero:
            checkpoint_path = _save_training_state(
                config,
                student=student,
                optimizer=optimizer,
                step=step,
                metadata=checkpoint_metadata,
            )
            metrics["checkpoint_path"] = str(checkpoint_path)
        if should_save:
            barrier(distributed)
        logger.log(step, metrics)


def run_mock_training(config: TrainConfig, max_steps: int) -> None:
    run_training(replace(config, teacher_type="mock", student_type="mock"), max_steps=max_steps)


def main() -> None:
    args = parse_args()
    try:
        config = derive_runtime_config(args)
        run_training(config, max_steps=args.max_steps)
    except (ImportError, NotImplementedError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
