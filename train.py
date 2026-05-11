"""Training scaffold for CSDM Mamba KD smoke paths."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data.dataset import MockTextDataset
from losses.cdm_loss import csdm_loss
from losses.kd_loss import kd_kl_loss
from models.cdm_engine import OffTrajectoryConfig
from models.student_mamba import MockStudentMamba, StudentOutput
from models.teacher_wrapper import HuggingFaceTeacherConfig, HuggingFaceTeacherWrapper, MockTeacherWrapper
from utils.logger import ConsoleLogger


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
class TrainConfig:
    seed: int = 42
    teacher_type: str = "mock"
    student_type: str = "mock"
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    top_k_kd: int | None = None
    loss: LossConfig = field(default_factory=LossConfig)
    mock: MockConfig = field(default_factory=MockConfig)
    hf_teacher: HuggingFaceRuntimeConfig = field(default_factory=HuggingFaceRuntimeConfig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mock", action="store_true", help="Run the mock-only Stage 3 scaffold.")
    parser.add_argument("--max_steps", type=int, default=2, help="Number of optimizer steps.")
    parser.add_argument("--teacher-type", choices=("mock", "hf"), default=None)
    parser.add_argument("--student-type", choices=("mock",), default=None)
    parser.add_argument("--teacher-model-name-or-path", default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default=None)
    parser.add_argument("--csdm-weight", type=float, default=None)
    parser.add_argument("--kd-weight", type=float, default=None)
    parser.add_argument("--ce-weight", type=float, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _nested_get(data: dict[str, Any], key: str, default: Any) -> Any:
    return data.get(key, default)


def _normalize_student_type(value: str) -> str:
    if value in {"mock", "mock_mamba"}:
        return "mock"
    raise ValueError(f"Unsupported student_type {value!r}; expected 'mock'.")


def _normalize_teacher_type(value: str) -> str:
    if value in {"mock", "hf"}:
        return value
    raise ValueError(f"Unsupported teacher_type {value!r}; expected 'mock' or 'hf'.")


def _optional_str(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def load_train_config(path: Path) -> TrainConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    loss_raw = raw.get("loss", {})
    mock_raw = raw.get("mock", {})
    teacher_raw = raw.get("teacher", {})
    student_raw = raw.get("student", {})
    hf_raw = raw.get("hf_teacher", raw.get("hf_teacher_example", {}))
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
    top_k_raw = raw.get("top_k_kd")
    top_k_kd = None if top_k_raw in (None, "") else int(top_k_raw)
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
        top_k_kd=top_k_kd,
        loss=loss,
        mock=mock,
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


def derive_runtime_config(args: argparse.Namespace) -> TrainConfig:
    """Apply CLI precedence: YAML, HF smoke defaults, then explicit CLI flags."""

    config = load_train_config(args.config)

    if args.mock:
        config = replace(config, teacher_type="mock", student_type="mock")
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

    if not args.mock and args.teacher_type is not None:
        config = replace(config, teacher_type=args.teacher_type)
    if not args.mock and args.student_type is not None:
        config = replace(config, student_type=args.student_type)
    if args.teacher_model_name_or_path is not None:
        config = replace(
            config,
            hf_teacher=replace(config.hf_teacher, model_name_or_path=args.teacher_model_name_or_path),
        )
    if args.local_files_only:
        config = replace(config, hf_teacher=replace(config.hf_teacher, local_files_only=True))
    if args.seq_len is not None:
        config = replace(config, mock=replace(config.mock, seq_len=args.seq_len))
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
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infinite_loader(loader: DataLoader[dict[str, Tensor]]) -> Iterator[dict[str, Tensor]]:
    while True:
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


def _top_k_pair(student_logits: Tensor, teacher_logits: Tensor, top_k: int | None) -> tuple[Tensor, Tensor]:
    if top_k is None or top_k <= 0 or top_k >= teacher_logits.shape[-1]:
        return student_logits, teacher_logits
    values, indices = teacher_logits.topk(top_k, dim=-1)
    student_topk = student_logits.gather(dim=-1, index=indices)
    return student_topk, values


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
        ignore_index=config.mock.ignore_index,
        positions_per_sequence=config.mock.positions_per_sequence,
    )
    ce_logits = output.on_logits[mask]
    ce_labels = labels[mask]
    ce = F.cross_entropy(ce_logits.float(), ce_labels, ignore_index=config.mock.ignore_index)

    on_masked = _masked_sequence_logits(output.on_logits, mask)
    off_masked = _masked_sequence_logits(output.off_logits, mask)
    teacher_masked = _masked_sequence_logits(teacher_logits, mask)
    fake_masked = _masked_sequence_logits(output.fake_logits, mask)

    kd_student, kd_teacher = _top_k_pair(on_masked, teacher_masked, config.top_k_kd)
    kd = kd_kl_loss(kd_student.float(), kd_teacher.float(), tau=config.loss.tau)
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


def _training_device(config: TrainConfig) -> torch.device:
    if config.teacher_type == "hf" and config.hf_teacher.device_map == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _teacher_vocab_size(teacher: MockTeacherWrapper | HuggingFaceTeacherWrapper) -> int:
    if isinstance(teacher, MockTeacherWrapper):
        return int(teacher.embedding.num_embeddings)
    vocab_size = getattr(getattr(teacher.model, "config", None), "vocab_size", None)
    if vocab_size is None:
        raise ValueError("HuggingFace teacher model config must expose vocab_size for smoke training.")
    return int(vocab_size)


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


def run_training(config: TrainConfig, max_steps: int, logger: ConsoleLogger | None = None) -> None:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if config.student_type != "mock":
        raise ValueError("Only mock student training is implemented.")

    set_seed(config.seed)
    device = _training_device(config)
    teacher = _build_teacher(config, device)
    teacher_vocab_size = _teacher_vocab_size(teacher)
    if teacher_vocab_size <= 1:
        raise ValueError(f"teacher vocab_size must be greater than 1, got {teacher_vocab_size}.")

    dataset = MockTextDataset(
        vocab_size=teacher_vocab_size,
        seq_len=config.mock.seq_len,
        num_samples=config.mock.num_samples,
        seed=config.seed,
        ignore_index=config.mock.ignore_index,
    )
    loader = DataLoader(dataset, batch_size=config.mock.batch_size, shuffle=False, drop_last=True)
    batches = infinite_loader(loader)

    student = MockStudentMamba(
        vocab_size=teacher_vocab_size,
        hidden_size=config.mock.hidden_size,
        off_config=OffTrajectoryConfig(),
    ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)
    logger = logger if logger is not None else ConsoleLogger()
    autocast_context = _autocast_context(config.mixed_precision, device)

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_metrics = {"total": 0.0, "ce": 0.0, "kd": 0.0, "csdm": 0.0}

        for micro_step in range(1, config.gradient_accumulation_steps + 1):
            batch = next(batches)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = torch.ones_like(input_ids) if config.teacher_type == "hf" else None
            with autocast_context:
                teacher_logits = teacher(input_ids, attention_mask=attention_mask)
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
        metrics["optimizer_step"] = step
        metrics["micro_step"] = config.gradient_accumulation_steps
        metrics["accumulation_steps"] = config.gradient_accumulation_steps
        metrics["accumulation_progress"] = f"{config.gradient_accumulation_steps}/{config.gradient_accumulation_steps}"
        if torch.cuda.is_available():
            metrics["cuda_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        logger.log(step, metrics)


def run_mock_training(config: TrainConfig, max_steps: int) -> None:
    run_training(replace(config, teacher_type="mock", student_type="mock"), max_steps=max_steps)


def main() -> None:
    args = parse_args()
    try:
        config = derive_runtime_config(args)
        run_training(config, max_steps=args.max_steps)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
