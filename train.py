"""Stage 3 mock training scaffold for CSDM Mamba KD."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field
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
from models.teacher_wrapper import MockTeacherWrapper
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
class TrainConfig:
    seed: int = 42
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    top_k_kd: int | None = None
    loss: LossConfig = field(default_factory=LossConfig)
    mock: MockConfig = field(default_factory=MockConfig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mock", action="store_true", help="Run the mock-only Stage 3 scaffold.")
    parser.add_argument("--max_steps", type=int, default=2, help="Number of optimizer steps.")
    return parser.parse_args()


def _nested_get(data: dict[str, Any], key: str, default: Any) -> Any:
    return data.get(key, default)


def load_train_config(path: Path) -> TrainConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    loss_raw = raw.get("loss", {})
    mock_raw = raw.get("mock", {})
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
    return TrainConfig(
        seed=int(_nested_get(raw, "seed", TrainConfig.seed)),
        mixed_precision=str(raw.get("mixed_precision", raw.get("precision", TrainConfig.mixed_precision))),
        gradient_accumulation_steps=int(
            _nested_get(raw, "gradient_accumulation_steps", TrainConfig.gradient_accumulation_steps)
        ),
        learning_rate=float(_nested_get(raw, "learning_rate", TrainConfig.learning_rate)),
        max_grad_norm=float(_nested_get(raw, "max_grad_norm", TrainConfig.max_grad_norm)),
        top_k_kd=top_k_kd,
        loss=loss,
        mock=mock,
    )


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


def run_mock_training(config: TrainConfig, max_steps: int) -> None:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MockTextDataset(
        vocab_size=config.mock.vocab_size,
        seq_len=config.mock.seq_len,
        num_samples=config.mock.num_samples,
        seed=config.seed,
        ignore_index=config.mock.ignore_index,
    )
    loader = DataLoader(dataset, batch_size=config.mock.batch_size, shuffle=False, drop_last=True)
    batches = infinite_loader(loader)

    teacher = MockTeacherWrapper(config.mock.vocab_size, config.mock.hidden_size).to(device)
    student = MockStudentMamba(
        vocab_size=config.mock.vocab_size,
        hidden_size=config.mock.hidden_size,
        off_config=OffTrajectoryConfig(),
    ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)
    logger = ConsoleLogger()

    use_bf16 = config.mixed_precision == "bf16" and device.type == "cuda"
    autocast_context = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    )

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_metrics = {"total": 0.0, "ce": 0.0, "kd": 0.0, "csdm": 0.0}

        for _ in range(config.gradient_accumulation_steps):
            batch = next(batches)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with autocast_context:
                teacher_logits = teacher(input_ids)
                output = student(input_ids)
                losses = compute_losses(output, teacher_logits, labels, config)
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
        if torch.cuda.is_available():
            metrics["cuda_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        logger.log(step, metrics)


def main() -> None:
    args = parse_args()
    if not args.mock:
        raise SystemExit("Only --mock Stage 3 training is implemented; real Llama/Mamba imports are not used.")
    config = load_train_config(args.config)
    run_mock_training(config, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
