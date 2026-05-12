"""Stage 4 mock evaluation CLI for CSDM Mamba KD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.vocab import get_tokenizer_vocab_size
from data.dataset import MockTextDataset
from evals.needle import evaluate_needle_scaffold
from evals.perplexity import evaluate_perplexity
from evals.perturbation_robustness import evaluate_perturbation_robustness
from models.cdm_engine import OffTrajectoryConfig
from models.student_mamba import MockStudentMamba
from models.teacher_wrapper import MockTeacherWrapper
from train import (
    _build_student,
    _build_teacher,
    _build_training_dataset,
    _load_training_tokenizer,
    _teacher_vocab_size,
    _training_device,
    _validate_vocab_for_training,
    derive_runtime_config,
    load_train_config,
    set_seed,
)
from utils.checkpointing import StudentCheckpointState, load_student_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mock", action="store_true", help="Run mock-only Stage 4 evaluation.")
    parser.add_argument("--student-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-type", choices=("mock", "hf"), default=None)
    parser.add_argument("--student-type", choices=("mock", "mamba"), default=None)
    parser.add_argument("--teacher-model-name-or-path", default=None)
    parser.add_argument("--hf-torch-dtype", choices=("float32", "float16", "bfloat16"), default=None)
    parser.add_argument("--hf-device-map", default=None)
    parser.add_argument("--tokenizer-name-or-path", default=None)
    parser.add_argument("--dataset-type", choices=("mock", "text", "jsonl"), default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--student-model-name-or-path", default=None)
    parser.add_argument("--student-vocab-size", type=int, default=None)
    parser.add_argument("--student-hidden-size", type=int, default=None)
    parser.add_argument("--student-num-layers", type=int, default=None)
    parser.add_argument("--student-state-extraction", choices=("last_hidden", "embedding", "none"), default=None)
    parser.add_argument("--off-state-mode", choices=("projection", "placeholder", "none"), default=None)
    parser.add_argument("--delta-alt-mode", choices=("delta_projection", "noise", "identity"), default=None)
    parser.add_argument("--off-logits-mode", choices=("lm_head", "projection_head", "placeholder"), default=None)
    parser.add_argument("--off-state-detach-direction", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default=None)
    parser.add_argument("--csdm-weight", type=float, default=None)
    parser.add_argument("--kd-weight", type=float, default=None)
    parser.add_argument("--ce-weight", type=float, default=None)
    parser.add_argument("--topk-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--topk-include-labels", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--topk-renormalize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-student-vocab-resize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("all", "perplexity", "perturbation", "needle"),
        default="all",
    )
    parser.add_argument("--max-batches", "--max_batches", dest="max_batches", type=int, default=None)
    return parser.parse_args()


def _build_mock_components(config_path: Path) -> tuple[Any, torch.device, DataLoader[Any], Any, Any]:
    config = load_train_config(config_path)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MockTextDataset(
        vocab_size=config.mock.vocab_size,
        seq_len=config.mock.seq_len,
        num_samples=config.mock.num_samples,
        seed=config.seed,
        ignore_index=config.mock.ignore_index,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.mock.batch_size,
        shuffle=False,
        drop_last=False,
    )
    teacher = MockTeacherWrapper(config.mock.vocab_size, config.mock.hidden_size).to(device)
    student = MockStudentMamba(
        vocab_size=config.mock.vocab_size,
        hidden_size=config.mock.hidden_size,
        off_config=OffTrajectoryConfig(),
    ).to(device)
    return config, device, dataloader, teacher, student


def _load_checkpoint_if_requested(
    student: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> StudentCheckpointState | None:
    if args.student_checkpoint is None:
        return None
    return load_student_from_checkpoint(
        student,
        args.student_checkpoint,
        strict=True,
        map_location=device,
    )


def _build_runtime_components(args: argparse.Namespace) -> tuple[Any, torch.device, DataLoader[Any], Any, Any]:
    config = derive_runtime_config(args)
    set_seed(config.seed)
    device = _training_device(config)
    tokenizer = _load_training_tokenizer(config)
    tokenizer_vocab_size = get_tokenizer_vocab_size(tokenizer) if tokenizer is not None else None
    teacher = _build_teacher(config, device)
    teacher_vocab_size = _teacher_vocab_size(teacher)
    _validate_vocab_for_training(
        config,
        tokenizer_vocab_size=tokenizer_vocab_size,
        teacher_vocab_size=teacher_vocab_size,
        tokenizer=tokenizer,
    )
    dataset = _build_training_dataset(config, tokenizer=tokenizer, vocab_size=teacher_vocab_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.mock.batch_size,
        shuffle=False,
        drop_last=False,
    )
    student = _build_student(config, teacher_vocab_size, device)
    return config, device, dataloader, teacher, student


def _checkpoint_metadata(
    args: argparse.Namespace,
    *,
    checkpoint_state: StudentCheckpointState | None,
    student_type: str,
) -> dict[str, Any]:
    checkpoint_metadata = checkpoint_state.metadata if checkpoint_state is not None else {}
    return {
        "student_checkpoint": None if args.student_checkpoint is None else str(args.student_checkpoint),
        "student_type": student_type,
        "checkpoint_loaded": checkpoint_state is not None,
        "checkpoint_step": None if checkpoint_state is None else checkpoint_state.step,
        "checkpoint_optimizer_step": None if checkpoint_state is None else checkpoint_state.optimizer_step,
        "checkpoint_project_stage": checkpoint_metadata.get("project_stage"),
        "checkpoint_student_type": checkpoint_metadata.get("student_type"),
        "checkpoint_student_vocab_size": checkpoint_metadata.get("student_vocab_size"),
        "checkpoint_student_hidden_size": checkpoint_metadata.get("student_hidden_size"),
    }


def run_mock_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    config, device, dataloader, teacher, student = _build_mock_components(args.config)
    checkpoint_state = _load_checkpoint_if_requested(student, args, device)
    metrics: dict[str, Any] = {}

    with torch.no_grad():
        if args.mode in {"all", "perplexity"}:
            metrics["perplexity"] = evaluate_perplexity(
                student=student,
                dataloader=dataloader,
                config=config,
                device=device,
                max_batches=args.max_batches,
            )
        if args.mode in {"all", "perturbation"}:
            metrics["perturbation"] = evaluate_perturbation_robustness(
                student=student,
                teacher=teacher,
                dataloader=dataloader,
                config=config,
                device=device,
                max_batches=args.max_batches,
            )
        if args.mode in {"all", "needle"}:
            metrics["needle"] = evaluate_needle_scaffold(
                config=config,
                max_batches=args.max_batches,
            )

    if args.mode == "all":
        if args.student_checkpoint is not None:
            metrics["metadata"] = _checkpoint_metadata(args, checkpoint_state=checkpoint_state, student_type="mock")
        return metrics
    result = metrics[args.mode]
    if args.student_checkpoint is not None:
        result = dict(result)
        result["metadata"] = _checkpoint_metadata(args, checkpoint_state=checkpoint_state, student_type="mock")
    return result


def run_runtime_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    config, device, dataloader, teacher, student = _build_runtime_components(args)
    checkpoint_state = _load_checkpoint_if_requested(student, args, device)
    metrics: dict[str, Any] = {}

    with torch.no_grad():
        if args.mode in {"all", "perplexity"}:
            metrics["perplexity"] = evaluate_perplexity(
                student=student,
                dataloader=dataloader,
                config=config,
                device=device,
                max_batches=args.max_batches,
            )
        if args.mode in {"all", "perturbation"}:
            metrics["perturbation"] = evaluate_perturbation_robustness(
                student=student,
                teacher=teacher,
                dataloader=dataloader,
                config=config,
                device=device,
                max_batches=args.max_batches,
            )
        if args.mode in {"all", "needle"}:
            metrics["needle"] = evaluate_needle_scaffold(
                config=config,
                max_batches=args.max_batches,
            )

    metadata = _checkpoint_metadata(args, checkpoint_state=checkpoint_state, student_type=config.student_type)
    if args.mode == "all":
        metrics["metadata"] = metadata
        return metrics
    result = dict(metrics[args.mode])
    result["metadata"] = metadata
    return result


def main() -> None:
    args = parse_args()
    if args.mock:
        metrics = run_mock_evaluation(args)
    else:
        metrics = run_runtime_evaluation(args)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
