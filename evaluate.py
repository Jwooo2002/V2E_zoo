"""Stage 4 mock evaluation CLI for CSDM Mamba KD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.dataset import MockTextDataset
from evals.needle import evaluate_needle_scaffold
from evals.perplexity import evaluate_perplexity
from evals.perturbation_robustness import evaluate_perturbation_robustness
from models.cdm_engine import OffTrajectoryConfig
from models.student_mamba import MockStudentMamba
from models.teacher_wrapper import MockTeacherWrapper
from train import load_train_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mock", action="store_true", help="Run mock-only Stage 4 evaluation.")
    parser.add_argument(
        "--mode",
        choices=("all", "perplexity", "perturbation", "needle"),
        default="all",
    )
    parser.add_argument("--max_batches", type=int, default=None)
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


def run_mock_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    config, device, dataloader, teacher, student = _build_mock_components(args.config)
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
        return metrics
    return metrics[args.mode]


def main() -> None:
    args = parse_args()
    if not args.mock:
        raise SystemExit("Only --mock Stage 4 evaluation is implemented; real Llama/Mamba imports are not used.")
    metrics = run_mock_evaluation(args)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
