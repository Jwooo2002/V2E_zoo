#!/usr/bin/env python
"""Run perturbation robustness benchmarks for CSDM-Mamba smoke models."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.vocab import get_tokenizer_vocab_size, validate_token_id_ranges
from evals.perturbation_robustness import compute_dual_perturbation_metrics, compute_perturbation_metrics
from losses.kd_loss import build_topk_indices
from models.cdm_engine import OffTrajectoryConfig
from models.student_mamba import MockStudentMamba
from train import (
    TrainConfig,
    _build_student,
    _build_teacher,
    _build_training_dataset,
    _load_training_tokenizer,
    _select_shared_valid_mask,
    _teacher_vocab_size,
    _training_device,
    _validate_vocab_for_training,
    load_train_config,
    set_seed,
)
from utils.checkpointing import StudentCheckpointState, load_student_from_checkpoint


PERTURBATION_MODES = ("identity", "noise", "delta_projection", "placeholder")
CSV_COLUMNS = (
    "mode",
    "kl_on",
    "kl_off",
    "delta_kl",
    "num_tokens",
    "num_batches",
    "topk_enabled",
    "top_k",
    "full_vocab.kl_on",
    "full_vocab.kl_off",
    "full_vocab.delta_kl",
    "full_vocab.num_tokens",
    "topk.kl_on",
    "topk.kl_off",
    "topk.delta_kl",
    "topk.num_tokens",
    "topk.top_k",
    "checkpoint_loaded",
    "student_checkpoint",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train_config.yaml")
    parser.add_argument("--mock", action="store_true", help="Use mock teacher/student/data without external models.")
    parser.add_argument("--teacher-type", choices=("mock", "hf"), default=None)
    parser.add_argument("--student-type", choices=("mock", "mamba"), default=None)
    parser.add_argument("--student-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-model-name-or-path", default=None)
    parser.add_argument("--student-model-name-or-path", default=None)
    parser.add_argument("--tokenizer-name-or-path", default=None)
    parser.add_argument("--dataset-type", choices=("mock", "text", "jsonl"), default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--student-vocab-size", type=int, default=None)
    parser.add_argument("--student-hidden-size", type=int, default=None)
    parser.add_argument("--student-num-layers", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", "--max_batches", dest="max_batches", type=int, default=2)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--topk-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--topk-include-labels", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--topk-renormalize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dual-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--position-wise", action="store_true")
    parser.add_argument("--position-metrics", dest="position_wise", action="store_true")
    parser.add_argument("--perturbation-mode", choices=PERTURBATION_MODES, default="delta_projection")
    parser.add_argument("--sweep", default=None, help="Comma-separated perturbation modes to evaluate.")
    return parser.parse_args(argv)


def _runtime_config(args: argparse.Namespace, *, mode: str) -> TrainConfig:
    config = load_train_config(args.config)
    if args.mock:
        config = replace(config, teacher_type="mock", student_type="mock", data=replace(config.data, dataset_type="mock"))
    if args.teacher_type is not None:
        config = replace(config, teacher_type=args.teacher_type)
    if args.student_type is not None:
        config = replace(config, student_type=args.student_type)
    if args.teacher_model_name_or_path is not None:
        config = replace(
            config,
            hf_teacher=replace(config.hf_teacher, model_name_or_path=args.teacher_model_name_or_path),
        )
    if args.student_model_name_or_path is not None:
        config = replace(
            config,
            mamba_student=replace(config.mamba_student, model_name_or_path=args.student_model_name_or_path),
        )
    if args.tokenizer_name_or_path is not None:
        config = replace(config, data=replace(config.data, tokenizer_name_or_path=args.tokenizer_name_or_path))
    if args.dataset_type is not None:
        config = replace(config, data=replace(config.data, dataset_type=args.dataset_type))
    if args.data_path is not None:
        config = replace(config, data=replace(config.data, path=args.data_path))
    if args.student_vocab_size is not None:
        config = replace(
            config,
            mamba_student=replace(config.mamba_student, vocab_size=args.student_vocab_size),
            student_vocab_size_explicit=True,
        )
    if args.student_hidden_size is not None:
        config = replace(config, mamba_student=replace(config.mamba_student, hidden_size=args.student_hidden_size))
    if args.student_num_layers is not None:
        config = replace(config, mamba_student=replace(config.mamba_student, num_layers=args.student_num_layers))
    if args.seq_len is not None:
        config = replace(config, mock=replace(config.mock, seq_len=args.seq_len), data=replace(config.data, seq_len=args.seq_len))
    if args.batch_size is not None:
        config = replace(config, mock=replace(config.mock, batch_size=args.batch_size))
    if args.mixed_precision is not None:
        config = replace(config, mixed_precision=args.mixed_precision)
    if args.local_files_only:
        config = replace(
            config,
            hf_teacher=replace(config.hf_teacher, local_files_only=True),
            mamba_student=replace(config.mamba_student, local_files_only=True),
            data=replace(config.data, local_files_only=True),
        )
    if args.topk_enabled is not None:
        config = replace(config, topk=replace(config.topk, enabled=args.topk_enabled))
    if args.top_k is not None:
        config = replace(config, topk=replace(config.topk, top_k=args.top_k))
    if args.topk_include_labels is not None:
        config = replace(config, topk=replace(config.topk, include_labels=args.topk_include_labels))
    if args.topk_renormalize is not None:
        config = replace(config, topk=replace(config.topk, renormalize_topk=args.topk_renormalize))
    if args.tau is not None:
        config = replace(config, loss=replace(config.loss, tau=args.tau))
    if config.teacher_type == "hf" and config.data.dataset_type != "mock" and config.data.tokenizer_name_or_path is None:
        config = replace(config, data=replace(config.data, tokenizer_name_or_path=config.hf_teacher.model_name_or_path))
    if config.student_type == "mamba":
        config = _apply_real_mamba_mode(config, mode)
    return config


def _apply_real_mamba_mode(config: TrainConfig, mode: str) -> TrainConfig:
    student_config = config.mamba_student
    if mode == "identity":
        student_config = replace(
            student_config,
            off_state_mode="projection",
            delta_alt_mode="identity",
            noise_sigma=0.0,
        )
    elif mode == "noise":
        student_config = replace(student_config, off_state_mode="projection", delta_alt_mode="noise")
    elif mode == "delta_projection":
        student_config = replace(student_config, off_state_mode="projection", delta_alt_mode="delta_projection")
    elif mode == "placeholder":
        student_config = replace(student_config, off_state_mode="placeholder", off_logits_mode="placeholder")
    else:
        raise ValueError(f"Unsupported perturbation mode {mode!r}.")
    return replace(config, mamba_student=student_config)


def _parse_modes(args: argparse.Namespace) -> list[str]:
    raw_modes = args.sweep.split(",") if args.sweep else [args.perturbation_mode]
    modes = [mode.strip() for mode in raw_modes if mode.strip()]
    if not modes:
        raise ValueError("at least one perturbation mode is required.")
    unknown = sorted(set(modes) - set(PERTURBATION_MODES))
    if unknown:
        raise ValueError(f"Unsupported perturbation mode(s): {', '.join(unknown)}.")
    return modes


def _mock_off_config(mode: str) -> OffTrajectoryConfig:
    if mode == "identity":
        return OffTrajectoryConfig(rho_min=0.0, rho_max=0.0, noise_sigma=0.0)
    if mode == "noise":
        return OffTrajectoryConfig(rho_min=0.0, rho_max=0.0, noise_sigma=0.01)
    if mode in {"delta_projection", "placeholder"}:
        return OffTrajectoryConfig()
    raise ValueError(f"Unsupported perturbation mode {mode!r}.")


def _build_components(config: TrainConfig, *, mode: str) -> tuple[Any, Any, DataLoader[dict[str, Tensor]], torch.device]:
    set_seed(config.seed)
    device = _training_device(config)
    tokenizer = _load_training_tokenizer(config)
    tokenizer_vocab_size = None
    if tokenizer is not None:
        tokenizer_vocab_size = get_tokenizer_vocab_size(tokenizer)
        if config.data.dataset_type != "mock":
            config = replace(config, mock=replace(config.mock, vocab_size=tokenizer_vocab_size))

    teacher = _build_teacher(config, device)
    teacher_vocab_size = _teacher_vocab_size(teacher)
    _validate_vocab_for_training(
        config,
        tokenizer_vocab_size=tokenizer_vocab_size,
        teacher_vocab_size=teacher_vocab_size,
        tokenizer=tokenizer,
    )
    dataset = _build_training_dataset(config, tokenizer=tokenizer, vocab_size=teacher_vocab_size)
    dataloader = DataLoader(dataset, batch_size=config.mock.batch_size, shuffle=False, drop_last=False)
    if config.student_type == "mock":
        student = MockStudentMamba(
            vocab_size=teacher_vocab_size,
            hidden_size=config.mock.hidden_size,
            off_config=_mock_off_config(mode),
        ).to(device)
    else:
        student = _build_student(config, teacher_vocab_size, device)
    teacher.eval()
    student.eval()
    return teacher, student, dataloader, device


def _call_teacher(teacher: Any, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
    if attention_mask is None:
        return teacher(input_ids)
    return teacher(input_ids, attention_mask=attention_mask)


def _call_student(student: Any, input_ids: Tensor, attention_mask: Tensor | None) -> Any:
    if attention_mask is None:
        return student(input_ids)
    return student(input_ids, attention_mask=attention_mask)


def _accumulate_position_metrics(
    position_state: dict[str, list[float] | list[int]],
    metrics: dict[str, Any],
) -> dict[str, list[float] | list[int]]:
    counts = metrics.get("position_num_tokens")
    on_values = metrics.get("position_kl_on")
    off_values = metrics.get("position_kl_off")
    if not isinstance(counts, list) or not isinstance(on_values, list) or not isinstance(off_values, list):
        return position_state
    if not position_state:
        position_state = {
            "on_sum": [0.0 for _ in counts],
            "off_sum": [0.0 for _ in counts],
            "counts": [0 for _ in counts],
        }
    for index, (count, on_value, off_value) in enumerate(zip(counts, on_values, off_values, strict=True)):
        count_int = int(count)
        position_state["on_sum"][index] = float(position_state["on_sum"][index]) + float(on_value) * count_int
        position_state["off_sum"][index] = float(position_state["off_sum"][index]) + float(off_value) * count_int
        position_state["counts"][index] = int(position_state["counts"][index]) + count_int
    return position_state


def _finalize_position_metrics(position_state: dict[str, list[float] | list[int]]) -> dict[str, list[float]]:
    if not position_state:
        return {}
    on_sum = [float(value) for value in position_state["on_sum"]]
    off_sum = [float(value) for value in position_state["off_sum"]]
    counts = [int(value) for value in position_state["counts"]]
    on_values: list[float] = []
    off_values: list[float] = []
    delta_values: list[float] = []
    for on_total, off_total, count in zip(on_sum, off_sum, counts, strict=True):
        if count <= 0:
            on_value = 0.0
            off_value = 0.0
        else:
            on_value = on_total / float(count)
            off_value = off_total / float(count)
        on_values.append(on_value)
        off_values.append(off_value)
        delta_values.append(off_value - on_value)
    return {"kl_on": on_values, "kl_off": off_values, "delta_kl": delta_values}


def _accumulate_metric_section(state: dict[str, float | int], metrics: dict[str, Any]) -> None:
    num_tokens = int(metrics["num_tokens"])
    state["num_tokens"] = int(state.get("num_tokens", 0)) + num_tokens
    state["kl_on_sum"] = float(state.get("kl_on_sum", 0.0)) + float(metrics["kl_on"]) * num_tokens
    state["kl_off_sum"] = float(state.get("kl_off_sum", 0.0)) + float(metrics["kl_off"]) * num_tokens


def _finalize_metric_section(state: dict[str, float | int]) -> dict[str, float | int]:
    num_tokens = int(state.get("num_tokens", 0))
    if num_tokens <= 0:
        raise ValueError("benchmark found no valid tokens.")
    kl_on = float(state["kl_on_sum"]) / float(num_tokens)
    kl_off = float(state["kl_off_sum"]) / float(num_tokens)
    return {
        "kl_on": kl_on,
        "kl_off": kl_off,
        "delta_kl": kl_off - kl_on,
        "num_tokens": num_tokens,
    }


def _checkpoint_output_metadata(state: StudentCheckpointState | None, checkpoint_path: Path | None) -> dict[str, Any]:
    metadata = state.metadata if state is not None else {}
    return {
        "student_checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_loaded": state is not None,
        "checkpoint_step": None if state is None else state.step,
        "checkpoint_optimizer_step": None if state is None else state.optimizer_step,
        "checkpoint_project_stage": metadata.get("project_stage"),
        "checkpoint_student_type": metadata.get("student_type"),
        "checkpoint_student_vocab_size": metadata.get("student_vocab_size"),
        "checkpoint_student_hidden_size": metadata.get("student_hidden_size"),
    }


def _run_mode(
    args: argparse.Namespace,
    mode: str,
) -> tuple[dict[str, Any], dict[str, list[float]], StudentCheckpointState | None]:
    config = _runtime_config(args, mode=mode)
    teacher, student, dataloader, device = _build_components(config, mode=mode)
    checkpoint_state = None
    if args.student_checkpoint is not None:
        checkpoint_state = load_student_from_checkpoint(student, args.student_checkpoint, strict=True, map_location=device)
    full_vocab_state: dict[str, float | int] = {}
    topk_state: dict[str, float | int] = {}
    num_batches = 0
    position_state: dict[str, list[float] | list[int]] = {}

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= args.max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            validate_token_id_ranges(
                input_ids,
                labels,
                vocab_size=int(getattr(student, "vocab_size", getattr(teacher, "vocab_size"))),
                ignored_label_id=config.vocab.ignored_label_id,
            )

            teacher_logits = _call_teacher(teacher, input_ids, attention_mask)
            output = _call_student(student, input_ids, attention_mask)
            on_logits = output.on_logits
            off_logits = on_logits if mode == "placeholder" else output.off_logits
            teacher_logits = teacher_logits.to(on_logits.device)
            labels = labels.to(on_logits.device)
            mask = _select_shared_valid_mask(
                labels=labels,
                ignore_index=config.vocab.ignored_label_id,
                positions_per_sequence=config.mock.positions_per_sequence,
            )
            if args.dual_report:
                batch_metrics = compute_dual_perturbation_metrics(
                    teacher_logits=teacher_logits,
                    on_logits=on_logits,
                    off_logits=off_logits,
                    labels=labels,
                    mask=mask,
                    tau=config.loss.tau,
                    top_k=config.topk.top_k,
                    include_labels=config.topk.include_labels,
                    renormalize_topk=config.topk.renormalize_topk,
                    position_wise=args.position_wise,
                )
                _accumulate_metric_section(full_vocab_state, batch_metrics["full_vocab"])
                _accumulate_metric_section(topk_state, batch_metrics["topk"])
                if args.position_wise:
                    position_state = _accumulate_position_metrics(position_state, batch_metrics["full_vocab"])
            else:
                topk_indices = None
                if config.topk.enabled:
                    topk_indices = build_topk_indices(
                        teacher_logits.detach().float(),
                        labels=labels,
                        top_k=config.topk.top_k,
                        include_labels=config.topk.include_labels,
                    )
                metrics = compute_perturbation_metrics(
                    teacher_logits=teacher_logits,
                    on_logits=on_logits,
                    off_logits=off_logits,
                    mask=mask,
                    tau=config.loss.tau,
                    topk_indices=topk_indices,
                    renormalize_topk=config.topk.renormalize_topk,
                    position_wise=args.position_wise,
                )
                _accumulate_metric_section(full_vocab_state, metrics)
                if args.position_wise:
                    position_state = _accumulate_position_metrics(position_state, metrics)
            num_batches += 1

    full_vocab = _finalize_metric_section(full_vocab_state)
    topk: dict[str, Any] | None = None
    if args.dual_report:
        topk = _finalize_metric_section(topk_state)
        topk.update(
            {
                "top_k": config.topk.top_k,
                "include_labels": config.topk.include_labels,
                "renormalize_topk": config.topk.renormalize_topk,
            }
        )
    summary = {
        "kl_on": full_vocab["kl_on"],
        "kl_off": full_vocab["kl_off"],
        "delta_kl": full_vocab["delta_kl"],
        "num_tokens": full_vocab["num_tokens"],
        "num_batches": num_batches,
        "topk_enabled": config.topk.enabled,
        "top_k": config.topk.top_k if config.topk.enabled else None,
        "checkpoint_loaded": checkpoint_state is not None,
        "checkpoint_step": None if checkpoint_state is None else checkpoint_state.step,
        "checkpoint_optimizer_step": None if checkpoint_state is None else checkpoint_state.optimizer_step,
        "full_vocab": full_vocab,
    }
    if topk is not None:
        summary["topk"] = topk
    return summary, _finalize_position_metrics(position_state), checkpoint_state


def _write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> None:
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
            writer.writeheader()
            metadata = payload.get("metadata", {})
            for mode, summary in payload["by_mode"].items():
                full_vocab = summary.get("full_vocab") or {}
                topk = summary.get("topk") or {}
                writer.writerow(
                    {
                        "mode": mode,
                        "kl_on": summary["kl_on"],
                        "kl_off": summary["kl_off"],
                        "delta_kl": summary["delta_kl"],
                        "num_tokens": summary["num_tokens"],
                        "num_batches": summary["num_batches"],
                        "topk_enabled": summary["topk_enabled"],
                        "top_k": "" if summary["top_k"] is None else summary["top_k"],
                        "full_vocab.kl_on": full_vocab.get("kl_on", ""),
                        "full_vocab.kl_off": full_vocab.get("kl_off", ""),
                        "full_vocab.delta_kl": full_vocab.get("delta_kl", ""),
                        "full_vocab.num_tokens": full_vocab.get("num_tokens", ""),
                        "topk.kl_on": topk.get("kl_on", ""),
                        "topk.kl_off": topk.get("kl_off", ""),
                        "topk.delta_kl": topk.get("delta_kl", ""),
                        "topk.num_tokens": topk.get("num_tokens", ""),
                        "topk.top_k": topk.get("top_k", ""),
                        "checkpoint_loaded": metadata.get("checkpoint_loaded", ""),
                        "student_checkpoint": metadata.get("student_checkpoint", ""),
                    }
                )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_batches <= 0:
        raise ValueError("--max-batches must be positive.")
    modes = _parse_modes(args)
    by_mode: dict[str, dict[str, Any]] = {}
    position_wise: dict[str, Any] = {}
    first_checkpoint_state = None
    for mode in modes:
        summary, positions, checkpoint_state = _run_mode(args, mode)
        by_mode[mode] = summary
        if first_checkpoint_state is None:
            first_checkpoint_state = checkpoint_state
        if args.position_wise and not position_wise:
            position_wise = positions
    first_mode = modes[0]
    config = _runtime_config(args, mode=first_mode)
    first_summary = by_mode[first_mode]
    return {
        "summary": first_summary,
        "full_vocab": first_summary["full_vocab"],
        "topk": first_summary.get("topk", {}),
        "by_mode": by_mode,
        "position_wise": position_wise,
        "metadata": {
            "teacher_type": config.teacher_type,
            "student_type": config.student_type,
            "dataset_type": config.data.dataset_type,
            "seq_len": config.mock.seq_len if config.data.dataset_type == "mock" else config.data.seq_len,
            "topk_enabled": config.topk.enabled,
            "top_k": config.topk.top_k if args.dual_report or config.topk.enabled else None,
            "dual_report": args.dual_report,
            "tau": config.loss.tau,
            "modes": modes,
            **_checkpoint_output_metadata(first_checkpoint_state, args.student_checkpoint),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_benchmark(args)
    _write_outputs(payload, args)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
