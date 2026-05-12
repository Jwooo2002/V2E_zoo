from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evals.perturbation_robustness import (
    compute_dual_perturbation_metrics,
    compute_perturbation_metrics,
    kl_teacher_student,
)
from losses.kd_loss import build_topk_indices
from models.student_mamba import MockStudentMamba
from train import load_train_config
from utils.checkpointing import save_training_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def test_kl_teacher_student_is_zero_for_identical_logits() -> None:
    logits = torch.randn(2, 3, 7)

    kl = kl_teacher_student(logits, logits, tau=1.0)

    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_kl_teacher_student_is_nonnegative_for_simple_inputs() -> None:
    teacher = torch.tensor([[[3.0, 0.0], [0.0, 3.0]]])
    student = torch.tensor([[[0.0, 3.0], [0.0, 3.0]]])

    kl = kl_teacher_student(teacher, student, tau=1.0)

    assert float(kl) >= 0.0
    assert torch.isfinite(kl)


def test_kl_teacher_student_mask_and_reduction_none() -> None:
    teacher = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    student = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]])
    mask = torch.tensor([[True, False]])

    per_position = kl_teacher_student(teacher, student, mask=mask, reduction="none")
    masked_mean = kl_teacher_student(teacher, student, mask=mask)
    unmasked_mean = kl_teacher_student(teacher, student)

    assert per_position.shape == (1, 2)
    assert torch.allclose(per_position[0, 1], torch.tensor(0.0))
    assert torch.allclose(masked_mean, torch.tensor(0.0), atol=1e-6)
    assert float(unmasked_mean) > float(masked_mean)


def test_topk_kl_path_returns_finite_scalar() -> None:
    teacher = torch.randn(2, 3, 9)
    student = torch.randn(2, 3, 9)
    indices = build_topk_indices(teacher, top_k=4, include_labels=False)

    kl = kl_teacher_student(teacher, student, topk_indices=indices, renormalize_topk=True)

    assert kl.shape == ()
    assert torch.isfinite(kl)


def test_compute_perturbation_metrics_and_position_wise_output() -> None:
    teacher = torch.randn(2, 4, 8)
    on = teacher + 0.01 * torch.randn(2, 4, 8)
    off = teacher + 0.05 * torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    indices = build_topk_indices(teacher, top_k=3, include_labels=False)

    metrics = compute_perturbation_metrics(
        teacher,
        on,
        off,
        mask=mask,
        topk_indices=indices,
        position_wise=True,
    )

    assert {"kl_on", "kl_off", "delta_kl", "num_tokens"}.issubset(metrics)
    assert int(metrics["num_tokens"]) == int(mask.sum())
    assert len(metrics["position_delta_kl"]) == teacher.shape[1]
    assert len(metrics["position_kl_on"]) == teacher.shape[1]
    assert len(metrics["position_kl_off"]) == teacher.shape[1]
    assert math.isfinite(float(metrics["delta_kl"]))


def test_dual_perturbation_metrics_report_full_vocab_and_topk() -> None:
    teacher = torch.tensor([[[8.0, 3.0, -2.0, -3.0], [0.0, 8.0, 3.0, -3.0]]])
    on = torch.tensor([[[8.0, 3.0, -2.0, -3.0], [0.0, 8.0, 3.0, -3.0]]])
    off = torch.tensor([[[8.0, -3.0, 3.0, -2.0], [0.0, 8.0, -3.0, 3.0]]])
    labels = torch.tensor([[2, -100]])
    mask = torch.tensor([[True, False]])

    dual = compute_dual_perturbation_metrics(
        teacher,
        on,
        off,
        labels=labels,
        mask=mask,
        top_k=1,
        include_labels=True,
    )
    legacy = compute_perturbation_metrics(teacher, on, off, mask=mask)

    assert set(dual) == {"full_vocab", "topk"}
    assert dual["full_vocab"]["num_tokens"] == 1
    assert dual["topk"]["num_tokens"] == 1
    assert dual["topk"]["top_k"] == 1
    assert dual["topk"]["include_labels"] is True
    assert math.isclose(float(dual["full_vocab"]["delta_kl"]), float(legacy["delta_kl"]), rel_tol=1e-6)
    assert math.isfinite(float(dual["topk"]["delta_kl"]))
    assert not math.isclose(float(dual["full_vocab"]["delta_kl"]), float(dual["topk"]["delta_kl"]), abs_tol=1e-8)


def test_dual_perturbation_topk_handles_ignored_labels() -> None:
    teacher = torch.randn(2, 3, 7)
    on = teacher + 0.01 * torch.randn(2, 3, 7)
    off = teacher + 0.03 * torch.randn(2, 3, 7)
    labels = torch.tensor([[1, -100, 3], [2, 4, -100]])
    mask = labels.ne(-100)

    metrics = compute_dual_perturbation_metrics(
        teacher,
        on,
        off,
        labels=labels,
        mask=mask,
        top_k=3,
        include_labels=True,
    )

    assert metrics["full_vocab"]["num_tokens"] == int(mask.sum())
    assert metrics["topk"]["num_tokens"] == int(mask.sum())
    assert math.isfinite(float(metrics["topk"]["kl_on"]))


def _run_benchmark(*args: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "scripts/run_perturbation_benchmark.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )
    return json.loads(result.stdout)


def _mock_student_checkpoint(tmp_path: Path) -> Path:
    config = load_train_config(ROOT / "configs" / "train_config.yaml")
    student = MockStudentMamba(config.mock.vocab_size, config.mock.hidden_size)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.fill_(0.02)
    return save_training_checkpoint(
        tmp_path,
        student,
        optimizer=None,
        step=9,
        optimizer_step=4,
        config={"student_type": "mock"},
        metadata={
            "project_stage": "9B-test",
            "student_type": "mock",
            "student_vocab_size": config.mock.vocab_size,
            "student_hidden_size": config.mock.hidden_size,
        },
        rng_state=False,
    )


def test_perturbation_benchmark_cli_mock_json() -> None:
    payload = _run_benchmark(
        "--config",
        "configs/train_config.yaml",
        "--mock",
        "--max-batches",
        "2",
        "--top-k",
        "8",
    )

    assert set(payload) == {"summary", "full_vocab", "topk", "by_mode", "position_wise", "metadata"}
    assert "delta_projection" in payload["by_mode"]
    summary = payload["summary"]
    for key in ("kl_on", "kl_off", "delta_kl"):
        assert math.isfinite(float(summary[key]))
    assert int(summary["num_tokens"]) > 0
    assert payload["full_vocab"]["delta_kl"] == summary["delta_kl"]
    assert math.isfinite(float(payload["topk"]["delta_kl"]))
    assert payload["topk"]["top_k"] == 8
    assert "full_vocab" in payload["by_mode"]["delta_projection"]
    assert "topk" in payload["by_mode"]["delta_projection"]


def test_perturbation_benchmark_loads_student_checkpoint_and_reports_metadata(tmp_path: Path) -> None:
    checkpoint_path = _mock_student_checkpoint(tmp_path)

    payload = _run_benchmark(
        "--config",
        "configs/train_config.yaml",
        "--mock",
        "--max-batches",
        "1",
        "--student-checkpoint",
        str(checkpoint_path),
    )

    metadata = payload["metadata"]
    assert metadata["student_checkpoint"] == str(checkpoint_path)
    assert metadata["checkpoint_loaded"] is True
    assert metadata["checkpoint_step"] == 9
    assert metadata["checkpoint_optimizer_step"] == 4
    assert metadata["checkpoint_project_stage"] == "9B-test"
    assert metadata["checkpoint_student_type"] == "mock"
    assert payload["summary"]["checkpoint_loaded"] is True
    assert payload["summary"]["checkpoint_step"] == 9
    assert "full_vocab" in payload
    assert "topk" in payload


def test_perturbation_benchmark_loads_mock_student_checkpoint(tmp_path: Path) -> None:
    checkpoint = _mock_student_checkpoint(tmp_path)
    payload = _run_benchmark(
        "--config",
        "configs/train_config.yaml",
        "--mock",
        "--max-batches",
        "2",
        "--student-checkpoint",
        str(checkpoint),
    )

    assert payload["metadata"]["checkpoint_loaded"] is True
    assert payload["metadata"]["student_checkpoint"] == str(checkpoint)
    assert payload["summary"]["checkpoint_loaded"] is True
    assert math.isfinite(float(payload["summary"]["delta_kl"]))


def test_perturbation_benchmark_writes_json_csv_and_sweeps(tmp_path: Path) -> None:
    output_json = tmp_path / "perturb.json"
    output_csv = tmp_path / "perturb.csv"

    payload = _run_benchmark(
        "--config",
        "configs/train_config.yaml",
        "--mock",
        "--max-batches",
        "2",
        "--sweep",
        "delta_projection,noise,identity",
        "--position-wise",
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
    )

    assert output_json.is_file()
    assert output_csv.is_file()
    assert set(payload["by_mode"]) == {"delta_projection", "noise", "identity"}
    for mode_summary in payload["by_mode"].values():
        assert "full_vocab" in mode_summary
        assert "topk" in mode_summary
    assert len(payload["position_wise"]["delta_kl"]) > 0
    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["metadata"]["modes"] == ["delta_projection", "noise", "identity"]
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["mode"] for row in rows] == ["delta_projection", "noise", "identity"]
    assert "full_vocab.kl_on" in rows[0]
    assert "full_vocab.delta_kl" in rows[0]
    assert "topk.kl_on" in rows[0]
    assert "topk.delta_kl" in rows[0]
    assert "topk.top_k" in rows[0]
    assert "checkpoint_loaded" in rows[0]


def test_perturbation_benchmark_mock_does_not_import_real_optional_modules() -> None:
    code = """
import builtins
import importlib

forbidden = ('transformers', 'mamba_ssm')
real_import = builtins.__import__
real_import_module = importlib.import_module

def blocked(name):
    return any(name == item or name.startswith(item + '.') for item in forbidden)

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if blocked(name):
        raise AssertionError(f'{name} imported by mock perturbation benchmark')
    return real_import(name, globals, locals, fromlist, level)

def guarded_import_module(name, package=None):
    if blocked(name):
        raise AssertionError(f'{name} imported through importlib by mock perturbation benchmark')
    return real_import_module(name, package)

builtins.__import__ = guarded_import
importlib.import_module = guarded_import_module
module = importlib.import_module('scripts.run_perturbation_benchmark')
module.main(['--config', 'configs/train_config.yaml', '--mock', '--max-batches', '1'])
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr


def test_perturbation_benchmark_never_passes_student_states_to_teacher(monkeypatch) -> None:
    import scripts.run_perturbation_benchmark as benchmark

    class TokenOnlyTeacher(torch.nn.Module):
        vocab_size = 11

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, input_ids: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            assert args == ()
            assert set(kwargs) <= {"attention_mask"}
            assert "h_t" not in kwargs
            assert "h_off" not in kwargs
            assert input_ids.ndim == 2
            self.calls += 1
            return torch.nn.functional.one_hot(input_ids, num_classes=self.vocab_size).float()

    teacher = TokenOnlyTeacher()
    student = MockStudentMamba(vocab_size=teacher.vocab_size, hidden_size=4)
    dataloader = DataLoader(
        [
            {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "labels": torch.tensor([2, 3, 4], dtype=torch.long),
            }
        ],
        batch_size=1,
    )

    def fake_build_components(config, *, mode: str):
        return teacher, student, dataloader, torch.device("cpu")

    monkeypatch.setattr(benchmark, "_build_components", fake_build_components)
    args = benchmark.parse_args(
        ["--config", "configs/train_config.yaml", "--mock", "--max-batches", "1", "--top-k", "4"]
    )

    summary, _positions, _checkpoint_state = benchmark._run_mode(args, "delta_projection")

    assert teacher.calls == 1
    assert int(summary["num_tokens"]) == 3
    assert "full_vocab" in summary
    assert "topk" in summary
