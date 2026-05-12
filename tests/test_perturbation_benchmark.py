from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from evals.perturbation_robustness import (
    compute_perturbation_metrics,
    kl_teacher_student,
)
from losses.kd_loss import build_topk_indices


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


def test_perturbation_benchmark_cli_mock_json() -> None:
    payload = _run_benchmark("--config", "configs/train_config.yaml", "--mock", "--max-batches", "2")

    assert set(payload) == {"summary", "by_mode", "position_wise", "metadata"}
    assert "delta_projection" in payload["by_mode"]
    summary = payload["summary"]
    for key in ("kl_on", "kl_off", "delta_kl"):
        assert math.isfinite(float(summary[key]))
    assert int(summary["num_tokens"]) > 0


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
    assert len(payload["position_wise"]["delta_kl"]) > 0
    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["metadata"]["modes"] == ["delta_projection", "noise", "identity"]
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["mode"] for row in rows] == ["delta_projection", "noise", "identity"]


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
