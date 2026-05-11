from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import MockTextDataset
from evals.needle import evaluate_needle_scaffold
from evals.perplexity import evaluate_perplexity
from evals.perturbation_robustness import evaluate_perturbation_robustness
from models.student_mamba import MockStudentMamba
from models.teacher_wrapper import MockTeacherWrapper
from train import load_train_config, set_seed


ROOT = Path(__file__).resolve().parents[1]


def _run_eval_cli(mode: str) -> dict[str, object]:
    result = subprocess.run(
        [
            sys.executable,
            "evaluate.py",
            "--config",
            "configs/train_config.yaml",
            "--mock",
            "--mode",
            mode,
            "--max_batches",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    return json.loads(result.stdout)


def test_eval_all_cli_smoke() -> None:
    metrics = _run_eval_cli("all")

    assert set(metrics) == {"perplexity", "perturbation", "needle"}
    assert set(metrics["perplexity"]) == {"loss", "perplexity", "num_tokens"}
    assert set(metrics["perturbation"]) == {"kl_on", "kl_off", "delta_kl", "num_tokens"}
    assert set(metrics["needle"]) == {"accuracy", "num_examples", "seq_len", "needle_position", "mode"}


def test_eval_perplexity_cli_smoke() -> None:
    metrics = _run_eval_cli("perplexity")

    assert math.isfinite(float(metrics["loss"]))
    assert math.isfinite(float(metrics["perplexity"]))
    assert int(metrics["num_tokens"]) > 0


def test_eval_perturbation_cli_smoke() -> None:
    metrics = _run_eval_cli("perturbation")

    assert math.isfinite(float(metrics["kl_on"]))
    assert math.isfinite(float(metrics["kl_off"]))
    assert math.isfinite(float(metrics["delta_kl"]))
    assert int(metrics["num_tokens"]) > 0


def test_eval_needle_cli_smoke() -> None:
    metrics = _run_eval_cli("needle")

    assert set(metrics) == {"accuracy", "num_examples", "seq_len", "needle_position", "mode"}
    assert math.isfinite(float(metrics["accuracy"]))
    assert int(metrics["num_examples"]) > 0
    assert metrics["mode"] == "synthetic_mock"


def test_eval_module_imports_do_not_require_real_llama_or_mamba_modules() -> None:
    code = """
import builtins
import importlib

forbidden = ('transformers', 'mamba_ssm')
real_import = builtins.__import__
real_import_module = importlib.import_module

def is_forbidden(name):
    return any(name == item or name.startswith(item + '.') for item in forbidden)

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if is_forbidden(name):
        raise AssertionError(f'{name} imported by mock eval module import')
    return real_import(name, globals, locals, fromlist, level)

def guarded_import_module(name, package=None):
    if is_forbidden(name):
        raise AssertionError(f'{name} imported through importlib by mock eval module import')
    return real_import_module(name, package)

builtins.__import__ = guarded_import
importlib.import_module = guarded_import_module
importlib.import_module('evals.perplexity')
importlib.import_module('evals.perturbation_robustness')
importlib.import_module('evals.needle')
importlib.import_module('evaluate')
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_direct_eval_has_no_gradients_and_finite_metrics() -> None:
    config = load_train_config(ROOT / "configs" / "train_config.yaml")
    set_seed(config.seed)
    device = torch.device("cpu")
    dataset = MockTextDataset(
        vocab_size=config.mock.vocab_size,
        seq_len=config.mock.seq_len,
        num_samples=8,
        seed=config.seed,
        ignore_index=config.mock.ignore_index,
    )
    dataloader = DataLoader(dataset, batch_size=config.mock.batch_size, shuffle=False)
    teacher = MockTeacherWrapper(config.mock.vocab_size, config.mock.hidden_size).to(device)
    student = MockStudentMamba(config.mock.vocab_size, config.mock.hidden_size).to(device)

    perplexity = evaluate_perplexity(student, dataloader, config, device, max_batches=2)
    perturbation = evaluate_perturbation_robustness(
        student,
        teacher,
        dataloader,
        config,
        device,
        max_batches=2,
    )
    needle = evaluate_needle_scaffold(config, max_batches=2)

    assert math.isfinite(float(perplexity["loss"]))
    assert math.isfinite(float(perplexity["perplexity"]))
    assert int(perplexity["num_tokens"]) > 0
    assert math.isfinite(float(perturbation["kl_on"]))
    assert math.isfinite(float(perturbation["kl_off"]))
    assert math.isfinite(float(perturbation["delta_kl"]))
    assert int(perturbation["num_tokens"]) > 0
    assert math.isfinite(float(needle["accuracy"]))
    assert all(parameter.grad is None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
