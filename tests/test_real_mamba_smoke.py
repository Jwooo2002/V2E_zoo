from __future__ import annotations

import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from models.student_mamba import MambaStudentConfig, RealMambaStudent


ROOT = Path(__file__).resolve().parents[1]


def test_student_mamba_import_still_does_not_require_mamba_ssm() -> None:
    code = """
import builtins
import importlib

real_import = builtins.__import__
real_import_module = importlib.import_module

def blocked(name):
    return name == 'mamba_ssm' or name.startswith('mamba_ssm.')

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if blocked(name):
        raise AssertionError('mamba_ssm imported at module import time')
    return real_import(name, globals, locals, fromlist, level)

def guarded_import_module(name, package=None):
    if blocked(name):
        raise AssertionError('mamba_ssm imported through importlib at module import time')
    return real_import_module(name, package)

builtins.__import__ = guarded_import
importlib.import_module = guarded_import_module
import models.student_mamba
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_real_mamba_student_missing_dependency_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> types.ModuleType:
        if name == "mamba_ssm":
            raise ImportError("missing mamba")
        return real_import_module(name, package)

    monkeypatch.setattr("models.student_mamba.importlib.import_module", fake_import_module)

    with pytest.raises(ImportError, match="mamba-ssm is required for RealMambaStudent"):
        RealMambaStudent(MambaStudentConfig(vocab_size=16, hidden_size=8, num_layers=1, torch_dtype="float32"))


def _mamba_available() -> bool:
    try:
        importlib.import_module("mamba_ssm")
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _mamba_available(), reason="mamba_ssm is not installed")
def test_real_mamba_tiny_forward_shapes_cpu() -> None:
    config = MambaStudentConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        torch_dtype="float32",
        device="cpu",
    )
    student = RealMambaStudent(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 5), dtype=torch.long)
    try:
        with torch.no_grad():
            output = student(input_ids)
    except RuntimeError as exc:
        pytest.skip(f"mamba_ssm CPU forward unsupported in this environment: {exc}")

    assert output.on_logits.shape == (2, 5, config.vocab_size)
    assert output.off_logits.shape == output.on_logits.shape
    assert output.fake_logits.shape == output.on_logits.shape
    assert output.h.shape[:2] == input_ids.shape
    assert output.h_off.shape == output.h.shape
    assert output.h_delta_alt.shape == output.h.shape
    assert output.off_logits.data_ptr() == output.on_logits.data_ptr()
    assert not output.fake_logits.requires_grad


def test_check_mamba_forward_script_supports_cpu() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_mamba_forward.py",
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--seq-len",
            "4",
            "--vocab-size",
            "32",
            "--hidden-size",
            "16",
            "--num-layers",
            "1",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )

    if not _mamba_available():
        assert result.returncode != 0
        payload = json.loads(result.stderr)
        assert payload["success"] is False
        assert payload["error_type"] == "ImportError"
        return
    if result.returncode != 0:
        payload = json.loads(result.stderr)
        pytest.skip(f"mamba_ssm CPU forward unsupported in this environment: {payload}")

    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["device"] == "cpu"
    assert payload["input_shape"] == [1, 4]
    assert payload["on_logits_shape"] == [1, 4, 32]
    assert payload["off_logits_shape"] == [1, 4, 32]
    assert payload["fake_logits_shape"] == [1, 4, 32]


@pytest.mark.skipif(not _mamba_available(), reason="mamba_ssm is not installed")
def test_real_mamba_tiny_forward_preserves_non_multiple_vocab_size() -> None:
    config = MambaStudentConfig(
        vocab_size=33,
        hidden_size=16,
        num_layers=1,
        torch_dtype="float32",
        device="cpu",
    )
    student = RealMambaStudent(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)

    try:
        with torch.no_grad():
            output = student(input_ids)
    except RuntimeError as exc:
        pytest.skip(f"mamba_ssm CPU forward unsupported in this environment: {exc}")

    assert output.on_logits.shape == (1, 4, config.vocab_size)
