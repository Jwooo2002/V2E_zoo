from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from models.student_mamba import MockStudentMamba, MambaStudentConfig, RealMambaStudent, _mamba_reference_kernel_patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "cdm_mamba_forward_script_tests",
    ROOT / "scripts" / "check_mamba_forward.py",
)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
check_mamba_forward = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = check_mamba_forward
SCRIPT_SPEC.loader.exec_module(check_mamba_forward)


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


def test_mock_student_mamba_state_shapes_remain_available() -> None:
    student = MockStudentMamba(vocab_size=29, hidden_size=11)
    input_ids = torch.randint(0, 29, (2, 6), dtype=torch.long)

    output = student(input_ids)

    assert output.h is not None
    assert output.h_off is not None
    assert output.h_delta_alt is not None
    assert output.h.shape == (2, 6, 11)
    assert output.h_off.shape == output.h.shape
    assert output.h_delta_alt.shape == output.h.shape


@pytest.mark.parametrize("state_extraction", ["last_hidden", "embedding", "none"])
def test_mamba_student_config_accepts_state_extraction_modes(state_extraction: str) -> None:
    config = MambaStudentConfig(state_extraction=state_extraction)

    assert config.state_extraction == state_extraction


def test_mamba_student_config_rejects_invalid_state_extraction() -> None:
    with pytest.raises(ValueError, match="Unsupported state_extraction"):
        MambaStudentConfig(state_extraction="private_cache")


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
    assert output.h is not None
    assert output.h_off is not None
    assert output.h_delta_alt is not None
    assert output.h.shape[:2] == input_ids.shape
    assert output.h_off.shape == output.h.shape
    assert output.h_delta_alt.shape == output.h.shape
    assert output.off_logits.data_ptr() == output.on_logits.data_ptr()
    assert not output.fake_logits.requires_grad


@pytest.mark.skipif(not _mamba_available(), reason="mamba_ssm is not installed")
@pytest.mark.parametrize("state_extraction", ["last_hidden", "embedding"])
def test_real_mamba_tiny_forward_exposes_state_modes_cpu(state_extraction: str) -> None:
    config = MambaStudentConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        torch_dtype="float32",
        device="cpu",
        state_extraction=state_extraction,
    )
    student = RealMambaStudent(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)

    try:
        with torch.no_grad():
            output = student(input_ids)
    except RuntimeError as exc:
        pytest.skip(f"mamba_ssm CPU forward unsupported in this environment: {exc}")

    assert output.h is not None
    assert output.h.shape == (1, 4, config.hidden_size)
    assert output.h_off is output.h
    assert output.h_delta_alt is output.h


@pytest.mark.skipif(not _mamba_available(), reason="mamba_ssm is not installed")
def test_real_mamba_tiny_forward_can_disable_state_exposure_cpu() -> None:
    config = MambaStudentConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        torch_dtype="float32",
        device="cpu",
        state_extraction="none",
    )
    student = RealMambaStudent(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)

    try:
        with torch.no_grad():
            output = student(input_ids)
    except RuntimeError as exc:
        pytest.skip(f"mamba_ssm CPU forward unsupported in this environment: {exc}")

    assert output.on_logits.shape == (1, 4, config.vocab_size)
    assert output.h is None
    assert output.h_off is None
    assert output.h_delta_alt is None


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
    assert payload["h_shape"] == [1, 4, 16]
    assert payload["h_off_shape"] == [1, 4, 16]
    assert payload["h_delta_alt_shape"] == [1, 4, 16]
    assert payload["state_extraction"] == "last_hidden"
    assert payload["expose_states"] is True
    assert payload["smoke_placeholder_off_logits"] is True
    assert payload["reference_forward"] is True
    assert payload["requested_reference_forward"] is False


def test_check_mamba_forward_script_supports_reference_flag_cpu() -> None:
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
            "--use-reference-forward",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )

    if not _mamba_available():
        assert result.returncode != 0
        return
    if result.returncode != 0:
        payload = json.loads(result.stderr)
        pytest.skip(f"mamba_ssm CPU reference forward unsupported in this environment: {payload}")

    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["reference_forward"] is True
    assert payload["requested_reference_forward"] is True


def test_check_mamba_forward_script_reports_null_state_shapes_for_none_mode() -> None:
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
            "--state-extraction",
            "none",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )

    if not _mamba_available():
        assert result.returncode != 0
        return
    if result.returncode != 0:
        payload = json.loads(result.stderr)
        pytest.skip(f"mamba_ssm CPU forward unsupported in this environment: {payload}")

    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["state_extraction"] == "none"
    assert payload["h_shape"] is None
    assert payload["h_off_shape"] is None
    assert payload["h_delta_alt_shape"] is None
    assert payload["smoke_placeholder_off_logits"] is True


def test_check_mamba_forward_script_reports_null_state_shapes_when_exposure_disabled() -> None:
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
            "--state-extraction",
            "last_hidden",
            "--no-expose-states",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )

    if not _mamba_available():
        assert result.returncode != 0
        return
    if result.returncode != 0:
        payload = json.loads(result.stderr)
        pytest.skip(f"mamba_ssm CPU forward unsupported in this environment: {payload}")

    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["state_extraction"] == "last_hidden"
    assert payload["expose_states"] is False
    assert payload["h_shape"] is None
    assert payload["h_off_shape"] is None
    assert payload["h_delta_alt_shape"] is None
    assert payload["smoke_placeholder_off_logits"] is True


def test_check_mamba_forward_error_payload_is_compact_for_tensor_repr() -> None:
    huge_tensor_repr = "causal_conv1d_fwd(): incompatible function arguments\nInvoked with: tensor([" + (
        "1.2345, " * 500
    ) + "])"
    payload = check_mamba_forward._compact_error_payload(
        TypeError(huge_tensor_repr),
        device="cuda",
        stage="causal_conv1d_cuda_fast_path",
    )
    dumped = json.dumps(payload)

    assert payload["success"] is False
    assert payload["error_type"] == "TypeError"
    assert payload["stage"] == "causal_conv1d_cuda_fast_path"
    assert "fused causal_conv1d fast path" in payload["probable_cause"]
    assert "Invoked with" not in payload["error_message"]
    assert "tensor([" not in payload["error_message"]
    assert len(payload["error_message"]) <= 300
    assert len(dumped) < 1200


def test_mamba_reference_kernel_patch_restores_missing_attrs(monkeypatch: pytest.MonkeyPatch) -> None:
    mamba_simple = types.ModuleType("mamba_ssm.modules.mamba_simple")
    selective_scan_interface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")

    def selective_scan_ref() -> None:
        return None

    selective_scan_interface.selective_scan_ref = selective_scan_ref
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> types.ModuleType:
        if name == "mamba_ssm.modules.mamba_simple":
            return mamba_simple
        if name == "mamba_ssm.ops.selective_scan_interface":
            return selective_scan_interface
        return real_import_module(name, package)

    monkeypatch.setattr("models.student_mamba.importlib.import_module", fake_import_module)

    assert not hasattr(mamba_simple, "selective_scan_fn")
    assert not hasattr(mamba_simple, "causal_conv1d_fn")
    with pytest.raises(RuntimeError, match="inside patch"):
        with _mamba_reference_kernel_patch(reference_causal_conv=True, reference_selective_scan=True):
            assert mamba_simple.selective_scan_fn is selective_scan_ref
            assert mamba_simple.causal_conv1d_fn is None
            raise RuntimeError("inside patch")

    assert not hasattr(mamba_simple, "selective_scan_fn")
    assert not hasattr(mamba_simple, "causal_conv1d_fn")


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
