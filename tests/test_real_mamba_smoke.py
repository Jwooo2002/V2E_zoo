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

from models.cdm_engine import OffTrajectoryConfig
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


def _install_fake_public_mamba(monkeypatch: pytest.MonkeyPatch, *, has_lm_head: bool = True) -> None:
    real_import_module = importlib.import_module

    class FakeMambaConfig:
        def __init__(self, **kwargs: object) -> None:
            self.d_model = int(kwargs["d_model"])
            self.vocab_size = int(kwargs["vocab_size"])

    class FakeBackbone(torch.nn.Module):
        def __init__(self, config: FakeMambaConfig) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(config.vocab_size, config.d_model)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embedding(input_ids)

    class FakeMambaLMHeadModel(torch.nn.Module):
        def __init__(
            self,
            config: FakeMambaConfig,
            device: str | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__()
            self.backbone = FakeBackbone(config)
            if has_lm_head:
                self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
            to_kwargs: dict[str, object] = {}
            if device is not None:
                to_kwargs["device"] = torch.device(device)
            if dtype is not None:
                to_kwargs["dtype"] = dtype
            if to_kwargs:
                self.to(**to_kwargs)

    fake_mamba = types.ModuleType("mamba_ssm")
    fake_mamba.__version__ = "fake"
    fake_config = types.ModuleType("mamba_ssm.models.config_mamba")
    fake_config.MambaConfig = FakeMambaConfig
    fake_mixer = types.ModuleType("mamba_ssm.models.mixer_seq_simple")
    fake_mixer.MambaConfig = FakeMambaConfig
    fake_mixer.MambaLMHeadModel = FakeMambaLMHeadModel
    fake_modules = {
        "mamba_ssm": fake_mamba,
        "mamba_ssm.models.config_mamba": fake_config,
        "mamba_ssm.models.mixer_seq_simple": fake_mixer,
    }

    def fake_import_module(name: str, package: str | None = None) -> types.ModuleType:
        if name in fake_modules:
            return fake_modules[name]
        if name.startswith("mamba_ssm."):
            raise ImportError(f"fake mamba module does not provide {name}")
        return real_import_module(name, package)

    monkeypatch.setattr("models.student_mamba.importlib.import_module", fake_import_module)


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


@pytest.mark.parametrize("off_state_mode", ["projection", "placeholder", "none"])
def test_mamba_student_config_accepts_off_state_modes(off_state_mode: str) -> None:
    config = MambaStudentConfig(off_state_mode=off_state_mode)

    assert config.off_state_mode == off_state_mode


@pytest.mark.parametrize("delta_alt_mode", ["delta_projection", "noise", "identity"])
def test_mamba_student_config_accepts_delta_alt_modes(delta_alt_mode: str) -> None:
    config = MambaStudentConfig(delta_alt_mode=delta_alt_mode)

    assert config.delta_alt_mode == delta_alt_mode


@pytest.mark.parametrize("off_logits_mode", ["lm_head", "projection_head", "placeholder"])
def test_mamba_student_config_accepts_off_logits_modes(off_logits_mode: str) -> None:
    config = MambaStudentConfig(off_logits_mode=off_logits_mode)

    assert config.off_logits_mode == off_logits_mode


def test_mamba_student_config_rejects_invalid_state_extraction() -> None:
    with pytest.raises(ValueError, match="Unsupported state_extraction"):
        MambaStudentConfig(state_extraction="private_cache")


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("off_state_mode", {"off_state_mode": "private_state"}),
        ("delta_alt_mode", {"delta_alt_mode": "private_delta"}),
        ("off_logits_mode", {"off_logits_mode": "private_logits"}),
    ],
)
def test_mamba_student_config_rejects_invalid_off_trajectory_modes(
    field_name: str,
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match=f"Unsupported {field_name}"):
        MambaStudentConfig(**kwargs)


def test_check_mamba_forward_parse_stage6e_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_mamba_forward.py",
            "--off-state-mode",
            "placeholder",
            "--delta-alt-mode",
            "identity",
            "--off-logits-mode",
            "projection_head",
            "--no-off-state-detach-direction",
        ],
    )

    args = check_mamba_forward.parse_args()

    assert args.off_state_mode == "placeholder"
    assert args.delta_alt_mode == "identity"
    assert args.off_logits_mode == "projection_head"
    assert args.off_state_detach_direction is False


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
    assert output.metadata is not None
    assert output.metadata["smoke_placeholder_off_logits"] is False
    assert output.metadata["off_logits_source"] == "lm_head"
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
    assert output.h_off is not None
    assert output.h_delta_alt is not None
    assert output.h_off.shape == output.h.shape
    assert output.h_delta_alt.shape == output.h.shape


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


def test_real_mamba_projection_off_state_changes_state_and_logits_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_public_mamba(monkeypatch)
    torch.manual_seed(0)
    config = MambaStudentConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        torch_dtype="float32",
        device="cpu",
        noise_sigma=0.0,
        off_state_mode="projection",
        delta_alt_mode="delta_projection",
        off_logits_mode="lm_head",
    )
    student = RealMambaStudent(
        config,
        off_config=OffTrajectoryConfig(delta_perturb_eps=0.25, noise_sigma=0.0, rho_min=1.0, rho_max=1.0),
    ).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)

    with torch.no_grad():
        output = student(input_ids)

    assert output.h is not None
    assert output.h_off is not None
    assert output.h_delta_alt is not None
    assert output.h_off.shape == output.h.shape
    assert output.h_delta_alt.shape == output.h.shape
    assert not torch.allclose(output.h_delta_alt, output.h)
    assert not torch.allclose(output.h_off, output.h)
    assert output.off_logits.shape == output.on_logits.shape
    assert output.metadata is not None
    assert output.metadata["smoke_placeholder_off_logits"] is False
    assert output.metadata["off_logits_source"] == "lm_head"
    assert output.metadata["off_state_source"] == "delta_perturbation_engine"
    assert output.metadata["delta_alt_source"] == "delta_projection"


def test_real_mamba_projection_head_off_logits_have_grad_and_fake_is_detached_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_public_mamba(monkeypatch)
    torch.manual_seed(0)
    config = MambaStudentConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        torch_dtype="float32",
        device="cpu",
        off_state_mode="projection",
        delta_alt_mode="delta_projection",
        off_logits_mode="projection_head",
        noise_sigma=0.0,
    )
    student = RealMambaStudent(
        config,
        off_config=OffTrajectoryConfig(delta_perturb_eps=0.25, noise_sigma=0.0, rho_min=1.0, rho_max=1.0),
    ).train()
    input_ids = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)

    output = student(input_ids)

    assert output.off_logits.requires_grad
    assert not output.fake_logits.requires_grad
    output.off_logits.sum().backward()
    assert student.off_projection_head.weight.grad is not None
    assert student.off_projection_head.weight.grad.abs().sum() > 0
    assert student.delta_perturb_proj.weight.grad is None


def test_real_mamba_lm_head_mode_falls_back_to_projection_head_when_public_lm_head_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_public_mamba(monkeypatch, has_lm_head=False)
    torch.manual_seed(0)
    config = MambaStudentConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        torch_dtype="float32",
        device="cpu",
        off_state_mode="projection",
        delta_alt_mode="delta_projection",
        off_logits_mode="lm_head",
        noise_sigma=0.0,
    )
    student = RealMambaStudent(
        config,
        off_config=OffTrajectoryConfig(delta_perturb_eps=0.25, noise_sigma=0.0, rho_min=1.0, rho_max=1.0),
    ).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)

    with torch.no_grad():
        output = student(input_ids)

    assert output.off_logits.shape == (1, 4, config.vocab_size)
    assert output.metadata is not None
    assert output.metadata["off_logits_source"] == "projection_head"
    assert output.metadata["smoke_placeholder_off_logits"] is False


def test_real_mamba_attached_off_state_direction_allows_delta_projection_grad_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_public_mamba(monkeypatch)
    torch.manual_seed(0)
    config = MambaStudentConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        torch_dtype="float32",
        device="cpu",
        off_state_mode="projection",
        delta_alt_mode="delta_projection",
        off_logits_mode="projection_head",
        noise_sigma=0.0,
        off_state_detach_direction=False,
    )
    student = RealMambaStudent(
        config,
        off_config=OffTrajectoryConfig(delta_perturb_eps=0.25, noise_sigma=0.0, rho_min=1.0, rho_max=1.0),
    ).train()
    input_ids = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)

    output = student(input_ids)

    output.off_logits.sum().backward()
    assert student.delta_perturb_proj.weight.grad is not None
    assert student.delta_perturb_proj.weight.grad.abs().sum() > 0


def test_real_mamba_placeholder_mode_reports_smoke_placeholder_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_public_mamba(monkeypatch)
    config = MambaStudentConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        torch_dtype="float32",
        device="cpu",
        off_state_mode="placeholder",
        off_logits_mode="placeholder",
    )
    student = RealMambaStudent(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)

    with torch.no_grad():
        output = student(input_ids)

    assert output.off_logits.data_ptr() == output.on_logits.data_ptr()
    assert output.metadata is not None
    assert output.metadata["smoke_placeholder_off_logits"] is True
    assert output.metadata["off_logits_source"] == "placeholder"


def test_real_mamba_off_state_none_exposes_clean_h_but_no_off_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_public_mamba(monkeypatch)
    config = MambaStudentConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        torch_dtype="float32",
        device="cpu",
        off_state_mode="none",
    )
    student = RealMambaStudent(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)

    with torch.no_grad():
        output = student(input_ids)

    assert output.h is not None
    assert output.h_off is None
    assert output.h_delta_alt is None
    assert output.off_logits.data_ptr() == output.on_logits.data_ptr()
    assert output.metadata is not None
    assert output.metadata["off_state_available"] is False
    assert output.metadata["delta_alt_available"] is False
    assert output.metadata["off_logits_source"] == "placeholder_no_off_state"


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
    assert payload["off_state_mode"] == "projection"
    assert payload["delta_alt_mode"] == "delta_projection"
    assert payload["off_logits_mode"] == "lm_head"
    assert payload["off_state_detach_direction"] is True
    assert payload["off_logits_source"] == "lm_head"
    assert payload["off_state_source"] == "delta_perturbation_engine"
    assert payload["delta_alt_source"] == "delta_projection"
    assert payload["off_state_available"] is True
    assert payload["delta_alt_available"] is True
    assert payload["off_logits_placeholder"] is False
    assert payload["smoke_placeholder_off_logits"] is False
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
    assert payload["smoke_placeholder_off_logits"] is False


def test_check_mamba_forward_script_reports_placeholder_off_logits_mode() -> None:
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
            "--off-state-mode",
            "placeholder",
            "--off-logits-mode",
            "placeholder",
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
    assert payload["off_state_mode"] == "placeholder"
    assert payload["off_logits_mode"] == "placeholder"
    assert payload["off_logits_source"] == "placeholder"
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
