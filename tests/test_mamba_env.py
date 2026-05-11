from __future__ import annotations

import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from utils.mamba_env import (
    MambaDependencyReport,
    check_mamba_dependencies,
    format_mamba_dependency_report,
)
import utils.mamba_env as mamba_env


ROOT = Path(__file__).resolve().parents[1]


def test_check_mamba_dependencies_returns_report() -> None:
    report = check_mamba_dependencies()

    assert isinstance(report, MambaDependencyReport)
    assert report.python_version
    assert isinstance(report.import_errors, dict)
    assert isinstance(report.gpu_names, list)


def test_missing_optional_mamba_deps_do_not_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> types.ModuleType:
        if name in {"mamba_ssm", "causal_conv1d"}:
            raise ImportError(f"missing optional {name}")
        return real_import_module(name, package)

    monkeypatch.setattr("utils.mamba_env.importlib.import_module", fake_import_module)

    report = check_mamba_dependencies()

    assert report.mamba_ssm_available is False
    assert report.causal_conv1d_available is False
    assert "mamba_ssm" in report.import_errors
    assert "causal_conv1d" in report.import_errors


def test_available_optional_mamba_deps_can_be_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module
    fake_mamba = types.ModuleType("mamba_ssm")
    fake_mamba.__version__ = "1.2.3"
    fake_conv = types.ModuleType("causal_conv1d")
    fake_conv.__version__ = "4.5.6"

    def fake_import_module(name: str, package: str | None = None) -> types.ModuleType:
        if name == "mamba_ssm":
            return fake_mamba
        if name == "causal_conv1d":
            return fake_conv
        return real_import_module(name, package)

    monkeypatch.setattr("utils.mamba_env.importlib.import_module", fake_import_module)

    report = check_mamba_dependencies()

    assert report.mamba_ssm_available is True
    assert report.mamba_ssm_version == "1.2.3"
    assert report.causal_conv1d_available is True
    assert report.causal_conv1d_version == "4.5.6"


def test_format_mamba_dependency_report_contains_python_and_torch_info() -> None:
    report = MambaDependencyReport(
        python_version="3.11.0",
        torch_version="2.4.0",
        torch_cuda_version=None,
        cuda_available=False,
        gpu_count=0,
        gpu_names=[],
        mamba_ssm_available=False,
        mamba_ssm_version=None,
        causal_conv1d_available=False,
        causal_conv1d_version=None,
        import_errors={"mamba_ssm": "ImportError: missing"},
    )

    text = format_mamba_dependency_report(report)

    assert "Python: 3.11.0" in text
    assert "PyTorch: 2.4.0" in text
    assert "mamba_ssm available: False" in text
    assert "Install hints:" in text


def test_check_mamba_env_script_exits_zero_without_required_mamba() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_mamba_env.py"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "Mamba dependency diagnostic" in result.stdout


def test_check_mamba_env_script_json_is_parseable() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_mamba_env.py", "--json"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "python_version" in payload
    assert "mamba_ssm_available" in payload
    assert "causal_conv1d_available" in payload


def test_require_mamba_fails_when_mamba_ssm_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    report = MambaDependencyReport(
        python_version="3.11.0",
        torch_version="2.4.0",
        torch_cuda_version=None,
        cuda_available=False,
        gpu_count=0,
        gpu_names=[],
        mamba_ssm_available=False,
        mamba_ssm_version=None,
        causal_conv1d_available=True,
        causal_conv1d_version="1.4.0",
        import_errors={"mamba_ssm": "ImportError: missing"},
    )
    monkeypatch.setattr(mamba_env, "check_mamba_dependencies", lambda: report)

    assert mamba_env.main(["--require-mamba", "--json"]) == 1


def test_require_mamba_fails_when_causal_conv1d_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    report = MambaDependencyReport(
        python_version="3.11.0",
        torch_version="2.4.0",
        torch_cuda_version=None,
        cuda_available=False,
        gpu_count=0,
        gpu_names=[],
        mamba_ssm_available=True,
        mamba_ssm_version="2.2.0",
        causal_conv1d_available=False,
        causal_conv1d_version=None,
        import_errors={"causal_conv1d": "ImportError: missing"},
    )
    monkeypatch.setattr(mamba_env, "check_mamba_dependencies", lambda: report)

    assert mamba_env.main(["--require-mamba", "--json"]) == 1


def test_require_mamba_succeeds_when_required_optional_deps_present(monkeypatch: pytest.MonkeyPatch) -> None:
    report = MambaDependencyReport(
        python_version="3.11.0",
        torch_version="2.4.0",
        torch_cuda_version=None,
        cuda_available=False,
        gpu_count=0,
        gpu_names=[],
        mamba_ssm_available=True,
        mamba_ssm_version="2.2.0",
        causal_conv1d_available=True,
        causal_conv1d_version="1.4.0",
        import_errors={},
    )
    monkeypatch.setattr(mamba_env, "check_mamba_dependencies", lambda: report)

    assert mamba_env.main(["--require-mamba", "--json"]) == 0


def test_mamba_env_module_import_does_not_import_optional_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real_import_module = importlib.import_module
    sys.modules.pop("utils.mamba_env", None)

    def tracked_import_module(name: str, package: str | None = None) -> types.ModuleType:
        if name in {"mamba_ssm", "causal_conv1d"}:
            calls.append(name)
            raise AssertionError(f"{name} should not be imported at module import time")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", tracked_import_module)

    module = importlib.import_module("utils.mamba_env")

    assert module.MambaDependencyReport
    assert calls == []


def test_mamba_env_module_import_guard_catches_direct_optional_imports() -> None:
    code = """
import builtins
real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'mamba_ssm' or name.startswith('mamba_ssm.'):
        raise AssertionError('mamba_ssm imported at module load')
    if name == 'causal_conv1d' or name.startswith('causal_conv1d.'):
        raise AssertionError('causal_conv1d imported at module load')
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded_import
import utils.mamba_env
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
