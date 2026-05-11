"""Optional Mamba dependency diagnostics.

This module intentionally does not import ``mamba_ssm`` or ``causal_conv1d``
at module import time. Use ``check_mamba_dependencies`` when an explicit
environment diagnostic is requested.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import importlib
from importlib import metadata
import json
import platform
from types import ModuleType
from typing import Sequence


@dataclass(frozen=True)
class MambaDependencyReport:
    python_version: str
    torch_version: str | None
    torch_cuda_version: str | None
    cuda_available: bool
    gpu_count: int
    gpu_names: list[str]
    mamba_ssm_available: bool
    mamba_ssm_version: str | None
    causal_conv1d_available: bool
    causal_conv1d_version: str | None
    import_errors: dict[str, str] = field(default_factory=dict)


def _package_version(module: ModuleType, distribution_names: tuple[str, ...]) -> str | None:
    version = getattr(module, "__version__", None)
    if isinstance(version, str):
        return version
    for name in distribution_names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def check_mamba_dependencies() -> MambaDependencyReport:
    """Inspect Mamba-related optional dependencies without requiring them.

    Optional imports happen only inside this explicit diagnostic call. The
    function does not instantiate Mamba modules and does not allocate large
    tensors.
    """

    import_errors: dict[str, str] = {}
    try:
        torch_module: ModuleType | None = importlib.import_module("torch")
        torch_error: str | None = None
    except Exception as exc:  # pragma: no cover - exact dependency failures vary.
        torch_module = None
        torch_error = f"{type(exc).__name__}: {exc}"
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    cuda_available = False
    gpu_count = 0
    gpu_names: list[str] = []

    if torch_module is None:
        if torch_error is not None:
            import_errors["torch"] = torch_error
    else:
        torch_version = getattr(torch_module, "__version__", None)
        torch_cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
        try:
            cuda_available = bool(torch_module.cuda.is_available())
            gpu_count = int(torch_module.cuda.device_count()) if cuda_available else 0
            gpu_names = [str(torch_module.cuda.get_device_name(index)) for index in range(gpu_count)]
        except Exception as exc:  # pragma: no cover - CUDA driver errors are environment-specific.
            import_errors["torch.cuda"] = f"{type(exc).__name__}: {exc}"
            cuda_available = False
            gpu_count = 0
            gpu_names = []

    try:
        mamba_ssm_module: ModuleType | None = importlib.import_module("mamba_ssm")
        mamba_error: str | None = None
    except Exception as exc:
        mamba_ssm_module = None
        mamba_error = f"{type(exc).__name__}: {exc}"
    mamba_ssm_available = mamba_ssm_module is not None
    mamba_ssm_version = (
        _package_version(mamba_ssm_module, ("mamba-ssm", "mamba_ssm")) if mamba_ssm_module is not None else None
    )
    if mamba_error is not None:
        import_errors["mamba_ssm"] = mamba_error

    try:
        causal_conv1d_module: ModuleType | None = importlib.import_module("causal_conv1d")
        causal_conv1d_error: str | None = None
    except Exception as exc:
        causal_conv1d_module = None
        causal_conv1d_error = f"{type(exc).__name__}: {exc}"
    causal_conv1d_available = causal_conv1d_module is not None
    causal_conv1d_version = (
        _package_version(causal_conv1d_module, ("causal-conv1d", "causal_conv1d"))
        if causal_conv1d_module is not None
        else None
    )
    if causal_conv1d_error is not None:
        import_errors["causal_conv1d"] = causal_conv1d_error

    return MambaDependencyReport(
        python_version=platform.python_version(),
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        cuda_available=cuda_available,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        mamba_ssm_available=mamba_ssm_available,
        mamba_ssm_version=mamba_ssm_version,
        causal_conv1d_available=causal_conv1d_available,
        causal_conv1d_version=causal_conv1d_version,
        import_errors=import_errors,
    )


def format_mamba_dependency_report(report: MambaDependencyReport) -> str:
    """Format a human-readable Mamba dependency report."""

    gpu_summary = ", ".join(report.gpu_names) if report.gpu_names else "none detected"
    lines = [
        "Mamba dependency diagnostic",
        f"Python: {report.python_version}",
        f"PyTorch: {report.torch_version or 'not available'}",
        f"PyTorch CUDA build: {report.torch_cuda_version or 'not available'}",
        f"CUDA available: {report.cuda_available}",
        f"GPU count: {report.gpu_count}",
        f"GPU names: {gpu_summary}",
        f"mamba_ssm available: {report.mamba_ssm_available}",
        f"mamba_ssm version: {report.mamba_ssm_version or 'not available'}",
        f"causal_conv1d available: {report.causal_conv1d_available}",
        f"causal_conv1d version: {report.causal_conv1d_version or 'not available'}",
    ]
    if report.import_errors:
        lines.append("Import errors:")
        for name in sorted(report.import_errors):
            lines.append(f"  - {name}: {report.import_errors[name]}")
    else:
        lines.append("Import errors: none")
    lines.extend(
        [
            "Install hints:",
            "  pip install causal-conv1d>=1.4.0 --no-build-isolation",
            "  pip install mamba-ssm --no-build-isolation",
            "  or: pip install mamba-ssm[causal-conv1d] --no-build-isolation",
            "These optional packages may compile CUDA extensions and are not required for mock tests.",
        ]
    )
    return "\n".join(lines)


def report_to_json(report: MambaDependencyReport) -> str:
    """Serialize a dependency report as stable JSON."""

    return json.dumps(asdict(report), sort_keys=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check optional Mamba runtime dependencies.")
    parser.add_argument(
        "--require-mamba",
        action="store_true",
        help="Exit nonzero if required Mamba optional packages are unavailable.",
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = check_mamba_dependencies()
    if args.json:
        print(report_to_json(report))
    else:
        print(format_mamba_dependency_report(report))
    if args.require_mamba and not (report.mamba_ssm_available and report.causal_conv1d_available):
        return 1
    return 0
