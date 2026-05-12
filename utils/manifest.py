"""Run manifest and registry utilities for reproducible smoke experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform as platform_module
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


RUN_SUBDIRS = ("configs", "logs", "checkpoints", "cache", "reports", "evals", "artifacts")


@dataclass(frozen=True)
class GitInfo:
    commit: str | None
    branch: str | None
    is_dirty: bool
    diff_summary: str | None


@dataclass(frozen=True)
class EnvInfo:
    python: str
    executable: str
    platform: str
    torch_version: str | None
    torch_cuda: str | None
    cuda_available: bool | None
    gpu_count: int | None
    gpu_names: list[str] = field(default_factory=list)
    transformers_version: str | None = None
    mamba_ssm_version: str | None = None
    causal_conv1d_version: str | None = None


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at: str
    project: str
    stage: str | None
    command: list[str]
    config_paths: list[str]
    output_dir: str
    git: dict[str, Any]
    env: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def generate_run_id(prefix: str = "run", timestamp: datetime | str | None = None, extra: Any | None = None) -> str:
    """Generate a filesystem-safe run id.

    Passing ``timestamp`` makes the output deterministic for the same prefix
    and ``extra`` value.
    """

    safe_prefix = _safe_name(prefix or "run")
    timestamp_text = _timestamp_text(timestamp)
    digest_source = json.dumps(
        {"prefix": safe_prefix, "timestamp": timestamp_text, "extra": extra},
        sort_keys=True,
        default=str,
    )
    suffix = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:6]
    return f"{safe_prefix}_{timestamp_text}_{suffix}"


def get_git_info(repo_root: str | Path | None = None) -> GitInfo:
    """Capture git commit/branch/status without failing outside a git repo."""

    root = Path(repo_root) if repo_root is not None else Path.cwd()
    commit = _git_output(root, "rev-parse", "HEAD")
    branch = _git_output(root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git_output(root, "status", "--short")
    if commit is None and branch is None and status is None:
        return GitInfo(commit=None, branch=None, is_dirty=False, diff_summary=None)
    diff_summary = status if status else None
    return GitInfo(
        commit=commit,
        branch=branch,
        is_dirty=bool(status),
        diff_summary=diff_summary,
    )


def get_env_info() -> EnvInfo:
    """Capture environment metadata while keeping optional deps optional."""

    torch_version = None
    torch_cuda = None
    cuda_available: bool | None = None
    gpu_count: int | None = None
    gpu_names: list[str] = []
    try:
        import torch

        torch_version = str(torch.__version__)
        torch_cuda = None if torch.version.cuda is None else str(torch.version.cuda)
        cuda_available = bool(torch.cuda.is_available())
        gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
        if cuda_available:
            gpu_names = [str(torch.cuda.get_device_name(index)) for index in range(gpu_count or 0)]
    except Exception:
        cuda_available = None
        gpu_count = None

    return EnvInfo(
        python=sys.version.split()[0],
        executable=sys.executable,
        platform=platform_module.platform(),
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        cuda_available=cuda_available,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        transformers_version=_package_version("transformers"),
        mamba_ssm_version=_package_version("mamba-ssm"),
        causal_conv1d_version=_package_version("causal-conv1d"),
    )


def create_run_dir(base_dir: str | Path, run_id: str) -> Path:
    safe_run_id = _safe_name(run_id)
    if safe_run_id != run_id:
        raise ValueError(f"run_id {run_id!r} is not filesystem-safe.")
    run_dir = Path(base_dir) / safe_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for subdir in RUN_SUBDIRS:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_config_files(config_paths: list[str | Path], configs_dir: str | Path) -> list[Path]:
    destination = Path(configs_dir)
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    seen: dict[str, int] = {}
    for raw_path in config_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        name = path.name
        count = seen.get(name, 0)
        seen[name] = count + 1
        if count:
            target_name = f"{path.stem}_{count}{path.suffix}"
        else:
            target_name = name
        target = destination / target_name
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def write_manifest(manifest: RunManifest | dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(manifest) if hasattr(manifest, "__dataclass_fields__") else dict(manifest)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_manifest(path: str | Path) -> RunManifest | dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest JSON must contain an object.")
    required = {"run_id", "created_at", "project", "command", "config_paths", "output_dir", "git", "env", "metadata"}
    if required.issubset(payload):
        return RunManifest(
            run_id=str(payload["run_id"]),
            created_at=str(payload["created_at"]),
            project=str(payload["project"]),
            stage=None if payload.get("stage") is None else str(payload["stage"]),
            command=[str(item) for item in payload.get("command", [])],
            config_paths=[str(item) for item in payload.get("config_paths", [])],
            output_dir=str(payload["output_dir"]),
            git=dict(payload.get("git", {})),
            env=dict(payload.get("env", {})),
            metadata=dict(payload.get("metadata", {})),
        )
    return payload


def manifest_dict(
    *,
    run_id: str,
    run_dir: str | Path,
    command: list[str],
    config_paths: list[str],
    stage: str | None = None,
    metadata: dict[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        project="cdm-mamba-kd",
        stage=stage,
        command=list(command),
        config_paths=list(config_paths),
        output_dir=str(run_dir),
        git=asdict(get_git_info(repo_root)),
        env=asdict(get_env_info()),
        metadata=dict(metadata or {}),
    )


def _timestamp_text(timestamp: datetime | str | None) -> str:
    if timestamp is None:
        value = datetime.now(timezone.utc)
    elif isinstance(timestamp, datetime):
        value = timestamp
    else:
        return _safe_name(str(timestamp))
    return value.strftime("%Y%m%d_%H%M%S")


def _safe_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "run"


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
