"""Read-only health checks for cached logits and checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ArtifactFileReport:
    path: str
    kind: str
    status: str
    size_bytes: int | None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class ArtifactHealthReport:
    root: str
    checked_count: int
    ok_count: int
    corrupt_count: int
    missing_count: int
    reports: list[ArtifactFileReport]

    @property
    def ok(self) -> bool:
        return self.corrupt_count == 0 and self.missing_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "ok": self.ok,
            "checked_count": self.checked_count,
            "ok_count": self.ok_count,
            "corrupt_count": self.corrupt_count,
            "missing_count": self.missing_count,
            "reports": [asdict(report) for report in self.reports],
        }


def discover_artifact_files(
    root: str | Path,
    *,
    include_cache: bool = True,
    include_checkpoints: bool = True,
    cache_sample_size: int | None = None,
) -> list[tuple[Path, str]]:
    """Find torch artifacts under conventional run cache/checkpoint folders."""

    root_path = Path(root)
    if not root_path.exists():
        return []
    discovered: list[tuple[Path, str]] = []
    if include_cache:
        cache_paths: list[Path] = []
        for path in sorted(root_path.glob("**/cache/**/*.pt")):
            if path.is_file():
                cache_paths.append(path)
        for path in sorted(root_path.glob("**/teacher_logits/*.pt")):
            if path.is_file():
                if path not in cache_paths:
                    cache_paths.append(path)
        discovered.extend((path, "teacher_cache") for path in _sample_paths(cache_paths, cache_sample_size))
    if include_checkpoints:
        for path in sorted(root_path.glob("**/checkpoints/*.pt")):
            if path.is_file():
                discovered.append((path, "checkpoint"))
    return discovered


def _sample_paths(paths: list[Path], sample_size: int | None) -> list[Path]:
    if sample_size is None:
        return paths
    if sample_size < 0:
        raise ValueError("cache_sample_size must be non-negative.")
    if sample_size == 0 or not paths:
        return []
    if sample_size >= len(paths):
        return paths
    if sample_size == 1:
        return [paths[0]]
    last_index = len(paths) - 1
    selected_indices = {
        round(index * last_index / (sample_size - 1))
        for index in range(sample_size)
    }
    return [paths[index] for index in sorted(selected_indices)]


def check_torch_artifact(path: str | Path, *, kind: str = "torch_artifact") -> ArtifactFileReport:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return ArtifactFileReport(
            path=str(artifact_path),
            kind=kind,
            status="missing",
            size_bytes=None,
            error_type="FileNotFoundError",
            error_message="artifact file does not exist",
        )
    size_bytes = artifact_path.stat().st_size
    try:
        # These are local experiment artifacts. Loading is the most direct
        # integrity check for the torch inline-container failures seen in long
        # runs, and map_location keeps CUDA tensors off the GPU during checks.
        torch.load(artifact_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - convert unknown torch load errors into a report.
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        return ArtifactFileReport(
            path=str(artifact_path),
            kind=kind,
            status="corrupt",
            size_bytes=size_bytes,
            error_type=type(exc).__name__,
            error_message=first_line,
        )
    return ArtifactFileReport(
        path=str(artifact_path),
        kind=kind,
        status="ok",
        size_bytes=size_bytes,
    )


def check_artifacts(
    root: str | Path,
    *,
    include_cache: bool = True,
    include_checkpoints: bool = True,
    cache_sample_size: int | None = None,
    max_files: int | None = None,
) -> ArtifactHealthReport:
    root_path = Path(root)
    artifacts = discover_artifact_files(
        root_path,
        include_cache=include_cache,
        include_checkpoints=include_checkpoints,
        cache_sample_size=cache_sample_size,
    )
    if max_files is not None:
        if max_files < 0:
            raise ValueError("max_files must be non-negative.")
        artifacts = artifacts[:max_files]

    reports = [check_torch_artifact(path, kind=kind) for path, kind in artifacts]
    ok_count = sum(1 for report in reports if report.status == "ok")
    corrupt_count = sum(1 for report in reports if report.status == "corrupt")
    missing_count = sum(1 for report in reports if report.status == "missing")
    return ArtifactHealthReport(
        root=str(root_path),
        checked_count=len(reports),
        ok_count=ok_count,
        corrupt_count=corrupt_count,
        missing_count=missing_count,
        reports=reports,
    )
