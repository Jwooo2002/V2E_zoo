"""Storage preflight helpers for long-running experiment stability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

_GIB = 1024**3


@dataclass(frozen=True)
class StoragePathReport:
    path: str
    checked_path: str
    free_bytes: int
    total_bytes: int
    used_bytes: int
    min_free_bytes: int

    @property
    def free_gib(self) -> float:
        return self.free_bytes / _GIB

    @property
    def min_free_gib(self) -> float:
        return self.min_free_bytes / _GIB

    @property
    def ok(self) -> bool:
        return self.free_bytes >= self.min_free_bytes


def nearest_existing_path(path: str | Path) -> Path:
    """Return ``path`` or its nearest existing parent for disk-usage checks."""

    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate
    for parent in candidate.parents:
        if parent.exists():
            return parent
    raise FileNotFoundError(f"No existing parent found for storage path: {path}")


def storage_path_report(path: str | Path, *, min_free_gb: float) -> StoragePathReport:
    if min_free_gb < 0:
        raise ValueError("min_free_gb must be non-negative.")
    checked_path = nearest_existing_path(path)
    usage = shutil.disk_usage(checked_path)
    return StoragePathReport(
        path=str(path),
        checked_path=str(checked_path),
        free_bytes=int(usage.free),
        total_bytes=int(usage.total),
        used_bytes=int(usage.used),
        min_free_bytes=int(min_free_gb * _GIB),
    )


def validate_storage_paths(paths: list[str | Path], *, min_free_gb: float) -> list[StoragePathReport]:
    """Validate that each path's filesystem has enough free space.

    Paths may point to directories that do not exist yet. The check uses the
    nearest existing parent so preflight can run before checkpoint/cache
    directories are created.
    """

    reports = [storage_path_report(path, min_free_gb=min_free_gb) for path in paths]
    failures = [report for report in reports if not report.ok]
    if failures:
        details = "; ".join(
            f"{report.path} checked at {report.checked_path}: "
            f"{report.free_gib:.2f} GiB free < {report.min_free_gib:.2f} GiB required"
            for report in failures
        )
        raise RuntimeError(f"Storage preflight failed: {details}")
    return reports
