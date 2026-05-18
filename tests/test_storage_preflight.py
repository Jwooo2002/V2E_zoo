from __future__ import annotations

from pathlib import Path

import pytest

from utils.storage import nearest_existing_path, storage_path_report, validate_storage_paths


def test_nearest_existing_path_uses_parent_for_future_directory(tmp_path: Path) -> None:
    future = tmp_path / "not" / "created" / "yet"

    assert nearest_existing_path(future) == tmp_path


def test_validate_storage_paths_accepts_zero_threshold_for_future_directory(tmp_path: Path) -> None:
    future = tmp_path / "cache" / "teacher_logits"

    reports = validate_storage_paths([future], min_free_gb=0.0)

    assert len(reports) == 1
    assert reports[0].path == str(future)
    assert reports[0].ok


def test_validate_storage_paths_rejects_negative_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_free_gb must be non-negative"):
        storage_path_report(tmp_path, min_free_gb=-1.0)


def test_validate_storage_paths_raises_clear_error_when_threshold_exceeds_disk(tmp_path: Path) -> None:
    report = storage_path_report(tmp_path, min_free_gb=0.0)
    impossible_gb = report.total_bytes / (1024**3) + 1.0

    with pytest.raises(RuntimeError, match="Storage preflight failed"):
        validate_storage_paths([tmp_path / "ckpt"], min_free_gb=impossible_gb)
