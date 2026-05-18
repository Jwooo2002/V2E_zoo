from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from utils.artifact_health import check_artifacts, check_torch_artifact, discover_artifact_files


ROOT = Path(__file__).resolve().parents[1]


def _write_run_artifacts(root: Path) -> tuple[Path, Path, Path]:
    cache_dir = root / "cache" / "teacher_logits"
    checkpoint_dir = root / "checkpoints"
    cache_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    valid_cache = cache_dir / "valid.pt"
    corrupt_cache = cache_dir / "corrupt.pt"
    checkpoint = checkpoint_dir / "checkpoint_step_1_opt_1.pt"
    torch.save({"logits": torch.zeros(1, 2, 3)}, valid_cache)
    corrupt_cache.write_bytes(b"partial torch artifact")
    torch.save({"student_state_dict": {"weight": torch.ones(2)}}, checkpoint)
    return valid_cache, corrupt_cache, checkpoint


def _write_many_artifacts(root: Path, *, cache_count: int = 5, checkpoint_count: int = 2) -> None:
    cache_dir = root / "cache" / "teacher_logits"
    checkpoint_dir = root / "checkpoints"
    cache_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    for index in range(cache_count):
        torch.save({"logits": torch.zeros(1, 1, 2), "index": index}, cache_dir / f"cache_{index:03d}.pt")
    for index in range(checkpoint_count):
        torch.save(
            {"student_state_dict": {"weight": torch.ones(2)}, "step": index},
            checkpoint_dir / f"checkpoint_step_{index}_opt_{index}.pt",
        )


def test_discover_artifact_files_finds_cache_and_checkpoints(tmp_path: Path) -> None:
    valid_cache, corrupt_cache, checkpoint = _write_run_artifacts(tmp_path)

    discovered = discover_artifact_files(tmp_path)
    discovered_paths = {path for path, _kind in discovered}

    assert {valid_cache, corrupt_cache, checkpoint}.issubset(discovered_paths)


def test_check_torch_artifact_reports_corrupt_file(tmp_path: Path) -> None:
    _valid_cache, corrupt_cache, _checkpoint = _write_run_artifacts(tmp_path)

    report = check_torch_artifact(corrupt_cache, kind="teacher_cache")

    assert report.status == "corrupt"
    assert report.kind == "teacher_cache"
    assert report.error_type is not None
    assert report.error_message


def test_check_artifacts_summarizes_run_directory(tmp_path: Path) -> None:
    _write_run_artifacts(tmp_path)

    report = check_artifacts(tmp_path)

    assert report.checked_count == 3
    assert report.ok_count == 2
    assert report.corrupt_count == 1
    assert report.missing_count == 0
    assert not report.ok


def test_check_artifacts_can_skip_cache(tmp_path: Path) -> None:
    _write_run_artifacts(tmp_path)

    report = check_artifacts(tmp_path, include_cache=False)

    assert report.checked_count == 1
    assert report.corrupt_count == 0
    assert report.ok


def test_cache_sampling_still_includes_all_checkpoints(tmp_path: Path) -> None:
    _write_many_artifacts(tmp_path, cache_count=5, checkpoint_count=2)

    report = check_artifacts(tmp_path, cache_sample_size=2)

    assert report.checked_count == 4
    assert sum(1 for item in report.reports if item.kind == "teacher_cache") == 2
    assert sum(1 for item in report.reports if item.kind == "checkpoint") == 2
    assert report.ok


def test_check_artifacts_rejects_negative_cache_sample_size(tmp_path: Path) -> None:
    _write_many_artifacts(tmp_path)

    try:
        check_artifacts(tmp_path, cache_sample_size=-1)
    except ValueError as exc:
        assert "cache_sample_size must be non-negative" in str(exc)
    else:
        raise AssertionError("negative cache_sample_size should raise")


def test_check_artifacts_rejects_negative_max_files(tmp_path: Path) -> None:
    _write_run_artifacts(tmp_path)

    try:
        check_artifacts(tmp_path, max_files=-1)
    except ValueError as exc:
        assert "max_files must be non-negative" in str(exc)
    else:
        raise AssertionError("negative max_files should raise")


def test_check_artifacts_cli_writes_json_and_can_fail_on_corrupt(tmp_path: Path) -> None:
    _write_run_artifacts(tmp_path)
    output_path = tmp_path / "health.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_artifacts.py",
            str(tmp_path),
            "--output-json",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    payload = json.loads(result.stdout)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["corrupt_count"] == 1
    assert saved == payload

    failed = subprocess.run(
        [
            sys.executable,
            "scripts/check_artifacts.py",
            str(tmp_path),
            "--fail-on-corrupt",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    assert failed.returncode == 1
    assert json.loads(failed.stdout)["corrupt_count"] == 1


def test_check_artifacts_cli_cache_sample_size_keeps_checkpoints(tmp_path: Path) -> None:
    _write_many_artifacts(tmp_path, cache_count=5, checkpoint_count=2)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_artifacts.py",
            str(tmp_path),
            "--cache-sample-size",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    payload = json.loads(result.stdout)
    kinds = [item["kind"] for item in payload["reports"]]

    assert payload["checked_count"] == 4
    assert kinds.count("teacher_cache") == 2
    assert kinds.count("checkpoint") == 2
