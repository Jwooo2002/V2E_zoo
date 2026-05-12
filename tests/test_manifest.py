from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from utils.manifest import (
    RunManifest,
    copy_config_files,
    create_run_dir,
    generate_run_id,
    get_env_info,
    get_git_info,
    load_manifest,
    write_manifest,
)


def test_generate_run_id_is_deterministic_and_safe() -> None:
    timestamp = datetime(2026, 5, 11, 22, 30, tzinfo=timezone.utc)

    first = generate_run_id(prefix="stage 8E", timestamp=timestamp, extra={"a": 1})
    second = generate_run_id(prefix="stage 8E", timestamp=timestamp, extra={"a": 1})
    changed = generate_run_id(prefix="stage 8E", timestamp=timestamp, extra={"a": 2})

    assert first == second
    assert changed != first
    assert first.startswith("stage_8E_20260511_223000_")
    assert "/" not in first
    assert "\\" not in first
    assert " " not in first


def test_get_env_info_has_python_and_executable() -> None:
    env = get_env_info()

    assert env.python
    assert env.executable
    assert env.platform
    assert isinstance(env.gpu_names, list)


def test_get_git_info_does_not_fail_outside_git(tmp_path: Path) -> None:
    info = get_git_info(tmp_path)

    assert info.commit is None
    assert info.branch is None
    assert info.is_dirty is False


def test_create_run_dir_creates_expected_subdirectories(tmp_path: Path) -> None:
    run_dir = create_run_dir(tmp_path, "run_20260511_223000_ab12cd")

    assert (run_dir / "manifest.json").parent == run_dir
    for name in ("configs", "logs", "checkpoints", "cache", "reports", "evals", "artifacts"):
        assert (run_dir / name).is_dir()


def test_create_run_dir_rejects_unsafe_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        create_run_dir(tmp_path, "../bad")


def test_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = RunManifest(
        run_id="run_1",
        created_at="2026-05-11T22:30:00+00:00",
        project="cdm-mamba-kd",
        stage="8E",
        command=["python", "train.py"],
        config_paths=["configs/train_config.yaml"],
        output_dir=str(tmp_path),
        git={"commit": "abc", "branch": "master", "is_dirty": False, "diff_summary": None},
        env={"python": "3.10"},
        metadata={"status": "planned"},
    )

    path = write_manifest(manifest, tmp_path / "manifest.json")
    loaded = load_manifest(path)

    assert isinstance(loaded, RunManifest)
    assert asdict(loaded) == asdict(manifest)


def test_copy_config_files(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    source.write_text("a: 1\n", encoding="utf-8")
    copied = copy_config_files([source], tmp_path / "run" / "configs")

    assert len(copied) == 1
    assert copied[0].read_text(encoding="utf-8") == "a: 1\n"
