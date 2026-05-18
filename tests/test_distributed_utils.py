from __future__ import annotations

from pathlib import Path

import pytest
import torch

from utils.distributed import (
    DistributedContext,
    RankZeroLogger,
    average_float_dict,
    effective_batch_size,
    get_device_for_rank,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_rank_zero,
    rank_local_dir,
    rank_zero_json_log,
    unwrap_model,
)


class RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[int, dict[str, object]]] = []

    def log(self, step: int, metrics: dict[str, object]) -> None:
        self.records.append((step, metrics))


def test_single_process_fallback_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)

    assert get_rank() == 0
    assert get_local_rank() == 0
    assert get_world_size() == 1
    assert is_distributed() is False
    assert is_rank_zero() is True


def test_parse_torchrun_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")

    assert get_rank() == 2
    assert get_local_rank() == 1
    assert get_world_size() == 4
    assert is_distributed() is True
    assert is_rank_zero() is False


def test_invalid_distributed_env_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "abc")

    with pytest.raises(ValueError, match="RANK"):
        get_rank()

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "0")
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        get_world_size()


def test_init_distributed_env_uses_env_without_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    context = init_distributed(mode="env", backend="gloo")

    assert context.enabled is True
    assert context.rank == 1
    assert context.local_rank == 0
    assert context.world_size == 2
    assert context.device == torch.device("cpu")


def test_init_distributed_env_requires_rank_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    with pytest.raises(ValueError, match="RANK"):
        init_distributed(mode="env")


def test_init_distributed_ddp_initializes_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"initialized": False, "init_backend": None, "destroyed": False}
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: state["initialized"])
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_init_process_group(*, backend: str, init_method: str) -> None:
        state["initialized"] = True
        state["init_backend"] = backend
        state["init_method"] = init_method

    def fake_destroy_process_group() -> None:
        state["destroyed"] = True
        state["initialized"] = False

    monkeypatch.setattr(torch.distributed, "init_process_group", fake_init_process_group)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", fake_destroy_process_group)

    context = init_distributed(mode="ddp", backend="gloo")

    assert context.enabled is True
    assert context.rank == 1
    assert context.world_size == 2
    assert context.backend == "gloo"
    assert context.process_group_initialized_by_us is True
    assert state["init_backend"] == "gloo"
    assert state["init_method"] == "env://"

    from utils.distributed import cleanup_distributed

    cleanup_distributed(context)

    assert state["destroyed"] is True


def test_init_distributed_env_does_not_initialize_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    def fail_init_process_group(*args, **kwargs) -> None:
        raise AssertionError("env mode should not initialize a process group")

    monkeypatch.setattr(torch.distributed, "init_process_group", fail_init_process_group)

    context = init_distributed(mode="env", backend="gloo")

    assert context.enabled is True
    assert context.process_group_initialized_by_us is False


def test_get_device_for_rank_returns_cpu_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert get_device_for_rank("auto") == torch.device("cpu")
    assert get_device_for_rank("cuda") == torch.device("cpu")


def test_rank_zero_json_log_filters_by_env_rank(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("RANK", "1")
    rank_zero_json_log({"step": 1, "loss": 2.0})
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("RANK", "0")
    rank_zero_json_log({"step": 1, "loss": 2.0})
    assert '"loss": 2.0' in capsys.readouterr().out


def test_effective_batch_size_uses_world_size() -> None:
    assert effective_batch_size(batch_size=2, gradient_accumulation_steps=8, world_size=1) == 16
    assert effective_batch_size(batch_size=2, gradient_accumulation_steps=8, world_size=2) == 32


def test_rank_local_dir_is_noop_without_distributed() -> None:
    context = DistributedContext(enabled=False)

    assert rank_local_dir("/tmp/cache", context) == "/tmp/cache"


def test_rank_local_dir_adds_rank_suffix_for_distributed() -> None:
    context = DistributedContext(enabled=True, rank=7, local_rank=1, world_size=8)

    assert rank_local_dir(Path("/tmp/cache"), context) == "/tmp/cache/rank_00007"


def test_rank_zero_logger_filters_nonzero_rank() -> None:
    recorder = RecordingLogger()
    logger = RankZeroLogger(recorder, DistributedContext(enabled=True, rank=1, local_rank=1, world_size=2))

    logger.log(1, {"loss": 1.0})

    assert recorder.records == []


def test_rank_zero_logger_keeps_rank_zero_records() -> None:
    recorder = RecordingLogger()
    logger = RankZeroLogger(recorder, DistributedContext(enabled=True, rank=0, local_rank=0, world_size=2))

    logger.log(1, {"loss": 1.0})

    assert recorder.records == [(1, {"loss": 1.0})]


def test_average_float_dict_is_noop_without_distributed() -> None:
    context = DistributedContext(enabled=False)

    assert average_float_dict({"b": 2.0, "a": 1.0}, context) == {"b": 2.0, "a": 1.0}


def test_unwrap_model_returns_module_attribute_when_present() -> None:
    module = torch.nn.Linear(2, 2)

    class Wrapper(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.module = wrapped

    assert unwrap_model(Wrapper(module)) is module
    assert unwrap_model(module) is module
