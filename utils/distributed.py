"""Small distributed-environment helpers for smoke-scale launch scaffolds."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None
    device: torch.device = torch.device("cpu")
    process_group_initialized_by_us: bool = False

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
    if name == "WORLD_SIZE":
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}.")
    elif value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    return value


def _dist_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    if _dist_initialized():
        return int(torch.distributed.get_rank())
    return _env_int("RANK", 0)


def get_local_rank() -> int:
    return _env_int("LOCAL_RANK", 0)


def get_world_size() -> int:
    if _dist_initialized():
        return int(torch.distributed.get_world_size())
    return _env_int("WORLD_SIZE", 1)


def is_distributed() -> bool:
    return _dist_initialized() or get_world_size() > 1


def is_rank_zero() -> bool:
    return get_rank() == 0


def distributed_env_requested() -> bool:
    return get_world_size() > 1


def init_distributed(
    *,
    mode: str = "none",
    backend: str | None = None,
    enabled: bool | None = None,
) -> DistributedContext:
    if enabled is not None:
        mode = "env" if enabled or distributed_env_requested() else "none"
    if mode not in {"none", "env", "ddp"}:
        raise ValueError("distributed mode must be one of: none, env, ddp.")
    if mode == "none":
        return DistributedContext(enabled=False, device=get_device_for_rank())

    world_size = get_world_size()
    if world_size <= 1:
        rank = get_rank()
        local_rank = get_local_rank()
        return DistributedContext(
            enabled=False,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=backend,
            device=get_device_for_rank(),
        )
    if not _dist_initialized():
        missing = [name for name in ("RANK", "LOCAL_RANK") if os.environ.get(name) in (None, "")]
        if missing:
            raise ValueError(
                "Distributed env mode requires RANK and LOCAL_RANK when WORLD_SIZE > 1; "
                f"missing: {', '.join(missing)}."
            )

    selected_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    local_rank = get_local_rank()
    device = get_device_for_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    initialized_by_us = False
    if mode == "ddp" and not _dist_initialized():
        torch.distributed.init_process_group(backend=selected_backend, init_method="env://")
        initialized_by_us = True

    rank = get_rank()
    world_size = get_world_size()
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=selected_backend,
        device=device,
        process_group_initialized_by_us=initialized_by_us,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.process_group_initialized_by_us and _dist_initialized():
        torch.distributed.destroy_process_group()


def barrier(context: DistributedContext | None = None) -> None:
    if context is not None and not context.enabled:
        return
    if _dist_initialized():
        torch.distributed.barrier()


def rank_local_dir(path: str | Path, context: DistributedContext) -> str:
    if not context.enabled:
        return str(path)
    return str(Path(path) / f"rank_{context.rank:05d}")


def unwrap_model(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def average_float_dict(values: dict[str, float], context: DistributedContext) -> dict[str, float]:
    if not context.enabled or not _dist_initialized():
        return dict(values)
    keys = sorted(values)
    tensor = torch.tensor([float(values[key]) for key in keys], dtype=torch.float64, device=context.device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / float(context.world_size)
    return {key: float(tensor[index].item()) for index, key in enumerate(keys)}


class RankZeroLogger:
    def __init__(self, logger: Any, context: DistributedContext) -> None:
        self.logger = logger
        self.context = context

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        if self.context.is_rank_zero:
            self.logger.log(step, metrics)


def rank_zero_print(*args: Any, **kwargs: Any) -> None:
    if is_rank_zero():
        print(*args, **kwargs)


def rank_zero_json_log(record: dict[str, Any]) -> None:
    if is_rank_zero():
        print(json.dumps(record, sort_keys=True), file=sys.stdout, flush=True)


def get_device_for_rank(preferred: str | None = None) -> torch.device:
    choice = "auto" if preferred in (None, "") else str(preferred).lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if choice == "cuda" or choice == "auto":
            local_rank = get_local_rank()
            device_count = torch.cuda.device_count()
            if local_rank >= device_count:
                raise ValueError(f"LOCAL_RANK={local_rank} is outside available CUDA device count {device_count}.")
            return torch.device("cuda", local_rank)
        return torch.device(choice)
    if choice == "auto":
        if torch.cuda.is_available():
            local_rank = get_local_rank()
            device_count = torch.cuda.device_count()
            if local_rank >= device_count:
                raise ValueError(f"LOCAL_RANK={local_rank} is outside available CUDA device count {device_count}.")
            return torch.device("cuda", local_rank)
        return torch.device("cpu")
    return torch.device(choice)


def effective_batch_size(batch_size: int, gradient_accumulation_steps: int, world_size: int | None = None) -> int:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            f"gradient_accumulation_steps must be positive, got {gradient_accumulation_steps}."
        )
    size = get_world_size() if world_size is None else int(world_size)
    if size <= 0:
        raise ValueError(f"world_size must be positive, got {size}.")
    return int(batch_size) * int(gradient_accumulation_steps) * size
