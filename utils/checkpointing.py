"""Checkpoint helpers for smoke-scale training and resume."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class TrainingCheckpointState:
    step: int
    optimizer_step: int
    config: dict[str, Any] | None
    metadata: dict[str, Any]
    path: Path
    student_state: dict[str, Any] | None = None
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    rng_state: dict[str, Any] | None = None


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    state = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint must contain a dict, got {type(state).__name__}.")
    return state


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    student: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    scheduler: Any | None = None,
    step: int,
    optimizer_step: int,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    rng_state: bool = True,
    filename: str | None = None,
) -> Path:
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}.")
    if optimizer_step < 0:
        raise ValueError(f"optimizer_step must be non-negative, got {optimizer_step}.")

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    filename = filename or f"checkpoint_step_{step}_opt_{optimizer_step}.pt"
    path = checkpoint_path / filename
    student_state = student.state_dict()
    optimizer_state = None if optimizer is None else optimizer.state_dict()
    scheduler_state = None if scheduler is None else scheduler.state_dict()
    payload: dict[str, Any] = {
        "student_state_dict": student_state,
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler_state,
        "student_state": student_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "step": int(step),
        "optimizer_step": int(optimizer_step),
        "config": config,
        "metadata": metadata or {},
    }
    if rng_state:
        payload["rng_state"] = _capture_rng_state()
    torch.save(payload, path)
    return path


def load_training_checkpoint(
    checkpoint_path: str | Path,
    student: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    load_optimizer: bool = True,
    load_rng_state: bool = True,
    restore_rng: bool | None = None,
) -> TrainingCheckpointState:
    path = Path(checkpoint_path)
    payload = load_checkpoint(path, map_location=map_location)
    student_state = payload.get("student_state_dict", payload.get("student_state"))
    if student_state is None:
        raise KeyError("training checkpoint is missing student_state_dict.")
    if student is not None:
        student.load_state_dict(student_state, strict=strict)

    if load_optimizer and optimizer is not None:
        optimizer_state = payload.get("optimizer_state_dict", payload.get("optimizer_state"))
        if optimizer_state is None:
            raise KeyError("training checkpoint is missing optimizer_state_dict.")
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = payload.get("scheduler_state_dict", payload.get("scheduler_state"))
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    should_restore_rng = load_rng_state if restore_rng is None else restore_rng
    if should_restore_rng:
        if "rng_state" not in payload:
            raise ValueError("training checkpoint is missing rng_state.")
        _restore_rng_state(payload["rng_state"])

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise TypeError("training checkpoint metadata must be a dict.")
    config = payload.get("config")
    if config is not None and not isinstance(config, dict):
        raise TypeError("training checkpoint config must be a dict when provided.")
    return TrainingCheckpointState(
        step=int(payload.get("step", 0)),
        optimizer_step=int(payload.get("optimizer_step", payload.get("step", 0))),
        config=config,
        metadata=metadata,
        path=path,
        student_state=dict(student_state),
        optimizer_state=payload.get("optimizer_state_dict", payload.get("optimizer_state")),
        scheduler_state=scheduler_state,
        rng_state=payload.get("rng_state", {}),
    )


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    directory = Path(checkpoint_dir)
    if not directory.exists():
        return None
    checkpoints = [path for path in directory.glob("checkpoint_step_*_opt_*.pt") if path.is_file()]
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_sort_key)


def _checkpoint_sort_key(path: Path) -> tuple[int, int, float]:
    match = re.match(r"checkpoint_step_(\d+)_opt_(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1)), int(match.group(2)), path.stat().st_mtime
    return -1, -1, path.stat().st_mtime


def _capture_rng_state() -> dict[str, Any]:
    python_state = random.getstate()
    torch_state = _as_cpu_uint8_rng_tensor(torch.get_rng_state(), key="torch_cpu")
    state: dict[str, Any] = {
        "torch_cpu": torch_state,
        "torch_cuda": None,
        "python_random": python_state,
        "numpy_random": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [
            _as_cpu_uint8_rng_tensor(cuda_state, key=f"torch_cuda[{index}]")
            for index, cuda_state in enumerate(torch.cuda.get_rng_state_all())
        ]
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        state["numpy_random"] = np.random.get_state()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise ValueError("rng_state must be a dict.")
    torch_state = _get_first_present(state, ("torch_cpu", "torch_rng_state", "torch"))
    if torch_state is None:
        raise ValueError("rng_state is missing torch_cpu.")
    torch.set_rng_state(_as_cpu_uint8_rng_tensor(torch_state, key="torch_cpu"))

    cuda_state = _get_first_present(state, ("torch_cuda", "cuda_rng_state_all", "cuda"))
    if cuda_state is not None:
        cuda_states = _normalize_cuda_rng_states(cuda_state)
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if len(cuda_states) != device_count:
                raise ValueError(
                    "rng_state['torch_cuda'] contains "
                    f"{len(cuda_states)} state(s), but CUDA reports {device_count} device(s)."
                )
            torch.cuda.set_rng_state_all(cuda_states)

    python_state = state.get(
        "python_random",
        state.get("python_random_state", state.get("python")),
    )
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy_random", state.get("numpy_random_state", state.get("numpy")))
    if numpy_state is not None:
        try:
            import numpy as np
        except ImportError:
            return
        np.random.set_state(numpy_state)


def _get_first_present(state: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in state:
            return state[key]
    return None


def _normalize_cuda_rng_states(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError("rng_state['torch_cuda'] must be a tensor or a list/tuple of tensors.")
    return [_as_cpu_uint8_rng_tensor(cuda_rng, key=f"torch_cuda[{index}]") for index, cuda_rng in enumerate(values)]


def _as_cpu_uint8_rng_tensor(value: Any, *, key: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    elif isinstance(value, (bytes, bytearray)):
        tensor = torch.tensor(list(value), dtype=torch.uint8)
    elif isinstance(value, (list, tuple)):
        tensor = torch.as_tensor(value)
    else:
        raise ValueError(f"rng_state[{key!r}] must be a torch.Tensor, bytes, bytearray, list, or tuple.")
    tensor = tensor.cpu()
    if tensor.numel() == 0:
        raise ValueError(f"rng_state[{key!r}] must not be empty.")
    if tensor.dtype != torch.uint8:
        if tensor.dtype == torch.bool or torch.is_floating_point(tensor) or torch.is_complex(tensor):
            raise ValueError(f"rng_state[{key!r}] must use integer byte values, got dtype {tensor.dtype}.")
        min_value = int(tensor.min().item())
        max_value = int(tensor.max().item())
        if min_value < 0 or max_value > 255:
            raise ValueError(f"rng_state[{key!r}] contains values outside the uint8 range.")
        tensor = tensor.to(dtype=torch.uint8)
    if tensor.dim() != 1:
        raise ValueError(f"rng_state[{key!r}] must be a 1D RNG state tensor.")
    return tensor.contiguous().clone()
