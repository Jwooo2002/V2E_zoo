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
            raise KeyError("training checkpoint is missing rng_state.")
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
    torch_state = torch.get_rng_state()
    state: dict[str, Any] = {
        "python_random_state": python_state,
        "torch_rng_state": torch_state,
        "python": python_state,
        "torch": torch_state,
    }
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
        state["cuda_rng_state_all"] = cuda_state
        state["cuda"] = cuda_state
    try:
        import numpy as np
    except ImportError:
        state["numpy_random_state"] = None
    else:
        numpy_state = np.random.get_state()
        state["numpy_random_state"] = numpy_state
        state["numpy"] = numpy_state
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise TypeError("rng_state must be a dict.")
    python_state = state.get("python_random_state", state.get("python"))
    if python_state is not None:
        random.setstate(python_state)
    torch_state = state.get("torch_rng_state", state.get("torch"))
    if torch_state is not None:
        torch.set_rng_state(torch_state.cpu() if hasattr(torch_state, "cpu") else torch_state)
    cuda_state = state.get("cuda_rng_state_all", state.get("cuda"))
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    numpy_state = state.get("numpy_random_state", state.get("numpy"))
    if numpy_state is not None:
        try:
            import numpy as np
        except ImportError:
            return
        np.random.set_state(numpy_state)
