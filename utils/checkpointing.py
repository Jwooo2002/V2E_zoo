"""Minimal torch checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    state = torch.load(Path(path), map_location=map_location)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint must contain a dict, got {type(state).__name__}.")
    return state
