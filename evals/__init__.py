"""Stage 4 mock evaluation scaffolds."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _ensure_repo_train_module() -> None:
    module = sys.modules.get("train")
    if module is not None and all(
        hasattr(module, name)
        for name in ("TrainConfig", "load_train_config", "set_seed", "_select_shared_valid_mask")
    ):
        return

    train_path = Path(__file__).resolve().parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("train", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load repository train.py from {train_path}")
    train_module = importlib.util.module_from_spec(spec)
    sys.modules["train"] = train_module
    spec.loader.exec_module(train_module)


_ensure_repo_train_module()

from evals.needle import evaluate_needle_scaffold
from evals.perplexity import evaluate_perplexity
from evals.perturbation_robustness import evaluate_perturbation_robustness

__all__ = [
    "evaluate_needle_scaffold",
    "evaluate_perplexity",
    "evaluate_perturbation_robustness",
]
