"""Simple console logging for mock training."""

from __future__ import annotations

import json
import math
import sys
from typing import Any


class ConsoleLogger:
    """Write parseable JSON metrics to stdout."""

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        payload: dict[str, Any] = {"step": step}
        for key, value in metrics.items():
            if isinstance(value, float):
                payload[key] = value if math.isfinite(value) else str(value)
            else:
                payload[key] = value
        print(json.dumps(payload, sort_keys=True), file=sys.stdout, flush=True)
