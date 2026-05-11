"""Run a tiny real-Mamba instantiate/forward smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from models.student_mamba import MambaStudentConfig, RealMambaStudent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def main() -> int:
    args = parse_args()
    try:
        device = _resolve_device(args.device)
        config = MambaStudentConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            torch_dtype="float32",
            device=str(device),
            use_pretrained=False,
        )
        student = RealMambaStudent(config).eval()
        input_ids = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(args.batch_size, args.seq_len),
            device=device,
            dtype=torch.long,
        )
        with torch.no_grad():
            output = student(input_ids)
        payload = {
            "success": True,
            "device": str(device),
            "input_shape": list(input_ids.shape),
            "on_logits_shape": list(output.on_logits.shape),
            "off_logits_shape": list(output.off_logits.shape),
            "fake_logits_shape": list(output.fake_logits.shape),
            "dtype": str(output.on_logits.dtype).replace("torch.", ""),
            "mamba_ssm_version": student.mamba_ssm_version,
        }
        print(json.dumps(payload, sort_keys=True))
        return 0
    except Exception as exc:
        payload = {
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
