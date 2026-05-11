"""Run a tiny real-Mamba instantiate/forward smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from models.student_mamba import MambaStudentConfig, RealMambaStudent  # noqa: E402

_MAX_ERROR_MESSAGE_CHARS = 300
_TENSOR_REPR_PATTERN = re.compile(r"tensor\([\s\S]*?(?:\)\s*,|\)\s*$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--use-reference-forward",
        action="store_true",
        help="Use smoke-only reference/non-fused Mamba kernels when available.",
    )
    parser.add_argument(
        "--state-extraction",
        choices=("last_hidden", "embedding", "none"),
        default="last_hidden",
        help="Student-side state scaffold to expose in the smoke output.",
    )
    parser.add_argument(
        "--no-expose-states",
        action="store_true",
        help="Disable h/h_off/h_delta_alt exposure in the smoke output.",
    )
    parser.add_argument(
        "--off-state-mode",
        choices=("projection", "placeholder", "none"),
        default="projection",
        help="Stage 6E student-side off-state approximation mode.",
    )
    parser.add_argument(
        "--delta-alt-mode",
        choices=("delta_projection", "noise", "identity"),
        default="delta_projection",
        help="Stage 6E student-side h_delta_alt approximation mode.",
    )
    parser.add_argument(
        "--off-logits-mode",
        choices=("lm_head", "projection_head", "placeholder"),
        default="lm_head",
        help="Projection used to map h_off to off_logits.",
    )
    parser.add_argument(
        "--off-state-detach-direction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach h_delta_alt - h before DeltaPerturbationEngine applies rho.",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def _is_causal_conv1d_fast_path_error(exc: BaseException) -> bool:
    text = str(exc)
    return "causal_conv1d" in text and (
        "incompatible function arguments" in text
        or "Expected x.is_cuda() to be true" in text
        or "causal_conv1d_fwd" in text
    )


def _compact_error_message(exc: BaseException) -> tuple[str, bool]:
    message = str(exc) or type(exc).__name__
    message = message.split("Invoked with:", 1)[0]
    message = _TENSOR_REPR_PATTERN.sub("tensor(<omitted>) ", message)
    message = " ".join(message.split())
    truncated = len(message) > _MAX_ERROR_MESSAGE_CHARS
    if truncated:
        message = f"{message[: _MAX_ERROR_MESSAGE_CHARS - 3]}..."
    return message, truncated


def _compact_error_payload(
    exc: BaseException,
    *,
    device: str | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    message, truncated = _compact_error_message(exc)
    payload: dict[str, Any] = {
        "success": False,
        "error_type": type(exc).__name__,
        "error_message": message,
        "error_truncated": truncated,
    }
    if device is not None:
        payload["device"] = device
    if stage is not None:
        payload["stage"] = stage
    if _is_causal_conv1d_fast_path_error(exc):
        payload["probable_cause"] = (
            "mamba_ssm import works, but the fused causal_conv1d fast path is "
            "incompatible with the installed causal-conv1d API."
        )
        payload["suggested_action"] = (
            "Use --use-reference-forward for Stage 6C smoke, or install a "
            "pinned compatible mamba-ssm + causal-conv1d wheel pair before "
            "CUDA fused training work."
        )
    else:
        payload["probable_cause"] = "Stage 6C-6E real-Mamba smoke could not complete in this environment."
        payload["suggested_action"] = (
            "Run scripts/check_mamba_env.py and retry CPU/reference smoke before real training work."
        )
    return payload


def _run_forward(args: argparse.Namespace, device: torch.device, *, use_reference_forward: bool) -> dict[str, Any]:
    config = MambaStudentConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        torch_dtype="float32",
        device=str(device),
        use_pretrained=False,
        use_reference_forward=use_reference_forward,
        state_extraction=args.state_extraction,
        expose_states=not args.no_expose_states,
        off_state_mode=args.off_state_mode,
        delta_alt_mode=args.delta_alt_mode,
        off_logits_mode=args.off_logits_mode,
        off_state_detach_direction=args.off_state_detach_direction,
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
    metadata = output.metadata or {}
    return {
        "success": True,
        "device": str(device),
        "input_shape": list(input_ids.shape),
        "on_logits_shape": list(output.on_logits.shape),
        "off_logits_shape": list(output.off_logits.shape),
        "fake_logits_shape": list(output.fake_logits.shape),
        "h_shape": _shape_or_none(output.h),
        "h_off_shape": _shape_or_none(output.h_off),
        "h_delta_alt_shape": _shape_or_none(output.h_delta_alt),
        "state_extraction": metadata.get("state_extraction", config.state_extraction),
        "expose_states": metadata.get("expose_states", config.expose_states),
        "off_state_mode": metadata.get("off_state_mode", config.off_state_mode),
        "delta_alt_mode": metadata.get("delta_alt_mode", config.delta_alt_mode),
        "off_logits_mode": metadata.get("off_logits_mode", config.off_logits_mode),
        "off_state_detach_direction": metadata.get(
            "off_state_detach_direction",
            config.off_state_detach_direction,
        ),
        "off_logits_source": metadata.get("off_logits_source"),
        "off_state_source": metadata.get("off_state_source"),
        "delta_alt_source": metadata.get("delta_alt_source"),
        "off_state_available": metadata.get("off_state_available"),
        "delta_alt_available": metadata.get("delta_alt_available"),
        "off_logits_placeholder": metadata.get(
            "off_logits_placeholder",
            output.off_logits.data_ptr() == output.on_logits.data_ptr(),
        ),
        "smoke_placeholder_off_logits": metadata.get(
            "smoke_placeholder_off_logits",
            output.off_logits.data_ptr() == output.on_logits.data_ptr(),
        ),
        "dtype": str(output.on_logits.dtype).replace("torch.", ""),
        "mamba_ssm_version": student.mamba_ssm_version,
        "reference_forward": use_reference_forward or device.type == "cpu",
        "requested_reference_forward": use_reference_forward,
    }


def _shape_or_none(tensor: torch.Tensor | None) -> list[int] | None:
    return None if tensor is None else list(tensor.shape)


def main() -> int:
    args = parse_args()
    try:
        device = _resolve_device(args.device)
        try:
            payload = _run_forward(args, device, use_reference_forward=args.use_reference_forward)
        except (RuntimeError, TypeError) as exc:
            if args.use_reference_forward or not _is_causal_conv1d_fast_path_error(exc):
                raise
            payload = _run_forward(args, device, use_reference_forward=True)
            payload["fallback"] = "reference_forward_after_causal_conv1d_fast_path_error"
        print(json.dumps(payload, sort_keys=True))
        return 0
    except Exception as exc:
        payload = _compact_error_payload(
            exc,
            device=args.device,
            stage="causal_conv1d_cuda_fast_path" if _is_causal_conv1d_fast_path_error(exc) else "mamba_forward",
        )
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
