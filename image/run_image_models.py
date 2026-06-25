import argparse
import json
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import torch


MODEL_ALIASES = {
    # 짧은 이름과 repo/folder 이름을 둘 다 허용한다.
    "i2e": "i2e",
    "eventgan": "eventgan",
    "event-gan": "eventgan",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run smoke tests for image-to-event repos.")
    parser.add_argument("data_dir", nargs="?", default="I2E/assets", help="테스트 입력 이미지 디렉토리.")
    parser.add_argument("--results-dir", default="results", help="테스트 결과와 preview PNG를 저장할 디렉토리.")
    parser.add_argument("--models", default="all", help="실행할 모델. all 또는 콤마 목록: i2e,eventgan.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="실행 디바이스. auto는 CUDA 가능 시 GPU.")
    parser.add_argument("--image-size", type=int, default=128, help="smoke test용 입력 resize 크기.")
    parser.add_argument("--i2e-ratio", type=float, default=0.12, help="I2E 이벤트 threshold ratio. 낮을수록 이벤트가 많아짐.")
    parser.add_argument("--eventgan-time-bins", type=int, default=9, help="EventGAN 출력 time bin 수.")
    parser.add_argument("--eventgan-checkpoint-dir", default="EventGAN/logs/EventGAN/checkpoints", help="EventGAN pretrained checkpoint 디렉토리.")
    return parser.parse_args()


def selected_models(models_arg):
    if models_arg == "all":
        return {"i2e", "eventgan"}
    models = set()
    for raw in models_arg.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in MODEL_ALIASES:
            raise ValueError(f"Unknown model '{raw}'")
        models.add(MODEL_ALIASES[key])
    return models


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def image_files(data_dir):
    files = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(Path(data_dir).glob(pattern))
    return sorted(files)


def load_rgb_tensor(path, size, device):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    arr = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def save_event_preview(event_frames, out_path):
    frames = event_frames.detach().float().cpu()
    pos = frames[:, 0].sum(dim=0).numpy()
    neg = frames[:, 1].sum(dim=0).numpy()
    pos = pos / max(float(pos.max()), 1.0)
    neg = neg / max(float(neg.max()), 1.0)
    image = np.zeros((frames.shape[-2], frames.shape[-1], 3), dtype=np.uint8)
    image[..., 2] = (pos * 255).astype(np.uint8)
    image[..., 0] = (neg * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), image)


def run_i2e(root, args, image_path, device, results_dir):
    sys.path.insert(0, str(root / "I2E"))
    from i2e import I2E

    model = I2E(ratio=args.i2e_ratio).to(device)
    model.eval()
    image = load_rgb_tensor(image_path, args.image_size, device)
    with torch.no_grad():
        events = model(image)
    event_frames = events[:, 0]
    npz_out = results_dir / "I2E_smoke_events.npz"
    np.savez_compressed(npz_out, event_frames=event_frames.cpu().numpy())
    out = results_dir / "I2E_smoke_event_frame.png"
    save_event_preview(event_frames, out)
    return {
        "input": str(image_path),
        "event_frames_shape": list(event_frames.shape),
        "event_count": int(events.sum().item()),
        "events": str(npz_out),
        "preview": str(out),
    }


def run_eventgan(root, args, image_paths, device, results_dir):
    eventgan_root = root / "EventGAN"
    sys.path.insert(0, str(eventgan_root / "EventGAN"))
    from models.unet import UNet
    from utils.viz_utils import gen_event_images

    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    second = cv2.imread(str(image_paths[1] if len(image_paths) > 1 else image_paths[0]), cv2.IMREAD_GRAYSCALE)
    if first is None or second is None:
        raise RuntimeError("Could not read EventGAN input images")
    first = cv2.resize(first, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
    second = cv2.resize(second, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
    pair = np.stack([first, second]).astype(np.float32)
    pair = (pair / 255.0 - 0.5) * 2.0
    images = torch.from_numpy(pair).unsqueeze(0).to(device)

    checkpoint_dir = root / args.eventgan_checkpoint_dir
    used_checkpoint = False
    if checkpoint_dir.exists() and (list(checkpoint_dir.rglob("*.pt")) or list(checkpoint_dir.rglob("*.pth"))):
        from models.eventgan_base import EventGANBase

        parsed = SimpleNamespace(
            checkpoint_dir=str(checkpoint_dir),
            n_image_channels=1,
            n_time_bins=args.eventgan_time_bins,
            sn=True,
        )
        model = EventGANBase(parsed)
        with torch.no_grad():
            event_volume = model.forward(images, is_train=False)
        used_checkpoint = True
    else:
        model = UNet(
            num_input_channels=2,
            num_output_channels=args.eventgan_time_bins * 2,
            skip_type="concat",
            activation="relu",
            num_encoders=4,
            base_num_channels=32,
            num_residual_blocks=2,
            norm="BN",
            use_upsample_conv=True,
            with_activation=True,
            sn=True,
            multi=False,
        ).to(device)
        model.eval()
        with torch.no_grad():
            event_volume = model(images)

    if isinstance(event_volume, (list, tuple)):
        event_volume = event_volume[-1]
    npz_out = results_dir / "EventGAN_smoke_event_volume.npz"
    np.savez_compressed(npz_out, event_volume=event_volume.detach().cpu().numpy())

    viz = gen_event_images(event_volume, "gen", device=str(device))
    time_preview = viz["gen_event_time_image"][0, 0].detach().cpu().numpy()
    time_preview = time_preview / max(float(time_preview.max()), 1e-6)
    time_out = results_dir / "EventGAN_smoke_event_time_image.png"
    cv2.imwrite(str(time_out), (time_preview * 255).astype(np.uint8))

    event_preview = viz["gen_event_image"][0, 0].detach().cpu().numpy()
    event_preview = event_preview / max(float(event_preview.max()), 1e-6)
    out = results_dir / "EventGAN_smoke_event_frame.png"
    cv2.imwrite(str(out), (event_preview * 255).astype(np.uint8))
    return {
        "inputs": [str(image_paths[0]), str(image_paths[1] if len(image_paths) > 1 else image_paths[0])],
        "event_shape": list(event_volume.shape),
        "used_checkpoint": used_checkpoint,
        "events": str(npz_out),
        "preview": str(out),
        "time_preview": str(time_out),
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    files = image_files(data_dir)
    if not files:
        raise RuntimeError(f"No image files found in {data_dir}")

    device = resolve_device(args.device)
    models = selected_models(args.models)
    summary = {"device": str(device), "data_dir": str(data_dir), "models": {}}

    if "i2e" in models:
        print("==> I2E smoke test", flush=True)
        summary["models"]["i2e"] = run_i2e(root, args, files[0], device, results_dir)
    if "eventgan" in models:
        print("==> EventGAN smoke test", flush=True)
        summary["models"]["eventgan"] = run_eventgan(root, args, files, device, results_dir)

    summary_path = results_dir / "image_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
