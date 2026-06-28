import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run EventGAN frame-pair to event-volume conversion.")
    parser.add_argument("data_dir", nargs="?", default="I2E/assets", help="입력 이미지 디렉토리. --prev-image/--next-image가 없으면 앞의 2장을 사용.")
    parser.add_argument("--prev-image", default=None, help="이전 grayscale/RGB 프레임 경로.")
    parser.add_argument("--next-image", default=None, help="다음 grayscale/RGB 프레임 경로.")
    parser.add_argument("--results-dir", default="results/eventgan", help="EventGAN 결과 저장 디렉토리.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="실행 디바이스.")
    parser.add_argument("--width", type=int, default=861, help="EventGAN 입력 resize 너비. 공식 demo 기본값 861.")
    parser.add_argument("--height", type=int, default=260, help="EventGAN 입력 resize 높이. 공식 demo 기본값 260.")
    parser.add_argument("--time-bins", type=int, default=9, help="EventGAN time bin 수. checkpoint 기본값 9.")
    parser.add_argument("--checkpoint-dir", default="EventGAN/logs/EventGAN/checkpoints", help="EventGAN checkpoint 디렉토리.")
    parser.add_argument("--synthetic-shift", type=int, default=0, help="next frame이 없을 때 같은 이미지를 x축으로 shift해서 pair 생성.")
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def image_files(data_dir):
    files = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(Path(data_dir).glob(pattern))
    return sorted(files)


def resolve_path(root, value):
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def choose_pair(root, args):
    prev = resolve_path(root, args.prev_image)
    nxt = resolve_path(root, args.next_image)
    if prev is not None:
        return prev, nxt

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    files = image_files(data_dir)
    if len(files) >= 2:
        return files[0], files[1]
    if len(files) == 1:
        return files[0], None
    raise RuntimeError(f"No image files found in {data_dir}")


def load_pair(prev_path, next_path, width, height, synthetic_shift):
    prev = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
    if prev is None:
        raise RuntimeError(f"Could not read image: {prev_path}")
    prev = cv2.resize(prev, (width, height), interpolation=cv2.INTER_AREA)

    if next_path is None:
        shift = synthetic_shift if synthetic_shift != 0 else 4
        nxt = np.roll(prev, shift=shift, axis=1)
    else:
        nxt = cv2.imread(str(next_path), cv2.IMREAD_GRAYSCALE)
        if nxt is None:
            raise RuntimeError(f"Could not read image: {next_path}")
        nxt = cv2.resize(nxt, (width, height), interpolation=cv2.INTER_AREA)

    pair = np.stack([prev, nxt]).astype(np.float32)
    pair = (pair / 255.0 - 0.5) * 2.0
    return pair


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    eventgan_root = root / "EventGAN"
    sys.path.insert(0, str(eventgan_root / "EventGAN"))
    from models.eventgan_base import EventGANBase
    from utils.viz_utils import gen_event_images

    device = resolve_device(args.device)
    prev_path, next_path = choose_pair(root, args)
    pair = load_pair(prev_path, next_path, args.width, args.height, args.synthetic_shift)
    images = torch.from_numpy(pair).unsqueeze(0).to(device)

    checkpoint_dir = root / args.checkpoint_dir
    if not checkpoint_dir.exists() or not (list(checkpoint_dir.rglob("*.pt")) or list(checkpoint_dir.rglob("*.pth"))):
        raise RuntimeError(f"No EventGAN checkpoint found in {checkpoint_dir}")

    options = SimpleNamespace(
        checkpoint_dir=str(checkpoint_dir),
        n_image_channels=1,
        n_time_bins=args.time_bins,
        sn=True,
    )
    model = EventGANBase(options)
    model.generator.to(device)
    with torch.no_grad():
        event_volume = model.forward(images, is_train=False)
    if isinstance(event_volume, (list, tuple)):
        event_volume = event_volume[-1]

    stem = f"{prev_path.stem}_{next_path.stem if next_path else 'shift'}"
    npz_out = results_dir / f"{stem}_eventgan_event_volume.npz"
    np.savez_compressed(npz_out, event_volume=event_volume.detach().cpu().numpy())

    viz = gen_event_images(event_volume, "gen", device=str(device))
    event_preview = viz["gen_event_image"][0, 0].detach().cpu().numpy()
    event_preview = event_preview / max(float(event_preview.max()), 1e-6)
    event_out = results_dir / f"{stem}_eventgan_event_frame.png"
    cv2.imwrite(str(event_out), (event_preview * 255).astype(np.uint8))

    time_preview = viz["gen_event_time_image"][0, 0].detach().cpu().numpy()
    time_preview = time_preview / max(float(time_preview.max()), 1e-6)
    time_out = results_dir / f"{stem}_eventgan_event_time_image.png"
    cv2.imwrite(str(time_out), (time_preview * 255).astype(np.uint8))

    summary = {
        "device": str(device),
        "prev_image": str(prev_path),
        "next_image": str(next_path) if next_path else None,
        "event_shape": list(event_volume.shape),
        "checkpoint_dir": str(checkpoint_dir),
        "events": str(npz_out),
        "preview": str(event_out),
        "time_preview": str(time_out),
    }
    summary_path = results_dir / "eventgan_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
