import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run I2E single-image to event-frame conversion.")
    parser.add_argument("data_dir", nargs="?", default="I2E/assets", help="입력 이미지 디렉토리.")
    parser.add_argument("--results-dir", default="results/i2e", help="I2E 결과 저장 디렉토리.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="실행 디바이스.")
    parser.add_argument("--image-size", type=int, default=128, help="입력 이미지를 정사각형으로 resize할 크기.")
    parser.add_argument("--ratio", type=float, default=0.12, help="I2E 이벤트 threshold ratio. 낮을수록 이벤트 증가.")
    parser.add_argument("--limit", type=int, default=0, help="처리할 이미지 최대 개수. 0이면 전체.")
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


def load_rgb_tensor(path, size, device):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    arr = image.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def save_preview(event_frames, out_path):
    frames = event_frames.detach().float().cpu()
    pos = frames[:, 0].sum(dim=0).numpy()
    neg = frames[:, 1].sum(dim=0).numpy()
    pos = pos / max(float(pos.max()), 1.0)
    neg = neg / max(float(neg.max()), 1.0)
    image = np.zeros((frames.shape[-2], frames.shape[-1], 3), dtype=np.uint8)
    image[..., 2] = (pos * 255).astype(np.uint8)
    image[..., 0] = (neg * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), image)


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
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No image files found in {data_dir}")

    sys.path.insert(0, str(root / "I2E"))
    from i2e import I2E

    device = resolve_device(args.device)
    model = I2E(ratio=args.ratio).to(device)
    model.eval()

    summary = {"device": str(device), "data_dir": str(data_dir), "images": []}
    for image_path in files:
        image = load_rgb_tensor(image_path, args.image_size, device)
        with torch.no_grad():
            events = model(image)
        event_frames = events[:, 0]  # T x 2 x H x W

        stem = image_path.stem
        npz_out = results_dir / f"{stem}_i2e_events.npz"
        preview_out = results_dir / f"{stem}_i2e_event_frame.png"
        np.savez_compressed(npz_out, event_frames=event_frames.cpu().numpy())
        save_preview(event_frames, preview_out)
        summary["images"].append(
            {
                "input": str(image_path),
                "event_frames_shape": list(event_frames.shape),
                "event_count": int(event_frames.sum().item()),
                "events": str(npz_out),
                "preview": str(preview_out),
            }
        )

    summary_path = results_dir / "i2e_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
