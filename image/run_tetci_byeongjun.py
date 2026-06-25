import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run TETCI_byeongjun RGB/grayscale image to synthetic event-style image conversion.")
    parser.add_argument("data_dir", nargs="?", default="I2E/assets", help="입력 이미지 디렉토리.")
    parser.add_argument("--results-dir", default="results/TETCI_byeongjun", help="TETCI_byeongjun 결과 저장 디렉토리.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="실행 디바이스.")
    parser.add_argument("--beta", type=float, default=0.5, help="LIF membrane decay beta. 낮을수록 이전 픽셀/시간 영향이 빨리 사라짐.")
    parser.add_argument("--threshold", type=float, default=0.65, help="LIF spike threshold. 낮을수록 spike/event 증가.")
    parser.add_argument("--thresholds", default=None, help="여러 threshold sweep. 예: 0.45,0.55,0.65,0.75")
    parser.add_argument("--no-spike-on-prob", type=float, default=0.2, help="원본 코드의 non-spike random white noise 확률.")
    parser.add_argument("--spike-value", type=int, default=200, help="spike 위치에 저장할 grayscale 값.")
    parser.add_argument("--seed", type=int, default=0, help="random noise 재현용 seed.")
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


def run_lif_scan(image_tensor, beta, threshold, device):
    try:
        import snntorch as snn

        lif = snn.Leaky(beta=beta, threshold=threshold).to(device)
        mem = torch.zeros(1, device=device)
        spikes = torch.zeros_like(image_tensor)
        for i in range(image_tensor.shape[0]):
            for j in range(image_tensor.shape[1]):
                spk, mem = lif(image_tensor[i, j], mem)
                spikes[i, j] = spk
        return spikes, True
    except ModuleNotFoundError:
        mem = torch.zeros(1, device=device)
        spikes = torch.zeros_like(image_tensor)
        for i in range(image_tensor.shape[0]):
            for j in range(image_tensor.shape[1]):
                mem = beta * mem + image_tensor[i, j]
                spk = (mem >= threshold).float()
                mem = mem * (1.0 - spk)
                spikes[i, j] = spk
        return spikes, False


def parse_thresholds(args):
    if args.thresholds is None:
        return [args.threshold]
    values = []
    for raw in args.thresholds.split(","):
        raw = raw.strip()
        if raw:
            values.append(float(raw))
    if not values:
        raise ValueError("--thresholds was provided but no values were parsed")
    return values


def process_image(path, args, threshold, device, rng, results_dir):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    image_tensor = torch.tensor(image, dtype=torch.float32, device=device) / 255.0
    spikes, used_snntorch = run_lif_scan(image_tensor, args.beta, threshold, device)

    spike_np = spikes.detach().cpu().numpy()
    no_spike = rng.choice([0, 255], spike_np.shape, p=[1.0 - args.no_spike_on_prob, args.no_spike_on_prob])
    output = np.where(spike_np == 1, args.spike_value, no_spike).astype(np.uint8)

    threshold_tag = f"th{threshold:.3g}".replace(".", "p")
    out_path = results_dir / f"{path.stem}_TETCI_byeongjun_{threshold_tag}.png"
    cv2.imwrite(str(out_path), output)
    return {
        "input": str(path),
        "threshold": threshold,
        "output": str(out_path),
        "shape": list(output.shape),
        "spike_count": int(spike_np.sum()),
        "used_snntorch": used_snntorch,
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
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No image files found in {data_dir}")

    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    thresholds = parse_thresholds(args)
    summary = {
        "device": str(device),
        "data_dir": str(data_dir),
        "beta": args.beta,
        "thresholds": thresholds,
        "images": [],
    }
    for threshold in thresholds:
        for path in files:
            summary["images"].append(process_image(path, args, threshold, device, rng, results_dir))

    summary_path = results_dir / "TETCI_byeongjun_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
