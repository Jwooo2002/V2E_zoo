import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


MODEL_ALIASES = {
    # CLI에서 짧은 이름(dvs)과 원래 폴더/프로젝트 이름(dvs-voltmeter)을 둘 다 허용한다.
    "dvs": "dvs",
    "dvs-voltmeter": "dvs",
    "voltmeter": "dvs",
    "rpg": "rpg",
    "rpg_vid2e": "rpg",
    "vid2e": "rpg",
    "v2e": "v2e",
    "senpi": "senpi",
    "senpi_ebi": "senpi",
    "v2ce": "v2ce",
    "v2ce-toolbox": "v2ce",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the video-to-event models on every video in a directory."
    )
    # 공통 파라미터: 입력 디렉토리, 실행할 모델, 결과/임시 저장 위치를 제어한다.
    parser.add_argument("data_dir", help="입력 비디오들이 들어있는 디렉토리.")
    parser.add_argument("--results-dir", default="results", help="모델 결과와 33ms 이벤트 프레임 PNG를 저장할 디렉토리.")
    parser.add_argument("--models", default="all", help="실행할 모델. all 또는 콤마 목록: dvs,rpg,v2e,senpi,v2ce. 폴더 이름 별칭도 허용.")
    parser.add_argument("--tmp-root", default="/tmp/v2e_zoo_batch_py", help="프레임 추출 등 중간 입력을 만드는 임시 디렉토리.")
    parser.add_argument("--python", default=sys.executable, help="각 모델 subprocess를 실행할 Python 경로. 기본값은 현재 Python.")
    parser.add_argument("--preview-ms", type=float, default=33.0, help="저장할 이벤트 프레임 누적 시간(ms). 30fps 기준 33ms가 1프레임 정도.")
    parser.add_argument("--skip-existing", action="store_true", help="이미 결과 파일이 있으면 해당 모델 실행은 건너뛰고 프리뷰만 다시 생성.")
    parser.add_argument("--max-frame-num", type=int, default=0, help="지원되는 모델에서 처리할 최대 프레임 수. 0이면 전체 프레임.")

    # rpg_vid2e 파라미터: 이벤트 발생 민감도와 refractory period를 제어한다.
    parser.add_argument("--rpg-contrast-neg", type=float, default=0.2, help="rpg_vid2e OFF 이벤트 contrast threshold. 낮을수록 OFF 이벤트가 많아짐.")
    parser.add_argument("--rpg-contrast-pos", type=float, default=0.2, help="rpg_vid2e ON 이벤트 contrast threshold. 낮을수록 ON 이벤트가 많아짐.")
    parser.add_argument("--rpg-refractory-ns", type=int, default=0, help="rpg_vid2e 픽셀별 refractory period(ns). 커질수록 같은 픽셀 연속 이벤트가 줄어듦.")

    # v2e 파라미터: threshold/noise/filter로 이벤트 수와 노이즈 성향을 조절한다.
    parser.add_argument("--v2e-pos-thres", type=float, default=0.2, help="v2e ON 이벤트 threshold. 낮을수록 ON 이벤트 증가.")
    parser.add_argument("--v2e-neg-thres", type=float, default=0.2, help="v2e OFF 이벤트 threshold. 낮을수록 OFF 이벤트 증가.")
    parser.add_argument("--v2e-sigma-thres", type=float, default=0.03, help="v2e threshold 랜덤 분산. 높이면 픽셀별 threshold 편차가 커짐.")
    parser.add_argument("--v2e-cutoff-hz", type=float, default=300, help="v2e photoreceptor low-pass cutoff(Hz). 낮을수록 빠른 변화가 더 부드러워짐.")
    parser.add_argument("--v2e-leak-rate-hz", type=float, default=0.01, help="v2e leak event rate(Hz). 높이면 시간 경과에 따른 누설 이벤트 증가.")
    parser.add_argument("--v2e-shot-noise-rate-hz", type=float, default=0.001, help="v2e shot noise rate(Hz). 높이면 랜덤 노이즈 이벤트 증가.")
    parser.add_argument("--v2e-refractory-period", type=float, default=0.0005, help="v2e refractory period(초). 커질수록 같은 픽셀 연속 이벤트가 줄어듦.")
    parser.add_argument("--v2e-disable-slomo", action=argparse.BooleanOptionalAction, default=True, help="v2e의 slomo 보간 사용 여부. 기본은 비활성화.")
    parser.add_argument("--v2e-extra-arg", action="append", default=[], help="v2e.py에 그대로 넘길 추가 인자. 여러 번 지정 가능.")

    # V2CE 파라미터: 입력 resize/crop 방식, 출력 이벤트 스케일, batch size를 제어한다.
    parser.add_argument("--v2ce-infer-type", default="pano", choices=["center", "pano"], help="V2CE 추론 방식. square 영상은 pano가 안전했고 center는 crop 크기 문제 가능.")
    parser.add_argument("--v2ce-height", type=int, default=260, help="V2CE 입력 resize 높이.")
    parser.add_argument("--v2ce-width", type=int, default=346, help="V2CE 입력 resize 너비.")
    parser.add_argument("--v2ce-ceil", type=int, default=10, help="V2CE 이벤트 프레임 intensity 상한/스케일 계수.")
    parser.add_argument("--v2ce-batch-size", type=int, default=1, help="V2CE stage1 batch size. GPU 메모리 부족 시 낮게 유지.")
    parser.add_argument("--v2ce-stage2-batch-size", type=int, default=24, help="V2CE stage2 batch size. GPU 메모리에 맞춰 조절.")
    parser.add_argument("--v2ce-extra-arg", action="append", default=[], help="v2ce.py에 그대로 넘길 추가 인자. 여러 번 지정 가능.")

    # SENPI 파라미터: 실행 디바이스만 선택한다.
    parser.add_argument("--senpi-device", default="auto", choices=["auto", "cuda", "cpu"], help="SENPI 실행 디바이스. auto는 CUDA 가능 시 GPU 사용.")
    return parser.parse_args()


def selected_models(models_arg):
    if models_arg == "all":
        return {"dvs", "rpg", "v2e", "senpi", "v2ce"}
    models = set()
    for raw in models_arg.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in MODEL_ALIASES:
            raise ValueError(f"Unknown model '{raw}'")
        models.add(MODEL_ALIASES[key])
    return models


def run(cmd, cwd=None):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def video_metadata(video):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, frames


def video_files(data_dir):
    patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
    files = []
    for pattern in patterns:
        files.extend(Path(data_dir).glob(pattern))
    return sorted(files)


def render_events(x, y, p, width, height, out_path):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if len(x) > 0:
        x = np.asarray(x, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)
        p = np.asarray(p)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x = x[valid]
        y = y[valid]
        p = p[valid]
        on = p > 0
        off = ~on
        img[y[off], x[off], 0] = 255
        img[y[on], x[on], 2] = 255
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(str(out_path), img)


def read_txt_events_window(path, time_limit, time_scale, delimiter=None):
    xs, ys, ps = [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(delimiter)
            if len(parts) < 4:
                parts = line.replace(",", " ").split()
            if len(parts) < 4:
                continue
            t = float(parts[0]) * time_scale
            if t >= time_limit:
                break
            xs.append(int(float(parts[1])))
            ys.append(int(float(parts[2])))
            ps.append(int(float(parts[3])))
    return xs, ys, ps


def preview_dvs(results_dir, name, width, height, preview_s):
    path = results_dir / f"DVS-Voltmeter_{name}.txt"
    if path.exists():
        x, y, p = read_txt_events_window(path, preview_s, 1e-6)
        render_events(x, y, p, width, height, results_dir / f"DVS-Voltmeter_{name}_event_frame_33ms.png")


def preview_rpg(results_dir, name, width, height):
    files = sorted((results_dir / f"rpg_vid2e_{name}" / name).glob("*.npz"))
    if files:
        z = np.load(files[0])
        render_events(z["x"], z["y"], z["p"], width, height, results_dir / f"rpg_vid2e_{name}_event_frame_33ms.png")


def preview_v2e(results_dir, name, width, height, preview_s):
    path = results_dir / f"v2e_{name}" / f"v2e_{name}.txt"
    if path.exists():
        x, y, p = read_txt_events_window(path, preview_s, 1.0, delimiter=",")
        render_events(x, y, p, width, height, results_dir / f"v2e_{name}_event_frame_33ms.png")


def preview_senpi(results_dir, name, preview_s):
    path = results_dir / f"senpi_ebi_{name}.npz"
    if not path.exists():
        return
    z = np.load(path)
    frames = z["event_frames"]
    if frames.ndim == 3 and frames.shape[0] > 0:
        frame = frames[0]
        img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        img[frame < 0, 0] = 255
        img[frame > 0, 2] = 255
        cv2.imwrite(str(results_dir / f"senpi_ebi_{name}_event_frame_33ms.png"), img)
        return
    events = z["events"]
    if events.size:
        limit_us = preview_s * 1e6
        selected = events[events[:, 0] < limit_us]
        width = int(z["shape"][2])
        height = int(z["shape"][1])
        render_events(selected[:, 1], selected[:, 2], selected[:, 3], width, height, results_dir / f"senpi_ebi_{name}_event_frame_33ms.png")


def preview_v2ce(results_dir, name, preview_s):
    matches = sorted((results_dir / f"V2CE-Toolbox_{name}").glob("*-events.npz"))
    if not matches:
        return
    stream = np.load(matches[0])["event_stream"]
    selected = stream[stream["timestamp"] < int(preview_s * 1e6)]
    if selected.size == 0:
        selected = stream[:0]
    width = int(selected["x"].max() + 1) if selected.size else 260
    height = int(selected["y"].max() + 1) if selected.size else 260
    width = max(width, 260)
    height = max(height, 260)
    render_events(
        selected["x"],
        selected["y"],
        selected["polarity"],
        width,
        height,
        results_dir / f"V2CE-Toolbox_{name}_event_frame_33ms.png",
    )


def write_previews(results_dir, name, models, width, height, preview_ms):
    preview_s = preview_ms / 1000.0
    if "dvs" in models:
        preview_dvs(results_dir, name, width, height, preview_s)
    if "rpg" in models:
        preview_rpg(results_dir, name, width, height)
    if "v2e" in models:
        preview_v2e(results_dir, name, width, height, preview_s)
    if "senpi" in models:
        preview_senpi(results_dir, name, preview_s)
    if "v2ce" in models:
        preview_v2ce(results_dir, name, preview_s)


def prepare_inputs(args, root_dir, video, name, work_root):
    rpg_input = work_root / "rpg_input"
    dvs_input = work_root / "dvs_input"
    shutil.rmtree(work_root, ignore_errors=True)
    rpg_input.mkdir(parents=True, exist_ok=True)
    dvs_input.mkdir(parents=True, exist_ok=True)
    run(
        [
            args.python,
            str(root_dir / "prepare_video_inputs.py"),
            str(video),
            "--name",
            name,
            "--rpg-root",
            str(rpg_input),
            "--dvs-root",
            str(dvs_input),
        ]
    )
    return rpg_input, dvs_input


def run_dvs(args, root_dir, results_dir, name, dvs_input):
    out = results_dir / f"DVS-Voltmeter_{name}.txt"
    if args.skip_existing and out.exists():
        return
    run(
        [
            args.python,
            "main.py",
            "--input_dir",
            str(dvs_input),
            "--output_dir",
            str(results_dir),
        ],
        cwd=root_dir / "DVS-Voltmeter",
    )
    raw = results_dir / f"{name}.txt"
    if raw.exists():
        raw.replace(out)


def run_rpg(args, root_dir, results_dir, name, rpg_input):
    out_dir = results_dir / f"rpg_vid2e_{name}"
    if args.skip_existing and list((out_dir / name).glob("*.npz")):
        return
    run(
        [
            args.python,
            "esim_torch/scripts/generate_events.py",
            "-i",
            str(rpg_input),
            "-o",
            str(out_dir),
            "-cn",
            str(args.rpg_contrast_neg),
            "-cp",
            str(args.rpg_contrast_pos),
            "-rp",
            str(args.rpg_refractory_ns),
        ],
        cwd=root_dir / "rpg_vid2e",
    )


def run_v2e(args, root_dir, results_dir, video, name, width, height):
    out_dir = results_dir / f"v2e_{name}"
    if args.skip_existing and (out_dir / f"v2e_{name}.txt").exists():
        return
    cmd = [
        args.python,
        "v2e.py",
        "-i",
        str(video),
        "-o",
        str(out_dir),
        "--overwrite",
        "--no_preview",
        "--output_width",
        str(width),
        "--output_height",
        str(height),
        "--pos_thres",
        str(args.v2e_pos_thres),
        "--neg_thres",
        str(args.v2e_neg_thres),
        "--sigma_thres",
        str(args.v2e_sigma_thres),
        "--cutoff_hz",
        str(args.v2e_cutoff_hz),
        "--leak_rate_hz",
        str(args.v2e_leak_rate_hz),
        "--shot_noise_rate_hz",
        str(args.v2e_shot_noise_rate_hz),
        "--refractory_period",
        str(args.v2e_refractory_period),
        "--dvs_text",
        f"v2e_{name}.txt",
        "--dvs_h5",
        f"v2e_{name}.h5",
        "--dvs_vid",
        f"v2e_{name}.mp4",
        "--vid_orig",
        "None",
        "--vid_slomo",
        "None",
    ]
    if args.v2e_disable_slomo:
        cmd.append("--disable_slomo")
    cmd.extend(args.v2e_extra_arg)
    run(cmd, cwd=root_dir / "v2e")


def run_senpi(args, root_dir, results_dir, name, dvs_input):
    out = results_dir / f"senpi_ebi_{name}.npz"
    if args.skip_existing and out.exists():
        return
    cmd = [
        args.python,
        str(root_dir / "run_senpi_video.py"),
        "--frames-dir",
        str(dvs_input / name),
        "--output",
        str(out),
    ]
    if args.senpi_device != "auto":
        cmd.extend(["--device", args.senpi_device])
    run(cmd, cwd=root_dir)


def run_v2ce(args, root_dir, results_dir, video, name):
    out_dir = results_dir / f"V2CE-Toolbox_{name}"
    if args.skip_existing and list(out_dir.glob("*-events.npz")):
        return
    weight = root_dir / "V2CE-Toolbox" / "weights" / "v2ce_3d.pt"
    if not weight.exists():
        print(f"missing {weight}; skipping V2CE for {name}", file=sys.stderr)
        return
    cmd = [
        args.python,
        "v2ce.py",
        "-i",
        str(video),
        "-o",
        str(out_dir),
        "--out_name_suffix",
        name,
        "-t",
        args.v2ce_infer_type,
        "--height",
        str(args.v2ce_height),
        "--width",
        str(args.v2ce_width),
        "--ceil",
        str(args.v2ce_ceil),
        "-b",
        str(args.v2ce_batch_size),
        "--stage2_batch_size",
        str(args.v2ce_stage2_batch_size),
        "--write_event_frame_video",
        "true",
        "-l",
        "info",
    ]
    if args.max_frame_num > 0:
        cmd.extend(["--max_frame_num", str(args.max_frame_num)])
    cmd.extend(args.v2ce_extra_arg)
    run(cmd, cwd=root_dir / "V2CE-Toolbox")


def process_video(args, root_dir, results_dir, models, video):
    name = video.stem
    width, height, fps, frames = video_metadata(video)
    print(f"==> {name}: {width}x{height}, fps={fps:.6g}, frames={frames}", flush=True)
    work_root = Path(args.tmp_root) / name
    rpg_input, dvs_input = prepare_inputs(args, root_dir, video, name, work_root)

    if "dvs" in models:
        print(f"==> DVS-Voltmeter: {name}", flush=True)
        run_dvs(args, root_dir, results_dir, name, dvs_input)
    if "rpg" in models:
        print(f"==> rpg_vid2e: {name}", flush=True)
        run_rpg(args, root_dir, results_dir, name, rpg_input)
    if "v2e" in models:
        print(f"==> v2e: {name}", flush=True)
        run_v2e(args, root_dir, results_dir, video, name, width, height)
    if "senpi" in models:
        print(f"==> SENPI: {name}", flush=True)
        run_senpi(args, root_dir, results_dir, name, dvs_input)
    if "v2ce" in models:
        print(f"==> V2CE-Toolbox: {name}", flush=True)
        run_v2ce(args, root_dir, results_dir, video, name)

    write_previews(results_dir, name, models, width, height, args.preview_ms)


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    models = selected_models(args.models)
    files = video_files(data_dir)
    if not files:
        raise RuntimeError(f"No video files found in {data_dir}")
    for video in files:
        process_video(args, root_dir, results_dir, models, video)
    print(f"done: {results_dir}")


if __name__ == "__main__":
    main()
