import argparse
import os

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--name", required=True)
    parser.add_argument("--rpg-root", required=True)
    parser.add_argument("--dvs-root", required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rpg_dir = os.path.join(args.rpg_root, args.name)
    rpg_img_dir = os.path.join(rpg_dir, "imgs")
    dvs_dir = os.path.join(args.dvs_root, args.name)
    os.makedirs(rpg_img_dir, exist_ok=True)
    os.makedirs(dvs_dir, exist_ok=True)

    timestamps = []
    info = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timestamp_s = frame_idx / fps
        timestamp_us = int(round(timestamp_s * 1e6))

        filename = f"{frame_idx:010d}.png"
        rpg_path = os.path.join(rpg_img_dir, filename)
        dvs_path = os.path.join(dvs_dir, filename)
        cv2.imwrite(rpg_path, gray)
        cv2.imwrite(dvs_path, gray)
        timestamps.append(f"{timestamp_s:.9f}")
        info.append(f"{dvs_path} {timestamp_us}")
        frame_idx += 1

    cap.release()
    with open(os.path.join(rpg_dir, "timestamps.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(timestamps) + "\n")
    with open(os.path.join(dvs_dir, "info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(info) + "\n")

    print(f"extracted {frame_idx} frames at {fps:.6g} fps")


if __name__ == "__main__":
    main()
