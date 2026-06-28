import argparse

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--v2e-args", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if args.v2e_args:
        print(f"--output_width {width} --output_height {height}")
    else:
        print(f"width={width} height={height} fps={fps:.6g} frames={frames}")


if __name__ == "__main__":
    main()
