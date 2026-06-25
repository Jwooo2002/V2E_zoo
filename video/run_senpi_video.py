import argparse
import glob
import os

import cv2
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, os.path.abspath("senpi_ebi"))
    from senpi.sim.params import make_params
    from senpi.sim.simulator import EventSimulator

    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not frame_paths:
        raise RuntimeError(f"No PNG frames found in {args.frames_dir}")

    frames = []
    for path in frame_paths:
        frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError(f"Could not read {path}")
        frames.append(frame.astype(np.float32) + 1.0)

    stack = torch.from_numpy(np.stack(frames, axis=0)).to(args.device)
    params = make_params()
    params["device"] = torch.device(args.device)
    params["return_events"] = 1
    sim = EventSimulator(params)
    events, event_frames = sim.forward(stack)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    events_np = np.empty((0, 4), dtype=np.float32)
    if isinstance(events, torch.Tensor):
        events_np = events.detach().cpu().numpy()
    event_frames_np = np.empty((0,), dtype=np.float32)
    if isinstance(event_frames, torch.Tensor):
        event_frames_np = event_frames.detach().cpu().numpy()

    np.savez_compressed(
        args.output,
        events=events_np,
        event_frames=event_frames_np,
        frame_count=len(frame_paths),
        shape=np.array(stack.shape),
    )
    print(
        "SENPI saved",
        args.output,
        "events",
        events_np.shape,
        "event_frames",
        event_frames_np.shape,
    )


if __name__ == "__main__":
    main()
