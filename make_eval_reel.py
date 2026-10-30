"""Join one test set's episode videos into a single labelled reel.

    ~/kairos/.venv/bin/python ~/kairos/make_eval_reel.py --tag smolvla_baseline --name lift_pot_standard

Reads outputs/eval/<tag>/<name>.json (episodes in the order they ran) and their videos, and writes
outputs/eval/<tag>/reel_<name>.mp4 at 1280x720. Each episode gets a banner with its number, seed, result,
running score and instruction; failed episodes have a red banner.
"""
import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

KAIROS = Path(__file__).resolve().parent
W, H, BANNER = 1280, 720, 64


def banner(frame, lines, ok):
    frame[:BANNER] = (frame[:BANNER] * 0.25 + np.array((30, 110, 40) if ok else (150, 30, 30)) * 0.75).astype(np.uint8)
    for i, (text, scale) in enumerate(lines):
        cv2.putText(frame, text, (14, 26 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2 if i == 0 else 1,
                    cv2.LINE_AA)
    return frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="smolvla_baseline")
    p.add_argument("--name", required=True, help="e.g. lift_pot_standard")
    p.add_argument("--fps", type=int, default=15)
    a = p.parse_args()
    out_dir = KAIROS / "outputs" / "eval" / a.tag
    res = json.loads((out_dir / f"{a.name}.json").read_text())
    eps = res["episodes"]
    out = out_dir / f"reel_{a.name}.mp4"
    writer = imageio.get_writer(out, fps=a.fps, codec="libx264", quality=7, macro_block_size=8,
                                ffmpeg_params=["-threads", "1"])
    n_ok = 0
    for i, ep in enumerate(eps, 1):
        n_ok += ep["success"]
        if not ep.get("video"):  # run without --videos
            continue
        head = (f"{a.name.replace('_', ' ')}  |  episode {i}/{len(eps)}  |  seed {ep['seed']}  |  "
                f"{'SUCCESS' if ep['success'] else 'FAIL'} ({ep['actions']} actions)  |  score {n_ok}/{i}")
        lines = [(head, 0.62), (f'"{ep["instruction"]}"', 0.55)]
        for frame in imageio.get_reader(KAIROS / "outputs" / ep["video"]):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            writer.append_data(banner(frame, lines, ep["success"]))
    writer.close()
    print(f"{out}  ({len(eps)} episodes, {n_ok} successes)")


if __name__ == "__main__":
    main()
