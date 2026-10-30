"""World-model Stage 2: encode the recorded attempts (collect_rollouts.py) with the frozen V-JEPA 2 encoder.

    nice -n 19 ~/kairos/.venv/bin/python ~/kairos/thermal_guard.py -- \
        ~/kairos/.venv-train/bin/python ~/kairos/encode_rollouts.py --task lift_pot

Same encoder and pooling as step5_vjepa_embeddings.py (4x4 grid x 1408 per camera). Only every --stride-th
frame is encoded: the world model looks 8 steps ahead, so frames 0, 4, 8, ... give every start and target it
needs, at a quarter of the time (~1 h instead of ~4 h for 200 episodes x 3 cameras).

Output: outputs/rollouts/<task>/vjepa2_stride<S>.pt with, per encoded frame, seed, frame index and one grid per
camera (fp16). States, actions and success stay in each episode's meta.npz. Resumable: finished episodes are
kept in a part file and skipped on restart.
"""
import argparse
import time
from pathlib import Path

import torch
from torchcodec.decoders import VideoDecoder

from step5_vjepa_embeddings import KAIROS, encode, load_encoder

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="lift_pot")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--root", type=Path, default=KAIROS / "outputs" / "rollouts", help="e.g. outputs/play")
    a = p.parse_args()
    root = (a.root if a.root.is_absolute() else KAIROS / a.root) / a.task
    out = root / f"vjepa2_stride{a.stride}.pt"
    part = out.with_suffix(".part.pt")
    episodes = sorted(d for d in root.glob("seed*") if d.is_dir() and not d.name.endswith(".tmp"))
    done = torch.load(part, weights_only=True) if part.exists() else {"seeds": []}
    todo = [d for d in episodes if int(d.name[4:]) not in set(done["seeds"])]
    print(f"{len(episodes)} episodes, {len(episodes) - len(todo)} already encoded, stride {a.stride}", flush=True)

    dtype = torch.float16
    encoder = load_encoder(a.device, dtype)
    t0, n_frames = time.time(), 0
    for i, d in enumerate(todo):
        grids = {}
        for cam in CAMERAS:
            video = VideoDecoder(str(d / f"{cam}.mp4"))
            imgs = video.get_frames_at(list(range(0, len(video), a.stride))).data.float().div(255)  # (n, 3, H, W)
            grids[cam] = torch.cat([encode(encoder, imgs[j:j + a.batch_size], a.device, dtype)[1]
                                    for j in range(0, len(imgs), a.batch_size)])
        n = len(grids[CAMERAS[0]])
        assert all(len(g) == n for g in grids.values()), d
        seed = int(d.name[4:])
        done["seeds"].append(seed)
        done.setdefault("seed", []).append(torch.full((n,), seed))
        done.setdefault("frame", []).append(torch.arange(n) * a.stride)
        for cam in CAMERAS:
            done.setdefault(cam, []).append(grids[cam])
        n_frames += 3 * n
        if i % 10 == 9 or i == len(todo) - 1:
            torch.save(done, part)
            rate = n_frames / (time.time() - t0)
            left = len(todo) - i - 1
            print(f"{time.strftime('%H:%M:%S')} {len(done['seeds'])}/{len(episodes)} episodes, {rate:.1f} frames/s, "
                  f"~{left * 3 * n / rate / 60:.0f} min left", flush=True)

    result = {"seed": torch.cat(done["seed"]), "frame": torch.cat(done["frame"]), "stride": a.stride,
              "cameras": CAMERAS, **{cam: torch.cat(done[cam]) for cam in CAMERAS}}
    torch.save(result, out)
    part.unlink()
    print(f"saved {out}: {len(result['seed'])} frames x {len(CAMERAS)} cameras, {out.stat().st_size / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
