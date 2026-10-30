"""Where does the jitter in evaluation videos come from? Compares, on a training demo (lift_pot ep 0):

  demo      - the recorded actions (what the robot was taught)
  sampling  - same observation, 5 predictions: how much the flow-matching noise changes the chunk
  within    - step-to-step jitter inside one predicted chunk (2nd difference, rad)
  boundary  - jump between the last executed action of one chunk and the first of the next (25 steps later)

    ~/kairos/.venv-train/bin/python ~/kairos/diag_smoothness.py
"""
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from action_smoothing import ChunkSmoother
from policy_server import CAMERAS, load_final

KAIROS = Path(__file__).resolve().parent
MODEL = KAIROS / "outputs" / "smolvla_baseline" / "final"
ARM = list(range(6)) + list(range(7, 13))  # joint dims, grippers excluded
N_EXEC = 25


def main():
    torch.manual_seed(0)
    policy, pre, post = load_final(MODEL, "cuda")
    policy.to("cuda").eval()
    ds = LeRobotDataset("local/robotwin_clean_2task", root=KAIROS / "data" / "robotwin_clean_2task")
    ep = ds.meta.episodes[0]
    lo, hi = ep["dataset_from_index"], ep["dataset_to_index"]
    task = ds[lo]["task"]

    def predict(i):
        f = ds[i]
        batch = {f"observation.images.{c}": f[f"observation.images.{c}"].unsqueeze(0) for c in CAMERAS}
        batch["observation.state"] = f["observation.state"].unsqueeze(0)
        batch["task"] = [task]
        with torch.inference_mode():
            return post(policy.predict_action_chunk(pre(batch))).squeeze(0).float().cpu().numpy()

    demo = np.stack([ds[i]["action"].numpy() for i in range(lo, hi)])[:, ARM]
    d2 = lambda a: np.abs(np.diff(a, n=2, axis=0)).mean()

    starts = list(range(lo, hi - 50, N_EXEC))
    chunks = [predict(i)[:, ARM] for i in starts]
    within = np.mean([d2(c[:N_EXEC]) for c in chunks])
    # executed action k=24 of chunk j vs action 0 of chunk j+1: should differ by one demo step
    boundary = np.mean([np.abs(chunks[j + 1][0] - chunks[j][N_EXEC - 1]).mean() for j in range(len(chunks) - 1)])
    demo_step = np.abs(np.diff(demo, axis=0)).mean()

    # Open-loop on the demo's own frames: the executed sequence (first N_EXEC of each chunk), raw vs --smooth.
    smoother = ChunkSmoother()
    raw_seq, smooth_seq = [], []
    for i in starts:
        torch.manual_seed(i)
        full = predict(i)
        raw_seq.append(full[:N_EXEC, ARM])
        smooth_seq.append(smoother.next(full, N_EXEC)[:N_EXEC, ARM])
    ref = demo[: len(starts) * N_EXEC]
    for name, seq in (("raw", np.concatenate(raw_seq)), ("smoothed", np.concatenate(smooth_seq))):
        seq = seq[: len(ref)]
        print(f"  executed {name:8s}: jitter {d2(seq):.5f} ({d2(seq) / d2(demo):.1f}x demo), "
              f"max step {np.abs(np.diff(seq, axis=0)).max():.4f}, error vs demo {np.abs(seq - ref).mean():.4f}")

    mid = lo + (hi - lo) // 2
    samples = np.stack([predict(mid)[:, ARM] for _ in range(5)])
    spread = samples.std(axis=0).mean()

    print(f"lift_pot demo episode 0, {hi - lo} frames, arm joints only, mean |rad|")
    print(f"  demo: step size {demo_step:.4f}, jitter (2nd diff) {d2(demo):.5f}")
    print(f"  policy: jitter inside a chunk {within:.5f}  ({within / d2(demo):.1f}x demo)")
    print(f"  policy: jump at chunk switch {boundary:.4f}  ({boundary / demo_step:.1f}x a normal demo step)")
    print(f"  policy: spread of 5 samples on the same frame {spread:.4f}  ({spread / demo_step:.1f}x a demo step)")


if __name__ == "__main__":
    main()
