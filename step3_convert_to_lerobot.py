"""Step 3a: convert RoboTwin clean demos (lift_pot, beat_block_hammer) into one LeRobot dataset.

    ~/kairos/.venv-train/bin/python ~/kairos/step3_convert_to_lerobot.py

Per frame:
  observation.images.cam_high / cam_left_wrist / cam_right_wrist   320x240 RGB (stored as video)
  observation.state   14 joints = left arm(6), left gripper, right arm(6), right gripper
  action              14 joints commanded for the next frame (recorded action[t] == state[t+1])
  task                one natural-language instruction per episode

Frames are saved every 15 physics steps at 250 Hz (= 16.7 Hz); LeRobot needs an integer fps, so 15 is
used. That only affects timestamps. All ~100 "seen" phrasings per episode go to
meta/instructions.json for phrasing augmentation during training.

Afterwards the image mean/std in meta/stats.json are recomputed per pixel (same LeRobot stats bug
found in the LIBERO project, see vla-libero-project/fix_image_stats.py).
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

KAIROS = Path(__file__).resolve().parent
DEMOS = KAIROS / "RoboTwin" / "data" / "demo_clean"
sys.path.insert(0, str(KAIROS / "RoboTwin" / "data"))
from decode_image_bit import decode_image_bit  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

TASKS = ["lift_pot", "beat_block_hammer"]
REPO_ID = "local/robotwin_clean_2task"
FPS = 15
CAMERAS = {"cam_high": "cam_head", "cam_left_wrist": "cam_left_wrist", "cam_right_wrist": "cam_right_wrist"}
JOINT_NAMES = (
    [f"left_joint{i}" for i in range(6)] + ["left_gripper"] + [f"right_joint{i}" for i in range(6)] + ["right_gripper"]
)


def joints(f, group):
    return np.concatenate(
        [f[f"{group}/{k}"][:] for k in
         ("left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states")],
        axis=1,
    ).astype(np.float32)


def true_image_stats(dataset, keys, n_samples=1500):
    """Per-channel mean/std over pixels, from an evenly spaced sample of frames."""
    idx = np.linspace(0, len(dataset) - 1, min(n_samples, len(dataset))).astype(int)
    sums = {k: torch.zeros(3, dtype=torch.float64) for k in keys}
    sq = {k: torch.zeros(3, dtype=torch.float64) for k in keys}
    count = 0
    for i in idx:
        item = dataset[int(i)]
        for k in keys:
            img = item[k].double()  # (3, H, W) in [0, 1]
            sums[k] += img.sum(dim=(1, 2))
            sq[k] += (img ** 2).sum(dim=(1, 2))
        count += img.shape[1] * img.shape[2]
    out = {}
    for k in keys:
        mean = sums[k] / count
        out[k] = (mean.numpy(), torch.sqrt((sq[k] / count - mean ** 2).clamp_min(0)).numpy())
    return out


def fix_image_stats(root):
    dataset = LeRobotDataset(REPO_ID, root=root)
    keys = [k for k in dataset.meta.stats if k.startswith("observation.images.")]
    stats_path = root / "meta" / "stats.json"
    on_disk = json.loads(stats_path.read_text())
    shutil.copy(stats_path, stats_path.with_suffix(".json.orig"))
    for k, (mean, std) in true_image_stats(dataset, keys).items():
        shape = np.array(on_disk[k]["std"]).shape
        print(f"  {k}: std {np.array(on_disk[k]['std']).ravel().round(4)} -> {std.round(4)}, mean {mean.round(4)}")
        on_disk[k]["mean"] = mean.reshape(shape).tolist()
        on_disk[k]["std"] = std.reshape(shape).tolist()
    stats_path.write_text(json.dumps(on_disk, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=KAIROS / "data" / "robotwin_clean_2task")
    ap.add_argument("--episodes-per-task", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.root.exists():
        shutil.rmtree(args.root)
    image = {"dtype": "video", "shape": (240, 320, 3), "names": ["height", "width", "channel"]}
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID, fps=FPS, root=args.root, robot_type="aloha-agilex",
        features={
            **{f"observation.images.{k}": image for k in CAMERAS},
            "observation.state": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
            "action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
        },
        image_writer_threads=4,
    )

    instructions, episode_index = [], 0
    for task in TASKS:
        src = DEMOS / task / "aloha_agilex"
        for ep in range(args.episodes_per_task):
            seen = json.loads((src / "instruction" / f"episode_{ep:07d}.json").read_text())["seen"]
            instruction = rng.choice(seen)
            with h5py.File(src / "data" / f"episode_{ep:07d}.hdf5", "r") as f:
                state, action = joints(f, "state"), joints(f, "action")
                frames = {k: f[f"vision/{v}/colors"][:] for k, v in CAMERAS.items()}
            for t in range(len(action)):
                dataset.add_frame({
                    **{f"observation.images.{k}": decode_image_bit(frames[k][t]) for k in CAMERAS},
                    "observation.state": state[t],
                    "action": action[t],
                    "task": instruction,
                })
            dataset.save_episode()
            instructions.append({"episode_index": episode_index, "robotwin_task": task, "source_episode": ep,
                                 "instruction": instruction, "seen": seen})
            episode_index += 1
            print(f"[{episode_index:3d}] {task} ep{ep:02d}: {len(action)} frames  \"{instruction}\"", flush=True)

    dataset.finalize()
    (args.root / "meta" / "instructions.json").write_text(json.dumps(instructions, indent=1))
    print(f"\nFinalized {episode_index} episodes. Recomputing image stats:")
    fix_image_stats(args.root)
    print(f"Done: {args.root}")


if __name__ == "__main__":
    main()
