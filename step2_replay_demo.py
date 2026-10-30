"""Step 2: replay a recorded demo in the simulator and check the robot repeats it.

Usage:  ~/kairos/.venv/bin/python ~/kairos/step2_replay_demo.py [task] [episode]
        (defaults: lift_pot 0)

Resets the sim with the demo's seed, sends the recorded joint actions one by one, and compares
the simulated joints and head-camera images with the recording. Saves to ~/kairos/outputs/:
  step2_<task>_ep<N>_compare.mp4   recorded head camera (left) vs replay (right)
  step2_<task>_ep<N>_replay.mp4    1920x1080 view of both arms during the replay
"""
import os
import sys
import importlib

import h5py
import imageio
import numpy as np

KAIROS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KAIROS)
import step1_move_robot as base  # noqa: E402  (sets up RoboTwin paths, cwd)
from sim_video import ShowcaseCamera, OUTPUTS  # noqa: E402

sys.path.insert(0, os.path.join(base.ROBOTWIN, "data"))
from decode_image_bit import decode_image_bit  # noqa: E402

TASK = sys.argv[1] if len(sys.argv) > 1 else "lift_pot"
EPISODE = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DEMO_DIR = os.path.join(base.ROBOTWIN, "data", "demo_clean", TASK, "aloha_agilex")


def load_demo():
    with open(os.path.join(DEMO_DIR, "seed.txt")) as f:
        seed = int(f.read().split()[EPISODE])
    with h5py.File(os.path.join(DEMO_DIR, "data", f"episode_{EPISODE:07d}.hdf5"), "r") as f:
        def joints(group):  # 14 = left arm(6) + left gripper + right arm(6) + right gripper
            return np.concatenate(
                [f[f"{group}/{k}"][:] for k in
                 ("left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states")],
                axis=1,
            )
        head = [decode_image_bit(b) for b in f["vision/cam_head/colors"][:]]
        return seed, joints("state"), joints("action"), head


def main():
    seed, states, actions, rec_head = load_demo()
    print(f"{TASK} episode {EPISODE}: seed {seed}, {len(actions)} steps")

    base.TASK = TASK
    env = getattr(importlib.import_module(f"envs.{TASK}"), TASK)()
    env.setup_demo(now_ep_num=EPISODE, seed=seed, **base.load_args())
    env.step_lim = 10_000
    showcase = ShowcaseCamera(env, fps=15)

    obs = env.get_obs()
    sim_states = [np.array(obs["joint_action"]["vector"])]
    sim_head = [obs["observation"]["head_camera"]["rgb"]]
    showcase.capture()
    success_step = None
    for t, action in enumerate(actions):
        env.take_action(action, action_type="qpos")
        obs = env.get_obs()
        sim_states.append(np.array(obs["joint_action"]["vector"]))
        sim_head.append(obs["observation"]["head_camera"]["rgb"])
        showcase.capture()
        if env.eval_success:
            success_step = t + 1
            break  # take_action ignores further actions once the task is solved
    success = bool(env.check_success())
    env.close_env()

    # Compare: sim state after action t vs recorded state t+1 (recorded action[t] == state[t+1]).
    n = len(sim_states) - 1
    err = np.abs(np.array(sim_states[1:]) - states[1:n + 1])
    arm_cols = [*range(0, 6), *range(7, 13)]
    start_err = np.abs(sim_states[0] - states[0]).max()
    img_err = np.mean([np.abs(sim_head[i].astype(int) - rec_head[i]).mean() for i in range(n + 1)])

    tag = f"step2_{TASK}_ep{EPISODE}"
    up = lambda im: np.kron(im, np.ones((3, 3, 1), dtype=np.uint8))  # 320x240 -> 960x720
    compare = [np.concatenate([up(rec_head[i]), up(sim_head[i])], axis=1) for i in range(n + 1)]
    compare_path = os.path.join(OUTPUTS, f"{tag}_compare.mp4")
    os.makedirs(OUTPUTS, exist_ok=True)
    imageio.mimsave(compare_path, compare, fps=15, quality=8, macro_block_size=8)
    replay_path = showcase.save(f"{tag}_replay.mp4")

    print("\n================ Replay check ================")
    print(f"Steps replayed:            {n} / {len(actions)}" + (f" (task solved at step {success_step})" if success_step else ""))
    print(f"Start pose difference:     {start_err:.4f} rad")
    print(f"Arm joint error (mean/max): {err[:, arm_cols].mean():.4f} / {err[:, arm_cols].max():.4f} rad")
    print(f"Gripper error (mean/max):  {err[:, [6, 13]].mean():.4f} / {err[:, [6, 13]].max():.4f}")
    print(f"Head camera pixel diff:    {img_err:.1f} / 255 (mean)")
    print(f"Task success in replay:    {success}")
    print(f"Videos:\n  {compare_path}\n  {replay_path}")
    ok = success and err[:, arm_cols].max() < 0.05
    print("STEP 2 OK: the replay matches the recording." if ok else "WARNING: replay differs from the recording.")


if __name__ == "__main__":
    main()
