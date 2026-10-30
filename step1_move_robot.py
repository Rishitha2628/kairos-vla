"""Step 1: load the aloha-agilex two-armed robot in a RoboTwin scene and move it.

Run from anywhere:  ~/kairos/.venv/bin/python ~/kairos/step1_move_robot.py
Saves videos to ~/kairos/outputs/: step1_robot_moving.mp4 (1920x1080 view of both arms)
and step1_head_camera.mp4 (what the policy will see, 320x240).
Add --view to watch it live in a 3D window (close the window to exit).
"""
import os
import sys
import importlib

import numpy as np
import yaml
import imageio

ROBOTWIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RoboTwin")
os.chdir(ROBOTWIN)
sys.path.insert(0, ROBOTWIN)

from envs import CONFIGS_PATH  # noqa: E402

sys.path.insert(0, os.path.dirname(ROBOTWIN))
from sim_video import ShowcaseCamera, OUTPUTS  # noqa: E402

TASK, TASK_CONFIG, SEED = "lift_pot", "demo_clean", 0
VIEW = "--view" in sys.argv


def load_args():
    with open(os.path.join(CONFIGS_PATH, f"{TASK_CONFIG}.yml")) as f:
        args = yaml.safe_load(f)
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml")) as f:
        embodiments = yaml.safe_load(f)
    robot_file = embodiments[args["embodiment"][0]]["file_path"]
    with open(os.path.join(robot_file, "config.yml")) as f:
        robot_cfg = yaml.safe_load(f)
    # CuRobo is not installed; mplib is enough for joint-space control.
    robot_cfg["planner"] = "mplib_RRT"
    args.update(
        task_name=TASK,
        task_config=TASK_CONFIG,
        left_robot_file=robot_file,
        right_robot_file=robot_file,
        dual_arm_embodied=True,
        left_embodiment_config=robot_cfg,
        right_embodiment_config=robot_cfg,
        embodiment_name=args["embodiment"][0],
        render_freq=1 if VIEW else 0,
        need_plan=True,
        save_data=False,
    )
    return args


def main():
    env = getattr(importlib.import_module(f"envs.{TASK}"), TASK)()
    env.setup_demo(now_ep_num=0, seed=SEED, is_test=True, **load_args())
    env.step_lim = 10_000

    showcase = ShowcaseCamera(env)
    showcase.record_physics()
    home = np.array(env.get_obs()["joint_action"]["vector"])  # 14 = (6 joints + gripper) x 2 arms
    print(f"\nRobot loaded. Joint vector size: {home.size}, home pose: {np.round(home, 3)}")

    frames, moved = [], 0.0
    n_steps = 40
    for t in range(n_steps * (3 if VIEW else 1)):
        phase = np.sin(2 * np.pi * t / n_steps)
        target = home.copy()
        target[[0, 7]] += 0.5 * phase            # swing both shoulders (base yaw)
        target[[1, 8]] += 0.4 * abs(phase)       # lift both arms
        target[[2, 9]] -= 0.4 * abs(phase)
        target[[6, 13]] = 0.5 + 0.5 * phase      # open/close both grippers
        env.take_action(target, action_type="qpos")
        obs = env.get_obs()
        frames.append(obs["observation"]["head_camera"]["rgb"])
        moved = max(moved, np.abs(np.array(obs["joint_action"]["vector"]) - home).max())

    if VIEW:
        print("\nDone moving. Close the viewer window to exit.")
        while not env.viewer.closed:
            env.viewer.render()

    os.makedirs(OUTPUTS, exist_ok=True)
    head_path = os.path.join(OUTPUTS, "step1_head_camera.mp4")
    imageio.mimsave(head_path, frames, fps=10)
    showcase_path = showcase.save("step1_robot_moving.mp4")
    env.close_env()
    print(f"\nLargest joint change during the test: {moved:.3f} rad")
    print(f"Saved videos:\n  {showcase_path}  ({showcase.n_frames} frames, 1920x1080)\n  {head_path}")
    print("STEP 1 OK: the robot moves." if moved > 0.05 else "WARNING: robot did not move.")


if __name__ == "__main__":
    main()
