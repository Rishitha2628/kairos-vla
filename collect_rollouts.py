"""World-model Stage 2: record SmolVLA's own attempts in RoboTwin, including mistakes, for world-model training.

    ~/kairos/.venv/bin/python ~/kairos/thermal_guard.py -- ~/kairos/.venv/bin/python ~/kairos/collect_rollouts.py \
        --model-dir outputs/smolvla_baseline/final --task lift_pot --episodes 200

The demos only show perfect grasps, so a world model trained on them cannot tell a good plan from a slightly
wrong one. Here the smoothed baseline runs on NEW scenes (seeds from 200000, so the 100000+ test scenes stay
unseen), and each executed plan is, at random:
  plain  (40%)  SmolVLA's plan as is
  arm    (30%)  arm joints drift linearly to a random offset (0.02-0.1 rad) over the executed part
  grip   (30%)  gripper commands shifted 2-6 steps earlier or later
Seeds the scripted expert can't solve are skipped, as in step4_eval.py (same expert cache).

--play (world-model generalisation, Stage 1): cheap "play" data for a world model that must work at new pot
positions. The pot is anywhere in --variant wide (+-10 cm, +-45 deg, demo range included); no expert check
(no expert is used at all); plans are plain 20% / arm 60% / grip 20%, arm drifts 0.05-0.25 rad, so the hands
wander ~5-15 cm around SmolVLA's usual path and also reach pots that are away from where it aims. meta.npz
also stores each hand's distance to its handle (dist, for checking the world model only, never for training).

Per episode, outputs/rollouts/<task>/seed<N>/:
  cam_high.mp4, cam_left_wrist.mp4, cam_right_wrist.mp4   the policy's cameras before every action + the final frame
  meta.npz   state (T+1, 14), action (T, 14) as executed, success, chunk_start, chunk_mode, chunk_param
"""
import argparse
import json
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import step4_eval as ev
from action_smoothing import ARM, ChunkSmoother

KAIROS = ev.KAIROS
GRIP = [6, 13]
MODES = ("plain", "arm", "grip")


def perturb(chunk, n_exec, rng, weights=(0.4, 0.3, 0.3), arm_size=(0.02, 0.1)):
    """Returns (chunk, mode, param). Only the executed part matters: the next plan starts from where we are."""
    mode = rng.choices(MODES, weights=weights)[0]
    chunk = chunk.copy()
    if mode == "arm":
        size = rng.uniform(*arm_size)
        off = np.array([rng.gauss(0, 1) for _ in ARM]) * size
        ramp = np.minimum(np.arange(1, len(chunk) + 1) / n_exec, 1.0)[:, None]
        chunk[:, ARM] += ramp * off
        return chunk, mode, size
    if mode == "grip":
        k = rng.choice([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6])  # k > 0: the gripper acts k steps earlier
        idx = np.clip(np.arange(len(chunk)) + k, 0, len(chunk) - 1)
        chunk[:, GRIP] = chunk[idx][:, GRIP]
        return chunk, mode, float(k)
    return chunk, mode, 0.0


def hand_dist(env):
    """Distance (m) of each hand (TCP) to its handle, as in the success check."""
    tcp = [np.array(env.robot.get_left_tcp_pose()[:3]), np.array(env.robot.get_right_tcp_pose()[:3])]
    return [float(np.linalg.norm(t - np.array(env.pot.get_contact_point(i)[:3]))) for i, t in enumerate(tcp)]


def record_frame(obs, writers, states, env=None, dists=None):
    for name, cam in ev.CAMERAS.items():
        writers[name].append_data(obs["observation"][cam]["rgb"])
    states.append(np.asarray(obs["joint_action"]["vector"], dtype=np.float32))
    if dists is not None:
        dists.append(hand_dist(env))


def run_episode(env, args_env, sock, seed, episode_idx, instruction, step_lim, n_exec, out_dir, play=False):
    env.setup_demo(now_ep_num=episode_idx, seed=seed, is_test=True, **args_env)
    env.step_lim = step_lim
    env.set_instruction(instruction=instruction)
    ev.request(sock, {"cmd": "reset", "noise_seed": seed})
    rng = random.Random(f"{seed}:perturb")
    smoother = ChunkSmoother()
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    writers = {name: imageio.get_writer(tmp / f"{name}.mp4", fps=25, codec="libx264", quality=8,
                                        macro_block_size=8, ffmpeg_params=["-threads", "1"]) for name in ev.CAMERAS}
    states, actions, chunks = [], [], []
    dists = [] if play else None
    mix = dict(weights=(0.2, 0.6, 0.2), arm_size=(0.05, 0.25)) if play else {}
    try:
        obs = env.get_obs()
        while not (env.eval_success or env.take_action_cnt >= env.step_lim):
            reply = ev.request(sock, {
                "cmd": "act",
                "images": {k: obs["observation"][v]["rgb"] for k, v in ev.CAMERAS.items()},
                "state": np.asarray(obs["joint_action"]["vector"], dtype=np.float32),
                "task": instruction,
            })
            chunk = smoother.next(reply["actions"], n_exec)
            chunk, mode, param = perturb(chunk, n_exec, rng, **mix)
            chunks.append((len(actions), MODES.index(mode), param))
            for action in chunk[:n_exec]:
                record_frame(obs, writers, states, env, dists)
                env.take_action(action, action_type="qpos")
                actions.append(action)
                obs = env.get_obs()
                if env.eval_success or env.take_action_cnt >= env.step_lim:
                    break
        record_frame(obs, writers, states, env, dists)  # the frame after the last action
    finally:
        for w in writers.values():
            w.close()
    success = bool(env.eval_success)
    c = np.array(chunks)
    np.savez(tmp / "meta.npz", state=np.stack(states), action=np.stack(actions).astype(np.float32),
             success=success, seed=seed, instruction=instruction,
             chunk_start=c[:, 0].astype(int), chunk_mode=c[:, 1].astype(int), chunk_param=c[:, 2],
             **({"dist": np.array(dists, np.float32)} if play else {}))
    tmp.rename(out_dir)
    env.close_env(clear_cache=(episode_idx % 5 == 4))
    return success, len(actions), [MODES[int(m)] for m in c[:, 1]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--task", default="lift_pot")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed-start", type=int, default=200000)
    p.add_argument("--n-action-steps", type=int, default=25)
    p.add_argument("--step-limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=KAIROS / "outputs" / "rollouts")
    p.add_argument("--variant", default="standard")
    p.add_argument("--play", action="store_true", help="play data: big arm drifts, no expert check (see top)")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    model_dir = (KAIROS / a.model_dir).resolve() if not a.model_dir.is_absolute() else a.model_dir
    out_root = (a.out if a.out.is_absolute() else KAIROS / a.out) / a.task  # cwd is RoboTwin/ by now
    out_root.mkdir(parents=True, exist_ok=True)
    cache_path = KAIROS / "outputs" / "eval" / "expert_cache" / f"{a.task}_standard.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    with open(Path(ev.CONFIGS_PATH) / "_eval_step_limit.yml") as f:
        step_lim = a.step_limit or ev.yaml.safe_load(f)[a.task]

    ev.base.TASK = a.task
    args_env = ev.base.load_args()
    env = getattr(ev.importlib.import_module(f"envs.{a.task}"), a.task)()
    ev.hard_variants.apply(env, a.variant)
    done = sorted(d for d in out_root.glob("seed*") if not d.name.endswith(".tmp"))
    for d in out_root.glob("seed*.tmp"):  # a half-written episode from a crash
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    print(f"{a.task}: {len(done)}/{a.episodes} episodes already recorded -> {out_root}", flush=True)
    server, sock = ev.start_server(model_dir, a.device, out_root / "server.log")
    n_done, n_ok = len(done), sum(bool(np.load(d / "meta.npz")["success"]) for d in done)
    try:
        seed = a.seed_start
        while n_done < a.episodes and seed < a.seed_start + 20 * a.episodes:
            out_dir = out_root / f"seed{seed}"
            if out_dir.exists():
                seed += 1
                continue
            check = {"solvable": True, "info": {"{A}": "060_kitchenpot/base0"}} if a.play else \
                ev.expert_check(env, args_env, seed, cache, cache_path)
            if not check["solvable"]:
                seed += 1
                continue
            candidates = ev.generate_episode_descriptions(a.task, [check["info"]], 1, seed=seed)[0]["seen"]
            instruction = random.Random(f"{seed}:select").choice(candidates) if candidates else a.task
            ok, n_act, modes = run_episode(env, args_env, sock, seed, n_done, instruction, step_lim,
                                           a.n_action_steps, out_dir, play=a.play)
            n_done, n_ok = n_done + 1, n_ok + ok
            print(f"  [{n_done:3d}/{a.episodes}] seed {seed}: {'SUCCESS' if ok else 'fail   '} {n_act:3d} actions | "
                  f"plans {' '.join(m[0] for m in modes)} | running {n_ok}/{n_done}", flush=True)
            seed += 1
    finally:
        try:
            ev.request(sock, {"cmd": "close"})
        except OSError:
            pass
        server.wait(timeout=60)
    print(f"\n{a.task}: {n_ok}/{n_done} successes recorded in {out_root}")


if __name__ == "__main__":
    main()
