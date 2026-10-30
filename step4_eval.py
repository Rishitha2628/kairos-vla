"""Step 4 / 8: evaluate a trained SmolVLA policy in RoboTwin.

    ~/kairos/.venv/bin/python ~/kairos/thermal_guard.py -- ~/kairos/.venv/bin/python ~/kairos/step4_eval.py \
        --model-dir outputs/smolvla_baseline/final --task lift_pot --variant standard

Follows RoboTwin's official protocol (scripts/eval_policy_xpolicylab.py):
  * seeds from 100000 upward (training demos used seeds 0-111);
  * before each seed the scripted expert must solve it, otherwise the seed is skipped. The expert runs on
    mplib (mplib_expert.py) since CuRobo isn't installed; results are cached per task/variant in
    outputs/eval/expert_cache/ so every model is tested on the same seeds;
  * the instruction is a RoboTwin "seen" phrasing generated for that episode's objects;
  * an episode ends on success or after the task's step limit (400 actions for both tasks).
--variant hard uses hard_variants.py (unseen pots, wider placement); --variant position: the training pot
at poses outside the demos' range.
--smooth filters each chunk and crossfades chunk switches (action_smoothing.py); results go to <tag>_smooth.
Every episode also records `jitter`: mean |2nd difference| of the executed arm-joint actions (rad).
--noise-seed N seeds the policy's sampling noise per episode (reproducible; a different N = a fresh draw on
the same scenes); results go to <tag>_ns<N>.
--n-samples K (diversity check): at every decision the policy draws K plans for the same observation; the
first is executed as usual, and how far apart the K plans are is saved to <task>_<variant>_plans.jsonl.
--planner DIR (with --n-samples K): the world model in DIR picks which of the K plans to execute
(planner.py); results go to <tag>_planner, the plans log also records each plan's score and the choice.

The policy runs in the Python 3.12 training env: this script starts policy_server.py itself and talks to
it over a local socket (policy_link.py). Each prediction is a 50-action chunk; the first
--n-action-steps of it are executed before asking again.

Outputs (resumable: finished seeds are skipped on rerun):
  outputs/eval/<tag>/<task>_<variant>.json        per-episode results + success rate
  outputs/eval/<tag>/videos/<task>_<variant>_seed<N>_<ok|fail>.mp4   only with --videos:
      every episode at 640x360; the first --hd-videos episodes at 1920x1080
"""
import argparse
import importlib
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

KAIROS = Path(__file__).resolve().parent
sys.path.insert(0, str(KAIROS))
import step1_move_robot as base  # noqa: E402  (sets cwd + sys.path for RoboTwin)
import hard_variants  # noqa: E402
import policy_link  # noqa: E402
from action_smoothing import ARM, ChunkSmoother  # noqa: E402
from mplib_expert import enable_mplib_expert  # noqa: E402
from sim_video import ShowcaseCamera  # noqa: E402
from envs import CONFIGS_PATH  # noqa: E402
from description.utils.generate_episode_instructions import generate_episode_descriptions  # noqa: E402

CAMERAS = {"cam_high": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--task", default="lift_pot")
    p.add_argument("--variant", default="standard", choices=["standard", "hard", "position"])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed-start", type=int, default=100000)
    p.add_argument("--n-action-steps", type=int, default=25)
    p.add_argument("--videos", action="store_true", help="record every episode (off by default: it slows the eval)")
    p.add_argument("--hd-videos", type=int, default=2)
    p.add_argument("--tag", default=None, help="results folder name (default: the model's run folder)")
    p.add_argument("--device", default="cuda", help="policy device")
    p.add_argument("--step-limit", type=int, default=None, help="override RoboTwin's limit (smoke tests)")
    p.add_argument("--smooth", action="store_true", help="smooth chunks + crossfade chunk switches")
    p.add_argument("--noise-seed", type=int, default=None, help="seed the policy's sampling noise per episode")
    p.add_argument("--n-samples", type=int, default=1, help="draw this many plans per decision and log their spread")
    p.add_argument("--planner", type=Path, default=None, help="world-model dir that picks among the --n-samples plans")
    p.add_argument("--candidates", choices=["samples", "varied"], default="samples",
                   help="varied: the planner also considers shifted-grasp / arm-drift versions of the plans")
    return p.parse_args()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(model_dir, device, log_path, extra=()):
    port = free_port()
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [str(KAIROS / ".venv-train" / "bin" / "python"), str(KAIROS / "policy_server.py"),
         "--model-dir", str(model_dir), "--port", str(port), "--device", device, *extra],
        cwd=KAIROS, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"policy server exited early, see {log_path}")
        if f"READY {port}" in Path(log_path).read_text():
            return proc, policy_link.connect(port)
        time.sleep(1)
    proc.kill()
    raise TimeoutError(f"policy server not ready after 10 min, see {log_path}")


def request(sock, msg):
    policy_link.send(sock, msg)
    return policy_link.recv(sock)


def expert_check(env, args_env, seed, cache, cache_path):
    key = str(seed)
    if key not in cache:
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args_env)
            enable_mplib_expert(env)
            info = env.play_once()
            ok = bool(env.plan_success and env.check_success())
            cache[key] = {"solvable": ok, "info": info.get("info", {}), "reason": "" if ok else "expert_failed"}
        except Exception as e:  # unstable scene, planner error, ...
            cache[key] = {"solvable": False, "info": {}, "reason": f"{type(e).__name__}: {e}"[:200]}
        finally:
            env.close_env()
        cache_path.write_text(json.dumps(cache, indent=1))
    return cache[key]


def plan_spread(samples, n_exec):
    """samples: (K, chunk, 14) plans for one observation. Arm joints in rad, gripper steps in actions."""
    arm = samples[:, :, ARM]
    pair = np.abs(arm[:, None] - arm[None])  # (K, K, chunk, 12)
    k = len(samples)
    off = ~np.eye(k, dtype=bool)
    moved = np.abs(arm[:, n_exec - 1] - arm[:, 0]).mean()  # how far a plan moves the arm in the executed part
    grip_t = [[int(np.argmax(np.abs(g - g[0]) > 0.5 * max(np.ptp(g), 1e-6))) if np.ptp(g) > 0.1 else -1
               for g in s[:, [6, 13]].T] for s in samples]  # first step each gripper changes a lot, -1 = never
    return {
        "std": float(arm.std(0).mean()),                                # spread across plans, whole chunk
        "pair_exec": float(pair[off][:, :n_exec].mean()),               # mean gap between two plans, executed part
        "pair_end_exec": float(pair[off][:, n_exec - 1].mean()),        # gap at the last executed step
        "max_end_exec": float(pair[off][:, n_exec - 1].mean(-1).max()), # most different pair at that step
        "moved_exec": float(moved),
        "grip_t": grip_t,
    }


def run_episode(env, args_env, sock, seed, episode_idx, instruction, step_lim, n_action_steps, cam, smoother,
                noise_seed=None, n_samples=1, plans_log=None):
    env.setup_demo(now_ep_num=episode_idx, seed=seed, is_test=True, **args_env)
    env.step_lim = step_lim
    env.set_instruction(instruction=instruction)
    camera = ShowcaseCamera(env, **cam) if cam else None
    if camera:
        camera.capture()
    request(sock, {"cmd": "reset", "noise_seed": None if noise_seed is None else noise_seed * 1_000_003 + seed})
    if smoother:
        smoother.reset()
    executed = []
    t0 = time.time()
    while not (env.eval_success or env.take_action_cnt >= env.step_lim):
        obs = env.get_obs()
        reply = request(sock, {
            "cmd": "act",
            "images": {k: obs["observation"][v]["rgb"] for k, v in CAMERAS.items()},
            "state": np.asarray(obs["joint_action"]["vector"], dtype=np.float32),
            "task": instruction,
            "n_samples": n_samples,
        })
        if n_samples > 1:
            extra = {"scores": [round(float(x), 4) for x in reply["scores"]], "chosen": reply["chosen"],
                     "chosen_kind": reply.get("chosen_kind")} if "scores" in reply else {}
            plans_log.write(json.dumps({"seed": seed, "action": int(env.take_action_cnt),
                                        **plan_spread(reply["samples"], n_action_steps), **extra}) + "\n")
            plans_log.flush()
        chunk = smoother.next(reply["actions"], n_action_steps) if smoother else reply["actions"]
        for action in chunk[:n_action_steps]:
            env.take_action(action, action_type="qpos")
            executed.append(action)
            if camera:
                camera.capture()
            if env.eval_success or env.take_action_cnt >= env.step_lim:
                break
    result = {"seed": seed, "success": bool(env.eval_success), "actions": int(env.take_action_cnt),
              "instruction": instruction, "seconds": round(time.time() - t0, 1),
              "jitter": round(float(np.abs(np.diff(np.asarray(executed)[:, ARM], n=2, axis=0)).mean()), 5)
              if len(executed) > 2 else None}
    env.close_env(clear_cache=(episode_idx % 5 == 4))
    return result, camera


def main():
    args = parse_args()
    model_dir = (KAIROS / args.model_dir).resolve() if not args.model_dir.is_absolute() else args.model_dir
    tag = args.tag or model_dir.parent.name + ("_smooth" if args.smooth else "") + \
        (f"_ns{args.noise_seed}" if args.noise_seed is not None else "") + ("_planner" if args.planner else "") + \
        ("_varied" if args.planner and args.candidates == "varied" else "")
    out_dir = KAIROS / "outputs" / "eval" / tag
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    cache_dir = KAIROS / "outputs" / "eval" / "expert_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.task}_{args.variant}"
    results_path = out_dir / f"{name}.json"
    cache_path = cache_dir / f"{name}.json"

    results = json.loads(results_path.read_text())["episodes"] if results_path.exists() else []
    done = {r["seed"] for r in results}
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    with open(os.path.join(CONFIGS_PATH, "_eval_step_limit.yml")) as f:
        step_lim = args.step_limit or yaml.safe_load(f)[args.task]

    base.TASK = args.task
    args_env = base.load_args()
    env = getattr(importlib.import_module(f"envs.{args.task}"), args.task)()
    hard_variants.apply(env, args.variant)

    print(f"{name}: model {model_dir}, {len(results)}/{args.episodes} episodes already done, step limit {step_lim}",
          flush=True)
    planner_dir = None if args.planner is None else (args.planner if args.planner.is_absolute() else KAIROS / args.planner)
    server, sock = start_server(model_dir, args.device, out_dir / f"server_{name}.log",
                                () if planner_dir is None else ("--planner", str(planner_dir),
                                                                "--candidates", args.candidates))
    plans_log = open(out_dir / f"{name}_plans.jsonl", "a") if args.n_samples > 1 else None

    def save():
        n_ok = sum(r["success"] for r in results)
        results_path.write_text(json.dumps({
            "model_dir": str(model_dir), "task": args.task, "variant": args.variant,
            "n_action_steps": args.n_action_steps, "step_limit": step_lim, "smooth": args.smooth,
            "noise_seed": args.noise_seed, "n_samples": args.n_samples,
            "planner": None if planner_dir is None else str(planner_dir), "candidates": args.candidates,
            "success_rate": n_ok / len(results) if results else None, "successes": n_ok,
            "episodes_run": len(results), "episodes": results,
        }, indent=1))

    try:
        seed = args.seed_start
        while len(results) < args.episodes and seed < args.seed_start + 20 * args.episodes:
            if seed in done:
                seed += 1
                continue
            check = expert_check(env, args_env, seed, cache, cache_path)
            if not check["solvable"]:
                print(f"  seed {seed}: skipped ({check['reason'][:60]})", flush=True)
                seed += 1
                continue
            candidates = generate_episode_descriptions(args.task, [check["info"]], args.episodes, seed=seed)[0]["seen"]
            instruction = random.Random(f"{seed}:select").choice(candidates) if candidates else args.task
            hd = len(results) < args.hd_videos
            cam = dict(width=1920, height=1080, fps=15) if hd else dict(width=640, height=360, fps=15)
            cam = cam if args.videos else None
            result, camera = run_episode(env, args_env, sock, seed, len(results), instruction, step_lim,
                                         args.n_action_steps, cam, ChunkSmoother() if args.smooth else None,
                                         args.noise_seed, args.n_samples, plans_log)
            result["pot_model"] = check["info"].get("{A}")
            result["video"] = camera.save(str(Path("eval") / tag / "videos" /
                                              f"{name}_seed{seed}_{'ok' if result['success'] else 'fail'}.mp4")) \
                if camera else None
            results.append(result)
            save()
            n_ok = sum(r["success"] for r in results)
            print(f"  [{len(results):2d}/{args.episodes}] seed {seed}: {'SUCCESS' if result['success'] else 'fail   '} "
                  f"{result['actions']:3d} actions {result['seconds']:5.0f}s | running {n_ok}/{len(results)} "
                  f"| \"{instruction[:50]}\"", flush=True)
            seed += 1
    finally:
        try:
            request(sock, {"cmd": "close"})
        except OSError:
            pass
        server.wait(timeout=60)

    save()
    n_ok = sum(r["success"] for r in results)
    print(f"\n{name}: {n_ok}/{len(results)} = {100 * n_ok / max(1, len(results)):.0f}% success -> {results_path}")


if __name__ == "__main__":
    main()
