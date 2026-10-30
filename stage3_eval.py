"""World-model generalisation, Stage 3: SmolVLA + a world-model reach planner at new pot positions.

    ~/kairos/.venv/bin/python ~/kairos/thermal_guard.py -- ~/kairos/.venv/bin/python ~/kairos/stage3_eval.py \
        --model-dir outputs/smolvla_baseline/final --world-model outputs/world_model3 --variant position --episodes 50

Same protocol, scenes and smoothed SmolVLA as step4_eval.py (noise seed 1). SmolVLA drives until the chunk it is
about to execute closes a gripper (its grasp). There, instead:
  1. refine  up to --refine-steps times: the world model (planner.ReachPlanner) picks a small arm correction
             that makes the wrist views look like the demos just before grasping; 8 steps are executed, then it
             re-plans from the new camera images. Stops early when no correction is predicted to help.
  2. close   both grippers close over 15 steps with the arm held still (--grab fixed), or SmolVLA's next
             chunk runs unchanged and SmolVLA grasps itself (--grab policy);
  3. SmolVLA takes over again (the lift). At most --max-grasps such interventions per episode.
--refine-steps 0 is the control: the same intercept-and-close without the world model.
Per episode the hand-handle distance before and after each refine phase is logged (from the simulator,
for analysis only; the planner never sees it).

Output: outputs/eval/<tag>/<task>_<variant>.json, tag = smolvla_baseline_smooth_ns1_reach<R>[_policygrab]
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

import step4_eval as ev
from action_smoothing import ChunkSmoother

KAIROS = ev.KAIROS
GRIP = [6, 13]


def hand_dist(env):
    tcp = [np.array(env.robot.get_left_tcp_pose()[:3]), np.array(env.robot.get_right_tcp_pose()[:3])]
    return [round(float(np.linalg.norm(t - np.array(env.pot.get_contact_point(i)[:3]))), 4) for i, t in enumerate(tcp)]


def run_episode(env, args_env, sock, seed, idx, instruction, step_lim, a):
    env.setup_demo(now_ep_num=idx, seed=seed, is_test=True, **args_env)
    env.step_lim = step_lim
    env.set_instruction(instruction=instruction)
    ev.request(sock, {"cmd": "reset", "noise_seed": a.noise_seed * 1_000_003 + seed})
    smoother, grasps, t0 = ChunkSmoother(), [], time.time()
    release = False  # --grab policy: SmolVLA's next chunk runs unchanged after a refine phase

    def done():
        return env.eval_success or env.take_action_cnt >= env.step_lim

    def execute(actions):
        for action in actions:
            env.take_action(action, action_type="qpos")
            if done():
                return

    def observe():
        obs = env.get_obs()
        return {k: obs["observation"][v]["rgb"] for k, v in ev.CAMERAS.items()}, \
            np.asarray(obs["joint_action"]["vector"], dtype=np.float32)

    while not done():
        images, state = observe()
        reply = ev.request(sock, {"cmd": "act", "images": images, "state": state, "task": instruction})
        chunk = smoother.next(reply["actions"], a.n_action_steps)[:a.n_action_steps]
        closes = (chunk[:, GRIP].min() < 0.45) and (state[GRIP].min() >= 0.45)
        if not closes or release or len(grasps) >= a.max_grasps:
            execute(chunk)
            release = False
            continue
        g = {"at_action": int(env.take_action_cnt), "dist_before": hand_dist(env), "steps": []}
        for _ in range(a.refine_steps):
            images, state = observe()
            r = ev.request(sock, {"cmd": "refine", "images": images, "state": state})
            g["steps"].append([round(r["score"], 4), round(r["stay"], 4)])
            if r["score"] > r["stay"] - a.min_gain:  # no correction is predicted to help
                break
            execute(r["actions"])
            if done():
                break
        g["dist_after"] = hand_dist(env)
        grasps.append(g)
        if done():
            break
        smoother.reset()
        if a.grab == "policy":
            release = True
            continue
        _, state = observe()
        close = np.repeat(state[None], 15, 0)
        close[:, GRIP] = np.linspace(state[GRIP], 0.0, 16)[1:]
        execute(close)
    return {"seed": seed, "success": bool(env.eval_success), "actions": int(env.take_action_cnt),
            "instruction": instruction, "seconds": round(time.time() - t0, 1), "grasps": grasps}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--world-model", type=Path, default=KAIROS / "outputs" / "world_model3")
    p.add_argument("--task", default="lift_pot")
    p.add_argument("--variant", default="position", choices=["standard", "position"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=100000)
    p.add_argument("--n-action-steps", type=int, default=25)
    p.add_argument("--noise-seed", type=int, default=1)
    p.add_argument("--refine-steps", type=int, default=8, help="0 = control without the world model")
    p.add_argument("--min-gain", type=float, default=0.002, help="stop refining below this predicted improvement")
    p.add_argument("--max-grasps", type=int, default=2)
    p.add_argument("--grab", default="fixed", choices=["fixed", "policy"],
                   help="after refining: fixed close with the arm held, or SmolVLA grasps itself")
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    model_dir = (KAIROS / a.model_dir).resolve() if not a.model_dir.is_absolute() else a.model_dir
    wm_dir = (KAIROS / a.world_model).resolve() if not a.world_model.is_absolute() else a.world_model
    tag = a.tag or f"{model_dir.parent.name}_smooth_ns{a.noise_seed}_reach{a.refine_steps}" + ("_policygrab" if a.grab == "policy" else "")
    out_dir = KAIROS / "outputs" / "eval" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{a.task}_{a.variant}"
    results_path = out_dir / f"{name}.json"
    cache_path = KAIROS / "outputs" / "eval" / "expert_cache" / f"{name}.json"
    results = json.loads(results_path.read_text())["episodes"] if results_path.exists() else []
    done = {r["seed"] for r in results}
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    with open(Path(ev.CONFIGS_PATH) / "_eval_step_limit.yml") as f:
        step_lim = ev.yaml.safe_load(f)[a.task]
    ev.base.TASK = a.task
    args_env = ev.base.load_args()
    env = getattr(ev.importlib.import_module(f"envs.{a.task}"), a.task)()
    ev.hard_variants.apply(env, a.variant)
    extra = ("--reach-planner", str(wm_dir)) if a.refine_steps > 0 else ()
    server, sock = ev.start_server(model_dir, "cuda", out_dir / f"server_{name}.log", extra)
    print(f"{name}: {tag}, {len(results)}/{a.episodes} done", flush=True)

    def save():
        n_ok = sum(r["success"] for r in results)
        results_path.write_text(json.dumps({
            "model_dir": str(model_dir), "world_model": str(wm_dir), "task": a.task, "variant": a.variant,
            "noise_seed": a.noise_seed, "refine_steps": a.refine_steps, "min_gain": a.min_gain,
            "max_grasps": a.max_grasps, "grab": a.grab, "success_rate": n_ok / len(results) if results else None,
            "successes": n_ok, "episodes_run": len(results), "episodes": results}, indent=1))

    try:
        seed = a.seed_start
        while len(results) < a.episodes and seed < a.seed_start + 20 * a.episodes:
            if seed in done:
                seed += 1
                continue
            check = ev.expert_check(env, args_env, seed, cache, cache_path)
            if not check["solvable"]:
                seed += 1
                continue
            candidates = ev.generate_episode_descriptions(a.task, [check["info"]], a.episodes, seed=seed)[0]["seen"]
            instruction = random.Random(f"{seed}:select").choice(candidates) if candidates else a.task
            r = run_episode(env, args_env, sock, seed, len(results), instruction, step_lim, a)
            env.close_env(clear_cache=(len(results) % 5 == 4))
            results.append(r)
            save()
            n_ok = sum(x["success"] for x in results)
            gs = " ".join(f"{max(g['dist_before']) * 100:.1f}->{max(g['dist_after']) * 100:.1f}cm({len(g['steps'])})"
                          for g in r["grasps"]) or "no grasp"
            print(f"  [{len(results):2d}/{a.episodes}] seed {seed}: {'SUCCESS' if r['success'] else 'fail   '} "
                  f"{r['actions']:3d} actions {r['seconds']:4.0f}s | {gs} | running {n_ok}/{len(results)}", flush=True)
            seed += 1
    finally:
        try:
            ev.request(sock, {"cmd": "close"})
        except OSError:
            pass
        server.wait(timeout=60)
    n_ok = sum(r["success"] for r in results)
    print(f"\n{name}: {n_ok}/{len(results)} = {100 * n_ok / max(1, len(results)):.0f}% success -> {results_path}")


if __name__ == "__main__":
    main()
