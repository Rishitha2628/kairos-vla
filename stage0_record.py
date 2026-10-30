"""World-model generalisation, Stage 0: record what "hands at the pot handles" looks like, at new pot positions.

    ~/kairos/.venv/bin/python ~/kairos/thermal_guard.py -- ~/kairos/.venv/bin/python ~/kairos/stage0_record.py \
        --model-dir outputs/smolvla_baseline/final --variant position --episodes 25

Question for Stage 0: do the wrist cameras see "about to grasp" the same way wherever the pot is? If so, the
demos' wrist views just before grasping are a goal a planner can aim for at any position.
For each seed (from 300000, so the 100000+ test seeds stay unseen), the same scene is run twice:
  expert  RoboTwin's scripted expert (measurement only, nothing is trained on it); kept if it succeeds.
          Frames every 15 physics steps, as in the demos.
  policy  the smoothed SmolVLA baseline, which mostly misses at new positions: frames every 2nd action.
Each frame stores, besides the 3 camera images, how far each hand (TCP) is from its handle (the success
check's contact points 0 and 1), and the gripper openings.

Per episode, outputs/stage0/<variant>/seed<N>_<expert|policy>/:
  cam_high.mp4, cam_left_wrist.mp4, cam_right_wrist.mp4, meta.npz (dist (T, 2) in m, grip (T, 2), success)
"""
import argparse
import json
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import step4_eval as ev
from action_smoothing import ChunkSmoother
from mplib_expert import enable_mplib_expert

KAIROS = ev.KAIROS


class Recorder:
    def __init__(self, env, out_dir):
        self.env, self.tmp = env, out_dir.with_name(out_dir.name + ".tmp")
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.writers = {name: imageio.get_writer(self.tmp / f"{name}.mp4", fps=15, codec="libx264", quality=8,
                                                 macro_block_size=8, ffmpeg_params=["-threads", "1"])
                        for name in ev.CAMERAS}
        self.dist, self.grip = [], []

    def frame(self, obs=None):
        env = self.env
        obs = obs or env.get_obs()
        for name, cam in ev.CAMERAS.items():
            self.writers[name].append_data(obs["observation"][cam]["rgb"])
        tcp = [np.array(env.robot.get_left_tcp_pose()[:3]), np.array(env.robot.get_right_tcp_pose()[:3])]
        handle = [np.array(env.pot.get_contact_point(i)[:3]) for i in (0, 1)]
        self.dist.append([np.linalg.norm(t - h) for t, h in zip(tcp, handle)])
        state = np.asarray(obs["joint_action"]["vector"])
        self.grip.append(state[[6, 13]])

    def close(self, out_dir, success):
        for w in self.writers.values():
            w.close()
        np.savez(self.tmp / "meta.npz", dist=np.array(self.dist, np.float32), grip=np.array(self.grip, np.float32),
                 success=success)
        if success is None:  # discard
            for f in self.tmp.iterdir():
                f.unlink()
            self.tmp.rmdir()
        else:
            self.tmp.rename(out_dir)


def run_expert(env, args_env, seed, out_dir):
    env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args_env)
    enable_mplib_expert(env)
    rec = Recorder(env, out_dir)
    env._take_picture = lambda: rec.frame()  # the expert's own frame hook (every save_freq physics steps)
    ok, info = False, {}
    try:
        info = env.play_once().get("info", {})
        ok = bool(env.plan_success and env.check_success())
    except Exception as e:  # unstable scene, planner error, ...
        print(f"  seed {seed}: expert error {type(e).__name__}: {e}"[:200], flush=True)
    finally:
        rec.close(out_dir, ok if ok else None)
        env.close_env()
    if ok:
        (out_dir / "info.json").write_text(json.dumps(info))
    return ok


def run_policy(env, args_env, sock, seed, step_lim, n_exec, out_dir, info):
    env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args_env)
    env.step_lim = step_lim
    candidates = ev.generate_episode_descriptions("lift_pot", [info], 1, seed=seed)[0]["seen"]  # as in step4_eval
    instruction = random.Random(f"{seed}:select").choice(candidates) if candidates else "lift_pot"
    env.set_instruction(instruction=instruction)
    ev.request(sock, {"cmd": "reset", "noise_seed": seed})
    smoother, rec = ChunkSmoother(), Recorder(env, out_dir)
    try:
        obs = env.get_obs()
        while not (env.eval_success or env.take_action_cnt >= env.step_lim):
            reply = ev.request(sock, {
                "cmd": "act", "images": {k: obs["observation"][v]["rgb"] for k, v in ev.CAMERAS.items()},
                "state": np.asarray(obs["joint_action"]["vector"], dtype=np.float32), "task": instruction})
            for action in smoother.next(reply["actions"], n_exec)[:n_exec]:
                if env.take_action_cnt % 2 == 0:
                    rec.frame(obs)
                env.take_action(action, action_type="qpos")
                obs = env.get_obs()
                if env.eval_success or env.take_action_cnt >= env.step_lim:
                    break
        rec.frame(obs)
    finally:
        rec.close(out_dir, bool(env.eval_success))
        env.close_env(clear_cache=True)
    return bool(env.eval_success)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--variant", default="position")
    p.add_argument("--episodes", type=int, default=25, help="expert successes to collect")
    p.add_argument("--seed-start", type=int, default=300000)
    p.add_argument("--n-action-steps", type=int, default=25)
    a = p.parse_args()
    task = "lift_pot"
    model_dir = (KAIROS / a.model_dir).resolve() if not a.model_dir.is_absolute() else a.model_dir
    out_root = KAIROS / "outputs" / "stage0" / a.variant
    out_root.mkdir(parents=True, exist_ok=True)
    for d in out_root.glob("*.tmp"):  # half-written episode from a crash
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    with open(Path(ev.CONFIGS_PATH) / "_eval_step_limit.yml") as f:
        step_lim = ev.yaml.safe_load(f)[task]
    ev.base.TASK = task
    args_env = ev.base.load_args() | {"save_freq": 15}
    env = getattr(ev.importlib.import_module(f"envs.{task}"), task)()
    ev.hard_variants.apply(env, a.variant)
    tried_path = out_root / "expert_tried.json"  # seeds whose expert failed, so a restart skips them
    tried = set(json.loads(tried_path.read_text())) if tried_path.exists() else set()
    server, sock = ev.start_server(model_dir, "cuda", out_root / "server.log")
    try:
        seed = a.seed_start
        while len(list(out_root.glob("seed*_expert"))) < a.episodes and seed < a.seed_start + 20 * a.episodes:
            exp_dir, pol_dir = out_root / f"seed{seed}_expert", out_root / f"seed{seed}_policy"
            if seed in tried and not exp_dir.exists():
                seed += 1
                continue
            if not exp_dir.exists():
                tried.add(seed)
                tried_path.write_text(json.dumps(sorted(tried)))
                if not run_expert(env, args_env, seed, exp_dir):
                    seed += 1
                    continue
            if not pol_dir.exists():
                info = json.loads((exp_dir / "info.json").read_text())
                ok = run_policy(env, args_env, sock, seed, step_lim, a.n_action_steps, pol_dir, info)
                d = np.load(exp_dir / "meta.npz")["dist"]
                print(f"  seed {seed}: expert ok ({len(d)} frames, closest {d.max(1).min() * 100:.1f} cm) | "
                      f"policy {'SUCCESS' if ok else 'fail'} | "
                      f"{len(list(out_root.glob('seed*_expert')))}/{a.episodes}", flush=True)
            seed += 1
    finally:
        try:
            ev.request(sock, {"cmd": "close"})
        except OSError:
            pass
        server.wait(timeout=60)


if __name__ == "__main__":
    main()
