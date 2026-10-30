"""Fine-tune SmolVLA on the RoboTwin 2-task clean set (lift_pot + beat_block_hammer).

One script for Steps 3, 6 and 7 so the three models differ only in the foresight target:
  --foresight none      Step 3 baseline
  --foresight vjepa     Step 6: also predict the V-JEPA 2 embedding of this demo's head camera
                        --foresight-frames (8 = 0.53 s) ahead
  --foresight shuffled  Step 7 control: same, but the "future" comes from another demo of the same task
                        (a fixed random pairing; same frame offset, clamped to that demo's length)
Foresight head (foresight.py): reads the action expert's output tokens (the trainable part - a head on the
frozen VLM could not change the policy), predicts the 4x4 grid of V-JEPA 2 features from Step 5, standardized
per dimension; loss = flow-matching loss + --foresight-weight * MSE. The head is only used in training: the
saved policy is a plain SmolVLA, evaluated exactly like the baseline.

Run under the thermal watchdog:
  ~/kairos/.venv-train/bin/python ~/kairos/thermal_guard.py -- \
      ~/kairos/.venv-train/bin/python ~/kairos/train_smolvla.py --output-dir ~/kairos/outputs/smolvla_baseline

Carried over from the LIBERO run (vla-libero-project/train_smolvla.py):
  * VLM frozen, only the action expert trains (fits in 6 GB);
  * delta_timestamps, so `action` is the true next chunk_size trajectory;
  * normalization via the processor pipeline, saved next to the weights;
  * dataset image stats already recomputed per pixel by step3_convert_to_lerobot.py.
New here:
  * instruction augmentation: each sample gets a random phrasing from the ~100 RoboTwin "seen"
    phrasings recorded for its own episode;
  * one rolling `latest/` checkpoint holding only the trainable weights + optimizer + step (the frozen
    VLM comes from the base model on resume), so the small disk holds it; --resume continues and
    SIGTERM saves before exiting. Only `final/` is a full 1.2 GB model;
  * CPU threads capped (--cpu-threads, --num-workers) so video decoding doesn't overheat the CPU;
  * self-cooling: above --cool-above C (CPU) or --gpu-cool-above C (GPU) the loop waits until
    --cool-to / --gpu-cool-to C before the next step,
    so the external watchdog (thermal_guard.py) rarely has to freeze the processes;
  * a transient DataLoader worker error rebuilds the loader instead of ending the run.
"""
import argparse
import glob
import json
import math
import random
import shutil
import signal
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

import foresight

KAIROS = Path(__file__).resolve().parent
REPO_ID = "local/robotwin_clean_2task"
CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=KAIROS / "data" / "robotwin_clean_2task")
    p.add_argument("--output-dir", type=Path, default=KAIROS / "outputs" / "smolvla_baseline")
    p.add_argument("--foresight", choices=["none", "vjepa", "shuffled"], default="none")
    p.add_argument("--foresight-frames", type=int, default=8, help="how far ahead (dataset frames, 15 fps)")
    p.add_argument("--foresight-weight", type=float, default=0.1)
    p.add_argument("--embeddings", type=Path, default=None,
                   help="Step 5 output (default <dataset-root>/vjepa2/cam_high_vitg.pt)")
    p.add_argument("--base", default="lerobot/smolvla_base")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--img", type=int, default=256, help="VLM input resolution (512 is 4x slower)")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=2, help="torch CPU threads in the main process")
    p.add_argument("--cool-above", type=float, default=85, help="CPU C at which training waits to cool")
    p.add_argument("--cool-to", type=float, default=82)
    p.add_argument("--gpu-cool-above", type=float, default=82, help="GPU C at which training waits to cool")
    p.add_argument("--gpu-cool-to", type=float, default=79)
    p.add_argument("--no-augment", action="store_true", help="always use the stored instruction")
    p.add_argument("--resume", action="store_true", help="continue from output-dir/latest")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def limit_worker_threads(_):
    torch.set_num_threads(1)


def cpu_temp_path():
    for hwmon in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if open(f"{hwmon}/name").read().strip() == "k10temp":
                return f"{hwmon}/temp1_input"
        except OSError:
            pass
    return None


def cpu_temp(path):
    return int(open(path).read()) / 1000 if path else 0.0


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return 0.0


def wait_until_cool(read, above, to, poll=0.5):
    """Wait while read() is at/above `above` until it drops to `to`. Returns seconds spent waiting."""
    if read() < above:
        return 0.0
    t0 = time.time()
    while read() > to:
        time.sleep(poll)
    return time.time() - t0


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest = args.output_dir / "latest"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    probe = LeRobotDataset(REPO_ID, root=args.dataset_root)
    state_dim = probe.meta.features["observation.state"]["shape"][0]
    action_dim = probe.meta.features["action"]["shape"][0]
    h, w, _ = probe.meta.features[f"observation.images.{CAMERAS[0]}"]["shape"]
    print(f"fps={probe.meta.fps} frames={probe.meta.total_frames} episodes={probe.meta.total_episodes}", flush=True)

    cfg = SmolVLAConfig(
        input_features={
            **{f"observation.images.{c}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w)) for c in CAMERAS},
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        resize_imgs_with_padding=(args.img, args.img),
        freeze_vision_encoder=True,
        train_expert_only=True,
        optimizer_lr=args.lr,
        device=args.device,
    )

    # Real next-chunk actions, not one action repeated.
    delta_timestamps = {"action": [i / probe.meta.fps for i in cfg.action_delta_indices]}
    dataset = LeRobotDataset(REPO_ID, root=args.dataset_root, delta_timestamps=delta_timestamps)

    def make_loader(num_workers=args.num_workers):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=True, drop_last=True, persistent_workers=num_workers > 0,
                          worker_init_fn=limit_worker_threads)

    loader = make_loader()

    phrasings = {e["episode_index"]: e["seen"]
                 for e in json.loads((args.dataset_root / "meta" / "instructions.json").read_text())}
    rng = random.Random(args.seed)

    def augment(batch):
        batch["task"] = [rng.choice(phrasings[int(i)]) for i in batch["episode_index"]]
        return batch

    print(f"loading {args.base} ...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args.base, config=cfg).to(args.device)
    policy.train()
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"params total={total / 1e6:.1f}M trainable={trainable / 1e6:.1f}M", flush=True)

    preprocessor, postprocessor = make_smolvla_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)
    params = [p for p in policy.parameters() if p.requires_grad]
    head = targets = None
    if args.foresight != "none":
        targets = foresight.ForesightTargets(args.embeddings or args.dataset_root / "vjepa2" / "cam_high_vitg.pt",
                                             args.dataset_root, args.foresight, args.foresight_frames, args.seed)
        targets.check_alignment(dataset)
        head = foresight.ForesightHead(policy, targets.dim).to(args.device)
        params += list(head.parameters())
        print(f"foresight {args.foresight}: {targets.describe()}, head {head.n_params() / 1e6:.1f}M params, "
              f"weight {args.foresight_weight}", flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=cfg.optimizer_betas,
                                  eps=cfg.optimizer_eps, weight_decay=cfg.optimizer_weight_decay)

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    step = 0
    if args.resume:
        state = torch.load(latest / "train_state.pt", map_location="cpu", weights_only=False)
        missing = policy.load_state_dict(state["trainable"], strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        rng.setstate(state["rng"])
        if head is not None:
            head.load_state_dict(state["head"])
        print(f"resumed at step {step}", flush=True)

    trainable_names = {n for n, p in policy.named_parameters() if p.requires_grad}

    def save(final=False):
        tmp = args.output_dir / "latest.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        if final:
            policy.save_pretrained(tmp)
            preprocessor.save_pretrained(tmp)
            postprocessor.save_pretrained(tmp)
            if head is not None:  # not needed to run the policy; kept for analysis
                torch.save(head.state_dict(), tmp / "foresight_head.pt")
        else:
            trainable = {n: t for n, t in policy.state_dict().items() if n in trainable_names}
            torch.save({"trainable": trainable, "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "step": step, "rng": rng.getstate(),
                        "head": head.state_dict() if head is not None else None},
                       tmp / "train_state.pt")
        (tmp / "train_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
        target = args.output_dir / ("final" if final else "latest")
        shutil.rmtree(target, ignore_errors=True)
        tmp.rename(target)
        if final:
            shutil.rmtree(latest, ignore_errors=True)
        print(f"saved {target} (step {step})", flush=True)

    stop = {"requested": False}

    def request_stop(signum, _):
        print(f"signal {signum}: saving and stopping after this step", flush=True)
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    log = open(args.output_dir / "train_log.csv", "a")
    if step == 0:
        log.write("step,loss,lr,samples_per_s,foresight_loss\n")
    print(f"training {args.steps} steps, batch {args.batch_size}, lr {args.lr}, img {args.img}, "
          f"augment {not args.no_augment}, foresight {args.foresight}", flush=True)

    temp_path = cpu_temp_path()
    running, running_fs, t0, cooled, loader_errors, window_start = [], [], time.time(), 0.0, 0, step
    data_iter = iter(loader)
    while step < args.steps and not stop["requested"]:
        cooled += wait_until_cool(lambda: cpu_temp(temp_path), args.cool_above, args.cool_to)
        # nvidia-smi costs ~30 ms per step (~3%); the GPU heats too fast to check less often.
        cooled += wait_until_cool(gpu_temp, args.gpu_cool_above, args.gpu_cool_to, poll=1.0)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        except RuntimeError as e:  # e.g. transient shared-memory failure in a worker
            loader_errors += 1
            # Workers hand batches over through /dev/shm, which fails under RAM pressure (no swap here).
            # After a few failures, load in the main process instead: slower, but no shared memory at all.
            workers = args.num_workers if loader_errors < 3 else 0
            print(f"DataLoader error #{loader_errors} at step {step}, rebuilding loader with {workers} workers: "
                  f"{str(e)[:160]}", flush=True)
            if loader_errors > 50:
                raise
            del data_iter
            loader = make_loader(workers)
            data_iter = iter(loader)
            continue

        if not args.no_augment:
            batch = augment(batch)
            if step < 2:  # prove it in the log, do not just claim it
                print(f"  [aug sample] {batch['task'][:2]}", flush=True)
        if head is not None:
            target = targets.lookup(batch["index"]).to(args.device, non_blocking=True)
        batch = preprocessor(batch)  # normalizes + tokenizes the instruction
        loss, _ = policy.forward(batch)
        flow_loss = loss
        if head is not None:
            fs_loss = head.loss(target)
            running_fs.append(fs_loss.item())
            loss = loss + args.foresight_weight * fs_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.optimizer_grad_clip_norm)
        optimizer.step()
        scheduler.step()

        running.append(flow_loss.item())  # the action loss alone, comparable with the baseline's
        step += 1
        if step % 100 == 0:
            avg = sum(running[-100:]) / len(running[-100:])
            sps = (step - window_start) * args.batch_size / (time.time() - t0)
            mem = torch.cuda.max_memory_allocated() / 2**30
            fs = sum(running_fs[-100:]) / len(running_fs[-100:]) if running_fs else float("nan")
            print(f"{time.strftime('%H:%M:%S')} Step {step}/{args.steps} | loss {avg:.4f} | foresight {fs:.4f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | {sps:.1f} samp/s | cooling {cooled:.0f}s | "
                  f"peak {mem:.2f}GiB", flush=True)
            log.write(f"{step},{avg:.6f},{scheduler.get_last_lr()[0]:.3e},{sps:.2f},{fs:.6f}\n")
            log.flush()
            t0, cooled, window_start = time.time(), 0.0, step
        if step % args.save_every == 0 and step < args.steps:
            save()

    if stop["requested"]:
        save()
        print(f"stopped at step {step}; continue with --resume", flush=True)
    else:
        save(final=True)
        print("Training complete!", flush=True)
    log.close()


if __name__ == "__main__":
    main()
