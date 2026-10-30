"""World-model Stage 1: train on the demos' V-JEPA summaries and test on held-out demos.

    ~/kairos/.venv/bin/python ~/kairos/thermal_guard.py -- ~/kairos/.venv-train/bin/python ~/kairos/train_world_model.py

Trains two models with the same recipe: one that sees the actions, and an action-blind one (actions zeroed),
which can only learn "what usually happens next" from the picture. On 5 held-out demos per task it reports
(standardized MSE of the predicted summary, lower is better):
  copy now        predict that nothing changes
  blind           the action-blind model
  real actions    the world model with the actions that were actually taken
  wrong actions   the same model, given another test sample's movement (relative to this state)
plus how often the real actions predict the true future better than the wrong ones, and a 3-hop rollout
(3 x H steps) where the model feeds itself its own predictions.

Output: outputs/world_model/{world_model.pt, blind.pt, results.json}
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from world_model import KAIROS, WorldModel, WorldModelData


def train(data, horizon, use_actions, args, device):
    torch.manual_seed(args.seed)
    model = WorldModel(horizon, use_actions=use_actions).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    rows = data.starts(horizon, "train")
    steps = args.epochs * math.ceil(len(rows) / args.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, s / 200) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    best, best_state, step, t0 = float("inf"), None, 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        for chunk in rows[torch.randperm(len(rows))].split(args.batch_size):
            b = data.batch(chunk, horizon, device)
            loss = F.mse_loss(model(b["grid"], b["state"], b["actions"]), b["target"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
        if epoch % 2 == 1 or epoch == args.epochs - 1:
            test = evaluate(model, data, horizon, device)["model"]
            if test < best:
                best, best_state = test, {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  {'actions' if use_actions else 'blind  '} epoch {epoch + 1:3d}/{args.epochs}  train {loss.item():.4f}  "
                  f"held-out {test:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def evaluate(model, data, horizon, device, wrong=False, seed=0):
    """Mean held-out MSE per sample: copy-now, the model, and (wrong=True) the model with another sample's movement."""
    model.eval()
    rows = data.starts(horizon, "test")
    gen = torch.Generator().manual_seed(seed)
    out = {"copy": [], "model": [], "wrong": []}
    for chunk in rows.split(256):
        b = data.batch(chunk, horizon, device)
        out["copy"].append(((b["grid"] - b["target"]) ** 2).mean((1, 2)).cpu())
        out["model"].append(((model(b["grid"], b["state"], b["actions"]) - b["target"]) ** 2).mean((1, 2)).cpu())
        if wrong:
            perm = torch.randperm(len(chunk), generator=gen)  # same scene, someone else's movement
            pred = model(b["grid"], b["state"], b["actions"][perm.to(device)])
            out["wrong"].append(((pred - b["target"]) ** 2).mean((1, 2)).cpu())
    out = {k: torch.cat(v) for k, v in out.items() if v}
    return {k: v.mean().item() for k, v in out.items()} | ({"per_sample": out} if wrong else {})


@torch.no_grad()
def rollout(model, data, horizon, hops, device):
    """Feed the model its own prediction `hops` times (true actions), vs copy-now, at the final step."""
    rows = data.starts(horizon * hops, "test")
    err_model, err_copy = [], []
    for chunk in rows.split(256):
        grid = data.grid[chunk].to(device).float()
        state = data.state[chunk]
        for h in range(hops):
            start = chunk + h * horizon
            steps = start[:, None] + torch.arange(horizon)[None]
            acts = data.encode_actions(data.action[steps], state).to(device)
            grid = model(grid, ((state - data.state_mean) / data.state_std).to(device), acts)
            state = data.action[steps[:, -1]]  # action[t] == state[t+1]
        target = data.grid[chunk + horizon * hops].to(device).float()
        first = data.grid[chunk].to(device).float()
        err_model.append(((grid - target) ** 2).mean((1, 2)).cpu())
        err_copy.append(((first - target) ** 2).mean((1, 2)).cpu())
    return torch.cat(err_model).mean().item(), torch.cat(err_copy).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=KAIROS / "outputs" / "world_model")
    args = p.parse_args()
    device = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)

    data = WorldModelData()
    H = args.horizon
    print(f"{len(data.starts(H, 'train'))} train / {len(data.starts(H, 'test'))} held-out samples "
          f"(held-out demos {data.holdout}), horizon {H}", flush=True)

    model = train(data, H, True, args, device)
    print(f"world model: {model.n_params() / 1e6:.1f}M params", flush=True)
    blind = train(data, H, False, args, device)

    ev = evaluate(model, data, H, device, wrong=True)
    ps = ev.pop("per_sample")
    blind_err = evaluate(blind, data, H, device)["model"]
    better = (ps["model"] < ps["wrong"]).float().mean().item()
    roll_model, roll_copy = rollout(model, data, H, 3, device)
    roll_blind, _ = rollout(blind, data, H, 3, device)
    res = {
        "horizon": H, "held_out_demos": data.holdout,
        "copy_now": ev["copy"], "blind": blind_err, "real_actions": ev["model"], "wrong_actions": ev["wrong"],
        "real_beats_wrong": better,
        f"rollout_{3 * H}_steps": {"copy_now": roll_copy, "blind": roll_blind, "real_actions": roll_model},
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.out / "results.json").write_text(json.dumps(res, indent=1))
    torch.save({"state_dict": model.state_dict(), "horizon": H}, args.out / "world_model.pt")
    torch.save({"state_dict": blind.state_dict(), "horizon": H, "use_actions": False}, args.out / "blind.pt")

    print(f"\nheld-out demos, {H} steps ahead (standardized MSE, lower is better):")
    print(f"  copy now (no change)     {ev['copy']:.4f}")
    print(f"  action-blind model       {blind_err:.4f}")
    print(f"  world model, real acts   {ev['model']:.4f}")
    print(f"  world model, wrong acts  {ev['wrong']:.4f}")
    print(f"  real actions predict the true future better than wrong ones in {100 * better:.0f}% of samples")
    print(f"  {3 * H}-step rollout: copy {roll_copy:.4f}, blind {roll_blind:.4f}, world model {roll_model:.4f}")


if __name__ == "__main__":
    main()
