"""World-model Stage 2/3: train on demos + recorded attempts (with mistakes), 3 cameras, plus a success scorer.

    nice -n 19 ~/kairos/.venv/bin/python ~/kairos/thermal_guard.py -- \
        ~/kairos/.venv-train/bin/python ~/kairos/train_world_model2.py

Data: the 100 demos (every frame) and the lift_pot attempts from collect_rollouts.py (every 4th frame,
encode_rollouts.py). Held out: the same 10 demos as Stage 1 and 40 of the attempts.
Models (same recipe as Stage 1, horizon 8):
  wm3     world model, 3 cameras (head + both wrists)
  blind3  the same without actions
  wm1     world model, head camera only (does seeing the grasp up close help?)
  value   success scorer on a 3-camera summary: target 0.99^(steps until success), 0 if the attempt failed
Tests on held-out data:
  A  prediction error: copy-now / blind / real actions / wrong actions, separately for demos and attempts
  B  small differences (as diag_world_model.py): real actions vs arm offsets and gripper shifts
  C  does imagining help judge an attempt? AUC for "this attempt will succeed" from the value of
     the scene now, the imagined scene 8 steps later, and the real scene 8 steps later
Output: outputs/world_model2/{wm3.pt, blind3.pt, wm1.pt, value.pt, results.json}
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from world_model import CELLS, DATASET, DIM, KAIROS, ValueModel, WorldModel

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
ARM = list(range(6)) + list(range(7, 13))
GRIP = [6, 13]


class SequenceData:
    """Every encoded frame of demos and attempts as one table of rows (grids for all cameras, fp16)."""

    def __init__(self, horizon=8, gamma=0.99, rollouts=KAIROS / "outputs" / "rollouts" / "lift_pot",
                 holdout_rollouts=40, seed=0):
        H = horizon
        demo_files = {c: DATASET / "vjepa2" / f"{c}_vitg.pt" for c in CAMERAS}
        roll = torch.load(rollouts / "vjepa2_stride4.pt", weights_only=True)
        n_demo = len(torch.load(demo_files[CAMERAS[0]], weights_only=True)["episode_index"])
        n_roll = len(roll["seed"])
        n = n_demo + n_roll
        self.grid = torch.empty(n, CELLS * len(CAMERAS), DIM, dtype=torch.float16)
        self.mean, self.std = [], []
        for k, cam in enumerate(CAMERAS):  # one camera at a time: each file is 0.5 GB
            g = torch.load(demo_files[cam], weights_only=True)["grid"].flatten(1, 2)
            s, s2 = torch.zeros(CELLS, DIM, dtype=torch.float64), torch.zeros(CELLS, DIM, dtype=torch.float64)
            for part in g.split(1024):
                s += part.double().sum(0)
                s2 += (part.double() ** 2).sum(0)
            mean = (s / len(g)).float()
            std = ((s2 / len(g) - (s / len(g)) ** 2).clamp_min(0).sqrt()).float().clamp_min(1e-4)
            cells = slice(k * CELLS, (k + 1) * CELLS)
            for src, dst in ((g, 0), (roll[cam].flatten(1, 2), n_demo)):
                for i in range(0, len(src), 1024):
                    part = src[i:i + 1024]
                    self.grid[dst + i:dst + i + len(part), cells] = ((part.float() - mean) / std).half()
            self.mean.append(mean)
            self.std.append(std)
            del g
            roll[cam] = None

        # demos: state/action per frame from the parquet
        table = pq.read_table(next((DATASET / "data").rglob("*.parquet")),
                              columns=["action", "observation.state", "episode_index", "frame_index"])
        d_action = torch.tensor(np.stack(table["action"].to_numpy(zero_copy_only=False)), dtype=torch.float32)
        d_state = torch.tensor(np.stack(table["observation.state"].to_numpy(zero_copy_only=False)), dtype=torch.float32)
        d_ep = torch.tensor(table["episode_index"].to_numpy())
        d_fr = torch.tensor(table["frame_index"].to_numpy())
        tasks = {r["episode_index"]: r["robotwin_task"]
                 for r in json.loads((DATASET / "meta" / "instructions.json").read_text())}
        rng = random.Random(seed)  # same held-out demos as Stage 1 (WorldModelData)
        demo_holdout = sorted(e for t in sorted(set(tasks.values()))
                              for e in rng.sample(sorted(e for e in tasks if tasks[e] == t), 5))

        self.kind = torch.zeros(n, dtype=torch.long)      # 0 demo, 1 attempt
        self.seq = torch.zeros(n, dtype=torch.long)       # episode index (demo) or seed (attempt)
        self.frame = torch.zeros(n, dtype=torch.long)
        self.next = torch.full((n,), -1, dtype=torch.long)  # row H steps later, -1 if none
        self.actions = torch.zeros(n, H, 14)
        self.state = torch.zeros(n, 14)
        self.value = torch.zeros(n)
        self.success = torch.zeros(n)
        self.test = torch.zeros(n, dtype=torch.bool)

        length = torch.bincount(d_ep)
        for e in range(len(length)):
            rows = torch.nonzero(d_ep == e).squeeze(1)
            L = len(rows)
            self.seq[rows], self.frame[rows], self.state[rows] = e, d_fr[rows], d_state[rows]
            self.value[rows] = gamma ** (L - 1 - d_fr[rows]).float()
            self.success[rows] = 1
            self.test[rows] = e in demo_holdout
            ok = rows[: max(0, L - H)]
            self.next[ok] = ok + H
            self.actions[ok] = d_action[ok[:, None] + torch.arange(H)[None]]

        seeds = sorted(set(roll["seed"].tolist()))
        roll_holdout = set(random.Random(seed).sample(seeds, holdout_rollouts))
        row_of = {}
        for i, (s, f) in enumerate(zip(roll["seed"].tolist(), roll["frame"].tolist())):
            row_of[(s, f)] = n_demo + i
        for s in seeds:
            meta = np.load(rollouts / f"seed{s}" / "meta.npz")
            act, st, ok = torch.from_numpy(meta["action"]), torch.from_numpy(meta["state"]), bool(meta["success"])
            T = len(act)
            for f in range(0, T + 1, roll["stride"]):
                r = row_of[(s, f)]
                self.kind[r], self.seq[r], self.frame[r], self.state[r] = 1, s, f, st[f]
                self.value[r] = gamma ** (T - f) if ok else 0.0
                self.success[r] = float(ok)
                self.test[r] = s in roll_holdout
                if f + H <= T and (s, f + H) in row_of:
                    self.next[r] = row_of[(s, f + H)]
                    self.actions[r] = act[f:f + H]
        self.delta_std = (self.actions[self.next >= 0] - self.state[self.next >= 0][:, None]).flatten(0, 1).std(0).clamp_min(1e-4)
        self.state_mean, self.state_std = self.state.mean(0), self.state.std(0).clamp_min(1e-4)
        self.horizon, self.gamma = H, gamma
        self.demo_holdout, self.roll_holdout = demo_holdout, sorted(roll_holdout)

    def rows(self, split, kind=None, with_next=True):
        m = self.test if split == "test" else ~self.test
        if kind is not None:
            m = m & (self.kind == kind)
        if with_next:
            m = m & (self.next >= 0)
        return torch.nonzero(m).squeeze(1)

    def encode_actions(self, actions, state):
        return (actions - state[:, None]) / self.delta_std / 10

    def batch(self, rows, device, cams=3, actions=None):
        c = slice(0, CELLS * cams)
        acts = self.actions[rows] if actions is None else actions
        return {
            "grid": self.grid[rows, c].to(device).float(),
            "state": ((self.state[rows] - self.state_mean) / self.state_std).to(device),
            "actions": self.encode_actions(acts, self.state[rows]).to(device),
            "target": self.grid[self.next[rows].clamp_min(0), c].to(device).float(),
        }


def cosine_lr(opt, steps):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, s / 200) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))


def train_wm(data, cams, use_actions, args, device, name):
    torch.manual_seed(args.seed)
    model = WorldModel(data.horizon, use_actions=use_actions, n_cams=cams).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    rows, test = data.rows("train"), data.rows("test")
    sched = cosine_lr(opt, args.epochs * math.ceil(len(rows) / args.batch_size))
    best, best_state, t0 = float("inf"), None, time.time()
    for epoch in range(args.epochs):
        model.train()
        for chunk in rows[torch.randperm(len(rows))].split(args.batch_size):
            b = data.batch(chunk, device, cams)
            loss = F.mse_loss(model(b["grid"], b["state"], b["actions"]), b["target"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        if epoch % 2 == 1 or epoch == args.epochs - 1:
            err = pred_error(model, data, test, device, cams).mean().item()
            if err < best:
                best, best_state = err, {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  {name:6s} epoch {epoch + 1:3d}/{args.epochs}  train {loss.item():.4f}  held-out {err:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def pred_error(model, data, rows, device, cams, actions=None):
    """Per-row MSE of the predicted summary (optionally with replacement actions, absolute)."""
    model.eval()
    out = []
    for i in range(0, len(rows), 256):
        r = rows[i:i + 256]
        b = data.batch(r, device, cams, None if actions is None else actions[i:i + 256])
        out.append(((model(b["grid"], b["state"], b["actions"]) - b["target"]) ** 2).mean((1, 2)).cpu())
    model.train()
    return torch.cat(out)


def train_value(data, args, device):
    torch.manual_seed(args.seed)
    model = ValueModel(n_cams=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    rows, test = data.rows("train", with_next=False), data.rows("test", kind=1, with_next=False)
    sched = cosine_lr(opt, args.value_epochs * math.ceil(len(rows) / args.batch_size))
    best, best_state, t0 = float("inf"), None, time.time()
    for epoch in range(args.value_epochs):
        model.train()
        for chunk in rows[torch.randperm(len(rows))].split(args.batch_size):
            loss = F.binary_cross_entropy_with_logits(model(data.grid[chunk].to(device).float()),
                                                      data.value[chunk].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if epoch % 2 == 1 or epoch == args.value_epochs - 1:
            v = value_of(model, data.grid[test].float(), device)
            err = F.binary_cross_entropy(v.clamp(1e-6, 1 - 1e-6), data.value[test]).item()
            if err < best:
                best, best_state = err, {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  value  epoch {epoch + 1:3d}/{args.value_epochs}  train {loss.item():.4f}  held-out attempts "
                  f"{err:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def value_of(model, grids, device):
    model.eval()
    return torch.cat([torch.sigmoid(model(g.to(device))).cpu() for g in grids.split(256)])


def auc(score, label):
    """Probability that a random success scores higher than a random failure."""
    order = torch.argsort(score)
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=score.dtype)
    pos = label > 0.5
    n_pos, n_neg = pos.sum().item(), (~pos).sum().item()
    return ((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--value-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=KAIROS / "outputs" / "world_model2")
    args = p.parse_args()
    device = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)

    data = SequenceData()
    for split in ("train", "test"):
        print(f"{split}: {len(data.rows(split, 0))} demo + {len(data.rows(split, 1))} attempt samples", flush=True)

    models = {"wm3": train_wm(data, 3, True, args, device, "wm3"),
              "blind3": train_wm(data, 3, False, args, device, "blind3"),
              "wm1": train_wm(data, 1, True, args, device, "wm1")}
    value = train_value(data, args, device)
    for name, m in models.items():
        torch.save({"state_dict": m.state_dict(), "horizon": data.horizon, "n_cams": m.cells // CELLS,
                    "use_actions": m.use_actions}, args.out / f"{name}.pt")
    torch.save({"state_dict": value.state_dict(), "n_cams": 3, "gamma": data.gamma}, args.out / "value.pt")
    torch.save({"mean": data.mean, "std": data.std, "delta_std": data.delta_std, "state_mean": data.state_mean,
                "state_std": data.state_std, "cameras": CAMERAS}, args.out / "normalization.pt")

    res = {"demo_holdout": data.demo_holdout, "attempt_holdout": data.roll_holdout, "A": {}, "B": {}, "C": {}}
    gen = torch.Generator().manual_seed(0)
    print("\nA. held-out prediction error, 8 steps ahead (lower is better)")
    for kind, label in ((0, "demos"), (1, "attempts")):
        rows = data.rows("test", kind)
        g = data.grid[rows].float()
        copy = ((g - data.grid[data.next[rows]].float()) ** 2).mean((1, 2)).mean().item()
        copy1 = ((g[:, :CELLS] - data.grid[data.next[rows], :CELLS].float()) ** 2).mean((1, 2)).mean().item()
        wrong = data.actions[rows][torch.randperm(len(rows), generator=gen)]
        wrong = wrong - wrong[:, :1] + data.actions[rows][:, :1]  # someone else's movement, from this pose
        r = {"copy_now_3cam": copy, "blind3": pred_error(models["blind3"], data, rows, device, 3).mean().item(),
             "wm3_real": pred_error(models["wm3"], data, rows, device, 3).mean().item(),
             "wm3_wrong": pred_error(models["wm3"], data, rows, device, 3, wrong).mean().item(),
             "copy_now_head": copy1, "wm1_real": pred_error(models["wm1"], data, rows, device, 1).mean().item()}
        res["A"][label] = r
        print(f"  {label:8s} 3 cams: copy {r['copy_now_3cam']:.4f}  blind {r['blind3']:.4f}  real {r['wm3_real']:.4f}  "
              f"wrong {r['wm3_wrong']:.4f} | head only: copy {r['copy_now_head']:.4f}  real {r['wm1_real']:.4f}")

    print("\nB. real actions predict the future better than slightly changed ones (50% = can't tell)")
    rows = data.rows("test")
    rows = rows[(data.frame[rows] >= 4)]
    for name in ("wm3", "wm1"):
        cams = models[name].cells // CELLS
        base = pred_error(models[name], data, rows, device, cams)
        real = data.actions[rows]
        out = {}
        for s in (0.013, 0.03, 0.1):
            ch = real.clone()
            ch[:, :, ARM] += torch.randn(len(rows), 1, len(ARM), generator=gen) * s * (torch.arange(1, 9) / 8)[None, :, None]
            out[f"arm {s}"] = (base < pred_error(models[name], data, rows, device, cams, ch)).float().mean().item()
        moving = (real[:, :, GRIP].amax(1) - real[:, :, GRIP].amin(1)).amax(1) > 0.1
        for k in (-4, -2, 2, 4):  # shift the gripper inside the window (edge values held)
            idx = (torch.arange(8) + k).clamp(0, 7)
            ch = real.clone()
            ch[:, :, GRIP] = real[:, idx][:, :, GRIP]
            e = pred_error(models[name], data, rows, device, cams, ch)
            out[f"gripper {'earlier' if k > 0 else 'later'} {abs(k)}"] = (base[moving] < e[moving]).float().mean().item()
        res["B"][name] = out | {"n_gripper_samples": int(moving.sum())}
        print(f"  {name}: " + "  ".join(f"{k} {100 * v:.0f}%" for k, v in out.items()) + f"  ({int(moving.sum())} gripper samples)")

    print("\nC. judging held-out attempts: AUC for 'this attempt will succeed' (0.5 = no idea, 1 = perfect)")
    rows = data.rows("test", kind=1)
    now = value_of(value, data.grid[rows].float(), device)
    with torch.no_grad():
        imagined = torch.cat([models["wm3"](**{k: v for k, v in data.batch(rows[i:i + 256], device, 3).items()
                                              if k != "target"}).cpu() for i in range(0, len(rows), 256)])
    pred = value_of(value, imagined, device)
    true_next = value_of(value, data.grid[data.next[rows]].float(), device)
    lab = data.success[rows]
    res["C"] = {"now": auc(now, lab), "imagined_8_ahead": auc(pred, lab), "real_8_ahead": auc(true_next, lab),
                "n": len(rows), "success_rate": lab.mean().item()}
    print(f"  value of the scene now {res['C']['now']:.3f} | imagined 8 steps ahead {res['C']['imagined_8_ahead']:.3f} | "
          f"real 8 steps ahead {res['C']['real_8_ahead']:.3f}   ({len(rows)} samples)")
    (args.out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
