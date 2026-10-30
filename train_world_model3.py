"""World-model generalisation, Stage 2: a world model that works at new pot positions, trained without any
expert demos there.

    nice -n 19 ~/kairos/.venv/bin/python ~/kairos/thermal_guard.py -- \
        ~/kairos/.venv-train/bin/python ~/kairos/train_world_model3.py

Data (3 cameras, V-JEPA 2 4x4 grids, horizon 8, same model as Stage 2 of the planner work):
  demos     the 50 lift_pot demos (normal positions; every frame)
  attempts  the 200 recorded SmolVLA attempts with small mistakes (normal positions; every 4th frame)
  play      the 200 play recordings: pot anywhere in +-10 cm / +-45 deg, big random arm drifts, no expert
Held out: 5 demos, 40 attempts, 30 play recordings (mostly new positions: only ~1/8 of the wide range is
inside the demos' range).
The grids are memory-mapped from the encoding files and standardized per batch: in RAM all together they
would be ~5 GB.

Tests on held-out play (what the planner needs; the goal score is stage0_analyze.py's, lower = hands look
closer to the demos' "just before grasping"):
  A  prediction error 8 steps ahead: copy-now / blind (no actions) / real actions / someone else's actions
  B  does the imagined future know where the hands end up? Spearman between the goal score of the imagined
     scene and the true hand-handle distance at that time, after 1 hop (8 steps) and 3 hops (24 steps,
     feeding the model its own predictions), next to the score of the real future (upper bound) and of the
     scene now (what you'd get without imagining)
  C  does it predict the change? Spearman between imagined change in goal score and the true change in
     hand distance over 24 steps, overall and when a hand is 2-6 cm from its handle
Output: outputs/world_model3/{wm3.pt, blind3.pt, normalization.pt, results.json}
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

import stage0_analyze
from world_model import CELLS, DATASET, DIM, KAIROS, WorldModel

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
ARM = list(range(6)) + list(range(7, 13))


class Data:
    """Rows of demos, attempts and play (kind 0/1/2). Grids stay in their (memory-mapped) source files."""

    def __init__(self, horizon=8, seed=0, holdout=(5, 40, 30)):
        H = horizon
        tasks = {r["episode_index"]: r["robotwin_task"] for r in json.loads((DATASET / "meta" / "instructions.json").read_text())}
        lift = sorted(e for e in tasks if tasks[e] == "lift_pot")
        demo = {c: torch.load(DATASET / "vjepa2" / f"{c}_vitg.pt", weights_only=True, mmap=True) for c in CAMERAS}
        self.sources = [  # per source: {camera: (N, 4, 4, 1408) fp16}
            {c: demo[c]["grid"] for c in CAMERAS},
            torch.load(KAIROS / "outputs" / "rollouts" / "lift_pot" / "vjepa2_stride4.pt", weights_only=True, mmap=True),
            torch.load(KAIROS / "outputs" / "play" / "lift_pot" / "vjepa2_stride4.pt", weights_only=True, mmap=True),
        ]
        # standardization: per camera and dimension, from the demos (as the planner's world model)
        self.mean, self.std = [], []
        for c in CAMERAS:
            g = demo[c]["grid"].flatten(1, 2)
            s, s2, n = torch.zeros(CELLS, DIM, dtype=torch.float64), torch.zeros(CELLS, DIM, dtype=torch.float64), 0
            for part in g.split(1024):
                s += part.double().sum(0)
                s2 += (part.double() ** 2).sum(0)
                n += len(part)
            self.mean.append((s / n).float())
            self.std.append(((s2 / n - (s / n) ** 2).clamp_min(0).sqrt()).float().clamp_min(1e-4))

        src, idx, kind, seq, frame, state, nxt, actions, dist, test = [], [], [], [], [], [], [], [], [], []

        def add(k, s, rows_idx, frames, st, act, d, held):
            """One sequence: rows_idx into source k, frames (n,), state per row, actions per step, dist per step."""
            base = len(src)
            pos = {int(f): i for i, f in enumerate(frames)}
            for i, f in enumerate(frames):
                f = int(f)
                src.append(k)
                idx.append(int(rows_idx[i]))
                kind.append(k)
                seq.append(s)
                frame.append(f)
                state.append(st[f])
                j = pos.get(f + H)
                nxt.append(base + j if j is not None else -1)
                actions.append(act[f:f + H] if j is not None else torch.zeros(H, 14))
                dist.append(d[f] if d is not None else torch.full((2,), float("nan")))
                test.append(held)

        rng = random.Random(seed)
        demo_held = set(rng.sample(lift, holdout[0]))
        table = pq.read_table(next((DATASET / "data").rglob("*.parquet")),
                              columns=["action", "observation.state", "episode_index", "frame_index"])
        d_action = torch.tensor(np.stack(table["action"].to_numpy(zero_copy_only=False)), dtype=torch.float32)
        d_state = torch.tensor(np.stack(table["observation.state"].to_numpy(zero_copy_only=False)), dtype=torch.float32)
        d_ep = torch.tensor(table["episode_index"].to_numpy())
        d_fr = torch.tensor(table["frame_index"].to_numpy())
        emb_ep = demo[CAMERAS[0]]["episode_index"]
        assert torch.equal(emb_ep, d_ep) and torch.equal(demo[CAMERAS[0]]["frame_index"], d_fr), "embeddings misaligned"
        for e in lift:
            rows = torch.nonzero(d_ep == e).squeeze(1)
            add(0, e, rows, d_fr[rows], d_state[rows], d_action[rows], None, e in demo_held)

        for k, (root, n_held) in ((1, (KAIROS / "outputs" / "rollouts" / "lift_pot", holdout[1])),
                                  (2, (KAIROS / "outputs" / "play" / "lift_pot", holdout[2]))):
            emb = self.sources[k]
            seeds = sorted(set(emb["seed"].tolist()))
            held = set(random.Random(seed).sample(seeds, n_held))
            all_seed, all_frame = emb["seed"], emb["frame"]
            for s in seeds:
                rows = torch.nonzero(all_seed == s).squeeze(1)
                meta = np.load(root / f"seed{s}" / "meta.npz")
                d = torch.from_numpy(meta["dist"]) if "dist" in meta else None
                add(k, s, rows, all_frame[rows], torch.from_numpy(meta["state"]),
                    torch.from_numpy(meta["action"]), d, s in held)

        self.src, self.idx = torch.tensor(src), torch.tensor(idx)
        self.kind, self.seq, self.frame = torch.tensor(kind), torch.tensor(seq), torch.tensor(frame)
        self.state, self.next = torch.stack(state).float(), torch.tensor(nxt)
        self.actions, self.dist, self.test = torch.stack(actions).float(), torch.stack(dist).float(), torch.tensor(test)
        ok = self.next >= 0
        self.delta_std = (self.actions[ok] - self.state[ok][:, None]).flatten(0, 1).std(0).clamp_min(1e-4)
        self.state_mean, self.state_std = self.state.mean(0), self.state.std(0).clamp_min(1e-4)
        self.horizon = H
        self.mean_all, self.std_all = torch.cat(self.mean), torch.cat(self.std)  # (48, 1408)

    def rows(self, split, kind=None, with_next=True):
        m = self.test if split == "test" else ~self.test
        if kind is not None:
            m = m & (self.kind == kind)
        if with_next:
            m = m & (self.next >= 0)
        return torch.nonzero(m).squeeze(1)

    def grids(self, rows):
        """(B, 48, 1408) standardized float, cameras one after another."""
        out = torch.empty(len(rows), CELLS * len(CAMERAS), DIM)
        for k, source in enumerate(self.sources):
            sel = torch.nonzero(self.src[rows] == k).squeeze(1)
            if len(sel):
                i = self.idx[rows[sel]]
                out[sel] = torch.cat([source[c][i].flatten(1, 2).float() for c in CAMERAS], 1)
        return (out - self.mean_all) / self.std_all

    def encode_actions(self, actions, state):
        return (actions - state[:, None]) / self.delta_std / 10

    def batch(self, rows, device, actions=None):
        acts = self.actions[rows] if actions is None else actions
        return {
            "grid": self.grids(rows).to(device),
            "state": ((self.state[rows] - self.state_mean) / self.state_std).to(device),
            "actions": self.encode_actions(acts, self.state[rows]).to(device),
            "target": self.grids(self.next[rows].clamp_min(0)).to(device),
        }


def cosine_lr(opt, steps):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, s / 200) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))


def train_wm(data, use_actions, args, device, name):
    torch.manual_seed(args.seed)
    model = WorldModel(data.horizon, use_actions=use_actions, n_cams=len(CAMERAS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    rows, test = data.rows("train"), data.rows("test", kind=2)
    sched = cosine_lr(opt, args.epochs * math.ceil(len(rows) / args.batch_size))
    best, best_state, t0 = float("inf"), None, time.time()
    for epoch in range(args.epochs):
        model.train()
        for chunk in rows[torch.randperm(len(rows))].split(args.batch_size):
            b = data.batch(chunk, device)
            loss = F.mse_loss(model(b["grid"], b["state"], b["actions"]), b["target"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        if epoch % 2 == 1 or epoch == args.epochs - 1:
            err = pred_error(model, data, test, device).mean().item()
            if err < best:
                best, best_state = err, {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  {name:6s} epoch {epoch + 1:3d}/{args.epochs}  train {loss.item():.4f}  held-out play {err:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def pred_error(model, data, rows, device, actions=None):
    model.eval()
    out = []
    for i in range(0, len(rows), 256):
        r = rows[i:i + 256]
        b = data.batch(r, device, None if actions is None else actions[i:i + 256])
        out.append(((model(b["grid"], b["state"], b["actions"]) - b["target"]) ** 2).mean((1, 2)).cpu())
    model.train()
    return torch.cat(out)


@torch.no_grad()
def imagine(model, data, rows, hops, device):
    """Feed the model its own predictions for `hops` x 8 real steps -> standardized grids (B, 48, 1408)."""
    out = []
    for i in range(0, len(rows), 256):
        r = rows[i:i + 256]
        grid = data.grids(r).to(device)
        cur = r
        for h in range(hops):
            st = data.state[cur]
            grid = model(grid, ((st - data.state_mean) / data.state_std).to(device),
                         data.encode_actions(data.actions[cur], st).to(device))
            cur = data.next[cur]
        out.append(grid.cpu())
    return torch.cat(out)


def goal_score(grid_std, data, goals):
    """Standardized grids (B, 48, 1408) -> stage0 goal score (wrist cameras, raw layer-normed features)."""
    raw = grid_std * data.std_all + data.mean_all
    emb = {c: {"grid": raw[:, k * CELLS:(k + 1) * CELLS]} for k, c in enumerate(CAMERAS) if c in stage0_analyze.WRISTS}
    # small chunks: the distance to every goal frame is (chunk, 100, 16, 1408) floats
    return np.concatenate([stage0_analyze.goal_score({c: {"grid": e["grid"][i:i + 32]} for c, e in emb.items()}, goals, "grid")
                           for i in range(0, len(raw), 32)])


def chain_end(data, rows, hops):
    """Row `hops` x 8 steps later, -1 where the sequence ends before."""
    cur = rows.clone()
    for _ in range(hops):
        cur = torch.where(cur >= 0, data.next[cur.clamp_min(0)], cur)
    return cur


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=KAIROS / "outputs" / "world_model3")
    args = p.parse_args()
    device = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    data = Data()
    for split in ("train", "test"):
        print(f"{split}: " + ", ".join(f"{len(data.rows(split, k))} {n}" for k, n in enumerate(("demo", "attempt", "play")))
              + " samples", flush=True)

    models = {"wm3": train_wm(data, True, args, device, "wm3"), "blind3": train_wm(data, False, args, device, "blind3")}
    for name, m in models.items():
        torch.save({"state_dict": m.state_dict(), "horizon": data.horizon, "n_cams": len(CAMERAS),
                    "use_actions": m.use_actions}, args.out / f"{name}.pt")
    torch.save({"mean": data.mean, "std": data.std, "delta_std": data.delta_std, "state_mean": data.state_mean,
                "state_std": data.state_std, "cameras": CAMERAS}, args.out / "normalization.pt")

    res = {"A": {}, "B": {}, "C": {}}
    gen = torch.Generator().manual_seed(0)
    rows = data.rows("test", kind=2)
    print(f"\nA. held-out play, prediction error 8 steps ahead (lower is better), {len(rows)} samples")
    g = data.grids(rows)
    wrong = data.actions[rows][torch.randperm(len(rows), generator=gen)]
    wrong = wrong - wrong[:, :1] + data.actions[rows][:, :1]  # someone else's movement, from this pose
    res["A"] = {"copy_now": ((g - data.grids(data.next[rows])) ** 2).mean((1, 2)).mean().item(),
                "blind": pred_error(models["blind3"], data, rows, device).mean().item(),
                "real": pred_error(models["wm3"], data, rows, device).mean().item(),
                "wrong": pred_error(models["wm3"], data, rows, device, wrong).mean().item()}
    print("  " + "  ".join(f"{k} {v:.4f}" for k, v in res["A"].items()))

    goals = stage0_analyze.demo_goals()
    print("\nB. does the imagined future know where the hands end up? Spearman(goal score, true hand distance)")
    for hops in (1, 3):
        end = chain_end(data, rows, hops)
        r = rows[end >= 0]
        end = end[end >= 0]
        err_end = data.dist[end].amax(1).numpy()
        s_now = goal_score(data.grids(r), data, goals)
        s_real = goal_score(data.grids(end), data, goals)
        s_im = goal_score(imagine(models["wm3"], data, r, hops, device), data, goals)
        s_blind = goal_score(imagine(models["blind3"], data, r, hops, device), data, goals)
        sp = stage0_analyze.spearman
        res["B"][f"{8 * hops}_steps"] = {"scene_now": sp(s_now, err_end), "imagined_blind": sp(s_blind, err_end),
                                         "imagined": sp(s_im, err_end), "real_future": sp(s_real, err_end),
                                         "n": len(r)}
        print(f"  {8 * hops:2d} steps: scene now {sp(s_now, err_end):.3f} | blind {sp(s_blind, err_end):.3f} | "
              f"imagined {sp(s_im, err_end):.3f} | real future {sp(s_real, err_end):.3f}   ({len(r)} samples)")
        if hops == 3:
            err_now = data.dist[r].amax(1).numpy()
            d_true = err_end - err_now
            near = (np.minimum(err_now, err_end) > 0.02) & (np.minimum(err_now, err_end) < 0.06)
            for label, m in (("all", np.ones_like(near)), ("2-6 cm", near)):
                c = {"imagined": sp(s_im[m] - s_now[m], d_true[m]), "blind": sp(s_blind[m] - s_now[m], d_true[m]),
                     "real_future": sp(s_real[m] - s_now[m], d_true[m]), "n": int(m.sum())}
                res["C"][label] = c
            print("\nC. 24 steps: Spearman(change in goal score, true change in hand distance)")
            for label, c in res["C"].items():
                print(f"  {label:7s} imagined {c['imagined']:.3f} | blind {c['blind']:.3f} | "
                      f"real future {c['real_future']:.3f}   ({c['n']} samples)")
    (args.out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
