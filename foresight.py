"""Foresight auxiliary task for train_smolvla.py (Steps 6 and 7).

ForesightTargets: for each dataset frame, the Step 5 V-JEPA 2 embedding the policy should predict.
  vjepa     the same demo, `frames` ahead (clamped to the last frame)
  shuffled  a different demo of the same task (fixed random pairing), same frame offset, clamped
            to that demo's length. Same kind of target, same statistics, but not this scene's future.
  Targets are the 4x4 grid features (16 x 1408), standardized per dimension over the whole dataset.

ForesightHead: reads the action expert's output tokens through a forward hook on action_out_proj (the layer
that turns them into velocities), mean-pools the chunk, and predicts the target with a 2-layer MLP.
"""
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForesightTargets:
    def __init__(self, embeddings_path, dataset_root, mode, frames, seed):
        d = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        grid = d["grid"].float().flatten(1)  # (N, 16 * 1408)
        self.mean, self.std = grid.mean(0), grid.std(0).clamp_min(1e-4)
        self.grid = ((grid - self.mean) / self.std).half()
        self.dim = self.grid.shape[1]
        ep, fr = d["episode_index"].tolist(), d["frame_index"].tolist()
        self.episode_index, self.frame_index = ep, fr  # row i = dataset frame i; see check_alignment

        row_of = {(e, f): i for i, (e, f) in enumerate(zip(ep, fr))}
        length = {}
        for e, f in zip(ep, fr):
            length[e] = max(length.get(e, 0), f + 1)

        self.partner = {e: e for e in length}
        if mode == "shuffled":
            tasks = {r["episode_index"]: r["robotwin_task"]
                     for r in json.loads((Path(dataset_root) / "meta" / "instructions.json").read_text())}
            rng = random.Random(seed)
            for task in sorted(set(tasks.values())):
                eps = sorted(e for e in length if tasks[e] == task)
                shift = rng.randrange(1, len(eps))  # a random rotation: never paired with itself
                order = eps[:]
                rng.shuffle(order)
                self.partner.update({order[i]: order[(i + shift) % len(order)] for i in range(len(order))})
            assert all(self.partner[e] != e for e in length)

        rows = []
        for e, f in zip(ep, fr):
            p = self.partner[e]
            rows.append(row_of[(p, min(f + frames, length[p] - 1))])
        self.target_row = torch.tensor(rows)
        self.mode, self.frames = mode, frames

    def check_alignment(self, dataset, n=200, seed=0):
        """Row i of the embeddings must be dataset frame i (Step 5 encodes in dataset order)."""
        assert len(dataset) == len(self.target_row), (len(dataset), len(self.target_row))
        for i in random.Random(seed).sample(range(len(dataset)), n):
            item = dataset.hf_dataset[i]
            assert (int(item["episode_index"]), int(item["frame_index"])) == \
                   (self.episode_index[i], self.frame_index[i]), f"row {i} misaligned"

    def lookup(self, dataset_index):
        return self.grid[self.target_row[dataset_index.long()]].float()

    def describe(self):
        ahead = f"{self.frames} frames ahead"
        return (f"target = {'same demo' if self.mode == 'vjepa' else 'another demo of the same task'}, {ahead}, "
                f"{self.dim}-dim (4x4 x 1408) standardized, {len(self.target_row)} frames")


class ForesightHead(nn.Module):
    def __init__(self, policy, target_dim, hidden=1024):
        super().__init__()
        proj = policy.model.action_out_proj
        self.net = nn.Sequential(nn.LayerNorm(proj.in_features), nn.Linear(proj.in_features, hidden), nn.GELU(),
                                 nn.Linear(hidden, target_dim))
        self._features = None
        proj.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        if self.training:
            self._features = inputs[0]  # (B, chunk, expert_hidden), already float32

    def loss(self, target):
        assert self._features is not None, "policy.forward must run before head.loss"
        pred = self.net(self._features.mean(1))
        self._features = None
        return F.mse_loss(pred, target)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
