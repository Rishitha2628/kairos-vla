"""World model on V-JEPA 2 summaries: (scene now, robot state, next H actions) -> scene H steps later.

The scene is the Step 5 V-JEPA 2 grid of the head camera (4x4 cells x 1408), standardized per dimension.
Actions are absolute joint targets (action[t] == state[t+1]); the model sees them relative to the current
state, so "move left by 0.1 rad" looks the same wherever the arm starts.

    tokens: 16 grid cells + 1 state + H actions  ->  transformer  ->  16 grid cells (as a change from now)

WorldModelData loads everything the model trains on (small: no images, the grid stays fp16).
"""
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

KAIROS = Path(__file__).resolve().parent
DATASET = KAIROS / "data" / "robotwin_clean_2task"
EMBEDDINGS = DATASET / "vjepa2" / "cam_high_vitg.pt"
CELLS, DIM = 16, 1408


class WorldModelData:
    """All frames, with per-frame grid (standardized), state, action, episode and task."""

    def __init__(self, embeddings=EMBEDDINGS, dataset=DATASET, holdout_per_task=5, seed=0):
        d = torch.load(embeddings, map_location="cpu", weights_only=True)
        grid = d["grid"].flatten(1, 2)  # (N, 16, 1408) fp16
        # mean/std and standardizing in pieces: the whole grid in fp32 would be another 1 GB per copy
        s, s2 = torch.zeros(CELLS, DIM, dtype=torch.float64), torch.zeros(CELLS, DIM, dtype=torch.float64)
        for part in grid.split(1024):
            s += part.double().sum(0)
            s2 += (part.double() ** 2).sum(0)
        n = len(grid)
        self.mean = (s / n).float()
        self.std = ((s2 / n - (s / n) ** 2) * n / (n - 1)).clamp_min(0).sqrt().float().clamp_min(1e-4)
        for part in grid.split(1024):  # in place, stays fp16
            part.copy_(((part.float() - self.mean) / self.std).half())
        self.grid = grid
        table = pq.read_table(next((dataset / "data").rglob("*.parquet")),
                              columns=["action", "observation.state", "episode_index", "frame_index", "index"])
        assert table["index"].to_pylist() == list(range(len(table))), "parquet rows must be in dataset order"
        assert table["episode_index"].to_pylist() == d["episode_index"].tolist(), "embeddings misaligned"
        assert table["frame_index"].to_pylist() == d["frame_index"].tolist(), "embeddings misaligned"
        self.action = torch.tensor(np.stack(table["action"].to_numpy(zero_copy_only=False)), dtype=torch.float32)
        self.state = torch.tensor(np.stack(table["observation.state"].to_numpy(zero_copy_only=False)), dtype=torch.float32)
        self.episode = d["episode_index"].long()
        self.frame = d["frame_index"].long()

        tasks = {r["episode_index"]: r["robotwin_task"] for r in json.loads((dataset / "meta" / "instructions.json").read_text())}
        self.task_names = sorted(set(tasks.values()))
        self.task = torch.tensor([self.task_names.index(tasks[int(e)]) for e in self.episode])
        rng = random.Random(seed)
        self.holdout = sorted(e for t in self.task_names
                              for e in rng.sample(sorted(e for e in tasks if tasks[e] == t), holdout_per_task))
        self.length = torch.zeros(int(self.episode.max()) + 1, dtype=torch.long)
        self.length.index_add_(0, self.episode, torch.ones_like(self.episode))
        # how far into the demo each frame is (0 = start, 1 = last frame): the label for the progress score
        self.progress = self.frame.float() / (self.length[self.episode] - 1).float()

        delta = self.action - self.state  # one-step change commanded by each action
        self.delta_std = delta.std(0).clamp_min(1e-4)
        self.state_mean, self.state_std = self.state.mean(0), self.state.std(0).clamp_min(1e-4)

    def starts(self, horizon, split):
        """Frame rows i with i+horizon in the same demo, for 'train' or 'test' (held-out demos)."""
        ok = self.frame + horizon < self.length[self.episode]
        held = torch.isin(self.episode, torch.tensor(self.holdout))
        return torch.nonzero(ok & (held if split == "test" else ~held)).squeeze(1)

    def batch(self, rows, horizon, device):
        """Inputs and target for starting rows: actions rows..rows+H-1 lead to the frame rows+H."""
        steps = rows[:, None] + torch.arange(horizon)[None]
        return {
            "grid": self.grid[rows].to(device).float(),
            "state": ((self.state[rows] - self.state_mean) / self.state_std).to(device),
            "actions": self.encode_actions(self.action[steps], self.state[rows]).to(device),
            "target": self.grid[rows + horizon].to(device).float(),
        }

    def encode_actions(self, actions, state):
        """actions (B, H, 14) absolute, state (B, 14) -> (B, H, 14) relative to the current state, scaled."""
        return (actions - state[:, None]) / self.delta_std / 10


class WorldModel(nn.Module):
    """n_cams cameras: the grid input/output is (B, 16 * n_cams, 1408), cameras one after another."""

    def __init__(self, horizon, d=384, layers=4, heads=6, dropout=0.1, use_actions=True, n_cams=1):
        super().__init__()
        self.horizon, self.use_actions, self.cells = horizon, use_actions, CELLS * n_cams
        self.cell_in = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, d))
        self.state_in = nn.Linear(14, d)
        self.action_in = nn.Linear(14, d)
        self.pos = nn.Parameter(torch.zeros(1, self.cells + 1 + horizon, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, DIM))
        nn.init.zeros_(self.out[1].weight)  # starts as "nothing changes"
        nn.init.zeros_(self.out[1].bias)

    def forward(self, grid, state, actions):
        """grid (B, cells, 1408) standardized, state (B, 14), actions (B, H, 14) -> predicted grid (B, cells, 1408)."""
        if not self.use_actions:
            actions = torch.zeros_like(actions)
        x = torch.cat([self.cell_in(grid), self.state_in(state)[:, None], self.action_in(actions)], 1) + self.pos
        x = self.blocks(x)
        return grid + self.out(x[:, :self.cells])

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class ValueModel(nn.Module):
    """Scene summary (B, cells, 1408) -> value in [0, 1]: how soon the episode will succeed (0 = it won't)."""

    def __init__(self, n_cams=1, d=256, layers=2, heads=4, dropout=0.1):
        super().__init__()
        cells = CELLS * n_cams
        self.cell_in = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, cells + 1, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, grid):
        x = torch.cat([self.cls.expand(len(grid), -1, -1), self.cell_in(grid)], 1) + self.pos
        return self.out(self.blocks(x)[:, 0]).squeeze(1)  # logit
