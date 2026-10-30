"""World-model Stage 4: pick one of SmolVLA's plans by imagining how each would turn out.

Used by policy_server.py --planner outputs/world_model2. At every decision:
  1. the V-JEPA 2 encoder summarizes the 3 camera images (same pooling as training, standardized);
  2. for each candidate plan, the world model (wm3) imagines `hops` x 8 steps of its actions, feeding itself
     its own predictions (the state for the next hop is the last commanded action: action[t] == state[t+1]);
  3. the success scorer rates each imagined scene; the plan with the highest score is executed.
hops=3 imagines 24 steps, about the 25 actions that are executed before the next decision.
candidates="varied" adds clearly different options to SmolVLA's near-identical plans (varied_candidates):
the kinds of change the world model saw in the recorded attempts (collect_rollouts.py).

GPU memory: SmolVLA (1.1 GB) + the V-JEPA encoder (1.9 GB) + the simulator's renderer (2.3 GB) don't fit in
6 GB, so the encoder's 40 transformer blocks live in RAM and each is moved to the GPU only while it runs
(offload_blocks; ~2 GB copied per decision, under a second).
"""
from pathlib import Path

import numpy as np
import torch

from step5_vjepa_embeddings import encode, load_encoder
from world_model import CELLS, ValueModel, WorldModel

ARM = list(range(6)) + list(range(7, 13))
GRIP = [6, 13]


def varied_candidates(samples, n_base=4, shifts=(-4, 4), arm_size=0.05, n_exec=25):
    """samples (n, chunk, 14) -> (candidates, kinds): all samples, plus for the first n_base of them
    the gripper shifted by each of `shifts` steps (> 0 = earlier) and one random arm drift of ~arm_size rad
    (reached at step n_exec, as in collect_rollouts.perturb). Uses torch's RNG (seeded per episode)."""
    cands, kinds = [samples], ["plan"] * len(samples)
    chunk = samples.shape[1]
    for i in range(min(n_base, len(samples))):
        for k in shifts:
            c = samples[i].copy()
            c[:, GRIP] = samples[i][np.clip(np.arange(chunk) + k, 0, chunk - 1)][:, GRIP]
            cands.append(c[None])
            kinds.append(f"grip{k:+d}")
        c = samples[i].copy()
        ramp = np.minimum(np.arange(1, chunk + 1) / n_exec, 1.0)[:, None]
        c[:, ARM] += ramp * torch.randn(len(ARM)).numpy() * arm_size
        cands.append(c[None])
        kinds.append("arm")
    return np.concatenate(cands).astype(np.float32), kinds


def offload_blocks(encoder, device):
    """Keep encoder.blocks in RAM and stream each block's weights to `device` just for its forward.
    Not pinned: CUDA's pinned allocator rounds blocks up and used 3.9 GB of RAM for these 1.9 GB of weights."""
    for name, child in encoder.named_children():
        if name != "blocks":
            child.to(device)
    for block in encoder.blocks:
        cpu = dict(block.state_dict())
        tensors = dict(block.named_parameters()) | dict(block.named_buffers())

        def load(module, inputs, cpu=cpu, tensors=tensors):
            for k, t in tensors.items():
                t.data = cpu[k].to(device)

        def unload(module, inputs, output, cpu=cpu, tensors=tensors):
            for k, t in tensors.items():
                t.data = cpu[k]

        unload(block, None, None)
        block.register_forward_pre_hook(load)
        block.register_forward_hook(unload)
    return encoder


class WorldModelPlanner:
    def __init__(self, model_dir, device, hops=3):
        model_dir = Path(model_dir)
        self.device, self.hops, self.dtype = device, hops, torch.float16
        norm = torch.load(model_dir / "normalization.pt", weights_only=True)
        self.cameras = norm["cameras"]
        self.mean = torch.cat(norm["mean"]).to(device)  # (cells, 1408), cameras one after another
        self.std = torch.cat(norm["std"]).to(device)
        self.delta_std, self.state_mean, self.state_std = (norm[k].to(device) for k in ("delta_std", "state_mean", "state_std"))
        ck = torch.load(model_dir / "wm3.pt", weights_only=True)
        self.horizon = ck["horizon"]
        self.wm = WorldModel(self.horizon, n_cams=ck["n_cams"]).to(device).eval()
        self.wm.load_state_dict(ck["state_dict"])
        self.value = None
        if (model_dir / "value.pt").exists():  # the Stage 2 success scorer (not in world_model3)
            vk = torch.load(model_dir / "value.pt", weights_only=True)
            self.value = ValueModel(n_cams=vk["n_cams"]).to(device).eval()
            self.value.load_state_dict(vk["state_dict"])
        self.encoder = offload_blocks(load_encoder("cpu", self.dtype), device)

    @torch.no_grad()
    def summarize(self, images):
        """images: {camera: uint8 (H, W, 3)} -> standardized grid (cells, 1408)."""
        x = torch.stack([torch.from_numpy(images[c]).permute(2, 0, 1) for c in self.cameras]).float().div(255)
        _, grid = encode(self.encoder, x, self.device, self.dtype)  # (cams, 4, 4, 1408)
        return (grid.to(self.device).float().flatten(0, 2) - self.mean) / self.std

    @torch.no_grad()
    def score(self, images, state, plans):
        """plans: (n, chunk, 14) absolute joint targets -> (n,) imagined value after hops x horizon steps."""
        n = len(plans)
        grid = self.summarize(images)[None].expand(n, -1, -1)
        plans = torch.as_tensor(plans, dtype=torch.float32, device=self.device)
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None].expand(n, -1)
        for h in range(self.hops):
            acts = plans[:, h * self.horizon:(h + 1) * self.horizon]
            grid = self.wm(grid, (st - self.state_mean) / self.state_std, (acts - st[:, None]) / self.delta_std / 10)
            st = acts[:, -1]
        return torch.sigmoid(self.value(grid)).cpu().numpy()


class ReachPlanner(WorldModelPlanner):
    """Stage 3 (generalisation): search small arm corrections that make the wrist views look like the demos just
    before grasping (stage0_analyze.py's goal score), imagined by the world model trained on play data.

    A candidate is one offset `delta` of the 12 arm joints, reached linearly over the first `n_exec` steps and
    then held; the gripper keeps its current opening. Cross-entropy search: `n` samples per round, the `elite`
    best refit the sampling mean/std, `rounds` rounds; delta is clipped to +-max_delta rad, so a single decision
    can only move the hands a few cm (re-planned from real images after every n_exec steps).
    """

    def __init__(self, model_dir, device, n=64, rounds=3, elite=8, sigma=0.04, max_delta=0.08, n_exec=8):
        super().__init__(model_dir, device, hops=1)
        self.n, self.rounds, self.elite, self.sigma, self.max_delta, self.n_exec = n, rounds, elite, sigma, max_delta, n_exec
        goals_path = Path(model_dir) / "goals.pt"
        if not goals_path.exists():
            import stage0_analyze
            g = stage0_analyze.demo_goals()
            torch.save({c: g[c]["grid"].half() for c in stage0_analyze.WRISTS}, goals_path)
        g = torch.load(goals_path, weights_only=True)
        self.wrists = [self.cameras.index(c) for c in ("cam_left_wrist", "cam_right_wrist")]
        self.goals = [g[c].to(device).float() for c in ("cam_left_wrist", "cam_right_wrist")]  # (100, 16, 1408)

    def goal_score(self, grid_std):
        """(B, cells, 1408) standardized -> (B,) lower = closer to "just before grasping"."""
        raw = grid_std * self.std + self.mean
        total = 0
        for k, goals in zip(self.wrists, self.goals):
            w = raw[:, k * CELLS:(k + 1) * CELLS]
            total = total + torch.stack([(w - g).abs().mean((1, 2)) for g in goals], 1).min(1).values
        return total

    def plans(self, state, deltas):
        """deltas (B, 12) -> absolute action sequences (B, horizon, 14)."""
        ramp = torch.clamp(torch.arange(1, self.horizon + 1, device=self.device) / self.n_exec, max=1.0)
        acts = state[None, None].repeat(len(deltas), self.horizon, 1)
        acts[:, :, ARM] += ramp[None, :, None] * deltas[:, None]
        return acts

    @torch.no_grad()
    def refine(self, images, state, generator=None):
        """-> (actions (n_exec, 14) float32, imagined score of the best correction, imagined score of staying put)."""
        grid = self.summarize(images)[None]
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        def score(deltas):
            acts = self.plans(st, deltas)
            g = self.wm(grid.expand(len(deltas), -1, -1), ((st - self.state_mean) / self.state_std)[None].expand(len(deltas), -1),
                        (acts - st[None, None]) / self.delta_std / 10)
            return self.goal_score(g)

        mu = torch.zeros(len(ARM), device=self.device)
        sigma = torch.full_like(mu, self.sigma)
        stay = score(mu[None])[0]
        best, best_score = mu, stay
        for _ in range(self.rounds):
            noise = torch.randn(self.n, len(ARM), device=self.device, generator=generator)
            deltas = (mu + sigma * noise).clamp(-self.max_delta, self.max_delta)
            s = score(deltas)
            top = torch.argsort(s)[:self.elite]
            if s[top[0]] < best_score:
                best, best_score = deltas[top[0]], s[top[0]]
            mu, sigma = deltas[top].mean(0), deltas[top].std(0).clamp_min(0.005)
        acts = self.plans(st, best[None])[0, :self.n_exec]
        torch.cuda.empty_cache()
        return acts.cpu().numpy().astype(np.float32), float(best_score), float(stay)
