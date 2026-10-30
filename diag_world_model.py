"""World-model Stage 1 check: can it tell the real actions from slightly different ones?

SmolVLA's candidate plans differ only a little (diversity check: ~0.01-0.03 rad, or gripper timing by a few
steps), so a useful picker must notice small differences, not just "a completely different movement".
On held-out demos, compares the prediction error with the real actions against the same actions changed by:
  arm offset s   every arm joint drifts linearly to a random offset of size s rad by the last step
  gripper +-k    gripper commands shifted k steps earlier/later (only samples where a gripper moves)
Reports how often the real actions predict the true future better (50% = can't tell them apart).

    ~/kairos/.venv-train/bin/python ~/kairos/diag_world_model.py
"""
import torch

from world_model import KAIROS, WorldModel, WorldModelData

ARM = list(range(6)) + list(range(7, 13))
GRIP = [6, 13]


@torch.no_grad()
def main():
    device = "cuda"
    data = WorldModelData()
    ck = torch.load(KAIROS / "outputs" / "world_model" / "world_model.pt", map_location="cpu", weights_only=True)
    H = ck["horizon"]
    model = WorldModel(H).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    rows = data.starts(H, "test")
    rows = rows[data.frame[rows] >= 4]  # room to shift the gripper earlier
    rows = rows[data.frame[rows] + H + 4 < data.length[data.episode[rows]]]
    gen = torch.Generator().manual_seed(0)

    def err(acts_abs, chunk):
        b = data.batch(chunk, H, device)
        pred = model(b["grid"], b["state"], data.encode_actions(acts_abs, data.state[chunk]).to(device))
        return ((pred - b["target"]) ** 2).mean((1, 2)).cpu()

    steps = rows[:, None] + torch.arange(H)[None]
    real = data.action[steps]
    base = torch.cat([err(real[i:i + 256], rows[i:i + 256]) for i in range(0, len(rows), 256)])
    print(f"{len(rows)} held-out samples, {H} steps ahead; real-action error {base.mean():.4f}")

    def report(name, changed, mask=None):
        e = torch.cat([err(changed[i:i + 256], rows[i:i + 256]) for i in range(0, len(rows), 256)])
        m = torch.ones(len(rows), dtype=torch.bool) if mask is None else mask
        win = (base[m] < e[m]).float().mean().item()
        print(f"  {name:16s} error {e[m].mean():.4f}   real actions better in {100 * win:4.0f}% of {int(m.sum())} samples")

    for s in (0.005, 0.013, 0.03, 0.1, 0.3):
        off = torch.randn(len(rows), 1, len(ARM), generator=gen) * s
        changed = real.clone()
        changed[:, :, ARM] += off * (torch.arange(1, H + 1) / H)[None, :, None]
        report(f"arm offset {s}", changed)

    moving = (real[:, :, GRIP].amax(1) - real[:, :, GRIP].amin(1)).amax(1) > 0.1  # a gripper opens/closes here
    for k in (-4, -2, 2, 4):
        changed = real.clone()
        changed[:, :, GRIP] = data.action[steps + k][:, :, GRIP]  # k > 0: gripper acts k steps earlier
        report(f"gripper {'earlier' if k > 0 else 'later'} {abs(k)}", changed, moving)


if __name__ == "__main__":
    main()
