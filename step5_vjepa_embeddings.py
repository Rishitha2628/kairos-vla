"""Step 5: encode every frame of the training demos with the frozen V-JEPA 2 encoder.

    ~/kairos/.venv/bin/python ~/kairos/thermal_guard.py -- ~/kairos/.venv-train/bin/python ~/kairos/step5_vjepa_embeddings.py

Encoder: V-JEPA 2 ViT-g/16 at 256 px (1B params, 1408-dim), taken from the V-JEPA 2-AC checkpoint already
in the HF cache (facebook/jepa-wms, vjepa2_ac_oss.pth.tar; its encoder is the frozen V-JEPA 2 encoder),
built with the vendored source from ~/vjepa2-maniskill and loaded with weights_only=True. fp16.

Each frame is encoded as a 2-frame clip of itself (tubelet 2), as V-JEPA 2-AC encodes single images, and the
256 patch tokens (16x16) are layer-normed, then pooled to keep the file small:
    global  (N, 1408)        mean over all patches
    grid    (N, 4, 4, 1408)  mean over 4x4 blocks of patches (keeps where things are)
N is the dataset's frame count, in dataset index order, so row i belongs to dataset frame i.

Output: <dataset-root>/vjepa2/<camera>_vitg.pt with global, grid, episode_index, frame_index and metadata.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

KAIROS = Path(__file__).resolve().parent
VJEPA_SRC = Path.home() / "vjepa2-maniskill" / "vjepa2_src"
CKPT_GLOB = "models--facebook--jepa-wms/snapshots/*/vjepa2_ac_oss.pth.tar"

IMG_SIZE, PATCH, TUBELET, DIM = 256, 16, 2, 1408
GRID = IMG_SIZE // PATCH  # 16 patches per side
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def load_encoder(device, dtype):
    sys.path.insert(0, str(VJEPA_SRC))
    from src.models import vision_transformer as vit

    ckpt = next((Path.home() / ".cache" / "huggingface" / "hub").glob(CKPT_GLOB))
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)  # build in fp16: the fp32 model alone would need 4 GB of RAM
    try:
        encoder = vit.vit_giant_xformers(patch_size=PATCH, img_size=(IMG_SIZE, IMG_SIZE), num_frames=64,
                                         tubelet_size=TUBELET, use_sdpa=True, use_SiLU=False, wide_SiLU=True,
                                         uniform_power=False, use_rope=True)
    finally:
        torch.set_default_dtype(prev)
    sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)["encoder"]
    sd = {k.replace("module.", "").replace("backbone.", ""): v.to(dtype) for k, v in sd.items()}
    missing, unexpected = encoder.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, unexpected[:5]
    print(f"encoder from {ckpt.name}: {sum(p.numel() for p in encoder.parameters()) / 1e6:.0f}M params, "
          f"{len(missing)} missing keys {missing[:3]}", flush=True)
    del sd
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


@torch.no_grad()
def encode(encoder, images, device, dtype):
    """images: (B, 3, H, W) float in [0, 1] -> global (B, DIM), grid (B, 4, 4, DIM), both fp16 on CPU."""
    x = F.interpolate(images.to(device).float(), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    x = ((x - MEAN.to(device)) / STD.to(device)).to(dtype)
    clip = x.unsqueeze(2).repeat(1, 1, TUBELET, 1, 1)  # (B, 3, 2, H, W)
    # RoPE angles are fp32; autocast reconciles them with fp16 values inside attention.
    with torch.autocast("cuda", dtype=dtype, enabled=device != "cpu"):
        h = encoder(clip)  # (B, 256, DIM)
    h = F.layer_norm(h.float(), (DIM,))
    grid = h.view(-1, GRID, GRID, DIM)
    grid = F.avg_pool2d(grid.permute(0, 3, 1, 2), kernel_size=GRID // 4).permute(0, 2, 3, 1)  # (B, 4, 4, DIM)
    return h.mean(1).half().cpu(), grid.half().cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=KAIROS / "data" / "robotwin_clean_2task")
    p.add_argument("--camera", default="cam_high")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="first N frames only (smoke test)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dtype = torch.float16
    out = a.out or a.dataset_root / "vjepa2" / f"{a.camera}_vitg.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = LeRobotDataset("local/robotwin_clean_2task", root=a.dataset_root)
    n = min(len(ds), a.limit or len(ds))
    key = f"observation.images.{a.camera}"
    encoder = load_encoder(a.device, dtype)

    glob_all, grid_all, ep_idx, fr_idx = [], [], [], []
    t0 = time.time()
    for start in range(0, n, a.batch_size):
        frames = [ds[i] for i in range(start, min(n, start + a.batch_size))]
        g, grid = encode(encoder, torch.stack([f[key] for f in frames]), a.device, dtype)
        glob_all.append(g)
        grid_all.append(grid)
        ep_idx += [int(f["episode_index"]) for f in frames]
        fr_idx += [int(f["frame_index"]) for f in frames]
        done = start + len(frames)
        if done % (a.batch_size * 100) == 0 or done == n:
            rate = done / (time.time() - t0)
            print(f"{time.strftime('%H:%M:%S')} {done}/{n} frames, {rate:.1f} frames/s, "
                  f"eta {(n - done) / rate / 60:.0f} min, gpu peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                  flush=True)

    torch.save({
        "global": torch.cat(glob_all), "grid": torch.cat(grid_all),
        "episode_index": torch.tensor(ep_idx), "frame_index": torch.tensor(fr_idx),
        "camera": a.camera, "encoder": "V-JEPA 2 ViT-g/16 256px (from vjepa2_ac_oss.pth.tar)",
        "preprocess": "resize to 256x256, ImageNet norm, frame repeated x2 (tubelet), layer_norm, mean-pooled",
        "fps": ds.meta.fps,
    }, out)
    print(f"saved {out}: {n} frames, global {tuple(torch.cat(glob_all).shape)}, "
          f"{out.stat().st_size / 2**20:.0f} MB, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
