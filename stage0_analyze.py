"""World-model generalisation, Stage 0 analysis: is "looks like the demos just before grasping" a goal that
works at new pot positions?

    nice -n 19 ~/kairos/.venv/bin/python ~/kairos/thermal_guard.py -- \
        ~/kairos/.venv-train/bin/python ~/kairos/stage0_analyze.py --variant position

Goal set: in each lift_pot demo, the last 2 frames before the grippers close on the handles (gripper still at
the half-open 0.5 it approaches with). The goal score of a frame (lower = closer to the goal) is, per wrist
camera, the distance of its V-JEPA 2 embedding to the nearest goal frame, summed over the two wrists:
  grid    mean |difference| of the 4x4 pooled grid (as the world model sees the scene)
  global  euclidean distance of the mean-pooled embedding
Recorded frames (stage0_record.py) carry the true distance of each hand to its handle; "hand error" = the
worse of the two hands. Only approach frames count (gripper >= 0.45: not yet closed).

Verdict needs, at new positions:
  * goal score rises with hand error (Spearman);
  * it tells "both hands within 2 cm" from "a hand more than 5 cm off" (AUC);
  * in each scene, the frame the score likes best is near the handles (hand error at the argmin),
    for the expert's run and against the policy's misses in the same scene.
Embeddings are cached in outputs/stage0/<variant>/vjepa2_wrists.pt.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torchcodec.decoders import VideoDecoder

from step5_vjepa_embeddings import KAIROS, encode, load_encoder

DATASET = KAIROS / "data" / "robotwin_clean_2task"
WRISTS = ["cam_left_wrist", "cam_right_wrist"]


def demo_goals():
    """-> {camera: {'grid': (G, 16, 1408), 'global': (G, 1408)}} from the lift_pot demos."""
    tasks = {r["episode_index"]: r["robotwin_task"] for r in json.loads((DATASET / "meta" / "instructions.json").read_text())}
    t = pq.read_table(next((DATASET / "data").rglob("*.parquet")),
                      columns=["observation.state", "episode_index", "frame_index"]).to_pandas()
    emb = {c: torch.load(DATASET / "vjepa2" / f"{c}_vitg.pt", weights_only=True) for c in WRISTS}
    ep, fr = emb[WRISTS[0]]["episode_index"].numpy(), emb[WRISTS[0]]["frame_index"].numpy()
    rows = []
    for e in sorted(e for e in tasks if tasks[e] == "lift_pot"):
        g = np.stack(t[t.episode_index == e].sort_values("frame_index")["observation.state"].values)[:, [6, 13]]
        closing = np.nonzero((g.max(1) < 0.45) & (np.arange(len(g)) > 10))[0]
        if not len(closing):
            continue
        k = closing[0]  # first frame the grippers are closing on the handles
        for f in (k - 2, k - 1):
            rows.append(np.nonzero((ep == e) & (fr == f))[0][0])
    rows = torch.tensor(rows)
    print(f"goal set: {len(rows)} demo frames from {len(rows) // 2} lift_pot demos")
    return {c: {"grid": emb[c]["grid"][rows].flatten(1, 2).float(), "global": emb[c]["global"][rows].float()}
            for c in WRISTS}


def encode_recordings(root, device):
    cache = root / "vjepa2_wrists.pt"
    runs = sorted(d for d in root.glob("seed*_*") if d.is_dir() and not d.name.endswith(".tmp"))
    done = torch.load(cache, weights_only=True) if cache.exists() else {}
    todo = [d for d in runs if d.name not in done]
    if todo:
        encoder = load_encoder(device, torch.float16)
        for i, d in enumerate(todo):
            out = {}
            for c in WRISTS:
                imgs = VideoDecoder(str(d / f"{c}.mp4"))[:].float().div(255)
                parts = [encode(encoder, imgs[j:j + 8], device, torch.float16) for j in range(0, len(imgs), 8)]
                out[c] = {"global": torch.cat([p[0] for p in parts]), "grid": torch.cat([p[1] for p in parts]).flatten(1, 2)}
            done[d.name] = out
            if i % 10 == 9 or i == len(todo) - 1:
                torch.save(done, cache)
                print(f"encoded {len(done)}/{len(runs)} runs", flush=True)
    return done


def goal_score(emb, goals, kind):
    total = 0
    for c in WRISTS:
        z, g = emb[c][kind].float(), goals[c][kind]
        d = (z[:, None] - g[None]).abs().mean((-1, -2)) if kind == "grid" else torch.cdist(z, g)
        total = total + d.min(1).values
    return total.numpy()


def auc(pos, neg):
    """P(score of a random positive < score of a random negative): lower score = closer to the goal."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None]
    return float((pos < neg).mean() + 0.5 * (pos == neg).mean()) if pos.size and neg.size else float("nan")


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="position")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    root = KAIROS / "outputs" / "stage0" / a.variant
    goals = demo_goals()
    emb = encode_recordings(root, a.device)
    results = {}
    for kind in ("grid", "global"):
        frames = {"expert": [], "policy": []}  # (score, hand error) of approach frames
        best = {"expert": [], "policy": []}    # hand error at each run's best-scoring frame
        pairs = []  # per scene: expert's best score < policy's best score?
        for name, e in emb.items():
            seed, who = name[4:].split("_")
            meta = np.load(root / name / "meta.npz")
            s, err, grip = goal_score(e, goals, kind), meta["dist"].max(1), meta["grip"].min(1)
            n = min(len(s), len(err))
            s, err, grip = s[:n], err[:n], grip[:n]
            ok = grip >= 0.45
            if not ok.any():
                continue
            frames[who].append(np.stack([s[ok], err[ok]], 1))
            i = np.argmin(np.where(ok, s, np.inf))
            best[who].append(err[i])
        for name in emb:
            seed, who = name[4:].split("_")
            if who != "expert" or f"seed{seed}_policy" not in emb:
                continue
            sc = {}
            for w in ("expert", "policy"):
                m = np.load(root / f"seed{seed}_{w}" / "meta.npz")
                s = goal_score(emb[f"seed{seed}_{w}"], goals, kind)
                n = min(len(s), len(m["dist"]))
                ok = m["grip"][:n].min(1) >= 0.45
                sc[w] = s[:n][ok].min() if ok.any() else np.inf
            pairs.append(sc["expert"] < sc["policy"])
        allf = np.concatenate(frames["expert"] + frames["policy"])
        near, far = allf[allf[:, 1] < 0.02, 0], allf[allf[:, 1] > 0.05, 0]
        r = {
            "frames": len(allf), "frames_within_2cm": len(near), "frames_beyond_5cm": len(far),
            "spearman_score_vs_hand_error": spearman(allf[:, 0], allf[:, 1]),
            "auc_within2cm_vs_beyond5cm": auc(near, far),
            "expert_best_frame_hand_error_cm_median": float(np.median(best["expert"]) * 100),
            "expert_best_frame_within_3cm": float(np.mean(np.array(best["expert"]) < 0.03)),
            "policy_best_frame_hand_error_cm_median": float(np.median(best["policy"]) * 100) if best["policy"] else None,
            "scenes_expert_grasp_beats_policy": float(np.mean(pairs)) if pairs else None,
            "n_scenes": len(pairs),
        }
        results[kind] = r
        print(f"\n[{kind}]")
        for k, v in r.items():
            print(f"  {k:40s} {v:.3f}" if isinstance(v, float) else f"  {k:40s} {v}")
    (root / "stage0_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
