"""Serve a trained SmolVLA policy to the simulator (runs in .venv-train, Python 3.12).

    ~/kairos/.venv-train/bin/python ~/kairos/policy_server.py --model-dir outputs/smolvla_baseline/final --port 5555

--model-dir may be a finished model (final/: full weights + processors) or a rolling training checkpoint
(latest/: only trainable weights in train_state.pt; rebuilt on top of the base model exactly as
train_smolvla.py builds it).

Requests (policy_link messages):
  {"cmd": "reset", "noise_seed": int (optional)}          -> {"ok": True}   (seeds the flow-matching noise)
  {"cmd": "act", "images": {cam: uint8 HxWx3}, "state": float32[14], "task": str, "n_samples": int (optional)}
                                                          -> {"actions": float32[chunk_size, 14]}
      with n_samples > 1 also "samples": float32[n, chunk_size, 14]; "actions" is samples[0]
      with --planner DIR and n_samples > 1, "actions" is the plan the world model rates best (planner.py),
      and the reply also has "scores" (n,) and "chosen"; --candidates varied also scores shifted-grasp and
      arm-drift versions of the plans (planner.varied_candidates) and adds "chosen_kind"
  {"cmd": "refine", "images": ..., "state": ...}          -> {"actions": float32[8, 14], "score", "stay"}
      with --reach-planner DIR: a small arm correction toward "just before grasping" (planner.ReachPlanner)
  {"cmd": "close"}                                        -> server exits
"""
import argparse
import json
import socket
from pathlib import Path

import numpy as np
import torch

import policy_link

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def load_final(model_dir, device):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, policy_action_to_transition, transition_to_batch, transition_to_policy_action,
    )

    policy = SmolVLAPolicy.from_pretrained(str(model_dir))
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(model_dir), config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(model_dir), config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    return policy, pre, post


def load_training_checkpoint(model_dir, device):
    from lerobot.configs.policies import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    targs = json.loads((model_dir / "train_args.json").read_text())
    dataset = LeRobotDataset("local/robotwin_clean_2task", root=targs["dataset_root"])
    feats = dataset.meta.features
    h, w, _ = feats[f"observation.images.{CAMERAS[0]}"]["shape"]
    chunk = int(targs["chunk_size"])
    img = int(targs["img"])
    cfg = SmolVLAConfig(
        input_features={
            **{f"observation.images.{c}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w)) for c in CAMERAS},
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=tuple(feats["observation.state"]["shape"])),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=tuple(feats["action"]["shape"]))},
        chunk_size=chunk, n_action_steps=chunk, resize_imgs_with_padding=(img, img),
        freeze_vision_encoder=True, train_expert_only=True, optimizer_lr=float(targs["lr"]), device=device,
    )
    policy = SmolVLAPolicy.from_pretrained(targs["base"], config=cfg)
    state = torch.load(model_dir / "train_state.pt", map_location="cpu", weights_only=False)
    missing = policy.load_state_dict(state["trainable"], strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    print(f"loaded training checkpoint at step {state['step']}", flush=True)
    pre, post = make_smolvla_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)
    return policy, pre, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--planner", type=Path, default=None, help="world-model dir: pick the best of n_samples plans")
    ap.add_argument("--candidates", choices=["samples", "varied"], default="samples")
    ap.add_argument("--reach-planner", type=Path, default=None, help="world-model dir for the 'refine' command")
    args = ap.parse_args()

    if (args.model_dir / "model.safetensors").exists():
        policy, pre, post = load_final(args.model_dir, args.device)
    else:
        policy, pre, post = load_training_checkpoint(args.model_dir, args.device)
    policy.config.device = args.device
    policy.to(args.device).eval()
    for pipeline in (pre, post):
        for step in getattr(pipeline, "steps", []):
            if getattr(step, "device", None) not in (None, "cpu"):
                step.device = args.device

    planner = None
    if args.planner is not None:
        from planner import WorldModelPlanner
        planner = WorldModelPlanner(args.planner, args.device)
        print(f"planner: world model from {args.planner}", flush=True)

    reach = None
    if args.reach_planner is not None:
        from planner import ReachPlanner
        reach = ReachPlanner(args.reach_planner, args.device)
        print(f"reach planner: world model from {args.reach_planner}", flush=True)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", args.port))
    server.listen(1)
    print(f"READY {args.port}", flush=True)
    conn, _ = server.accept()
    with conn:
        while True:
            msg = policy_link.recv(conn)
            if msg["cmd"] == "reset":
                policy.reset()
                if msg.get("noise_seed") is not None:
                    torch.manual_seed(msg["noise_seed"])
                policy_link.send(conn, {"ok": True})
            elif msg["cmd"] == "act":
                batch = {f"observation.images.{c}": torch.from_numpy(msg["images"][c]).permute(2, 0, 1)
                         .float().div(255).unsqueeze(0) for c in CAMERAS}
                batch["observation.state"] = torch.from_numpy(msg["state"]).float().unsqueeze(0)
                batch["task"] = [msg["task"]]
                batch = pre(batch)
                n = int(msg.get("n_samples", 1))
                if n > 1:  # same observation, n independent noise draws
                    batch = {k: v.repeat(n, *[1] * (v.dim() - 1)) if torch.is_tensor(v) and v.shape[:1] == (1,) else v
                             for k, v in batch.items()}
                with torch.inference_mode():
                    chunk = policy.predict_action_chunk(batch)  # (n, chunk_size, 14), normalized
                samples = post(chunk).float().cpu().numpy().astype(np.float32)
                reply = {"actions": samples[0]}
                if n > 1:
                    reply["samples"] = samples
                if planner is not None and n > 1:
                    if args.candidates == "varied":
                        from planner import varied_candidates
                        cands, kinds = varied_candidates(samples)
                    else:
                        cands, kinds = samples, ["plan"] * len(samples)
                    scores = planner.score(msg["images"], msg["state"], cands)
                    reply["chosen"] = int(np.argmax(scores))
                    reply["chosen_kind"] = kinds[reply["chosen"]]
                    reply["scores"] = scores.astype(np.float32)
                    reply["actions"] = cands[reply["chosen"]]
                    # hand cached blocks back: the simulator's renderer shares this 6 GB GPU
                    torch.cuda.empty_cache()
                policy_link.send(conn, reply)
            elif msg["cmd"] == "refine":
                actions, best, stay = reach.refine(msg["images"], msg["state"])
                policy_link.send(conn, {"actions": actions, "score": best, "stay": stay})
            elif msg["cmd"] == "close":
                policy_link.send(conn, {"ok": True})
                break


if __name__ == "__main__":
    main()
