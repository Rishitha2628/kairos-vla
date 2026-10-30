# kairos: V-JEPA 2 world models for SmolVLA on RoboTwin 2.0

Can a world model built on [V-JEPA 2](https://github.com/facebookresearch/vjepa2) make a VLA
([SmolVLA](https://huggingface.co/lerobot/smolvla_base)) better, and above all make it work where its demos
never went? Bimanual aloha-agilex in [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin), task
`lift_pot` (plus `beat_block_hammer` for the first part), all on one laptop (RTX 3060 6 GB, 15 GB RAM).

## Results

SmolVLA, 20k steps × batch 8, trained on the 50 clean demos of each task. Evaluation follows RoboTwin's
protocol: seeds from 100000, only seeds the scripted expert can solve, 400-action limit, 25 of each
50-action chunk executed. `--smooth` filters each chunk and crossfades chunk switches.
Variants of `lift_pot`: **standard** (the demos' range: pot 0, ±5 cm, ±22.5°), **position** (the same pot at
poses outside that range, up to ±10 cm, ±45°), **hard** (unseen pots 1–6, ±10 cm, ±45°).

### 1. V-JEPA 2 foresight as an auxiliary loss: no effect

The action expert also predicts V-JEPA 2 features of the scene ~0.5 s ahead (training only).

| | baseline | foresight |
|---|---|---|
| lift_pot standard, smoothed, 150 episodes | 107 (71.3%) | 108 (72.0%) |
| beat_block_hammer standard, smoothed, 30 episodes | 18 (60.0%) | 22 (73.3%) |
| lift_pot hard, smoothed, 30 episodes | 1 (3.3%) | 1 (3.3%) |

An early lead on lift_pot (77% vs 65%, raw, on 30 and 57 episodes) vanished at 150 smoothed episodes. With the VLM frozen and the head
used only in training, the extra loss does not change what the policy does.

### 2. World model as a planner over SmolVLA's plans: no gain, then worse

- **Diversity check:** three reruns with different sampling noise solve 107 / 109 / 112 of the same 150
  scenes; picking the best of the three per scene would give 82%, so a perfect picker could gain ~9 pts.
- **World model** (`world_model.py`): 4×4 V-JEPA 2 grid per camera + robot state + the next 8 actions →
  the grid 8 steps later (transformer, residual). Trained on the demos plus 200 recorded SmolVLA attempts
  with deliberate mistakes (arm drift, gripper timing). With all 3 cameras it tells real actions from
  0.03 rad arm offsets 82% of the time. A success scorer on the imagined scene reaches AUC 0.87.
- **Planner** (`planner.py`): at every decision, imagine 24 steps of each candidate and run the best one.

| lift_pot standard, smoothed, same 150 scenes | success |
|---|---|
| SmolVLA alone (3 reruns) | 107 / 109 / 112 |
| planner, picks among 8 SmolVLA samples | 112 (74.7%) |
| planner, 20 candidates (8 samples + gripper ±4 steps + arm drifts) | **86 (57.3%)** |

With 20 candidates the planner chose the deliberately drifted arm plans 61% of the time (they are 20% of
the candidates) and lost 33–40 scenes the baseline solved: it exploits the world model's errors.

### 3. Generalising to new pot positions (in progress)

SmolVLA reproduces its demos: at **position** poses it solves **2 / 50 (4%)**. The goal here is to let a
world model reach those poses with no expert demos there.

- **Stage 0, goal signal** (`stage0_record.py`, `stage0_analyze.py`): wrist cameras move with the hands,
  so "hands at the handles" should look the same at any pot pose. On 15 scenes at new positions, the
  distance of the wrist views to the demos' last frames before grasping tracks the true hand–handle
  distance: Spearman 0.80; AUC 0.99 for "both hands within 2 cm" vs "a hand more than 5 cm off" (also
  on expert frames alone); in every expert run the best-scoring frame is within 3 cm (median 0.8 cm).
  Weaker between 2 and 6 cm.
- **Stage 1, play data** (`collect_rollouts.py --play --variant wide`): 200 recordings with the pot
  anywhere in ±10 cm / ±45° and large random arm drifts on SmolVLA's plans. No expert.
- **Stage 2, world model on play data** (`train_world_model3.py`): does the imagined future know where
  the hands end up at new positions? *Running.*
- **Stage 3, planner** that searches its own arm motions toward the goal signal, then (a) a fixed
  close-and-lift or (b) SmolVLA takes over the grasp. *Next.*

Result files: `results/eval/<run>/<task>_<variant>.json` (per episode), `results/world_model/`.

## Pipeline

```bash
bash get_demos.sh                                     # 50 clean demos per task
python step2_replay_demo.py                           # check replay in the sim
python step3_convert_to_lerobot.py                    # LeRobot dataset
python step5_vjepa_embeddings.py --camera cam_high    # V-JEPA 2 grids (also both wrist cameras)
python train_smolvla.py                               # baseline (--foresight for part 1)
python step4_eval.py --model-dir outputs/smolvla_baseline/final --task lift_pot --variant standard --smooth
```

Parts 2 and 3: `run_diversity.sh`, `run_stage2.sh`, `run_stage3.sh`, `run_position_test.sh`,
`run_stage0_analyze.sh`, `run_stage2_gen.sh` (each lists its commands).

| file | role |
|---|---|
| `step1_move_robot.py` … `step5_vjepa_embeddings.py` | sim setup, demo replay, LeRobot conversion, eval, V-JEPA 2 encoding |
| `train_smolvla.py`, `foresight.py` | SmolVLA training, optional V-JEPA 2 foresight head |
| `policy_server.py`, `policy_link.py` | SmolVLA runs in its own venv; the sim talks to it over a socket |
| `mplib_expert.py` | RoboTwin's scripted expert on mplib (CuRobo not installed) |
| `hard_variants.py` | standard / position / hard / wide pot placements |
| `action_smoothing.py` | chunk filtering and crossfade (`--smooth`) |
| `world_model.py`, `train_world_model*.py`, `diag_world_model.py` | world models and their tests |
| `collect_rollouts.py`, `encode_rollouts.py` | recorded attempts / play data and their V-JEPA 2 grids |
| `planner.py` | world-model planner used by `policy_server.py --planner` |
| `stage0_record.py`, `stage0_analyze.py` | goal-signal test at new pot positions |
| `thermal_guard.py`, `auto_resume.py` | pause when the laptop runs hot; restart after crashes |

## Setup

Two venvs: `.venv` (Python 3.10, RoboTwin + sapien, `robotwin_requirements.txt`) runs the sim;
`.venv-train` (Python 3.12, LeRobot at commit `fdbfc01`) trains and serves SmolVLA and V-JEPA 2.
RoboTwin at `RoboTwin/` (commit `ea8b211`) with `results/patches/robotwin_curobo_placeholder.patch`
applied. V-JEPA 2 ViT-g/16 256px weights come from `download_models.py`.
