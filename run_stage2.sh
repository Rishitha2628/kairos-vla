#!/bin/bash
# World-model Stage 2: record 200 lift_pot attempts (with mistakes), then encode the demos' wrist cameras.
cd "$(dirname "$0")"
.venv/bin/python auto_resume.py --done "outputs/eval/.never" -- \
  .venv/bin/python thermal_guard.py -- .venv/bin/python collect_rollouts.py \
  --model-dir outputs/smolvla_baseline/final --task lift_pot --episodes 200
for cam in cam_left_wrist cam_right_wrist; do
  [ -f "data/robotwin_clean_2task/vjepa2/${cam}_vitg.pt" ] && continue
  .venv/bin/python auto_resume.py --done "data/robotwin_clean_2task/vjepa2/${cam}_vitg.pt" -- \
    .venv/bin/python thermal_guard.py -- .venv-train/bin/python step5_vjepa_embeddings.py --camera "$cam"
done
