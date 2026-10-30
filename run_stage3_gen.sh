#!/bin/bash
# Stage 3 at new pot positions (50 scenes each): SmolVLA + world-model reach planner, then the same
# intercept-and-close without the world model (control). Baseline SmolVLA alone: outputs/eval/smolvla_baseline_smooth_ns1/lift_pot_position.json
cd "$(dirname "$0")"
for R in 8 0; do
  .venv/bin/python auto_resume.py --done outputs/eval/.never -- \
    .venv/bin/python thermal_guard.py -- .venv/bin/python stage3_eval.py \
    --model-dir outputs/smolvla_baseline/final --world-model outputs/world_model3 --variant position --episodes 50 --refine-steps $R
done
