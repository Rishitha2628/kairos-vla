#!/bin/bash
# Stage 3 option (b): 8 world-model corrections, then SmolVLA grasps itself (no fixed close).
cd "$(dirname "$0")"
.venv/bin/python auto_resume.py --done outputs/eval/.never -- \
  .venv/bin/python thermal_guard.py -- .venv/bin/python stage3_eval.py \
  --model-dir outputs/smolvla_baseline/final --world-model outputs/world_model3 --variant position --episodes 50 \
  --refine-steps 8 --grab policy
