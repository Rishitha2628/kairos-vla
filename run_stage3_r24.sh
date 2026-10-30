#!/bin/bash
# Stage 3 with a longer reach phase: up to 24 world-model corrections (reach8 used all 8 and was still improving).
cd "$(dirname "$0")"
.venv/bin/python auto_resume.py --done outputs/eval/.never -- \
  .venv/bin/python thermal_guard.py -- .venv/bin/python stage3_eval.py \
  --model-dir outputs/smolvla_baseline/final --world-model outputs/world_model3 --variant position --episodes 50 --refine-steps 24
