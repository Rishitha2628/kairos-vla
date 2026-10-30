#!/bin/bash
# After encode_rollouts.py finishes: train the Stage 2/3 world models + success scorer.
cd "$(dirname "$0")"
until [ -f outputs/rollouts/lift_pot/vjepa2_stride4.pt ] && ! pgrep -f encode_rollouts.py > /dev/null; do sleep 30; done
.venv/bin/python thermal_guard.py -- .venv-train/bin/python train_world_model2.py
