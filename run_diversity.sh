#!/bin/bash
# Diversity check: rerun the smoothed baseline on the same lift_pot scenes with fresh sampling noise.
#   ns1: draws 8 plans per decision and logs their spread (executes the first); ns2: a third plain draw.
cd "$(dirname "$0")"
for args in "--noise-seed 1 --n-samples 8" "--noise-seed 2"; do
  .venv/bin/python auto_resume.py --done "outputs/eval/.never" -- \
    .venv/bin/python thermal_guard.py -- .venv/bin/python step4_eval.py --model-dir outputs/smolvla_baseline/final \
    --task lift_pot --variant standard --episodes 150 --smooth $args
done
