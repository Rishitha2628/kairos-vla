#!/bin/bash
# Extend lift_pot standard to more episodes for baseline vs foresight (finished episodes are kept, same seeds).
#   ./run_long_lift_pot.sh 150
cd "$(dirname "$0")"
N=${1:-150}
for smooth in --smooth ""; do
  for model in smolvla_baseline smolvla_foresight; do
    .venv/bin/python auto_resume.py --done "outputs/eval/.never" -- \
      .venv/bin/python thermal_guard.py -- .venv/bin/python step4_eval.py --model-dir "outputs/$model/final" \
      --task lift_pot --variant standard --episodes "$N" $smooth
  done
done
