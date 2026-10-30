#!/bin/bash
# Step 4/8: evaluate one model on all test sets, one after another, then (with --videos) build a labelled reel per test.
#   ./run_eval.sh outputs/smolvla_baseline/final            # raw actions (RoboTwin protocol)
#   ./run_eval.sh outputs/smolvla_baseline/final --smooth   # smoothed actions -> outputs/eval/<tag>_smooth
# Extra args go to step4_eval.py. Each test runs under thermal_guard + auto_resume; step4_eval.py skips
# finished episodes on restart.
cd "$(dirname "$0")"
MODEL=$1; shift
EXTRA=("$@")
TAG=$(basename "$(dirname "$MODEL")")
[[ " ${EXTRA[*]} " == *" --smooth "* ]] && TAG=${TAG}_smooth
for t in "lift_pot standard" "lift_pot hard" "beat_block_hammer standard"; do
  read -r task variant <<< "$t"
  .venv/bin/python auto_resume.py --done "outputs/eval/.never" -- \
    .venv/bin/python thermal_guard.py -- .venv/bin/python step4_eval.py --model-dir "$MODEL" --task "$task" \
    --variant "$variant" "${EXTRA[@]}"
  [[ " ${EXTRA[*]} " == *" --videos "* ]] &&
    nice -n 19 .venv/bin/python make_eval_reel.py --tag "$TAG" --name "${task}_${variant}"
done
