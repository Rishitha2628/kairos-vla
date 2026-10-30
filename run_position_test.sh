#!/bin/bash
# After the standard-test planner run: the training pot at new poses (--variant position), 50 scenes,
# baseline only (a planner that picks among SmolVLA's plans cannot reach poses SmolVLA does not aim for).
cd "$(dirname "$0")"
until tr '\r' '\n' < outputs/eval_planner_varied_stdout.log | grep -qE "^lift_pot_standard: [0-9]+/150 ="; do sleep 60; done
while pgrep -f "step4_eval.py" > /dev/null; do sleep 30; done
COMMON="--model-dir outputs/smolvla_baseline/final --task lift_pot --variant position --episodes 50 --smooth --noise-seed 1"
.venv/bin/python auto_resume.py --done outputs/eval/.never -- \
  .venv/bin/python thermal_guard.py -- .venv/bin/python step4_eval.py $COMMON
