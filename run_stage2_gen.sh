#!/bin/bash
# After the play data is encoded (encode_rollouts.py --root outputs/play): train + test the Stage 2 world model.
cd "$(dirname "$0")"
until grep -q "^\[auto_resume\].*finished" outputs/play_encode_stdout.log; do sleep 60; done
nice -n 19 ionice -c 3 .venv/bin/python thermal_guard.py -- .venv-train/bin/python train_world_model3.py
