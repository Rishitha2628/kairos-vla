#!/bin/bash
# Stop stage0_record.py once 15 scenes have both runs (enough for the go/no-go), then run stage0_analyze.py.
cd "$(dirname "$0")"
ROOT=outputs/stage0/position
until [ "$(ls -d $ROOT/seed*_policy 2>/dev/null | grep -vc '\.tmp$')" -ge 15 ]; do sleep 30; done
kill 609856 609862 609863 2>/dev/null   # auto_resume, thermal_guard, stage0_record (exact PIDs)
sleep 10
kill 610013 2>/dev/null   # its policy_server.py
sleep 20
rm -rf $ROOT/*.tmp
echo "[run_stage0_analyze] $(date '+%F %T') recording stopped at $(ls -d $ROOT/seed*_policy | wc -l) scenes"
nice -n 19 ionice -c 3 .venv/bin/python thermal_guard.py -- .venv-train/bin/python stage0_analyze.py --variant position
